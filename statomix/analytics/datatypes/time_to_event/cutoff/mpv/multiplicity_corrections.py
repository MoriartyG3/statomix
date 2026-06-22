"""
Multiplicity corrections for `MinimumPValue` sweeps.

Deliberately kept separate from `MinimumPValue` itself: `get_mpv_df()`
stays fast and side-effect-free (one log-rank + one Cox fit per
threshold), and nothing about the core sweep changes whether or not a
correction is ever applied. Corrections are opt-in, applied to an
already-computed `mpv_df`, and you choose which one (if any) fits your
situation -- see `compare_corrections()` below for a side-by-side view
before committing to one.

Why "minimum p-value" needs its own correction, not generic FDR
-----------------------------------------------------------------
`MinimumPValue` scans many thresholds over *one* underlying covariate and
asks "what is the smallest log-rank p-value anywhere in this scan?". This
is fundamentally different from testing many independent hypotheses and
asking which ones survive:

- The thresholds are not independent tests of different hypotheses --
  they're one family of highly correlated test statistics (adjacent
  thresholds barely move the group assignment of anyone except the
  handful of subjects near the cutoff), all aimed at the *same*
  underlying question ("is there a cutpoint that separates survival in
  this covariate?").
- Benjamini-Hochberg / FDR control is built for the "many separate
  discoveries, how many are false" setting (e.g. testing thousands of
  genes). It is not designed for "I took the min over one correlated
  family" and using it here would not correctly bound the false-positive
  rate of the *minimum*. It is intentionally NOT offered here, to avoid
  the wrong tool being reached for by default. See `compare_corrections`
  docstring for which generic corrections (Bonferroni) *do* still apply,
  with caveats.

Methods provided
-----------------
1. `bonferroni_correction` -- generic, conservative, fast. Multiplies the
   raw minimum p-value by the number of valid thresholds tested. Treats
   thresholds as if independent, which they are not (adjacent thresholds
   are strongly correlated), so this is an upper bound on the true
   adjusted p-value -- typically far too conservative, but cheap and a
   useful sanity check / worst case.

2. `lausen_schumacher_correction` -- analytic approximation derived
   specifically for the minimum p-value over an ordered continuous
   covariate (Lausen & Schumacher, 1992, Biometrics 48:73-85, building on
   Miller & Siegmund 1982). Uses the standardized log-rank statistic's
   range over the scanned thresholds and an asymptotic formula for the
   tail probability of its maximum. Much less conservative than
   Bonferroni and fast (no resampling), but it's an asymptotic
   approximation -- accuracy depends on sample size and how extreme the
   range of group-size proportions scanned is.

3. `permutation_correction` -- the gold-standard, assumption-light
   option for this exact problem (Miller & Siegmund's original proposal
   uses this idea). Repeatedly shuffles the (time, event) pairs against
   the covariate, reruns the *entire threshold sweep* on each shuffled
   dataset, and records the minimum log-rank p-value each time. The
   adjusted p-value is the fraction of permutations whose minimum
   p-value is at least as extreme as the one observed in the real data.
   This directly estimates the null distribution of "the smallest
   p-value you'd see by chance when scanning this many (correlated)
   thresholds with this sample size," so it requires no asymptotic or
   independence assumptions -- the tradeoff is compute cost: each
   permutation reruns a full threshold sweep.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from lifelines.statistics import logrank_test


@dataclass
class CorrectionResult:
    """Result of applying one multiplicity correction to an MPV sweep.

    Attributes
    ----------
    method : str
        Name of the correction applied.
    raw_p_value : float
        The unadjusted minimum log-rank p-value from the observed sweep.
    adjusted_p_value : float
        The multiplicity-corrected p-value. Always >= raw_p_value
        (correcting for multiple comparisons can only make the result
        less significant, never more).
    threshold : float
        The threshold at which the raw minimum p-value was observed.
    n_thresholds_tested : int
        Number of valid thresholds included in the sweep this correction
        was applied to.
    details : dict
        Method-specific extra information (e.g. permutation null
        distribution summary stats, or the Lausen-Schumacher range
        statistic).
    """

    method: str
    raw_p_value: float
    adjusted_p_value: float
    threshold: float
    n_thresholds_tested: int
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CorrectionResult(method={self.method!r}, "
            f"threshold={self.threshold:.4g}, "
            f"raw_p_value={self.raw_p_value:.4g}, "
            f"adjusted_p_value={self.adjusted_p_value:.4g}, "
            f"n_thresholds_tested={self.n_thresholds_tested})"
        )


def _best_valid_row(mpv_df: pd.DataFrame) -> pd.Series:
    if "valid_split" not in mpv_df.columns:
        raise ValueError("mpv_df does not look like MinimumPValue output (no 'valid_split' column).")
    valid = mpv_df[mpv_df["valid_split"]]
    if valid.empty:
        raise ValueError("No valid threshold rows in mpv_df (all splits were degenerate or errored).")
    if "p_value" not in valid.columns:
        raise ValueError("mpv_df has no 'p_value' column to correct.")
    return valid.loc[valid["p_value"].idxmin()]


def bonferroni_correction(mpv_df: pd.DataFrame) -> CorrectionResult:
    """Bonferroni correction: adjusted_p = min(raw_p * n_tests, 1.0).

    `n_tests` is the number of *valid* thresholds in `mpv_df` (rows with
    `valid_split == True`) -- degenerate splits that were never actually
    tested are correctly excluded from the multiplier.

    This treats each threshold's test as independent. They aren't:
    moving the threshold by one observation only changes the group
    assignment of that one subject, so consecutive tests are highly
    correlated and the true effective number of independent tests is
    much smaller than `n_tests`. Bonferroni therefore over-corrects --
    treat its adjusted p-value as a conservative upper bound, not a
    best estimate.
    """
    best_row = _best_valid_row(mpv_df)
    n_tests = int(mpv_df["valid_split"].sum())
    raw_p = float(best_row["p_value"])
    adjusted_p = min(raw_p * n_tests, 1.0)

    return CorrectionResult(
        method="bonferroni",
        raw_p_value=raw_p,
        adjusted_p_value=adjusted_p,
        threshold=float(best_row["threshold"]),
        n_thresholds_tested=n_tests,
        details={
            "note": (
                "Conservative upper bound: assumes independent tests, "
                "which adjacent thresholds are not."
            )
        },
    )


def lausen_schumacher_correction(
    mpv_df: pd.DataFrame,
    surv_df_mpv: pd.DataFrame,
    target_col_name: str | None = None,
    min_group_fraction: float = 0.1,
) -> CorrectionResult:
    """Lausen & Schumacher (1992) analytic correction for the minimum
    p-value over an ordered continuous covariate.

    Approximates the null distribution of the *maximum* standardized
    log-rank statistic (equivalently, the minimum p-value) over all
    cutpoints that split the cohort into two groups with size fraction
    between `min_group_fraction` and `1 - min_group_fraction`, using the
    asymptotic formula from Lausen & Schumacher (1992) / Miller &
    Siegmund (1982):

        P(max |Z(t)| > b)  ~=  4 * phi(b) / b
                              + phi(b) * b * (1/eps2 - 1) * [ (1/eps1 - 1/eps2) ]
                              (Davies' approximation; see Notes)

    where `phi` is the standard normal density, `b` is the observed
    maximum standardized log-rank statistic, and `eps1`/`eps2` are the
    smallest/largest group-fraction bounds actually scanned.

    Parameters
    ----------
    mpv_df : pandas.DataFrame
        Output of `MinimumPValue.get_mpv_df()`.
    surv_df_mpv : pandas.DataFrame
        The same data originally passed to `MinimumPValue` (needed to
        compute group-size fractions per threshold, since `mpv_df` itself
        only stores group counts post-hoc per row).
    target_col_name : str, optional
        Name of the grouping/covariate column in `surv_df_mpv`. If not
        given, inferred as the one column that isn't "time"/"event".
    min_group_fraction : float, default 0.1
        Thresholds producing a group-size split more extreme than this
        (e.g. a 5%/95% split when `min_group_fraction=0.1`) are excluded
        from the correction. This mirrors standard practice: very
        unbalanced splits make the asymptotic approximation unreliable
        and are rarely of practical interest anyway. Must be in
        (0, 0.5).

    Returns
    -------
    CorrectionResult
        `details` includes the standardized statistic `b`, and `eps1`,
        `eps2` (the min/max group-size fractions actually used).

    Notes
    -----
    This is an asymptotic approximation; it assumes proportional hazards
    and a reasonably large sample size relative to how fine
    `search_resolution` is. With very small samples or extremely fine
    threshold grids, prefer `permutation_correction` instead, which makes
    no such asymptotic assumption.

    Raises
    ------
    ValueError
        If `min_group_fraction` is out of range, or no valid thresholds
        remain after filtering by `min_group_fraction`.
    """
    if not 0 < min_group_fraction < 0.5:
        raise ValueError(
            f"min_group_fraction must be in (0, 0.5), got {min_group_fraction!r}"
        )

    if target_col_name is None:
        grouping_cols = [c for c in surv_df_mpv.columns if c not in ("time", "event")]
        if len(grouping_cols) != 1:
            raise ValueError(
                "Could not infer target_col_name automatically; pass it explicitly."
            )
        target_col_name = grouping_cols[0]

    valid = mpv_df[mpv_df["valid_split"]].copy()
    if valid.empty:
        raise ValueError("No valid threshold rows in mpv_df.")

    n_total = len(surv_df_mpv)
    target_col = surv_df_mpv[target_col_name]

    # group0_n is the size of the "<=threshold" group, stored per row by
    # MinimumPValue. Use that directly rather than recomputing from
    # surv_df_mpv, so this stays consistent with whatever rows mpv_df
    # actually contains.
    if "group0_n" not in valid.columns:
        raise ValueError("mpv_df is missing 'group0_n' -- unexpected MinimumPValue output format.")

    valid["_group_fraction"] = valid["group0_n"] / n_total
    fraction_ok = (valid["_group_fraction"] >= min_group_fraction) & (
        valid["_group_fraction"] <= 1 - min_group_fraction
    )
    valid_filtered = valid[fraction_ok]

    if valid_filtered.empty:
        raise ValueError(
            "No thresholds remain after filtering by min_group_fraction="
            f"{min_group_fraction}. Try lowering it, or check that "
            "surv_df_mpv matches the data used to build mpv_df."
        )

    # Lausen & Schumacher's formula treats a fraction t and its mirror
    # image 1-t symmetrically (a 20%/80% split carries the same
    # information as an 80%/20% split). Fold everything into the
    # smaller-fraction side so eps1 <= eps2 <= 0.5.
    folded_fractions = valid_filtered["_group_fraction"].apply(lambda f: min(f, 1 - f))
    eps1 = max(folded_fractions.min(), 1.0 / n_total)
    eps2 = folded_fractions.max()

    best_row = valid_filtered.loc[valid_filtered["p_value"].idxmin()]
    raw_p = float(best_row["p_value"])

    # Recover the standardized log-rank statistic b from the two-sided
    # p-value: p = 2 * (1 - Phi(b))  =>  b = Phi^{-1}(1 - p/2).
    b = stats.norm.ppf(1 - raw_p / 2)

    if not np.isfinite(b) or b <= 0:
        adjusted_p = 1.0
    else:
        phi_b = stats.norm.pdf(b)
        # Davies' (1987) approximation to P(max |Z(t)| > b) over
        # eps1 <= t <= eps2, as used by Lausen & Schumacher (1992).
        term1 = 4 * phi_b / b
        term2 = phi_b * b * ((1.0 / eps1 - 1.0 / eps2) - (1.0 / (1 - eps2) - 1.0 / (1 - eps1))) / 2.0
        # term2 can be negative for numerically degenerate eps bounds;
        # clip contributions to keep the result a valid probability.
        term2 = max(term2, 0.0)
        adjusted_p = min(term1 + term2, 1.0)
        # A multiplicity-corrected p-value should never end up smaller
        # than the raw single-test p-value it's correcting -- if the
        # approximation ever produces that (possible near the numerical
        # edges of the eps1/eps2 range), floor it at raw_p.
        adjusted_p = max(adjusted_p, raw_p)

    return CorrectionResult(
        method="lausen_schumacher",
        raw_p_value=raw_p,
        adjusted_p_value=adjusted_p,
        threshold=float(best_row["threshold"]),
        n_thresholds_tested=int(len(valid_filtered)),
        details={
            "standardized_statistic_b": float(b) if np.isfinite(b) else None,
            "eps1": float(eps1),
            "eps2": float(eps2),
            "min_group_fraction": min_group_fraction,
            "note": (
                "Asymptotic approximation (Davies 1987 / Lausen & Schumacher "
                "1992). Assumes proportional hazards and adequate sample size."
            ),
        },
    )


def permutation_correction(
    minimum_p_value_obj,
    n_permutations: int = 1000,
    random_seed: int | None = 42,
    show_progress: bool = True,
    fast_logrank_only: bool = True,
) -> CorrectionResult:
    """Permutation-based correction: the most direct, assumption-light
    answer to "how surprising is this minimum p-value, given how many
    correlated thresholds were scanned?".

    Procedure
    ---------
    1. Run (or reuse) the real threshold sweep, take the minimum log-rank
       p-value -- call it `p_obs`.
    2. Repeat `n_permutations` times: randomly shuffle the covariate
       column relative to (time, event) -- this breaks any real
       association while preserving the marginal distributions of the
       covariate, the censoring pattern, and the survival times exactly
       -- then rerun the *entire* threshold sweep on the shuffled data
       and record its minimum p-value.
    3. The adjusted p-value is the fraction of permutations whose minimum
       p-value is <= `p_obs` (i.e. as extreme or more extreme than what
       was actually observed), with the standard +1/+1 continuity
       correction so the adjusted p-value is never reported as exactly 0
       regardless of `n_permutations`.

    This is the most defensible correction for this specific problem: it
    makes no assumption about independence between thresholds or about
    the asymptotic distribution of the test statistic, because it
    empirically rebuilds the actual null distribution of "minimum
    p-value when scanning this many thresholds on data this size," using
    the real sweep procedure each time. The cost is computational: each
    permutation reruns the full sweep.

    Parameters
    ----------
    minimum_p_value_obj : MinimumPValue
        The fitted `MinimumPValue` instance for the real data sweep, used
        as the template for permutations (its `surv_df_mpv`,
        `target_col_name`, `use_synthetic_cutoffs`, `search_resolution`,
        and `alpha` are all reused unchanged for every permuted sweep).
    n_permutations : int, default 1000
        More permutations -> more precise adjusted p-value, at
        proportionally higher cost. 1000 gives a minimum resolvable
        p-value of about 1/1001; if you need to resolve much smaller
        p-values precisely, increase this.
    random_seed : int or None, default 42
        Base seed; permutation `i` uses `random_seed + i` (when not
        None), so the whole correction is reproducible without every
        permutation being identical.
    show_progress : bool, default True
    fast_logrank_only : bool, default True
        If True (recommended), each permutation's sweep uses
        `lifelines.statistics.logrank_test` directly instead of
        constructing a full `BinaryClassSurv` (which also fits a Cox
        model and two KM curves per threshold). The minimum p-value
        across thresholds only depends on the log-rank statistic, so
        skipping the unused Cox/KM fitting makes each permutation
        dramatically cheaper with no change in the result. Set False
        only if you need to debug against the exact `BinaryClassSurv`
        code path (much slower: a full Cox fit per threshold per
        permutation).

    Returns
    -------
    CorrectionResult
        `details` includes the full null distribution of permuted
        minimum p-values (`null_min_p_values`), so you can inspect or
        re-plot it (e.g. as a histogram against `p_obs`).

    Raises
    ------
    ValueError
        If `n_permutations < 1`, or the real sweep has no valid
        threshold (nothing to correct).
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}")

    mpv = minimum_p_value_obj
    real_mpv_df = mpv.get_mpv_df()
    best_row = _best_valid_row(real_mpv_df)
    p_obs = float(best_row["p_value"])

    surv_df = mpv.surv_df_mpv
    target_col_name = mpv.target_col_name
    thresholds = mpv._get_thresholds()  # reuse the exact same grid as the real sweep

    time = surv_df["time"].to_numpy()
    event = surv_df["event"].to_numpy()
    covariate = surv_df[target_col_name].to_numpy()

    null_min_p_values = np.empty(n_permutations, dtype=float)
    iterator = tqdm(range(n_permutations)) if show_progress else range(n_permutations)

    for i in iterator:
        rng = np.random.default_rng(random_seed + i if random_seed is not None else None)
        shuffled_covariate = rng.permutation(covariate)

        if fast_logrank_only:
            null_min_p_values[i] = _fast_min_logrank_p(
                time, event, shuffled_covariate, thresholds
            )
        else:
            shuffled_df = pd.DataFrame(
                {"time": time, "event": event, target_col_name: shuffled_covariate}
            )
            shuffled_mpv = type(mpv)(
                surv_df_mpv=shuffled_df,
                surv_label=mpv.surv_label,
                use_synthetic_cutoffs=mpv.use_synthetic_cutoffs,
                search_resolution=mpv.search_resolution,
                show_progress=False,
                alpha=mpv.alpha,
                skip_invalid_thresholds=True,
            )
            shuffled_result_df = shuffled_mpv.get_mpv_df()
            valid_shuffled = shuffled_result_df[shuffled_result_df["valid_split"]]
            null_min_p_values[i] = (
                valid_shuffled["p_value"].min() if not valid_shuffled.empty else 1.0
            )

    # +1/+1 continuity correction: guarantees adjusted_p > 0 always, and
    # is the standard unbiased estimator for a permutation-test p-value.
    n_as_extreme = int(np.sum(null_min_p_values <= p_obs))
    adjusted_p = (n_as_extreme + 1) / (n_permutations + 1)

    return CorrectionResult(
        method="permutation",
        raw_p_value=p_obs,
        adjusted_p_value=adjusted_p,
        threshold=float(best_row["threshold"]),
        n_thresholds_tested=int(real_mpv_df["valid_split"].sum()),
        details={
            "n_permutations": n_permutations,
            "n_as_extreme_or_more": n_as_extreme,
            "null_min_p_values": null_min_p_values,
            "null_mean": float(np.mean(null_min_p_values)),
            "null_5th_percentile": float(np.percentile(null_min_p_values, 5)),
        },
    )


def _fast_min_logrank_p(
    time: np.ndarray,
    event: np.ndarray,
    covariate: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """Minimum log-rank p-value across `thresholds`, computed directly
    via `logrank_test` with no `BinaryClassSurv`/Cox overhead. Used only
    inside the permutation loop, where only the minimum p-value matters
    and per-threshold descriptive stats (median survival, HR, etc.) are
    discarded anyway.
    """
    best_p = 1.0
    for threshold in thresholds:
        is_low = covariate <= threshold
        n_low, n_high = is_low.sum(), (~is_low).sum()
        if n_low == 0 or n_high == 0:
            continue
        try:
            result = logrank_test(
                time[is_low],
                time[~is_low],
                event_observed_A=event[is_low],
                event_observed_B=event[~is_low],
            )
        except Exception:
            # A degenerate permuted split (e.g. all-censored on one side)
            # shouldn't crash the whole permutation run -- just skip it,
            # consistent with MinimumPValue's own skip_invalid_thresholds
            # behavior on the real sweep.
            continue
        if result.p_value < best_p:
            best_p = result.p_value
    return best_p


def compare_corrections(
    minimum_p_value_obj,
    mpv_df: pd.DataFrame | None = None,
    n_permutations: int = 1000,
    min_group_fraction: float = 0.1,
    random_seed: int | None = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run all three corrections and return a side-by-side comparison.

    This is the recommended entry point: rather than picking one
    correction up front, run all three and look at how much they agree.
    Bonferroni and Lausen-Schumacher are cheap and will usually bracket
    the permutation result (Bonferroni above, Lausen-Schumacher close to
    or below it); if the permutation correction disagrees substantially
    with both, that's a signal the analytic assumptions (independence for
    Bonferroni, asymptotics for Lausen-Schumacher) may not hold well for
    your sample size or threshold grid, and the permutation result should
    be trusted over the other two.

    Parameters
    ----------
    minimum_p_value_obj : MinimumPValue
        Fitted instance for the real data.
    mpv_df : pandas.DataFrame, optional
        Reuse an already-computed sweep instead of recomputing it
        (needed for Bonferroni/Lausen-Schumacher; the permutation
        correction always reruns the sweep on the real data once more
        internally for consistency, which is cheap relative to the
        permutations themselves).
    n_permutations, random_seed, show_progress
        Passed through to `permutation_correction`.
    min_group_fraction
        Passed through to `lausen_schumacher_correction`.

    Returns
    -------
    pandas.DataFrame
        One row per method, with columns: method, raw_p_value,
        adjusted_p_value, threshold, n_thresholds_tested.
    """
    if mpv_df is None:
        mpv_df = minimum_p_value_obj.get_mpv_df()

    results = []

    bonf = bonferroni_correction(mpv_df)
    results.append(bonf)

    global_raw_p = bonf.raw_p_value  # the true global minimum, for comparison below

    try:
        ls = lausen_schumacher_correction(
            mpv_df,
            surv_df_mpv=minimum_p_value_obj.surv_df_mpv,
            target_col_name=minimum_p_value_obj.target_col_name,
            min_group_fraction=min_group_fraction,
        )
        results.append(ls)
    except ValueError as exc:
        warnings.warn(f"Skipping Lausen-Schumacher correction: {exc}", UserWarning)

    perm = permutation_correction(
        minimum_p_value_obj,
        n_permutations=n_permutations,
        random_seed=random_seed,
        show_progress=show_progress,
    )
    results.append(perm)

    comparison = pd.DataFrame(
        [
            {
                "method": r.method,
                "raw_p_value": r.raw_p_value,
                "adjusted_p_value": r.adjusted_p_value,
                "threshold": r.threshold,
                "n_thresholds_tested": r.n_thresholds_tested,
            }
            for r in results
        ]
    )

    # Lausen-Schumacher restricts to thresholds within
    # [min_group_fraction, 1 - min_group_fraction] *before* finding its
    # own minimum p-value, so its raw_p_value/threshold can legitimately
    # differ from Bonferroni's and the permutation correction's (which
    # both use the unrestricted global minimum). Flag this explicitly so
    # the differing raw_p_value isn't mistaken for a bug or a
    # disagreement between methods on the same test.
    if (comparison["method"] == "lausen_schumacher").any():
        ls_raw_p = comparison.loc[comparison["method"] == "lausen_schumacher", "raw_p_value"].iloc[0]
        if not np.isclose(ls_raw_p, global_raw_p):
            comparison["note"] = ""
            comparison.loc[comparison["method"] == "lausen_schumacher", "note"] = (
                f"raw_p_value computed only over thresholds with group fraction in "
                f"[{min_group_fraction}, {1 - min_group_fraction}] -- not directly "
                f"comparable to the other rows' global minimum "
                f"(raw_p_value={global_raw_p:.4g})"
            )

    return comparison
