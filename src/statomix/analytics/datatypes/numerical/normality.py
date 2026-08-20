"""
normality.py
=============

A toolkit for testing whether a 1-D numeric sample plausibly comes from a
normal distribution, and for figuring out WHICH test to trust when several
disagree.

DESIGN NOTE: data is passed to the constructor (Normality(series)), not via
a separate set_data() call. This means a Normality object is either fully
valid or doesn't exist -- there's no intermediate state where the object
exists but holds no/stale data, and no way to accidentally call a report
method against the wrong dataset by reusing one instance across calls.

WHY "WHICH TEST" IS NOT JUST A SAMPLE-SIZE QUESTION
----------------------------------------------------
A common rule of thumb is "Shapiro-Wilk for n<=50, something else above
that." That rule only addresses validity at a given n — it says nothing
about which test has the best POWER for your n, what KIND of departure
from normality each test is good at catching, or whether your data has
properties (ties, outliers) that quietly break a test's assumptions. This
module's diagnostic methods (get_tie_diagnostics, get_outlier_diagnostics,
get_power_note, recommend_test_for_purpose) exist to cover those gaps.

THE SIX TESTS, AND WHAT EACH IS ACTUALLY GOOD AT
--------------------------------------------------
- Shapiro-Wilk (`shapiro`): Based on the correlation between the data and
  its expected normal order statistics. One of the most powerful
  all-purpose tests across small AND large samples (the "n<=50" rule is
  largely a historical/computational artifact, not a deep statistical
  limit). Good default choice when you have no other reason to prefer
  something else.
- Kolmogorov-Smirnov (`ks`): Compares the empirical CDF to a normal CDF.
  AS IMPLEMENTED HERE, parameters (mean/std) are estimated from the same
  sample being tested — this is technically invalid for a KS test (which
  assumes known, not estimated, parameters) and biases p-values upward,
  making the test too lenient. Kept for reference/comparison only. Most
  sensitive to deviations near the CENTER of the distribution, weakest in
  the tails of the three "shape" tests here.
- Anderson-Darling (`anderson`): Like KS, but weights the TAILS of the
  distribution more heavily. Best choice when you specifically care about
  extreme-value behavior (risk modeling, control limits, capability
  indices). Valid for n>20 or so.
- D'Agostino-Pearson (`dagostino`): Built from sample skewness and
  kurtosis (3rd/4th moments). Powerful against asymmetry and tail-weight
  problems specifically, but can MISS other shape issues (e.g.
  bimodality) that don't show up in skew/kurtosis. Best for n>50;
  moment-based, so sensitive to a few extreme outliers.
- Jarque-Bera (`jb`): Also skew/kurtosis-based, asymptotic (large-sample)
  test. Same strengths/blind spots as D'Agostino, more common in
  econometrics. Most reliable at large n.
- Lilliefors (`lilliefors`): KS corrected for estimated parameters — the
  statistically valid version of what `ks` is trying to do above. Prefer
  this over `ks` whenever mean/std are estimated from the sample (the
  normal case).

A MORE COMPLETE DECISION CHECKLIST
------------------------------------
1. What do you actually care about? Overall shape, center, or tails?
   -> see `recommend_test_for_purpose()`.
2. Is your sample size enough to have real power, or so large that any
   test will reject trivial/practically-irrelevant deviations?
   -> see `get_power_note()`.
3. Are parameters (mean/std) known in advance, or estimated from this
   same sample? (Almost always the latter in practice.)
   -> use `lilliefors`, not `ks`, when estimated.
4. Does the data have a lot of repeated/tied values (rounding, Likert
   scales, sensor quantization)? KS-family tests degrade with heavy ties.
   -> see `get_tie_diagnostics()`.
5. Are a handful of outliers driving the result? Moment-based tests
   (Jarque-Bera, D'Agostino) are very sensitive to a few extreme points.
   -> see `get_outlier_diagnostics()`.
6. What will you DO with the normality verdict? Many downstream
   procedures (t-test, ANOVA, linear regression) are fairly robust to
   mild non-normality at moderate-to-large n (Central Limit Theorem) —
   strict rejection by a test doesn't always mean the downstream method
   will misbehave.
7. Look at it, don't just test it: `qq_plot()` and
   `hist_with_normal_curve()` show WHERE and HOW MUCH a distribution
   deviates, which a p-value alone cannot.

For a full narrative walkthrough of all of this with worked examples, see
the companion document "Choosing a Normality Test — A Practical Guide"
(provided separately alongside this module).
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors

from scipy.stats import probplot
from scipy.stats import shapiro, kstest, normaltest, jarque_bera, anderson, skew, kurtosis


class Normality:
    """
    Runs and reports on a battery of normality tests for a 1-D dataset,
    plus diagnostics to help decide WHICH test(s) to trust.

    Data is supplied at construction time (not via a separate set_data()
    call), so each instance is tied to exactly one dataset for its whole
    lifetime. This makes it safe to hold onto multiple instances (e.g.
    one per DataFrame column) without any risk of one instance's data
    being silently overwritten by a later call elsewhere in the code.

    See the module-level docstring (top of this file) for a guide to test
    selection criteria beyond sample size: what each test is sensitive to,
    how ties/outliers affect reliability, and how to pick a test based on
    what you're actually using the normality check for.

    Quick start:
        >>> n = Normality(df["col"])
        >>> n.get_normality_report_full()        # run all 6 tests
        >>> n.get_consensus()                     # majority vote summary
        >>> n.qq_plot()                            # visual check

    Choosing a test:
        >>> n.get_full_diagnostics()                        # ties, outliers, power, shape — all at once
        >>> n.recommend_test_for_purpose("tails")            # purpose-driven recommendation
        >>> n.get_tie_diagnostics()                          # is KS/Lilliefors reliable here?
        >>> n.get_outlier_diagnostics()                      # are a few points driving the verdict?
        >>> n.get_power_note()                               # is n too small/large to trust the p-value alone?
    """

    def __init__(self, series: pd.Series, alpha=0.05, ddof=1):
        """
        Set the dataset to be tested.

        Args:
            series (pd.Series or array-like): 1-D numeric data.
            alpha (float): Significance level used for all tests' verdicts.
            ddof (int): Delta degrees of freedom for std calculation.
                        Use 0 for population std, 1 (default) for sample std.
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        assert series.ndim == 1, "Input data must be a 1-dimensional array (single row)."

        n_total = len(series)
        self.series = series.copy().dropna()
        n_dropped = n_total - len(self.series)
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} NaN value(s) out of {n_total} "
                f"({n_dropped / n_total:.1%}). Tests run on remaining "
                f"{len(self.series)} observations."
            )

        if len(self.series) < 3:
            raise ValueError(
                f"Need at least 3 non-NaN observations to test normality, "
                f"got {len(self.series)}."
            )

        self.alpha = alpha
        self.ddof = ddof
        self.mean = np.mean(self.series)
        self.std = np.std(a=self.series, ddof=self.ddof)

        # name -> bound method
        self._test_dispatch = {
            "shapiro": self._shapiro,
            "ks": self._ks,
            "anderson": self._anderson,
            "dagostino": self._dagostino,
            "jb": self._jb,
            "lilliefors": self._lilliefors,
        }
        self._implemented_tests = list(self._test_dispatch.keys())

    def _verdict(self, p_value, alpha=None):
        """Single source of truth for the reject/fail-to-reject decision."""
        return p_value > (alpha if alpha is not None else self.alpha)

    def get_normality_report_full(self, as_dataframe=True):
        """
        Run every implemented normality test.

        Args:
            as_dataframe (bool): If True (default), return a tidy
                pd.DataFrame indexed by test name. If False, return the
                raw list of dicts.

        Returns:
            pd.DataFrame or list[dict]
        """
        results = [self._test_dispatch[name]() for name in self._implemented_tests]

        if not as_dataframe:
            return results

        return pd.DataFrame(results).set_index("test_type")

    def get_normality_report_default(self):
        n = len(self.series)
        ties = self.get_tie_diagnostics()

        # Shapiro is the preferred all-purpose test whenever its p-value is reliable.
        if n <= 5000:
            return self._shapiro()

        # Above 5000, avoid KS-family tests if ties are heavy.
        if not ties["ks_family_reliable"]:
            return self._anderson()

        return self._lilliefors()

    def get_recommended_tests(self):
        """
        Suggest which tests are statistically appropriate for the current
        sample size, since not every test is reliable at every n.

        Returns:
            list[str]: Subset of self._implemented_tests recommended for
            the current sample size.
        """
        n = len(self.series)
        recommended = []
        if n <= 50:
            recommended.append("shapiro")
        if n > 20:
            recommended.append("anderson")
        if n > 50:
            recommended.extend(["dagostino", "lilliefors"])
        if not recommended:
            # very small samples: Shapiro is still the standard fallback
            recommended.append("shapiro")
        recommended.append("jb")  # asymptotic, but commonly reported regardless
        return recommended

    def get_tie_diagnostics(self, warn_threshold=0.10):
        """
        Check how much repeated/duplicate values ("ties") are present.
        KS-family tests (KS, Lilliefors) assume a continuous distribution
        with no ties; heavily tied data (rounded measurements, Likert
        scales, sensor quantization) inflates their statistic in ways the
        reported p-value does not account for, making them unreliable.
        Shapiro-Wilk, Anderson-Darling, D'Agostino, and Jarque-Bera
        tolerate ties much better.

        Args:
            warn_threshold (float): Fraction of duplicated values above
                which KS-family tests should be flagged as unreliable.

        Returns:
            dict: {
                'n_unique': int,
                'n_total': int,
                'tie_fraction': float,       # 0 = all unique, ->1 = heavily tied
                'ks_family_reliable': bool,  # False if tie_fraction exceeds threshold
            }
        """
        n_total = len(self.series)
        n_unique = self.series.nunique()
        tie_fraction = 1 - (n_unique / n_total)

        return {
            "n_unique": int(n_unique),
            "n_total": int(n_total),
            "tie_fraction": tie_fraction,
            "ks_family_reliable": tie_fraction <= warn_threshold,
        }

    def get_outlier_diagnostics(
        self,
        method="iqr",
        z_thresh=3.0,
        modified_z_thresh=3.5,
    ):
        """
        Flag potential outliers, since a handful of extreme points can
        drive rejection in moment-based tests (Jarque-Bera, D'Agostino —
        both use the 3rd/4th moments and are very sensitive to extremes)
        and in Shapiro-Wilk / Anderson-Darling. Useful to check before
        trusting a "non-normal" verdict: is it a real shape issue, or a
        few bad/extreme points?

        Args:
            method (str):
                - 'iqr'            : Tukey's fences (1.5 * IQR)
                - 'zscore'         : |z| > z_thresh
                - 'modified_zscore': |modified_z| > modified_z_thresh
            z_thresh (float):
                Threshold used when method='zscore'. Default = 3.0.
            modified_z_thresh (float):
                Threshold used when method='modified_zscore'.
                Default = 3.5 (Iglewicz & Hoaglin recommendation).

        Returns:
            dict: {
                'method': str,
                'threshold': float | str,
                'n_outliers': int,
                'outlier_fraction': float,
                'outlier_values': list[float],
            }
        """
        data = self.series

        if method == "iqr":
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = data[(data < lower) | (data > upper)]
            threshold = "1.5*IQR"

        elif method == "zscore":
            std = data.std(ddof=self.ddof)

            if std == 0:
                return {
                    "method": method,
                    "threshold": z_thresh,
                    "n_outliers": 0,
                    "outlier_fraction": 0.0,
                    "outlier_values": [],
                    "computable": False,
                    "note": "Standard deviation is zero; z-scores cannot be computed.",
                }

            z = (data - data.mean()) / std
            outliers = data[z.abs() > z_thresh]
            threshold = z_thresh

        elif method == "modified_zscore":
            median = data.median()
            mad = np.median(np.abs(data - median))

            if mad == 0:
                return {
                    "method": method,
                    "threshold": modified_z_thresh,
                    "n_outliers": 0,
                    "outlier_fraction": 0.0,
                    "outlier_values": [],
                    "computable": False,
                    "note": "MAD is zero; modified z-scores cannot be computed.",
                }

            modified_z = 0.6745 * (data - median) / mad
            outliers = data[np.abs(modified_z) > modified_z_thresh]
            threshold = modified_z_thresh

        else:
            raise ValueError(
                "method must be one of {'iqr', 'zscore', 'modified_zscore'}"
            )

        return {
            "method": method,
            "threshold": threshold,
            "n_outliers": int(len(outliers)),
            "outlier_fraction": len(outliers) / len(data),
            "outlier_values": sorted(outliers.tolist()),
            "computable": True,
        }

    def get_power_note(self):
        """
        Contextualize sample size against statistical power, since
        "valid sample-size range" and "good power" are not the same thing.

        - At small n (<20-30): every test has low power; failure to
          reject normality is weak evidence FOR normality, just an
          absence of strong evidence against it.
        - At large n (>~300-500): every test becomes highly powered
          against even trivial, practically meaningless deviations
          (e.g. rounding error). A "reject" verdict may be statistically
          real but practically irrelevant for downstream use (most
          parametric tests, e.g. t-test/ANOVA, are fairly robust to mild
          non-normality at large n thanks to the CLT).

        Returns:
            dict: {'n': int, 'regime': str, 'note': str}
        """
        n = len(self.series)
        if n < 20:
            regime = "low power"
            note = (
                "Sample is small; all tests have limited power here. A "
                "'fail to reject normality' result is weak evidence, not "
                "proof of normality. Lean on the QQ plot / histogram, not "
                "just the p-value."
            )
        elif n <= 300:
            regime = "moderate power"
            note = (
                "Sample size is in a reasonable range for most tests to "
                "have decent power without being oversensitive."
            )
        else:
            regime = "high power / oversensitive"
            note = (
                "Sample is large; tests will likely flag even small, "
                "practically unimportant deviations as statistically "
                "significant. Check effect size (skew/kurtosis magnitude, "
                "QQ-plot) rather than relying on the p-value alone, and "
                "consider whether your downstream procedure actually "
                "requires strict normality."
            )
        return {"n": n, "regime": regime, "note": note}

    def recommend_test_for_purpose(self, purpose="general"):
        """
        Recommend a primary test (plus rationale) based on what you are
        actually checking normality FOR, not just sample size — different
        downstream uses care about different parts of the distribution.

        Args:
            purpose (str): One of:
                - 'general': Overall best all-purpose check.
                - 'tails': Care about extreme-value / tail behavior
                  (risk models, control limits, capability indices).
                - 'parametric_test': About to run a t-test/ANOVA/regression
                  and want to sanity-check the normality assumption.
                - 'symmetry_only': Mainly worried about skew, not exact
                  shape (e.g. choosing mean vs. median as a summary).

        Returns:
            dict: {'purpose': str, 'recommended_test': str, 'rationale': str}
        """
        n = len(self.series)
        ties = self.get_tie_diagnostics()

        purposes = {
            "general": (
                "shapiro" if n <= 5000 else "lilliefors",
                "Shapiro-Wilk remains one of the most powerful all-purpose "
                "tests across a wide range of sample sizes, not just n<=50 "
                "(that cutoff is largely a legacy of older implementations). "
                "Falls back to Lilliefors above n=5000, where scipy's "
                "Shapiro-Wilk implementation documents that the p-value "
                "(not the W statistic itself) may no longer be accurate.",
            ),
            "tails": (
                "anderson",
                "Anderson-Darling weights the tails of the distribution "
                "more heavily than KS-family or Shapiro-Wilk, making it "
                "the most relevant choice when extreme-value behavior is "
                "what actually matters downstream.",
            ),
            "parametric_test": (
                "shapiro",
                "Shapiro-Wilk is the conventional pre-check before a "
                "t-test/ANOVA. Note that these downstream tests are "
                "fairly robust to mild non-normality at moderate-to-large "
                "n (Central Limit Theorem), so also check "
                "get_power_note() and the effect size, not just the verdict.",
            ),
            "symmetry_only": (
                "dagostino",
                "D'Agostino-Pearson is built directly from skewness and "
                "kurtosis, so it (and get_distribution_shape()) directly "
                "answers whether asymmetry is the concern, without being "
                "thrown off by other shape issues like multimodality.",
            ),
        }

        if purpose not in purposes:
            raise ValueError(f"purpose must be one of {list(purposes.keys())}")

        test, rationale = purposes[purpose]

        if not ties["ks_family_reliable"] and test in ("ks", "lilliefors"):
            rationale += (
                f" CAUTION: data has a high tie fraction "
                f"({ties['tie_fraction']:.1%}); KS-family tests are "
                f"unreliable here. Consider 'shapiro' or 'anderson' instead."
            )

        return {"purpose": purpose, "recommended_test": test, "rationale": rationale}

    def get_full_diagnostics(self):
        """
        One-stop summary combining sample size/power, ties, outliers,
        and distribution shape — the full context needed to decide which
        normality test(s) to trust, beyond a bare sample-size rule.

        Returns:
            dict with keys: 'power', 'ties', 'outliers', 'shape'
        """
        return {
            "power": self.get_power_note(),
            "ties": self.get_tie_diagnostics(),
            "outliers": self.get_outlier_diagnostics(),
            "shape": self.get_distribution_shape(),
        }

    def get_normality_report(self, test_type):
        """
        Generate a normality report for the data using the specified test type.

        Args:
            test_type (str): One of:
                - 'shapiro': Shapiro-Wilk test. Best for n <= 5000.
                - 'ks': One-sample Kolmogorov-Smirnov test against
                  N(mean, std) estimated FROM THE SAME DATA. Note: this is
                  statistically invalid as a goodness-of-fit test (KS
                  assumes known, not estimated, parameters), which biases
                  p-values upward. Kept for reference/comparison only —
                  prefer 'lilliefors' for KS-style testing in practice.
                - 'anderson': Anderson-Darling test. More weight on tails
                  than KS. Best for n > 20.
                - 'dagostino': D'Agostino-Pearson test (based on skew +
                  kurtosis). Best for n > 50.
                - 'jb': Jarque-Bera test (based on skew + kurtosis,
                  asymptotic). Most reliable for large n.
                - 'lilliefors': KS test corrected for estimated
                  parameters. The statistically appropriate alternative
                  to 'ks' above.

        Raises:
            ValueError: If `test_type` is not one of the implemented tests.

        Returns:
            dict: Results of the selected normality test.

        Example:
            >>> report = instance.get_normality_report('shapiro')
        """
        if test_type not in self._implemented_tests:
            raise ValueError(
                f"Test not implemented, must be one of {self._implemented_tests}"
            )

        return self._test_dispatch[test_type]()

    def get_consensus(self, alpha=None):
        """
        Majority-vote summary across all implemented tests. Useful because
        individual tests can disagree, especially near the alpha boundary
        or with small/moderate, heavy-tailed samples.

        Args:
            alpha (float, optional): Override the significance level used
                for this consensus check without changing the instance default.

        Returns:
            dict: {
                'votes_normal': int,
                'votes_total': int,
                'consensus_normal': bool,   # majority vote
                'unanimous': bool,
            }
        """
        report = self.get_normality_report_full(as_dataframe=True)

        if alpha is not None:
            # Recompute votes at a different alpha where p-values are available.
            # Anderson-Darling's "p" column is numeric (interpolated), so this
            # works uniformly across all tests.
            votes = (report["p"] > alpha).sum()
        else:
            votes = report["normal"].sum()

        total = len(report)
        return {
            "votes_normal": int(votes),
            "votes_total": int(total),
            "consensus_normal": votes > total / 2,
            "unanimous": votes == total or votes == 0,
        }

    def get_distribution_shape(self):
        """
        Report skewness and kurtosis, which explain *why* a test might
        have rejected normality (e.g. heavy tail vs. asymmetry), and
        contextualize results from skew/kurtosis-based tests (D'Agostino, JB).

        Returns:
            dict: {
                'skewness': float,        # 0 for a symmetric distribution
                'kurtosis_excess': float, # 0 for normal (Fisher definition)
                'skew_interpretation': str,
                'kurtosis_interpretation': str,
            }
        """
        s = skew(self.series)
        k = kurtosis(self.series, fisher=True)  # excess kurtosis; normal = 0

        if abs(s) < 0.5:
            skew_msg = "approximately symmetric"
        elif abs(s) < 1:
            skew_msg = "moderately skewed"
        else:
            skew_msg = "highly skewed"
        skew_msg += " (right/positive)" if s > 0 else " (left/negative)" if s < 0 else ""

        if abs(k) < 0.5:
            kurt_msg = "approximately normal tail weight (mesokurtic)"
        elif k > 0:
            kurt_msg = "heavier tails than normal (leptokurtic)"
        else:
            kurt_msg = "lighter tails than normal (platykurtic)"

        return {
            "skewness": s,
            "kurtosis_excess": k,
            "skew_interpretation": skew_msg,
            "kurtosis_interpretation": kurt_msg,
        }

    def get_sensitivity(self, alphas=(0.01, 0.05, 0.10)):
        """
        Show whether each test's verdict flips across common alpha levels,
        which tells you how borderline a result is rather than just a
        binary pass/fail at alpha=0.05.

        Args:
            alphas (tuple[float]): Significance levels to check.

        Returns:
            pd.DataFrame: rows = tests, columns = alphas, values = bool verdicts.
        """
        report = self.get_normality_report_full(as_dataframe=True)
        out = pd.DataFrame(index=report.index)
        for a in alphas:
            # Anderson's "p" is numeric (interpolated); all tests now share this.
            out[f"normal_at_alpha={a}"] = report["p"] > a
        return out
    
    def get_effect_size(self):
        """
        Quantify HOW FAR the data is from normal, independent of sample
        size -- the thing a p-value alone cannot tell you (see
        get_power_note(): at large n, trivial deviations still produce
        small p-values; at small n, real deviations may not).
 
        Combines three sample-size-independent measures:
        - Shapiro-Wilk's 1-W: W is a standardized (0 to 1) measure of how
          well the data's order statistics correlate with what's expected
          under normality; W=1 is perfect, so 1-W grows with departure.
          Unlike skewness/kurtosis, this is sensitive to departures
          that DON'T show up as asymmetry or heavy tails -- e.g.
          bimodality, which a skew/kurtosis-only view can miss entirely
          (calibration: a symmetric two-cluster bimodal sample can show
          skewness near 0 while 1-W is still substantial).
        - Skewness: standardized asymmetry (0 = symmetric).
        - Excess kurtosis: standardized tail-weight deviation (0 = normal
          tail weight).
 
        None of these three scale up with n the way a p-value does, so
        they're the right tool for "is this deviation big enough to
        matter" rather than "is this deviation statistically detectable."
 
        Returns:
            dict: {
                'one_minus_w': float,         # Shapiro-Wilk 1-W
                'one_minus_w_interpretation': str,
                'skewness': float,
                'kurtosis_excess': float,
                'overall_interpretation': str,
            }
        """
        one_minus_w = 1 - self._shapiro()["stat"]
 
        # Thresholds calibrated against standard normal (~0.002), uniform
        # and t(df=3) (~0.03-0.04), moderate skew/heavy tails (~0.15-0.21),
        # and severe lognormal-style skew (~0.57).
        if one_minus_w < 0.02:
            w_msg = "negligible deviation from normal"
        elif one_minus_w < 0.05:
            w_msg = "small deviation from normal"
        elif one_minus_w < 0.15:
            w_msg = "moderate deviation from normal"
        else:
            w_msg = "large deviation from normal"
 
        shape = self.get_distribution_shape()
        s, k = shape["skewness"], shape["kurtosis_excess"]
 
        # If skew/kurtosis both look mild but 1-W says otherwise, that
        # combination itself is informative -- flag it explicitly rather
        # than silently leaving the contradiction for the user to notice.
        if one_minus_w >= 0.15 and abs(s) < 0.5 and abs(k) < 0.5:
            overall = (
                f"{w_msg}; skewness and kurtosis both look mild, which "
                f"suggests the departure is in OVERALL SHAPE rather than "
                f"simple asymmetry or tail weight (e.g. multimodality) -- "
                f"worth a look at the histogram/QQ plot."
            )
        else:
            overall = (
                f"{w_msg} (1-W={one_minus_w:.3f}); "
                f"{shape['skew_interpretation']}; "
                f"{shape['kurtosis_interpretation']}."
            )
 
        return {
            "one_minus_w": one_minus_w,
            "one_minus_w_interpretation": w_msg,
            "skewness": s,
            "kurtosis_excess": k,
            "overall_interpretation": overall,
        }

    def _shapiro(self):
        # scipy documents that for N > 5000 the W statistic remains accurate
        # but the p-value may not be -- capture that warning explicitly and
        # surface it in the result rather than letting it print to stderr
        # and get lost, especially when looping over many columns.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shapiro_stat, shapiro_p = shapiro(self.series)
            p_value_reliable = not any(
                "p-value may not be accurate" in str(w.message) for w in caught
            )

        return {
            "test_type": "Shapiro-Wilk",
            "stat": shapiro_stat,
            "p": shapiro_p,
            "normal": self._verdict(shapiro_p),
            "recommended_sample_size": "<=50",
            "p_value_reliable": p_value_reliable,
        }

    def _ks(self):
        """
        NOTE: parameters (mean, std) are estimated from the same sample
        being tested. This is statistically invalid for a true KS
        goodness-of-fit test and biases p-values upward (i.e. makes the
        data look more "normal" than it is). Provided for reference /
        comparison against 'lilliefors', which corrects for this.
        """
        ks_stat, ks_p = kstest(self.series, "norm", args=(self.mean, self.std))

        return {
            "test_type": "Kolmogorov-Smirnov",
            "stat": ks_stat,
            "p": ks_p,
            "normal": self._verdict(ks_p),
            "recommended_sample_size": ">50 (prefer 'lilliefors' instead)",
            "p_value_reliable": True,
        }

    def _anderson(self):
        # scipy.stats.anderson's `method` parameter (which returns a direct,
        # table-interpolated p-value) was only added in SciPy 1.17. On older
        # scipy, passing method='interpolate' raises TypeError ("unexpected
        # keyword argument"). To stay correct across scipy versions, try the
        # modern call first and fall back to replicating the same
        # interpolation manually against the older critical_values /
        # significance_level table if the keyword isn't supported.
        try:
            anderson_result = anderson(self.series, dist="norm", method="interpolate")
            stat = anderson_result.statistic
            p_value = anderson_result.pvalue
        except TypeError:
            with warnings.catch_warnings():
                # older scipy emits a FutureWarning here nudging users
                # toward the method= kwarg we just confirmed is unavailable;
                # suppress it since there's nothing actionable to do.
                warnings.simplefilter("ignore", FutureWarning)
                anderson_result = anderson(self.series, dist="norm")

            stat = anderson_result.statistic
            # significance_level is expressed in PERCENT (e.g. 15, 10, 5,
            # 2.5, 1), paired with critical_values. Interpolate the p-value
            # the same way scipy's own method='interpolate' does: sort by
            # critical value ascending, then linearly interpolate. This is
            # the same fix as before, just done by hand instead of asking
            # scipy to do it -- and converted to a fraction (not percent)
            # this time, matching self.alpha's units.
            sig_levels_pct = np.asarray(anderson_result.significance_level, dtype=float)
            crit_vals = np.asarray(anderson_result.critical_values, dtype=float)
            order = np.argsort(crit_vals)
            p_value = float(
                np.interp(stat, crit_vals[order], sig_levels_pct[order] / 100.0)
            )

        return {
            "test_type": "Anderson-Darling",
            "stat": stat,
            "p": p_value,
            "normal": self._verdict(p_value),
            "recommended_sample_size": ">20",
            "p_value_reliable": True,
        }

    def _dagostino(self):
        dagostino_stat, dagostino_p = normaltest(self.series)

        return {
            "test_type": "D'Agostino-Pearson",
            "stat": dagostino_stat,
            "p": dagostino_p,
            "normal": self._verdict(dagostino_p),
            "recommended_sample_size": ">50",
            "p_value_reliable": True,
        }

    def _jb(self):
        jb_stat, jb_p = jarque_bera(self.series)

        return {
            "test_type": "Jarque-Bera",
            "stat": jb_stat,
            "p": jb_p,
            "normal": self._verdict(jb_p),
            "recommended_sample_size": "large n (asymptotic test)",
            "p_value_reliable": True,
        }

    def _lilliefors(self):
        lilliefors_stat, lilliefors_p = lilliefors(self.series)

        return {
            "test_type": "Lilliefors",
            "stat": lilliefors_stat,
            "p": lilliefors_p,
            "normal": self._verdict(lilliefors_p),
            "recommended_sample_size": ">50",
            "p_value_reliable": True,
        }

    @staticmethod
    def _fmt_p(p_value, ndigits=3):
        """
        Format a p-value for on-plot display only -- this never touches
        the raw float stored in dicts/DataFrames returned by
        get_normality_report() etc., which always keep the exact value.

        Convention: print the exact value to `ndigits` decimal places
        when it's representable at that precision (down to the smallest
        nonzero value, e.g. 0.001); below that threshold, collapse to
        "<0.001" (APA-style) rather than a long/tiny decimal or
        scientific notation, since the exact magnitude rarely matters
        for an at-a-glance plot annotation once it's that small.

        Returns just the value portion (e.g. "0.948" or "<0.001") --
        callers are responsible for their own "p = " / "p-value: " /
        "p" column label, since some contexts (a table column already
        labeled "p") don't want a repeated prefix.
        """
        threshold = 10 ** (-ndigits)
        if p_value < threshold:
            return f"<{threshold:.{ndigits}f}"
        return f"{p_value:.{ndigits}f}"

    # Short, fixed-width labels for the all-tests table -- the raw
    # test_type strings ("Kolmogorov-Smirnov", "D'Agostino-Pearson") are
    # too long to keep six rows aligned and compact on a plot.
    _SHORT_TEST_NAMES = {
        "Shapiro-Wilk": "Shapiro",
        "Kolmogorov-Smirnov": "KS",
        "Anderson-Darling": "Anderson",
        "D'Agostino-Pearson": "DAgostino",
        "Jarque-Bera": "JarqueBera",
        "Lilliefors": "Lilliefors",
    }

    def _annotation_text(self, test_type):
        """
        Build the 'TestName / stat / p / verdict' text block used by
        qq_plot() and hist_with_normal_curve() when annotate=True.
        """
        result = self.get_normality_report(test_type)
        verdict = "normal" if result["normal"] else "not normal"
        lines = [
            result["test_type"],
            f"stat = {result['stat']:.4f}",
            f"p = {self._fmt_p(result['p'])}",
            f"({verdict} at \u03b1={self.alpha})",
        ]
        if not result.get("p_value_reliable", True):
            lines.append("(p-value may be unreliable)")
        return "\n".join(lines)

    def _annotation_text_all(self):
        """
        Build a compact, fixed-width table of all 6 tests' stat/p/verdict
        for qq_plot() and hist_with_normal_curve() when annotate_all=True.
        Each row: short test name, statistic, p-value (scientific notation
        for very small values), a checkmark/cross for the verdict, and a
        trailing '*' if that test's p-value may not be reliable (e.g.
        Shapiro-Wilk above n=5000) -- see the footnote line appended below
        the table when any '*' is present.
        """
        report = self.get_normality_report_full(as_dataframe=True).reset_index()

        header = f"{'Test':<11s} {'stat':>9s} {'p':>8s}"
        rows = [header, "-" * len(header)]
        any_unreliable = False

        for _, row in report.iterrows():
            short_name = self._SHORT_TEST_NAMES[row["test_type"]]
            mark = "\u2713" if row["normal"] else "\u2717"
            reliable = row.get("p_value_reliable", True)
            star = "" if reliable else "*"
            if not reliable:
                any_unreliable = True
            rows.append(
                f"{short_name:<11s} {row['stat']:>9.3f} "
                f"{self._fmt_p(row['p']):>8s} {mark}{star}"
            )

        if any_unreliable:
            rows.append("* p-value may be unreliable")

        return "\n".join(rows)

    def qq_plot(
        self,
        save_path=None,
        plot=True,
        figsize=(10, 5),
        annotate=False,
        annotate_test="shapiro",
        annotate_loc="upper left",
        annotate_all=False,
    ):
        """
        Draw a Q-Q plot of this instance's data against the normal distribution.

        Args:
            save_path (str, optional): If given, save the figure to this path.
            plot (bool): If True, display the figure; otherwise close it
                (useful for headless/batch saving).
            figsize (tuple): Figure size.
            annotate (bool): If True, overlay a text box with one chosen
                test's name, statistic, and p-value directly on the plot.
                Default False -- existing calls are unaffected. Ignored
                if annotate_all=True.
            annotate_test (str): Which test's result to display when
                annotate=True. One of self._implemented_tests (e.g.
                'shapiro', 'anderson'). Ignored if annotate=False or if
                annotate_all=True.
            annotate_loc (str): Corner to place the annotation box in.
                One of 'upper left', 'upper right', 'lower left',
                'lower right'. Ignored if neither annotate nor
                annotate_all is True.
            annotate_all (bool): If True, overlay a compact table with
                ALL 6 implemented tests' statistic, p-value, and verdict,
                instead of a single test. Takes priority over annotate/
                annotate_test if both are set. Default False.
        """
        fig = plt.figure(figsize=figsize)
        probplot(self.series, dist="norm", plot=plt)

        if annotate_all:
            self._add_annotation(plt.gca(), None, annotate_loc, all_tests=True)
        elif annotate:
            if annotate_test not in self._implemented_tests:
                raise ValueError(
                    f"annotate_test must be one of {self._implemented_tests}"
                )
            self._add_annotation(plt.gca(), annotate_test, annotate_loc)

        if save_path is not None:
            plt.savefig(save_path)

        if plot:
            plt.show()
        else:
            plt.close(fig)

    def hist_with_normal_curve(
        self,
        bins=30,
        save_path=None,
        plot=True,
        figsize=(10, 5),
        annotate=False,
        annotate_test="shapiro",
        annotate_loc="upper right",
        annotate_all=False,
    ):
        """
        Plot a histogram of the data overlaid with the fitted normal PDF.
        Often more persuasive to stakeholders than a bare p-value.

        Args:
            bins (int): Number of histogram bins.
            save_path (str, optional): If given, save the figure to this path.
            plot (bool): If True, display the figure; otherwise close it.
            figsize (tuple): Figure size.
            annotate (bool): If True, overlay a text box with one chosen
                test's name, statistic, and p-value directly on the plot.
                Default False -- existing calls are unaffected. Ignored
                if annotate_all=True.
            annotate_test (str): Which test's result to display when
                annotate=True. One of self._implemented_tests (e.g.
                'shapiro', 'anderson'). Ignored if annotate=False or if
                annotate_all=True.
            annotate_loc (str): Corner to place the annotation box in.
                One of 'upper left', 'upper right', 'lower left',
                'lower right'. Default 'upper right', since a normal-ish
                histogram peaks in the middle and leaves both top corners
                clear; switch to a 'lower' corner if your data is heavily
                skewed and the curve's peak sits near the top of a corner.
                Ignored if neither annotate nor annotate_all is True.
            annotate_all (bool): If True, overlay a compact table with
                ALL 6 implemented tests' statistic, p-value, and verdict,
                instead of a single test. Takes priority over annotate/
                annotate_test if both are set. Default False. The table
                is taller than the single-test box -- if it overlaps the
                histogram bars, try a different annotate_loc or a larger
                figsize.
        """
        from scipy.stats import norm as _norm

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.series, bins=bins, density=True, alpha=0.6, edgecolor="white")

        x = np.linspace(self.series.min(), self.series.max(), 200)
        ax.plot(x, _norm.pdf(x, self.mean, self.std), linewidth=2)
        ax.set_title("Histogram with Fitted Normal Curve")

        if annotate_all:
            self._add_annotation(ax, None, annotate_loc, all_tests=True)
        elif annotate:
            if annotate_test not in self._implemented_tests:
                raise ValueError(
                    f"annotate_test must be one of {self._implemented_tests}"
                )
            self._add_annotation(ax, annotate_test, annotate_loc)

        if save_path is not None:
            fig.savefig(save_path)

        if plot:
            plt.show()
        else:
            plt.close(fig)

    def _add_annotation(self, ax, test_type, loc, all_tests=False):
        """
        Draw the test-stat/p-value text box on the given Axes at one of
        the four corners, in axes-fraction coordinates so it stays put
        regardless of the data's actual scale.

        Args:
            test_type (str or None): Ignored if all_tests=True.
            all_tests (bool): If True, draw the compact all-6-tests table
                instead of a single test's box.
        """
        positions = {
            "upper left": dict(x=0.03, y=0.97, va="top", ha="left"),
            "upper right": dict(x=0.97, y=0.97, va="top", ha="right"),
            "lower left": dict(x=0.03, y=0.03, va="bottom", ha="left"),
            "lower right": dict(x=0.97, y=0.03, va="bottom", ha="right"),
        }
        if loc not in positions:
            raise ValueError(f"annotate_loc must be one of {list(positions.keys())}")
        pos = positions[loc]

        text = self._annotation_text_all() if all_tests else self._annotation_text(test_type)
        ax.text(
            pos["x"], pos["y"], text,
            transform=ax.transAxes,
            verticalalignment=pos["va"],
            horizontalalignment=pos["ha"],
            fontsize=9 if not all_tests else 8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    def plot_dashboard(
        self,
        show_all_tests=True,
        test_type="shapiro",
        bins=30,
        save_path=None,
        plot=True,
        figsize=(15, 5),
    ):
        """
        Draw a single combined figure: Q-Q plot, histogram with fitted
        normal curve, and a dedicated panel showing test result(s) --
        either all 6 implemented tests as a table, or one chosen test
        shown larger. This is the "one figure to share" view; qq_plot()
        and hist_with_normal_curve() remain available individually if you
        only need one of the two plots.

        Args:
            show_all_tests (bool): If True (default), the third panel is
                a table with all 6 implemented tests' statistic, p-value,
                and color-coded verdict. If False, the third panel shows
                just the one test named by `test_type`, larger and
                easier to read at a glance.
            test_type (str): Which test to show when show_all_tests=False.
                One of self._implemented_tests (e.g. 'shapiro', 'anderson').
                Ignored if show_all_tests=True.
            bins (int): Number of histogram bins (passed through to the
                histogram panel).
            save_path (str, optional): If given, save the figure to this path.
            plot (bool): If True, display the figure; otherwise close it
                (useful for headless/batch saving).
            figsize (tuple): Overall figure size. The three panels split
                this width roughly 1 : 1 : 0.7-0.9 (table vs. single-test
                panel), not evenly, since the third panel needs less
                horizontal room than the two plots.

        Returns:
            matplotlib.figure.Figure: the created figure (also useful if
            you want further customization before saving/showing it
            yourself).
        """
        from scipy.stats import norm as _norm

        if not show_all_tests and test_type not in self._implemented_tests:
            raise ValueError(
                f"test_type must be one of {self._implemented_tests}"
            )

        third_panel_width = 0.9 if show_all_tests else 0.7
        fig, axes = plt.subplots(
            1, 3, figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1, third_panel_width]},
        )

        # Panel 1: Q-Q plot
        probplot(self.series, dist="norm", plot=axes[0])
        axes[0].set_title("Q-Q Plot")

        # Panel 2: histogram with fitted normal curve
        axes[1].hist(self.series, bins=bins, density=True, alpha=0.6, edgecolor="white")
        x = np.linspace(self.series.min(), self.series.max(), 200)
        axes[1].plot(x, _norm.pdf(x, self.mean, self.std), linewidth=2)
        axes[1].set_title("Histogram with Fitted Normal Curve")

        # Panel 3: test result(s)
        axes[2].axis("off")
        if show_all_tests:
            self._draw_results_table(axes[2])
            axes[2].set_title("Test Results")
        else:
            self._draw_single_result_box(axes[2], test_type)
            axes[2].set_title("Test Result")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)

        if plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _draw_results_table(self, ax):
        """
        Render all 6 implemented tests as a real matplotlib table (not a
        text annotation -- this panel has dedicated room, so a proper
        table reads better than monospace text alignment tricks) with
        color-coded verdicts and a footnote if any p-value may be
        unreliable (e.g. Shapiro-Wilk above n=5000).
        """
        report = self.get_normality_report_full(as_dataframe=True).reset_index()

        cell_text = []
        unreliable_rows = []
        for i, row in report.iterrows():
            mark = "\u2713" if row["normal"] else "\u2717"
            if not row.get("p_value_reliable", True):
                mark += "*"
                unreliable_rows.append(i)
            cell_text.append([
                row["test_type"],
                f"{row['stat']:.4f}",
                self._fmt_p(row["p"]),
                mark,
            ])

        table = ax.table(
            cellText=cell_text,
            colLabels=["Test", "Stat", "p", "Normal"],
            loc="center", cellLoc="center",
            colWidths=[0.42, 0.2, 0.22, 0.18],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.0)

        for i, row in report.iterrows():
            cell = table[i + 1, 3]  # +1 to skip the header row
            cell.set_text_props(
                color="#1f7a3d" if row["normal"] else "#a3242f",
                weight="bold",
            )

        if unreliable_rows:
            ax.text(
                0.5, -0.06, "* p-value may be unreliable",
                transform=ax.transAxes, ha="center", fontsize=8, style="italic",
            )

    def _draw_single_result_box(self, ax, test_type):
        """
        Render one test's result as a large, centered box -- used by
        plot_dashboard() when show_all_tests=False. The verdict line is
        drawn separately so it can be color-coded (green/red) the same
        way the all-tests table's verdict column is.
        """
        from matplotlib.patches import FancyBboxPatch

        result = self.get_normality_report(test_type)
        verdict_color = "#1f7a3d" if result["normal"] else "#a3242f"
        verdict_text = "Normal" if result["normal"] else "Not normal"

        body_lines = [
            result["test_type"],
            "",
            f"Statistic: {result['stat']:.4f}",
            f"p-value: {self._fmt_p(result['p'])}",
        ]
        if not result.get("p_value_reliable", True):
            body_lines.append("(p-value may be unreliable)")

        # Background box behind everything, drawn first (zorder=0) so the
        # text sits on top of it.
        ax.add_patch(FancyBboxPatch(
            (0.08, 0.20), 0.84, 0.62, transform=ax.transAxes,
            boxstyle="round,pad=0.02", facecolor="#f4f5fb",
            edgecolor="gray", zorder=0,
        ))

        # Vertical offset for the verdict/alpha lines scales with how many
        # body lines there are, so the extra "(p-value may be unreliable)"
        # line (when present) doesn't crowd into the verdict line below it.
        body_top_y = 0.66
        line_height = 0.07
        verdict_y = body_top_y - len(body_lines) * line_height * 0.5 - 0.06

        ax.text(
            0.5, body_top_y, "\n".join(body_lines),
            transform=ax.transAxes, fontsize=13,
            ha="center", va="center", linespacing=1.8, zorder=1,
        )
        ax.text(
            0.5, verdict_y, f"Verdict: {verdict_text}",
            transform=ax.transAxes, fontsize=13, weight="bold",
            ha="center", va="center", color=verdict_color, zorder=1,
        )
        ax.text(
            0.5, verdict_y - 0.08, f"(\u03b1 = {self.alpha})",
            transform=ax.transAxes, fontsize=11,
            ha="center", va="center", color="gray", zorder=1,
        )
