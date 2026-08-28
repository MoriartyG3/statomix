# Choosing a Normality Test — A Practical Guide

This guide accompanies `normality.py`. It explains how to decide which
normality test (or tests) to trust for a given dataset, beyond the common
but incomplete "use Shapiro for small n, something else for large n" rule,
and documents the `Normality` / `NormalityBatch` API as currently written.

---

## 0. The API at a glance

`Normality` takes its data at **construction time** — there's no separate
`set_data()` step. An instance is either fully valid or doesn't exist;
its data can never be swapped out later, which removes a whole class of
bugs where reusing one instance across multiple datasets silently returns
results for the wrong one.

```python
from normality import Normality, NormalityBatch

n = Normality(df["revenue"], alpha=0.05, ddof=1)   # alpha/ddof optional, shown for clarity

n.series   # the cleaned (NaN-dropped) pd.Series being tested
n.mean     # sample mean
n.std      # sample std (ddof as given)
n.alpha    # significance level used for all verdicts
n.ddof     # delta degrees of freedom used for std
```

Constructing with fewer than 3 valid (non-NaN) observations raises
`ValueError` immediately, rather than producing a `Normality` object that
would fail mysteriously later. NaNs are dropped automatically with a
`UserWarning` telling you how many were removed.

For many columns at once, use `NormalityBatch`, which creates one
independent `Normality` instance per column internally — see Section 9.

---

## 1. Why sample size alone isn't enough

The textbook heuristic — Shapiro-Wilk for n ≤ 50, Kolmogorov-Smirnov /
Lilliefors above that — is about **valid operating range**, not about
which test is actually best for your situation. It ignores:

- **What kind of departure from normality you're worried about** (skew?
  heavy tails? multimodality?)
- **How much statistical power you actually have** at your n
- **Whether your data violates a test's assumptions** in ways that don't
  show up in the p-value (ties, a few outliers)
- **What you're going to do with the result** — different downstream
  procedures tolerate different kinds of non-normality differently

A complete decision process weighs all of these. The sections below walk
through each one.

---

## 2. The six tests, side by side

| Test | What it's built from | Most sensitive to | Typical valid range | Caveats |
|---|---|---|---|---|
| **Shapiro-Wilk** | Correlation between data and expected normal order statistics | Overall shape | Small to large n — one of the most powerful all-purpose tests at almost any n | The "n≤50 only" rule is largely a legacy of older, computationally limited implementations, not a real statistical ceiling. Above n=5000, scipy documents that its p-value specifically (not the statistic) may not be accurate — see Section 10. |
| **Kolmogorov-Smirnov (KS)** | Max distance between empirical and theoretical CDF | Deviations near the *center* of the distribution | Any n, **only if parameters are known in advance** | If mean/std are estimated from the same sample (the usual case), the test is invalid — p-values are biased upward (too lenient). Use **Lilliefors** instead. |
| **Lilliefors** | KS statistic, with a correction for estimated parameters | Same as KS (center-weighted), corrected | n > ~50, the practically valid version of KS for estimated parameters | Still less sensitive to tail deviations than Anderson-Darling |
| **Anderson-Darling** | Like KS, but weights tails more heavily | Tail / extreme-value behavior | n > ~20 | Best choice when downstream use cares about extremes (risk models, control limits) |
| **D'Agostino-Pearson** | Sample skewness + kurtosis (3rd/4th moments) | Asymmetry and tail weight specifically | n > ~50 | Can *miss* shape problems that don't show up in skew/kurtosis (e.g. a symmetric bimodal distribution can pass). Moment-based ⇒ sensitive to a few extreme points. |
| **Jarque-Bera** | Also skewness + kurtosis, asymptotic | Same blind spots as D'Agostino | Large n (asymptotic test) | Common in econometrics; same outlier sensitivity as D'Agostino |

**Rule of thumb:** if you only run one test and have no special concerns,
Shapiro-Wilk is a reasonable default at almost any sample size up to
about 5000 observations. Reach for the others when you have a specific
reason (see below).

---

## 3. What are you actually worried about?

This is the single biggest thing a sample-size rule misses. Pick the test
that's sensitive to the *kind* of deviation that matters for your use case.
`recommend_test_for_purpose(purpose)` encodes this directly:

```python
n.recommend_test_for_purpose("general")          # -> shapiro (or lilliefors above n=5000)
n.recommend_test_for_purpose("tails")            # -> anderson
n.recommend_test_for_purpose("parametric_test")  # -> shapiro
n.recommend_test_for_purpose("symmetry_only")    # -> dagostino
```

- **General-purpose check, no special concern** → Shapiro-Wilk.
- **Tail / extreme-value behavior matters** (you're about to compute a
  parametric VaR, set control-chart limits, or run a process capability
  index) → **Anderson-Darling**. It's built to weight the tails, where KS
  and even Shapiro can be comparatively less sensitive.
- **About to run a t-test, ANOVA, or linear regression** → Shapiro-Wilk is
  the conventional pre-check. But remember: these procedures are fairly
  **robust to mild non-normality at moderate-to-large n** thanks to the
  Central Limit Theorem. A "reject" verdict doesn't automatically mean
  your downstream test will misbehave — check the *effect size* (how far
  off is the skew/kurtosis, really?) before abandoning a parametric
  approach.
- **Only care about symmetry** (e.g., deciding whether to report the mean
  or median as a summary statistic) → **D'Agostino-Pearson**, or simply
  look at the skewness number directly via `get_distribution_shape()`.
  Don't worry about multimodality or other shape issues these tests are
  not built to catch.
- **You suspect multimodality or another shape oddity that skew/kurtosis
  won't catch** → Don't lean on D'Agostino/Jarque-Bera here; use
  Shapiro-Wilk or Anderson-Darling, and **look at a histogram**.

If your data has heavy ties, `recommend_test_for_purpose()` automatically
appends a caution to its rationale when it would otherwise recommend a
KS-family test (see Section 5).

---

## 4. Sample size vs. statistical power (they're not the same thing)

A test being "valid" at a given n doesn't mean it has good power there.
`get_power_note()` gives a quick read on this:

- **Small n (< ~20–30):** Every normality test has limited power.
  "Failed to reject normality" is **weak evidence**, not proof — you
  simply don't have enough data to detect anything but a gross departure.
  Lean more on visual diagnostics (QQ plot) than the p-value here.
- **Moderate n (~30–300):** The "sweet spot" where most tests have
  reasonable power without being oversensitive to trivial deviations.
- **Large n (> ~300–500):** Tests become powerful enough to flag *any*
  real-world deviation — including ones that don't matter in practice
  (e.g. rounding error, slight measurement granularity). A statistically
  significant "non-normal" result at n=5,000 may be **practically
  irrelevant**. At this scale, check the magnitude of skew/kurtosis and
  look at the QQ plot rather than trusting the p-value alone — and ask
  whether your downstream procedure actually needs strict normality, or
  is robust enough (CLT) not to care.

---

## 5. Two assumption violations that quietly break a test

### a) Ties (repeated values)

KS-family tests (KS, Lilliefors) assume the underlying distribution is
continuous, with effectively zero probability of exact ties. Real data
often violates this — rounded measurements, Likert/survey scales, sensor
quantization, currency amounts. Heavy ties inflate the KS-family
statistic in ways the reported p-value doesn't correct for, making the
test unreliable (typically *too eager to reject*).

**Check:**
```python
n.get_tie_diagnostics()
# {'n_unique': ..., 'n_total': ..., 'tie_fraction': ..., 'ks_family_reliable': bool}
```
If `ks_family_reliable` is `False` (tie fraction above 10% by default),
**prefer Shapiro-Wilk, Anderson-Darling, D'Agostino, or Jarque-Bera**
over KS/Lilliefors — they tolerate ties much better.

### b) Outliers

Moment-based tests (Jarque-Bera, D'Agostino-Pearson) are built from the
3rd and 4th sample moments (skewness, kurtosis), which are **extremely**
sensitive to a small number of extreme values. A handful of bad data
points or genuine extreme observations can single-handedly cause these
tests to reject normality, even when the bulk of the data looks fine.

**Check:**
```python
n.get_outlier_diagnostics(method="iqr")      # Tukey's 1.5*IQR fences
n.get_outlier_diagnostics(method="zscore")   # |z| > 3 by default
```
If a few points are driving the result, decide whether they're data
errors (fix/remove) or real signal (then the rejection may be legitimate
— your data really does have heavy tails).

---

## 6. Visual diagnostics: do this regardless of which test you pick

A p-value tells you "reject" or "fail to reject" — it doesn't tell you
**where** or **how much** a distribution deviates from normal. Two
distributions can both "fail" Shapiro-Wilk for completely different
reasons (one skewed, one bimodal, one just has fat tails), and the fix
for each is different.

```python
n.qq_plot()                  # points hugging the diagonal = good fit
n.hist_with_normal_curve()   # histogram with fitted normal PDF overlay
```

- **QQ plot**: points hugging the diagonal line = good fit. Curvature at
  the ends = tail problems (heavy or light tails). An S-curve = skew.
  Jumps or clusters = multimodality or ties.
- **Histogram with fitted normal curve overlay**: often the most
  intuitive for a non-technical audience, and quickly reveals
  multimodality that skew/kurtosis-based tests might miss entirely.

Use these together with, not instead of, the numeric tests.

---

## 7. A practical workflow

1. **Look first.** `n.qq_plot()` and `n.hist_with_normal_curve()`. Form a
   rough hypothesis about what (if anything) looks off — skew, heavy
   tails, multimodality, a few outliers.
2. **Check for ties and outliers.** `n.get_tie_diagnostics()` and
   `n.get_outlier_diagnostics()` — these can silently invalidate or
   distort specific tests before you even look at p-values.
3. **Pick a primary test based on what you're worried about**
   (`n.recommend_test_for_purpose(purpose)`), not just sample size.
4. **Run the full battery anyway** for context (`n.get_normality_report_full()`,
   `n.get_consensus()`) — if 5 of 6 tests agree, that's stronger evidence
   than any single test; if they sharply disagree, that disagreement
   itself is informative (often points to tails or outliers driving a
   moment-based test, or to ties distorting a KS-family test).
5. **Contextualize against sample size** (`n.get_power_note()`). At small
   n, treat a "pass" as weak evidence. At large n, treat a "fail" with
   skepticism unless the effect size (`n.get_distribution_shape()`,
   visual deviation) is also meaningful.
6. **Ask what you're using this for.** If the next step is a t-test or
   ANOVA, remember these are fairly robust to mild non-normality at
   reasonable sample sizes — a statistically significant rejection isn't
   automatically a practical problem.

Or get everything from steps 2, 4, and 5 in one call:
```python
n.get_full_diagnostics()   # {'power': ..., 'ties': ..., 'outliers': ..., 'shape': ...}
```

---

## 8. Mapping this guide to `normality.py`

| Guide section | Method |
|---|---|
| Run all six tests | `get_normality_report_full()` |
| Run one specific test | `get_normality_report(test_type)` |
| Single best test based on n | `get_normality_report_default()` |
| Majority vote across tests | `get_consensus()` |
| Skewness / kurtosis + interpretation | `get_distribution_shape()` |
| Does the verdict flip across alpha levels? | `get_sensitivity()` |
| Tie fraction / is KS-family reliable? | `get_tie_diagnostics()` |
| Outlier scan (IQR or z-score) | `get_outlier_diagnostics()` |
| Sample-size vs. power context | `get_power_note()` |
| Purpose-driven test recommendation | `recommend_test_for_purpose(purpose)` |
| Everything above, in one call | `get_full_diagnostics()` |
| Visual check: QQ plot | `qq_plot()` |
| Visual check: histogram + normal curve | `hist_with_normal_curve()` |

---

## 9. A correctness note on Shapiro-Wilk at large n

scipy's Shapiro-Wilk implementation documents that **above n=5000, the W
statistic remains accurate but the p-value may not be**. This module
checks for that condition automatically:

- `get_normality_report('shapiro')` returns a `p_value_reliable` flag —
  `False` whenever scipy raises the underlying accuracy warning.
- `get_normality_report_default()` and `recommend_test_for_purpose()`
  automatically switch to **Lilliefors** above n=5000, rather than
  silently returning a Shapiro-Wilk p-value that may not be trustworthy.

This matters for batch processing in particular: across many columns
with varying row counts, some may cross the 5000-row threshold while
others don't. `NormalityBatch.summary()` reflects the correct per-column
routing automatically.

---

## 10. A portability note on Anderson-Darling across scipy versions

`scipy.stats.anderson` gained an explicit `method` parameter (which
returns a direct, table-interpolated p-value) in **SciPy 1.17**. On older
scipy installs, passing it raises `TypeError`. `_anderson()` in this
module handles both cases transparently:

- On scipy ≥ 1.17, it uses `method="interpolate"` directly.
- On older scipy, it falls back to the legacy call (`critical_values` /
  `significance_level` arrays) and reproduces the same linear
  interpolation by hand, converting `significance_level` from percent to
  a fraction so it's directly comparable to `alpha`.

Both paths were verified to produce identical statistics and p-values on
the same data. You don't need to do anything differently based on your
scipy version — `get_normality_report('anderson')` behaves the same
either way, including the `p_value_reliable` field (always `True` for
this test, since both code paths return a usable p-value).

---

## 11. Quick reference: decision flowchart (text form)

```
Start
  │
  ├─ Plot histogram + QQ plot. Anything obviously off? Note it.
  │
  ├─ High tie fraction (rounded/discrete data)?
  │     Yes → avoid KS / Lilliefors; use Shapiro, Anderson, JB, or D'Agostino
  │     No  → KS-family is fine to use
  │
  ├─ Outliers present?
  │     Yes → check if they're errors; be cautious trusting JB/D'Agostino alone
  │     No  → moment-based tests are safe to trust
  │
  ├─ What matters most?
  │     Overall fit, no special concern → Shapiro-Wilk
  │     Tail behavior specifically       → Anderson-Darling
  │     Symmetry only                    → D'Agostino-Pearson / skewness
  │     Pre-check for t-test/ANOVA        → Shapiro-Wilk (+ remember CLT robustness)
  │
  ├─ Sample size?
  │     n < 20      → low power; treat "pass" as weak evidence; lean on visuals
  │     20–300      → reasonable power; trust the test(s) above
  │     300–5000    → very high power; treat "fail" with skepticism unless
  │                   effect size (skew/kurtosis, QQ deviation) is also meaningful
  │     n > 5000    → as above, AND prefer Lilliefors over Shapiro-Wilk
  │                   (p-value reliability boundary)
  │
  └─ Decide, with the caveats above in mind — not from a single p-value.
```

