# Scientific and behavioral changes

Structural refactors are intended to preserve established artifacts.  The
items below are deliberate corrections and can change scientific results or
metadata; they are therefore isolated and documented rather than treated as
incidental formatting changes.

## Normality diagnostics

- The raw one-sample Kolmogorov-Smirnov result fitted with the sample mean and
  standard deviation is retained for compatibility but marked
  `p_value_reliable=False`.  It is excluded from consensus because ordinary KS
  calibration assumes parameters known independently of the tested sample.
- Lilliefors is preferred when a KS-family diagnostic with estimated normal
  parameters is needed.
- Shapiro-Wilk p-values are treated as reliable only through the documented
  5,000-observation range; the recommendation falls back for larger samples.
- Inputs now reject non-finite or constant samples and invalid `alpha`/`ddof`
  values explicitly.

## Survival analyses

- Required rows are selected consistently, incomplete rows are counted and
  dropped, durations must be finite and non-negative, and event indicators
  must be boolean or exact 0/1 values.
- P-value labels compare the unrounded p-value with thresholds.  A value such
  as `0.004` can no longer be mislabeled `P<0.001` because of premature
  rounding.
- Hazard-ratio interpretation rejects non-finite and non-positive ratios.

## Minimum-p-value threshold search

- Cutoff analysis is separated into a descriptive `ThresholdScan` and an
  inferential `MaximallySelectedLogRank`. The orchestration API requires the
  caller to choose `mode="exploratory"` or `mode="inferential"` explicitly.
- Candidates are unique patient partitions constrained by `minprop` and
  `maxprop`. Tied values and synthetic grid points that produce the same split
  are evaluated once. Legacy IQR and percent-trim candidate filters are not
  accepted for schema-2 scans.
- Inferential mode reports one global maximally selected log-rank p-value. The
  fast method is the Lausen--Schumacher (1992) Brownian-bridge approximation;
  the alternative is conditional Monte Carlo with the corrected
  `(1 + extreme_count) / (B + 1)` estimator, Monte Carlo uncertainty, an
  explicit seed, and stated exchangeability assumptions.
- Cox-Wald p-values, hazard ratios, and ordinary confidence intervals remain
  descriptive after cutoff selection. No Cox analogue of the log-rank maxstat
  p-value is reported. Outputs and plots carry visible post-selection labels.
- The generic correction registry is unchanged. Optional Bonferroni, Holm,
  Holm-Sidak, Hochberg, Benjamini-Hochberg, and Benjamini-Yekutieli columns are
  retained as row-wise exploratory sensitivity analyses, not as substitutes
  for maxstat inference.
- Cox and log-rank rows now have separate eligibility fields. Invalid or failed
  rows are excluded consistently from every correction and plot. A Cox fit
  failure no longer suppresses a valid log-rank result, and the log-rank test
  statistic is persisted alongside its p-value.
- Markers are family-specific. A Cox-Wald minimum is not drawn or serialized as
  if it were the log-rank optimum. Inferential selection always uses the
  log-rank maxstat maximum.
- Schema-2 artifacts record an input-row hash and all scan settings in
  `scan_config.json`, then use their SHA-256 configuration fingerprint in the
  storage path. Settings include alpha, mode, candidate bounds, grid
  configuration, corrections, maxstat method, permutation count, seed, batch
  size, and uncertainty level.
- Legacy MPV artifacts are not rewritten. They can be loaded read-only through
  `MinimumPValue.load_legacy_artifact()` while new results use versioned paths.
- The scientific limitation is explicit: the global p-value accounts for the
  cutoff search but does not externally validate the chosen cutoff or remove
  selection bias from its hazard ratio/confidence interval.

## Workflow behavior

- `Cleaner.inherit_curated_state()` is a new opt-in path for data derived from
  a completed parent Cleaner group. It preserves curated datatypes and
  survival labels, recomputes target-dependent profiles, and applies parent
  category recoding only to explicitly changed columns. It requires inherited
  event values to be binary `0`/`1`, durations to be finite and non-negative,
  and each inherited endpoint to contain at least one complete row. Existing
  Cleaner calls are unchanged.
- Requested analysis-configuration versions are honored or rejected with a
  precise version-selection error; they are no longer silently ignored.
- Project configuration generation passes the requested dataset version and
  config version through to every dataset analyzer.
- An existing summary workbook no longer prevents a missing survival report
  from being repaired.
- Read-like version lookup does not create Zarr groups as a side effect.
- Empty and missing datatype branches retain their explicit no-op contracts.

These corrections should be reviewed as expected deltas in a before/after
artifact comparison.  All other differences are regressions until explained.
