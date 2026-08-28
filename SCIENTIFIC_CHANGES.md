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

- Every candidate split receives a row with `valid_split`, `invalid_reason`,
  and structured error fields; failed candidates are no longer silently
  discarded.
- Raw Cox and log-rank p-values remain in the output.  Holm step-down adjusted
  p-values are added by default across the successful threshold family and
  drive significance-dependent markers.  `multiplicity_method="none"`
  restores exploratory raw-p-value selection.
- Plot x-axes and reference markers use the actual threshold values rather
  than dataframe row positions.
- Existing MPV metadata is retained and augmented with lifecycle status rather
  than reset during object construction.

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
