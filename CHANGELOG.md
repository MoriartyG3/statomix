# Changelog

## 0.3.0

- Restore the canonical `analytics`, `pipelines`, `dataset`, and `project`
  implementation hierarchy while retaining the separated `curation`,
  `storage`, `core`, and `reporting` backends.
- Remove the unreleased duplicate `analysis` and `workflows` namespaces.
- Split descriptive threshold construction from maximally selected log-rank
  inference.
- Require explicit exploratory or inferential cutoff-analysis mode.
- Add CRAN-compatible log-rank score process and Lausen--Schumacher (1992)
  global p-value.
- Add conditional Monte Carlo inference with finite p-value correction,
  uncertainty reporting, reproducible seeds, and an exhaustive small-sample
  validation route.
- Deduplicate candidate patient partitions and replace threshold IQR/trimming
  with explicit `minprop`/`maxprop` bounds.
- Preserve the generic multiplicity registry as an exploratory sensitivity
  layer with family-specific eligibility.
- Keep valid log-rank results when Cox fitting fails; persist log-rank test
  statistics and family-specific markers.
- Add schema-2 fingerprinted artifacts and read-only legacy loading.
- Label Cox hazard ratios, confidence intervals, and per-cutoff p-values as
  post-selection descriptive outputs.
