# Maxstat validation contract

The maxstat implementation is pinned to these reference definitions:

- CRAN `maxstat` 0.7-26 `cmaxstat` for candidate indices, expectation,
  permutation variance, standardized absolute score process, and first-maximum
  cutoff selection;
- CRAN `maxstat` 0.7-26 `pLausen92` for the Lausen--Schumacher large-sample
  p-value;
- CRAN `exactRankTests` 0.8-37 `cscores.Surv` and `irank` for log-rank scores,
  including tied-time behavior;
- Phipson and Smyth (2010) for the non-zero Monte Carlo estimator
  `(1 + extreme_count) / (B + 1)`.

Primary sources:

- <https://cran.r-project.org/package=maxstat>
- <https://github.com/cran/maxstat/blob/master/R/maxstat.test.R>
- <https://github.com/cran/maxstat/blob/master/R/maxstat.R>
- <https://github.com/cran/exactRankTests/blob/master/R/cscores.R>
- <https://doi.org/10.2202/1544-6115.1585>

## Automated checks

`tests/test_maxstat.py` specifies:

- unique partition construction for tied values and synthetic grids;
- tied-time score parity with `exactRankTests` semantics;
- Lausen--Schumacher alpha-0.05 quantiles from the reference package's
  published reproduction of Table 2;
- independent exhaustive enumeration for a small conditional permutation
  distribution;
- the finite Monte Carlo p-value correction and uncertainty fields;
- zero-variance rejection;
- required analysis mode;
- family-specific invalid-row exclusion;
- survival of a valid log-rank result when Cox fitting fails;
- provenance mismatch rejection; and
- read-only loading of legacy Parquet artifacts.

The broader suite retains the generic multiplicity, plotting, survival input,
public API, artifact parity, and workflow regression checks.

## Interpretation boundary

Passing these checks supports parity of the implemented statistic and p-value
algorithms. It cannot establish an externally valid cutoff, correct selection
bias in a reported Cox hazard ratio/confidence interval, or justify the
exchangeability and censoring assumptions for a particular dataset.
