# Survival cutoff scans and maxstat inference

Statomix 0.3 separates two questions that were previously mixed together:

1. `ThresholdScan` describes what happens across candidate cutoffs.
2. `MaximallySelectedLogRank` provides one global p-value for a cutoff selected
   by maximizing a standardized log-rank statistic.

`MinimumPValue` remains the artifact/plot orchestrator, but it now requires an
explicit `mode="exploratory"` or `mode="inferential"` choice.

## Candidate partitions

The unit of analysis is a unique patient partition, not a formatted cutoff.
Tied predictor values and several synthetic grid points can produce the same
`<=`/`>` allocation. Such duplicates are collapsed to one canonical observed
threshold. Candidate group sizes are constrained with `minprop` and `maxprop`.

The implementation follows the CRAN `maxstat` indexing convention:
`floor(n * minprop) <= lower_n <= floor(n * maxprop)`, with both groups
non-empty. In a small sample, the realized lower-group proportion can be less
than `minprop` by less than one observation.

The former `trunc_pct` and `iqr_multiplier` candidate filters are legacy-only
and rejected for schema-2 scans. `use_synthetic_cutoffs=True` remains available
in exploratory mode, but duplicate partitions are still removed. Inferential
mode uses all eligible observed partitions.

## Exploratory mode

Use exploratory mode to inspect raw cutoff-specific results or registered
row-wise sensitivity corrections:

```python
mpv = MinimumPValue(
    surv_label="OS",
    surv_df_mpv=survival_frame,
    root_group=root_group,
    mode="exploratory",
    minprop=0.10,
    maxprop=0.90,
    correction_methods=["holm", "fdr_by"],
    selection_method="holm",
    selection_family="log_rank",
)
mpv.create_mpv_data(replace=True)
```

Raw, adjusted, and Cox-Wald outputs are visibly labeled as exploratory or
post-selection descriptive. A row-wise correction does not turn the selected
cutoff into a validated prognostic rule and is not the package's primary
maxstat inference.

## Inferential mode

The fast large-sample route uses the Lausen--Schumacher (1992) Brownian-bridge
approximation, matching `maxstat::pLausen92`:

```python
mpv = MinimumPValue(
    surv_label="OS",
    surv_df_mpv=survival_frame,
    root_group=root_group,
    mode="inferential",
    minprop=0.10,
    maxprop=0.90,
    maxstat_method="lausen_1992",
)
mpv.create_mpv_data(replace=True)

result = mpv.maxstat_result
print(result.optimal_threshold, result.statistic, result.p_value)
```

The conditional Monte Carlo alternative permutes the fixed survival scores
relative to the ordered predictor and evaluates the maximum statistic across
the same candidate partitions:

```python
mpv = MinimumPValue(
    surv_label="OS",
    surv_df_mpv=survival_frame,
    root_group=root_group,
    mode="inferential",
    maxstat_method="conditional_monte_carlo",
    n_permutations=99_999,
    random_state=20260831,
)
mpv.create_mpv_data(replace=True)
```

The simulated p-value is `(1 + extreme_count) / (B + 1)`, so it is never zero.
The result includes a Monte Carlo standard error and a corrected
Clopper--Pearson interval. This procedure depends on exchangeability under the
global null; it is not assumption-free. Only explicit exhaustive enumeration
for a sufficiently small sample is labeled exact.

The inference engine can also be used without artifact orchestration:

```python
from statomix.analytics.datatypes.survival import MaximallySelectedLogRank

analysis = MaximallySelectedLogRank(
    predictor=frame["biomarker"],
    time=frame["time"],
    event=frame["event"],
    minprop=0.10,
    maxprop=0.90,
)
result = analysis.fit(method="lausen_1992")
process = analysis.process_df
```

## Generic row-wise corrections

The generic multiplicity registry is unchanged. These methods remain
available for explicitly exploratory sensitivity analyses:

| Name | Error criterion | Dependence guidance |
| --- | --- | --- |
| `none` | None | Raw exploratory p-values |
| `bonferroni` | FWER | Valid under arbitrary dependence |
| `holm` | FWER | Valid under arbitrary dependence |
| `holm_sidak` | FWER | Independence or suitable dependence |
| `hochberg` | FWER | Independence or suitable positive dependence |
| `fdr_bh` | FDR | Independence or positive regression dependence |
| `fdr_by` | FDR | Valid under arbitrary dependence |

Cox-Wald and log-rank values are separate families. Each family uses its own
explicit eligibility mask, and invalid/failed rows remain `NaN` in every
adjusted column. Family-specific markers prevent a Cox-selected cutoff from
being presented as a log-rank-selected cutoff.

## Artifacts and compatibility

Schema-2 artifacts are stored under a path containing a SHA-256 fingerprint of
the analyzed rows and all scan settings, including mode, `minprop`/`maxprop`,
synthetic-grid settings, alpha, row-wise correction settings, maxstat method,
permutation count, seed, batch size, and confidence level. Every folder
contains `scan_config.json`.
Inferential folders also contain:

- `maxstat_result.json` and `maxstat_result.parquet`;
- `maxstat_process.parquet` and `.csv`;
- `plot_maxstat_process.png`.

Legacy MPV paths are never rewritten by a schema-2 scan. Use
`MinimumPValue.load_legacy_artifact(path)` for a read-only table load.

## Interpretation limits

The maxstat p-value tests a global association while accounting for the cutoff
search. It does not validate the cutoff in external data. It also does not
remove selection bias from the hazard ratio or its ordinary Cox confidence
interval at the selected cutoff. Those estimates are retained as labeled,
secondary descriptions and require independent validation or a dedicated
post-selection method for confirmatory interpretation.
