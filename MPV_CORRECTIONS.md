# Minimum-p-value corrections

`MinimumPValue` is an exploratory scan over candidate survival cutoffs. Raw
p-values are stored and plotted by default. Multiplicity correction is opt-in
because adjusted values answer a different inferential question and no single
method is appropriate for every analysis.

## Configuration

Request one method with a string or several methods with a sequence:

```python
mpv = MinimumPValue(
    surv_label="OS",
    surv_df_mpv=survival_frame,
    root_group=root_group,
    correction_methods=["holm", "fdr_bh", "fdr_by"],
    selection_method="fdr_bh",
)
mpv.create_mpv_data(replace=True)
```

Omitting both options uses raw p-values:

```python
mpv = MinimumPValue(
    surv_label="OS",
    surv_df_mpv=survival_frame,
    root_group=root_group,
)
```

Supported names are:

| Name | Error criterion | Dependence guidance |
| --- | --- | --- |
| `none` | None | Exploratory raw p-values |
| `bonferroni` | FWER | Valid under arbitrary dependence |
| `holm` | FWER | Valid under arbitrary dependence |
| `holm_sidak` | FWER | Independence or suitable dependence |
| `hochberg` | FWER | Independence or suitable positive dependence |
| `fdr_bh` | FDR | Independence or positive regression dependence |
| `fdr_by` | FDR | Valid under arbitrary dependence |

Adjacent threshold tests are strongly dependent. Holm or Bonferroni provide
conservative family-wise error control without a dependence assumption;
Benjamini-Yekutieli provides dependence-robust false-discovery-rate control.
Benjamini-Hochberg is often less conservative, but its dependence assumptions
must be justified. These corrections do not turn a data-selected cutoff into
an externally validated clinical cutoff.

## Results and plots

Cox-PH and log-rank p-values are corrected as separate families, using each
column's own finite test count. The Parquet/CSV output retains the raw columns
and adds columns such as:

```text
cox_ph.p_value_holm
cox_ph.p_value_fdr_bh
log_rank.p_value_holm
log_rank.p_value_fdr_bh
```

The output also records `cox_ph.multiplicity.n_tests`,
`log_rank.multiplicity.n_tests`, `multiplicity.methods`, and
`multiplicity.selection_method`.

Every run writes:

- `plot_p_values_none.png` for the uncorrected view;
- `plot_p_values_<method>.png` for each requested correction;
- `plot_p_values_all_corrections.png` for a two-panel comparison of every
  configured method;
- the existing dashboard and diagnostic plots, whose threshold markers use
  `selection_method`.

Calling `plot_p_values(correction="holm")` selects a configured view at
runtime. Existing immutable MPV artifacts must be regenerated with
`replace=True` before requesting corrections that were not calculated in the
stored table.

The former `multiplicity_method="holm"` argument still maps to a one-method
configuration for compatibility, but emits `DeprecationWarning`. New code
should use `correction_methods` and `selection_method` explicitly.
