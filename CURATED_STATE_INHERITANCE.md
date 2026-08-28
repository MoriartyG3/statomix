# Curated-state inheritance

`Cleaner.inherit_curated_state()` supports a dataset created from another
dataset's `cleaner/.../curated_data/df.parquet`. It is intentionally separate
from replaying a Cleaner configuration on raw data.

Replaying the parent edit schemas would apply renames, removals, and category
edits twice. Curated-state inheritance instead treats the target source
dataframe as post-curation data, validates its relationship to the parent, and
materializes independent target artifacts.

## Contract

The operation:

- requires a complete parent `curated_data` group;
- maps parent curated column names to target names explicitly;
- requires every intentional value change to be declared;
- optionally aligns and validates observations by a unique row key;
- rejects missing parent columns and, in strict mode, target-only columns;
- verifies that all undeclared columns remain exactly equal to the parent;
- recomputes column and survival profiles from target values;
- preserves parent curated datatypes and survival endpoint labels;
- preserves an inherited survival-typed column that has no saved semantic
  profile, without inventing a new endpoint or survival pair;
- reapplies parent categorical recoding only to explicitly changed columns;
- validates inherited event columns as binary `0`/`1` and durations as
  numeric, non-negative values;
- writes identity edit schemas because the input is already curated;
- records parent artifact hashes and the target source-data hash in Zarr
  metadata.

It does not copy parent descriptive statistics, analysis workbooks, or plots.

## Corrected-survival example

```python
source_dataset = project.datasets["Discovery Cohort"]
target_dataset = project.datasets[
    "Validation_Cohort_Corrected_Survival"
]

inheritance = target_dataset.cleaner.inherit_curated_state(
    source_cleaner=source_dataset.cleaner,
    source_version=1,
    source_config_version=1,
    target_version=1,
    target_config_version=1,
    column_mapping={
        "OS Months": "OS_months",
        "DFS Months": "DFS_months",
        "LRC Months": "LRC_months",
    },
    changed_columns=[
        "OS Event",
        "OS_months",
        "DFS Event",
        "DFS_months",
        "LRC Event",
        "LRC_months",
    ],
    row_key="Patient ID",
    strict=True,
    apply_parent_category_edits=True,
    replace=True,
)

print(inheritance["survival_pairs"])
```

Expected endpoint labels are inherited from the parent configuration. For the
example above, the printed list should contain `OS`, `DFS`, and `LRC`. An empty
list means that the selected parent Cleaner configuration itself has no saved
survival pairs.

`replace=False` is the default. It refuses to overwrite any existing target
Cleaner artifacts. Use `replace=True` only when the target configuration is a
disposable or deliberately replaceable candidate, as in the example.

`apply_parent_category_edits=True` is also the default. This matters when a
corrected event column reintroduces the parent's pre-curation labels: only the
declared changed columns receive the inherited categorical recoding. Set it to
`False` only when those replacement values are already in the parent's final
curated representation.

After inheritance, configure the analyzer against the same exact Cleaner
version and configuration:

```python
target_dataset.configure_analyzer(
    version=1,
    config_version=1,
)

target_dataset.analyzer.create_summary_report(
    version=1,
    config_version=1,
)
```

If that analyzer configuration was already executed before inheritance, its
reports describe the previous Cleaner artifacts. Use a fresh analyzer
version/configuration, or explicitly remove only those stale candidate
analysis outputs before regenerating them. Cleaner inheritance never deletes
analysis results implicitly.

## When not to use it

If the target dataframe was derived from the parent's raw source dataframe,
reuse and apply the parent's curation configuration once instead. This API is
only for data loaded from a completed curated-data group.
