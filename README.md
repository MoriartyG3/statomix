# Statomix

Statomix is a human-in-the-loop statistical curation and analysis library for
tabular biomedical data. The package keeps the established domain-facing
`Project`, `Dataset`, `Cleaner`, `Analyzer`, and `analytics/datatypes`
organization while separating reusable curation, storage, and reporting
backends.

```python
from statomix import Project

project = Project(project_name="study")
dataset = project.add_dataset(df=source_df, dataset_name="cohort_a")

dataset.cleaner.create_col_report()
# Curate the generated workbook using the existing workflow.
dataset.cleaner.create_col_edit_schema()
```

Datasets derived from an existing Cleaner's curated dataframe can inherit its
semantic state without applying rename/removal/category transformations twice:

```python
derived.cleaner.inherit_curated_state(
    source_cleaner=parent.cleaner,
    source_version=1,
    source_config_version=1,
    column_mapping={"OS Months": "OS_months"},
    changed_columns=["OS Event", "OS_months"],
    row_key="Patient ID",
    strict=True,
)
```

See [CURATED_STATE_INHERITANCE.md](CURATED_STATE_INHERITANCE.md) for the
validation, overwrite, lineage, and analyzer-regeneration contracts.

Survival cutoff analysis now separates descriptive `ThresholdScan` output from
one global `MaximallySelectedLogRank` test. `MinimumPValue` requires an
explicit exploratory or inferential mode; generic row-wise corrections remain
available only as labeled sensitivity analyses. See
[MPV_CORRECTIONS.md](MPV_CORRECTIONS.md) for candidate rules, methods,
artifacts, and interpretation limits. The pinned reference and regression
contract is in [MAXSTAT_VALIDATION.md](MAXSTAT_VALIDATION.md).

## Package layout

- `statomix.analytics`: descriptive, normality, survival, threshold, and
  multiplicity methods organized by datatype;
- `statomix.pipelines`: Cleaner and Analyzer orchestration;
- `statomix.dataset` and `statomix.project`: user-facing composition;
- `statomix.core`: immutable contracts, errors, registries, and shared result
  models;
- `statomix.storage`: Zarr hierarchy, canonical paths, serializers, and
  atomic file writes;
- `statomix.curation`: column, categorical, and survival curation;
- `statomix.reporting`: presentation-only Excel renderers.

There are no parallel `analysis` or `workflows` namespaces. See
[NAMESPACE_MIGRATION.md](NAMESPACE_MIGRATION.md) if a notebook was written
against an earlier refactor branch.

See [ARCHITECTURE.md](ARCHITECTURE.md) for dependency rules and
[SCIENTIFIC_CHANGES.md](SCIENTIFIC_CHANGES.md) for the deliberately changed
scientific behavior.

## Development checks

```bash
uv sync --group dev
uv run black --check src tests
uv run ruff check src tests
uv run pytest -q
uv build
```

To compare artifacts produced by a reference checkout and this refactor:

```bash
uv run statomix-compare reference-output candidate-output
```

The comparator checks the artifact inventory, exact Parquet values and dtypes,
decoded PNG pixels, and semantic Excel workbook content.  Expected differences
from the scientific corrections must be reviewed against
`SCIENTIFIC_CHANGES.md`; they should not be silently accepted as refactor
noise.
