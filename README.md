# Statomix

Statomix is a human-in-the-loop statistical curation and analysis library for
tabular biomedical data.  This branch keeps the established `Project`,
`Dataset`, `Cleaner`, and `Analyzer` interaction while separating domain
logic from persistence and report rendering.

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

Minimum-p-value scans use raw p-values by default and can calculate several
corrections in one run. See [MPV_CORRECTIONS.md](MPV_CORRECTIONS.md) for the
configuration, generated plots, and statistical interpretation.

## Package layout

- `statomix.core`: immutable contracts, errors, registries, and version
  selection;
- `statomix.storage`: Zarr hierarchy, canonical paths, serializers, and
  atomic file writes;
- `statomix.curation`: column, categorical, and survival curation;
- `statomix.analysis`: descriptive, normality, survival, and multiplicity
  methods;
- `statomix.reporting`: presentation-only Excel renderers;
- `statomix.workflows`: project/dataset orchestration;
- legacy `analytics`, `pipelines`, `dataset`, and `project` modules remain as
  compatibility facades.

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
