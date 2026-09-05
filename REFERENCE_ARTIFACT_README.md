# Statomix reference-artifact overlay

This overlay targets Statomix `development` commit `4aee89d` and adds:

- persistent `analysis` and `reference` dataset roles;
- direct immutable artifacts for explicitly declared reference columns;
- optional renaming while selecting reference columns;
- explicit typed survival-event encoding;
- explicit duration units, endpoint bindings, definitions and declarations;
- source-Parquet checksum linkage;
- Analyzer guards for reference-only datasets;
- exclusion of reference datasets from project analysis workbooks;
- compatibility with existing keyed column updates;
- regression tests for persistence, validation, immutability and keyed updates.

The overlay never edits a project store. It changes package source and tests only.

## 1. Preflight

Run from the Statomix repository root before extraction:

```bash
reference_zip="/path/to/statomix_reference_artifact_overlay.zip"

unzip -p "$reference_zip" \
    check_reference_artifact_overlay.py |
    python - "$reference_zip"
```

Extract only after the checker passes:

```bash
unzip -o "$reference_zip"
```

## 2. Format and test

```bash
uv run --active black \
    src/statomix/core/artifacts.py \
    src/statomix/dataset/__init__.py \
    src/statomix/dataset/base.py \
    src/statomix/dataset/dataset.py \
    src/statomix/dataset/roles.py \
    src/statomix/pipelines/analyzer/analyzer.py \
    src/statomix/pipelines/reference/__init__.py \
    src/statomix/pipelines/reference/reference.py \
    src/statomix/project/project.py \
    src/statomix/reporting/excel/project_analysis_config.py \
    src/statomix/storage/artifacts.py \
    tests/test_reference_artifacts.py

uv run --active ruff check src tests

uv run --active pytest -q \
    tests/test_reference_artifacts.py \
    tests/test_keyed_updates.py \
    tests/test_transformer.py \
    tests/test_row_exclusion.py

uv run --active pytest -q
uv build

git diff --check
git status --short
git diff --stat
```

## 3. Register OCAT-900 as reference-only

```python
ocat_900_dataset = project.add_dataset(
    df=df_ocat_900_survival,
    dataset_name="ocat_900_survival",
    display_label="OCAT 900 authoritative survival",
    dataset_role="reference",
)
```

After reopening the project:

```python
ocat_900_dataset = project.datasets["ocat_900_survival"]
assert ocat_900_dataset.dataset_role == "reference"
```

## 4. Create the direct reference artifact

This preserves the authoritative source names. The raw 900-row source Parquet
is not modified.

```python
from statomix.transformation import MONTHS


endpoint_definitions = {
    "OS": (
        "Overall survival event as recorded; elapsed time "
        "from randomisation."
    ),
    "DFS": (
        "Disease-free survival event as recorded; elapsed time "
        "from randomisation."
    ),
    "LRC": (
        "Locoregional control event as recorded; elapsed time "
        "from randomisation."
    ),
}

ocat_900_artifact = ocat_900_dataset.create_reference_artifact(
    version=1,
    config_version=1,
    identifier="Patient ID",
    column_mapping={
        "Patient ID": "Patient ID",
        "OS Event": "OS Event",
        "OS_months": "OS_months",
        "DFS Event": "DFS Event",
        "DFS_months": "DFS_months",
        "LRC Event": "LRC Event",
        "LRC_months": "LRC_months",
    },
    event_columns={
        "OS Event": {0: False, 1: True},
        "DFS Event": {0: False, 1: True},
        "LRC Event": {0: False, 1: True},
    },
    duration_units={
        "OS_months": MONTHS,
        "DFS_months": MONTHS,
        "LRC_months": MONTHS,
    },
    endpoints={
        "OS": {
            "event": "OS Event",
            "duration": "OS_months",
            "definition": endpoint_definitions["OS"],
        },
        "DFS": {
            "event": "DFS Event",
            "duration": "DFS_months",
            "definition": endpoint_definitions["DFS"],
        },
        "LRC": {
            "event": "LRC Event",
            "duration": "LRC_months",
            "definition": endpoint_definitions["LRC"],
        },
    },
    reason=(
        "Import the reviewed OCAT-900 survival fields as the "
        "authoritative update source for the OCAT subset."
    ),
    name="authoritative_ocat_survival",
)
```

To rename during import, change only the output side of `column_mapping` and
use those output names in `duration_units` and `endpoints`. For example:

```python
column_mapping={
    "OS Duration": "OS_months",
}
```

## 5. Apply the existing keyed update

If the reference artifact preserves its source names:

```python
ocat_corrected_artifact = ocat_dataset.transformer.create_keyed_update_data(
    base=ocat_base_artifact,
    updates=ocat_900_artifact,
    base_key="Patient ID",
    update_key="Patient ID",
    column_mapping={
        "OS Event": "OS Event",
        "OS Duration": "OS_months",
        "DFS Event": "DFS Event",
        "DFS Duration": "DFS_months",
        "LRC Event": "LRC Event",
        "LRC Duration": "LRC_months",
    },
    endpoint_mapping={
        "OS": "OS",
        "DFS": "DFS",
        "LRC": "LRC",
    },
    version=1,
    config_version=1,
    reason=(
        "Replace the OCAT survival values with the reviewed "
        "authoritative OCAT-900 survival source."
    ),
    name="corrected_ocat_survival",
)
```

Use an unused Transformer version/configuration in the real project.

## 6. Commit only intended repository files

`REFERENCE_ARTIFACT_DELIVERY.json` and
`check_reference_artifact_overlay.py` are delivery helpers and normally should
not be committed. `REFERENCE_ARTIFACT_README.md` may be committed if desired.
