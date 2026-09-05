# Statomix typed row-exclusion overlay

This overlay adds a reusable `ExcludeRows` operation to the immutable
Transformer pipeline. It selects complete values from a column curated as an
`Identifier`; it never accepts a pandas query, expression, callback, or row
number as the selection instruction.

The operation creates a new child artifact. It does not modify the Cleaner
artifact, the months artifact, survival-event values, endpoint bindings, units,
categorical ranks, or prior Analyzer reports.

## 1. Inspect the repository before extraction

Run from the Statomix repository root:

```bash
cd /workspace/MoriartyG3/python_packages/statomix

git branch --show-current
git status --short
```

The expected branch is `development`. Existing unrelated changes may remain,
but stop if any of the files named by the preflight checker were modified.

Set the archive path:

```bash
row_exclusion_zip="/path/to/statomix_row_exclusion_overlay.zip"
```

Run the read-only compatibility check:

```bash
unzip -p "$row_exclusion_zip" \
    check_row_exclusion_overlay.py | \
    python - "$row_exclusion_zip"
```

Only continue when it prints both `PASS` lines.

## 2. Extract and validate

```bash
unzip -o "$row_exclusion_zip"
```

Format and statically check the supplied files:

```bash
uv run --active black \
    src/statomix/transformation/specifications.py \
    src/statomix/transformation/rows.py \
    src/statomix/transformation/operations.py \
    src/statomix/transformation/__init__.py \
    src/statomix/pipelines/transformer/transformer.py \
    src/statomix/storage/artifacts.py \
    src/statomix/reporting/excel/transformation.py \
    tests/test_row_exclusion.py

uv run --active ruff check src tests

uv run --active pytest -q \
    tests/test_row_exclusion.py \
    tests/test_transformer.py

uv run --active pytest -q
uv build

git diff --check
git status --short
git diff --stat
```

## 3. Create the reviewed N0 child artifact

Restart the notebook kernel after installing the overlay, then reopen the
project:

```python
from statomix import Project
from statomix.storage.artifacts import load_artifact
from statomix.transformation import ExcludeRows

project = Project(
    project_name="Germinal Center Study",
    project_dir=(
        "/workspace/MoriartyG3/projects/germinal_centers/"
        "statomix_projects"
    ),
)

n0 = project.datasets["n0_raw"]
n0_months = n0.transformer.artifact(
    version=1,
    config_version=1,
)
```

Create Transformer version 2 from the already-converted months artifact:

```python
exclusion_reason = (
    "Age at surgery recorded as zero; surgery-based PFS and LRC "
    "durations cannot be established from the available source."
)

n0_reviewed = n0.transformer.create_data(
    source=n0_months,
    operations=[
        ExcludeRows(
            identifier="patientID",
            values=("CAIB-T00004423OC",),
            reason=exclusion_reason,
        )
    ],
    version=2,
    config_version=1,
    name="exclude_unusable_surgery_time_origin",
)

print(n0_reviewed.artifact_id)
print(n0_reviewed.path("df"))
print(n0_reviewed.path("exclusions"))
print(n0_reviewed.path("audit"))
```

Do not change the `version=2, config_version=1` numbers after a completed
artifact has been written. If that slot already exists with a different parent
or operation, choose a genuinely unused Transformer version.

## 4. Verify the exact result

This comparison proves that the new dataframe is exactly the months dataframe
minus the reviewed patient, with its original index and all remaining values
unchanged:

```python
import pandas as pd
from pandas.testing import assert_frame_equal

before = load_artifact(n0_months)
after = load_artifact(n0_reviewed)

patient_id = "CAIB-T00004423OC"
expected_df = before.df.loc[
    before.df["patientID"].ne(patient_id)
].copy()

assert before.df["patientID"].eq(patient_id).sum() == 1
assert after.df["patientID"].eq(patient_id).sum() == 0
assert len(after.df) == len(before.df) - 1
assert_frame_equal(after.df, expected_df)

event_columns = sorted(
    {
        pair.event_profile.col_name
        for pair in before.pairs.pairs.values()
    }
)
assert_frame_equal(
    after.df[event_columns],
    expected_df[event_columns],
)

assert after.metadata == before.metadata
assert after.ranks == before.ranks
assert {
    label: (
        pair.event_profile.col_name,
        pair.time_profile.col_name,
    )
    for label, pair in after.pairs.pairs.items()
} == {
    label: (
        pair.event_profile.col_name,
        pair.time_profile.col_name,
    )
    for label, pair in before.pairs.pairs.items()
}

excluded_rows = pd.read_parquet(
    n0_reviewed.path("exclusions")
)
display(excluded_rows)

assert excluded_rows["identifier_column"].tolist() == [
    "patientID"
]
assert excluded_rows["reason"].tolist() == [exclusion_reason]
assert n0_reviewed.manifest["parents"][0]["artifact_id"] == (
    n0_months.artifact_id
)

print(
    "PASS: one reviewed patient excluded from a new child artifact; "
    "months, events, endpoint bindings, units, ranks, lineage, and "
    "parent provenance preserved."
)
```

The transformation workbook contains an `Excluded Rows` sheet. The same rows
are stored in checksummed, machine-readable form at
`n0_reviewed.path("exclusions")`.

## 5. Configure a new Analyzer report

Use a new, unused Analyzer version/configuration. Do not reuse or overwrite an
Analyzer report made from the pre-exclusion artifact.

```python
from statomix.transformation import MONTHS

n0.configure_analyzer_from_artifact(
    source=n0_reviewed,
    version=3,
    config_version=1,
    survival_evaluation={
        "OS": {"unit": MONTHS, "time_points": [12, 24, 36, 48, 60]},
        "LRC": {"unit": MONTHS, "time_points": [12, 24, 36, 48, 60]},
        "PFS": {"unit": MONTHS, "time_points": [12, 24, 36, 48, 60]},
    },
)
```

## 6. Inspect and commit

Do not commit the delivery manifest or checker unless you want them maintained
in the repository. Stage only package code, tests, and optionally this README:

```bash
git add \
    src/statomix/transformation/specifications.py \
    src/statomix/transformation/rows.py \
    src/statomix/transformation/operations.py \
    src/statomix/transformation/__init__.py \
    src/statomix/pipelines/transformer/transformer.py \
    src/statomix/storage/artifacts.py \
    src/statomix/reporting/excel/transformation.py \
    tests/test_row_exclusion.py \
    ROW_EXCLUSION_README.md

git diff --cached --check
git diff --cached --stat
git diff --cached
git status --short
```

After reviewing the staged diff:

```bash
git commit -m "feat: add audited identifier-based row exclusions"
git push origin development
```

The two delivery-only files can then be removed from the repository working
tree:

```bash
rm ROW_EXCLUSION_DELIVERY.json check_row_exclusion_overlay.py
```
