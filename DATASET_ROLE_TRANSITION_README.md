# Audited dataset-role transition overlay

This overlay targets Statomix commit `6106455` on the `development` branch.
It adds `Project.set_dataset_role()` so an existing dataset can be reclassified
without rewriting its source dataframe or deleting prior pipeline artifacts.

## Behavior

- Requires an explicit dataset name, supported role, and nonempty reason.
- Persists the role on the dataset group and in the project dataset registry.
- Appends an ordered UTC-stamped transition record.
- Synchronizes the live `Dataset`, `Analyzer`, and reference builder objects.
- Treats an already-applied role as an idempotent no-op.
- Leaves source Parquet and existing Cleaner/Analyzer files unchanged.

## 1. Preflight

Run from the Statomix repository before extracting:

```bash
cd /workspace/MoriartyG3/python_packages/statomix

role_transition_zip="/workspace/MoriartyG3/python_packages/statomix_dataset_role_transition_overlay.zip"

unzip -p "$role_transition_zip" \
    check_dataset_role_transition_overlay.py |
    python - "$role_transition_zip"
```

Extraction is allowed only after the checker reports both PASS messages.

## 2. Extract

```bash
unzip -o "$role_transition_zip"
```

## 3. Format and verify

```bash
uv run --active black \
    src/statomix/project/project.py \
    tests/test_dataset_role_transition.py

uv run --active ruff check src tests

uv run --active pytest -q \
    tests/test_dataset_role_transition.py \
    tests/test_reference_artifacts.py
```

If the focused tests pass, run the complete checks:

```bash
uv run --active pytest -q
uv build

git diff --check
git status --short
git diff --stat
```

## 4. Stage and commit

Do not commit the delivery manifest or checker.

```bash
git add \
    DATASET_ROLE_TRANSITION_README.md \
    src/statomix/project/project.py \
    tests/test_dataset_role_transition.py

git diff --cached --check
git diff --cached --stat
git status --short

git commit -m "feat: add audited dataset role transitions"
git push origin development
```

Move the two delivery-only helpers outside the repository afterward:

```bash
mv -n \
    DATASET_ROLE_TRANSITION_DELIVERY.json \
    /workspace/MoriartyG3/python_packages/

mv -n \
    check_dataset_role_transition_overlay.py \
    /workspace/MoriartyG3/python_packages/
```

## 5. Correct the existing OCAT-900 role

Restart the notebook kernel after installing the committed code, reopen the
project, and run:

```python
ocat_900_dataset = project.datasets["ocat_900_survival"]

project.set_dataset_role(
    dataset_name="ocat_900_survival",
    dataset_role="reference",
    reason=(
        "This dataset is an authoritative survival update source for "
        "the OCAT subset and is not an independent analysis cohort."
    ),
)

assert ocat_900_dataset.dataset_role == "reference"
assert ocat_900_dataset.analyzer.dataset_role == "reference"
assert ocat_900_dataset.reference.dataset_role == "reference"

role_history = list(
    ocat_900_dataset.groups["root"].attrs["dataset_role_history"]
)

print("Current role:", ocat_900_dataset.dataset_role)
print("Latest transition:", role_history[-1])
```

The previously created column report remains on disk as historical review
material. It is not used to construct the direct reference artifact.
