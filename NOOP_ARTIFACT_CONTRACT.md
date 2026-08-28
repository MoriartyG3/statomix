# Statomix no-op artifact contract

The no-op foundation was introduced on `feat/noop-cleaner-contract`; this
refactor starts from commit
`4b3843fb6a4e1436b7199e9afc2eb2d99bb70611` and preserves that contract.

## Invariant

An empty edit schema is an identity element. For a dataframe `df` and empty
schema collection `E`:

```text
apply(df, E) == df
```

The implementation never inserts dummy rows, dummy categories, or synthetic
survival pairs. Empty parquet artifacts contain zero rows while retaining their
required columns.

Every edit artifact has an explicit no-op factory:

```python
ColEditSchema.empty()
CatMetaEditSchema.empty()
SurvEditSchema.empty()
SurvCatMetaEditSchema.empty()
SurvPairs.empty()
```

Saving and loading one of these objects is supported. Passing the empty edit
schemas to `apply_curation_schemas()` leaves the dataframe unchanged, including
its index and dtypes.

## Applicability rules

- Applicability is calculated from `col_profiles_curated.parquet`, after the
  user has accepted or corrected the inferred datatypes.
- No categorical columns: omit the human report, create an empty
  `cat_meta_edit_schema.parquet`, record `not_applicable`, and continue.
- No survival columns: omit both human survival reports and create empty
  survival profiles, survival edit schema, survival pairs, and survival-event
  categorical schema.
- No numerical columns: no new Cleaner artifact is invented. Analyzer
  numerical summaries and normality diagnostics return zero-row dataframes
  with stable columns.
- Analyzer categorical and survival descriptives also use stable zero-row
  contracts. The survival contract includes every default probability and
  RMST time point (12, 24, 36, 48, and 60).
- Datatype maps always contain every `DataTypes` header plus
  `Survival Labels`, even when the associated option list is empty.
- Excel data validation is omitted when its source list contains no values;
  an invalid reversed range is never generated.

## Metadata

Cleaner configuration metadata now contains:

```text
curated_datatype_counts:
  Identifier: <int>
  Numerical: <int>
  Categorical: <int>
  Survival: <int>
  DateTime: <int>
  Free Text: <int>

procedure_status:
  <procedure>:
    status: completed | not_applicable
    reason: <machine-readable reason>
    input_count: <int>
    output_count: <int>
```

The schema alone deliberately cannot distinguish these two cases:

1. applicable columns existed but the user specified zero edits;
2. no applicable columns existed.

The persisted status and counts provide that audit distinction.

## Run the checks

The commands below are intentionally separated so a failure identifies one
contract layer at a time.

### 1. Create or synchronize the environment

```bash
cd /path/to/statomix_noop_contract
uv sync --group dev
```

Expected: the project and development dependencies synchronize without a
resolver error. A private-repository authentication error for `fileverse17`
means GitHub credentials are not available in the environment.

### 2. Check empty artifact serialization and identity

```bash
uv run pytest -q tests/test_empty_artifacts.py
```

Expected: three tests pass. A parquet error involving a zero-column dataframe
means an artifact serializer bypassed `frame_from_rows()`.

### 3. Check absent datatype handling in the Analyzer

```bash
uv run pytest -q tests/test_datatype_absence.py
```

Expected: six tests pass. `KeyError: "name"`, `KeyError: "surv_label"`, or an
invalid Excel validation range indicates that an old empty-data path is still
being used.

### 4. Check Cleaner branch progression

```bash
uv run pytest -q tests/test_cleaner_not_applicable.py
```

Expected: one test passes without requesting any curated categorical or
survival Excel file.

### 5. Run the full suite

```bash
uv run pytest -q
```

Expected: ten tests pass.

### 6. Check formatting and static rules

```bash
uv run black --check src tests
uv run ruff check src tests
```

Interpretation: these are repository-wide checks and can expose pre-existing
style findings in untouched source files. Inspect whether each finding is
inside a changed file before treating it as a regression.

### 7. Build the package

```bash
uv build
```

Expected: one source distribution and one wheel are created under `dist/`.

## Test matrix

The included tests cover:

- round-trip loading of every typed empty schema;
- empty column and survival profile artifacts;
- the dataframe identity invariant;
- a total datatype inventory;
- no numerical columns;
- no categorical columns;
- no survival pairs;
- no analyzable columns beyond an identifier;
- analysis configuration with empty dropdown sources;
- all Cleaner categorical and survival calls on a numerical-only dataset.

The tests do not assert that datatype inference is scientifically correct.
They assert that, once the curated datatype inventory is known, empty branches
are represented consistently and do not break downstream execution.
