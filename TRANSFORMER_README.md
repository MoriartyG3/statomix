# Statomix Transformer overlay

Baseline: development commit `b4300afabedd6561741e9b2014b472984ae08174`.

This overlay adds deterministic column transformations, portable artifact
references, strict row-wise concatenation, and explicit Analyzer artifact binding.
Existing Cleaner APIs and legacy Analyzer configurations are retained.

## Installation

Start in your Statomix repository. Commit or otherwise back up local edits first.
The archive contains full replacement files, not a git patch. It contains no
source datasets, project stores, credentials, dependency environments, or git data.

Before extraction, run the included standard-library baseline checker:

```bash
unzip -p /path/to/statomix_transformer_overlay.zip check_transformer_overlay.py | \
    python - /path/to/statomix_transformer_overlay.zip
```

Replace the archive path with its actual downloaded location. This command does
not import Statomix. It checks every replacement against the baseline checksum,
rejects conflicting new files, and accepts an already-applied identical overlay.
If it reports a conflict, stop and compare your local changes; do not force an
overwrite. The checker is a preflight, not a lock against concurrent local edits.

If it passes, extract from the repository root:

```bash
unzip -o /path/to/statomix_transformer_overlay.zip
uv run --active ruff check src tests
uv run --active pytest -q tests/test_transformer.py
uv run --active pytest -q
uv build
git diff --check
git status --short
git diff --stat
```

No tests or Statomix procedures were executed while preparing this delivery.
Static parsing and lint/format checks, if available, are recorded separately in
`TRANSFORMER_DELIVERY.json`. Passing your full suite is required before commit.

Restart Jupyter's kernel after updating the package. Continue on `development`;
no new branch or automatic commit is required by this overlay.

## Concepts

- `dataset.curated_artifact(...)`: a content-pinned, read-only snapshot of a
  completed Cleaner output. This does not edit historical files.
- `dataset.transformer.create_data(...)`: save a child from an ordered operation
  plan. It returns a `DatasetArtifactRef`, not an untracked DataFrame.
- `dataset.transformer.artifact(...)`: retrieve a completed child after reopening.
- `project.combine_datasets(...)`: produce a new registered dataset from two or
  more selected artifacts; returns the new Dataset.
- `dataset.configure_analyzer_from_artifact(...)`: bind an independent Analyzer
  configuration to an exact artifact and explicit survival evaluation units.

Versions pin the ordered parents; configurations pin operations and execution
fingerprints. Changed parents require a new Transformer version. Changed
operations or implementation fingerprints require a new configuration. Use
`artifact(...)` to retrieve old outputs without requesting re-execution.

Files live below
`datasets/<name>/transformer/versionN/configM/data/`:

- `df.parquet`, including typed categorical ranks and semantic metadata
- `col_profiles.parquet` and `surv_pairs.parquet`
- `metadata.json`: units, category domains, endpoint definitions
- `row_lineage.parquet`: immediate parent artifact and row ordinal for each row
- `specification.json`: exact operations, reason, runtime/source fingerprint
- `audit.xlsx`: operation counts, parent artifact hashes, specification
- `manifest.json`: completion marker, file hashes, ordered parent references

Output hashes are in the manifest, rather than in the workbook that is itself
hashed. File checksums identify stored bytes; they do not promise byte-identical
Parquet serialization across library versions. Display labels are not identities.

## Notebook: arithmetic on a completed Cleaner artifact

```python
from statomix.storage.artifacts import load_artifact
from statomix.transformation import Affine, Ratio

dataset = project.datasets["n0_raw"]
parent = dataset.curated_artifact(version=1, config_version=1)

# Inspect saved, curated names before constructing an operation.
print(list(load_artifact(parent).df.columns))
```

Supply the actual existing names to the following reusable function. This supports
creating a weighted feature, then replacing that new feature within the child:

```python
def create_weighted_feature(dataset, parent, *, first_column, second_column,
                            output_column, version, config_version):
    operations = [
        Affine(
            output=output_column,
            terms=((first_column, 2.0), (second_column, -0.5)),
            offset=3.0,
            reason="Prespecified weighted feature: 2*A - 0.5*B + 3",
        ),
        Ratio(
            output=output_column,
            numerator=output_column,
            denominator=2.0,
            mode="replace",
            reason="Scale the weighted feature by one half",
        ),
    ]
    return dataset.transformer.create_data(
        source=parent,
        operations=operations,
        version=version,
        config_version=config_version,
        name="weighted_feature",
    )
```

Zero coefficients still participate in missing-value propagation. Scalars are
dimensionless coefficients; the affine offset uses the common input unit. Known
affine input units must agree exactly. Ratios derive their unit scale and
dimension from their operands when both are known; otherwise the output unit is
unknown. This v1 unit algebra does not simplify arbitrary compound dimensions.
For unit conversion use `ConvertUnit`, not an arithmetic rename.

## Notebook: convert verified days to elapsed months

The caller must verify which columns are in days. No unit is inferred from its
name. `MONTHS` means 365.25/12 days, not calendar months. No rounding is applied.

```python
from statomix.transformation import ConvertUnit, DAYS, MONTHS

def convert_verified_duration_columns(dataset, parent, *, duration_columns,
                                      version, config_version):
    operations = [
        ConvertUnit(
            source=column,
            output=column,
            source_unit=DAYS,
            target_unit=MONTHS,
            mode="replace",
            reason="Verified source days; elapsed months defined as 365.25/12 days",
        )
        for column in duration_columns
    ]
    return dataset.transformer.create_data(
        source=parent,
        operations=operations,
        version=version,
        config_version=config_version,
        name="durations_in_months",
    )
```

If the source unit is unknown, `source_unit` plus `reason` is an explicit assertion
recorded in the child; it does not silently modify the parent. If already known,
the source unit must match. Attempting days-to-months again on a months child
raises an error. For a different convention, construct `Unit("months", "time",
your_days_per_month)` and use that same unit in analysis and other cohorts.

Creating a new duration does not automatically rebind an endpoint. Pass
`bind_endpoints=("OS",)` to `ConvertUnit` when deliberately replacing OS's
duration reference with the new output column. Event data and declarations are
unchanged. Generic arithmetic cannot replace a bound duration, event, identifier,
or categorical column. Unsupported survival types remain blocked.

## Notebook: combine N0 and Priyanka *after* conversion

Before exporting the Cleaner references, supply reviewed endpoint definitions:
`endpoint_definitions={label: definition_and_time_origin, ...}` and a `reason`.
These declarations must agree across cohorts. Equal text is an auditable human
assertion, not proof of clinical equivalence.

```python
def combine_validation_artifacts(project, *, n0_months, priyanka_months,
                                 n0_mapping, priyanka_mapping,
                                 patient_id_column, dataset_name):
    return project.combine_datasets(
        sources=[n0_months, priyanka_months],
        mappings=[n0_mapping, priyanka_mapping],
        identity_columns=[patient_id_column],
        dataset_name=dataset_name,
        display_label="Combined validation cohorts",
        reason="Pool reviewed N0 and Priyanka artifacts already in months",
        cohort_column="source_cohort",
    )
```

Mappings must cover **all** source columns and map them to the same common schema.
If names already match, build each mapping as
`{name: name for name in load_artifact(reference).df.columns}`. Mismatched physical
dtypes require explicit upstream harmonization: there is no silent cast, column
projection, deduplication, or schema union. Missing or overlapping identity keys
raise before dataset registration. In particular, this operation does NOT apply
the Priyanka/N0 exclusion rule on your behalf.

The combined artifact is available at:

```python
combined_ref = combined_dataset.transformer.artifact(
    version=1,
    config_version=1,
)
```

Its source DataFrame is already in months. Parent transformation plans are stored
as provenance, not replayed. A copy of that DataFrame, including rank/unit footer
metadata, is the newly registered dataset's source. Do not rerun days conversion.

## Analyzer binding and viewing exact generated plots

Provide an evaluation entry for every endpoint, with the same Unit as its saved
duration. Time points are in that unit; no default month assumption is made for
artifact-bound analysis. An empty list requests no probability/RMST evaluations.

```python
def summarize_month_artifact(dataset, reference, *, version, config_version):
    state = load_artifact(reference)
    evaluation = {
        label: {"unit": MONTHS, "time_points": [12, 24, 36, 48, 60]}
        for label in state.pairs.pairs
    }
    dataset.configure_analyzer_from_artifact(
        source=reference,
        version=version,
        config_version=config_version,
        survival_evaluation=evaluation,
    )
    return dataset.analyzer.create_summary_report(
        version=version,
        config_version=config_version,
    )
```

Choose an unused Analyzer configuration. Existing legacy configurations are not
retargeted. New survival reports have `surv/report_manifest.json`, which maps
endpoint labels to exact PNG filenames and checksums (filenames are hash-based
so endpoint labels cannot escape the output directory). To view them:

```python
import json
from IPython.display import Image, display

def show_saved_survival_plots(analyzer, *, version, config_version):
    bundle = analyzer._find_group_bundle(
        version=version, config_version=config_version
    )
    directory = bundle["config"]["path"] / "surv"
    report = json.loads((directory / "report_manifest.json").read_text())
    for record in report["plots"]:
        print(record["endpoint"])
        display(Image(filename=str(directory / record["path"])))
```

## Integrity and recovery

- Completed bundle files are immutable. Modification or missing files raises on
  load, including transitive parent checks. Parents must remain available.
- Arithmetic preserves the input index, including duplicate index labels. Row
  identity is the artifact ID plus row ordinal, not the pandas index alone.
- Concatenation creates a fresh RangeIndex and complete immediate-parent lineage.
- Writer locks prevent concurrent writers using this API. A killed process can
  leave a lock or unregistered dataset directory. Inspect and confirm no writer
  remains before manual recovery; the API does not delete these automatically.
- Staging failures publish no completed artifact. Filesystem rename assumes a
  local same-filesystem store. This is not a distributed transaction or a
  power-loss durability guarantee.
- A combined dataset is registered only after its transformed output completes.
  Failure before registration can leave an inspectable, unregistered directory.
- Legacy Cleaner artifacts stay readable without new manifests or unit fields.
  Their units remain unknown unless you explicitly declare them.
- Moving the entire project preserves references; partial moves and cross-project
  references are not supported in v1.

## Limits

No filtering, joins, learned preprocessing, arbitrary callbacks, automatic
clinical reconciliation, cross-project imports, or new survival estimators.
Concatenation is strict and analysis-ready only after all compatibility checks
pass; conflicting data must be curated separately. There is no automatic new
Cleaner configuration synthesized from the derived output. The existing
single-parent inheritance writer is unchanged; use artifact APIs for these new
derivations to preserve their metadata.

New artifact-bound survival summaries explicitly honor units. Direct calls to
legacy plotting/statistical APIs retain their existing parameter defaults; users
calling those low-level APIs must still pass appropriate units/labels themselves.
Everything remains in-memory pandas execution. Large datasets may need a future
streaming implementation. Integer inputs beyond the exact float64 range are
rejected rather than silently rounded; other arithmetic follows float64 precision.

## Suggested verification order

1. Run the new tests, then the full suite.
2. Review `git diff`, including the three changed integration files.
3. On a fresh Transformer configuration, convert one known duration.
4. Inspect the new Parquet metadata and `audit.xlsx`; verify source hashes stay fixed.
5. Bind a fresh Analyzer configuration and inspect its time-unit labels.
6. Attempt a deliberately conflicting concatenation and confirm it fails.
7. Only then combine your reviewed cohort artifacts and commit on development.
