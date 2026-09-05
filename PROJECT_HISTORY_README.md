# Statomix project-history report overlay

This overlay adds a read-only project lineage report to Statomix. It targets the
clean `development` commit:

```text
accb5087f45589944861a5034e0a9656f1ad1bae
```

The preflight checker must pass before extraction. It verifies the archive,
branch, exact commit, clean working tree, replacement-file checksum, and that
all new paths are absent.

## 1. Verify and extract

Run from the Statomix repository root:

```bash
cd /workspace/MoriartyG3/python_packages/statomix

history_zip="/workspace/MoriartyG3/python_packages/statomix_project_history_overlay.zip"

unzip -p "$history_zip" \
    check_project_history_overlay.py | \
    python - "$history_zip"

unzip -o "$history_zip"
```

## 2. Format and run focused checks

```bash
uv run --active black \
    src/statomix/history \
    src/statomix/project/project.py \
    tests/test_project_history.py

uv run --active ruff check src tests

uv run --active pytest -q \
    tests/test_project_history.py
```

## 3. Run the full verification

```bash
uv run --active pytest -q
uv build

git diff --check
git status --short
git diff --stat
```

Do not commit until the focused and full checks pass.

## 4. Create the Germinal Center Study history report

Restart the notebook kernel after installing the overlay, reopen the project,
and create the report outside the project store:

```python
from pathlib import Path

from statomix import Project


project = Project(
    project_name="Germinal Center Study",
    project_dir=Path(
        "/workspace/MoriartyG3/projects/germinal_centers/"
        "statomix_projects"
    ),
)

history_report = project.create_history_report(
    output_dir=Path(
        "/workspace/MoriartyG3/projects/germinal_centers/"
        "statomix_history_reports"
    ),
    verify_checksums=True,
    include_files=False,
)

print("History ID:", history_report.history_id)
print("Interactive HTML:", history_report.html_path)
print("Static SVG:", history_report.svg_path)
print("JSON graph:", history_report.json_path)
print("Excel audit:", history_report.audit_path)
```

Open `history_report.html_path` from the Jupyter file browser. The HTML is a
standalone, self-contained report: it does not require a running Python widget,
web server, or CDN. It provides dataset, pipeline, role, status, report-node,
and text filters. Selecting a node shows its reason, shape, version,
configuration version, path, and recorded metadata.

The SVG is the deterministic static fallback. The JSON is the normalized graph
for programmatic use. The Excel workbook contains Nodes, Edges, Warnings, and
Node Attributes sheets; `include_files=True` also adds a Files sheet and exposes
per-file details in the HTML node panel.

## 5. Read the graph directly

The discovery API can be used without creating any output files:

```python
from statomix.history import discover_project_history


history = discover_project_history(
    project=project,
    verify_checksums=True,
    include_files=False,
)

display(history.nodes_frame())
display(history.edges_frame())
display(history.warnings_frame())
```

Discovery is read-only. Report files are written atomically outside the project
store. A content-derived history ID makes an unchanged report idempotent and
prevents accidental overwriting of a different history snapshot.

## 6. Commit after verification

The checker and delivery manifest are installation aids and should be moved out
of the repository before committing:

```bash
mv -n PROJECT_HISTORY_DELIVERY.json \
    /workspace/MoriartyG3/python_packages/

mv -n check_project_history_overlay.py \
    /workspace/MoriartyG3/python_packages/

git add \
    PROJECT_HISTORY_README.md \
    src/statomix/history \
    src/statomix/project/project.py \
    tests/test_project_history.py

git diff --cached --check
git diff --cached --stat
git status --short

git commit -m \
    "feat: add project artifact history reports"

git push origin development
```

## Design boundaries

- Source dataframes, Cleaner outputs, reference artifacts, Transformer outputs,
  Analyzer configurations, and generated reports have distinct node types.
- Edges are directed from an input/parent toward its derived consumer.
- Transformer parent roles remain typed, including base/update parents and
  concatenated inputs.
- Stored checksums are verified when available. Cleaner/source files without a
  stored expected checksum are marked as computed, not falsely called verified.
- Missing files, malformed manifests, checksum mismatches, dangling edges, and
  cycles are recorded as visible warnings/errors.
- Dataset display labels are presentation metadata; stable dataset keys remain
  the graph identity.
- A missing artifact reason is displayed as not recorded. The report never
  manufactures a reason.
- The history report does not create Cleaner, Transformer, Analyzer, or dataset
  versions and does not write inside the Statomix project store.
