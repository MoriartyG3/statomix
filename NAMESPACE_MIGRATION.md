# Namespace migration

Statomix 0.3 keeps the original domain-facing organization. The temporary
`statomix.analysis` and `statomix.workflows` namespaces existed only on
unreleased refactor branches and have been removed to avoid two names for the
same implementation.

## Analytics imports

| Temporary refactor import | Canonical import |
| --- | --- |
| `statomix.analysis.descriptive.categorical` | `statomix.analytics.datatypes.categorical` |
| `statomix.analysis.descriptive.numerical` | `statomix.analytics.datatypes.numerical` |
| `statomix.analysis.normality` | `statomix.analytics.datatypes.numerical` |
| `statomix.analysis.multiplicity` | `statomix.analytics.multiplicity` |
| `statomix.analysis.survival` | `statomix.analytics.datatypes.survival` |
| `statomix.analysis.survival.thresholds` | `statomix.analytics.datatypes.survival.thresholds` |

The threshold implementation is grouped beneath
`statomix.analytics.datatypes.survival.thresholds.mpv`:

```python
from statomix.analytics.datatypes.survival import (
    BinaryClassSurv,
    MaximallySelectedLogRank,
    MultiClassSurv,
    SingleClassSurv,
    ThresholdScan,
)
from statomix.analytics.datatypes.survival.thresholds import MinimumPValue
```

## Orchestration imports

| Temporary refactor import | Canonical import |
| --- | --- |
| `statomix.workflows.project.Project` | `statomix.project.project.Project` |
| `statomix.workflows.dataset.Dataset` | `statomix.dataset.dataset.Dataset` |
| `statomix.workflows.cleaner.Cleaner` | `statomix.pipelines.cleaner.cleaner.Cleaner` |
| `statomix.workflows.dataset_analyzer.Analyzer` | `statomix.pipelines.analyzer.analyzer.Analyzer` |
| `statomix.workflows.group_analyzer.GroupAnalyzer` | `statomix.pipelines.analyzer.group_analyzer.GroupAnalyzer` |
| `statomix.workflows.project_analyzer.Analyzer` | `statomix.project.analyzer.analyzer.Analyzer` |

The root convenience imports remain available:

```python
from statomix import Dataset, Project
```

## What did not move back

The following packages represent genuine backend responsibilities and remain
separate:

- `statomix.curation` for semantic/profile reconstruction;
- `statomix.storage` for layout, atomic writes, serialization, hashing, and
  version persistence;
- `statomix.core` for dependency-neutral contracts, result models, and errors;
- `statomix.reporting` for Excel and presentation rendering.

This migration changes Python import locations only. It does not intentionally
change existing Zarr groups, filenames, workbook schemas, no-op contracts, or
the documented schema-2 survival-threshold behavior.
