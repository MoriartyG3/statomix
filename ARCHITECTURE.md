# Statomix architecture

The refactor follows the same principles used in Multiomix: explicit values,
keyword-oriented internal calls, dependency-light contracts, one logging
entry point, backend adapters at the edge, and compatibility at public
boundaries.

## Dependency direction

- `core` is dependency-light and imports no Statomix implementation layer.
- `storage` depends only on `core` plus backend libraries.
- `analysis` owns statistical computation; `curation` may consume its
  descriptive summaries.
- `reporting` turns analysis and storage contracts into presentation files.
- `workflows` compose all of those layers and retain the historical
  notebook-facing methods.

New analysis code does not own Excel presentation or workflow orchestration.
The existing survival classes retain lazy validation helpers, and
`MinimumPValue` retains its Zarr artifact adapter, solely to preserve public
methods and persisted paths.  Those compatibility seams are deliberately
local and do not run during module import; new features should put persistence
in `storage` or `workflows`.

## Stability boundary

The following are compatibility requirements:

- existing imports under `statomix.analytics`, `statomix.pipelines`,
  `statomix.dataset`, and `statomix.project`;
- the public `Project`/`Dataset`/`Cleaner`/`Analyzer` method names and default
  call sequence;
- Zarr group names, artifact filenames, workbook sheet names, and persisted
  empty-artifact schemas;
- legacy nested dictionaries returned by pipeline group selection.

Internally, immutable dataclasses (`GroupBundle`, `AnalyzerInputPaths`, and
procedure status values) replace unvalidated string-key dictionaries.  The
storage adapter converts them back to the legacy shape at the public boundary.

## Read and write semantics

Read-like operations use `find_version_group`/`find_config_group` and never
create missing groups.  Write workflows retain explicit `require_group`
semantics.  File artifacts that can be rendered off-store use a sibling
temporary file followed by an atomic same-filesystem replacement.

Curated-state inheritance is a distinct workflow boundary. It consumes an
immutable parent `curated_data` contract and writes independent target
artifacts. Pure schema/row validation and profile reconstruction live in
`statomix.curation.inheritance`; filesystem and Zarr orchestration live in
`statomix.workflows.cleaner_inheritance`. Existing Cleaner methods are not
special-cased and their default behavior is unchanged.

## Extension points

New statistical operations belong in `statomix.analysis` and can implement the
small `Analysis` protocol.  They can be registered through `AnalysisRegistry`
without changing workflow orchestration.  New storage or reporting formats
should be adapters that consume domain results instead of being embedded in
the statistical classes.
