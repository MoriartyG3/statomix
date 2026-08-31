# Statomix architecture

Statomix uses a domain-first public hierarchy and service-oriented backend
boundaries. Package names follow the established user-facing organization;
modularity does not create a second vocabulary for the same operation.

## Dependency direction

- `core` is dependency-light and imports no higher Statomix layer.
- `storage` depends on `core` plus backend libraries.
- `analytics` owns statistical computation in the established
  `analytics/datatypes` hierarchy.
- `curation` may consume analytics summaries but remains a separate domain
  service.
- `reporting` turns analytics and storage contracts into presentation files.
- `pipelines` compose analytics, curation, reporting, and storage.
- `dataset` and `project` provide the user-facing orchestration objects.

The forbidden direction is equally important: `analytics`, `curation`, and
`storage` do not import `pipelines`, `dataset`, or `project`.

New analysis code does not own Excel presentation or workflow orchestration.
The existing survival classes retain lazy validation helpers, and
`MinimumPValue` retains its Zarr artifact adapter as an analytics orchestrator.
Pure candidate construction lives in `ThresholdScan`, and the scan-level
inferential procedure lives in `MaximallySelectedLogRank`. Those components do
not depend on project or pipeline orchestration.

## Stability boundary

The following are compatibility requirements:

- canonical imports under `statomix.analytics`, `statomix.pipelines`,
  `statomix.dataset`, and `statomix.project`;
- the public `Project`/`Dataset`/`Cleaner`/`Analyzer` method names and default
  call sequence;
- existing Zarr group names, artifact filenames, workbook sheet names, and
  persisted empty-artifact schemas, except for explicitly versioned scientific
  schemas such as threshold-scan schema 2, which use new fingerprinted paths
  and never rewrite legacy results;
- legacy nested dictionaries returned by pipeline group selection.

Internally, immutable dataclasses (`GroupBundle`, `AnalyzerInputPaths`, and
procedure status values) replace unvalidated string-key dictionaries.  The
storage adapter converts them back to the legacy shape at the public boundary.

## Read and write semantics

Read-like operations use `find_version_group`/`find_config_group` and never
create missing groups.  Write workflows retain explicit `require_group`
semantics.  File artifacts that can be rendered off-store use a sibling
temporary file followed by an atomic same-filesystem replacement.

Curated-state inheritance is a distinct pipeline boundary. It consumes an
immutable parent `curated_data` contract and writes independent target
artifacts. Pure schema/row validation and profile reconstruction live in
`statomix.curation.inheritance`; filesystem and Zarr orchestration live in
`statomix.pipelines.cleaner.inheritance`. Existing Cleaner methods are not
special-cased and their default behavior is unchanged.

## Extension points

New statistical operations belong in the appropriate
`statomix.analytics.datatypes` domain and can implement the small `Analysis`
protocol. They can be registered through `AnalysisRegistry` without changing
pipeline orchestration. New storage or reporting formats
should be adapters that consume domain results instead of being embedded in
the statistical classes.

Row-wise p-value corrections follow the same pattern: correction definitions
and backend mappings live in `statomix.analytics.multiplicity`, while the MPV
orchestrator consumes the registry for exploratory sensitivity output. The
maximally selected log-rank global p-value is a separate statistical procedure
and is intentionally not registered as a generic correction.

## Namespace ownership rule

Each concept has one implementation owner. Compatibility adapters are allowed
only when a genuinely separate backend owns the implementation—for example,
`pipelines.base` exposing storage-backed versioning or historical Cleaner
curation imports exposing the `curation` service. A façade must never redirect
`analytics` to a second analysis tree or `pipelines` to a second workflow tree.
