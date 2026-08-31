# Verification strategy

The test suite has four layers:

1. contract tests for immutable values, no-op schemas, and version selection;
2. scientific unit tests for p-value labeling, survival validation, normality
   reliability, Holm adjustment, unique cutoff partitions, CRAN maxstat
   reference values, and exhaustive permutation parity;
3. compatibility tests that import the historical module paths and exercise
   absent-datatype workflows;
4. artifact comparison between a run of `feat/noop-cleaner-contract` and the
   refactored branch.

The artifact comparator is intentionally strict.  It compares directory
inventory first, then uses format-aware comparison for Parquet, Excel, JSON,
YAML, and PNG outputs.  Other files are compared by SHA-256.  It does not run a
pipeline itself, so both checkouts must receive the same input data and runtime
parameters.

```bash
uv run statomix-compare /path/to/reference /path/to/candidate
```

The expected scientific deltas are listed in
`SCIENTIFIC_CHANGES.md`.  Do not weaken the comparator globally to accept
them; review those files/columns explicitly.

The maxstat-specific reference versions, assumptions, and check inventory are
documented in `MAXSTAT_VALIDATION.md`. Conditional Monte Carlo tests use small,
pinned seeds; exhaustive enumeration is limited to small fixtures.
