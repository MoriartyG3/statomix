from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "statomix"


def test_canonical_modules_export_the_public_implementations() -> None:
    from statomix.analytics.datatypes.numerical import Normality
    from statomix.analytics.datatypes.numerical.normality import (
        Normality as CanonicalNormality,
    )
    from statomix.analytics.datatypes.survival import (
        BinaryClassSurv,
        MaximallySelectedLogRank,
        SingleClassSurv,
        ThresholdScan,
    )
    from statomix.analytics.datatypes.survival.binary_class_surv import (
        BinaryClassSurv as CanonicalBinaryClassSurv,
    )
    from statomix.analytics.datatypes.survival.single_class_surv import (
        SingleClassSurv as CanonicalSingleClassSurv,
    )
    from statomix.pipelines.base import BasePipeline
    from statomix.storage.versioning import BasePipeline as StorageBasePipeline

    assert CanonicalNormality is Normality
    assert CanonicalBinaryClassSurv is BinaryClassSurv
    assert CanonicalSingleClassSurv is SingleClassSurv
    assert MaximallySelectedLogRank.__name__ == "MaximallySelectedLogRank"
    assert ThresholdScan.__name__ == "ThresholdScan"
    assert BasePipeline is StorageBasePipeline


def test_removed_duplicate_namespaces_are_absent() -> None:
    assert not (PACKAGE_ROOT / "analysis").exists()
    assert not (PACKAGE_ROOT / "workflows").exists()


def test_runtime_source_does_not_import_removed_namespaces() -> None:
    forbidden_imports = ("statomix.analysis", "statomix.workflows")
    offenders: list[str] = []

    for source_path in PACKAGE_ROOT.rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        if any(import_path in source for import_path in forbidden_imports):
            offenders.append(str(source_path.relative_to(PACKAGE_ROOT)))

    assert offenders == []
