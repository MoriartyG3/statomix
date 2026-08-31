from __future__ import annotations


def test_legacy_modules_reexport_refactored_implementations() -> None:
    from statomix.analysis.normality import Normality
    from statomix.analysis.survival import (
        BinaryClassSurv,
        MaximallySelectedLogRank,
        SingleClassSurv,
        ThresholdScan,
    )
    from statomix.analytics.datatypes.numerical.normality import (
        Normality as LegacyNormality,
    )
    from statomix.analytics.datatypes.survival.binary_class_surv import (
        BinaryClassSurv as LegacyBinaryClassSurv,
    )
    from statomix.analytics.datatypes.survival.single_class_surv import (
        SingleClassSurv as LegacySingleClassSurv,
    )
    from statomix.pipelines.base import BasePipeline
    from statomix.storage.versioning import BasePipeline as RefactoredBasePipeline

    assert LegacyNormality is Normality
    assert LegacyBinaryClassSurv is BinaryClassSurv
    assert LegacySingleClassSurv is SingleClassSurv
    assert MaximallySelectedLogRank.__name__ == "MaximallySelectedLogRank"
    assert ThresholdScan.__name__ == "ThresholdScan"
    assert BasePipeline is RefactoredBasePipeline
