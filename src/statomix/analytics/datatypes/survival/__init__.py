"""Legacy survival-analysis namespace."""

from .binary_class_surv import BinaryClassSurv
from .multi_class_surv import MultiClassSurv
from .single_class_surv import SingleClassSurv

__all__ = ["BinaryClassSurv", "MultiClassSurv", "SingleClassSurv"]
