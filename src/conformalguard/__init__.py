"""Public API for ConformalGuard."""

from conformalguard.guard import ConformalGuard, GuardConfig
from conformalguard.suite import (
    SuiteResult,
    available_datasets,
    load_builtin_dataset,
    run_builtin_suite,
)

__all__ = [
    "ConformalGuard",
    "GuardConfig",
    "SuiteResult",
    "__version__",
    "available_datasets",
    "load_builtin_dataset",
    "run_builtin_suite",
]

__version__ = "0.2.0"
