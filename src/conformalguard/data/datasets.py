"""Dataset containers used by ConformalGuard."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Features, target, and provenance for one benchmark dataset."""

    name: str
    source: str
    source_id: int
    X: pd.DataFrame
    y: pd.Series