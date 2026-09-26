"""Perturbation package: targeted gender swaps and matched-magnitude controls."""
from .controls import ControlSet, SynonymControls
from .gender import GenderSwapper
from .types import Change, Perturbation

__all__ = [
    "Change",
    "Perturbation",
    "GenderSwapper",
    "ControlSet",
    "SynonymControls",
]
