"""Preprocessing namespace for frame-wise video transformations.

Purpose:
    Prepare loaded video data for detection/tracking through grayscale,
    normalization, and binarization operations.

Inputs:
    A valid ``Casa`` dictionary containing video arrays.

Outputs:
    Updated arrays in ``casa['video']`` and preprocessing metadata in
    ``casa['meta']``.

Methods:
    - ``grayscale(...)``
    - ``normalization.<method>(...)``
    - ``binarization.<method>(...)``
"""
from . import binarization
from . import normalization
from .grayscale import grayscale

__all__ = [
    "grayscale",
    "binarization",
    "normalization",
]
