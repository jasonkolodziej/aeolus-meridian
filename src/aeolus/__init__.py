"""AEOLUS / MERIDIAN hurricane forecast system.

Reference implementation of Scope v2.1. The package is organised around the two
things v2.1 changed:

* :mod:`aeolus.data` and :mod:`aeolus.training.curriculum` implement the
  train/serve consistency policy (§4.6) -- ERA5 pretrains, GDAS serves, and
  final best-track never becomes a model input.
* :mod:`aeolus.inference.scheduler` implements the corrected operational timing
  (§6.2) -- cycle t runs on the t-6 NWP cycle, gated on TC-Vitals arrival, with
  products due ahead of the t+3:00 advisory.
"""

from __future__ import annotations

__version__ = "2.1.0"

from .data.sources import Flavor, Role
from .time_utils import cycle_label, select_nwp_cycle

__all__ = ["__version__", "Flavor", "Role", "cycle_label", "select_nwp_cycle"]
