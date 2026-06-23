"""Training labels derived from macro events and market resolutions."""

from __future__ import annotations

from kalshi_train.targets.fed_cut import FedCutExample, build_fed_cut_examples

__all__ = ["FedCutExample", "build_fed_cut_examples"]
