from __future__ import annotations

from .data_loader import load_feature_sets


class FeatureRegistry:
    def __init__(self, feature_sets_path: str):
        self.feature_sets_path = feature_sets_path
        self.feature_sets = load_feature_sets(feature_sets_path)

    def get(self, name: str) -> list[str]:
        if name not in self.feature_sets:
            raise KeyError(f"Feature set {name!r} not found. Available: {sorted(self.feature_sets)}")
        return list(self.feature_sets[name])

