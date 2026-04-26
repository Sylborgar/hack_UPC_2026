from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from .config import CBRConfig
from .schemas import PreprocessArtifacts
from .utils import l2_normalize


class BlockPreprocessor:
    def __init__(self, config: CBRConfig):
        self.config = config
        self.preprocessors: dict[str, object] = {}
        self.columns_by_block: dict[str, list[str]] = {}
        self.block_slices: dict[str, tuple[int, int]] = {}

    def fit_transform(self, df: pd.DataFrame, columns_by_block: dict[str, list[str]]) -> PreprocessArtifacts:
        self.columns_by_block = {k: list(v) for k, v in columns_by_block.items()}
        block_matrices: dict[str, np.ndarray] = {}
        weighted: list[np.ndarray] = []
        start = 0
        missing_parts = []
        for name, cols in self.columns_by_block.items():
            if not cols:
                continue
            matrix = self._fit_transform_block(df, name, cols)
            block_matrices[name] = matrix
            end = start + matrix.shape[1]
            self.block_slices[name] = (start, end)
            start = end
            weighted.append(matrix * np.sqrt(self.config.all_blocks()[name].weight))
            missing_parts.append(df[cols].isna().mean(axis=1))
        final = l2_normalize(np.concatenate(weighted, axis=1).astype(np.float32))
        missing_row_ratio = pd.concat(missing_parts, axis=1).mean(axis=1).to_numpy(dtype=np.float32) if missing_parts else np.zeros(len(df), dtype=np.float32)
        return PreprocessArtifacts(final, block_matrices, self.preprocessors, self.block_slices, missing_row_ratio)

    def transform(self, df: pd.DataFrame) -> PreprocessArtifacts:
        block_matrices: dict[str, np.ndarray] = {}
        weighted: list[np.ndarray] = []
        for name, cols in self.columns_by_block.items():
            if not cols:
                continue
            matrix = self._transform_block(df, name, cols)
            block_matrices[name] = matrix
            weighted.append(matrix * np.sqrt(self.config.all_blocks()[name].weight))
        final = l2_normalize(np.concatenate(weighted, axis=1).astype(np.float32))
        return PreprocessArtifacts(final, block_matrices, self.preprocessors, self.block_slices, np.zeros(len(df), dtype=np.float32))

    def _fit_transform_block(self, df: pd.DataFrame, name: str, cols: list[str]) -> np.ndarray:
        if name in self.config.categorical_blocks:
            values = df[cols].fillna(self.config.missing_values.categorical_strategy).astype(str)
            kwargs = {"handle_unknown": "ignore"}
            try:
                enc = OneHotEncoder(sparse_output=True, min_frequency=None, **kwargs)
            except TypeError:
                enc = OneHotEncoder(sparse=True, **kwargs)
            raw = enc.fit_transform(values)
            self.preprocessors[name] = {"type": "categorical", "encoder": enc, "columns": cols}
            arr = raw.astype(np.float32).toarray() if sparse.issparse(raw) else raw.astype(np.float32)
            return l2_normalize(arr)
        numeric = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy=self.config.missing_values.numeric_strategy)
        scaler = RobustScaler() if self.config.scaler == "robust" else StandardScaler()
        arr = imputer.fit_transform(numeric)
        arr = scaler.fit_transform(arr)
        self.preprocessors[name] = {"type": "numeric", "imputer": imputer, "scaler": scaler, "columns": cols}
        return l2_normalize(np.nan_to_num(arr).astype(np.float32))

    def _transform_block(self, df: pd.DataFrame, name: str, cols: list[str]) -> np.ndarray:
        proc = self.preprocessors[name]
        if proc["type"] == "categorical":
            values = df.reindex(columns=cols).fillna(self.config.missing_values.categorical_strategy).astype(str)
            raw = proc["encoder"].transform(values)
            arr = raw.astype(np.float32).toarray() if sparse.issparse(raw) else raw.astype(np.float32)
            return l2_normalize(arr)
        numeric = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        arr = proc["imputer"].transform(numeric)
        arr = proc["scaler"].transform(arr)
        return l2_normalize(np.nan_to_num(arr).astype(np.float32))

