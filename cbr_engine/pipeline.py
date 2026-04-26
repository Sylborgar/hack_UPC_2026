from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np

from .config import CBRConfig
from .data_loader import load_cases, write_dataframe
from .feature_builder import FeatureBuilder
from .index_store import make_index_store
from .preprocessing import BlockPreprocessor
from .reporting import write_build_report
from .retriever import CBRRetriever
from .utils import ensure_dir, timer, utc_timestamp, write_json
from .validation import validate_input_dataframe


def build_cbr_index(data_path: str, feature_sets_path: str, config: CBRConfig, output_dir: str | None = None, force: bool = False) -> dict[str, object]:
    output = Path(output_dir or config.outputs.index_dir)
    if output.exists() and force:
        shutil.rmtree(output)
    ensure_dir(output)
    timings: dict[str, float] = {}
    with timer() as t:
        df = load_cases(data_path)
    timings["load_data_seconds"] = t["seconds"]
    df, validation_report = validate_input_dataframe(df, config)
    with timer() as t:
        built = FeatureBuilder(config, feature_sets_path).build(df)
        pre = BlockPreprocessor(config)
        artifacts = pre.fit_transform(df, built.columns_by_block)
    timings["build_features_seconds"] = t["seconds"]
    with timer() as t:
        store = make_index_store(config.index.backend).build(artifacts.matrix)
    timings["build_index_seconds"] = t["seconds"]

    ids = df[config.id_column].to_numpy()
    metadata_cols = list(dict.fromkeys([c for c in config.metadata_columns + config.outcome_columns if c in df.columns]))
    metadata = df[metadata_cols].copy()
    metadata["__row_pos"] = np.arange(len(df))
    metadata["missing_row_ratio"] = artifacts.missing_row_ratio
    np.save(output / "ids.npy", ids)
    np.save(output / "matrix.npy", artifacts.matrix.astype(np.float32))
    block_dir = ensure_dir(output / "block_matrices")
    for name, mat in artifacts.block_matrices.items():
        np.save(block_dir / f"{name}.npy", mat.astype(np.float32))
    metadata_path = write_dataframe(metadata, output / "metadata.parquet", index=False)
    joblib.dump({"preprocessor": pre, "block_slices": artifacts.block_slices}, output / "preprocessors.joblib")
    store.save(output)
    manifest = {
        "timestamp": utc_timestamp(),
        "num_cases": int(len(df)),
        "vector_dim": int(artifacts.matrix.shape[1]),
        "blocks": list(built.columns_by_block),
        "weights": config.block_weights(),
        "columns_by_block": built.columns_by_block,
        "backend": store.backend_name,
        "dataset_path": str(data_path),
        "feature_set_name": config.feature_set_name,
        "mode": config.mode,
        "validation": validation_report,
        "timings": timings,
        "metadata_path": str(metadata_path.name),
    }
    write_json(output / "build_manifest.json", manifest)
    write_json(output / "feature_report.json", built.report.to_dict())
    config.save_yaml(output / "config_resolved.yaml")
    write_build_report(output, manifest, built.report.to_dict())
    return manifest


def load_retriever(index_dir: str, config: CBRConfig | None = None) -> CBRRetriever:
    index_path = Path(index_dir)
    config = config or CBRConfig.from_yaml(index_path / "config_resolved.yaml")
    return CBRRetriever.load(index_path, config)
