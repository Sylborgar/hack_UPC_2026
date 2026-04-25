"""
config.py
---------
Configuración centralizada del pipeline de feature engineering visual.
Todos los parámetros configurables se definen aquí como dataclass.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class VisualFeatureConfig:
    """
    Configuración principal del pipeline.
    
    Se puede sobreescribir desde CLI o instanciar directamente en código.
    """

    # ── Rutas ────────────────────────────────────────────────────────────
    data_dir: Path = Path(".")
    output_dir: Path = Path("outputs")
    assets_subdir: str = "assets"

    # Nombres de CSVs de entrada
    creatives_csv: str = "creatives.csv"
    creative_summary_csv: str = "creative_summary.csv"

    # Columna que en creatives.csv apunta al asset
    asset_path_column: str = "asset_path"
    creative_id_column: str = "creative_id"
    campaign_id_column: str = "campaign_id"

    # ── Modelos ───────────────────────────────────────────────────────────
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # Backbone CNN: "resnet50" | "efficientnet_b0" | "efficientnet_b3" | "convnext_tiny"
    cnn_model_name: str = "resnet50"

    # ── PCA ───────────────────────────────────────────────────────────────
    clip_pca_components: int = 64
    cnn_pca_components: int = 64

    # ── Procesamiento ─────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 0          # DataLoader workers (0 = main process)
    image_size: int = 224         # Resize antes de CNN/CLIP
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # ── Visualización ─────────────────────────────────────────────────────
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    color_by: str = "campaign_id"  # columna para colorear los plots

    # ── Miscelánea ────────────────────────────────────────────────────────
    random_seed: int = 42
    log_level: str = "INFO"

    # ── Subdirectorios de salida (calculados) ─────────────────────────────
    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / self.assets_subdir

    def setup_output_dirs(self) -> None:
        """Crea todos los directorios de salida necesarios."""
        for d in [self.output_dir, self.figures_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
