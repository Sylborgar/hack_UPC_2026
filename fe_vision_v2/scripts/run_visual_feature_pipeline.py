"""
run_visual_feature_pipeline.py
--------------------------------
Punto de entrada CLI para el pipeline de feature engineering visual.

Uso:
    python scripts/run_visual_feature_pipeline.py \
        --data_dir "C:/Users/sylbo/.../Smadex_Creative_Intelligence_Challenge" \
        --output_dir "outputs" \
        --clip_model "openai/clip-vit-base-patch32" \
        --cnn_model "resnet50" \
        --clip_pca_components 64 \
        --cnn_pca_components 64
"""

import argparse
import logging
import sys
from pathlib import Path

# Permitir importar src/ desde cualquier directorio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VisualFeatureConfig
from src.pipeline import VisualFeaturePipeline


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smadex Creative Intelligence — Visual Feature Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Ruta base del dataset (contiene creatives.csv, creative_summary.csv y assets/).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directorio donde se guardarán todos los outputs.",
    )

    # Modelos
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Nombre del modelo CLIP en HuggingFace Hub.",
    )
    parser.add_argument(
        "--cnn_model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"],
        help="Backbone CNN a usar.",
    )

    # PCA
    parser.add_argument(
        "--clip_pca_components",
        type=int,
        default=64,
        help="Número de componentes PCA para embeddings CLIP.",
    )
    parser.add_argument(
        "--cnn_pca_components",
        type=int,
        default=64,
        help="Número de componentes PCA para embeddings CNN.",
    )

    # Proceso
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de batch para extracción de embeddings.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Tamaño de resize para CNN (no afecta a CLIP que usa su propio procesador).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Columnas configurables
    parser.add_argument("--asset_path_column", type=str, default="asset_path")
    parser.add_argument("--creative_id_column", type=str, default="creative_id")
    parser.add_argument("--campaign_id_column", type=str, default="campaign_id")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = VisualFeatureConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        clip_model_name=args.clip_model,
        cnn_model_name=args.cnn_model,
        clip_pca_components=args.clip_pca_components,
        cnn_pca_components=args.cnn_pca_components,
        batch_size=args.batch_size,
        image_size=args.image_size,
        log_level=args.log_level,
        asset_path_column=args.asset_path_column,
        creative_id_column=args.creative_id_column,
        campaign_id_column=args.campaign_id_column,
    )

    pipeline = VisualFeaturePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
