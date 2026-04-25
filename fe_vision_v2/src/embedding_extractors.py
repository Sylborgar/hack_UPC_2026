"""
embedding_extractors.py
-----------------------
Extrae embeddings visuales usando CLIP (HuggingFace Transformers) y
backbones CNN pretrained de torchvision (ResNet50, EfficientNet, ConvNeXt).

Diseñado para facilitar el cambio de modelo vía config.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.config import VisualFeatureConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Extractor
# ─────────────────────────────────────────────────────────────────────────────

class CLIPEmbeddingExtractor:
    """
    Extrae embeddings visuales con openai/clip-vit-base-patch32 (o compatible).

    Cada embedding tiene dimensión 512 (ViT-B/32).

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Cargando modelo CLIP: {config.clip_model_name} en {self.device}")
        self.model = CLIPModel.from_pretrained(config.clip_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.model.eval()

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae un embedding por imagen.

        Returns
        -------
        pd.DataFrame con columnas: creative_id, clip_0 … clip_N
        """
        ids, embeddings = [], []

        for start in tqdm(
            range(0, len(df), self.config.batch_size),
            desc="CLIP embeddings",
        ):
            batch = df.iloc[start : start + self.config.batch_size]
            batch_ids, batch_embs = self._process_batch(batch)
            ids.extend(batch_ids)
            embeddings.extend(batch_embs)

        emb_arr = np.array(embeddings)
        dim = emb_arr.shape[1]
        emb_df = pd.DataFrame(emb_arr, columns=[f"clip_{i}" for i in range(dim)])
        emb_df.insert(0, self.config.creative_id_column, ids)
        logger.info(f"CLIP embeddings: {emb_df.shape[0]} creatividades, dim={dim}")
        return emb_df

    # ─────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────

    def _process_batch(self, batch: pd.DataFrame):
        ids, images = [], []
        for _, row in batch.iterrows():
            cid = row[self.config.creative_id_column]
            img_path = row["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                ids.append(cid)
            except Exception as exc:
                logger.warning(f"CLIP skip {cid}: {exc}")

        if not images:
            return [], []

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            feats = self.model.get_image_features(pixel_values=pixel_values)

            # Fix defensivo
            if not isinstance(feats, torch.Tensor):
                if hasattr(feats, "image_embeds"):
                    feats = feats.image_embeds
                elif hasattr(feats, "pooler_output"):
                    feats = self.model.visual_projection(feats.pooler_output)
                else:
                    raise TypeError(f"Salida CLIP inesperada: {type(feats)}")

            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        embs = feats.cpu().numpy().tolist()
        return ids, embs



# ─────────────────────────────────────────────────────────────────────────────
# CNN Extractor
# ─────────────────────────────────────────────────────────────────────────────

class CNNEmbeddingExtractor:
    """
    Extrae embeddings de la penúltima capa de una CNN pretrained.

    Modelos soportados (config.cnn_model_name):
      - "resnet50"
      - "efficientnet_b0"
      - "efficientnet_b3"
      - "convnext_tiny"

    Es fácil añadir más modelos extendiendo ``_build_model``.

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Cargando modelo CNN: {config.cnn_model_name} en {self.device}")
        self.model, self.embed_dim = self._build_model(config.cnn_model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = self._build_transform(config.image_size)

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae embeddings CNN para todas las imágenes.

        Returns
        -------
        pd.DataFrame con columnas: creative_id, cnn_0 … cnn_N
        """
        ids, embeddings = [], []

        for start in tqdm(
            range(0, len(df), self.config.batch_size),
            desc="CNN embeddings",
        ):
            batch = df.iloc[start : start + self.config.batch_size]
            batch_ids, batch_embs = self._process_batch(batch)
            ids.extend(batch_ids)
            embeddings.extend(batch_embs)

        emb_arr = np.array(embeddings)
        dim = emb_arr.shape[1] if emb_arr.ndim == 2 else self.embed_dim
        emb_df = pd.DataFrame(emb_arr, columns=[f"cnn_{i}" for i in range(dim)])
        emb_df.insert(0, self.config.creative_id_column, ids)
        logger.info(f"CNN embeddings: {emb_df.shape[0]} creatividades, dim={dim}")
        return emb_df

    # ─────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────

    def _process_batch(self, batch: pd.DataFrame):
        ids, tensors = [], []
        for _, row in batch.iterrows():
            cid = row[self.config.creative_id_column]
            img_path = row["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                tensors.append(self.transform(img))
                ids.append(cid)
            except Exception as exc:
                logger.warning(f"CNN skip {cid}: {exc}")

        if not tensors:
            return [], []

        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self.model(batch_tensor)

        return ids, embs.cpu().numpy().tolist()

    # ─────────────────────────────────────────────────────────────────────
    # Construcción de modelos (extensible)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_model(model_name: str):
        """
        Carga el backbone y elimina la cabeza de clasificación.

        Para añadir un nuevo modelo basta con añadir un bloque elif aquí
        y devolver (model_sin_cabeza, dim_embedding).
        """
        if model_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Eliminar la capa fc final → salida de avgpool (2048-d)
            model = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
            embed_dim = 2048

        elif model_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            # classifier es Sequential([Dropout, Linear]) → lo quitamos
            base.classifier = nn.Identity()
            model = base
            embed_dim = 1280

        elif model_name == "efficientnet_b3":
            base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            base.classifier = nn.Identity()
            model = base
            embed_dim = 1536

        elif model_name == "convnext_tiny":
            base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            # Quitar head.fc
            base.classifier = nn.Sequential(base.classifier[0], nn.Flatten())
            model = base
            embed_dim = 768

        else:
            raise ValueError(
                f"CNN '{model_name}' no soportado. "
                "Opciones: resnet50, efficientnet_b0, efficientnet_b3, convnext_tiny"
            )

        return model, embed_dim

    @staticmethod
    def _build_transform(image_size: int) -> transforms.Compose:
        """Preprocesado estándar ImageNet para CNN."""
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
