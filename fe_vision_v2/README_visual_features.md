# Smadex Creative Intelligence — Visual Feature Pipeline

Pipeline modular de feature engineering visual para el Smadex Creative Intelligence Challenge.  
Extrae embeddings (CLIP, CNN) y ~45 features interpretables de imágenes publicitarias, aplica PCA y genera visualizaciones exploratorias.

---

## Estructura del proyecto

```
smadex_visual/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuración centralizada (dataclass)
│   ├── data_loader.py             # Carga y merge de CSVs + resolución de rutas
│   ├── image_validator.py         # Validación de imágenes + informe de calidad
│   ├── visual_features.py         # Features interpretables con OpenCV
│   ├── embedding_extractors.py    # Embeddings CLIP y CNN
│   ├── pca_reducer.py             # PCA sobre embeddings
│   ├── visualizer.py              # UMAP/t-SNE + distribuciones
│   └── pipeline.py                # Orquestador principal
├── scripts/
│   └── run_visual_feature_pipeline.py   # Punto de entrada CLI
├── requirements.txt
└── README_visual_features.md
```

---

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota GPU**: Si tienes CUDA, instala primero PyTorch con soporte CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/).  
> El pipeline detecta automáticamente GPU/CPU.

---

## Ejecución

```bash
python scripts/run_visual_feature_pipeline.py \
    --data_dir "C:\Users\sylbo\Documents_local\hack_UPC_2026\Smadex_Creative_Intelligence_Challenge" \
    --output_dir "outputs" \
    --clip_model "openai/clip-vit-base-patch32" \
    --cnn_model "resnet50" \
    --clip_pca_components 64 \
    --cnn_pca_components 64
```

### Parámetros disponibles

| Parámetro | Default | Descripción |
|---|---|---|
| `--data_dir` | _(requerido)_ | Carpeta raíz del dataset |
| `--output_dir` | `outputs` | Directorio de salida |
| `--clip_model` | `openai/clip-vit-base-patch32` | Modelo CLIP de HuggingFace |
| `--cnn_model` | `resnet50` | `resnet50` / `efficientnet_b0` / `efficientnet_b3` / `convnext_tiny` |
| `--clip_pca_components` | `64` | Componentes PCA para CLIP |
| `--cnn_pca_components` | `64` | Componentes PCA para CNN |
| `--batch_size` | `32` | Batch size para extracción |
| `--image_size` | `224` | Resize para CNN |
| `--log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `--asset_path_column` | `asset_path` | Columna de ruta en creatives.csv |
| `--creative_id_column` | `creative_id` | Columna de ID en creatives.csv |
| `--campaign_id_column` | `campaign_id` | Columna de campaña |

---

## Outputs generados

```
outputs/
├── image_quality_report.csv                           # Informe de validación
├── creative_visual_features.parquet                   # Features interpretables + metadatos
├── creative_clip_embeddings.parquet                   # Embeddings CLIP (512-d)
├── creative_cnn_embeddings.parquet                    # Embeddings CNN (2048-d ResNet50)
├── creative_visual_features_with_embeddings_pca.parquet  # Todo unificado con PCA
├── models/
│   ├── pca_clip.joblib                                # Modelo PCA CLIP
│   └── pca_cnn.joblib                                 # Modelo PCA CNN
└── figures/
    ├── clip_umap_campaign_id.png
    ├── clip_umap_creative_status.png
    ├── cnn_umap_campaign_id.png
    ├── feature_distributions.png
    └── feature_correlation_heatmap.png
```

### Descripción de cada archivo

| Archivo | Descripción |
|---|---|
| `image_quality_report.csv` | Una fila por creative con `status` (ok/missing/corrupt), dimensiones |
| `creative_visual_features.parquet` | ~45 features visuales interpretables + metadatos de los CSVs |
| `creative_clip_embeddings.parquet` | Embeddings 512-d de CLIP, normalizados L2 |
| `creative_cnn_embeddings.parquet` | Embeddings 2048-d de ResNet50 (penúltima capa) |
| `creative_visual_features_with_embeddings_pca.parquet` | Features + PCA CLIP (64-d) + PCA CNN (64-d) + metadatos |
| `models/pca_*.joblib` | Modelos PCA serializados para inferencia futura |
| `figures/*.png` | Visualizaciones UMAP/t-SNE y distribuciones |

---

## Features visuales extraídas

### Geometría básica
`width`, `height`, `aspect_ratio`, `image_area`

### Color
`brightness_mean/std`, `contrast`, `saturation_mean/std`, `hue_mean/std`,  
`dominant_color_r/g/b/h/s/v`, `colorfulness`

### Textura
`edge_density` (Canny), `visual_complexity` (entropía), `sharpness_laplacian_var`

### Layout y localización
`centroid_x/y_visual_mass`, `visual_mass_top/bottom/left/right/center/border_ratio`,  
`quadrant_*_density`, `symmetry_horizontal/vertical_score`,  
`main_object_bbox_*`, `main_object_is_centered`,  
`saliency_center_bias`, `background_uniformity_score`, `empty_space_ratio`

---

## Flujo de la pipeline

```
creatives.csv + creative_summary.csv
          │
          ▼
   CreativeDataLoader           ← merge + resolución de rutas
          │
          ▼
    ImageValidator              ← ok / missing / corrupt → image_quality_report.csv
          │
     (válidas)
     ┌────┴────────────────┐
     ▼                     ▼
HandcraftedVisual    CLIPEmbedding    CNNEmbedding
FeatureExtractor     Extractor        Extractor
     │                     │               │
     ▼                     ▼               ▼
visual_features      clip_embeddings  cnn_embeddings
  .parquet              .parquet         .parquet
     │                     │               │
     └──────────┬───────────┘               │
                ▼                           ▼
           PCAReducer                  PCAReducer
          (clip→64d)                  (cnn→64d)
                │                           │
                └─────────────┬─────────────┘
                              ▼
              creative_visual_features_with_embeddings_pca.parquet
                              │
                              ▼
                    EmbeddingVisualizer
                    (UMAP/t-SNE + plots)
```

---

## Cambiar de modelo CNN

En la llamada CLI, basta con:

```bash
--cnn_model efficientnet_b0
# o
--cnn_model convnext_tiny
```

Para añadir un nuevo backbone, edita `src/embedding_extractors.py` → método `_build_model` y añade un bloque `elif`.

---

## Requisitos del sistema

- Python 3.10+
- 8 GB RAM mínimo (16 GB recomendado para datasets grandes)
- GPU opcional (CUDA ≥ 11.8) — el pipeline es funcional en CPU

---

## Notas sobre las heurísticas visuales

Las features de layout usan un **mapa de activación** combinado:

```
activation = 0.6 × edge_map + 0.4 × saturation_norm
```

Esto captura regiones visualmente activas sin necesitar segmentación semántica.  
La "masa visual" en cada región (top, bottom, cuadrantes…) es la suma de activación normalizada.  
El bounding box del objeto principal se obtiene con `findContours` sobre un umbral adaptativo de este mapa.

Son heurísticas suficientemente robustas y rápidas para un hackathon, con alta interpretabilidad.
