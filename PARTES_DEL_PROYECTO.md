# Partes del proyecto

## 1) Lanzador unificado
- build_visual_features.py
- Punto de entrada único para ejecutar toda la pipeline.

## 2) Paquete visual_features

### cli.py
- Define todos los argumentos de línea de comandos.

### devices.py
- Selecciona automáticamente CPU/GPU (cuda/mps/cpu).

### data_io.py
- Carga CSVs.
- Valida estructura mínima de datos.
- Resuelve rutas de imágenes.
- Valida/carga imágenes y guarda parquet.

### image_features.py
- Extrae features interpretables de imagen:
  - brightness_mean
  - saturation_mean
  - contrast
  - dominant_h, dominant_s, dominant_v
  - edge_density
  - visual_complexity
  - empty_space_ratio
  - aspect_ratio, width, height

### ocr_features.py
- OCR opcional con easyocr o pytesseract.
- Genera:
  - word_count
  - char_count
  - has_price
  - has_discount
  - has_cta
  - has_urgency_terms

### models.py
- Carga modelos pretrained:
  - CLIP (openai/clip-vit-base-patch32)
  - ResNet50 (torchvision)
- Extrae embeddings CLIP y ResNet.

### dimensionality.py
- Reduce embeddings con PCA (64 por defecto).

### pipeline.py
- Orquesta el flujo completo:
  - lectura de datos
  - validación de imágenes
  - extracción de features y embeddings
  - OCR opcional
  - PCA
  - merge con creative_summary
  - guardado de archivos finales

### main.py
- Punto de entrada interno del paquete.

## 3) Outputs generados
- creative_image_features.parquet
- creative_image_embeddings_raw.parquet
- creative_model_features.parquet
