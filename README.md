# Creative Memory Copilot - Smadex Creative Intelligence

Proyecto final del HackUPC 2026 para el reto de Smadex: construir un copiloto que ayude a un marketer a entender que creatividades funcionan, cuales se estan cansando o repitiendo, y que deberia testear despues.

La idea central no es un clasificador opaco, sino una memoria creativa explicable. Cada creative es un caso historico multimodal: datos tabulares, KPIs temporales, imagen, embeddings visuales, features de layout, OCR/grounding con Florence-2 y textos de anuncio. Sobre esa memoria se construye un CBR, Case-Based Reasoning, que recupera casos parecidos, compara patrones ganadores y genera acciones.

## Lo diferencial

El proyecto no se queda en "ranking por ROAS". La parte fuerte es el join multimodal con EDA y una memoria creativa explicable:

- EDA serio de leakage: se separan metadata, features disponibles antes o durante los primeros dias, y outcomes de todo el ciclo de vida.
- Vision por varias capas: features interpretables con OpenCV, embeddings CNN ResNet50, embeddings CLIP y Florence-2 como capa semantica/espacial.
- Florence-2 como diferencial: OCR con regiones, phrase grounding y spatial fingerprints para saber donde estan texto, CTA, producto, logo, badges, jerarquia visual y zonas de composicion.
- CBR multimodal: no predice solo un numero; busca casos historicos parecidos y explica por bloques por que son similares.
- Dashboard final: tres pantallas para responder las tres preguntas del challenge, mas mapa 2D y busqueda por brief.

## Dataset

Datos originales en `Smadex_Creative_Intelligence_Dataset_FULL/`:

| Archivo | Filas x columnas | Uso |
|---|---:|---|
| `advertisers.csv` | 36 x 4 | contexto de anunciante |
| `campaigns.csv` | 180 x 14 | configuracion de campanas |
| `campaign_summary.csv` | 180 x 22 | KPIs agregados de campana |
| `creatives.csv` | 1080 x 32 | metadata y assets de cada creative |
| `creative_summary.csv` | 1080 x 59 | KPIs agregados por creative |
| `creative_daily_country_os_stats.csv` | 192315 x 14 | serie diaria por creative, pais y OS |
| `assets/*.png` | 1080 imagenes | material visual del anuncio |

La tabla final de casos esta en:

```text
dataset/final/creative_cbr_cases_final.csv
dataset/final/creative_cbr_cases_final.parquet
dataset/final/cbr_feature_sets.json
dataset/final/creative_memory.db
```

El EDA avanzado parte de una tabla multimodal de `1080 x 401` y produce una tabla final CBR de `1080 x 336`. La seleccion queda trazada en:

```text
dataset/output/advanced_eda_summary.json
dataset/output/advanced_eda_column_audit.csv
dataset/output/advanced_eda_reduction_summary.csv
dataset/output/advanced_eda_high_corr_pairs.csv
```

## Pipeline completa

```text
Raw CSVs + assets
        |
        v
Feature preprocessing tabular
        |
        +--> daily fact limpia a creative_id x date
        +--> first_3d / first_7d / first_14d
        +--> lifecycle targets separados
        |
        v
Vision v2
        |
        +--> OpenCV interpretable features
        +--> ResNet50 embeddings -> PCA 64
        +--> CLIP embeddings -> PCA 64
        |
        v
Florence-2 spatial fingerprint
        |
        +--> OCR con region
        +--> phrase grounding
        +--> layout, CTA, logo, producto, badges, jerarquia
        |
        v
Join tabular + vision + CV prompt
        |
        v
EDA avanzado multimodal
        |
        +--> auditoria de columnas
        +--> leakage control
        +--> columnas correlacionadas/duplicadas
        +--> feature sets por ventana temporal
        |
        v
CBR multimodal
        |
        +--> indice de similitud por bloques
        +--> vecinos historicos
        +--> explicabilidad por bloque
        |
        v
Preguntas del challenge + dashboard web
```

## Feature Engineering

### 1. Datos tabulares

Se unieron advertisers, campaigns, creatives, summaries y daily stats. El grano diario original es:

$$creative\_id \times date \times country \times os$$

Para modelado se agrego tambien a:

$$creative\_id \times date$$

Desde ahi se crearon ventanas tempranas:

- `first_3d_*`: evidencia disponible tras 3 dias.
- `first_7d_*`: evidencia disponible tras 7 dias.
- `first_14d_*`: evidencia disponible tras 14 dias.
- `lifecycle_*` y `overall_*`: outcomes historicos completos, usados para explicar y evaluar, no para similitud prelaunch.

Tambien se extrajeron variables de texto y creatividad:

- longitud de headline, subhead y CTA;
- flags de accion, urgencia, beneficio, trust y social proof;
- target age parseado;
- contexto de campana, vertical, formato, idioma, OS, objetivo y presupuesto.

### 2. Vision interpretable

En `fe_vision_v2/` se extraen features visuales con OpenCV:

- geometria: `width`, `height`, `aspect_ratio`, `image_area`;
- color: brillo, saturacion, hue, color dominante, colorfulness;
- textura: edge density, complejidad visual, nitidez;
- layout: centro de masa visual, distribucion por cuadrantes, simetria, empty space, main object bbox.

Estas variables son importantes porque se pueden explicar a negocio: "mas clutter", "producto mas centrado", "CTA en zona baja", "mas espacio vacio", etc.

### 3. CNN ResNet50

Se usa ResNet50 como extractor visual generalista. La salida original es de 2048 dimensiones y se reduce con PCA a 64:

```text
fe_vision_v2/outputs/creative_cnn_embeddings.parquet
fe_vision_v2/outputs/models/pca_cnn.joblib
```

Por que usarlo: ResNet50 captura textura, formas, composicion y patrones visuales locales. Es util para detectar anuncios visualmente parecidos aunque no compartan el mismo texto o concepto semantico.

### 4. CLIP

Se usa `openai/clip-vit-base-patch32`, tambien reducido a 64 componentes:

```text
fe_vision_v2/outputs/creative_clip_embeddings.parquet
fe_vision_v2/outputs/models/pca_clip.joblib
```

Por que usarlo: CLIP alinea imagen y lenguaje. Captura mejor el concepto del anuncio: gaming, comida, viaje, producto, UGC, promo, estilo, etc. Fue especialmente util para detectar repeticion creativa y near duplicates. En el analisis de similitud CLIP:

- `1080` creatives;
- `64` dimensiones PCA;
- similitud media aleatoria casi `0.00018`;
- similitud media del vecino mas cercano `0.9017`;
- `195` pares near-duplicate con umbral `0.98`.

Esto indica que CLIP separa bien el espacio creativo y descubre familias visuales repetidas.

### 5. Florence-2, la capa diferencial

En `cv/` se usa `microsoft/Florence-2-base` sobre los 1080 creatives. Florence-2 genera:

- OCR con regiones: donde esta cada texto y cuanto ocupa.
- Phrase grounding: localizacion de CTA, producto, logo, badges y elementos visuales.
- JSON estructurado por creative.
- Spatial fingerprint con features como `text_area_total_rel`, `cta_y_relative`, `logo_in_top_left`, `product_zone_coverage`, `visual_hierarchy_score`, `brand_clarity_score`, `whitespace_balance`.

Artefactos:

```text
cv/output/creative_spatial_features.parquet
cv/output/creative_spatial_with_kpis.parquet
cv/output/creative_prompt_analysis_florence.jsonl
cv/output/creative_spatial_elements.jsonl
cv/output/spatial_kpi_correlations_strong.csv
cv/output/visualizaciones/*_visualized.png
```

Lo importante: Florence-2 permite pasar de "esta imagen se parece" a "esta imagen tiene el CTA abajo, mucho texto, producto centrado, badge arriba a la derecha y marca visible". Esa explicabilidad visual es lo que hace que el sistema sea util para decisiones creativas.

Algunos hallazgos fuertes del cruce Florence/KPI:

- En fintech, mayor `text_area_total_rel` se asocia con mayor `overall_cvr` (`spearman=0.547`, `n=180`).
- En travel, mayor `text_area_total_rel` se asocia con mayor `overall_cvr` (`spearman=0.495`).
- En gaming, mayor `avg_text_area_rel` se asocia con mayor `overall_cvr` (`spearman=0.477`).
- En global, mayor `image_aspect_ratio` se asocia con menor `overall_ctr` (`spearman=-0.463`).
- En global, mayor `ocr_element_count` se asocia con menor `overall_cvr` (`spearman=-0.450`).

No se usan como causalidad absoluta, sino como evidencia para generar hipotesis de test.

## EDA y trampas encontradas

Esta parte fue clave. El dataset tiene columnas muy tentadoras que mejoran cualquier modelo, pero algunas no son validas segun el momento de decision.

### Trampa 1: `impressions_last_7d`

En `creative_daily_country_os_stats.csv` existe `impressions_last_7d`. El preprocessing la elimina porque el EDA detecto que es un rolling de 7 filas anteriores, no necesariamente de 7 dias calendario. Como el grano incluye pais y OS, 7 filas no equivalen a 7 dias.

Decision:

```text
drop impressions_last_7d
recalcular ventanas temporales desde creative_day
```

### Trampa 2: `last_7d_*`

En `creative_summary.csv` aparecen columnas como:

```text
last_7d_ctr
last_7d_cvr
last_7d_impressions
last_7d_clicks
last_7d_conversions
```

Son muy informativas, pero representan el final de la vida del creative. Para una decision prelaunch o early no se pueden usar: son leakage. En el proyecto se conservan como evidencia historica si hace falta, pero no entran en los feature sets de similitud.

### Trampa 3: outcomes con pinta de features

Columnas como `creative_status`, `fatigue_day`, `perf_score`, `overall_roas`, `overall_ipm`, `overall_ctr`, `ctr_decay_pct`, `lifecycle_*` son necesarias para responder preguntas, pero no para buscar vecinos cuando simulamos una decision previa al resultado.

### Trampa 4: correlaciones perfectas y duplicados

El EDA detecto redundancias fuertes:

- `width`, `image_quality_width`, `cv_image_width`, `prompt_image_width`: correlacion `1.0`.
- `height`, `asset_pixels`, `vis_image_area`, `prompt_image_height`: correlacion `1.0`.
- `text_density` y `cv_text_density_meta`: correlacion `1.0`.
- `has_price` y `cv_has_price_meta`: correlacion `1.0`.
- `headline_chars` y `txtsel_headline_length`: correlacion `1.0`.

Por eso se genera `advanced_eda_reduction_summary.csv`:

| Metrica | Valor |
|---|---:|
| columnas de entrada | 401 |
| drop always | 39 |
| candidatas correlacionadas | 44 |
| columnas finales de caso | 336 |
| features candidatas de similitud | 287 |
| features prelaunch | 257 |
| features early 3d | 270 |
| features early 7d | 281 |
| features early 14d | 287 |

## Metadata JSON y feature sets

El contrato del modelo esta en `dataset/final/cbr_feature_sets.json`. La idea es que una columna puede estar en la tabla final, pero no necesariamente se puede usar para todos los modelos.

Resumen conceptual:

```json
{
  "metadata_keep_cols": [
    "creative_id", "campaign_id", "advertiser_id", "asset_file",
    "advertiser_name", "app_name", "vertical", "format", "language",
    "theme", "hook_type", "cta_text", "headline", "subhead"
  ],
  "target_cols": [
    "creative_status", "fatigue_day", "has_fatigue", "perf_score",
    "overall_ctr", "overall_cvr", "overall_ipm", "overall_roas",
    "ctr_decay_pct", "cvr_decay_pct", "peak_rolling_ctr_5"
  ],
  "full_period_outcome_cols": [
    "lifecycle_days", "lifecycle_spend_usd", "lifecycle_impressions",
    "lifecycle_clicks", "lifecycle_conversions", "lifecycle_revenue_usd",
    "lifecycle_ctr", "lifecycle_cvr", "lifecycle_ipm", "lifecycle_roas"
  ],
  "feature_sets": {
    "prelaunch_feature_cols": "solo datos disponibles antes de lanzar",
    "early_3d_feature_cols": "prelaunch + first_3d_*",
    "early_7d_feature_cols": "prelaunch + first_3d_* + first_7d_*",
    "early_14d_feature_cols": "prelaunch + first_3d_* + first_7d_* + first_14d_*"
  }
}
```

En la pipeline final de la app se usa `prelaunch_feature_cols`. Esto fuerza una comparacion limpia: el CBR no mira el resultado futuro para decidir que creatividades son similares.

## Modelo inteligente: CBR explicable

El CBR vive en `cbr_engine/`. Cada caso es un creative. El vector de similitud se compone por bloques:

| Bloque | Peso final | Que captura |
|---|---:|---|
| `clip` | 0.45 | concepto visual-semantico del anuncio |
| `cnn` | 0.25 | similitud visual de textura, formas y composicion |
| `visual_numeric` | 0.20 | Florence-2 + OpenCV + layout interpretable |
| `text_numeric` | 0.05 | rasgos del copy |
| `categorical_context` | 0.05 | vertical, formato, idioma, tema, hook, tono |

La similitud se calcula normalizando cada bloque y combinando similitudes coseno ponderadas:

$$
S(q, c) =
\frac{\sum_b w_b \cdot cos(x_{q,b}, x_{c,b})}{\sum_b w_b}
$$

Cada bloque conserva su similitud individual, de forma que la explicacion puede decir si un match viene de CLIP, CNN, contexto o features visuales.

Artefactos del indice:

```text
outputs/cbr_index_app/
  ids.npy
  matrix.npy
  metadata.csv
  block_matrices/*.npy
  preprocessors.joblib
  index_sklearn.joblib
  build_manifest.json
  feature_report.json
```

El indice final tiene:

- `1080` casos;
- vector multimodal de `386` dimensiones;
- backend sklearn en esta corrida;
- 21600 relaciones de vecinos en SQLite (`20` vecinos por creative).

Metricas offline del smoke evaluation:

| Metrica | Valor |
|---|---:|
| Pearson outcome correlation @5 | 0.549 |
| Spearman outcome correlation @5 | 0.530 |
| label consistency @5 | 0.591 |
| fatigue retrieval consistency @5 | 0.749 |
| NDCG @5 | 0.771 |
| self-neighbor violations | 0 |

## Respuesta a las tres preguntas

### 1. Which creatives are working best?

Modulo: `creative_intelligence/questions.py` y outputs en `best_creatives/outputs/`.

Se construye un ranking robusto, no solo ROAS. El score combina percentiles de:

$$
best\_score =
0.40 \cdot perf
+ 0.25 \cdot roas
+ 0.20 \cdot ipm
+ 0.10 \cdot cvr
+ 0.05 \cdot ctr
+ status\_bonus
$$

Antes se aplica filtro de evidencia por spend o impresiones para evitar declarar winner a un creative con poca muestra.

Outputs:

```text
best_creatives/outputs/question1_winner_ranking.csv
best_creatives/outputs/question1_top_global.csv
best_creatives/outputs/question1_top_by_campaign.csv
best_creatives/outputs/question1_top_by_vertical.csv
best_creatives/outputs/question1_top_by_vertical_format.csv
best_creatives/outputs/question1_winner_explanations.jsonl
```

Ejemplo de top global real:

- `500634`, travel, rewarded video, `best_score=0.918`, accion `Scale`.
- `500685`, gaming, rewarded video, `best_score=0.915`, accion `Scale`.
- `500129`, travel, rewarded video, `best_score=0.908`, accion `Scale`.

El CBR no decide el winner; lo explica. Para cada winner busca similares y resume cuantos vecinos fueron top performers, stable o fatigued.

### 2. Which creatives look repetitive or tired?

Modulo: `creative_intelligence/questions.py` y outputs en `fatigue_repetition/outputs/`.

Se calculan dos riesgos:

Fatiga:

$$
tired =
0.35 \cdot fatigue\_label
+ 0.25 \cdot ctr\_decay
+ 0.20 \cdot cvr\_decay
+ 0.20 \cdot early\_fatigue
$$

Repeticion:

$$
repetition =
0.50 \cdot similarity\_density
+ 0.25 \cdot same\_metadata
+ 0.25 \cdot neighbor\_density
$$

Outputs:

```text
fatigue_repetition/outputs/question2_creative_health.csv
fatigue_repetition/outputs/question2_tired_creatives.csv
fatigue_repetition/outputs/question2_repetitive_creatives.csv
fatigue_repetition/outputs/question2_health_summary.csv
fatigue_repetition/outputs/question2_health_explanations.jsonl
```

Resumen de salud:

| Estado | Creatives |
|---|---:|
| healthy + distinct | 604 |
| healthy + moderate_repetition | 151 |
| at_risk + distinct | 89 |
| tired + distinct | 71 |
| tired + moderate_repetition | 64 |
| healthy + high_repetition | 52 |
| tired + high_repetition | 14 |

### 3. What should we test next?

Modulo: `creative_intelligence/questions.py` y outputs en `next_tests/outputs/`.

Para cada creative:

1. Busca vecinos similares con CBR.
2. Separa vecinos `top_performer`.
3. Compara el creative actual contra los winners parecidos.
4. Detecta diferencias accionables: formato, hook, CTA, layout, clutter, brand clarity, jerarquia, texto, producto.
5. Propone accion: `Scale`, `Keep`, `Refresh`, `Pause or rebuild`, `Test next variation`, `Gather evidence`.

Outputs:

```text
next_tests/outputs/question3_next_tests.csv
next_tests/outputs/question3_priority_tests.csv
next_tests/outputs/question3_recommendation_cards.jsonl
```

Importante: la recomendacion no copia el vecino mas cercano. Usa el vecino como evidencia, pero decide por patron agregado de winners similares.

## Dashboard y web

La app esta en `app/` y se arranca con:

```powershell
python -m app.api
```

URL local:

```text
http://127.0.0.1:8000
```

La base de datos es:

```text
dataset/final/creative_memory.db
```

Tablas principales:

| Tabla | Filas |
|---|---:|
| `creatives` | 1080 |
| `question1_winner_ranking` | 1080 |
| `question2_creative_health` | 1080 |
| `question3_next_tests` | 1080 |
| `creative_neighbors` | 21600 |
| `creative_explanations` | 1080 |
| `creative_landscape` | 1080 |

### Pantalla 1: Best creatives

Responde "Which creatives are working best?". Muestra ranking, creative, KPIs, accion recomendada y explicacion. Al seleccionar un caso:

- carga imagen original;
- muestra vecinos similares;
- muestra similitud por bloques CLIP/CNN/contexto;
- explica por que el creative funciona.

### Pantalla 2: Worst / tired

Responde "Which creatives look repetitive or tired?". Ordena por riesgo de fatiga y repeticion. Para cada caso muestra:

- `tired_score`;
- `repetition_score`;
- estado de fatiga;
- estado de repeticion;
- vecinos parecidos;
- que se hizo bien y que se esta saturando.

### Pantalla 3: Next test

Responde "What should we test next?". Permite escribir un brief, por ejemplo:

```text
Quiero hacer un anuncio de Gaming con gameplay visible, recompensa y CTA directo.
```

El sistema infiere vertical, busca winners cercanos y weak cases cercanos, y devuelve:

- patrones a hacer;
- patrones a evitar;
- casos ganadores de referencia;
- casos debiles a no copiar;
- explicacion con Groq si hay API key.

### Mapa de creatives

La app incluye `Creative landscape`, un mapa 2D generado desde el mismo vector multimodal del CBR. Usa UMAP si esta disponible y t-SNE/PCA como fallback. Permite ver:

- todos los casos;
- clusters por similitud multimodal;
- estado top/stable/fatigued/underperformer;
- caso seleccionado;
- vecinos CBR;
- matches del brief.

## Explicabilidad

Hay dos niveles:

1. Explicabilidad de similitud: el CBR devuelve `similarity_clip`, `similarity_cnn`, `similarity_visual_numeric`, `similarity_text_numeric` y `similarity_categorical_context`.
2. Explicabilidad visual: las features de Florence-2 y OpenCV transforman la imagen en descriptores accionables: CTA, layout, texto, producto, logo, badges, jerarquia visual y whitespace.

Esto permite frases como:

```text
Este caso se parece sobre todo por CLIP y visual_numeric:
comparte concepto de travel/rewarded, CTA bajo, producto claro y baja complejidad.
Los winners parecidos tienen menos clutter y mayor claridad de marca.
```

## Como ejecutar

Instalar dependencias:

```powershell
pip install -r requirements.txt
pip install -r app/requirements.txt
```

Reconstruir preguntas y base de datos:

```powershell
python -m creative_intelligence.run_questions --cbr-backend engine
```

Arrancar app:

```powershell
python -m app.api
```

Construir indice CBR manualmente:

```powershell
python scripts/build_cbr_index.py --data_path dataset/final/creative_cbr_cases_final.parquet --feature_sets_path dataset/final/cbr_feature_sets.json --config configs/cbr_prelaunch.yaml --output_dir outputs/cbr_index --force
```

Consultar un caso:

```powershell
python scripts/query_cbr.py --creative_id 500634 --index_dir outputs/cbr_index --config configs/cbr_prelaunch.yaml --k 10 --same_vertical --same_format
```

## Estructura del repositorio

```text
app/                         API local + frontend
best_creatives/              outputs pregunta 1
cbr_engine/                  motor CBR multimodal
configs/                     configuraciones CBR por modo/modelo
creative_intelligence/       pipeline de preguntas y DB final
cv/                          Florence-2 spatial fingerprint
dataset/                     joins, EDA avanzado y tabla final
fatigue_repetition/          outputs pregunta 2
feature/                     preprocessing tabular y EDA inicial
fe_vision_v2/                CLIP, ResNet50, OpenCV, PCA, UMAP
next_tests/                  outputs pregunta 3
outputs/                     indices, evaluaciones, calibraciones
scripts/                     CLIs para CBR
tests/                       tests del motor CBR
```

## Conclusion

El resultado final es un Creative Memory Copilot: una herramienta que combina vision, datos tabulares, EDA de leakage y razonamiento por casos para convertir datos historicos en decisiones creativas. El valor diferencial esta en que no solo dice "este anuncio va bien", sino que ensena los casos parecidos, explica que elementos visuales y de contexto sostienen la decision, detecta fatiga/repeticion y propone el siguiente test con evidencia.
