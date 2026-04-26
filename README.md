# Advanced Creative CBR Engine

Este paquete implementa un sistema CBR estatico para Creative Intelligence en AdTech. No es un modelo predictivo clasico: no aprende una funcion cerrada `X -> y`, sino que recupera casos historicos parecidos, explica por que son parecidos y reutiliza sus outcomes como evidencia para decidir `SCALE`, `PAUSE`, `PIVOT` o `INVESTIGATE`.

## Diagnostico del diseno anterior

La version simple en `creative_intelligence/cbr.py` era util como baseline, pero tenia limites importantes:

- Pesos fijos manuales: `context_weight`, `numeric_weight` y `embedding_weight` eran arbitrarios. Podian devolver vecinos visualmente parecidos con contexto irrelevante, o contexto correcto pero bajo parecido creativo.
- Riesgo de leakage: para `prelaunch` no deben usarse outcomes futuros como `creative_status`, `fatigue_day`, `has_fatigue`, `perf_score`, `overall_ctr`, `overall_cvr`, `overall_ipm`, `overall_roas`, `ctr_decay_pct`, `cvr_decay_pct` o `peak_rolling_ctr_5`. Tambien se excluyen `lifecycle_*` y full-period outcomes. En `prelaunch` se bloquean `first_3d_*`, `first_7d_*` y `first_14d_*`; en `early_3d` se permite `first_3d_*` pero no ventanas futuras; en `early_7d` se permiten `first_3d_*` y `first_7d_*`; en `early_14d` se bloquean lifecycle y overall.
- Sesgo en matching: `campaign_id`, `advertiser_id` y `app_name` pueden dominar la similitud y hacer que el sistema solo encuentre copias internas. Por defecto son metadata/filtros, no features de similitud.
- Sobreajuste del retrieval: calibrar pesos contra todo el dataset infla la calidad. Este repo incluye evaluacion offline y calibracion con splits configurables; se recomienda holdout temporal.
- Baja explicabilidad: una distancia coseno global no explica el match. Ahora se devuelven similitudes por bloque: `clip`, `cnn`, `visual_numeric`, `text_numeric`, `business_early` y `categorical_context`.
- Performance: sklearn brute sirve al inicio, pero se usa FAISS `IndexFlatIP` si esta instalado. Se persisten matriz, ids, metadata, matrices por bloque, preprocessors, indice y manifiestos.

## Modos

- `prelaunch`: solo usa informacion disponible antes del lanzamiento. Excluye `first_3d_*`, `first_7d_*`, `first_14d_*`, `lifecycle_*` y `overall_*`.
- `early_3d`: permite senales de los primeros 3 dias y bloquea ventanas futuras.
- `early_7d`: permite primeros 3 y 7 dias, bloquea 14 dias, lifecycle y overall.
- `early_14d`: permite early windows hasta 14 dias, bloquea lifecycle y overall.

## Como evita leakage

`FeatureBuilder` lee `cbr_feature_sets.json`, selecciona `feature_set_name`, descarta columnas ausentes, outcomes, metadata no permitida, prefijos prohibidos por modo y columnas con demasiados missing. El reporte `feature_report.json` deja trazabilidad de columnas usadas y descartadas.

## Similitud global vs bloques

Cada bloque se imputa, escala y normaliza L2 por separado. Despues se aplica el peso del bloque y se normaliza la matriz final. FAISS usa inner product sobre vectores normalizados, equivalente a cosine. El retriever devuelve `global_similarity` y columnas como `similarity_clip`, `similarity_cnn`, `similarity_visual_numeric`, `similarity_text_numeric` y `similarity_categorical_context`.

## Ejecucion

Arrancar la web con todo ya integrado:

```powershell
python -m app.api
```

La app usa `dataset/final/creative_memory.db`. Si la DB falta o fue generada con el CBR antiguo, se reconstruye automaticamente con `cbr_engine` antes de levantar el servidor.

Construir indice:

```powershell
python scripts/build_cbr_index.py --data_path dataset/final/creative_cbr_cases_final.parquet --feature_sets_path dataset/final/cbr_feature_sets.json --config configs/cbr_prelaunch.yaml --output_dir outputs/cbr_index --force
```

Consultar:

```powershell
python scripts/query_cbr.py --creative_id 12345 --index_dir outputs/cbr_index --config configs/cbr_prelaunch.yaml --k 10 --same_vertical --same_format
```

Evaluar retrieval:

```powershell
python scripts/evaluate_cbr_retrieval.py --data_path dataset/final/creative_cbr_cases_final.parquet --feature_sets_path dataset/final/cbr_feature_sets.json --config configs/cbr_prelaunch.yaml --output_dir outputs/cbr_eval
```

Calibrar pesos:

```powershell
python scripts/calibrate_cbr_weights.py --data_path dataset/final/creative_cbr_cases_final.parquet --feature_sets_path dataset/final/cbr_feature_sets.json --config configs/cbr_prelaunch.yaml --output_dir outputs/cbr_calibration --n_trials 100
```

## Metricas offline

`evaluation.py` calcula correlacion entre outcome real y outcome agregado de vecinos, consistencia de label, `ndcg_at_k`, hit rate de top performers y consistencia de fatiga. Tambien valida que el propio query no aparezca como vecino.

## Calibracion de pesos

`weight_calibration.py` usa random search con pesos normalizados para los bloques activos. Optimiza `neighbor_outcome_correlation` o `ndcg_at_k` segun config y guarda `calibration_trials.parquet`, `best_weights.json` y `calibration_report.md`. Para evitar sobreajuste, usa la evaluacion offline como criterio y el README recomienda reservar un holdout temporal final.

## Integracion en Streamlit

Desde Streamlit puedes cargar el retriever una vez con cache:

```python
import streamlit as st
from cbr_engine import CBRConfig, load_retriever

@st.cache_resource
def get_retriever():
    cfg = CBRConfig.from_yaml("configs/cbr_prelaunch.yaml")
    return load_retriever("outputs/cbr_index", cfg)

retriever = get_retriever()
neighbors, trace = retriever.retrieve_by_id(creative_id, k=10, filters={"same_vertical": True})
enriched = retriever.enrich(neighbors)
st.dataframe(neighbors)
st.write(enriched["recommendation"])
st.write(enriched["explanation"]["marketer_explanation"])
```

## Artefactos persistidos

El build guarda `index.faiss` o `index_sklearn.joblib`, `ids.npy`, `matrix.npy`, `metadata.parquet`, `block_matrices/*.npy`, `preprocessors.joblib`, `feature_report.json`, `config_resolved.yaml`, `build_manifest.json` y `build_report.md`.
