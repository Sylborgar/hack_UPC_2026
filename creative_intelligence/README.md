# Creative Intelligence Pipeline

Este modulo responde las tres preguntas del challenge usando una unica tabla limpia:

```text
dataset/final/creative_cbr_cases_final.parquet
dataset/final/cbr_feature_sets.json
```

## Pregunta 1: Which creatives are working best?

Nucleo: ranking robusto con filtro de confianza.

Usa outcomes de performance para ordenar:

- `perf_score`
- `overall_roas`
- `overall_ipm`
- `overall_ctr`
- `overall_cvr`
- `lifecycle_spend_usd`
- `lifecycle_impressions`
- `creative_status`

El CBR no decide el winner. Se usa como capa explicativa: para cada winner, busca casos similares y resume cuantos vecinos fueron `top_performer`, `stable` o `fatigued`.

Outputs:

```text
best_creatives/outputs/question1_winner_ranking.csv
best_creatives/outputs/question1_top_global.csv
best_creatives/outputs/question1_top_by_campaign.csv
best_creatives/outputs/question1_top_by_vertical.csv
best_creatives/outputs/question1_top_by_vertical_format.csv
best_creatives/outputs/question1_winner_explanations.jsonl
```

## Pregunta 2: Which creatives look repetitive or tired?

Nucleo:

- `tired`: senales temporales/de performance.
- `repetitive`: similitud con otros creatives mediante CBR.

Fatiga usa:

- `creative_status`
- `has_fatigue`
- `fatigue_day`
- `ctr_decay_pct`
- `cvr_decay_pct`

Repeticion usa:

- similitud media de vecinos
- densidad de vecinos muy similares
- coincidencia de `format`, `theme`, `hook_type`, `cta_text`
- ratio de vecinos fatigados

Outputs:

```text
fatigue_repetition/outputs/question2_creative_health.csv
fatigue_repetition/outputs/question2_tired_creatives.csv
fatigue_repetition/outputs/question2_repetitive_creatives.csv
fatigue_repetition/outputs/question2_health_summary.csv
fatigue_repetition/outputs/question2_health_explanations.jsonl
```

## Pregunta 3: What should we test next?

Nucleo: CBR como memoria creativa.

Para cada creative:

1. Busca vecinos similares.
2. Separa vecinos `top_performer`.
3. Compara el creative actual contra esos winners.
4. Propone una accion: `Scale`, `Keep`, `Refresh`, `Pause or rebuild`, `Test next variation`, `Gather evidence`.
5. Genera una tarjeta estructurada para que un LLM redacte la explicacion sin inventar.

Outputs:

```text
next_tests/outputs/question3_next_tests.csv
next_tests/outputs/question3_priority_tests.csv
next_tests/outputs/question3_recommendation_cards.jsonl
```

## Ejecutar

```bash
.venv/bin/python -m creative_intelligence.run_questions
```

Por defecto, el CBR usa `prelaunch_feature_cols`, que evita usar performance futura para calcular similitud.

La ejecucion tambien crea una base SQLite para la app:

```text
dataset/final/creative_memory.db
```

Tablas principales:

```text
creatives
question1_winner_ranking
question2_creative_health
question3_next_tests
creative_neighbors
creative_explanations
creative_landscape
metadata
```

La recomendacion de `question3_next_tests` no copia el vecino mas parecido. Usa el vecino mas cercano como evidencia, pero la propuesta se basa en el patron agregado de vecinos similares que fueron `top_performer`.

El mapa `creative_landscape` usa UMAP si esta instalado. Si no, genera una proyeccion 2D con t-SNE sobre PCA desde el mismo vector multimodal del CBR.

## App local

Instala dependencias de la app si hace falta:

```bash
.venv/bin/python -m pip install -r app/requirements.txt
```

Crea o edita `.env` en la raiz:

```bash
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.55
APP_HOST=127.0.0.1
APP_PORT=8000
```

Arranca API y frontend:

```bash
.venv/bin/python -m app.api
```

Abre:

```text
http://127.0.0.1:8000
```
