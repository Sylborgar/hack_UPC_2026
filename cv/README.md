# Florence-2 Spatial Fingerprint Pipeline

Pipeline para extraer features espaciales interpretables desde assets creativos y cruzarlos con KPIs.

## Que hace

- Usa `Florence-2` para OCR con region (`<OCR_WITH_REGION>`).
- Usa `Florence-2` para grounding por frases (`<CAPTION_TO_PHRASE_GROUNDING>`).
- Convierte detecciones en un `spatial fingerprint` por creativo.
- Genera un JSON estructurado por creativo con el esquema de analisis espacial para correlacion con KPIs.
- Exporta features y merge con KPIs de `creative_summary.csv`.
- Calcula correlaciones de Spearman globales y por vertical.

## Mejoras aplicadas sobre la idea inicial

- Evita depender de demasiadas llamadas por imagen (se centra en OCR + grounding).
- Incluye `fallback` robusto para no romper el pipeline si una imagen falla.
- Soporta `resume` para continuar ejecuciones largas en CPU.
- Mantiene features explicables para decisiones de marketing.

## Estructura

- `build_visual_features.py`: lanzador unico.
- `visual_features/cli.py`: argumentos de linea de comandos.
- `visual_features/devices.py`: seleccion de dispositivo.
- `visual_features/data_io.py`: carga/validacion de datos y escritura de outputs.
- `visual_features/florence.py`: wrapper de modelo y tareas Florence-2.
- `visual_features/spatial_features.py`: parseo OCR/grounding y calculo de features.
- `visual_features/pipeline.py`: orquestacion completa.
- `visual_features/main.py`: entrypoint del paquete.

## Instalacion

```bash
cd cv
pip install -r requirements.txt
```

## Ejemplos de uso

Ejecutar sobre 20 creativos:

```bash
cd cv
python build_visual_features.py --limit 20 --save-elements-jsonl
```

Ejecutar sobre IDs concretos:

```bash
cd cv
python build_visual_features.py --creative-ids 500008,500011 --save-elements-jsonl
```

Reanudar una corrida previa:

```bash
cd cv
python build_visual_features.py --resume --save-elements-jsonl

Ejecutar y exportar solo el JSON estructurado del prompt:

```bash
cd cv
python build_visual_features.py --limit 20 --disable-florence --analysis-jsonl-name creative_prompt_analysis.jsonl
```
```

## Outputs

Se escriben en `cv/output/`:

- `creative_spatial_features.parquet` (o `.csv` si parquet no esta disponible)
- `creative_spatial_with_kpis.parquet` (o `.csv`)
- `creative_prompt_analysis.jsonl` (JSON por creativo con estructura del prompt)
- `spatial_kpi_correlations.csv`
- `spatial_kpi_correlations_strong.csv`
- `creative_spatial_elements.jsonl` (si `--save-elements-jsonl`)
- `run_summary.json`

## Nota de rendimiento

- En CPU, Florence-2 puede tardar varios segundos por imagen.
- Para iterar rapido, usa `--limit` o `--creative-ids`.
- Para corrida completa (1,080 creativos), usa `--resume` para tolerar interrupciones.
