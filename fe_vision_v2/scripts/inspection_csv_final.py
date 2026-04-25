"""
inspection_csv_final.py
-----------------------
Exploracion del CSV final con foco en metadata y diccionario de columnas.

Genera:
- metadata.json
- columns_dictionary.csv
- missing_values.csv
- numeric_summary.csv
- categorical_top_values.csv
- report.md

Ejemplo:
python scripts/inspection_csv_final.py \
  --input outputs/creative_visual_features_with_embeddings_pca.csv \
  --output_dir outputs/inspection_csv_final
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Exploracion del CSV final y diccionario de columnas.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("outputs/creative_visual_features_with_embeddings_pca.csv"),
		help="Ruta al CSV final.",
	)
	parser.add_argument(
		"--output_dir",
		type=Path,
		default=Path("outputs/inspection_csv_final"),
		help="Carpeta de salida para reportes.",
	)
	parser.add_argument(
		"--top_n_categories",
		type=int,
		default=8,
		help="Top N categorias por columna categorica.",
	)
	return parser.parse_args()


def _embedding_info(col: str) -> tuple[bool, str, int | None, str | None]:
	m = re.match(r"^(clip_pca|cnn_pca)_(\d+)$", col)
	if m:
		family = m.group(1)
		idx = int(m.group(2))
		desc = f"Componente PCA {idx} del embedding {family.split('_')[0].upper()}."
		return True, family, idx, desc

	m = re.match(r"^(clip|cnn)_(\d+)$", col)
	if m:
		family = m.group(1)
		idx = int(m.group(2))
		desc = f"Dimension {idx} del embedding {family.upper()} (sin PCA)."
		return True, family, idx, desc

	if col.startswith("clip_"):
		return True, "clip", None, "Feature derivada de embedding CLIP."
	if col.startswith("cnn_"):
		return True, "cnn", None, "Feature derivada de embedding CNN."

	return False, "", None, None


def _column_group(col: str) -> str:
	if col.endswith("_id") or col in {"creative_id", "campaign_id"}:
		return "identificadores"

	is_emb, family, _, _ = _embedding_info(col)
	if is_emb:
		if "pca" in family:
			return "embeddings_pca"
		return "embeddings_raw"

	if col in {
		"advertiser_name",
		"app_name",
		"vertical",
		"format",
		"language",
		"creative_launch_date",
		"theme",
		"hook_type",
		"cta_text",
		"headline",
		"subhead",
		"dominant_color",
		"emotional_tone",
		"asset_file",
		"creative_status",
	}:
		return "metadata_creative"

	if col.startswith("overall_") or col.startswith("first_7d_") or col.startswith("last_7d_"):
		return "kpis"
	if "impressions" in col or "clicks" in col or "conversions" in col:
		return "kpis"
	if col in {
		"total_spend_usd",
		"total_revenue_usd",
		"total_days_active",
		"fatigue_day",
		"ctr_decay_pct",
		"cvr_decay_pct",
		"peak_rolling_ctr_5",
		"peak_day_impressions",
		"perf_score",
	}:
		return "kpis"

	if (
		col.startswith("brightness")
		or col.startswith("contrast")
		or col.startswith("saturation")
		or col.startswith("hue")
		or col.startswith("dominant_color_")
		or col.startswith("colorfulness")
		or col.startswith("edge_")
		or col.startswith("visual_")
		or col.startswith("centroid_")
		or col.startswith("quadrant_")
		or col.startswith("symmetry_")
		or col.startswith("main_object_")
		or col.startswith("saliency_")
		or col.startswith("background_")
		or col.startswith("empty_space_")
	):
		return "vision_features"

	if col in {"width_x", "height_x", "width_y", "height_y", "aspect_ratio", "image_area"}:
		return "dimensiones"

	if col in {"duration_sec", "text_density", "copy_length_chars", "readability_score"}:
		return "copy_video"

	if col in {
		"brand_visibility_score",
		"clutter_score",
		"novelty_score",
		"motion_score",
		"faces_count",
		"product_count",
		"has_price",
		"has_discount_badge",
		"has_gameplay",
		"has_ugc_style",
	}:
		return "creative_attributes"

	return "otros"


def _column_description(col: str) -> str:
	exact = {
		"creative_id": "Identificador unico del creativo.",
		"campaign_id": "Identificador de la campana.",
		"advertiser_name": "Nombre del anunciante.",
		"app_name": "Nombre de la app o producto promocionado.",
		"vertical": "Vertical de negocio (gaming, fintech, etc.).",
		"format": "Formato de anuncio (banner, interstitial, rewarded_video, etc.).",
		"width_x": "Ancho original del creativo en pixeles (origen tabla base).",
		"height_x": "Alto original del creativo en pixeles (origen tabla base).",
		"width_y": "Ancho usado en el pipeline visual (post-join).",
		"height_y": "Alto usado en el pipeline visual (post-join).",
		"language": "Idioma principal del creativo.",
		"creative_launch_date": "Fecha de lanzamiento del creativo.",
		"theme": "Tema narrativo o estilo del anuncio.",
		"hook_type": "Tipo de gancho inicial del mensaje.",
		"cta_text": "Texto del call-to-action.",
		"headline": "Titular principal del anuncio.",
		"subhead": "Subtitulo o texto secundario.",
		"dominant_color": "Color dominante anotado en metadata.",
		"emotional_tone": "Tono emocional del creativo.",
		"duration_sec": "Duracion del asset en segundos.",
		"text_density": "Densidad de texto dentro de la creatividad.",
		"copy_length_chars": "Longitud del copy en caracteres.",
		"readability_score": "Score de legibilidad del texto.",
		"brand_visibility_score": "Visibilidad de marca estimada.",
		"clutter_score": "Nivel de saturacion visual.",
		"novelty_score": "Score de novedad percibida.",
		"motion_score": "Score de movimiento visual.",
		"faces_count": "Numero de caras detectadas.",
		"product_count": "Numero de productos detectados.",
		"has_price": "Flag binario: muestra precio.",
		"has_discount_badge": "Flag binario: muestra badge de descuento.",
		"has_gameplay": "Flag binario: incluye gameplay.",
		"has_ugc_style": "Flag binario: estilo UGC.",
		"asset_file": "Ruta o nombre del asset.",
		"creative_status": "Estado de rendimiento/fatiga del creativo.",
		"fatigue_day": "Dia estimado de fatiga del creativo.",
		"total_days_active": "Dias activos del creativo.",
		"total_spend_usd": "Gasto total en USD.",
		"total_impressions": "Impresiones totales.",
		"total_clicks": "Clicks totales.",
		"total_conversions": "Conversiones totales.",
		"total_revenue_usd": "Ingresos totales atribuidos en USD.",
		"overall_ctr": "CTR global.",
		"overall_cvr": "CVR global.",
		"overall_ipm": "Installs/conversiones por mil impresiones.",
		"overall_roas": "ROAS global.",
		"first_7d_ctr": "CTR de los primeros 7 dias.",
		"last_7d_ctr": "CTR de los ultimos 7 dias.",
		"ctr_decay_pct": "Cambio porcentual del CTR entre inicio y final.",
		"first_7d_cvr": "CVR de los primeros 7 dias.",
		"last_7d_cvr": "CVR de los ultimos 7 dias.",
		"cvr_decay_pct": "Cambio porcentual del CVR entre inicio y final.",
		"peak_rolling_ctr_5": "Mejor CTR en ventana movil de 5 dias.",
		"peak_day_impressions": "Impresiones del dia pico.",
		"first_7d_impressions": "Impresiones de los primeros 7 dias.",
		"first_7d_clicks": "Clicks de los primeros 7 dias.",
		"first_7d_conversions": "Conversiones de los primeros 7 dias.",
		"last_7d_impressions": "Impresiones de los ultimos 7 dias.",
		"last_7d_clicks": "Clicks de los ultimos 7 dias.",
		"last_7d_conversions": "Conversiones de los ultimos 7 dias.",
		"perf_score": "Score global de rendimiento.",
		"aspect_ratio": "Relacion de aspecto del asset (ancho/alto).",
		"image_area": "Area de imagen en pixeles.",
		"brightness_mean": "Media de brillo.",
		"brightness_std": "Desviacion estandar de brillo.",
		"contrast": "Contraste global de la imagen.",
		"saturation_mean": "Media de saturacion.",
		"saturation_std": "Desviacion estandar de saturacion.",
		"hue_mean": "Media del matiz (hue).",
		"hue_std": "Desviacion estandar del matiz (hue).",
		"colorfulness": "Indice de colorido percibido.",
		"edge_density": "Densidad de bordes detectados.",
		"visual_complexity": "Complejidad visual estimada.",
		"sharpness_laplacian_var": "Nitidez estimada (varianza del Laplaciano).",
		"centroid_x_visual_mass": "Coordenada X del centro de masa visual normalizada en [0,1].",
		"centroid_y_visual_mass": "Coordenada Y del centro de masa visual normalizada en [0,1].",
		"saliency_center_bias": "Tendencia de la saliencia hacia el centro.",
		"background_uniformity_score": "Uniformidad del fondo.",
		"empty_space_ratio": "Proporcion de espacio vacio.",
	}

	if col in exact:
		return exact[col]

	is_emb, _, _, emb_desc = _embedding_info(col)
	if is_emb and emb_desc:
		return emb_desc

	m = re.match(r"^dominant_color_([rgbhsv])$", col)
	if m:
		comp = m.group(1).upper()
		return f"Componente {comp} del color dominante calculado por vision."

	if col.startswith("visual_mass_"):
		return "Distribucion de masa visual en una region especifica (ratio)."

	if col.startswith("quadrant_") and col.endswith("_density"):
		return "Densidad visual en un cuadrante de la imagen."

	if col.startswith("symmetry_"):
		return "Score de simetria visual (horizontal o vertical)."

	if col.startswith("main_object_bbox_"):
		return "Coordenada o area del bounding box del objeto principal."

	if col.startswith("main_object_center_"):
		return "Coordenada del centro del objeto principal normalizada en [0,1]."

	if col in {"main_object_is_centered"}:
		return "Flag binario: el objeto principal esta centrado."

	if col.endswith("_impressions"):
		return "Metica de impresiones para una ventana especifica."

	if col.endswith("_clicks"):
		return "Metrica de clicks para una ventana especifica."

	if col.endswith("_conversions"):
		return "Metrica de conversiones para una ventana especifica."

	if col.endswith("_ctr"):
		return "Click-through-rate para una ventana especifica."

	if col.endswith("_cvr"):
		return "Conversion-rate para una ventana especifica."

	if col.endswith("_score"):
		return "Score numerico derivado por modelo o regla de negocio."

	return "Descripcion no mapeada automaticamente; revisar definicion en pipeline." 


def build_column_dictionary(df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	n_rows = len(df)
	mem_bytes_per_col = df.memory_usage(deep=True)

	for col in df.columns:
		is_emb, emb_family, emb_idx, _ = _embedding_info(col)
		non_null = int(df[col].notna().sum())
		missing = n_rows - non_null

		rows.append(
			{
				"column": col,
				"dtype": str(df[col].dtype),
				"group": _column_group(col),
				"description": _column_description(col),
				"is_embedding": is_emb,
				"embedding_family": emb_family if is_emb else "",
				"embedding_index": emb_idx if emb_idx is not None else "",
				"non_null_count": non_null,
				"missing_count": missing,
				"missing_pct": (missing / n_rows * 100.0) if n_rows else np.nan,
				"unique_count": int(df[col].nunique(dropna=True)),
				"memory_mb": float(mem_bytes_per_col[col]) / (1024**2),
			}
		)

	return pd.DataFrame(rows).sort_values(["group", "column"]).reset_index(drop=True)


def build_categorical_summary(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
	cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
	out_rows: list[dict[str, object]] = []

	for col in cat_cols:
		vc = df[col].value_counts(dropna=False).head(top_n)
		for val, cnt in vc.items():
			out_rows.append(
				{
					"column": col,
					"value": "<NA>" if pd.isna(val) else str(val),
					"count": int(cnt),
					"pct": float(cnt) / len(df) * 100.0 if len(df) else np.nan,
				}
			)

	return pd.DataFrame(out_rows)


def main() -> None:
	args = parse_args()
	input_path = args.input
	output_dir = args.output_dir

	if not input_path.exists():
		raise FileNotFoundError(f"No se encontro el CSV: {input_path}")

	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(input_path, low_memory=False)

	n_rows, n_cols = df.shape
	dup_rows = int(df.duplicated().sum())
	total_mem_mb = float(df.memory_usage(deep=True).sum()) / (1024**2)

	dict_df = build_column_dictionary(df)
	missing_df = dict_df[["column", "missing_count", "missing_pct"]].sort_values(
		"missing_pct", ascending=False
	)

	numeric_df = df.select_dtypes(include=[np.number])
	numeric_summary = numeric_df.describe().T.reset_index().rename(columns={"index": "column"})

	categorical_summary = build_categorical_summary(df, args.top_n_categories)

	emb_df = dict_df[dict_df["is_embedding"] == True].copy()  # noqa: E712
	emb_counts = emb_df["embedding_family"].value_counts().to_dict()

	metadata = {
		"input_file": str(input_path),
		"rows": n_rows,
		"columns": n_cols,
		"duplicates": dup_rows,
		"total_memory_mb": round(total_mem_mb, 3),
		"numeric_columns": int(len(numeric_df.columns)),
		"categorical_columns": int(len(df.columns) - len(numeric_df.columns)),
		"groups_count": dict_df["group"].value_counts().to_dict(),
		"embedding_columns": int(len(emb_df)),
		"embedding_breakdown": emb_counts,
		"missing_cells_pct": float(df.isna().sum().sum()) / (n_rows * n_cols) * 100.0,
	}

	metadata_path = output_dir / "metadata.json"
	dict_path = output_dir / "columns_dictionary.csv"
	missing_path = output_dir / "missing_values.csv"
	numeric_path = output_dir / "numeric_summary.csv"
	cat_path = output_dir / "categorical_top_values.csv"
	report_path = output_dir / "report.md"

	with metadata_path.open("w", encoding="utf-8") as f:
		json.dump(metadata, f, ensure_ascii=False, indent=2)

	dict_df.to_csv(dict_path, index=False)
	missing_df.to_csv(missing_path, index=False)
	numeric_summary.to_csv(numeric_path, index=False)
	categorical_summary.to_csv(cat_path, index=False)

	top_missing = missing_df.head(10)
	top_missing_lines = []
	for _, r in top_missing.iterrows():
		top_missing_lines.append(
			f"- {r['column']}: {int(r['missing_count'])} ({r['missing_pct']:.2f}%)"
		)

	group_counts = dict_df["group"].value_counts()
	group_lines = [f"- {g}: {int(c)} columnas" for g, c in group_counts.items()]

	emb_lines = [f"- {k}: {v} columnas" for k, v in emb_counts.items()] if emb_counts else ["- No embeddings"]

	report = (
		"# Exploracion CSV Final\n\n"
		f"Archivo: {input_path}\n\n"
		"## Metadata\n"
		f"- Filas: {n_rows}\n"
		f"- Columnas: {n_cols}\n"
		f"- Duplicados exactos: {dup_rows}\n"
		f"- Memoria aproximada: {total_mem_mb:.2f} MB\n"
		f"- % celdas faltantes: {metadata['missing_cells_pct']:.2f}%\n\n"
		"## Grupos de columnas\n"
		+ "\n".join(group_lines)
		+ "\n\n## Embeddings detectados\n"
		+ "\n".join(emb_lines)
		+ "\n\n## Top columnas con missing\n"
		+ "\n".join(top_missing_lines)
		+ "\n\n## Archivos generados\n"
		f"- {metadata_path}\n"
		f"- {dict_path}\n"
		f"- {missing_path}\n"
		f"- {numeric_path}\n"
		f"- {cat_path}\n"
	)

	report_path.write_text(report, encoding="utf-8")

	print("Exploracion completada")
	print(f"- metadata: {metadata_path}")
	print(f"- diccionario de columnas: {dict_path}")
	print(f"- reporte: {report_path}")
	print(f"- embeddings detectados: {len(emb_df)} -> {emb_counts}")


if __name__ == "__main__":
	main()
