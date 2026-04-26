from __future__ import annotations

import json
import mimetypes
import os
import re
import sqlite3
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "dataset" / "final" / "creative_memory.db"
ASSETS_DIR = ROOT / "Smadex_Creative_Intelligence_Dataset_FULL" / "assets"
VISUALIZATIONS_DIR = ROOT / "cv" / "output" / "visualizaciones"
STATIC_DIR = ROOT / "app" / "static"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def load_env() -> None:
    for path in [ROOT / ".env", ROOT / "app" / ".env"]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if value and key not in os.environ:
                os.environ[key] = value


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def asset_url(asset_file: Any) -> str | None:
    if not asset_file:
        return None
    return f"/api/assets/{Path(str(asset_file)).name}"


def visualized_url(asset_file: Any) -> str | None:
    if not asset_file:
        return None
    visualized_name = f"{Path(str(asset_file)).stem}_visualized.png"
    if not (VISUALIZATIONS_DIR / visualized_name).exists():
        return None
    return f"/api/visualizations/{visualized_name}"


class AppState:
    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Missing DB: {DB_PATH}")
        with db_connect() as conn:
            self.case_index = pd.read_sql_query(
                """
                SELECT
                    c.creative_id, c.campaign_id, c.advertiser_name, c.app_name,
                    c.vertical, c.format, c.language, c.theme, c.hook_type, c.cta_text,
                    c.headline, c.subhead, c.dominant_color, c.emotional_tone,
                    c.objective, c.primary_theme, c.target_age_segment, c.target_os,
                    c.asset_file, c.creative_status, c.perf_score, c.overall_roas,
                    c.overall_ipm, c.overall_ctr, c.overall_cvr,
                    c.text_density, c.clutter_score, c.brand_visibility_score,
                    c.motion_score, c.novelty_score, c.has_gameplay, c.has_ugc_style,
                    c.has_discount_badge, c.prompt_visual_hierarchy_score,
                    q1.best_score, q1.winner_segment, q1.global_rank, q1.vertical_rank,
                    q1.format_rank, q1.action_hint AS q1_action_hint,
                    q2.fatigue_status, q2.repetition_status, q2.tired_score,
                    q2.repetition_score, q2.creative_health_risk,
                    q3.recommended_action, q3.next_test, q3.priority_score,
                    q3.suggestion_1, q3.suggestion_2, q3.suggestion_3
                FROM creatives c
                LEFT JOIN question1_winner_ranking q1 USING (creative_id)
                LEFT JOIN question2_creative_health q2 USING (creative_id)
                LEFT JOIN question3_next_tests q3 USING (creative_id)
                """,
                conn,
            )
            self.landscape = pd.read_sql_query("SELECT * FROM creative_landscape", conn)

        self.case_index["asset_url"] = self.case_index["asset_file"].map(asset_url)
        self.case_index["visualized_url"] = self.case_index["asset_file"].map(visualized_url)
        self.verticals = sorted(self.case_index["vertical"].dropna().astype(str).unique().tolist())
        self.formats = sorted(self.case_index["format"].dropna().astype(str).unique().tolist())
        self._build_brief_index()

    def _build_brief_index(self) -> None:
        text_cols = [
            "vertical",
            "format",
            "language",
            "theme",
            "hook_type",
            "cta_text",
            "headline",
            "subhead",
            "dominant_color",
            "emotional_tone",
            "objective",
            "primary_theme",
            "target_age_segment",
            "target_os",
            "creative_status",
            "winner_segment",
            "fatigue_status",
            "repetition_status",
        ]
        corpus = (
            self.case_index[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
            .tolist()
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode")
        self.brief_matrix = self.vectorizer.fit_transform(corpus)

    def list_creatives(self, view: str, filters: dict[str, str], limit: int, offset: int) -> dict[str, Any]:
        df = self.case_index.copy()
        df = apply_filters(df, filters)
        df = apply_search(df, filters.get("q", ""))
        if view == "best":
            df = df.sort_values("best_score", ascending=False, na_position="last")
        elif view == "tired":
            df = df[
                df["fatigue_status"].isin(["tired", "at_risk"])
                | df["repetition_status"].isin(["high_repetition", "moderate_repetition"])
            ].sort_values(
                ["tired_score", "creative_health_risk"], ascending=False
            )
        elif view == "repetitive":
            df = df[df["repetition_status"].isin(["high_repetition", "moderate_repetition"])].sort_values(
                "repetition_score", ascending=False
            )
        elif view == "recommendations":
            df = df.sort_values("priority_score", ascending=False, na_position="last")
        else:
            df = df.sort_values("creative_id")
        total = len(df)
        page = df.iloc[offset : offset + limit]
        return {"total": total, "items": [case_card(row) for _, row in page.iterrows()]}

    def match_brief(self, brief: str, vertical: str | None, fmt: str | None, limit: int = 8) -> dict[str, Any]:
        requested_vertical = vertical
        inferred_vertical, vertical_reason = infer_vertical(brief, self.verticals)
        vertical = requested_vertical or inferred_vertical
        filters = {"vertical": vertical or "", "format": fmt or ""}
        df = apply_filters(self.case_index.copy(), filters)
        if df.empty:
            df = self.case_index.copy()
            vertical_reason = "fallback_all_verticals"

        query_text = expand_brief_terms(brief, vertical)
        query = self.vectorizer.transform([query_text])
        if query.nnz:
            sims = linear_kernel(query, self.brief_matrix[df.index]).ravel()
        else:
            sims = np.zeros(len(df), dtype=float)
        work = df.copy()
        work["brief_similarity"] = sims
        work["best_score_fill"] = pd.to_numeric(work["best_score"], errors="coerce").fillna(0)
        work["risk_fill"] = pd.to_numeric(work["creative_health_risk"], errors="coerce").fillna(0)
        work["priority_fill"] = pd.to_numeric(work["priority_score"], errors="coerce").fillna(0)

        good_mask = work["creative_status"].eq("top_performer") | work["winner_segment"].isin(
            ["top_winner", "strong_candidate"]
        )
        bad_mask = (
            work["creative_status"].isin(["underperformer", "fatigued"])
            | work["fatigue_status"].eq("tired")
            | work["best_score_fill"].le(0.35)
        )
        good = work[good_mask].copy()
        bad = work[bad_mask].copy()
        if good.empty:
            good = work.sort_values("best_score_fill", ascending=False).head(max(limit * 2, 10)).copy()
        if bad.empty:
            bad = work.sort_values("best_score_fill", ascending=True).head(max(limit * 2, 10)).copy()

        if np.nanmax(sims) <= 0:
            good["match_score"] = 0.70 * good["best_score_fill"] + 0.30 * good["priority_fill"]
            bad["match_score"] = 0.50 * (1 - bad["best_score_fill"]) + 0.50 * bad["risk_fill"]
        else:
            good["match_score"] = (
                0.58 * good["brief_similarity"] + 0.30 * good["best_score_fill"] + 0.12 * good["priority_fill"]
            )
            bad["match_score"] = (
                0.50 * bad["brief_similarity"] + 0.25 * (1 - bad["best_score_fill"]) + 0.25 * bad["risk_fill"]
            )
        good = good.sort_values("match_score", ascending=False).head(limit)
        bad = bad.sort_values("match_score", ascending=False).head(limit)
        guidance = build_guidance(good, bad)
        suggested_vertical = vertical or mode_or_none(good["vertical"]) or mode_or_none(work.nlargest(limit, "best_score_fill")["vertical"])
        return {
            "brief": brief,
            "filters": {
                "vertical": vertical,
                "format": fmt,
                "requested_vertical": requested_vertical,
                "suggested_vertical": suggested_vertical,
                "vertical_reason": "manual" if requested_vertical else vertical_reason,
                "query_has_known_terms": bool(query.nnz),
            },
            "best_cases": [brief_case(row, "do") for _, row in good.iterrows()],
            "avoid_cases": [brief_case(row, "avoid") for _, row in bad.iterrows()],
            "do": guidance["do"],
            "dont": guidance["dont"],
            "evidence": guidance["evidence"],
        }


def apply_filters(df: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    vertical = filters.get("vertical")
    fmt = filters.get("format")
    if vertical:
        df = df[df["vertical"].astype(str).str.lower().eq(vertical.lower())]
    if fmt:
        df = df[df["format"].astype(str).str.lower().eq(fmt.lower())]
    return df


def apply_search(df: pd.DataFrame, query: str | None) -> pd.DataFrame:
    if not query:
        return df
    text = normalize_text(query)
    if not text:
        return df
    if text.isdigit():
        id_mask = df["creative_id"].astype(str).str.contains(text, na=False)
    else:
        id_mask = pd.Series(False, index=df.index)
    search_cols = [
        "creative_id",
        "headline",
        "subhead",
        "app_name",
        "advertiser_name",
        "vertical",
        "format",
        "theme",
        "hook_type",
        "cta_text",
    ]
    haystack = df[search_cols].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)
    return df[id_mask | haystack.str.contains(re.escape(text), na=False)]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def infer_vertical(brief: str, verticals: list[str]) -> tuple[str | None, str]:
    text = normalize_text(brief)
    aliases = {
        "gaming": [
            "gaming",
            "game",
            "juego",
            "jugador",
            "gameplay",
            "playable",
            "reward",
            "rewarded",
            "recompensa",
            "nivel",
            "puzzle",
            "battle",
        ],
        "fintech": [
            "fintech",
            "finance",
            "finanzas",
            "banco",
            "bank",
            "crypto",
            "inversion",
            "dinero",
            "money",
            "card",
            "tarjeta",
            "loan",
            "prestamo",
            "ahorro",
        ],
        "food_delivery": [
            "food",
            "delivery",
            "comida",
            "restaurante",
            "restaurant",
            "pizza",
            "burger",
            "hamburguesa",
            "pedido",
            "takeaway",
        ],
        "travel": [
            "travel",
            "viaje",
            "viajes",
            "hotel",
            "flight",
            "vuelo",
            "vuelos",
            "booking",
            "vacaciones",
            "escapada",
            "destino",
        ],
        "ecommerce": [
            "ecommerce",
            "e-commerce",
            "shop",
            "shopping",
            "tienda",
            "compra",
            "compras",
            "producto",
            "catalogo",
            "oferta",
            "descuento",
        ],
        "entertainment": [
            "entertainment",
            "entretenimiento",
            "ocio",
            "diversion",
            "divertida",
            "fun",
            "streaming",
            "musica",
            "music",
            "video",
            "serie",
            "series",
            "pelicula",
            "movie",
            "show",
            "content",
            "contenido",
        ],
    }
    available = set(verticals)
    scores: dict[str, int] = {}
    for vertical, terms in aliases.items():
        if vertical not in available:
            continue
        score = 0
        for term in terms + [vertical, vertical.replace("_", " ")]:
            if re.search(rf"\b{re.escape(normalize_text(term))}\b", text):
                score += 1
        if score:
            scores[vertical] = score
    if not scores:
        return None, "not_inferred"
    return max(scores.items(), key=lambda item: item[1])[0], "text_alias"


def expand_brief_terms(brief: str, vertical: str | None) -> str:
    text = normalize_text(brief)
    expansions = []
    keyword_expansions = {
        "gameplay": "gaming playable rewarded_video has_gameplay motion_score recompensa",
        "recompensa": "gaming rewarded_video reward cta_text",
        "reward": "gaming rewarded_video recompensa",
        "descuento": "discount promo badge has_discount_badge price ecommerce",
        "oferta": "discount promo badge ecommerce",
        "precio": "price has_price discount ecommerce",
        "ugc": "ugc creator testimonial social proof",
        "testimonio": "social proof trust ugc",
        "comida": "food_delivery delivery restaurant",
        "viaje": "travel booking hotel flight",
        "crypto": "fintech finance money trust",
        "banco": "fintech finance trust card",
    }
    for key, extra in keyword_expansions.items():
        if key in text:
            expansions.append(extra)
    if vertical:
        expansions.append(vertical.replace("_", " "))
        expansions.append(vertical)
    return " ".join([text, *expansions]).strip()


def case_card(row: pd.Series) -> dict[str, Any]:
    fields = [
        "creative_id",
        "campaign_id",
        "advertiser_name",
        "app_name",
        "vertical",
        "format",
        "theme",
        "hook_type",
        "cta_text",
        "headline",
        "subhead",
        "asset_url",
        "visualized_url",
        "creative_status",
        "best_score",
        "winner_segment",
        "q1_action_hint",
        "fatigue_status",
        "repetition_status",
        "recommended_action",
        "next_test",
        "perf_score",
        "overall_roas",
        "overall_ipm",
        "tired_score",
        "repetition_score",
        "priority_score",
        "text_density",
        "clutter_score",
        "brand_visibility_score",
        "motion_score",
        "novelty_score",
        "has_gameplay",
        "has_ugc_style",
        "has_discount_badge",
        "prompt_visual_hierarchy_score",
    ]
    return clean_json({field: row.get(field) for field in fields})


def brief_case(row: pd.Series, kind: str) -> dict[str, Any]:
    card = case_card(row)
    card.update(
        clean_json(
            {
                "kind": kind,
                "brief_similarity": row.get("brief_similarity"),
                "match_score": row.get("match_score"),
            }
        )
    )
    return card


def build_guidance(good: pd.DataFrame, bad: pd.DataFrame) -> dict[str, Any]:
    do: list[str] = []
    dont: list[str] = []
    evidence: dict[str, Any] = {}

    for col, label in [
        ("format", "formato"),
        ("theme", "tema"),
        ("hook_type", "hook"),
        ("cta_text", "CTA"),
        ("emotional_tone", "tono emocional"),
    ]:
        good_mode = mode_or_none(good[col]) if col in good else None
        bad_mode = mode_or_none(bad[col]) if col in bad else None
        if good_mode:
            do.append(f"Usa {label} parecido a '{good_mode}' en las primeras variantes.")
        if bad_mode and bad_mode != good_mode:
            dont.append(f"Evita apoyar todo el concepto en {label}='{bad_mode}'.")
        evidence[col] = {"best_mode": good_mode, "avoid_mode": bad_mode}

    numeric_rules = [
        ("text_density", "densidad de texto", "mantén el mensaje más limpio", "evita saturar la pieza con texto"),
        ("clutter_score", "clutter visual", "simplifica la composición", "evita una composición demasiado cargada"),
        ("brand_visibility_score", "visibilidad de marca", "asegura una marca visible", "no escondas demasiado la marca"),
        ("motion_score", "movimiento", "añade más dinamismo", "evita una pieza demasiado estática"),
        ("novelty_score", "novedad", "busca un ángulo más novedoso", "no repitas patrones demasiado vistos"),
        ("prompt_visual_hierarchy_score", "jerarquía visual", "refuerza la jerarquía visual", "evita que CTA, producto y texto compitan"),
    ]
    for col, label, positive, negative in numeric_rules:
        if col not in good.columns or col not in bad.columns:
            continue
        good_med = pd.to_numeric(good[col], errors="coerce").median()
        bad_med = pd.to_numeric(bad[col], errors="coerce").median()
        if pd.isna(good_med) or pd.isna(bad_med):
            continue
        evidence[col] = {"best_median": round(float(good_med), 4), "avoid_median": round(float(bad_med), 4)}
        delta = good_med - bad_med
        if abs(delta) < 0.05:
            continue
        if col in {"text_density", "clutter_score"}:
            if delta < 0:
                do.append(f"{positive}: los buenos casos tienen menor {label}.")
                dont.append(f"{negative}: aparece más en los casos débiles.")
        elif delta > 0:
            do.append(f"{positive}: los buenos casos tienen mayor {label}.")
        else:
            dont.append(f"Revisa {label}: los casos débiles lo tienen más alto.")

    return {"do": dedupe(do)[:6], "dont": dedupe(dont)[:6], "evidence": evidence}


def mode_or_none(series: pd.Series) -> str | None:
    values = series.dropna().astype(str)
    if values.empty:
        return None
    return values.mode().iloc[0]


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


STATE: AppState | None = None


def state() -> AppState:
    global STATE
    if STATE is None:
        STATE = AppState()
    return STATE


class CreativeHandler(BaseHTTPRequestHandler):
    server_version = "CreativeMemoryHTTP/1.0"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/api/health":
                self.json_response({"ok": True, "db": str(DB_PATH), "groq_configured": bool(os.getenv("GROQ_API_KEY"))})
            elif path == "/api/options":
                self.json_response({"verticals": state().verticals, "formats": state().formats})
            elif path == "/api/creatives":
                view = query.get("view", ["best"])[0]
                limit = clamp_int(query.get("limit", ["40"])[0], 1, 2000)
                offset = clamp_int(query.get("offset", ["0"])[0], 0, 100000)
                filters = {
                    "vertical": query.get("vertical", [""])[0],
                    "format": query.get("format", [""])[0],
                    "q": query.get("q", [""])[0],
                }
                self.json_response(state().list_creatives(view, filters, limit, offset))
            elif path.startswith("/api/creative/"):
                creative_id = int(path.rsplit("/", 1)[-1])
                self.json_response(get_creative_detail(creative_id))
            elif path == "/api/landscape":
                self.json_response(get_landscape(query))
            elif path.startswith("/api/assets/"):
                self.serve_asset(path.rsplit("/", 1)[-1])
            elif path.startswith("/api/visualizations/"):
                self.serve_visualization(path.rsplit("/", 1)[-1])
            else:
                self.serve_static(path)
        except Exception as exc:
            self.json_response({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, status=500)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            body = self.read_json_body()
            if parsed.path == "/api/explain":
                self.json_response(explain_creative(body))
            elif parsed.path == "/api/brief/match":
                brief = str(body.get("brief", "")).strip()
                if not brief:
                    self.json_response({"ok": False, "error": "brief is required"}, status=400)
                    return
                result = state().match_brief(
                    brief,
                    body.get("vertical") or None,
                    body.get("format") or None,
                    clamp_int(body.get("limit", 8), 5, 10),
                )
                self.json_response({"ok": True, **result})
            elif parsed.path == "/api/brief/explain":
                self.json_response(explain_brief(body))
            else:
                self.json_response({"ok": False, "error": "Not found"}, status=404)
        except Exception as exc:
            self.json_response({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, status=500)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def json_response(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(clean_json(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_asset(self, filename: str) -> None:
        safe_name = Path(filename).name
        path = ASSETS_DIR / safe_name
        self.serve_file(path)

    def serve_visualization(self, filename: str) -> None:
        safe_name = Path(filename).name
        path = VISUALIZATIONS_DIR / safe_name
        self.serve_file(path)

    def serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            target = STATIC_DIR / "index.html"
        else:
            target = (STATIC_DIR / path.lstrip("/")).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())):
                self.send_error(403)
                return
            if target.is_dir():
                target = target / "index.html"
        self.serve_file(target)

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def get_creative_detail(creative_id: int) -> dict[str, Any]:
    with db_connect() as conn:
        creative = fetch_one(conn, "SELECT * FROM creatives WHERE creative_id = ?", [creative_id])
        q1 = fetch_one(conn, "SELECT * FROM question1_winner_ranking WHERE creative_id = ?", [creative_id])
        q2 = fetch_one(conn, "SELECT * FROM question2_creative_health WHERE creative_id = ?", [creative_id])
        q3 = fetch_one(conn, "SELECT * FROM question3_next_tests WHERE creative_id = ?", [creative_id])
        exp = fetch_one(conn, "SELECT * FROM creative_explanations WHERE creative_id = ?", [creative_id])
        landscape = fetch_one(conn, "SELECT * FROM creative_landscape WHERE creative_id = ?", [creative_id])
        neighbors = fetch_all(
            conn,
            """
            SELECT
                n.*,
                c.headline AS neighbor_headline,
                c.subhead AS neighbor_subhead,
                c.asset_file AS neighbor_asset_file,
                c.app_name AS neighbor_app_name
            FROM creative_neighbors n
            LEFT JOIN creatives c ON c.creative_id = n.neighbor_id
            WHERE n.creative_id = ?
            ORDER BY n.neighbor_rank
            LIMIT 12
            """,
            [creative_id],
        )
    if creative:
        creative["asset_url"] = asset_url(creative.get("asset_file"))
        creative["visualized_url"] = visualized_url(creative.get("asset_file"))
    for neighbor in neighbors:
        neighbor["asset_url"] = asset_url(neighbor.get("neighbor_asset_file") or f"assets/creative_{neighbor['neighbor_id']}.png")
    if exp:
        exp["q1"] = parse_json_field(exp.pop("q1_json", None))
        exp["q2"] = parse_json_field(exp.pop("q2_json", None))
        exp["q3"] = parse_json_field(exp.pop("q3_json", None))
    return {
        "creative": creative,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "explanations": exp,
        "neighbors": neighbors,
        "landscape": landscape,
    }


def get_landscape(query: dict[str, list[str]]) -> dict[str, Any]:
    df = state().landscape.copy()
    filters = {"vertical": query.get("vertical", [""])[0], "format": query.get("format", [""])[0]}
    df = apply_filters(df, filters)
    return {"items": clean_json(df.to_dict(orient="records"))}


def fetch_one(conn: sqlite3.Connection, sql: str, params: list[Any]) -> dict[str, Any] | None:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def fetch_all(conn: sqlite3.Connection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def explain_creative(body: dict[str, Any]) -> dict[str, Any]:
    creative_id = int(body.get("creative_id"))
    focus = str(body.get("focus", "all"))
    detail = get_creative_detail(creative_id)
    evidence = {
        "focus": focus,
        "creative": compact_creative(detail["creative"]),
        "q1": detail["explanations"].get("q1") if detail.get("explanations") else None,
        "q2": detail["explanations"].get("q2") if detail.get("explanations") else None,
        "q3": detail["explanations"].get("q3") if detail.get("explanations") else None,
        "neighbors": detail.get("neighbors", [])[:8],
    }
    prompt = (
        "Explica este creative a un marketer. Usa solo la evidencia JSON. "
        "Estructura la respuesta con: Diagnóstico, Evidencia, Por qué este Next Test, Qué hacer ahora. "
        "Sé claro, breve y accionable. Si la evidencia es débil, dilo.\n\n"
        + json.dumps(clean_json(evidence), ensure_ascii=False)
    )
    return groq_completion(prompt, fallback=evidence, fallback_text=fallback_creative_text(evidence))


def explain_brief(body: dict[str, Any]) -> dict[str, Any]:
    brief = str(body.get("brief", "")).strip()
    if not brief:
        return {"ok": False, "error": "brief is required"}
    match = state().match_brief(
        brief,
        body.get("vertical") or None,
        body.get("format") or None,
        clamp_int(body.get("limit", 8), 5, 10),
    )
    prompt = (
        "Eres un creative strategist para mobile advertising. A partir de estos casos historicos, "
        "propón un plan creativo para el brief. Distingue claramente: Hacer, Evitar, Primer test. "
        "No inventes métricas ni causas fuera del JSON.\n\n"
        + json.dumps(clean_json(compact_match_for_llm(match)), ensure_ascii=False)
    )
    response = groq_completion(prompt, fallback=match, fallback_text=fallback_brief_text(match))
    response["match"] = match
    return response


def compact_creative(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keep = [
        "creative_id",
        "campaign_id",
        "advertiser_name",
        "app_name",
        "vertical",
        "format",
        "theme",
        "hook_type",
        "cta_text",
        "headline",
        "subhead",
        "creative_status",
        "perf_score",
        "overall_roas",
        "overall_ipm",
        "asset_url",
        "visualized_url",
    ]
    return {key: row.get(key) for key in keep}


def compact_match_for_llm(match: dict[str, Any]) -> dict[str, Any]:
    def compact_case(item: dict[str, Any]) -> dict[str, Any]:
        keep = [
            "creative_id",
            "vertical",
            "format",
            "theme",
            "hook_type",
            "cta_text",
            "headline",
            "creative_status",
            "match_score",
            "best_score",
            "overall_roas",
            "overall_ipm",
            "text_density",
            "clutter_score",
            "brand_visibility_score",
            "motion_score",
            "novelty_score",
        ]
        return {key: item.get(key) for key in keep if key in item}

    return {
        "brief": match.get("brief"),
        "filters": match.get("filters"),
        "do": match.get("do", [])[:6],
        "dont": match.get("dont", [])[:6],
        "best_cases": [compact_case(item) for item in match.get("best_cases", [])[:3]],
        "avoid_cases": [compact_case(item) for item in match.get("avoid_cases", [])[:3]],
    }


def groq_completion(prompt: str, fallback: Any, fallback_text: str) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False,
            "provider": "groq",
            "error": "GROQ_API_KEY is not configured",
            "text": fallback_text,
            "fallback": clean_json(fallback),
        }

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.55"))
    max_tokens = clamp_int(os.getenv("GROQ_MAX_TOKENS", "350"), 128, 1200)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Creative Memory Copilot. Explain marketing evidence in Spanish. "
                    "Use only the provided structured evidence and avoid unsupported claims."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", "curl/8.5.0")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return {"ok": True, "provider": "groq", "model": model, "text": text, "raw": data}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "provider": "groq",
            "error": error_body,
            "text": fallback_text,
            "fallback": clean_json(fallback),
        }
    except Exception as exc:
        return {"ok": False, "provider": "groq", "error": str(exc), "text": fallback_text, "fallback": clean_json(fallback)}


def fallback_creative_text(evidence: dict[str, Any]) -> str:
    creative = evidence.get("creative") or {}
    q1 = evidence.get("q1") or {}
    q2 = evidence.get("q2") or {}
    q3 = evidence.get("q3") or {}
    neighbors = evidence.get("neighbors") or []
    similar_top = q1.get("similar_cases", {}).get("top_performer_count") if isinstance(q1.get("similar_cases"), dict) else None
    similar_total = q1.get("similar_cases", {}).get("similar_case_count") if isinstance(q1.get("similar_cases"), dict) else None
    lines = [
        f"Diagnóstico: creative {creative.get('creative_id')} en {creative.get('vertical')} / {creative.get('format')} con estado {creative.get('creative_status')}.",
        (
            "Evidencia: "
            f"ROAS {creative.get('overall_roas')}, IPM {creative.get('overall_ipm')}, "
            f"fatiga {q2.get('fatigue_status')}, repetición {q2.get('repetition_status')}."
        ),
    ]
    if similar_top is not None and similar_total:
        lines.append(f"Casos similares: {similar_top} de {similar_total} vecinos fueron top performers.")
    if q3.get("next_test"):
        lines.append(f"Por qué este Next Test: se propone porque combina el estado actual, el riesgo de fatiga/repetición y el patrón de vecinos similares. Next test: {q3.get('next_test')}")
    suggestions = [q3.get("suggestion_1"), q3.get("suggestion_2"), q3.get("suggestion_3")]
    suggestions = [s for s in suggestions if s]
    if suggestions:
        lines.append("Qué hacer ahora: " + " ".join(str(s) for s in suggestions))
    if neighbors:
        lines.append(f"Evidencia CBR: se han usado {min(len(neighbors), 8)} vecinos cercanos para contextualizar la decisión.")
    return "\n".join(lines)


def fallback_brief_text(match: dict[str, Any]) -> str:
    filters = match.get("filters", {})
    do = match.get("do", [])[:4]
    dont = match.get("dont", [])[:4]
    best = match.get("best_cases", [])[:3]
    avoid = match.get("avoid_cases", [])[:3]
    lines = [
        f"Vertical sugerida: {filters.get('suggested_vertical') or filters.get('vertical') or 'sin vertical fija'}.",
        "Hacer: " + (" ".join(do) if do else "usar los patrones de los winners más cercanos y mantener el mensaje claro."),
        "Evitar: " + (" ".join(dont) if dont else "copiar patrones de casos con bajo score, fatiga o repetición alta."),
    ]
    if best:
        ids = ", ".join(str(item.get("creative_id")) for item in best)
        lines.append(f"Casos ganadores de referencia: {ids}.")
    if avoid:
        ids = ", ".join(str(item.get("creative_id")) for item in avoid)
        lines.append(f"Casos a evitar como referencia principal: {ids}.")
    lines.append("Primer test: crea una variante que combine el formato/hook más repetido entre winners con menor clutter y CTA directo.")
    return "\n".join(lines)


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except Exception:
        return minimum


def run() -> None:
    load_env()
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = clamp_int(os.getenv("APP_PORT", "8000"), 1, 65535)
    state()
    server = ThreadingHTTPServer((host, port), CreativeHandler)
    print(f"Creative Memory Copilot running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
