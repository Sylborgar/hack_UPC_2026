const state = {
  activeView: "home",
  selected: null,
  lastBrief: null,
  landscape: [],
  mapNeighbors: new Set(),
  mapPointById: new Map(),
  mapSearchSeq: 0,
  map: {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    dragging: false,
    moved: false,
    lastX: 0,
    lastY: 0,
  },
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "n/a";
  const n = Number(value);
  if (Number.isFinite(n)) return n.toFixed(digits);
  return esc(value);
}

function creativeImageBlock(c) {
  const detailUrl = c.visualized_url || c.asset_url;
  const hasImage = Boolean(detailUrl);
  return `
    <div class="creative-media">
      <img class="hero-img" data-creative-image src="${esc(c.asset_url)}" alt="Creative ${esc(c.creative_id)}">
      <button
        class="soft-btn full-btn image-detail-toggle"
        data-image-toggle
        data-original-url="${esc(c.asset_url)}"
        data-detail-url="${esc(detailUrl || "")}"
        data-image-title="${esc(c.headline || c.app_name || `Creative ${c.creative_id}`)}"
        ${hasImage ? "" : "disabled"}
      >${c.visualized_url ? "Detail" : "Open image"}</button>
    </div>
  `;
}

function bindImageToggle(panel) {
  panel.querySelectorAll("[data-image-toggle]").forEach((button) => {
    if (!button.dataset.detailUrl) return;
    button.addEventListener("click", () => {
      openImageModal(button.dataset.detailUrl, button.dataset.imageTitle || "Creative detail");
    });
  });
}

function bindImageModal() {
  $("imageModalClose").addEventListener("click", closeImageModal);
  $("imageModalBackdrop").addEventListener("click", closeImageModal);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeImageModal();
  });
}

function openImageModal(url, title) {
  if (!url) return;
  $("imageModalTitle").textContent = title || "Creative detail";
  $("imageModalImg").src = url;
  $("imageModal").classList.remove("hidden");
  $("imageModal").setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeImageModal() {
  const modal = $("imageModal");
  if (!modal || modal.classList.contains("hidden")) return;
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
  $("imageModalImg").removeAttribute("src");
  document.body.classList.remove("modal-open");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function init() {
  bindTabs();
  bindControls();
  bindMapControls();
  bindImageModal();
  await loadOptions();
  await loadList("home");
  await runBriefMatch(false);
}

async function loadOptions() {
  const data = await api("/api/options");
  for (const id of ["verticalFilter", "briefVertical"]) {
    const select = $(id);
    for (const vertical of data.verticals) {
      select.insertAdjacentHTML("beforeend", `<option value="${esc(vertical)}">${esc(vertical)}</option>`);
    }
  }
  for (const id of ["formatFilter", "briefFormat"]) {
    const select = $(id);
    for (const format of data.formats) {
      select.insertAdjacentHTML("beforeend", `<option value="${esc(format)}">${esc(format)}</option>`);
    }
  }
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });
}

async function switchView(view, preferredId = null) {
  state.activeView = view;
  document.querySelector(".filters-band").classList.toggle("hidden", view === "next");
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  document.querySelectorAll(".workspace").forEach((panel) => panel.classList.remove("active"));
  $(`view-${view}`).classList.add("active");
  if (["home", "best", "tired"].includes(view)) await loadList(view, preferredId);
  if (view === "map") {
    await loadLandscape();
    if ($("searchInput").value.trim()) await searchMapSelection();
  }
}

function bindControls() {
  $("refreshBtn").addEventListener("click", () => refreshActiveView());
  $("verticalFilter").addEventListener("change", () => refreshActiveView());
  $("formatFilter").addEventListener("change", () => refreshActiveView());
  $("searchInput").addEventListener("input", debounce(() => {
    if (state.activeView === "map") {
      searchMapSelection();
    } else if (["home", "best", "tired"].includes(state.activeView)) {
      loadList(state.activeView);
    }
  }, 220));
  $("matchBriefBtn").addEventListener("click", () => runBriefMatch(false));
  $("explainBriefBtn").addEventListener("click", () => runBriefMatch(true));
}

function refreshActiveView() {
  if (["home", "best", "tired"].includes(state.activeView)) loadList(state.activeView);
  if (state.activeView === "map") loadLandscape();
}

function filtersQuery(includeSearch = true) {
  const params = new URLSearchParams();
  const vertical = $("verticalFilter").value;
  const format = $("formatFilter").value;
  const q = $("searchInput").value.trim();
  if (vertical) params.set("vertical", vertical);
  if (format) params.set("format", format);
  if (includeSearch && q) params.set("q", q);
  return params;
}

async function loadList(view, preferredId = null) {
  const params = filtersQuery(true);
  params.set("view", view === "home" ? "all" : view);
  params.set("limit", view === "home" ? 1200 : 80);
  const data = await api(`/api/creatives?${params.toString()}`);
  const target = view === "home" ? $("homeList") : view === "best" ? $("bestList") : $("tiredList");
  target.innerHTML = data.items.map((item) => caseCard(item, view)).join("");
  target.querySelectorAll(".case-card").forEach((button) => {
    button.addEventListener("click", () => selectCreative(Number(button.dataset.id), view));
  });
  const selected = preferredId || (data.items[0] ? data.items[0].creative_id : null);
  if (selected) await selectCreative(Number(selected), view);
  if (!data.items.length) {
    const panel = view === "home" ? $("homeDetail") : view === "best" ? $("bestDetail") : $("tiredDetail");
    panel.innerHTML = `<div class="empty-state">No hay casos con esos filtros.</div>`;
  }
}

function caseCard(item, view) {
  const primary = view === "best"
    ? `Score ${fmt(item.best_score)} · ROAS ${fmt(item.overall_roas, 2)}`
    : view === "tired"
      ? `Tired ${fmt(item.tired_score)} · Repeat ${fmt(item.repetition_score)}`
      : `ROAS ${fmt(item.overall_roas, 2)} · IPM ${fmt(item.overall_ipm, 2)}`;
  const action = view === "best" ? item.q1_action_hint || item.winner_segment : item.recommended_action || item.fatigue_status;
  return `
    <button class="case-card" data-id="${item.creative_id}">
      <img class="thumb" src="${esc(item.asset_url)}" alt="Creative ${esc(item.creative_id)}" loading="lazy">
      <span>
        <span class="case-title">${esc(item.headline || item.app_name || item.creative_id)}</span>
        <span class="meta-line">#${esc(item.creative_id)} · ${esc(item.vertical)} · ${esc(item.format)} · ${esc(item.hook_type || "")}</span>
        <span class="metric-row">
          <span class="badge ${badgeKind(item.creative_status)}">${esc(item.creative_status)}</span>
          <span class="badge">${esc(action || "")}</span>
        </span>
        <span class="meta-line">${primary}</span>
      </span>
    </button>
  `;
}

function badgeKind(status) {
  if (status === "top_performer") return "good";
  if (status === "fatigued") return "warn";
  if (status === "underperformer") return "bad";
  return "";
}

async function selectCreative(id, view) {
  state.selected = id;
  document.querySelectorAll(".case-card").forEach((card) => {
    card.classList.toggle("active", Number(card.dataset.id) === id);
  });
  const detail = await api(`/api/creative/${id}`);
  if (view === "home") {
    renderHomeDetail($("homeDetail"), detail);
  } else {
    const panel = view === "best" ? $("bestDetail") : $("tiredDetail");
    renderQuestionDetail(panel, detail, view);
    explainSelected(id, view);
  }
  if (state.activeView === "map") drawLandscape();
}

function renderHomeDetail(panel, detail) {
  const c = detail.creative || {};
  const q1 = detail.q1 || {};
  const q2 = detail.q2 || {};
  const q3 = detail.q3 || {};
  panel.innerHTML = `
    <div class="detail-layout">
      ${creativeImageBlock(c)}
      <div class="detail-copy">
        <h2>${esc(c.headline || c.app_name || c.creative_id)}</h2>
        <p class="meta-line">#${esc(c.creative_id)} · ${esc(c.app_name)} · ${esc(c.vertical)} · ${esc(c.format)}</p>
        <div class="metric-row">
          <span class="badge ${badgeKind(c.creative_status)}">${esc(c.creative_status)}</span>
          <span class="badge">${esc(c.theme)}</span>
          <span class="badge">${esc(c.hook_type)}</span>
          <span class="badge">${esc(c.cta_text)}</span>
        </div>
        ${featureSection("Performance", [
          ["Perf score", c.perf_score],
          ["Best score", q1.best_score],
          ["ROAS", c.overall_roas],
          ["IPM", c.overall_ipm],
          ["CTR", c.overall_ctr],
          ["CVR", c.overall_cvr],
        ])}
        ${featureSection("Health", [
          ["Fatigue", q2.fatigue_status],
          ["Repetition", q2.repetition_status],
          ["Risk", q2.creative_health_risk],
          ["Fatigue day", c.fatigue_day],
          ["CTR decay", c.ctr_decay_pct],
          ["CVR decay", c.cvr_decay_pct],
        ])}
        ${featureSection("Creative characteristics", [
          ["Language", c.language],
          ["Dominant color", c.dominant_color],
          ["Emotional tone", c.emotional_tone],
          ["Objective", c.objective],
          ["Age segment", c.target_age_segment],
          ["Target OS", c.target_os],
          ["Text density", c.text_density],
          ["Clutter", c.clutter_score],
          ["Brand visibility", c.brand_visibility_score],
          ["Motion", c.motion_score],
          ["Novelty", c.novelty_score],
          ["Gameplay", c.has_gameplay],
          ["UGC style", c.has_ugc_style],
          ["Discount badge", c.has_discount_badge],
        ])}
        <div class="evidence-box">
          <strong>Next test</strong>
          <p>${esc(q3.next_test || "No recommendation")}</p>
          <p class="meta-line">${esc([q3.suggestion_1, q3.suggestion_2, q3.suggestion_3].filter(Boolean).join(" "))}</p>
        </div>
      </div>
    </div>
  `;
  bindImageToggle(panel);
}

function featureSection(title, rows) {
  return `
    <section class="feature-section">
      <h2>${esc(title)}</h2>
      <div class="feature-grid">
        ${rows.map(([label, value]) => `
          <div class="feature-cell">
            <span>${esc(label)}</span>
            <strong>${fmtMaybe(value)}</strong>
          </div>
        `).join("")}
      </div>
    </section>
  `;
}

function fmtMaybe(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  const n = Number(value);
  if (Number.isFinite(n)) return Math.abs(n) >= 10 ? n.toFixed(2) : n.toFixed(3);
  return esc(value);
}

function renderQuestionDetail(panel, detail, view) {
  const c = detail.creative || {};
  const q1 = detail.q1 || {};
  const q2 = detail.q2 || {};
  const q3 = detail.q3 || {};
  panel.innerHTML = `
    <div class="detail-layout">
      ${creativeImageBlock(c)}
      <div class="detail-copy">
        <h2>${esc(c.headline || c.app_name || c.creative_id)}</h2>
        <p class="meta-line">#${esc(c.creative_id)} · ${esc(c.app_name)} · ${esc(c.vertical)} · ${esc(c.format)} · ${esc(c.theme)}</p>
        <div class="metric-row">
          <span class="badge ${badgeKind(c.creative_status)}">${esc(c.creative_status)}</span>
          <span class="badge good">${esc(q3.recommended_action || q1.action_hint || "")}</span>
          <span class="badge">${esc(q2.fatigue_status || "")}</span>
          <span class="badge">${esc(q2.repetition_status || "")}</span>
        </div>
        <div class="stats-grid">
          <div class="stat"><span>Best score</span><strong>${fmt(q1.best_score)}</strong></div>
          <div class="stat"><span>ROAS</span><strong>${fmt(c.overall_roas, 2)}</strong></div>
          <div class="stat"><span>IPM</span><strong>${fmt(c.overall_ipm, 2)}</strong></div>
          <div class="stat"><span>Risk</span><strong>${fmt(q2.creative_health_risk)}</strong></div>
        </div>
        <div class="evidence-box">
          <strong>Next test</strong>
          <p>${esc(q3.next_test || "No recommendation")}</p>
          <p class="meta-line">${esc([q3.suggestion_1, q3.suggestion_2, q3.suggestion_3].filter(Boolean).join(" "))}</p>
        </div>
        <div class="llm-box loading" id="llmBox-${view}">Generating explanation...</div>
        <h2 style="margin-top:18px;">Similar cases</h2>
        <div class="neighbor-strip">
          ${(detail.neighbors || []).slice(0, 8).map(neighborCard).join("")}
        </div>
      </div>
    </div>
  `;
  bindImageToggle(panel);
  panel.querySelectorAll("[data-neighbor-id]").forEach((button) => {
    button.addEventListener("click", () => openHomeCase(Number(button.dataset.neighborId)));
  });
}

function neighborCard(n) {
  const blockLine = n.similarity_clip === null || n.similarity_clip === undefined
    ? ""
    : `<div class="meta-line">CLIP ${fmt(n.similarity_clip)} · CNN ${fmt(n.similarity_cnn)} · ctx ${fmt(n.similarity_categorical_context)}</div>`;
  const finalLine = n.final_score === null || n.final_score === undefined ? "" : ` · final ${fmt(n.final_score)}`;
  return `
    <button class="neighbor" data-neighbor-id="${esc(n.neighbor_id)}">
      <strong>#${esc(n.neighbor_rank)} · ${esc(n.neighbor_id)}</strong>
      <div class="meta-line">${esc(n.neighbor_headline || n.neighbor_app_name || "")}</div>
      <div class="meta-line">${esc(n.neighbor_status)} · sim ${fmt(n.similarity)}${finalLine}</div>
      ${blockLine}
      <div class="meta-line">ROAS ${fmt(n.neighbor_roas, 2)} · IPM ${fmt(n.neighbor_ipm, 2)}</div>
    </button>
  `;
}

async function explainSelected(id, view) {
  const box = $(`llmBox-${view}`);
  if (!box) return;
  box.classList.add("loading");
  box.textContent = "Generating explanation...";
  const focus = view === "best" ? "q1" : "q2";
  const data = await api("/api/explain", {
    method: "POST",
    body: JSON.stringify({ creative_id: id, focus }),
  });
  box.classList.remove("loading");
  box.innerHTML = `${data.ok ? "" : `<p class="meta-line">Fallback local: ${esc(data.error || "Groq unavailable")}</p>`}${formatText(data.text || "")}`;
}

async function runBriefMatch(withExplanation) {
  let brief = $("briefInput").value.trim();
  if (!brief) {
    brief = "Nuevo anuncio mobile con CTA claro";
    $("briefInput").value = brief;
  }
  const body = {
    brief,
    vertical: $("briefVertical").value || null,
    format: $("briefFormat").value || null,
    limit: 8,
  };
  $("matchBriefBtn").disabled = true;
  $("matchBriefBtn").textContent = "Matching...";
  try {
    const data = await api("/api/brief/match", { method: "POST", body: JSON.stringify(body) });
    state.lastBrief = data;
    renderBriefMatch(data);
    if (withExplanation) await explainBrief(body);
  } finally {
    $("matchBriefBtn").disabled = false;
    $("matchBriefBtn").textContent = "Find patterns";
  }
}

function renderBriefMatch(data) {
  $("doList").innerHTML = (data.do || []).map((item) => `<li>${esc(item)}</li>`).join("");
  $("dontList").innerHTML = (data.dont || []).map((item) => `<li>${esc(item)}</li>`).join("");
  $("bestMatches").innerHTML = (data.best_cases || []).map(miniCard).join("");
  $("avoidMatches").innerHTML = (data.avoid_cases || []).map(miniCard).join("");
  const filters = data.filters || {};
  $("briefInference").innerHTML = `
    <span>Vertical: ${esc(filters.vertical || filters.suggested_vertical || "all")}</span>
    <span>${esc(filters.vertical_reason || "not_inferred")}</span>
    <span>${filters.query_has_known_terms ? "known terms" : "fallback ranking"}</span>
  `;
  $("briefExplanation").classList.add("hidden");
  document.querySelectorAll("[data-mini-id]").forEach((card) => {
    card.addEventListener("click", () => openHomeCase(Number(card.dataset.miniId)));
  });
  if (state.activeView === "map") drawLandscape();
}

function miniCard(item) {
  return `
    <button class="mini-card" data-mini-id="${esc(item.creative_id)}">
      <img class="thumb" src="${esc(item.asset_url)}" alt="Creative ${esc(item.creative_id)}" loading="lazy">
      <div>
        <div class="case-title">${esc(item.headline || item.creative_id)}</div>
        <div class="meta-line">#${esc(item.creative_id)} · ${esc(item.vertical)} · ${esc(item.format)} · ${esc(item.creative_status)}</div>
        <div class="metric-row">
          <span class="badge ${item.kind === "do" ? "good" : "bad"}">match ${fmt(item.match_score)}</span>
          <span class="badge">ROAS ${fmt(item.overall_roas, 2)}</span>
        </div>
      </div>
    </button>
  `;
}

async function explainBrief(body) {
  const box = $("briefExplanation");
  box.classList.remove("hidden");
  box.classList.add("loading");
  box.textContent = "Generating strategy...";
  const data = await api("/api/brief/explain", { method: "POST", body: JSON.stringify(body) });
  box.classList.remove("loading");
  box.innerHTML = `${data.ok ? "" : `<p class="meta-line">Fallback local: ${esc(data.error || "Groq unavailable")}</p>`}${formatText(data.text || "")}`;
  if (data.match) {
    state.lastBrief = data.match;
    renderBriefMatch(data.match);
    box.classList.remove("hidden");
    box.innerHTML = `${data.ok ? "" : `<p class="meta-line">Fallback local: ${esc(data.error || "Groq unavailable")}</p>`}${formatText(data.text || "")}`;
  }
}

async function openHomeCase(id) {
  $("searchInput").value = String(id);
  $("verticalFilter").value = "";
  $("formatFilter").value = "";
  await switchView("home", id);
}

async function loadLandscape() {
  const params = filtersQuery(false);
  const data = await api(`/api/landscape?${params.toString()}`);
  state.landscape = data.items || [];
  if (state.selected) {
    await selectMapCase(state.selected, { center: true, ensureVisible: true });
  } else {
    drawLandscape();
  }
}

async function searchMapSelection() {
  if (state.activeView !== "map") return;
  const q = $("searchInput").value.trim();
  const seq = ++state.mapSearchSeq;
  if (!q) {
    state.mapNeighbors = new Set();
    drawLandscape();
    return;
  }

  const params = filtersQuery(true);
  params.set("view", "all");
  params.set("limit", "1");
  let data = await api(`/api/creatives?${params.toString()}`);
  if (!data.items.length && ($("verticalFilter").value || $("formatFilter").value)) {
    const loose = new URLSearchParams({ view: "all", limit: "1", q });
    data = await api(`/api/creatives?${loose.toString()}`);
  }
  if (seq !== state.mapSearchSeq || !data.items.length) return;
  await selectMapCase(Number(data.items[0].creative_id), { center: true, ensureVisible: true });
}

function bindMapControls() {
  $("zoomInBtn").addEventListener("click", () => zoomMap(1.25));
  $("zoomOutBtn").addEventListener("click", () => zoomMap(0.8));
  $("resetMapBtn").addEventListener("click", () => {
    state.map.scale = 1;
    state.map.offsetX = 0;
    state.map.offsetY = 0;
    drawLandscape();
  });
  const canvas = $("landscapeCanvas");
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomMap(event.deltaY < 0 ? 1.12 : 0.88, event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });
  canvas.addEventListener("mousedown", (event) => {
    state.map.dragging = true;
    state.map.moved = false;
    state.map.lastX = event.clientX;
    state.map.lastY = event.clientY;
  });
  window.addEventListener("mousemove", (event) => {
    if (!state.map.dragging) return;
    const dx = event.clientX - state.map.lastX;
    const dy = event.clientY - state.map.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 3) state.map.moved = true;
    state.map.offsetX += dx;
    state.map.offsetY += dy;
    state.map.lastX = event.clientX;
    state.map.lastY = event.clientY;
    drawLandscape();
  });
  window.addEventListener("mouseup", () => {
    state.map.dragging = false;
  });
  canvas.addEventListener("click", (event) => {
    if (state.map.moved) return;
    const rect = canvas.getBoundingClientRect();
    const point = nearestMapPoint(event.clientX - rect.left, event.clientY - rect.top);
    if (point) selectMapPoint(point.creative_id);
  });
}

function zoomMap(factor, x = null, y = null) {
  const canvas = $("landscapeCanvas");
  const rect = canvas.getBoundingClientRect();
  const cx = x === null || x === undefined ? rect.width / 2 : x;
  const cy = y === null || y === undefined ? rect.height / 2 : y;
  const nextScale = Math.max(0.65, Math.min(8, state.map.scale * factor));
  const actual = nextScale / state.map.scale;
  state.map.offsetX = cx - rect.width / 2 - (cx - rect.width / 2 - state.map.offsetX) * actual;
  state.map.offsetY = cy - rect.height / 2 - (cy - rect.height / 2 - state.map.offsetY) * actual;
  state.map.scale = nextScale;
  drawLandscape();
}

function drawLandscape() {
  const canvas = $("landscapeCanvas");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, rect.width, rect.height);
  drawMapGrid(ctx, rect);
  state.mapPointById = new Map();
  const highlight = highlightSets();
  for (const point of state.landscape) {
    const { x, y } = pointToCanvas(point, rect);
    const id = Number(point.creative_id);
    state.mapPointById.set(id, { ...point, x, y });
    ctx.beginPath();
    ctx.arc(x, y, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = statusColor(point.creative_status);
    ctx.globalAlpha = highlight.any.size && !highlight.any.has(id) ? 0.24 : 0.78;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  drawHighlightSet(ctx, highlight.neighbors, "#2f6f8f", 6.5, "");
  drawHighlightSet(ctx, highlight.best, "#2aa79b", 8, "best match");
  drawHighlightSet(ctx, highlight.avoid, "#d86161", 8, "weak match");
  drawSelectedPoint(ctx);
  updateMapInfo(highlight);
}

function drawMapGrid(ctx, rect) {
  ctx.strokeStyle = "#e7e1f0";
  ctx.lineWidth = 1;
  const step = 80 * state.map.scale;
  const startX = ((state.map.offsetX % step) + step) % step;
  const startY = ((state.map.offsetY % step) + step) % step;
  for (let x = startX; x < rect.width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rect.height);
    ctx.stroke();
  }
  for (let y = startY; y < rect.height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }
}

function pointToCanvas(point, rect) {
  const baseX = ((Number(point.landscape_x) + 1) / 2) * (rect.width - 32) + 16;
  const baseY = ((Number(point.landscape_y) + 1) / 2) * (rect.height - 32) + 16;
  return {
    x: (baseX - rect.width / 2) * state.map.scale + rect.width / 2 + state.map.offsetX,
    y: (baseY - rect.height / 2) * state.map.scale + rect.height / 2 + state.map.offsetY,
  };
}

function nearestMapPoint(x, y) {
  let best = null;
  let bestDist = Infinity;
  for (const point of state.mapPointById.values()) {
    const dist = (point.x - x) ** 2 + (point.y - y) ** 2;
    if (dist < bestDist) {
      best = point;
      bestDist = dist;
    }
  }
  return bestDist <= 18 ** 2 ? best : null;
}

async function selectMapPoint(id) {
  await selectMapCase(id, { center: false, ensureVisible: false });
}

async function selectMapCase(id, options = {}) {
  const { center = true, ensureVisible = true } = options;
  state.selected = Number(id);
  const detail = await api(`/api/creative/${id}`);
  state.mapNeighbors = new Set((detail.neighbors || []).map((item) => Number(item.neighbor_id)));

  if (ensureVisible && !state.landscape.some((point) => Number(point.creative_id) === Number(id))) {
    $("verticalFilter").value = "";
    $("formatFilter").value = "";
    const data = await api("/api/landscape");
    state.landscape = data.items || [];
  }

  renderMapDetail(detail);
  if (center) centerMapOnCreative(id);
  drawLandscape();
}

function centerMapOnCreative(id) {
  const point = state.landscape.find((item) => Number(item.creative_id) === Number(id));
  if (!point) return;
  const canvas = $("landscapeCanvas");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  state.map.scale = Math.max(state.map.scale, 1.35);
  const baseX = ((Number(point.landscape_x) + 1) / 2) * (rect.width - 32) + 16;
  const baseY = ((Number(point.landscape_y) + 1) / 2) * (rect.height - 32) + 16;
  state.map.offsetX = -((baseX - rect.width / 2) * state.map.scale);
  state.map.offsetY = -((baseY - rect.height / 2) * state.map.scale);
}

function renderMapDetail(detail) {
  const c = detail.creative || {};
  const q1 = detail.q1 || {};
  const q2 = detail.q2 || {};
  const q3 = detail.q3 || {};
  $("mapDetail").innerHTML = `
    <div class="map-media">
      <img class="map-thumb" src="${esc(c.asset_url)}" alt="Creative ${esc(c.creative_id)}">
      <button
        class="soft-btn full-btn image-detail-toggle"
        data-image-toggle
        data-detail-url="${esc(c.visualized_url || c.asset_url || "")}"
        data-image-title="${esc(c.headline || c.app_name || `Creative ${c.creative_id}`)}"
      >Detail</button>
    </div>
    <h2>${esc(c.headline || c.app_name || c.creative_id)}</h2>
    <p class="meta-line">#${esc(c.creative_id)} · ${esc(c.vertical)} · ${esc(c.format)}</p>
    <div class="metric-row">
      <span class="badge ${badgeKind(c.creative_status)}">${esc(c.creative_status)}</span>
      <span class="badge">score ${fmt(q1.best_score)}</span>
      <span class="badge">risk ${fmt(q2.creative_health_risk)}</span>
    </div>
    <div class="evidence-box">
      <strong>Next test</strong>
      <p>${esc(q3.next_test || "No recommendation")}</p>
    </div>
    <h2 class="map-neighbor-title">Similar CBR cases</h2>
    <div class="map-neighbor-list">
      ${(detail.neighbors || []).slice(0, 6).map(neighborCard).join("")}
    </div>
    <button class="primary-btn full-btn" id="openMapCaseBtn">Open in Home</button>
  `;
  bindImageToggle($("mapDetail"));
  $("openMapCaseBtn").addEventListener("click", () => openHomeCase(Number(c.creative_id)));
  $("mapDetail").querySelectorAll("[data-neighbor-id]").forEach((button) => {
    button.addEventListener("click", () => selectMapCase(Number(button.dataset.neighborId), { center: true, ensureVisible: true }));
  });
}

function highlightSets() {
  const bestCases = state.lastBrief && state.lastBrief.best_cases ? state.lastBrief.best_cases : [];
  const avoidCases = state.lastBrief && state.lastBrief.avoid_cases ? state.lastBrief.avoid_cases : [];
  const best = new Set(bestCases.map((item) => Number(item.creative_id)));
  const avoid = new Set(avoidCases.map((item) => Number(item.creative_id)));
  const neighbors = state.mapNeighbors || new Set();
  const any = new Set([...best, ...avoid, ...neighbors]);
  if (state.selected) any.add(Number(state.selected));
  return { best, avoid, neighbors, any };
}

function drawHighlightSet(ctx, ids, color, radius, label) {
  for (const id of ids) {
    const point = state.mapPointById.get(Number(id));
    if (!point) continue;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    if (label) {
      ctx.fillStyle = color;
      ctx.font = "700 11px Inter, sans-serif";
      ctx.fillText(label, point.x + radius + 4, point.y + 4);
    }
  }
}

function drawSelectedPoint(ctx) {
  if (!state.selected) return;
  const selected = state.mapPointById.get(Number(state.selected));
  if (!selected) return;
  ctx.beginPath();
  ctx.arc(selected.x, selected.y, 11, 0, Math.PI * 2);
  ctx.strokeStyle = "#4b2fa3";
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = "#211a32";
  ctx.font = "700 12px Inter, sans-serif";
  ctx.fillText(String(state.selected), selected.x + 13, selected.y - 8);
}

function updateMapInfo(highlight) {
  const info = $("mapInfo");
  if (!info) return;
  const selected = state.selected || "ninguno";
  const method = state.landscape[0] ? state.landscape[0].landscape_method : "unknown";
  info.innerHTML = `
    <span>Seleccionado: ${esc(selected)}</span>
    <span>Similar CBR: ${highlight.neighbors.size}</span>
    <span>Brief winners: ${highlight.best.size}</span>
    <span>Brief weak cases: ${highlight.avoid.size}</span>
    <span>Projection: ${esc(method)}</span>
    <span>Zoom: ${state.map.scale.toFixed(2)}x</span>
  `;
}

function statusColor(status) {
  if (status === "top_performer") return "#2aa79b";
  if (status === "fatigued") return "#b88712";
  if (status === "underperformer") return "#d86161";
  return "#6d49d8";
}

function formatText(text) {
  return esc(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

function debounce(fn, ms) {
  let timeout = null;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), ms);
  };
}

window.addEventListener("resize", () => {
  if (state.activeView === "map") drawLandscape();
});

init().catch((err) => {
  console.error(err);
});



