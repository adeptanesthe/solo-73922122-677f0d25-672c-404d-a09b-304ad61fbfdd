/* 野生动物追踪串号校正与计数台 —— 前端逻辑 */
"use strict";

const PALETTE = [
  "#ff4d4f", "#36cfc9", "#ffa940", "#73d13d", "#9254de",
  "#40a9ff", "#f759ab", "#a0d911", "#ffc53d", "#597ef7",
  "#ff7a45", "#5cdbd3", "#b37feb", "#95de64", "#69c0ff",
];

const state = {
  sessionId: null,
  version: 0,
  nFrames: 0,
  detections: [],
  judgments: [],          // [{a, b, type}]，a/b 为框号
  params: { w_dist: 1, w_app: 10000, penalty: 15000 },
  images: [],             // HTMLImageElement
  currentFrame: 0,
  selection: [],          // 已选框号（至多 2 个）
  result: null,           // 当前展示的求解结果
  jobId: null,
  pollTimer: null,
  locateJobId: null,      // 冲突定位任务
  locateTimer: null,
  locateRunning: false,
  conflict: null,         // {pairKeys:Set<string>, judgments:[...], checks:int}
  locateMessage: null,    // {text, tone}
  limitJudgments: 8,      // 会话下发的定位条数上限
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const detById = (id) => state.detections.find((d) => d.id === id);
const pairKey = (a, b) => [a, b].sort().join("||");

function trajIndexOf(boxId) {
  if (!state.result || !state.result.feasible) return null;
  for (let i = 0; i < state.result.trajectories.length; i++) {
    if (state.result.trajectories[i].boxes.some((b) => b.id === boxId)) return i;
  }
  return null;
}

/* ---------------------------------------------------------------- API */

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return resp.json();
}

function showError(targetId, err) {
  const el = $(targetId);
  el.textContent = `错误：${err.message}`;
  el.classList.remove("muted");
  el.style.color = "#ff9d97";
}

/* ------------------------------------------------------------ 会话载入 */

function loadSession(payload) {
  state.sessionId = payload.session_id;
  state.version = payload.version;
  state.nFrames = payload.n_frames;
  state.detections = payload.detections;
  state.judgments = payload.judgments;
  state.params = payload.params;
  state.currentFrame = 0;
  state.selection = [];
  state.result = null;
  state.jobId = null;
  state.locateJobId = null;
  state.locateRunning = false;
  state.conflict = null;
  if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
  if (state.locateTimer) { clearTimeout(state.locateTimer); state.locateTimer = null; }
  if (payload.limits && payload.limits.location_judgments) {
    state.limitJudgments = payload.limits.location_judgments;
  }

  state.images = [];
  for (let i = 0; i < state.nFrames; i++) {
    const img = new Image();
    img.src = `/api/sessions/${state.sessionId}/frames/${i}`;
    img.onload = () => { if (i === state.currentFrame) draw(); };
    state.images.push(img);
  }

  $("p-wdist").value = payload.params.w_dist;
  $("p-wapp").value = payload.params.w_app;
  $("p-penalty").value = payload.params.penalty;

  const info = $("session-info");
  info.classList.add("muted");
  info.style.color = "";
  info.innerHTML =
    `会话 <b>${payload.session_id}</b>：${payload.n_frames} 帧、` +
    `${payload.detections.length} 个检测框。` +
    (payload.description ? `<br>${payload.description}` : "");

  $("solve-btn").disabled = false;
  $("no-session").style.display = "none";
  renderAll();
}

async function uploadSession() {
  const imgs = $("img-input").files;
  const jsons = $("json-input").files;
  if (imgs.length === 0 || imgs.length > 10) {
    alert("请选择 1~10 张帧图片。");
    return;
  }
  if (jsons.length !== 1) {
    alert("请选择 1 个检测 JSON 文件。");
    return;
  }
  const fd = new FormData();
  for (const f of imgs) fd.append("images", f);
  fd.append("detections", jsons[0]);
  try {
    const payload = await api("/api/sessions", { method: "POST", body: fd });
    loadSession(payload);
  } catch (err) {
    showError("session-info", err);
  }
}

async function loadExample() {
  try {
    const payload = await api("/api/example", { method: "POST" });
    loadSession(payload);
  } catch (err) {
    showError("session-info", err);
  }
}

/* ------------------------------------------------------------ 画面渲染 */

function detsInFrame(f) {
  return state.detections.filter((d) => d.frame === f);
}

function draw() {
  const img = state.images[state.currentFrame];
  if (!img || !img.complete || img.naturalWidth === 0) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  for (const d of detsInFrame(state.currentFrame)) {
    const [x, y, w, h] = d.rect;
    const ti = trajIndexOf(d.id);
    const color = ti !== null ? PALETTE[ti % PALETTE.length] : "#ffe600";
    const selected = state.selection.includes(d.id);

    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeStyle = selected ? "#ffffff" : color;
    ctx.strokeRect(x, y, w, h);

    const label = `#${d.id}` + (ti !== null ? ` T${ti + 1}` : "");
    ctx.font = "bold 12px sans-serif";
    const tw = ctx.measureText(label).width;
    const ly = y >= 16 ? y - 16 : y + h + 2;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(x, ly, tw + 6, 15);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 3, ly + 12);
  }
}

function renderFrameNav() {
  const n = state.nFrames || 0;
  $("frame-indicator").textContent =
    n === 0 ? "— / —" : `第 ${state.currentFrame + 1} / ${n} 帧（本帧 ${detsInFrame(state.currentFrame).length} 框）`;
  $("prev-frame").disabled = n === 0 || state.currentFrame === 0;
  $("next-frame").disabled = n === 0 || state.currentFrame >= n - 1;
}

function switchFrame(delta) {
  const next = state.currentFrame + delta;
  if (next < 0 || next >= state.nFrames) return;
  state.currentFrame = next;
  renderFrameNav();
  draw();
}

/* ------------------------------------------------------------ 选择与标记 */

function renderSelection() {
  const info = $("selection-info");
  if (state.selection.length === 0) {
    info.textContent = "未选择框。点击画面中的检测框进行选择（可跨帧选两个）。";
  } else {
    const parts = state.selection.map((id) => {
      const d = detById(id);
      return d ? `帧 ${d.frame} 的 #${id}` : `#${id}`;
    });
    let text = `已选：${parts.join(" 与 ")}`;
    if (state.selection.length === 2) {
      const j = findJudgment(state.selection[0], state.selection[1]);
      if (j) text += `（当前判断：${j.type === "same" ? "同只" : "异只"}）`;
    }
    info.textContent = text;
  }
  const two = state.selection.length === 2;
  $("mark-same").disabled = !two;
  $("mark-diff").disabled = !two;
  $("mark-clear").disabled = !two || !findJudgment(state.selection[0], state.selection[1]);
}

function findJudgment(a, b) {
  const key = pairKey(a, b);
  return state.judgments.find((j) => pairKey(j.a, j.b) === key) || null;
}

function onCanvasClick(ev) {
  if (!state.sessionId) return;
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const px = (ev.clientX - rect.left) * sx;
  const py = (ev.clientY - rect.top) * sy;
  const dets = detsInFrame(state.currentFrame);
  for (let i = dets.length - 1; i >= 0; i--) {
    const [x, y, w, h] = dets[i].rect;
    if (px >= x && px <= x + w && py >= y && py <= y + h) {
      toggleSelect(dets[i].id);
      return;
    }
  }
}

function toggleSelect(id) {
  const i = state.selection.indexOf(id);
  if (i >= 0) {
    state.selection.splice(i, 1);
  } else {
    if (state.selection.length >= 2) state.selection.shift();
    state.selection.push(id);
  }
  renderSelection();
  draw();
}

async function setJudgment(type) {
  if (state.selection.length !== 2) return;
  const [a, b] = state.selection;
  const key = pairKey(a, b);
  const rest = state.judgments.filter((j) => pairKey(j.a, j.b) !== key);
  const next = type ? [...rest, { a, b, type }] : rest;
  invalidateResult(); // 修改即撤下旧结果与冲突标记
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/judgments`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ judgments: next }),
    });
    state.version = resp.version;
    state.judgments = next;
    renderJudgments();
    renderSelection();
    draw();
  } catch (err) {
    alert(`保存判断失败：${err.message}`);
  }
}

async function removeJudgment(a, b) {
  const key = pairKey(a, b);
  const next = state.judgments.filter((j) => pairKey(j.a, j.b) !== key);
  invalidateResult();
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/judgments`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ judgments: next }),
    });
    state.version = resp.version;
    state.judgments = next;
    renderJudgments();
    renderSelection();
    draw();
  } catch (err) {
    alert(`删除判断失败：${err.message}`);
  }
}

function isConflictJudgment(j) {
  return !!state.conflict && state.conflict.pairKeys.has(pairKey(j.a, j.b));
}

function renderJudgments() {
  const ul = $("judgment-list");
  ul.innerHTML = "";
  $("judgment-count").textContent = `（${state.judgments.length} 条）`;
  for (const j of state.judgments) {
    const da = detById(j.a);
    const db = detById(j.b);
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = `tag ${j.type === "same" ? "same" : "diff"}`;
    tag.textContent = j.type === "same" ? "同只" : "异只";
    const text = document.createElement("span");
    text.textContent = `帧${da ? da.frame : "?"} #${j.a} ↔ 帧${db ? db.frame : "?"} #${j.b}`;
    li.append(tag, text);
    if (isConflictJudgment(j)) {
      li.classList.add("conflict");
      const mark = document.createElement("span");
      mark.className = "conflict-mark";
      mark.textContent = "⚑ 冲突组";
      li.append(mark);
    }
    const del = document.createElement("button");
    del.textContent = "删除";
    del.onclick = () => removeJudgment(j.a, j.b);
    li.append(del);
    ul.appendChild(li);
  }
  renderConflictPanel();
}

/* ------------------------------------------------------------ 参数 */

async function applyParams() {
  if (!state.sessionId) return;
  const wDist = parseFloat($("p-wdist").value);
  const wApp = parseFloat($("p-wapp").value);
  const penalty = parseFloat($("p-penalty").value);
  if (![wDist, wApp, penalty].every((v) => Number.isFinite(v) && v >= 0)) {
    alert("参数必须是有限非负数。");
    return;
  }
  invalidateResult(); // 修改即撤下旧结果
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/params`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ w_dist: wDist, w_app: wApp, penalty }),
    });
    state.version = resp.version;
    state.params = { w_dist: wDist, w_app: wApp, penalty };
  } catch (err) {
    alert(`保存参数失败：${err.message}`);
  }
}

/* ------------------------------------------------------------ 求解 */

function invalidateResult() {
  // 输入一改，旧结果与旧冲突标记都必须撤下
  clearConflict();
  if (state.result !== null) {
    state.result = null;
    const banner = $("result-banner");
    banner.className = "banner stale";
    banner.textContent = "输入已修改，旧结果已撤下。请重新计算。";
    $("result-summary").innerHTML = "";
    $("trajectory-list").innerHTML = "";
    draw();
  }
  renderConflictPanel();
}

/* ------------------------------------------------------------ 冲突定位 */

function clearConflict() {
  state.conflict = null;
  state.locateMessage = null;
  if (state.locateTimer) { clearTimeout(state.locateTimer); state.locateTimer = null; }
  state.locateJobId = null;
  state.locateRunning = false;
}

function setLocateMessage(text, tone) {
  state.locateMessage = text ? { text, tone: tone || "muted" } : null;
}

function renderConflictPanel() {
  const panel = $("conflict-panel");
  const hint = $("conflict-hint");
  const status = $("conflict-status");
  const btn = $("locate-btn");
  const group = $("conflict-group");
  const n = state.judgments.length;
  const overLimit = n > state.limitJudgments;
  const infeasible = !!(
    state.result && state.result.feasible === false && state.result.reason === "infeasible"
  );
  const show = overLimit || infeasible || !!state.conflict || state.locateRunning ||
    !!state.locateMessage;
  panel.hidden = !show;
  if (!show) {
    hint.textContent = "";
    status.textContent = "";
    group.innerHTML = "";
    btn.hidden = true;
    return;
  }

  group.innerHTML = "";
  if (state.conflict) {
    for (const j of state.conflict.judgments) {
      const da = detById(j.a);
      const db = detById(j.b);
      const li = document.createElement("li");
      const tag = document.createElement("span");
      tag.className = `tag ${j.type === "same" ? "same" : "diff"}`;
      tag.textContent = j.type === "same" ? "同只" : "异只";
      const text = document.createElement("span");
      text.textContent =
        `帧${da ? da.frame : "?"} #${j.a} ↔ 帧${db ? db.frame : "?"} #${j.b}`;
      li.append(tag, text);
      group.appendChild(li);
    }
  }

  if (overLimit) {
    hint.textContent =
      `当前 ${n} 条判断，超过 ${state.limitJudgments} 条，暂不提供冲突定位；仍可正常求解。`;
    btn.hidden = true;
  } else if (infeasible) {
    hint.textContent = state.locateRunning
      ? "正在逐条检查判断组合的可行性…"
      : `当前已确认无解。点击定位，返回一组单独即可导致不可行的判断` +
        `（删去其中任意一条即可行）。`;
    btn.hidden = false;
    btn.disabled = state.locateRunning;
  } else {
    hint.textContent = state.conflict
      ? "已定位的冲突判断如下；改判其中任意一条后，旧标记会自动撤下。"
      : "";
    btn.hidden = true;
  }
  const msg = state.locateMessage;
  status.textContent = msg ? msg.text : "";
  status.style.color =
    msg && msg.tone === "err" ? "#ff9d97"
      : msg && msg.tone === "warn" ? "#ffd887"
        : "";
}

async function locateConflict() {
  if (!state.sessionId || state.locateRunning) return;
  setLocateMessage("正在启动冲突定位…");
  renderConflictPanel();
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/locate-conflict`, {
      method: "POST",
    });
    if (resp.status === "too_many") {
      setLocateMessage(resp.message, "warn");
      renderConflictPanel();
      return;
    }
    state.locateJobId = resp.job_id;
    state.locateRunning = true;
    setLocateMessage(`检查中（输入版本 v${resp.version}）…`);
    renderConflictPanel();
    pollLocate(resp.job_id, resp.version);
  } catch (err) {
    state.locateRunning = false;
    setLocateMessage(`冲突定位未完成：${err.message} 不猜测冲突判断。`, "err");
    renderConflictPanel();
  }
}

async function pollLocate(jobId, version) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (jobId !== state.locateJobId) return; // 已被更新的定位任务取代
    if (job.status === "running") {
      state.locateTimer = setTimeout(() => pollLocate(jobId, version), 400);
      return;
    }
    state.locateRunning = false;
    if (job.status === "error") {
      setLocateMessage(
        `冲突定位未完成：${job.error || "定位任务失败"}。不猜测冲突判断。`, "err");
      renderConflictPanel();
      return;
    }
    if (job.version !== state.version) {
      setLocateMessage(
        `定位完成，但其基于旧输入（v${job.version}，当前 v${state.version}），` +
        "结果已作废；未标记任何判断，请重新定位。", "warn");
      renderConflictPanel();
      return;
    }
    const r = job.result || {};
    if (r.status === "found") {
      state.conflict = {
        pairKeys: new Set(
          (r.pair_keys && r.pair_keys.length)
            ? r.pair_keys
            : r.judgments.map((j) => pairKey(j.a, j.b))
        ),
        judgments: r.judgments,
        checks: r.checks,
      };
      setLocateMessage(
        `${r.message}（共 ${r.checks} 次可行性检查）`, "err");
    } else if (r.status === "incomplete") {
      state.conflict = null;
      setLocateMessage(r.message || "冲突定位未完成：可行性检查超时或出错。不猜测冲突判断。", "err");
    } else if (r.status === "too_many") {
      state.conflict = null;
      setLocateMessage(r.message, "warn");
    } else if (r.status === "feasible") {
      state.conflict = null;
      setLocateMessage(r.message || "当前判断组合可行，未发现冲突。", "warn");
    } else {
      state.conflict = null;
      setLocateMessage("冲突定位未完成：返回了无法识别的结果。不猜测冲突判断。", "err");
    }
    renderJudgments();
  } catch (err) {
    state.locateRunning = false;
    setLocateMessage(`冲突定位未完成：${err.message} 不猜测冲突判断。`, "err");
    renderConflictPanel();
  }
}

async function solve() {
  if (!state.sessionId) return;
  const status = $("solve-status");
  status.classList.add("muted");
  status.style.color = "";
  setLocateMessage(null);
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/solve`, { method: "POST" });
    state.jobId = resp.job_id;
    status.textContent = `计算中（输入版本 v${resp.version}）… 可继续修改判断与参数。`;
    if (state.pollTimer) clearTimeout(state.pollTimer);
    pollJob(resp.job_id);
  } catch (err) {
    showError("solve-status", err);
  }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (jobId !== state.jobId) return; // 已被更新的任务取代
    if (job.status === "running") {
      state.pollTimer = setTimeout(() => pollJob(jobId), 400);
      return;
    }
    const status = $("solve-status");
    if (job.status === "error") {
      status.textContent = job.error || "求解失败。";
      status.style.color = "#ff9d97";
      return;
    }
    if (job.version !== state.version) {
      // 迟到结果：输入在计算期间已变化，丢弃不展示
      status.textContent =
        `任务完成，但其基于旧输入（v${job.version}，当前 v${state.version}），结果已作废。请重新计算。`;
      status.style.color = "#ffd887";
      const banner = $("result-banner");
      banner.className = "banner stale";
      banner.textContent = "存在基于旧输入的迟到结果，已自动丢弃，未覆盖当前输入。";
      return;
    }
    status.textContent = `计算完成（输入版本 v${job.version}）。`;
    status.style.color = "";
    state.result = job.result;
    renderResult();
    renderJudgments();
    renderConflictPanel();
    draw();
  } catch (err) {
    showError("solve-status", err);
  }
}

function renderResult() {
  const banner = $("result-banner");
  const summary = $("result-summary");
  const list = $("trajectory-list");
  const r = state.result;
  if (!r) return;
  if (r.feasible) {
    // 可行结果意味着旧的冲突标记已不成立
    state.conflict = null;
    setLocateMessage(null);
  }
  if (!r.feasible) {
    const known = r.reason === "infeasible" || (r.message || "").startsWith("无解");
    banner.className = "banner err";
    banner.textContent = known
      ? (r.message || "无解：约束冲突。")
      : `未得到结果：${r.message || "求解失败。"}`;
    summary.innerHTML = known
      ? "<b>不提供动物计数。</b>"
      : "<b>本次未得到可用结果，不提供动物计数。</b>";
    list.innerHTML = "";
    return;
  }
  banner.className = "banner ok";
  banner.textContent = r.message || "求解成功。";
  summary.innerHTML =
    `<div><span class="count">${r.animal_count}</span>只动物（按轨迹条数计）</div>` +
    `<div>轨迹条数：${r.n_trajectories}</div>` +
    `<div>总代价：${r.total_cost.toFixed(3)}` +
    `（关联代价 ${r.assoc_cost.toFixed(3)} + 轨迹罚分 ${r.penalty_cost.toFixed(3)}）</div>`;
  list.innerHTML = "";
  r.trajectories.forEach((t, i) => {
    const div = document.createElement("div");
    div.className = "traj";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = PALETTE[i % PALETTE.length];
    const boxes = document.createElement("span");
    boxes.className = "boxes";
    boxes.textContent = `轨迹 T${i + 1}：` + t.boxes.map((b) => `帧${b.frame} #${b.id}`).join(" → ");
    const len = document.createElement("span");
    len.className = "len";
    len.textContent = `（${t.length} 框）`;
    div.append(sw, boxes, len);
    list.appendChild(div);
  });
}

/* ------------------------------------------------------------ 汇总渲染 */

function renderAll() {
  renderFrameNav();
  renderSelection();
  renderJudgments();
  draw();
  const banner = $("result-banner");
  banner.className = "banner muted";
  banner.textContent = "尚无结果。";
  $("result-summary").innerHTML = "";
  $("trajectory-list").innerHTML = "";
  renderConflictPanel();
}

/* ------------------------------------------------------------ 事件绑定 */

$("upload-btn").addEventListener("click", uploadSession);
$("example-btn").addEventListener("click", loadExample);
$("prev-frame").addEventListener("click", () => switchFrame(-1));
$("next-frame").addEventListener("click", () => switchFrame(1));
$("mark-same").addEventListener("click", () => setJudgment("same"));
$("mark-diff").addEventListener("click", () => setJudgment("different"));
$("mark-clear").addEventListener("click", () => setJudgment(null));
$("solve-btn").addEventListener("click", solve);
$("locate-btn").addEventListener("click", locateConflict);
canvas.addEventListener("click", onCanvasClick);
for (const id of ["p-wdist", "p-wapp", "p-penalty"]) {
  $(id).addEventListener("change", applyParams);
}
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.key === "ArrowLeft") switchFrame(-1);
  if (ev.key === "ArrowRight") switchFrame(1);
});
