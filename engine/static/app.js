const POLL_MS = 3000;
const CHART_POLL_MS = 10000;

function $(id) { return document.getElementById(id); }
function pnlClass(val) { return val > 0 ? "pnl-positive" : val < 0 ? "pnl-negative" : ""; }
function fmtPnl(val) { return (val > 0 ? "+" : "") + "$" + val.toFixed(2); }
function fmtPct(val) { return (val > 0 ? "+" : "") + val.toFixed(1) + "%"; }
function fmtVol(v) {
  const n = parseFloat(v || 0);
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return n.toFixed(0);
}
function shortTs(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function fetchJSON(url) { return (await fetch(url)).json(); }

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------
async function refreshStatus() {
  const d = await fetchJSON("/api/status");
  const bal = $("balance");
  if (d.balance !== null && d.balance !== undefined) {
    bal.textContent = "$" + d.balance.toFixed(2);
  } else {
    bal.textContent = "—";
  }
  $("api-key").textContent = d.api_key || "";
  const badge = $("engine-badge");
  badge.textContent = d.engine_state;
  badge.className = "engine-badge " + d.engine_state;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
async function refreshStats() {
  const d = await fetchJSON("/api/stats");
  const retEl = $("return-pct");
  retEl.textContent = fmtPct(d.total_return_pct);
  retEl.className = "stat-card-value " + (d.total_return_pct >= 0 ? "positive" : "negative");
  $("win-rate").textContent = d.win_rate + "%";
  $("total-bets").textContent = d.total_bets;
  $("total-matches").textContent = d.total_matches + " (" + d.skipped_matches + " skip)";
  $("max-dd").textContent = d.max_drawdown + "%";
}

// ---------------------------------------------------------------------------
// Upcoming / Positions / History / Log (unchanged logic, minor cleanup)
// ---------------------------------------------------------------------------
const _upcomingExpanded = new Set();
function _matchKey(m) { return `${m.match_date||""}|${m.team1||""}|${m.team2||""}`; }

function _previewBlock(m) {
  if (!m.preview) return '<div class="preview-pending">Computing preview…</div>';
  if (!m.preview.length) return '<div class="preview-pending">No preview available</div>';
  const t1Abbr = TEAM_ABBREVS[m.team1] || m.team1 || "T1";
  const t2Abbr = TEAM_ABBREVS[m.team2] || m.team2 || "T2";
  const lines = m.preview.map(s => {
    const winnerAbbr = s.toss_winner === m.team1 ? t1Abbr : t2Abbr;
    const t1p = s.team1_win_prob * 100;
    const fav = t1p >= 50
      ? `${t1Abbr} ${t1p.toFixed(0)}%`
      : `${t2Abbr} ${(100 - t1p).toFixed(0)}%`;
    return `<div class="preview-row"><span class="preview-scen">${winnerAbbr} won toss, ${s.toss_decision}</span><span class="preview-fav">${fav}</span></div>`;
  }).join("");
  return `<div class="preview-block"><div class="preview-title">Pre-toss preview · last-known XIs · informational only</div>${lines}</div>`;
}

async function refreshUpcoming() {
  const data = await fetchJSON("/api/upcoming");
  const body = $("upcoming-body"), empty = $("upcoming-empty");
  if (!data.length) { body.innerHTML = ""; empty.style.display = ""; return; }
  empty.style.display = "none";

  const liveKeys = new Set(data.map(_matchKey));
  for (const k of Array.from(_upcomingExpanded)) {
    if (!liveKeys.has(k)) _upcomingExpanded.delete(k);
  }

  body.innerHTML = data.map(m => {
    const pred = m.model_prediction;
    const modelStr = pred ? `${(pred.model_prob*100).toFixed(0)}% ${pred.team}` : "—";
    const mktStr = pred ? `${(pred.market_price*100).toFixed(0)}¢` : "—";
    const edgeStr = pred ? fmtPct(pred.edge*100) : "—";
    const statusBadge = m.status === "awaiting_toss"
      ? '<span class="badge badge-awaiting">Awaiting toss</span>'
      : pred ? '<span class="badge badge-signal">Signal ready</span>'
      : `<span class="badge">${m.status||"—"}</span>`;
    const key = _matchKey(m);
    const open = _upcomingExpanded.has(key);
    const caret = `<span class="row-caret">${open ? "▾" : "▸"}</span>`;
    const main = `<tr class="upcoming-row${open ? " open" : ""}" data-key="${key}"><td class="caret-cell">${caret}</td><td>${m.match_date||""}</td><td>${m.team1||"?"} vs ${m.team2||"?"}</td><td>${modelStr}</td><td>${mktStr}</td><td>${edgeStr}</td><td>${statusBadge}</td></tr>`;
    const detail = `<tr class="upcoming-detail${open ? "" : " hidden"}" data-key="${key}"><td></td><td colspan="6">${_previewBlock(m)}</td></tr>`;
    return main + detail;
  }).join("");
}

async function refreshPositions() {
  const data = await fetchJSON("/api/positions");
  const body = $("positions-body"), empty = $("positions-empty");
  if (!data.length) { body.innerHTML = ""; empty.style.display = ""; return; }
  empty.style.display = "none";
  body.innerHTML = data.map(p => {
    const cur = p.current_price || p.entry_price;
    const unrealized = (cur - p.entry_price) * (p.contracts || 0);
    return `<tr><td>${p.team1||"?"} vs ${p.team2||"?"}</td><td>${p.team} (${p.side})</td><td>${p.entry_price.toFixed(2)}¢</td><td>${cur.toFixed(2)}¢</td><td>${(p.entry_price-0.15).toFixed(2)}¢</td><td class="${pnlClass(unrealized)}">${fmtPnl(unrealized)}</td></tr>`;
  }).join("");
}

let _historyData = [];
let _historyFilter = "all";
function renderHistory() {
  const body = $("history-body"), empty = $("history-empty");
  let data = _historyData;
  const isSkipStatus = s => s && (s==="skipped"||s==="no_market"||s.startsWith("skipped_"));
  if (_historyFilter === "bets") data = data.filter(h => !isSkipStatus(h.status));
  else if (_historyFilter === "skipped") data = data.filter(h => isSkipStatus(h.status));
  if (!data.length) { body.innerHTML = ""; empty.style.display = ""; return; }
  empty.style.display = "none";
  body.innerHTML = data.slice(0,100).map(h => {
    const pnl = h.pnl||0, isSkip = isSkipStatus(h.status);
    const pickTeam = h.chosen_team || h.team || "";
    let result;
    if (h.status==="no_market") result='<span class="badge badge-nomarket">No Market</span>';
    else if (isSkip) result=`<span class="badge badge-skipped" title="${h.status||""}">Skipped</span>`;
    else if (h.status==="stop_loss") result='<span class="badge badge-stopped">Stopped</span>';
    else if (pnl>0) result='<span class="badge badge-win">Win</span>';
    else result='<span class="badge badge-loss">Loss</span>';
    const modelStr = h.model_prob ? `${(h.model_prob*100).toFixed(0)}% ${pickTeam}` : "—";
    const mktStr = h.market_price?`${(h.market_price*100).toFixed(0)}¢`:"—";
    const edgeStr = h.edge?fmtPct(h.edge*100):"—";
    const pnlStr = isSkip?"—":fmtPnl(pnl);
    return `<tr class="${isSkip?"row-skipped":""}"><td>${h.match_date||""}</td><td>${h.team1||"?"} vs ${h.team2||"?"}</td><td>${modelStr}</td><td>${mktStr}</td><td>${edgeStr}</td><td>${result}</td><td class="${isSkip?"":pnlClass(pnl)}">${pnlStr}</td></tr>`;
  }).join("");
}
async function refreshHistory() { _historyData = await fetchJSON("/api/history"); renderHistory(); }

async function refreshLog() {
  const data = await fetchJSON("/api/events");
  const list = $("log-list");
  if (!data.length) { list.innerHTML = '<div class="empty-state">No events yet</div>'; return; }
  list.innerHTML = data.slice(0,100).map(e =>
    `<div class="log-entry"><span class="log-ts">${shortTs(e.ts)}</span><span class="log-type ${e.type}">${e.type}</span><span class="log-msg">${e.message}</span></div>`
  ).join("");
}

// ---------------------------------------------------------------------------
// Win Probability Chart — Kalshi-style with %, dynamic range, endpoint labels
// ---------------------------------------------------------------------------
const TEAM_ABBREVS = {
  "Mumbai Indians": "MI", "Chennai Super Kings": "CSK",
  "Royal Challengers Bengaluru": "RCB", "Kolkata Knight Riders": "KKR",
  "Rajasthan Royals": "RR", "Sunrisers Hyderabad": "SRH",
  "Delhi Capitals": "DC", "Punjab Kings": "PBKS",
  "Lucknow Super Giants": "LSG", "Gujarat Titans": "GT",
};
const T1_COLOR = "#4ade80";
const T2_COLOR = "#60a5fa";

const _chanceCharts = {};
let _lastChartRefresh = 0;

async function _loadChartData(ticker) {
  try {
    const trades = await fetchJSON(`/api/trades/${ticker}?limit=1000`);
    return trades.map(t => ({ x: new Date(t.ts).getTime(), y: t.price * 100 }));
  } catch { return []; }
}

function _buildContinuousLines(t1Raw, t2Raw) {
  const events = [];
  for (const p of t1Raw) events.push({ ts: p.x, t1: p.y });
  for (const p of t2Raw) events.push({ ts: p.x, t2: p.y });
  events.sort((a, b) => a.ts - b.ts);

  const t1Out = [], t2Out = [];
  let lastT1 = null, lastT2 = null;

  for (const e of events) {
    if (e.t1 !== undefined) { lastT1 = e.t1; if (lastT2 === null) lastT2 = 100 - e.t1; }
    if (e.t2 !== undefined) { lastT2 = e.t2; if (lastT1 === null) lastT1 = 100 - e.t2; }
    const t1v = lastT1 !== null ? lastT1 : (lastT2 !== null ? 100 - lastT2 : 50);
    const t2v = lastT2 !== null ? lastT2 : (lastT1 !== null ? 100 - lastT1 : 50);
    t1Out.push({ x: e.ts, y: t1v });
    t2Out.push({ x: e.ts, y: t2v });
  }
  return [t1Out, t2Out];
}

function _dynamicRange(t1, t2) {
  let lo = 100, hi = 0;
  for (const p of t1) { if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y; }
  for (const p of t2) { if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y; }
  const pad = Math.max((hi - lo) * 0.2, 5);
  return { min: Math.max(0, Math.floor((lo - pad) / 5) * 5), max: Math.min(100, Math.ceil((hi + pad) / 5) * 5) };
}

const endLabelPlugin = {
  id: "endLabels",
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    const area = chart.chartArea;
    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      if (!meta.visible || !ds.data.length) return;
      const last = meta.data[meta.data.length - 1];
      if (!last) return;
      const val = ds.data[ds.data.length - 1].y;
      const abbr = ds._abbrev || ds.label;
      const y = last.y;

      ctx.save();

      const pctText = Math.round(val) + "%";
      ctx.font = "bold 16px -apple-system, BlinkMacSystemFont, sans-serif";
      const pctW = ctx.measureText(pctText).width;
      ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
      const abbrW = ctx.measureText(abbr).width;
      const blockW = Math.max(pctW, abbrW);

      const rx = area.right + 8;
      const abbrY = y - 10;
      const pctY = y + 4;

      ctx.fillStyle = ds.borderColor;
      ctx.globalAlpha = 0.8;
      ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textBaseline = "top";
      ctx.fillText(abbr, rx, abbrY);

      ctx.globalAlpha = 1;
      ctx.font = "bold 16px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText(pctText, rx, pctY);

      ctx.restore();
    });
  }
};

const crosshairPlugin = {
  id: "crosshair",
  afterDraw(chart) {
    if (!chart._active || !chart._active.length) return;
    const ctx = chart.ctx;
    const x = chart._active[0].element.x;
    const area = chart.chartArea;
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, area.top);
    ctx.lineTo(x, area.bottom);
    ctx.stroke();
    ctx.restore();
  }
};

function _renderChanceChart(canvasId, t1Label, t2Label, t1Data, t2Data) {
  const [t1, t2] = _buildContinuousLines(t1Data, t2Data);
  const range = _dynamicRange(t1, t2);
  const t1Abbr = TEAM_ABBREVS[t1Label] || t1Label;
  const t2Abbr = TEAM_ABBREVS[t2Label] || t2Label;

  const existing = _chanceCharts[canvasId];
  if (existing) {
    existing.data.datasets[0].data = t1;
    existing.data.datasets[0]._abbrev = t1Abbr;
    existing.data.datasets[1].data = t2;
    existing.data.datasets[1]._abbrev = t2Abbr;
    existing.options.scales.y.min = range.min;
    existing.options.scales.y.max = range.max;
    existing.update("none");
    return;
  }

  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  _chanceCharts[canvasId] = new Chart(canvas, {
    type: "line",
    data: {
      datasets: [
        {
          label: t1Label,
          _abbrev: t1Abbr,
          data: t1,
          borderColor: T1_COLOR,
          backgroundColor: "transparent",
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: T1_COLOR,
          borderWidth: 2,
          tension: 0.3,
          order: 1,
        },
        {
          label: t2Label,
          _abbrev: t2Abbr,
          data: t2,
          borderColor: T2_COLOR,
          backgroundColor: "transparent",
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: T2_COLOR,
          borderWidth: 2,
          tension: 0.3,
          order: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { right: 60 } },
      interaction: { mode: "index", intersect: false },
      onHover: (e, elements, chart) => { chart._active = elements; },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(10,14,20,0.92)",
          borderColor: "#252d38",
          borderWidth: 1,
          titleColor: "#7d8590",
          bodyColor: "#e6edf3",
          bodyFont: { size: 13, weight: "600" },
          padding: { x: 12, y: 8 },
          displayColors: true,
          boxWidth: 8,
          boxHeight: 8,
          usePointStyle: true,
          callbacks: {
            title: ctx => new Date(ctx[0].parsed.x).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
            label: ctx => {
              const a = ctx.dataset._abbrev || ctx.dataset.label;
              return ` ${a}  ${ctx.parsed.y.toFixed(1)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "time",
          time: { tooltipFormat: "HH:mm:ss", displayFormats: { minute: "h:mma", hour: "ha" } },
          ticks: { color: "#484f58", font: { size: 10 }, maxTicksLimit: 6 },
          grid: { color: "rgba(37,45,56,0.3)", drawTicks: false },
          border: { display: false },
        },
        y: {
          min: range.min,
          max: range.max,
          ticks: {
            color: "#484f58",
            font: { size: 10 },
            callback: v => v + "%",
            stepSize: 5,
            padding: 8,
          },
          grid: { color: "rgba(37,45,56,0.3)", drawTicks: false, lineWidth: ctx => ctx.tick.value === 50 ? 1.5 : 0.5 },
          border: { display: false },
        },
      },
      animation: { duration: 0 },
    },
    plugins: [endLabelPlugin, crosshairPlugin],
  });
}

// ---------------------------------------------------------------------------
// Live Matches
// ---------------------------------------------------------------------------
async function refreshLive() {
  const data = await fetchJSON("/api/live");
  const container = $("live-container");
  const dot = $("live-dot");

  if (!data.length) {
    container.innerHTML = '<div class="empty-state">No live matches right now</div>';
    if (dot) dot.style.display = "none";
    return;
  }
  if (dot) dot.style.display = "inline-block";

  const existingCards = container.querySelectorAll(".live-card");
  const needsRebuild =
    existingCards.length !== data.length ||
    Array.from(existingCards).some((c, i) => c.dataset.event !== (data[i].event_ticker || ""));

  if (needsRebuild) {
    for (const id of Object.keys(_chanceCharts)) {
      _chanceCharts[id].destroy();
      delete _chanceCharts[id];
    }
    container.innerHTML = data.map((m, i) => `
      <div class="live-card" data-event="${m.event_ticker||""}" data-idx="${i}">
        <div class="live-card-header">
          <div class="live-status-text"></div>
          <div style="display:flex;align-items:center;gap:10px">
            <span class="live-freshness"></span>
            <div class="live-status"></div>
          </div>
        </div>
        <div class="live-scoreboard">
          <div class="live-team"><div class="live-team-name t1-name"></div><div class="live-team-score t1-score"></div><div class="live-team-overs t1-overs"></div></div>
          <div class="live-vs">VS</div>
          <div class="live-team"><div class="live-team-name t2-name"></div><div class="live-team-score t2-score"></div><div class="live-team-overs t2-overs"></div></div>
        </div>
        <div class="live-odds-row">
          <div class="live-odds-card t1-card"><div class="live-odds-label t1-odds-label"></div><div class="live-odds-value t1-odds-val"></div><div class="live-odds-vol t1-vol-val"></div></div>
          <div class="live-odds-card t2-card"><div class="live-odds-label t2-odds-label"></div><div class="live-odds-value t2-odds-val"></div><div class="live-odds-vol t2-vol-val"></div></div>
        </div>
        <div class="chance-chart-wrap">
          <div class="chance-chart-title">Win Probability</div>
          <canvas id="chance-chart-${i}"></canvas>
        </div>
        <div class="live-bet-area"></div>
      </div>`).join("");
  }

  const now = Date.now();
  const shouldRefreshCharts = now - _lastChartRefresh > CHART_POLL_MS;

  for (let i = 0; i < data.length; i++) {
    const m = data[i];
    const card = container.querySelector(`[data-idx="${i}"]`);
    if (!card) continue;

    card.querySelector(".live-status-text").textContent = m.status_text || "";
    const sAge = m.scores_age_secs !== null ? m.scores_age_secs.toFixed(0) + "s" : "—";
    const oAge = m.odds_age_secs !== null ? m.odds_age_secs.toFixed(0) + "s" : "—";
    card.querySelector(".live-freshness").textContent = `Score ${sAge} · Odds ${oAge}`;

    const stateEl = card.querySelector(".live-status");
    const isLive = m.match_state === "LIVE";
    stateEl.textContent = isLive ? "LIVE" : m.match_state || "PRE";
    stateEl.className = "live-status" + (isLive ? "" : " pre");

    card.querySelector(".t1-name").textContent = m.team1;
    card.querySelector(".t1-score").textContent = m.t1_score || "—";
    card.querySelector(".t1-overs").textContent = m.t1_score_info || "";
    card.querySelector(".t2-name").textContent = m.team2;
    card.querySelector(".t2-score").textContent = m.t2_score || "—";
    card.querySelector(".t2-overs").textContent = m.t2_score_info || "";

    card.querySelector(".t1-odds-label").textContent = m.team1;
    card.querySelector(".t1-odds-val").textContent = m.t1_kalshi_odds !== null ? (m.t1_kalshi_odds * 100).toFixed(0) + "¢" : "—";
    card.querySelector(".t1-vol-val").textContent = m.t1_volume_24h ? "Vol " + fmtVol(m.t1_volume_24h) : "";
    card.querySelector(".t2-odds-label").textContent = m.team2;
    card.querySelector(".t2-odds-val").textContent = m.t2_kalshi_odds !== null ? (m.t2_kalshi_odds * 100).toFixed(0) + "¢" : "—";
    card.querySelector(".t2-vol-val").textContent = m.t2_volume_24h ? "Vol " + fmtVol(m.t2_volume_24h) : "";

    let betHtml = "";
    if (m.active_bet) {
      const b = m.active_bet;
      const pnlVal = b.unrealized_pnl || 0;
      const pnlStr = (pnlVal >= 0 ? "+" : "") + "$" + pnlVal.toFixed(2);
      const pnlCls = pnlVal >= 0 ? "pnl-positive" : "pnl-negative";
      const curStr = b.current_price !== null ? (b.current_price * 100).toFixed(0) + "¢" : "—";
      betHtml = `<div class="live-bet-strip">
        <div class="live-bet-label">Active Bet</div>
        <div class="live-bet-detail"><span>Side:</span> ${b.team} YES</div>
        <div class="live-bet-detail"><span>Entry:</span> ${(b.entry_price*100).toFixed(0)}¢</div>
        <div class="live-bet-detail"><span>Now:</span> ${curStr}</div>
        <div class="live-bet-detail"><span>Qty:</span> ${Math.round(b.contracts)}</div>
        <div class="live-bet-detail"><span>Cost:</span> $${b.bet_amount.toFixed(2)}</div>
        <div class="live-bet-detail ${pnlCls}"><span>P&L:</span> ${pnlStr}</div>
      </div>`;
    }
    card.querySelector(".live-bet-area").innerHTML = betHtml;

    const chartId = "chance-chart-" + i;
    if (m.t1_ticker && m.t2_ticker && typeof Chart !== "undefined" && shouldRefreshCharts) {
      Promise.all([_loadChartData(m.t1_ticker), _loadChartData(m.t2_ticker)]).then(
        ([d1, d2]) => _renderChanceChart(chartId, m.team1, m.team2, d1, d2)
      );
    }
  }

  if (shouldRefreshCharts) _lastChartRefresh = now;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
async function refresh() {
  try {
    await Promise.all([
      refreshStatus(), refreshStats(), refreshLive(),
      refreshUpcoming(), refreshPositions(), refreshHistory(), refreshLog(),
    ]);
  } catch (e) { console.error("Refresh error:", e); }
}

document.querySelectorAll("#history-panel .filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#history-panel .filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    _historyFilter = btn.dataset.filter;
    renderHistory();
  });
});

$("upcoming-body").addEventListener("click", e => {
  const row = e.target.closest("tr.upcoming-row");
  if (!row) return;
  const key = row.dataset.key;
  if (_upcomingExpanded.has(key)) _upcomingExpanded.delete(key);
  else _upcomingExpanded.add(key);
  refreshUpcoming();
});

$("upcoming-expand-all").addEventListener("click", () => {
  document.querySelectorAll("#upcoming-body tr.upcoming-row").forEach(tr => {
    if (tr.dataset.key) _upcomingExpanded.add(tr.dataset.key);
  });
  refreshUpcoming();
});

$("upcoming-collapse-all").addEventListener("click", () => {
  _upcomingExpanded.clear();
  refreshUpcoming();
});

refresh();
setInterval(refresh, POLL_MS);
