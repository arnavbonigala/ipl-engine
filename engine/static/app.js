const POLL_MS = 5000;

function $(id) { return document.getElementById(id); }

function pnlClass(val) {
  return val > 0 ? "pnl-positive" : val < 0 ? "pnl-negative" : "";
}

function fmtPnl(val) {
  const sign = val > 0 ? "+" : "";
  return sign + "$" + val.toFixed(2);
}

function fmtPct(val) {
  const sign = val > 0 ? "+" : "";
  return sign + val.toFixed(1) + "%";
}

function shortTs(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

async function refreshStatus() {
  const data = await fetchJSON("/api/status");
  $("balance").textContent = "$" + data.bankroll.toFixed(2);
  $("wallet").textContent = data.wallet_address || "—";
  const badge = $("engine-badge");
  badge.textContent = data.engine_state;
  badge.className = "engine-badge " + data.engine_state;
}

async function refreshStats() {
  const data = await fetchJSON("/api/stats");
  const retEl = $("return-pct");
  retEl.textContent = fmtPct(data.total_return_pct);
  retEl.className = "stat-value " + (data.total_return_pct >= 0 ? "positive" : "negative");
  $("win-rate").textContent = data.win_rate + "%";
  $("total-bets").textContent = data.total_bets;
  $("total-matches").textContent = data.total_matches + " (" + data.skipped_matches + " skipped)";
  $("max-dd").textContent = data.max_drawdown + "%";
}

async function refreshUpcoming() {
  const data = await fetchJSON("/api/upcoming");
  const body = $("upcoming-body");
  const empty = $("upcoming-empty");

  if (!data.length) {
    body.innerHTML = "";
    empty.style.display = "";
    return;
  }
  empty.style.display = "none";

  body.innerHTML = data.map(m => {
    const pred = m.model_prediction;
    const modelStr = pred
      ? `${(pred.model_prob * 100).toFixed(0)}% ${pred.team}`
      : "—";
    const mktStr = pred ? `${(pred.market_price * 100).toFixed(0)}¢` : "—";
    const edgeStr = pred ? fmtPct(pred.edge * 100) : "—";
    const statusBadge = m.status === "awaiting_toss"
      ? '<span class="badge badge-awaiting">Awaiting toss</span>'
      : pred
        ? '<span class="badge badge-signal">Signal ready</span>'
        : `<span class="badge">${m.status || "—"}</span>`;
    return `<tr>
      <td>${m.match_date || ""}</td>
      <td>${m.team1 || "?"} vs ${m.team2 || "?"}</td>
      <td>${modelStr}</td>
      <td>${mktStr}</td>
      <td>${edgeStr}</td>
      <td>${statusBadge}</td>
    </tr>`;
  }).join("");
}

async function refreshPositions() {
  const data = await fetchJSON("/api/positions");
  const body = $("positions-body");
  const empty = $("positions-empty");

  if (!data.length) {
    body.innerHTML = "";
    empty.style.display = "";
    return;
  }
  empty.style.display = "none";

  body.innerHTML = data.map(p => {
    const cur = p.current_price || p.entry_price;
    const unrealized = (cur - p.entry_price) * (p.contracts || 0);
    const stop = p.entry_price - 0.15;
    return `<tr>
      <td>${p.team1 || "?"} vs ${p.team2 || "?"}</td>
      <td>${p.team} (${p.side})</td>
      <td>${p.entry_price.toFixed(2)}¢</td>
      <td>${cur.toFixed(2)}¢</td>
      <td>${stop.toFixed(2)}¢</td>
      <td class="${pnlClass(unrealized)}">${fmtPnl(unrealized)}</td>
    </tr>`;
  }).join("");
}

let _historyData = [];
let _historyFilter = "all";

function renderHistory() {
  const body = $("history-body");
  const empty = $("history-empty");
  let data = _historyData;

  if (_historyFilter === "bets")
    data = data.filter(h => !["skipped", "no_market"].includes(h.status));
  else if (_historyFilter === "skipped")
    data = data.filter(h => ["skipped", "no_market"].includes(h.status));

  if (!data.length) {
    body.innerHTML = "";
    empty.style.display = "";
    return;
  }
  empty.style.display = "none";

  body.innerHTML = data.slice(0, 100).map(h => {
    const pnl = h.pnl || 0;
    const isSkip = h.status === "skipped" || h.status === "no_market";
    let result;
    if (h.status === "no_market")
      result = '<span class="badge badge-nomarket">No Market</span>';
    else if (h.status === "skipped")
      result = `<span class="badge badge-skipped" title="${h.skip_reason || ""}">Skipped</span>`;
    else if (h.status === "stop_loss")
      result = '<span class="badge badge-stopped">Stopped</span>';
    else if (pnl > 0)
      result = '<span class="badge badge-win">Win</span>';
    else
      result = '<span class="badge badge-loss">Loss</span>';

    const modelStr = isSkip
      ? (h.model_prob ? `${(h.model_prob * 100).toFixed(0)}% ${h.team}` : "—")
      : `${(h.model_prob * 100).toFixed(0)}% ${h.team}`;
    const mktStr = h.market_price ? `${(h.market_price * 100).toFixed(0)}¢` : "—";
    const edgeStr = h.edge ? fmtPct(h.edge * 100) : "—";
    const pnlStr = isSkip ? (h.skip_reason || "—") : fmtPnl(pnl);
    const rowClass = isSkip ? "row-skipped" : "";

    return `<tr class="${rowClass}">
      <td>${h.match_date || ""}</td>
      <td>${h.team1 || "?"} vs ${h.team2 || "?"}</td>
      <td>${modelStr}</td>
      <td>${mktStr}</td>
      <td>${edgeStr}</td>
      <td>${result}</td>
      <td class="${isSkip ? "" : pnlClass(pnl)}">${pnlStr}</td>
    </tr>`;
  }).join("");
}

async function refreshHistory() {
  _historyData = await fetchJSON("/api/history");
  renderHistory();
}

async function refreshLog() {
  const data = await fetchJSON("/api/events");
  const list = $("log-list");
  const empty = $("log-empty");

  if (!data.length) {
    list.innerHTML = '<div class="empty-state">No events yet</div>';
    return;
  }
  if (empty) empty.style.display = "none";

  list.innerHTML = data.slice(0, 100).map(e => {
    return `<div class="log-entry">
      <span class="log-ts">${shortTs(e.ts)}</span>
      <span class="log-type ${e.type}">${e.type}</span>
      <span class="log-msg">${e.message}</span>
    </div>`;
  }).join("");
}

async function refresh() {
  try {
    await Promise.all([
      refreshStatus(),
      refreshStats(),
      refreshUpcoming(),
      refreshPositions(),
      refreshHistory(),
      refreshLog(),
    ]);
  } catch (e) {
    console.error("Refresh error:", e);
  }
}

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    _historyFilter = btn.dataset.filter;
    renderHistory();
  });
});

refresh();
setInterval(refresh, POLL_MS);
