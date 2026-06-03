const state = {
  sessionId: crypto.randomUUID(),
  metrics: null,
};

const els = {
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  tabs: document.querySelectorAll(".tab"),
  views: document.querySelectorAll(".view"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  transcript: document.querySelector("#transcript"),
  sessionLabel: document.querySelector("#sessionLabel"),
  newSessionButton: document.querySelector("#newSessionButton"),
  turnCount: document.querySelector("#turnCount"),
  toolCallCount: document.querySelector("#toolCallCount"),
  latencyValue: document.querySelector("#latencyValue"),
  sessionCount: document.querySelector("#sessionCount"),
  toolStrip: document.querySelector("#toolStrip"),
  eventList: document.querySelector("#eventList"),
  refreshMetricsButton: document.querySelector("#refreshMetricsButton"),
  langfuseStatus: document.querySelector("#langfuseStatus"),
  langfuseHost: document.querySelector("#langfuseHost"),
  mcpServers: document.querySelector("#mcpServers"),
  toolPrefixing: document.querySelector("#toolPrefixing"),
  errorList: document.querySelector("#errorList"),
  runEvalButton: document.querySelector("#runEvalButton"),
  scoreRing: document.querySelector("#scoreRing"),
  overallScore: document.querySelector("#overallScore"),
  toolCoverage: document.querySelector("#toolCoverage"),
  answerQuality: document.querySelector("#answerQuality"),
  avoidanceScore: document.querySelector("#avoidanceScore"),
  scenarioList: document.querySelector("#scenarioList"),
};

function init() {
  els.sessionLabel.textContent = `Session ${state.sessionId.slice(0, 8)}`;
  bindEvents();
  checkHealth();
  refreshMetrics();
  addMessage("assistant", "Ready. Try a multi-turn task that should use one of your MCP tools.");
}

function bindEvents() {
  els.tabs.forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });

  els.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = els.messageInput.value.trim();
    if (!message) return;
    els.messageInput.value = "";
    addMessage("user", message);
    await sendChat(message);
  });

  els.newSessionButton.addEventListener("click", () => {
    state.sessionId = crypto.randomUUID();
    els.sessionLabel.textContent = `Session ${state.sessionId.slice(0, 8)}`;
    els.transcript.innerHTML = "";
    addMessage("assistant", "New session started.");
  });

  els.refreshMetricsButton.addEventListener("click", refreshMetrics);
  els.runEvalButton.addEventListener("click", runEvaluation);
}

function switchView(viewName) {
  els.tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === viewName));
  els.views.forEach((view) => view.classList.toggle("is-active", view.id === `${viewName}View`));
  refreshMetrics();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Server not healthy");
    els.statusDot.classList.add("is-ok");
    els.statusText.textContent = "Server connected";
  } catch {
    els.statusDot.classList.remove("is-ok");
    els.statusText.textContent = "Server unavailable";
  }
}

async function sendChat(message) {
  const button = els.chatForm.querySelector("button");
  button.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: state.sessionId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Chat failed");
    addMessage("assistant", data.content, data);
    await refreshMetrics();
  } catch (error) {
    addMessage("error", error.message);
    await refreshMetrics();
  } finally {
    button.disabled = false;
    els.messageInput.focus();
  }
}

async function refreshMetrics() {
  const response = await fetch("/api/metrics");
  const metrics = await response.json();
  state.metrics = metrics;
  renderMetrics(metrics);
}

async function runEvaluation() {
  els.runEvalButton.disabled = true;
  els.runEvalButton.textContent = "Running";
  try {
    const response = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenarios_path: "evals/scenarios/tool_multiturn.json" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Evaluation failed");
    renderEvaluation(data);
    await refreshMetrics();
  } catch (error) {
    renderEvalError(error.message);
    await refreshMetrics();
  } finally {
    els.runEvalButton.disabled = false;
    els.runEvalButton.textContent = "Run eval";
  }
}

function addMessage(role, text, meta = null) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = `<div>${escapeHtml(text).replaceAll("\n", "<br>")}</div>`;
  if (meta?.tool_calls?.length) {
    const names = meta.tool_calls.map((tool) => tool.name).join(", ");
    node.innerHTML += `<small>Tools: ${escapeHtml(names)} · ${meta.latency_ms} ms</small>`;
  } else if (meta?.latency_ms) {
    node.innerHTML += `<small>${meta.latency_ms} ms</small>`;
  }
  els.transcript.appendChild(node);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function renderMetrics(metrics) {
  const chat = metrics.chat;
  els.turnCount.textContent = chat.turns;
  els.toolCallCount.textContent = chat.tool_calls;
  els.latencyValue.textContent = `${chat.avg_latency_ms} ms`;
  els.sessionCount.textContent = chat.sessions;

  els.toolStrip.innerHTML = "";
  const toolEntries = Object.entries(chat.tool_counts);
  if (toolEntries.length === 0) {
    els.toolStrip.innerHTML = `<span class="pill">No tools called yet</span>`;
  } else {
    toolEntries.forEach(([name, count]) => {
      els.toolStrip.innerHTML += `<span class="pill">${escapeHtml(name)} · ${count}</span>`;
    });
  }

  els.eventList.innerHTML = chat.recent_events.length
    ? chat.recent_events.map(renderEvent).join("")
    : `<div class="event">No chat events yet.</div>`;

  const obs = metrics.observability;
  els.langfuseStatus.textContent = obs.langfuse_configured ? "Configured" : "Not configured";
  els.langfuseHost.innerHTML = obs.langfuse_host
    ? `<a href="${escapeAttr(obs.langfuse_host)}" target="_blank" rel="noreferrer">${escapeHtml(obs.langfuse_host)}</a>`
    : "-";
  els.mcpServers.textContent = obs.mcp_servers.length ? obs.mcp_servers.join(", ") : "None enabled";
  els.toolPrefixing.textContent = obs.mcp_tool_name_prefix ? "Enabled" : "Disabled";

  els.errorList.innerHTML = metrics.errors.length
    ? metrics.errors.map((error) => `<div class="error-item"><strong>${escapeHtml(error.where)}</strong>${escapeHtml(error.message)}</div>`).join("")
    : `<div class="event">No recent errors.</div>`;

  if (metrics.evaluation) renderEvaluation(metrics.evaluation);
}

function renderEvent(event) {
  const tools = event.tool_names.length ? event.tool_names.join(", ") : "No tools";
  return `<div class="event">
    <strong>${escapeHtml(event.session_id.slice(0, 8))} · ${event.latency_ms} ms</strong>
    <span>${escapeHtml(tools)}</span>
  </div>`;
}

function renderEvaluation(report) {
  setScore(els.overallScore, report.overall_score);
  setScore(els.toolCoverage, report.required_tool_coverage);
  setScore(els.answerQuality, report.answer_quality);
  setScore(els.avoidanceScore, report.forbidden_tool_avoidance);
  const degrees = Math.round((report.overall_score || 0) * 360);
  els.scoreRing.style.background = `conic-gradient(var(--accent) ${degrees}deg, #d9e1df ${degrees}deg)`;
  els.scenarioList.innerHTML = report.scenarios?.length
    ? report.scenarios.map(renderScenario).join("")
    : `<div class="scenario">No scenario details available.</div>`;
}

function renderScenario(scenario) {
  return `<article class="scenario">
    <strong>${escapeHtml(scenario.id)} · ${formatPercent(scenario.score)}</strong>
    <div>${escapeHtml(scenario.description)}</div>
    <div class="pill">tool coverage ${formatPercent(scenario.required_tool_coverage)}</div>
  </article>`;
}

function renderEvalError(message) {
  els.scenarioList.innerHTML = `<div class="error-item"><strong>Evaluation failed</strong>${escapeHtml(message)}</div>`;
}

function setScore(element, value) {
  element.textContent = value === undefined || value === null ? "-" : formatPercent(value);
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

init();
