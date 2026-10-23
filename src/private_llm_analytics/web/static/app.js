// Private LLM Analytics Web Console (v0.5)
// Vanilla JavaScript Client - Local Only

let currentConfig = null;
let lastReleaseData = null;

document.addEventListener("DOMContentLoaded", () => {
  initConsole();
});

async function initConsole() {
  await fetchConfig();
  await refreshDatasetStatus();
  await refreshPrivacyStatus();
  await refreshReleaseHistory();
  await refreshContinualStatus();

  setupEventListeners();
}

// -------------------------------------------------------------
// API Calls & Rendering
// -------------------------------------------------------------

async function fetchConfig() {
  try {
    const res = await fetch("/api/config");
    currentConfig = await res.json();
    renderConfigOptions(currentConfig);
  } catch (err) {
    showError("Failed to load configuration from backend: " + err);
  }
}

function renderConfigOptions(cfg) {
  // 1. Providers checkboxes
  const provContainer = document.getElementById("provider-checkboxes");
  provContainer.textContent = "";
  cfg.providers.forEach((p) => {
    const label = document.createElement("label");
    label.className = "checkbox-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "provider-filter";
    input.value = p;
    input.checked = true;
    input.addEventListener("change", applyResultFilter);
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + p));
    provContainer.appendChild(label);
  });

  // 2. Tasks checkboxes
  const taskContainer = document.getElementById("task-checkboxes");
  taskContainer.textContent = "";
  cfg.tasks.forEach((t) => {
    const label = document.createElement("label");
    label.className = "checkbox-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "task-filter";
    input.value = t;
    input.checked = true;
    input.addEventListener("change", applyResultFilter);
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + t));
    taskContainer.appendChild(label);
  });

  // 3. Recipients dropdown
  const recipSelect = document.getElementById("select-recipient");
  recipSelect.textContent = "";
  cfg.recipients.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    recipSelect.appendChild(opt);
  });
  if (cfg.recipients.includes("provider_a")) {
    recipSelect.value = "provider_a";
  }

  // 4. Default Epsilon & Delta
  document.getElementById("input-epsilon").value = cfg.default_epsilon;
  document.getElementById("input-delta").value = cfg.default_delta;

  // 5. Candidate Windows
  const windowSelect = document.getElementById("select-window");
  windowSelect.textContent = "";
  cfg.candidate_windows.forEach((w) => {
    const opt = document.createElement("option");
    opt.value = w;
    opt.textContent = `${w} days`;
    if (w === 30) opt.selected = true;
    windowSelect.appendChild(opt);
  });
  const optAll = document.createElement("option");
  optAll.value = "all";
  optAll.textContent = "All time";
  windowSelect.appendChild(optAll);
}

async function refreshDatasetStatus() {
  try {
    const res = await fetch("/api/dataset/status");
    const data = await res.json();
    const statusEl = document.getElementById("dataset-status-badge");
    const infoEl = document.getElementById("dataset-info");

    if (data.loaded) {
      statusEl.className = "badge badge-success";
      statusEl.textContent = "Loaded";
      infoEl.textContent = `Rows: ${data.rows.toLocaleString()} | Active Users: ${data.active_users.toLocaleString()} | Source: ${data.source}`;
    } else {
      statusEl.className = "badge badge-warning";
      statusEl.textContent = "No Data Loaded";
      infoEl.textContent = "Please click 'Load Demo Data' or upload a telemetry file (CSV/Parquet).";
    }
  } catch (err) {
    showError("Failed to fetch dataset status: " + err);
  }
}

async function refreshPrivacyStatus() {
  try {
    const recipient = document.getElementById("select-recipient").value || "provider_a";
    const res = await fetch(`/api/privacy/status?recipient=${encodeURIComponent(recipient)}`);
    const data = await res.json();

    document.getElementById("budget-eps-used").textContent = data.effective_epsilon.toFixed(4);
    document.getElementById("budget-eps-cap").textContent = data.epsilon_cap.toFixed(4);
    document.getElementById("budget-headroom").textContent = data.remaining_headroom.toFixed(4);
    document.getElementById("budget-delta").textContent = data.delta_global.toExponential(1);
    document.getElementById("budget-accountant-mode").textContent = data.accountant_mode;

    const fillEl = document.getElementById("budget-progress-fill");
    fillEl.style.width = `${Math.min(100, data.spend_percentage)}%`;

    if (data.spend_percentage > 90) {
      fillEl.className = "progress-fill danger";
    } else if (data.spend_percentage > 70) {
      fillEl.className = "progress-fill warning";
    } else {
      fillEl.className = "progress-fill";
    }
  } catch (err) {
    showError("Failed to fetch privacy status: " + err);
  }
}

async function refreshReleaseHistory() {
  try {
    const res = await fetch("/api/releases");
    const items = await res.json();
    const tbody = document.getElementById("history-table-body");
    tbody.textContent = "";

    if (!items || items.length === 0) {
      const emptyRow = document.createElement("tr");
      const emptyCell = document.createElement("td");
      emptyCell.colSpan = 7;
      emptyCell.style.textAlign = "center";
      emptyCell.style.color = "var(--text-light)";
      emptyCell.textContent = "No historical releases recorded yet.";
      emptyRow.appendChild(emptyCell);
      tbody.appendChild(emptyRow);
      return;
    }

    items.forEach((item) => {
      const tr = document.createElement("tr");
      const shortId = item.release_id ? item.release_id.substring(0, 8) + "..." : "-";
      const ts = item.timestamp ? item.timestamp.replace("T", " ").substring(0, 19) : "-";

      const tdId = document.createElement("td");
      tdId.style.fontFamily = "monospace";
      tdId.textContent = shortId;

      const tdTs = document.createElement("td");
      tdTs.textContent = ts;

      const tdMetric = document.createElement("td");
      const strongMetric = document.createElement("strong");
      strongMetric.textContent = item.metric;
      tdMetric.appendChild(strongMetric);

      const tdRecipient = document.createElement("td");
      tdRecipient.textContent = item.recipient;

      const tdBudget = document.createElement("td");
      tdBudget.textContent = `ε = ${item.epsilon}, δ = ${item.delta}`;

      const tdMech = document.createElement("td");
      tdMech.textContent = item.mechanism;

      const tdStatus = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = item.cache_hit ? "badge badge-success" : "badge badge-primary";
      badge.textContent = item.cache_hit ? "Reused (0 Cost)" : "Committed";
      tdStatus.appendChild(badge);

      tr.appendChild(tdId);
      tr.appendChild(tdTs);
      tr.appendChild(tdMetric);
      tr.appendChild(tdRecipient);
      tr.appendChild(tdBudget);
      tr.appendChild(tdMech);
      tr.appendChild(tdStatus);

      tbody.appendChild(tr);
    });
  } catch (err) {
    showError("Failed to fetch release history: " + err);
  }
}

async function refreshContinualStatus() {
  try {
    const res = await fetch("/api/continual/status");
    const data = await res.json();
    const statusBadge = document.getElementById("continual-status-badge");
    const msgEl = document.getElementById("continual-msg");
    const queryBtn = document.getElementById("btn-continual-query");

    if (document.getElementById("cont-metric")) {
      document.getElementById("cont-metric").textContent = data.supported_metric === "interaction_volume" ? "Interaction Volume" : data.supported_metric;
      document.getElementById("cont-epoch-days").textContent = data.epoch_days ?? 90;
      document.getElementById("cont-bounds").textContent = `${data.daily_l2_bound ?? 10.0} / ${data.epoch_group_l12_bound ?? 40.0}`;
      document.getElementById("cont-topo").textContent = data.tree_topology ?? "binary";
    }

    if (data.enabled) {
      statusBadge.className = "badge badge-success";
      statusBadge.textContent = "Enabled";
      msgEl.textContent = data.message || `Mode: ${data.mode} | Epoch Horizon: ${data.epoch_days} days.`;
      queryBtn.disabled = false;
    } else {
      statusBadge.className = "badge badge-warning";
      statusBadge.textContent = "Disabled (Default)";
      msgEl.textContent = data.message || "Continual observation requires trusted configuration and an authoritative finalized stream state.";
      queryBtn.disabled = true;
    }
  } catch (err) {
    showError("Failed to fetch continual status: " + err);
  }
}

// -------------------------------------------------------------
// User Action Handlers
// -------------------------------------------------------------

function setupEventListeners() {
  // Demo Data
  document.getElementById("btn-load-demo").addEventListener("click", async () => {
    clearAlerts();
    try {
      const res = await fetch("/api/dataset/demo", { method: "POST" });
      if (!res.ok) throw await res.json();
      await refreshDatasetStatus();
      showSuccess("Demo synthetic telemetry dataset generated successfully.");
    } catch (err) {
      showError(err.error || "Failed to load demo data.");
    }
  });

  // Upload Data
  document.getElementById("btn-upload").addEventListener("click", () => {
    document.getElementById("file-input").click();
  });

  document.getElementById("file-input").addEventListener("change", async (e) => {
    clearAlerts();
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/dataset/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw await res.json();
      await refreshDatasetStatus();
      showSuccess(`Dataset '${file.name}' validated and loaded successfully.`);
    } catch (err) {
      showError(err.error || "Dataset upload failed.");
    }
  });

  // Recipient change
  document.getElementById("select-recipient").addEventListener("change", () => {
    refreshPrivacyStatus();
  });

  // Planner Recommendation
  document.getElementById("btn-get-recommendation").addEventListener("click", async () => {
    clearAlerts();
    const metric = getSelectedMetric();
    const recipient = document.getElementById("select-recipient").value;

    try {
      const res = await fetch("/api/planner/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          metric: metric,
          recipient: recipient,
          planning_horizon_days: 90,
        }),
      });

      if (!res.ok) throw await res.json();
      const plan = await res.json();
      renderPlannerRecommendation(plan);
    } catch (err) {
      showError(err.error || "Failed to get planner recommendation.");
    }
  });

  // Confirmation Modal
  document.getElementById("btn-open-confirm").addEventListener("click", () => {
    clearAlerts();
    const metric = getSelectedMetric();
    const recipient = document.getElementById("select-recipient").value;
    const eps = parseFloat(document.getElementById("input-epsilon").value);
    const delta = parseFloat(document.getElementById("input-delta").value);
    const windowVal = document.getElementById("select-window").value;

    if (isNaN(eps) || eps <= 0) {
      showError("Epsilon must be a positive number.");
      return;
    }
    if (isNaN(delta) || delta <= 0 || delta >= 1) {
      showError("Delta must be strictly between 0 and 1.");
      return;
    }

    document.getElementById("confirm-metric").textContent = metric;
    document.getElementById("confirm-recipient").textContent = recipient;
    document.getElementById("confirm-eps").textContent = eps;
    document.getElementById("confirm-delta").textContent = delta;
    document.getElementById("confirm-window").textContent = windowVal === "all" ? "All time" : `${windowVal} days`;

    document.getElementById("confirm-modal").classList.add("active");
  });

  document.getElementById("btn-cancel-release").addEventListener("click", () => {
    document.getElementById("confirm-modal").classList.remove("active");
  });

  // Execute Release
  document.getElementById("btn-confirm-release").addEventListener("click", async () => {
    document.getElementById("confirm-modal").classList.remove("active");
    clearAlerts();

    const metric = getSelectedMetric();
    const recipient = document.getElementById("select-recipient").value;
    const eps = parseFloat(document.getElementById("input-epsilon").value);
    const delta = parseFloat(document.getElementById("input-delta").value);
    const windowVal = document.getElementById("select-window").value;

    const payload = {
      metric: metric,
      recipient: recipient,
      epsilon: eps,
      delta: delta,
      window_days: windowVal === "all" ? null : parseInt(windowVal, 10),
    };

    try {
      const res = await fetch("/api/release", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw await res.json();
      const data = await res.json();
      lastReleaseData = data;
      renderReleaseResult(data);
      await refreshPrivacyStatus();
      await refreshReleaseHistory();
      showSuccess(
        data.cache_hit
          ? "Existing canonical release reused. No additional privacy spend."
          : `DP release executed and committed to persistent ledger. Incremental ε cost: ${data.incremental_privacy_cost}`
      );
    } catch (err) {
      showError(err.error || "Failed to execute differential privacy release.");
    }
  });

  // Continual Query
  document.getElementById("btn-continual-query").addEventListener("click", async () => {
    clearAlerts();
    const startDay = parseInt(document.getElementById("continual-start-day").value, 10);
    const endDay = parseInt(document.getElementById("continual-end-day").value, 10);
    const recipient = document.getElementById("select-recipient").value;

    try {
      const res = await fetch("/api/continual/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_day: startDay,
          end_day: endDay,
          metric: "interaction_volume",
          recipient: recipient,
        }),
      });

      if (!res.ok) throw await res.json();
      const data = await res.json();
      renderContinualResult(data);
      await refreshPrivacyStatus();
      showSuccess(`Continual interval query [${startDay}, ${endDay}] executed.`);
    } catch (err) {
      showError(err.error || "Continual query failed.");
    }
  });
}

function getSelectedMetric() {
  const radios = document.getElementsByName("metric-choice");
  for (const r of radios) {
    if (r.checked) return r.value;
  }
  return "interaction_volume";
}

// -------------------------------------------------------------
// Rendering Helpers
// -------------------------------------------------------------

function renderPlannerRecommendation(plan) {
  const panel = document.getElementById("planner-result-panel");
  panel.style.display = "block";

  document.getElementById("plan-clipping").textContent = plan.recommended_clipping ?? "N/A";
  document.getElementById("plan-granularity").textContent = plan.recommended_granularity;
  document.getElementById("plan-window").textContent = `${plan.recommended_window_days} days`;
  document.getElementById("plan-freq").textContent = `Every ${plan.recommended_frequency_days} days`;

  document.getElementById("plan-clip-loss").textContent = plan.estimated_clipping_loss ?? "N/A";
  document.getElementById("plan-noise-loss").textContent = plan.estimated_dp_noise ?? "N/A";
  document.getElementById("plan-total-loss").textContent = plan.estimated_utility ?? "N/A";
  document.getElementById("plan-feasible").textContent = plan.budget_feasible ? "Feasible" : "Infeasible";
  document.getElementById("plan-feasible").className = plan.budget_feasible ? "badge badge-success" : "badge badge-warning";
  document.getElementById("plan-reason").textContent = plan.decision_reason;
}

function renderReleaseResult(data) {
  const panel = document.getElementById("release-result-panel");
  panel.style.display = "block";

  document.getElementById("res-metric").textContent = data.metric;
  document.getElementById("res-recipient").textContent = data.recipient;
  document.getElementById("res-eps-delta").textContent = `ε = ${data.epsilon}, δ = ${data.delta}`;
  document.getElementById("res-noise-scale").textContent = data.noise_scale;

  const cacheStatusEl = document.getElementById("res-cache-status");
  cacheStatusEl.textContent = data.cache_hit ? "Reused (0 Cost)" : "New Release Committed";
  cacheStatusEl.className = data.cache_hit ? "badge badge-success" : "badge badge-primary";

  document.getElementById("res-release-id").textContent = data.release_id || data.release_key;

  applyResultFilter();
}

function applyResultFilter() {
  if (!lastReleaseData || !lastReleaseData.results) return;

  const tbody = document.getElementById("results-table-body");
  tbody.textContent = "";

  const results = lastReleaseData.results;
  let cellEntries = [];

  if (results.cell_volumes) {
    cellEntries = Object.entries(results.cell_volumes).map(([k, v]) => ({
      cell: k,
      value: Number(v).toFixed(2),
      unit: "interactions",
    }));
  } else if (results.cell_shares) {
    cellEntries = Object.entries(results.cell_shares).map(([k, v]) => ({
      cell: k,
      value: (Number(v) * 100).toFixed(2) + "%",
      unit: "workflow share",
    }));
  } else if (results.prevalence_rates) {
    cellEntries = Object.entries(results.prevalence_rates).map(([k, v]) => ({
      cell: k,
      value: (Number(v) * 100).toFixed(2) + "%",
      unit: "prevalence",
    }));
  }

  // Filter by selected display providers and tasks
  const selectedProviders = Array.from(
    document.querySelectorAll("input[name='provider-filter']:checked")
  ).map((el) => el.value);
  const selectedTasks = Array.from(
    document.querySelectorAll("input[name='task-filter']:checked")
  ).map((el) => el.value);

  const filtered = cellEntries.filter((item) => {
    const parts = item.cell.split(":");
    if (parts.length === 2) {
      const [prov, tsk] = parts;
      return selectedProviders.includes(prov) && selectedTasks.includes(tsk);
    }
    return true;
  });

  if (filtered.length === 0) {
    const emptyRow = document.createElement("tr");
    const emptyCell = document.createElement("td");
    emptyCell.colSpan = 3;
    emptyCell.style.textAlign = "center";
    emptyCell.style.color = "var(--text-light)";
    emptyCell.textContent = "No cells match the current filter selection.";
    emptyRow.appendChild(emptyCell);
    tbody.appendChild(emptyRow);
    return;
  }

  filtered.forEach((item) => {
    const tr = document.createElement("tr");

    const tdCell = document.createElement("td");
    const strong = document.createElement("strong");
    strong.textContent = item.cell;
    tdCell.appendChild(strong);

    const tdVal = document.createElement("td");
    tdVal.style.fontFamily = "monospace";
    tdVal.style.fontSize = "14px";
    tdVal.textContent = item.value;

    const tdUnit = document.createElement("td");
    tdUnit.style.color = "var(--text-light)";
    tdUnit.textContent = item.unit;

    tr.appendChild(tdCell);
    tr.appendChild(tdVal);
    tr.appendChild(tdUnit);
    tbody.appendChild(tr);
  });
}

function renderContinualResult(data) {
  const panel = document.getElementById("continual-result-panel");
  panel.style.display = "block";

  document.getElementById("cont-query-id").textContent = data.query_id;
  document.getElementById("cont-nodes-used").textContent = data.nodes_used.map((n) => `[${n[0]}, ${n[1]}]`).join(", ");
  document.getElementById("cont-levels").textContent = data.levels_used.join(", ");
  document.getElementById("cont-reused").textContent = data.reused_levels.length > 0 ? data.reused_levels.join(", ") : "None";
  document.getElementById("cont-new-exposed").textContent = data.newly_exposed_levels.length > 0 ? data.newly_exposed_levels.join(", ") : "None";

  const tbody = document.getElementById("continual-table-body");
  tbody.textContent = "";
  data.cell_names.forEach((name, i) => {
    const tr = document.createElement("tr");
    const val = data.vector[i] !== undefined ? data.vector[i].toFixed(2) : "0.00";

    const tdName = document.createElement("td");
    const strongName = document.createElement("strong");
    strongName.textContent = name;
    tdName.appendChild(strongName);

    const tdVal = document.createElement("td");
    tdVal.style.fontFamily = "monospace";
    tdVal.textContent = val;

    tr.appendChild(tdName);
    tr.appendChild(tdVal);
    tbody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// Alerts
// -------------------------------------------------------------

function clearAlerts() {
  const container = document.getElementById("alert-container");
  container.textContent = "";
}

function showError(msg) {
  const container = document.getElementById("alert-container");
  container.textContent = "";
  const div = document.createElement("div");
  div.className = "alert alert-danger";
  div.textContent = typeof msg === "string" ? msg : JSON.stringify(msg);
  container.appendChild(div);
}

function showSuccess(msg) {
  const container = document.getElementById("alert-container");
  container.textContent = "";
  const div = document.createElement("div");
  div.className = "alert alert-success";
  div.textContent = typeof msg === "string" ? msg : JSON.stringify(msg);
  container.appendChild(div);
}
