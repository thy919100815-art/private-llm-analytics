// Private LLM Analytics Web Console (v0.5)
// Vanilla JavaScript Client - Local Only (Simplified Chinese Localization)

let currentConfig = null;
let lastReleaseData = null;

const METRIC_NAMES = {
  interaction_volume: "交互量",
  user_prevalence: "用户使用率",
  user_normalized_share: "用户归一化占比",
  success_rate: "成功率",
};

const TASK_NAMES = {
  technical: "技术类",
  document: "文档类",
  analysis: "分析类",
};

const UNIT_NAMES = {
  interactions: "次交互",
  "workflow share": "工作流占比",
  prevalence: "用户使用率",
  success_rate: "任务成功率",
};

const GRANULARITY_NAMES = {
  daily: "按日（Daily）",
  weekly: "按周（Weekly）",
  biweekly: "双周（Bi-weekly）",
  monthly: "按月（Monthly）",
};

const SOURCE_NAMES = {
  demo: "演示数据",
  uploaded: "本地上传",
};

const PROFILE_DETAILS = {
  interaction_volume: {
    scheme: "用户级 L2 裁剪 + 高斯向量机制",
    privacyUnit: "用户",
    adjacency: "增加/删除一个用户",
    contribution: "用户交互向量 L2 裁剪",
    sensitivity: "L2 敏感度 = C",
    mechanism: "高斯向量机制",
    accounting: "RDP",
    continual: "支持二叉树结构",
  },
  user_prevalence: {
    scheme: "用户参与二值化 + 高斯直方图机制",
    privacyUnit: "用户",
    adjacency: "增加/删除一个用户",
    contribution: "用户参与状态二值化（k=9）",
    sensitivity: "L2 敏感度 = 3.0",
    mechanism: "高斯直方图机制",
    accounting: "RDP",
    continual: "暂不支持",
  },
  user_normalized_share: {
    scheme: "用户归一化贡献 + 高斯向量机制 + 单纯形归一化",
    privacyUnit: "用户",
    adjacency: "增加/删除一个用户",
    contribution: "用户交互向量 L1 归一化（范数上限 1.0）",
    sensitivity: "L2 敏感度 = 1.0",
    mechanism: "高斯向量机制",
    accounting: "RDP",
    continual: "暂不支持",
  },
  success_rate: {
    scheme: "共享用户级裁剪 + 成功数/总数联合高斯机制 + 比率后处理",
    privacyUnit: "用户",
    adjacency: "增加/删除一个用户",
    contribution: "用户交互向量 L2 裁剪（共享缩放因子 a_u）",
    sensitivity: "联合 L2 敏感度 = sqrt(2) × C_rate",
    mechanism: "联合高斯机制（18维联合向量单次发布）",
    accounting: "RDP（单次 RDP 计费）",
    continual: "暂不支持",
  },
};

function updateProfileSchemePanel(metric) {
  let d = PROFILE_DETAILS[metric] || PROFILE_DETAILS.interaction_volume;
  if (currentConfig && currentConfig.statistics) {
    const backendStat = currentConfig.statistics.find(s => s.id === metric || s.statistic_id === metric);
    if (backendStat) {
      d = {
        scheme: backendStat.profile_name_zh || d.scheme,
        privacyUnit: backendStat.privacy_unit === "user" ? "用户" : backendStat.privacy_unit,
        adjacency: backendStat.adjacency === "add_remove_user" ? "增加/删除一个用户" : backendStat.adjacency,
        contribution: backendStat.contribution_definition || d.contribution,
        sensitivity: backendStat.sensitivity_description || d.sensitivity,
        mechanism: backendStat.profile_name_zh || d.mechanism,
        accounting: (backendStat.accounting_family || "").toUpperCase() || d.accounting,
        continual: backendStat.supports_continual ? "支持二叉树结构" : "暂不支持",
      };
    }
  }
  const nameEl = document.getElementById("profile-display-name");
  if (nameEl) nameEl.textContent = d.scheme;
  const unitEl = document.getElementById("detail-privacy-unit");
  if (unitEl) unitEl.textContent = d.privacyUnit;
  const adjEl = document.getElementById("detail-adjacency");
  if (adjEl) adjEl.textContent = d.adjacency;
  const contEl = document.getElementById("detail-contribution");
  if (contEl) contEl.textContent = d.contribution;
  const sensEl = document.getElementById("detail-sensitivity");
  if (sensEl) sensEl.textContent = d.sensitivity;
  const mechEl = document.getElementById("detail-mechanism");
  if (mechEl) mechEl.textContent = d.mechanism;
  const accEl = document.getElementById("detail-accounting");
  if (accEl) accEl.textContent = d.accounting;
  const contnEl = document.getElementById("detail-continual");
  if (contnEl) contnEl.textContent = d.continual;
}

document.addEventListener("DOMContentLoaded", () => {
  initConsole();
});

async function initConsole() {
  await fetchConfig();
  await refreshDatasetStatus();
  await refreshPrivacyStatus();
  await refreshReleaseHistory();
  await refreshContinualStatus();

  updateProfileSchemePanel(getSelectedMetric());
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
    showError("无法从后端加载配置：" + (err.error || err.detail || err));
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
    const displayTask = TASK_NAMES[t] || t;
    label.appendChild(document.createTextNode(" " + displayTask));
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
    opt.textContent = `最近 ${w} 天`;
    if (w === 30) opt.selected = true;
    windowSelect.appendChild(opt);
  });
  const optAll = document.createElement("option");
  optAll.value = "all";
  optAll.textContent = "全部时间";
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
      statusEl.textContent = "已加载";
      const displaySource = SOURCE_NAMES[data.source] || data.source;
      infoEl.textContent = `数据行数: ${data.rows.toLocaleString()} | 活跃用户数: ${data.active_users.toLocaleString()} | 数据来源: ${displaySource}`;
    } else {
      statusEl.className = "badge badge-warning";
      statusEl.textContent = "未加载数据";
      infoEl.textContent = "请加载演示数据，或上传本地遥测数据文件（CSV / Parquet）。";
    }
  } catch (err) {
    showError("无法获取数据集状态：" + (err.error || err.detail || err));
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
    document.getElementById("budget-accountant-mode").textContent = "RDP → (ε, δ)";

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
    showError("无法获取隐私预算状态：" + (err.error || err.detail || err));
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
      emptyCell.textContent = "暂无历史发布记录。";
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
      strongMetric.textContent = METRIC_NAMES[item.metric] || item.metric;
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
      badge.textContent = item.cache_hit ? "已复用（无额外消耗）" : "已提交";
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
    showError("无法获取历史发布记录：" + (err.error || err.detail || err));
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
      document.getElementById("cont-metric").textContent =
        data.supported_metric === "interaction_volume" ? "交互量" : (METRIC_NAMES[data.supported_metric] || data.supported_metric);
      document.getElementById("cont-epoch-days").textContent = data.epoch_days ?? 90;
      document.getElementById("cont-bounds").textContent = `${data.daily_l2_bound ?? 10.0} / ${data.epoch_group_l12_bound ?? 40.0}`;
      document.getElementById("cont-topo").textContent =
        data.tree_topology === "binary" ? "二叉树" : (data.tree_topology ?? "二叉树");
    }

    if (data.enabled) {
      statusBadge.className = "badge badge-success";
      statusBadge.textContent = "已启用";
      msgEl.textContent = data.message || `模式: ${data.mode} | Epoch 时间跨度: ${data.epoch_days} 天。`;
      queryBtn.disabled = false;
    } else {
      statusBadge.className = "badge badge-warning";
      statusBadge.textContent = "未启用（默认）";
      msgEl.textContent = "持续观测功能需要可信配置，以及已经正式确认的权威数据流状态。";
      queryBtn.disabled = true;
    }
  } catch (err) {
    showError("无法获取持续观测状态：" + (err.error || err.detail || err));
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
      showSuccess("演示遥测数据已成功加载。");
    } catch (err) {
      showError(translateError(err) || "演示数据加载失败。");
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
      showSuccess(`数据集“${file.name}”校验通过并已成功加载。`);
    } catch (err) {
      showError(translateError(err) || "数据集上传失败。");
    }
  });

  // Recipient change
  document.getElementById("select-recipient").addEventListener("change", () => {
    refreshPrivacyStatus();
  });

  // Metric selection change -> update read-only DP Profile panel
  document.querySelectorAll("input[name='metric-choice']").forEach((radio) => {
    radio.addEventListener("change", () => {
      updateProfileSchemePanel(getSelectedMetric());
    });
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
      showError(translateError(err) || "获取效用规划器建议失败。");
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
      showError("隐私参数 ε 必须为正数。");
      return;
    }
    if (isNaN(delta) || delta <= 0 || delta >= 1) {
      showError("隐私参数 δ 必须严格介于 0 和 1 之间。");
      return;
    }

    document.getElementById("confirm-metric").textContent = METRIC_NAMES[metric] || metric;
    document.getElementById("confirm-recipient").textContent = recipient;
    document.getElementById("confirm-eps").textContent = eps;
    document.getElementById("confirm-delta").textContent = delta;
    document.getElementById("confirm-window").textContent = windowVal === "all" ? "全部时间" : `最近 ${windowVal} 天`;

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
          ? "已复用现有标准发布结果，本次没有产生额外隐私消耗。"
          : `差分隐私结果已生成，并写入持久化隐私记账记录。增量 ε 消耗: ${data.incremental_privacy_cost}`
      );
    } catch (err) {
      showError(translateError(err) || "差分隐私发布失败。");
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
      showSuccess(`持续观测区间查询 [${startDay}, ${endDay}] 已完成。`);
    } catch (err) {
      showError(translateError(err) || "持续观测区间查询失败。");
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

  document.getElementById("plan-clipping").textContent = plan.recommended_clipping ?? "无";
  document.getElementById("plan-granularity").textContent =
    GRANULARITY_NAMES[plan.recommended_granularity] || plan.recommended_granularity;
  document.getElementById("plan-window").textContent = `${plan.recommended_window_days} 天`;
  document.getElementById("plan-freq").textContent = `每 ${plan.recommended_frequency_days} 天`;

  document.getElementById("plan-clip-loss").textContent = plan.estimated_clipping_loss ?? "无";
  document.getElementById("plan-noise-loss").textContent = plan.estimated_dp_noise ?? "无";
  document.getElementById("plan-total-loss").textContent = plan.estimated_utility ?? "无";
  document.getElementById("plan-feasible").textContent = plan.budget_feasible ? "预算可行" : "预算不可行";
  document.getElementById("plan-feasible").className = plan.budget_feasible ? "badge badge-success" : "badge badge-warning";
  document.getElementById("plan-reason").textContent = plan.decision_reason;
}

function renderReleaseResult(data) {
  const panel = document.getElementById("release-result-panel");
  panel.style.display = "block";

  document.getElementById("res-metric").textContent = METRIC_NAMES[data.metric] || data.metric;
  document.getElementById("res-recipient").textContent = data.recipient;
  document.getElementById("res-eps-delta").textContent = `ε = ${data.epsilon}, δ = ${data.delta}`;
  document.getElementById("res-noise-scale").textContent = data.noise_scale;

  const cacheStatusEl = document.getElementById("res-cache-status");
  cacheStatusEl.textContent = data.cache_hit ? "已复用（无额外隐私消耗）" : "新结果已发布";
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
  } else if (results.cell_success_rates) {
    cellEntries = Object.entries(results.cell_success_rates).map(([k, v]) => ({
      cell: k,
      value: (Number(v) * 100).toFixed(2) + "%",
      unit: "success_rate",
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
    emptyCell.textContent = "当前筛选条件下没有匹配的统计单元。";
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
    tdUnit.textContent = UNIT_NAMES[item.unit] || item.unit;

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
  document.getElementById("cont-reused").textContent = data.reused_levels.length > 0 ? data.reused_levels.join(", ") : "无";
  document.getElementById("cont-new-exposed").textContent = data.newly_exposed_levels.length > 0 ? data.newly_exposed_levels.join(", ") : "无";

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
// Alerts & Error Translation
// -------------------------------------------------------------

function translateError(err) {
  if (!err) return "";
  let msg = "";
  let errorType = "";
  if (typeof err === "string") {
    msg = err;
  } else if (typeof err === "object") {
    msg = err.error || err.detail || "";
    errorType = err.error_type || "";
    if (!msg && !errorType) {
      msg = JSON.stringify(err);
    }
  }

  if (
    errorType === "BudgetExhaustedError" ||
    msg.includes("BudgetExhaustedError") ||
    msg.includes("budget exhausted") ||
    msg.includes("not authorized under the current lifetime privacy cap") ||
    msg.includes("Lifetime privacy cap exceeded")
  ) {
    return "当前全生命周期隐私预算不足，无法执行本次发布。";
  }
  if (msg.includes("No telemetry data loaded") || msg.includes("no telemetry")) {
    return "尚未加载遥测数据。请先加载演示数据或上传本地遥测数据文件。";
  }
  if (
    errorType === "ProductionSafetyConfigurationError" ||
    msg.includes("ProductionSafetyConfigurationError") ||
    msg.includes("Continual interval queries require an authoritative finalized stream configuration")
  ) {
    return "持续观测区间查询需要权威的已完成数据流配置，当前 Web 控制台不会自动初始化该状态。";
  }
  if (msg.includes("Delta must be strictly between 0 and 1")) {
    return "隐私参数 δ 必须严格介于 0 和 1 之间。";
  }
  if (msg.includes("Epsilon must be a positive number")) {
    return "隐私参数 ε 必须为正数。";
  }
  if (msg.includes("Failed to execute differential privacy release")) {
    return "差分隐私发布失败。";
  }
  if (msg.includes("Failed to load demo data")) {
    return "演示数据加载失败。";
  }
  if (msg.includes("Dataset upload failed")) {
    return "数据集上传失败。";
  }
  return msg || "操作遇到错误。";
}

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
