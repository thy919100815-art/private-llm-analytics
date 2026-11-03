// Private LLM Analytics Web Console (v1.0.x / v1.1)

// Vanilla JavaScript Client - Local Only (Simplified Chinese Localization)



let currentConfig = null;

let lastReleaseData = null;

function setDownloadButtonsEnabled(enabled) {

  const btnCsv = document.getElementById("btn-download-csv");

  const btnJson = document.getElementById("btn-download-json");

  if (btnCsv) {

    btnCsv.disabled = !enabled;

    btnCsv.title = enabled ? "下载 DP 处理后的 CSV 文件" : "请先成功生成 DP 发布结果后下载";

  }

  if (btnJson) {

    btnJson.disabled = !enabled;

    btnJson.title = enabled ? "下载 DP 处理后的 JSON 文件" : "请先成功生成 DP 发布结果后下载";

  }

}

let previousEffectiveEps = 0.0;



const METRIC_NAMES = {

  interaction_volume: "交互量",

  user_prevalence: "用户使用率",

  user_normalized_share: "用户归一化占比",

  success_rate: "成功率",

  top_k_tasks: "热门任务 Top-K",

};



const TASK_NAMES = {
  document: "文档类",
  technical: "技术类",
  analysis: "分析类",
  coding: "代码编写",
  translation: "多语言翻译",
  summarization: "文本摘要",
  customer_support: "客服支持",
  creative_writing: "创意写作",
  data_extraction: "信息抽取",
  qa: "问答咨询",
  code_review: "代码审查",
  brainstorming: "头脑风暴",
};



const UNIT_NAMES = {

  interactions: "次交互",

  "workflow share": "工作流占比",

  prevalence: "用户使用率",

  success_rate: "任务成功率",

  top_k: "排名与估计人数",

};



const GRANULARITY_NAMES = {

  daily: "按日（Daily）",

  weekly: "按周（Weekly）",

  biweekly: "双周（Bi-weekly）",

  monthly: "按月（Monthly）",

  full_9_cell: "全网格（9单元）",

};



const SOURCE_NAMES = {

  demo: "演示数据",

  uploaded: "本地上传",

};



const PROVIDER_NAMES = {

  provider_a: "OpenAI",

  provider_b: "Kimi",

  provider_c: "DeepSeek",

  internal: "内部 AI 治理团队",

};



function getProviderDisplayName(providerId) {

  if (!providerId) return "";

  if (currentConfig && currentConfig.recipient_details) {

    const detail = currentConfig.recipient_details.find((r) => r.recipient_id === providerId);

    if (detail && detail.display_name && detail.display_name !== providerId) {

      return detail.display_name;

    }

  }

  return PROVIDER_NAMES[providerId] || providerId;

}



function formatCellName(cell) {

  if (!cell || typeof cell !== "string") return cell || "-";

  if (cell.includes(":")) {

    const parts = cell.split(":", 2);

    const pDisp = getProviderDisplayName(parts[0]);

    return `${pDisp} × ${parts[1]}`;

  }

  return getProviderDisplayName(cell);

}



function renderMathString(mathStr, isDisplay = true) {

  const container = document.createElement("div");

  container.className = isDisplay ? "mech-math-box math-display" : "math-inline";

  if (!mathStr) return container;



  if (typeof katex !== "undefined" && typeof katex.renderToString === "function") {

    try {

      container.innerHTML = katex.renderToString(mathStr.trim(), {

        displayMode: isDisplay,

        throwOnError: false,

      });

      return container;

    } catch (err) {

      console.warn("KaTeX render error:", err);

    }

  }

  container.textContent = mathStr;

  return container;

}



function autoRenderMath(element) {

  if (!element) return;

  if (typeof renderMathInElement === "function") {

    try {

      renderMathInElement(element, {

        delimiters: [

          { left: "$$", right: "$$", display: true },

          { left: "\\[", right: "\\]", display: true },

          { left: "\\(", right: "\\)", display: false },

          { left: "$", right: "$", display: false },

        ],

        throwOnError: false,

      });

    } catch (err) {

      console.warn("auto-render error:", err);

    }

  }

}



async function triggerBlobDownload(url, defaultFilename) {

  try {

    showNotification("正在准备下载文件...", "info");

    const resp = await fetch(url);

    if (!resp.ok) {

      let errMsg = `下载失败 (HTTP ${resp.status})`;

      try {

        const errJson = await resp.json();

        errMsg = errJson.error || errJson.detail || errMsg;

      } catch (_) {}

      showNotification(errMsg, "error");

      return;

    }

    let filename = defaultFilename;

    const disposition = resp.headers.get("Content-Disposition");

    if (disposition && disposition.includes("filename=")) {

      const match = disposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);

      if (match && match[1]) {

        filename = match[1].replace(/['"]/g, "").trim();

      }

    }

    const blob = await resp.blob();

    const blobUrl = window.URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.style.display = "none";

    a.href = blobUrl;

    a.download = filename;

    document.body.appendChild(a);

    a.click();

    document.body.removeChild(a);

    window.URL.revokeObjectURL(blobUrl);

    showNotification(`已下载: ${filename}`, "success");

  } catch (err) {

    showNotification(`下载异常: ${err.message}`, "error");

  }

}



document.addEventListener("DOMContentLoaded", () => {

  initConsole();

});



window.addEventListener("load", () => {

  autoRenderMath(document.body);

});



async function initConsole() {

  await fetchConfig();

  await refreshDatasetStatus();

  await refreshPrivacyStatus();

  await refreshReleaseHistory();

  await refreshContinualStatus();



  renderMechanismCard(getSelectedMetric());

  renderRecipientPolicy(getSelectedRecipient());

  setupEventListeners();

  autoRenderMath(document.body);

}



// -------------------------------------------------------------

// API Calls & Initial Setup

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



function getSelectedRecipient() {

  const sel = document.getElementById("select-recipient");

  return sel ? sel.value : "provider_a";

}



function getSelectedMetric() {

  const radios = document.getElementsByName("metric-choice");

  for (const r of radios) {

    if (r.checked) return r.value;

  }

  return "interaction_volume";

}



function isTrustedInternalRecipient(recipientId) {

  if (!currentConfig || !currentConfig.recipient_details) {

    return recipientId === "internal";

  }

  const detail = currentConfig.recipient_details.find((r) => r.recipient_id === recipientId);

  return detail ? (detail.role === "trusted" && detail.exact_results === true) : (recipientId === "internal");

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

    const pDisp = getProviderDisplayName(p);

    label.appendChild(document.createTextNode(" " + pDisp + (pDisp !== p ? ` (${p})` : "")));

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



  // 3. Recipients dropdown with display names

  const recipSelect = document.getElementById("select-recipient");

  recipSelect.textContent = "";



  const details = cfg.recipient_details || [];

  if (details.length > 0) {

    details.forEach((rd) => {

      const opt = document.createElement("option");

      opt.value = rd.recipient_id;

      opt.textContent = `${rd.display_name} (${rd.recipient_id})`;

      recipSelect.appendChild(opt);

    });

  } else {

    cfg.recipients.forEach((r) => {

      const opt = document.createElement("option");

      opt.value = r;

      opt.textContent = r;

      recipSelect.appendChild(opt);

    });

  }



  if (cfg.recipients.includes("provider_a")) {

    recipSelect.value = "provider_a";

  } else if (cfg.recipients.length > 0) {

    recipSelect.value = cfg.recipients[0];

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



// -------------------------------------------------------------

// Mechanism Card Rendering (Sections A - I)

// -------------------------------------------------------------



function renderMechanismCard(metric) {

  if (!currentConfig || !currentConfig.mechanism_cards) return;

  const card = currentConfig.mechanism_cards[metric];

  if (!card) return;



  const titleEl = document.getElementById("mech-card-title");

  if (titleEl) {

    titleEl.textContent = `${card.display_title} — ${card.profile_name_zh}`;

  }



  const bodyEl = document.getElementById("mech-card-body");

  if (!bodyEl) return;

  bodyEl.textContent = "";



  // Section A: 统计目标

  const secA = createSection(

    "A. 统计目标（Statistical Target）",

    card.target,

    `隐私保护单位: ${card.privacy_unit} | 邻接关系: ${card.adjacency}`

  );

  bodyEl.appendChild(secA);



  // Section B: 原始数据示例

  if (card.raw_example) {

    const secB = document.createElement("div");

    secB.className = "mech-section";



    const title = document.createElement("div");

    title.className = "mech-section-title";

    title.textContent = "B. 原始数据示例（Raw Telemetry Example）";

    secB.appendChild(title);



    const note = document.createElement("p");

    note.className = "card-subtitle";

    note.style.color = "var(--primary)";

    note.style.marginBottom = "4px";

    note.textContent = `[说明] ${card.raw_example.note}`;

    secB.appendChild(note);



    const tbl = document.createElement("table");

    tbl.className = "mech-table";

    const thead = document.createElement("thead");

    const trHead = document.createElement("tr");

    (card.raw_example.columns || []).forEach((c) => {

      const th = document.createElement("th");

      th.textContent = c;

      trHead.appendChild(th);

    });

    thead.appendChild(trHead);

    tbl.appendChild(thead);



    const tbody = document.createElement("tbody");

    (card.raw_example.rows || []).forEach((row) => {

      const tr = document.createElement("tr");

      (card.raw_example.columns || []).forEach((c) => {

        const td = document.createElement("td");

        td.textContent = row[c] !== undefined ? row[c] : "-";

        tr.appendChild(td);

      });

      tbody.appendChild(tr);

    });

    tbl.appendChild(tbody);

    secB.appendChild(tbl);

    bodyEl.appendChild(secB);

  }



  // Section C: 用户级贡献

  if (card.user_contribution) {

    const mathText = card.user_contribution.formula || "";

    const desc = `${card.user_contribution.explanation || ""}${card.user_contribution.numeric_example ? `\n示例: ${card.user_contribution.numeric_example}` : ""}`;

    const secC = createSection("C. 用户级贡献转换（User-Level Contribution）", desc, null, mathText);

    bodyEl.appendChild(secC);

  }



  // Section D: 贡献限制

  if (card.contribution_bound) {

    const mathText = card.contribution_bound.formula || "";

    const secD = createSection("D. 贡献限制与裁剪（Contribution Bounding）", card.contribution_bound.explanation || "", null, mathText);

    bodyEl.appendChild(secD);

  }



  // Section E: 敏感度

  if (card.sensitivity) {

    const mathText = card.sensitivity.formula || "";

    const boundNotes = [];

    if (card.sensitivity.conservative_bound) boundNotes.push(card.sensitivity.conservative_bound);

    if (card.sensitivity.tight_bound) boundNotes.push(card.sensitivity.tight_bound);

    const noteText = boundNotes.length > 0 ? boundNotes.join("  |  ") : null;

    const secE = createSection("E. 敏感度界定（Sensitivity Calibration）", card.sensitivity.explanation || "", noteText, mathText);

    bodyEl.appendChild(secE);

  }



  // Section F: DP 机制

  if (card.dp_mechanism) {

    const mathText = card.dp_mechanism.equation || "";

    const secF = createSection("F. 差分隐私机制（Differential Privacy Mechanism）", card.dp_mechanism.explanation || "", null, mathText);

    bodyEl.appendChild(secF);

  }



  // Section G: 后处理

  if (card.post_processing) {

    const steps = card.post_processing.steps || [];

    const secG = createSection("G. 后处理计算（Post-Processing）", card.post_processing.explanation || "", null, steps);

    bodyEl.appendChild(secG);

  }



  // Section H: 发布结果代表什么

  if (card.result_meaning) {

    const content = `语义解读: ${card.result_meaning.interpretation || ""} | 示例: ${card.result_meaning.example || ""} | 度量单位: ${card.result_meaning.units || ""}`;

    const secH = createSection("H. 发布结果代表什么（Semantic Interpretation）", content);

    bodyEl.appendChild(secH);

  }



  // Section I: 隐私记账

  if (card.privacy_accounting) {

    const content = `记账族: ${card.privacy_accounting.accounting || ""} | 机制: ${card.privacy_accounting.mechanism || ""}\n${card.privacy_accounting.explanation || ""}`;

    const secI = createSection("I. 隐私记账与持久化（Privacy Accounting & Replay）", content);

    bodyEl.appendChild(secI);

  }



  // Render all math formulas in the body

  autoRenderMath(bodyEl);

}



function createSection(title, desc, note, mathContent) {

  const sec = document.createElement("div");

  sec.className = "mech-section";



  const t = document.createElement("div");

  t.className = "mech-section-title";

  t.textContent = title;

  sec.appendChild(t);



  if (desc) {

    const d = document.createElement("div");

    d.className = "mech-section-desc";

    d.textContent = desc;

    sec.appendChild(d);

  }



  if (Array.isArray(mathContent)) {

    // Array of formula steps

    const stepsContainer = document.createElement("div");

    stepsContainer.style.margin = "6px 0";

    mathContent.forEach((stepStr, idx) => {

      const stepBox = renderMathString(stepStr, true);

      stepsContainer.appendChild(stepBox);

      if (idx < mathContent.length - 1) {

        const arrow = document.createElement("div");

        arrow.style.cssText = "text-align: center; color: var(--primary); font-size: 14px; margin: 2px 0;";

        arrow.textContent = "↓";

        stepsContainer.appendChild(arrow);

      }

    });

    sec.appendChild(stepsContainer);

  } else if (mathContent && typeof mathContent === "string" && mathContent.trim().length > 0) {

    const m = renderMathString(mathContent, true);

    sec.appendChild(m);

  }



  if (note) {

    const n = document.createElement("p");

    n.className = "card-subtitle";

    n.style.marginTop = "4px";

    n.textContent = note;

    sec.appendChild(n);

  }



  return sec;

}



// -------------------------------------------------------------

// Recipient Policy Visibility

// -------------------------------------------------------------



function renderRecipientPolicy(recipientId) {

  if (!currentConfig || !currentConfig.recipient_details) return;

  const detail = currentConfig.recipient_details.find((r) => r.recipient_id === recipientId);

  if (!detail) return;



  const nameEl = document.getElementById("recip-display-name");

  if (nameEl) nameEl.textContent = detail.display_name;



  const idEl = document.getElementById("recip-stable-id");

  if (idEl) idEl.textContent = detail.recipient_id;



  const roleEl = document.getElementById("recip-role-desc");

  if (roleEl) roleEl.textContent = detail.role === "trusted" ? "可信内部管理员" : "外部供应商";



  const provEl = document.getElementById("recip-providers-desc");

  if (provEl) provEl.textContent = detail.providers.join(", ") === "*" ? "全部模型供应商" : detail.providers.join(", ");



  const exactEl = document.getElementById("recip-exact-perm");

  if (exactEl) {

    exactEl.textContent = detail.exact_results ? "允许（内部可见）" : "严格禁止";

    exactEl.className = detail.exact_results ? "badge badge-success" : "badge badge-warning";

  }



  const crossEl = document.getElementById("recip-cross-perm");

  if (crossEl) crossEl.textContent = detail.cross_provider_comparison ? "允许" : "禁止";



  const scopeBadge = document.getElementById("recip-scope-badge");

  if (scopeBadge) scopeBadge.textContent = detail.privacy_scope_mode;



  const metricsEl = document.getElementById("recip-metrics-desc");

  if (metricsEl) metricsEl.textContent = detail.metrics.join(", ") === "*" ? "全部指标" : `${detail.metrics.length} 项标准指标`;



  // Update visibility of internal accuracy panel

  const isTrusted = detail.exact_results && detail.role === "trusted";

  const internalPanel = document.getElementById("internal-accuracy-panel");

  if (internalPanel && !isTrusted) {

    internalPanel.style.display = "none";

  }

}



// -------------------------------------------------------------

// Status Refreshers

// -------------------------------------------------------------



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

      infoEl.textContent = `数据行数: ${data.rows.toLocaleString()} | 活跃用户数: ${(data.user_count || data.active_users || 0).toLocaleString()} | 数据来源: ${displaySource}`;



      // Update Dataset Overview Grid

      const recEl = document.getElementById("dataset-records-count");

      if (recEl) recEl.textContent = `${data.rows.toLocaleString()} 条`;



      const usrEl = document.getElementById("dataset-users-count");

      if (usrEl) usrEl.textContent = `${(data.user_count || data.active_users || 0).toLocaleString()} 个`;



      const provEl = document.getElementById("dataset-providers-count");

      if (provEl) provEl.textContent = `${data.provider_count || 3} 种`;



      const taskEl = document.getElementById("dataset-tasks-count");

      if (taskEl) taskEl.textContent = `${data.task_count || 3} 类`;



      const timeEl = document.getElementById("dataset-timerange");

      if (timeEl) {

        if (data.time_range_display) {

          timeEl.textContent = data.time_range_display;

        } else if (data.time_range && typeof data.time_range === "object" && data.time_range.start && data.time_range.end) {

          timeEl.textContent = `${data.time_range.start} ～ ${data.time_range.end}`;

        } else if (typeof data.time_range === "string") {

          timeEl.textContent = data.time_range;

        } else {

          timeEl.textContent = "全量时间";

        }

      }



      // Update Plain-Language Flow Chart Counts

      const flowRaw = document.getElementById("flow-raw-count");

      if (flowRaw) flowRaw.textContent = `${data.rows.toLocaleString()} 条日志`;



      const flowUsr = document.getElementById("flow-user-count");

      if (flowUsr) flowUsr.textContent = `${(data.user_count || data.active_users || 0).toLocaleString()} 个员工`;

    } else {

      statusEl.className = "badge badge-warning";

      statusEl.textContent = "未加载数据";

      infoEl.textContent = "尚未加载遥测数据。请加载演示数据，或上传本地遥测数据文件（CSV / Parquet）。";



      const recEl = document.getElementById("dataset-records-count");

      if (recEl) recEl.textContent = "0 条";

      const usrEl = document.getElementById("dataset-users-count");

      if (usrEl) usrEl.textContent = "0 个";

      const provEl = document.getElementById("dataset-providers-count");

      if (provEl) provEl.textContent = "-";

      const taskEl = document.getElementById("dataset-tasks-count");

      if (taskEl) taskEl.textContent = "-";

      const timeEl = document.getElementById("dataset-timerange");

      if (timeEl) timeEl.textContent = "未加载";

    }

  } catch (err) {

    showError("无法获取数据集状态：" + (err.error || err.detail || err));

  }

}



async function refreshPrivacyStatus() {

  try {

    const recipient = getSelectedRecipient();

    const res = await fetch(`/api/privacy/status?recipient=${encodeURIComponent(recipient)}`);

    const data = await res.json();



    previousEffectiveEps = data.effective_epsilon;

    document.getElementById("budget-eps-used").textContent = data.effective_epsilon.toFixed(4);

    document.getElementById("budget-eps-cap").textContent = data.epsilon_cap.toFixed(4);

    document.getElementById("budget-headroom").textContent = data.remaining_headroom.toFixed(4);

    document.getElementById("budget-delta").textContent = data.delta_global.toExponential(1);

    document.getElementById("budget-accountant-mode").textContent = "RDP → (ε, δ)";



    const fillEl = document.getElementById("budget-progress-fill");

    fillEl.style.width = `${Math.min(100, data.spend_percentage)}%`;



    // Render Epoch details

    if (data.epoch) {

      const ep = data.epoch;

      const drEl = document.getElementById("epoch-date-range");

      if (drEl) drEl.textContent = `${ep.epoch_start} ～ ${ep.epoch_end}`;

      const dayEl = document.getElementById("epoch-current-day");

      if (dayEl) dayEl.textContent = `${ep.current_day}`;

      const stEl = document.getElementById("epoch-status");

      if (stEl) {

        stEl.textContent = ep.status;

        stEl.className = ep.status === "ACTIVE" ? "badge badge-success" : "badge badge-warning";

      }

      const resPctEl = document.getElementById("epoch-reserve-pct");

      if (resPctEl) resPctEl.textContent = `${ep.reserve_percentage_remaining.toFixed(1)}%`;



      // Render Workloads breakdown

      if (ep.workloads) {

        const wls = ep.workloads;

        const updateWl = (key, spentId, barId) => {

          const item = wls[key];

          if (item) {

            const spentEl = document.getElementById(spentId);

            if (spentEl) {

              spentEl.textContent = `${item.spent_percentage.toFixed(1)}% (${item.release_count} 次 | ε: ${item.spent_epsilon.toFixed(2)}/${item.allocated_epsilon.toFixed(2)})`;

            }

            const barEl = document.getElementById(barId);

            if (barEl) {

              barEl.style.width = `${Math.min(100, item.spent_percentage)}%`;

            }

          }

        };



        updateWl("interaction_volume", "wl-spent-volume", "wl-bar-volume");

        updateWl("success_rate", "wl-spent-success_rate", "wl-bar-success_rate");

        updateWl("user_prevalence", "wl-spent-prevalence", "wl-bar-prevalence");

        updateWl("user_normalized_share", "wl-spent-share", "wl-bar-share");

        updateWl("top_k_tasks", "wl-spent-top_k", "wl-bar-top_k");



        // Reserve

        const resItem = wls["reserve"];

        if (resItem) {

          const resSpentEl = document.getElementById("wl-spent-reserve");

          if (resSpentEl) {

            const remPct = Math.max(0, 100 - resItem.spent_percentage);

            resSpentEl.textContent = `已用 ${resItem.spent_percentage.toFixed(1)}% (${resItem.release_count} 次) | 剩余 ${remPct.toFixed(1)}%`;

          }

          const resBarEl = document.getElementById("wl-bar-reserve");

          if (resBarEl) {

            resBarEl.style.width = `${Math.min(100, resItem.spent_percentage)}%`;

          }

        }

      }

    }



    // Render Lifetime Exposure Monitor

    if (data.lifetime_exposure) {

      const lt = data.lifetime_exposure;

      const countEl = document.getElementById("lifetime-epochs-count");

      if (countEl) countEl.textContent = `${lt.total_epochs_count}`;

      const ltEpsEl = document.getElementById("lifetime-effective-eps");

      if (ltEpsEl) ltEpsEl.textContent = lt.lifetime_effective_epsilon.toFixed(4);

      const noticeEl = document.getElementById("lifetime-notice");

      if (noticeEl) noticeEl.textContent = lt.notice;

    }

  } catch (err) {

    showError("无法获取隐私预算状态：" + (err.error || err.detail || err));

  }

}



async function refreshReleaseHistory() {

  try {

    const res = await fetch("/api/releases");

    const releases = await res.json();

    const tbody = document.getElementById("history-table-body");

    tbody.textContent = "";



    if (releases.length === 0) {

      const tr = document.createElement("tr");

      tr.innerHTML = `<td colspan="8" style="text-align:center; color: var(--text-light);">暂无已确认的权威发布记录</td>`;

      tbody.appendChild(tr);

      return;

    }



    releases.forEach((r) => {

      const tr = document.createElement("tr");

      const shortId = (r.release_id || r.release_key || "").substring(0, 8);

      const isCommitted = r.status === "COMMITTED" || r.cache_hit;

      const statusBadge = isCommitted

        ? `<span class="badge badge-success">已提交</span>`

        : `<span class="badge badge-warning">${r.status || "草稿"}</span>`;



      const displayMetric = METRIC_NAMES[r.metric] || r.metric;

      const displayRecipient = r.recipient_name || r.recipient;



      tr.innerHTML = `

        <td><code style="font-size: 11px;">${shortId}...</code></td>

        <td>${r.timestamp ? r.timestamp.substring(0, 19).replace("T", " ") : "-"}</td>

        <td><strong>${displayMetric}</strong></td>

        <td>${displayRecipient}</td>

        <td>ε = ${r.epsilon}, δ = ${r.delta}</td>

        <td><span class="badge">${r.mechanism || "高斯向量"}</span></td>

        <td>${statusBadge}</td>

        <td>

          <button class="btn-action-sm" onclick="downloadReleaseCsv('${r.release_id || r.release_key}', '${r.recipient}')">CSV</button>

          <button class="btn-action-sm" onclick="downloadReleaseJson('${r.release_id || r.release_key}', '${r.recipient}')">JSON</button>

        </td>

      `;

      tbody.appendChild(tr);

    });

  } catch (err) {

    showError("无法加载发布历史：" + (err.error || err.detail || err));

  }

}



async function refreshContinualStatus() {

  try {

    const res = await fetch("/api/continual/status");

    const data = await res.json();

    const badge = document.getElementById("continual-status-badge");

    const btn = document.getElementById("btn-continual-query");

    const msg = document.getElementById("continual-msg");



    if (data.enabled) {

      badge.className = "badge badge-success";

      badge.textContent = "已启用（权威流）";

      btn.disabled = false;

      msg.textContent = `持续观测状态就绪。二叉树拓扑结构: ${data.tree_topology.levels} 层，总叶子容量 ${data.tree_topology.leaf_capacity} 天。`;

      document.getElementById("cont-metric").textContent = METRIC_NAMES[data.supported_metric] || data.supported_metric;

      document.getElementById("cont-epoch-days").textContent = data.epoch_days;

      document.getElementById("cont-bounds").textContent = `${data.day_bound} / ${data.epoch_bound}`;

    } else {

      badge.className = "badge badge-warning";

      badge.textContent = "未启用（默认）";

      btn.disabled = true;

      msg.textContent = data.reason || "持续观测功能未开启。";

    }

  } catch (err) {

    // Graceful fallback

  }

}



// -------------------------------------------------------------

// Event Listeners Setup

// -------------------------------------------------------------



function setupEventListeners() {

  // Demo Data Load

  document.getElementById("btn-load-demo").addEventListener("click", async () => {

    clearAlerts();

    try {

      const res = await fetch("/api/dataset/demo", { method: "POST" });

      if (!res.ok) throw await res.json();

      const data = await res.json();

      lastReleaseData = null;

      setDownloadButtonsEnabled(false);

      const resContainer = document.getElementById("release-result-card");

      if (resContainer) resContainer.style.display = "none";

      await refreshDatasetStatus();

      showSuccess(`演示遥测数据已成功加载。共加载 ${data.rows.toLocaleString()} 条记录。`);

    } catch (err) {

      showError("加载演示数据失败：" + (err.error || err.detail || err));

    }

  });



  // CSV Sample Template Download

  document.getElementById("btn-download-sample-csv").addEventListener("click", () => {

    triggerBlobDownload("/api/dataset/sample-template", "sample_private_llm_analytics.csv");

  });



  // Data Format Sample Modal

  document.getElementById("btn-view-sample-modal").addEventListener("click", () => {

    document.getElementById("data-format-modal").classList.add("active");

  });

  document.getElementById("btn-close-sample-modal").addEventListener("click", () => {

    document.getElementById("data-format-modal").classList.remove("active");

  });



  // Input Preview Modal (Trusted Admin Only)

  document.getElementById("btn-view-input-preview").addEventListener("click", async () => {

    try {

      const res = await fetch(`/api/dataset/preview?limit=20`);

      if (!res.ok) {

        const err = await res.json();

        throw new Error(err.error || err.detail || `HTTP ${res.status}`);

      }

      const data = await res.json();

      renderInputPreviewModal(data);

    } catch (e) {

      showError("无法获取输入数据预览：" + (e.message || e));

    }

  });



  document.getElementById("btn-close-input-preview").addEventListener("click", () => {

    document.getElementById("input-preview-modal").classList.remove("active");

  });



  // File Upload

  document.getElementById("btn-upload").addEventListener("click", () => {

    document.getElementById("file-input").click();

  });



  document.getElementById("file-input").addEventListener("change", async (e) => {

    const file = e.target.files[0];

    if (!file) return;



    clearAlerts();

    const formData = new FormData();

    formData.append("file", file);



    try {

      const res = await fetch("/api/dataset/upload", {

        method: "POST",

        body: formData,

      });

      if (!res.ok) throw await res.json();

      const data = await res.json();

      await refreshDatasetStatus();

      showSuccess(`文件 "${file.name}" 上传成功，共导入 ${data.rows.toLocaleString()} 条遥测记录。`);

    } catch (err) {

      showError("上传数据文件失败：" + (err.error || err.detail || err));

    } finally {

      e.target.value = "";

    }

  });



  // Recipient change

  document.getElementById("select-recipient").addEventListener("change", (e) => {

    renderRecipientPolicy(e.target.value);

    refreshPrivacyStatus();

  });



  // Metric selection change -> update Mechanism Card

  document.querySelectorAll("input[name='metric-choice']").forEach((radio) => {

    radio.addEventListener("change", () => {

      renderMechanismCard(getSelectedMetric());

    });

  });



  // Privacy Preset Cards

  document.querySelectorAll(".preset-card").forEach((card) => {

    card.addEventListener("click", () => {

      document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("active"));

      card.classList.add("active");

      const eps = parseFloat(card.dataset.eps);

      document.getElementById("input-epsilon").value = eps;

      updatePresetIndicator(eps, card.dataset.desc);

    });

  });



  // Toggle Advanced Privacy Settings

  document.getElementById("btn-toggle-advanced-privacy").addEventListener("click", () => {

    const panel = document.getElementById("advanced-privacy-panel");

    panel.style.display = panel.style.display === "none" ? "block" : "none";

  });



  // Planner Recommendation

  document.getElementById("btn-get-recommendation").addEventListener("click", async () => {

    clearAlerts();

    const metric = getSelectedMetric();

    const recipient = getSelectedRecipient();



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

    const recipient = getSelectedRecipient();

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

    const recipient = getSelectedRecipient();

    const isOverride = document.getElementById("chk-manual-override")?.checked;

    let eps = null;

    let delta = null;

    if (isOverride) {

      eps = parseFloat(document.getElementById("input-epsilon").value);

      delta = parseFloat(document.getElementById("input-delta").value);

    }

    const windowVal = document.getElementById("select-window").value;



    const payload = {

      metric: metric,

      recipient: recipient,

      epsilon: eps,

      delta: delta,

      window_days: windowVal === "all" ? null : parseInt(windowVal, 10),

      use_epoch_planner: !isOverride,

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

      setDownloadButtonsEnabled(true);

      renderReleaseResult(data);

      await refreshPrivacyStatus();

      await refreshReleaseHistory();

      showSuccess(

        data.cache_hit

          ? "已复用现有标准发布结果，本次没有产生额外隐私消耗。"

          : `差分隐私结果已生成，并写入持久化隐私记账记录。增量 ε 消耗: ${data.incremental_privacy_cost}`

      );



      // If trusted internal recipient, fetch accuracy evaluation and utility summary

      if (isTrustedInternalRecipient(recipient)) {

        await loadAccuracyEvaluation(data.release_id || data.release_key, recipient);

        await loadUtilitySummary(recipient);

      } else {

        const evalPanel = document.getElementById("internal-accuracy-panel");

        if (evalPanel) evalPanel.style.display = "none";

      }

    } catch (err) {

      lastReleaseData = null;

      setDownloadButtonsEnabled(false);

      const resContainer = document.getElementById("release-result-card");

      if (resContainer) resContainer.style.display = "none";

      showError(translateError(err) || "差分隐私发布失败。");

    }

  });



  // Downloads for current release

  document.getElementById("btn-download-csv").addEventListener("click", () => {

    if (!lastReleaseData) return showError("尚未生成发布结果，无法下载。");

    const relId = lastReleaseData.release_id || lastReleaseData.release_key;

    const recipient = lastReleaseData.recipient;

    triggerBlobDownload(`/api/release/download/csv?release_id=${encodeURIComponent(relId)}&recipient=${encodeURIComponent(recipient)}`, `private_llm_analytics_release_${relId}.csv`);

  });



  document.getElementById("btn-download-json").addEventListener("click", () => {

    if (!lastReleaseData) return showError("尚未生成发布结果，无法下载。");

    const relId = lastReleaseData.release_id || lastReleaseData.release_key;

    const recipient = lastReleaseData.recipient;

    triggerBlobDownload(`/api/release/download/json?release_id=${encodeURIComponent(relId)}&recipient=${encodeURIComponent(recipient)}`, `private_llm_analytics_release_${relId}.json`);

  });



  // Downloads for Internal Evaluation Report

  document.getElementById("btn-download-eval-csv").addEventListener("click", () => {

    if (!lastReleaseData) return showError("尚未生成发布结果，无法下载。");

    const relId = lastReleaseData.release_id || lastReleaseData.release_key;

    const recipient = getSelectedRecipient();

    triggerBlobDownload(`/api/evaluation/download/csv?release_id=${encodeURIComponent(relId)}&recipient=${encodeURIComponent(recipient)}`, `private_llm_analytics_evaluation_${relId}.csv`);

  });



  document.getElementById("btn-download-eval-json").addEventListener("click", () => {

    if (!lastReleaseData) return showError("尚未生成发布结果，无法下载。");

    const relId = lastReleaseData.release_id || lastReleaseData.release_key;

    const recipient = getSelectedRecipient();

    triggerBlobDownload(`/api/evaluation/download/json?release_id=${encodeURIComponent(relId)}&recipient=${encodeURIComponent(recipient)}`, `private_llm_analytics_evaluation_${relId}.json`);

  });



  // Epsilon-Utility Sweep Experiment Button

  document.getElementById("btn-run-epsilon-sweep").addEventListener("click", async () => {

    const btn = document.getElementById("btn-run-epsilon-sweep");

    btn.disabled = true;

    btn.textContent = "实验分析中...";

    try {

      const metric = getSelectedMetric();

      const res = await fetch("/api/evaluation/sweep", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

          metric: metric,

          recipient: "internal",

          candidate_epsilons: [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],

          n_trials: 5,

        }),

      });



      if (!res.ok) throw await res.json();

      const sweepData = await res.json();

      renderEpsilonSweepResults(sweepData);

      showSuccess("ε 与准确性关系实验完成（生产隐私账本 0 消耗）。");

    } catch (err) {

      showError("ε 扫描实验失败：" + (err.error || err.detail || err));

    } finally {

      btn.disabled = false;

      btn.textContent = "分析 ε-准确性关系";

    }

  });



  // Continual Query

  document.getElementById("btn-continual-query").addEventListener("click", async () => {

    clearAlerts();

    const startDay = parseInt(document.getElementById("continual-start-day").value, 10);

    const endDay = parseInt(document.getElementById("continual-end-day").value, 10);

    const recipient = getSelectedRecipient();



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



  // Advance Epoch

  const btnAdv = document.getElementById("btn-advance-epoch");

  if (btnAdv) {

    btnAdv.addEventListener("click", async () => {

      if (!confirm("确认推进到下一个 90 天 Privacy Epoch？\n本周期将归档关闭，新周期将重新获得业务配额，历史隐私暴露将持续在 Lifetime Exposure 中累计。")) {

        return;

      }

      try {

        const res = await fetch("/api/privacy/epoch/advance", { method: "POST" });

        if (!res.ok) throw await res.json();

        await refreshPrivacyStatus();

        showSuccess("已成功推进到新一个 90 天 Privacy Epoch。");

      } catch (e) {

        showError("推进 Epoch 失败：" + (e.message || e.detail || e));

      }

    });

  }



  // Admin Policy Modal

  document.getElementById("btn-open-policy-modal").addEventListener("click", async () => {

    clearAlerts();

    try {

      const res = await fetch("/api/admin/privacy-policy");

      if (!res.ok) throw await res.json();

      const pol = await res.json();



      document.getElementById("admin-input-eps-cap").value = pol.lifetime_epsilon_cap;

      document.getElementById("admin-input-global-delta").value = pol.global_delta;

      document.getElementById("admin-input-def-eps").value = pol.default_query_epsilon;

      document.getElementById("admin-input-def-delta").value = pol.default_query_delta;

      document.getElementById("admin-cur-spent").textContent = pol.current_effective_epsilon.toFixed(4);

      document.getElementById("admin-input-reason").value = "";



      const tbody = document.getElementById("admin-policy-audit-tbody");

      tbody.textContent = "";

      if (pol.audit_history && pol.audit_history.length > 0) {

        pol.audit_history.forEach((h) => {

          const tr = document.createElement("tr");

          tr.innerHTML = `

            <td>${h.timestamp ? h.timestamp.substring(0, 19).replace("T", " ") : "-"}</td>

            <td><strong>${h.actor}</strong></td>

            <td>${h.epsilon_cap}</td>

            <td>${h.effective_epsilon_at_change}</td>

            <td>${h.reason || "-"}</td>

          `;

          tbody.appendChild(tr);

        });

      } else {

        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-light);">暂无调整记录</td></tr>`;

      }



      document.getElementById("admin-policy-modal").classList.add("active");

    } catch (err) {

      showError("无法加载管理员隐私政策：" + (err.error || err.detail || err));

    }

  });



  document.getElementById("btn-cancel-policy").addEventListener("click", () => {

    document.getElementById("admin-policy-modal").classList.remove("active");

  });



  document.getElementById("btn-save-policy").addEventListener("click", async () => {

    const epsCap = parseFloat(document.getElementById("admin-input-eps-cap").value);

    const globDelta = parseFloat(document.getElementById("admin-input-global-delta").value);

    const defEps = parseFloat(document.getElementById("admin-input-def-eps").value);

    const defDelta = parseFloat(document.getElementById("admin-input-def-delta").value);

    const reason = document.getElementById("admin-input-reason").value;



    if (isNaN(epsCap) || epsCap <= 0) {

      alert("全生命周期 ε 上限必须为正数。");

      return;

    }

    if (isNaN(globDelta) || globDelta <= 0 || globDelta >= 1) {

      alert("全局 δ 必须严格介于 0 和 1 之间。");

      return;

    }

    if (isNaN(defEps) || defEps <= 0) {

      alert("默认单次查询 ε 必须为正数。");

      return;

    }

    if (isNaN(defDelta) || defDelta <= 0 || defDelta >= 1) {

      alert("默认单次查询 δ 必须严格介于 0 和 1 之间。");

      return;

    }



    try {

      const res = await fetch("/api/admin/privacy-policy", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

          recipient: "internal",

          lifetime_epsilon_cap: epsCap,

          global_delta: globDelta,

          default_query_epsilon: defEps,

          default_query_delta: defDelta,

          reason: reason,

        }),

      });



      if (!res.ok) throw await res.json();

      document.getElementById("admin-policy-modal").classList.remove("active");

      await refreshPrivacyStatus();

      await fetchConfig();

      showSuccess("全生命周期隐私安全政策已成功更新并记录审计日志。");

    } catch (err) {

      alert("更新隐私政策失败：" + (err.error || err.detail || err));

    }

  });

}



function updatePresetIndicator(eps, desc) {

  const labelEl = document.getElementById("preset-active-label");

  const descEl = document.getElementById("tradeoff-desc");

  const fillEl = document.getElementById("tradeoff-fill");



  if (labelEl) labelEl.textContent = `当前选择：ε = ${eps}`;

  if (descEl) descEl.textContent = desc || `ε = ${eps}：数值越小隐私保护越强，数值越大统计精度越高。`;



  if (fillEl) {

    if (eps <= 0.5) fillEl.style.width = "25%";

    else if (eps <= 1.0) fillEl.style.width = "50%";

    else fillEl.style.width = "80%";

  }

}



// -------------------------------------------------------------

// Input Preview Modal Rendering

// -------------------------------------------------------------



function renderInputPreviewModal(data) {

  const tbody = document.getElementById("input-preview-tbody");

  tbody.textContent = "";



  const rows = data.rows || [];

  if (rows.length === 0) {

    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-light);">暂无数据记录</td></tr>`;

  } else {

    rows.forEach((r) => {

      const tr = document.createElement("tr");

      tr.innerHTML = `

        <td><code>${r.user_id}</code></td>

        <td>${r.timestamp}</td>

        <td>${r.provider_name || r.provider} <code style="font-size:10px; color:var(--text-light);">(${r.provider})</code></td>

        <td>${r.task_name || r.task}</td>

        <td><span class="badge ${r.success === '成功' ? 'badge-success' : 'badge-warning'}">${r.success}</span></td>

      `;

      tbody.appendChild(tr);

    });

  }



  document.getElementById("input-preview-modal").classList.add("active");

}



// -------------------------------------------------------------

// Internal Accuracy Evaluation & Utility Summary

// -------------------------------------------------------------



async function loadAccuracyEvaluation(releaseId, recipient) {

  const panel = document.getElementById("internal-accuracy-panel");

  if (!panel) return;

  panel.style.display = "block";



  try {

    const res = await fetch(`/api/evaluation/accuracy?release_id=${encodeURIComponent(releaseId)}&recipient=${encodeURIComponent(recipient)}`);

    if (!res.ok) throw await res.json();

    const evalData = await res.json();

    renderAccuracyEvaluation(evalData);

  } catch (err) {

    console.warn("Could not load internal evaluation:", err);

  }

}



function renderAccuracyEvaluation(evalData) {

  document.getElementById("eval-primary-name").textContent = evalData.primary_metric_name || "核心指标";

  document.getElementById("eval-primary-val").textContent = evalData.primary_metric_display || "-";

  document.getElementById("eval-mae-val").textContent = evalData.mae !== null ? evalData.mae.toFixed(2) : "-";

  document.getElementById("eval-rmse-val").textContent = evalData.rmse !== null ? evalData.rmse.toFixed(2) : "-";



  const thead = document.getElementById("accuracy-table-head");

  const tbody = document.getElementById("accuracy-table-body");

  tbody.textContent = "";



  const metric = evalData.metric;



  if (metric === "interaction_volume") {

    thead.innerHTML = `

      <tr>

        <th>统计单元</th>

        <th>真实未加噪 (M_raw)</th>

        <th>单人截断后 (M_clip)</th>

        <th>DP 发布值 (M_dp)</th>

        <th>截断偏差</th>

        <th>DP 噪声误差</th>

        <th>总误差</th>

        <th>相对误差 (%)</th>

      </tr>

    `;



    (evalData.rows || []).forEach((r) => {

      const tr = document.createElement("tr");

      tr.innerHTML = `

        <td><strong>${formatCellName(r.cell)}</strong></td>

        <td>${r.raw_exact.toFixed(2)}</td>

        <td>${r.clipped_exact.toFixed(2)}</td>

        <td style="font-family: monospace; font-weight: 700;">${r.dp_value.toFixed(2)}</td>

        <td style="color: ${r.clipping_bias < 0 ? '#dc2626' : 'var(--text-muted)'};">${r.clipping_bias.toFixed(2)}</td>

        <td style="color: ${r.dp_noise_error < 0 ? '#dc2626' : '#16a34a'};">${r.dp_noise_error.toFixed(2)}</td>

        <td style="font-weight: 600;">${r.total_error.toFixed(2)}</td>

        <td>${r.relative_error_pct.toFixed(2)}%</td>

      `;

      tbody.appendChild(tr);

    });

  } else if (metric === "top_k_tasks") {

    thead.innerHTML = `

      <tr>

        <th>排名</th>

        <th>真实热门任务</th>

        <th>真实频次</th>

        <th>DP 估计热门任务</th>

        <th>DP 估计频次</th>

        <th>命中状态</th>

      </tr>

    `;



    (evalData.rows || []).forEach((r) => {

      const tr = document.createElement("tr");

      tr.innerHTML = `

        <td><strong>#${r.rank}</strong></td>

        <td>${TASK_NAMES[r.exact_task] || r.exact_task} <code style="font-size:10px;">(${r.exact_task})</code></td>

        <td>${r.exact_count}</td>

        <td>${TASK_NAMES[r.dp_task] || r.dp_task} <code style="font-size:10px;">(${r.dp_task})</code></td>

        <td style="font-family: monospace;">${r.dp_noisy_count.toFixed(2)}</td>

        <td><span class="badge ${r.hit ? 'badge-success' : 'badge-warning'}">${r.hit ? '命中' : '未命中'}</span></td>

      `;

      tbody.appendChild(tr);

    });

  } else {

    // prevalence, normalized_share, success_rate

    thead.innerHTML = `

      <tr>

        <th>统计单元</th>

        <th>真实精确比率 / 占比</th>

        <th>DP 估计值</th>

        <th>百分点误差 (pp)</th>

        <th>状态说明</th>

      </tr>

    `;



    (evalData.rows || []).forEach((r) => {

      const tr = document.createElement("tr");

      const ppErr = r.pp_error !== null && r.pp_error !== undefined ? `${r.pp_error.toFixed(2)} pp` : "-";

      tr.innerHTML = `

        <td><strong>${formatCellName(r.cell)}</strong></td>

        <td>${r.exact_rate !== null ? (r.exact_rate * 100).toFixed(2) + "%" : (r.exact_share !== null ? (r.exact_share * 100).toFixed(2) + "%" : "-")}</td>

        <td style="font-family: monospace; font-weight: 700;">${r.dp_rate !== null && r.dp_rate !== undefined ? (r.dp_rate * 100).toFixed(2) + "%" : (r.dp_share !== null && r.dp_share !== undefined ? (r.dp_share * 100).toFixed(2) + "%" : "受抑制")}</td>

        <td style="color: #be123c; font-weight: 600;">${ppErr}</td>

        <td><span class="badge ${r.suppressed ? 'badge-warning' : 'badge-success'}">${r.suppressed ? '小样本受抑制' : '正常发布'}</span></td>

      `;

      tbody.appendChild(tr);

    });

  }

}



async function loadUtilitySummary(recipient) {

  try {

    const res = await fetch(`/api/evaluation/summary?recipient=${encodeURIComponent(recipient)}`);

    if (!res.ok) throw await res.json();

    const summary = await res.json();

    renderUtilitySummaryTable(summary);

  } catch (err) {

    console.warn("Could not load utility summary:", err);

  }

}



function renderUtilitySummaryTable(summary) {

  const tbody = document.getElementById("utility-summary-tbody");

  tbody.textContent = "";



  const explanations = {

    user_prevalence: "员工使用率以百分点 (pp) 衡量，高斯扰动对高覆盖率模型影响轻微。",

    user_normalized_share: "用户工作流占比每人权重归一，各类别分布保持整体一致性。",

    interaction_volume: "交互总量包含高频员工 L2 裁剪偏差与 DP 高斯噪声。",

    success_rate: "任务成功率为分子分母安全比率，小样本单元实施阈值抑制。",

    top_k_tasks: "Top-K 热门任务发现旨在准确保留前位核心场景，高频任务排序稳健。",

  };



  summary.forEach((item) => {

    const tr = document.createElement("tr");

    let ratingBadge = `<span class="badge badge-success">优秀</span>`;

    if (item.numeric_value !== null && item.numeric_value > 20.0) {

      ratingBadge = `<span class="badge badge-warning">良好</span>`;

    }



    tr.innerHTML = `

      <td><strong>${item.metric_name_zh}</strong> <code style="font-size:10px; color:var(--text-light);">(${item.metric_id})</code></td>

      <td>${item.primary_metric}</td>

      <td style="font-family: monospace; font-weight: 700; color: #be123c;">${item.value_display}</td>

      <td>${ratingBadge}</td>

      <td style="font-size: 11.5px; color: var(--text-muted);">${explanations[item.metric_id] || "-"}</td>

    `;

    tbody.appendChild(tr);

  });

}



function renderEpsilonSweepResults(sweepData) {

  const container = document.getElementById("epsilon-sweep-container");

  container.style.display = "block";



  const tbody = document.getElementById("epsilon-sweep-tbody");

  tbody.textContent = "";



  const ratings = {

    0.1: "极高隐私 / 噪声较大",

    0.25: "高隐私 / 适合监管合规",

    0.5: "较强隐私 / 强合规场景推荐",

    1.0: "推荐平衡 / 商业分析主流",

    2.0: "较宽松 / 内部高精度分析",

    4.0: "宽松上限 / 接近无噪精度",

  };



  (sweepData.points || []).forEach((pt) => {

    const tr = document.createElement("tr");

    const eps = pt.epsilon;

    const sigma = (1.0 / eps).toFixed(2);

    const evalText = ratings[eps] || "均衡";



    tr.innerHTML = `

      <td><strong>ε = ${eps}</strong></td>

      <td style="font-family: monospace;">σ ≈ ${sigma}</td>

      <td style="font-family: monospace; font-weight: 700; color: #be123c;">${pt.mean_error}</td>

      <td style="font-family: monospace;">±${pt.std_error}</td>

      <td style="font-family: monospace;">${pt.median_error}</td>

      <td style="font-family: monospace;">${pt.p90_error}</td>

      <td><span class="badge ${eps <= 0.5 ? 'badge-primary' : (eps <= 1.0 ? 'badge-success' : 'badge-warning')}">${evalText}</span></td>

    `;

    tbody.appendChild(tr);

  });

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

  document.getElementById("res-recipient").textContent = data.recipient_name || data.recipient;

  document.getElementById("res-recipient-id-tag").textContent = data.recipient_name ? `(${data.recipient})` : "";

  document.getElementById("res-eps-delta").textContent = `ε = ${data.epsilon}, δ = ${data.delta}`;

  document.getElementById("res-noise-scale").textContent = data.noise_scale;



  const cacheStatusEl = document.getElementById("res-cache-status");

  cacheStatusEl.textContent = data.cache_hit ? "标准重放（无额外隐私消耗）" : "新结果已发布";

  cacheStatusEl.className = data.cache_hit ? "badge badge-success" : "badge badge-primary";



  const effAfter = data.privacy_summary ? data.privacy_summary.effective_epsilon_after : null;

  const transEl = document.getElementById("res-lifetime-trans");

  if (effAfter !== null) {

    if (data.cache_hit) {

      transEl.textContent = `${effAfter.toFixed(4)} (未变)`;

    } else {

      transEl.textContent = `${previousEffectiveEps.toFixed(4)} → ${effAfter.toFixed(4)}`;

    }

  } else {

    transEl.textContent = "-";

  }



  const costEl = document.getElementById("res-incremental-cost");

  if (data.cache_hit) {

    costEl.textContent = "0.0 (Canonical Replay)";

  } else {

    costEl.textContent = `+${data.incremental_privacy_cost.toFixed(4)}`;

  }



  document.getElementById("res-release-id").textContent = data.release_id || data.release_key;



  applyResultFilter();

}



function applyResultFilter() {

  if (!lastReleaseData || !lastReleaseData.results) return;



  const tbody = document.getElementById("results-table-body");

  const thead = document.getElementById("results-table-head");

  const diagBox = document.getElementById("result-diagnostic-box");

  tbody.textContent = "";

  diagBox.style.display = "none";



  const results = lastReleaseData.results;

  const metric = lastReleaseData.metric;



  // Handle Top-K tasks

  if (metric === "top_k_tasks" || results.top_k) {

    thead.innerHTML = `

      <tr>

        <th>排名 (Rank)</th>

        <th>任务类别 (Task)</th>

        <th>有噪 DP 估计频次</th>

        <th title="参与率计算方式：该任务有噪估计参与人数 / 企业全量公开员工基数">任务估算参与率 (分母: 公开员工基数 ${results.public_population_size || 50000} 人)</th>

      </tr>

    `;



    const topKList = results.top_k || [];

    if (topKList.length === 0) {

      diagBox.style.display = "block";

      diagBox.textContent = "诊断信息：当前任务分类域内未发现满足阈值的 Top-K 任务。";

      return;

    }



    topKList.forEach((item) => {

      const tr = document.createElement("tr");

      tr.innerHTML = `

        <td><strong>#${item.rank}</strong></td>

        <td>${TASK_NAMES[item.task] || item.task} <code style="font-size:11px; color:var(--text-light);">(${item.task})</code></td>

        <td style="font-family: monospace; font-size: 14px;">${Number(item.noisy_count).toFixed(2)}</td>

        <td style="color: var(--text-light);">${item.estimated_prevalence !== null ? (Number(item.estimated_prevalence) * 100).toFixed(2) + "%" : "未指定公开分母"}</td>

      `;

      tbody.appendChild(tr);

    });

    return;

  }



  // Handle standard cell results

  thead.innerHTML = `

    <tr>

      <th>统计单元</th>

      <th>DP 估计值</th>

      <th>单位 / 状态</th>

    </tr>

  `;



  let cellEntries = [];

  let unit = "interactions";



  if (results.cell_volumes) {

    cellEntries = Object.entries(results.cell_volumes).map(([k, v]) => ({

      cell: k,

      value: Number(v).toFixed(2),

      unit: "interactions",

    }));

  } else if (results.cell_shares) {

    unit = "workflow share";

    cellEntries = Object.entries(results.cell_shares).map(([k, v]) => ({

      cell: k,

      value: (Number(v) * 100).toFixed(2) + "%",

      unit: "workflow share",

    }));

  } else if (results.prevalence_rates) {

    unit = "prevalence";

    cellEntries = Object.entries(results.prevalence_rates).map(([k, v]) => ({

      cell: k,

      value: (Number(v) * 100).toFixed(2) + "%",

      unit: "prevalence",

    }));

  } else if (results.success_rates || results.cell_success_rates) {

    unit = "success_rate";

    const rates = results.success_rates || results.cell_success_rates;

    const suppressed = new Set(results.suppressed_cells || []);

    cellEntries = Object.entries(rates).map(([k, v]) => ({

      cell: k,

      value: v !== null ? (Number(v) * 100).toFixed(2) + "%" : "受保护抑制 (Suppressed)",

      unit: "success_rate",

      isSuppressed: suppressed.has(k) || v === null,

    }));

  }



  // Diagnostic: check if raw results empty

  if (cellEntries.length === 0) {

    diagBox.style.display = "block";

    diagBox.textContent = "诊断原因 [NO RESULT FIELD]: 后端返回结果中未包含匹配当前统计指标的数据字段。";

    return;

  }



  // Filter by selected display providers and tasks (client-side only, zero privacy spend)

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

    diagBox.style.display = "block";

    if (selectedProviders.length === 0 || selectedTasks.length === 0) {

      diagBox.textContent = "诊断原因 [DISPLAY FILTER HID ALL]: 当前显示筛选未勾选任何 Provider 或任务类别。勾选上方复选框即可显示结果，不会重新计算或消耗隐私预算。";

    } else {

      diagBox.textContent = "诊断原因 [RECIPIENT FILTERED]: 当前接收方权限或所选筛选条件下没有可显示的统计单元。";

    }



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

    strong.textContent = formatCellName(item.cell);

    tdCell.appendChild(strong);

    if (item.cell.includes(":")) {

      const codeSpan = document.createElement("span");

      codeSpan.style.cssText = "font-size: 11px; color: var(--text-light); margin-left: 6px; font-family: monospace;";

      codeSpan.textContent = `[${item.cell}]`;

      tdCell.appendChild(codeSpan);

    }



    const tdVal = document.createElement("td");

    tdVal.style.fontFamily = "monospace";

    tdVal.style.fontSize = "14px";

    if (item.isSuppressed) {

      tdVal.style.color = "var(--warning)";

      tdVal.style.fontWeight = "600";

    }

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

    strongName.textContent = formatCellName(name);

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

// Global Download Handlers (for onclick bindings)

// -------------------------------------------------------------



function downloadReleaseCsv(relId, recipient) {

  triggerBlobDownload(`/api/release/download/csv?release_id=${encodeURIComponent(relId)}&recipient=${encodeURIComponent(recipient)}`, `private_llm_analytics_release_${relId}.csv`);

}



function downloadReleaseJson(relId, recipient) {

  triggerBlobDownload(`/api/release/download/json?release_id=${encodeURIComponent(relId)}&recipient=${encodeURIComponent(recipient)}`, `private_llm_analytics_release_${relId}.json`);

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

    errorType = err.type || err.error_type || "";

    if (!msg && !errorType) {

      msg = JSON.stringify(err);

    }

  }



  if (

    errorType === "PrivacyBudgetExceededError" ||

    msg.includes("budget exhausted") ||

    msg.includes("Privacy budget exhausted") ||

    msg.includes("Lifetime privacy cap exceeded")

  ) {

    return "当前全生命周期隐私预算不足，无法执行本次发布。";

  }

  if (

    errorType === "RecipientSecurityViolation" ||

    msg.includes("Recipient access violation") ||

    msg.includes("strictly forbidden")

  ) {

    return "接收方访问控制安全限制：" + msg;

  }

  if (msg.includes("No telemetry data loaded") || msg.includes("no telemetry")) {

    return "尚未加载遥测数据。请先加载演示数据或上传本地遥测数据文件。";

  }

  if (

    errorType === "ProductionSafetyConfigurationError" ||

    msg.includes("ProductionSafetyConfigurationError")

  ) {

    return "生产安全配置校验失败：" + msg;

  }

  if (msg.includes("Delta must be strictly between 0 and 1")) {

    return "隐私参数 δ 必须严格介于 0 和 1 之间。";

  }

  if (msg.includes("Epsilon must be a positive number") || msg.includes("Epsilon must be a finite positive")) {

    return "隐私参数 ε 必须为正数。";

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

