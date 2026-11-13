"""Authoritative DP Mechanism Cards for Web Console Explainability (v1.0.x / v1.1).

Provides rich, human-readable yet mathematically exact explanations for each of
the five approved statistic-specific differential privacy profiles:
1. User Prevalence (user_prevalence)
2. User-Normalized Share (user_normalized_share)
3. Interaction Volume (interaction_volume)
4. Success Rate (success_rate)
5. Long-Tail Top-K Tasks (top_k_tasks)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional

from private_llm_analytics.statistics.registry import StatisticProfileRegistry


@dataclass(frozen=True)
class MechanismCardSection:
    """A standardized section within a statistic's mechanism explanation card."""
    title: str
    content: str
    code_or_math: Optional[str] = None
    table: Optional[List[Dict[str, Any]]] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class MechanismCard:
    """Complete differential privacy mechanism explanation card for one statistic."""
    statistic_id: str
    display_title: str
    display_name_zh: str
    profile_name_zh: str
    privacy_unit: str
    adjacency: str
    target: str
    raw_example: Dict[str, Any]
    user_contribution: Dict[str, Any]
    contribution_bound: Dict[str, Any]
    sensitivity: Dict[str, Any]
    dp_mechanism: Dict[str, Any]
    post_processing: Dict[str, Any]
    result_meaning: Dict[str, Any]
    privacy_accounting: Dict[str, Any]
    utility_decomposition: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_mechanism_card(statistic_id: str, app_config: Optional[Any] = None) -> MechanismCard:
    """Retrieve the authoritative mechanism explanation card for a given statistic."""
    stat_id = statistic_id.lower().strip()

    # Extract dynamic configuration bounds where available
    c_volume = 20.0
    c_rate = 20.0
    min_noisy_den = 1.0
    k_prevalence = 9
    k_tasks = 5
    top_k = 5
    d_domain = 9
    n_tasks_taxonomy = 25

    if app_config is not None:
        if getattr(app_config, "taxonomy", None) is not None:
            n_prov = len(app_config.taxonomy.providers)
            n_task = len(app_config.taxonomy.tasks)
            d_domain = n_prov * n_task
            n_tasks_taxonomy = n_task
        elif getattr(app_config, "scenario", None) is not None:
            n_prov = len(app_config.scenario.providers)
            n_task = len(app_config.scenario.tasks)
            d_domain = n_prov * n_task
            n_tasks_taxonomy = n_task

        metrics_cfg = getattr(app_config, "metrics", None)
        if metrics_cfg and hasattr(metrics_cfg, "metrics"):
            m = metrics_cfg.metrics.get("interaction_volume")
            if m and getattr(m, "contribution", None) and getattr(m.contribution, "clip_norm", None):
                c_volume = float(m.contribution.clip_norm)
            m_sr = metrics_cfg.metrics.get("success_rate")
            if m_sr and getattr(m_sr, "contribution", None):
                c_rate = float(getattr(m_sr.contribution, "clip_norm_rate", 20.0) or 20.0)
                min_noisy_den = float(getattr(m_sr.contribution, "min_noisy_denominator", 1.0) or 1.0)
            m_prev = metrics_cfg.metrics.get("user_prevalence")
            if m_prev and getattr(m_prev, "contribution", None) and getattr(m_prev.contribution, "max_cells_per_user", None):
                k_prevalence = int(m_prev.contribution.max_cells_per_user)
            m_tk = metrics_cfg.metrics.get("top_k_tasks")
            if m_tk and getattr(m_tk, "contribution", None):
                k_tasks = int(getattr(m_tk.contribution, "max_tasks_per_user", 5) or 5)
                top_k = int(getattr(m_tk.contribution, "top_k", 5) or 5)

    if stat_id == "user_prevalence":
        tight_bound = math.sqrt(min(k_prevalence, d_domain))
        conservative_bound = math.sqrt(k_prevalence)
        return MechanismCard(
            statistic_id="user_prevalence",
            display_title="用户使用率 / User Prevalence",
            display_name_zh="用户使用率",
            profile_name_zh="用户参与二值化 + 高斯直方图机制",
            privacy_unit="用户（完整用户遥测轨迹）",
            adjacency="增加/删除一个用户（Add/Remove One User）",
            target="统计在当前时间窗口内，公开员工名册中有多少比例的用户至少使用过某个 Provider × Task 统计单元。",
            raw_example={
                "note": "示例数据，仅用于解释机制",
                "columns": ["user", "provider", "task", "interactions"],
                "rows": [
                    {"user": "u001", "provider": "provider_a", "task": "technical", "interactions": 20},
                    {"user": "u001", "provider": "provider_a", "task": "document", "interactions": 3},
                    {"user": "u002", "provider": "provider_a", "task": "technical", "interactions": 0},
                    {"user": "u003", "provider": "provider_a", "task": "technical", "interactions": 5},
                ],
            },
            user_contribution={
                "formula": r"b_{u,j} = \mathbf{1}[m_{u,j} > 0] \in \{0, 1\}^D",
                "explanation": "将每个用户的原始交互记录汇总转换为长度为 D 的二值参与指示向量。只要用户在该单元中发生过至少 1 次交互，对应位置记为 1，否则记为 0。",
                "numeric_example": "用户 u001 在 technical 和 document 有交互，对应向量为 (1, 1, 0, ...)；用户 u002 无交互，对应向量为 (0, 0, 0, ...)。",
            },
            contribution_bound={
                "formula": r"\|b_u\|_0 \le \min(k, D)",
                "configured_k": k_prevalence,
                "dimension_D": d_domain,
                "explanation": f"单用户最多计入 k={k_prevalence} 个单元的有效使用。若用户在更多单元活跃，则截断/子抽样至最多 k 个单元，以限制单用户对全局分布的最大影响。",
            },
            sensitivity={
                "formula": r"\Delta_2 = \sqrt{\min(k, D)} \le \sqrt{k}",
                "conservative_bound": f"Delta_2 <= sqrt(k) = {conservative_bound:.4f}",
                "tight_bound": f"Delta_2 = sqrt(min({k_prevalence}, {d_domain})) = {tight_bound:.4f}",
                "rate_sensitivity": r"\Delta_2 / N_{\text{public}} = \sqrt{k} / N_{\text{public}}",
                "coordinate_sensitivity": r"1 / N_{\text{public}}",
                "explanation": f"由于每个分量均为 0 或 1，L2 范数等于参与单元数开平方。生产实现采用保守上界 Delta_2 = sqrt({k_prevalence}) = {conservative_bound:.4f}；在已配置 D 单元的有限域中紧界为 {tight_bound:.4f}。除以公开基准人数 N_public 后获得使用率。",
            },
            dp_mechanism={
                "equation": r"\tilde{S} = S + \mathcal{N}(0, \sigma^2 I), \quad S = \sum_u b_u",
                "explanation": "首先对所有用户的二值参与向量求和得到各单元总参与人数 S，随后向求和向量添加高斯噪声 Z ~ N(0, sigma^2 I)。",
            },
            post_processing={
                "steps": [
                    r"\hat S_j = \max(0, \tilde S_j)",
                    r"\hat p_j = \operatorname{clip}\left(\frac{\hat S_j}{N_{\text{public}}}, 0, 1\right)",
                ],
                "explanation": "后处理首先将噪声参与人数截断至非负数（>= 0），随后严格除以固定的公开员工基准人数 N_public，并将使用率严格截断在 [0, 1] 区间。严禁使用私有活跃用户数作为外部除数分母。",
            },
            result_meaning={
                "example": "provider_a × technical: DP 用户使用率 = 37.8%",
                "interpretation": "在公开员工总体中，经过严格的用户级差分隐私保护后，估计有约 37.8% 的员工使用过该模型能力。",
                "units": "百分比（%）",
            },
            privacy_accounting={
                "mechanism": "高斯直方图机制（Analytic Gaussian Mechanism）",
                "accounting": "RDP（Rényi 差分隐私组合）",
                "explanation": "每次发布生成一个关于阶数 alpha 的 RDP 损失向量，并累加到持久化 SQLite 账本中。对相同语义参数发起的重复查询（Canonical Replay）命中缓存，新增隐私消耗严格为 0。",
            },
        )

    elif stat_id == "user_normalized_share":
        return MechanismCard(
            statistic_id="user_normalized_share",
            display_title="用户归一化占比 / User-Normalized Share",
            display_name_zh="用户归一化占比",
            profile_name_zh="用户归一化贡献 + 高斯向量机制 + 单纯形归一化",
            privacy_unit="用户（完整用户遥测轨迹）",
            adjacency="增加/删除一个用户（Add/Remove One User）",
            target="统计一个典型用户的 LLM 工作流如何分配在不同 Provider × Task 类别上，而不是让重度用户完全主导总量。",
            raw_example={
                "note": "示例数据，仅用于解释机制",
                "columns": ["task", "raw_interactions", "normalized_share"],
                "rows": [
                    {"task": "technical (技术类)", "raw_interactions": 60, "normalized_share": "0.60 (60%)"},
                    {"task": "document (文档类)", "raw_interactions": 30, "normalized_share": "0.30 (30%)"},
                    {"task": "analysis (分析类)", "raw_interactions": 10, "normalized_share": "0.10 (10%)"},
                ],
            },
            user_contribution={
                "formula": r"x_u = \frac{m_u}{\|m_u\|_1} \in \Delta^{D-1}",
                "explanation": "将每个用户的多维交互向量 m_u 按自身 L1 范数进行归一化。若用户总交互为 0，则贡献全 0 向量。归一化后每个用户的总权重严格为 1.0，体现典型工作流分配。",
                "numeric_example": "用户交互 (60, 30, 10)，其 L1 范数为 100，归一化贡献向量为 (0.6, 0.3, 0.1)。",
            },
            contribution_bound={
                "formula": r"\|x_u\|_1 \le 1.0, \quad \|x_u\|_2 \le \|x_u\|_1 \le 1.0",
                "explanation": "由于用户工作流向量被限制在概率单纯形上，各分量非负且和为 1，因此该向量的 L2 范数恒不超过 1.0。",
            },
            sensitivity={
                "formula": r"\Delta_2 = 1.0",
                "configured_value": 1.0,
                "explanation": "任意单个用户的加入或退出，对全量累加和向量 V 的 L2 距离改变最大为 1.0，敏感度精确且保守地校准为 Delta_2 = 1.0。",
            },
            dp_mechanism={
                "equation": r"\tilde V = V + \mathcal{N}(0, \sigma^2 I), \quad V = \sum_u x_u",
                "explanation": "将所有活跃用户的归一化向量进行向量求和得到 V，并向求和向量添加方差为 sigma^2 的各向同性高斯噪声。",
            },
            post_processing={
                "steps": [
                    r"y_j = \max(0, \tilde V_j)",
                    r"\hat s_j = \frac{y_j}{\sum_{i=1}^D y_i}",
                    r"\text{若 } \sum_i y_i = 0, \quad \hat s_j = \frac{1}{D}",
                ],
                "explanation": "采用“非负截断 + 单纯形重新归一化”（nonnegative clamping + simplex renormalization）。注意：本机制并非欧氏单纯形投影，而是将所有负值坐标置 0 后按坐标和重新归一化。若分母全为 0，则平滑回退至 1/D 均匀分布。",
            },
            result_meaning={
                "example": "technical: 45.2%, document: 34.1%, analysis: 20.7%",
                "interpretation": "典型员工日常 LLM 使用中，约 45.2% 的精力分配在技术类任务，34.1% 分配在文档处理。",
                "units": "百分比（工作流占比 %）",
            },
            privacy_accounting={
                "mechanism": "高斯向量机制（Gaussian Vector Mechanism）",
                "accounting": "RDP（Rényi 差分隐私组合）",
                "explanation": "使用 Delta_2 = 1.0 与配置的 (epsilon, delta) 计算高斯噪声尺度 sigma 并记账。相同查询重放 0 隐私成本。",
            },
        )

    elif stat_id == "interaction_volume":
        c_val = c_volume
        return MechanismCard(
            statistic_id="interaction_volume",
            display_title="交互量 / Interaction Volume",
            display_name_zh="交互量",
            profile_name_zh="用户级 L2 裁剪 + 高斯向量机制",
            privacy_unit="用户（完整用户遥测轨迹）",
            adjacency="增加/删除一个用户（Add/Remove One User）",
            target="统计不同 Provider × Task 单元中的总交互规模。",
            raw_example={
                "note": "示例数据，仅用于解释机制",
                "columns": ["provider_task", "raw_count", "clipped_count"],
                "rows": [
                    {"provider_task": "provider_a:technical", "raw_count": 30, "clipped_count": 12.0},
                    {"provider_task": "provider_a:document", "raw_count": 40, "clipped_count": 16.0},
                    {"provider_task": "provider_a:analysis", "raw_count": 0, "clipped_count": 0.0},
                ],
            },
            user_contribution={
                "formula": r"\bar m_u = m_u \min\left(1, \frac{C}{\|m_u\|_2}\right)",
                "explanation": "统计每位用户在各单元的原始交互次数向量 m_u。当用户的 L2 范数超过阈值 C 时，按比例缩小该向量，保持各方向比例不变。",
                "numeric_example": f"设 C={c_val:.1f}。用户原始向量 (30, 40, 0) 的 L2 范数为 50，裁剪因子为 20/50 = 0.4，裁剪后贡献为 (12, 16, 0)。",
            },
            contribution_bound={
                "formula": r"\|\bar m_u\|_2 \le C",
                "configured_C": c_val,
                "explanation": f"经过用户级 L2 裁剪后，任意单个用户的贡献向量的 L2 范数严格小于或等于 C={c_val:.1f}。",
            },
            sensitivity={
                "formula": r"\Delta_2 = C",
                "configured_value": c_val,
                "explanation": f"增加或删除一个用户，全量累加和向量的最大 L2 变化量恰好为 C={c_val:.1f}。",
            },
            dp_mechanism={
                "equation": r"\tilde M = M + \mathcal{N}(0, \sigma^2 I), \quad M = \sum_u \bar m_u",
                "explanation": "汇总所有用户裁剪后的贡献向量 M，并加入尺度为 sigma 的高斯噪声向量。",
            },
            post_processing={
                "steps": [
                    r"\hat M_j = \max(0, \tilde M_j)",
                ],
                "explanation": "将加噪后的估计值截断至非负数（>= 0），因为交互总量在物理现实中不能为负数。",
            },
            result_meaning={
                "example": "provider_a × technical: 10,887 次交互",
                "interpretation": "在指定观测窗口内，该模型与任务单元累计发生约 10,887 次交互。",
                "units": "次交互（interactions）",
            },
            privacy_accounting={
                "mechanism": "高斯向量机制（Gaussian Vector Mechanism）",
                "accounting": "RDP（支持二叉树持续观测与小样本效用规划）",
                "explanation": "以敏感度 Delta_2 = C 计算 RDP 成本并写入 SQLite 账本。支持二叉时间树（Binary Tree）跨窗口复用与零成本 Canonical Replay。",
            },
            utility_decomposition={
                "explanation": "面向可信内部管理员的效用误差分解体系：",
                "definitions": [
                    {"term": "原始精确值 (Raw Exact)", "symbol": "M_raw", "meaning": "未经裁剪的真实统计值"},
                    {"term": "裁剪精确值 (Clipped Exact)", "symbol": "M_clip", "meaning": "施加用户贡献上限 C 后的确定性统计值"},
                    {"term": "DP 发布值 (DP Output)", "symbol": "M_DP", "meaning": "注入高斯差分隐私噪声后的最终发布值"},
                    {"term": "贡献截断偏差 (Clipping Bias)", "symbol": "M_clip - M_raw", "meaning": "因防止单用户主导而导致的结构性偏差"},
                    {"term": "DP 噪声误差 (DP Noise Error)", "symbol": "M_DP - M_clip", "meaning": "为满足差分隐私注入的高斯随机扰动"},
                    {"term": "总误差 (Total Error)", "symbol": "M_DP - M_raw", "meaning": "最终发布值与真实值之间的总体差异"},
                ],
            },
        )

    elif stat_id == "success_rate":
        c_r = c_rate
        joint_sens = math.sqrt(2) * c_r
        return MechanismCard(
            statistic_id="success_rate",
            display_title="成功率 / Success Rate",
            display_name_zh="成功率",
            profile_name_zh="共享用户级裁剪 + 成功数/总数联合高斯机制 + 比率后处理",
            privacy_unit="用户（完整用户遥测轨迹）",
            adjacency="增加/删除一个用户（Add/Remove One User）",
            target="统计每个 Provider × Task 单元中成功交互占总交互的比例。",
            raw_example={
                "note": "示例数据，仅用于解释机制",
                "columns": ["user", "task", "total_interactions (n_u)", "success_interactions (s_u)"],
                "rows": [
                    {"user": "u001", "task": "technical", "total_interactions (n_u)": 30, "success_interactions (s_u)": 24},
                    {"user": "u001", "task": "document", "total_interactions (n_u)": 40, "success_interactions (s_u)": 32},
                ],
            },
            user_contribution={
                "formula": r"a_u = \min\left(1, \frac{C_{\text{rate}}}{\|n_u\|_2}\right), \quad \bar n_u = a_u n_u, \quad \bar s_u = a_u s_u",
                "explanation": "每位用户拥有总交互数向量 n_u 与成功交互数向量 s_u（恒满足 0 <= s_{u,j} <= n_{u,j}）。采用共享缩放因子 a_u 统一对成功数与总数进行等比裁剪，在限制重度用户影响的同时严格保留用户的内部成功率比例不变。",
                "numeric_example": f"设 C_rate={c_r:.1f}。用户总数向量 (30, 40) 的 L2 范数为 50，成功向量为 (24, 32)。共享因子 a_u = 20/50 = 0.4。裁剪后总数 nbar_u = (12, 16)，成功数 sbar_u = (9.6, 12.8)，各单元内部成功率依然精确保持 80%。",
            },
            contribution_bound={
                "formula": r"w_u = \begin{bmatrix} \bar s_u \\ \bar n_u \end{bmatrix}, \quad \|w_u\|_2 \le \sqrt{2} C_{\text{rate}}",
                "configured_C_rate": c_r,
                "explanation": f"将裁剪后的成功向量与总数向量拼接为联合向量 w_u = [\bar s_u ; \bar n_u]。由于 ||sbar_u||_2 <= ||nbar_u||_2 <= C_rate，拼接向量的 L2 范数平方 <= 2 * C_rate^2。",
            },
            sensitivity={
                "formula": r"\Delta_2 \le \sqrt{2} C_{\text{rate}}",
                "configured_value": round(joint_sens, 4),
                "explanation": f"联合向量的总体 L2 敏感度严格校准为 Delta_2 <= sqrt(2) * C_rate = sqrt(2) * {c_r:.1f} ≈ {joint_sens:.4f}。",
            },
            dp_mechanism={
                "equation": r"\begin{bmatrix} \tilde S \\ \tilde N \end{bmatrix} = \begin{bmatrix} S \\ N \end{bmatrix} + Z, \quad Z \sim \mathcal{N}(0, \sigma^2 I)",
                "explanation": "核心设计：采用单次联合高斯机制（Single Joint Gaussian Mechanism），一次性向 [S; N] 注入高斯噪声。严格产生单次 RDP 隐私消耗事件，严禁拆分为两次独立的全预算发布。",
            },
            post_processing={
                "steps": [
                    r"\hat N_j = \max(\tilde N_j, 0)",
                    r"\hat S_j = \operatorname{clip}(\tilde S_j, 0, \hat N_j)",
                    r"\text{若 } \hat N_j < \tau, \quad \text{返回 } \text{SUPPRESSED / null (抑制发布)}",
                    r"\text{否则 } \hat r_j = \frac{\hat S_j}{\hat N_j}",
                ],
                "explanation": rf"后处理步骤：1. 分母截断至非负数 \hat N_j = \max(\tilde N_j, 0)；2. 分子截断至 [0, \hat N_j]，确保物理合理性；3. 安全阈值保护：若估计总数 \hat N_j < \tau ({min_noisy_den:.1f})，则直接返回 SUPPRESSED / null（受保护抑制），防止小样本噪声放大；4. 否则计算比率 \hat r_j = \hat S_j / \hat N_j。",
            },
            result_meaning={
                "example": "provider_a × technical: 成功率 = 92.4% (或样本不足时显示“受保护抑制”)",
                "interpretation": "在经差分隐私保护后，该单元中成功完成的大模型交互比例约为 92.4%。",
                "units": "百分比（成功率 %）",
            },
            privacy_accounting={
                "mechanism": "联合高斯机制（Joint Gaussian Vector Mechanism）",
                "accounting": "单次 RDP 记账（Single RDP Charging）",
                "explanation": f"以敏感度 Delta_2 = sqrt(2) * C_rate = {joint_sens:.4f} 进行单次 RDP 记账。Canonical Replay 重放 0 隐私成本。",
            },
        )

    elif stat_id == "top_k_tasks":
        tight_tk = math.sqrt(min(k_tasks, n_tasks_taxonomy))
        conservative_tk = math.sqrt(k_tasks)
        return MechanismCard(
            statistic_id="top_k_tasks",
            display_title="热门任务 Top-K / Long-Tail Top-K Tasks",
            display_name_zh="热门任务 Top-K",
            profile_name_zh="任务参与约束 + 高斯向量机制 + 差分隐私 Top-K 选择",
            privacy_unit="用户（完整用户遥测轨迹）",
            adjacency="增加/删除一个用户（Add/Remove One User）",
            target="在当前配置的有限任务分类体系（Taxonomy）中，寻找用户覆盖率最高的前 K 个任务。",
            raw_example={
                "note": "示例数据，仅用于解释机制",
                "columns": ["task_id", "task_name", "user_active"],
                "rows": [
                    {"task_id": "t_01", "task_name": "代码生成与审查", "user_active": "参与 (1)"},
                    {"task_id": "t_02", "task_name": "多语言文档翻译", "user_active": "参与 (1)"},
                    {"task_id": "t_03", "task_name": "数据清洗与分析", "user_active": "参与 (1)"},
                    {"task_id": "t_04", "task_name": "营销文案润色", "user_active": "未参与 (0)"},
                ],
            },
            user_contribution={
                "formula": r"v_{u,t} = \mathbf{1}[\text{用户使用过任务 } t] \in \{0, 1\}^{|T|}",
                "explanation": f"候选任务空间为当前系统配置的有限任务 Taxonomy (大小为 |T|={n_tasks_taxonomy})，并非开放/无界字符串宇宙。每位用户产生一个表示任务使用情况的二值向量 v_u。",
            },
            contribution_bound={
                "formula": r"\|v_u\|_0 \le \min(k_{\text{tasks}}, |T|)",
                "configured_k_tasks": k_tasks,
                "taxonomy_size": n_tasks_taxonomy,
                "explanation": f"单用户最多计入 k_tasks={k_tasks} 个任务的活跃状态。超出者通过确定性截断或子抽样限制在 k_tasks 以内。",
            },
            sensitivity={
                "formula": r"\Delta_2 = \sqrt{\min(k_{\text{tasks}}, |T|)} \le \sqrt{k_{\text{tasks}}}",
                "conservative_bound": f"Delta_2 <= sqrt(k_tasks) = {conservative_tk:.4f}",
                "tight_bound": f"Delta_2 = sqrt(min({k_tasks}, {n_tasks_taxonomy})) = {tight_tk:.4f}",
                "explanation": f"二值稀疏向量的 L2 敏感度受限于活跃任务数。生产环境采用保守敏感度 Delta_2 = sqrt({k_tasks}) = {conservative_tk:.4f}，在有限域中紧界为 {tight_tk:.4f}。",
            },
            dp_mechanism={
                "equation": r"\tilde H = H + \mathcal{N}(0, \sigma^2 I), \quad H = \sum_u v_u",
                "explanation": "首先在配置的任务集合上对所有用户的参与向量求和得到全局任务频次直方图 H，随后向全部任务直方图添加高斯噪声向量产生有噪直方图 H_tilde。",
            },
            post_processing={
                "steps": [
                    r"\hat H_t = \max(0, \tilde H_t)",
                    r"\text{按 } \hat H_t \text{ 降序排序，选取前 } K \text{ 个任务}",
                ],
                "explanation": "Top-K 排序截断为纯后处理（Post-Processing）。排序和筛选完全在加噪后的 DP 估计值上进行，严禁使用任何未经加噪的内部精确排序或频次。",
            },
            result_meaning={
                "example": "Top-1: 代码生成 (估计用户数: 1,420), Top-2: 文档翻译 (估计用户数: 980)",
                "interpretation": "在企业有限任务体系中，经过差分隐私保护后识别出员工使用人数最多的前 K 个核心应用场景。",
                "units": "排名与估计参与人数",
            },
            privacy_accounting={
                "mechanism": "高斯直方图机制 + 差分隐私排序选择",
                "accounting": "RDP（单次高斯直方图记账）",
                "explanation": "依据敏感度 Delta_2 对全量直方图加噪进行单次 RDP 记账。根据差分隐私后处理不变性，后续的 Top-K 提取不产生任何额外隐私消耗。",
            },
        )

    else:
        raise ValueError(f"Unknown statistic ID for mechanism card: '{statistic_id}'")


def get_all_mechanism_cards(app_config: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all 5 mechanism cards formatted for JSON/Web consumption."""
    stats = [
        "user_prevalence",
        "user_normalized_share",
        "interaction_volume",
        "success_rate",
        "top_k_tasks",
    ]
    cards = {}
    for sid in stats:
        cards[sid] = get_mechanism_card(sid, app_config=app_config).to_dict()
    return cards
