# PVLDB 2027 论文写作蓝图：标题、贡献、大纲、定理/图表映射与相关工作检索清单

> **范围说明**：本文件基于截至提交 `v3` 最终一致性复核（`results/sdg_mh/long_horizon_420d_v3_final/`）的全部实测与形式化证书结果整理，仅映射已验证结论与真实数据，不新增研究机制或实验结论。

---

## 一、拟定论文标题（候选）

- **主标题（推荐）**：
  *No Epochs, No Expiry: Multi-Horizon User-Time Differential Privacy and Unified Calendar Planning for Perpetual Analytics*
- **备选副标题**：
  *Perpetual Multi-Horizon User-Time Differential Privacy: Combining Daily Gaussian Sealing with Phase-Joint Calendar Planning*

---

## 二、两大核心技术贡献（严格收敛主线）

1. **贡献 1：Multi-Horizon User-Time Differential Privacy 与每日 Gaussian 连续发布理论刻画**
   - 提出 `(u, I)`-邻接（User-Time Interval Adjacency）多时间尺度隐私承诺曲线 $\bar\rho(h)$（同时覆盖 $h \in \{1, 7, 30, 90, 365\}$ 离散锚点与 $h \ge 365$ 仿射线性尾部 $\bar\rho(h) = a_{\text{tail}} + \beta_{\text{tail}} h$），消除固定 Epoch 重置漏洞及 $1/t^2$ 终身衰减导致的渐近效用坍塌。
   - 证明在静态加性日查询工作负载下，每日独立离散 Gaussian 封存 $\mathcal{N}_{\mathbb{Z}}(0, \sigma_t^2)$ 兼具无边界平稳效用、各向同性矢量分配最优化（$\eta_k = C_k^2 / \sum_j C_j^2$）与优于传统 Binary Tree（`Tree-2l` / `Tree-U` + Honaker GLS）及 $w$-event (`W-BD`) 的全跨度滚动精度（在 $[1, 420]$ 相同包络下）。
2. **贡献 2：统一 zCDP 账本下多尺度可加与窗口级非可加（Distinct）统计的 Calendar 联合规划与有限到无限安全认证**
   - 建立精确数据依赖窗口相交计数公式 $n_f(s, h; \phi_f)$（纠正仅按发布日期落入 $I$ 计费的 $5\times$ 低估缺陷），实现用户跨多日窗口级 $k$-任务去重直方图裁剪（$\Delta^2 = m + 2k$）。
   - 证明联合日历 210 天周期平移恒等式 $L_{\max}(h + P) = L_{\max}(h) + P \cdot \text{slope}$；在实证发现 `Original_Planner` 与 `Week1_v2` 曲线均不满足全局凹性后，提出仅依赖 $h \ge H_0 = 365$ 仿射尾部的 **575 天有限ENUM 到无限时域归纳归约定理**。
   - 构造混合离散-连续凸优化（210 规范相位枚举 $\times$ SLSQP 切平面）+ **100 位 `Decimal` 有向舍入 Lagrangian 对偶证书（`DECIMAL100_DIRECTED_ROUNDING`）** + 严格零容差账本提交闭环（区分 `SOLVER_CONVERGED` / `FEASIBLE_CERTIFIED` / `NEAR_GLOBAL_OPTIMAL_CERTIFIED` / `GLOBAL_OPTIMAL_PROVED`）。

---

## 三、PVLDB 论文完整章节大纲

1. **Section 1: Introduction**
   - 长期持续企业分析（无 Epoch 清零、无期限边界）与混合负载矛盾（每日加性计数 vs 7d/30d/90d 去重活跃统计）。
   - 现有范式局限：`Epoch` 重置/拒服、`LifetimeDecay` 方差爆炸、Binary Tree 仅支持可加且受线性尾部约束放大短程噪声、$w$-event 无法跨 $h \in [1, 420]$ 满足多尺度包络。
   - 本文方法与实测定位概览。
2. **Section 2: Problem Formulation & Multi-Horizon User-Time Privacy**
   - `(u, I)`-邻接与 Multi-Horizon zCDP 承诺曲线 $\bar\rho(h)$（非全局凹但仿射线性尾部）。
   - 与 $w$-event DP、User-level DP、Binary Tree 有限时域证书的形式化关系与语义边界。
3. **Section 3: Unified Privacy Ledger & Perpetual Daily Sealing**
   - 单进程事务原子批准入（`admit_and_commit_batch`、`strict_zero_tol=True`、`conservative_epsilon_drift_bound` $\Delta\varepsilon \le 3.22 \times 10^{-11}$）。
   - Canonne–Kamath–Steinke 离散 Gaussian 加噪与只读后处理查询层（窗口先求和后裁剪 + `Failures <= Requests` 一致性）。
4. **Section 4: Periodic Distinct Histograms & Exact Calendar Accounting**
   - 窗口级 $k$-任务裁剪（`RollingDistinctState`）与敏感度推导 $\Delta^2 = m + 2k$。
   - 数据窗口相交定理（Theorem 1：$L = W_f + h - 1$）与 $5\times$ 反例。
   - 210 天周期平移与无凹性 $h \ge 365$ 仿射尾部 575 天归约定理（Theorems 2 & 3）。
5. **Section 5: Joint Phase-Budget Optimization & Verifiable Optimality Certification**
   - 可分离凸目标 $\sum c_j / v_j$ 与保守缩放（`CONSERVATIVE_BACKOFF_REL = 1e-11`）。
   - 210 相位枚举 + `Decimal100` 有向舍入弱对偶下界定理（Theorem 4）与四级可发布认证层级。
6. **Section 6: Experimental Evaluation**
   - 端到端实验设计（2,000 用户 $\times$ 420 天 $\times$ 8 独立数据种子 $\times$ 5 CRN MC = 40 次/配置，$df=7$；保留 90 天与 $C_r$ 消融对照）。
   - Exp 1：420 天跨周期安全轨迹检验（$t=210, 365, 420$ 严格正余量零违规）。
   - Exp 2：`BudgetOnly` vs `JointPlanning` 同查询日期（发布日 / 每日仪表盘 / 固定周日）配对统计对比与负向权衡披露。
   - Exp 3：$C_r \in \{30, 50, 100, 200\}$ 裁剪偏置 vs 噪声 Pareto 拐点消融。
   - Exp 4：相同 `[1, 420]` 加性包络下与 `Tree-2l`、`Tree-U`、`W-BD-30` 公平对比（统一后处理与非可比性声明）。
7. **Section 7: Related Work**
8. **Section 8: Limitations, Threat Model & Conclusion**

---

## 四、定理 / 性质 与 图表 / 代码 / 证据映射表

| 论文编号 | 核心内容 | 形式状态 | 对应源码与检验函数 | 对应图表 / 数据产出 |
|---|---|---|---|---|
| **Def 1 & Curve Props** | `(u,I)` 邻接及 `Week1_v2` / `Original` 非全局凹验证 | `MATHEMATICALLY_PROVED` | `calendar_accounting.py`: `PrivacyCommitmentCurve` | [PVLDB_2027_THEORY_APPENDIX_AND_LIMITATIONS.md](file:///f:/Private%20LLM%20Analytics/docs/reports/PVLDB_2027_THEORY_APPENDIX_AND_LIMITATIONS.md) Appendix B.1 |
| **Theorem 1** | 数据依赖窗口相交闭式计数 $n_f(s,h;\phi_f)$（$L=W_f+h-1$）与 $5\times$ 反例 | `MATHEMATICALLY_PROVED` | `calendar_accounting.py`: `count_intersections_fast()` / `test_calendar_accounting.py` | Appendix A.1–A.2 |
| **Theorem 2 & 3** | 210 天周期平移恒等式与无凹性 575 天有限 $\to$ 无限仿射尾部归约 | `MATHEMATICALLY_PROVED`（限静态周期日历） | `calendar_accounting.py`: `verify_periodicity_and_long_term_security()` | 420d Ledger 轨迹表 + `fig1_420d_cross_period_ledger_trajectory.png` |
| **Theorem 4 & Lemma** | `Decimal100` 有向舍入 Lagrangian 对偶下界 + `1e-12` 账本漂移上界（$\le 3.22\times 10^{-11}$） | `NUMERICAL_CERTIFIED` (`rel_gap <= 7.45e-9`) | `joint_planner.py`: `certify_lagrangian_dual_bound_exact()`, `ledger.py`: `conservative_epsilon_drift_bound()` | [solver_runtime_and_dual_bounds.csv](file:///f:/Private%20LLM%20Analytics/results/sdg_mh/long_horizon_420d_v3_final/solver_runtime_and_dual_bounds.csv) + 取消反例测试 |
| **Table 1 / Fig 1 (Paper)** | 同查询日期下 `D_JointPlanning` vs `B_BudgetOnly`（发布日 / 每日 / 周日陈旧度与 Holm/Wilcoxon $p$） | `EMPIRICAL_VALIDATED` ($K=8$, $df=7$) | [run_long_horizon_420d_audit_v3.py](file:///f:/Private%20LLM%20Analytics/experiments/sdg_mh/run_long_horizon_420d_audit_v3.py) | [audit_v3_paired_with_holm_wilcoxon.csv](file:///f:/Private%20LLM%20Analytics/results/sdg_mh/long_horizon_420d_v3_final/audit_v3_paired_with_holm_wilcoxon.csv) & [fig1_v3_same_query_date_d_vs_b.png](file:///f:/Private%20LLM%20Analytics/results/sdg_mh/long_horizon_420d_v3_final/figures/fig1_v3_same_query_date_d_vs_b.png) |
| **Table 2 / Fig 2 (Paper)** | 统一后处理下 `SDG-MH` vs `Tree-2l` / `Tree-U` / `W-BD-30`（限定 `[1,420]` 加性任务） | `EMPIRICAL_VALIDATED` ($K=8$, $df=7$) | `baselines.py` + `run_long_horizon_420d_audit_v3.py` | [audit_v3_competitive_baselines_summary.csv](file:///f:/Private%20LLM%20Analytics/results/sdg_mh/long_horizon_420d_v3_final/audit_v3_competitive_baselines_summary.csv) & [fig2_v3_uniform_postproc_baselines.png](file:///f:/Private%20LLM%20Analytics/results/sdg_mh/long_horizon_420d_v3_final/figures/fig2_v3_uniform_postproc_baselines.png) |
| **Table 3 / Fig 3 (Paper)** | $C_r \in \{30,50,100,200\}$ 偏置-噪声 Pareto 消融（冻结 420d 消融结果） | `EMPIRICAL_VALIDATED` | [run_long_horizon_420d_and_ablation.py](file:///f:/Private%20LLM%20Analytics/experiments/sdg_mh/run_long_horizon_420d_and_ablation.py) | [cr_clipping_ablation_420d.csv](file:///f:/Private%20LLM%20Analytics/results/sdg_mh/long_horizon_420d/cr_clipping_ablation_420d.csv) & [fig2_cr_clipping_pareto_ablation.png](file:///f:/Private%20LLM%20Analytics/results/sdg_mh/long_horizon_420d/figures/fig2_cr_clipping_pareto_ablation.png) |

---

## 五、相关工作精准检索与对比核查任务（供写作轮直接执行）

1. **Continual Counting & Binary/Matrix Factorization Mechanisms**
   - 核查引文与准确边界：Chan–Shi–Song (TISSEC 2011) / Dwork et al. (STOC 2010) 二叉树计数；Honaker (2015) GLS 最优重构；Denisov–McMahan–Rush–Smith–Thakurta (NeurIPS 2022) / Fichtenberger–Henzinger–Upadhyay (ICML 2023) Matrix Mechanism / Toeplitz-banded factorization 在**有限流 vs 无限多尺度线性尾部**下与 SDG-MH 的方差交叉点。
2. **$w$-Event & Time-Decaying Differential Privacy on Streams**
   - 核查引文与对比点：Kellaris–Papadopoulos–Xiao–Papadias (PVLDB 2014) `W-BD` / `W-BA`；Cummings–et al. / Perrier–et al. 用户时间区间邻接与滑动窗口；Wang et al. (SIGMOD/PVLDB) 周期流隐私；重点核对单窗口 $w$ 预算在长程叠加下如何突破多尺度包络。
3. **User-Level & Pan-Private / Distinct-Count Streaming DP**
   - 核查引文与对比点：Epasto et al. / Amin et al. (NeurIPS 2019) 多贡献用户裁剪偏置-方差权衡（对比 $C_r$ 消融）；Mir–Muthukrishnan–Nikolov–Wright (PODS 2011) / Card–et al. Pan-private distinct count / Flajolet-Martin 对比本文严格有界 $k$-任务滚动去重精确机制。
4. **LLM / AI Telemetry Privacy-Preserving Analytics Systems**
   - 核查近期工业界与数据库系统文献（WildChat / ShareGPT 遥测隐私分析、企业级交互审计平台中的多维可加+Top-K 任务监控架构），明确本文作为长周期混合工作负载隐私调度系统的定位。
