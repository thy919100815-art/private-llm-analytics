# PVLDB 2027 论文理论附录、实验说明与局限性声明
## Theoretical Appendix, Long-Horizon 420d Experimental Protocol & Explicit Limitations

---

## Appendix A: 数据依赖窗口相交计数公式与闭式证明

### A.1 定义与等价充要条件
设某定期日历发布族 $f$ 具有统计窗口长度 $W_f \in \mathbb{N}^+$、发布步长 $S_f \in \mathbb{N}^+$ 与相位 $\phi_f \in \{0, \dots, S_f-1\}$。第 $t_k$ 日（$t_k \equiv \phi_f \pmod{S_f}$）执行的发布统计数据覆盖日期区间为：
$$B_f(t_k) = [t_k - W_f + 1, t_k].$$
在 Multi-Horizon User-Time Differential Privacy 语义下，设任意受保护的用户时间跨度区间为 $I = [s, s + h - 1]$。发布事件 $t_k$ 受到区间 $I$ 内用户数据变更影响的充要条件是两区间相交（$B_f(t_k) \cap I \ne \emptyset$）：
$$\max(t_k - W_f + 1, s) \le \min(t_k, s + h - 1) \iff s \le t_k \le s + h + W_f - 2.$$
因此，区间长度 $L = W_f + h - 1$，真实受影响发布次数 $n_f(s, h; \phi_f)$ 为：
$$n_f(s, h; \phi_f) = \left| \{ t \in [s, s + L - 1] \mid t \equiv \phi_f \pmod{S_f} \} \right| = \begin{cases} 1 + \left\lfloor \frac{L - 1 - \Delta}{S_f} \right\rfloor, & \Delta \le L - 1 \\ 0, & \Delta > L - 1 \end{cases}$$
其中 $\Delta = (\phi_f - s) \bmod S_f \in \{0, \dots, S_f - 1\}$。

### A.2 独立反例：禁止仅按“发布日期落入 $I$”计费
若错误统计 $n_f^{\text{naive}} = |\{t \in [s, s+h-1] \mid t \equiv \phi_f \pmod{S_f}\}|$，考虑 $W_f = 30, S_f = 7, \phi_f = 0, s = 0, h = 1$：
- 错误口径（$t \in [0, 0]$）：仅计 $t=0$（**1 次发布**）；
- 正确相交口径（$t \in [0, 29]$）：包括 $t \in \{0, 7, 14, 21, 28\}$（**5 次发布**）。
该误差低估单日隐私曝光达 $5\times$。本仓库 `calendar_accounting.py:count_intersections_fast()` 与离散枚举 `count_intersections_oracle()` 严格执行 $L = W_f + h - 1$（`test_data_window_intersection_counterexample_vs_release_date_only` 已回归验证）。

---

## Appendix B: 承诺曲线非凹性实证与无凹性仿射尾部归约定理

### B.1 规范承诺曲线的非全局凹性
将目标 $(\varepsilon, \delta = 10^{-6})$ 锚点按 $\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}$ 反演至 zCDP 空间后，相邻节点割线斜率为：

| 承诺曲线 | $[1, 7]$ 斜率 | $[7, 30]$ 斜率 | $[30, 90]$ 斜率 | $[90, 365]$ 斜率 ($\beta_{\text{tail}}$) | 全局凹性判断 |
|---|---:|---:|---:|---:|:---:|
| **`Original_Planner`** | `0.00102022` | `0.00077494` | `0.00076604` | **`0.00079364`** ($+2.76\times 10^{-5}$) | **非全局凹 (`False`)** |
| **`Week1_v2`** | `0.00125250` | `0.00059393` | **`0.00076979`** ($+1.76\times 10^{-4}$) | **`0.00079569`** ($+2.59\times 10^{-5}$) | **非全局凹 (`False`)** |

### B.2 仿射尾部归约定理（无凹性前提）
由于两条曲线在 $\rho(h)$ 空间均不满足全局凹性，任何“依赖全局凹性证明 575 天无限归约”的论证均不成立。现给出仅依赖 **$h \ge H_0 = 365$ 仿射尾部线性定义**的严格证明：
- 设 $\bar\rho(h) = a_{\text{tail}} + \beta_{\text{tail}} h$ 对所有 $h \ge H_0$（要求 $H_0 \ge \text{horizons}[-1] = 365$）严格成立；设联合周期 $P = \operatorname{lcm}(7, 7, 30) = 210$。
- **充分条件**：(1) $\forall h \in [1, H_0 + P = 575], L_{\max}(h) \le \bar\rho(h)$；(2) $\text{slope} = d + \sum_f x_f / S_f \le \beta_{\text{tail}}$。
- **证明**：对任意 $h > 575$，唯一分解为 $h = h' + k P$，其中 $k \in \mathbb{N}^+$ 且 $h' \in [H_0, H_0 + P - 1] = [365, 574]$。
  因 $h', h \ge H_0$ 均严格处于仿射区间，直接成立等式：
  $$\bar\rho(h) - \bar\rho(h') = k P \beta_{\text{tail}}.$$
  由 210 天周期平移恒等式：
  $$L_{\max}(h) - L_{\max}(h') = k P \cdot \text{slope}.$$
  相减即得：
  $$S(h) = \bar\rho(h) - L_{\max}(h) = S(h') + k P (\beta_{\text{tail}} - \text{slope}) \ge S(h') \ge 0. \quad \blacksquare$$

---

## Appendix C: 四级最优性证书、Primal/Dual 下界与账本容差误差界

### C.1 四级优化与可发布证书严格分层
为杜绝将“数值迭代收敛”混同为“数学全局最优证明”，规划引擎强制分离以下四个正交状态层级（`PlanningResult.optimality_certification_tier`）：
1. **`SOLVER_CONVERGED`**：连续 SLSQP 切平面迭代在 `SOLVER_VIOLATION_TOL=1e-8` 内终止，但存在微弱负松弛或未经零违规认证；**严禁标为 `OPTIMAL` 或发布**。
2. **`FEASIBLE_CERTIFIED`**：经 `_certify_and_shrink_budget()` 保守缩放并严格通过 575 天零容差独立安全证书（`min_slack >= 0.0`），但全相位空间对偶下界未完整封口（例如固定预算或存在未收敛相位）。
3. **`NEAR_GLOBAL_OPTIMAL_CERTIFIED`**：策略通过严格零违规发布认证，**全部 210 个规范相位类**均有已认证对偶下界或被单调下界检查严格证明空可行域（`_is_phase_provably_infeasible`），且在**保守收缩后重算**的相对最优性间隙 $\text{rel\_gap} = \frac{f(\mathbf{v}_{\text{cert}}) - LB_{\text{global}}}{LB_{\text{global}}} \le 10^{-4}$。
4. **`GLOBAL_OPTIMAL_PROVED`**：仅保留给零数值间隙且无缩减退让的纯符号闭式全局证明；IEEE-754 数值规划与保守收缩解**严格不标为 `GLOBAL_OPTIMAL_PROVED`**（故 `is_math_global_optimal == False`）。

### C.2 可验证 Lagrangian 对偶下界定理
对固定相位 $\vec\phi$，考虑有效线性约束子集 $A \mathbf{v} \le \mathbf{b}$（包含尾部增长率约束与近紧 $(h, s)$ 包络约束）及任何非负乘子 $\boldsymbol{\lambda} \ge \mathbf{0}$。令 $\mathbf{g}(\boldsymbol{\lambda}) = A^T \boldsymbol{\lambda}$，对偶函数关于盒约束 $\mathbf{v} \in [\mathbf{l}, \mathbf{u}]$ 具有精确闭式解：
$$q(\boldsymbol{\lambda}) = \sum_{j=1}^4 \left(\frac{c_j}{\hat{v}_j} + g_j(\boldsymbol{\lambda})\hat{v}_j\right) - \boldsymbol{\lambda}^T \mathbf{b}, \quad \hat{v}_j = \begin{cases} \operatorname{clip}(\sqrt{c_j / g_j}, l_j, u_j), & g_j > 0 \\ u_j, & g_j \le 0 \end{cases}.$$
- **v2 证书修正说明**：PR #4 初版曾采用 `float64` 下 `q_best * (1.0 - 1e-14)` 作为浮点扣减，但反例测试（`test_naive_float64_dual_deduction_fails_cancellation_counterexample`）表明在大对偶乘子（$\boldsymbol{\lambda} \sim 10^{14}$–$10^{15}$）引起的相消截断下，朴素 `1e-14` 比例扣减可高估真对偶值 $+0.2166$ 并突破弱对偶上界。现已替换为 `certify_lagrangian_dual_bound_exact()`：**100 位高精度 `Decimal` 有向舍入（`ROUND_FLOOR` / `ROUND_CEILING`）+ 精确有理下界（`1/7, 1/30`）+ `math.nextafter(..., -inf)`** 向下安全转换（`DECIMAL100_DIRECTED_ROUNDING`），无条件消除有界精度内的上偏舍入越界。

### C.3 账本 `1e-12` 容差保守数值误差论证
- 通用接口 `UnifiedPrivacyLedger` 的浮点求和容差 `PRIVACY_LEDGER_TOL = 1e-12` 对应在最小锚点（`Original_Planner`, $\bar\rho(1)=0.0141986, \delta=10^{-6}$）下的最坏 $(\varepsilon, \delta)$ 边界漂移严格上界为：
  $$\Delta\varepsilon \le \left(1 + \sqrt{\frac{\ln(1/\delta)}{\bar\rho(1)}}\right) \cdot 10^{-12} \le 3.22 \times 10^{-11}.$$
- 在 `JointCalendarPlanner` 提交路径 `commit_plan_to_ledger()` 中，由于保守收缩预留了 $> +10^{-13}$ 的物理安全余量（远大于 420–575 天 IEEE-754 累加舍入误差 $< 3 \times 10^{-14}$），接口启用 `strict_zero_tol=True`，实现数学意义与代码层面的**双重零违规提交**。

---

## Appendix D: 实验设计与显式限制条件 (Explicit Scope & Limitations)

1. **基线冻结与长程增量隔离**：
   - 既有 16 组 90 天基准结果（`results/calendar_planning/true_e2e_benchmark_results.csv`）保持完全冻结不变；
   - 新增 420 天跨周期端到端对比与 $C_r \in \{30, 50, 100, 200\}$ 裁剪消融全部写入独立目录 `results/sdg_mh/long_horizon_420d/`，并修复了日期字符串格式超出 31 天的问题（`datetime.date(2025, 1, 1) + timedelta(days=day-1)`）。
2. **420 天跨周期与多随机种子统计总结（2,000 用户 $\times$ 3 数据种子 $\times$ 10 MC = 30 次/配置）**：
   - **长程窗口样本扩展**：相比 90 天实验仅有 $N=1$ 个 90d Distinct 窗口，420 天流水线覆盖 **$60$ 个 7d 窗口、$56$ 个 30d 窗口与 $12$ 个 90d 窗口**。
   - **跨周期安全轨迹实测**：在检查点 $t \in \{210, 365, 420\}$，全部配置的真实账本最坏消耗 $\rho_{\text{ledger}}(h)$ 处处满足 $\rho_{\text{ledger}}(h) \le L_{\max}^{(575)}(h) \le \bar\rho(h)$（零违规、`strict_zero_tol=True` 通过，第 210 天受冷启动截断略低于平稳极限，第 365/420 天平稳贴合并保留严格正余量 $+0.0012 \sim +0.0170$）。
   - **Week1_v2 DistinctHeavy 端到端改善**：7d / 30d / 90d Distinct MAPE 由 Group B 的 `2.88% ± 0.05% / 2.55% ± 0.05% / 2.78% ± 0.07%` 改善至 Group D 的 **`2.79% ± 0.05% / 2.46% ± 0.04% / 2.61% ± 0.08%`**。
   - **$C_r$ 裁剪消融 Pareto 拐点**：在固定 `Week1_v2` 每日预算 $d \approx 0.000471$ 与各向同性配比 $\eta = C_r^2/(C_r^2+C_f^2)$ 下，$C_r=30 \to 50 \to 100 \to 200$ 的端到端 Total RMSE 呈经典 U 型曲线：**`3337.54 ± 9.92`（偏置主导）$\to$ `2346.22 ± 23.90`（$C_r=50$ 平衡极小值）$\to$ `3175.34 ± 22.78` $\to$ `6195.86 ± 35.92`（噪声主导）**。
3. **显式论文限制条件与审稿风险 (Limitations & Paper Risks for PVLDB Camera-Ready)**：
   - **小幅优化收益与有限样本置信区间重叠风险**：在 `Original / DistinctHeavy` 420 天 30 次有限采样中，虽然 Group D 优化器抽象目标严格降低 `+1.12%`（`940.89 -> 930.33`）且解析 $\sigma$ 更低，但由于 $\sigma$ 绝对差距仅 $\sim 1\%$，有限次离散高斯随机噪声下样本 MAPE 置信区间（`2.83% ± 0.05%` vs `2.87% ± 0.04%`）可能出现统计重叠；Default 工作负载下更因尾部约束瓶颈使得 $D \equiv B$（0% 增益）。论文必须严格避免在全负载下夸大物理 RMSE 增益，而应聚焦于高冲突日历与短程紧密约束下的可行域释放效应。
   - **静态日历限定**：210 天周期恒等式与 575 天归纳归约严格限于固定周期静态日历（假设 A1–A4）；数据自适应触发查询流（Adaptive Online Revisit）标记为 `NOT_PROVED`，必须依赖运行时 `UnifiedPrivacyLedger` 逐次拦截。
   - **合成工作负载重尾偏差**：企业合成遥测采用 $\text{LogNormal}$ 日请求量分布，$C_r=50$ 处存在 `-1439.56` 负偏置；真实长尾分布下最优 $C_r^*$ 需基于公开先验或平滑敏感度校准。
   - **单进程崩溃语义边界**：`UnifiedPrivacyLedger` 与 `SealedStore` 在单进程线程安全加锁与局部写入截断下验证通过；跨节点分布式崩溃恢复标记为 `NOT_VALIDATED`。

---

## Appendix E: v2 论文级统计复核、CRN 配对实证与时间序列竞争基线对比 (`long_horizon_420d_v2_audit`)

1. **修正 PR #5 分层伪重复与种子滑动重叠问题（$K=8 \times M=5 = 40$ 次运行，独立单元 $df=7$）**：
   - 修复 PR #5 中 `trial_seed + day` 导致的相邻 MC 抽样共享 `419/420` 日常种子与 `59/60` 周窗种子、以及 `B` / `D` 未采用 Common Random Numbers (CRN) 配对的统计缺陷；改用碰撞无关的 `SeedSequence([data_seed, mc_trial, day, stream_id])`，严格以**数据种子为独立统计单元**计算分层 95% CI（$df=7$）及方差分解（ICC）。
   - 例如 `requests_clipping_bias` 的类内相关系数 $\text{ICC}=1.000$，证明 PR #5 朴素池化 CI（$\pm 6.67$）低估了真实跨数据种子不确定性（$\pm 18.40$）达 $2.76\times$。
2. **`Original / DistinctHeavy` 此前反转现象的根因与配对重测结论**：
   - 修复种子滑动重叠并采用 $K=8$ CRN 配对后，`Original / DistinctHeavy` 下 `D_JointPlanning` 相对 `B_BudgetOnly` 在 **7d Distinct（`-1.14%`, $p=0.0001$, `8/8` 种子）与 90d Distinct（`-3.99%`, $p<10^{-4}$, `8/8` 种子）** 上稳定更优；此前 PR #5 中的反转主要由 Seed 42 重叠噪声异常值（`59/60` 种子重叠）与非配对方差造成。
   - **诚实披露的负向权衡与非显著项**：(a) 因 `DistinctHeavy` 目标函数权重 `(0.10, 0.40, 0.40, 0.10)` 主动压缩每日预算 `d`，`D_JointPlanning` 的每日加性 Requests RMSE 上升 `+0.38%`（`Week1_v2`）至 `+0.53%`（`Original`，`0/8` 种子更优）；(b) 在 **30d Distinct** 上，由于 `D` 将相位 $\phi_{30}$ 从 `0`（周日）移至 `4`（周四），受周期内工作日比例提高影响（真实均值由 `1965.9` 增至 `1972.1`，`+0.32%`），不同日期窗口抽样误差使 `30d MAPE` 差值仅 `-0.023%` ~ `-0.025%`（$p=0.26$–$0.33$，**`K=8` 下未达统计显著**）。
3. **同一语义下的竞争基线对比（`Week1_v2`, $K=8$, 相同剩余加性包络与 `C_r=50, C_f=10`）**：
   - **可比加性工作负载**：在 420 天滚动 Requests/Failures 查询上，`SDG-MH` 的 1d / 7d Requests RMSE 为 **`2310.5 ± 29.7` / `10919.2 ± 324.2`**，显著优于严格对齐相同包络的 `Tree-2l (Honaker GLS)`（`4378.6 ± 61.7` / `12736.0 ± 199.4`）、`W-BD-30 (MH-Calibrated)`（`3565.6 ± 236.3` / `15477.0 ± 1310.2`）和 `Tree-U`（`27742.4` / `50365.7`）；在 90d Requests 上所有方法趋近共同的 `C_r=50` 裁剪偏置下界（`~129k`），而在低偏置指标 `Failures 30d` 上 `SDG-MH`（`7667.4 ± 510.3`）仍优于 `Tree-2l`（`9682.6 ± 777.3`）。
   - **非可比性声明**：`Tree` 与 `W-BD` 无法直接支撑带 $k=3$ 任务裁剪的跨期去重 Distinct 直方图，表中严格标为 `N/A (Non-Additive Union)`；`W-BD-30` 若仅按单点 $\bar\rho_{\text{add}}(30)$ 校准将在 $h=365$ 超标 `1.35x`。

---

## Appendix F: v3 最终一致性复核修正（128 位单射种子、统一后处理、同查询日期口径与 Holm-Bonferroni 检验，`long_horizon_420d_v3_final`）

1. **修复 PR #6 `uint32` 碰撞并实证零碰撞（50,520 个实验种子）**：
   - 将 `derive_crn_seed` 升级为 128 位单射位打包编码 `(ds << 96) | (mc << 64) | (day << 32) | stream_id`（`stream_id < 500000` 以容纳 `CalendarJob +500003` 任务偏移）；全量 50,520 个种子检测实现 **0 碰撞**。
2. **统一全方法窗口级非负与一致性后处理（`Failures <= Requests`）**：
   - `SDG-MH`、`Tree-2l`、`Tree-U` 与 `W-BD-30` 对每个滚动窗口统一先求和再执行单次 $\hat R = \max(0, \tilde R), \hat F = \min(\hat R, \max(0, \tilde F))$；消除了 PR #6 中 `W-BD-30` 日内正裁剪偏置累积（`Failures 30d` 由 `20410.14` 修正为 `14172.86`，`-30.56%`；`90d` `-53.94%`）及 `Tree-U` `Failures 1d` 越界（`-47.27%`）。
3. **同查询请求日期集合下的发布日、最近可用结果与陈旧度分离（`B` vs `D`）**：
   - 在完全相同的每日仪表盘查询集合下（$K=8$，`df=7`）：7d Distinct（$p_{\text{holm}} \le 0.0001$，Wilcoxon $p=0.0078$，`8/8` 种子）与 90d Distinct（$p_{\text{holm}} = 0.0000$，`8/8` 种子）在**发布日**与**同日查询**下均显著改善；30d 同日平均陈旧度相近（`2.99d` vs `3.00d`），但 `common_daily_30d_served_mape` 仍因噪声/工作周期波动保持统计不显著（$p_{\text{holm}} = 1.000$，Wilcoxon $p = 0.547$–`0.742`），而若业务锚定周日固定查询，非约束相位 $\phi_{30}=4$ 带来显式 `+3.0d` 陈旧度及 `+7.55` 滞后漂移 RMSE。
4. **有限时域 vs 无限时域证书界线**：
   - 竞争基线性能对比严格限定于共同 `[1, 420]` 时域、共同剩余可加 zCDP 包络及共同滚动可加查询；明确披露 `SDG-MH` 持有无限时域（$h \ge 1$）证书，而 `Tree (L=9)` 仅覆盖 $T \le 512$ 有限二叉树。
   - 理论与实验三重分层边界：`DECIMAL100_DIRECTED_ROUNDING` 在有条件输入假设（`rho_bars` 为操作性 RHS 上界、$c_j \ge 0, l_j > 0$）下构成严格 100 位浮点舍入证书，不可替代符号无限闭式解（`is_math_global_optimal = False`）。
