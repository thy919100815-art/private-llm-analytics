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
由弱对偶定理与 $10^{-14}$ 浮点向下安全扣减，$LB_\phi = (1 - 10^{-14}) q(\boldsymbol{\lambda}) \le f_\phi^*$ 无条件成立。

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
