# Private LLM Analytics v0.6 — Statistic-Specific DP Mechanism Framework

## 1. Executive Summary & Design Principle

In differential privacy analytics for enterprise LLM telemetry, statistical queries cannot be treated merely as a scalar metric coupled with an arbitrary choice of Gaussian or Laplace perturbation. 

In **Private LLM Analytics v0.6**, differential privacy mechanisms are formally tied to **statistic-specific DP release profiles**. Each supported enterprise statistic owns a complete, mathematically closed release pipeline:

$$\text{Statistical Target} \longrightarrow \text{User Contribution} \longrightarrow \text{Bounding} \longrightarrow \text{Sensitivity} \longrightarrow \text{DP Mechanism} \longrightarrow \text{Privacy Cost (RDP)} \longrightarrow \text{Post-Processing} \longrightarrow \text{Estimand}$$

Production queries select the **statistical target**; the system automatically binds the **approved DP profile**. Arbitrary client-side selection of mechanism families, sensitivities, noise scales, or RNG seeds is rejected by design.

---

## 2. Mathematical Formalization & The Four Approved Profiles

### 2.1 Profile 1: User Prevalence (`user_prevalence`)
- **Profile ID**: `binary_prevalence_gaussian` (v1)
- **Target**: Proportion of eligible enterprise users who adopted each of the 9 Provider $\times$ Task capabilities.
- **Privacy Unit**: Individual enterprise user trajectory within the observation window.
- **Adjacency**: Add/remove user ($D \sim D'$ differing by one user $u$).
- **Per-User Contribution**: Binary indicator vector $b_u \in \{0, 1\}^9$, where $b_{u,j} = \mathbb{I}[m_{u,j} > 0]$ for user cell interaction count $m_{u,j}$.
- **Contribution Bound**: Dimension $k = 9$ fixed cells per user. $\|b_u\|_2 \le \sqrt{9} = 3.0$.
- **Sensitivity**: Exact $L_2$ sensitivity $\Delta_2 = \sqrt{k} = 3.0$.
- **DP Mechanism**: Analytic Gaussian Vector Mechanism calibrated to $(\epsilon, \delta)$ with sensitivity $\Delta_2 = 3.0$:
  $$\tilde{S} = \sum_{u} b_u + \mathcal{N}(0, \sigma^2 I_9)$$
- **Accounting**: Single Gaussian RDP charge $\rho(\alpha) = \frac{9\alpha}{2\sigma^2}$.
- **Post-Processing**:
  1. Negative counts clamped: $\hat{S}_j = \max(0, \tilde{S}_j)$.
  2. Divided strictly by fixed **public** eligible workforce denominator $N_{\text{public}}$: $\hat{p}_j = \hat{S}_j / N_{\text{public}}$.
  3. Clamped into valid probability range $[0.0, 1.0]$.
- **Released Estimand**: $\hat{p}_j \in [0, 1]$ prevalence rate per cell. Active sample user count is never leaked.

### 2.2 Profile 2: User-Normalized Share (`user_normalized_share`)
- **Profile ID**: `simplex_share_gaussian` (v1)
- **Target**: Distribution of user workflow effort across the 9 Provider $\times$ Task capabilities.
- **Privacy Unit**: User trajectory.
- **Adjacency**: Add/remove user.
- **Per-User Contribution**: Normalized simplex vector $s_u \in \Delta^9$ where $s_{u,j} = \frac{m_{u,j}}{\sum_k m_{u,k}}$ if $\sum_k m_{u,k} > 0$, else $0$.
- **Contribution Bound**: $L_1$ norm bound $\|s_u\|_1 \le 1.0 \implies \|s_u\|_2 \le 1.0$.
- **Sensitivity**: Exact $L_2$ sensitivity $\Delta_2 = 1.0$.
- **DP Mechanism**: Analytic Gaussian Vector Mechanism on bounded sum $\sum_u s_u$:
  $$\tilde{V} = \sum_u s_u + \mathcal{N}(0, \sigma^2 I_9)$$
- **Accounting**: Gaussian RDP charge $\rho(\alpha) = \frac{\alpha}{2\sigma^2}$.
- **Post-Processing**:
  1. Non-negativity clamp: $\hat{V}_j = \max(0, \tilde{V}_j)$.
  2. Simplex projection/renormalization: $\hat{s}_j = \hat{V}_j / \sum_k \hat{V}_k$ (uniform $\frac{1}{9}$ if all zero).
- **Released Estimand**: Public share vector $\hat{s} \in \Delta^9$ summing to $1.0$, marginal provider shares, and marginal task shares.

### 2.3 Profile 3: Interaction Volume (`interaction_volume`)
- **Profile ID**: `clipped_volume_gaussian` (v1)
- **Target**: Aggregate volume of LLM interactions across Provider $\times$ Task capabilities.
- **Privacy Unit**: User trajectory.
- **Adjacency**: Add/remove user.
- **Per-User Contribution**: Raw interaction counts $m_u \in \mathbb{R}^9_{\ge 0}$.
- **Contribution Bound**: User-level $L_2$ clipping at norm threshold $C$:
  $$\bar{m}_u = m_u \cdot \min\left(1, \frac{C}{\|m_u\|_2}\right) \implies \|\bar{m}_u\|_2 \le C$$
- **Sensitivity**: Exact $L_2$ sensitivity $\Delta_2 = C$.
- **DP Mechanism**: Analytic Gaussian Vector Mechanism:
  $$\tilde{M} = \sum_u \bar{m}_u + \mathcal{N}(0, \sigma^2 I_9)$$
- **Accounting**: Gaussian RDP charge $\rho(\alpha) = \frac{\alpha C^2}{2\sigma^2}$. Supports continual observation tree decomposition and utility-aware parameter planning.
- **Post-Processing**: Coordinates clamped to non-negative: $\hat{M}_j = \max(0, \tilde{M}_j)$.
- **Released Estimand**: Differential-privacy bounded interaction volume estimates.

### 2.4 Profile 4: Success Rate (`success_rate`)
- **Profile ID**: `bounded_success_rate_joint_gaussian` (v1)
- **Target**: Task execution success rate $r_j = \frac{\sum_u s_{u,j}}{\sum_u n_{u,j}}$ per cell $j \in \{1,\dots,9\}$.
- **Privacy Unit**: User trajectory.
- **Adjacency**: Add/remove user.
- **Per-User Contribution**: User interaction attempts $n_u \in \mathbb{R}^9_{\ge 0}$ and successful interactions $s_u \in \mathbb{R}^9_{\ge 0}$ satisfying coordinate-wise $0 \le s_{u,j} \le n_{u,j}$.
- **Contribution Bounding & Sensitivity Derivation**: See Section 3 below. Joint vector $w_u = [\bar{s}_u; \bar{n}_u] \in \mathbb{R}^{18}$ has proved sensitivity $\Delta_2 = \sqrt{2} C_{\text{rate}}$.
- **DP Mechanism**: **ONE single 18-dimensional joint Gaussian release**:
  $$[\tilde{S}; \tilde{N}] = \sum_u [\bar{s}_u; \bar{n}_u] + \mathcal{N}(0, \sigma^2 I_{18})$$
  with noise scale $\sigma$ calibrated via analytic Gaussian mechanism for sensitivity $\sqrt{2} C_{\text{rate}}$ and $(\epsilon, \delta)$.
- **Accounting**: Exactly **ONE joint RDP cost**:
  $$\rho(\alpha) = \frac{\alpha (\sqrt{2} C_{\text{rate}})^2}{2\sigma^2} = \frac{\alpha C_{\text{rate}}^2}{\sigma^2}$$
- **Post-Processing**:
  1. Noisy total attempts clamped: $\hat{N}_j = \max(0, \tilde{N}_j)$.
  2. Noisy successes clamped: $\hat{S}_j = \text{clip}(\tilde{S}_j, 0, \hat{N}_j)$.
  3. Denominator threshold check against public $\tau = \text{min\_noisy\_denominator}$ (e.g. 1.0 or 5.0):
     - If $\hat{N}_j < \tau$: cell rate is suppressed (`null`) with status `"noisy denominator below reporting threshold"`.
     - Else: rate is computed and clamped into $[0.0, 1.0]$: $\hat{r}_j = \text{clip}\left(\frac{\hat{S}_j}{\hat{N}_j}, 0.0, 1.0\right)$.
- **Released Estimand**: Derived success rates $\hat{r}_j \in [0, 1]$ (or suppressed), noisy attempts $\hat{N}_j$, and noisy successes $\hat{S}_j$. Raw exact counts and active user counts are strictly non-disclosed.

---

## 3. Formal Sensitivity Proof for Success Rate

### Theorem 1 (Joint Bounded Success-Rate Sensitivity)
Let user attempts be $n_u \in \mathbb{R}^d_{\ge 0}$ and user successes be $s_u \in \mathbb{R}^d_{\ge 0}$ with coordinate-wise constraint $0 \le s_{u,j} \le n_{u,j}$ for all $j \in \{1,\dots,d\}$.
Let user clipping threshold $C_{\text{rate}} > 0$ be applied to total attempts using shared scaling factor:
$$a_u = \min\left(1, \frac{C_{\text{rate}}}{\|n_u\|_2}\right)$$
Define the clipped user attempt vector and clipped user success vector as:
$$\bar{n}_u = a_u n_u, \quad \bar{s}_u = a_u s_u$$
Then the joint user contribution vector $w_u = \begin{bmatrix} \bar{s}_u \\ \bar{n}_u \end{bmatrix} \in \mathbb{R}^{2d}$ satisfies:
$$\|w_u\|_2 \le \sqrt{2} C_{\text{rate}}$$
Consequently, under add/remove single user adjacency, the global $L_2$ sensitivity of the joint sum $\sum_u w_u$ is:
$$\Delta_2 = \sqrt{2} C_{\text{rate}}$$

### Proof:
1. By definition of $a_u = \min\left(1, \frac{C_{\text{rate}}}{\|n_u\|_2}\right)$, the norm of the clipped attempt vector satisfies:
   $$\|\bar{n}_u\|_2 = a_u \|n_u\|_2 = \min(\|n_u\|_2, C_{\text{rate}}) \le C_{\text{rate}}$$
2. Because $0 \le s_{u,j} \le n_{u,j}$ and $a_u \ge 0$, we have $0 \le \bar{s}_{u,j} \le \bar{n}_{u,j}$ for each coordinate $j$.
3. Squaring both sides of the non-negative inequality gives $\bar{s}_{u,j}^2 \le \bar{n}_{u,j}^2$ for all $j$.
4. Summing across all $d$ coordinates:
   $$\|\bar{s}_u\|_2^2 = \sum_{j=1}^d \bar{s}_{u,j}^2 \le \sum_{j=1}^d \bar{n}_{u,j}^2 = \|\bar{n}_u\|_2^2 \le C_{\text{rate}}^2$$
5. For the concatenated joint vector $w_u \in \mathbb{R}^{2d}$:
   $$\|w_u\|_2^2 = \|\bar{s}_u\|_2^2 + \|\bar{n}_u\|_2^2 \le C_{\text{rate}}^2 + C_{\text{rate}}^2 = 2 C_{\text{rate}}^2$$
6. Taking the square root:
   $$\|w_u\|_2 \le \sqrt{2} C_{\text{rate}}$$
7. Under add/remove user adjacency, any neighboring datasets $D, D'$ differ by at most one user $u^*$:
   $$\left\| \sum_{u \in D} w_u - \sum_{u \in D'} w_u \right\|_2 = \|w_{u^*}\|_2 \le \sqrt{2} C_{\text{rate}}$$
Thus, the global $L_2$ sensitivity is bounded by $\sqrt{2} C_{\text{rate}}$. $\blacksquare$

---

## 4. Single Joint Release vs. Naive Budget Split

A naive approach to private ratio queries splits privacy budget $\epsilon = \epsilon_1 + \epsilon_2$, perturbing numerator and denominator independently. 
In contrast, Private LLM Analytics v0.6 releases $[\bar{s}; \bar{n}]$ in **one joint Gaussian mechanism**:
1. Calibrated to $\Delta_2 = \sqrt{2} C_{\text{rate}}$ under the entire budget $(\epsilon, \delta)$.
2. Preserves cross-coordinate correlation without double privacy charging.
3. Consumes only ONE entry in the persistent lifetime RDP ledger.

---

## 5. Web Console Architecture & Information Exposure Boundaries

- **User Control**: Browser selects the statistic and target $(\epsilon, \delta)$ and observation window.
- **Read-Only Scheme Panel**: Displays the authoritative backend profile name and rules.
- **Collapsible Mechanism Details**: Displays public mathematical properties (privacy unit, adjacency, contribution bound, sensitivity formula, DP mechanism, accounting mode, continual support).
- **Security Boundary**: The browser has no controls for selecting mechanism family, sensitivity, noise scale, or RNG seed. All internal states, raw interaction counts, active user denominators, and private vectors remain within the trusted server.
