from pathlib import Path
import json, re
P=Path(__file__).resolve().parents[1]
def edit(name, replacements):
    p=P/name
    s=p.read_text(encoding='utf-8')
    for old,new in replacements:
        if old not in s: print('NOT FOUND',name,old[:65])
        s=s.replace(old,new)
    p.write_text(s,encoding='utf-8')

edit('main.tex',[(
'实验与测试结果表明，全套 542 项工程测试套件全量通过，在 50,000 用户量级的企业级基准数据上，任务成功率百分点误差小于 0.1 pp，系统能够为企业在外部接收方环境下安全开展多模型效能度量提供具有数学严格证明的差分隐私保障。',
'本文以 50,000 用户企业级合成基准分析统计效用，并将理论模型、工程测试与实现限制分别说明。静态指标在用户级加/删邻接和理想高斯采样假设下具有所列敏感度界；时间树按代际内每日范数之和约束，采用逐层联合敏感度与 RDP 组合。当前统计 RNG 不宣称 CSPRNG 安全属性，发布键配置绑定及部分默认值仍有待完善；测试仅支持其覆盖范围内的结论，不能替代完整安全证明。')])
edit('sections/00_abstract_notation.tex',[(
'成功率有效发布的最小加噪分母阈值，低于该值执行抑制','成功率加噪分母阈值；本文显式配置 $\\tau=1.0$，低于该值抑制'),(
'持续时间序列观测中的每日截断界与代际（Epoch）综合截断界','持续观测的每日 $L_2$ 界与代际每日 $L_2$ 范数之和界'),(
'本文所声明的所有隐私上界均在此邻接关系下严格推导并获得成立保证。','本文的静态与时间树数学界均采用此邻接关系；实现条件及尚未完成的工程保证见第~\\ref{sec:engineering}、\\ref{sec:limitations} 节。')])
edit('sections/01_background.tex',[(
'在设定的严格用户级邻接关系与','在设定的用户级邻接关系与'),('具备严格数学稳定性保证','具有所述数学隐私上界'),('在无需担忧单名员工行为轨迹被外部逆向反推的前提下，安全向','在限定个体隐私损失的前提下，向'),('基于全维度语义哈希实现严格的幂等发布控制','基于已绑定的语义字段实现幂等发布控制（配置绑定限制见第~\\ref{sec:accounting_release} 节）'),('系统均无条件阻断执行，绝不向不可信接收方降级回退原始数据','系统在已实现的校验路径中拒绝执行，并禁止向不可信接收方回退原始数据')])
edit('sections/02_requirements_threat_model.tex',[(
'（隐私损失参数放大约至 $k\\varepsilon$；而在近似差分隐私下，$\\delta$ 项也会放大至约 $k e^{(k-1)\\varepsilon} \\delta$）',
'（若记录级机制满足 $(\\varepsilon,\\delta)$-DP，则标准链式界为 $(k\\varepsilon,\\delta_k)$-DP，其中 $\\delta_k=\\delta\\sum_{i=0}^{k-1}e^{i\\varepsilon}$；亦可采用较松上界 $\\delta_k\\le k e^{(k-1)\\varepsilon}\\delta$，二者不是近似等式）'),(
'系统处理的核心数据源为企业大模型网关沉淀的交互遥测记录。','系统面向企业大模型网关交互遥测记录；本文效用评测使用程序生成的合成基准，不代表真实员工行为。'),(
'在企业实际环境中，遥测日志呈现出显著的','作为需求假设，企业遥测日志可能呈现'),('此类操作绝不可能凭空削弱已成立的隐私保证','在不额外访问私有数据的条件下，此类操作保持已成立的隐私保证')])
edit('sections/03_architecture.tex',[(
r'操作系统物理真随机数控制\\(\texttt{OS Entropy RNG})',r'统计随机采样（非 CSPRNG）\\(\texttt{NumPy RNG})'),(
'基于精确的解析高斯机制（Analytic Gaussian Mechanism）求解最优噪声标准差 $\\sigma$；生产模式下严密阻断任何调用方对随机数发生器（RNG）的控制，确保噪声样本源自操作系统提供的熵源（OS-provided entropy）初始化的随机数生成器。对于流式时间序列，该层挂载时间二叉树机制实现增量加噪。',
'基于解析高斯条件数值求解噪声标准差 $\\sigma$。静态生产发布拒绝调用方注入 RNG/seed，使用操作系统熵初始化的统计 RNG；不宣称 CSPRNG 安全属性。时间树当前另用 NumPy 全局 RNG，限制见第~\\ref{sec:engineering} 节。'),(
r'$\varepsilon_{\text{current}} + \Delta\varepsilon \le \varepsilon_{\text{cap}}$',r'$\varepsilon_{\text{projected}}(\delta_{\text{global}}) \le \varepsilon_{\text{cap}}$')])
edit('sections/05_dp_profiles.tex',[(
r'默认阈值 $\tau = 5.0$',r'本文 YAML 生效阈值 $\tau = 1.0$'),('分子物理约束','分子值域约束'),('确保成功数绝不超过总调用数','使后处理成功数不超过后处理总量'),(
'随后仅基于加噪得分向量 $\\tilde{c}$ 进行降序排序，取前 $K$ 个元素作为输出集合 $\\operatorname{TopK}(\\tilde{c})$。',
'随后仅基于加噪得分向量 $\\tilde{c}$ 降序排序。代码遍历有限任务列表并在达到 $K$ 项时停止，无额外阈值时等价于取 $K_{\\text{eff}}=\\min(K,T)$ 项；有阈值时还可更少。当前 $K=5,T=3$，实际输出 3 项。这里的计数是参与任务的独立用户数，不是交互次数。')])

edit('sections/07_accounting_release.tex',[(
'呈现等价的','呈现可保证的'),('执行精确的对偶凸优化转换','在预设离散 RDP 阶数集合上最小化标准 RDP-to-DP 上界'),(
'相较于经典高级组合定理（Advanced Composition Theorem），RDP 组合消除了复杂的松弛常数项，显著延长了企业在给定隐私上限下的有效可用发布次数。',
'这是合法的保守上界，不等价于对所有连续 $\\alpha>1$ 求全局最优，也不代表精确隐私损失。零成本空历史按代码返回 $0$；非空历史采用上述离散网格。RDP 组合的效用收益取决于发布参数。'),(
'系统立即抛出 \\texttt{PrivacyBudgetExceededError}，无条件熔断并拒绝执行，坚决恪守 Fail-Closed 安全底线。','系统抛出 \\texttt{PrivacyBudgetExceededError} 并拒绝本次发布。'),(
'从而形式化抵御了多供应商联合反演攻击','以对该组全部已计费发布的联合可见信息进行组合约束'),(
'系统对查询请求的所有本征要素提取确定性哈希签名：',
'当前代码对规范 JSON 字典计算 SHA-256；下式概括其已绑定字段，$h_{\\text{profile}}$ 表示现有 Profile 指纹：'),(
r'  \text{ReleaseKey} = \text{SHA256}\Big(\text{fp}_{\text{dataset}} \,\|\, \text{window} \,\|\, \text{metric\_id} \,\|\, \mathcal{S}_{\text{scope}} \,\|\, \varepsilon \,\|\, \delta \,\|\, \text{taxonomy\_ver}\Big)',
r'''  \begin{split}
  \text{ReleaseKey}=\text{SHA256}\big(\operatorname{CanonicalJSON}(&\text{dataset},\text{window},\text{metric},\mathcal S_{\text{scope}},\varepsilon,\delta,\\
  &C,C_{\text{rate}},k,N_{\text{public}},h_{\text{profile}},\text{versions},\ldots)\big).
  \end{split}'''),(
'仅当 \\texttt{ReleaseKey} 为全系统首次出现时，才由操作系统提供的熵源（OS-provided entropy）初始化的随机数生成器生成单次高斯样本并持久化归档。',
'首次获准提交的静态发布使用由操作系统熵初始化的统计 RNG 进行高斯采样并持久化，不据此声明密码学安全性。'),(
'该机制从工程体系上确保了相同语义的分析任务在任意高频调用下，均无法被外部实体用于统计均值恢复，同时保障了前端控制台反复刷新时的体验与零额外预算支出。',
'对同一已提交对象的重放不提供独立噪声样本，因而不增加预算。当前键还绑定机制/校准版本、贡献模式、人口模式、供应商/任务标签等，但未完整绑定运行时的 $\\tau$、Top-K 的 $K$ 与参与上限，以及独立 taxonomy 版本号。本次变参检查发现这些配置可变化而键不变；现有 Profile 指纹不足以填补此缺口。\\textbf{REMAINING LIMITATION：}应以实际生效参数构造 \\texttt{mechanism\\_config\\_hash}，纳入完整 contribution、后处理、taxonomy 版本和组合策略；实现前不能宣称跨配置重放正确。')])

edit('sections/10_engineering.tex',[(
'\\subsection{生产环境真随机数控制与注入防御}',r'\subsection{生产环境随机数采样与注入防御}'),(
'差分隐私的数学安全性从根本上依赖于噪声采样的不可预测性。若外部攻击者能够预测、操纵或重放随机数生成器（RNG）的种子（Seed），差分隐私保证将完全坍塌。',
'理论高斯机制假设按规定分布采样且随机状态不向接收方泄露。实际伪随机采样、浮点计算及种子控制需要单独审计；拒绝调用方传入种子并不等价于实现密码学安全随机数。'),(
'生产模式熵源强制锁定：','静态生产模式采样：'),(
'系统无条件采用基于操作系统提供的熵源（OS-provided entropy）初始化的随机数生成器（\\texttt{np.random.default\\_rng()}）',
'静态发布编排器采用 \\texttt{np.random.default\\_rng()}：由操作系统提供的熵初始化统计随机数生成器，用于当前高斯采样实现；该实现不宣称具备密码学 CSPRNG 安全属性'),(
'系统确立了“任何异常无条件阻断、绝不静默放行、绝不降级明文”的\\textbf{全闭环安全准则（Fail-Closed）}',
'系统以参数校验、预算预留和权限检查实施\\textbf{故障拒绝准则（Fail-Closed）}，以下结论限于对应实现路径'),('HTTP 403 强安全违例','HTTP 403 权限违例'),(
'数据库已提交记账快照} $\\equiv$ \\textbf{RESTful API JSON 载荷} $\\equiv$ \\textbf{Web 控制台呈现视图} $\\equiv$ \\textbf{CSV / JSON 导出文件',
'数据库已提交快照} $\\longrightarrow$ \\textbf{API 过滤视图} $\\longrightarrow$ \\textbf{Web / CSV / JSON 呈现'),(
'禁止在浏览器端执行任何隐式的数学重加噪或二次缩放','不在浏览器端重新抽取隐私噪声；显示百分比和舍入属于后处理')])
p=P/'sections/10_engineering.tex';s=p.read_text(encoding='utf8');mark='\\subsection{评测实验与生产环境状态严格隔离}'
s=s.replace(mark,'当前默认 BitGenerator 为 PCG64。时间树节点路径 \\texttt{continual/state.py} 仍调用 \\texttt{np.random.normal}，使用全局统计 RNG，可能受进程内全局 seed 影响。\\textbf{REMAINING LIMITATION：}尚未实现统一 CSPRNG 高斯采样器，也未完成浮点采样的端到端安全证明；静态接口的注入防御不能直接推广至时间树。\n\n'+mark)
p.write_text(s,encoding='utf8')

edit('sections/09_utility.tex',[(
'基于内部真实遥测明细','基于输入遥测明细（本章为合成数据）'),('曼哈顿距离（$L_1$ Distance）','总变差距离（TV Distance）'),(r'$L_1(s_{\text{raw}}, \hat{s}) = \frac{1}{2}',r'$\operatorname{TV}(s_{\text{raw}}, \hat{s}) = \frac{1}{2}'),(
'在基准企业环境（在册员工 $N_{\\text{public}} = 50,000$ 人、包含 3,011,232 条真实交互记录的数据集',
'在 50,000 用户企业级合成基准中（公开总体 $N_{\\text{public}} = 50,000$，包含 3,011,232 条合成交互记录的数据集'),('50,000 人基准实测表','50,000 用户合成基准实测表'),('真实用户数','原始参与用户数'),('在册的 50,000 名员工中','合成场景的 50,000 名用户中'),('至少调用过大模型一次的员工总数','至少调用过大模型一次的用户总数'),('全公司宏观使用率','该合成总体使用率'),('处理任意任务的员工为','处理任意任务的用户为'),('这一特定单元的独立员工数客观上恰好为','这一单元的独立用户数为'),(
'7 个百分点的差距是企业员工客观工作分工的反映（有 3,577 名员工在观测周期内仅使用大模型撰写文档或分析报表，从不编写代码）。',
'该差距反映合成场景中的用户行为分布：3,577 名用户未参与该供应商的 Technical 单元；不能据此推断他们从不执行其他供应商的编程任务，更不能推断真实员工分工。'),(
'该实测有力推翻了“差分隐私导致数据剧烈失真”的常见误解。','这说明该次合成基准试验中，此单元的噪声误差较小。'),('展现出极高的可用性','用于评估基准场景的误差'),('在 1,000 用户与 50,000 用户基准下的实测结果','在 1,000 用户与 50,000 用户合成基准下的历史示例结果'),('(企业基准)','(合成基准)'),(
'实测表明，在万级员工规模下，高斯机制注入的噪声相较于数十万级的真实样本量影响微弱，成功率估计误差迅速收敛至 $0.09\\text{ pp}$（相对误差小于 $0.11\\%$），完全满足采购评估与 SLA 质量仲裁的商业需求。',
'该历史示例在合成企业基准下给出 $0.09\\text{ pp}$ 的误差，不能外推真实企业 SLA。本文统一采用 $C_{\\text{rate}}=20,\\tau=1.0$；仓库所存 Parquet 不含 success 列，Success Rate 需由带成功标记的合成生成器另行构造。原表完整种子与生成快照尚未留存，本次不将这组历史数值计为已复现实验。'),(
'在真实业务中，部分自动化脚本或重度研发人员单月调用成百上千次，其超过 20 次的部分全部被几何截断。',
'当前合成用户的计数向量超出 $L_2$ 界 $C=20$ 时按比例收缩；这不是把每位用户的总调用次数直接截为 20。'),(
'在当前预置的 3 任务分类体系下，真实调用量排名前三位依次为：Technical (48,432 人)、Document (46,379 人)、Analysis (45,734 人)。系统在注入高斯噪声（$\\sigma = 9.45$）后，排序位次完全保持一致，Top-3 命中率达到 $100\\%$，Jaccard 相似度为 $1.00$。系统在报告中客观注明：当前受管字典基数相对有限，该指标的高命中率存在字典规模较小的客观前提。',
'对存储的合成 Parquet 重新按任务跨供应商去重，原始独立参与用户数依次为 Document (48,490)、Technical (48,432)、Analysis (48,311)。这不是交互调用量排名，原文混入了单供应商单元计数。当前 $K=5,T=3$，代码实际取 $K_{\\text{eff}}=3$，仍按保守敏感度 $\\sqrt{5}$ 加噪（$\\sigma=9.4467$）。无额外阈值时，选择全部三个任务使 Top-3 集合命中率和 Jaccard 必为 1，不能据此证明排序效用；位次稳定性须另作多次试验。')])
edit('sections/12_limitations.tex',[(
'由于真实企业环境存在大量依赖自动化工具或深度集成 IDE 的高频研发人员，固定截断导致整体交互总量呈现约 $39.8\\%$ 的结构性低估（Under-counting Bias）。',
'在当前合成企业基准的所示统计单元中，固定截断产生约 $39.8\\%$ 的结构性低估（Under-counting Bias）；该结论不能外推至真实企业总体。'),(
'利用历史已过期、不属于当前敏感窗口的公共或历史交互分布估算相对稳定的经验阈值 $C$','利用真正公开、获得相应授权且不在保护范围内的数据，或已按预算释放的 DP 统计估计阈值 $C$；历史数据不会因时间过期自动免除用户级隐私成本')])
p=P/'sections/12_limitations.tex';s=p.read_text(encoding='utf8');s+='\n\\paragraph{本次审计保留的实现限制（REMAINING LIMITATION）：}统计 RNG 与时间树全局 RNG 尚未升级为统一的密码学安全采样路径；ReleaseKey 对运行时分母阈值、Top-K 配置及 taxonomy 版本的绑定不完整；加载 YAML 的成功率阈值为 $1.0$，但若省略该字段，部分配置模型和直接构造入口仍默认 $5.0$，应显式传参。第~\\ref{sec:continual} 节证明针对固定流配置、前缀封存、独立理想高斯噪声与逐层记账模型，不是对任意代码路径的安全认证。合成基准、有限测试与 SQLite 事务均不能证明不存在其他漏洞；SQLite 也不等同于不可篡改账本。\n';p.write_text(s,encoding='utf8')

edit('appendix/appendix_config.tex',[(
'本附录提供生产环境的核心配置架构解析与基准模板。','本附录提供与当前仓库 YAML 对齐的核心配置节选；不是对所有入口默认值已统一的声明。'),(
'top_k: 5                  # 最终加噪截断输出的前 K 个任务','top_k: 5                  # T=3 时实际最多输出 min(K,T)=3 项'),(
'delta: 4.0e-6               # 全局累计松弛项 Delta','delta: 4.0e-6               # basic 模式/内部作用域的累加 delta 上限'),(
'epsilon: 4.0                # 全局生命周期累计隐私预算上限 Epsilon','epsilon: 4.0                # basic 模式/内部作用域的 epsilon 上限'),(
'delta_global: 1.0e-6','delta_global: 1.0e-6         # 外部 RDP 转换和 lifetime cap 的 delta'),('50,000 名员工','50,000 名合成用户')])
p=P/'appendix/appendix_config.tex';s=p.read_text(encoding='utf8');s=s.replace('\\subsection{全生命周期隐私与规划器配置 (privacy.yaml)}','本文所有 Success Rate 配置示例均显式设置 \\texttt{min\\_noisy\\_denominator: 1.0}。YAML 加载后和 Profile 默认值为 $1.0$；部分兼容构造入口仍默认 $5.0$，因此不可省略此字段。Top-K 的配置 $K=5$ 保留当前代码行为，实际输出数受任务字典大小限制。\n\n\\subsection{全生命周期隐私与规划器配置 (privacy.yaml)}')
s=s.replace('\\subsection{多接收方权限与作用域配置 (recipients.yaml)}','\\textbf{字段语义：}外部 \\texttt{mode: rdp} 发布以 \\texttt{accountant.delta\\_global} 为唯一 RDP 转换目标，cap 为 \\texttt{accountant.epsilon\\_cap}。静态 \\texttt{internal} 作用域在当前编排器中走 basic 组合，使用 \\texttt{global\\_budget} 的 $4.0,4\\times10^{-6}$；显式 basic 模式亦使用这组值。两组 delta 含义不同，不能相加或互换。单次 \\texttt{default\\_query\\_budget.delta} 只用于机制标定；持续观测使用 RDP 路径及其配置。\n\n\\subsection{多接收方权限与作用域配置 (recipients.yaml)}')
p.write_text(s,encoding='utf8')

# Programmatic Table 4: values originate in recalculate.py, never handwritten.
audit=json.loads((P/'revision/numerical_audit.json').read_text(encoding='utf8'))
p=P/'sections/06_gaussian.tex';s=p.read_text(encoding='utf8')
start=s.index('\\begin{tabularx}',s.index('\\label{tab:sigma_comparison}'));end=s.index('\\end{tabularx}',start)+len('\\end{tabularx}')
labels=['User Prevalence','Normalized Share','Interaction Volume','Success Rate','Top-K Tasks']
params=[r'$k=9$',r'$\|x_u\|_2\le1$',r'$C=20$',r'$C_{\text{rate}}=20$',r'$k_t=5$（保守）']
table=r'''\begin{tabularx}{\textwidth}{Xcccc}
\toprule
\textbf{统计指标} & \textbf{配置} & $\boldsymbol{\Delta_2}$ & $\boldsymbol{\sigma_{\rm classical}}$ & $\boldsymbol{\sigma_{\rm analytic}}$ \\
\midrule
'''
for label,param,row in zip(labels,params,audit['gaussian']):
    table+=f"{label} & {param} & ${row['sensitivity']:.4f}$ & ${row['classical']:.4f}$ & ${row['analytic']:.4f}$ \\\\\n"
table+=r'\bottomrule'+'\n'+r'\end{tabularx}'
s=s[:start]+table+s[end:]
s=s.replace('约 $19.5\\%$（噪声方差降低逾 $35.2\\%$）','约 $20.2711\\%$（噪声方差降低约 $36.4330\\%$）')
s=s.replace('其噪声矢量范数增长与维度 $d$ 的解耦特性更契合高维统计需求','其每坐标噪声尺度由 $L_2$ 敏感度控制；总噪声范数通常仍随 $\\sqrt d$ 增长')
s=s.replace('在初始包围区间 $[10^{-5}\\Delta_2, \\, \\sigma_{\\text{classical}} \\times 2.5]$ 内求解精确根 $\\sigma^*$，收敛绝对容差设为 $10^{-12}$。该机制不仅对任意 $\\varepsilon > 0$ 与 $\\delta \\in (0, 1)$ 均全局成立，而且从数学上保证了在满足给定 $(\\varepsilon, \\delta)$ 保证下的高斯噪声方差达到理论极限最小值。',
'下界为 $10^{-5}\\Delta_2$，上界取 $\\max(2.5b,100\\Delta_2)$，其中 $b=\\Delta_2\\sqrt{2\\ln(1.25/\\delta)}/\\min(\\varepsilon,1)$；绝对及相对容差均为 $10^{-12}$。解析条件适用于任意 $\\varepsilon>0$，但代码采用浮点求根，不宣称极端参数下的数值完备性。最小方差结论限于给定敏感度的各向同性高斯机制族，不是所有 DP 机制的全局最优。')
s+='\n表中数值由 \\texttt{revision/recalculate.py} 调用项目函数并以独立 CDF 求根交叉核对。单位敏感度经典系数为 $5.2988025269$，解析系数为 $4.2246788893$。降幅分别按 $1-\\sigma_a/\\sigma_c$ 与 $1-(\\sigma_a/\\sigma_c)^2$ 计算。当前 $T=3$ 的 Top-K 紧界为 $\\sqrt3$，但实现保守使用 $\\sqrt5$，故表格保留实际校准尺度。\n'
p.write_text(s,encoding='utf8')
