# MixedPara Probe 迁移约束

本文件作用于整个仓库。涉及 MixedPara、MegaLens、Probe、Trace 或并行分析的修改，必须遵守
以下规则。

## 0. 当前交付范围：Probe

- 当前迁移主线交付 probe producer、事件与字段约定、scope/lifecycle 边界、trace-off
  开销约束、Trace 聚合与加载、并行分析、采集结果检查和真实训练环境验证。
- PIG、ProgressiveDecoupler、Mitigation、typed PIG projection 以及为其服务的新
  consumer/reducer 不进入当前路线。
- 保留 MixedPara 已有的 PP、DP、TP、EP 分析，以及用于组合这些结果的 Hybrid Analysis。
  当前不增加新的诊断公式、图结构、ranking、mitigation 或 analyzer feature。
- event catalog、scanner 和机器可检查清单只维护 probe 可采集性、事件角色、字段一致性和
  已保留分析器所需的部分，不单独建设面向 PIG 的消费架构。
- 下一任务优先选择能够在目标训练路径上产生真实 trace 证据的 probe 切片；CPU/static
  测试用于锁定调用边界，GPU/NCCL 结论必须来自对应硬件运行。

## 1. 原始设计优先

- 在不破坏 Megatron-LM-FL 既有训练正确性、公开 API 和后端兼容性的前提下，优先还原
  MixedPara 的原始设计。
- 默认参考源码 checkout 为 `/home/chlience/flagos-project/MixedPara`；开始实现前，确认该
  checkout 与仓库内机器清单记录的 baseline revision 一致。
- 开始实现前，锁定仓库内机器清单记录的 MixedPara baseline、源文件、符号和可观察行为。
- 默认保持原版的模块边界、调用顺序、事件名与字段、scope 边界、时间语义、聚合方法、公式、
  阈值、状态传播、返回结构、异常行为和报告副作用。
- 来源一致迁移中不得依据猜测增加 probe、字段、指标、公式、collective、wait、stream
  dependency、synchronize 或训练功能。缺少证据的能力使用 `partial`、`unavailable` 或
  `unknown + reason` 表达。

## 2. 迁移与修订分离

- 来源一致迁移、兼容适配、语义修正、FL 路径覆盖、观测扩展和延期能力必须明确分类。
- 能够独立提交时，先建立来源一致基线，再用后续小提交完成兼容适配或语义修正，避免把目标
  扩展混入“与原版一致”的声明。
- 只有以下依据能够支持偏离原版：

  1. 可定位的 Megatron-LM-FL API、backend、topology 或构建差异；
  2. 可复现的正确性或语义问题；
  3. 明确的 consumer、真实 backend 或用户批准的产品需求。
- 修订保持最小范围。若证据不足以确定行为，保留现有兼容路径并记录边界；会显著改变迁移方向
  时先请求用户确认。

## 3. 保留可审查的设计 Diff

每一项偏离 MixedPara 的修改都必须在实现前或同一任务内留下可审查的设计 diff，并持续更新
仓库外的 `/home/chlience/flagos-project/docs/adaptations.md`。事件、字段和生命周期语义同步
更新 `/home/chlience/flagos-project/docs/event-reference.md`；机器可检查的 producer、
consumer、事件集合和目标新增事件同步更新仓库内
`tests/megalens/fixtures/probe_scan_gate.json` 及对应 scanner 测试。

设计 diff 至少包含：

- Source：MixedPara baseline revision、文件、符号和原始行为；
- Target constraint：Megatron-LM-FL 的实际差异、复现或调用路径；
- Delta：逐项说明新增、删除或改写了什么，并标注差异分类；
- Reason：为什么需要该差异，以及原样照搬会产生的具体结果；
- Invariants：参数、返回、异常、调用顺序、collective/wait/sync 数量、tensor 生命周期和
  trace-off 行为中必须保持的部分；
- Alternatives：评估过但未采用的方案及原因；
- Validation：source/target differential、定向回归、受影响 gate、scanner、wheel 或 GPU
  证据；
- Residual boundary：尚未证明的能力、兼容妥协和后续依赖。

设计 diff 必须能够让 reviewer 区分以下三类内容：原版 MixedPara 行为、Megatron-LM-FL 必需
适配、目标侧新增能力。提交说明、测试名称和 Roadmap 验收项应能定位到同一能力边界。

## 4. Roadmap 编号与提交格式

- 功能、测试和构建提交必须对应仓库外
  `/home/chlience/flagos-project/docs/roadmap.md` 中的一个独立验收项。
- `G*` 用于训练路径 Probe，`A*` 用于 Trace 聚合与并行 Analysis，`V*` 用于扫描、门禁、
  训练配置和证据清单，`E*` 用于打包、FlagScale 覆盖层和工作环境。
- 一个提交默认只包含一个验收编号及其直接回归。纯文档和治理调整使用 `[META]`，不得混入
  功能实现或测试语义修订。
- 提交标题使用 `[编号]` 前缀；提交信息先写中文摘要，再写英文摘要，保持两个独立段落：

  ```text
  feat: [G4.2] 保持原生 P2P Work 生命周期

  feat: [G4.2] Preserve the native P2P Work lifecycle
  ```

## 5. 功能优先与适度验证

- 默认工作场景为受控的开发、CI 和训练环境。当前任务优先闭合用户指定的真实功能、来源一致性、
  兼容性和可复现运行路径，并保持实现范围最小。
- 正确性、数据安全、已有公开接口和已经批准的 fail-closed 协议继续作为必要约束。新增防护需要
  至少一项依据：可复现故障、实际外部信任边界、明确 consumer、生产事故或用户批准的需求。
- 缺少上述依据时，不为恶意输入、身份伪造、极端 TOCTOU、罕见信号窗口、路径攻击或其他假设性
  对抗场景增加复杂状态机、多层校验和额外抽象。此类建议记录到 residual boundary，不阻塞当前
  功能切片。
- code review 先区分功能阻断、语义偏差、兼容问题和可选加固。只有前三类默认进入当前修复范围；
  可选加固需要说明实际收益、复杂度和触发条件，再决定是否排期。
- 优先使用能够直接证明功能的最小验证集：真实调用路径、定向回归和必要的集成运行。除非用户
  明确要求 TDD，默认先完成实现，再补与实际行为对应的测试。

## 6. 验收要求

- 对来源一致部分，优先提供 source/target differential、AST/签名/默认值比较或等价的行为回归。
- 对修订部分，验证修改只影响已声明的证据路径，并保持训练调用、返回、异常、同步数量和
  trace-off fast path。
- CPU/static 证据不能外推为 GPU、NCCL、stream completion 或物理通信时长结论；缺少运行时
  证据时明确记录 capability status 和 reason。
- 提交前检查 staged diff，确保只包含当前设计 diff 对应的文件；完成后更新当前验证结果、
  Roadmap 状态、残余边界和下一动作。文档不记录提交时间线或一次性运行路径。
