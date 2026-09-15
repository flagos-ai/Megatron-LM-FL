# Megatron-LM-FL Upstream Upgrade Skills

这组 skills 用于把 Megatron-LM-FL 升级到任意已固定 SHA 的 NVIDIA Megatron-LM release，同时保留和验证 FL fork 的 plugin、侵入式功能、多芯片平台、override、CI/CD、测试、训练与打包能力。

skills 不绑定具体 release、PR、vendor 列表或一次性路径。每次升级都从 Git refs 和动态发现结果生成独立 artifacts。FlagScale Begin、FlagScale End、FlagScale Add 等注释只作为所有权提示，不是 FL 修改的完整边界。

## 核心原则

- 先分类和决策，后修改源码。
- 区分 history_base、sync_tree_base、release_base、fork 和 target。
- 从 target 开始集成，只重放经批准的 FL 语义；禁止用旧 fork 文件或整棵目录覆盖 target。
- 提前 cherry-pick 的上游 commit/PR 必须做 provenance 判断。
- clean merge、语法通过、import 成功或 marker 数量相等都不是语义兼容证明。
- 不可用硬件、外部 CI、数据、checkpoint、registry 或 scheduler 必须记录为 owned external gate。
- 新情况无法由现有方法表达时，重新打开 skill gap 并更新 skill。
- 未经授权，不创建集成分支、不修改源码、不触发 CI/硬件任务，也不 push 或创建 PR。

## Skill 目录

| Skill | 负责的问题 | 主要交付 |
|---|---|---|
| [mg-classify-fork-delta](mg-classify-fork-delta/SKILL.md) | 固定多层基线，盘点 fork delta、provenance、动作和问题域 | inventory、decisions、routing、gap ledger |
| [mg-integrate-upstream-conflicts](mg-integrate-upstream-conflicts/SKILL.md) | 分析 fork 与 target 同时修改的文本和语义冲突 | conflict ledger、resolution decisions |
| [mg-audit-plugin-overrides](mg-audit-plugin-overrides/SKILL.md) | 审计 overridable surface、registry、vendor 和签名漂移 | override audit、matrix、decisions |
| [mg-preserve-platform-patches](mg-preserve-platform-patches/SKILL.md) | 审计 PlatformBase、多芯片实现和裸 CUDA/device 假设 | capability matrix、device allowlist |
| [mg-integrate-runtime-features](mg-integrate-runtime-features/SKILL.md) | 按完整功能链处理侵入式 runtime 修改 | feature ledger、lifecycle、redesign decisions |
| [mg-audit-training-integration](mg-audit-training-integration/SKILL.md) | 审计参数、配置、构造、训练循环和 checkpoint/output | training lifecycle ledger |
| [mg-integrate-build-packaging](mg-integrate-build-packaging/SKILL.md) | 审计 pyproject、MANIFEST、依赖、Docker、wheel/install | build audit、validation gates |
| [mg-audit-cicd](mg-audit-cicd/SKILL.md) | 静态审计 workflow、config、runner、引用和 failure masking | CI audit、workflow matrix |
| [mg-run-upgrade-test-matrix](mg-run-upgrade-test-matrix/SKILL.md) | 生成并执行按能力和硬件分层的测试矩阵 | test matrix、logs、external gates |
| [mg-fl-upstream-sync](mg-fl-upstream-sync/SKILL.md) | 校验 handoff、编排集成、处理 repository support | intake validation、execution ledger |

每个 SKILL.md 是对应 skill 的权威执行规范；references/ 保存详细契约，scripts/ 提供确定性发现、决策合成和 validator，agents/openai.yaml 提供调用元数据。

## 完整流程

### 1. 固定升级边界并分类

使用 mg-classify-fork-delta。

输入包括历史 ancestry 证据 history_base、上次同步使用的精确 NVIDIA tree sync_tree_base、上一个正式 release_base、当前 fork 和目标 target。

步骤：

1. 解析 refs 为完整 SHA，记录 worktree、remote、tag 和 ancestry。
2. 计算 sync_tree_base..fork 的完整差异。
3. 动态发现 plugin、override、platform、vendor、FlagScale marker、CI、测试、训练和打包 surface。
4. 分析提前 cherry-pick/backport 的 upstream provenance。
5. 为每个 delta 选择 UPSTREAM_ONLY、UPSTREAM_COVERS、UPSTREAM_PLUS_FL_DELTA、REPLAY_FL、REDESIGN 或 MANUAL。
6. 分配一个 primary domain 和零到多个 secondary domains。
7. 生成 skill-gap ledger；未知情况不能为了归零被塞入相近类别。

交付 inventory、provenance、effective decisions、domain routing、skill gaps 和设计基线。验收要求每个 delta 恰好一次，unclassified=0、unrouted=0、refs 一致且 worktree 不变。

### 2. 分析 upstream 冲突

使用 mg-integrate-upstream-conflicts。

1. 以 sync_tree_base、fork、target 做三方 merge-tree。
2. 区分文本冲突、自动文本合并和 clean merge 下的语义风险。
3. 逐路径记录 FL invariant、target 变化、符号、provenance、策略、owner 和 observing test。
4. 邀请 secondary domain handlers 参与审查。
5. 对 P0 与 REDESIGN 要求显式审批。

验收要求所有 both-changed 路径闭合，禁止 blanket ours/theirs 和 whole-file replacement。

### 3. 审计 plugin 和 override

使用 mg-audit-plugin-overrides。

1. 通过 Git object 和 AST 动态发现 decorator、registration、registry key、vendor 和 fallback。
2. 比较参数、默认值、同步/异步、static/class method、继承和 binding。
3. 检查 target 被删除、移动或重设计的情况。
4. 为每个 registry identity 记录 disposition、owner 和 focused test。

验收要求不存在未解释的 missing、stale、duplicate 或 signature mismatch，所有动态 vendor 都被覆盖。

### 4. 审计多芯片平台层

使用 mg-preserve-platform-patches。

1. 动态发现 PlatformBase、实现、注册 key、选择顺序和 cur_platform consumers。
2. 比较方法/property、参数、默认值、继承和 capability。
3. 扫描 torch.cuda、CUDA、NCCL、NVTX 和 device/RNG/stream/memory/graph 假设。
4. 将真正 CUDA-specific 的位置放入细粒度 reviewed allowlist。
5. 为每个 vendor 建立 test 或 external hardware gate。

验收要求平台合约与设备假设都有解释，禁止字符串式批量替换。

### 5. 恢复侵入式 runtime feature

使用 mg-integrate-runtime-features。

1. 把多文件 delta 重组成语义 feature，不按文件或 marker block 生硬重放。
2. 追踪配置、构造/import、dispatch/schedule、execution/communication、output/checkpoint/metrics 和 observing test。
3. 建立 definition/import/call 与 feature dependency。
4. 对 feature split、merge 和 redesign 使用独立人工 decision。
5. 从 target 架构重放最小 FL 行为。

验收要求每个 feature 有 invariant、owner、target change、策略、依赖、failure mode 和测试或 external gate。

### 6. 审计训练集成

使用 mg-audit-training-integration。

1. 对比 public import、argument declaration/validation 和 config propagation。
2. 检查 model、optimizer、scheduler、data 和 parallel-state 构造。
3. 检查 train/eval、pipeline schedule、precision 和 distributed execution。
4. 检查 checkpoint、metrics、logging 和 output 副作用。
5. 绑定 unit、integration、functional test 或 external gate。

验收要求每条训练路由都有 invariant、target relationship、strategy、owner 和 observing evidence。

### 7. 审计 build 和 packaging

使用 mg-integrate-build-packaging。

1. 对比 pyproject 的 build system、metadata、dependencies、extras 和 tool sections。
2. 审计 MANIFEST、package inclusion 和 plugin import surface。
3. 审计 Docker base/stage/args、shell entrypoint 和测试安装说明。
4. 区分 source inclusion、runtime/build/optional dependency、镜像 provisioning 和 test-only asset。
5. 在获批隔离环境安排 editable install、wheel content、clean import 和 image gates。

静态审计不等于 build 成功；每行必须有 owner、requirement、策略和验证 gate。

### 8. 审计 CI/CD

使用 mg-audit-cicd。

1. 静态解析 trigger、permission、job、runner、matrix、reusable workflow 和 secret surface。
2. 解析本地 action、脚本、配置、Dockerfile 和测试引用，区分外部、动态和生成文件。
3. 动态关联平台注册与 CI hardware config。
4. 检查 YAML、shell、continue-on-error、failure masking、ignored tests 和条件 job。
5. 不触发远程 workflow。

验收要求 CI 路由及 backend/config 差异均有 owner/disposition；动态引用与 failure masking 不被静默忽略。

### 9. 生成并执行测试矩阵

使用 mg-run-upgrade-test-matrix。

发现阶段动态盘点 unit groups、functional families、recipes、golden environments、platform tests、CI hardware configs 和 FlagScale E2E，并按 static、import/CPU、focused unit、single accelerator、distributed、functional/golden、FlagScale E2E 分层。

每行记录硬件、进程数、数据/checkpoint/service、命令模板和 evidence。未授权或不可用条件保持 external-gate 或 not-run。

执行阶段需要单独授权，并记录 commit、环境、时间、命令、exit code、日志和 artifacts。只有 exit code 与日志共同证明的结果才可以是 pass。

### 10. 编排、审批和实际集成

使用 mg-fl-upstream-sync。

1. 校验 classifier bundle、effective decisions、routing、skill gaps 和 approval hashes。
2. intake validator 通过且用户授权后，才从 target SHA 创建隔离分支。
3. 按 plugin/override、platform/vendor、runtime、training、build/package、CI/CD、tests、repository support 分批集成。
4. 每批更新 execution ledger 并运行 observing tests。
5. repository support 处理 README、PR template、ignore rule、maintenance script、recipe 和 golden value 等跨域资产。
6. 出现新方法缺口时停止集成、更新 skill 并重跑相关阶段。
7. 最终重跑受影响 audits、routing、gap detection 和 test matrix。

验收要求所有 delta ID 有应用状态和证据；本地 gates 通过；外部 gates 有 owner；merge、push、PR 仍需分别授权。

## Artifact 与决策模型

每个专项通常使用三层 artifact：

1. Candidate facts：脚本从固定 Git objects 自动生成，不人工修改。
2. Reviewed decisions：人工记录 owner、invariant、strategy、reason、evidence、test 或 external gate。
3. Effective ledger：compose script 合并 facts 与 decisions，再由 validator 验收。

这种分离能区分“本次升级需要人工判断”和“skill 方法没有覆盖”。

建议把 artifacts 放在仓库外，例如：

    /share/project/zhaoyingli/flagos/mg-temp/<upgrade-name>/

不要把分析、日志、wheel 或 image artifacts 写入源码树。

## 审批门

| Gate | 允许动作 | 进入条件 |
|---|---|---|
| A：分类与审计 | 只读 Git 分析、生成 artifacts、更新 skills | refs 固定，源码不变 |
| B：决策审批 | 填写 decisions 和 external gates | candidate facts 稳定且 skill gaps 已处置 |
| C：源码集成 | 创建隔离分支、分批修改源码 | intake 对精确 hashes 通过，用户明确 GO |
| D：外部执行 | 安装依赖、使用硬件/scheduler、触发 CI | 用户批准具体环境和命令范围 |
| E：发布 | merge、push、PR、tag、release | 最终 diff、execution ledger 和测试证据获批 |

一个 Gate 的授权不会自动授权后续 Gate。

## 当前 core_v0.18.2 实例

Artifacts 位于：

    /share/project/zhaoyingli/flagos/mg-temp/core-v0.18.2-analysis/

最终 handoff 位于：

    /share/project/zhaoyingli/flagos/mg-temp/core-v0.18.2-analysis/final-handoff/

当前 mg-skills-v1.1：

- 10 个正式 skills，31/31 项检查通过；
- 284/284 个 fork deltas 已路由；
- unrouted=0、conflicting_routes=0；
- skill gaps=0；
- source worktree未被 skill 建设修改。

这表示通用方法已覆盖，不表示实际升级已经完成。当前 intake 仍因 bundle is not approved 关闭；审批前不得创建集成分支或修改源码。

重点查看 final-handoff 下的 final-coverage-report.md、manual-decisions.md、repo-upgrade-runbook.md、go-no-go-checklist.md 和 approval-template.json。

## 维护规则

- 新增或修改 skill 后运行 skill-creator validator、Python compile 和回归测试。
- 更新 mg-classify-fork-delta/references/skill-coverage.json 的版本与 evidence，再重新生成 routing。
- 新模式影响决策正确性、验收可信度或可能复现时，必须更新 skill 并加入回归场景。
- 一次性仓库异常记录在本次 artifacts 中，不硬编码进通用规则。
- 单个 skill 内不增加 README、CHANGELOG 或安装指南；详细规范放在 SKILL.md 和 references/，套件级说明维护在本文件。
