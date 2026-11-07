# 训练引擎与任务面：设计与实施

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 状态 | as-built 设计 + 整改实施设计（待实施） |
| 日期 | 2026-11-06 |
| 审计基线 | v0.6.0 工作区；行号为该快照，实施时以符号名为锚 |
| 读者 | 框架研发、评审人、训练平台 |
| 关联 | [总览与路线](overview.md)、[架构总览](../framework_design.md)、[加固计划](../hardening_plan.md)（H1-4/H1-5 及其训练面条目已实施项不重复） |

## 1. 范围与验收判定

范围：`src/autovision/engine/`（双引擎）、`src/autovision/tasks/`
（顶层契约 + 抽查 detection/semantic）、`src/autovision/runtime/`、
`src/autovision/config/`、`src/autovision/optim/`、augment 调度抽查。
判定：**核心路径达标**——task 训练循环、checkpoint 原子性、NaN/OOM
恢复、DDP 验证汇聚、early stopping/resume 状态机均达到企业级水平，
决策几乎都有编号注释与测试（含进程级故障注入）。尚无 P1；4 个 P2
（TRAIN-01..04）修复后整体达标，P3 作为验收后第一迭代。

## 2. 现状设计（as-built）

### 2.1 双引擎

- `Engine.fit`（`src/autovision/engine/engine.py:851` 起）：epoch 制
  预训练循环，面向 PretextRecipe（`forward==training_step`、
  `configure_optimizers`、`clip_gradients`、`on_save/on_load_checkpoint`
  钩子）。
- `TaskEngine.fit_task`（同文件 `:1607` 起）：step 制微调循环，增加
  梯度累积、EMA、周期验证 + best/last 双检查点、early stopping、
  窗口触发的 dataloader 重建。两引擎共享 `_write_atomic`、RNG 捕获、
  NaN/OOM 恢复原语。
- 训练/推理双层契约：`src/autovision/tasks/train_model.py`
  （TrainModel，训练容器）与 `src/autovision/tasks/task_model.py`
  （TaskNetwork，纯推理可导出），互不 import 训练机制。

### 2.2 配置三方言同源

唯一校验入口 `pydantic_model_validate`（`src/autovision/config/validate.py:81`）；
`PydanticConfig` 强制 `extra="forbid"` + strict + validate_assignment
（`src/autovision/config/base.py:41`）。API/CLI/YAML 三条路最终都过
同一 Pydantic args；`no_auto`/`assert_config_resolved` 把未解析
`"auto"` 定义为框架 bug 而非用户错误。

### 2.3 checkpoint 与 resume

- `_write_atomic`（`src/autovision/runtime/checkpoint.py:53`）：
  mkstemp 唯一 tmp、写后 fsync、`os.replace`、目录 fsync、3 次退避
  重试、陈旧 tmp 清扫（`src/autovision/runtime/fs_utils.py:29`，
  1h 龄期保护）。加载 `weights_only=True`，无 pickle 代码执行面。
- fit_task 的 `_restore_last_state`（`engine.py:1880`）恢复模型（含
  FSDP）、optimizer、scheduler、AMP scaler、按 rank 分片的 RNG、
  best_metric 基线（sidecar 优先）、early-stopping 计数、sampler
  epoch、fit_args 漂移检查。NaN 回滚复用同一恢复函数并加
  `validate_finite` 全量巡检。

### 2.4 回调

`CallbackRegistry`（`src/autovision/engine/callbacks.py:41`）：6 个
生命周期事件、每处理器独立 try/except 隔离、仅 rank zero 派发。

## 3. 关键流程

**fit_task 单步**：step hooks → loader 重建（窗口切换）→ 取批 →
zero_grad → 累积窗口（`no_sync` 除最后 microbatch）→ loss/accum →
D32 窗口级腐败判定（MIN all_reduce）→ `scaler.unscale_` → clip →
非 AMP 路径有限性断言 → `scaler.step/update`（skipped 不消耗调度步）→
EMA → 日志窗口 → 周期保存 → 验证 → best/early-stop。

**DDP**：`WORLD_SIZE` 判定 → dataset≥world 守卫 → `init_process_group`
（nccl/hccl/gloo 按设备）→ DDP 包装 → 拒绝自定义 sampler →
DistributedSampler + 每 rank 独立 loader 种子流 → 验证用 strided
sampler 精确划分 + metric state 汇聚 + val loss 分子/分母分别 reduce
→ best 决策 broadcast 对齐 → rank0 写失败经写屏障传播。

**已正确处理的易错点**（不重复展开）：NaN 分级恢复（回滚→预算→loud）、
OOM 降批（DDP fail-fast）、SIGKILL 注入测试、AMP scaler 持久化、
空 val 集合构造期拒绝、AMP overflow 不消耗调度步、batch=1+BN 告警、
梯度累积下无限迭代器保证窗口完整。

## 4. 审计发现清单

### 4.1 P2

| 编号 | 问题 | 证据 |
|---|---|---|
| TRAIN-01 | 预训练 resume 无 fit_args/数据漂移守卫，且 `scheduler.load_state_dict` 把旧 `max_epochs` 一并恢复——续训调大 max_epochs 不生效，余弦静默按旧终点衰减到底停在 end_value | `src/autovision/engine/engine.py:1306-1307`、`:1349-1363`；`src/autovision/optim/schedulers.py:36-39`；对比 fit_task 有 D17 漂移检查（`engine.py:1516`、`:1984`） |
| TRAIN-02 | pretrain 导出走裸 `torch.save` 直写终路径（同语义的 fit_task 导出已按 H1-5 修复为原子写） | `src/autovision/cli/train.py:335-347` vs `engine.py:2181` |
| TRAIN-03 | 多 rank resume 每个 rank 独立读 checkpoint；输出目录非共享盘时 start_step/best_metric 分叉 → 集合通信挂满 timeout 才报错，根因不可见 | `engine.py:1895`、`:1933-1944`、`:1957-1959` |
| TRAIN-04 | `best_metric.json` sidecar 裸 `write_text` 直写终路径（读取端有容错，但违反"所有 durable artifact 走 tmp+rename"的自定纪律） | `engine.py:2752-2756` |

### 4.2 P3

| 编号 | 问题 | 一句话方向 |
|---|---|---|
| TRAIN-05 | warm-start 的 `getattr(train_model, "load_train_state_dict", ...)` 永远落空（该方法只在 TaskNetwork 上）；结构不匹配只抛 torch 原生 size mismatch 长报错 | 删除死 getattr；load 失败时做 key 级 diff 摘要（缺失/多出/形状不匹配三类计数） |
| TRAIN-06 | NaN 守卫每步 `torch.isfinite(loss)` 强制 device→host 同步，与 `_TrainWindow` 注释声称的避免每步 sync 自相矛盾 | 异步标志位或窗口级检查折中（正确性动机保留） |
| TRAIN-07 | `accumulated_loss` 累加后从未读取（死代码） | 删除 |
| TRAIN-08 | 预训练日志消费不对称（rank0+logger_fn 时按 log 点消费，其余 rank 每步消费） | 统一为按 log 点 |
| TRAIN-09 | fit_task 把 step 存进 `CheckpointMetadata.epoch`，语义复用 | 元数据加 `step` 字段，`epoch` 语义归位（跨版本读取兼容注记） |
| TRAIN-10 | 数据指纹每次启动全量 `list(value[:])` 物化再 JSON+sha256，10 万图级启动延迟可观 | 流式哈希或缓存（按 args 指纹记忆化） |
| TRAIN-11 | `_OOM_MAX_ATTEMPTS`/`_EPOCH_ARCHIVE_KEEP` 重复定义两次 | 收敛为单处 |
| TRAIN-12 | resume 一个已完成的 run 静默返回（steps_run=0，无提示） | 输出 info "already completed at step N" |
| TRAIN-13 | 无 SIGTERM/SIGINT 优雅停机（只能靠周期 checkpoint） | 文档化为已知边界；可选 handler 触发一次保存 |
| TRAIN-14 | dataloader worker 崩溃非 OOM 直接 re-raise，整个 run 终止 | 文档化；可选有限重试策略 |
| TRAIN-15 | `Engine.fit` 对 batch_sampler 型 DataLoader 不健壮（len/整除/`batch_size` 属性访问） | 入口类型守卫 + 可读报错 |

## 5. 整改实施设计

> 状态：**TRAIN-01..05、07、09、11、12、15 已实施（工作区，随 0.6.1
> 发布）**；TRAIN-06/08/10/13/14 登记为验收后第一批迭代项（见下）。
> 关键不变量：不改参数的 resume 必须与不中断训练的 LR 轨迹逐位一致
> （golden 测试锚定，tests/test_pretrain_resume.py）。
> 实施与设计的偏差：fit_task 侧调度器形状漂移按"指名 warning"实施
> 而非强制重设——任务调度器是各模型自有的任意对象，形状字段无法像
> CosineWarmupScheduler 一样安全枚举重置；`CheckpointMetadata.step`
> 为新增可选字段，旧 checkpoint 读取为 None。TRAIN-13（无 SIGTERM
> 优雅停机）按设计文档的"文档化"路线落进 docs/known_limits.md。
>
> **实施后复审结论（只读代理复审 + 训练子系统 255 项回归全绿）：可合入，
> 无 P0/P1。** 复审驱动的追加闭环：① TRAIN-03 的对账移出
> `last_path.exists()` 守卫——"单 rank 缺文件"场景（非共享盘）现在同样
> 被对账捕获（缺文件 rank 以 step 0/无基线进入对账，与已恢复 rank 的
> 分叉被指名报错），docstring 同步扩写；计划中的 2-rank gloo 端到端
> 用例登记为验收后第一迭代项。② golden 测试钉住"恢复后首步 LR ==
> checkpoint 保存值"（P3-1，防加载顺序回退）。
> 复审登记的其余 P3（漂移 warning 接线测试、`_write_atomic` 提升为
> 公开导出、fit_args 快照含 lr、指纹记忆化随 TRAIN-10）随批次排期。

### 5.1 TRAIN-01 预训练 resume 补漂移守卫 + 调度器扩展生效（P2）

**问题**：`Engine._build_checkpoint` 的 engine_state 只存
optimizer/scheduler/rng；`_resume` 直接 `scheduler.load_state_dict`
回灌旧 `max_epochs`。

**设计**：

1. engine_state 增加 `fit_args_digest`（normalize 后 fit_args 的
   sha256）与 `data_fingerprint`（复用数据层指纹）。`_resume` 比对：
   不一致 → 逐字段 diff 输出 warning（复用 fit_task 漂移检查的 diff
   helper，消灭两套实现）。换 normalization 续训（现静默）同样被
   diff 捕获。
2. 调度器两段式恢复：先按**当前** fit_args 重建 scheduler，再
   `load_state_dict` 只回灌步计数（`last_epoch` 等非派生字段），
   最后显式重设派生属性（`max_epochs`/`total_steps`）为当前值——
   `CosineWarmupScheduler` 的状态导出相应拆分（计数 vs 派生）。
3. resume 时扩展 `max_epochs` 的行为在 CHANGELOG 标注（0.7.0）：
   "续训调大 max_epochs 后 LR 按新终点继续衰减"。

**涉及文件**：`src/autovision/engine/engine.py`（`_build_checkpoint`、
`_resume`）、`src/autovision/optim/schedulers.py`（state 拆分）。

**测试计划**：resume 不改参数 → LR 轨迹与不中断训练逐位一致（golden
锚定防回退）；resume 时 max_epochs 10→20 → 前 10 epoch 轨迹一致、
11–20 按 20 终点继续衰减；fit_args 漂移 → warning 含 diff 字段名。
`tests/test_checkpoint.py`、`tests/test_engine_train_mode.py` 回归。

**验收标准**：三个新用例通过；预训练路径具备与 fit_task 同级的漂移
可见性。

### 5.2 TRAIN-02 pretrain 导出原子化（P2）

**设计**：`cli/train.py` 导出段改为复用 `_write_atomic`（tmp 命名
沿用 `<target>.tmp-<pid>` 约定，保证 fs_utils 孤儿清扫识别），并照
H1-5 的形状包 try + `_rank_zero_write_barrier`（若在分布式上下文）。
**测试计划**：注入 `torch.save` 抛错 → 目标路径无残留 tmp、旧工件
不变（对齐 `tests/test_engine_cache_and_retry.py` 既有用例形状）。

### 5.3 TRAIN-03 多 rank resume 一致性守卫（P2）

**设计**：`_restore_last_state` 完成后做一次集合通信对账：
`all_reduce(MIN)` 与 `all_reduce(MAX)` 各取 `start_step` 与
`best_metric`，min≠max → 抛 `AutoVisionError`，消息列出不一致字段并
指出最可能原因（输出目录未挂共享盘 / rank 间 checkpoint 视图不一致）。
成本：resume 一次性两次小集合通信，可忽略。**测试计划**：2-rank gloo
（`tests/test_distributed.py` 基建，`AUTOVISION_DDP_TESTS` 门控）注入
单 rank 缺 checkpoint → 快速失败且消息可操作；共享盘正常路径 → 对账
无感通过。

### 5.4 TRAIN-04 sidecar 原子写（P2）

**设计**：`best_metric.json` 写入改走
`src/autovision/runtime/fs_utils.py` 的原子写 helper（tmp + fsync +
`os.replace`）。**测试计划**：写失败注入 → 旧 sidecar 保持可解析；
`tests/test_checkpoint.py` 回归。

## 6. 实施顺序

TRAIN-01（最大，含调度器状态拆分）→ 04 → 02 → 03；P3 中 TRAIN-05
的报错质量与 TRAIN-13 的文档化建议随阶段 3 一并做。每项独立 commit，
标题 `fix(engine): TRAIN-nn <摘要>`。

## 7. 风险与回滚

- TRAIN-01 的调度器两段式恢复是行为修正：不改参数的 resume 必须
  golden 逐位一致（这是首要回归锚）；扩展 max_epochs 的场景从
  "静默停底"变为"按新终点衰减"，CHANGELOG 标注。
- TRAIN-03 在"合法的非共享盘用户"身上会把原来的 30 分钟挂起变成
  快速失败——这是修复目标；错误消息给出共享盘/单 rank 调试两条出路。
- 全部修复不触碰 D32/NaN/OOM 恢复语义，`tests/test_nan_recovery.py`、
  `tests/test_engine_recovery.py`、`tests/test_crash_recovery*.py`
  作为常驻回归。
