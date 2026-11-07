# 导出与模型注册表：设计与实施

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 状态 | as-built 设计 + 整改实施设计（待实施） |
| 日期 | 2026-11-06 |
| 审计基线 | v0.6.0 工作区；行号为该快照，实施时以符号名为锚 |
| 读者 | 框架研发、评审人、MLOps / 发布工程 |
| 关联 | [总览与路线](overview.md)、[架构总览](../framework_design.md)、[加固计划](../hardening_plan.md)（H1-6/H1-7/H1-8 及其导出面条目已实施项不重复）、[已知限制](../known_limits.md) |

## 1. 范围与验收判定

范围：`src/autovision/export/`（ONNX/int8/TensorRT/OpenVINO 管线、
parity 门禁、工件元数据、benchmark）与 `src/autovision/registry/`
（模型注册表、部署门禁、downloadable、GTR/自持资产）。
判定：**优秀工程库，补 1 个 P1 与关键 P2 后可验收**。"临时文件 +
分层门禁 + `os.replace` 发布"贯穿四条导出路径且被测试逐条钉死（含
ENOSPC）；哈希链审计、逐字节晋升校验、钉住 sha256 下载均扎实。

## 2. 现状设计（as-built）

### 2.1 导出管线（八步）

`ONNXExportMixin.export_onnx`（`src/autovision/export/export_onnx.py:389`）
内层 `_trace_and_write_onnx`（同文件 `:523`）按序：前置 blocker 扫描
（`export_blocker` / `dynamic_spatial_blocker`，动态轴 legacy 拒绝）
→ eval/fp32 → `deploy()` → 导出安全模块交换 → `torch.export` 严格
trace（batch 维 Dim 上限 128）→ 失败时静态轴降级 legacy TorchScript
→ FP16 转换（ORT block list + 精度策略 + 去重 cast + 拓扑重排）→
onnxslim → value_info 修复 → 元数据写入（真实 opset 回读）→ 验证
门禁 → 截断守卫 + 孤儿清扫 → `os.replace` 原子发布。任何一步失败
清理临时图、旧工件完好。

### 2.2 parity 门禁

| 门禁 | 容差 | 说明 |
|---|---|---|
| fp32 ONNX | atol 5e-3 / rtol 1e-1 | `src/autovision/export/parity_checks.py:30` |
| int8 ONNX | atol 2.0 / rtol 0.5 | 只抓结构性破坏 |
| TRT 引擎 | 按构建决策点解析出的实际精度选档（H3-4a 已实施） | 张量级 + e2e 最终框（box 1.0px） |
| fp16 | CUDA EP 首位（H1-8 已实施）；无 CUDA EP 时降级"执行+有限性"冒烟并告警 | 本文 EXPORT-02 处理其残留缺口 |

三种比较不变量按解码形态分派：稠密逐元素 / in-graph top-k 用行和
（置换容忍）/ 无选择 query 贪婪行配对。

### 2.3 注册表与下载

- 布局 `{root}/{dev,staging,prod}/{name}/{version}/`，manifest 最后
  原子写，`path` 读取时防目录逃逸；注册/晋升对已存在条目拒绝
  （不可变）；晋升副本逐字节 sha256 校验；门禁针对晋升后副本求值，
  失败 rmtree + `promote_failed` 审计（H3-4j 已实施）。
- `current.json` 指针原子替换；审计日志 sha256 哈希链 + fsync，
  读取时全链验证；`@_locked` = FileLock(30s) + 线程 RLock。
- 门禁协议 `check(entry) -> list[str]`；默认 Sha256/ModelCard/
  License/Benchmark/Parity；`gates=[]` 为显式旁路。
- 下载：每目标 FileLock → (path,size,mtime,sha) 记忆化复验（H3-4i
  已实施）→ `.part` + fsync + `os.replace` → sha 不匹配删文件换
  mirror；GTR/自持资产整族 sha256 钉死。

## 3. 关键流程

- **export 命令**：`out` 后缀分派 → 写 `{stem}.{uuid}.tmp.onnx` →
  八步管线 → 失败 `_drop_onnx_artifacts`（旧工件不动）→
  `_move_onnx_artifacts` 发布 → sidecar 写入。
- **promote**：目标 stage 校验 → 一站定位 → 不可变检查 → copy2 →
  sha 逐字节比对 → sidecar sha 比对 → 逐 gate 求值（异常折算失败）→
  失败 rmtree + 审计 + raise → manifest → current 指针 → 审计。
  全程持锁。
- **下载**：注册名 → 缓存路径 → 锁 → 记忆化命中直返 → 损坏删重下 →
  全失败返回最后一个错误。

## 4. 审计发现清单

### 4.1 P1

| 编号 | 问题 | 证据 |
|---|---|---|
| EXPORT-01 | 注册/晋升在"目录已建、manifest 未写"窗口被杀 → 残留目录触发 "versions are immutable" 拒绝，该 (name, version) 永久无法重试，只能人工清理；错误信息还把用户引向"不可变"的错误结论（`_list_versions` 按 manifest 过滤会跳过残目录，显式 `get()` 却报 Corrupt，行为自相矛盾） | `src/autovision/registry/model_registry.py:216-221`、`:223`、`:284-289`、`:500-503` |

### 4.2 P2

| 编号 | 问题 | 证据 |
|---|---|---|
| EXPORT-02 | fp16 在无 CUDA EP 机器上 verify=True fail-open：降级冒烟门禁后工件照常发布，无机器可读的"验证强度"标记，下游无法区分全量 parity 与 smoke-only | `src/autovision/export/export_onnx.py:843-859`、`:951-980`；`src/autovision/export/artifact_metadata.py:77-89` |
| EXPORT-03 | 注册表 `_copy_sidecars` 只复制三种后缀，TRT 的 `.meta.json` 与 OV 的 `<stem>.json` 身份 sidecar 被丢弃——从注册表路径加载时身份门禁整体消失且无告警 | `src/autovision/registry/model_registry.py:560-570`；`src/autovision/export/export_onnx.py:1477-1492` |
| EXPORT-04 | 注册表读路径全程持锁做全量 sha256（2GB 工件 4–10 秒，写者全阻塞、30s 锁超时伪失败），且无 digest 记忆化（downloadable/user_held 均有，策略不一致） | `model_registry.py:132-159`、`:512-536` |
| EXPORT-05 | dynamic-spatial 第二探针输入未种子化，违背同函数第一个比较输入"种子随机"的注释论证，门禁可偶发误报 | `src/autovision/export/export_onnx.py:914-925`、`:938-942` |
| EXPORT-06 | int8 parity 门禁 `zip(fp32_outputs, int8_outputs)` 对输出数不匹配静默截断；`equal_nan=True` 让 NaN 输出过闸（TRT 侧对应检查是显式 raise） | `export_onnx.py:1122`、`:1130-1134` |
| EXPORT-07 | `gates=[]` 旁路在审计日志不可见：成功审计 details 只有 `from_stage`，"explicit, audited bypass" 的承诺未兑现 | `model_registry.py:265-267`、`:358-362`、`:641-651` |
| EXPORT-08 | `register_downloadable` 名称未做路径校验，含 `..` 的名称可穿越缓存目录（对比模型注册表有 `_validate_key_part`） | `src/autovision/registry/downloadable.py:83-86`、`:130` |
| EXPORT-09 | 工件已发布、sidecar 未更新的竞态窗口（三类载体共有）；OV 场景旧 `.json` 还是身份载体，窗口内并发读者会拿旧身份判新工件 | `export_onnx.py:438-443`、`:1436-1437`；`src/autovision/export/openvino_helpers.py:168-187` |
| EXPORT-10 | >2GB external-data 图会被 value_info 修复/元数据写入的内联加载破坏（管线实际不支持自己注释里承认存在的图类，fail-closed 但报错晦涩） | `export_onnx.py:1532-1538`、`:1559-1570`、`:116-143` |

### 4.3 P3

| 编号 | 问题 | 一句话方向 |
|---|---|---|
| EXPORT-11 | TRT 失败时 staged 图刻意保留，与旧好引擎同 stem 共存；staged 名不含 pid 并发互踩 | 失败也清理 + 名带 pid |
| EXPORT-12 | 动态 batch 上限 128 未在 API 文档标注 | docstring + 报错指引 |
| EXPORT-13 | benchmark p90 取索引偏移（n=20 时近 p95）；robust 方法论不覆盖 ORT 路径 | 最近秩 p90；robust 贯通 |
| EXPORT-14 | CLI `register metadata=<字符串>` 时 `dict(...)` 裸崩 | CLI 解析层校验 |
| EXPORT-15 | `list_entries` 对单个损坏条目无容错 | 坏项跳过 + warning |
| EXPORT-16 | `GTR_ASSET_ROOT` import 时固化 env；HF revision 未钉；`verify_gtr_asset` 无记忆化 | env 延迟读取；钉 revision；补记忆化 |
| EXPORT-17 | GTR 默认候选含 research 树路径，与 docstring "intentionally not shipped" 矛盾 | 移除默认候选或改文案 |
| EXPORT-18 | OFFLINE 检查之前已删校验失败的缓存文件（离线下破坏性清除） | OFFLINE 分支保护现场 |
| EXPORT-19 | 单 URL 下载无重试/退避；无 Content-Length 校验与总量上限；`.lock` 永不清理 | 有限重试 + 上限 + 锁清理 |
| EXPORT-20 | register 对源文件无稳定性检查（可登记写一半的快照）；告警集合无界；审计追加 O(n²) | 稳定性检查 + 有界 + 追加定位优化 |
| EXPORT-21 | promote 成功后无条件移动 current 指针（旧版重晋升会回拉生产指针），与 register 策略不对称 | 文档化语义或加 `set_current=False` |
| EXPORT-22 | `_verify_onnx` 对 self 做 deepcopy，大模型导出内存翻倍 | 评估去掉或 lazy device move |

## 5. 整改实施设计

> 状态：**EXPORT-01..10、18、19 已实施 + P3 落地 9 项（13/14/15/16/
> 17/20/21/22 + VerifyStrengthGate），随 0.6.1 发布**；P3-11 登记为
> 设计冲突项、P3-19/24 登记为性能批次（见下）。
> 实施与设计的偏差（复审轮次修正）：① EXPORT-03 的 sidecar 名必须
> 对齐**加载器**的 `with_suffix` 语义（`model.engine` →
> `model.meta.json`）——设计文档写的 `<engine>.meta.json`（str+suffix）
> 是加载器永远找不到的名字，复现时纠正；② EXPORT-09 的 TRT 分支按
> write_artifact_metadata 的 with_suffix 语义落地为"sidecar 先行发布到
> 最终名 + 旧 sidecar 备份恢复"，而非设计里的 staged→rename 搬运；③
> EXPORT-10 从管线入口检查改为 onnxslim 后按 .data 实际体积门控

> （dynamo 在 trace 时合法地把全部初始化器外置，入口检查会误杀
> 正常导出）；④ **EXPORT-11 登记为设计冲突**：失败也清理 staged 图
> 与 H3-4b 的已测试决策（失败保留供调试，test_deployment.py
> TestTensorRTStagedOnnx 钉住）直接矛盾——pid 命名部分同样与
> debug-adjacent 命名冲突，均未实施，需先重设计 debug 工作流。
>
> **实施后复审结论：复审判定"需修复后合入"，全部 P1/P2 已闭环。**
> 复审实证：① P1-1——TRT 备份恢复被逐迭代快照打穿（第 2/3 次
> publish 前重拍读到新字节覆盖旧身份备份，失败恢复写回新身份配旧
> 引擎）；已改为**单次快照、循环外**。② P1-2——`write_spec` 公共
> 参数被内部调用硬编码覆盖（TRT/OV/int8 中间工件残留 decode
> contract）；已改为转发调用方值。③ P2-1——`verify=False` 的导出
> 元数据记 `"full"`；已引入第三态 `"unverified"`（gate 与消费告警
> 同步）。④ P2-2——EXPORT-02 零测试；已补 fp32/full、verify=False/
> unverified、gate 无强度拒绝与 TRT 备份恢复四个测试。⑤ P2-3——
> 0 字节 audit.jsonl 触发 UnboundLocalError 且发生在 manifest 写入后
> （半完成注册）；已初始化 non_empty 并补测试。⑥ P2-4——ONNX
> `_move_onnx_artifacts` 与 OV json 发布的失败路径无旧 sidecar 恢复；
> 已补 spec/json 的备份恢复。⑦ P3-a/b/c/d/e/f + EXPORT-01 gc 子命令
> 与 EXPORT-04 list verify 参数的偏差已在偏差清单登记或转排期。

### 5.1 EXPORT-01 注册表孤儿条目自愈（P1）

**设计**：

1. 判重依据从"目标目录存在"改为"目标目录存在 **且**
   `manifest.json` 可读"：manifest 存在 → 维持不可变拒绝（现行为）；
   目录存在但 manifest 缺失 → 视为孤儿条目。
2. 孤儿处置分两态（复用 `src/autovision/runtime/fs_utils.py` 的龄期
   思想，但以条目目录的 mtime 为准）：
   - 目录最新修改时间早于龄期（建议 10 分钟）→ 判定上个进程已死，
     rmtree 孤儿 + warning，继续本次注册/晋升；
   - 目录新鲜 → 判定并发注册进行中，报 `RegistryError`（消息区分
     "正在进行的注册"与"需要人工清理的孤儿"两种情形并给出路径）。
3. `get()` 对"目录存在但 manifest 缺失"的条目从 Corrupt 改为
   "incomplete entry（可能注册被中断）"，与 `list_versions` 的过滤
   行为对齐。
4. 显式管理出口：`registry gc --dry-run`（CLI 子动作）列出孤儿清单，
   `--apply` 执行清理（审计日志记 `gc` 事件）。

**测试计划**：`tests/test_model_registry.py` 新增：模拟被杀残留
（mkdir + 拷贝一半 + 无 manifest，mtime 回拨过龄期）→ 重试注册成功
且 warning；新鲜孤儿目录 → 报"并发进行中"；`gc --dry-run/--apply`
两态；manifest 存在的重复注册仍拒绝。`tests/test_registry_hardening2.py`
回归。

**验收标准**：kill 场景恢复无需人工；错误信息三态互不混淆。

### 5.2 EXPORT-02 验证强度进元数据（P2）

**设计**：元数据增加键 `autovision_verify_strength`（`full` /
`smoke-only`）：全量 parity 通过 → full；冒烟降级 → smoke-only。
消费侧：`verify_metadata` 读到 smoke-only 时输出一次 warning
（"该工件导出时未做数值对拍，请在 CUDA ORT 上复验"）；部署门禁新增
opt-in 的 `VerifyStrengthGate`（要求 full 才可晋升 prod）。model card
模板展示该字段。

**测试计划**：CPU 环境 fp16 导出 → 元数据含 smoke-only；fp32 →
full；gate 开启时 smoke-only 晋升被拒（`tests/test_registry_gates.py`
新增）。**验收标准**：下游机器可读地区分两种验证强度。

### 5.3 EXPORT-03 sidecar 复制补齐身份载体（P2）

**设计**：`_copy_sidecars` 白名单追加两类：`<artifact>.meta.json`
（TRT）与 `<stem>.json`（OV：xml 的身份 sidecar，仅当工件后缀为
`.xml` 时收集同名 `.json`）；晋升 sha 比对覆盖新增 sidecar。
**测试计划**：注册 `.engine`（带 `.meta.json`）与 `.xml`（带
`.json`）→ 晋升后从注册表路径构造 `InferenceEngine` →
`verify_engine_artifact` 拿到身份（不再整体跳过）；`tests/test_deployment.py`
回归。

### 5.4 EXPORT-04 读路径锁瘦身 + 记忆化（P2）

**设计**：

1. digest 记忆化（downloadable 同款 `(path, size, mtime, sha)` memo，
   进程内 + 与条目 manifest 挂钩失效）。
2. `get()`/`list_entries()` 增加 `verify: bool = True` 参数：False 时
   只读 manifest/指针（锁内毫秒级），哈希移出锁外按需执行；默认
   True 保持现语义。高频读路径（`current()`）内部改走 verify=False +
   记忆化。
3. 文档写明三档校验语义：注册/晋升（强制全量）→ `current()`（记忆化
   复验）→ `get(verify=False)`（信任 manifest）。

**测试计划**：memo 命中时哈希调用计数为 0；`get(verify=False)` 不做
全量哈希（计时/计数断言）；并发读写无死锁（既有并发用例扩展）。

### 5.5 EXPORT-05 探针种子化（P2）

**设计**：第二空间探针输入改用与第一输入同源的
`torch.Generator().manual_seed(0)`；注释补一句"两个输入都从同一种子
派生，保证门禁决策逐 run 稳定"。**测试计划**：连续两次导出同模型 →
探针输入逐位一致（张量哈希断言）；`tests/test_dynamic_spatial_export.py`
回归。

### 5.6 EXPORT-06 int8 门禁补结构性检查（P2）

**设计**：输出数不匹配 → 显式 raise（消息含两侧输出名清单，对齐
TRT 侧语义）；`torch.testing.assert_close` 改 `equal_nan=False`，
fp32 参考含 NaN 时给出"参考图本身产生 NaN（坏模型/坏校准）"的可读
错误。**测试计划**：构造输出数不一致的双图 → 报错；NaN 参考 → 门禁
失败带指引。

### 5.7 EXPORT-07 旁路进审计（P2）

**设计**：promote 审计 details 增加 `gates`（通过的门禁名列表或
`"none"`）与 `bypass`（bool）；`audit_log` 输出透出。**测试计划**：
`gates=[]` 晋升 → 审计条目含 `bypass: true`；正常晋升 →
`bypass: false` 且 gates 全列；审计链验证不受影响。

### 5.8 EXPORT-08 downloadable 名称校验（P2）

**设计**：`register_downloadable` 复用模型注册表 `_validate_key_part`
的校验语义（拒绝 `/`、`\`、`..`、NUL、绝对成分），抽为共享 helper
（放 `src/autovision/registry/` 公共位置）供两处使用。**测试计划**：
`"../x"`、绝对路径名、含 NUL 名 → 注册即报错；合法名回归。

### 5.9 EXPORT-09 工件 + sidecar 同事务发布（P2）

**设计**：统一"sidecar 先行、工件最后"的发布顺序（工件 rename 即
发布点，读者要么看到旧工件 + 旧 sidecar，要么新工件 + 新 sidecar）：
ONNX：postprocess.json 先 replace 再发布图；TRT：`.meta.json` 与
`.postprocess.json` 先 replace 再发布引擎；OV：`.json` → `.bin` →
`.xml`（xml 最后）。各失败路径清理已 replace 的新 sidecar（恢复旧
sidecar 不可能——记录 warning 指引重新导出，但旧工件未被动过，窗口
内身份以旧 sidecar 对旧工件仍然正确）。

**测试计划**：注入"sidecar 发布后、工件发布前失败" → 旧工件 + 旧
sidecar 存活且身份校验一致；成功路径顺序断言（spy replace 调用序）。

### 5.10 EXPORT-10 大图 fail-fast（P2）

**设计**：管线入口检测 external-data（`onnx.load(...,
load_external_data=False)` 检查 initializer 的 data_location）：存在
→ 立即抛可读错误（">2GB external-data 导出暂不支持：管线后处理会
内联整图，请使用 ≤2GB 静态导出或拆分模型"），并在 `known_limits.md`
登记。全量 external-data 支持另立 backlog 条目。**测试计划**：构造
小体积 external-data 模型（几十 MB 即可触发检测分支）→ 导入期报错
而非在后处理深处晦涩失败。

## 6. 实施顺序

EXPORT-01（可用性）→ 03（一行级高价值）→ 02 → 07 → 06 → 05 → 08
→ 04 → 09 → 10。每项独立 commit，标题 `fix(export): EXPORT-nn <摘要>`
或 `fix(registry): EXPORT-nn <摘要>`。

## 7. 风险与回滚

- EXPORT-01 的孤儿清理有误删风险——龄期守卫 + `gc --dry-run` 人工
  确认双保险；回滚 revert 即恢复旧"不可变"语义。
- EXPORT-04 引入 `verify=False` 读路径：默认值不变（True），调用方
  显式选择降级，审计面不受影响。
- EXPORT-09 的发布顺序重排被既有原子性测试
  （`tests/test_export_atomicity.py`、`tests/test_deployment.py` 的
  保旧工件用例）全覆盖，回归不过即回滚。
