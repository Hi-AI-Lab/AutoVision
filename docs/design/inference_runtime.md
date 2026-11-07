# 推理运行时与跟踪：设计与实施

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 状态 | as-built 设计 + 整改实施设计（待实施） |
| 日期 | 2026-11-06 |
| 审计基线 | v0.6.0 工作区；行号为该快照，实施时以符号名为锚 |
| 读者 | 框架研发、评审人、边缘/流媒体平台 |
| 关联 | [总览与路线](overview.md)、[架构总览](../framework_design.md)、[加固计划](../hardening_plan.md)（H1-2/H1-3 及其推理面条目已实施项不重复）、[已知限制](../known_limits.md) |

## 1. 范围与验收判定

范围：`src/autovision/results.py` 与 `src/autovision/keys.py`（契约层）、
`src/autovision/inference/`（engine/stream/multi_stream/video）、
`src/autovision/tracking/`（ByteTrack/BoT-SORT/analytics）、
`src/autovision/visualize/`（annotators/effects/zones 抽查）。
判定：**有条件通过**——单一容器契约、三段式管线背压、tracker 单流
状态机、multi_stream 可恢复重连与熔断、ByteTrack golden 快照均属同类
项目上乘实现；修复 INFER-01（多路流写手隔离）、INFER-02（float 语义
图渲染崩溃，已复现）、INFER-03（run_stream 逐帧隔离）后可验收。

## 2. 现状设计（as-built）

### 2.1 契约层

- `InferenceResult`（`src/autovision/results.py:262`）：9 个类型化
  字段 + extra/metadata/task；`from_record` 归一化 postprocess 裸 dict
  （`boxes`→`bboxes` 别名、未类型化键进 extra）；`__post_init__`
  禁止 extra 影子化类型化字段；校验集中在 `_validate`
  （形状、primary 行数一致、panoptic 一致性）。
- RLE 边界：`to_json(masks="rle")` 对实例栈逐个 COCO RLE，语义整数
  label map 走无损自研 RLE，panoptic 双通道分别 RLE；float 语义图有
  弃用闸门（加固计划 H1-11：0.6.1 警告、0.7.0 拒绝）；`masks="dense"`
  显式 opt-in 且受 `max_elements*100` 硬闸。
- `ResultValidationError` 双继承 `AutoVisionError`/`ValueError`
  （`src/autovision/errors.py:62`）。

### 2.2 管线与状态机

- `run_stream` 三段式：reader 线程 → `read_queue(prefetch)` → 主线程
  （batch/推理/tracker）→ `write_queue(writer_buffer)` → writer 线程
  （`src/autovision/inference/stream.py:379`）。背压默认阻塞不丢帧；
  `max_lag_frames` 显式丢旧（加固计划 H3-5b 已实施）；sentinel 必须
  落地 + reader 死亡兜底（H1-2 已实施）+ writer 30s 看门狗 + 有界
  drain。
- tracker/zone 单流状态机：实例按 stream 归属、`track=False` + zone
  运行时守卫、重连按 `reconnect_generation` 统一 reset、退役轨迹经
  `drain_removed_tracks()` 恰好发布一次（H1-3 已实施，flush 内排空）。
- multi_stream：每 source 一个 reader 线程 + 有界队列；**单个
  batcher 线程是唯一引擎调用者**（规避 TRT/OV runner 非线程安全）；
  per-source JSONL/视频/统计；drop_oldest 默认；指数退避 + 可恢复
  reconnect 预算 + per-source 熔断（H3-5d/f 已实施）；逐包序列化
  隔离（H3-1e 已实施）。
- 引擎参数两类：结构性参数（backend/artifact/device/providers）构造期
  校验一次；阈值类（threshold）每次 `run_batch` 传入。工件自描述
  元数据在加载期校验身份（task/类数/image_size）。

## 3. 关键流程

- **单帧**：`preprocess_image` → `preprocess_batch` → `_forward`
  （torch/ONNX/OpenVINO/TRT 只换 forward）→ 网络 `postprocess` →
  `InferenceResult.from_record(...).to_json(...)` →
  `annotate_result` 按字段分派渲染。
- **run_stream flush**：查 reconnect generation 并重置状态 → 批量
  推理 → 逐帧按序 tracker → smoother → zone → 写 JSONL（raw
  predictions + tracks + zones；JSONL 在 tracker 之前序列化，平滑只
  影响渲染）→ 渲染入写队列 → `drain_removed_tracks`。
- **multi_stream**：N reader 入队 → batcher 每轮每 source 取至多
  batch_size 帧 → 一次 run_batch → 逐包：abort 源跳过 → 序列化失败
  仅记该源 → 渲染 + 分辨率变化轮换写手文件名。

## 4. 审计发现清单

### 4.1 P1

| 编号 | 问题 | 证据 |
|---|---|---|
| INFER-01 | multi_stream 渲染写手创建/写入无隔离：`AnnotatedVideoWriter(...)` 构造与 `writer.write(...)` 都不在 try 内，单源磁盘满/编解码失败会穿透循环冲出 `run()`，summary.json 不可达（丢 summary），与模块"一路源故障不拖垮其他源"的核心承诺矛盾；同函数内 `draw_prediction` 失败却是被捕获降级的，隔离策略不一致 | `src/autovision/inference/multi_stream.py:572-576`、`:590`、`:596-610` |

### 4.2 P2

| 编号 | 问题 | 证据 |
|---|---|---|
| INFER-02 | float 语义 label map 渲染崩溃（已复现 `IndexError`）：直接分支用浮点数组做索引；尺寸不一致分支 cast 成 int 反而不崩——行为随分辨率而异；与序列化路径"警告+接受到 0.7.0"的策略不一致 | `src/autovision/visualize/annotators.py:617`、`:606-612`；对照 `src/autovision/results.py:664-685` |
| INFER-03 | run_stream 无逐帧隔离：`from_record(...).to_json(...)` 抛 `ResultValidationError` 或 tracker 对 NaN 分数/Inf 坐标抛 `AutoVisionError` 都直接终止整个 run——对 24/7 RTSP 场景是可用性缺口；JSON/渲染路径都专门做了 NaN 容忍，tracker 路径却一帧即死 | `src/autovision/inference/stream.py:177`、`:542-544`；`src/autovision/tracking/byte_track.py:701-711`（已复现） |
| INFER-04 | 无元数据工件整体绕过身份闸门：`read_metadata` 对不可读/缺失返回 `{}` 且只记 debug，`verify_metadata` 对空 dict 全部跳过——被剥离元数据的错误工件可静默加载运行 | `src/autovision/inference/engine.py:73-79`；`src/autovision/export/artifact_metadata.py:210-229`、`:288-290` |
| INFER-05 | 推理后端输出数量/形状无运行时对拍：任意输出原样转 tensor 交给 postprocess；记录数检查在消费方才有，且 multi_stream 只记 warning——后端图与 torch 参考布局不一致时静默错误坐标或晦涩异常 | `src/autovision/inference/engine.py:131-141`、`:159-182` |
| INFER-06 | `TraceAnnotator._history` 无淘汰：setdefault 只增，长跑服务每个出现过的 track_id 留一条 deque 永不清除（对比 TrackSmoother/LineZone 均有 max_age 驱逐） | `src/autovision/visualize/effects.py:361`、`:446-448` |

### 4.3 P3

| 编号 | 问题 | 一句话方向 |
|---|---|---|
| INFER-07 | stream.py 全程裸字符串键，未 import `src/autovision/keys.py` | 统一引契约常量 |
| INFER-08 | `_track_record` 在记录无 labels 时重建 Track 丢失 `det_index`，join-back 契约在该分支失效 | 透传 det_index |
| INFER-09 | `engine.run_batch([])` 抛裸 RuntimeError（stack 空列表） | 前置校验抛 `AutoVisionError` |
| INFER-10 | multi_stream 每帧 `flush()` JSONL，N 源高频 I/O 放大 | 按帧数/间隔刷 |
| INFER-11 | 16-bit 帧渲染 `max()>1.5 → /255` 启发式回绕损坏（已复现） | dtype 分支显式归一 |
| INFER-12 | effects docstring 声称 tracker_id 写进 result.extra，实际无人写入——文档与代码失真 | 改文档或接通写入 |
| INFER-13 | `TrackSmoother` 对同帧重复 id 双计（公共 API 未防御） | 输入去重或断言 |
| INFER-14 | run_stream `track_ids` set 全程累积，数月长跑缓慢无界 | 窗口化或退化统计 |
| INFER-15 | `write_queue.put(rendered)` 无超时无 stop_event 检查，writer 卡死时主循环无看门狗可依 | 有界 put + 看门狗 |
| INFER-16 | `VideoFrameStream` 对同一实例二次迭代共享 capture 无守卫 | 迭代状态断言 |
| INFER-17 | effects 的 mask 尺寸不符静默跳过，与 MaskAnnotator 的 resize+debug 策略不一致 | 对齐策略 |
| INFER-18 | 被熔断 abort 的 live 源 reader 继续解码到预算耗尽 | 尽早退出解码循环 |
| INFER-19 | multi_stream 时间戳用 `time.time()`，NTP 回拨可乱序 | 单调时钟 |

## 5. 整改实施设计

> 状态：**INFER-01..09、11、12、13、16 已实施（工作区，随 0.6.1 发布）**；
> INFER-10/14/15/17/18/19 登记为验收后第一批迭代项（见下）。
> 实施与设计的偏差：INFER-03 的失败阈值固化为模块常量
> `_MAX_CONSECUTIVE_FRAME_ERRORS=10`（未暴露为参数，避免 API 膨胀）；
> INFER-04 的豁免口是 `InferenceEngine(require_metadata=False)` 默认值
> （库用途不变），CLI/serving 两个生产入口传 True；INFER-06 的驱逐在
> annotate 末尾执行且当前帧 id 受保护（上界在任何流失率下精确成立）；
> INFER-12 按文档修正实施（撤销 docstring 失真声明，未接通 extra 写入——
> 那是行为新增，另行排期）；INFER-19 按"墙钟时间戳是 JSONL 工件契约"
> 登记为有意取舍而非缺陷。
>
> **实施后复审结论（只读代理复审 + 推理子系统 350 项回归全绿）：可合入，
> 无 P0/P1。** 复审驱动的追加闭环：① multi_stream 清理路径防护
> （P2-1）——`_abort_source` 与 run 级 finally 的 close 全部 best-effort
> （MP4 finalize 恰在磁盘满时最可能抛，不得反杀 containment）；summary
> 保存失败改为 `raise run_error from save_error`（根因不被掩盖）。② skip
> 模式可见性补齐（P2-2）——run 结束时的汇总 warning、
> `StreamRunResult.summary()` 携带 failed_frames、CLI 经 kwargs 透传
> `on_frame_error` 天然可用。③ 连续失败计数随重连 generation 重置；
> engine 的 known 键集合改用 keys.py 常量、空 dict 记录报错；docstring
> 记载 require_metadata（torch backend 下为 no-op）；TraceAnnotator 空帧
> 也驱逐、阈值边界措辞统一为"超过 10 帧"。④ 测试补齐：恰好 10 连坏 +
> 健康帧重置、writer 写入中途失败、Inf/负值 float label map。
> 复审登记的剩余 P3（require_metadata 接线测试、负值 int16 渲染回绕、
> aborted 源 straggler 计数）随批次排期。

### 5.1 INFER-01 multi_stream 写手隔离（P1）

**设计**：

1. `AnnotatedVideoWriter` 构造移入 per-packet/per-source try；失败 →
   该源标记 aborted（携带原因 `writer setup failed: ...`），按既有
   abort 语义跳过后续帧，其他源继续。
2. `writer.write(...)` 包 try；失败同上，计数进 `render_failures`
   （source 统计已有结构可挂）。
3. 保证 `finally` 后的 `result.save()` 恒可达：`run()` 主循环不得让
   单源异常穿透（兜底：最外层 catch 转 run 级错误记录后重抛前先保存
   summary）。
4. 对齐原则：写手失败与序列化失败同等级（降级该源、不拖垮全局）；
   与 `draw_prediction` 的既有降级一致。

**涉及文件**：`src/autovision/inference/multi_stream.py`。

**测试计划**：`tests/test_multi_stream.py` 新增：source 2 的 writer
工厂抛错 → run 正常返回、source 1 输出完整、summary 含 source 2 的
error 条目与 frames_dropped；写入中途抛错 → 该源 aborted、其余源
不中断；`tests/test_soak.py` 的 multi_stream 老化用例保持绿。

**验收标准**：两个新用例通过；summary 恒产出。

### 5.2 INFER-02 float 语义图渲染统一策略（P2）

**设计**：`SemanticMaskAnnotator` 的两个分支统一先做
`labels = np.rint(labels).astype(np.int64)`（NaN → 0 前景边界并在
debug 日志说明），再 clip/mod 索引；策略与序列化侧 H1-11 对齐：
0.6.x 渲染容忍 + debug 日志，0.7.0 随序列化一起拒绝 float 多类图。
docstring 写明"float label map 按四舍五入取整渲染，弃用路径同
to_json(masks='rle')"。

**涉及文件**：`src/autovision/visualize/annotators.py`。

**测试计划**：float label map（尺寸一致/不一致两种）→ annotate 返回
有效 ndarray 不抛错（修复前第一个用例失败）；float {0,1} 与 int 版
逐像素一致；`tests/test_visualize.py` 回归。

### 5.3 INFER-03 run_stream 逐帧隔离（P2）

**设计**（含一个决策点）：

1. `_track_record` 与 `to_json` 的调用包 per-frame try；失败 →
   `frames_failed += 1`、按序写一条错误行到 JSONL（含 frame index 与
   错误摘要），继续下一帧；连续失败超过阈值（建议
   `max_consecutive_frame_errors=10`）→ 升级为终止 run（防"全线坏"
   时静默空转）。
2. **决策点（建议方案 A）**：默认行为从"一帧即死"改为"跳帧 + 计数 +
   汇总"；新增 `on_frame_error="raise"|"skip"`（默认 skip）供需要
   fail-fast 的离线场景。理由：长跑监控场景可用性优先，且失败可见
   （错误行 + StreamRunResult 计数 + 结束 warning）。
3. tracker 输入校验（NaN/Inf）触发同一通道，不直接抛穿。

**涉及文件**：`src/autovision/inference/stream.py`、
`src/autovision/tracking/byte_track.py`（校验错误的归类标记）。

**测试计划**：`tests/test_stream_pipeline.py` 新增：NaN 分数 +
track=True → run 完成、frames_failed==1、错误行存在、其余帧正常
（修复前该用例失败）；`on_frame_error="raise"` → 终止（保留旧行为
出口）；连续 10 坏帧 → 终止。**验收标准**：三用例通过；
`StreamRunResult` 新计数有测试锚定。

### 5.4 INFER-04 元数据闸门可选 fail-closed（P2）

**设计**：`InferenceEngine` 增加构造参数 `require_metadata: bool =
False`：True 且 `read_metadata` 为空 → `AutoVisionError`（消息指引
重新导出或显式降级）；False 时把现状的 debug 日志升级为一次 warning
（可见性）。`autovision-cli predict` 与 serving 的 `create_app` 传
True（面向生产的入口 fail-closed，库用途保持宽松）。

**涉及文件**：`src/autovision/inference/engine.py`、
`src/autovision/serving/app.py`、`src/autovision/cli/_app.py`。

**测试计划**：无元数据 + require → 报错；无元数据 + 默认 → warning；
有元数据但身份不匹配 → 既有行为保持；`tests/test_inference_runtime.py`、
`tests/test_serving.py` 回归。**验收标准**：旧工件（无元数据）在
生产入口不再静默通过。

### 5.5 INFER-05 后端输出契约运行时校验（P2）

**设计**：`run_batch` 在 postprocess 返回后校验：(a) 记录数 ==
batch 内样本数，否则 `AutoVisionError`（消息含两侧数量与 backend
名）；(b) 每条记录的键 ∈ `src/autovision/keys.py` 已知集合（未知键
列表报出，extra 白名单键除外）。消费方既有检查保留（防御深度）。

**涉及文件**：`src/autovision/inference/engine.py`。

**测试计划**：Fake engine 返回少一条记录 / 未知键 → 报错带 backend
名；`tests/test_inference_runtime.py` 回归。

### 5.6 INFER-06 TraceAnnotator 历史上界（P2）

**设计**：新增 `max_tracks: int = 512` 构造参数：`annotate` 时按
"最后出现帧序"驱逐最旧且已死亡（不在当前帧）的 id，保持
`len(_history) <= max_tracks`；`max_tracks=None` 保持无界（显式
opt-out）。与 TrackSmoother 的 max_age 驱逐互补（一个按年龄、一个按
总量）。

**涉及文件**：`src/autovision/visualize/effects.py`。

**测试计划**：10 万 id 序列 → `_history` 规模有界且渲染输出在驱逐
边界处确定性（同输入同输出）；`tests/test_visualize_effects.py`
回归。

## 6. 实施顺序

INFER-01 → INFER-02 → INFER-03（可用性三连，建议同批联调）；
INFER-04/05（生产入口语义）；INFER-06。P3 中 INFER-11（16bit 回绕）
与 INFER-12（文档失真）建议随批顺手修。

## 7. 风险与回滚

- INFER-03 改变默认失败语义（终止 → 跳帧）：这是行为变更，CHANGELOG
  显著标注；`on_frame_error="raise"` 是旧行为的显式出口；长跑验收
  用 `tests/test_soak.py` 补一条跳帧累积断言。
- INFER-04 在生产入口对旧工件从"静默通过"变"报错"——预期行为，
  错误消息给出重新导出指引；known_limits 同步改写。
- INFER-01/02 是纯健壮性修复，无行为面变化，revert 即回滚。
