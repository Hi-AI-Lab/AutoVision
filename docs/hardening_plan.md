# 工程加固设计与实施计划

> 本页是 2026-11 全仓库代码审计（7 个子系统并行审计，关键结论均经运行时复现）
> 的**修复设计与实施计划**。审计发现的定位与既有边界见[已知限制](known_limits.md)，
> 分层与模块职责见[架构总览](framework_design.md)，版本与弃用规则见
> [发布流程](release_process.md)。
>
> 状态：**设计评审中**。目标载体：v0.6.1（PATCH，P0/P1）与 v0.7.0（MINOR，
> P2 与机制化）。文中的行号以 v0.6.0 工作区为准，实施时以符号名为锚。

## 1. 背景与总体判定

审计结论：架构与工程纪律显著高于平均自研框架水准（训练循环、导出门禁、
合规治理达到 A- 级），但企业级验收判定为**有条件不通过**。核心矛盾是
"**设计写到了、代码没接上**"：原子写纪律、生命周期清理、fail-closed
原则在多处只覆盖了好走的路径。共发现 1 个 P0、11 个 P1、40 余个 P2，
其中 4 个 P1 属于"不崩溃但结果错"或"不报错但永久挂起"——这类静默
故障对企业验收最致命。

| 子系统 | 评级 | 关键缺口 |
| --- | --- | --- |
| 训练引擎 + 任务面 | B+ | DDP 种子三处脱钩、`export_weights` 非原子 |
| 导出链路 + runtime | A- | TRT/OpenVINO 发布非原子、fp16 门禁路径缺失 |
| 数据层 + 配置层 | B+ | 下载 pin 全空、截断图静默解码、三处静默丢标注 |
| 契约层 results/keys | B+ | float 语义 mask 静默二值化、dense 无上限 |
| CLI/指标/注册表/治理 | B+ | 误差分析匹配器排序错位（污染难例采样） |
| Serving | 参考实现 A- / 生产 D | fail-open 认证、并发与像素资源无上界 |
| 推理运行时 + 跟踪 | C+ | 两个无界泄漏 + 一个可复现永久挂起 + 8 个死测试 |

## 2. 严重度定义

- **P0**：默认部署姿态下可直接造成安全/资源事故。
- **P1**：静默数据损坏、静默挂起、系统性错误结果，或核心机制形同虚设。
- **P2**：口径漂移、边界健壮性缺口、性能缺陷、错误信息质量；不影响主路径
  正确性但会在生产环境累积为事故。

## 3. 全局设计原则（所有修复共同遵守）

1. **计量路径 fail-fast**：mAP/混淆矩阵/误差分析等"出数字"的路径上，
   宁可抛异常不可静默少计；可视化路径可以容错降级（保持现状）。
2. **原子发布全覆盖**：凡"写终路径"的产物（checkpoint、ONNX、TRT 引擎、
   OpenVINO IR、导出权重、JSONL）一律 tmp + fsync + `os.replace`，门禁
   失败保留旧工件。
3. **生命周期契约必须被框架自己的消费方兑现**：tracker 的
   `drain_removed_tracks()`、GMC 的 `reset()` 这类已设计的清理钩子，
   参考实现先接上。
4. **seed 全链路贯通**：运行 seed 必须到达 DistributedSampler、loader
   generator（按 rank 偏移）、worker_init_fn 三处。
5. **意图→测试锚定**：每个设计编号（D1–D36 及本文 H 编号）至少一条
   测试显式引用其编号（见 H4-3），防止"文档说做了、代码没做"复发。
6. **弃用走 AGENTS.md §3.4**：行为翻转先告警跨一个 MINOR，警告文案写明
   `deprecated in X.Y.Z` 与 `remove in A.B.C`。

## 4. 批次划分与版本映射

| 批次 | 内容 | 目标版本 | 估算 |
| --- | --- | --- | --- |
| 批次 0 | 测试防线修复（死测试） | 0.6.1 | 0.5 人日 |
| 批次 1 | P0/P1 正确性与安全性修复（H1 系列 + H2-1） | 0.6.1 | 6–8 人日 |
| 批次 2 | Serving 加固余项（H2 系列） | 0.6.1–0.7.0 | 3–4 人日 |
| 批次 3 | P2 汇总修复（H3 系列，按子系统分组） | 0.7.0 | 8–10 人日 |
| 批次 4 | 机制化防复发（H4 系列） | 0.7.0 起持续 | 5–6 人日 |

依赖关系：批次 0 先行（恢复 8 个测试的回归保护后，批次 1 的修改才有
护栏）；批次 1 内 H1-2/H1-3（流与跟踪）应同批联调；批次 4 的 H4-3
（意图锚定检查）落地后回填本文所有 H 编号的测试引用。

---

## 5. 批次 0：测试防线修复

### H0-1 恢复被嵌套吞掉的 8 个流式管线测试（P1）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`tests/test_stream_pipeline.py:186-187` 的缩进事故把 8 个
测试函数（zone 防护、NaN→null、平滑不污染 JSONL、缓冲区校验、分类渲染等）
嵌进了模块级函数 `test_redact_source_hides_url_credentials` 的函数体内。
`pytest --collect-only` 仅收集 8/16。`run_stream` 对
`track=False` + zone 的运行时防护（`stream.py:325-330`）因此处于零回归
保护状态。

**设计**：把 8 个函数恢复为 `TestPipeline` 的方法（带 `self`）或模块级
函数（去 `self`），删除嵌套。同时新增一条布局守卫（见 H4-4），从机制上
杜绝同类事故。

**测试计划**：

```bash
.venv/bin/python -m pytest tests/test_stream_pipeline.py --collect-only -q
# 验收：收集 16 个用例
.venv/bin/python -m pytest tests/test_stream_pipeline.py -q
```

**验收标准**：collect 16/16；全部通过；若恢复后有用例失败，按失败项
单独开缺陷（它们暴露的是防护代码自身的回归，不是测试的问题）。

---

## 6. 批次 1：P0/P1 正确性与安全性修复

### H1-1 误差分析匹配器 IoU 行错位（P1，一行修复 + 回归测试）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`src/autovision/metrics/error_analysis.py:195-208`。
`ious` 的行序是类内**张量序**（`boxes_p[pred_index]`），而循环按**分数
排名**取行（`for position, p in enumerate(order)` 后 `row = ious[position]`）。
仅当输入预测恰好按分数降序排列时二者重合。已复现：0.9 分压 GT 的框被判
FP、0.3 分无关框被判 TP。激活面是离线模式
（`analyze-errors predictions=… annotations=…`，外部 JSON 不保证有序），
后果链：报告/混淆矩阵/`hard_examples.json` 全部失真，且难例文件会污染
难例加权采样（`error_analysis.py:483-485`）。所有测试夹具恰好是降序分数，
故零拦截。mAP 主路径委托 pycocotools，不受影响。

**设计**：行索引改为与排序一一对应。`argsort` 的返回值本身就是
"排名 → 类内位置" 的映射，直接复用：

```python
sorted_idx = scores_p[pred_index].argsort(descending=True)
order = pred_index[sorted_idx]
...
for position, p in enumerate(order.tolist()):
    ...
    if ious is not None:
        row = ious[sorted_idx[position]]   # 原: ious[position]
```

**测试计划**（`tests/test_error_analysis.py` 新增）：

```python
def test_match_detections_with_unsorted_scores_attributes_correctly() -> None:
    """乱序分数输入：高分压 GT 的框必须判 TP，低分无关框必须判 FP。"""
```

夹具：同图两类框，pred0=0.30 在空白处、pred1=0.90 正压 GT，张量序保持
未排序；断言 `summarize_matches` 的 tp/fp 归因与 `Match` 行内容。
参数化补一个"全同分"与"单预测"的退化 id 用例。

**验收标准**：新测试通过；既有误差分析/混淆测试全绿；
`hard_examples` 采样路径打点核对一次。

### H1-2 消除 run_stream 永久挂起（P1，可复现）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`src/autovision/inference/stream.py:380-388`，reader
结束时若队列满，sentinel（`None`）只重试 5 秒即放弃（仅 warning）；
主循环 `stream.py:532-544` 的唯一退出条件是收到 sentinel 或
`writer_error`——sentinel 丢失后 `except queue.Empty: continue`
（`:536-538`）永久空转。已复现：3 帧视频、`prefetch=2`、`batch_size=1`、
首批推理 sleep 6s → 线程 20 秒后仍存活。触发面：CPU 大模型首批 >5s、
GPU 争用、TRT 大 batch。

**设计**（双保险，两层都做）：

1. **sentinel 无限重试**（根因修复）：reader 线程的 sentinel 投递改为
   `while not stop_event.is_set()` 循环内 `put(timeout=0.2)`。该循环只
   阻塞 reader 线程自身；主循环每次 `flush()` 后必然腾出队列空间，
   sentinel 终将入队。`stop_event` 保证外部中止路径仍可退出。
   删除 5 秒 deadline 与 `logger.warning`。
2. **主循环 reader 存活守卫**（兜底，覆盖 reader 崩溃未投 sentinel 的
   另一入口）：

```python
except queue.Empty:
    if (
        not reader_thread.is_alive()
        and read_queue.empty()
        and not reader_error
        and not saw_sentinel
    ):
        # reader 已死且队列已空却没有 sentinel：视为异常终止而非空转。
        break
```

   循环出口后按现有 `reader_error` 语义处理；无错误则记录 warning 并
   正常收尾（`finally` 块已能安全停机）。

**测试计划**（`tests/test_stream_pipeline.py`，确定性时序构造）：
FakeEngine 首批 `time.sleep(6)`（测试内 monkeypatch 缩短为 0.5s 量级）、
`prefetch=2`、3 帧源，断言 `run_stream` 在限定墙钟内返回且 JSONL 完整；
第二个用例：reader 抛异常且不投 sentinel，断言主循环以 `reader_error`
退出。

**验收标准**：两个新用例通过；既有 16 个流测试全绿；挂起类缺陷在
长跑老化测试（H4-1）中不再出现。

### H1-3 ByteTrack 僵尸轨迹清理（P1×2，两个无界泄漏）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：

1. `src/autovision/tracking/byte_track.py:602-603`：匹配失败的
   unconfirmed 轨迹置 `state=_REMOVED` 但**不从 `self.tracks` 移除**；
   唯一做移除的 `_retire_tracks`（`:627-648`）只处理 `_LOST`。已复现：
   100 帧闪烁检测后 `len(tracker.tracks)` 从 2 涨到 100。`:630-631` 的
   docstring 声称 unconfirmed 移除"dropped silently"，与实现不符——
   实际是滞留。每帧 `_predict_tracks` 对全量僵尸跑 Kalman。
2. `drain_removed_tracks()`（`byte_track.py:421`）是已设计的下游清理
   契约（design D2），但 `grep -rn drain_removed_tracks src/` 零调用——
   `tracker._removed` 在 `run_stream` 长跑中以每帧一条的速度无界堆积。

**设计**：

1. `_unconfirmed_association` 中对 unmatched unconfirmed 轨迹：置
   `_REMOVED` 后同步从 `self.tracks` 移除（对齐官方 yolox 语义）；修正
   `_retire_tracks` docstring。移除动作放在 association 阶段末尾统一
   过滤，避免迭代中改列表：

```python
removed_ids = {unconfirmed[i].track_id for i in unmatched_new}
for t in unconfirmed:
    if t.track_id in removed_ids:
        t.state = _REMOVED
self.tracks = [t for t in self.tracks if t.track_id not in removed_ids]
```

2. `run_stream` 在 `flush()` 内、zone/smoother 更新之后调用
   `tracker.drain_removed_tracks()`，返回值并入运行统计
   （`stats["tracks_removed"]`），不改 JSONL 输出语义（被移除轨迹本就
   不发布）。

**测试计划**：`tests/test_tracking.py` 新增
`test_unconfirmed_removal_does_not_leak`：100 帧大跳变高分检测，断言
`len(tracker.tracks) <= 3`（活跃 + 缓冲期上限）且 golden 生命周期快照
不变；`tests/test_stream_pipeline.py` 新增长跑用例：1000 帧闪烁检测 +
`track=True`，断言 `tracker._removed` 长度有界（该用例同时是 H4-1
老化测试的首个固化）。

**验收标准**：golden 跟踪测试逐位不变（此修复不改变发布轨迹，只清理
内部列表）；两个泄漏用例通过。

### H1-4 训练引擎 DDP 种子三件套（P1，复现性断裂）

> 状态：已实施（工作区，待发布于 0.6.1；含 H3-3b sampler epoch 恢复）。

**现状与证据**：

1. `src/autovision/engine/engine.py:780-782`（`Engine.fit`）与
   `:1500-1502`（`fit_task`）：`DistributedSampler(..., shuffle=True)`
   用默认 `seed=0`，运行 `seed` 未传入——多卡 shuffle 顺序与 seed 脱钩，
   单卡/多卡同 seed 数据序不同。
2. `engine.py:858-869`：`Engine.fit` 每 epoch 重建的 `epoch_loader`
   丢弃 `worker_init_fn`/`generator`/`prefetch_factor`/
   `persistent_workers`——SSL 预训练多卡下所有 worker 重放同一增广序列
   （对比 `fit_task` 的 `_with_sampler`（`:2372-2387`）携带了它们，
   同文件内行为不对称，属遗漏）。
3. `src/autovision/cli/common_helpers.py:265` 与
   `src/autovision/cli/train_helpers.py:163`：`make_dataloader_generator(seed)`
   未传 `rank`，而签名（`engine.py:434`）支持 `rank` 偏移。

**设计**：

1. 两处 sampler 构造加 `seed=seed`（`fit`/`fit_task` 的 `seed` 参数已
   在作用域内）；恢复分支同时 `sampler.set_epoch(...)` 对齐
   `fit` 的既有行为，并给 `_InfiniteCycleIterator` 增加 epoch 透传
   （顺带修 P2 "恢复不恢复 sampler epoch"，见 H3-1）。
2. `epoch_loader` 重建补齐四个字段，与 `_with_sampler` 共用同一个
   字段拷贝 helper，消灭两处不对称。
3. 两个 CLI 调用点改为 `make_dataloader_generator(seed, rank=rank)`；
   `rank` 取 `torch.distributed.get_rank()`（进程组未初始化时 0）。

**兼容性**：这改变了多卡训练的数据顺序（修复复现性断裂本身）。在
CHANGELOG 标注"多卡 run 的数据顺序与 0.6.0 不同"；`resume` 场景在
`check_resume_fit_args` 的漂移告警中已有覆盖，不额外加机制。

**测试计划**：`tests/test_distributed.py`（torchrun gloo 2-rank，已有
基建）新增：两 rank 的 loader generator 状态不同（per-rank 偏移）；
`tests/test_engine_recovery.py` 新增：`fit()` DDP 重建路径断言
`worker_init_fn`/`generator` 传递到重建的 loader（用注入 loader 检查
属性，不依赖真数据）。

**验收标准**：多卡 smoke 中同 seed 两次 run 的首 epoch 采样序一致；
单卡路径回归不变。

### H1-5 `export_weights` 原子化 + 写屏障（P1）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`engine.py:1788-1799` 直接 `torch.save(..., path)` 写
终路径（对比 `runtime/checkpoint.py:53` 的 `_write_atomic`：tmp + fsync
+ `os.replace` + 3 次退避重试）；调用点 `engine.py:2125-2130`（周期导出
last）与 `:2335-2339`（早停导出）在 `is_rank_zero` 内**无 try、无
`_rank_zero_write_barrier`**（`:325`），而同文件 best 导出路径
（`:2285-2312`）有完整防护。磁盘满/权限故障时：半截 `exported_last.pt`
留在目标路径，rank0 死亡后其余 rank 要等 NCCL watchdog（默认 1800s）。
`engine.py:2016-2021` 注释声称的 "peers fail fast" 对此路径不成立。

**设计**：

1. `export_weights` 的落盘段改为复用 `_write_atomic`
   （`runtime/checkpoint.py`），tmp 文件命名沿用 `<target>.tmp-<pid>`
   约定，保证 fs_utils 的孤儿清扫（1h mtime 保护）能识别。
2. 两处调用点包 `try/except` + `export_ok` 标志 +
   `_rank_zero_write_barrier(device, world_size, export_ok)`，照抄
   best 导出路径（`:2285-2312`）的形状。

**测试计划**：`tests/test_engine_cache_and_retry.py` 新增：注入
`torch.save` 抛错，断言目标路径无残留 tmp、旧工件（若存在）不变、
DDP 下非 0 rank 收到屏障错误而非挂起（2-rank gloo 用例）。

**验收标准**：写失败注入用例通过；既有 checkpoint 重试测试全绿。

### H1-6 TensorRT 引擎原子发布（P1）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`src/autovision/export/tensorrt_helpers.py:302-303`
`engine_path.write_bytes(engine_blob)` 直写终路径；调用方
`export_onnx.py:1268-1285` 先 build → 写 → verify，
verify 失败再 `unlink`（`:1203-1205`）——旧好引擎先被覆盖再被删，
数分钟构建毁于一次数值漂移（GTR×TRT 11.4 的已知漂移场景正是触发面，
见 `docs/known_limits.md`）。

**设计**（重排时序，而不是只加 tmp）：

1. `tensorrt_helpers.py`：引擎 blob 写入改为
   `tmp = engine_path.with_name(engine_path.name + ".tmp-<pid>")` →
   write → fsync → `os.replace`（与 ONNX 链 `export_onnx.py:342-358`
   同款）。
2. `export_onnx.py` 的 TRT 流程重排为：build → **写 tmp 引擎** →
   用 tmp 路径构造 `TensorRTEngineRunner` 跑 parity → 通过后
   `os.replace` 发布 → 写 sidecar（`.meta.json`/`.postprocess.json`
   同样先 tmp 后 replace）。verify 失败时只删 tmp，**旧引擎原地不动**。
   `unlink` 语义保留给"旧引擎不存在且本次失败"的场景。

**测试计划**：`tests/test_deployment.py`（mock build）新增：
a) 预置一个旧引擎文件 + 注入 verify 失败 → 断言旧引擎字节不变、无
`.tmp-` 残留；b) 成功路径断言发布与 sidecar 同批原子出现。

**验收标准**：两个用例通过；`test_export_atomicity.py` 全绿。

### H1-7 OpenVINO IR 原子发布（P1）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`src/autovision/export/openvino_helpers.py:129`
`save_model(ov_model, str(xml_path))` 直写终路径；`:131-139` verify
失败后 finally 只删中间 ONNX，**坏 IR 留在部署目标位置**（注释自称
"mirroring the ONNX exporter"，但 ONNX 公开路径失败时 tmp 被删、旧工件
存活，语义正好相反）。`export_onnx.py:324-328` 承诺的
"previous good artifact survives" 对 `.xml` 目标不成立。

**设计**：与 H1-6 同形——IR（`.xml` + `.bin` 两个文件）写 tmp
（`.tmp-<pid>` 后缀成对），verify 通过后成对 `os.replace`；失败删
tmp 对。`.bin` 的成对替换需在同一临界区完成（先 replace `.bin` 再
`.xml`，与 OpenVINO 运行时"按 xml 找 bin"的读取顺序一致）。

**测试计划**：`tests/test_deployment.py` 新增镜像用例（预置旧 IR +
verify 失败 → 旧 `.xml`/`.bin` 不变）。

**验收标准**：同 H1-6。

### H1-8 fp16 ONNX 门禁路径兑现（P1，承诺与代码不符）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`src/autovision/export/export_onnx.py:765` 的警告文案
承诺 "Re-verify on a CUDA-enabled onnxruntime build ... for exact
parity"，但 `:782` 无条件
`providers=["CPUExecutionProvider"]`——CUDA EP 可用时承诺的路径在代码中
不存在，fp16 图的 CUDA-EP 数值从未被门禁覆盖。

**设计**：对拍 session 的 provider 选择改为探测：

```python
available = ort.get_available_providers()
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available
    else ["CPUExecutionProvider"]
)
```

CUDA EP 生效时输出 info 日志（记录 device/供应商版本）；不可用时保留
现有 warning。CI（无 GPU）行为不变；带 GPU 的 nightly（`trt-nightly.yml`
同机）补跑 fp16 CUDA-EP parity 一个用例。

**测试计划**：单测用 monkeypatch 模拟 provider 列表断言选择逻辑；真实
CUDA-EP 用例带 `requires_cuda` 门控进 nightly job。

**验收标准**：CPU CI 全绿；nightly 报告出现 fp16 CUDA parity 一行。

### H1-9 数据集注册表补齐 pin + VOC 转 HTTPS（P1）

> 状态：部分实施（工作区）：机制/脚本/守卫测试就绪，pin 值需有网环境计算
> （`scripts/compute_dataset_pins.py` 计算 + `scripts/apply_dataset_pins.py`
> 写回，支持 `--check`/`--force`/回读验证）；VOC https 探测因无出网未改。

**现状与证据**：`src/autovision/data/downloads.py:55-103`，
`DATASET_REGISTRY` 全部 `DatasetFile` 仅 `url=`，无一有 `sha256`/
`size_bytes`——下游的 sha256 校验（`:341-348`）、size 校验（`:335-340`）、
磁盘预检（`:189-205`）、严格模式 `AUTOVISION_REQUIRE_DATASET_HASHES=1`
（`:550-556` 拒绝一切注册条目）**全部是死代码**。`known_limits.md`
已如实登记"内置 COCO/VOC 数据集条目没有官方摘要，默认只告警"。
叠加 VOC 条目走 `http://` 明文（`:84-86`），无 pin + 明文使 MITM 可
注入恶意 tar。

**设计**：

1. 为 4 个条目的可直链文件补 `sha256` 与 `size_bytes`：从官方当前
   工件一次性计算，摘要与计算环境（镜像、日期、上游页面）登记进
   `docs/provenance/LEDGER.md`。COCO 官方页面发布过 md5；sha256 由
   实施时从官方源计算并双人复核（脚本
   `scripts/compute_dataset_pins.py` 新增，输出可粘贴的 registry 片段）。<!-- docpath-ok -->
2. VOC URL 迁移到 `https://host.robots.ox.ac.uk/...`；实施时验证
   https 可用性与证书链，若上游不支持 https 则改用镜像清单并保持
   pin 强制。
3. **机制化防回退**：新增测试
   `test_dataset_registry_entries_are_pinned`，断言每个可直链
   `DatasetFile` 必须携带 `sha256`（`manual_instructions` 条目豁免）；
   否则 CI 失败。这样"新条目必须带 pin"成为结构约束。
4. `known_limits.md` 的"下载完整性"段同步改写：内置条目已钉摘要，
   仅第三方手工数据集不受保护。

**兼容性**：pin 生效后，上游换文件（同 URL 不同字节）会导致校验失败
——这是 fail-closed 的预期行为；报错信息沿用现有
"downloaded file does not match pinned sha256" 文案并附 LEDGER 指引。

**测试计划**：`tests/test_downloads_resume.py` 增加用例：pin 存在时
篡改响应字节 → 校验失败删除临时文件；既有 9 个续传场景回归。

**验收标准**：`AUTOVISION_REQUIRE_DATASET_HASHES=1` 下
`download_dataset("coco2017-val")` 可成功完成（本地小文件 smoke 用
伪造注册表 + 本地 HTTP 服务即可，不需要真下 18GB）。

### H1-10 截断图像全局开关隔离（P1，静默数据损坏）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`src/autovision/data/file_helpers.py:67`（读图设
`ImageFile.LOAD_TRUNCATED_IMAGES = False`）与 `:107`（读 mask 设
`True`）改的是**同一进程级全局**。分割任务 DataLoader worker 内图与
掩码交替读取：一次 mask 读取把全局置 True 后，其后任何走 PIL 回退
路径的截断 JPEG 都会静默解码部分像素进训练——恰好违背 `:65-67` 注释
自声明的 "Truncated images must fail loudly"。

**设计**：全局翻转改为**调用点局部作用域**。新增上下文管理器：

```python
@contextmanager
def _truncated_image_loading() -> Iterator[None]:
    from PIL import ImageFile
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
```

`_open_mask_pil` 内 `with _truncated_image_loading():` 包住解码；
`_open_image_pil` 不再触碰全局（改为断言当前值为 False，若为 True
说明有第三方库改动过，记 warning 不阻断）。掩码保持现有宽松语义
（历史数据集的掩码文件可能被截断过，行为不回退、不破坏）。

**测试计划**：`tests/` 新增顺序敏感用例：先 `_open_mask_pil` 再
`_open_image_pil` 读截断 JPEG，断言图片读取**仍然报错**（修复前该
用例失败）。

**验收标准**：新用例通过；`open_mask` 对截断掩码的行为不变。

### H1-11 float 语义 mask 的 RLE 静默二值化（P1，弃用路径）

> 状态：已实施 0.6.1 阶段：deprecation warning（工作区）；0.7.0 raise 待做。

**现状与证据**：`src/autovision/results.py:582-590`：`to_json(masks="rle")`
对 semantic `mask` 的"是否 label map"判定只看 dtype 是否整数。float
dtype 的 `(H,W)` 多类别图被按 `>0.5` 静默二值化——类别 1/2 无差别融为
前景，无警告。已复现。

**设计**（按 AGENTS.md §3.4 走一个 MINOR 的弃用）：

- **0.6.1**：float mask 进入 RLE 路径时计算
  `np.unique(mask_array)`：若唯一值 ⊆ `{0.0, 1.0}` → 保持二值化并输出
  一次 deprecation 级 warning（文案写明 `deprecated in 0.6.1,
  float label maps will be rejected in 0.7.0; convert to integer
  dtype`）；若存在 >1 的值 → 同样 warning + 维持现状二值化（不新增
  崩溃）。唯一值缓存一次计算避免热路径重复扫。
- **0.7.0**：非 bool/integer 的 `(H,W)` mask 在 RLE 路径抛
  `AutoVisionError`，报错文案给出两条出路（转 int label map，或
  `masks="dense"`/`"none"`）。

**测试计划**：`tests/test_mask_serialization.py` 新增：float 多类 mask
触发 warning（`pytest.warns`）；float {0,1} mask 输出与现状逐字节一致
（golden 锚定不静默变更行为）；0.7.0 的 raise 用例随该版本实施时打开
（同文件参数化，`pytest.param(..., marks=...)` 按版本门控）。

**验收标准**：现有 mask 序列化测试全绿；新 warning 用例通过。

---

## 7. 批次 2：Serving 加固

### H2-1 认证默认 fail-closed（P0）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`src/autovision/serving/app.py:144-147`：
`if api_key is None: return True`——默认无鉴权（fail-open）。缓解仅是
默认绑 loopback（`server.py:43`）。`host=0.0.0.0` 上线即无鉴权推理
接口。`known_limits.md:8` 如实登记了这一点，但生产姿态应当 fail-closed。

**设计**（ enforcement 放在 `serve()`，`create_app` 保持库用途不变）：

```python
# server.py
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

def serve(model=None, *, host="127.0.0.1", api_key=None, ...):
    key_present = api_key is not None or Env.SERVING_API_KEY.value
    if host not in _LOOPBACK and not key_present:
        if not _allow_insecure():          # env: AUTOVISION_SERVE_ALLOW_INSECURE
            raise AutoVisionError(
                "Refusing to bind a non-loopback address without an API key. "
                "Set api_key= / AUTOVISION_SERVING_API_KEY, or set "
                "AUTOVISION_SERVE_ALLOW_INSECURE=1 to accept an open endpoint."
            )
```

- `api_key` 显式传入时行为不变；环境变量通道不变；K8s 探针端点
  （`/healthz`、`/readyz`）保持公开（`app.py:253` 既有设计）。
- `deploy/k8s/` 参考清单同步补 `AUTOVISION_SERVING_API_KEY` 注入示例。
- `known_limits.md` 的 serving 段改写：默认姿态从"可选 key"变为
  "非回环绑定强制 key"。

**测试计划**：`tests/test_serving.py` 新增：非回环 + 无 key →
`AutoVisionError`；非回环 + env key → 启动；env 允许不安全 → 启动
且 `/metrics` 记录 `insecure` 标志；回环 + 无 key → 行为不变。

**验收标准**：新用例通过；`CHANGELOG` 在 0.6.1 的**安全修复**节
显著标注（semver 允许 PATCH 内的安全行为翻转，需明示）。

### H2-2 并发与像素资源上界（P1×2）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：

1. `_read_body_limited` 只有单请求 8MB 上限（`app.py:115-141`）；
   排队协程数无上界，`serve()` 未暴露 uvicorn 的 `limit_concurrency`
   （`server.py:92-94`）。数千并发慢速连接 → 内存无界增长。
2. base64 解码内存放大：像素预算按**单图**计（`app.py:76` 默认 40M
   像素），已实测 6000×6000 纯色 PNG 仅 ~156KB 编码；`_predict_one`
   在引擎返回前持有整个 batch 的解码结果（`app.py:490-496`），
   `max_batch=16` → 单请求 ~1.7GB 瞬时内存。

**设计**：

1. `serve()` 新增 `limit_concurrency: int = 64`、
   `timeout_keep_alive: int = 30` 两个透传参数，进 uvicorn kwargs；
   docstring 标注"生产应再叠加网关层配额"。
2. `create_app` 新增 `max_request_pixels: int | None = None`（默认
   `max_batch * max_image_pixels`），`_predict_one` 解码循环内累计
   已解码像素，超限返回 413（错误文案区分单图超限与总量超限）。
3. `max_image_pixels` 默认值 40M → 16M（4000×4000，覆盖常见工业相机
   单帧），CHANGELOG 标注默认值变更与调整指引。

**测试计划**：并发上限用 uvicorn 参数透传断言（config 捕获）；像素
总量用例：batch=4 × 15M 像素图 → 413；batch 内累计边界值正好通过。

**验收标准**：`known_limits.md:118` 的"不限制一次请求的总像素"条目
删除并改为新语义描述。

### H2-3 poison 按变体隔离（P2→0.7.0，行为变更）

> 状态：已实施（工作区，随 0.7.0 切换）：poison Event 与并发信号量均按
> variant 划分；`/predict` 在健康变体内路由，`/readyz` 全部中毒才 503；
> `/healthz`/`/readyz` 携带 `poisoned_variants` 明细。测试：双 variant
> 参数化（单卡死另一 variant 仍 200、readyz 200 带明细；全中毒 503）。

**现状与证据**：`app.py:394-398` 单个全局 `inference_poisoned` Event；
`:706-723` 一个 variant 超时使**所有** variant 的请求 503，`/readyz`
整体摘除。多 variant（A/B）部署中非关键 variant 卡死拖垮全进程。

**设计**：poison 标志改为 per-variant（dict[variant_name, Event]）。
`/predict` 检查被路由 variant 的 poison（未指定 variant 的请求：任一
健康 variant 可服务即可）；`/readyz` 语义改为**全部** poisoned 才 503，
响应体保留 per-variant 明细（字段已存在，行为收紧即可）。0.6.1 先
输出 deprecation note（`/readyz` 语义变更预告），0.7.0 切换。

**测试计划**：`tests/test_serving.py` 既有 poison 测试组改造为
双 variant 参数化：仅一个卡死时另一个仍 200、readyz 200 且明细标记。

### H2-4 api_key 不落环境变量（P2）

> 状态：已实施（工作区，待发布于 0.6.1）。

**现状与证据**：`server.py:107-112` 把 `app_kwargs`（含显式传入的
`api_key`）整体 `json.dumps` 进 `AUTOVISION_SERVE_CONFIG`——同 UID
进程可经 `/proc/<pid>/environ` 读取，与 `app.py:270-273` "key 不进
argv/进程列表"的设计初衷自相矛盾。

**设计**：workers>1 路径检测 `config` 中含 `api_key` → `AutoVisionError`
并指引改用 `AUTOVISION_SERVING_API_KEY` 环境变量（子进程天然继承）。
workers=1 不经 env，不受影响。

**测试计划**：workers>1 + 显式 key → 报错；workers>1 + env key → 正常。

### H2-5 其余 serving P2（合并实施）

| 编号 | 问题 | 证据 | 设计方向 |
| --- | --- | --- | --- |
| H2-5a（已实施） | A/B 分桶键是客户端可控的 `X-Request-ID` | `app.py:434,475-482` | 分桶键改为服务端逐请求签发的 routing key；`X-AB-Variant` 加 `allow_variant_override: bool = False` 开关（默认忽略请求头，开启后未知变体仍 400） |
| H2-5b（已实施） | `X-Request-ID` 未清洗进日志（日志伪造） | `app.py:445-449` | 校验 `[0-9a-zA-Z-]{1,64}`，不合格截断+替换 |
| H2-5c（已实施） | `AutoVisionError` 原文回传（4xx 通道泄漏路径） | `app.py:889-899,878-888` | 对外映射为稳定短语 + request_id；原文进服务端日志 |
| H2-5d（已实施） | 400/413/429/503/504 无延迟指标 | `app.py:912-940` | 非 200/500 分支同样 `metrics.observe`，label 加 `error_class` |
| H2-5e（已实施） | graceful shutdown 默认无限等待 | `server.py:50,86-88` | `timeout_graceful_shutdown` 默认 30s，docstring 指引 |
| H2-5f（已实施） | depth 任务被摘要静默不可用 | `results.py:569` + serving | `/predict` 响应对 `summarized=true` 的 depth 附加 `depth_truncated: true` 字段；`known_limits.md` 注明应走文件/对象存储通道 |

---

## 8. 批次 3：P2 汇总修复

按子系统分组的 P2 清单（每项：一句话设计方向 + 目标批次内排期）。
行号锚定 v0.6.0。

### 8.1 契约层与消费方

| 编号 | 问题 | 证据 | 设计方向 |
| --- | --- | --- | --- |
| H3-1a（已实施）| dense 模式大数组无硬上限 | `results.py:613,621` | `masks="dense"` 时校验 `numel <= max_elements * 100`，超限抛 `AutoVisionError`（显式 opt-in 行为保留，但有一道闸） |
| H3-1b（已实施）| `(H,W,2)` 缺 `segment_ids` 误判为 N 个实例 | `results.py:391-408` | `_masks_are_per_instance` 增加"最后一维为 2 且 ndim==3"的 panoptic 启发式兜底 + 构造期 warning |
| H3-1c（已实施）| 混淆矩阵静默吞未声明类别 | `metrics/confusion.py:72-85` | `class_ids` 显式模式下遇未见 predicted/true 类别 → `logger.warning` 一次（含类名集合） |
| H3-1d（已实施）| `_summarize` 对非数值数组崩溃 | `results.py:149-157` | TypeError 分支扩大到 `np.min/max/mean` 调用；非数值摘要只输出 `shape/dtype/length` |
| H3-1e（已实施）| multi_stream 逐源序列化无隔离 | `inference/multi_stream.py:418-423` | `to_json` 包 per-source try，失败计 `frames_failed` 并写错误行，不影响其他源 |
| H3-1f（已实施）| `select()` 前置条件与构造校验两套判定 | `results.py:436-442` | 抽 `_is_panoptic_like()` 共用；`select(0)` 空结果抛 `AutoVisionError` 而非裸 `IndexError` |
| H3-1g（已实施）| `mask_to_bool` 整数路径把 -1 当前景 | `results.py:204-206` | 整数路径改为 `array > 0`（与 float 路径、panoptic -1=unassigned 约定对齐）；CHANGELOG 行为变更标注 |
| H3-1h（已实施）| 异常类型违反 errors.py 契约 | `results.py` 全部裸 `ValueError`；`annotators.py:438,449` raise 基类 | 新增 `ResultValidationError(AutoVisionError)` 替换裸 `ValueError`（旧异常类型是其父类，调用方 `except ValueError` 不破坏）；annotator 改抛具体子类 |
| H3-1i（已实施）| MaskAnnotator 分辨率不匹配静默跳过 | `visualize/annotators.py:356-357` | 与 SemanticMaskAnnotator 对齐：NEAREST resize 到帧尺寸，不一致时 debug 日志 |
| H3-1j（已实施）| NaN score 渲染为 "cat nan"；`filter_scores` 静默丢 NaN 行 | `annotators.py:140`、`results.py:469-476` | 渲染统一显示 `n/a`；`filter_scores` 对含 NaN 输入 warning 一次 |

### 8.2 数据层

| 编号 | 问题 | 证据 | 设计方向 |
| --- | --- | --- | --- |
| H3-2a（已实施）| YOLO/OBB 加载器静默跳过字段数不符行 | `yolo_object_detection_dataset.py:108-109` 等 | 0.6.1 计数 + 一次 warning（含文件与行号、丢弃条数）；0.7.0 默认 `raise`，保留 `on_malformed_line="skip"\|"raise"` 参数 |
| H3-2b（已实施）| 重复 image_id / 未声明 category_id 静默覆盖丢弃 | `coco_object_detection_dataset.py:101-124` | 重复 id → `AutoVisionError`（构造期即崩，符合"加载器 fail-loud"）；未声明 category_id → warning 计数（与 `merge_coco` 对齐） |
| H3-2c（已实施）| COCO 实例分割 O(图×标注) 双重循环 | `coco_instance_segmentation_dataset.py:58-64` | 预分组 `defaultdict(list)`（与检测加载器同形）；val2017 级数据构建时间从分钟级回到秒级 |
| H3-2d（已实施）| 实例分割/关键点加载器缺 path 遍历防护 | `coco_instance_segmentation_dataset.py:79` 等 | 抽 `_resolve_under(root, file_name)` helper（检测加载器 `:95-100` 同款），三处统一 |
| H3-2e（已实施）| 截断 JSON / 缺键裸崩 | `coco_object_detection_dataset.py:51-52` 等 | 统一包装为带文件名的 `AutoVisionError`（versioning.py:92-98 同款），缺键报出缺失键名 |
| H3-2f（已实施）| YOLO 归一化坐标零校验；`read_text()` 缺省编码 | `convert.py:104-116`、`:92` | 转换路径校验 `0<=v<=1` 且 `w,h>0`（越界报错含行号）；全部标签读取显式 `encoding="utf-8"` 并 `utf-8-sig` 容 BOM |
| H3-2g（已实施）| mmap refcount exists→open→increment 竞态 | `mmap_helpers.py:164-181` | 检查+增量移入 FileLock 临界区；Windows 的 `os.replace`-on-open 问题记录进 `known_limits.md`（不支持场景显式报错） |
| H3-2h（已实施）| 快照对文件消失/不可读裸崩；非原子写 | `versioning.py:201-213,294` | stat/open 包 `AutoVisionError`（含路径）；快照 JSON 走 `_write_atomic` |
| H3-2i（已实施）| stem 扁平化掩码匹配跨目录错配 | `image_dataset.py:113-116` 等 | 匹配键改为相对路径（保持 `.stem` 作为回退并 warning） |
| H3-2j（已实施）| quality 报告 markdown 注入 | `quality.py:188-189` | detail 转义扩展到换行/反引号/竖线 |

### 8.3 训练引擎与任务面

| 编号 | 问题 | 证据 | 设计方向 |
| --- | --- | --- | --- |
| H3-3a（已实施）| 非 rank0 RNG 不落盘，恢复后增广多样性坍缩 | `engine.py:1694-1741` | engine_state 增加 per-rank RNG 分片（`rng_rank_{i}`，rank0 汇总保存），恢复时各自读取 |
| H3-3b（已实施）| 恢复不恢复 sampler epoch | `engine.py:1285` | `_InfiniteCycleIterator` 接受起始 epoch；恢复路径透传（与 H1-4-1 合并实施） |
| H3-3c（已实施）| DDP 验证 padding 重复计入指标 | `engine.py:1505-1507,2187-2196` | metric merge 前按 sampler padding 计数加权剔除（`DistributedSampler` 的补齐索引可精确还原重复样本） |
| H3-3d（已实施）| 空 val shard 按 0 平均 | `engine.py:2183` | `sum/count` 两项分别 all_reduce（count=0 的 rank 不贡献分子分母） |
| H3-3e（已实施）| `world_size > len(dataset)` 空 epoch 无告警 | `engine.py:745-750` | 入口守卫扩展：分片后每 rank 样本数为 0 → 显式报错 |
| H3-3f（已实施）| `STEP_TIMEOUT_SEC` 声明未接线 | `runtime/env.py:194-196` | 接线到 D32 裁定/日志循环的步级看门狗，或删除声明（二选一，倾向接线） |
| H3-3g（已实施）| 恢复时数据集变化无检测 | `engine.py:1725-1737` | fit_args 快照附加数据集指纹（复用 data 层 `_data_fingerprint`），resume 时比对告警 |
| H3-3h（已实施）| batch=1+BN 无 SyncBN 转换 | 检测配置默认 | `fit_task` 检测 DDP + BN 模块 + `batch_size==1` 组合 → warning 指引 `convert_sync_batchnorm`（自动转换留给用户显式开启） |

### 8.4 导出与 registry

| 编号 | 问题 | 证据 | 设计方向 |
| --- | --- | --- | --- |
| H3-4a（已实施）| `precision="auto"` 用 fp16 宽容差校验可能 fp32 的引擎 | `export_onnx.py:1090-1099`、`tensorrt_helpers.py:308-312` | build 结果回读实际精度（flag/layer 精度），按实际精度选容差 |
| H3-4b（已实施）| TRT staged ONNX 未过门禁却留存 | `tensorrt_helpers.py:461-476` | 与 int8 路径对齐：构建成功后删除 staged ONNX（或 `keep_onnx=True` 显式保留） |
| H3-4c（已实施）| verify 钩子 `precision` 参数 12 处仅 3 处使用 | 各任务 task_model | 未使用处补 fp16 容差分支（复用 GTR 家族的分档表），或显式声明该任务仅支持 fp32 校验并 warning |
| H3-4d（已实施）| 工件不携带工具链版本 | `registry/assets.py` metadata schema | `.meta.json`/`.postprocess.json` 增加 `toolchain: {torch, onnx_opset, onnxruntime, tensorrt, openvino}`；InferenceEngine 加载时版本不匹配 → warning |
| H3-4e（已实施）| 同名 str marker 生成两个不同 Dim | `export_onnx.py:76` | marker 字符串按名字缓存复用同一 `Dim` 对象 |
| H3-4f（已实施）| 多输入 parity 喂同一张量 | `export_onnx.py:791-795` | 输入数 >1 时显式报"暂不支持多输入对拍"（当前单输入契约下安全） |
| H3-4g（已实施）| registry 版本混排崩溃 | `model_registry.py:89-95,330` | `_version_sort_key` 统一返回 `(0, tuple)` / `(1, str)` 可全序键；混排 warning |
| H3-4h（已实施）| 许可证 allowlist 双份漂移；模型卡 license 不归一 | `deployment_gates.py:30-32` vs `compliance/licenses.py:18-29`、`report.py:278` | allowlist 收敛为 `compliance/licenses.py` 单一来源；`report` 比对前过 `normalize_license` |
| H3-4i（已实施）| `downloadable` 整文件入内存/无锁/重复哈希 | `downloadable.py:113,166` | 分块流式写 + FileLock 互斥 + (path,size,mtime) 摘要缓存（user_held 同款） |
| H3-4j（已实施）| 晋升失败无审计记录 | `model_registry.py:295-302` | 失败也 `_append_audit`（event="promote_failed"），hash-chain 不变 |

### 8.5 推理运行时与跟踪

| 编号 | 问题 | 证据 | 设计方向 |
| --- | --- | --- | --- |
| H3-5a（已实施）| multi_stream `batch_size` 死参数 | `multi_stream.py:373-378` | 批组装内层循环 `for _ in range(batch_size)` 按队列深度取帧；docstring 更新 |
| H3-5b（已实施）| run_stream 无丢帧策略，live 延迟无界 | `stream.py:362-367` | 新增 `max_lag_frames` 参数（默认 None=现状；设置后 reader 在队满时 drop_oldest 并计数进 stats） |
| H3-5c（已实施）| 断流重连后 tracker/GMC 状态不重置 | `stream.py:311-321`、`gmc.py:67-69` | reader 通知重连事件 → 主循环调用 `tracker.reset()`/`gmc.reset()`/zone 清理；`StreamRunResult.reconnects` 语义修正（区分 clean EOF 与放弃重连） |
| H3-5d（已实施）| 重连预算不随恢复重置 | `multi_stream.py:333-337` | 成功重连后连续健康帧超过阈值（如 `fps*30`）时重置 `attempts` 计数 |
| H3-5e（已实施）| 变分辨率：单路崩溃/多路覆盖输出 | `engine.py:201`、`multi_stream.py:429-438` | 单路：帧尺寸与首帧不符 → 明确 `AutoVisionError`（提示固定分辨率要求）；多路：writer 轮换文件名 `annotated-{seq}.mp4` |
| H3-5f（已实施）| multi_stream 吞推理异常，持续 OOM 假成功 | `multi_stream.py:386-395` | 滑动窗口失败率 > 0.5 且 ≥10 帧 → 中止该源并标记 error；summary 增加 `degraded: true` |
| H3-5g（已实施）| 相机索引 docstring 与实现不符 | `video.py:85-86,119` | docstring 明确"索引需传 int；字符串按文件路径解析"；int 类型分支保留 |

### 8.6 CLI、指标与治理

| 编号 | 问题 | 证据 | 设计方向 |
| --- | --- | --- | --- |
| H3-6a（已实施）| CLI 错误面三种行为 | `cli/_app.py:436-450,859` 等 | 统一：`AutoVisionError` → stderr 一行 + exit 1；用法错误 → `BadParameter` exit 2；`plot-curves` 缺参改 `BadParameter` |
| H3-6b（已实施）| wrapper 命令静默忽略未知键 | `_app.py:534-538` | `download`/`predict` 等补 `_reject_unknown_keys`（data-snapshot 同款） |
| H3-6c（已实施）| mAP headline F1 逐操作点类集合不一致；`auto_raise` 跨轮副作用 | `metrics/map_evaluator.py:326-350,182-190` | 文档声明口径（val 输出 footer 注明"headline 为 argmax-F1 点读数，非 COCO summary"）；`auto_raise` 改为只读提示不修改状态 |
| H3-6d（已实施）| crowd-ignore 忽略类别一致性 | `error_analysis.py:306-311` | ignore 判定要求同类 crowd（对齐 pycocotools），注释更新 |
| H3-6e（已实施）| 离线模式类别 id 按排序映射易错位 | `cli/analyze_errors.py:120-124` | predictions JSON 侧若带 `categories` 则按其显式映射；否则 warning 声明排序假设 |
| H3-6f（已实施）| HPO 拼错键逐 trial 失败被吞 | `cli/_app.py:737-742` | trial 失败原因聚合进最终错误（首个 TypeError 原文透出） |
| H3-6g（部分：实测 31.8%<40%，阈值保持 30）| coverage 30% 弱阈值；mypy 白名单小 | `.github/workflows/ci.yml:27,102-123` | coverage 阈值提升路线：0.7.0 → 40%（核心 19 文件 70%），myty 白名单每迭代 +2 模块（results.py、inference/stream.py 优先） |
| H3-6h（已实施）| AGENTS.md "30 条命令" 漂移 | 实测 31 条 | 文档改为"31 条"（README 全表补漏 `tile-dataset`），`tests/test_infrastructure.py` 新增对账用例：AGENTS.md 声明数 vs `_app.py` 注册数 |

---

## 9. 批次 4：机制化（防复发）

### H4-1 长时运行老化测试套件

> 状态：已实施（tests/test_soak.py，AUTOVISION_RUN_SOAK=1 门控）。

新增 `tests/test_soak.py`（本计划新增文件；标记 `@pytest.mark.soak`，默认跳过，<!-- docpath-ok -->
`AUTOVISION_RUN_SOAK=1` 开启并纳入 nightly）：

- 1000 帧闪烁检测 + `track=True`：断言 `len(tracker.tracks)`、
  `len(tracker._removed)`、RSS 增长均有界（对应 H1-3）。
- 慢批 + EOF 时序（对应 H1-2）：确定性 sleep 构造，断言墙钟上限。
- multi_stream 100 帧 × 3 源：断言 per-source 统计闭合、无句柄泄漏
  （`psutil` 可选依赖，缺失时跳过 RSS 断言）。

### H4-2 并发测试基建

> 状态：已实施。mmap 双进程 refcount（`tests/test_mmap.py::
> test_double_process_open_race_keeps_refcount_exact`，4 进程参数化）；
> 并发下载互斥（`tests/test_registry_hardening2.py::
> test_concurrent_resolve_serializes_on_file_lock`）；serving 并发上限
> 透传断言（`tests/test_serving.py` uvicorn config 捕获）。

- mmap refcount：`tests/test_mmap.py` 新增双进程用例
  （`multiprocessing` + barrier），断言 refcount 终值正确、无幽灵引用
  （对应 H3-2g 的回归防护）。
- 并发下载：两个进程 resolve 同一未缓存资产，断言产物 sha256 正确且
  无交错字节（对应 H3-4i）。
- serving 并发上限：uvicorn config 捕获断言（对应 H2-2）。

### H4-3 意图→测试锚定（design-anchor 检查）

> 状态：已实施（scripts/check_design_anchors.py + allowlist，CI lint job 接线）。

新增 `scripts/check_design_anchors.py`（本计划新增文件；进 CI lint job）：<!-- docpath-ok -->

1. 输入：全仓库扫描 `D\d+[a-z]?` 与 `H\d+-\d+[a-z]?` 编号的出现点
   （源码注释/docstring、docs、tests）。
2. 规则：每个编号必须至少被一个 `tests/` 下的文件引用（docstring 或
   注释中显式写编号），否则 exit 1 并列出缺失清单。
3. 豁免表：`scripts/design_anchor_allowlist.toml`，条目带 `expires`
   日期，过期即失败（沿用 provenance allowlist 的既有机制）。
4. 首次运行生成完整清单，已知缺口按批次回填；本文档所有 H 编号在
   对应批次完成时必须已有锚定测试。

### H4-4 测试布局守卫

> 状态：已实施（scripts/check_test_layout.py，CI lint job 接线）。

新增 `scripts/check_test_layout.py`（本计划新增文件；进 CI lint job）：AST 扫描 <!-- docpath-ok -->
`tests/**/*.py`，断言所有 `test_*` 函数/方法定义在模块顶层或类体
第一层，禁止嵌套定义（H0-1 的结构性防复发）。成本 O(文件数)，
无第三方依赖。

### H4-5 导出侧 golden 数值基线

> 状态：已实施（工作区）。`tests/test_export_golden.py` 为全部 8 个任务面
> （det / obb / pose / depth / semseg / cls / instance / panoptic）建立
> ORT CPU 输出快照（`tests/golden/export_<face>.json`，atol 1e-5）：
> 固定种子微型模型 → `export_onnx(verify=True)` → 固定种子输入 →
> ORT 输出逐值比对（名称/顺序/形状断言 + 数值容差）。录制走
> "不存在即生成并失败一次"策略（禁止 record-and-skip），与 byte_track
> golden 同语义。

`tests/golden/` 现仅覆盖跟踪生命周期/蒸馏损失/训练轨迹。新增：

- 固定种子小模型（每任务面一个最小 TaskNetwork 变体）+ 固定输入 →
  ONNX 导出 → ORT CPU 推理输出快照（`tests/golden/export_<task>.json`，
  容差 atol 1e-5）。捕获：decode/后处理数值漂移、opset 变化引入的
  数值差、fp16 黑名单算子集合变化。
- 快照生成走"不存在即生成并失败一次"的策略（禁止 record-and-skip）：
  首次运行写文件后断言失败并提示二次运行验证，杜绝 golden 误删后
  CI 首轮放行（对应 `tests/test_byte_track_golden.py:137-140` 的
  既有弱点，同批改造该文件）。

### H4-6 CI 阈值演进与文档对账

> 状态：部分实施——命令数对账测试已落地（H3-6h）；coverage 阈值保持 30
> （H3-6g：实测 31.8%，40% 随核心文件补测后提升）；`AUTOVISION_GOLDEN_STRICT`
> 不再需要——byte_track golden 已改为无条件 record-then-fail（H4-5），
> 严格性是默认行为而非环境开关。

- coverage：`--cov-fail-under` 30 → 40（0.7.0），随批次递增。
- 文档对账：`tests/test_infrastructure.py` 增加断言——CLI 注册命令数
  与 AGENTS.md 声明一致（防 P3 类文档漂移）。
- `AUTOVISION_GOLDEN_STRICT=1` 在 CI test job 设置，golden 缺失即失败。

---

## 10. 测试矩阵与验收门禁（每批次 Definition of Done）

每个批次合并前必须全部满足：

```bash
# 1. 静态检查
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --config-file pyproject.toml <本批触达模块>

# 2. 目标回归（按批次选择对应文件，全部 -q 通过）
# 批次0: tests/test_stream_pipeline.py
# 批次1: tests/test_error_analysis.py tests/test_tracking.py
#        tests/test_stream_pipeline.py tests/test_engine_recovery.py
#        tests/test_distributed.py tests/test_engine_cache_and_retry.py
#        tests/test_deployment.py tests/test_downloads_resume.py
#        tests/test_mask_serialization.py tests/test_serving.py
# 批次2/3: 见各编号的测试计划

# 3. 全量测试（双 Python 版本由 CI 执行）
.venv/bin/python -m pytest tests -q

# 4. 工程门禁
.venv/bin/python scripts/check_provenance.py
.venv/bin/python scripts/check_layering.py
.venv/bin/python scripts/check_docs_links.py
.venv/bin/mkdocs build --strict
```

外加：`CHANGELOG.md` 按节归位（修复/变更/安全修复）；涉及行为变更的
条目在 `docs/known_limits.md` 同步改写；新公共参数写入
`docs/` 对应页面；本文档中被实施编号的"状态"行更新为
`已实施于 vX.Y.Z`。

## 11. 兼容性与弃用策略汇总

| 变更 | 类型 | 处置 |
| --- | --- | --- |
| serving 非回环绑定强制 key（H2-1） | 安全修复 | 0.6.1 直接生效，CHANGELOG 安全节显著标注 |
| `max_image_pixels` 默认 40M→16M、请求总像素上限（H2-2） | 行为变更 | 0.6.1 生效；超限报 413 带调整指引 |
| `/readyz` 全 poisoned 才 503（H2-3） | 行为变更 | 已随本工作区切换（原计划 0.6.1 预告 → 0.7.0 切换，因批次 3 同批发布而合并） |
| A/B 分桶键服务端签发、`X-AB-Variant` 需 `allow_variant_override=True`（H2-5a） | 安全修复 | 同 X-Request-ID 不再粘住变体；需要定向的测试/网关显式开启 |
| float mask RLE 二值化 → 拒绝（H1-11） | 弃用 | 0.6.1 warning → 0.7.0 raise |
| YOLO 畸形行静默跳过 → raise（H3-2a） | 弃用 | 0.6.1 warning + 计数 → 0.7.0 raise + `on_malformed_line` 参数 |
| `mask_to_bool` 整数 -1 语义（H3-1g） | 行为变更 | 0.7.0 生效，CHANGELOG 标注 |
| 多卡数据顺序变化（H1-4） | 修复副作用 | CHANGELOG 标注；resume 漂移告警已覆盖 |
| dataset pin 生效（H1-9） | 收紧 | fail-closed 报错带 LEDGER 指引 |
| `ResultValidationError`（H3-1h） | 新增子类 | 旧 `except ValueError` 代码不破坏 |

## 12. 风险与回滚

- **H1-2/H1-3（流与跟踪）**：sentinel 无限重试的理论风险是 reader
  线程在"队列永不再空"时悬挂——但主循环不消费就不会有下一帧，等价于
  现状挂起，无新增风险面；主循环守卫兜底。回滚单元：两修复分别独立
  commit，可单独 revert。
- **H1-4（种子）**：改变多卡数据顺序，任何依赖旧顺序的回归对比会
  失效。回滚：revert 即恢复；CHANGELOG 明示版本分界。
- **H1-9（pin）**：上游换文件会阻断下载。缓解：LEDGER 登记 + 报错
  文案给出 `AUTOVISION_OFFLINE=1` 与镜像替代路径；不提供全局豁免开关
  （严格模式已是 opt-in，此处按默认强制）。
- **H2-1（fail-closed）**：现有"0.0.0.0 + 无 key"用户升级即启动失败。
  这是预期行为；文档与错误信息双通道指引。
- **H1-11/H3-2a（弃用翻转）**：0.7.0 的 raise 可能打断存在脏数据的
  老用户。缓解：0.6.1 起的 warning 计数可在训练日志中检索
  `deprecation` 关键字提前发现。

## 13. 实施顺序（关键路径）

```
批次0 (H0-1, H4-4 提前落地)
  └→ 批次1: H1-1 ─ H1-2 ─ H1-3（流/跟踪联调）
             H1-4 ─ H1-5（引擎）
             H1-6 ─ H1-7 ─ H1-8（导出）
             H1-9 ─ H1-10（数据）
             H1-11 ─ H2-1（契约/服务）
  └→ 批次2: H2-2 … H2-5
  └→ 批次3: 按 8.1→8.5→8.6 顺序（8.3 的 H3-3b 依赖 H1-4 合入）
  └→ 批次4: H4-3/H4-4 随 0.7.0 合入并回填锚定
```

每个编号独立 commit，标题前缀 `fix(hardening): H1-x <摘要>`；每批
完成后在本文档对应小节追加 `> 已实施于 vX.Y.Z（PR #…）`。
