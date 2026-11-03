# 设计提案：ultralytics 8.4 借鉴实施设计

> 状态：设计提案（未实施）。本文档是对
> `ultralytics 8.4.177`（下称"上游"）逐子系统分析后形成的可执行设计，
> 每一项都给出了 AutoVision 侧的现状核对、接口草图、涉及文件与验收标准。
> 实施顺序与依赖见 §6；合规边界见 §2。

---

## 1. 背景与结论摘要

上游 8.4.x 相对大家熟悉的旧版有几个结构性变化，影响借鉴判断：

- `nms.py` 从 `ops.py` 独立为纯 torch 实现；`tuner.py` 新增 Ray Tune 路径
  但独立遗传进化 `Tuner` 仍在（`use_ray=False` 默认路径；首轮分析误判为
  "已删除"，已勘误）；新增 `utils/autodevice.py`（空闲 GPU 选择）、
  `events.py`（遥测）、`triton.py`（Triton 服务客户端）。
  另：`utils/instance.py` 本版已无 `xyxyr` 旋转存储，OBB 走 `(N,5)` 数组 +
  ops 层转换；`preserve_obb` 是 segment 层面的角度保持裁剪。
- **depth 与 semantic segmentation 升为正式任务面**，与 AutoVision 的 8 任务
  面对齐，其导出元数据、指标与增广设计对 AutoVision 有直接参照价值。
- 导出侧新增格式×参数×精度三张**声明式矩阵**与 `try_export` 门禁装饰器。
- `AutoBackend` 重构为 `nn/backends/` 插件 + **导出工件元数据读回**。

**结论**：AutoVision 在结果契约（`InferenceResult`）、导出 parity 门禁、
sha256 下载、Pydantic 配置、HPO（ASHA）、训练基建（EMA / NaN 守卫 / 原子
checkpoint）上已经覆盖甚至严格于上游，**这些不是借鉴对象**。真正值得借鉴的
集中在四块：

1. **导出工件元数据闭环**（写全 + 推理端校验读回）——P0；
2. **追踪器架构**（退役通知、检测索引回写、多路流生命周期、GMC、钩子化与
   BoT-SORT）——P0/P1；
3. **训练侧 DX 基建**（autobatch、扫描缓存、box_candidates、图像读取缓存、
   下载续传）——P0/P1/P2；
4. **solutions 产品层**（快速穿越补偿计数、多区域计数、测速、热力图）——P3。

每项的编号 `D1`–`D31` 在全文通用（D1–D14 第一轮；D15–D31 第二轮，
2026-11-03 追加），实施 PR 应在标题引用编号。

**第二轮（同日追加）**：在 D1–D14 落地核实完成后，对上游做了第二轮全库
逐子系统分析（engine / data / utils·nn·optim·models / 本仓库缺口盘点），
新发现 D15–D31 共 17 项，集中于：训练侧加固（AMP 事前守卫、resume 白名单、
save_period 序列化卫生、OOM 降批、回调注册表）、推理侧（构建期/运行期
参数二分 + handle 级模型复用、空闲 GPU 自动选卡）、数据面（COCO 交换
健壮性、增广四目标钩子协议、扫描缓存模板化、深度/语义协议）、评估面
（指标聚合容器 + 实时混淆矩阵）。另确认两处首轮勘误（见 §1 更正与 §5
新增不借鉴项）。

## 2. 合规红线（AGPL）与溯源要求

上游是 **AGPL-3.0**，AutoVision 是 Apache-2.0。`NOTICE` 以两段纪律界定
边界：其一，可选运行时集成（extras + 运行时下载）**必须永不把上游源码或
权重拷入本仓库**；其二（2026-11-03 新增），AGPL 源可作**架构参考**驱动
本仓库内的净室重实现，但禁止逐行翻译与逐字复制注释 / docstring /
常量表 / 测试数据，落地文件逐个登记台账并附净室声明，商业分发前复核。

因此本文档所有设计项遵循以下红线，全部适用于 §6 的每个 PR：

1. **净室重实现**：只借鉴"设计思想 / 算法步骤 / 接口形态"，实现必须在
   AutoVision 的接口与风格内重写；禁止对照上游逐行翻译，禁止复制上游注释、
   docstring、常量表与测试数据。
2. **运行时依赖不引入**：不在核心路径 import 上游包；`encoders/ultralytics`
   的惰性导入边界维持不变（GMC 等涉及 `cv2` 的新功能按 `video` extra 处理，
   见 D6）。
3. **溯源登记**：每项落地时在 `docs/provenance/LEDGER.md` §1 追加一行，
   方式标注为"架构参考；AutoVision 实现"（模板见 §9）；
   涉及许可边界判断的（尤其 D6/D7 的上游 tracker 语义），商业发布前按台账
   头部要求做法务复核。
4. **参考索引仅指路**：文中给出的上游 `file:line` 是给实现者**读设计**用的，
   不是拷贝清单。

## 3. 现状盘点：不重复建设的部分

以下能力已核对确认存在，相关 PR 不得重复实现；改造它们时只做增量。

| 领域 | AutoVision 现状 | 上游对应物（不借鉴原因） |
|---|---|---|
| 结果契约 | `InferenceResult` + `keys.py` 键名契约 + RLE 边界约束 | engine/results.py（`Results`/`BaseTensor` 无明显优势） |
| 导出门禁 | `export/parity_checks.py`（ORT parity、贪心行匹配）+ `e2e_parity.py` + INT8 报告 + `registry/deployment_gates.py` | `exporter.py` / `benchmarks.py`（结构不同；仅吸收计时方法论，见 D5） |
| 数据下载 | `data/downloads.py`：sha256 钉死 + 原子发布 + tar/zip 成员防护 | 上游无哈希校验（仅吸收断点续传，见 D11） |
| 配置校验 | Pydantic + `difflib` 模糊建议（`config/validate.py`） | frozenset 类型注册表（同理念，无优势） |
| HPO | `hpo/`（ASHA + 搜索空间） | Ray Tune（更重，不引入） |
| 数据质检 | `cli/data_check.py`（含 near-duplicate）、`data/versioning.py` 快照 | `verify_image_label` |
| 训练基建 | `engine/engine.py`：EMA、NaN/Inf 梯度守卫、原子 checkpoint、resume、DDP | 相当 |
| 事件/观测 | `observability/events` + loggers | `events.py`（遥测形态不同，不借鉴） |
| 自动预标注 | `cli/pseudo_label.py`（teacher→COCO） | 上游 data/annotator.py 的 SAM 链路（依赖 AGPL 上游，不引入） |
| SAHI | `tasks/shared/sahi.py` + `tiling_utils.py`（推理侧 tile 合并） | `solutions` 无对应；训练侧切片见 D9 |

## 4. 分项设计

优先级含义：P0 = 立即做（低成本、高确定性收益）；P1 = 近期（1–2 个迭代）；
P2 = 中期（按数据/需求驱动）；P3 = 按需（产品需求触发再启动）。

### 4.1 D1 导出工件元数据补全与一致性校验（P0）

**动机与上游参考**：上游把 `task/names/stride/imgsz/batch/end2end` 等全部写
进 ONNX `metadata_props`（上游 engine/exporter.py 的元数据 dict 与各格式写入点），
推理端 `AutoBackend` 加载即读回，自动配置 batch / imgsz / 后处理。AutoVision
的 `export/export_onnx.py::onnx_export_metadata()` 目前只写
`autovision_version / license_info / model_name / image_normalize /
num_classes`，且 `inference/engine.py` 完全不读回。

**架构差异声明（非目标）**：AutoVision 的推理契约是"pre/post 留在
`TaskNetwork`，artifact 只换 forward"（`inference/engine.py` 模块
docstring），这与上游 AutoBackend 的"artifact 自足"路线不同，**不做
artifact-only 推理**。D1 的目标限定为：工件自描述 + 网络↔工件一致性校验。

**设计**：

1. 新模块 `autovision.export.artifact_metadata`：

   ```python
   META_SCHEMA_VERSION = 1

   def build_metadata(network: TaskNetwork) -> dict[str, str]: ...
   def read_metadata(artifact: Path, backend: str) -> dict[str, str]: ...
   def verify_metadata(
       metadata: dict[str, str], network: TaskNetwork, *, artifact: Path
   ) -> None: ...
   ```

2. 元数据 schema（全字符串键值，前缀 `autovision_`）：

   | 键 | 值 | 说明 |
   |---|---|---|
   | `autovision_meta_version` | `"1"` | schema 版本，未来演进依据 |
   | `autovision_task` | 如 `detection` | 任务面名 |
   | `autovision_class_path` | `TaskNetwork` 子类全路径 | 诊断用 |
   | `autovision_image_size` | JSON `[H, W]` | 网络输入尺寸 |
   | `autovision_num_classes` | 整数（已有，保留） | |
   | `autovision_names` | JSON 类别名列表 | 超过 512 类截断并写 `names_truncated=true` |
   | `autovision_input_layout` | `"NCHW"` | 预留 NHWC |
   | `autovision_image_normalize` | JSON（已有，保留） | |
   | `autovision_version` / `license_info` | （已有，保留） | |

   各格式载体：ONNX → `metadata_props`（沿用 `_write_onnx_metadata`）；
   OpenVINO → `rt_info`（经 `export/openvino_helpers.py`）；TensorRT →
   engine 旁路 `<stem>.meta.json` sidecar（与既有量化 metadata 的旁路发布
   方式一致，原子写）。

3. `InferenceEngine.__init__`（`backend != "torch"` 分支）调用
   `verify_metadata`：比较 `task / num_classes / image_size`；不一致抛
   `AutoVisionError`，报错文案给出两侧实际值与修复建议（重新导出或换
   network）。metadata 缺失（旧工件）→ `logger.debug` 降级放行，保持兼容。

4. `export_onnx` 导出完成后的日志/回显附带元数据摘要（见 D14c）；
   parity 门禁报告新增一行 `artifact_metadata: ok/mismatch`。

**涉及文件**：`export/export_onnx.py`、新增 `autovision.export.artifact_metadata`、
`inference/engine.py`、`export/openvino_helpers.py`、
`export/tensorrt_helpers.py`（sidecar 读写）、`export/e2e_parity.py`（报告行）。

**测试**：三种格式 write→read roundtrip；task/类别数不匹配各一条失败用例；
无 metadata 旧工件兼容；`test_gtr_export_gates.py` 补 metadata 一致性断言。

**成本**：2–3 人日。**风险**：低；TRT sidecar 需与既有量化 metadata 的写入
顺序（原子发布前）核对，避免双文件竞争。

### 4.2 D2 追踪器状态卫生：退役通知、检测索引回写、多路流生命周期（P0）

**动机与上游参考**：上游 tracker 钩子把"检测在原始预测里的索引"保留为输出
最后一列（`trackers/track.py`），下游可溯源；tracker 退役的轨迹通过
`removed_stracks` 暴露给 solution 层清理状态（`solutions/solutions.py` 的
`forget_tracks`）；多路流按 batch slot 一流一 tracker，源切换时 reset
（`persist` 语义）。

**现状核对（含一处更正）**：`tracking/byte_track.py` 的模块级 `_KALMAN`
单例**只读**（常量矩阵、所有方法纯函数），多路流下并不会互踩——此前分析
中的"状态污染"判断不成立，无需为正确性去单例；但实现上改为
`BYTETracker.__init__` 内持有实例以消除模块级可变全局的歧义，属顺手项。
真正的缺口是下面三条。

**设计**：

1. **检测索引回写**：`Track` dataclass 增加字段
   `det_index: int = -1`。`BYTETracker.update()` 在三段匹配中维护
   "track ← 检测"配对的原始检测下标（stage 2/3 的低分/剩余池来自同一输入
   数组，索引可直接传递）；未匹配跟新的 track 填 `-1`。行为变化仅为新增
   字段，向后兼容。

2. **退役通知**：`BYTETracker` 新增
   `def drain_removed_tracks(self) -> list[Track]:`——把本帧被置为
   `removed` 的轨迹（unconfirmed 未发布即删的除外，它们从未发布、无需
   通知）以 `state="removed"` 的 `Track` 形式返回一次后清空。`update()`
   返回值不变。

3. **多路流生命周期**：`inference/multi_stream.py` 目前完全没有 tracking
   集成。设计约定（落地时机随 D13 或独立小 PR）：每路流一个
   `BYTETracker` 实例（keyed by stream id），source 重连/切换时调用
   `tracker.reset()`——为此给 `BYTETracker` 增加 `reset()`
   （清 `tracks / _next_id / frame_id`），与 `LineZone.reset()` 对齐。
   在 `docs/framework_design.md` 的 stream 小节补一段生命周期约定：
   "tracker/zone 状态机属于单路流，禁止跨流共享或缓存在流循环之外"
   （与 AGENTS.md §7 既有条目呼应）。

**涉及文件**：`tracking/byte_track.py`、`inference/stream.py`（消费
`det_index` 与退役通知的可选路径）、`inference/multi_stream.py`（注释级
约定先行）。

**测试**：三段匹配的 `det_index` 正确性（构造高/低分混合序列）；
`drain_removed_tracks` 只返回一次、unconfirmed 不出现；`reset()` 后 id
从 1 重新计数；既有 `test_tracking` 全绿。

**成本**：1–2 人日。

### 4.3 D3 box_candidates 劣化目标过滤与 OBB 保持裁剪（P0）

**动机与上游参考**：上游任何几何增广后按"宽高下限、长宽比上限、面积比"
过滤劣化框（`data/augment.py::box_candidates`，seg/OBB 阈值不同），OBB 用
多边形∩画布再 `minAreaRect` 保角度（`utils/instance.py::clip` 的
`preserve_obb` 分支）。AutoVision 的
`augment/detection/transform.py`（仿射 + mosaic）与
`oriented_transform.py` 目前没有等价过滤；小目标经旋转/裁剪后产生的
碎框会直接污染训练标签。

**设计**：

1. 新模块 `autovision.augment.detection.box_candidates`（纯函数，
   numpy/torch 双支持，无新依赖）：

   ```python
   def box_candidates(
       boxes_before: Tensor, boxes_after: Tensor, *,
       min_wh: float = 2.0, max_aspect_ratio: float = 100.0,
       min_area_ratio: float = 0.1,
   ) -> Tensor: ...          # (N,) bool

   def obb_candidates(
       polys_before: Tensor, polys_after: Tensor, canvas_hw: tuple[int, int], *,
       min_area_ratio: float = 0.01,
   ) -> Tensor: ...          # 多边形面积比 + 裁剪后仍为凸四边形
   ```

2. 接线点（过滤后必须同步删除对应 label，避免框/标签错位）：
   - `transform.py`：仿射参数应用之后、`_build_mosaic` 拼接 + clip 之后；
   - `oriented_transform.py::__call__`：对 `polygons` 用 `obb_candidates`。
   - SAHI/tiling 不在此处接线（推理侧已有 per-label 过滤）。

3. **OBB 保持裁剪**：`geometry/` 新增
   `clip_polygon_to_canvas(poly, (H, W)) -> poly | None`
   （Sutherland–Hodgman 纯 numpy），供 `obb_candidates` 与后续 tile 场景
   复用；不引入 shapely。

**涉及文件**：如上 + 两个 transform 的 args（阈值进
`TaskTransformArgs` 子类，带默认值，方便任务面覆盖）。

**测试**：合成用例（旋转 45° 出界、缩放到 1px、长宽比爆炸）验证过滤；
label 同步删除；OBB 裁剪保持角度（与 minAreaRect 对照的近似断言）。

**成本**：1–2 人日。

### 4.4 D4 autobatch 训练批量自动探测（P1）

**动机与上游参考**：上游 `utils/autobatch.py` + `torch_utils.profile_ops`：
候选 batch 逐个做 forward+backward 显存剖析（模拟 assigner/loss 的中间
分配）、显存目标 = 可用显存 × fraction（默认 0.6）、对 (batch, mem) 做一阶
多项式拟合外推最优值、OOM 候选回退前一安全档、CPU/MPS/cudnn.benchmark
直接跳过。AutoVision **没有**任何自动批量能力（已 grep 确认），
`cli/train.py` 的 `batch_size: int` 需要用户手猜。

**设计**：

1. 新模块 `autovision.engine.autobatch`：

   ```python
   def probe_train_batch_size(
       step_fn: Callable[[int], None],   # 接收 batch_size，内部完成
                                          # 取数 -> forward -> backward -> step
       *, device: torch.device,
       fraction: float = 0.6,
       candidates: Sequence[int] | None = None,  # 缺省 [1,2,4,8,16]+(32,64)
       dataset_size: int | None = None,
   ) -> int: ...
   ```

   剖析协议：每个候选 `torch.cuda.reset_peak_memory_stats()` →
   `step_fn(b)` 前向+反向（autocast 与正式训练一致）→ 记录
   `max_memory_allocated + max_memory_reserved` 增量；OOM（`OutOfMemoryError`
   / RuntimeError 匹配）标记该候选为天花板；拟合仅在显存随 batch 单调的
   子集上做，`finally` 里 `empty_cache()`。AutoVision 不照搬"模拟
   assigner 分配"——各任务 step 形态差异大，直接用**真实 step 闭包**
   （detection/pose 用含 matcher 的完整 loss，pretext recipe 用其
   `training_step`），显存画像天然含 assigner/队列分支。

2. CLI/参数接入：`cli/train.py` 的 `batch_size` 接受 `int | "auto"`，
   新增 `batch_fraction: float = 0.6`。`"auto"` 时在构建 dataloader **之前**
   探测（dataloader 依赖 batch_size，顺序必须是 probe → build）；探测用的
   step 闭包由 `train_task/` 各任务注册（与既有 transform_args 注册机制同
   源）。`get_global_batch_size` 流程不变，probe 返回的是**每设备** batch。

3. DDP：单卡探测（rank 0，其余 rank 等待 barrier），结果 broadcast；
   world_size>1 时目标显存按单卡可用计算。

4. 护栏：CPU/MPS/`cudnn.benchmark=True`/数据集为空 → 返回默认值并
   `logger.info` 说明跳过原因（照搬上游的护栏清单，不照搬实现）。

**涉及文件**：新增 `autovision.engine.autobatch`；`cli/train.py`、
`cli/train_helpers.py`、`cli/train_task/`（step 闭包注册）。

**测试**：mock 显存曲线的单调拟合/回退分支（不依赖 GPU）；GPU 可用时的
smoke（CI 标记 skipif）；`--batch-size auto` 端到端最小训练 1 epoch。

**成本**：3–5 人日。**风险**：探测与真实训练的显存有偏差（fragmentation），
默认 fraction 0.6 留足余量；文档注明 probe 结果是建议值，可被
`AUTOVISION_AUTOBATCH_MAX` 上限钳制。

### 4.5 D5 基准计时稳定性与指标下限门禁（P1）

**动机与上游参考**：上游 `utils/benchmarks.py` 两个方法论级技巧：**迭代
2σ 剔离群**计时样本（最多 3–5 轮），以及**按实测单次耗时自适应运行次数**
（凑够最小测量窗口而非固定步数）；其矩阵式验证中失败保留为"❎"行不中断、
`verbose=0.x` 直接作为全部格式 metric 的下限断言。AutoVision
`export/benchmark.py::benchmark_inference` 目前固定 warmup=5/steps=20，
无离群处理；parity 门禁有 atol/rtol 但没有"跨后端 metric 下限"这一
CI 级断言形态。

**设计**：

1. 新模块 `autovision.export.timing`：

   ```python
   def iterative_sigma_clip(samples: list[float], *, sigma: float = 2.0,
                            max_rounds: int = 5) -> np.ndarray: ...
   def adaptive_run_count(single_run_s: float, *, target_s: float = 10.0,
                          floor: int = 20, ceiling: int = 500) -> int: ...
   ```

2. `benchmark_inference(..., *, robust: bool = False,
   min_measure_seconds: float = 10.0)`：`robust=True` 时 warmup 后先试跑
   3 次估单次耗时 → `adaptive_run_count` 定步数 → 采集后
   `iterative_sigma_clip`，`BenchmarkResult` 新增
   `latency_ms_mean_robust / latency_ms_std / outliers_dropped` 字段。
   默认 `False` 保持既有基准表可比；`docs/benchmarks.md` 与
   `--robust` CLI 旗标在 0.5 周期再翻转默认。

3. 指标下限门禁：`registry/deployment_gates.py` 新增 gate
   `metric_floor`（参数：跨后端逐一声明 `{backend: floor}` 或统一
   floor），parity/benchmark 报告未达 floor 即 gate 失败——把上游
   `verbose=floor` 的断言形态移植进 AutoVision 既有 gate 框架，而不是新造
   报告体系。

**涉及文件**：新增 `autovision.export.timing`；`export/benchmark.py`、
`registry/deployment_gates.py`、`cli/export.py`（`--robust` 透传）。

**测试**：合成延迟样本（注入尖峰）验证 clip 与自适应次数；gate 通过/失败
两分支；`benchmark_inference(robust=True)` 在 CPU 上 smoke。

**成本**：2–3 人日。

### 4.6 D6 GMC 相机运动补偿（P1）

**动机与上游参考**：上游 `trackers/utils/gmc.py` 提供
sparseOptFlow / ORB / SIFT / ECC / none 五种相机运动补偿，downscale=2，
可用检测框排除前景关键点，失败回退单位阵。固定机位之外（无人机/车载/
手持）没有 GMC 的 LineZone 计数会大量 ID switch。AutoVision tracking
目前是纯 numpy（无 cv2）。

**设计**：

1. 新模块 `autovision.tracking.gmc`：

   ```python
   class GMCEstimator:
       def __init__(self, method: str = "sparse_opt_flow", *,
                    downscale: int = 2, detect_shadows: bool = False): ...
       def apply(self, frame_gray: np.ndarray,
                 det_boxes: np.ndarray | None = None) -> np.ndarray:  # (2,3)
   ```

   内部保存上一帧（降采样灰度图 + 关键点），`apply` 估计本帧相对上一帧的
   2×3 仿射；`det_boxes` 提供时先在框内做 mask 再取特征（排除前景）。
   任何估计失败返回单位阵并 `logger.warning`（每 tracker 生命周期只警告
   一次）。

2. 依赖边界（AGENTS.md §3.2）：GMC 需要 `cv2`（`video` extra）。实现必须
   **函数内惰性 import**，缺失时抛出带 `pip install -e '.[video]'` 的
   `AutoVisionError`；`tracking/__init__` 顶层不 import `gmc`。

3. `BYTETracker` 接线：构造参数
   `motion_compensation: str = "none"`；`update()` 在 `predict()` 之后、
   stage 1 之前调用 GMC，把 2×3 仿射作用于所有存活 track 的框（中心平移，
   宽高不变；旋转分量由 method 选择约束——默认 `sparse_opt_flow` 只出
   平移，ECC 才可能带尺度）。GMC 状态（上帧图/关键点）属于 tracker 实例，
   随 D2 的 `reset()` 一并清理。

4. `inference/stream.py`：`run_stream(..., motion_compensation=...)`
   透传；CLI 帮助文案注明"固定机位无需开启"。

**涉及文件**：新增 `autovision.tracking.gmc`；`tracking/byte_track.py`、
`inference/stream.py`。

**测试**：合成平移序列（两帧错切的同一组框）验证估计≈真值；cv2 缺失时的
报错文案；`motion_compensation="none"` 时行为与现状逐位一致（回归锚点）。

**成本**：3–5 人日。**风险**：ECC/ORB 参数在不同分辨率下敏感，先只默认
暴露 `sparse_opt_flow`，其余 method 标注 experimental。

### 4.7 D7 ByteTracker 钩子化与 BoT-SORT（P1）

**动机与上游参考**：上游把 `BYTETracker.update` 拆成
`_split_detections / _pre_first_association / _first_association /
_second_association / _unconfirmed_association / _init_new_tracks` 等可覆写
钩子，`BOTSORT` 只覆写 3 个方法即为完整实现（XYWH 卡尔曼、GMC、IoU 与
ReID 距离取 min + 双阈值门控、EMA 平滑外观特征）。AutoVision 的
`update()` 是单体函数，任何变体（含 D6 的 GMC 接线）都要复制整个函数。

**设计**：

1. **第一步（纯重构，无行为变化）**：把 `update()` 拆为实例方法钩子：
   `_split_detections(boxes, scores, classes) -> (high, low)`、
   `_pre_association(tracks) -> None`（GMC 挂点，D6 从"直接改 update"改为
   落这里）、`_first_association(pool, high)`、`_second_association(...)`、
   `_unconfirmed_association(...)`、`_spawn_new_tracks(dets)`、
   `_retire_tracks()`。`update()` 保留为固定编排（模板方法）。
   **等价性验收**：固定 3 组合成序列（含低分池/未确认/退役路径），重构前后
   `Track` 序列逐位一致（golden fixture 落 `tests/golden/`）。

2. **第二步**：新增 `autovision.tracking.bot_sort`：
   `BOTSORT(BYTETracker)`，覆写：
   - `_pre_association`：调 D6 的 GMC 并对 track 状态做仿射更新；
   - `_first_association`：代价 = `min(IoU 距离, 外观距离)`，受
     `proximity_thresh`（空间门控）与 `appearance_thresh` 双阈值约束；
   - 卡尔曼换 XYWH 状态（新增 `_KalmanFilterXYWH`，与现有 XYAH 并存，
     共享求解写法）。
   **ReID 明确出范围**：AutoVision 没有嵌入基础设施，且上游 ReID 权重许可
   需单独评估。接口预留 `appearance_encoder: AppearanceEncoder | None =
   None`（Protocol，输入 crop batch 输出 L2 归一特征），默认 `None` 时
   BOTSORT 退化为"GMC + XYWH 卡尔曼的 ByteTrack"，不引入第二个模型。

3. 入口：`tracking/__init__` 导出 `BOTSORT`；`inference/stream.py` 与
   tracking 相关 CLI 增加 `tracker="bytetrack"|"botsort"`。

**涉及文件**：`tracking/byte_track.py`（重构）、新增 `autovision.tracking.bot_sort`、
`tracking/__init__.py`、`inference/stream.py`、`tests/golden/`。

**测试**：golden 等价性（第一步）；BOTSORT 在合成平移相机序列上 ID switch
少于 bytetrack 基线（能力断言，非逐位）；`appearance_encoder=None` 退化
路径与 bytetrack 行为差异仅来自 GMC/KF。

**成本**：5–8 人日（两步分开合入）。**风险**：重构引入行为漂移——由
golden fixture 锚定；AGPL 语义边界——只参考论文级算法描述与已登记的
MIT ByteTrack 论文，不读上游实现细节做逐行翻译（净室纪律，见 §2）。

### 4.8 D8 数据集扫描缓存（P2）

**动机与上游参考**：上游首次全量校验后把解析结果存 `*.cache`，缓存哈希掺入
`use_keypoints/nc/kpt_shape` 等**扫描参数**（参数变了自动重扫）+ 文件
路径/size/mtime_ns，扫描骨架只留 5 个钩子，并产出
"X images, Y backgrounds, Z corrupt" 普查。AutoVision 大数据集每次启动
重复扫描/校验；`versioning.dataset_snapshot` 是摘要级（发布溯源用），
不是进程启动加速。

**设计**：

1. 新模块 `autovision.data.scan_cache`：

   ```python
   SCAN_CACHE_SCHEMA = "1"

   def scan_fingerprint(paths: Sequence[Path], task_params: Mapping[str, Any]) -> str
   def load_scan_cache(root: Path, fingerprint: str) -> ScanReport | None
   def store_scan_cache(root: Path, fingerprint: str, report: ScanReport) -> None
   ```

2. 缓存文件 `<dataset-root>/.autovision_scan_cache.json`，原子写
   （tmp + `os.replace`），内容为**校验结论**而非解析标注：
   per-file `{path: ok | background | corrupt}` + 计数器 + 摘要行。
   标注解析仍在 dataset `__init__`/`__getitem__`，避免缓存成为第二份
   标注事实源。

3. 指纹输入：schema 版本 + `(relpath, size, mtime_ns)` 序列的 sha256 +
   影响校验的任务参数（task、`num_channels`、`kpt_shape`、过滤 args——
   各 dataset 类声明 `scan_cache_params()`）。任一变化 → 自动重扫。

4. 开关：dataset 构造 `use_scan_cache: bool = True`；环境变量
   `AUTOVISION_SCAN_CACHE=0` 全局关闭；`data_check --no-cache` 旁路缓存
   强制全扫（质检语义不能被缓存遮蔽）。

**涉及文件**：新增 `autovision.data.scan_cache`；`data/object_detection_dataset.py`
等各 dataset 接线（逐任务面小 PR）；`cli/data_check.py`。

**测试**：mtime/参数变化失效；损坏 JSON 重建；并发写安全（tmp+replace）；
`--no-cache` 旁路。

**成本**：3–4 人日（含 2 个任务面的接线示范）。

### 4.9 D9 大图训练侧切片（P2）

**动机与上游参考**：上游 data/split_dota.py 的训练侧切片：滑窗（crop/gap，末行末列
贴边不越界）、窗口保留率阈值 `im_rate_thr=0.6`（几乎全 padding 的窗口
丢弃，全丢则保 rate 最大的一批）、目标-窗口按 `IoF ≥ 0.7` 分配（一目标可
入多窗）、tile 文件名编码源坐标。AutoVision 只有推理侧 SAHI 合并，
训练侧大图（遥感/航拍/全景）没有切片工具。

**设计**：

1. 新模块 `autovision.data.split_large_images`，几何复用
   `tasks/shared/tiling_utils.py::tile_image`（不重写滑窗）：

   ```python
   def split_large_image_dataset(
       images_dir: Path, labels_dir: Path | None, out_dir: Path, *,
       crop: int = 1024, gap: float = 0.2, iof_thr: float = 0.7,
       min_window_rate: float = 0.6,
       label_format: Literal["hbb", "obb"] = "hbb",
   ) -> Path:  # out_dir
   ```

2. 标签处理：HBB 用纯 numpy IoF（交集/目标面积）；OBB 用 D3 产出的
   `clip_polygon_to_canvas` 做多边形∩窗口再算 IoF；被窗口截断且
   `IoF < iof_thr` 的目标丢弃（与 box_candidates 同一动机）。
   输出 YOLO 归一化标签 + `tiles.jsonl` 清单（tile→源图与窗口坐标，
   供溯源与推理侧 SAHI 参数对齐）。

3. CLI：`autovision tile-dataset`（`cli/` 新命令，参数与函数签名同源）。

**涉及文件**：新增 `autovision.data.split_large_images`、`cli/` 新命令、
`tasks/shared/tiling_utils.py`（如需导出窗口枚举的小重构）。

**测试**：合成大图 + 已知目标布局，断言窗口数/目标分配/保留率回退；
OBB 截断丢弃；清单可回溯。

**成本**：3–4 人日。

### 4.10 D10 图像读取缓存：npy 边车与 RAM 共享缓存（P2）

**动机与上游参考**：上游两级缓存：与图同目录同名 `.npy` 边车（mtime、
通道数/dtype staleness 校验，损坏自动 unlink 回源），以及把全部图打包成
**单个连续共享 uint8 tensor** 的 RAM 缓存（fork COW 共享，`/dev/shm`
余量不足 2× 时降级并告警）。AutoVision 的 `image_dataset.py` 每次样本读取
都走 `open_image`（JPEG 解码）；`mmap_helpers` 面向大图整册，小图高频
解码是训练吞吐瓶颈。

**设计**：

1. 新模块 `autovision.data.image_cache`：

   ```python
   def open_image_cached(path: Path, *, num_channels: int,
                         mode: Literal["none", "disk", "ram"],
                         pool: SharedImagePool | None = None) -> np.ndarray
   ```

   - `disk`：边车 `<stem>.npy`；staleness = 源图 `mtime_ns` 变化或
     npy 的 shape/dtype 不符 → 重建；npy 损坏 → unlink + 回源。
     写入用 tmp + `os.replace`，容忍多 worker 并发重建（后者胜出）。
   - `ram`：`SharedImagePool` 在**构建 dataloader 的进程**里分配单个
     连续 uint8 tensor（fork 后 COW 共享；pin_memory worker 走共享内存
     语义）；容量预估 = 抽样 30 张 × N × 1.2，`/dev/shm`（或页缓存）
     不足则降级 `disk` 并 `logger.warning`。池一旦建好只读，天然
     worker 安全。

2. 接线：dataset 构造参数 `image_cache: Literal["none","disk","ram"] =
   "none"`；`open_image` 语义不变（`image_cache` 是外层装饰，不进
   `file_helpers` 的公开契约）。与 `mmap_helpers` 的关系写入
   `docs/framework_design.md`：小图 npy 边车 / 大图整册 mmap / 全集
   RAM 池，三者按图尺寸与数据集规模选择。

**涉及文件**：新增 `autovision.data.image_cache`；`data/image_dataset.py` 及
分任务 dataset 的构造透传。

**测试**：staleness 三态（mtime 变/shape 变/损坏）；并发写仅留完整文件；
RAM 池在 num_workers>0 下取数与直读一致；`/dev/shm` 不足降级。

**成本**：4–5 人日。

### 4.11 D11 数据集下载断点续传与重试（P2）

**动机与上游参考**：上游 `utils/downloads.py::safe_download` 的传输纪律：
失败重试保留 `.part` 并发 `Range: bytes=<n>-`；416（无可续）删除重下；
服务器忽略 Range 返 200 时重置；5xx 指数退避（5/10/20s），非
408/416/429 的 4xx fail-fast；下载前磁盘空间预检（1.5× 安全系数）。
AutoVision 的 `data/downloads.py` 已有 sha256 + 原子发布 + 成员防护，
但 docstring 明示"does not resume partial HTTP transfers"——大权重/数据集
在弱网下反复从头下。

**设计**（全部为 `data/downloads.py` 内增量，不换 HTTP 栈）：

1. `_download_file` 增加重试循环：可重试集合 `{408, 429} ∪ 5xx`；
   退避 `min(5 * 2**i, 60)s`；其他 4xx 立即抛。
2. 续传：重试时若 `<dest>.part` 存在且服务器支持（首次请求带
   `Range`，响应 206 才续），从已收字节数续写；206 校验
   `Content-Range` 起点一致，不一致删除重下；200（忽略 Range）重置。
3. 磁盘预检：`shutil.disk_usage(dest).free >= 1.5 × size_bytes`（size
   未知时跳过），不足直接抛可读错误。
4. sha256 校验、原子 `os.replace` 发布、tar/zip 成员防护**全部保留不动**
   （这部分 AutoVision 比上游强，不回退）。

**涉及文件**：`data/downloads.py`；`registry/gtr_download.py` 若复用同一
底层则一并受益（核对后决定是否抽公共函数）。

**测试**：本地 HTTP 测试服务器模拟 206/200/416/503 序列；断点续传后
sha256 仍通过；fail-fast 分支。

**成本**：2–3 人日。

### 4.12 D12 rect 批次与连续分片 DDP sampler（P3，证据驱动）

**动机与上游参考**：上游按长宽比排序分组、stride 取整的 batch 形状
（`data/base.py::set_rectangle`），DDP 下每 rank 分**连续**段保住排序局部性
（`build.py::ContiguousDistributedSampler`），rect 与 mosaic 互斥。
收益是 padding 接近零；代价是 sampler/letterbox/dataset 三处联动，
且与 shuffle/ mosaic/ image_weights 交互复杂。

**设计（门槛前置）**：先在 dataloader 调试指标里加
`padding_ratio`（batch 内真实面积 / 画布面积），跑一次 COCO 基准确认
浪费超过 15% 再立项。届时实现 `data/samplers.py` 新增
`AspectRatioGroupBatchSampler`（epoch 内分块排序 + 种子化块内 shuffle）
与 DDP 连续分片变体；mosaic 激活时自动禁用并告警（沿用上游互斥语义）。
本项**不做无证据预实现**。

**成本**（若立项）：5–8 人日。

### 4.13 D13 solutions 产品层与 LineZone 快速穿越补偿（P3，按产品需求触发）

**动机与上游参考**：上游 `solutions/` 以一个 `BaseSolution`（模板方法 +
track 复用 + 结构化 `SolutionResults` + track/solution 分相计时）承载
约 20 个 ~100 行的方案。其中对 AutoVision 最有价值的三个能力点：
**快速穿越补偿**（目标一帧跨过线时用前后位置连线与线段求交补判，
`solutions/object_counter.py`）、**多命名区域计数**（`add_region` API）、
**速度估计**（双线过线计时 × `meter_per_pixel` 标定，速度锁定后不再更新）。

**现状核对**：AutoVision `tracking/analytics.py` 的 `LineZone` 用
`_side_of(track)` 判当前帧符号位——高速目标两帧符号位相同即漏计，
补偿缺失属实；多区域/测速/热力图不存在。

**设计**：

1. **先修 `LineZone`（独立小 PR，不建 solutions 层）**：`update()` 为每个
   track 记录上一帧锚点；`side` 未翻转时追加"线段相交"判定
   （`geometry/` 新增 `segments_intersect(p1, p2, a, b) -> bool`，纯
   numpy/标量即可）；相交即按方向补一次 in/out。这是对既有功能的行为
   增强，需要回归用例覆盖"两帧跨线"场景。

2. **solutions 层（触发后立项）**：新顶层包 `src/autovision/solutions/`
   （注意 `scripts/check_layering.py` 分层：solutions 依赖
   tracking/visualize/geometry，不得反向）：

   ```python
   class SolutionResult(TypedDict):
       frame: np.ndarray          # 叠加后画面（visualize 管线）
       payload: dict[str, Any]    # 结构化计数/测速结果

   class BaseSolution:
       def process(self, frame, tracks: Sequence[Track],
                   frame_id: int) -> SolutionResult: ...
   ```

   首批方案：`RegionCounter`（多命名 `PolygonZone`）、
   `SpeedEstimator`、`Heatmap`（numpy 累积 + `visualize` 调色，不引入
   cv2/matplotlib）。`run_stream(..., solution=...)` 挂接点 + 每个 solution
   独立 `ops` 式计时字段进 JSONL（复用 stream 现有 `time_s` 结构）。
   D2 的 `drain_removed_tracks` 用于各方案清理 per-id 状态。

**成本**：LineZone 补偿 1–2 人日；solutions 层首批三方案 5–8 人日。

### 4.14 D14 工程小件打包（P3）

四个 ≤1 人日的独立小项，可随相邻 PR 捎带：

| 编号 | 项 | 设计 | 参考 |
|---|---|---|---|
| a | serving/引擎 warmup | FastAPI lifespan 与 `InferenceEngine` 构造后各跑一次**全管线** `run_batch`（含 numpy 后处理解码）——首次请求延迟尖峰常来自后处理而非 forward | 上游 warmup 连 NMS 一起预热 |
| b | 导出产物最小尺寸护栏 | `export_onnx` 在 parity 之前断言 `artifact.stat().st_size > 4096`，小于即判损坏失败（空图/空权重类事故的廉价护栏） | 上游 `@try_export` |
| c | 导出完成回显 | 导出成功日志附一段可直接粘贴的 `InferenceEngine(model=..., artifact=..., backend=...)` 用法与元数据摘要（配合 D1） | 上游"打印可复制的 predict 命令" |
| d | 预测参数与引擎生命周期文档 | `DetectionModel`（`cli/model_handle.py`）已缓存 network；在 `docs/framework_design.md` 明确"结构性参数（backend/artifact/device）重建、阈值类参数每调用传入"的约定，避免未来出现每请求重载 | 上游 `PREDICTOR_SETUP_KEYS` |

### 4.15 D15 推理引擎构建期/运行期参数二分与 handle 级模型复用（P1）

**动机与上游参考**：上游 `engine/model.py` 定义 <!-- docpath-ok -->
`PREDICTOR_SETUP_KEYS`（device/dnn/data/nms/compile/channels_last/quantize/tracker，
上游 engine/model.py:32），把预测参数二分：任一构建期键变化才重建 predictor
并重新 `setup_model()`（model.py:546-568），conf/iou/save 等运行期键热更新；
`fuse()/load()/train()` 等改变模型本体的人口处显式置空 predictor。

**现状核对**：`cli/model_handle.py::predict_to_files`（model_handle.py:372
附近）每次调用都 `load_task_model(model, device=...)` 重建网络并新建推理
路径，全库对 `InferenceEngine` 无跨调用复用；`DetectionModel` handle
（`detection(...).train().val().predict().export()`）是无状态句柄。
D14d 只落了文档约定（`docs/framework_design.md` §3.1），未落机制。

**设计**：

1. `cli/model_handle.py` 新增模块级
   `ENGINE_SETUP_KEYS: tuple[str, ...] = ("backend", "artifact", "device",
   "precision")`；`DetectionModel` 实例持有 `_engine_cache` 与
   `_engine_signature`（构建期参数签名 dict）。
2. 新私有方法 `_resolve_engine(**overrides)`：签名一致 → 复用缓存引擎；
   不一致 → 重建 `InferenceEngine` 并替换缓存。运行期参数（conf/iou/
   `max_image_pixels` 等每次调用传入）不进签名。
3. `export()` / `load()` / `train()` 等改变权重或工件的入口返回前置空
   `_engine_cache`（对应上游 fuse/load/train 后置空语义）。
4. 批量预测入口统一走 `_resolve_engine`；device 惰性解析（首次预测时），
   handle 构造不触 GPU。
5. `_resolve_engine` 加 `threading.Lock`，签名对比与替换在锁内；对象
   pickle/跨进程传递时缓存字段被剔除（`__getstate__`）。

**涉及文件**：`cli/model_handle.py`；`inference/engine.py`（如需暴露轻量
签名辅助）。

**测试**：同签名两次预测只构建一次引擎（构建计数 mock）；backend/device
变化触发重建；`export()` 后缓存失效；锁内并发安全。

**成本**：2–3 人日。**风险**：低；注意 handle 序列化边界。

### 4.16 D16 AMP 事前守卫（P1）

**动机与上游参考**：上游 `utils/checks.py::check_amp`（checks.py:1004-1080）
三级防线：CPU/MPS 直接 False → 已知 AMP 缺陷 GPU 黑名单正则短路 → 用小
模型对同一 batch 跑 FP32 与 autocast 推理、`torch.allclose(atol=0.5)` 实测
判定；下载失败跳过、断言失败禁 AMP。DDP 集成两细节（上游 trainer.py:412-421）：
探针会污染 default_callbacks，先备份后恢复；AMP 决策以 int tensor
`dist.broadcast`（gloo 不支持 bool）。

**现状核对**：AutoVision 无任何 AMP 检查（grep 确认）。AMP 由 `precision`
参数驱动：`engine/engine.py::_build_grad_scaler`（engine.py:214-231）与
autocast 推导（engine.py:457-460, 520-525, 1240-1262）；NaN/Inf 梯度守卫
是事后防线（跳 step）。事前探针缺失——AMP 数值异常环境会静默劣化。

**设计**：

1. 新模块 `autovision.engine.amp_check`：

   ```python
   def check_amp_support(
       model: torch.nn.Module, *, device: torch.device,
       batch: torch.Tensor | None = None, atol: float = 0.5,
   ) -> bool: ...
   ```

   - CPU/MPS → False（`logger.info` 原因）；
   - 实测：FP32 forward vs `autocast` forward 输出逐 tensor
     `allclose(atol)`（dict 输出逐键检查，None 值键跳过，取最小）；batch
     缺省按模型输入规格构造随机张量；
   - 任何异常 → `logger.warning` + False（fail-closed 到 FP32）。
   - **净室纪律**：探针用调用方传入的模型本体（已构建的 TaskNetwork），
     不像上游那样下载外部权重。
2. TaskEngine 接线：`precision` 开 AMP 时在构建 scaler 前跑一次探针；
   False → 自动降级 FP32 并 `logger.warning` 说明。参数 `amp_check: bool
   = True` 可关。
3. DDP：rank0 探针 → int tensor `dist.broadcast`，各 rank 同一决策。

**涉及文件**：新增 `engine/amp_check.py`；`engine/engine.py`（scaler 构建
前接线）；trainer_args / `cli/train.py`（`amp_check` 开关）。

**测试**：CPU 强制 False；mock allclose 失败 → 降级路径；异常注入 →
False；DDP broadcast（gloo 冒烟，CI 标记）；GPU 实测 smoke 标记 skipif。

**成本**：2–3 人日。**风险**：低；dict 多键输出的逐键 allclose 需处理
嵌套与非 tensor 键。

### 4.17 D17 resume 白名单加固、早停状态持久化与 possible_stop（P1）

**动机与上游参考**：上游 `trainer.py::check_resume`（trainer.py:1022-1067）
用 ckpt args 整体替换当前 args，但仅白名单键可被 ckpt 覆盖，白名单外差异
**逐键列出两侧值并告警**；早停 stopper 状态 `{best_fitness, best_epoch}`
序列化进 ckpt（trainer.py:802）并在 resume 恢复（trainer.py:1084）；
`EarlyStopping.possible_stop`（torch_utils.py:1147）在距 patience 还差
1 步时置位，trainer 借此**提前开启验证**（trainer.py:651）——否则
val=False 时早停没有 fitness 可判。

**现状核对**：`docs/known_limits.md:24-28` 明确承认"checkpoint 不恢复
mid-epoch sampler 位置、不校验配置/版本差异"。`engine/engine.py` 已有早停
（`early_stopping_patience/min_delta`，engine.py:794-795、1485-1549）与
resume，但早停状态是否入 checkpoint 需实现时核对；无 possible_stop。

**设计**：

1. **resume 参数白名单**：resume 路径新增 `RESUME_OVERRIDABLE_KEYS`
   （建议集：`epochs`、`precision`、`early_stopping_patience`、
   `early_stopping_min_delta`、`save_every_num_steps`、
   `val_every_num_steps`、`log_every_num_steps`——以 checkpoint 实际保存
   的训练参数收窄）；ckpt 保存的训练参数仅白名单键可覆盖本次调用参数，
   白名单外差异逐键 `logger.warning`（列两侧值）。环境/版本差异校验仍按
   known_limits 现状不做（超范围）。
2. **早停状态持久化**：checkpoint 新增
   `early_stopping: {best_metric, best_step, steps_since_improve}`；
   resume 恢复，避免"resume 后早停立即误触发"或耐心重置。旧 ckpt 无该键
   → 新鲜状态（兼容）。
3. **possible_stop**：早停逻辑增加 `possible_stop` 属性（距 patience 差
   1 步置位）；验证调度据此在既定 `val_every_num_steps` 之外追加一次
   验证（仅早停启用时生效）。

**涉及文件**：`engine/engine.py`（resume 恢复路径、早停状态、验证调度）。

**测试**：白名单覆盖生效 / 白名单外差异告警列表逐键断言；早停状态
save→resume→连续计数 roundtrip；possible_stop 触发补验证；旧 ckpt 兼容。

**成本**：3–4 人日。**风险**：中——动训练循环核心；**必须先落 resume
行为回归锚点（golden 式用例）再改**；白名单语义与既有 `trainer_args`
dict 白名单风格一致。

### 4.18 D18 空闲 GPU 自动选卡（P1）

**动机与上游参考**：8.4 新增 `utils/autodevice.py`：把选卡编码进 device
语法而非新参数——`device="0,-1"` 中 `-1` 是"自动选一张空闲卡"占位符。
`select_idle_gpu`（autodevice.py:135-195）：显存占比+利用率阈值过滤 →
按 `(utilization, -free_mem, random.random())` 排序（**随机 tiebreaker**，
注释写明动机：多进程同时启动时各卡看起来同样空闲的选卡竞态）→ 取前 N，
不足时告警并**移除占位符而非抛错**（缩卡优于崩溃）。`parse_device`
（torch_utils.py:226-243）负责替换；CUDA_VISIBLE_DEVICES 物理 id↔torch id
双向翻译（torch_utils.py:214-224）；NVML 缺失全类软失败（warning + 早退）。

**现状核对**：AutoVision `devices` 支持 "auto"（`cli/train.py`），语义是
数量自动而非空闲选卡；共享服务器上固定取 0/1 号卡会撞车。无 NVML 集成。

**设计**：

1. 新模块 `autovision.runtime.autodevice`：

   ```python
   def list_gpu_stats() -> list[GPUStat]: ...
   def select_idle_gpus(count: int, *, candidates=None,
                        min_memory_fraction: float = 0.2,
                        min_free_fraction: float = 0.2) -> list[int]: ...
   def resolve_devices(spec: str | int | Sequence[int | str]) -> list[int]: ...
   ```

2. 语法：device 字符串支持 `-1` 占位符（`"0,-1"`、`"-1,-1"`），逐个替换
   为空闲卡；排序键 `(utilization, -memory_free, rng.random())`，rng 可
   注入种子（测试需要）；选不满时移除占位符并告警。整数与 `"auto"` 语义
   不变（向后兼容）。
3. NVML 经 `nvidia-ml-py` **函数内惰性导入**；缺失 → `-1` 降级为顺序取卡
   + `logger.warning`（报错文案含 `pip install nvidia-ml-py`）。
4. CVD 桥接：`CUDA_VISIBLE_DEVICES` 限制下请求 id 超出 torch 可见数视为
   物理 id 翻译（重写上游判定规则）。
5. 接线：`runtime/env.py` 的 device 解析与 `cli/train.py` devices 统一走
   `resolve_devices`。

**涉及文件**：新增 `runtime/autodevice.py`；`runtime/env.py`、
`cli/train.py`。

**测试**：mock pynvml 的过滤/排序/tiebreaker/不足移除；CVD 翻译表；无
NVML 降级；`-1` 语法解析（全部 CPU 可测）。

**成本**：2–3 人日。**风险**：低；随机 tiebreaker 需种子注入点。

### 4.19 D19 COCO 交换格式健壮性（P1）

**动机与上游参考**：上游两件已解问题：① COCO 多段掩码合并为单多边形
（`converter.py::merge_multi_segment` + `min_index`，converter.py:548-611
——按与当前链尾最近点对两轮连接）；② 多边形白名单校验（list、len≥6、
偶数、全数值），不合法时用 bbox 的 8 点多边形兜底 + **全文件级一次性
告警**（converter.py:327-343, 360-364）。

**现状核对**：AutoVision COCO 导入在 `data/` 侧（`tests/test_interchange.py`
存在）；多段 RLE/多边形拼接行为需实现时定位现行为——若已有等价逻辑，
本项收敛为校验兜底。

**设计**：

1. `geometry/polygons.py` 或 interchange 解析模块新增：
   - `merge_multi_segment(segments: list[np.ndarray]) -> np.ndarray`——
     多段按最近点对首尾相接成单链（净室重写算法语义，不引入 shapely）；
   - `sanitize_polygon(poly, *, fallback_box, warn_once) -> np.ndarray |
     None`——白名单校验（类型/长度/偶数/有限数值），不合格退 bbox 矩形
     8 点；文件级告警聚合（每文件最多一条，附坏行计数）。
2. 接线：COCO→AutoVision 转换路径逐 annotation 调用；segment 缺失或不
   合法但 bbox 存在 → 兜底而不丢样本。
3. 纯 numpy；与既有 RLE→掩码路径分工明确（本项只管多边形列表）。

**涉及文件**：`data/interchange.py`（或实际 COCO 解析模块）、 <!-- docpath-ok -->
`geometry/polygons.py`。

**测试**：两段/三段拼接正确性（已知几何）；非法多边形兜底 + 聚合告警
单次；合法路径回归不变。

**成本**：2–3 人日。**风险**：低。

### 4.20 D20 save_period 周期存档与 checkpoint 序列化卫生（P2）

**动机与上游参考**：上游 `trainer.py::save_model`（trainer.py:766-835）：
`(epoch % save_period == 0)` 落 `epoch{N}.pt`；单次序列化到 `io.BytesIO`
再写多份（last/best/epochN 共享一份字节）；保存前 EMA 非有限张量从 live
model resync（EMA 一旦被 NaN 污染会永远跳过保存）；fp16 溢出
`torch.nan_to_num_` 钳制；NHWC 权重归一回 NCHW contiguous（否则 fuse 的
`.view()` 崩）。

**现状核对**：AutoVision 只有 `save_last/save_best/save_every_num_steps`
（engine.py:779-781、1092-1113）；无按 epoch 周期滚动归档；checkpoint 写入
已有原子发布。

**设计**：

1. `save_period_num_epochs: int = 0`（0=关闭，默认行为不变）；>0 时
   `epoch % save_period == 0` 追加 `<name>_epoch{N}.pt`，滚动保留最近
   K=3 份（可配）。
2. 序列化卫生三件（可独立小 PR 捎带）：fp16 权重 `nan_to_num_` 钳制；
   NHWC→NCHW contiguous 归一；单次 `torch.save` 到 BytesIO 后多路原子写。
3. EMA 恢复路径核对：保存前发现 EMA 非有限张量 → 从 live model resync
   或拒绝保存并告警（与既有 NaN 守卫语义对齐后择一，实现时定）。

**涉及文件**：`engine/engine.py`（保存循环）。

**测试**：周期存档生成与滚动清理；BytesIO 多路写内容一致；NHWC 归一；
NaN EMA 分支；默认参数行为逐位不变（回归锚点）。

**成本**：2–3 人日。

### 4.21 D21 首轮 OOM 自动降批（P2）

**动机与上游参考**：上游 `trainer.py:565-599`：捕获 4 类显存错误串
（out of memory / CUBLAS_STATUS_ALLOC_FAILED / CUDNN_STATUS_INTERNAL_ERROR
/ unable to find an engine）→ batch 减半并重建 pipeline，**仅首个 epoch、
单卡、最多 3 次**；降批时 `close()` 旧 dataloader（释放 worker 与预取
批次）保留 optimizer 状态；for/else 保证只有完整 epoch 才复位计数器。

**现状核对**：AutoVision 无 OOM 自动处理（grep 无命中），OOM 直接抛错。
D4 autobatch 是事前探测，本项是训中兜底，互补；两者语义需核对不冲突
（探测建议值 + 训中降批取更小）。

**设计**：

1. TaskEngine 训练循环捕获 `torch.OutOfMemoryError` 与匹配上述错误串的
   RuntimeError；仅当前 epoch 为该次 fit 的首个 epoch 且尝试 <3 时：
   每设备 batch 减半、重建 dataloader（保留 optimizer/scheduler/EMA
   状态）、`logger.warning`，继续当前 epoch。
2. 第 3 次后放弃并抛原始错误（文案附人工建议：手动调小 batch 或用
   `batch_size="auto"`）。
3. 单机多卡：只改每设备 batch，`get_global_batch_size` 语义不变。

**涉及文件**：`engine/engine.py`。

**测试**：mock 显存错误 → 减半/重建调用序列；3 次后放弃；非首 epoch 不
触发；完整 epoch 后计数复位。

**成本**：2–3 人日。**风险**：中——dataloader 重建与 sampler/缓存状态
交互；与 D4 的互斥语义需写清。

### 4.22 D22 引擎级回调注册表（P2）

**动机与上游参考**：上游 23 命名事件（`callbacks/base.py:128-156`）+
`default_callbacks` + add/set/clear 四件套；构造时**两层拷贝**防实例间
共享列表（trainer.py:182-184）。注意上游 `run_callbacks` **无异常隔离**
（四处同构 `for cb in ...: cb(self)`）——AutoVision 现有"异常仅记录不
中断"更稳，必须保留，不得"借鉴"该缺陷。

**现状核对**：`engine/engine.py` 现有 5 固定事件
（`on_train_start/on_step_end/on_validation_end/on_early_stop/on_train_end`，
engine.py:806-810）鸭子回调 + `step_hooks`；预训练 Engine 与 TaskEngine
两套循环各一份事件面。

**设计**：

1. 新模块 `autovision.engine.callbacks`：`CALLBACK_EVENTS` 元组（扩展现有
   5 事件：`on_epoch_start/on_epoch_end/on_checkpoint_save/on_fit_start`
   ……以实现时真实需要收窄，**不一次到位 23 个**）+ `CallbackRegistry`
   （`add/set/clear` + `run(event, payload)`，逐回调 try/except +
   `logger.warning`）。
2. 两套循环统一持有一个 registry；`fit(callbacks=...)` 旧鸭子列表继续
   接受（适配层转换），非破坏。
3. `step_hooks`（含 dataloader 重初始化标记）语义不变；registry 是事件
   面、hooks 是计算面，分工写入 `docs/framework_design.md`。

**涉及文件**：新增 `engine/callbacks.py`；`engine/engine.py`。

**测试**：注册/覆盖/清除；异常隔离（单回调抛错不中断、告警一次）；两
循环事件序列快照；旧 `callbacks=` 列表兼容。

**成本**：3–4 人日。**风险**：低-中；事件面扩张克制。

### 4.23 D23 训练基建杂项打包（P2）

四个 ≤2 人日的独立小项（同 D14 体例，可随相邻 PR 捎带）：

| 编号 | 项 | 设计 | 参考 |
|---|---|---|---|
| a | worker 播种 | dataloader 级 `generator=torch.Generator().manual_seed((seed+rank)%2**64)` + `worker_init_fn` 以 `torch.initial_seed()%2**32` 派生 numpy/random 种子；`seed_everything` 文档补三层可复现说明 | 上游 build.py:232-236, 398-414 |
| b | worker 上限 | `num_workers = min(num_workers, ceil(samples/batch))`；单批 loader 强制 0 worker；prefetch 训练 4 / 验证 2 | 上游 build.py:394-417 |
| c | model_info/FLOPs | `runtime/model_info.py`：参数量分层（total/trainable/frozen 已有）+ thop 缺失双回退的 GFLOPs 估算；`fit` 横幅与模型卡附加 | 上游 torch_utils.py:546-671 |
| d | fuse_conv_bn | `runtime/fuse.py`：通用 Conv+Norm 融合（QAT amax 按 BN 折叠比例同步缩放语义入注释）；任务局部重参不动；fuse 后 val 对拍开关（与 parity 门禁同精神） | 上游 torch_utils.py:383-404 |

**成本**：各 1–2 人日。

### 4.24 D24 增广统一四目标钩子协议（P2，架构前置）

**动机与上游参考**：上游 `BaseTransform`（augment.py:28-121）：
`get_params` 一次产出共享随机参数 → `apply_image / apply_instances /
apply_semantic / apply_depth` 四钩子，图、框、语义掩码、深度图共享同一
几何变换。`known_limits.md:170-171` 承认 mosaic/mixup/zoom-out/color-jitter
未移植；本项是移植这些复合增广的**架构前提**，避免各任务各写一份几何
变换。

**现状核对**：AutoVision 增广为方法制（`augment/transform_args.py::
MethodTransformArgs` + 各方法类，torchvision v2 基础）；检测增广在
`augment/detection/`（D3 已接线）。任务间几何同步由各自 transform 体系
保证。

**设计**（协议先行，落地随 mosaic/mixup 移植 PR）：

1. 新基类 `autovision.augment.core.GeoTransform`：
   `sample_params(rng, hw) -> Params`（冻结 dataclass）+
   `apply_image/apply_instances/apply_semantic_mask/apply_depth` 四抽象
   方法 + `__call__` 固定编排（模板方法）。
2. 值约定：语义掩码 INTER_NEAREST + 边界值 255（ignore）；深度
   INTER_NEAREST + 边界值 0（禁止在无效区边缘制造伪有效像素）；关键点
   出界 visibility 置 0；无 pose 对称表时翻转自动禁用并告警。
3. `close_augmentation` 调度：训练末期一次性关闭随机几何/混合类增广并
   重建 pipeline（与 D12 rect 互斥语义对齐）。
4. 既有 MethodTransformArgs 体系保留；新协议只作为复合增广的实现层，
   不推翻方法制。

**涉及文件**：新增 `augment/core.py`；落地随复合增广移植。

**测试**：同一 Params 下四目标几何逐位对应；close 调度单次触发。

**成本**：协议 1–2 人日；随复合增广移植另计。

### 4.25 D25 扫描缓存模板方法化（P2）

**动机与上游参考**：上游 `cache_labels` 是 5 钩子模板方法
（dataset.py:120-165：get_label_files / get_cache_hash / verify_args /
result_to_label / scan_summary），Depth/Semantic 各改 2–3 钩子即接入，
普查行 "X images, Y backgrounds, Z corrupt"；depth 的 cache hash 掺入
`depth_scale`（参数变 → 自动重扫）。

**现状核对**：D8 的 `data/scan_cache.py` 已落地，接入点是"文件清单校验
层"（COCO/YOLO 清单无独立扫描阶段）；深度/语义任务面未接入。

**设计**：

1. `scan_cache.py` 增加钩子协议（鸭子即可）：`scan_files()`、
   `fingerprint_params()`、`verdict(item)`、`summarize(counts)`；
2. 深度/语义 dataset 以 2–3 个钩子接入（深度 verdict = 纵横比 2% +
   dtype 校验，见 D26；语义 = LUT 可映射性）；
3. 缓存结论仍只存 verdict 不存标注（D8 纪律不变）；`--no-cache` 与
   `AUTOVISION_SCAN_CACHE=0` 旁路不变。

**涉及文件**：`data/scan_cache.py`、深度/语义 dataset。

**测试**：两任务面接入后指纹敏感性、verdict 正确、旁路不变。

**成本**：2–3 人日。

### 4.26 D26 深度/语义数据协议对照（P2）

**动机与上游参考**：深度 = uint16 PNG + depth_scale 元数据（默认 1000
毫米）+ 0=无效像素 + 保存侧非有限/非正值归零 + uint16 量程守卫
（上游 utils.py:65-129）；读取侧纵横比 log 误差 ≤2% 校验（utils.py:324-366）；
深度 val 用拉伸而非 padding（padding 污染边缘像素，dataset.py:558-561）。
语义 = label_mapping 编译 256 项正向/逆向 LUT 一次查表 + `ignore_label`
协议（dataset.py:950-978）；`single_cls` 对语义直接 raise。

**现状核对**：AutoVision 深度/语义数据面现有约定需逐项盘点（实现第一
步）；等价项不重复建设，只补缺失校验。

**设计**：

1. 深度：读取校验增加 `verify_depth_pair`（纵横比 log 误差 ≤2%，npy 走
   mmap 只验形状）；letterbox 对齐确认 INTER_NEAREST；深度 val 不 padding。
2. 语义：label_mapping 编译 256 项 LUT（`np.take` 一次映射）+ `ignore_label`
   键（缺省 255）；`single_cls` 对语义 raise。
3. 不强制迁移既有数据格式；推荐协议写入 `docs/framework_design.md`。

**涉及文件**：深度/语义 dataset 与 `data/` 校验层；文档。

**成本**：2–3 人日（含盘点）。

### 4.27 D27 指标聚合容器与实时混淆矩阵（P2）

**动机与上游参考**：上游 `DetMetrics`：stats 惰性（逐 batch 只 append
原始 tp/conf/cls，结尾 concat 重算——DDP 合并友好，metrics.py:1146-1186）；
`Metric` 全 property 从 `all_ap` 派生；SegmentMetrics/PoseMetrics 用
property `fget` 组合扩展（metrics.py:1363, 1516）；验证循环内实时更新
混淆矩阵；per-image 指标"空 GT+空预测"记 1.0 而非 0/0（metrics.py:1086-1089）；
DDP 三种聚合分工（stats `gather_object` / 混淆矩阵 `reduce SUM` /
per-image 单独 gather，detect/val.py:312-343）。

**现状核对**：AutoVision 无容器类；命名不统一（Collector/Evaluator 后缀
混用），共享 `update/reset/compute + export_state/merge_state` 鸭子协议；
混淆矩阵**事后**从 error_analysis 的 Match 构建（`metrics/confusion.py::
from_matches`）。

**设计**：

1. 新增 `metrics/containers.py::TaskMetrics` 基类：`stats: dict[str, list]`
   惰性缓冲 + `process() -> dict[str, float]` + `export_state/merge_state`
   保持既有鸭子协议；**不改现有 Collector/Evaluator 公开行为**，容器供
   新任务与渐进重构采用。
2. 混淆矩阵实时路径：验证循环内匹配结果直接
   `ConfusionMatrix.update(matches)`，与事后 `from_matches` 并存；同一
   Match 结构保证两路径最终一致。
3. per-image 指标"空 GT+空预测"记 1.0。
4. 命名统一走弃用政策（AGENTS.md §3.4）：Collector 别名标注
   `deprecated in X.Y.Z / remove in A.B.C` 跨一个 MINOR，**不做静默别名**。

**涉及文件**：新增 `metrics/containers.py`；`metrics/confusion.py`、
`metrics/error_analysis.py`。

**测试**：惰性 stats 两 rank merge；实时与事后混淆矩阵逐位一致（同一
数据双路径）；空/空 记 1.0。

**成本**：3–5 人日。**风险**：中；命名迁移涉及公共 API 弃用流程。

### 4.28 D28 视频源协议增量（P3）

逐项小改（实现时逐条评估接线点，`inference/video.py` / `stream.py`）：

1. **grab/retrieve 抽帧**：抽帧路径每帧 `grab()` 保持解码器同步、仅命中
   stride 时 `retrieve()` 解码（上游 loaders.py:174-176 语义）；
2. **批不跨视频**：混合图像/视频清单批量预测时，视频播完立即刷出当前批
   （loaders.py:463-473）；
3. 清单文件输入：`.txt`/`.csv` 逐行路径清单（相对清单父目录解析）；
4. PIL/ndarray 通道归一化矩阵（`I;16` 用 `>>8` 防截断、BGRA→BGR、灰度+
   alpha，loaders.py:565-603 语义）。

**成本**：合计 2–3 人日。**不做**：LoadScreenshots、YouTube 源、mss
（场景过窄，见 §5）。

### 4.29 D29 数据与评估小件打包（P3）

| 编号 | 项 | 设计 | 参考 |
|---|---|---|---|
| a | IO 体检 | `data_check` 增加抽样（≤5 文件）stat ping 与读吞吐测量，超阈值（10ms/50MB/s）告警建议本地存储；小文件（<16KB）豁免吞吐判断 | 上游 utils.py:147-223 |
| b | max_det 抬升 | 检测验证入口扫标签最大目标数，超 max_det 且用户未显式指定时自动抬升 + 告警（防静默召回损失） | 上游 detect/val.py:63-95 |
| c | 数据集 YAML 协议对照 | 声明式 path/split 分离、names/nc 互洽校验；download 钩子仅允许 URL/脚本路径（**禁止 exec**，上游 utils.py:824 为反例） | 上游 utils.py:713-832 |

**成本**：各 1–2 人日。

### 4.30 D30 INT8 校准清单（P3，触发式）

立项 INT8 校准时按单执行（上游实测坑，exporter.py:1047-1097、1438-1460、
1847-1859）：① 校准集用 val split；② `drop_last=True`（静态 batch 必需）；
③ 校准 loader `workers=0`；④ 尾批不足静态 batch 时 **tile 补齐**（静态图
拒绝变批）；⑤ **先导 ONNX 再 import tensorrt**（import 顺序 bug）；⑥
QAT 模型跳过校准。**成本**：随 INT8 立项计。

### 4.31 D31 声明式导出参数矩阵（P3）

触发条件：导出 CLI 参数组合继续膨胀。设计：格式→参数白名单→量化支持集
三张声明表 + `validate_args`（用户传非默认值而白名单不含 → 报错列出该
格式合法 quantize 选项；上游 exporter.py:155-527 语义）。AutoVision 4 后端
规模下当前收益有限；`try_export` 的体积护栏 D14b 已覆盖。

## 5. 明确不借鉴的清单

| 上游设计 | 不借鉴理由 |
|---|---|
| `parse_model` YAML 模型动物园（YAML → 字符串模块名反射建网） | 与 AutoVision 的 encoder/registry 路线冲突；AGPL 边界要求上游仅存在于隔离 encoder 模块 |
| `Results`/`BaseTensor` 结果容器与 `_keys` 协议 | `InferenceResult` + `keys.py` 已覆盖且更严格（RLE、`max_elements`） |
| cfg `default.yaml` + frozenset 类型注册表 | Pydantic 方案是严格超集，已有模糊建议 |
| Ray Tune 集成 | `hpo/` ASHA 已覆盖且更轻 |
| 纯 torch NMS（`TorchNMS`） | torchvision 是核心依赖；仅记录 class-offset 技巧备查 |
| GA4 匿名遥测（`events.py`） | 观测体系形态不同，隐私定位不同 |
| 24 后端 AutoBackend 全家桶 | 超出定位；`InferenceEngine` 的 torch/onnx/trt/openvino 四后端已覆盖部署矩阵 |
| 遗传超参进化（`engine/tuner.py` 独立 `Tuner`） | 与 ASHA 定位重叠；MongoDB 分布式协调引入重依赖（首轮勘误：该 Tuner 在 8.4 仍在，Ray 是新增路径而非替代） | <!-- docpath-ok -->
| SAM 自动标注链（上游 data/annotator.py） | 依赖 AGPL 上游权重/包；`pseudo_label` 已提供 teacher 标注路径 |

第二轮新增确认：

| 上游设计 | 不借鉴理由 |
|---|---|
| `run_callbacks` 无异常隔离（trainer/predictor/validator/exporter 四处同构） | 上游缺陷，回调异常直接中断主流程；AutoVision 保留"逐回调 try/except + 记录不中断"，不得反向借鉴 |
| `utils/triton.py` Triton 远程客户端 | 无远程 serving 后端需求；唯一可记的"输出 dtype 跟随首个输入 dtype"契约 |
| `optim/muon.py` MuSGD 混合优化器 | 仅大规模长 schedule 训练有价值，依赖较新 torch 的 foreach/bfloat16 | <!-- docpath-ok -->
| `LoadScreenshots` / YouTube 源（mss、pytube 系） | 桌面截屏与第三方反爬接口场景过窄且脆弱 |
| `check_det_dataset` 的 `exec()` 下载钩子 | 任意代码执行面；只借鉴声明式 URL/脚本路径形态 |
| `YOLO.__init__` 的 `__class__/__dict__` 实例替换 hack | 可读性差；AutoVision 用工厂函数/类选择器 |
| `utils/__init__.py` 无 `__all__` 平铺 + import 副作用（monkey patch、settings 写盘） | 违背显式导入与延迟初始化原则 |
| GroundingDataset 硬编码实例数断言 | 把固定数据集计数写死进库的反例 |
| 9 宫格 mosaic | 第二布局维护成本高、实践少用；借鉴 mosaic 只做 4 宫格 |
| `NMSModel` 运行时怪癖清单（TF/CoreML/INT8 端到端导出） | 服务于 NMS 嵌入导出路线；AutoVision 后处理留在 `TaskNetwork` |
| `check_amp` 下载外部权重做校验 | 净室实现改用调用方自有模型或随机张量对照 |
| `segment2box` 慢路径精确裁剪解 | 复杂度/收益比不佳；快路径两分支可记 |
| Albumentations 锚点反射检测 + 失点补位 hack | 强耦合第三方内部 API，除非 albumentations 成为一等公民否则跳过 |

## 6. 实施顺序与依赖

```text
P0（立即，互不依赖，可并行）：
  D1 元数据闭环        2–3 人日
  D2 tracker 状态卫生   1–2 人日
  D3 box_candidates    1–2 人日

P1（近期）：
  D4 autobatch         3–5 人日   （独立）
  D5 计时方法论+floor  2–3 人日   （独立）
  D6 GMC               3–5 人日   （依赖 D2 的 reset()； cv2=video extra）
  D7 钩子化+BoT-SORT   5–8 人日   （依赖 D6 的挂点设计；第一步可先行）

P2（中期，数据/需求驱动）：
  D8 扫描缓存          3–4 人日
  D9 大图切片          3–4 人日   （依赖 D3 的 clip_polygon_to_canvas）
  D10 图像读取缓存     4–5 人日
  D11 下载续传         2–3 人日

P3（按需）：
  D13a LineZone 穿越补偿 1–2 人日 （依赖 D2 的退役通知语义，非硬依赖）
  D13b solutions 层      5–8 人日 （触发条件：产品需求；依赖 D2/D6/D13a）
  D12 rect 批次          5–8 人日 （触发条件：padding_ratio>15% 证据）
  D14 小件               各 ≤1 人日（a/b 随 D1 捎带）

第二轮（D15–D31，2026-11-03 追加；互不依赖者可并行）：
P1：
  D15 参数二分+handle 复用   2–3 人日（独立）
  D16 AMP 事前守卫           2–3 人日（独立）
  D17 resume 加固+早停持久化 3–4 人日（独立；先落回归锚点再动 engine）
  D18 空闲 GPU 选卡          2–3 人日（独立）
  D19 COCO 交换健壮性        2–3 人日（独立）
P2：
  D20 save_period+序列化卫生 2–3 人日（独立）
  D21 OOM 降批               2–3 人日（独立；与 D4 探测的互斥语义需核对）
  D22 回调注册表             3–4 人日（独立）
  D23 训练杂项 a–d           各 1–2 人日
  D24 增广四目标协议         1–2 人日（协议先行评审；落地随 mosaic/mixup 移植）
  D25 扫描缓存模板化         2–3 人日（依赖 D8）
  D26 深度/语义协议对照      2–3 人日（含盘点；深度校验与 D25 共享）
  D27 指标容器+实时混淆矩阵  3–5 人日（含命名弃用流程）
P3：
  D28 视频源增量             2–3 人日
  D29 小件 a–c               各 1–2 人日
  D30 INT8 校准清单          触发式（INT8 立项时）
  D31 导出参数矩阵           触发式（CLI 参数膨胀时）
```

注：D4 的顺延尾巴（`probe_train_batch_size` 的任务面闭包与 CLI
`batch_size="auto"` 实测接线）仍挂在第一轮 D4 名下，优先级等同本轮 P1。

合入纪律（每项适用）：最小改动 + 目标测试；行为变化项更新
`CHANGELOG.md`；公共 API 变更遵守弃用政策（AGENTS.md §3.4）；溯源登记
见 §9。

## 7. 通用验收标准

每个 PR 除仓库常规门禁（`ruff check src tests scripts`、mypy、
`mkdocs build --strict` 零警告、相关 parity/回归测试）外：

1. **类型提示与 docstring**：新公共函数/类 Google 风格参数说明；
   测试 AAA 结构、`pytest.param(..., id=...)`。
2. **行为锚点**：凡涉及既有行为改造（D2/D7/D13a），必须先落
   golden/回归 fixture，改造后逐位一致或显式声明差异。
3. **依赖边界**：惰性 import + 明确 `pip install -e '.[extra]'` 报错
   （D6 的 cv2、D7 若引入可选 ReID encoder）。
4. **文档**：面向用户的功能（D4/D9/D11/D13）更新对应 `docs/` 页面；
   架构约定（D2 的流生命周期、D10 的缓存选型）更新
   `docs/framework_design.md`。
5. **基准可比性**：D5 落地后既有基准表数值不得无声明漂移
   （robust 默认关闭，翻转默认走 0.5 的行为变更流程）。

## 8. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| AGPL 净室纪律被"参考实现"侵蚀 | 许可污染整个发行物 | §2 红线写入每个 PR 模板；实现者只读设计摘要不逐行对照；`scripts/check_provenance.py` 保持绿色；商业发布前法务复核台账 |
| tracker 重构引入行为漂移 | 下游计数/可视化静默变化 | D7 第一步 golden 等价性先行合入，BoT-SORT 作为纯新增 |
| autobatch 显存画像偏差 | 建议值过大导致训练 OOM | fraction 默认 0.6 + `AUTOVISION_AUTOBATCH_MAX` 钳制 + 文档声明是建议值 |
| 基准数值漂移 | 外部可比性受损 | robust 计时默认关闭；翻转默认走行为变更流程并更新 `docs/benchmarks.md` |
| 缓存正确性（D8/D10） | 陈旧缓存遮蔽数据问题 | 指纹含 mtime_ns + 任务参数；`data_check --no-cache` 与 `AUTOVISION_SCAN_CACHE=0` 旁路；缓存只存校验结论不存标注 |
| cv2 进入 tracking 路径 | 违反可选依赖约束 | D6 强制惰性 import + video extra 报错文案；CI 增加无 cv2 环境的 tracking 冒烟 |
| 元数据 schema 演进 | 旧工件误报不匹配 | `autovision_meta_version` + 缺失即放行的兼容策略；mismatch 报错给出两侧实际值 |

## 9. 溯源登记模板

各项落地时在 `docs/provenance/LEDGER.md` §1 表格追加：

```markdown
| `<落地文件>` | [ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（<D 编号>） | 本设计文档 §<x>、NOTICE |
```

`NOTICE` 的 AGPL 段落于 2026-11-03 二次改写为**协议式**结构：红线句
"must never have source or weights copied into this repository" **逐字保留**
在运行时依赖段落原位不动；其后的"架构参考"段改为三条纪律——
**Permitted**（读上游代码提取设计：组件结构、接口形态、算法步骤、边界
处理，再依据本设计文档与论文级语义在 AutoVision 风格内实现）、
**Prohibited**（复制或逐行翻译上游源码、复制注释/docstring/常量表/测试
数据、隔离 extra 边界之外的运行时导入）、**Required**（每个落地文件在
台账登记为架构参考 + 净室声明 + 复核日期，商业分发前复核边界）。协议
不再枚举具体落地特性（避免清单过时），登记事实以本台账与本设计文档
§10 为准。台账"代码 lineage"下的**净室声明**（实现依据本设计文档与
算法论文语义，未对照上游源码逐行翻译）随每轮落地追加并注明复核日期。

## 10. 落地状态（2026-11-03）

| 项 | 状态 | 落点 | 备注 |
|---|---|---|---|
| D1 | ✅ 已落地 | `export/artifact_metadata.py`、`export/export_onnx.py`、`inference/engine.py` | 三载体（ONNX 图内 / OpenVINO `<xml>.json` / TRT `<engine>.meta.json`）；`tests/test_artifact_metadata.py`（19 例） |
| D2 | ✅ 已落地 | `tracking/byte_track.py` | `det_index` / `drain_removed_tracks` / `reset()` / 实例化 Kalman；`tests/test_tracking.py` +6 例 |
| D3 | ✅ 已落地（范围收敛） | `augment/detection/box_candidates.py`、`geometry/polygons.py`、mosaic 接线 | `obb_candidates` 未单独落地：OBB 管线（翻转+缩放）无劣化变换、无接线点，其多边形裁剪需求由 D9 的 `clip_polygon_to_canvas` 承接；`tests/test_box_candidates.py` + `tests/test_geometry_polygons.py` |
| D4 | ✅ 核心已落地；CLI 接线顺延 | `engine/autobatch.py` | 探测循环 + 纯拟合策略 + 可 mock 测试（9 例）；`batch_size="auto"` 的任务面闭包接入留给各任务面 PR（模块 docstring 有契约） |
| D5 | ✅ 已落地 | `export/timing.py`、`export/benchmark.py`、`registry/deployment_gates.py` | robust 计时默认关闭；`MetricFloorGate` opt-in；`tests/test_timing.py` + gates 测试 |
| D6 | ✅ 已落地 | `tracking/gmc.py`、`byte_track.py`、`inference/stream.py` | 五方法 + 前景剔除 + 降采样还原；`tests/test_tracking_gmc.py` |
| D7 | ✅ 已落地 | `byte_track.py` 钩子化 + `tracking/bot_sort.py` | golden 快照 `tests/golden/byte_track_lifecycle.json` 逐位锚定；ReID 出范围（`_first_association` 覆写为扩展点） |
| D8 | ✅ 已落地 | `data/scan_cache.py`、`object_detection_dataset.py`、`cli/data_check.py`、`runtime/env.py` | 接入点为文件清单校验层（COCO/YOLO 清单无独立扫描阶段）；CLI 旗标为 `no_cache=true`（key=value 壳既有风格） |
| D9 | ✅ 已落地 | `data/split_large_images.py`、`cli/_app.py`（`tile-dataset`）、`tasks/shared/tiling_utils.py` | OBB 裁剪/多边形 IoF 在模块内实现 |
| D10 | ✅ 已落地 | `data/image_cache.py`、`data/image_dataset.py` | 偏差记录：返回 torch tensor（沿 `open_image` 既有契约）；mtime 记录用 npy 内 header 数组（同文件原子发布） |
| D11 | ✅ 已落地 | `data/downloads.py` | `.part` 改为确定性隐藏名以支持跨进程续传（sha256 兜底并发） |
| D12 | ⏸ 证据门未触发 | — | `padding_ratio` 指标未落地前不实现（§4.12 前置条件） |
| D13a | ✅ 已落地 | `tracking/analytics.py` | chord 真交点判定（端点触碰不算），每 track 每帧至多一次 |
| D13b | ⏸ 按产品需求触发 | — | solutions 层未立项 |
| D14 | ✅ a/b/c/d 全部落地 | `serving/app.py`、`export_onnx.py`、`docs/framework_design.md` §3.1 | |

第二轮（D15–D31）：

| 项 | 状态 | 落点 | 备注 |
|---|---|---|---|
| D15 | ✅ 已落地 | `cli/model_handle.py` | `predict(backend=..., artifact=...)` + 签名缓存（backend/artifact/device/providers/openvino_device 五键——上游无 precision 参数，键集按本仓库实际）；**LRU-2 槽**（2026-11-03 压力探针实证单槽在交替双签名下 1600 调用 1569 次重建，2 槽覆盖该模式；更多配置仍会颠簸，docstring 已声明；审查修正：dict 赋值不重排——真 LRU 需 pop+重插，有方向性回归测试区分 LRU/FIFO）；`train()` 换网与缓存失效同锁；pickle 剔除缓存且 pre-D15 旧 pickle 兼容（`setdefault` 双键；`export()` 不失效缓存为已记录偏差——签名含 artifact，行为良性）；`tests/test_model_handle_engine.py`（16 例） |
| D16 | ✅ 已落地（语义修正） | `engine/amp_check.py`、`engine/engine.py`、`cli/train_task/object_detection.py` | 探针为**三态**：False=确认不安全降 fp32 / True=安全 / **None=无法探测（保持请求精度）**——设计原稿的"缺输入即降级"在 GPU 机器上会误伤无 `model_input_spec` 的模型，实测修正；DDP 广播三态 int8（True=1/False=0/不可判=-1，二值编码会把不可判静默降级）；trainer_args `amp_check` 仅 OD 面接线（其余任务面跟进）；`tests/test_amp_check.py`（15 例） |
| D17 | ✅ 已落地 | `engine/engine.py` | `engine_state` 新增 `no_improve`（resume 续接早停耐心）+ `fit_args` 快照（`check_resume_fit_args` 白名单 `RESUME_OVERRIDABLE_KEYS` 外漂移告警、现值治理）；possible_stop 距阈值 1 次验证时提前验证一次（patience>=2 守卫）；`tests/test_resume_hardening.py`（5 例）+ 既有早停回归全绿 |
| D18 | ✅ 已落地（范围收敛；降级语义二次修正） | `runtime/autodevice.py`、`engine/engine.py::resolve_device`、`cli/common_helpers.py::resolve_num_devices` | 偏差：多进程 torchrun 由 LOCAL_RANK 映射设备，`"0,-1"` 的完整多卡语义需启动器配合——本版交付 `resolve_device("-1")` 单进程选卡 + `resolve_device_spec` 供启动器预解析 + 逗号 spec 进程计数。**降级语义（2026-11-03 真机探针修正）**：阈值筛光 → 两段式回退（stage-2 以零阈值选最不忙可见卡）→ 仍不足**抛 `AutoVisionError`**（原"顺序填满→cpu"会在单卡繁忙机上产出 `cuda:0,cpu` 混搭——多进程第二个 rank 落 CPU，实证为 P1 缺陷）；`tests/test_autodevice.py`（22 例 1 跳过；审查修正：claimed 过滤统一 torch 命名空间——CVD 翻译下显式 id 曾可被占位符重复选中） |
| D19 | ✅ 已落地（范围收敛） | `geometry/polygons.py`、`data/coco_instance_segmentation_dataset.py` | `merge_multi_segment` **未落地**：本仓库无 COCO→YOLO-seg 转换器，无消费方，落地 `is_valid_polygon`/`polygon_from_bbox` + 数据集白名单兜底（聚合告警）；顺带修复 compressed RLE（string counts）误送 `frPyObjects` 的既有缺陷；`tests/test_coco_polygon_sanitization.py`（15 例） |
| D20 | ✅ 已落地 | `engine/engine.py`（`_build_checkpoint`/`_prune_epoch_archives`）、`runtime/checkpoint.py`（`serialize`/`save_serialized`） | `save_period_num_epochs` 落在预训练 `Engine.fit`（epoch 制；task 面为 step 制，等价物 `save_every_num_steps` 已有）；卫生三件：fp16 溢出**克隆钳制**（不动 live 权重）、4-d 权重 contiguous 归一、单次 `serialize()` 多路原子写（存档 epoch 才走 BytesIO，常规保存保持流式避免 RAM 尖峰）；滚动保留最近 3 份 |
| D21 | ✅ 已落地（单进程限定） | `engine/engine.py`（`_is_oom_error`/`_clone_loader_with_batch_size` + step 循环 try/except） | **只做单进程**（world_size==1 且非 FSDP）：DDP 下 OOM 是非对称事件，单 rank 捕获后无法安全同步重建——与上游"仅单卡"纪律一致；batch 减半 + loader 克隆重建（sampler 实例保续）、optimizer/scheduler 保留、OOMed step 跳过、3 次上限、batch=1 时不降级直接 raise |
| D22 | ✅ 已落地 | 新增 `engine/callbacks.py`；TaskEngine/Engine 接线 | `CallbackRegistry`（6 事件，克制扩张）+ `adapt()` 鸭子兼容（旧 `callbacks=` 列表不变）+ `on_checkpoint_save` 新事件；异常隔离逐 handler（严于上游：上游 `run_callbacks` 无隔离）；`registry.run` 不分 rank，rank 门控仍在引擎调用侧 |
| D23 | ✅ a/c/d 落地；b 已存在 | a: `common_helpers.build_task_dataloader(seed=)`/`train_helpers.get_dataloader(seed=)` + OD/预训练调用点；c: 新增 `runtime/model_info.py` + 横幅 GFLOPs；d: 新增 `runtime/fuse.py` | **b 偏差：无需实现**——`build_task_dataloader` 已有 worker≤batches + 单批零 worker（含实测备注，common_helpers.py:252-254）；c 的 thop 惰性导入、缺失输出 n/a（不伪 0）；d 为保守 Sequential(conv,bn) 模式 + `assert_fused_parity` 数值门禁（eval 语义，QAT 出范围） |
| D24 | ✅ 协议落地 | 新增 `augment/core.py`（`GeoTransform` 四目标模板 + `nearest_resize_to` + 255/0 值约定） | 协议先行；复合增广（mosaic/mixup）移植时以此为实现层，方法制 `MethodTransformArgs` 体系不变 |
| D25 | ⏸ 盘点后顺延 | — | 深度/语义数据集确实未接 scan_cache（grep 确认）——真实缺口，但接线需各数据集定义校验钩子，属独立工作量；D8 的清单校验层保持现状 |
| D26 | ⏸ 盘点：核心已对齐 | — | 深度存储**已是** uint16 PNG + `png_depth_scale`（默认 1000 毫米，depth_estimation_dataset.py:78-82）+ ignore 约定；语义**已是** ignore_index 255。剩余 delta 仅：深度/图纵横比校验、语义 label_mapping LUT 化——随 D25 顺延 |
| D27 | ✅ 容器落地 | 新增 `metrics/containers.py`（`TaskMetrics` + `empty_empty_score`） | stats 惰性缓冲 + `export_state/merge_state` 保持既有鸭子协议；现有 Collector/Evaluator 公开行为不变（渐进迁移，命名弃用流程未启动）；验证循环内实时混淆矩阵接线属各任务面跟进（同 D4 模式） |
| D28 | ⏸ 盘点后顺延 | — | 当前推理面**没有**帧 stride 采样特性，grab/retrieve 抽帧增量依附于该未来功能；批不跨视频/清单文件/PIL 通道归一化随特性立项一并评估 |
| D29 | ✅ a/b 落地；c 对照后不落地 | a: `data/quality.py::probe_storage_speed` + `cli/data_check.py` 接线；b: `metrics/map_evaluator.py`（`auto_raise` 跟踪 + compute 期抬升）+ `evaluate_object_detection(max_det=)` | a 抽样 ≤5、stat 10ms / 吞吐 50MB/s 阈值、<16KB 豁免、只告警不判罚；b 利用"maxDets 在 compute 期才消费"的实现细节做**零额外 IO** 的自动抬升（opt-in：`max_det=None` 时启用，显式传值钉死协议）；c 偏差：AutoVision 已用 Pydantic 数据配置协议（path/split 分离 + 类型校验），与上游 YAML 协议等价，不引入第二套 |
| D30 | ⏸ 触发式 | — | INT8 校准立项时按单执行（§4.30） |
| D31 | ⏸ 触发式 | — | 导出 CLI 参数膨胀时再立项（§4.31） |

本轮验证：`ruff check src tests` 全绿；mypy 新模块零错误（`common_helpers.py`
的 4 条既有错误位于 `get_num_workers`，非本轮引入）；`mkdocs build --strict`
零警告；`check_layering.py` / `check_provenance.py` 绿色。审查轮（独立子代理
复查）修复：DDP 三态编码、`_resolve_engine` 锁内取网络（防 train/predict
竞态缓存陈旧权重）、`resolve_device` 多 token 告警、autodevice 的
"torch 无 CUDA 也降级"实现与测试 CVD 环境隔离、`polygon_from_bbox`
返回 None（畸形 bbox 不再穿透）、数据集 bbox 兜底 None 安全、OD
`amp_check` 严格 bool 校验、banner 先解析后打印。

第二轮（D20–D29）审查轮修复：删除 OOM try 的复制 except handler（手术
残留死代码）；try 嵌回 `engine.step` span（恢复 dataload 嵌套语义，
observability 测试注释重新成立）；`_with_sampler`/`_clone_loader_with_batch_size`
透传 `worker_init_fn`/`generator`（否则 DDP 重建与 OOM 重建静默丢 D23a
播种）；`make_dataloader_generator` 加 rank 偏移；GFLOPs 探针放到模型
权重设备（GPU 机器上恒 n/a 的实证修复）；`on_train_end` 两循环统一传
result 对象；存档 metadata epoch 1-based 与文件名对齐、保留份数提常量；
`callbacks.run` 未知事件改为显式拒绝；fuse 拒绝 affine=False BN、parity
先比输出数量；`merge_state` 从合并 GT 重推最密图计数（DDP auto_raise
不再低估）；`evaluate_object_detection` 校验 `max_det>=1`；补
save_period 端到端、OOM 预算耗尽、batch=1、重建保播种四个测试盲区；
known_limits 补记 D23a resume 播种流不保证与首跑逐位一致。

第三轮（2026-11-03，企业级验收测试落地）：按分层验收方案补齐两个 L2/L3
锚点——`tests/test_training_trajectory_golden.py`（金标准轨迹：同 seed 的
微型训练 loss 序列 + 数据顺序逐位锚定 `tests/golden/training_trajectory.json`，
同 seed 逐位确定性已实证，任何训练循环数值静默漂移在此失败）与
`tests/test_crash_recovery_integration.py`（SIGKILL 崩溃重启 e2e：原子写
耐久性 + D17 状态续接 + 恢复后早停预算 + stale-tmp 清扫，posix 限定，
~10s）。两项把"friendly 双跑 resume"与"mock 显存"升级为敌对路径实证。

合规执行：溯源台账 §1 新增 6 行 + 净室声明；`NOTICE` AGPL 段落改写为
"红线句原文保留 + 架构参考纪律"两段式（红线句逐字未动）；
`scripts/check_provenance.py` / `check_layering.py` 保持绿色。
