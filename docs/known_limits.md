# 已知限制与路线图

> 本页汇总对外可见的边界与工程注记；导出相关细节见[导出与部署](export.md)。

## 定位

训练 + 参考推理框架。训练、评测、导出是生产路径；跟踪与 serving 是参考实现。
serving 的认证自 0.6.1 起**默认 fail-closed**：非回环地址（非
`127.0.0.1`/`::1`/`localhost`）且未提供 `api_key=` 与
`AUTOVISION_SERVING_API_KEY` 时拒绝启动；显式设置
`AUTOVISION_SERVE_ALLOW_INSECURE=1` 可接受开放端点（`create_app` 作为库
入口保持原有 opt-in 语义）。其余能力：请求体/像素上限（单图默认 16M 像素，
一次请求总像素默认 `max_batch × max_image_pixels`，超限 413）、连接层上限
（`limit_concurrency`，默认 64）、优雅停机默认 30s、线程池推理、有界排队
（`queue_timeout_s`，饱和返回 503）、单次引擎调用
超时（`inference_timeout_s`，默认 120s；超时返回 504 并将该实例标记为不可再用，
此后 `/readyz` 返回 503（`poisoned=true`）、`/healthz` 携带 `poisoned` 字段，
编排器应摘除流量并重启进程）、`/readyz`、TLS、多 worker（`api_key=` 在
workers>1 下被拒绝，key 必须走环境变量）、进程内 token-bucket
限流、预测分布漂移
（PSI，`/drift`）与 A/B/canary 分流；`/docs` 与 `/openapi.json` 默认关闭。
全局配额、告警、跨进程一致限流、A/B 统计显著性与金丝雀自动回滚仍属生产网关职责。

## 运行与完整性

- **确定性**：`AUTOVISION_DETERMINISTIC=1` 开启 torch 确定性算子与 cuDNN
  deterministic（部分算子无确定性实现会报错，且通常更慢）；默认关闭。
- **fp16**：CUDA/NPU 上 `precision="16-mixed"` 使用 `GradScaler`，其状态随
  checkpoint 保存/恢复；bf16/fp32 不需要。AMP 溢出由 scaler 跳过该步并回退
  scale，不会被非有限梯度检查中止。
- **断点续训**：checkpoint 保存/恢复 Python/NumPy/Torch RNG 状态，但不恢复
  mid-epoch sampler 位置，也不校验配置/版本差异；续训时优化器状态会覆盖新的
  `lr` 等参数。需要严格轨迹复现时请从 epoch 边界续训。加载跨版本 checkpoint
  会记录 warning（不阻断）。checkpoint/导出/manifest 写入均先清扫同目标
  超过 1 小时的崩溃残留临时文件，再开始本次写入。D17 后 resume 会续接早停
  耐心计数并对白名单外超参漂移逐键告警（当前调用参数治理）；dataloader 的
  worker 播种流（D23a）从运行起点的 seed 重建，与首次运行的 shuffle 顺序
  不保证逐位一致。
- **mmap 缓存完整性**：缓存文件打开时校验，损坏/截断会报带重建指引的错误
  （删除文件或 `overwrite=True` 重建）。运行中文件被外部截断（与他人共享
  `AUTOVISION_DATA_CACHE_DIR` 时的外部写入者）会导致 SIGBUS 进程崩溃——这是
  OS 层约束；请勿让不受信进程写缓存目录。
- **下载完整性**：权重下载校验钉住的 sha256；数据集下载写入临时文件并原子
  替换，缓存压缩包在解压前重新校验 size/sha256（有钉住值时）。内置 COCO/VOC
  数据集条目没有官方摘要，默认只告警。设置
  `AUTOVISION_REQUIRE_DATASET_HASHES=1` 可拒绝未钉摘要的数据集下载；
  `AUTOVISION_OFFLINE=1` 同时拒绝数据集与权重下载。
  `scripts/compute_dataset_pins.py` 可从已下载工件计算 sha256/size 并输出
  可粘贴的 registry 片段（加固计划 H1-9；补齐 pin 前条目列在
  `tests/test_data_hardening.py` 的 `UNPINNED_ALLOWLIST`，清空该表是目标状态）。
- **数据加载严格性（0.6.1 起）**：COCO 注释中的重复 `image id` 现在在加载期
  报错（此前静默覆盖）——检测/实例分割/关键点三个面统一（DATA-08）；
  `category_id` 未在 `categories` 声明的标注会计数并汇总 warning（此前静默
  丢弃）；引用未声明 `image_id` 的孤儿标注同样计数告警（DATA-29）；
  `ignore_classes` 中未声明的 id 输出一次 warning（DATA-30）；YOLO/OBB
  标签的字段数不符行与"字段数合法但值非法"的行仍被跳过，但会同通道计数并
  输出带文件与行号的 deprecation warning，0.7.0 起默认改为报错
  （DATA-06）；命名未知类别的行计数并汇总 warning（DATA-07）；
  二进制/GBK 标签文件报错带文件名（DATA-21）；声明 `width/height <= 0`
  的图像在构造期报错（DATA-20）；COCO `file_name` 的目录穿越防护
  （`resolve_under`）统一到检测/实例分割/关键点三个加载器与 Label Studio
  interchange（DATA-10）。`resolve_under` 的包含判断按解析后真实路径
  （`Path.resolve()`，跟随符号链接）计算：**指向数据根之外的符号链接
  图像/掩码会被拒绝**——用符号链接挂载共享存储的图像树需要把真实目录
  作为 `images_dir` 传入。分割任务在读取期校验图-掩码空间尺寸一致
  （语义/全景，DATA-11）。分割掩码解码保持宽松（允许历史截断掩码），但该
  宽松旗标不再泄漏到图片解码——截断图片始终响亮报错（H1-10）。
  `open_image` 的返回通道数强制等于 `num_channels`（1 通道请求对 RGB/RGBA
  源返回 1 通道；非法通道数报错，DATA-09）。
- **推理结果序列化**：`to_json(masks="rle")` 对 float dtype 的语义分割
  label map 会输出 deprecation warning（当前仍按 `> 0.5` 二值化；0.7.0 起
  拒绝，请转整数 dtype 或改用 `masks="dense"`/`"none"`）；`masks="dense"`
  在元素数超过 `max_elements×100` 时直接报错而不是构造无界 JSON
  （加固计划 H1-11/H3-1a）。
- **旧 teacher checkpoint**：当前导出的 task model 可被 pickle-free 加载器读取；
  旧导出若嵌入 `__class__` 等对象，默认拒绝。只有可信本地产物才可设置
  `AUTOVISION_TRUST_TEACHER_CHECKPOINT=1` 启用完整 unpickler（RCE 风险自担）。
- **mmap 缓存键**：包含标签文件（COCO JSON / YOLO `.txt` 等）与图像文件
  （路径、大小与 mtime）的双指纹，以及 `ignore_classes` 指纹——编辑标签或
  增删图像都会重建缓存（DATA-03；修复前 YOLO 增删图像不失效）；
  `overwrite=True` 会强制重建，删除动作仅由 rank0 执行（DATA-12）。残余
  边界：DDP + `overwrite=True` 下早于 rank0 到达的非 0 rank 可能在删除前
  打开**旧** mmap 并 pin 引用——该 rank 本轮训练的行集合与 rank0 静默
  分叉（挂死已闭合，分叉窗口未根治；根治需要版本化发布协议，属 P3
  遗留项）。
  user-held 权重解析现在强制校验 sha256（按路径/大小/mtime 缓存摘要）。
- **指标口径（METRIC 系列）**：OBB mAP 的 PR 构建（背景 FP 计入）与
  AP 包络积分均已修正，**历史 OBB 数字与 0.6.x 之前不可比**；headline
  precision/recall/F1 为单操作点读数（`headline_score_threshold`，默认
  0.50），**与 0.6.x 之前的 argmax-F1 扫描读数不可比**（AP/AR 不变）；
  含 crowd 数据的训练期 mAP 因链路接通而上升。合规 CLI 默认
  fail-closed（`fail_on="error"`），`strict=` 弃用。
- **导出/注册表边界（EXPORT 系列）**：fp16 在无 CUDA EP 机器上导出的
  工件携带 `autovision_verify_strength=smoke-only`（数值未对拍；
  `VerifyStrengthGate` 可选强制 full）；TRT/OV 的身份 sidecar 按
  加载器 `with_suffix` 命名（`model.meta.json` / `model.json`）；>2GB
  external-data 导出在 onnxslim 后 fail-fast（管线不支持该规模）；
  `register_downloadable` 名称穿越/NUL/绝对路径拒绝（命名空间斜杠
  合法）；OFFLINE 下损坏缓存保留现场不删除；`promote` 支持
  `set_current=False`（默认行为不变）。TRT 失败路径保留 staged ONNX
  是 H3-4b 的已测试决策（调试工作流），与 EXPORT-11 审计建议冲突、
  未实施失败清理。
- **serving 语义（SERVE 系列）**：引擎/序列化边界故障返回 **500**
  （客户端输入错误保持 400；依赖 400 的监控规则需迁移）；解码与推理
  超时分离（`decode_timeout_s` 默认 `min(10, inference_timeout_s / 2)`，
  解码超时不毒化）；`warmup_timeout_s` 默认 60s（0 禁用 warmup）；
  限流桶键按端点命名空间化，`trust_proxy_headers` 弃用（0.6.1 告警、
  0.7.0 移除），改用 `trusted_proxy_hops`；registry CLI 对未知键在
  执行前拒绝；CLI 辅助层配置错误统一 `ConfigValidationError`
  （其是 `AutoVisionError` 子类，既有 `except AutoVisionError` 不变）。
  warmup 超时命名启动失败，但 `asyncio.to_thread` 中的挂死线程无法被
  取消——**永不返回的 wedge（CUDA 停滞）会使进程退出本身被默认执行器的
  join 阻塞**，编排器最终需要 SIGKILL；就绪探针在该窗口内保持失败。
- **数据快照规格**：`dataset_snapshot` 原生理解 Ultralytics `data.yaml`
  （`path` 锚定相对根、`names` 只作元数据记录）；带显式 `format:` 键的
  spec 走通用规则（DATA-01）。校验会检出"快照后新增的 split"
  （`added_split`，DATA-02）。
- **训练引擎边界（TRAIN 系列）**：任务调度器由各模型自有实现，resume
  扩展 `steps` 时其衰减形状被 checkpoint 固定（指名 warning，不会静默）；
  预训练 `CosineWarmupScheduler` 的续训扩展 `max_epochs` 按新终点衰减。
  训练进程**无 SIGTERM/SIGINT 优雅停机**：被杀的 run 依赖最近一次周期
  checkpoint（原子写保证可用），正在写的 tmp 依赖 1 小时龄期清扫；
  dataloader worker 崩溃（非 OOM）终止整个 run。`CheckpointMetadata`
  的新增 `step` 字段在旧 checkpoint 上为 None；fit_task 的 `epoch`
  字段历史上承载 0-based step（TRAIN-09）。
- **EoMT 权重**：`load_weights=True` 强制 DINOv2/DINOv3 许可 opt-in；
  hub checkpoint 未在 `WEIGHT_MANIFEST` 中钉 sha256 时默认拒绝，只有显式
  `AUTOVISION_ALLOW_UNPINNED_WEIGHTS=1` 才加载（可信镜像）。class head 会按
  目标类别数/query 数重建；EoMT 现可走 ONNX 导出（legacy TorchScript 路径，
  动态 batch 关闭）。
- **推理运行时边界（INFER 系列）**：`run_stream` 默认
  `on_frame_error="skip"`——坏帧被跳过并写入 JSONL 错误行（连续超过 10 帧失败
  仍中止；计数在健康帧与重连时重置）；需要旧行为传 `on_frame_error="raise"`。CLI predict 与 serving
  对**无元数据**的 ONNX/TRT/OpenVINO 工件拒绝构建引擎（`InferenceEngine`
  直接构造仍可用但输出 warning）；float 语义 label map 的渲染按四舍五入
  容忍（0.7.0 随序列化一起拒绝）；multi_stream 的 JSONL 时间戳是
  `time.time()` 墙钟（NTP 回拨可乱序——单调时钟会破坏工件的绝对时间
  语义，属有意取舍）。
- **可视化/视频工具**：`visualize/effects.py` 与 `geometry/boxes.py` /
  `geometry/masks.py` 的框、mask、polygon 工具只依赖 NumPy/Pillow；
  `mask_to_polygons` 惰性依赖 OpenCV（`video` extra）。`process_video` /
  `VideoSink` / `ImageSink` 是同步、RGB 约定的轻量管线，不替代
  `run_stream` 的 reader/infer/writer 三段式、后端切换与 JSONL 持久化。
- **运行清单**：每次训练在输出目录写 `run_manifest.json`（命令、git commit、
  环境白名单、依赖版本、GPU）；resume 写独立的 `run_manifest.resume-*.json`，
  不覆盖原始记录。
- **早停/回调**：task training 支持 `early_stopping_patience` /
  `early_stopping_min_delta`（CLI `trainer_args`）与 Python API 的 callback 钩子；
  pretrain 仍按 epochs 跑满。

## 企业能力边界

- **训练并行**：DDP（torchrun）、FSDP 任务训练
  （`trainer_args={"parallel": "fsdp"}`，full-state checkpoint/resume/导出，
  2-rank 单卡测试）与梯度检查点已实现；FSDP 的 EMA 更新需按
  `AUTOVISION_FSDP_EMA_INTERVAL` 做 full-param gather，整模型未做 auto-wrap。
  DeepSpeed / ZeRO / 弹性训练未接入；多机 NCCL 与故障注入未在公开 CI 验证。
- **注册表**：文件系统后端与 MLflow 后端都执行标准部署门禁
  （`gates=None` 默认 fail-closed）；rollback/audit 仍以文件系统后端为准。
- **CI**：公开 CI 为 Ubuntu + Python 3.10/3.11 的 CPU 任务，测试作业安装
  `serving`/`video`/`transformers` 测试依赖；新增核心子集 coverage 下限、bandit
  （medium/high）、核心模块 mypy 与 `scripts/smoke_enterprise.py` 冒烟门禁。TensorRT/OpenVINO/多机 DDP 等重环境
  用例按门控变量跳过；真实 TensorRT engine/parity 已在 RTX 4090D + TensorRT 11.4 上验证
  int8 QDQ 路径（含 `export_tensorrt(verify=True)` 门禁）；公开 CI/夜间
  self-hosted workflow（`.github/workflows/trt-nightly.yml`）继续覆盖。
- **serving**：TLS、多 worker、进程内限流、漂移与 A/B/canary 分流已提供；
  `/metrics` 与 `/drift` 在设置 API key 后同样要求鉴权（K8s 探针保持公开）；
  限流桶与漂移窗口在多 worker 下各自独立（跨 worker 聚合需指标后端/网关），
  `/drift` 是服务级聚合（A/B 混合流量会混合分布）；无模型热加载（用滚动重启），
  `deploy/k8s/` 提供 Deployment/Service/HPA/PDB/PVC 参考清单；全局配额、告警、
  A/B 统计显著性与自动回滚仍需平台承接（见上文定位）。

## 许可状态与 STA 迁移

- **DEIMv2 组件已删除（2026-10-17）**：`SpatialPriorModule` 已从代码库移除，
  DINOv3 ViT 变体保留但改用 no-STA adapter；发行物不再包含 DEIMv2 代码，
  可按 Apache-2.0 发布。处置记录见[溯源台账](provenance/LEDGER.md)。
- **旧 STA-era ViT 检查点不受支持**：任何含 `backbone.sta.*` 的基线 `.pt`
  或用户自持 ViT 资产（`dinov3-vitt16` / `dinov3-vitt16plus` /
  `dinov3-vits16`）会被加载器明确拒绝，解析入口在读取/哈希容器前即失败，
  需重新训练/导出 no-STA 检查点。DINOv3 ConvNeXt 及用户自持的 4 个
  ConvNeXt 资产不受影响；`USER_HELD_LOADABLE_ASSETS` 只含这 4 个，
  3 个 ViT 留在 `USER_HELD_UNSUPPORTED_ASSETS` 仅供审计
  （`allow_unsupported=True`）。
- **受限权重仍然存在**：DINOv2 权重 CC-BY-NC 4.0、DINOv3 权重 DINOv3
  License、用户自持检测器禁再分发/商用；这些是权重条款，与已删除的 STA
  代码无关。DEIMv2 外部权重仍为非商用。

## 适配缺口

- albumentations 适配层未接入（变换基于 torchvision v2）。
- super-gradients 分割 wrapper 未接入（CustomizableDetector 检测家族可用）。
- 难例加权采样（`hard_examples=`）与长尾采样（`long_tail_sampler=`）仅支持
  单机；DDP 的 DistributedSampler 会覆盖采样权重，组合被显式拒绝。
- Label Studio 的 `keypointlabels` 以 `keypoint_radius` 小框转换为检测框；姿态
  训练应使用原生关键点格式（COCO keypoints / YOLO-pose）。
- Label Studio 的旋转矩形（`rotation != 0`）按轴对齐框导入，旋转角保留在
  标注字段；旋转框训练仍使用 YOLO-OBB / DOTA 格式。
- 多路流运行时 v1 不做跨路跟踪 / ReID；tracking 仅单路 `run_stream`。
  每路渲染 writer 在该路分辨率变化时重建。`LineZone` / `PolygonZone` /
  `TrackSmoother` 与 reader/infer/writer 三段式管线也只接入单路
  `run_stream`；`run_multi_stream` 保持每源独立批处理。
- serving 的 `max_image_pixels` 按单图限制，`max_request_pixels`（默认
  `max_batch × max_image_pixels`）限制一次请求的解码总像素（超限 413）；
  RLE 响应字节数无上限，高并发大图 / 稠密 mask 场景需在网关层再加总预算。
  depth 任务的深度图超过 `max_json_elements` 时只返回 shape/dtype 摘要并在
  每条结果上标记 `depth_truncated: true`——完整深度图应走文件 / 对象存储
  通道（H2-5f），JSON 通道不承载像素级深度数据。

## 检测谱系状态

`n` / `s` 为 spec-frozen（规格冻结：组件与配方已承诺，预训练权重与 COCO AP
待发布）；`m` / `l` / `x` 骨干线方向已定（timm ConvNeXt，tiny→m / small→l /
base→x，选型标准为可导出 + 许可干净），实现待落地。`detr/gtrvit-s` 为 planned。只有 `released` 档带预训练权重：发布门禁要求权重 +
模型卡（许可 / sha256 / seed）+ 全量 AP + 可复现延迟。机器可读状态与变体清单见
[模型库](model_zoo.md)。

## 导出与部署

- GTR 全部五条路线（`detr/gtrvit-*`、`obb/gtr`、`pose/gtr`、`depth/gtr`、
  `semseg/gtr`）已可导出 ONNX/TensorRT/OpenVINO：trace 期经
  `prepare_onnx_export` 钩子换用 GLA 纯 ONNX 降级
  （`shared/gtr/gla_onnx.py`），eager 路径不变；导出为固定尺寸图（
  `dynamic_spatial=True` 会被快速拒绝），五条路线的端到端门禁在
  `tests/test_gtr_export_gates.py`。fp16 导出可用：GLA 的 fp32 状态段经
  `onnx_fp32_op_types`（MatMul/Resize/ConstantOfShape）钉住以保持图一致
  （ConstantOfShape 是 GLA causal 掩码的 trace 产物，多 chunk 尺寸下会被
  ORT float16 转换器漏转），但 GTR 的计算以 GEMM 为主，fp16 主要惠及
  Conv 段（ORT-GPU/OpenVINO），ORT CPU 反而更慢——CPU 部署用 fp32 图，
  TensorRT 建议由 fp32 图自行管精度。
- **TensorRT 版本边界**：GTR 展开纯图的 fp32 引擎在 TensorRT 11.4
  （11.4.0.106）上构建成功但数值错误（det/depth 输出系统性偏离 torch 与
  onnxruntime 参考；上游受支持环境是 TensorRT 10.14）。`export_tensorrt`
  在 TensorRT ≥ 11 上构建前会主动告警；引擎对拍门禁会拒绝并删除这类引擎
  （fail-closed）；该 TRT 版本上勿对 GTR 关闭 verify，部署走受支持的
  TensorRT 版本或上游 plugin 路线（见 docs/export.md）。
- GTR 的 eager 前向依赖 fla 的 Triton 核（CUDA-only）：CPU 上构造
  `backend='torch'` 的推理引擎会被可读报错拒绝（用 `backend='onnx'` 或
  `device='cuda'`）；CPU eager 部署走 ONNX 后端。TensorRT / OpenVINO 导出
  的 parity 参考经同一钩子回放导出安全路径：OpenVINO 参考固定在 CPU（此前
  会在 eager fla 核上崩掉全部 GTR 路线），TensorRT 在模型驻留 CPU 时回退并
  告警、GPU 驻留保持 eager-vs-engine 强校验。
- **fla 版本横幅**：GTR 依赖的 fla 在 Python < 3.11（及 Triton < 3.3.0）导入
  时即打印升级建议 WARNING（`fla/utils.py` 的 `check_environments()`，
  logger 名 `fla.utils`），每个导入 GTR 栈的进程各出现一次。仓库本身支持
  Python 3.10+，这是第三方包的环境建议而非功能错误；Triton 低于 3.3 的
  提示是真实风险信号（GLA 核可能出错），请勿在库代码里静默，根治方式是
  升级解释器/Triton。导出的 `.pt` 不携带模型家族元数据（GTR 检测导出与
  其它 DETR 导出同为 `DETRObjectDetection` + 纯数值超参），因此示例脚本对
  `--weights` 的 GTR-on-CPU 预检基于加载后网络的模块树内省
  （`examples/tasks/_common.py` 的 `gtr_eager_device_guard`），属示例层
  护栏而非格式契约。
- **NaN 恢复的 DDP 边界**：回滚恢复假定全体 rank 可见同一 `last.ckpt`
  （共享文件系统部署）。DDP 下任一 rank 非对称死亡后（OOM 自愈本就是
  单进程路径，`world_size > 1` 时直接 re-raise），`fit`/`fit_task` 的
  finally 会销毁进程组，对端在下一次集合通信上快速失败而非静默挂死；
  但不保证所有 NCCL 配置下都是即时失败，极端情况仍可能等到 watchdog
  超时。
- **验证 loss 非有限不触发回滚**：训练窗口已逐步校验权重/梯度，验证侧
  NaN 视为测量故障（坏验证数据/数值问题）：逐次告警、该轮按既有早停
  节奏计入 no-improve、best 跟踪不受污染（非有限值不能成为基线）。如需
  对验证数据问题触发训练回滚，属新需求而非缺陷。
- **升级兼容（存档期 checkpoint）**：save_period 存档路径 `last.ckpt`
  的元数据 epoch 已从 1-based 修正为 0-based（随 D32 生效）；由旧版本
  写出的存档期 `last.ckpt` 在 resume/回滚时仍会静默跳过一个 epoch——
  两种纪元约定无法从文件本身区分，跨版本续训前请确认 `last.ckpt` 由
  ≥ 本版本写出。
- **断流重连沿用打开时流属性**：重连只重开 capture，`info`（宽高/fps）
  保持 open-time 快照（帧序号延续）；RTSP 重协商改变分辨率的流，下游按
  旧几何渲染可能错位——监控场景建议固定编码参数。
- ONNX FP16 parity 门禁需要 CUDA execution provider：无 CUDA EP 的构建（如
  CPU-only onnxruntime）退化为「执行 + 有限性」smoke 门禁（输出含非有限值即
  失败）并告警；ONNX 导出的示例输入跟随模型设备（GPU 上训练后可直接导出）。
  即使有 CUDA EP，fp16 的 pose / depth / semseg 与任意精度的 det / obb parity
  仍为行求和不变式（fp16 简并权重存在 decode 相邻真实漂移、det/obb 图内
  top-k 会整行换入近平局候选，逐元素对齐不可行）；fp32 的 pose / depth /
  semseg 为逐元素对拍。
- 动态 H/W 支持面、int8 首层 fp32 配方、TensorRT 11 适配见[导出与部署](export.md)。
- int8 全量 AP 报告（`scripts/measure_int8_accuracy.py`）需要兼容的
  ONNX Runtime / TensorRT 环境；`export` extra 已要求 `numpy>=1.25`
  （onnx/onnxscript 的导入前提），重装该 extra 后即可复测。当前开发 venv
  仍是旧 numpy，所以本地 ONNX 集成测试按环境门控跳过。
- `examples/cpp/` 是参考运行时，不是 ABI/SDK；完整构建需要 ONNX Runtime 与
  OpenCV 开发库，host-only 后处理测试只需 g++。
- GTR 部署工程：eager 侧 Triton GLA 核已迁移（2× 提速），但 `torch.compile` 对
  GLA 段负收益、CUDA Graphs 与自研核原地写接口冲突（需 op 改造）；导出路线用
  纯 ONNX 降级（无需插件）；官方自研核 + `gla_plugin.cu` 插件路线需自行编译。
  TensorRT 引擎从展开纯图构建（无 plugin 节点）：上游 plugin 路线才是延迟
  路线（官方读数 det S fp16 b1 ≈ 2.3 ms），纯图引擎延迟应预期明显更高。
- 官方训练增广（mosaic / mixup / zoom-out / color-jitter）未移植；姿态官方配方为
  `precision="32-true"`（`use_amp: False`）。

## 离线与权重

- 蒸馏教师默认请求预训练权重（`teacher_pretrained=True`）；离线时传
  `teacher_pretrained=False`。
- 预训练权重清单按 commit + LFS sha256 全量钉住
  （`src/autovision/encoders/weight_hashes.py`）。
- 受限许可权重（DINOv2 CC-BY-NC 4.0 / DINOv3 License）默认不下载，需显式自证；
  用户自持库禁再分发 / 商用，见[用户自持权重](user_held_weights.md)。
