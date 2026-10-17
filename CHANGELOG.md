# 变更日志

本文件记录 AutoVision 的对外变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本语义遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)；发布流程与版本政策见
`docs/release_process.md`（站点页「发布流程」）。

`0.1.x` 为内部开发线（未对外发布，无变更记录）；本文件自 `0.2.0` 起对外。

## [Unreleased]

### 新增

- **数据闭环（P1）**：
  - 伪标注 / 自训练：`autovision.tasks.detection.shared.pseudo_label` +
    `scripts/self_train_detect.py`（教师 `.pt` → 伪标注 → 学生训练 → 标注 val 评测
    → 报告），CLI `pseudo-label`（逐类阈值、SAHI 可选、批量推理）。
  - 误差分析：`autovision.metrics.error_analysis` + CLI `analyze-errors`（在线 /
    离线两模式）：TP/FP/FN 匹配、FP 归因（duplicate / classification /
    localization / background）、按类别与 COCO 面积分档切片、top-FP/FN 画廊
    （`report.md` / `report.json`）。
  - 转换扩展：PASCAL VOC 解析器、YOLO-pose → COCO 关键点、YOLO-seg 多边形 →
    COCO 分割、COCO 合并（`convert-data from=voc | yolo-pose | yolo-seg | merge`）。
  - pose 增广：关键点感知 mosaic + mixup（默认关闭、步窗口可调），mosaic 几何与
    检测族共享同一实现。
- 检测 mAP 指标新增 `map_small` / `map_medium` / `map_large`（COCO 面积分档，
  此前固定 `"all"`；headline 数字不变）。

- **部署广度（P1）**：
  - ONNX 导出支持**动态 H/W**（`export_onnx(dynamic_spatial=True)`，含第二尺寸
    torch↔ONNXRuntime parity 门禁；`input_dynamic_shapes` 规格位被导出器消费），
    并要求 dynamo 导出器（legacy 路径会把尺寸烘成常量——fail-closed）。
  - **OpenVINO** 一等后端：`export_openvino`（ONNX→IR + 编译后对拍门禁 +
    metadata 边车）、`export out=model.xml`、benchmark 后端
    `openvino`（设备可切 CPU/GPU/AUTO，纳入 parity 门禁）。
  - **TensorRT int8**（QDQ 路线）：`export_tensorrt(precision="int8",
    calibration_data=...)`（QDQ ONNX 先过 int8↔fp32 门禁，再以 strongly-typed
    网络构建引擎），benchmark 的 `tensorrt` 后端支持 `precision="int8"` +
    `quantization={...}`（首层 fp32 配方用 `nodes_to_exclude`）。
- int8 QDQ 链与 TensorRT 的互操作修复：int32 常量 `DequantizeLinear` 折叠回
  fp32（TRT 11 解析器要求）；激活量化默认**对称**（`activation_symmetric`，
  非对称 zero-point 被 TRT 拒绝）。
- `export` extra 显式声明 `onnxscript`（dynamo 导出器的依赖，缺失时静默回退
  legacy）；新增 `openvino` extra。
- **int8 首层 fp32 配方自动建议**：`suggest_first_layer_nodes`（导出自
  `autovision.export`）枚举 fp32 图的前导 Conv 节点；TensorRT int8 构建失败
  的提示直接给出可粘贴的 `nodes_to_exclude` 配方，复现脚本
  `scripts/measure_new_backends.py --quantization-exclude auto` 一键套用；
  OpenVINO IR 支持 `precision="fp16"`（编译后对拍门禁照常）。

- **超参搜索（B2）**：`autovision.hpo`（`SearchSpace` 分布规格 + `run_search`
  随机 / 网格 / ASHA 驱动 + `task_trial` 复用训练入口与 Pydantic 覆盖面）与
  CLI `hpo`；产出 `report.md` + `trials.jsonl` + `best.json`，已完成试验目录
  自动复用（断点续跑），ASHA 逐 rung 以框架原生 resume 续训而非重训。
- **VOC 可直接训练（B3）**：`download → convert-data from=voc → train` 三步配方
  入 README；转换器保持 VOC 规范类名（排序 id）、`difficult → iscrowd` ignore
  区；端到端训练测试（CPU）在案。

- **推理侧能力（P2 战略项的一部分）**：
  - **ByteTrack 跟踪**（`autovision.tracking`，A3）：卡尔曼运动模型 + 三阶段
    IoU 关联（高分池融合检测分数 / 低分池 / 未确认轨迹），class-aware；官方
    门限齐备（新轨迹需 `track_thresh + 0.1` 且连续命中才发布，低分下限 0.1，
    首帧激活例外）；`predict source=... track=true` → JSONL 逐帧 `tracks`
    （`age` = 创建以来帧数）+ 渲染 id 标注，tracker 按源视频真实帧率折算
    丢失窗口。
  - **参考 serving**（`autovision.serving`，F2；`serving` extra）：FastAPI
    `/predict`（批处理按输入顺序、与评测同源前后处理）+ `/healthz` +
    Prometheus `/metrics`（请求按端点与状态、图片数、延迟 summary + 峰值；
    5xx 同样计数）；显式边界（无鉴权/TLS/编排，漂移检测等按产品触发）。
  - **数据版本化**（`autovision.data.versioning` + `data-snapshot` CLI）：内容
    寻址快照（根相对清单 sha256，`deep=true` 含内容哈希）+ 漂移报告
    （added/removed/resized/content_digest）+ 搬迁副本校验；spec 内相对路径按
    spec 文件目录解析，写入默认不覆盖（`overwrite=true`），deep 校验要求
    deep 快照（浅快照显式报错）。

### 修复

- **HF 骨干（DINOv2/DINOv3 系）的 ONNX 导出在 transformers ≥5 环境下全线失败**：
  首个 eager 前向会先安装 output-capturing hooks（在 `threading.Lock` 下），
  而 dynamo 无法 trace `with <lock>`，严格导出直接失败——现于 trace 前做一次
  warm-up 前向（数值透明）。同批修复：严格 trace 失败且空间轴为**静态**时回退
  legacy TorchScript 导出（此前不回退、直接抛 dynamo 内部错误——文档承诺的
  回退路径此前只覆盖第二步之后）；动态轴失败改为**可读的 AutoVisionError**
  （含"模型在输入尺寸上有 Python 级分支/守卫"的处置指引）。token→patch 网格的
  折叠改由**输入尺寸**推导（`H // patch`），动态 H/W 下不再对符号形状调
  `math.isqrt`：HF/ViT 线的 eager 报错与既有行为一致；timm 包装器在 patch size
  可发现时按输入推导网格（`dynamic_img_size` 类模型由"报错"变为可用，token 数
  与网格不匹配仍报错）。
- int8 首层 fp32 配方此前需手工枚举 fp32 图节点名（见上「自动建议」）。

- CLI 训练入口的显式设备串（`devices="cpu"` / `"cuda:1"`）此前在 9 个任务入口以
  `int(devices)` 崩溃——现按单设备解析（与 `engine.resolve_device` 语义一致）。
- `convert-data` 的文档示例此前无法执行（`out=` / `images=` / `annotations=` 与
  函数参数名不匹配、连字符键直接报错）——已修并加回归测试。
- 新版 CLI 键名统一支持 `--kebab-case`（自动映射到下划线参数名）。
- **步窗口增强在 `num_workers>0` 下静默不生效**（先存缺陷，检测 / 旋转框 / 姿态三族同存）：
  含步窗口调度的训练 loader 现以非持久 worker 构建，窗口边界跨越时把当前 step
  重新下发到 worker。

## [0.2.0] - 2026-10-16

首个对外发布版本（clean-provenance 发布线）。

### 新增

- **任务面 8 个**：检测、旋转框、实例 / 全景 / 语义分割、姿态、深度、分类；
  Python API 与 `autovision-cli`（20 条命令）共享同一套 Pydantic 严格校验。
- **检测谱系 `detr/*`**（39 个注册变体）：DINOv2 / DINOv3 ViT 与 ConvNeXt、
  HGNetV2 装配、GTR 解码器装配；档位谱系 `n` / `s` 冻结为公开档位。
- **GTR 多任务家族**（MIT，26 个官方权重，sha256 全钉住）：检测 / 实例分割 /
  语义分割 / 深度 / 旋转框 / 姿态（s/m/l/x，旋转框 s/x）；官方 EMA 权重直载，
  各任务加载与前向数值对拍（检测解码器谱系 bit-equal）。
- **姿态估计任务面**：COCO 关键点数据面、OKS 指标与官方损失（matcher / 损失 /
  CDN 与官方实现逐位一致）。
- **自监督预训练与蒸馏**：DistillationV3、SimCLR、DINO、DINOv2、DINOv3、
  DenseCL、DetCon B/S、训练中 kNN 探针。
- **运行时**：DDP（预训练 + 任务训练双引擎）、混合精度、原子 checkpoint +
  resume、自描述 checkpoint 格式、NaN 捕获、JSONL / TensorBoard / MLflow / W&B
  记录器、OTel（opt-in，默认关闭）。
- **导出与部署**：ONNX（fp32 / fp16，默认数值对拍门禁）、TensorRT（batch-only
  profile）、int8 QDQ 量化链、`export` CLI、benchmark 三后端（torch / ONNX /
  TensorRT）parity 门禁。
- **参考推理运行时** `autovision.inference`：视频文件 / RTSP / HTTP 流 + batch
  聚合、torch / ONNX / TensorRT 后端、与评测同源的前后处理、
  `predict --source` 输出逐帧 JSONL + 渲染视频。
- **数据层**：11 类数据集、检测增广全套（mosaic / mixup / copyblend /
  box-aware eraser，逐 step 窗口调度）、mmap 文件清单（内容指纹 / 原子发布 /
  碰撞守卫）。
- **权重供应链**：31 个 HuggingFace 资产（revision + sha256 钉住 + 许可门控）、
  GTR 26 个官方权重（MIT）、用户自持资产库（仓库外，不可再分发，见模型库页面）。
- **文档**：公开文档站点与模型库页面（每模型许可 / sha256 / 官方指标 / 加载
  代码）、协议化基准表 `docs/benchmarks.md`。


### 变更

- 版本号对齐 `0.2` 线：`0.1.0` → `0.2.0`；打包版本改为单一源
  （`src/autovision/_version.py`，pyproject 动态读取）。
- 文档树重构为站点导航结构（as-built 架构总览 + 模型库 + 基准 + 贡献 + 发布
  流程）；历史设计分册不再随公开树分发。

### 安全与合规

- 溯源与分层门禁（`scripts/check_provenance.py` / `scripts/check_layering.py`）
  在 CI 强制。
- 受限许可权重默认不下载（DINOv2 CC-BY-NC 4.0 / DINOv3 License 需显式自证
  接受条款后启用）。
- AGPL 组件（`ultralytics`）仅允许 extras 隔离加载：不进核心依赖、不在模块级
  import。
