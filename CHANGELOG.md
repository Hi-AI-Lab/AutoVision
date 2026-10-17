# 变更日志

本文件记录 AutoVision 的对外变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本语义遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)；发布流程与版本政策见
`docs/release_process.md`（站点页「发布流程」）。

`0.1.x` 为内部开发线（未对外发布，无变更记录）；本文件自 `0.2.0` 起对外。

## [Unreleased]

（下一版本开发中。）

## [0.3.0] - 2026-10-17

### 许可与兼容性

- **删除 DEIMv2 派生的 STA（2026-10-17）**：`SpatialPriorModule` 从代码库
  移除，`DETRVariantSpec` 不再携带 `use_sta` / `conv_inplane`；DINOv3 ViT
  变体保留并改用 no-STA adapter，DINOv3 ConvNeXt 与其他家族不变。发行物
  不再包含 DEIMv2 代码，许可阻塞解除。
- **旧 STA-era ViT 检查点不再兼容**：含 `backbone.sta.*` 的基线 `.pt` 与
  用户自持的 `dinov3-vitt16` / `dinov3-vitt16plus` / `dinov3-vits16` 会被
  加载器显式拒绝；需使用重新训练/导出的 no-STA 检查点，或改用 ConvNeXt
  资产。用户自持的 4 个 ConvNeXt 资产不受影响。

### 新增

- **许可与致谢文档**：README 新增 `致谢（Acknowledgements）`；
  新增 `THIRD_PARTY_LICENSES.md`（GTR MIT 全文）与
  `docs/provenance/LEDGER.md`；`NOTICE` 补齐 RF-DETR / PicoDet /
  ByteTrack / SAHI 归属，并记录 DEIMv2 派生 STA 的删除与兼容性变化。
- **发布依赖**：`export` extra 下限提升为 `numpy>=1.25`（onnx / onnxscript
  在导入时使用 `numpy.exceptions`，numpy 1.23/1.24 下会直接失败）；核心训练
  安装仍保持 `numpy>=1.23`，与 super-gradients 的互斥由既有 extras 冲突声明
  处理。

- **黄金路径 / 部署契约（roadmap phase 1–2 代码基础）**：
  - **任务级模型目录**：`autovision.registry.catalog` 汇总 RF-DETR N/S/M/L、
    GTR s/m/l/x、自研 DETR 谱系与 RT-DETR 的许可 / 权重 / AP / 可导出状态；
    `list-models --task detection --pretrained`、`model-info <key>` 可查。
  - **预训练检测句柄**：`autovision.detection(<key>, pretrained=True)` /
    `DetectionModel.from_pretrained` 惰性加载发布权重（RF-DETR 全链；GTR 在
    GLA/CUDA 可用时走官方 EMA 权重装配，否则给出可读安装/硬件指引）；
    `predict / val / export` 不再要求先训练。
  - **训练配方**：`recipe=finetune | coco-finetune | low-data` 或 YAML 路径，
    只覆盖 `model_args` / `transform_args` / `trainer_args`，显式参数优先。
  - **版本化后处理规格**：ONNX / OpenVINO / TensorRT 导出件旁写
    `<artifact>.postprocess.json`（schema_version=1；top-k、阈值、原图映射、
    resize/mean/std），附带 NumPy 参考解码器；分割 / 姿态 / 深度 / 分类导出
    不写 sidecar。
  - **端到端框级 parity**：`autovision.export.verify_end_to_end_parity`
    比较最终框 / 分数 / 类别（宽松匹配 + 像素容差 + missing/extra 计数），
    可直接输出 JSON 报告；不再只比张量求和。
  - **OpenVINO 运行时**：`InferenceEngine(backend="openvino")`、`run_stream`、
    serving 与 CLI `predict` 均支持 `.xml` IR（`openvino_device=CPU|GPU|AUTO`）。
  - 修复 legacy ONNX 导出回退时多传一个位置参数导致的
    `TypeError`（dynamo 不可用时的真正回退路径）。
  - 新增测试：模型目录、预训练句柄、配方（含 1 步真实训练）、后处理规格、
    框级 parity、OpenVINO 运行时接线。

- **数据体检与难例闭环（phase 3 代码基础）**：
  - **数据体检**：`autovision.data.quality` + CLI `data-check`：坏图、精确/近重复
    图、跨 split 泄漏、框越界/非法尺寸、原始类别 id、类别分布与 val 缺类；
    产出 `report.json` / `report.md` / `manifest.json`，`fail_on` 控制退出码。
  - **难例清单**：`analyze-errors` 额外写 `hard_examples.json`（逐图 TP/FP/FN、
    FP 归因、漏检面积）；`split=` 记录来源，采样器只消费 `train`，避免验证集
    泄漏进训练。
  - **难例采样**：`autovision.data.hard_examples` 的 manifest + 确定性
    `WeightedRandomSampler`；`train_object_detection(..., hard_examples=...)`
    按 `min(max_repeat, 1 + alpha × 归一化误差)` 加权。v1 单机；DDP 组合被
    显式拒绝（DistributedSampler 会覆盖权重）。
  - **多轮自训练**：`scripts/self_train_detect.py --rounds N --min-improvement`；
    逐轮产物（`rounds/r{i}/`）、学生提升后升级为下一轮教师、不升即停；
    顶层 `summary.json` / `summary.md` 记录 stop_reason 与最优模型。
    `--rounds 1` 保持原有单轮布局。
  - **长尾采样**：`autovision.data.samplers` 的 `class_balanced` /
    `repeat_factor`；`train_object_detection(..., long_tail_sampler=...)`
    接入训练，与 `hard_examples` 互斥，v1 单机。
  - **标注工具互通**：`autovision.data.interchange` 的 CVAT XML（bbox /
    polygon）与 Label Studio JSON（rectangle / polygon / keypoint-as-box）→
    COCO；`convert-data from=cvat | label-studio`，含 CLI 与 loader 往返测试。
  - 新增测试：数据体检（核心 + CLI 夹具）、难例 manifest/采样/误差分析输出/
    1 步真实训练接入、多轮自训练（真实 2 轮）、长尾采样权重/确定性/训练接入、
    CVAT / Label Studio 转换。

- **生态与企业能力（phase 4 代码基础）**：
  - **插件协议**：`autovision.plugins` 的 `autovision.plugins` entry-point
    发现、`PluginInfo` / `PLUGIN_API_VERSION`、kind 校验、失败收集 /
    strict 模式；`model_source` 自动接入编码器目录；CLI
    `plugins action=list|check`；`examples/plugin-demo` 可安装模板。
  - **企业合规报告**：`autovision.compliance` 汇总框架/依赖许可/模型卡+
    sha256/数据快照；CLI `compliance-report` 输出 `compliance.json` +
    `compliance.md`，支持 `strict` / `no_timestamp`。
  - **私有注册表**：`current` 指针、`rollback`、append-only `audit.jsonl`、
    模型卡/后处理边车随产物复制；部署门禁
    `Sha256Gate` / `ModelCardGate` / `LicenseGate` / `BenchmarkGate` /
    `ParityGate` / `DataSnapshotGate`；CLI
    `registry action=register|promote|rollback|current|list|audit`。
  - **LTS / SLA 文档**：`docs/lts_policy.md`、`docs/support_policy.md`
    （6 个月 LTS 节奏、12/6 个月支持窗、P1–P3 响应目标）。
  - **硬件伙伴包模板**：`deploy/packs/_template/` 的 `convert.py`（校验
    ONNX/spec、执行厂商命令）与 `parity.py`（目标端 JSONL 对拍）；一伙伴
    一包，不预研厂商矩阵。

- **谱系发布治理（P0）**：
  - `SPECTRUM` 档位状态改为三态 `planned / spec-frozen / released`；
    `detr/n`、`detr/s` 明确为 spec-frozen（规格冻结，预训练权重 / AP 待发布）。
  - 新增 `autovision.registry.spectrum_release` 发布门禁：`released` 必须有
    已发布资产、模型卡（许可 + sha256）、训练 seed、全量 COCO AP、延迟记录
    与存在的复现脚本。
  - 新增 `scripts/train_spectrum.py`、`scripts/eval_spectrum_coco.py` 骨架
    （支持 `--dry-run` 与 coco8 smoke 路径）。
  - 新增 RF-DETR 五档模型卡（官方 COCO 指标 + pinned sha256，
    `autovision.registry.model_cards`；Base 标记上游弃用）；基准表补
    RF-DETR 官方 T4 口径（N/S/M/L）。
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
  - **多路流运行时**：`MultiStreamRunner` / `run_multi_stream`（`predict
    sources=...`）：每路 reader 线程 + 单一共享 batcher、每路
    `predictions.jsonl` / `annotated.mp4`、断流指数退避重连、队列
    `drop_oldest|block` 背压、`summary.json` 每路帧数/丢弃/重连/错误。
    v1 不做跨路跟踪。
  - **PicoDet NMS 后处理规格**：`postprocess.json` 新增 `family="picodet"` +
    class-offset NMS；NumPy 参考解码与 torch 后处理逐框对拍（含重复框抑制）。
  - **int8 vs fp32 AP 报告**：`scripts/measure_int8_accuracy.py` +
    `autovision.export.int8_report`：同一 split/前后处理下导出 fp32/int8、
    跑全量 COCO AP、输出 `report.json`/`report.md`（delta、门禁、校准集、
    硬件）；`--max-ap-drop` 控制退出码。
  - **C++ ONNX Runtime 示例**：`examples/cpp/`（header-only decode + host
    test + ORT/OpenCV main + CMake）；host 测试由 g++ 真实编译运行，
    完整 runtime 构建依赖 ORT/OpenCV 开发库。
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

### 变更

- **PyPI 打包**：发行名改为 `autovision-framework`（import 名仍 `autovision`，
  CLI 仍 `autovision-cli`）；新增 PyPI 专用 README、PEP 639 许可元数据、
  sdist 的 GTR / MSDA CUDA-C++ 源清单，以及 tag 触发的 trusted-publishing
  CI job。`super-gradients` 与 `export` / `transformers` / `rfdetr` / `video`
  的 extras 互斥已显式声明。

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
