# AutoVision 对标 YOLO 生态：差距分析与完善路线

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 日期 | 2026-10-16 |
| 状态 | 内部分析稿（决策依据，非对外发布产物） |
| 输入 | 外部评审清单《类 YOLO 视觉框架成功要素》（六层：算法 / 工程 / 数据闭环 / 部署 / 生态 / 生产级） |
| 基础事实 | 全仓只读盘点 @ commit `3893ef2`（GTR multi-task S5b），盘点结论均可回溯到具体文件 |

> 后续执行设计（各阶段 API、数据结构、测试与验收）见
> [90 天路线设计与实施手册](roadmap_90d_design.md)。

---

## 0. 结论摘要

**已经建成、且同行难以复制的三样**（本框架真正的护城河）：

1. **合规与溯源工程**：Apache-2.0 + `scripts/check_provenance.py` / `check_layering.py` 双门禁 + 权重许可门控（CC-BY-NC / DINOv3 License 显式自证）+ clean-cut 发布手册。这正好对上 YOLO 清单里的痛点 #8（许可证风险、供应商锁定）——Ultralytics 的 AGPL 是大量商业项目无法直接使用的硬约束。
2. **数值一致性验证**：导出对拍门禁（ONNX fp32/fp16/int8、TensorRT parity）、GTR 六任务 bit-exact 权重加载、金标准快照（`tests/golden/`）。这是清单痛点 #2（部署碎片化、转换掉精度）的直接答案，且"默认开启、失败即拒绝"在开源项目里几乎看不到第二家。
3. **协议化基准**：`docs/benchmarks.md` 的不可变协议 + 只引官方数字 + 自家数字必附复现脚本 + 勘误节。透明基准（清单第 1 层要求）不是口号，是可审计的表格纪律。

**三处真缺口**（按影响排序）：

1. **规模谱系没打完**：检测谱系 n/s 已冻结、m/l/x 待定；GTR 检测 m/l/x 权重已入库但 model 变体只接了 S。这是"n/s/m/l/x 规模可调"轴唯一没打满的地方，也是补齐成本最低的地方。
2. **部署工具链离"多端"最远**：只有 ONNX/TensorRT 两个后端；无推理运行时（无视频/RTSP、无 serving、CLI predict 只吃 `.pt` 且 batch=1、动态形状仅 batch）；四个 GTR 任务模型的导出门禁覆盖缺失（默认 `verify=True` 直接报错）。
3. **生态采用漏斗几乎为零**：mkdocs 导航全部指向已删除文件（站点构建必挂）；无 CONTRIBUTING/CHANGELOG/issue 模板/发布节奏；无对外模型库页面；权重库一半不可再分发。

**战略立场**：不逐条复刻 Ultralytics。用户从 YOLO 迁移的真实驱动力是 AGPL 合规与转换掉精度，本框架恰好是对着这两个痛点建的。完善方向是强化"许可干净 + 可验证一致性"这条窄而硬的差异化，部署广度靠 ONNX 生态借力（见 §4）。

---

## 1. 现状快照

| 维度 | 现状 | 关键位置 |
|---|---|---|
| 代码规模 | 332 个源文件 / 约 5.9 万行；1600+ 个测试（盘点计 1710 个测试函数，含金标准快照） | `src/autovision/`、`tests/golden/` |
| 任务面 | 8 个：检测、旋转框、实例/全景/语义分割、姿态、深度、分类 | `src/autovision/tasks/` |
| 权重资产 | 67 个钉死检查点：31 HF（revision+sha256+许可门控）、26 GTR（MIT）、3 D-FINE（Apache）、7 用户自持（不可再分发） | `encoders/weight_hashes.py`、`registry/gtr_assets.py`、`registry/user_held_assets.py` |
| 导出与部署 | ONNX fp32/fp16 + int8 QDQ、TensorRT fp32/fp16，默认数值对拍门禁；benchmark 三后端 | `src/autovision/export/` |
| 数据与增广 | 11 类数据集；检测增广全套（mosaic/mixup/copyblend/box-aware eraser + step 窗口调度）；mmap 清单 + 指纹 | `src/autovision/data/`、`augment/detection/` |
| 运行时 | DDP、AMP、原子 checkpoint + resume、JSONL/TB/MLflow/W&B + OTel（opt-in）、NaN 捕获 | `runtime/`、`engine/`、`observability/`、`debug/` |
| 模型注册表 | 文件系统 dev/staging/prod + MLflow 后端 + model card | `registry/model_registry.py`、`mlflow_registry.py`、`model_card.py` |

---

## 2. 六层对照

### 2.1 算法层（精度 / 速度 / 规模可调）

**现状**

- 检测谱系 39 个变体；n/s 于 2026-10-15 冻结为公开档位，m/l/x 骨干线待定（README「状态与已知边界」）。
- GTR 任务族（分割/语义/深度/姿态/旋转框）已接 s/m/l/x（旋转框为 s/x），benchmarks.md 已读出 gtrpose s/m/l/x 四档 AP（70.05 / 74.04 / 74.70 / 75.48）。
- **GTR 检测家族只接 S**：`registry["gtr-s"]` / `registry["gtrvit-s"]`（`tasks/detection/detr/variants.py`），而 `gtr_{m,l,x}_coco` 权重已在资产库注册。
- benchmarks.md 自家行的 Params/GFLOPs 空白、TRT 延迟列标注「待 P3」；benchmark harness 只报 images/s 与延迟 p50/p90，无参数量/算力/功耗统计。
- 无 tracking 任务。

**缺口**

- 检测规模轴 m/l/x 悬而未决——"n/s/m/l/x"谱系唯一未打满的轴。
- 缺参数量/算力透明列（清单明确要求"在 mAP、FPS、参数量、功耗之间给出透明基准"）。

**动作**

- A1：检测 m/l/x 决策——接线 GTR 检测变体（权重、装配路径、对拍基建现成）或正式冻结方向并同步 README。
- A2：benchmark 增加参数量/FLOPs（可选功耗采样），回填 benchmarks.md 自家行。
- A3（P2）：tracker（ByteTrack/OC-SORT）作为推理侧能力接入。

### 2.2 工程层（开箱即用）

**现状**：统一 API/CLI + 同一套 Pydantic 校验、DDP/AMP/resume、日志三件套 + OTel、快速开始可直接跑。

**缺口**

- **无 `export` CLI 命令**：导出/量化仅 Python API 可及；`train-instance-segmentation` / `train-oriented-object-detection` / `train-panoptic-segmentation` 亦未注册为 CLI 命令（`src/autovision/cli/train_task/` 已实现）。
- 无超参搜索（optuna/ray/hyperopt 零依赖）。
- 自定义数据集无插件钩子（固定 Pydantic 类）；VOC 只登记下载、无解析器；`convert-data` 仅支持检测 bbox；albumentations 适配未接（README 已承认）。
- 版本治理：`0.1.0`（`pyproject.toml` / `src/autovision/_version.py`）与 tag `v0.2-clean` 不一致；无成文 semver/弃用政策。

**动作**

- B1（P0）：`export` CLI（`autovision-cli export model=... format=onnx|engine`），顺带补齐三个未注册的 train 命令。
- B2（P1）：HPO——随机/网格搜索（复用现有 config override 机制）+ ASHA（Ray Tune 作可选 extra）。
- B3（P1）：VOC 解析器 + `convert-data` 扩展到关键点/分割。
- B4（P0）：版本号对齐 + CHANGELOG（Keep a Changelog）+ 弃用政策落仓库。

### 2.3 数据闭环

**现状（比预期强）**：11 类数据集（COCO det/instseg/kp、YOLO det/OBB、ImageFolder、SSL 目录/文件列表、两视图、配对分割、panoptic、depth）；检测增广全套（`StepScheduledCompose` + mosaic/mixup/copyblend/box-aware eraser，逐 step 窗口调度）；mmap 清单 + 内容指纹（缓存键）；4 数据集下载注册表；coco8 合成冒烟集；抽帧 CLI。

**缺口**

- 无自动标注/伪标注、无难例挖掘/主动学习、无标注工具对接、无数据版本化。
- VOC 只下载不解析；`convert-data` 仅 bbox；pose 训练增广未移植（仅 resize+hflip，官方 mosaic/mixup 明确未移植）。

**动作**

- C1（P1）：伪标注/自训练闭环——SAHI 切片推理 + 强检测器 + 现有对拍基建，直接对向"数据少、标注贵、长尾差"。
- C2（P1）：误差分析报告——per-class AP 已有，补 top-FP/FN 可视化画廊 + 按目标尺寸/类别切片（COCO per-area 目前固定 `"all"`，值得放开）。
- C3：数据版本化——基于现有指纹做内容寻址快照。
- C4（P1）：pose 增广移植（训练复现的前提）。

### 2.4 部署工具链（最大缺口）

**现状（一致性这条比 YOLO 严）**

- ONNX fp32/fp16，默认对拍门禁（atol 5e-3 / rtol 1e-1）；int8 QDQ 独立链 + 校准读取器（仅单输入模型）；TensorRT fp32/fp16。
- benchmark 三后端（torch/onnx/tensorrt）+ parity 门禁。
- 动态形状仅 batch（1..128）；TRT 显式拒绝动态 H/W。
- **四个 GTR 任务模型的 `verify_onnx_export_outputs` 已补齐（P0 已实现，见 §3.1）**；同轮排查发现更硬的问题：GTR GLA 前向的 `chunk_gla`/融合 RMSNorm 核为自定义 CUDA op，无 ONNX 降级——`gtr/*` 与 `detr/gtrvit-*` 全族**当前不可导出**，导出入口现以明确报错拒绝（`GTRViTBackbone.export_blocker`）。

**缺口**

- 后端面：OpenVINO / TFLite / NCNN / RKNN / CoreML / 昇腾转换 / 地平线全无（昇腾仅 `torch_npu` 设备放置，`runtime/accelerator.py`）。
- 无 serving（HTTP/gRPC）；无视频/RTSP 推理（只有抽帧 CLI）；CLI predict 仅 `.pt`、每次前向 batch=1（`cli/model_handle.py`）。
- TRT INT8 校准缺失；无 QAT/FP8；无 C++/移动端示例。

**动作**

- D0（P0）：GTR 四模型补 `verify_onnx_export_outputs`——旗舰模型导出必须先过门禁。
- D1（P0）：参考推理运行时——视频/RTSP 循环 + batch + ONNX/TRT 路径 + 与评测同源的前后处理 + `predict --source`（demo 与客户 POC 的刚需，把"前后处理一致"从承诺变成产品）。
- D2（P1）：ONNX 动态 H/W（`ModelInputSpec.input_dynamic_shapes` 字段已存在但导出器未用）。
- D3（P1）：OpenVINO 作后端 #3（CPU 装机量大、经 ONNX 路线成本低）；边缘二选一 RKNN 或 TFLite（按目标客户定）。
- D4（P1）：TRT INT8 校准（校准读取器已存在，只差接 TRT builder）。
- D5：其余后端走"ONNX hub + 转换文档/脚本"，C++ 只做参考 runtime 的最小示例。

### 2.5 生态与社区

**现状**：67 个钉死权重 + `list-models` + model card；NOTICE/溯源纪律；clean-cut 发布手册与孤儿分支就绪（远程待配置）。

**缺口**

- **mkdocs 导航全部指向已删除文件**（`docs/design/*`、`docs/provenance/*`、`docs/index.md`、`migration-0.2.md` 在工作区已删除且未提交）→ 站点构建必挂。
- 无 CONTRIBUTING / CODE_OF_CONDUCT / CHANGELOG / issue·PR 模板 / 发布流程。
- 无对外模型库页面（model zoo + 结果表是 YOLO 采用漏斗的核心）。
- 权重库一半不可再分发（DINOv2 CC-BY-NC / DINOv3 门控 / 7 自持），外部用户"开箱即用"体验受限。

**动作**

- E1（P0）：发布就绪包——docs 导航重写 + 版本对齐 + CHANGELOG + CONTRIBUTING + issue 模板。
- E2（P0）：模型 zoo 页面（每模型：许可 / sha256 / AP / 延迟 / 一行加载代码），以 GTR MIT 全族为旗舰；自有训练产物发布 HuggingFace（已有 HF 供应链经验）。
- E3：公开 API 稳定性白皮书（企业采用的前提）。

### 2.6 生产级能力

**现状**：训练侧可观测齐全（JSONL/TB/MLflow/W&B + OTel opt-in + NaN 捕获）；原子 checkpoint + resume；注册表已有 dev/staging/prod + sha256 promote（回滚语义的种子）；model card。

**缺口**：无推理侧监控/漂移检测/A-B/金丝雀（无 serving 则无推理监控）；无 FSDP/DeepSpeed；无告警。

**动作**

- F1：canary/回滚 SOP 文档化（注册表能力已在，文档化成本极低，企业交付叙事需要）。
- F2（P2）：serving 参考实现（FastAPI + 批处理 + 指标端点）作为推理监控前提；若定位仍是"训练框架"，在 README 明确写为边界。
- F3：FSDP/DeepSpeed 按大模型需求排期。

---

## 3. 优先级路线

| 优先级 | 条目 | 内容 | 退出门禁 |
|---|---|---|---|
| **P0** | 导出面闭环 | B1 export CLI；D0 GTR 四模型 verify hook + GLA 导出阻断显式化；A1 检测规模轴决策 | ✅ `autovision-cli export` 可用；GTR 全族 verify 钩子就位、导出以明确报错拒绝（「verify=True 导出通过」重定性为 G2 前置）；✅ m/l/x 方向落文档（timm ConvNeXt 线） |
| **P0** | 参考推理运行时 | D1 视频/RTSP/batch + ONNX/TRT + 同源前后处理 | ✅ 视频源 + 活网络流（MJPEG/HTTP，与 RTSP 同路径）端到端出画面 + JSONL；✅ torch/ONNX 路径逐帧一致（盒子差 ~1e-5 px）；RTSP 传输层现场验证待相机环境 |
| **P0** | 透明基准补全 | A2 参数量/FLOPs 工具 + 回填 benchmarks.md（含 TRT b1/b4 延迟） | ✅ `measure_model_cost`（params/MACs/FLOPs，benchmark 报告携带）+ ✅ benchmarks.md §2.1 新表（成本 + TRT-fp16 b1/b4，GTR-GLA 行以阻断原因入表）；✅ 结果表禁空列（§1.0 待训练块）；✅ 数字全部配套复现脚本（含补写的 `scripts/eval_gtr_pose_ap.py`、`scripts/measure_spectrum_benchmarks.py`） |
| **P0** | 发布就绪包 | E1 docs 导航/版本/CHANGELOG/CONTRIBUTING；E2 model zoo 页 | ✅ `mkdocs build --strict` 零警告（CI docs 任务 + 结构契约测试）；✅ 公开树断链检查全绿（相对链接 + 反引号仓库路径，`scripts/check_docs_links.py`）；✅ 版本号单一源 `0.2.0`（pyproject 动态读取）+ 发布 tag `vX.Y.Z` 一致性测试在案（打 tag 时强制）；模型库页与注册表零漂移（`gen_model_zoo.py --check`） |
| **P1** | 部署广度 | D2 动态 H/W；D3 OpenVINO + RKNN/TFLite 二选一；D4 TRT INT8 | ✅ D2 动态 H/W（导出 + 第二尺寸 parity 门禁 + 多尺寸运行，见 §3.6）；✅ D3 OpenVINO 过 parity + 出数（10.18ms@64px CPU，见 §3.6）；⏳ 边缘槽 TFLite 方向定案、工具链阻断已记录（parity/读数未产出，见 §3.6）；✅ D4 TRT int8 引擎过 parity（int8 容差）+ 出数（1.60ms@640 b1） |
| **P1** | 数据闭环 | C1 伪标注闭环；C2 误差分析报告；C3/C4 转换与 pose 增广 | ✅ 端到端自训练脚本产出可评测模型（`scripts/self_train_detect.py` + 报告；见 §3.5）；✅ 误差报告出 top-FP/FN 画廊（`report.md` 内嵌链接全部可解析，测试锁定）；C3/C4 见 §3.5 |
| **P1** | 工程欠账 | B2 HPO；B3 VOC/多任务转换 | ✅ 一次随机搜索出报告（6 个真实试验 + `report.md`/`trials.jsonl`/`best.json`，最优 trial map_50 0.75，见 §3.7）；✅ VOC 可直接训练（`download → convert-data from=voc → train` 配方 + 端到端训练测试 + 真实读数，见 §3.7）；多任务转换（关键点 / 分割）见 §3.5 C3 |
| **P2** | 战略项 | F2 serving + 推理监控；A3 tracking；数据版本化；昇腾/地平线按客户需求 | 按产品/客户触发（已交付参考实现：✅ ByteTrack 跟踪接入推理运行时；✅ 参考 serving（FastAPI + `/metrics`，边界明示）；✅ 数据版本化（内容寻址快照 + 漂移报告）；昇腾/地平线：按客户触发，见 §3.8 决策记录） |

### 3.1 P0「导出面闭环」执行记录（2026-10-16）

**已交付**（全量测试 1725 通过 / 14 跳过 / 0 失败；ruff / 溯源 / 分层门禁全绿）：

1. **B1 export CLI**：`autovision-cli export model=<导出件.pt> out=<file.onnx|file.engine|file.trt> [precision=...]`（`src/autovision/cli/export.py`，走 `load_task_model` + 网络自身的导出 mixin，参数校验与 Python API 同源）。同轮补齐三个此前仅 Python API 的训练命令：`train-oriented-object-detection`、`train-instance-segmentation`、`train-panoptic-segmentation`（CLI 共 20 条命令）。新测试 `tests/test_cli_export.py` 走真实命令，默认 `verify=True` 对拍门禁在导出管线内执行。
2. **D0 GTR 导出验证面**：
   - 四个 GTR 任务模型（pose/depth/semseg/OBB）补齐 `verify_onnx_export_outputs`（输出配对：pose=logits+keypoints、depth=单深度图、semseg=单 logits、OBB=logits+boxes）。
   - **重定性（实测发现）**：GTR GLA 前向三条路径全部不可导出——CPU 前向直接失败（Triton 核需 CUDA）；CUDA fp32 走 fla Triton 核，`torch.export` 无法追踪；CUDA fp16 + deploy 走自研核，`rmsnorm_gated` 为未包装的 pybind 调用，同样不可追踪；且即使追踪通过，`chunk_gla` 系列自定义 op 也无 ONNX 降级（TRT 需官方 `gla_plugin.cu`）。因此 **「GTR 全族 verify=True 导出通过」在 P0 不可达**，上调为 G2 部署工程的前置项。
   - 交付形态改为：`GTRViTBackbone` 声明 `export_blocker`，导出 mixin 扫描模块树后在入口以明确 `AutoVisionError` 拒绝——覆盖 `gtr/*` 四任务模型与 `detr/gtrvit-*`（含未来任意组合，无需逐模型声明），替代不可读的 tracer 栈回溯。verify 钩子随 G2 落地即生效。新测试 `tests/test_gtr_export_gates.py`。
3. **A1 检测规模轴决策**：m/l/x 骨干线定为 **timm ConvNeXt（tiny→m / small→l / base→x，`timm_cnn` 家族）**，决策标准 = 可导出（纯 CNN 无自定义核）+ 许可干净；GTR GLA 线不承担 m/l/x（导出阻断；其角色保持 `gtr-s`/`gtrvit-s` 组件档）。落地位置：`variants.py` 的 `SPECTRUM` 注释（机器可读决策面）+ README「状态与已知边界」。变体接线与权重为后续实现项。
4. 顺带修复：`tests/test_deployment.py` 的 SAHI 测试桩缺 `_module_device`（仓库既有失败，与本项无关）。

**遗留 → 下一步**：① G2 GLA 导出工程（自定义 op 包装 + ONNX 降级/插件）——完成后四个 verify 钩子自动生效；② timm ConvNeXt m/l/x 线实现；③ benchmarks.md 自家行补全仍为独立 P0 项（本项未动）。

### 3.2 P0「参考推理运行时」执行记录（2026-10-16）

**已交付**（新测试 11 项全绿 + CLI 端到端烟测；ruff / 溯源 / 分层门禁全绿）：

1. **核心运行时 `autovision.inference`**（新包，分层表 rank 1，与 engine 同级）：
   - `InferenceEngine(model, backend="torch"|"onnx"|"tensorrt", artifact=...)`——**同源前后处理**的结构化实现：网络自己的 `preprocess_image`/`preprocess_batch`/`postprocess` 原样复用，只替换 forward（与导出对拍门禁同一契约；torch 后端镜像导出侧 reference 语义：`eval` + `deploy`）。ONNX 后端按图的输入 dtype 供数（fp16 图无 CUDA EP 时显式报错），输出统一转 fp32 供 postprocess。
   - `run_stream(source, out_dir, ...)`——视频文件/RTSP 循环 + batch 聚合 + `predictions.jsonl`（逐帧、原始像素、大张量自动摘要）+ `annotated.mp4`（惰性创建，兼容上报 0×0 的活流）。
   - 视频 I/O 用 OpenCV（`video` extra，新声明；FFmpeg 后端，文件/RTSP/HTTP 同一 reader）+ 打开/读取超时（默认 10s/5s，网络源尽力而为）。
2. **复用优化**：`TensorRTEngineRunner` 从 benchmark 内联实现提炼到 `export/tensorrt_helpers.py`（含输入 dtype 自适应——修掉 fp16 引擎潜在供数 dtype 隐患），benchmark 与推理运行时共用一份引擎执行代码。
3. **CLI**：`predict` 增加 `source=` 分发（视频/RTSP + `backend=`/`artifact=`/`batch_size=`/`render=`/`max_frames=`），`images=` 路径不变。
4. **验证读数**：torch vs ONNX(fp32) 同一批帧逐帧对拍——标签全等、盒子最大差 **~1.5e-5 px**、分数最大差 **~1.5e-8**；批不变性（bs1 vs bs4）盒子差 0、分数差 ~2e-8（CPU 归约顺序）；CLI 端到端 6 帧视频 → JSONL 6 行 + `annotated.mp4` 6 帧；**活网络流端到端**——进程内 MJPEG-over-HTTP 源（纯 stdlib，与 RTSP 同一 URL/网络流代码路径）→ 逐帧 JSONL + 渲染视频全通。

**遗留 → 下一步**：RTSP 传输层现场验证（本环境无可用的 RTSP 服务端/ffmpeg 二进制；URL 网络流路径已由活 MJPEG 源端到端覆盖、不可达 URL 错误路径已锁测试，建议在有真实相机/媒体服务器的环境补一次现场读数）；TRT 后端（当时本机无 tensorrt 包未执行——其执行代码与 TRT 11 适配在 §3.3 落地并实测）。

### 3.3 P0「透明基准补全」执行记录（2026-10-16）

**已交付**（新测试 23 项全绿；ruff / 溯源 / 分层 / 文档链接门禁全绿）：

1. **A2 成本工具**：`autovision.export.model_cost`——`measure_model_cost` / `count_parameters` / `ModelCost`（params、trainable、MACs、FLOPs、输入形状）；计数用 PyTorch `FlopCounterMode`（零新依赖），**MAC = 2 FLOPs 的 torch 口径与官方表交叉验证成立**：`detr/gtrvit-s` 读出 33.9 GFLOPs / 12.12M，GTR 官方 GTR-S 为 33.8 / 12.1。`benchmark()` 报告（JSON + markdown）自动携带参数量/FLOPs（部署前快照——deploy 融合会改变计数，见 §3.3 注）；`autovision.measure_model_cost` 顶层导出。
2. **benchmarks.md**：新增 **§2.1 模型成本与 TensorRT 延迟**表（5 模型 × Params/GFLOPs/TRT-fp16 b1·b4 engine 与全链），`detr/gtrvit-s` 的 TRT 行以阻断原因入表（GLA 核无 ONNX 降级）；§1 结果表移除无读数行（detr/n、detr/s 移入 §1.0 待训练块，结果表禁空列）；§2 增加口径标定补记；勘误节记 5 条。§3 对手表交叉验证：本表 FLOPs 口径与官方一致。
3. **实测读数**（RTX 4090D / torch 2.14.1+cu130 / TRT 11.4）：
   - **TRT-fp16 engine-only p50**：detr/n 2.70 / 3.46ms（b1/b4），detr/s 2.83 / 3.54，detr/gtr-s 2.73 / 3.43，rtdetr/resnet18 1.92 / 3.10。
   - **TRT-fp16 全链 p50**（预处理 → H2D → engine，§1 协议口径）：11.3–14.4ms（b1）/ 19.7–23.6ms（b4）——**CPU 预处理（1280×720→640²）在 b1 链路占 ~80%**，预处理下沉是与引擎优化同等重要的部署项（已写入 §2.1 解读）。
   - **torch 复现标定**：§2 口径确认为未 deploy 的 eager forward（detr/n 19.5 vs 表内 19.3 ✓；gtr-s/gtrvit-s 偏高 2.3–5.0ms，原脚本未入库、残余差不可考，入勘误）。
4. **复现脚本**（benchmarks.md 引用自此全部可解析）：`scripts/measure_spectrum_benchmarks.py`（成本 + torch/TRT 延迟一体，引擎可缓存复用）；`scripts/eval_gtr_pose_ap.py`（§1 gtrpose 四行的官方协议 AP 复现——数据面 → 变换 → 网络 → 官方后处理 → pycocotools；已用官方 `gtrpose_s_coco` 权重 + 合成 keypoints 数据集做机制验证，全量 COCO val2017 运行需在有数据集的环境执行）。
5. **顺带修复（真实缺陷，TRT 11 + CUDA 模型导出路径）**：
   - TRT 11 移除了 `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` 与 `BuilderFlag.FP16`/精度约束 flags（以及 `IBuilder.platform_has_fast_fp16`）——原实现按 TRT 10.x API 硬取属性，在新版直接崩溃；现为版本容错：缺失即跳过，**fp16 引擎改经 fp16-ONNX + strongly-typed 网络路线构建**（与旧 builder-flag 路线语义等价，FP32 黑名单继续保护溢出敏感算子）。版本容错测试用假对象覆盖 TRT 10/11 两种 API 形状。
   - **ONNX 导出在 CUDA 驻留模型上必炸**（示例输入建在 CPU，与模型设备不匹配——训练后导出的常规场景）：示例输入现跟随模型设备；CUDA 门控的真机导出测试在案。
   - `export` CLI/benchmark 顺带受益（同一条导出链）。

**遗留 → 下一步**：§1.0 的 detr/n、detr/s 训练读数（档位训练 + 权重入库为独立切片）；`scripts/eval_gtr_pose_ap.py` 的全量 COCO val2017 现场复跑；TRT 引擎在 CI 环境不可复现（需 GPU+TRT，读数已附脚本与硬件登记）。

### 3.4 P0「发布就绪包」执行记录（2026-10-16）

**已交付**（`tests/test_infrastructure.py` 重写 + `tests/test_model_zoo.py` 新增，25 项：24 过 / 1 条件跳过；ruff（src/tests/scripts）/ 溯源 / 分层 / 断链 / `mkdocs build --strict` 全绿）：

1. **E1 docs 导航重写**：
   - `mkdocs.yml` 按新树重写：导航 7 页（首页 / 模型库 / 基准 / 架构总览 / 贡献指南 / 发布流程 / 变更日志），material 主题 + `zh`，pymdownx 包含扩展；契约写在配置头部（导航 ⊆ docs、docs ⊆ 导航 ∪ 排除、strict 零警告、新公开页必须注册）。
   - 新 `docs/index.md`（站点首页：三条差异化主线 + 快速开始 + 能力速览 + 导航表）。
   - `docs/framework_design.md` 从「11 分册索引（全部指向已删除文件）」重写为 **as-built 架构总览**：分层契约（rank 表 + 两条硬禁 + 债务表语义）、模块地图（20 个顶层包/模块，以代码路径为准）、任务面、四类权重资产模型、门禁表、数值一致性承诺。历史设计分册不再随公开树分发。
   - `CONTRIBUTING.md` / `CHANGELOG.md` 在仓库根（GitHub 面），站点页 `docs/contributing.md` / `docs/changelog.md` 经 `pymdownx.snippets` 一行包含——单一来源，无内容复制。
   - **内部稿处置决定**：`docs/strategy/gap-analysis.md` 自带「内部分析稿（非对外发布产物）」状态，故以 `exclude_docs` 排除出公开站点（仓库内保留并持续维护）；对应测试锁定「不在导航、被显式排除、排除项不陈旧」。
2. **E1 版本对齐（单一源）**：
   - 打包版本改为动态读取（`[tool.setuptools.dynamic] version = {attr = "autovision._version.__version__"}`），`_version.py` 升 **0.2.0**——版本号从此只有一个书写位置（打包 / `autovision.__version__` / 发布检查同源）。实测 editable 安装后 `importlib.metadata.version("autovision") == autovision.__version__ == "0.2.0"`。
   - 新增 **tag 一致性测试**：HEAD 携带 `vX.Y.Z` 发布 tag 时，tag 必须等于 `v{__version__}`（未打 tag 的开发提交自动跳过；发布清单要求打 tag 后执行）。历史 tag `v0.2-clean` 保留原义（0.2 线干净谱系快照）。
3. **E1 CHANGELOG / CONTRIBUTING / 发布流程 / 模板**：
   - `CHANGELOG.md`（Keep a Changelog 1.1.0；声明 0.1.x 内部线、`[0.2.0] - 2026-10-16` 首次对外段、`[Unreleased]` 常驻）。
   - `CONTRIBUTING.md`：环境、**提交前门禁全表**（与 CI 逐条一致）、测试规范（对拍优先 / 金标准重录 / 门控变量）、文档规范（导航注册 / 断链 / 数字纪律）、许可与溯源红线（AGPL 仅 extras 隔离、权重 revision+sha256+许可三元组、二进制不进仓、NOTICE 登记）、提交与 PR 流程。
   - `docs/release_process.md`：SemVer 政策、版本单一源、弃用政策（先警示、至少跨一个 MINOR 移除）、6 步发布清单（含发布快照分支 + tag）、权重发布政策（HuggingFace 发布核对项）。
   - `.github/ISSUE_TEMPLATE/bug_report.yml` / `feature_request.yml`（含「不贴受限权重直链」合规项与许可预检）+ `.github/PULL_REQUEST_TEMPLATE.md`（门禁复选 + 数值证据要求 + 许可清单）。
4. **E2 模型库页**：
   - `scripts/gen_model_zoo.py` 从注册表**机器生成** `docs/model_zoo.md`（223 行）：GTR 全族 26 资产表（任务 / 变体 / 资产名 / 官方指标 / 官方延迟 / sha256）＋检测档位谱系（直读 `SPECTRUM`）＋HF 权重清单 31（按许可分组 + 接入键 + revision + sha256）＋可下载 warm-start 资产 3 ＋用户自持库 7 ＋门控环境变量表；附三类加载代码（`gtr/*` 任务族 `model_args.checkpoint` 直载；检测两线装配 `warm_start` / `backbone_weights`；骨干与开放词表接线）。
   - **加载写法实机验证**（RTX 4090D，本机权重库）：`gtr/*` 五任务族 `model_args={"checkpoint": "<资产名>"}` 全过（pose 走完 2 步训练 + OKS 验证；semseg / depth / seg / obb 构造级权重加载通过）；检测两线装配 `decoder_coverage=1.0`、`skipped_shape=0`，`detr/gtr-s` 与 `detr/gtrvit-s`（骨干 + encoder + decoder 全权重）各自 2 步训练闭环通过。
   - 防漂移与自证：`--check` 字节级比对 + `tests/test_model_zoo.py`（漂移 / 资产完备 / 指标列 / 许可与 revision 逐行核对）；生成器内置断言（官方延迟表覆盖全部检测资产、任务组与许可族展示覆盖、HF 清单每项必须有策展角色、开放词表文本塔必须 sha256 钉住）。
   - 口径决策：sha256 显示前 12 位（完整值以注册表为准）；延迟列只填官方发表值（GTR 检测档），自家读数仍归 `docs/benchmarks.md` 协议表。
5. **验收门禁（新增，机械化）**：
   - `scripts/check_docs_links.py`：公开树两类断链检查（Markdown 相对链接；反引号仓库路径按「仓库根 → `src/autovision/` 包根」两级解析、允行号后缀），fenced 代码块跳过链接检查，行内代码与 HTML 注释不参与链接扫描；行级豁免标记 `<!-- docpath-ok -->`（勘误类历史引用）。
   - GitHub / GitLab 两个 CI 新增 docs 任务（断链 + `mkdocs build --strict`）；`v*` tag 推送触发套件（tag ↔ 版本一致性检查在打 tag 的提交上执行）；lint 门禁扩到 `scripts/`；`pyproject.toml` 新增 `docs` extra（mkdocs-material）。
   - `tests/test_infrastructure.py` 重写为两组契约：站点结构（导航 ⊆ docs、docs ⊆ 导航 ∪ 排除（含 gitignore 风格模式语义）、strategy 不进站、strict 构建、链接检查器自测）与发布包（版本单源、CHANGELOG 跟随版本、tag 一致、贡献入口、模板可解析）。
6. **顺带修复**：`docs/benchmarks.md` 两处失效脚本引用（`measure_spectrum_latency_torch.py` → `measure_spectrum_benchmarks.py`；smoke 脚本归档改写为测试落点）按「只增不改」协议登记勘误节（2026-10-16 两行，数字未变）。

**遗留 → 下一步**：
- ① `uv.lock` 陈旧为**先存问题**（与本次改动无关）：`uv lock` 因 `super-gradients` × `transformers` 的 protobuf 版本互斥未列入 `[tool.uv] conflicts` 而报「不可满足」；锁文件亦缺 `video` / `docs` extra。修法：补冲突声明后重生成，或正式改以 pip 为准（CONTRIBUTING 已标注当前按 pip）。
- ② 站点 `site_url` 待对外仓库 URL 确定后设置（当前 sitemap 为空是未设 URL 的直接结果，不影响构建与验收）。
- ③ `docs/benchmarks.md` §1.0 / §2 的空列属 A2 在途工作（本项未动其数字）。
- ④ 对外发布动作：按 `docs/release_process.md` 清单执行；打 `v0.2.0` 后运行 `pytest tests/test_infrastructure.py -k release_tag` 验证 tag 与版本一致。

### 3.5 P1「数据闭环」执行记录（2026-10-16）

**已交付**（新测试 58 项：`test_pseudo_label` 12 + `test_error_analysis` 18 + `test_self_train` 2 + `test_convert_extended` 14 + `test_pose_augment` 12；ruff / 溯源 / 分层 / 断链全绿；全量套件 1823 过 / 16 跳过 / 0 失败）：

1. **C1 伪标注 / 自训练闭环**（门禁：端到端脚本产出可评测模型 ✅）：
   - 核心 `tasks/detection/shared/pseudo_label.py`：`generate_pseudo_labels`——教师检测模型（batch 推理，可选 SAHI 切片）→ COCO 实例标注 json（可直接作为训练集消费）；类别映射沿用评估器约定（`sorted(categories)` 内部序 → 原始 id）、支持**逐类阈值**（长尾：稀有类提高置信门槛）、SAHI 下取最低活跃阈值传给内部过滤再逐类判定；产出统计（逐类计数 + 分数 min/mean/max）随 `PseudoLabelResult` 返回。
   - `data/convert.py` 增 `merge_coco`（标注集合并：id 重编 + 类别按名联合，标注 + 伪标注半监督入口）。
   - **门禁脚本 `scripts/self_train_detect.py`**：教师 `.pt` → 伪标注（可选与标注集合并，跨根路径自动绝对化）→ 学生训练 → 在标注 val 上评测 → `report.json`（教师/阈值/伪标注统计/val 读数/导出件路径）。测试 `tests/test_self_train.py` 跑**真实脚本**（CPU，2 步学生）断言 `report.json` 的 val 读数与导出模型存在。
   - CLI：`autovision-cli pseudo-label model= images= out= [data=] [threshold=] [sahi=true]`（22 条命令）。
2. **C2 误差分析**（门禁：误差报告出 top-FP/FN 画廊 ✅）：
   - 核心 `metrics/error_analysis.py`：逐类贪心匹配（COCO 风格，crowd GT 按 ignore 区处理）→ TP/FP/FN + **FP 归因**（duplicate / classification / localization / background，TIDE 风格互斥）；切片：逐类 + COCO 面积三档（small <32² / medium / large >96² px²）；**画廊**：全局 top-k（FP 按数量、分数破平；FN 按数量、漏检面积破平）+ 逐类最差样本，渲染 TP 绿 / FP 红（归因 + 分数）/ FN 橙 MISS 叠加图；产出 `report.json`（逐条匹配记录）+ `report.md`（汇总 / 逐类 / 面积表 + 画廊内嵌，链接经测试验证全部可解析）。
   - `COCOMAPEvaluator` 放开面积切片：`map_small/map_medium/map_large` 三个新键（此前固定 `"all"`；headline 数字不变，空评路径同步给键）。
   - CLI：`autovision-cli analyze-errors model= data= out= [iou=] [top-k=]`（在线）或 `predictions= annotations= [images=]`（离线，消费 `predict` 的 predictions.json）。
3. **C3 转换扩展**：
   - `data/convert.py` 新增 `voc_to_coco`（PASCAL VOC 树解析：`Annotations/*.xml` + `ImageSets/Main` 分片过滤；1-based inclusive → 0-based xywh 无损换算；`difficult` → `iscrowd=1` ignore 区并保留原标志）、`yolo_pose_to_coco`（Ultralytics YOLO-pose → COCO person_keypoints，`num_keypoints` 按可见点计数，与 `CocoKeypointDataset` 的官方过滤口径一致——测试经关键点数据集**往返加载**验证）、`yolo_seg_to_coco`（YOLO-seg 多边形 → COCO 分割：绝对多边形 + 边界框 + **鞋带公式**面积）。
   - `convert-data` 扩到 `from=voc | yolo-pose | yolo-seg | merge`；键名归一（`--kebab-case` → 下划线）与 `out=` / `images=` / `annotations=` 别名。
   - 顺带修复**先存缺陷**：`convert-data` 的文档示例从未可用（`out=`/`images=` 与函数参数名不匹配、连字符键直接 TypeError）——已修并加回归测试锁定示例契约。
4. **C4 pose 增广移植**：
   - 几何抽取：`augment/mosaic_geometry.py`（检测侧三件套提为共享模块 + 新增 `apply_affine_to_keypoints`），检测变换改为引用共享实现（行为零变更，74 项既有增广测试全绿）。
   - `tasks/pose_estimation/gtr/transforms.py`：**关键点感知 mosaic**（2×2 画布 + 滚动缓存 + 单一整画布仿射同时作用于框与关键点；画出画布的关键点置 v=0、框退化则连带人员整体丢弃——行对齐契约保持）与 **mixup**（当前帧 beta∈[0.45,0.55] 混合 + 标注合并）；两者默认关闭（`None`）——默认路径与既有实现逐位一致（既有 pose 测试全绿）；步窗口 `*_step_start/stop`（start 含、stop 不含）经引擎 step-hook 下发，窗口边界跨越触发 loader 重置并以当前 step 重新下发到 worker（含步窗口的 loader 关闭持久 worker；见下文评审轮 F3），配置层校验「有窗口必有 prob / stop > start」（pydantic 构造期强制）。
   - 实机读数（RTX 4090D）：`mosaic_prob=1.0 + mixup_prob=1.0 + mixup_step_stop=2` 的 3 步训练闭环通过（OKS 验证正常出数）；确定性几何测试（恒等仿射下 2×2 粘贴坐标精确、平移 -70px 后仅右侧两片存活且关键点落点 22px 精确）。

**评审轮修复（独立评审，2026-10-16）**：1 blocker + 3 should-fix + 6 nits 全部处理：

- F1（blocker）：误差分析内部 id → 类名映射误按类名排序 → 改按**类别 id** 排序（COCO 形态类名不再错位；回归测试用 `zebra/ant` 形态锁定）。
- F2：命中 crowd（ignore 区）GT 的检测按 pycocotools ``dtIg`` 语义记为 ``ignored``（既非 TP 也非 FP），汇总与 precision/recall 不计入、报告单列一行。
- F3（两段修复）：pose 步窗口在 `num_workers>0` 下永不生效——(a) 变换补**重初始化握手**（窗口边界跨越置 `requires_dataloader_reinitialization`）；(b) 根因修复：**含步窗口的 loader 以 `persistent_workers=False` 构建**（持久 worker 跨迭代存活，重置迭代器不会重新下发变换；非持久则每次重置以当前 step 重新 pickle）——顺带修掉**检测 / 旋转框族同存的先存缺陷**（其 mosaic/mixup 窗口在 `num_workers>0` 下同样静默不生效）。worker 内探针验证：修复前 worker 侧 `_mosaic` 调用为 0，修复后按窗口在 worker 进程内触发（检测族 6 步 6 次 worker 侧 mosaic）。
- F4：自训练脚本拦截 0 伪标注（空标注训练集会被静默跳过、产出无意义学生——改为显式拒绝并给出处置建议）。
- nits（含一项按 pycocotools 顺序的语义收紧：先取最佳 IoU 匹配、命中 crowd 才记 `ignored`；更强的常规重叠保留 classification）：转换器拒绝负类索引；VOC 元素按**本地名**匹配（默认/xsi 命名空间的导出树可读）；`merge_coco` 对丢弃的标注告警；面积分档边界失败即报；离线分析同名词干歧义拒绝；`class_thresholds` 键语义澄清为**类别 id**（+测试）；`list_images` 递归（与训练数据面一致）；pose 窗口校验改 pydantic `model_validator`（所有构造路径强制）；报告注明 `-1.0` 分档约定；`pseudo-label` 的 `data=` 无有效标注时失败即报。

**遗留 → 下一步**：
- ① 官方 pose 增广的 zoom-out / color-jitter 与官方 mosaic/mixup 超参（比例、窗口）仍需按官方配置核对（当前参数镜像检测族端口，已在代码 docstring 标注「未核对官方值」）——训练复现切片的前提。
- ② 自训练的教师选择与迭代轮次编排（多轮 round、教师升级策略）未做——单轮闭环已通，多轮编排留给实际项目触发。
- ③ 误差分析的离线模式按「预测 json 的图像集 ⊆ 标注 json」处理；若后续要支持预测集为子集抽样（部分图像），需放宽 `_from_offline` 的匹配要求。
- ④ C3 的 `yolo_seg_to_coco` 仅支持单行单多边形（Ultralytics txt 惯例）；多多边形/带洞场景（COCO RLE）未覆盖。

---

### 3.6 P1「部署广度」执行记录（2026-10-16）

**已交付**（新测试 18 项：`test_dynamic_spatial_export` 6 + `test_openvino_backend` 6 + `test_trt_int8` 6（1 项 GPU 门控跳过）；全量套件 1840 过 / 17 跳过 / 0 失败；ruff / 溯源 / 分层 / 断链全绿）：

1. **D2 动态 H/W**：
   - `export_onnx(dynamic_spatial=True)`：H/W 轴动态（`Dim` 界从样例尺寸推导，避免与模型内部守卫冲突）；`ModelInputSpec.input_dynamic_shapes` 规格位被导出器消费（Dim / 轴名 / (min,max) 三种标记）。
   - **门禁**：导出时自动跑"第二尺寸 parity 探针"（大样例下采样 2/3、小样例上采样 3/2；torch vs ORT 同尺寸对比）——纯追踪的假动态图会被抓出。**fail-closed**：dynamo 导出器不可用时拒绝（legacy TorchScript 会把尺寸烘成常量，实测其"动态"图在其他尺寸触发 ORT broadcast 错误）。
   - 读数：`rtdetr/resnet18` 64px 权重，ONNX Runtime 在 48/64/96/160 全部运行、输出形状一致；探针尺寸 96 通过。
   - 顺带修复：`onnxscript`（dynamo 导出器依赖）此前缺失 → 所有导出静默回退 legacy；已加入 `export` extra。
2. **D3 OpenVINO（一等后端 ✅）**：
   - `export_openvino`（mixin）／`out=model.xml`（CLI）／`export/openvino_helpers.py`：ONNX（fp32/fp16，走既有门禁）→ IR 转换 → 编译 → **同 hook 的 torch↔OpenVINO 对拍门禁** → metadata 边车（IR 不能带 ONNX metadata）。
   - benchmark 新后端 `openvino`（设备 CPU/GPU/AUTO 可切、可复用既有 IR、纳入 parity 门禁、报告 device/IR 字段）。读数（64px，torch/onnx/openvino 同尺寸同批）：15.84 / 10.43 / **10.18 ms**（98.2 imgs/s），parity ✅。
   - 新 extra `openvino`（Apache-2.0，pip 可装；CI 测试门控 `importorskip`）。
3. **D3 边缘槽（⏳ 方向定案 + 阻断记录，未过 parity）**：TFLite/LiteRT 为选定方向，但两条阻断证据在案：(a) LiteRT 转换栈经 `xdsl`（由 `ai-edge-litert`/`litert-converter` 引入）硬钉 `typing-extensions<4.13`，与框架环境（≥4.13）**不可共存**（实测安装后 `import autovision` 因 pydantic_core 缺 `Sentinel` 崩溃，已回退并记录）；(b) 隔离环境（/tmp 独立 venv）里 `onnx2tf` 在本族图上失败于 FPN 的 **nearest `Resize`**（`node_upsample_nearest2d`），按其建议加 `-b 1` 静态形状重试仍无可交付产物。交付形态 = 转换脚本 `scripts/convert_onnx_to_tflite.py`（隔离环境用法 + 阻断说明 + 目标端 parity 归平台侧）+ 本节记录；**边缘后端的 parity/benchmark 读数未在本环境产出**（RKNN 为厂商 SDK，同样不可 pip 安装验证）。建议：按目标客户选型后再做该平台的对拍与出数（D5 政策：非一等后端走 ONNX hub + 转换脚本/文档）。
4. **D4 TRT INT8（✅）**：
   - `export_tensorrt(precision="int8", calibration_data|calibration_reader, quantization_config=...)`：先跑 `export_onnx_int8` 的 **int8↔fp32 对拍门禁**，再以 QDQ 图 + strongly-typed 网络构建引擎（TRT 11 路线；旧版另有 INT8 标志兜底）；`keep_int8_onnx` 可选保留中间图。
   - **两处真实互操作缺陷修复**（实例发现、门禁驱动）：① ORT QDQ 会产出 int32 常量输入的 `DequantizeLinear`（bias），TRT 11 解析器直接拒绝 → 新增折叠器（`_sanitize_qdq_for_tensorrt`，按 DQ 的 axis 广播还原 fp32；实测折叠 54 个节点）；② ORT 默认**非对称**激活量化（zero_point≠0）被 TRT 拒 → `Int8QuantizationConfig.activation_symmetric` 默认 True（对称，双运行时通用）。
   - 读数（640×640 b1）：torch 16.95ms → **int8 引擎 1.60ms**（p90 1.61，625 imgs/s，52MB 引擎），parity ✅（int8 容差 atol 2.0/rtol 0.5）。已知边界：TRT 无 int8 tactic 的融合（本模型：主干 Conv+Relu+MaxPool）需首层 fp32 配方（`nodes_to_exclude`，benchmark 的 `tensorrt.quantization` 字段可表达），构建失败时错误信息给出该处置建议（评审轮修复：TRT 的 `build_serialized_network` **返回 None** 而非抛异常，提示原先挂在异常路径上从未触发；现挂在 None 路径，文案改为「TRT 融合名未必等于 ONNX 名 + 首层 fp32 配方」）；int8 parity 裕度记录：boxes dim-1 和 max|diff| 1.784（容差 atol 2.0，约 11% 裕度，单权重抽样）。

**评审轮修复（独立评审，2026-10-16）**：4 项 should-fix + 若干 nits 全处理，其中一项为**全导出链的既有缺陷**：
- **parity 比较输入**：导出/benchmark 门禁原先用规格的全零样例张量比较——全零图上 DETR 顶-k 选择存在并列、元素级对比会抖动（评审实测同一图同一权重两跑一过一次失败；640 两分类小模型在静态导出下也会假失败）。现改为**带种子的随机张量**（追踪仍用样例）；benchmarks §4 已登记勘误。
- **fail-closed 覆盖面**：原仅覆盖 `dynamic_spatial` 旗标；现覆盖 `input_dynamic_shapes` 规格标记路径（任一非 batch 轴动态 → legacy 回退拒绝）。
- **int8 构建失败提示**：改为挂在 `None` 返回路径（真实失败路径），文案不再承诺节点名对应关系。
- **中间产物清理**：dynamo 外部数据边车（`*.onnx.data`，102MB 级）、失败时残留的 fp32 参考图与 int8 中间图，均在成功/失败路径清理（测试断言补强）。
- **`verify_parity` 顺序依赖**：torch 后端强制排第一（原按 dict 顺序，`{"onnx": ..., "torch": ...}` 会静默跳过全部门禁对比）。
- **第二轮复验（同一位评审）**：9 项判定中 8 项 VERIFIED；残留 5 项继续修复——benchmark 的 parity 比较输入同样改随机张量（原仅导出侧改到；三次 trial 稳定通过）、`export_onnx_int8` 自身的中间产物清理（成功/失败/门禁失败三路径，探针验证零残留）、OV 门禁失败路径的中间 ONNX 清理、三处过期文案（后端清单/脚本提示/计数）、并记录一项"并发负载下一次未复现的失败"待观察。**修复引入的回归**（reader 样本计数因门禁移入 try 而虚增 1）被全量套件抓出并修复（`test_quantization` 复跑 25 过）。
- 其余：`export_tensorrt`/`export_onnx_int8` 接受校准**路径**（README 的 CLI 示例此前会 TypeError；共享 `resolve_calibration_samples`）、IR 目标路径规则（`model.onnx` 不再被当成目录）、QDQ 折叠后清理无用初始化器、`keep_int8_onnx` 语义（保留折叠后图）写入 docstring、`benchmark`/`export` 包的后端清单与导出面补齐（`OpenVINOBackendArgs`/`OpenVINOEngineRunner`）、`Dim` 名称标记的 min 界注意事项入 docstring、校准解析惰性化。

**第二轮（2026-10-17）**：③ 收口（家族矩阵 + 两个真实缺陷）、④ 收口（自动配方）、② 部分收口（fp16 IR），并复测四后端读数（见 `docs/benchmarks.md` §2.3）。新测试 5 项：`test_dynamic_spatial_export` 6 → 11。

1. **遗留③ = ViT 系动态 H/W 验证 → 两个真实缺陷 + 家族矩阵**：
   - **缺陷 A（阻断，静态导出全线失败）**：HF 骨干（`detr/dinov2-*`、`detr/dinov3-vit*`）在 transformers 5.x 下**连静态 ONNX 导出都失败**——首个 eager 前向会安装 output-capturing hooks（在 `threading.Lock` 下），dynamo 无法 trace `with <lock>`，严格 trace（`torch.export`）直接报错。修复：trace 前做一次 **warm-up 前向**（hooks 落地，trace 走已安装快路径；数值透明）。回归测试用一个"首次前向持锁"的模型锁定（无 warm-up 时该测试失败）。
   - **缺陷 B（动态 H/W 的符号化阻塞）**：token→patch 网格的折叠用 `math.isqrt(token_count)`（`dinov2_wrapper` 与 `timm_wrapper` 各一处），动态 H/W 下 token 数是符号值、`isqrt` 不可 trace。修复：**由输入尺寸推导网格**（`H // patch`），eager 的非方图/网格不匹配报错原样保留（trace 内改用 reshape 兜底）；静态路径逐位不变（金标准/对拍测试把关）。
   - **缺陷 C（回退与报错面）**：严格 trace 失败此前**从不走 legacy 回退**（文档承诺的回退只覆盖"第二步之后"），且错误直接抛 dynamo 内部信息。修复：空间轴**静态**时回退 legacy TorchScript（记录告警，`verify` 门禁照常）；**动态**轴时抛可读的 `AutoVisionError`——`Constraints violated` 场景额外给出"模型在输入尺寸上有 Python 级分支"的处置指引（导出静态 H/W 或用 CNN 骨干）。回归测试 3 条（含"动态轴必须 fail-closed 不回退"）。
   - **家族矩阵实测**（256px、`dynamic_spatial=True`、含第二尺寸探针）：`rtdetr/resnet18` ✅（既有）、`detr/hgnetv2-b0` ✅（36s；另在 192/320 两尺寸 torch↔ORT 对拍 ✅）、`detr/s` ✅（34s）、`detr/gtr-s` ✅（37s）；`detr/dinov2-vittest-notpretrained` 静态 ✅（修复 A 后）、动态 ❌ = 上游 transformers 的 ViT 嵌入**在输入尺寸上有 Python 级分支**（patch 数命中"原生尺寸"才走快路径），严格导出无法符号化 → 可读拒绝；`detr/dinov3-convnext-tiny` 动态 ❌（同类 guard 违例）；`detr/gtrvit-s` 本身带 `export_blocker`（GLA 无 ONNX 降级）不适用。**结论：动态 H/W 的支持面 = torchvision/CNN 骨干 + HGNetV2 谱系；HF 骨干线以可读错误拒绝（边界写入 README 与 §2.3）。**
   - 导出面回归读数：`dynamic_spatial / cli_export / tensorrt_export / gtr_export_gates / quantization / openvino` 六套件 **66 过 / 1 跳过**；gated 的 TRT int8 套件（`AUTOVISION_TRT_TESTS=1`）**6 过**（含 640px int8 引擎构建 + parity，本轮改用 `suggest_first_layer_nodes` 取配方）；全量套件 **1930 过 / 18 跳过 / 0 失败**（第二轮前基线 1923/18，+7 = 新增测试）。
2. **遗留④ = int8 首层 fp32 配方自动建议（✅）**：新增 `suggest_first_layer_nodes`（导出自 `autovision.export`，按图拓扑序枚举前导 Conv 节点名）；**TRT int8 构建失败**（`build_serialized_network` 返回 None）的提示现在直接从 staged fp32 图给出可粘贴的 `quantization_config` 配方；复现脚本 `measure_new_backends.py --quantization-exclude auto` 一键套用。实机验证：640px 真实图取到 `node_Conv_1952/1954` → int8 引擎一次构建通过（p50 1.535ms）。
3. **遗留②（部分收口）**：OpenVINO **fp16 IR** 导出 + 编译后 IR 级对拍门禁在 CPU 通过（日志 `OpenVINO parity gate passed on device 'CPU'`）；中间 fp16 ONNX 的 torch↔ORT 对拍因本环境无 CUDA EP 而跳过（既有行为）。GPU/NPU 设备读数仍待硬件。
4. **四后端同尺寸复测（640×640，2 步检查点，parity 全开）**：torch 14.92/16.73ms、onnx（CPU EP）188.31/187.00ms、openvino（CPU）109.06/125.52ms、**TRT int8 1.535ms（639.7 img/s）**；表与口径见 `docs/benchmarks.md` §2.3。
5. **附带发现并记录（parity 门禁的前提）**：`benchmark(verify_parity=True)` 的对比以**数值收敛的检查点**为前提——规格串模型是随机权重，未训练框头在同一权重的 **GPU↔CPU torch 前向之间**就能差 ~0.9（归一化），门禁报出的是模型条件数而非后端保真度。已写入 `benchmark()` docstring + 规格串下显式告警，§2.3 记录。

**评审轮修复（独立评审，2026-10-17）**：2 blocker + 6 should-fix + nits 全处理：

- **B1（blocker）**：本轮的 warm-up 回归测试**不可证伪**——评审把 `src` 复制到 `/tmp`、仅删掉 warm-up 一行后该测试仍通过（严格 trace 失败会回退 legacy，同样产出文件，掩盖了缺陷）。修复：测试改为断言 `caplog` 中**无**"legacy TorchScript"告警（锁定 dynamo 路径必须完成）；复核：同一 no-warm 副本上该测试**失败**、仓库上通过。
- **B2（blocker）**：复现脚本 `--quantization-exclude auto` 的探针导出经 `deploy()` **污染被测模型**——`benchmark()` 的成本快照（params/FLOPs）刻意在部署前采集，改前 `benchmark_torch.json` 的 params = 33,439,416（部署后值）而非 26,358,456。修复：探针对检查点 `deepcopy`、探针图（含 `.data`）用后即清；复核：auto 运行 params = 26,358,456、输出目录零残留。
- **S1**：`--num-classes` 只作用于探针、未传给 `benchmark()`（评审 spy 复现）→ 已透传（spy 复核 7）；`--model` 既非文件也非规格串时给出可读报错（exit 2）。
- **S2**：legacy 回退路径会把**被标记的 batch 轴**做成动态，与同一导出刚发出的"batch 轴保持 STATIC"告警矛盾（`dynamic_spatial` 分支亦不可达）→ 现按 dynamo 语义跳过 axis-0 标记、删除不可达分支；复核：标记 + `dynamic_batch_size=False` 的 legacy 图 batch 维为静态 1。
- **S3**："ViT 嵌入按 patch 数分支"的解释被套用到 CNN（`detr/dinov3-convnext-tiny` 实为 HF stage-grid 守卫）→ 报错提示与 README / §2.3 改为通用表述（分支/守卫都在输入尺寸上；建议改用可符号化骨干）。
- **S4**：§2.3 的长尾描述与同段 p90 自相矛盾（20 步的 p90 即次大值 ⇒ 每轮 ≥2/20 次 >845ms，而原文写"2–3/60"）→ 按两组读数各自的口径改写。
- **S5**：README 的 int8 首层前提过于绝对（评审复测：不带排除的构建**也成功**，历史失败为间歇性）→ 改为"受支持的兜底配方"表述，§2.3 同步加注。
- **S6**：README 与两个 CI 都写 `pytest tests -q`（console script 形态），评审实测本机 **34 个收集错误**（site-packages 同名 `tests` 包遮蔽 + console script 无 CWD 路径；本轮新增的两个文件亦在其列）→ `pyproject.toml` 增 `[tool.pytest.ini_options] pythonpath = ["."]`；复核：console script 与 `python -m pytest` 收集一致、零错误。
- nits：量化器 docstring（"从不用于决策"→"用于受支持配方与构建失败提示"）、TRT 提示注释（fp32 兄弟图通常已被清理，QDQ 图是主来源）、benchmark 规格串告警移到后端校验之后且口径泛化（GPU↔CPU 与 CPU↔deepcopy 同现象）、脚本对"给了 auto 却无 tensorrt-int8"给出提示、`--image-size` 与检查点规格尺寸的约束写入 help、CHANGELOG 的非方图表述按 HF/ViT 线与 timm 包装器分别说明。
- 复核读数：全量套件 **1930 过 / 18 跳过 / 0 失败**（修复前后计数一致）；导出六套件 66 过 / 1 跳过；gated TRT int8 六项 6 过；`ruff` / 溯源 / 分层 / 断链 / 模型库 `--check` / `mkdocs --strict` 全绿。

**遗留 → 下一步**：
- ① 边缘平台（RKNN/TFLite）的目标端 parity 与读数（依赖客户选型与厂商工具链；脚本与阻断记录已就绪）。
- ② OpenVINO 的 GPU/NPU 设备读数与 `ov_config`（PERFORMANCE_HINT/AFFINITY）复测——640px CPU 存在**稀疏长尾**（60 次直测 2–3 次 350–850ms，p50 稳定 ~100–125ms；64px 无此现象），已作为观察项记入 §2.3。
- ③ 已收口（家族矩阵 + 边界文档）；HF 骨干若未来上游把输入尺寸分支去掉（测试已写成"可用即可用"的双分支形态），动态 H/W 可随之开启。
- ④ 已收口（自动建议）；残留：TRT 的**融合名**与 ONNX 节点名仍可能不一致（提示已说明按构建日志迭代）。

### 3.7 P1「工程欠账」执行记录（2026-10-16）

**已交付**（新测试 40 项：`test_hpo` 38 + `test_voc_trainable` 2；ruff / 溯源 / 分层 / 断链 / mkdocs strict 全绿）：

1. **B2 HPO（门禁：一次随机搜索出报告 ✅）**：
   - 新包 `autovision.hpo`（分层 rank 1，与 engine 同级；`scripts/check_layering.py` LAYERS 显式登记）：
     - `space.py`：点号覆盖键 → 分布规格（`choice` / `uniform` / `loguniform` / `randint`，裸列表 = choice）；种子化采样（同种子同点集）；`grid()` 笛卡尔积（连续轴显式报错）；`split_overrides` 把采样点还原成训练入口的 kwargs——**复用与 CLI 完全同一套 Pydantic 覆盖面**（`model_args.*` 等合并进对应字典），拼错/越界的键由配置校验在 trial 1 报错而不是被静默忽略。
     - `search.py`：`run_search` 三种策略（random / grid / asha）；试验函数契约 `fn(params, steps, run_dir)`；失败试验记录后继续（不中断搜索）；已完成试验目录命中 `metrics.json` 即**复用**（断点续跑）；产物 `report.md`（空间摘要 + 按目标排序的试验表 + 最优配置）+ `trials.jsonl` + `best.json`。
     - `asha.py`：原生 ASHA（零依赖）——预算阶梯 `min, min·eta, …, max_budget`（末级钉在 max_budget）、每 rung 提升前 `ceil(n/eta)`、提升的配置**在同一 run 目录续训**（`resume_interrupted=True`，`steps` 为总量目标；**不能同时传 `checkpoint=`**——引擎显式拒绝该组合，见评审轮 F1）；逐 rung 指标落 `metrics.rung<r>.json`（中断后从丢失的 rung 重入）。Ray Tune 用户可用同一 trial 函数接其 `ASHAScheduler`（文档在案，不作捆绑依赖）。
     - `task_trial(task, base)`：把任意 `autovision.train_*` 入口接成试验函数（objective 读 `best_metrics`，如 `val_metric/map_50`）。
   - CLI：`autovision-cli hpo space=search.yaml out=... task=object_detection data=... model=... [strategy=random|grid|asha] [trials=] [budget=] [objective=] [asha_max_budget=]`（23 条命令）。
   - **门禁读数**（本机 GPU：`NVIDIA GeForce RTX 4090 D`，`devices=auto`；coco8，6 个真实试验 × 300 步，空间 = lr×wd×ema_warmup×batch）：
     最优 trial（`lr=8.8e-4, wd=1e-4, ema_warmup=100, bs=4`）**map_50 = 0.75**，其余 5 个
     0.0（高 lr / 长 warmup 在 300 步预算下学不动）——搜索把可用区间挑出来了，报告与 JSONL 落盘。
   - 测试覆盖：规格校验（7 类非法输入）、种子确定性、网格积与连续轴拒绝、报告/最优/失败记录/复用、ASHA 的 rung 结构与逐 run 步数单调（续训证据）、`task_trial` 的 resume 约定（monkeypatch 入口）、CLI 端到端（真实 2 步训练 → 报告）。
2. **B3 VOC 可直接训练（门禁 ✅）**：
   - `download → convert-data from=voc → train` 三步配方入 README（含 ignore 区语义说明）。
   - 端到端测试（CPU，`test_voc_trainable`）：合成 VOC 树 → CLI 转换（train/val 分片 + 规范 20 类名 + `difficult → iscrowd`）→ 训练 2 步 → val 出数；另测自定义类表（`class_names=`）。
   - **真实读数**：20 类 VOC 格式语料（120 图 / 224 标注）转换正确（类名 = VOC 规范 20 类排序）；可学语料（8 类、位置-类别相关、96 训练图）2000 步 GPU 训练 → **map 0.0029 / map_50 0.0103 / recall 0.125**（合成重噪声语料 + 2000 步下的正常量级；管线全链可训可评）。
   - **VOC 官方语料获取受限（记录）**：官方下载源（host.robots.ox.ac.uk → www.robots.ox.ac.uk）在本环境限速至 ~14 kB/s（460MB tar 不可用；pjreddie 镜像已失效返回 HTML），真实 VOC2007 全量跑分留待有快镜像/离线语料的环境；转换与训练路径已由合成规范语料端到端覆盖。

**评审轮修复（独立评审，2026-10-16）**：1 blocker + 6 should-fix + 8 nits 全处理：

- **F1（blocker）**：ASHA 提升在真实训练入口上必然失败——`task_trial` 同时传 `checkpoint=` 与 `resume_interrupted=True`，而引擎显式拒绝该组合（`'checkpoint' and 'resume_interrupted' cannot both be set`），导致提升 rung 全部 `failed`、搜索静默停止。修复：只传 `resume_interrupted=True`（运行目录非空时），`steps` 作为总量目标由引擎续训；**真实回归测试在案**（coco8 CPU，2 rung ASHA：rung 0 两个 ok、rung 1 提升 rung
  ok 且继续训练）。
- F3（同源）：首个 checkpoint 之前崩溃的试验目录此前被非空目录守卫永久卡死（每次重跑都 failed）；修复后 `resume_interrupted` 走引擎的"No last.ckpt found"路径自愈。
- F2：`metrics.json` 复用读取改为容错（截断 / 非映射 / 非数值 → 告警后重跑，不再中断整个搜索）。
- F4：README 的 VOC 配方修复两处（原配方按原文执行直接 ConfigValidationError——评审实测）：缺 `val` 分片；`images` 误放顶层（应置于 `train`/`val` 分片内，配置为 extra=forbid）。修复后按 README 原文逐字执行验证通过。
- F5/F6：CLI 的 `task=` 名称在开跑前解析（拼错即 `BadParameter`，不再以"failed trial + exit 0"收场）；缺 `space=` 与非法数值参数给出可读报错。
- F7：ASHA 失败 rung 不再把空指标镜像进 `metrics.json`（污染后续复用）。
- nits：空间校验（NaN/Inf 界、元数、randint 整数性、点号空段）、`split_overrides` 前缀冲突（`a.b` + `a.b.c`）、grid 去重、保留键（`out`/`steps`/`checkpoint`/`resume_interrupted`）显式拒绝并文档化、`task_trial` 各任务族 objective 名文档化、`force` 语义（忽略复用但可能续训）文档化、CHANGELOG 结构（三个 `### 新增` 合并为一）、分层门禁可见化（`task_trial` 改为静态可见的 `cli.train_task` 导入，`ALLOWED_UPWARD` 登记 `hpo→cli` 并验证该条目为活条目）、§3.7 设备归属补证据（本机 GPU 实测名）。

**遗留 → 下一步**：
- ① 真实 VOC2007 全量训练读数（依赖镜像/离线语料；配方与代码路径已就绪）。
- ② Ray Tune 集成仅有文档路径（未捆绑依赖、未在案测试）——如需多机/并行 HPO 再引入并补测试。
- ③ HPO 目前串行执行（并行 trials 需要 Ray/多进程编排）；`grid` 与连续轴混用时需改用嵌套 choice 或 random。
- ④ VOC2012（与 2007 转换器兼容但未在注册表）与 VOC 分割标注（`SegmentationClass`）未纳入。

### 3.8 P2「战略项」执行记录（2026-10-16）

**背景**：P2 行退出门禁为「按产品/客户触发」。本轮把其中三项落成**参考实现**（可验证、可评审），第 4 项（昇腾/地平线）保持按客户触发并记录决策依据。新测试 31 项（评审轮后 44 项）：`test_tracking` 15→20 + `test_data_versioning` 9→15 + `test_serving` 7→9。

1. **A3 tracking（ByteTrack 参考实现 ✅）**：
   - 新包 `autovision.tracking`（LAYERS rank 5，纯数学：numpy + scipy 分配）：`BYTETracker` —— 匀速卡尔曼（`xyah` 状态、官方过程/测量噪声参数化，逐式核对零误差）+ **三阶段关联**（官方语义：激活轨迹含 lost 对高分池、分数融合 IoU；仍追踪者对低分池 0.5；未确认轨迹对剩余高分框 0.7、未命中即移除），class-aware（不同类不互抢 id，文档化与官方 MOT 单类实现的差异）。官方门限齐备：新轨迹需 `track_thresh + 0.1`（`det_thresh`）且连续命中才发布（首帧激活例外），低分池下限 0.1。逐帧 API 吃绝对像素 `xyxy`（框架原生检测输出）。
   - 接入推理运行时：`run_stream(..., track=True, track_thresh=, track_buffer=)` → JSONL 逐帧 `tracks`（track_id/bbox/score/class_id/age）+ 渲染 `#id 类别 分数` + `StreamRunResult.tracks`（唯一 id 数）；`predict source=... track=true` 经 kwargs 透传（CLI 无需改动）。
   - 测试：移动目标单 id、并行双目标不互换、类变更换新 id、丢帧缓冲内同 id 复活、超缓冲后退役换 id、低分框不新建/可维持、帧率缩放缓冲、种入确定性、参数校验、端到端流（静态查询网格下 ≥1 个 id 跨全部帧）。
2. **数据版本化（内容寻址快照 ✅）**：
   - 新 `data/versioning.py` + CLI `data-snapshot`：**明确与 mmap 缓存键（路径+mtime）区分**——快照按分片记录根路径、文件数/字节数、`(相对路径, 大小)` 清单 sha256；`deep=true` 追加**文件内容** sha256。根相对摘要 ⇒ 目录整体搬迁/复制可用副本 spec 校验通过（测试锁定）。
   - `verify_dataset_snapshot` 输出结构化漂移报告（added / removed / resized / content_digest + 涉及文件，明细字段 `manifest`），CLI 漂移时非零退出；同尺寸内容改写只有 deep 快照能抓（测试锁定该边界）。spec 内**相对路径按 spec 文件目录解析**（与训练入口同规则，基准目录记入快照，换 CWD 校验不漂移）；写入默认不覆盖（`overwrite=true` 显式放行）；`deep=true` 校验只对 deep 快照有效（浅快照显式报错，不做空校验）。
   - 实机读数：对真实 `coco8` 快照（deep）：2 分片，train 9 文件 / 3015 字节，摘要 `aee45d6a5587…`；复校验 `ok=true` 零差异。
3. **F2 参考 serving（✅ 参考实现，边界明示）**：
   - 新包 `autovision.serving`（rank 1，复用 `InferenceEngine` ⇒ 与评测/流同源前后处理；新 `serving` extra = fastapi + uvicorn）：
     - `POST /predict`：`{"images": [b64, ...]}`（或单张），批处理按输入顺序返回、`max_batch` 上限（413）、`class_names` 映射；批=逐张等价（测试断言数值一致）。
     - `GET /healthz`（后端/网络类/artifact/threshold）；`GET /metrics` —— **零依赖** Prometheus 文本注册表（counter/gauge/summary：请求按端点与状态、图片数、延迟 summary）。
   - CLI `serve model=... port=...`（缺 extra 给出安装指引）。
   - **边界（明确写入 README/模块 docstring）**：无鉴权、无 TLS、单进程、无编排；漂移检测 / A-B / 金丝雀不在范围——`/metrics` 是其起点，按产品需求在其上构建。F2 原文「若定位仍是训练框架，在 README 明确写为边界」的要求以「交付参考实现 + 边界声明」满足。
4. **昇腾 / 地平线（⏳ 按客户触发；决策记录）**：
   - **昇腾**：软件侧适配已在（`runtime/accelerator.py`：设备解析/CUDA→NPU 回退、hccl DDP 后端、bf16 能力与混合精度守卫；NPU 路径以 mock 测试在案），**无真机验证**。触发器：客户明确要求昇腾训练/推理 + 可用真机；届时工作 = 真机冒烟（设备/DDP/AMP/导出）+ 后端选项文档，代码面预计较小。
   - **地平线（J5/J6 等）**：仓库零覆盖。触发器：客户选择地平线硬件；届时工作 = 官方工具链（要厂商 SDK，pip 不可安装）导出转换 + 目标端 parity 读数，路径与 D3 边缘槽同构（先 ONNX hub + 转换脚本/文档，见 §3.6）。
   - 结论：两项**不做预研投入**（保持 README「已知边界」表述），按客户需求立项。

**评审轮修复（独立评审，2026-10-16）**：2 blocker + 8 should-fix + 若干 nits 全部处理（新测试 31 → 44：tracking 15→20、versioning 9→15、serving 7→9，全部为对应缺陷的回归）：

- B1（blocker，数据版本化）：spec 文件里的**相对路径按进程 CWD 解析**（评审复现：从带"诱饵目录"的 CWD 生成快照，钉住的是错误的数据树，且记录路径无法区分）——现按 **spec 文件所在目录**解析（与训练入口同一规则），基准目录记入快照（`spec_base_dir`），任意 CWD 可复校验；回归测试含诱饵树。
- B2（blocker，跟踪）：**一个低分检测可同时驱动两条轨迹**（评审复现：两条同位置轨迹一 tracked 一 lost，单张 0.3 分框同时被阶段 2 的 tracked 池与阶段 3 的 lost 池消费，双路卡尔曼更新 + 死 id 借已分配的检测复活）——阶段 3（自创的"lost 借低分复活"）删除，关联按官方重构为三阶段且**每检测 1:1 消费**；回归测试锁定「单低分框只维持一条轨迹、且永不复活 lost 轨迹」。
- S4（跟踪）：官方门限缺失/未文档化（低分池无 0.1 下限、新轨迹无 `det_thresh`、无"未确认一帧不发布"确认门、首帧激活例外）——全部按官方实现（`det_thresh = track_thresh + 0.1`、低分池 `0.1 < score < track_thresh`、未确认轨迹 0.7 关联且未命中即移除、首帧新轨迹即时激活），docstring 的"差异清单"只保留真实差异（class-aware、xyxy 逐帧 API、无 mot20 模式）。实测影响：随机权重 tiny 模型分数 ~0.02，旧实现会给它们建轨迹（官方语义下永不建）——流测试改用分数抬升的模型（几何不动），README/CHANGELOG 的"官方语义"表述现与代码一致。
- S6（跟踪）：`age` 实为命中次数——现为 `frame_id - start_frame`（创建以来帧数），流测试断言跨 8 帧 id 的 age 恰为 0..7。
- S1（数据版本化）：`deep=True` 校验浅快照是**静默空转**（无记录内容摘要，docstring/CLI 却称内容检查）——现显式报错并给出"重新 deep 快照"的指引。
- S2（数据版本化）：同分片**重复根标签**时清单键碰撞、深摘要按标签重解析根导致**子树漏算**（评审复现：第二个根的子树内容改写不被发现）——行内改携带各自解析路径（不再由标签反推），重复标签在收集期显式拒绝。
- S3（流）：tracker 的 `frame_rate` 硬编码 30（真实 fps 就在同一函数里）——现取源视频真实帧率（未知回退 30），`track_buffer` 语义与文档一致。
- S5（serving）：**5xx 不上 `/metrics`**（仅 AutoVisionError 被捕获；其余经 FastAPI 500 静默）——现 500 路径同样计数（status="500" + errors_total + 延迟观测，用崩溃引擎回归）。
- S7/S8（测试/文档）：serving 顺序断言原为恒真套话（`... or True`）——改为真实顺序断言（A,B,A → 位置对齐）；`create_app` docstring 误称 `model=` 可收 `InferenceEngine`——已更正。
- nits：`tracker.update` 严格校验（形状/非有限值/类别 dtype → AutoVisionError，杜绝 NaN 框流入 JSONL）；`_jsonable` 覆盖 numpy 数组/标量（不再中途 TypeError）；无类别帧的轨迹标注不再伪装 class 0 名称；`/metrics` 的 HELP 行不再依赖调用顺序（调用点统一 + 声明升级），summary 的峰值移入独立 gauge 族（`*_seconds_max`）；快照写入加覆盖守卫、布尔严格解析（`deep=False` 不再被 `bool()` 反转）、坏 JSON 包成 AutoVisionError；漂移明细字段更名 `manifest` + `truncated` 标记 + `manifest_omitted` 标记；符号链接范围入 docstring；`data-snapshot` 拒绝未知键（拼写守卫）；删除死代码 `_SplitSnapshot`。
- 复核读数：`ruff` / 溯源 / 分层 / 断链 / 模型库 `--check` / `mkdocs --strict` 全绿；P2 三套件 **44 过**；全量套件 **1923 过 / 18 跳过 / 0 失败**（修复前基线 1910/18，+13 即新增回归）；coco8 实机读数不变（2 分片，train 9 文件 / 3015 字节，摘要 `aee45d6a5587…`，复校验零差异）。
- **对评审原始复现脚本的复跑**（同一批脚本，修复后重跑）：B2 复现（两轨迹 + 单个 0.3 分框）由「1 检测 → 2 轨迹」变为 **1:1**；S4 复现（0.01 / 0.0 / −0.5 分框与 0.0 分"复活"lost）全部**不再建轨迹 / 不再复活**；S1 复现（`deep=True` 校验浅快照）由静默 `True` 变为**显式报错**；S2 复现（重复标签）由"子树漏算且校验通过"变为**收集期显式拒绝**；B1 复现（CWD 诱饵树）记录的根路径与清单键均为 **spec 目录树**（`images/real.png`，非诱饵 `decoy.png`）；S3 复现（10fps 源 + 15 帧遮挡）由「同 id 跨遮挡」变为 10 帧窗口外**换新 id**；S5 复现（引擎 `RuntimeError`）500 现被 `/metrics` 计数；`/metrics` 曝光文本逐族带 HELP/TYPE（summary 仅 `_count`/`_sum`，峰值为独立 gauge 族）。

**遗留 → 下一步**：
- ① 跟踪侧：OC-SORT 备选（ByteTrack 首发）、跨摄像头 ReID、track 级指标（MOTA/IDF1，需标注视频）未做。
- ② serving：鉴权/限流/多 worker/TLS 与漂移检测、A-B、金丝雀按产品需求触发；ONNX/TRT 后端的 serving 组合已在 `create_app(backend=/artifact=)` 支持（本环境以 torch 后端测试）。
- ③ 数据版本化：远端快照注册表（把快照 JSON 推到对象存储/注册中心）与数据集「拉取即校验」流程未做；当前形态 = 本地快照 + 校验。
- ④ 昇腾/地平线：按客户立项（触发器与工作范围见上）。

## 4. 战略立场（含"不做什么"）

1. **不做 Ultralytics 式的 15 个后端**。部署广度以 ONNX 为 hub 借力社区转换，一等公民维护控制在 3–4 个（ONNX / TensorRT / OpenVINO / 边缘一个）。
2. **护城河叙事保持锐利**：许可干净（Apache + 全链溯源）+ 可验证一致性（门禁 + bit-exact + 协议化基准）。营销面与工程面同一套证据。
3. **基准纪律不可退让**：只引官方发表值、自家数字必附复现脚本、只增不改 + 勘误节（`docs/benchmarks.md` 既定协议）。
4. **GTR MIT 全族是旗舰干净模型库**：26 个检查点覆盖 6 任务 s/m/l/x，是外部用户拿得到的、许可无虞的完整谱系；DINOv2/v3 相关路径受权重许可限制，只作内部/合规自证路径。
5. **roadmap 重排**：原 M4+ 已超期完成（分割/分类、MLflow 注册表、int8 PTQ、OTel），把本文件 §3 缺口按优先级排入正式路线图。

## 5. 发布前红线状态（2026-10-16 更新）

- ✅ **mkdocs 导航与新树不一致（构建必失败）** → 已解决（E1，见 §3.4）：导航按新树重写，`mkdocs build --strict` 零警告；CI docs 任务 + 站点结构契约测试锁定。
- ✅ **版本号 `0.1.0` 与 tag `v0.2-clean` 不一致** → 已解决（E1，见 §3.4）：版本号单一源 `0.2.0`（pyproject 动态读取 `_version.py`）；发布 tag 统一 `vX.Y.Z`，由测试在打 tag 的提交上强制与版本一致；历史 tag `v0.2-clean` 保留原义。
- ✅ **本文件暂未接入 `mkdocs.yml` 导航** → 已处置（E1，见 §3.4）：按本文件「内部分析稿（非对外发布产物）」状态，以 `exclude_docs` 排除出公开站点（仓库内保留），测试锁定排除项不陈旧。
- ⏳ **工作区变更未提交**（含 `docs/design/*`、`docs/provenance/*` 全量删除；本次改动后仍为在途状态）：对外发布动作前按 `docs/release_process.md` 清单提交，并刷新发布快照分支（孤儿单提交）后打 `v0.2.0` tag。
- ⚠️ **`uv.lock` 陈旧（先存问题）**：`uv lock` 因 `super-gradients` × `transformers` 冲突未列入 `[tool.uv] conflicts` 而报「不可满足」，且锁缺 `video` / `docs` extra；修复前以 pip 为准（CONTRIBUTING 已标注）。
