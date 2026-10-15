# AutoVision 实现蓝图 · 分册 29：DETR 产品化路线——对标 YOLO 级成熟度的差距分析与实施计划

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（设计评审稿，待实施；v1.1 修订内容见 §12 修订记录） |
| 状态 | 设计完成；分阶段实施计划见 §6，总验收 gate 见 §7 |
| 决策依据 | 2026-10-15 产品化方向决策：将框架从「研究训练框架」推进到「YOLO 级成熟度的实时检测产品」；本册 §1.2 给出该目标的操作性定义（六判据 J1–J6） |
| 关联分册 | 03（模型层）、04（蒸馏配方）、05（任务模型与评测）、08（部署与模型注册表）、10（里程碑/风险表格式、semver 与交付形态 §17–§18）、12（缺口状态 §4.3 检测家族补全）、19（D-FINE 基线对齐）、23–26（跨模型蒸馏 / FP8 训练 / token indexer / step 工程）、27（开放词汇）、28（清洁溯源、D1 权重决策）、30（任务优先目录结构 as-built，谱系脚手架所在） |
| 受影响模块 | `src/autovision/tasks/detection/detr/`、`tasks/detection/shared/`、`export/`、`cli/`、`registry/`、权重供应链（`docs/provenance/LEDGER.md` P3/P4 待办项）、`docs/`、CI（基准回归） |
| 兼容承诺 | 公开 API 键名（`pretrain/train_*/embed/benchmark` 等）不破坏；`detr/<backbone>` 既有注册键不动，谱系键（§3.2）为纯新增；`docs/benchmarks.md` 基准表只增不改，修正走勘误节（§4.4） |

> **标注约定**：【现状】仓库已验证事实（出处 `runs/`、`src/`、`docs/`，可复查）；【设计】本册决策（附编号 D0–D10 及理由，含 D1b）；【目标】可验收目标（与 §7 判据挂钩）；【上游】宽松许可参考实现；【风险】风险与缓解。

---

## 1. 背景与目标定义

### 1.1 动机

本仓库工程底座（配置/引擎/导出/测试/溯源治理）已达到成熟开源框架水准，但对外呈现仍是「研究训练框架」：没有自有模型族谱系、没有公开的精度-延迟基准、没有官方权重 zoo、没有发版节奏。YOLO 系列十年的成功要素不在单点算法，而在**产品成熟度**：谱系化模型、公开帕累托数据、极简 API、部署纵深、持续演进。本册定义「达到 YOLO 的程度」的可验收含义，并给出分阶段实施计划。

### 1.2 「YOLO 的程度」操作性定义（六判据）

| # | 判据 | 可验收含义 |
|---|---|---|
| J1 | 有名字、有谱系、有权重 | `detr/{n,s,m,l,x}` 五档注册，≥3 档有官方权重，可下载、可复现、有模型卡 |
| J2 | 有公开的精度-延迟表 | `docs/benchmarks.md` 含自家各档 COCO val AP + TRT/ONNX 延迟，与 ≥6 个对手官方档位同表，全部有出处 |
| J3 | 帕累托前沿有竞争力 | N/S 档：同延迟下 AP 与 YOLOv8/11 同档差距 ≤0.5（第一版进同一张表，不要求反超） |
| J4 | 五行代码闭环 | `train → val → predict → export` ≤5 行 API / 3 条 CLI；coco8 级冒烟 <2 分钟，进 CI |
| J5 | 部署纵深 | N/S 档 FP32/FP16/INT8 三档延迟-精度公开，INT8 AP 损失 <1%；一个公开可复现的边缘端示例 |
| J6 | 演进机制 | semver + changelog + PyPI；连续两个 minor 周期按期交付，每版更新基准表 |

### 1.3 现状盘点

| 维度 | 现状【现状】 | 相对 YOLO 水准 |
|---|---|---|
| 工程底座 | Pydantic 严格配置 + Registry、双 DDP 引擎、原子 checkpoint/resume、全量 1505 测试通过（2026-10-14 Phase 4 终验，LEDGER）、金标准快照、ruff/CI、provenance allowlist + LEDGER | 持平或更强 |
| 数据格式 | COCO、YOLO、YOLO-OBB（8 分量）、mmap 清单（分册 06） | 持平 |
| 训练数据 | COCO2017 就绪：train2017 118,287 / val2017 5,000，官方数精确核对，落盘 `~/Research/Project/Dino-Dome-Research/dataset/coco/`（LEDGER Phase 4） | J2 前置条件已消除 |
| 任务覆盖 | 检测（DETR 装配族（variants/optimizer 谱系脚手架，2026-10-15 拆分）/ D-FINE / RT-DETR / RF-DETR / PicoDet / OBB / 开放词汇脚手架）、分类、分割；depth/pose 脚手架 | 检测够用，广度非本册目标 |
| 训练配方 | SIoU（分册 15）、box-aware eraser（分册 16）、多尺度、EMA、Dome LR、DistillationV3 全规格（分册 04）、SSL 骨干微调闭环 | 有独有亮点，**无标准化 COCO recipe** |
| 部署 | ONNX FP16 + 数值对拍门禁（atol 5e-3）、TensorRT 导出、`benchmark` CLI、SAHI（分册 08） | 骨架已有，缺 INT8 与公开基准 |
| 模型族 | DETR 为装配体（RT-DETRv2/D-FINE 解码器 + 多来源骨干），**无谱系、无 COCO 官方成绩、无权重 zoo** | **最大差距** |
| 实测性能 | VisDrone：RF-DETR-n 50ep mAP 0.2789 / AP50 0.4925（`runs/visdrone_rfdetr_nano_800_full-erase-siou-multiscale-as90-50ep`）；DEIMv2-DINOv3-S mAP 0.2123@8088 步（未完训；该线已移除，历史 run 仅存档）；**无 COCO 标准协议成绩** | 无公开可比数字，J2 无从谈起 |
| 生态 | MkDocs 站骨架、CLI 已有；无 PyPI 发版、无版本化 release、无教程/示例库 | **第二大差距** |

### 1.4 战略定位

- **不学**：dense prediction + 标签分配的技术路线（DETR 系有端到端集合预测的答案，本仓库已在此路线上）；全任务大而全（分类/分割/depth/pose 保持脚手架，资源全部押检测）。
- **学**：产品形态（谱系/权重/基准表/极简 API）、发版节奏、生态运营。
- **差异化身份（设计立场）**：**「YOLO 级工程成熟度 + DETR 端到端范式 + SSL 预训练闭环」**。本仓库独有的牌：SSL 预训练 → 蒸馏 → 检测微调 → FP8 低精度训练全链路（分册 23–26 蓝图）、NMS-free、开放词汇（分册 27）、小目标密集场景（VisDrone/Dome，YOLO 的传统弱项区）。叙事是「另一个范式的同等成熟度产品」，不是「又一个 YOLO」。
- 【设计 D0】**研究组件不进谱系第一版**。谱系用成熟组件打基准；Dome/开放词汇等验证后按 §5 门禁在后续 minor 融合，避免产品线被实验不稳定拖累（FSDC/DEIMv2/ECViT 已于 2026-10-15 整体移除，见 §12）。

---

## 2. 差距矩阵

| 判据 | YOLO 参照 | 现状 | 差距 | 承接 |
|---|---|---|---|---|
| J1 谱系+权重 | YOLOv8/11 N–X 全档官方权重 | `detr/<backbone>` 组件级键，无谱系、无 zoo | 谱系定义、缩放规则、训练、发布管线 | P1 |
| J2 公开基准表 | 官方 README 基准表（AP+延迟） | 无 COCO 成绩，VisDrone 数字散在 runs | 协议、复现脚本、对手数字政策 | P0 |
| J3 帕累托竞争力 | N/S 档同延迟 AP 领先 | 未知（未测） | 蒸馏短程 recipe + N/S 档训练 | P1 |
| J4 五行闭环 | `model.train()/predict()` + coco8 | API 已接近，缺 predict/val、冒烟数据集、自动下载 | API 面 + 数据入口 | P2 |
| J5 部署纵深 | FP16/INT8 + 全平台导出 | FP16 对拍门禁 + TRT 已有 | INT8 PTQ、动态 shape、端侧示例 | P3 |
| J6 演进机制 | 数月一代、每代有数字 | 里程碑文化已有，无发版执行 | semver/PyPI/changelog/节奏 | P4 |

---

## 3. 产品定义：DETR 谱系

### 3.1 谱系表

【设计 D1】谱系以「官方宽松许可上游组件装配」为纪律（承接分册 28）：骨干与解码器组件必须有可追溯宽松许可出处并登记 LEDGER；许可不明组件不得入谱系。

| 档 | 骨干 | 解码器 | 输入尺寸 | 对标锚 | 起点路线 |
|---|---|---|---|---|---|
| `detr/n` | HGNetV2-N | D-FINE N 宽度配方 | 560–640 | YOLOv8n / YOLO11n | **蒸馏优先（D1b）** |
| `detr/s` | HGNetV2-S | D-FINE-S 装配 | 640 | YOLOv8s / YOLO11s | 装配 warm-start 微调 |
| `detr/m` | ConvNeXt-T | D-FINE-M 装配 | 640–800 | YOLOv8m / RT-DETR-R18 | 装配 warm-start 微调 |
| `detr/l` | ConvNeXt-S/B | D-FINE-L 装配 | 800 | RT-DETR-R34 / YOLOv8l | 装配 warm-start 微调 |
| `detr/x` | ConvNeXt-B / DINOv2-S（可选档） | D-FINE-L 装配或蒸馏 | 1024 | RT-DETR-R50 / YOLOv8x | 装配或蒸馏 |

【设计 D1b·v1.1】**N 档为蒸馏优先档**：`dfine_n` 官方权重不存在（LEDGER Phase 4 实测：HF `Peterande/D-FINE` 仅 s/m/l，已从资产清单排除），且低宽度配置下形状驱动装配的覆盖率不可靠（ecvitt(192)+dfine_s 仅 27% 的诚实拒绝先例）。N 档起点 = 12ep 蒸馏（DistillationV3；教师默认 RF-DETR-n，备选 dfine_s 蒸 decoder 知识），P0 内决策冻结。教师池【现状】：RF-DETR 家族（Apache，transformers 通道已验证）、DINOv2（Apache）、`dfine_{s,m,l}` 官方 COCO 权重（Apache，缓存已就位）——DEIMv2 已于 2026-10-15 移除，不再作为教师候选。

【设计 D1b 附注】骨干候选修订：ECViT 线移除（2026-10-15），n/s 档由 HGNetV2 顶上——与 RT-DETR/D-FINE 官方骨干同构、timm 权重齐全、卷积为主边缘友好；M/L 档 ConvNeXt（torchvision/timm，VisDrone runs 已有大量实证）。宽度/深度倍率与输入尺寸在 P1 以实际 latency profile 冻结；本表为**结构承诺**，非数值承诺。DINOv3 骨干因 Meta 定制许可（分册 28 D1-a：用户自持 + opt-in 门控）不进默认谱系，仅作为可选变体存在。

### 3.2 命名与模型串

【设计 D2】谱系键 `detr/{n,s,m,l,x}` 为「官方配方 + 官方权重」入口（N 档为官方蒸馏配方 + 蒸馏权重，D1b）；既有 `detr/<backbone>` 组件级键保留不动，作为自定义装配入口。两者共享同一模型实现，仅默认超参/权重来源不同。训练与推理入口统一走既有 `train_object_detection(model="detr/n", ...)` 与 `load_task_model(...)`。

### 3.3 单一定义 + 缩放规则

【设计 D3】谱系由**单一模型定义 + 结构缩放规则**生成（宽度/深度/分辨率/解码器层数四元组表），禁止每档手写复制代码。缩放规则作为分册 03 的增补节落档；配方可复现性由金标准快照机制覆盖。

### 3.4 权重 zoo 与模型卡

【设计 D4】权重发布复用既有权重供应链机制（manifest + commit/LFS sha256 钉住，`scripts/update_weight_hashes.py`；license 字段必填——LEDGER R7 运行时警告的延伸：谱系权重发布起 license 字段为硬性）。模型卡必填字段：训练数据、schedule、种子、AP（自测）、三精度延迟、许可证、已知局限。离线/镜像消费端（`AUTOVISION_MIRROR_URL`/`AUTOVISION_OFFLINE`，分册 12 §3 待办）在 P1 一并收口——权重 zoo 发布后该待办从「应该做」变为「必须做」。

---

## 4. 基准与评测协议

### 4.1 COCO 标准协议

【设计 D5】COCO 2017 train2017（118k）/ val2017（5k）；AP 口径 = pycocotools 标准，与 torchmetrics 评测管线对拍一致（分册 05）。三档 schedule 固化为配置：**12ep**（蒸馏短程，主打）、**24ep**（中程）、**72ep**（监督参照，用于量化 D1b 蒸馏增益；仅 N 档必须、其余档选做）。对外声明数字一律标注 schedule 档位与种子；关键档位 ≥2 seeds 报均值。

### 4.2 延迟协议

【设计 D6】固定协议，表头登记硬件型号：TRT FP16，batch=1 与 batch=4 两档，含 H2D 拷贝与预处理，warmup 50 次取 p50；三精度列 FP32/FP16/INT8（INT8 自 P3 起）。自家数字必须附复现脚本；不同硬件数字不同表，不混排。

### 4.3 对手数字政策

【设计 D7】对手（YOLOv8/11、RT-DETR、D-FINE、LW-DETR、DEIMv2——均为外部方法，与本项目集成决策无关）数字**只引用官方仓库/论文发表值**，逐条给出处与引用日期；不复现对手（避免复现偏差争议，也省算力）。自家数字全部可复现。每版发布复核对手数字是否更新。

### 4.4 基准表格式与勘误

`docs/benchmarks.md`：一表（AP）/ 一表（延迟）/ 一表（对手出处清单）；**只增不改**，修正进文末勘误节（日期 + 原因）。CI nightly 对自家数字做回归（延迟漂移 >3% 告警，衔接分册 10 R8）。

---

## 5. 研究线与产品线的关系

【设计 D8】研究组件（Dome 密度引导、开放词汇、token indexer、FP8；FSDC/DEIMv2/ECViT 已移除，不在列）进谱系的统一门禁：

1. 研究线 runs 在 ≥2 数据集 × ≥2 seeds 复现增益（VisDrone + COCO 或其子集）；
2. 延迟回退 ≤2%（P0 协议实测）；
3. loss 轨迹可录金标准快照（训练稳定性证据）；
4. 许可审计过（组件与教师权重均需 license 字段）。

FP8 训练（分册 24）先落 train 路径（对所有模型生效），不与谱系发布绑定。开放词汇（分册 27）按同一门禁，作为谱系之上的可选能力层。

---

## 6. 分阶段实施计划

| 里程碑 | 周期 | 内容 | 退出门禁 |
|---|---|---|---|
| **P0 基准锚定** | 1–2 周 | COCO 协议管线（数据已就绪：train 118,287 / val 5,000 官方数核对，见 §1.3；评测口径对拍、三档 recipe 配置化）；`docs/benchmarks.md` v0（对手官方数字 + 出处清单）；延迟协议落地并测自家基线（rf-detr-n / dfine 装配态）；装配器换骨干验证（HGNetV2-S + dfine_s，ECViT 移除后首验）；训练吞吐实测（冻结 §8 算力账）；N 档路线决策冻结（D1b） | 基准表入库；AP/延迟两表均有可复现脚本；nightly 基准回归跑通；N 档路线冻结 |
| **P1 谱系化 + 权重 zoo** | 4–6 周 | 冻结谱系表（D1/D1b）并落组件冻结纪律（D10）；HGNetV2 宽度配方入 `detr/variants.py`（现状尺寸表仅 ViT/DINOv3/ConvNeXt）；单一定义 + 缩放规则（D3，增补分册 03）；S/M/L 装配 warm-start 微调；N 档 12ep 蒸馏（RF-DETR-n / dfine_s 教师，DistillationV3）；权重发布管线（模型卡 / 哈希 / license 硬性）；`detr/{n,s}` 加载闭环；镜像/离线待办收口 | 每档 COCO AP + 三精度延迟进表；权重可下载、哈希钉住、模型卡齐全；N 档达 J3（进表） |
| **P2 易用性** | 3–4 周（可与 P1 并行） | `model.train()/val()/predict()/export()` API 面；CLI `predict`/`val`；coco8 冒烟数据集；数据集自动下载（COCO/VisDrone/VOC）与 YOLO↔COCO 转换 CLI；自动 batch；结果目录约定 + 训练曲线 | README quickstart ≤5 行完成训练→验证→预测→导出；coco8 冒烟 <2min 进 CI（J4） |
| **P3 部署纵深** | 3–4 周 | INT8 PTQ（TRT 熵校准）；FP8 训练路径落地（分册 24，不绑谱系）；动态 shape/batch 导出；CPU ONNX Runtime 路径；Jetson/视频流 + SAHI 公开示例 | N/S 档 FP32/FP16/INT8 三档表公开；INT8 AP 损失 <1%；端到端示例可复现（J5） |
| **P4 演进机制与生态** | 持续；6–8 周/版 | 发行名决策（§11/R10）；setuptools-scm + Keep a Changelog + PyPI 首发 + CI 发版（分册 10 §17 既定蓝图，执行）；MkDocs diataxis 补全（tutorials/how-to/reference + 配置自动导出页）；每版三件套：新权重 / 基准表更新 / changelog；技术报告与 VisDrone 榜单提交；示例 gallery | **P2b 十二文件逐文件重表达专项完成（LEDGER Phase 5 专项，PyPI 发布硬前置）**；0.2.0 出 PyPI；文档站上线；连续两个 minor 按期交付（J6） |

阶段依赖：P0 → P1 严格串行（谱系档位冻结依赖 P0 延迟 profile 与算力实测）；P2 与 P1 并行；P3 依赖 P1 的 N/S 权重；P4 从 P1 起持续。

---

## 7. 总验收 gate（「达到 YOLO 程度」的最终判据）

| 判据 | 验收标准 | 证据物 |
|---|---|---|
| J1 | 五档注册，≥3 档官方权重可下载可复现（算力不足按 §8 裁剪时：注册保五档，权重档数与 J3 档位同步降级） | 注册表 + manifest + 模型卡 |
| J2 | 自家 ≥3 档 + ≥6 个对手官方档位同表，全有出处 | `docs/benchmarks.md` + 复现脚本 |
| J3 | N/S 档 TRT FP16 延迟 ≤ 对手同档 ×1.1，且 AP ≥ 对手 −0.5 | 基准表 + 测量记录 |
| J4 | quickstart ≤5 行 / 3 条 CLI，coco8 <2min | CI 冒烟测试 |
| J5 | N/S INT8 损失 <1% AP；边缘端示例公开 | 对拍报告 + 示例 |
| J6 | semver + changelog + PyPI（发行名决策前置，R10/§11）；连续 2 个 minor 按期 | 发布记录 |

---

## 8. 算力预算与最小可行路径

【设计 D9·估算，P0 实测吞吐后冻结】成本锚：COCO train2017 118k 张；72ep 监督 ≈ 12ep 的 6×；12ep 在线蒸馏（DistillationV3，教师在线前向）约为 72ep 监督的 1/5–1/6。

- P1 训练量（按 v1.1 路线，D9）：N 档 12ep 蒸馏 ×≥2（recipe 迭代）+ N 档 72ep 监督参照 ×1（D1b 增益锚）+ S/M/L 装配 warm-start 短程微调各 ≥1（起点含 COCO decoder 权重，按 0.3–0.5 个 12ep 当量折算）≈ **6–9 个「12ep 当量」**。
- 最小可行：**4×A100（或 ≥300 TFLOPS bf16/卡等价）× 4 周**；舒适：8×A100 × 2 周窗口 × 2。
- 延迟测量：1×T4 或 Orin（可租，按协议 §4.2 固定登记）。
- 算力不足裁剪顺序：砍 L/X → 保 N/S/M → 极限只保 N/S（J1 退到 ≥2 档，J3 不变）。

---

## 9. 风险登记册

| # | 风险 | 概率/影响 | 缓解 | 兜底 |
|---|---|---|---|---|
| R1 | COCO 算力不足 | 高/高 | 12ep 蒸馏路线（≈1/6 成本）；N/S 先行；§8 裁剪顺序 | 只发 N 档，主打 VisDrone/小目标主场叙事 |
| R2 | N/S 档 AP 不及 YOLO 同档 | 中/高 | 蒸馏教师选强（RF-DETR / dfine 官方权重系，§3.1 D1b）；D-FINE 解码器已对齐基线（分册 19） | 对标锚退到 D-FINE/LW-DETR 档位；COCO 表照发不虚饰，差异化叙事转移到 NMS-free/小目标/开放词汇 |
| R3 | 教师/组件权重许可 | 中/高 | 教师选择许可审计先行（RF-DETR / D-FINE / DINOv2 均 Apache；DINOv3 Meta 定制许可走既有 opt-in 门控，不入谱系）；license 字段硬性（D4） | 教师退到 DINOv2（Apache）线 |
| R4 | 导出精度静默劣化 | 中/高 | 沿用双门禁（数值对拍 + 指标对拍，分册 10 R4）；每谱系档 FP16/INT8 黑名单必填 | per-model 豁免登记 |
| R5 | 基准表腐化（对手数字过时/自家漂移） | 中/中 | 勘误节纪律（§4.4）；nightly 自家回归 + 3% 告警；每版复核对手出处 | 季度人工复评 |
| R6 | API/注册键兼容破坏 | 低/高 | 本册兼容承诺 + 分册 02 §4.3 弃用政策；谱系键纯新增 | 旧键 alias 一个版本（R16 更名先例） |
| R7 | 维护带宽不足，节奏失守 | 中/中 | P4 项目优先级裁剪表；自动化发版减少人工 | minor 周期 6–8 周降级到 12 周，并在文档公示 |
| R8 | AGPL 边界失守（ultralytics 对比基线时混入） | 低/高 | 既有 extras 隔离 + import-linter 门禁不动摇；产品谱系零 AGPL 组件；对比表只引用 YOLO 官方数字，不 import 其代码 | 依赖替换预案（分册 10 R2） |
| R9 | 研究线方向漂移带动谱系组件变动（先例：2026-10-15 一日移除三条线、两周两次改名） | 中/高 | 组件冻结纪律（D10）：谱系只进许可审计过的成熟组件；组件变动走 minor + 新权重条目 | N 档蒸馏配方独立于研究线，可单独维持 |
| R10 | PyPI 发行名被占用（`autovision` 为 2021 年休眠项目，§11） | 高/中 | P4 启动前决策发行名（默认倾向 `autovision-cv`，import 名不变）；同步核对 conda/docker 镜像同名占用 | (b)/(c) 备选路径（§11） |
| R11 | "AutoVision" 商标拥挤（Omron AutoVISION 在先使用 + 美加同类申请，§11） | 中/高 | 对外发布/营销前商标 counsel 清查；此前对外一律以内部代号表述（如 "Autovision 训练框架"） | 更名预案（随 R10 一并决策） |

---

## 10. 维护约定

- **与分册 10 的关系**：本册 P0–P4 细化并取代分册 10 §18 中「M4+ 检测家族补全」条目的排期依据；GA 语义与交付形态沿用分册 10 §16–§17。
- **版本节奏**：GA 前 MINOR 每 6–8 周（分册 10 §17）；带宽不足按 R7 预案降级并公示。
- **基准勘误**：`docs/benchmarks.md` 只增不改；修正进勘误节并注明日期与原因。
- **模型卡**：每权重一份（字段见 §3.4）；权重更新 = 新条目，不覆盖旧条目。
- **谱系组件冻结（设计 D10）**：骨干/解码器/宽度配方一经 P1 冻结，不随研究线漂移；组件替换或配方变更 = minor 版本 + 新权重条目 + 基准表新列，不原地覆盖。
- **跨册影响**：谱系键、API 面、导出格式变化按分册 02 §4.3 弃用政策执行，并同步受影响分册与 ADR。

---

## 11. 命名与发布合规风险（2026-10-15 核查，发布前必读）

品牌定名 **AutoVision**（全库 VTrain/Autovision 已统一，见 `docs/migration-0.2.md`）。两项外部核查结果**直接影响 J6 的 PyPI 判据**（已并入风险登记册 R10/R11）：

1. **PyPI 名称被占用**：`pypi.org/project/autovision` 已存在（2021 年休眠项目，v0.0.1，1.1 kB）。`autovision` 无法作为 PyPI 包名首发。可选路径：(a) 换发行名（如 `autovision-cv`），import 名保持 `autovision`；(b) 联系占用者（休眠项目，转让概率低）；(c) 更名。**需在 P4 启动前决策**（R10）。
2. **商标拥挤**："AutoVision" 在视觉软件领域已有在先使用——Omron Microscan **AutoVISION**（工业机器视觉配置平台，与本项目领域直接相邻），另有美国（75330426，类别 9）与加拿大（1877720）的 AUTOVISION 软件类商标申请。**对外发布与营销前需商标 counsel 出具清查意见**；在此之前避免将 "AutoVision" 用作对外商品名（内部代号不受影响）（R11）。

---

## 12. 修订记录

- **v1.1（2026-10-15）**：① 谱系表修订——ECViT 骨干线移除（commit `588530d`/`d23b6cb`），n/s 档候选改 HGNetV2-N/S，M/L 档 ConvNeXt，增「起点路线」列；② 新增 D1b：`dfine_n` 官方权重不存在（LEDGER Phase 4 实测），N 档定为蒸馏优先档，P0 内冻结决策；③ 教师池同步——DEIMv2 移除（commit `0d8c0ec`），教师收敛为 RF-DETR / DINOv2 / dfine 官方权重；④ P4 增硬前置门禁：P2b 十二文件逐文件重表达专项（LEDGER 定向复扫结论：结构性相似，改名无法稀释）；⑤ 新增 D10 谱系组件冻结纪律与风险 R9（方向漂移）；⑥ 现状盘点同步：COCO2017 数据就绪、检测家族清单更新。v1.0→v1.1 期间仓库事实基础：commit `3833f5a`…`d23b6cb`（task-first 重构、R16 二次更名 `avdetr→detr`、三条线移除、Apache 装配链路验证）。
- **v1.1 同日复核修正（「分析-修复-完善-优化-测试」轮）**：① 测试基数 1503→1505（LEDGER Phase 4 终验）；② 关联分册补 30；③ P1 补「HGNetV2 宽度配方入 `detr/variants.py`」缺口项（现状尺寸表仅 ViT/DINOv3/ConvNeXt，grep 实证）；④ §7 J1 增 §8 裁剪降级注记，消除「≥3 档权重」与「极限只保 N/S」的表述冲突；⑤ §8 算力账按 v1.1 路线重写（装配微调 0.3–0.5 当量折算）；⑥ §4.1 72ep 参照用途与 D1b 挂钩；⑦ 环境变量名经代码核实（`runtime/env.py`、`encoders/weight_verification.py`）；⑧ 命名合规核查并入为 §11（原追加节节号与风险登记册 §9 冲突且位于修订记录之后，重排为 §11/§12，登记 R10/R11，J6 与 P4 挂钩）。
- **P1 代码侧实施记录（2026-10-15）**：谱系注册表落地——`detr/variants.py` 增 timm HGNetV2 宽度配方（`hgnetv2-b0/b1`；宽度经缓存 dfine_s checkpoint **实证**：hidden 256 / enc FFN 1024 / dec 3 层 / points (4,4,4)）与谱系键 `detr/{n,s}`（D10 冻结元数据 `SPECTRUM`；m/l/x 声明 planned、不占注册键）；`TimmCNNBackbone`（timm features_only 三阶金字塔 + OFFLINE/MIRROR hub 环境桥 `bridge_hub_env`）；dfine_official 装配器重建（形状驱动 + decoder 覆盖闸；实测 dfine_s→detr/s **172 张量、decoder 覆盖率 1.0、0 形状冲突**）；AVWarmStart v1 训练侧消费入口（`TrainArgs.warm_start` → `load_warm_start_into_network`，license 硬性 R7 + 覆盖闸 0.5）；权重发布管线 `registry/model_card.py`（license 必填、sha256 64-hex、延迟精度白名单）；`docs/benchmarks.md` v0（D5/D6/D7 协议脚手架）；配方脚本 `scripts/train_spectrum_n_distill.py`（**蒸馏模式诚实拒绝**——doc 23 教师未接线，列为 P1 remaining）与 `scripts/finetune_spectrum_s.py`（CUDA/COCO/权重三门禁）。测试 `tests/test_p1_spectrum.py` ×20（真 dfine_s 装配、license 拒载、detr/{n,s} coco8 训练闭环；10.3s）；detection 回归 147 过；ruff/layering/provenance 全绿。
- **本地权重资产盘点（2026-10-15，`Dino-Dome-Research/weight/`）**：① **GTR 家族**（Intellindust-AI-Lab「Gated Token Recurrence」，GitHub 公开，checkpoint **MIT**）——det/seg/pose/obb/depth/semseg × S/M/L/X 全任务，COCO det AP S=53.6/M=57.3/L=58.9/X=59.4（EMA 口径）；键形实证：decoder 与 `dfine_official` **同构**（hidden 192、heads 8、points (4,4,4)），骨干为自研线性注意力块（`blocks/norm/patch_embed`，非 HGNetV2）；对谱系的意义：**N 档蒸馏教师候选升级**（MIT、本地、四档齐，已记入 `SPECTRUM["n"].note` 与 N 脚本门禁信息）；hidden-192 变体可高覆盖装配 GTR decoder（方向决策待定，未擅动谱系）。② `dinov2|dinov3|edgecrafter/*-ltdetr-coco`（Lightly 产出 legacy 基线，约 11 个）——`detector_checkpoint` 既有消费入口，**仅限本地研究，发布红线**（LEDGER P3-1 待替换项）；DINOv3 骨干叠加 Meta 许可（D1-a 用户自持）。③ `dinov3/convnext-{tiny,small,base,large}`（无 ltdetr 后缀）——Meta DINOv3 骨干权重（D1-a opt-in 线）。

