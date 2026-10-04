# VTrain 企业级视觉训练框架设计文档（实现蓝图）· 总览

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图；自 v3.0 起按子系统拆分为 11 个分册，本文件为总览与索引） |
| 状态 | 评审稿 → 开发基线 |
| 日期 | 2026-09-25 |
| 参考基线 | lightly-train 0.17.0 源码全量剖析（415 个源文件，7 个子系统，关键机制逐条核验，出处标注到 `文件:行`） |
| 读者 | 框架研发团队、算法团队负责人、平台/基础设施团队、安全合规 |
| 占位名 | VTrain（正式命名待定） |

> **设计目标一句话**：构建一个对标 lightly-train 的企业级视觉模型训练框架——一条命令覆盖「自监督预训练/蒸馏 → 下游微调 → 统一评测 → 导出部署」，且在质量工程、安全合规、可观测性、发布治理四个维度达到企业内部可长期维护、多团队共用、可交付生产的标准。
>
> **v3.0 相对 v2.0 的变化**：以 lightly-train 0.17.0 的**实测源码**为基线重写为实现蓝图。每个子系统先给出「基线是怎么实现的」（精确到签名、公式、默认值、调用链），再给出「VTrain 怎么实现」（模块布局、类契约、数据结构）。v2.0 中与源码不符的表述已修正（修正清单见分册 11 附录 A.0）。
>
> **v3.1 变化**：按子系统拆分为分册。正文 § 编号保持**全局章节编号**不变，跨册引用（如「见 §7.3」「附录 A」）仍然有效；章节与分册的映射见下表。

**标注约定**（全部分册适用）：

- **【基线】** lightly-train 0.17.0 已验证事实，附源码出处（路径相对 `src/lightly_train/`）。
- **【设计】** VTrain 自己的决策，基线中不存在。
- **【差异】** 有意偏离基线的决策，附理由与 ADR 编号。

---

## 分册目录

| 册 | 文件 | 内容 | 全局章节 |
|---|---|---|---|
| 01 | [01-overview-and-architecture.md](design/01-overview-and-architecture.md) | 愿景与定位、干系人、范围、技术选型、成功指标；NFR 十条（含基线差距）；分层架构、八条架构原则、核心领域模型、ADR 治理 | §1–§3 |
| 02 | [02-public-api-and-config.md](design/02-public-api-and-config.md) | 基线 API 面与启示、VTrain API 面、版本弃用、插件系统；Pydantic 严格校验、三配置方言、auto 解析链与顺序、四类注册表 | §4–§5 |
| 03 | [03-model-layer.md](design/03-model-layer.md) | ModelWrapper 协议族、Package 注册表与 9 包机制、DINOv2/v3 vendoring 与权重供应链、检测任务协议 | §6 |
| 04 | [04-pretraining-methods.md](design/04-pretraining-methods.md) | Method 基类契约（含 lr 缩放/无 wd 清单）、方法注册表、DistillationV3 完整数值规格、其他方法速览、kNN 探针设计 | §7 |
| 05 | [05-task-models.md](design/05-task-models.md) | TaskModel/TrainModel 双层契约、检测家族损失与训练参数、参数组结构、类别头迁移、评测指标与 EvalProtocol | §8 |
| 06 | [06-data-layer.md](design/06-data-layer.md) | 格式与统一标注表示、mmap 文件清单方案、批组装与 dataloader、规模分桶桶表、检测变换与 StepScheduledCompose | §9 |
| 07 | [07-runtime-and-experiments.md](design/07-runtime-and-experiments.md) | 双引擎事实与单引擎决策（四原语）、分布式 rank 链、精度与数值健康、回调、checkpoint/状态格式、容错、run 目录与 logger 体系 | §10–§11 |
| 08 | [08-assets-and-deployment.md](design/08-assets-and-deployment.md) | 便携模型格式与 load_model、ONNX 导出管线（FP16 黑名单+对拍门禁）、TensorRT、SAHI、embed、模型注册表 | §12 |
| 09 | [09-quality-security-observability.md](design/09-quality-security-observability.md) | 基线测试体系实况与七层测试策略、门禁；安全合规基线反例与 VTrain 措施；观测能力与 OTel 设计 | §13–§15 |
| 10 | [10-process-roadmap-risks.md](design/10-process-roadmap-risks.md) | 支持矩阵与交付形态、工程组织、里程碑与基线移植映射、风险登记册 | §16–§19 |
| 11 | [11-appendix-baseline-analysis.md](design/11-appendix-baseline-analysis.md) | 机制对照总表（含 v2.0→v3.0 修正清单）、基线源码地图、环境变量全表、术语表 | 附录 A–D |

## 章节快速定位

| 章节 | 分册 | 章节 | 分册 |
|---|---|---|---|
| §1 概述 · §2 NFR · §3 总体架构 | [01](design/01-overview-and-architecture.md) | §9 数据层与数据治理 | [06](design/06-data-layer.md) |
| §4 公共 API · §5 配置系统 | [02](design/02-public-api-and-config.md) | §10 训练运行时 · §11 实验管理 | [07](design/07-runtime-and-experiments.md) |
| §6 模型层 | [03](design/03-model-layer.md) | §12 模型资产与部署 | [08](design/08-assets-and-deployment.md) |
| §7 预训练方法层 | [04](design/04-pretraining-methods.md) | §13 质量 · §14 安全 · §15 可观测 | [09](design/09-quality-security-observability.md) |
| §8 任务模型层 | [05](design/05-task-models.md) | §16–§19 交付/流程/路线图/风险 · 附录 A–D | [10](design/10-process-roadmap-risks.md) / [11](design/11-appendix-baseline-analysis.md) |

## 阅读指引（按角色）

| 读者 | 建议路径 |
|---|---|
| 框架研发（实现） | 01 → 02 → 按负责子系统精读 03–08 → 09/10（门禁与路线图）→ 11（实现时逐条对照基线） |
| 算法团队 | 01 §1（定位与范围）、04（蒸馏配方与数值）、05（任务模型）、11 附录 A.0（修正清单） |
| 平台 / MLOps | 01 §2（NFR）、07（运行时/容错/run 目录）、09 §15（可观测）、10 §16（交付形态） |
| 安全合规 | 09 §14（安全合规基线反例）、06 §9.1（数据治理）、11 附录 C（环境变量面） |

## 维护约定

- 分册内修改按各分册自评审；**跨章影响**（如 checkpoint/导出格式、公共 API 签名、全局章节编号）必须同步受影响分册并按 §3.4 落 ADR。
- 新增内容优先扩展既有分册；只有新子系统才新增分册，并同步更新本索引的「分册目录」「章节快速定位」。
- 引用规范：`§N.M` 一律用全局章节编号；基线出处写 `文件:行`（相对 `src/lightly_train/`）；分册间链接用相对路径 `xx.md`。

---

*本文档为 VTrain 实现蓝图。所有【基线】条目已在 lightly-train 0.17.0 源码上核验（含数值与行号），实现期可直接作为移植与对照基准；【设计】条目按 §18 路线图落入里程碑，关键决策按 §3.4 落 ADR；NFR 表（§2）是 GA 验收的唯一标准来源。*
