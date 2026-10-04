# VTrain 实现蓝图 · 分册 01：概述 · 质量属性 · 总体架构

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §1–§3 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 1. 概述

### 1.1 愿景与定位

VTrain 是一个**企业内部的视觉模型训练基础设施**，覆盖模型开发全生命周期：

```
无标注图像 ──► 预训练/蒸馏 ──► 骨干权重 ──► 下游微调 ──► 统一评测 ──► 导出部署 ──► 模型资产库
                (Method)      (backbone)    (TaskModel)   (EvalProtocol) (ONNX/TRT)  (Registry)
```

与 lightly-train 0.17.0 的定位差异（基于源码核验）：

| 维度 | lightly-train 0.17.0【基线】 | VTrain【设计】 |
|---|---|---|
| 商业模式/许可 | AGPL-3.0 本体；`LICENSE_INFO` 文案注入每个 checkpoint 与 ONNX 元数据（`_license.py:12`） | 企业内部平台，Apache-2.0 本体，AGPL 依赖物理隔离 |
| 任务范围 | 7 个微调 API：图像分类（单头/多头、多类/多标签）、实例/语义/全景分割、目标检测、旋转框检测、深度估计（`_commands/train_task.py`） | 检测首发，任务注册表按需扩展 |
| 评估能力 | 预训练期仅 `train_loss` 及分量（`_methods/method.py:132`）；微调期 torchmetrics mAP（`_metrics/`）；**无独立 evaluate 命令、无预训练质量探针** | 内置 kNN/线性探针 + 统一评测协议 + `evaluate`/`benchmark` 命令 |
| 引擎形态 | 双形态：预训练用 `pytorch_lightning.Trainer`（`_commands/train_helpers.py:203`），任务微调用 `lightning_fabric.Fabric` 手写循环（`_commands/train_task.py:1705`） | 单引擎薄封装（ADR-0001，理由更新见 §10.1） |
| 供应链 | 权重下载**零哈希校验**（三处 TODO：`_models/dinov3/dinov3_package.py:409` 等）；遥测默认**开** | sha256 强校验；遥测默认关 |
| 工程体系 | 5 个 CI workflow、无 GPU runner、无覆盖率配置、版本号静态字符串手动发版 | 完整 CI/CD 门禁、ADR 治理、SLA、全自动发版 |

### 1.2 干系人与用户画像

| 角色 | 使用方式 | 核心诉求 |
|---|---|---|
| 算法研究员 | Python API 写训练脚本、改方法/模型 | 改动局部化、实验快速迭代、结果可复现 |
| 应用工程师 | CLI/YAML 调用现成配方、导出部署 | 报错友好、文档全、产物可部署 |
| 平台/MLOps | K8s/Slurm 上批量跑训练、监控 | 容错、可观测、资源利用率、统一镜像 |
| 框架研发 | 维护框架本体 | 清晰的模块边界、测试体系、ADR 流程 |
| 安全合规 | 审计许可、数据、供应链 | 许可矩阵、SBOM、数据治理留痕 |

### 1.3 范围

**P0（GA 必须交付）**

- 目标检测微调：COCO/YOLO 数据格式，DETR 系（RT-DETRv2/D-FINE 路线，参照基线 LT-DETR）+ 轻量 dense 头路线（参照 PicoDet）
- 蒸馏预训练：任意教师（含 DINOv2/v3）→ 任意学生骨干（对标基线 DistillationV3）
- 统一评测协议 + benchmark 对比命令
- 导出：ONNX（动态 batch/FP16 + 数值对拍门禁）+ TensorRT + SAHI 切片推理
- 单机多卡 + torchrun 多节点
- 完整 CI/CD、测试体系、文档站、Docker 交付

**P1（GA 后两个 minor 版本内）**

- 语义/实例分割任务、分类任务
- 模型注册表后端（MLflow/文件系统）与 promotion 门禁
- 华为 Ascend NPU 适配、8-bit 优化器（基线已有 bitsandbytes 适配先例，`_optim/adamw8bit_args.py`）、int8 PTQ
- OpenTelemetry 全链路追踪

**明确不做**

- 数据标注/可视化平台（集成外部工具，不自研）
- 在线服务/SaaS 多租户
- 通用 AutoML/NAS
- LLM/多模态

### 1.4 技术选型基线

| 领域 | 选型 | 理由 |
|---|---|---|
| 深度学习 | PyTorch ≥2.4 | 基线下限 torch≥2.1（`pyproject.toml:31`）；VTrain 需要 `torch.export` + dynamo ONNX（基线要求 ≥2.5，Dim.STATIC 需 ≥2.6，`_export/onnx_helpers.py:27-54`），故取 2.4 为下限、以 2.6 为推荐 |
| 训练引擎 | PyTorch Lightning（薄封装） | 基线验证了 Trainer（预训练）与 Fabric（任务）双形态；VTrain 单引擎封装，保留切换自由（ADR-0001） |
| 配置校验 | Pydantic v2 | 基线全量使用（但 pydantic 是经由 `lightly` 的传递依赖，未直接声明）；VTrain 直接声明依赖 |
| 数据增强 | albumentations ≥2.0 | 基线约束为 `>=1.3.1,!=1.4.*`（1.4 有 bug、1.3 被 SuperGradients 拖住，`pyproject.toml:27-28`），并在代码里散落 `albumentations≥2.0` 才可用的分支（如 geometry 记录，`_transforms/view_transform.py:204`）。**教训成立但需修正表述：不是锁死旧版，而是兼容矩阵拖累**。VTrain 锁 ≥2.0 并建适配层 |
| 评测 | torchmetrics detection | 基线用 `MeanAveragePrecision`，backend 可选 pycocotools/faster_coco_eval（`_metrics/mean_average_precision.py:23-29`）；VTrain 延续并封装统一协议 |
| 包管理 | uv + lockfile | 基线已验证该体系：31MB `uv.lock` 入库 + `uv lock --exclude-newer <日期>` 冻结解析（`Makefile:299-313`）；VTrain 沿用并加 Renovate |
| CI | GitLab CI（平台无关模板） | 矩阵化多平台测试 |
| 文档 | MkDocs Material（diataxis 结构） | 基线为 Sphinx+furo+autoapi，配置参考靠 `docs/prebuild.py` 把 Pydantic 默认值导出为 md（值得借鉴）；VTrain 用 MkDocs + 自研 schema 导出 |
| 可观测 | structlog + NVML + 可选 OTel | 基线：nvidia-ml-py 已是核心依赖（`pyproject.toml:36`）、CUDA 利用率 0.2s 采样线程（`_training_step_timer.py:172`）；VTrain 升级为结构化日志 + 一等指标 |

### 1.5 成功指标

| 指标 | 目标 |
|---|---|
| 新任务（如分割）接入框架的开发量 | ≤ 1 个子包（~2 人周），不改编译/核心代码（基线参照：PicoDet 检测 = `_task_models/picodet_object_detection/` 独立子包 + 共享组件） |
| 新骨干模型接入 | 仅实现适配协议（~1 文件），注册即可用（基线参照：torchvision 家族 3 个 wrapper 文件） |
| 从数据到可部署 ONNX 的命令数 | ≤ 4 条 |
| 同配置复跑结果 | 确定性模式下逐 step loss 一致 |
| 训练 24h 中断后恢复 | ≤ 5 分钟（自动 resume），无数据损坏（基线已具备 resume 全状态恢复，`_train_task_state.py` + `fabric.load`） |
| 框架层吞吐开销 | 相对裸 PyTorch 循环 < 5% |
| 文档 | 任一公开 API 有 docstring + 类型标注，覆盖率 100%（基线 mypy strict 全量标注可作工程模板） |

---


## 2. 质量属性与非功能需求（NFR）

企业级的定义：以下每一条都有**可测量的验收标准**，并落入 CI/CD 门禁或运维 SOP。与基线的差距即实现重点。

| # | 质量属性 | 需求 | 验收方式 | 基线现状（差距来源） |
|---|---|---|---|---|
| N1 | **可靠性** | 瞬时故障后自动恢复训练；checkpoint 原子写 | 故障注入测试；resume 后指标连续 | 原子写（mmap tmp+rename、ckpt 由 torch.save）已具备；自动拉起靠外部调度器，框架内无看门狗/心跳 |
| N2 | **可复现性** | 同 seed + 同环境 + 确定性模式 → 逐 step loss 一致；run 100% 携带环境指纹 | 数值金标准回归测试（§13 L4） | seed_everything(workers=True) 已有；但 **checkpoint/run 不存 seed、配置不落盘**，环境信息只进日志 |
| N3 | **性能** | 框架开销 < 5%；GPU busy ≥ 90%；性能劣化 > 3% 阻断合入 | nightly 吞吐基准趋势门禁 | 基线有 data_wait、profiling/* 指标与吞吐打印（`train_task_helpers.py:1132`），无趋势基线与门禁 |
| N4 | **兼容性** | 支持矩阵内全绿；弃用周期 ≥ 2 minor 且带 DeprecationWarning | 矩阵 CI + 弃用 Lint | 基线弃用实践成熟（`resume`→`resume_interrupted`、`reuse_class_head` 均带运行时告警），但无自动化矩阵（仅 3.8-minimal vs 3.13-maximal 两轴） |
| N5 | **安全** | 依赖无阻断级漏洞；权重 100% 哈希校验；核心包零 AGPL import | 安全扫描 + import 边界检查 | 基线：权重下载零校验、遥测默认开、ultralytics 靠 Makefile 排除出镜像 |
| N6 | **质量** | core 行覆盖 ≥ 85% 且增量不降；mypy strict；ruff 全绿 | PR 门禁 | 基线 mypy strict + ruff 全量；**无覆盖率工具链** |
| N7 | **可观测** | 每 run 100% 产出结构化元数据（配置/git/数据指纹/环境） | run 目录审计脚本 | 基线只有日志与 logger hyperparams，无机器可读 run 元数据 |
| N8 | **可维护** | 依赖单向、边界清晰；破坏性变更走 ADR；vendor 白名单制 | 架构守卫测试（import-linter） | 基线边界靠约定 + vendored 代码 license 头（`dev_tools/*.tmpl`），无自动守卫 |
| N9 | **交付效率** | PR 门禁 < 30 min；发版全自动 | CI 时长统计 | 基线 PR 门禁轻量（≤4 workflow），发版手动 |
| N10 | **可移植** | 同一份代码跑开发机/Slurm/K8s；数据源支持本地/NFS/S3 | 交付形态验收 | 基线 Slurm 感知（CPUS_PER_TASK、rank 探测链）、fsspec 读 YAML；无 K8s 模板与 S3 数据面 |

---


## 3. 总体架构

### 3.1 分层架构

```
┌────────────────────────────────────────────────────────────────────┐
│ L6 接口层    Python API (vtrain.*) ｜ CLI (typer 薄壳) ｜ YAML 配置   │
├────────────────────────────────────────────────────────────────────┤
│ L5 命令层    pretrain ｜ finetune ｜ evaluate ｜ benchmark ｜        │
│              export ｜ predict ｜ load_model ｜ list_*               │
│              （每命令 = 三段式: config 校验 → 组装 → 驱动）            │
├────────────────────────────────────────────────────────────────────┤
│ L4 核心域层                                                        │
│   Method        预训练方法(LightningModule)                          │
│   TaskModel     推理态任务模型(纯 torch, 三段式, 可导出)              │
│   TrainModel    训练态包装(TaskModel+criterion+metrics+参数组)       │
│   ModelWrapper  骨干适配协议(ModelWrapper/MultiScale 两个组合面)      │
│   DataModule    数据模块(格式适配+mmap 清单)                         │
│   EvalProtocol  统一评测协议                                         │
├────────────────────────────────────────────────────────────────────┤
│ L3 组件层    _configs ｜ _transforms ｜ _optim ｜ _metrics ｜        │
│              _checkpoint ｜ _export ｜ _sahi ｜ _loggers ｜ _debug   │
├────────────────────────────────────────────────────────────────────┤
│ L2 运行时层  _engine(Lightning 薄封装)｜ _distributed ｜ _fault      │
│              (容错: 原子ckpt/看门狗/auto-resume)｜ _device           │
├────────────────────────────────────────────────────────────────────┤
│ L1 基础设施层  _storage(fsspec)｜ _cache ｜ _registry(模型资产)      │
│                ｜ _telemetry ｜ _license                             │
├────────────────────────────────────────────────────────────────────┤
│ L0 外部依赖   torch ｜ lightning ｜ albumentations ｜ torchmetrics ｜│
│               pycocotools ｜ [extras] timm·ultralytics·onnx·tensorrt │
└────────────────────────────────────────────────────────────────────┘
   依赖规则：只允许上层 import 下层；L4 域层不 import L5/L6（import-linter 强制）
```

【基线】lightly-train 的实际分层与此同构，但有三个值得注意的「越层」事实，VTrain 立为反例：

1. 命令层直接驱动运行时：`_commands/train_task.py` 的手写 Fabric 循环（`train_task.py:1705-2001`）承担了训练循环、grad-accum、EMA 钩子、验证调度、checkpoint 调度的全部编排——这是基线最复杂的 700 行，也是行为回归最集中的地方。VTrain 必须把该循环收进 `_engine`，命令层只做组装。
2. 方法层反向依赖命令层辅助：`_methods` 与 `_commands/train_helpers.py` 互相调用（方法注册表被命令层消费，`MethodArgs` 的 `ClassVar` 被 `get_epochs` 消费）。VTrain 把「方法元数据」（default_steps/default_epochs/参考 batch）收进 L4 的方法描述对象，命令层只读。
3. transform/数据层知道任务模型细节：`_task_models/ltdetr_object_detection/transforms.py` 里定义检测变换默认值、被 `_transforms/ltdetr_transforms/` 的协议反向约束。VTrain 以「家族自带 transforms + 框架只定义协议」为规则，与基线一致但写入 import-linter。

### 3.2 架构原则

1. **API 即契约**：公共 API 是唯一稳定面，内部模块随时可重构；公共函数 = 「kwargs → Pydantic 校验 → from_config()」三段式（基线已验证该模式：`pretrain()` 的函数体就是 `pydantic_model_validate(TrainConfig, locals())` + `train_from_config`，`_commands/train.py:241-242`）。
2. **`"auto"` 惰性解析**：超参默认 `"auto"`，按依赖链在命令流中顺序解析（精确顺序见 §5.3），最终配置落盘审计。
3. **协议优先于继承**：跨边界扩展点（模型/数据集/评测/日志）一律 `Protocol` + 注册表。基线为 `@runtime_checkable` Protocol（模型层）+ 类级 `ClassVar` 钩子（任务模型层）+ 纯 dict 注册表（方法/优化器层）三种形态并存——VTrain 统一为 Protocol + 注册表，但保留基线「结构化子类型、不要求继承」的精髓。
4. **训练/推理解耦**：TaskModel（推理三段式）与 TrainModel（训练壳）分离；导出物只依赖前者。基线对此的执行非常严格：`TaskModel` docstring 明言 "Must be pure PyTorch and not rely on Fabric or Lightning"（`_task_models/task_model.py:31`）。
5. **适配器隔离**：许可证不兼容（AGPL）或 API 不稳定的依赖物理隔离到 `_adapters/` 子包。基线先例：ultralytics 因 AGPL 被 Makefile 显式排除出 Docker 镜像（`Makefile:283-297` 注释 "as it has an AGPL license and we should not distribute it"）。
6. **vendor 白名单制**：确需 vendor 的第三方代码放入 `_vendor/<name>/`，附 `ORIGIN.md`（上游 repo+commit、本地改动清单、升级脚本），CI 检查。基线先例：DINOv2/DINOv3/EoMT/RT-DETR/D-FINE/PicoDet 等十余处 vendored 代码全部带 license 头模板（`dev_tools/licenseheader.tmpl` + 各专属模板，`licenseheaders` 工具在 CI 校验头存在）。
7. **一切皆可审计**：run、checkpoint、导出物、数据集均携带 schema 版本与元数据。基线 checkpoint 已内嵌 `lightly_train` 元数据块（`_checkpoint.py:29`），但无 schema_version、配置不落盘——VTrain 补齐。
8. **引擎可替换**：Lightning 藏在 `_engine` 薄接口后，接口按「我们要什么」（fit/resume/分布式/日志钩子/step 级回调）定义，不按 Lightning 的类结构定义（§10.1 论证为何必须抽象）。
9. **同名模块三分法（命名即领域）**：同一上游模型名（dinov2/dinov3/eomt 等）会在三个领域各出现一次，靠包路径前缀区分职责，依赖单向、禁止跨域混用：

   ```
   ┌────────────────────────────────────────────────────────────────────┐
   │ _methods/<name>/            训练配方：预训练方法(LightningModule)     │
   │                             只关心「怎么训」                        │
   ├────────────────────────────────────────────────────────────────────┤
   │ _models/<name>/             骨干封装：backbone wrapper + 权重供应链   │
   │                             只关心「模型本体」(叶子层)              │
   ├────────────────────────────────────────────────────────────────────┤
   │ _task_models/<name>_eomt_*/ 部署面任务模型：纯 torch 三段式、可导出    │
   │                             只关心「怎么推理/导出」                 │
   └────────────────────────────────────────────────────────────────────┘
      依赖方向：_methods → _models ← _task_models（_task_models 不 import
      训练机制，原则 4 的执行细则；import-linter 强制）
   ```

   定位规则：找「dinov2 的 X」先按领域定包——训练行为在 `_methods/`，模型结构/权重在 `_models/`，推理/导出在 `_task_models/`。例如 `_methods/dinov2/`（DINOv2 自监督配方）与 `_task_models/dinov2_eomt_semantic_segmentation/`（EoMT 语义分割部署模型）共享 dinov2 骨干但互不感知。

### 3.3 核心领域模型（与基线类名对齐）

```text
ModelSpec          模型声明： "pkg/name" ｜ 本地路径 ｜ nn.Module ｜ ModelWrapper 实例
                   【基线】parse_model_name: split("/", maxsplit=1)，dinov2_vit→dinov2 兼容改名
                   （_models/package_helpers.py:143-157）；custom 实例 pickle 进 checkpoint
Method             预训练方法实例(LightningModule 子类)：training_step_impl(batch)->TrainingStepResult
                   【基线】TrainingStepResult(loss, log_dict)——注意是单一已求和 loss，非分量字典
TaskModel          推理态：preprocess_image/preprocess_batch → forward_backend → postprocess
                   【基线】持有 _init_args，暴露 class_path；deploy() 切换部署模式
TrainModel         训练态：training_step(fabric,batch,step)->TaskStepResult；get_optimizer；
                   get_export_state_dict（EMA 权重优先）；方法名刻意与 Lightning 一致以配合
                   Fabric 的 _FabricModule 分发（_task_models/train_model.py:89-100 注释）
DatasetSpec        数据声明：格式+路径+划分+评测配置【设计】；基线为 TaskDataArgs + 隐式
                   data_config_file 字段（exclude=True，用于相对路径解析）
EvalProtocol       评测协议【设计】；基线最接近物 = TaskMetricArgs(watch_metric) + torchmetrics
Run                一次训练/评测的产物容器【设计】；基线 = out 目录约定（§11.1）
Artifact           可持久化资产【设计】；基线事实：ckpt 与 exported .pt 两种（§10.5/§12.1）
ModelRecord        模型资产库条目【设计】；基线仅有 DOWNLOADABLE_MODEL_URL_AND_HASH 注册表
                   （_task_models/task_model_helpers.py:92-162）
```

### 3.4 ADR（架构决策记录）治理

- 所有**不可逆/影响面大**的决策必须落 ADR，存 `docs/adr/NNNN-<title>.md`；模板：背景 → 备选方案 → 决策 → 后果 → 回退路径。
- 本文档关键决策实现期各建 ADR。首批清单：`0001-训练引擎选型-单引擎封装.md`、`0002-checkpoint-与导出物-schema.md`、`0003-单视图蒸馏配方.md`、`0004-vendor与许可边界.md`。
- 破坏性变更 PR 必须引用对应 ADR 编号，否则 review 拒绝。

---
