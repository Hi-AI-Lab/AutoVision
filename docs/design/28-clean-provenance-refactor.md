# VTrain 实现蓝图 · 分册 28：清洁溯源重构——代码、目录与架构去 lightly-train 影响设计

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0（设计评审稿，待实施） |
| 状态 | 设计完成；实施计划见 §8，总验收 gate 见 §9 |
| 审计依据 | 2026-10-14 三路代码溯源审计（逐行 diff + token 级相似度 + 权重/资产审计，方法与复现见附录 C） |
| 关联分册 | 01（总架构）、09（质量与安全）、11（基线附录）、19（解码器基线对齐）、20（DINOv3 ViT masked forward） |
| 受影响模块 | `_methods/dinov3/`、`_models/ecvit/`、`_task_models/object_detection_components/`、`_task_models/ltdetr_object_detection/`、`_transforms/`、`_registry/`、`_embedding/`、`scripts/`、全库包结构（§4） |
| 兼容承诺 | 公开 API 键名（`model="…"`、`method_args`、`pretrain/train_*` 等）保持稳定；checkpoint 的 state-dict 张量布局不变；包结构迁移后提供旧路径 remap（§4.3）；切换期间旧 warm-start 权重加载路径保留至 Phase 4 验收 |

> **标注约定**：【审计】溯源审计已核实的事实（出处 `文件:行`，路径相对 `src/autovision/` 或 lightly-train 仓库根）；【现状】本仓库现状；【设计】本文档决策（附编号 R1–R… 及理由）；【上游】宽松许可参考源；【风险】风险与缓解。

---

## 1. 背景与问题定义

### 1.1 三层影响面

本仓库目标许可是 **Apache-2.0**（仓库根 `LICENSE`）。参考系统 lightly-train 0.17.0 是 **AGPL-3.0**（强传染：混入衍生表达则整个作品分发时必须整体转 AGPL，且不可逆）。2026-10-14 溯源审计确认影响面分三层：

**第一层：代码衍生（法律层）。** 全库无 `import lightly` 依赖、无整文件逐字复制、测试无 fixture 拷贝；但存在约 15 个文件与 lightly-train 对应文件构成"近似复述/局部逐字"的衍生关系。按处置难度分三档：

| 档 | 判定 | 文件 | 依据 |
|---|---|---|---|
| 🔴 P0 | lightly 原创、近乎逐字、**无宽松上游** | `_methods/dinov3/dinov3.py` 的 `_align_cross_view_pair`（:439-500）与 `_roi_align_view`（:502-545）；`_methods/dinov3/constrained_crop.py` 的 `sample_high_overlap_box`（:187-215） | token 相似度 41.4% / 36.1%，最长逐字块 36 行；lightly 对应文件仅带 Lightly AGPL 版权头 |
| 🟡 P1 | 逐字度高，但**存在 Apache-2.0 原始上游**（lightly 自己就是从上游 vendored 的） | `dfine_official_decoder.py`（~65%）、`ecvit.py`（~72%）、`hybrid_encoder.py`（~63%）、`rtdetrv2_decoder.py`（~59%）、`backbone_wrappers.py` 的 `SpatialPriorModule`（逐字）、`dfine_utils.py`（核心 4 函数）、`dfine_criterion.py` 的 `_get_go_indices`、`denoising.py`、`matcher.py`、`ema.py`、`flat_cosine.py` | lightly NOTICE 明示上游：D-FINE、RT-DETR、EdgeCrafter、DEIMv2、DEIM 均 Apache-2.0 |
| 🟢 P2 | 已重写，仅配方常量/结构同源（常量是事实，不受版权保护） | `detcon.py`、`detcon_transform.py`、`dino.py`、`simclr.py`、`method.py`、`method_helpers.py`、`_scaling.py`、`view_transform.py`、`box_ops.py`（1 个函数） | token 相似度 4%–43%，最长逐字块 ≤16 行且多为默认参数；算法上游（DeepMind detcon、facebookresearch/dino、Lightning）均 Apache/MIT |

另有一处**非 AGPL 但非 Apache 的二级风险**：`_models/ecvit/ecvit.py:111-175` 内置的 `RopePositionEmbedding` 抄自 lightly vendored 的 DINOv3 代码（Meta 自定义 DINOv3 许可），随 P0 一并处理。

**第二层：架构与命名镜像（身份层）。** 顶层包结构 17 个中有 **15 个与 lightly_train 完全同名**（`_methods/_models/_task_models/_transforms/_embedding/_commands/_configs/_data/_debug/_events/_export/_loggers/_metrics/_optim/_visualize`），加上 `_scaling.py` 同名同 API，以及 `Method/MethodArgs/TaskModel/ModelWrapper/EmbeddingModel/TrainingStepResult/get_teacher/DOWNLOADABLE_MODEL_URL_AND_HASH` 等术语逐一对位。目录名与术语本身不受版权保护，**不构成法律风险**；但它是"是否独立演化"的最直观审计信号，且与第 2.4 节的 README 表述（"对标 lightly-train 0.17.0 的机制，按 Apache-2.0 重新实现"）共同构成持续的参考关系。本册按用户要求将其纳入重构范围（§4），与法律整改分开推进、分开验收。

**第三层：二进制资产（资产层）。** 这是当前**最硬的暴露点**：`_task_models/ltdetr_object_detection/coco_checkpoint.py:90-118` 向 Lightly 的 S3 桶注册了 3 个 LT-DETR COCO checkpoint（URL+SHA256 逐字来自 lightly registry）；`scripts/visdrone_ltdetr_*` 与 `/home/hdjs/Research/Project/Dino-Dome-Research/weight/` 另有约 8 个 Lightly 产出的 `.pt` 做 warm-start。lightly-train 仓库**对 checkpoint 文件本身没有任何许可声明**（README 只许可软件；商业条款暗示模型训练/运行需授权；NOTICE 只覆盖 vendored 源码），DINOv3 骨干的 checkpoint 还叠加 Meta DINOv3 许可义务。详见 §6。

### 1.2 目标

1. **法律层**：src/ 内不存在任何 lightly-train 衍生表达；每个实质移植文件有可追溯的宽松许可出处与署名。
2. **架构层**：顶层包结构、模块命名、内部术语与 lightly-train 无一一对应关系，建立本框架自己的领域语言；分层依赖有机器可检查的契约。
3. **资产层**：仓库内及运行时下载的全部二进制权重 100% 具有明确许可字段；lightly 产出权重全部下线，warm-start 改由宽松许可官方权重自建。
4. **流程层**：NOTICE + 溯源台账 + CI 红线固化，防止复发。

### 1.3 非目标

- 不改变任何已交付行为的语义（检测/分类/分割精度、导出图在容差内不变）。
- 不重构训练引擎、数据层等已确认自有原创的子系统（`_engine`、`_registry`、`_observability`、`_data` 核心逻辑仅随包迁移移动位置）。
- 不追求与 lightly-train 的 checkpoint **文件格式**互通（`AVWarmStart v1` 新格式见 §6.3，legacy 只读保留到 Phase 4 结束）。

---

## 2. 现状架构与镜像分析

### 2.1 顶层包结构对照

【审计】lightly_train 0.17.0 顶层与 autovision 顶层对照：

| 职责 | lightly_train | autovision 现状 | 镜像程度 |
|---|---|---|---|
| 预训练策略 | `_methods/` | `_methods/` | 完全同名 |
| 模型包装 | `_models/` + `model_wrappers.py` | `_models/` | 同名 |
| 任务模型 | `_task_models/` | `_task_models/` | 同名 |
| 变换 | `_transforms/` | `_transforms/` | 同名 |
| 嵌入导出 | `_embedding/` | `_embedding/` | 同名 |
| 命令 | `_commands/` + `_cli.py` | `_commands/`（+ `pyproject` 指向 `autovision._cli`） | 同名 |
| 配置 | `_configs/` | `_configs/` | 同名 |
| 数据 | `_data/` | `_data/` | 同名 |
| 调试/事件/导出/日志/指标/优化器/可视化 | `_debug/ _events/ _export/ _loggers/ _metrics/ _optim/ _visualize/` | 同名七件套 | 同名 |
| 缩放启发 | `_scaling.py` | `_scaling.py`（同 API：`interpolate/get_bucket_value`） | 同名同 API |
| 训练引擎 | 依赖 PyTorch Lightning | `_engine/`（自研） | **自有** |
| 资产注册 | `task_model_helpers.py` 内 `DOWNLOADABLE_MODEL_URL_AND_HASH` | `_registry/`（自有结构，但 `coco_checkpoint.py` 复用了该常量名与 3 条 URL/哈希数据） | 结构自有 |
| 观测 | — | `_observability/` | 自有 |

### 2.2 术语与约定镜像

| lightly 术语/约定 | autovision 现状 | 处置（R 编号） |
|---|---|---|
| `Method` / `MethodArgs` | 同名（`_methods/method.py`） | R10 |
| `TaskModel` / `task_model.py` | 同名 | R11 |
| `ModelWrapper` / `EmbeddingModel` | 同名（`_models/`） | R12 |
| `TrainingStepResult` | 同名 | R10 |
| `get_teacher` / `prepare_teacher` | 同名（`_methods/method_helpers.py`） | R10 |
| `DOWNLOADABLE_MODEL_URL_AND_HASH` | `_registry/downloadable.py` 复用该常量名 | R13 |
| checkpoint payload 约定 `{"train_model": {"ema_model.model.<k>": t}, "model_init_args", "model_class_path"}` | `coco_checkpoint.py` 按此约定读取 | R14 |
| `lightly_csv` 嵌出格式名 | `_embedding/embed.py`（且列序与 lightly 不兼容） | R15 |
| 检测器族名 "LT-DETR"（LT = Lightly Train） | `ltdetr_object_detection/`、scripts、README | R16 |

### 2.3 依赖方向现状

【现状】（箭头=import 方向，简化）：

```
scripts → autovision(_cli/_commands) → _engine → (_methods, _task_models)
_methods → _models, _transforms, _optim, _scaling
_task_models → object_detection_components → _models(backbone), _transforms/detection_transforms
_models → (HF transformers / timm / torchvision 第三方)
_embedding/_registry → _models
```

问题：`_task_models/object_detection_components` 事实上是检测族的共享内核，但物理上寄生在 `_task_models/` 之下；`_methods` 与 `_task_models` 之间通过 distillation 教师构建存在隐性耦合（`method_helpers.get_teacher` 直接引用任务模型包装）。目标契约见 §4.2。

### 2.4 溯源审计清单（完整版见附录 A）

按处置动作汇总（文件 → 动作 → 上游/验收）：

| # | 文件 | 动作 | 来源/说明 |
|---|---|---|---|
| 1 | `_methods/dinov3/dinov3.py` | **P0 clean-room**（2 个函数） | 规格源：分册 20；无上游 |
| 2 | `_methods/dinov3/constrained_crop.py` | **P0 clean-room**（1 个函数，可借 torchvision BSD 数学） | §5.1 S-1 |
| 3 | `_models/ecvit/ecvit.py` | **P1 re-port** ← EdgeCrafter + **P0 RoPE 替换**（timm/transformers Apache 实现） | §5.1 S-4、§5.2 |
| 4 | `_task_models/object_detection_components/dfine_official_decoder.py` | **P1 re-port** ← D-FINE | §5.2 |
| 5–11 | `hybrid_encoder.py`、`rtdetrv2_decoder.py`、`denoising.py`、`matcher.py`、`ema.py`、`dfine_utils.py`、`dfine_criterion.py` | **P1 re-port** ← RT-DETR / D-FINE | §5.2 |
| 12 | `ltdetr_object_detection/backbone_wrappers.py` | **P1 re-port**（`SpatialPriorModule`）← 本地 DEIMv2-main | §5.2 |
| 13 | `flat_cosine.py` | **P1 re-port** ← DEIM | §5.2 |
| 14 | `object_detection_components/_msda_src/` | 核对头部 → 补 Detectron2（Apache）署名 | §5.2 注 |
| 15 | `_optim/lars.py`、`_methods/simclr|dino|detcon|method*.py`、`_transforms/view_transform.py`、`_scaling.py` | **P2 注释/署名修正**（代码保留） | §5.3 |
| 16 | `box_ops.py` `sanitize_boxes_cxcywh_normalized` | 改名 + 注释 | R17 |
| 17 | `dino.py:302` | 事实错误修正（lightly 的 DINOLoss 并不 normalize teacher） | §5.3 |
| 18 | `coco_checkpoint.py` + scripts 权重链路 | **P3 资产切换** | §6 |
| 19 | `weights/yolo26n.pt`（Ultralytics AGPL 权重） | 移出仓库 | R18 |

---

## 3. 重构原则与方法论

### 3.1 版权边界（判断一件东西"能不能留"）

| 可以留（事实/思想） | 必须清除（表达） |
|---|---|
| 算法思想、论文机制、架构概念 | 逐字或近似逐字的代码体（含翻译式改名） |
| 超参数值、默认配置、配方数字 | 具有独创性的注释/文档字符串文本 |
| URL、文件名、SHA256、格式约定 | lightly 特有的文件组织在单文件内的结构复刻 |
| 类名与 API 用通用词（`SimCLR`、`DINO`、`LARS`——论文/通用词） | `Ported from lightly-train` 类 lineage 声明（指向 AGPL 源的声明一律清除） |

【设计 R1】注释中的事实性对标允许保留在 **docs/**（如"机制对标 lightly-train 0.17.0 的 X 行为"），**src/ 内一律清零**——src 里保留这类句子只会诱导后续贡献者"照着补齐"，且是审计的第一负面信号。

### 3.2 Clean-room 规程（P0 用）

适用于无宽松上游的 lightly 原创代码：

1. **规格先行**：由"已读过 lightly 代码"的人（或依据本仓设计文档/论文）写行为规格文档：输入输出张量形状与 dtype、边界条件、随机性语义、与哪些损失/变换衔接。规格中**不得出现** lightly 的代码片段、标识符命名、逐行结构描述。
2. **人员隔离**：实现者未读过 lightly 对应文件；实现时只看规格 + 本仓设计分册 + 论文 + 宽松许可第三方算子。
3. **独立表达**：循环结构、中间变量命名、实现策略（组合现成算子 vs 手写）必须与原实现有可见差异；验收含 token 相似度阈值（§9）。
4. **记录**：规格文档、实现 PR、评审记录入 `docs/provenance/` 台账（§7.1）。

### 3.3 Re-port 规程（P1 用）

适用于存在 Apache-2.0 原始上游的文件：

1. 克隆上游仓库，**记录 commit hash 入台账**（附录 D 列出仓库清单与本地镜像）。
2. 以上游文件为底，保留上游版权头，追加 `Modified by Autovision (Apache-2.0)` 行。
3. 重新施加本仓增量。增量清单以**当前单测 + golden 测试锁定的行为**为准（附录 A 每项列了增量要点）；**不再为加载 lightly checkpoint 保留任何增量**——权重链路在 Phase 4 整体切换（§6），过渡期 legacy loader 只读保留。
4. 验收双向 diff：对上游 diff 只含本仓增量；对 lightly 相似度 <15%（lightly 自己的修改不带入）。
5. 台账登记一行：`文件 → 上游 repo@commit → license → 修改摘要 → 复核人`。

### 3.4 命名独立性（R2）

- 建立本框架自己的领域语言（§4.1、附录 B），顶层包不再与 lightly_train 一一同名。
- 论文/通用词不动：`simclr`、`dino`、`detcon`、`simclr`/`ntxent`、`lars`、`adamw` 等是社区词汇，**不改**（改了反而不可读）。
- 公开 API 的用户可见键名保持稳定（R9，§4.4）：`model="torchvision/resnet18"`、`method_args`、`pretrain/train_object_detection/...` 全部不变；包重命名只动 import 路径与内部类名。

---

## 4. 目标架构设计

### 4.1 新顶层包结构

【设计 R3】**丢弃 `_` 私有前缀的扁平包风格**（lightly 的标志性布局习惯），改为按框架自身管线域（pretrain → finetune → evaluate → export）组织的公开子包。完整迁移映射：

```
src/autovision/
├── cli/                 # ← _commands/（train_task 子包随之），_cli.py 入口
├── config/              # ← _configs/
├── data/                # ← _data/（含 _serialize/ → data/serialize/）
├── augment/             # ← _transforms/；detection_transforms/ → augment/detection/
├── pretext/             # ← _methods/（自监督预训练策略层）
│   ├── simclr/ dino/ detcon/ densecl/ dinov2/ dinov3/ distillv2/ distillv3/
│   ├── recipe.py        # ← method.py（R10：Method→PretextRecipe）
│   └── teachers.py      # ← method_helpers.py（R10：get_teacher→build_teacher）
├── encoders/            # ← _models/（骨干/模型包装层）
│   ├── torchvision/ timm/ super_gradients/ ultralytics/ custom/
│   ├── dinov2/ dinov3/  #  HF transformers 包装（现状已自有）
│   ├── ecvit/ rtdetr/ rfdetr/
│   └── protocols.py     # ← model_wrappers.py（R12：ModelWrapper→EncoderProtocol 等）
├── tasks/               # ← _task_models/
│   ├── image_classification/
│   ├── segmentation/    #  linear / eomt(dinov2|dinov3) 各变体
│   ├── detection/
│   │   ├── components/  # ← object_detection_components/（提升为检测族共享内核）
│   │   ├── rtdetr/ deimv2/ dfine/ rfdetr/ picodet/ fsdc/
│   │   ├── oriented/    # ← rtdetr_oriented_object_detection/
│   │   └── avdetr/      # ← ltdetr_object_detection/（R16 已更名：AV-DETR）
│   └── eomt_core/       # ← eomt/（若被分割族共享）
├── engine/              # ← _engine/（自有原创，仅移动）
├── optim/               # ← _optim/ + _scheduler.py
├── metrics/             # ← _metrics/
├── features/            # ← _embedding/（embed 命令实现；R15：格式改名 csv_features）
├── registry/            # ← _registry/ + 新增 assets.py（§6.3 资产清单）
├── export/              # ← _export/
├── observability/       # ← _observability/ + _loggers/ + _events/（合并观测面）
├── debug/               # ← _debug/
├── visualize/           # ← _visualize/
├── scaling.py           # ← _scaling.py（保留实现，P2 已重述注释）
└── errors.py
```

要点：

- **R16（检测器族改名）**：`LT-DETR` 名称中的 LT 即 Lightly Train，属于对参考系统的持续背书。架构本体是 RT-DETRv2/D-FINE 组装 + EdgeCrafter/DINOv3 骨干，改名不损失信息。已定名 `avdetr`（Autovision DETR，2026-10-14 决策，registry 旧键 alias 一个版本）；改名波及：包名、registry 键 `ltdetr/…`、scripts 文件名、README、docs 13/14/19/23 的表述（docs 保留历史名并在新文档首行注明"原 LT-DETR"）。
- **R15**：`lightly_csv` → `csv_features`。现实现与 lightly 的列序本就不兼容（labels 第二列 vs 末列），改名同时修正语义；`embed(format=…)` 的旧值 `lightly_csv` 保留一个弃用别名两个版本。
- 迁移是**机械操作**：脚本化 `git mv` + import 重写 + `__init__` 导出修正 + 全测试。不与任何行为修改混在同一 PR。

### 4.2 分层依赖契约

【设计 R4】用 `import-linter`（dev 依赖，CI 执行）固化单向分层，高层可 import 低层、禁止反向：

```
cli → engine → {pretext, tasks} → {augment, features} → encoders → {data, optim, metrics, config, registry, scaling}
```

配套规则：

1. `tasks → pretext` 禁止直接 import；蒸馏教师构建经 `encoders` 协议注入（现状 `method_helpers.get_teacher` 直连任务模型包装，随 R10 收敛到 `pretext/teachers.py` + 协议）。
2. `encoders` 不得 import `tasks`/`pretext`/`engine`。
3. 第三方重模型依赖（transformers/timm）只允许出现在 `encoders/` 内部（现状已基本如此，`_models/*` 之外 grep transformers 归零）。
4. `tasks/detection/components` 只被 `tasks/detection/*` import。

### 4.3 checkpoint 兼容层

【设计 R14】两件事：

1. **legacy 只读加载**：`tasks/detection/avdetr/warm_start.py` 内置

   ```python
   LEGACY_CLASS_PATH_REMAP = {
       # 老 lightly 产物 checkpoint 内嵌的类路径 → 本仓新路径（含 ltdetr→avdetr 迁移前后的两级映射）
       "lightly_train._task_models.ltdetr_object_detection.task_model.": "autovision.tasks.detection.avdetr.network.",
       "autovision._task_models.ltdetr_object_detection.task_model.":   "autovision.tasks.detection.avdetr.network.",
   }
   ```

   加载时对 payload 的 `model_class_path` 做前缀重映射并打 INFO 日志；state-dict 键名（`backbone.`/`encoder.`/`decoder.` 前缀与张量形状）不变，不需要张量级 remap。
2. **新格式 `AVWarmStart v1`**（自建 warm-start 用，§6.2）：

   ```python
   {"format": "av_warm_start_v1",
    "model": state_dict,            # 直接是目标网络键名
    "spec": {"family": "avdetr", "backbone": "...", "classes": 80, "created": "..."},
    "license": {"id": "Apache-2.0", "origin": "<repo@commit 或 自训 run id>"}}
   ```

   `license` 为必填字段——资产清单与 checkpoint 自描述都把许可当一等元数据（§6.3），从机制上杜绝"无主权重"再次入库。

### 4.4 公开 API 策略（R9）

- `pretrain/train_object_detection/train_image_classification/train_semantic_segmentation/embed/benchmark/list_models/list_methods/load_*` 及参数名**全部不变**；`autovision-cli` 入口不变。
- 变化的只是包路径（`from autovision.pretext import …`）与弃用别名（`lightly_csv` 格式名、`ltdetr/…` registry 键保留 alias 两版本 + DeprecationWarning）。
- 提供 `docs/migration-0.2.md`：旧 import 路径 → 新路径速查表（由迁移脚本自动生成）。

### 4.5 AGPL 隔离模式（推广既有实践）

【现状】`ultralytics` 集成已采用"optional extra + 运行时下载权重 + 源码零拷贝"的隔离模式。【设计 R18】将其命名为本仓标准模式并写入贡献规程（§7.4）：**任何 AGPL/自定义许可的第三方**只允许 (a) optional extra 依赖 + 运行时适配器，或 (b) 子进程/独立服务边界；**禁止**拷贝其源码与权重入库。`weights/yolo26n.pt` 据此移出仓库，需要时由用户按 ultralytics 隔离模式自取。

---

## 5. 代码整改设计

### 5.1 P0：clean-room 重写（4 项）

> 规程见 §3.2。每一项先落规格文档到 `docs/provenance/specs/`，评审后再实现。

**S-1 `pretext/dinov3/constrained_crop.py::sample_high_overlap_box`**

- 行为规格：在父裁剪窗内采样一个子框，使其与给定目标几何的**重叠面积比落在 [0.05, 0.40]**；输入目标框与父窗尺寸，输出子框坐标；随机性为均匀可复现（固定 generator）。
- 实现指引：面积–长宽比参数采样语义与 torchvision `RandomResizedCrop.get_params` 同族（BSD-3-Clause，可直接引用其数学并署名），把目标面积从 `[0.08,1]·窗` 换成约束区间采样；禁止打开 lightly `dinov31/constrained_crop.py`。
- 验收：区间断言的统计测试（N≥1000 次采样全部落在约束内）；token 相似度 <15%；8 维几何张量布局 `[x0,y0,x1,y1,w,h,hflip,vflip]` 保持（张量布局是接口事实，非表达）。

**S-2 `pretext/dinov3/network.py::_roi_align_view`**（现 `dinov3.py:502-545`）

- 行为规格：按每样本几何张量从视图特征图/图像上双线性采样到目标网格，并按 `hflip/vflip` 标志翻转。等价于 `torchvision.ops.roi_align` + 翻转组合。
- 实现指引：**优先组合现成算子**（torchvision roi_align，BSD-3）而非手写 im2col；本仓只写几何换算与批处理外壳。
- 验收：与"roi_align+flip 组合参照实现"数值一致（fp32 max abs err < 1e-6，固定 seed 单测）。

**S-3 `pretext/dinov3/network.py::_align_cross_view_pair`**（现 `dinov3.py:439-500`）

- 行为规格：对 student/teacher 双视图 patch 特征做几何对齐——视图 B 的采样框映射到视图 A 坐标系，取 A 对应位置特征，产出对齐特征对供 PaKA/对比损失（分册 14/20）。唯一规格来源：**分册 20（`20-dinov3-vit-masked-forward.md`）与分册 14** + arXiv 2509.05606（PaKA）。
- 规程：若分册 20 的描述不足以无歧义实现，**先扩写分册 20** 再实现（把规格补全作为交付物）。
- 验收：smoke 训练对拍（小数据集、固定 seed，loss 曲线形态一致 + 最终 loss 相对差 <2%）；相似度 <15%。

**S-4 `encoders/ecvit/rope.py`：替换 DINOv3 许可的 RoPE**

- 现状：`ecvit.py:111-175` 的 `RopePositionEmbedding` 是 lightly `_models/dinov3/dinov3_src/layers/rope_position_encoding.py` 的删减拷贝（Meta DINOv3 License，非 AGPL 但与 Apache 分发不兼容）。
- 设计：以 **timm 的 rotary 实现或 HF transformers 的 rope 模块**（均 Apache-2.0）为底重写，保序保 buffer 形状以维持 state-dict 兼容；若两者算子语义不足以逐比特对齐，允许数值等价（<1e-5）+ 旧 checkpoint 加载测试做验收，不追求逐比特。
- 验收：ECViT 前向数值回归 + 现有 ecvit 相关单测全过。

### 5.2 P1：从 Apache 原始源重新移植（规程 §3.3）

| 目标文件（新路径） | 上游 | 上游文件 | 本仓需保留的增量（以现有单测/golden 锁定为准） |
|---|---|---|---|
| `tasks/detection/components/dfine_official_decoder.py` | D-FINE（Peterande/D-FINE，Apache-2.0） | `src/zoo/dfine/dfine_decoder.py` | 输出 dict 契约、`max_gt_cap`、TensorRT gather 采样器、open-vocab `TextCrossAttention`、decoder 钩子面 |
| `tasks/detection/components/hybrid_encoder.py`、`rtdetrv2_decoder.py`、`denoising.py`、`matcher.py`、`ema.py` | RT-DETR（lyuwenyu/RT-DETR，Apache-2.0；D-FINE 仓内同源文件可交叉验证） | 对应 `rtdetrv2_*` 文件 | `bilinear_grid_sample`（TensorRT）、`max_gt_cap`、逐图 query mask、分数缓存 |
| `tasks/detection/components/dfine_utils.py`、`dfine_criterion.py` | D-FINE | `dfine_utils.py`/`dfine_criterion.py` | 本仓缓存层、`matched_pair_iou`、`flatten_match_indices`、重构后的 `fg_loss/ddf_loss`、向量化 `_get_go_indices` |
| `tasks/detection/avdetr/backbone_wrappers.py`（`SpatialPriorModule`） | **本地镜像 `DEIMv2-main`**（Apache-2.0） | 对应 spatial prior / STA 文件 | 本仓三骨干合并的 wrapper 面、ViT 中间层 tap |
| `augment/`（`flat_cosine.py` 所在处） | DEIM（Intellindust-AI-Lab/DEIM，Apache-2.0） | scheduler 文件 | 本仓 step-based 接口适配 |
| `encoders/ecvit/ecvit.py`（骨架部分） | EdgeCrafter（Intellindust-AI-Lab/EdgeCrafter，Apache-2.0） | ECViT backbone 实现 | `ECViTModelWrapper` 的多层级特征输出契约（P3-P5 + projector）、`ConvNormLayer/RopePositionEmbedding/Mlp` 的本仓 vendored 决策改为"上游有则引上游，无则 S-4 方式重写" |
| `tasks/detection/components/_msda_src/`（CUDA 扩展） | Detectron2（facebookresearch/detectron2，Apache-2.0；社区通用 MSDA 代码） | `detectron2/layers/csrc/deformable/` | 仅核对 + 补署名，不改代码 |

【设计 R5】移植验收的**双向 diff**：

```bash
# 1) 对上游：diff 应只含本仓增量（增量行数与本表"增量"列相符）
diff -u upstream/file.py autovision/new/file.py | diffstat
# 2) 对 lightly：token 相似度必须 <15%（lightly 自有修改不带入）
python scripts/check_provenance.py --reference-dir /home/hdjs/Research/Software/lightly-train-0.17.0
```

【风险 R-RISK1】重移植可能引入数值漂移。缓解：每个文件移植前先补"行为快照"单测（小张量、固定 seed，断言现实现输出）；移植后逐文件跑快照 + `tests/golden/` 数值门禁；Phase 2 结束跑一次 VisDrone 小规模微调对拍（§9）。

### 5.3 P2：注释、署名与零碎项（代码保留，只动文本/命名）

规则（R1）：

- **lineage 声明**一律改为指向宽松上游或论文：`Ported from lightly-train 0.17.0 …` → `Based on D-FINE (Apache-2.0) <repo>, modified by Autovision`。
- **事实性对标**（"配方数字与 lightly-train 0.17.0 一致"）在 src/ 内清零，需要时移入 docs/。
- 具体修正点（全部来自审计命中清单）：

| 位置 | 修正 |
|---|---|
| `dfine_official_decoder.py:4`、`dfine_utils.py`、`dfine_criterion.py` | 随 P1 重移植自然消除 |
| `dino.py:302-304` | 事实错误：lightly 的 DINOLoss **不** normalize teacher；改为"normalize 为本仓增量（对应 facebookresearch/dino 的用法）" |
| `method.py:42`、`method_helpers.py:67` | 教师准备逻辑重述为本仓语义（eval + 冻结 + 可选权重注入），移除 "mirrors/mirroring lightly-train" |
| `simclr.py:56`、`detcon*.py` 头部 | "lightly defaults" → "SimCLR/DetCon 论文标准配方（数值与 DeepMind 开源配置一致）"；detcon 文件头补 DeepMind detcon（Apache-2.0）参考声明 |
| `_scaling.py` | 保留实现；文件头注明通用插值启发式，不改 API |
| `box_ops.py:24` | `sanitize_boxes_cxcywh_normalized` → `clip_normalize_boxes`（R17） |
| `rfdetr.py:6,17`、`model_registry.py:19`、`ltdetr config.py:11,888-901,957`、`task_model.py:465`、5 个 `scripts/visdrone_ltdetr_*.py` 的 `/tmp/lightly_ckpts` 注释 | 随 R16 改名与 P3 资产切换一并重写 |
| `embed.py` `lightly_csv` | → `csv_features`（R15） |
| 全库 `grep -rin lightly src/` | Phase 5 验收时清零（docs/ 允许事实性保留） |

---

## 6. 权重与资产管理设计

### 6.1 资产许可矩阵（现状 → 目标）

| 资产 | 现状 | 目标 | 替代来源与许可 |
|---|---|---|---|
| `edgecrafter_ecvitt/ecvittplus/ecvits_ltdetr_coco_*.pt`（S3 注册 3 件） | Lightly S3，无许可声明 | **下架注册表** | EdgeCrafter 官方 ECViT 权重（Apache-2.0）作骨干 init → COCO 自训检测头 → 产出自有 `AVWarmStart v1` |
| `dinov3_convnext_{tiny,base,large}_ltdetr_coco_*.pt`、`dinov3_vits16_ltdetr_coco_*.pt`、`*_coco_full.pt`（本地 ~5 件） | Lightly 产出（其一为改名件） | **下线** | decoder 侧用 D-FINE 官方 COCO（Apache-2.0）；骨干侧决策点 D1（下） |
| `weights/yolo26n.pt` | Ultralytics（AGPL 家族）权重入库 | **移出仓库** | ultralytics 隔离模式（R18）按需运行时获取 |
| DINOv3 HF 权重（运行时下载） | Meta DINOv3 License | 决策点 D1 | 见下 |

【决策点 D1 —— **已决策（2026-10-14）：D1-a**】

- **D1-a（已采纳）**：DINOv3 骨干线保留，但仅在**用户自行动作**（optional extra + 用户本机持有权重 + 环境旗标 `AUTOVISION_DINOV3_WEIGHTS=1` 自证已接受 Meta DINOv3 License）下启用，仓库与文档不做分发；研究产出在文档中附 DINOv3 License 义务注记（见 NOTICE）。检测 warm-start 骨干从 EdgeCrafter（Apache）与 DINOv2（Apache）两条线自建。实现：`encoders/dinov3/dinov3_package.py` 权重门控（`-notpretrained` 与显式 `weights_path` 不受影响）。
- ~~D1-b：研究主线切换 DINOv2（Apache-2.0），DINOv3 全线退出默认路径。~~（未采纳；DINOv2 线仍是 warm-start 的 Apache 路线之一）

### 6.2 自建 warm-start 管线

【设计 R6】新增 `scripts/build_warm_start_coco.py`：

1. 输入：Apache 官方组件权重（EdgeCrafter ECViT / DINOv2 / D-FINE decoder 官方 COCO 权重）。
2. 组装 `avdetr` 目标网络（backbone init ← 官方；decoder init ← D-FINE；新初始化层按现 warm-start 规则跳过 class-count 相关张量）。
3. COCO 标准配方短程微调（复用现有检测训练管线），产出 `AVWarmStart v1` checkpoint + 台账行（run id、数据、license 字段）。
4. 验收：VisDrone 微调起点 mAP 与"现状 Lightly warm-start"差 ≤0.5 AP（内部研究可接受线；若不达标调配方重训）。

### 6.3 资产清单（asset manifest）

【设计 R7】`registry/assets.py` 取代 `coco_checkpoint.py` 里的 3 条硬编码注册：

```python
ASSET_MANIFEST: dict[str, AssetEntry] = {
    "d-fine/dfine_s_coco": AssetEntry(
        urls=["https://…/dfine_s_coco.pth"],          # 官方源
        sha256="…",
        license=LicenseRef(id="Apache-2.0", origin="Peterande/D-FINE@<commit>"),
    ), …
}
```

- `license` 为**必填**；缺失即 CI 失败（§7.2 扫描 `LicenseRef` 构造）。
- 常量名 `DOWNLOADABLE_MODEL_URL_AND_HASH` 废除（R13），下载器 API 改 `resolve_asset(name)`。
- 运行时下载域名校验：URL host 白名单进 config，默认仅含官方源域名；lightly S3 host 进入禁词表。

---

## 7. 防护机制（防复发）

### 7.1 NOTICE 与溯源台账

- 仓库根新增 `NOTICE`：列全部 vendored/移植组件（D-FINE、RT-DETR、EdgeCrafter、DEIM/DEIMv2、Detectron2 MSDA、torchvision、timm、transformers、DeepMind detcon 参考、facebookresearch/dino 参考、Lightning 参考），每项含来源、许可、修改摘要——格式参照 lightly-train 自己的 NOTICE（该格式是社区惯例）。
- `docs/provenance/LEDGER.md`：逐文件台账（§3.3 第 5 步产物），P0/P1 每个交付 PR 必须带一行。
- `docs/provenance/specs/`：P0 clean-room 规格文档。

### 7.2 CI 红线：`scripts/check_provenance.py`

【设计 R8】单一工具两种模式：

- **模式一（默认，每次 CI）禁词扫描**：扫描 `src/ tests/ scripts/`，命中即失败：

  ```
  lightly_train  lightly-train  from lightly  import lightly
  Copyright (c) Lightly  lightly_ckpts  DOWNLOADABLE_MODEL_URL_AND_HASH
  lightly-train-checkpoints  .s3.us-east-1.amazonaws.com  LT-DETR(过渡期白名单)
  ```

  `docs/` 允许事实性提及（白名单文件 `provenance_allowlist.txt`，条目须带理由与复审日期）。
- **模式二（`--reference-dir`，本地挂对照源时启用）相似度扫描**：按附录 C 算法对全库 token 相似度评分，输出超阈值（25%）文件报告；Phase 5 验收与季度复审跑全量。

接入 pre-commit（快速禁词）+ CI（全量）。

### 7.3 git 策略

- 现状：14 个 commit，起点为 "pre-remediation snapshot"（整改前快照在历史里）。
- 【设计 R19】整改完成后 **clean cut**：从验收通过 commit 打 `v0.2-clean` tag；对外发布用**新仓库或 `git filter-repo` 截去 pre-remediation 历史**，只发布 Phase 0 之后的历史。私有历史可保留（不分发即不触发 AGPL 义务），但任何公开发布（含 push 到公共远端）前必须完成 clean cut。

### 7.4 贡献规程

`CONTRIBUTING.md` 增补"第三方引入清单"：引入任何第三方代码/权重前，必须 (1) 确认许可在 Apache-2.0 兼容白名单内（Apache/MIT/BSD/自研宽松），(2) 台账登记，(3) 走 R18 隔离模式（若为 AGPL/自定义），(4) `check_provenance.py` 通过。评审人对照 §3.1 版权边界表。

---

## 8. 实施计划

| Phase | 内容 | 交付物 | 验收 gate | 估工 |
|---|---|---|---|---|
| **0 防护基线** | NOTICE + 台账骨架 + `check_provenance.py` 入 CI + 审计快照归档 | §7.1/7.2 全套 | CI 红线生效（允许现存量白名单） | 2 人日 |
| **1 P0 clean-room** | S-1/S-2/S-3/S-4（§5.1），spec 先行 | 规格文档 + 新实现 + 单测 | 相似度 <15%；数值/统计验收 | 4–6 人日 |
| **2 P1 re-port** | §5.2 表逐文件，规程 §3.3 | 重移植 + NOTICE/台账行 + 行为快照测试 | 双向 diff 达标；golden 全过 | 5–8 人日 |
| **3 架构迁移** | §4 包结构 + R10–R17 术语 + remap + import-linter + migration 文档 | 新目录树 | 全测试过；旧 import/旧 checkpoint 加载测试过；公开 API 兼容测试过 | 3–4 人日 |
| **4 资产切换** | D1 决策 + §6.2 自建 warm-start + manifest（R7）+ 下线 lightly 资产 | 自有 COCO warm-start + 资产清单 | 资产 license 字段 100%；VisDrone 对拍 ≤0.5 AP | 代码 2–3 人日 + 训练算力 |
| **5 总验收与 clean cut** | §9 全项 + R19 | `v0.2-clean` tag + 发布审计记录 | §9 全绿 | 1–2 人日 |

依赖关系：Phase 1、2 可并行；Phase 3 依赖 1、2（避免迁移中改语义）；Phase 4 的代码部分可与 3 并行，训练部分依赖 D1 决策；Phase 5 收口。

---

## 9. 验收标准（总 gate）

1. `grep -rinE "lightly" src/ scripts/ tests/` 命中为 **0**（docs/ 白名单除外）。
2. `check_provenance.py --reference-dir <lightly-train-0.17.0>` 全库 token 相似度 **<15%**（允许白名单：`_msda_src` 等已署名 Apache 文件对上游高相似、对 lightly 低相似）。
3. 测试全绿 + `tests/golden/` 数值门禁 + VisDrone 小规模微调对拍（ΔAP ≤0.5）。
4. 公开 API 兼容测试：旧键名/旧格式别名可用并出 DeprecationWarning；`LEGACY_CLASS_PATH_REMAP` 覆盖全部历史 checkpoint 样本。
5. NOTICE/LEDGER/ASSET_MANIFEST 完整；**所有二进制权重资产 license 字段非空**；lightly S3 host 出现在禁词表且仓库内无引用。
6. import-linter 分层契约全绿。
7. clean cut 完成：`v0.2-clean` 起点之后的历史不含 pre-remediation 快照。

---

## 10. 风险与回滚

| 风险 | 影响 | 缓解 / 回滚 |
|---|---|---|
| 重移植引入数值漂移（R-RISK1） | 精度回落 | 逐文件行为快照先行；golden 门禁；Phase 2 末对拍；单文件级回滚（每文件独立 PR） |
| S-3 clean-room 规格不足导致行为偏差 | SSL 探针指标变化 | 先扩写分册 20；smoke 对拍 gate；必要时按行为等价迭代（实现仍独立） |
| 包迁移破坏隐性 import（scripts/notebooks 外部引用） | 工具链断裂 | 迁移脚本生成速查表；旧路径 shim 包保留一个版本（`autovision/_compat/…`） |
| D1 决策延迟 | Phase 4 阻塞 | D1-a 与 D1-b 的骨干线代码先行解耦（backbone registry 已按家族注册） |
| 术语改名与外部协作（论文/报告）暂不一致 | 表述混乱 | docs 保留 "原 LT-DETR" 对照注记；README 迁移表 |
| 过渡期双权重链路并存 | 磁盘/加载混淆 | legacy loader 只读 + 日志标注 deprecated；Phase 5 移除 |

---

## 附录 A：受影响文件完整清单（审计输出）

> 相似度为 token 级（附录 C）；"上游"指宽松许可原始出处。

| 文件（现路径） | 判定 | 相似度/证据 | 处置 | 上游 |
|---|---|---|---|---|
| `_methods/dinov3/dinov3.py` | 🔴 衍生（近逐字函数） | 41.4%；`:439-500`、`:502-545` ← LT `_methods/dinov31/dinov31.py:337-450` | P0 S-2/S-3 | 无（lightly 原创） |
| `_methods/dinov3/constrained_crop.py` | 🔴 衍生 | 36.1%；`:187-215` ← LT `dinov31/constrained_crop.py:150-162` | P0 S-1 | torchvision 数学（BSD-3） |
| `_models/ecvit/ecvit.py` | 🟡 衍生 + 🔴 RoPE | 72%；`:111-175` RoPE ← LT dinov3_src（DINOv3 License） | P0 S-4 + P1 | EdgeCrafter（Apache） |
| `object_detection_components/dfine_official_decoder.py` | 🟡 近逐字移植 | ~65% 行、~95% 函数体；自述 "Ported from lightly-train" | P1 | D-FINE（Apache） |
| `object_detection_components/dfine_utils.py` | 🟡 核心衍生 | 4 核心函数与 LT 逐字级；自述 "Ported 1:1" | P1 | D-FINE |
| `object_detection_components/dfine_criterion.py` | 🟡 重构衍生 | `_get_go_indices` 逐字块；VFL/损失数学同 D-FINE | P1 | D-FINE + RT-DETR |
| `object_detection_components/hybrid_encoder.py` | 🟡 近逐字 | ~63% 行 | P1 | RT-DETR |
| `object_detection_components/rtdetrv2_decoder.py` | 🟡 衍生 | ~59% 行；MSDA 段 ~2/3 同 | P1 | RT-DETR |
| `object_detection_components/denoising.py` | 🟡 重写（同 CDN 算法） | ~35% 数学行 | P1 | RT-DETR |
| `object_detection_components/matcher.py` | 🟡 重写 | focal cost 公式逐字 | P1 | RT-DETR / D-FINE |
| `object_detection_components/ema.py` | 🟡 重写 | 低 | P1 | RT-DETR |
| `flat_cosine.py` | 🟡 重写 | 低 | P1 | DEIM |
| `ltdetr_object_detection/backbone_wrappers.py` | 🟡 `SpatialPriorModule` 逐字 | `:58-123` ← LT `dino_vit_wrapper.py:58-124` | P1 | DEIMv2（本地镜像） |
| `object_detection_components/_msda_src/` | 🟡 待核签 | 社区通用 MSDA CUDA 扩展 | 署名核对 | Detectron2（Apache） |
| `object_detection_components/dfine_decoder.py`（DFINEv2Decoder） | 🟢 概念衍生 | ~5%（一个 zero-init 惯用法，LT `:794-795`） | P2 注释 | D-FINE 数学 + 本仓合成 |
| `object_detection_components/box_ops.py` | 🟢 独立 | 1 个函数同名同实现（`:24-33`） | R17 改名 | — |
| `_methods/detcon/detcon.py`、`detcon_transform.py` | 🟢 重写 | 28.7% / 21.4%；最长块 16 行（参数默认） | P2 注释 | DeepMind detcon（Apache） |
| `_methods/dino/dino.py` | 🟢 重写 | 26.2%；`:302` 事实错误注释 | P2 修正 | facebookresearch/dino（Apache） |
| `_methods/simclr/simclr.py` | 🟢 重写 | 30.6%（小文件，骨架+默认值） | P2 注释 | SimCLR（Apache 诸源） |
| `_methods/method.py`、`method_helpers.py` | 🟢 API 镜像 | 21.7% / 4.3% | P2 注释（R10 随 Phase 3） | — |
| `_scaling.py` | 🟢 重写、API 镜像 | 43%（10 行通用逻辑） | P2 注释 | — |
| `_transforms/view_transform.py`、`transform_args.py` | 🟢 独立 | 6.9% / 12% | 无动作 | facebookresearch/dino 管线 |
| `_models/dinov2/*`、`_models/dinov3/*`（HF 包装） | 🟢 独立 | ~13%，零匹配块 | 无动作 | HF transformers（Apache） |
| `_models/rfdetr/rfdetr.py` | 🟢 独立 | 概念+命名约定 | P2 注释 | 本仓自有（transformers 底座） |
| `ltdetr_object_detection/config.py` | 🟢 事实复用 | ~2%（变体配方为事实） | R16 改名随迁 | — |
| `ltdetr_object_detection/task_model.py`、`train_model.py`、`transforms.py` | 🟢 重写 | 8% / 10% / 1% | R16 随迁 | — |
| `ltdetr_object_detection/coco_checkpoint.py` | 🟢 自有胶水 | 0%（但 URL/哈希逐字、S3 host） | P3 §6 | 资产切换 |
| `_embedding/embed.py` | 🟢 自有 | 格式名引用 | R15 | — |
| `weights/yolo26n.pt` | 🔴 资产 | Ultralytics AGPL 权重 | R18 移除 | — |

## 附录 B：术语映射表（R10–R17 汇总）

| 现名 | 新名 | 说明 |
|---|---|---|
| `_methods/` `Method` `MethodArgs` `TrainingStepResult` | `pretext/` `PretextRecipe` `PretextArgs` `StepOutput` | 公开参数名 `method_args` 保留（R9） |
| `method_helpers.get_teacher` | `pretext/teachers.py::build_teacher` | 经协议注入任务模型 |
| `_task_models/` `TaskModel` | `tasks/` `TaskNetwork` | |
| `_models/` `ModelWrapper` `EmbeddingModel` | `encoders/` `EncoderProtocol` `FeatureExtractor` | |
| `_transforms/` | `augment/` | |
| `_embedding/` `lightly_csv` | `features/` `csv_features` | 旧值别名两版本 |
| `DOWNLOADABLE_MODEL_URL_AND_HASH` | `ASSET_MANIFEST` / `resolve_asset` | |
| `ltdetr_object_detection/`、registry 键 `ltdetr/…` | `tasks/detection/avdetr/`、`avdetr/…`（**已更名，2026-10-14**） | registry 旧键 `ltdetr/…` alias 一个版本 + DeprecationWarning |
| `sanitize_boxes_cxcywh_normalized` | `clip_normalize_boxes` | |
| checkpoint payload `{"train_model", "model_init_args", "model_class_path"}` | `AVWarmStart v1`（§4.3） | legacy 只读保留至 Phase 4 末 |

## 附录 C：审计方法与复现

1. **token 化**：剥除注释/docstring/字符串字面量；`lightly_train`→`A`、`autovision`→`B` 归一。
2. **相似度**：`difflib.SequenceMatcher` 对 token 行序列求 ratio；"逐字块"= 最长连续相同 token 行。
3. **人工复核**：相似度 >20% 的文件对做逐行 diff，区分"算法必然同形"（如 softmax+gather 标准写法）与"表达性重合"（常量布局、命名习惯、错误结构同源）。
4. **复现**：Phase 0 交付 `scripts/check_provenance.py`（模式二即本审计的脚本化），`--reference-dir` 指向 lightly-train-0.17.0 源码树；`provenance_allowlist.txt` 白名单化 docs 引用。

## 附录 D：上游参考源清单

| 上游 | 许可 | 获取 | 本地镜像 |
|---|---|---|---|
| D-FINE | Apache-2.0 | `git clone https://github.com/Peterande/D-FINE`（记 commit 入台账） | 无 |
| RT-DETR | Apache-2.0 | `git clone https://github.com/lyuwenyu/RT-DETR` | 无（D-FINE 仓内同源文件可交叉验证） |
| EdgeCrafter | Apache-2.0 | `git clone https://github.com/Intellindust-AI-Lab/EdgeCrafter` | 无 |
| DEIM / DEIMv2 | Apache-2.0 | `git clone https://github.com/Intellindust-AI-Lab/DEIMv2` | `~/Research/Software/DEIMv2-main` |
| Dome-DETR | Apache-2.0（分册 13 引用） | — | `~/Research/Software/Dome-DETR-master` |
| RF-DETR | Apache-2.0 | — | `~/Research/Software/rf-detr-1.11.1` |
| Detectron2（MSDA CUDA） | Apache-2.0 | `git clone https://github.com/facebookresearch/detectron2` | 无 |
| torchvision（RandomResizedCrop 数学 / roi_align） | BSD-3-Clause | 随环境 | 随环境 |
| timm / transformers（RoPE、DINOv2/v3 包装） | Apache-2.0 | 随环境 | 随环境 |
| facebookresearch/dino（DINOLoss 参考） | Apache-2.0 | GitHub | 无 |
| google-deepmind/detcon（DetCon 配方参考） | Apache-2.0 | GitHub | 无 |
| Lightning（LARS 原始出处之一） | Apache-2.0 | GitHub | 无 |

> 台账规则：所有 re-port PR 必须在 `docs/provenance/LEDGER.md` 记录 `上游 repo@commit`；clone 后不要跟随上游 main 漂移。
