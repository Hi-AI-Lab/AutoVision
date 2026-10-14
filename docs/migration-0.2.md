# 迁移指南 0.1 → 0.2（Phase 3 包结构迁移）

分册 28（`docs/design/28-clean-provenance-refactor.md`）§4 的架构迁移落地
记录。**公开 API 完全不变**（`pretrain / train_* / embed / benchmark /
list_models / list_methods / load_*` 及全部参数键名）；变化的只有包路径与
内部模块组织。

## 旧 → 新 import 路径速查

| 旧（0.1） | 新（0.2） |
|---|---|
| `autovision._commands` | `autovision.cli` |
| `autovision._configs` | `autovision.config` |
| `autovision._data`（含 `_serialize` → `serialize`） | `autovision.data` |
| `autovision._transforms`（含 `detection_transforms` → `detection`） | `autovision.augment` |
| `autovision._methods` | `autovision.pretext` |
| `autovision._models` | `autovision.encoders` |
| `autovision._task_models` | `autovision.tasks` |
| `autovision._task_models.object_detection_components` | `autovision.tasks.detection.components` |
| `autovision._task_models.{rtdetr,deimv2,dfine,rfdetr,picodet,fsdc,rtdetr_oriented,ltdetr}_object_detection` | `autovision.tasks.detection.<同名>` |
| `autovision._engine` | `autovision.engine` |
| `autovision._optim` | `autovision.optim` |
| `autovision._scheduler` | `autovision.optim.schedulers` |
| `autovision._metrics` | `autovision.metrics` |
| `autovision._embedding` | `autovision.features` |
| `autovision._registry` | `autovision.registry` |
| `autovision._export` | `autovision.export` |
| `autovision._observability`（`_loggers`/`_events` 并入为子包） | `autovision.observability`（`.loggers` / `.events`） |
| `autovision._debug` | `autovision.debug` |
| `autovision._visualize` | `autovision.visualize` |
| `autovision._scaling` | `autovision.scaling` |
| `autovision._optimizer_type` | `autovision.optimizer_type` |
| `autovision._cli`（CLI 入口模块） | `autovision.cli._app`（console script 不变：`autovision-cli`） |

单文件模块保留顶层不动：`_checkpoint / _accelerator / _compat /
_distributed / _env / _license / _logging / _model_wrappers / _plot /
_system / _torch_compile / types / errors / _version`。

## 兼容性承诺（一个版本周期）

1. **顶层 shim**：迁移完成初期旧包名曾以弃用 shim 过渡（发
   `DeprecationWarning`，re-export 新包公开名）。**已于 2026-10-14 移除**：
   仓库内外均无引用，shim 未随任何版本发布，0.2 起旧顶层包名不再可用，
   请直接按上表改为新路径。深层子模块路径（如 `autovision._methods.dino.dino`）
   从来不是公开 API，始终不在兼容范围。
2. **`embed` 格式名**：`format="lightly_csv"` 改名 `"csv_features"`（输出
   布局不变）；旧值保留一个版本，发 `DeprecationWarning`（R15）。
3. **checkpoint**：state-dict 键与张量布局不变；旧 payload 的
   `model_class_path` 由
   `tasks/detection/ltdetr_object_detection/coco_checkpoint.py::
   LEGACY_CLASS_PATH_REMAP` 自动重映射并打日志（§4.3）。
4. **CLI 入口**：`autovision-cli` console script 行为不变（入口点内部路径
   更新为 `autovision.cli._app:cli_entrypoint`）。

## 新增的机器检查

- `scripts/check_layering.py`（CI provenance job）：§4.2 单向分层契约，
  上行 import 需在 `ALLOWED_UPWARD` 带理由登记（当前 8 项共享工具类债务）。
- `scripts/check_provenance.py`：§7.2 禁词 + 相似度双模式。

## 仍待办（Phase 3 收尾，见台账）

- R16 检测器族更名（`ltdetr_object_detection` → 终名待定），含 registry
  键 alias。
- Phase 4 资产切换后，`DownloadableModel.license` 从可选转必填。

## 检测器族更名（R16，已落地）

检测器族 LT-DETR → **AV-DETR**（LT 即 Lightly Train 缩写，R16 去品牌化）：

| 旧 | 新 |
|---|---|
| 包 `tasks/detection/ltdetr_object_detection/` | `tasks/detection/avdetr/` |
| 类 `LTDETR*`（`AVDETRObjectDetection` 等 9 个） | `AVDETR*` |
| registry 键 `ltdetr/<backbone>` | `avdetr/<backbone>`（**旧键 alias 一个版本**，发 DeprecationWarning；`train_object_detection(model="ltdetr/…")` 与 `resolve_avdetr_variant` 均接受） |
| `scripts/visdrone_ltdetr_*.py`（19 个） | `scripts/visdrone_avdetr_*.py` |

**不改**：基线资产文件名（`*_ltdetr_coco_*.pt`，Lightly S3 外部键，Phase 4
替换前原样）；`LEGACY_CLASS_PATH_REMAP` 旧键（追加 avdetr 目标代际）；
docs/ 历史分册中的 LT-DETR 表述（历史名，新文档首行已注）。

## 内部类名术语（R10–R12，已落地）

内部类与辅助模块改名（**公开 API 键名不变**：`method_args`、
`list_methods`、`get_method_cls`、`optimizer_args_cls`、
`transform_cls` 等全部保留）：

| 旧 | 新 |
|---|---|
| `pretext/method.py` 模块 | `pretext/recipe.py` |
| `pretext/method_helpers.py` 模块 | `pretext/teachers.py` |
| `Method` | `PretextRecipe` |
| `MethodArgs`（`method_args.py`，模块名不变） | `PretextArgs` |
| `TrainingStepResult` | `StepOutput` |
| `TaskModel`（`task_model.py`，模块名不变） | `TaskNetwork` |
| `ModelWrapper` 协议（`model_wrapper.py`，模块名不变） | `EncoderProtocol` |
| `EmbeddingModel`（`embedding_model.py`，模块名不变） | `FeatureExtractor` |
| `teachers.get_teacher` | `teachers.build_teacher` |
