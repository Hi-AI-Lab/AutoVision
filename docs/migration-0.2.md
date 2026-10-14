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

## 0.3 任务优先目录重组（design doc 30，2026-10-15）

`tasks/` 一级改为按任务类型分组、二级按模型家族，公开 API 与全部类名
不变；旧包路径的导出/checkpoint 由
`task_model_helpers._MODULE_PATH_MIGRATIONS` 前缀映射自动迁移（第四代
类路径迁移，覆盖 `model_class_path` 与 `init_args` 内嵌类引用）。

| 旧（0.2） | 新（0.3） |
|---|---|
| `tasks.dinov2_eomt_semantic_segmentation` | `tasks.semantic_segmentation.dinov2_eomt` |
| `tasks.dinov3_eomt_semantic_segmentation` | `tasks.semantic_segmentation.dinov3_eomt` |
| `tasks.linear_semantic_segmentation` | `tasks.semantic_segmentation.linear` |
| `tasks.dinov2_eomt_instance_segmentation` | `tasks.instance_segmentation.dinov2_eomt` |
| `tasks.dinov3_eomt_instance_segmentation` | `tasks.instance_segmentation.dinov3_eomt` |
| `tasks.dinov2_eomt_panoptic_segmentation` | `tasks.panoptic_segmentation.dinov2_eomt` |
| `tasks.dinov3_eomt_panoptic_segmentation` | `tasks.panoptic_segmentation.dinov3_eomt` |
| `tasks.image_classification` | `tasks.classification` |
| `tasks.detection.rtdetr_oriented_object_detection` | `tasks.oriented_detection.rtdetr` |
| `tasks.detection.{rtdetr,dfine,deimv2,rfdetr,picodet,fsdc}_object_detection` | `tasks.detection.<家族短名>` |
| `tasks.detection.components.{ema,flat_cosine}` | `tasks.shared.{ema,flat_cosine}` |
| `tasks.detection.components`（检测架构核） | `tasks.detection.shared` |
| `tasks.detection.components.{box_ops,obb}`（纯数学） | `autovision.geometry.{box_ops,obb}`（新增顶层包；augment/metrics 的两条 ALLOWED_UPWARD 债务就此消除） |
| `tasks.detection.components.{obb_decoder,obb_criterion}` | `tasks.oriented_detection.shared` |
| `tasks.detection.components.{tiling_utils,sahi}` | `tasks.shared.{tiling_utils,sahi}` |
| `tasks.detection.components.{sfs_ff,dbfsaf,wavelet_downsample}` | `tasks.detection.fsdc.<同名>` |
| `autovision.tasks.depth_estimation` / `pose_estimation` | 新建脚手架（无家族，见包 docstring 模板） |

不变：`tasks.detection.avdetr`、`tasks.eomt`（跨三任务架构核）、
registry/CLI 键名与 `LEGACY_CLASS_PATH_REMAP` 全部历史键。

**FSDC 家族删除（2026-10-15）**：`tasks/detection/fsdc/`、registry 键
`fsdc/*`、CLI 分派分支、`tests/test_fsdc_native.py` 与 4 个
`scripts/*fsdc*` 运行脚本移除；deimv2 的离线测试变体 `fsdc-vittest`
更名 `deimv2-vittest`（纯测试用，无发布模型）。旧 FSDC 导出/checkpoint
自此不可加载（类已不存在）。历史设计文档 21/22 保留。
