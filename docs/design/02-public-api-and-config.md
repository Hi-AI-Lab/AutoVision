# VTrain 实现蓝图 · 分册 02：公共 API · 配置系统

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §4–§5 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 4. 公共 API 与扩展契约

### 4.1 基线 API 面与启示【基线】

`__init__.py.__all__`（`__init__.py:60-85`）导出：

```python
# 预训练（Trainer 路径）
pretrain(*, out, data, model, method="distillation", method_args=None, embed_dim=None,
         epochs="auto", batch_size=128, num_workers="auto", devices="auto", num_nodes=1,
         resume_interrupted=False, checkpoint=None, overwrite=False, accelerator="auto",
         strategy="auto", precision="auto", float32_matmul_precision="auto", seed=0,
         loggers=None, callbacks=None, optim="auto", optim_args=None, transform_args=None,
         loader_args=None, trainer_args=None, model_args=None, resume=None /*已弃用*/) -> None
# 任务微调（Fabric 路径，7 个任务各一个函数，参数面一致）
train_object_detection(*, out, data, model, steps="auto", batch_size="auto", ..., precision="bf16-mixed",
                       gradient_accumulation_steps="auto", save_checkpoint_args=None,
                       torch_compile_args=None, debug_args=None, ...) -> None
train_image_classification / train_image_classification_multihead /
train_semantic_segmentation(_multihead) / train_instance_segmentation /
train_panoptic_segmentation / train_object_detection
# 推理/资产
embed(*, out, data, checkpoint, format="torch", image_size=(224,224), ...) -> None
export(*, out, checkpoint, part="model", format="package_default", ...) -> None
export_onnx(*, out, checkpoint, batch_size=1, height=None, width=None, precision="32-true",
            simplify=True, verify=True, ...) -> None
predict_semantic_segmentation(*, out, data, model, ...) -> None
benchmark_object_detection(*, out, dataset_name, data, model, ...) -> BenchmarkResult
load_model(model, device=None) -> TaskModel
list_models() -> list[str];  list_methods() -> list[str]
```

启示（VTrain 吸收 / 规避）：

- **吸收**：keyword-only 参数 + 全默认值 + `"auto"`；`resume_interrupted` 与 `checkpoint` 互斥校验（`train_helpers.py:462-469`）；CLI 只有 5 条命令且与 API 共享校验（`CLITrainConfig` 字符串方言，§5.2）；`list_models()` 直接返回排序后的 `"pkg/model"` 字符串联合。
- **规避**：任务训练无 CLI（基线 `_cli.py` 不含 train_object_detection 等）——VTrain 全命令进 CLI；`train()` 是带 FutureWarning 的弃用别名仍占着 API 面——VTrain 弃用即从 `__all__` 移除；`predict` 语义分散（任务模型方法 vs `predict_semantic_segmentation` 命令）——VTrain 收敛为统一 `predict()`。

### 4.2 VTrain API 面【设计】

```python
import vtrain

# ── 预训练/蒸馏（P0）────────────────────────────────
vtrain.pretrain(
    out,                                  # run 目录（必填，禁止默认）
    data,                                 # 图像目录 / 文件列表 / DatasetSpec
    model,                                # student: "pkg/name" | nn.Module | 注册名
    *, method="distillation",
    method_args=None,                     # teacher="dinov3/vitb16", queue_size, ...
    embed_dim=None, epochs="auto", batch_size=128,
    optim="auto", optim_args=None, transform_args=None,
    devices="auto", num_nodes=1, precision="auto",
    seed=0, checkpoint=None, resume_interrupted="auto",  # 互斥；auto=发现 last.ckpt 即续
    callbacks=None, trainer_args=None,    # Lightning 逃生舱
)

# ── 微调（P0）──────────────────────────────────────
vtrain.finetune(
    out, data, model, *,
    task="detection",                     # 注册表：detection | (P1) segmentation | classification
    model_args=None, steps="auto", batch_size="auto",
    optim="auto", optim_args=None,
    transform_args=None,
    eval_args=None,                       # 评测协议！与训练增强解耦（一等公民）
    devices="auto", precision="auto", seed=0,
    checkpoint=None, resume_interrupted="auto",
    callbacks=None, trainer_args=None,
)

# ── 评测/导出/推理（P0）────────────────────────────
vtrain.evaluate(model_or_run, data, eval_args=None, out=None) -> EvalReport
vtrain.benchmark(models: list[str | Path], data, eval_args=None, out=...) -> BenchmarkReport
vtrain.export(model, out, format="onnx", export_args=None)      # onnx | tensorrt
vtrain.predict(model, images, out=None, sahi_args=None)          # 含切片推理
vtrain.load_model(path_or_name, device=None) -> TaskModel        # 统一推理入口

# ── 内省与资产 ─────────────────────────────────────
vtrain.list_models(task=None); vtrain.list_methods(); vtrain.list_datasets()
vtrain.register_model(name, base=..., build=...)   # 装饰器：组合式模型注册
vtrain.register_criterion(name)                    # 自定义损失插件
vtrain.register_dataset_format(name, adapter)      # 自定义数据格式插件
```

API 设计约束：

- 全部 keyword-only（`out/data/model` 除外），新增参数只加不带默认行为变更的关键字。
- 所有函数返回**结果对象**（`FinetuneResult`/`BenchmarkReport`...），携带解析后配置与产物路径，可程序化串联。基线反例：`pretrain()` 返回 `None`，产物只能靠文件系统约定发现。
- `resume_interrupted="auto"`：比基线的 bool 更进一步——run 目录存在 `last.ckpt` 即自动续训（平台批量任务零配置容错）。基线语义（bool，且与 `checkpoint` 互斥）保留为显式路径。

### 4.3 版本与弃用策略【设计】

- 语义化版本 `MAJOR.MINOR.PATCH`；`MAJOR=0` 期间以 MINOR 视作破坏线。
- 弃用流程：`@deprecated(since, removal)` + DeprecationWarning + CHANGELOG 弃用节 + 迁移指引 → 两个 minor 后移除。基线实践参照：`resume` 参数、`reuse_class_head`、`ModelFormat("ultralytics")`、`teacher_url` 均以运行时警告方式弃用且保留一个以上版本。
- CI 弃用检查：测试套件在 `-W error::DeprecationWarning` 下运行。
- **配置 schema 版本化**：run/checkpoint 内配置带 `schema_version`；旧版本由迁移器注册表升级。基线反例：无任何 config 版本机制，checkpoint 仅记录 `version=框架版本` 字符串且从不分支（`_checkpoint.py:58-70`）。

### 4.4 插件系统【设计】

```toml
# 插件包 pyproject.toml
[project.entry-points."vtrain.models"]
my_backbone = "my_pkg.models:MY_PACKAGE"        # 实现 Package 协议的单例

[project.entry-points."vtrain.criteria"]
wiou = "my_pkg.losses:WIoUCriterion"

[project.entry-points."vtrain.data_formats"]
dota = "my_pkg.data:DOTAFormatAdapter"
```

`importlib.metadata.entry_points()` 惰性发现；注册表合并时重名即报错；插件 import 失败不影响核心启动。

【差异】基线的扩展注册全部是**进程内 dict**：方法注册表 `_method_name_to_cls()`（`_methods/method_helpers.py:53-67`，类名小写 + 别名映射）、优化器注册表 `_OPTIM_TYPE_TO_ARGS`（`_optim/optimizer_helpers.py:26-31`）、任务模型注册表 `TASK_TRAIN_MODEL_CLASSES`（`_commands/train_task_helpers.py:139-154`）、模型注册表 `ModelRegistry`（装饰器注册，`_configs/model_registry.py:18-89`）。第三方不改源码就无法接入——VTrain 以 entry-points 替代，内置注册表保留 dict 形态作为实现细节。

---


## 5. 配置系统

### 5.1 基线校验机制【基线】

单一 Pydantic 基类（`_configs/config.py:23-70`）——VTrain 照抄整套策略：

```python
class PydanticConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",              # 未知键即错（报错友好机制的基石）
        frozen=False,                # 允许原地 resolve_auto（见 5.3）
        strict=True,                 # 严格类型（CLI 方言局部放宽）
        validate_assignment=True,    # resolve_auto 的每次赋值都会被重新校验
        validate_default=True,
        protected_namespaces=tuple(  # 使 model/model_args 成为合法字段名的关键技巧
            n for n in dir(BaseModel) if n.startswith("model_")),
    )
```

- 子类 `model_config` 与基类**合并**：`TrainConfig` 加 `arbitrary_types_allowed=True`（接受 nn.Module），`CLITrainConfig` 关闭之（强制字符串方言）。
- 单一校验入口 `_configs/validate.py:40-53` `pydantic_model_validate(model, obj)`：捕获 `ValidationError`，逐条经 `_pydantic_error_msg` 美化后抛 `ConfigValidationError("Found N errors in the config!...")`。
- 错误美化规则（`validate.py:93-185`，无 did-you-mean）：
  - `missing` → `Missing key: 'a.b'`；`extra_forbidden` → `Unknown key: 'b'`
  - 类型错误 → `Invalid type for key 'a': Input should be a valid integer but got '1' with type 'str'`（键路径点分、列表下标 `[i]`）
- 错误层级（`errors.py`，33 行）：`LightlyTrainError` → `UnknownModelError` / `ConfigError(→ConfigValidationError)` / `UnresolvedAutoError`。

### 5.2 三配置方言与 YAML 装载【基线】

```
TrainConfig          富类型（Path/nn.Module/dict|Pydantic 联合）      ← Python API
FunctionTrainConfig  子配置字段收窄为纯 dict                          ← Python API 内部
CLITrainConfig       全字符串（out:str, model:str...）                ← CLI（OmegaConf k=v）
```

三者继承收敛，校验逻辑只写一遍（`_commands/train.py:526-580`）。任务侧同构：`TrainTaskConfig` + 每任务子类；检测数据用 **discriminated union**（`Field(discriminator="format")`，默认 `"yolo"` 由 `set_default_data_format` 预注入，`train_task.py:2087-2101`、`_commands/data_helpers.py:86-93`）。

YAML 装载的关键技巧（VTrain 必抄）：`TrainTaskConfig` 的 `data` 字段带 `mode="before"` 校验器——路径则 fsspec 打开 + `yaml.safe_load`；**先按 union 全体成员的合法键并集过滤掉外来键，再交给 extra="forbid" 的 Pydantic**，否则用户 YAML 里的无关键会直接炸（`_commands/data_helpers.py:19-59`）。YAML 路径记入隐藏字段 `data_config_file`（`exclude=True`），供后续相对路径解析。

### 5.3 auto 解析机制【基线】

- **表示**：无专用类型，`X | Literal["auto"]` 字符串哨兵；递归检测 `PydanticConfig.has_auto()`（`config.py:59-86`）。
- **解析**：每个配置类自带 `resolve_auto(<依赖参数>) -> None`（鸭子类型方法，无公共接口），**原地变异**字段（`frozen=False` + `validate_assignment=True` 保证变异仍被校验）。
- **兜底**：标量读取用 `no_auto(v)`（残留即抛 `UnresolvedAutoError`，报错文案明言"非用户错误"）；整配置用 `assert_config_resolved(config)`。
- **解析顺序由命令流决定**（这是设计文档必须固化的事实）：

```
pretrain 流(_commands/train.py):
  transform_args → float32_matmul_precision → epochs → strategy/precision
  → batch_size/num_workers → optim_args(wrapped_model) → method_args(scaling_info, optim_args, wrapped_model)
  注: epochs="auto" 依据 MethodArgs 的 ClassVar(default_steps 优先于 default_epochs)，
      epochs = ceil(default_steps * batch_size / dataset_size)（train_helpers.py:420-452）

task 流(_commands/train_task.py):
  data(YAML) → save_checkpoint_args → checkpoint 解析(可能改写 model 名) → steps/model 家族
  → batch_size(默认 TrainModelArgs.default_batch_size)/num_workers → gradient_accumulation_steps
  → model_args → metric_args → transform_args(!! 需要 total_steps/train_num_batches/grad_accum，
     所以必须在 dataloader 建好之后) → logger_args
```

【设计】VTrain 沿用同一依赖链，并加两条基线没有的约束：解析完成强制 `out/config.resolved.yaml` 落盘（审计锚点，基线只在日志与 logger hyperparams 里出现且不可复跑）；错误美化增加 difflib 最近邻拼写建议（独立单测覆盖）。

### 5.4 注册表机制【基线】

| 注册表 | 机制 | 出处 |
|---|---|---|
| 方法 | 类名小写 dict + 别名（`distillation`→DistillationV3，`distillationv1`→Distillation）；`HIDDEN_METHODS` 支持隐藏 | `_methods/method_helpers.py:19-67` |
| 优化器 | `OptimizerType` enum + `_OPTIM_TYPE_TO_ARGS` dict | `_optim/optimizer_type.py:11-15`、`optimizer_helpers.py:26-31` |
| 任务模型 | `TASK_TRAIN_MODEL_CLASSES`，按 `task` + `is_supported_model(model_name)` 双键分发 | `_commands/train_task_helpers.py:139-154, 777-784` |
| 模型（任务） | `ModelRegistry[ConfigT]`：装饰器注册字符串别名 → Pydantic 配置类，元数据含 `DownloadableCheckpoint(url, sha256)` 与 `ModelAlias(alias_for)`；`include_in_model_list=False` 可藏模型 | `_configs/model_registry.py:18-89` |
| 可下载权重 | `DOWNLOADABLE_MODEL_URL_AND_HASH`（url + sha256 字段存在但**只用于展示**，下载路径无校验） | `task_model_helpers.py:92-162` |

### 5.5 VTrain 实现规格【设计】

```
src/vtrain/_configs/
├── base.py           # PydanticConfig（照抄 5.1 全部设置）+ ConfigError 层级
├── validate.py       # pydantic_model_validate + prettify(带 did-you-mean) + no_auto/assert_resolved
├── migrations/       # schema_version → 迁移器注册表（v1 起即占位）
└── registry.py       # 泛型 Registry[T]（覆盖 5.4 四类内置注册表的公共机制）
```

字段级约定：所有对外配置继承 `PydanticConfig`；`X | Literal["auto"]` 只允许出现在明确列入「auto 解析链」的字段；嵌套子配置一律 `dict | Config | None` 联合（API 直传实例、YAML/CLI 传 dict）。

---
