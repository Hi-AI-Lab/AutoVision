# 30 · tasks 任务优先目录重组（task-first layout, as-built）

> 编号说明：初稿误用 29，与并行分册《29 · AV-DETR productization roadmap》
> 撞号，落地时改为 30。提交信息中的 "design doc 29" 均指本文。

日期：2026-10-15。状态：**已落地**（四个阶段提交，验收见 §5）。
取代设计文档 28 §4 中 `tasks/` 一节的目标树；28 的其余内容（溯源整改、
clean-cut、迁移机制）不受影响。

## 1. 动机

28 落地后的 `tasks/` 是"混排"：检测按家族分组，分割却按
`<backbone>_eomt_<task>` 平铺六包，共享代码两处安家（`eomt/` 顶层、
`detection/components/` 埋两层深，9815 行 / 32 文件）。三个具体异味：

1. `components/` 消费者分布显示 `ema/flat_cosine` 被分类任务反向引用
   （唯一跨任务 reach-through）；`dfine_*` 五件套约 3000 行最大消费者
   是 avdetr；`sfs_ff/dbfsaf/wavelet_downsample` 只服务 fsdc。
2. 有向框（OBB）有独立 metrics 模块、CLI 子命令与公开 API，任务地位
   上不是检测的子目录。
3. `detection/avdetr_object_detection/` 一类命名与父目录冗余；
   `dinov3_*` 包跨包继承 `dinov2_*` 包。

目标：**一级 = 任务类型，二级 = 模型家族**，与 `cli/train_task/` 子命令、
`metrics/` 模块一一对应；新增任务类型有固定动作清单
（四件套模板 + metrics + CLI 子命令 + registry 键）。

## 2. As-built 树

```
src/autovision/tasks/
├── task_model.py / train_model.py / task_model_io.py / task_model_helpers.py
├── shared/                  # 跨任务共享：训练设施（平铺）+ 架构核（子包 eomt/）
├── detection/
│   ├── shared/              # RT-DETR/DFINE 架构核（原 components 主体，
│   │                        #   含 msda + _msda_src CUDA 源）
│   ├── detr/                # 自研主家族（原 avdetr，2026-10-15 再更名）
│   └── rtdetr/ dfine/ deimv2/ rfdetr/ picodet/         # 去后缀
├── oriented_detection/      # OBB 提升为同级任务
│   ├── shared/              #   obb_decoder obb_criterion
│   └── rtdetr/              #   原 rtdetr_oriented_object_detection
├── semantic_segmentation/   # dinov2_eomt dinov3_eomt linear
├── instance_segmentation/   # dinov2_eomt dinov3_eomt
├── panoptic_segmentation/   # dinov2_eomt dinov3_eomt
├── classification/          # 原 image_classification
├── depth_estimation/        # 脚手架（四件套模板见包 docstring）
└── pose_estimation/         # 脚手架（同上）
```

新增顶层包 `autovision/geometry/`（`box_ops.py`、`obb.py` 纯数学，
无 nn 模块）：augment 与 metrics 改为向下导入，**撤销两条
`ALLOWED_UPWARD` 债务**（box math / OBB math）。任务分组目录为
namespace 包（无 `__init__.py`），与 `detection/` 既有惯例一致。

## 3. 兼容性

- **公开 API 零变化**：全部类名、registry/CLI 键、函数签名不变；
  变化的只有模块路径（速查表见 `docs/migration-0.2.md` §"0.3"）。
- **第四代类路径迁移**：`task_model_helpers._MODULE_PATH_MIGRATIONS`
  （15 条模块前缀映射）同时作用于 `model_class_path`
  （`load_task_model`）与 `init_args` 内嵌类引用
  （`_verify_autovision_global`）；前三代
  `LEGACY_CLASS_PATH_REMAP` 键全部保留。
- provenance allowlist：avdetr `config.py` 的历史注释豁免随注册表拆分
  同步为 `config.py` + `variants.py` 两条（理由刷新：R16 已落地）。

## 4. 阶段记录

| 阶段 | 提交 | 内容 |
|---|---|---|
| 1 | 3833f5a | `tasks/shared/`（ema/flat_cosine，解除分类→检测依赖）；depth/pose 脚手架；layering 脚本删除 shim 死配置 |
| 2 | 381bf41 | 15 包重组 + 第四代类路径迁移 + migration-0.2 §0.3 速查表 |
| 3 | 4a1514c | components 按消费者分流（shared/detection·shared/oriented·shared/fsdc/geometry） |
| 4 | c39e20e | avdetr 拆分：`variants.py`（注册表 403 行，config 1135→790）+ `optimizer.py`（`build_avdetr_optimizer` 逐字提取，train_model 1200→1045） |

未做（记录取舍）：teacher 蒸馏与 checkpoint 键修补留在
`AVDETRObjectDetectionTrain` 内（与 self 状态耦合，无可用的纯函数接缝）；
`ms_deformable_attention.py`（零消费者的纯 PyTorch MSDeformAttn 导出
备胎）保留在 `detection/shared/`，待 torch.export 路线定夺。

## 5. 验收

- 全量测试：**1535 passed / 14 skipped / 0 failed**（每阶段各跑一次）。
- 三守卫：ruff（src/tests）全绿；layering 全绿且债务条目 8→6；
  provenance 全绿（allowlist 路径同步）。
- 残留扫描：`detection.components`、旧包路径在 src/tests/scripts 的
  .py/.sh/.toml/.ipynb 全文件类型下仅剩三类**有意保留**——迁移映射键、
  `LEGACY_CLASS_PATH_REMAP` 历史字符串、历史测试文件名 docstring。
- 已知存量（非本次引入，移交后续）：`scripts/` 三个 E402、16 个 I001
  （ledger 期间引入，`ruff check scripts` 范围）；LEDGER.md 有并行会话
  的未提交改动，本重构未附台账条目。

## 6. 附记：FSDC 家族删除（2026-10-15）

方向调整，`tasks/detection/fsdc/`（含 stage 3 归入的 sfs_ff/dbfsaf/
wavelet_downsample）、`fsdc/*` registry 键与 CLI 分派、
`tests/test_fsdc_native.py`、4 个 `scripts/*fsdc*` 一并移除；deimv2
离线测试变体 `fsdc-vittest` 更名 `deimv2-vittest`。详见
`docs/migration-0.2.md` §0.3 附注。

## 7. 附记：家族更名 avdetr → detr（2026-10-15）

自研检测器族 AV-DETR 更名 **DETR**（目录/registry 键/9 个类/函数/19 个
脚本），`avdetr/*`、`ltdetr/*` 双旧键弃用别名；第五代类路径迁移
（模块前缀 + 改名类精确条目）见 `docs/migration-0.2.md` 对应节。

## 8. 附记：eomt 并入 shared/（2026-10-15）

归属规则定为一元：**只服务一个任务的代码进任务目录，跨任务的代码进
`tasks/shared/`**（任务内多家族共享用 `<task>/shared/`）。
`tasks/eomt/` 是唯一跨任务却在顶层安家的包，按规则并入
`tasks/shared/eomt/`（子目录保持架构内聚）。eomt 零依赖任务包，天然
满足 shared/ 契约；六个公开类路径不变，仅 17 处导入改写 + 一条
`tasks.eomt. → tasks.shared.eomt.` 前缀迁移作保险。顶层自此为
8 个任务目录 + 1 个 `shared/`。
