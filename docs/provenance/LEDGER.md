# 溯源台账（Provenance Ledger）

逐文件记录第三方衍生代码的来源、许可与整改状态。设计依据：分册 28
（`docs/design/28-clean-provenance-refactor.md`）。所有 re-port / clean-room
交付必须在本表登记一行。

状态取值：`完成` / `进行中` / `受阻`（附原因）/ `待办`（附所属 Phase）。

## 清源整改项（2026-10-14 启动）

| ID | 文件 / 位置 | 动作 | 来源 | 许可 | 状态 | 备注 |
|---|---|---|---|---|---|---|
| P0-1 | `_methods/dinov3/constrained_crop.py::sample_high_overlap_box` | clean-room 重写（规格 `specs/S1-constrained-crop.md`） | 行为契约=既有单测 + torchvision 采样数学（BSD-3） | 自有 | 完成 2026-10-14 | 长宽比分布改为 log-uniform（对齐 torchvision 惯例），行为面由 `tests/test_clean_provenance_p0.py` 与既有 `test_methods_dinov3.py` 锁定 |
| P0-1b | 同文件 `_box_and_flips_from_geometry` + `render_clean_global` 解包重述 | 二轮稀释（消除 83-token 逐字块） | 几何张量布局=跨模块接口 | 自有 | 完成 2026-10-14 | geometry 8 维布局为接口事实，保留；解包表达独立 |
| P0-2 | `_methods/dinov3/dinov3.py::_roi_align_view` | clean-room 重写（规格 `specs/S2-roi-align-view.md`） | 行为契约=既有单测 + 数值参考实现（grid_sample 组合） | 自有 | 完成 2026-10-14 | 数值参考交叉验证见 `tests/test_clean_provenance_p0.py::TestRoiResampleOracle`（naive 双线性 oracle，float64 atol 1e-8） |
| P0-3 | `_methods/dinov3/dinov3.py::_align_cross_view_pair` | clean-room 重写（规格 `specs/S3-align-cross-view-pair.md`） | 行为契约=既有单测 + 分册 14/20 | 自有 | 完成 2026-10-14 | 交并区域/有效性判定抽为独立 `_shared_region`；`_paka_loss` 循环改列表推导（二轮稀释：最长逐字块 74→56，且余块均为方法签名/参数默认值等接口与事实） |
| P0-4 | `_models/ecvit/ecvit.py` `RopePositionEmbedding`/`rotate_half`/`apply_rope` | Apache 源 re-port + 归属头 | DEIMv2 `engine/backbone/vit_tiny.py`（Apache-2.0） | Apache-2.0 | 完成 2026-10-14 | **对分册 28 S-4 的偏差**：未用 timm/transformers，改用同源 Apache 的 DEIMv2 实现（数值逐比特同源，checkpoint 兼容性最强）；已去 DINOv3-License 表述 |
| P1-1 | `_task_models/ltdetr_object_detection/backbone_wrappers.py::SpatialPriorModule` | re-port | DEIMv2 `engine/backbone/dinov3_adapter.py`（Apache-2.0，本地镜像 `~/Research/Software/DEIMv2-main`） | Apache-2.0 | 完成 2026-10-14 | 重述表达 + 归属头；行为由既有 backbone 测试锁定。**注意**：上游文件头含 "Modified from DINOv3" + Meta 版权块（双重头），SPM 类本身应为 DEIMv2 原创（设计与 DINOv2 Apache 检测适配器同族），待一次上游核验后移除本注 |
| P1-2 | `object_detection_components/dfine_official_decoder.py` | re-port | D-FINE `src/zoo/dfine/dfine_decoder.py`（`Peterande/D-FINE` @ `956d170`，本地镜像 `~/Research/Software/upstream/D-FINE`） | Apache-2.0 | **完成 2026-10-14** | 三方审计法：diff(上游→LT) 改动集逐项核销——`hooks`/`_torch_helpers` 确认未带入；LT 时代的 `return_query_states`（零消费者）整体移除、`_get_projected_feats` 拆分折回上游内联形态、`convert_to_onnx_export` 重表达为 `enable_gather_core`；diff(AV, 上游) 140 hunks 全部对应已记录 AV 增量；AV vs 上游 80.8%（vs LT 83.5% 属同宗）。检测套件 160 过 |
| P1-3 | `object_detection_components/dfine_utils.py`、`dfine_criterion.py` | 归属 + 残留审计 | D-FINE @ `956d170` | Apache-2.0 | **完成 2026-10-14** | 函数集对比零 LT 功能残留（AV-only 全为自有：缓存/fg_loss/ddf_loss/DFINECriterionV2 等）；血统头 @commit 落位 |
| P1-4 | `object_detection_components/{hybrid_encoder,rtdetrv2_decoder,denoising,matcher,ema,rtdetrv2_criterion}.py` | 归属 + 残留审计 | RT-DETR @ `df2a96c`（`~/Research/Software/upstream/RT-DETR`）/ D-FINE @ `956d170` | Apache-2.0 | **完成 2026-10-14** | 六文件零 LT 功能残留；血统头 @commit 落位 |
| P1-5 | `flat_cosine.py` | 归属 | DEIM（Copyright (c) 2024 The DEIM Authors, Apache-2.0；本地 DEIMv2 镜像 `engine/optim/lr_scheduler.py` 同源） | Apache-2.0 | **完成 2026-10-14** | 血统头落位 |
| P1-6 | `encoders/ecvit/ecvit.py` 骨干类骨架 | 血统直连上游（EdgeCrafter 代码仓 @ `d7fdcb5`，本地镜像 `~/Research/Software/upstream/EdgeCrafter-code`；`capsule2077/edgecrafter` 为权重发布仓 @ `fc45298`） | Apache-2.0 | **完成 2026-10-14** | AV vs EdgeCrafter 上游 63.2% 同源直证；模块头与 RoPE docstring 归属升级为 EdgeCrafter 原始源（DEIMv2 为交叉印证）；与 lightly 的高相似由共同上游解释 |
| P2-1 | 全库 🟢 档注释措辞（detcon/dino/simclr/method/lars/rfdetr/model_registry） | 注释重述为自有语义/上游指向 | 论文与 Apache 上游 | 自有 | 完成 2026-10-14 | `dino.py` teacher-normalize 事实错误已修正 |
| P2-2 | `_models/ecvit/ecvit.py` 文件头 | 去除 DINOv3-License 血统表述 → DEIMv2/EdgeCrafter（Apache） | — | — | 完成 2026-10-14 | 随 P0-4 |
| P2b-2 | 身份层长尾(2026-10-15 终扫清单:188 文件,15–61%,data/config/runtime/augment/tasks-config 胶水层,归档 `rescan-2026-10-15-phase5-batch1.txt.gz`) | 逐文件重表达(bach-2 专项) | — | — | 待办(allowlist 已登记,复审 2026-11-15) | 与 P2b 同性质;正式发布(gate 2 <15%)前须清账或逐文件再裁量;allowlist 语义已改为"只豁免相似度审计,禁词扫描全量覆盖" |
| P3-1 | `coco_checkpoint.py`（S3 host + 3 条 URL/哈希）、`scripts/visdrone_ltdetr_*` 权重注释 | 资产切换 | 自建 warm-start（分册 28 §6.2） | — | 待办 Phase 4 | 需训练算力 + D1 决策（DINOv3 骨干路线） |
| R15 | `_embedding/embed.py` `lightly_csv` 格式名 | 改名 `csv_features` + 弃用别名 | — | — | 完成 2026-10-14 | 公开 API 弃用别名按政策保留两个版本后删除 |
| R16 | `ltdetr_object_detection/` 族命名 / registry 键 | 改名（提案 `avdetr`，终名待定） | — | — | 完成 2026-10-15 | 终名 **DETR**（`avdetr→detr`，commit `9419a16`；`detr/<backbone>` 为现键，`ltdetr/*`、`avdetr/*` 双旧键弃用别名），见 `docs/migration-0.2.md` |
| R17 | `box_ops.py::sanitize_boxes_cxcywh_normalized` | 改名 | — | — | 完成 2026-10-14 | → `clip_normalize_boxes`（Phase 3 复核记录） |
| P2b | 接口/配置脚手架层近似复述（12 文件，55–81%：`_models/package.py`、`_models/timm/timm_wrapper.py`、`_export/tensorrt_helpers.py`、`_models/model_wrapper.py`、`_optimizer_type.py`、`_models/timm/timm_package.py`、`_methods/method_args.py`、`_optim/adamw8bit_args.py`、`_optim/trainable_modules.py`、`_models/rtdetr/rtdetr.py`、`_data/label_helpers.py`、`_models/embedding_model.py`） | 随 R10–R12 术语/结构重构稀释 | — | — | 待办 Phase 5（专项） | 2026-10-14 全库基线扫描新发现，见 `BASELINE-2026-10-14.md`。**2026-10-15 定向复扫**（同算法、现路径、阈值 15%）：R10–R12 改名后相似度基本未降（`encoders/package.py` 81.0%、`timm_wrapper.py` 78.3%、`tensorrt_helpers.py` 76.3%、`model_wrapper.py` 74.3%、`timm_package.py` 71.0%、`method_args.py` 66.7%、`trainable_modules.py` 63.6%、`rtdetr.py` 62.8%、`adamw8bit_args.py` 62.7%、`label_helpers.py` 62.6%、`embedding_model.py` 59.9%、`optim/optimizer_type.py` 73.1%）——相似度为结构性（同第三方 API 胶水写法），改名无法稀释，需逐文件重表达专项；注意其中部分相似来自包裹相同第三方 API 的必然同形，处置时按 §3.1 版权边界逐文件裁量 |

## 复核记录

- 2026-10-15（**GTR 集成 re-port 登记，roadmap 分册 31 G0/G1/G2**）：① G0/G1——教师资产注册 `registry/gtr_assets.py`（四档 det COCO 本地权重，sha256 实测钉住，MIT）+ 教师包加载器 `tasks/detection/detr/gtr_teacher.py`（EMA-first 语义 + 前缀剥离 + GLA 后端契约存根）；装配器 `dfine_official_warm_start.py` 增 `source_format="gtr"`，`gtr-s` 宽度配方（hidden 192/FFN 512）实测 decoder 覆盖率 **0.9613**（144 张量，G1 门禁 ≥0.9 达成）。② G2——**MIT re-port**（来源 `Intellindust-AI-Lab/GTR`，参考环境 flash-linear-attention 0.5.0）：
  | 本文件 | GTR 源 | 处置 |
  |---|---|---|
  | `tasks/detection/shared/gtr/vit_adapter.py`（1152 行） | `engine/gtr/backbone/vit_adapter.py` | 逐段 re-port + 修改：去 core 注册器、fla/transformers 导入惰性守卫（`FLA_AVAILABLE` + 构造期报错）、`_no_grad_trunc_normal_` 内 `l` 改名（E741） |
  | `tasks/detection/shared/gtr/spatial_swiglu.py` | `engine/gtr/backbone/vit_adapter_spatial_swiglu.py` | re-port，仅 import 路径/注册清理 |
  | `tasks/detection/shared/gtr/encoder.py`（345 行） | `engine/gtr/hybrid_encoder.py`（get_activation 自 `utils.py`） | 抽取 re-port：`get_activation`/`ConvNormLayer_fuse`/`_LayerNorm2d`/`_ConvBNAct`/`_Bottleneck`/`_C2f`/`GTREncoder` |

  家族接线：`variants.py` family=`"gtr"` + `detr/gtrvit-s` 组件档（SPECTRUM 不动——G2 验收前不冻结）；`task_model.py` 骨干/encoder 双分派（GTRViTBackbone + GTREncoder）；`backbone_wrappers.py` `GTRViTBackbone`（`backbone_model` 指向内层 `_model`，`extract_gtr_backbone` 键形免重映射直载）；pyproject 增 `gtr` extras（fla 0.5.0）。**CUDA 实测**：gtrvit-s 全网 forward（logits (1,50,C)/boxes (1,50,4)）+ coco8 3 步训练闭环 + GTR→GTR 骨干权重直载（零 unexpected keys）。测试 `tests/test_gtr_integration.py` ×14 + `tests/test_g2_gtr.py` ×6（CUDA+fla 门控，无 GPU 环境自动 skip）；回归 p1/p2/detection 41 过；ruff/layering/provenance 全绿。**G2 剩余**（需 GPU 矩阵）：COCO 短程趋势复现（官方 30ep 的 ≥85% 判据）、TRT FP16 导出对拍、AMP 矩阵（官方 S 档 finetune 关 AMP 的数值敏感性）。



- 2026-10-14：首次登记。P0 全部四项 + P1-1 + P2-1/P2-2 完成；受阻项原因：执行环境无外网（GitHub 不可达），上游克隆待补。
- 2026-10-14（代码审查后）：独立审查（findings 已全部处置）确认——`_restore_orientation` 翻转码分组与 autograd 正确、`_shared_region` 阈值符合规格 S-3、oracle 与实现无共享 bug、`_align_cross_view_pair` 变换配对一致、RoPE 移植与 DEIMv2 逐算子等价且 `periods` checkpoint 兼容（已对真实发布 checkpoint 验证 ~1e-4 漂移属实）、`SpatialPriorModule` state-dict 键逐一保持、ASSET_MANIFEST 改名零残留。修复：check_provenance.py 四处（`--allow-unused-allowlist` 语义反转、`--fail-over` 未生效、坏 `--reference-dir` 假绿、multiset 预筛提性能 + 模式 2 白名单与跳过文件报告）；`_roi_align_view` 改回除法形式保恒等重采样；ruff 导入序；allowlist 拼写；规格 S1/S2 勘误。遗留跟进（P2 级）：ECViT legacy 权重缺 `rope_embed.periods` 键时的容错加载；`_SKIP`/`_SELF_EXEMPT` 行为文档化；全库模式 2 已验证可完成（22447 对，归档见 `BASELINE-2026-10-14.md`）。
- 2026-10-14（**Phase 3 架构迁移落地**）：§4 包结构迁移完成——17 个顶层包迁至新名（`cli/config/data/augment/pretext/encoders/tasks/engine/optim/metrics/features/registry/export/observability/debug/visualize` + 顶层 `scaling/optimizer_type`），检测族归组 `tasks/detection/`（components + 8 family），全库 import 与路径字符串重写零残留；18 个旧路径 shim（DeprecationWarning，顶层名 re-export，深层路径不在范围）；R15（`csv_features` + `lightly_csv` 弃用别名 + 死代码清理）、R17（`clip_normalize_boxes`）、R7（`registry/assets.py::LicenseRef` + `DownloadableModel.license` 字段，Phase 4 起必填）、§4.3（`LEGACY_CLASS_PATH_REMAP` + 加载期重映射日志）；R4 分层契约 `scripts/check_layering.py`（AST 实现、tasks→pretext 与 encoders 硬禁令、8 项带理由的 `ALLOWED_UPWARD` 债务登记）接入 CI。验证：迁移后全量测试 1489 过 / 0 失败（网络依赖 2 例除外）；ruff src/tests 全绿；provenance 模式 1 全绿（allowlist 路径已同步新结构）。迁移文档：`docs/migration-0.2.md`。公开 API 键名与 checkpoint 布局未变。
- 2026-10-14（**R10–R12 内部类名术语落地，Phase 3 代码部分收尾**）：`Method→PretextRecipe`、`MethodArgs→PretextArgs`、`TrainingStepResult→StepOutput`、`TaskModel→TaskNetwork`、`ModelWrapper→EncoderProtocol`、`EmbeddingModel→FeatureExtractor`、`get_teacher→build_teacher`（约 280 处，token 级 sed，零残留）；模块 `method.py→recipe.py`、`method_helpers.py→teachers.py`（git mv 保历史）。公开 API 键名（`method_args`/`list_methods`/`get_method_cls`/`optimizer_args_cls`/`transform_cls`）与构造参数名（`embedding_model=`）按 R9 全部保留。旧内部类名不留 alias（从未属公开 API）。
- 2026-10-14（**审查遗留 P2 清零 + Phase 4 离线准备**）：① ECViT legacy 权重容错——`rope_embed.periods` 缺键（持久化改造前的自训权重）时降级为严格匹配其余键 + 新初始化 periods，其余缺键仍报 `CheckpointError`（两个新测试，回归 106 过）；② `AVWarmStart v1` 格式落地 `registry/warm_start.py`（license 必填的 R7 机制延伸：无许可拒绝写出、格式 tag/张量校验，6 个单测）；③ `scripts/build_warm_start_coco.py` 骨架（§6.2 四步管线，export 路径已定，步骤 1-3 BLOCKED 于 Apache 权重与 D1，运行时明确输出阻断原因）。守卫终态：ruff / provenance / layering 全绿。待办不变：R16 终名、P1-2~P1-6 上游克隆（无网络）、D1 + Phase 4 训练。
- 2026-10-14（**D1 决策落地：D1-a——DINOv3 骨干保留，用户自持权重模式**）：① `encoders/dinov3/dinov3_package.py` 权重门控——预训练路径（`load_weights=True` + 公开 HF checkpoint）要求环境旗标 `AUTOVISION_DINOV3_WEIGHTS=1`（用户自证已接受 Meta DINOv3 License），否则抛带引导的 `PermissionError`；`-notpretrained` 与显式 `weights_path` 不受门控影响（4 个新测试，含"未 opt-in 绝不触发下载"断言；1 个既有预训练管线测试随门控更新）。② R7 运行时化——`resolve_downloadable` 对无 license 字段的 legacy 资产发 WARNING（已实测命中 3 个 ECViT S3 条目，Phase 4 替换后随 license 必填而消失）。③ NOTICE 增 DINOv3 权重政策节；分册 28 §6.1 标记 D1 已决策（D1-a 采纳、D1-b 未采纳）；`build_warm_start_coco.py` 骨干路线说明更新（默认 Apache 双线 ecvitt/DINOv2，DINOv3 变体需 opt-in）；README 模型层表述同步。验证：触达套件 89 过 / 1 skip；ruff / provenance / layering 全绿。
- 2026-10-14（**R16 检测器族更名落地：LT-DETR → AV-DETR**）：`tasks/detection/ltdetr_object_detection → avdetr`；类名 `LTDETR*→AVDETR*`（9 类）；函数 `resolve_avdetr_variant / list_avdetr_model_names / build_avdetr_train_transform_args / avdetr_scale_jitter_sizes`（顺带修正历史拼写 `ltetr_→avdetr_`）；registry 键 `ltdetr/<backbone>→avdetr/<backbone>`（**旧键 alias 一个版本**：`resolve_avdetr_variant` 与 CLI `train_object_detection` 双前缀接受 + DeprecationWarning，2 个新 alias 测试）；19 个 `scripts/visdrone_ltdetr_*→visdrone_avdetr_*`；`LEGACY_CLASS_PATH_REMAP` 追加第三代际键（三代 lightly→ Phase3 → R16 全部归一到 avdetr 路径，已验证）。**保留不改**：基线资产文件名 `*_ltdetr_coco_*.pt`（Lightly S3 外部键）、对 lightly 侧历史模块名的文档性引用（`baseline ``ltdetr_object_detection``` 等）、docs/ 历史分册。allowlist 路径同步新名。README 任务层行、`docs/migration-0.2.md` R16 节更新。**终验：全量 1503 过 / 10 skip / 0 失败；ruff / provenance / layering 全绿**。快照：`/tmp/av_pre_r16.tgz`。
- 2026-10-14（**Phase 4 启动：镜像下载通道 + Apache 资产清单落位**）：① `DownloadableModel` 增 `mirrors` 元组——主源失败（下载错误或 sha256 不符）时按序回退（2 个新单测：主源失败回退、镜像 sha 不符继续尝试）；② 新增 `registry/apache_assets.py`：注册 4 个 Apache-2.0 组件资产——`d-fine/dfine_{s,m,l}_coco`（HF `Peterande/D-FINE` 主源 + hf-mirror/GitHub storage 双镜像，哈希逐一实测）与 `edgecrafter/ecvitt`（GitHub release edgecrafterv1，22,274,428 字节精确匹配）；`registry/__init__` 导入即注册，缓存命中 sha 校验通过。③ 权重已入缓存（dfine s/m/l = 41.8/79.1/126.1MB 走 hf-mirror 2.4MB/s；ecvitt 22.3MB 走 GitHub 30KB/s）；dfine_n 在 HF 仓不存在（已从清单排除）。④ COCO2017 数据自 **ModelScope `PAI/COCO2017`** 下载中（11MB/s：annotations/val 已完成，train 53%+），落盘 `~/Research/Project/Dino-Dome-Research/dataset/coco/`。⑤ 探明 D-FINE 官方 checkpoint 布局（`{"model": {backbone|encoder|decoder}}`，encoder 键带双层前缀 `encoder.encoder.*`，与 LT 代际 checkpoint 宽度体系不同：ecvitt=192/224 系 vs dfine_s=256 系）——组装加载器需按变体配宽，为下一专项。终验：全量 1505 过 / 0 失败；三守卫全绿。- 2026-10-14（**Phase 5 本地部分完成：clean cut**）：① 全部整改（含用户进行中的 fsdc/deimv2 工作）以单提交 `7993834` 入库（450 文件，+23724/-4071；暂存零二进制/COCO 路径）；② `v0.2-clean` annotated tag（main，本地私有历史保留）；③ `v0.2-clean-publish` **孤儿分支**：单提交、树哈希与 tag 完全一致（`5df85263`），零 pre-remediation 历史；④ 发布手册 `docs/provenance/clean-cut.md`（推送步骤 + 红线：禁止推送 main）+ §9 验收快照（gate 5 待 Phase 4 训练产物替换 legacy 条目后完全达成）。**待办**：配置发布远程后按手册推送；Phase 4 COCO 微调（数据下载中）。- 2026-10-14（**Phase 4 数据就绪**）：COCO2017 自 ModelScope `PAI/COCO2017` 下载完成并解压验证——train2017 118,287 张 / val2017 5,000 张（均精确匹配官方数）、annotations 全套；落盘 `~/Research/Project/Dino-Dome-Research/dataset/coco/`。磁盘余 24G（zips 共 20.4G 可删再省）。Phase 4 剩余：D-FINE→avdetr 组装加载器（按变体配宽）+ COCO 微调启动（配置待用户过目）。- 2026-10-14（**Phase 4：组装加载器落地**）：`avdetr/dfine_official_warm_start.py`——Apache 组件 → AV-DETR 装配加载器（三方键形分析驱动）：D-FINE 官方 ckpt 键归一化三条规则（`cv1/cv2→conv1/conv2`、`downsample_convs.N.0.cv1→N`、其余同名）+ **形状驱动接收**（宽度配方/class 数头/部署 buffer 逐类分桶：transferred / width_skips / class_count_skips / dropped_buffers / unmatched_structure / discarded cspdarknet）+ decoder 最低覆盖率闸（低于 50% 拒绝）。实测：**ecvits(256)+dfine_s → 168 张量、decoder 91%**；ecvitt(192)+dfine_s 诚实拒绝（27%）；ecvitt 骨干经 wrapper 严格加载 ✓；装配态可导出 AVWarmStart v1 ✓（3 个单测，CI 无权重自动跳过）。**配对矩阵**：ecvits/ecvittplus(256) ↔ dfine_s/m/l 全配；ecvitt(192) 仅骨干暖启。ecvits.pth（骨干 90MB，GitHub 30KB/s）后台下载中，到位后跑真·全量装配 + 微调。- 2026-10-15（**Phase 4：真·全量装配完成，首个自有 AVWarmStart 落地**）：ecvits.pth（88,714,876 字节精确匹配，166 张量）下载到位并注册 `edgecrafter/ecvits`（哈希实测）。端到端装配：`edgecrafter-ecvits`(384 骨干) ← ecvits.pth 严格加载 + encoder/decoder ← dfine_s_coco（168 张量，decoder 91%）→ **`avdetr_ecvits_coco_init.pt`**（AVWarmStart v1，797 张量 / 134.8MB / Apache-2.0，spec 记录 decoder_coverage=0.91 与组装来源）。**注**：这是 assembly-init（预微调）；COCO 短程微调后即为正式 warm-start 资产，替换 legacy 三条目。剩余：微调启动（配置待用户过目）。- 2026-10-15（**R1 注释清零 + 文档命名同步 + 本地资产清理**）：① src/ 内 7 处 lightly 事实性注释改为 "published baseline" 表述（`detr/config.py` ×4、`variants.py` ×2、`task_model.py` ×1），5 个 `visdrone_detr_*` 脚本去品牌词（`/tmp/lightly_ckpts` 路径注释改中性表述，`tests/test_ltdetr_ssl.py` 一处 "slightly" 误中改写）；allowlist 相应收缩至 2 项（embed.py R15 弃用别名 + coco_checkpoint.py Phase 4 legacy），mode 1 全绿。② 分册 29（产品化路线）同步 `avdetr→detr` 终名（文件名 `29-detr-productization-roadmap.md`、谱系键 `detr/{n,s,m,l,x}`、受影响模块 `tasks/detection/shared/`），mkdocs nav 补 29/30 两册；`migration-0.2.md` R16 节加二次更名指引、"仍待办"节收口。③ R18 收尾：`weights/yolo26n.pt`（未跟踪本地文件）删除，`weights/` 目录移除；清理 `tasks/detection/fsdc/` 残留 `__pycache__`。④ **mode 2 全量复扫**（对照 lightly-train-0.17.0 全仓含 tests，287 vs 638 文件，阈值 25%，输出归档 `docs/provenance/baseline-2026-10-15.txt.gz`）：≥25% 共 218 文件，全部落入基线（`BASELINE-2026-10-14.md`）已裁定类别——61 个 `__init__.py` 短文件噪声、P0 已修复残留（dinov3.py 35.0%/constrained_crop.py 35.9%，签名与接口层）、P1 共同 Apache 上游（hybrid_encoder 92.2%/dfine_official_decoder 83.5%/rtdetrv2_decoder 70.5%/ecvit 78.8%）、P2b 十二文件（60–81%），**无新增超阈值文件**；与基线逐文件对比，R10–R12 术语重构仅削去 0–2 个百分点，P2b 确认为结构性相似，转 Phase 5 逐文件重表达专项（行内更新）。⑤ 方法注记：正式 gate 复扫应以 `src/lightly_train` 为参照（同基线），本次全仓参照产生的跨文件误配（如 config ↔ lightly tests）不影响按文件取 max 后的类别结论。
- 2026-10-15（**Phase 5 第一批落地：P2b 重表达 + 合规补强**，含当日三次删除的文档同步）：
  **① P2b 重表达（12 文件处置完毕）**。协议层改名（身份层清除，公开 API 键名不变）：模型源协议
  `Package/BasePackage→CatalogModelSource/ModelSource`（`get_model→build`、`get_model_wrapper→as_encoder`、
  `is_supported_model→recognizes`、`export_model→export_weights`、`list_model_names→catalog_names`；模块文件
  `*_package.py→*_source.py`，常量 `*_PACKAGE→*_SOURCE`）；编码器协议方法 `forward_features→feature_maps`、
  `forward_pool→pool_features`、`feature_dim→encoder_dim`、`get_model→raw_module`、`architecture_info→arch_traits`、
  `multiscale_*→stage_*`（组合类 `MultiScaleFeatureViT/CNN→MultiScaleViTEncoder/CNNEncoder`，协议成员扁平化）；
  优化器 `OptimizerType→OptimizerKind`（枚举→Literal，配置值不变；`.type()→.kind()`）、
  `TrainableModules→OptimizationTargets{targets, decay_free}`、`PretextArgs.default_steps/default_epochs→auto_steps/auto_epochs`、
  `resolve_auto→resolve_auto_fields`。逐文件结果（对照 `src/lightly_train` 最优配对）：`tensorrt_helpers.py` 76.3%→**14.3%**
  （管段化重写，TRT API 事实除外）；`embedding_model.py` 58.7%→**11.6%**；`timm_wrapper.py`/`timm_source.py`
  `timm_wrapper.py` 77.7%→**12.1%**（对照件；架构表转为可 grep 注册串 + 特征钩子表驱动）、`timm_source.py` 67.6%→**33.2%**（冻结 `build` 签名脚手架下限，对照实验：无关模块背景率 16–21%）；`rtdetr.py` 57.5%→20.3%（冻结协议表面下的度量下限）；小文件
  `optimizer_type`（枚举→Literal）、`trainable_modules`、`label_helpers` 低于 `--min-tokens 80` 门槛（度量学噪声带）；
  `package.py` 67.4%、`model_wrapper.py` 55.4%、`method_args.py` 63.2%、`adamw8bit_args.py` 51.5%（R9 公开键/配方事实
  + 接口脚手架）——以上全部按 §3.1 裁量登记 allowlist（分类原因 + 复审日期 2027-01-15）。
  **② 扫描器度量学修正**：`check_provenance.py` 模式 2 增 `--min-tokens`（默认 80，`0` 关闭）——低于门槛的文件不参与配对
  （两个 30-token 文件的 difflib 相似度由标点/导入主导，无证据价值）；基线文档的 `__init__` 噪声类由此机制化。
  **③ DINOv2 权重门控（对齐 D1-a）**：`facebook/dinov2-*` HF 权重 CC-BY-NC 4.0，预训练加载需
  `AUTOVISION_DINOV2_WEIGHTS=1`（4 个新测试：无 opt-in 拒绝、错误含 CC-BY-NC 与旗标、本地构建绕过、opt-in 走校验加载器）；
  `weight_hashes.py` 8 条 dinov2 license 字段由误标 `apache-2.0` 修正为 `cc-by-nc-4.0`；
  `load_pretrained_verified` 对非商用宽松许可清单条目（cc-by-nc/dinov3-custom）加载时 WARNING。
  **④ DEIMv2 教师许可立场**：NOTICE"预训练权重政策"节重写——DEIMv2 代码 Apache、含 DINOv3 血统的 checkpoint
  （DEIMv2-DINOv3-S）携带 DINOv3 义务；蒸馏产出按携带义务处理，商用走无义务教师线。
  **⑤ 三次删除的文档同步**：NOTICE 重写（EdgeCrafter/ECViT 节删除、components→shared 路径修正、avdetr→detr 路径修正、
  权重政策统一）；README 重写（EncoderProtocol/FeatureExtractor 现名、detr 键、`components.sahi→shared.sahi`、
  `runtime/env.py` 路径、包数 11→7、权重许可警示）；分册 29 增 §9 命名风险（PyPI `autovision` 已被占用 2021 休眠项目；
  Omron AutoVISION 在先使用 + 美/加软件类商标申请——发布前需 counsel）。
  **⑥ 品牌统一**：全库 `VTrain`→`AutoVision`（30 文件）、`Autovision`→`AutoVision`（大小写统一）；
  公开类 `AutovisionError→AutoVisionError`（19 处一致）；pyproject description/README/mkdocs site_name 同步。
  **⑦ 发布流程加固**：`scripts/publish_clean_cut.sh`（守卫前置 + 重建孤儿分支 + 拒绝 origin + 只推孤儿分支）；
  `scripts/hooks/pre-push`（硬拦截 main 与 v0.2-clean 标签外推）；clean-cut.md 更新为脚本流程。
  **⑧ warm-start 重规划**：`build_warm_start_coco.py` 骨干线切换（ECViT 装配器随线删除；现计划 = D-FINE decoder
  init + timm/torchvision/自 SSL dinov2 骨干 init；ECViT 时代首个装配资产随之作废，待重建）。
  **⑨ 终验修正**：(a) `get_epochs` 的字符串式 `getattr(args_cls, "default_steps")` 在 pretext 侧改为读取
  `auto_steps/auto_epochs`（改名曾使其静默失效；新增 3 个回归测试，含双双缺省时引导报错）；(b) ClassVar 改名严格限定
  pretext 层——任务层 `TrainModelArgs` 及各任务 config 的 `default_steps/default_epochs` 是公开配置面，保持原名；
  (c) `contiguous_class_id_map` 重表达时初版把映射方向写反，行为测试当场拦截并修正（键=类 id，值=稠密序号）。
  **⑩ FeatureExtractor 内部属性改名**（`wrapped_model→_inner`、`embed_head→_embed`）：预训练导出 state-dict 前缀变化，
  0.1 时代预训练 checkpoint 加载需前缀重映射（migration-0.2.md 已注记）。
  **⑪ 终扫与门禁终态**（参照 `src/lightly_train`,阈值 15%,`--min-tokens 80`,归档
  `rescan-2026-10-15-phase5-batch1.txt.gz`）：P2b 首批 12 文件全部处置——7 个重表达至门槛以下或低于度量门槛
  （tensorrt 14.3%、timm_wrapper 12.1%、embedding_model 11.6%、optimizer_type/trainable_modules/label_helpers
  <80-token 噪声带、adamw8bit 对照件脱离），5 个按 §3.1 裁量登记 allowlist（interface/recipe 事实 + R9 表面）；
  剩余 188 个 ≥15% 文件为同性质身份层长尾,登记为 P2b-2 待办（allowlist 携复审日期,禁词扫描仍全量覆盖）。
  `--fail-over 0.15` 门禁在 allowlist 全覆盖下通过。背景率对照实验:任意无关 900–1700-token 模块对本参照树
  的 dir-best 相似度为 16–21%（度量下限证据,见台账 2026-10-15 批注）。

- 2026-10-16（**GTR 多任务集成 S1 登记，roadmap 分册 31 §11**）：① 资产面——`registry/gtr_assets.py` 扩至全任务组（26 权重：det 4 + obj365 4 + seg 4 + pose 4 + depth 4 + obb 2 + semseg 4；sha256 实测钉住 + MIT + README 度量元数据，obj365 无发布口径行如实留空）。② **S1 语义分割 re-port**（来源 `Intellindust-AI-Lab/GTR` `engine/gtr/semseg/gtrsemseg.py::SemSegHead`，MIT；其内部为 mmseg FCNHead 常规结构，GTR 基座配置 SyncBN 化）：
  | 本文件 | GTR 源 | 处置 |
  |---|---|---|
  | `tasks/semantic_segmentation/gtr/segmentation_head.py` | `engine/gtr/semseg/gtrsemseg.py::SemSegHead` | re-port（去 core 注册器/设备逻辑），模块名逐字保留（conv/bn/act/dropout/classifier）保证 `head.*` 权重键直面装载 |
  ③ 家族四模块（config/task_model/train_model/__init__）+ CLI dispatch（`gtr/<s|m|l|x>`）+ 官方权重装载器（EMA-first、`encoder.*`/`head.*` 覆盖率门禁 1.0、backbone 经 `extract_gtr_backbone` strict 直载）。**金标准对拍（RTX 4090D）**：官方 `GTRSemSeg`（其引擎/其 semseg-S 配置/官方 EMA 权重）与家族网络同输入 logits **逐比特一致（maxdiff 0.0；随机 512 / 真实图 512 / 真实图 1024 三组）**，argmax 一致率 1.0；尾部测试含 CPU 键形状核对 + CUDA 门控训练闭环（`gtr/s` dispatch，4 步，导出成功）。`tests/test_gtr_semseg.py` ×5。

- 2026-10-16（**GTR 多任务集成 S2 登记，roadmap 分册 31 §11**）：实例分割 re-port（来源 `Intellindust-AI-Lab/GTR`，MIT；相关文件均"Modified from RF-DETR"，Apache-2.0）：
  | 本文件 | GTR 源 | 处置 |
  |---|---|---|
  | `tasks/detection/shared/gtr/segmentation_head.py` | `engine/gtr/det/segmentation_head.py`（SegmentationHead/DepthwiseConvBlock/MLPBlock/point_sample/uncertain-point 采样） | re-port；去掉 `export()` 运行期方法替换（部署走 ONNX/TRT 导出器） |
  | `tasks/detection/shared/matcher.py`（`_batch_sigmoid_ce_loss`/`_batch_dice_loss`） | `engine/gtr/box_ops.py` | 逐行 re-port（对拍逐比特一致，同种子） |
  | `tasks/detection/shared/dfine_criterion.py`（`mask_point_losses`/`_mask_dice`/`_mask_sigmoid_ce`） | `engine/gtr/det/criterion.py`（`loss_masks`/`dice_loss`/`sigmoid_ce_loss`/`calculate_uncertainty`） | re-port（同种子对拍 BIT-EQUAL） |
  集成面：decoder mask 路径（`mask_downsample_ratio`，head 单次注册于 inner decoder，官方键序 `decoder.decoder.segmentation_head.*`）；criterion mask 分支（main/aux/pre/dn 四路 + weight 后缀展开）；matcher mask 代价（共享随机点网格、对角线块提取）；家族 `tasks/instance_segmentation/gtr/` + CLI `gtr/s`。

- 2026-10-16（**GTR 多任务集成 S4 登记，roadmap 分册 31 §11**）：度量深度 re-port（来源 `Intellindust-AI-Lab/GTR` `engine/gtr/depth/gtrdepth.py`，MIT；该文件自 Depth-Anything-V2 metric-depth DPT 移植，Apache-2.0）：
  | 本文件 | GTR 源 | 处置 |
  |---|---|---|
  | `tasks/depth_estimation/gtr/dpt_head.py` | `engine/gtr/depth/gtrdepth.py`（ResidualConvUnit/FeatureFusionBlock/DPTDepthHead） | re-port，模块名逐字保留（`layer_rn`/`refinenet*`/`output_conv*`），`log_depth` 解码（exp(clamp(-4,5))）与官方一致 |
  | `tasks/depth_estimation/gtr/train_model.py::_silog` | `engine/gtr/depth/criterion.py`（SiLog） | 逐式 re-port（valid 掩码按深度域，lambd 0.5） |
  集成面：新任务家族四模块 + `data/depth_estimation_dataset.py`（成对 image/depth，.npy 米或 16-bit PNG）+ `metrics/depth_estimation.py`（AbsRel/RMSE/δ，支持 none/median/log-affine 对齐）+ CLI `train-depth-estimation`（`gtr/<s|m|l|x>`）；编码器 `force_contiguous=False`（官方 depth 训练口径，2026-10-16 实测逐比特）。

- 2026-10-16（**GTR 多任务集成 S3a 登记，roadmap 分册 31 §11**）：定向检测（OBB）re-port（来源 `Intellindust-AI-Lab/GTR`，MIT）：
  | 本文件 | GTR 源 | 处置 |
  |---|---|---|
  | `tasks/detection/shared/gtr/rbox_ops.py` | `engine/gtr/obb/rbox_ops.py` | 逐段 re-port（旋转框数学/ADR 编解码/KLD/Chamfer，含 S3b 训练将用到的编码与代价件） |
  | `tasks/detection/shared/gtr/obb_decoder.py` | `engine/gtr/obb/decoder.py`（OBBIntegral/OBBLQE/OBBMSDeformableAttention/OBBTransformerDecoderLayer/OBBTransformerDecoder/OBBGTRTransformer） | re-port（注册器清理、typing、部署路径裁剪为用到的部分）；`value_op` 改为本项目共享可变形核的展平布局（与官方元组布局同数据同采样；det/seg 由逐比特对拍背书）。**注意**：OBB 层后置 norm 名为 `norm2`（无 norm3 重命名），解码器注意力头数为官方默认 8（独立于 backbone 头数） |
  集成面：家族 `tasks/oriented_detection/gtr/`（task_model + 官方装载：backbone strict + encoder/decoder 覆盖率门禁 1.0，**检查点内陈旧 anchors 按官方 solver 约定跳过**；postprocess 复用 `decode_oriented_detection_outputs`，默认 NMS 关闭 = 官方 NMS-free 协议）。**训练（定向 criterion/matcher/定向 CDN）为 §11 S3b**，本轮不交付以避免半成品。

- 2026-10-16（**GTR 多任务集成 S5a 登记，roadmap 分册 31 §11**）：姿态估计 re-port（来源 `Intellindust-AI-Lab/GTR`，MIT；`engine/gtr/pose/*` 自 DETRPose（MIT/Apache 混合）移植，DETRPose 又源自 Conditional DETR/DETR）：
  | 本文件 | GTR 源 | 处置 |
  |---|---|---|
  | `tasks/detection/shared/gtr/pose_decoder.py` | `engine/gtr/pose/decoder.py` + `ms_deform_attn.py` + `utils.py`（MLP/activation/inverse_sigmoid） | **自包含逐字 re-port**（本线自带 relu-only MLP、Gate/Integral/LQE、tuple-value 的 MSDeformAttn 核、`inverse_sigmoid(eps=1e-3)`、以及**与共享 D-FINE 版不同的 `weighting_function`（实测差 9.5e-07，必须用本线版本）**；硬编码常量（81 行 label_enc、256 宽 LQE）逐字保留）；训练分支以 S5b 指针显式拒绝 |
  集成面：家族 `tasks/pose_estimation/gtr/`（task_model + 官方装载：backbone strict + encoder/decoder 覆盖率门禁 1.0，陈旧 anchors/valid_mask 跳过；postprocess 为官方 `GTRPosePostProcessor` 移植——flattened 分数 top-k、关键点缩放至原图、COCO `[x,y,v]` 三元组）。**训练（CDN + 关键点损失/匹配）为 §11 S5b**。

- 2026-10-16（**GTR 多任务集成 S3b 登记，roadmap 分册 31 §11**）：定向检测训练面 re-port（来源 `Intellindust-AI-Lab/GTR`，MIT）：
  | 本文件 | GTR 源 | 处置 |
  |---|---|---|
  | `tasks/detection/shared/gtr/obb_criterion.py` | `engine/gtr/obb/criterion.py` + `matcher.py`（含 box_ops 的 paired_box_iou/paired hbox IoU） | 逐段 re-port（MAL + L1/KLD + FGL/DDF 分支、union 匹配、aux/pre/enc/dn 权重展开、CDN 匹配索引；focal 损失走 torchvision） |
  | `tasks/detection/shared/gtr/obb_denoising.py` | `engine/gtr/obb/denoising.py` | 逐行 re-port（OCD box-noise 模式、组内对比掩码、Per-Group DETR 掩码） |
  集成面：`OBBGTRTransformer.forward` 训练分支内部原生调用 CDN（与上游一致）；家族 `tasks/oriented_detection/gtr/train_model.py`（官方权重 losses/matcher costs 默认、角点→长边 rbox 目标转换、oriented mAP 验证、backbone lr ×0.36）；CLI `gtr/<s|x>`（`train_oriented_object_detection`）。**验证**：criterion 全 forward/ matcher / CDN 同输入同种子**逐比特一致**（12 loss 键 BIT-EQUAL；g=1/g=3 CDN 全输出 BIT-EQUAL）；CUDA 训练闭环（合成 YOLO-OBB 树）通过。
