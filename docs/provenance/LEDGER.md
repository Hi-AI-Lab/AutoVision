# 溯源台账（Provenance ledger）

> 状态：人工维护的模块/文件级台账，覆盖仓库内 vendored、ported 与
> architecture-reference 代码。它不是法律意见；商业发布前需法务复核。
> 完整非 Apache 许可文本见仓库根 `THIRD_PARTY_LICENSES.md`；许可边界汇总见
> `NOTICE`；权重许可见[模型库](../model_zoo.md)。

## 1. 代码 lineage

| 组件 / 文件 | 上游 | 许可 | 方式 | 归属位置 |
|---|---|---|---|---|
| `tasks/shared/flat_cosine.py` | [Intellindust DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 | 重实现（step-based 接口） | 文件头、NOTICE |
| `tasks/detection/shared/dfine_decoder.py`、`dfine_official_decoder.py`、`dfine_criterion.py`、`dfine_utils.py` | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 | ported | NOTICE、LICENSE |
| `tasks/detection/shared/rtdetrv2_decoder.py`、`hybrid_encoder.py`、`rtdetrv2_criterion.py`、`matcher.py`、`denoising.py`、`tasks/shared/ema.py` | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 | ported | NOTICE、LICENSE |
| `tasks/detection/shared/_msda_src/`、`msda.py`、`ms_deformable_attention.py` | Detectron2 / Deformable DETR lineage | Apache-2.0 | vendored | 文件头、NOTICE、LICENSE |
| `tasks/detection/shared/gtr/*`、`tasks/detection/shared/gtr/csrc/*` | [Intellindust-AI-Lab/GTR](https://github.com/Intellindust-AI-Lab/GTR) | MIT | ported / vendored | 文件头、NOTICE、THIRD_PARTY_LICENSES |
| `tasks/detection/shared/gtr/gla_onnx.py`（GLA 纯 ONNX 降级） | [Intellindust-AI-Lab/GTR](https://github.com/Intellindust-AI-Lab/GTR) `tensorrt_plugin` 导出补丁；chunk-GLA 递推遵循 [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | MIT | ported（导出期 swap，eager 不变） | 文件头、NOTICE |
| `tasks/detection/shared/matcher.py` 的 LSAP 路径 | [torch-linear-assignment](https://github.com/ivan-chai/torch-linear-assignment) | Apache-2.0 | 可选依赖（未 vendored） | NOTICE、pyproject extra |
| `tasks/detection/picodet/` | [PaddleDetection PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection) | Apache-2.0 | 架构参考；AutoVision 实现 | README 致谢、NOTICE |
| `tracking/byte_track.py` | [ByteTrack](https://github.com/ifzhang/ByteTrack)（ECCV 2022） | MIT（上游） | 算法参考；AutoVision 实现 | 文件头、README 致谢 |
| `tasks/shared/sahi.py`、`tasks/shared/tiling_utils.py` | [SAHI](https://github.com/obss/sahi) | MIT（上游） | 算法参考；AutoVision 实现 | README 致谢 |
| `visualize/effects.py`、`visualize/skeletons.py`、`geometry/boxes.py`、`geometry/masks.py`、`inference/video.py` | [Roboflow supervision 0.30](https://github.com/roboflow/supervision) | MIT | API/算法参考；AutoVision 实现（Pillow/NumPy 重写，不引入运行时依赖） | 文件头、NOTICE、THIRD_PARTY_LICENSES |
| `encoders/rfdetr/`、`tasks/detection/rfdetr/` | [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) / HF transformers | Apache-2.0 | wrapper；权重按需下载 | 模型卡、模型库 |

| `export/artifact_metadata.py`、`export/timing.py`、`export/export_onnx.py`（TRT 身份旁路、产物最小尺寸护栏）、`registry/deployment_gates.py`（MetricFloorGate） | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（D1/D5/D14b） | 设计提案 §4.1/§4.5/§4.14、NOTICE |
| `tracking/gmc.py`、`tracking/bot_sort.py`、`tracking/byte_track.py`（钩子化重构）、`tracking/analytics.py`（chord 穿越补偿）、`inference/stream.py`（frame/补偿接线） | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics)（GMC/钩子/退役通知）；BoT-SORT（arXiv 2206.14651）阶段结构 | AGPL-3.0 / 论文 | 架构参考；AutoVision 净室实现（D2/D6/D7/D13a） | 设计提案 §4.2/§4.6/§4.7/§4.13、NOTICE |
| `augment/detection/box_candidates.py`（含 mosaic 接线与 args 阈值） | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) `box_candidates` 语义 | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（D3） | 设计提案 §4.3、NOTICE |
| `engine/autobatch.py` | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) autobatch 策略（线性拟合 + OOM 回退） | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（D4；step 闭包按任务面注入） | 设计提案 §4.4、NOTICE |
| `data/scan_cache.py`、`data/image_cache.py`、`data/downloads.py`（断点续传/重试/磁盘预检）、`data/split_large_images.py`（训练侧切片 + CLI `tile-dataset`） | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) 缓存/续传/split_dota 设计 | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（D8/D9/D10/D11；sha256 校验为 AutoVision 既有更强机制） | 设计提案 §4.8–§4.11、NOTICE |
| `serving/app.py`（lifespan 全管线预热）、`docs/framework_design.md` §3.1（引擎/流状态生命周期约定） | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) warmup/预测器复用语义 | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（D14a/D14d） | 设计提案 §4.14、NOTICE |
| `cli/model_handle.py`（`_ENGINE_SETUP_KEYS` 引擎缓存二分）、`engine/amp_check.py`（三态 AMP 事前探针）、`runtime/autodevice.py`（`-1` 空闲选卡 + CVD 翻译）、`engine/engine.py`（resume 白名单 `check_resume_fit_args` / 早停状态持久化 / possible_stop）、`geometry/polygons.py`（`is_valid_polygon` / `polygon_from_bbox`）、`data/coco_instance_segmentation_dataset.py`（多边形白名单兜底 + compressed-RLE 修复） | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) `PREDICTOR_SETUP_KEYS`/`check_amp`/`autodevice`/`check_resume`/多边形白名单设计 | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（D15/D16/D17/D18/D19） | 设计提案 §4.15–§4.19、NOTICE |
| `engine/engine.py`（save_period 单次序列化多路写 / fp16 克隆钳制 / OOM 降批 / 回调注册接线）、`runtime/checkpoint.py`（`serialize`/`save_serialized`）、`engine/callbacks.py`、`runtime/model_info.py`（thop 惰性）、`runtime/fuse.py`（Sequential Conv+BN 折叠 + parity 门禁）、`augment/core.py`（四目标钩子协议）、`metrics/containers.py`（惰性 stats + 空/空约定）、`data/quality.py`（存储体检）、`metrics/map_evaluator.py`（maxDets 自动抬升）、`cli/common_helpers.py`/`cli/train_helpers.py`（worker 播种） | [Ultralytics 8.4.177](https://github.com/ultralytics/ultralytics) save_model/BytesIO、OOM 降批、回调清单、model_info、fuse、BaseTransform、DetMetrics、check_file_speeds、max_det 抬升设计 | AGPL-3.0（上游） | 架构参考；AutoVision 净室实现（D20/D21/D22/D23/D24/D27/D29；OOM 降批限定单进程、回调隔离严于上游、fuse 只走 Sequential 模式） | 设计提案 §4.20–§4.29、NOTICE |

**净室声明（Ultralytics AGPL 架构参考，2026-11-03）**：上述 D 系列实现依据
`docs/design_ultralytics_adoption.md` 与上游论文级语义（BoT-SORT、ByteTrack、
Lucas–Kanade/ECC 等公开算法）在本仓库接口风格内重写；实现者未对照上游源码
逐行翻译，未复制上游注释、docstring、常量表或测试数据。D15–D19 落地时按
同一纪律执行：探针不下载外部权重（改用调用方自有模型）、选卡算法以本仓库
device 语法重设计（`-1` 占位符 + 降级填满而非删占位符）、resume 白名单语义
反转为"当前调用参数治理 + 漂移告警"。D20–D29 同一纪律：OOM 降批按集合
通信安全收敛为单进程、回调隔离逐 handler（严于上游的裸循环）、fuse 数学
重写并加本仓库 parity 门禁、bf16/fp16 探针 dtype 错配备忘在案。`NOTICE`
的 AGPL 段落（2026-11-03 二次改写，协议式三条款）保留 "must never have
source or weights copied" 红线原文。商业分发前需按台账头部要求法务复核
本节。

## 2. 已处置：DEIMv2（STA 删除）

2026-10-17 的许可审计发现：`ViTDetectionBackbone` 曾包含
`SpatialPriorModule`，与 DEIMv2 上游 `SpatialPriorModulev2` 在图层顺序、
通道宽度与 state-dict 键上逐项对应。上游 DEIMv2 License 只允许非商用使用、
修改与再分发，商用需要单独许可；这会使组合发行物不能再按 Apache-2.0 商用。

处置结果（用户决策：删 STA，保留 DINOv3 ViT，保留 ConvNeXt）：

- `SpatialPriorModule` 已从代码库删除，`DETRVariantSpec` 不再携带
  `use_sta` / `conv_inplane` 字段，`ViTDetectionBackbone` 对 DINOv2 与
  DINOv3 ViT 统一使用 no-STA adapter；
- DINOv3 ViT 变体（`dinov3-vits16` / `vitb16` / `vitl16` / `vitt16` /
  `vitt16plus` 及别名）保留，架构按 no-STA 重新定义；
- DINOv3 ConvNeXt 与 timm / GTR 家族不受影响；
- 代码库不再包含 DEIMv2 代码，`THIRD_PARTY_LICENSES.md` 不再附 DEIMv2
  License 全文，仅保留 GTR MIT 全文；DEIM 的 flat-cosine 参考仍为
  Apache-2.0；
- 兼容性：任何含 `backbone.sta.*` 的旧 ViT 检查点（基线 `.pt` 与用户自持
  3 个 ViT 资产）不能忠实加载，加载器会显式报错，需重新训练/导出；用户自持
  的 4 个 ConvNeXt 资产不受影响。

## 3. 权重与数据

- GTR 权重：MIT；再分发时附 MIT 通知（见 `THIRD_PARTY_LICENSES.md`）。
- RF-DETR 权重：Apache-2.0；从上游下载，不由本仓库重分发。
- DINOv2 权重：CC-BY-NC 4.0（非商用）；默认不下载，需显式自证。
- DINOv3 权重 / 基于 DINOv3 的教师：DINOv3 License；商用需遵守
  “Built with DINOv3”等义务；默认不下载。
- DEIMv2 转换权重（如 `DEIMv2_*`）：上游 DEIMv2 权重受 DEIMv2 License
  约束；HF 转换卡的许可需逐一复核，不能仅凭转换仓库声明推定可商用。
  AutoVision 代码不再依赖 DEIMv2，但外部权重条款不因此改变。
- 用户自持 7 个检测器：`DINOv3-License+NOASSERTION`，禁再分发/商用。
  其中 `dinov3-vitt16` / `dinov3-vitt16plus` / `dinov3-vits16` 为 STA-era，
  当前加载器拒绝；`dinov3-convnext-*` 4 个仍可加载。
- 数据集（COCO / VOC / Cityscapes / DOTA / NYU / Objects365 / ImageNet）：
  不随仓库分发；使用与再分发遵循各自条款。

## 4. 待办与复核

| 项 | 状态 | 处置 |
|---|---|---|
| DEIMv2 `SpatialPriorModule` | **DONE（已删除）** | 删除 STA；ViT 改用 no-STA adapter；旧检查点显式拒绝 |
| DINOv3 ViT no-STA 重新训练/导出 | OPEN | 旧 STA-era 基线/用户自持 ViT 检查点需重训后才能使用 |
| DEIMv2 转换权重许可 | OPEN | 外部权重；逐个核对 HF 模型卡与上游条款 |
| 上游 Apache NOTICE 文件 | OPEN | 法务复核 D-FINE / RT-DETR / DEIM / Detectron2 是否有 NOTICE 需随分发保留 |
| RF-DETR / PicoDet / ByteTrack / SAHI 致谢 | DONE | README `致谢` + NOTICE |
| GTR MIT 全文 | DONE | `THIRD_PARTY_LICENSES.md` |
| 缺失的旧设计/台账引用 | DONE | NOTICE 改为指向本文件，不再引用已删除的 `docs/design/*` |

## 5. 维护规则

新增 vendored / ported / architecture-reference 代码时：

1. 在源码文件头写明上游、许可、修改点；
2. 在本台账新增一行（组件、上游、许可、方式、归属位置）；
3. 非 Apache/MIT 条款在 `THIRD_PARTY_LICENSES.md` 附全文；
4. 更新 `NOTICE` 与模型卡；
5. 运行 `python scripts/check_provenance.py`、`scripts/check_docs_links.py`、
   `mkdocs build --strict`。

> 本台账基于当前源码与上游公开许可整理；如与上游最新条款冲突，以上游为准，
> 并立即停止商业发布。
