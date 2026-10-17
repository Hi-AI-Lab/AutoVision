# 已知限制与路线图

> 本页汇总对外可见的边界与工程注记；导出相关细节见[导出与部署](export.md)。

## 定位

训练 + 参考推理框架。训练、评测、导出是生产路径；跟踪与 serving 是参考实现，
不提供鉴权、TLS、多进程编排，漂移检测 / A-B / 金丝雀不在范围内（`/metrics` 是
起点，按产品需求在其上构建）。

## 许可状态与 STA 迁移

- **DEIMv2 组件已删除（2026-10-17）**：`SpatialPriorModule` 已从代码库移除，
  DINOv3 ViT 变体保留但改用 no-STA adapter；发行物不再包含 DEIMv2 代码，
  可按 Apache-2.0 发布。处置记录见[溯源台账](provenance/LEDGER.md)。
- **旧 STA-era ViT 检查点不受支持**：任何含 `backbone.sta.*` 的基线 `.pt`
  或用户自持 ViT 资产（`dinov3-vitt16` / `dinov3-vitt16plus` /
  `dinov3-vits16`）会被加载器明确拒绝，需重新训练/导出 no-STA 检查点。
  DINOv3 ConvNeXt 及用户自持的 4 个 ConvNeXt 资产不受影响。
- **受限权重仍然存在**：DINOv2 权重 CC-BY-NC 4.0、DINOv3 权重 DINOv3
  License、用户自持检测器禁再分发/商用；这些是权重条款，与已删除的 STA
  代码无关。DEIMv2 外部权重仍为非商用。

## 适配缺口

- albumentations 适配层未接入（变换基于 torchvision v2）。
- super-gradients 分割 wrapper 未接入（CustomizableDetector 检测家族可用）。
- 难例加权采样（`hard_examples=`）与长尾采样（`long_tail_sampler=`）仅支持
  单机；DDP 的 DistributedSampler 会覆盖采样权重，组合被显式拒绝。
- Label Studio 的 `keypointlabels` 以 `keypoint_radius` 小框转换为检测框；姿态
  训练应使用原生关键点格式（COCO keypoints / YOLO-pose）。
- Label Studio 的旋转矩形（`rotation != 0`）按轴对齐框导入，旋转角保留在
  标注字段；旋转框训练仍使用 YOLO-OBB / DOTA 格式。
- 多路流运行时 v1 不做跨路跟踪 / ReID；tracking 仅单路 `run_stream`。
  每路渲染 writer 在该路分辨率变化时重建。

## 检测谱系状态

`n` / `s` 为 spec-frozen（规格冻结：组件与配方已承诺，预训练权重与 COCO AP
待发布）；`m` / `l` / `x` 骨干线方向已定（timm ConvNeXt，tiny→m / small→l /
base→x，选型标准为可导出 + 许可干净），实现待落地。`detr/gtrvit-s` 为 planned。只有 `released` 档带预训练权重：发布门禁要求权重 +
模型卡（许可 / sha256 / seed）+ 全量 AP + 可复现延迟。机器可读状态与变体清单见
[模型库](model_zoo.md)。

## 导出与部署

- GTR 模型（含 `detr/gtrvit-*`）不可导出 ONNX/TensorRT：导出入口以明确报错拒绝
  （GLA 自定义 CUDA 核无 ONNX 降级，见 `GTRViTBackbone.export_blocker`）；四个 GTR
  任务模型的导出 verify 钩子已就位，随部署工程落地生效。
- ONNX FP16 parity 门禁在无 CUDA execution provider 时跳过并告警；ONNX 导出的示例
  输入跟随模型设备（GPU 上训练后可直接导出）。
- 动态 H/W 支持面、int8 首层 fp32 配方、TensorRT 11 适配见[导出与部署](export.md)。
- int8 全量 AP 报告（`scripts/measure_int8_accuracy.py`）需要兼容的
  ONNX Runtime / TensorRT 环境；`export` extra 已要求 `numpy>=1.25`
  （onnx/onnxscript 的导入前提），重装该 extra 后即可复测。当前开发 venv
  仍是旧 numpy，所以本地 ONNX 集成测试按环境门控跳过。
- `examples/cpp/` 是参考运行时，不是 ABI/SDK；完整构建需要 ONNX Runtime 与
  OpenCV 开发库，host-only 后处理测试只需 g++。
- GTR 部署工程：Triton GLA 核已迁移（2× 提速），但 `torch.compile` 对 GLA 段负收益、
  CUDA Graphs 与自研核原地写接口冲突（需 op 改造）、TensorRT 部署需官方
  `gla_plugin.cu`。
- 官方训练增广（mosaic / mixup / zoom-out / color-jitter）未移植；姿态官方配方为
  `precision="32-true"`（`use_amp: False`）。

## 离线与权重

- 蒸馏教师默认请求预训练权重（`teacher_pretrained=True`）；离线时传
  `teacher_pretrained=False`。
- 预训练权重清单按 commit + LFS sha256 全量钉住
  （`src/autovision/encoders/weight_hashes.py`）。
- 受限许可权重（DINOv2 CC-BY-NC 4.0 / DINOv3 License）默认不下载，需显式自证；
  用户自持库禁再分发 / 商用，见[用户自持权重](user_held_weights.md)。
