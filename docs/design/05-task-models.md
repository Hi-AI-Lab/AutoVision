# VTrain 实现蓝图 · 分册 05：任务模型层（微调）

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §8 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 8. 任务模型层（微调）

### 8.1 双层模型契约【基线】

```python
class TaskModel(nn.Module):
    """纯 PyTorch；不 import Lightning/Fabric；可导出。基线 task_model.py:30-176"""
    def __init__(self, init_args: dict, ignore_args=None)   # 捕获 locals() 为 _init_args（剔除 load_weights）
    class_path -> str                                       # "{module}.{ClassName}"，动态重建的锚点
    def deploy(self) -> Self                                # 递归调子模块 convert_to_deploy()；is_deploy_mode
    def preprocess_image(self, image)  -> (Tensor, metadata)   # 逐图、host 侧、含 resize
    def preprocess_batch(self, batch)  -> Tensor               # batch 级、GPU、可烘进导出图（仅归一化）
    def forward_backend(self, x) -> Any                        # 原始输出（导出对象就是它）
    def postprocess(self, raw, metadata, **kw) -> list[Any]
    def predict(self, image) / predict_batch(self, images)
    def load_train_state_dict(self, state_dict) -> None        # 剥 ema_model.model. 前缀加载

class TrainModel(nn.Module):
    task: ClassVar[str];  task_model_cls: ClassVar[type[TaskModel]]
    train_model_args_cls / task_metric_args_cls / train_transform_cls / val_transform_cls / torch_compile_args_cls
    def training_step(self, fabric, batch, step) -> TaskStepResult    # 命名与 Lightning 一致以配合 Fabric 分发
    def validation_step(self, fabric, batch, step) -> TaskStepResult  # 验证用 EMA 模型
    def get_optimizer(self, total_steps, global_batch_size) -> (Optimizer, LRScheduler)
    def get_export_state_dict(self) -> dict                # 默认 state_dict()；检测器覆写为仅 ema_model.* 键
    def clip_gradients(self, fabric, optimizer) -> Tensor | None
    def on_train_batch_end(self) -> None                   # EMA 更新钩子

@dataclass
class TaskStepResult:
    loss: Tensor; metrics: TaskMetric; log_dict: dict[str, float]
    visualization: TaskStepVisualization | None            # 惰性渲染 val 可视化（_visualize/ 渲染器）
```

导出类型面（`_task_models/task_model_io.py`）：`TensorSpec(shape, dtype, is_batched)`、`ModelInputSpec(input_specs, input_dynamic_shapes)`（**只允许 batch 维动态**，校验器强制）、`BaseModelOutput` 子类自动注册为 pytree 节点以穿越 `torch.export`；检测实例 `ObjectDetectionOutput(logits, boxes)`（boxes 为归一化 cxcywh）。

### 8.2 检测家族（P0 对标）【基线】

> **修正（对照 v2.0 §8.2）**：基线**没有 YOLO 检测器模型**——"yolo" 只是数据格式；检测器家族为 **LT-DETR**（RT-DETRv2 / D-FINE 双 decoder + HybridEncoder，骨干可挂 DINOv2/v3-ViT/ConvNeXt/ECViT）与 **PicoDet**（轻量 dense 头）。

| 家族 | 损失（权重为源码常量） | 关键训练参数 |
|---|---|---|
| LT-DETR (RT-DETRv2 decoder) | 匈牙利匹配(cost = 2.0·focal_cls + 5.0·L1 + 2.0·GIoU, α0.25 γ2.0) + `loss_vfl 1.0 + loss_bbox 5.0 + loss_giou 2.0`；辅助层/denoising/encoder 输出同名进入总和 | `default_batch_size=32`、`default_steps=266_112`、lr 5e-4 × sqrt(global/32)、`backbone_lr_factor=0.05`、AdamW(wd 1e-4)、flat-cosine 调度、clip 0.1 |
| LT-DETR (D-FINE decoder) | 上表 + FGL/DDF：`loss_fgl 0.15 + loss_ddf 1.5`（角点分布 focal + 逐层解耦蒸馏） | 同上 |
| PicoDet | SimOTA(o2m, topk10) + TaskAligned(o2o) 双分配；总损失 = o2o + 0.5×dense(VFL 1.0 + GIoU 2.0 + DFL 0.25) | SGD(momentum .9, wd 4e-5)、lr 0.1 线性缩放 @80、`default_steps=90_000`、CosineWarmup |
| LT-DETR 实例分割 | EdgeCrafter 准则 = D-FINE + `loss_mask 5.0 + loss_dice 5.0`（点采样 12544 点/ratio16） | MaskAware 匹配器 |

AMP 数值健康（基线把经验固化在解码器里，VTrain 收敛为 `fp32_modules` 配置）：RT-DETRv2/D-FINE 的 bbox refinement 段落显式 `autocast(enabled=False)`（注释明言 "known overflow hotspot, especially with EMA weights"，`object_detection_components/rtdetrv2_decoder.py:366-391`）；匹配器代价矩阵 `nan_to_num(C, nan=1.0)`。

EMA：`ModelEMA(decay=0.9999, warmup_steps=2000)`；`get_export_state_dict` 只保留 `ema_model.*` 键；验证与导出均用 EMA 权重——**导出物即 EMA**，这是基线质量的关键实践，VTrain 原样吸收。

### 8.3 参数组结构【基线】

```python
lr = model_args.lr * math.sqrt(global_batch_size / default_batch_size)
backbone_lr = lr * model_args.backbone_factor
groups = [ {name:"backbone",       lr: backbone_lr},
           {name:"backbone_no_wd", lr: backbone_lr, weight_decay: 0},     # 非空才加
           {name:"detector"},                                             # 基础 lr
           {name:"detector_no_wd", weight_decay: 0} ]
```

骨干切分细节：ViT 骨干只有 `backbone.backbone_model`（预训练主体）进低 lr 组，连接层（sta/convs/norms）进 detector 组；ECViT 的 projector 全速 lr；ConvNeXt 整骨干低 lr（`ltdetr_object_detection/train_model.py:764-882`）。调度器二选一：`linear`（LinearLR start_factor=0.01 预热）或 `flat-cosine`（预热→平台→余弦→尾段，`object_detection_components/flat_cosine.py`）。

### 8.4 类别头迁移（"换 nc 加载 COCO 预训练"）【基线】

基线以 `load_state_dict` pre-hook 实现（无 `load_pretrained(nc)` 方法，VTrain 把该机制暴露为协议方法）：

- score head / denoising embedding：checkpoint 类数**多**则截断行；**少**则保留 checkpoint 行、用新初始化权重补齐（`object_detection_components/hooks.py:31-228`）。
- PicoDet GFL 分类头：形状不匹配即整头重初始化并记日志。
- 加载路径统一 `strict=False` + 剥 `ema_model.model.`（优先）或 `model.` 前缀；`reuse_class_head` 参数已弃用，语义固化为「类数匹配才复用，否则重初始化」。

### 8.5 评测与指标【基线→设计】

【基线】torchmetrics `MeanAveragePrecision`（backend `pycocotools`/`faster_coco_eval` 可选，iou 阈值默认 0.5:0.95、max_det [1,10,100]）；`watch_metric="val_metric/map"` 驱动 best 选择（`get_watch_metric_mode` 按指标名推断 min/max）；**评测前后处理无 letterbox**——纯 resize 到训练尺寸，decode = sigmoid + flatten topk(300) + 无 NMS + cxcywh→xyxy ×原图尺寸；大图走 SAHI。

【设计】VTrain `EvalProtocol`：把 resize 规则（plain/letterbox 可选）、匹配规则（topk/NMS 参数）、指标实现、协议指纹封装为一等公民，训练内 val 与 `evaluate`/`benchmark` 命令共用；指标对拍门禁复用同一协议（§12.4）。新增 letterbox 选项时以协议指纹区分，杜绝「同名不同规」。

### 8.6 任务扩展路径【设计】

任务注册表 `task → {TaskModel 子包, DataModule 子包, EvalProtocol, 指标集}`。基线印证：新增 PicoDet = 独立子包（backbone esnet.py / neck csp_pan.py / head pico_head.py / losses / assigner / postprocessor / config / transforms）+ `TASK_TRAIN_MODEL_CLASSES` 注册一行 + 共享 `_metrics`、`_transforms/ltdetr_transforms`、`object_detection_components`。VTrain 的验收用例：新增分割任务核心域层零改动。

---
