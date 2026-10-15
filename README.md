# Autovision (VTrain)

企业级视觉模型训练框架的实现(设计文档见 [docs/framework_design.md](docs/framework_design.md) 及 [docs/design/](docs/design/) 分册)。对标 lightly-train 0.17.0 的机制,按 Apache-2.0 重新实现。

## 已实现(M0–M2 预训练主干 + M4+ 首批任务/方法扩展)

| 子系统 | 模块 | 对应设计章节 |
|---|---|---|
| 配置系统 | `config/`(PydanticConfig 严格校验 + did-you-mean 报错美化 + Registry) | §5 |
| 优化器 | `optim/`(AdamW/SGD/自研 LARS、无 weight-decay 参数规则、按架构 auto wd)、`optim/schedulers.py`(CosineWarmup) | §7.1 |
| 模型层 | `encoders/`(ModelWrapper 协议族、包注册表 + custom 兜底、EmbeddingModel;**11 个包**:torchvision、**timm(架构前缀表 → auto wd)**、**super_gradients(CustomizableDetector)**、ultralytics(AGPL 隔离)、**dinov2(transformers)、dinov3(transformers 门控; 权重用户自持 D1-a)**、rfdetr(transformers 门控)、rtdetr wrapper) | §6 |
| 数据层 | `data/`(图像/ImageFolder 分类/成对分割数据集、mmap 文件清单:引用计数/原子发布/碰撞守卫/数据指纹) | §9.1–9.3 |
| 变换 | `augment/`(ViewTransform 管线、MethodTransform 多视图协议、参数模型 + auto 解析) | §9.5 |
| 方法层 | `pretext/`(**DistillationV3** 全规格、**SimCLR**(自研 NT-Xent)、**DINO**(EMA 教师 + 中心化损失 + 多裁剪 + 动量/温度/权重衰减三调度 + 末层冻结)、**kNN 探针**(`method_args.knn_eval`)) | §7 |
| 运行时 | `engine/`(step 循环、预训练与**任务训练双 DDP**、混合精度、原子 checkpoint、resume、`on_before_optimizer_step` 钩子)、`_checkpoint.py`(自描述格式) | §10 |
| 任务层 | 检测(RT-DETRv2 + **AV-DETR SSL 骨干微调**:`avdetr/dinov2-vits14` 等 37 变体(原 LT-DETR,R16 更名),ViT 中间层 tap + STA 空间先验 + 骨干/detector 分组 LR,打通自家预训练骨干 → 检测微调闭环;D-FINE 解码器可用)、**分类(ImageFolder + top-k 指标)**、**分割(线性头 + mIoU/混淆矩阵)**;任务指标支持跨 rank 状态汇聚(DDP 验证) | §8 |
| 部署 | ONNX(FP16 黑名单 + 数值对拍门禁)、**TensorRT 导出**(batch-only profile、attention FP32 强制)、**benchmark**(torch/ONNX/TensorRT backend + 可选 parity 门禁) | §12 |
| 命令/API | `pretrain`/`train_object_detection`/`train_image_classification`/`train_semantic_segmentation`/`embed`/`benchmark`/`list_models`/`list_methods`/`load_*` | §4 |

## 快速开始

```python
import autovision

result = autovision.pretrain(
    out="runs/exp1",
    data="path/to/images",            # 目录或文件列表
    model="torchvision/resnet18",     # 学生骨干
    epochs=2,
    batch_size=4,
    method_args={"knn_eval": {"enabled": True, "k": 5}},  # 训练中 kNN 探针
)
print(result.exported_model_path, result.metrics)
```

目标检测微调:

```python
result = autovision.train_object_detection(
    out="runs/det1",
    data={  # 或 YAML 路径
        "format": "coco",
        "train": {"annotations": "train.json", "images": "images/"},
        "val": {"annotations": "val.json", "images": "images/"},
    },
    model_args={"backbone": "torchvision/resnet18", "image_size": (640, 640)},
)
print(result.best_metrics["val_metric/map"])
```

图像分类 / 语义分割微调:

```python
autovision.train_image_classification(
    out="runs/cls1",
    data={"train": "data/train/", "val": "data/val/"},  # ImageFolder 布局
    model="torchvision/resnet18",
)

autovision.train_semantic_segmentation(
    out="runs/seg1",
    data={"train": {"images": "images/", "masks": "masks/"}},
    num_classes=3,
    model="torchvision/resnet18",
)
```

导出与部署:

```python
model = autovision.load_task_model(result.exported_model_path)   # 推理
model.export_onnx("model.onnx", precision="fp16")                # FP16 + 数值对拍门禁
model.export_tensorrt("model.engine", precision="fp16")          # 需安装 tensorrt
autovision.benchmark("rtdetr/resnet18", out="bench/", num_classes=2,
                     backends={"torch": {}, "onnx": {"provider": "cpu"}},
                     verify_parity=True)
```

大图切片推理(SAHI):瓦片合并组件见
`autovision.tasks.detection.components.sahi`
(`tile_image` + `combine_object_detection_tiles`);检测类任务模型提供
`predict_sahi(image, tile_size, overlap, threshold, ...)` 便捷入口
(默认 tile 取模型输入尺寸、全图 resize 视图作全局瓦片一并推理)。

命令行(与 Python API 共享同一套 Pydantic 校验):

```bash
autovision-cli pretrain out=runs/e1 data=images/ model=torchvision/resnet18 epochs=2 \
    method_args='{"queue_size": 4, "teacher_pretrained": false}'
autovision-cli train-object-detection out=runs/d1 data=data.yaml steps=2000
autovision-cli train-image-classification out=runs/c1 data=data_dir steps=500
autovision-cli train-semantic-segmentation out=runs/s1 data=seg.yaml num_classes=3
autovision-cli benchmark out=bench/ model=rtdetr/resnet18 num_classes=2
autovision-cli list-models | list-methods | embed out=emb.pt checkpoint=...
```

## 开发

```bash
pip install -e ".[dev]"
pytest tests            # 1050+ 个测试(含金标准 loss 轨迹快照)
ruff check src tests    # lint(CI 同款规则集,配置见 pyproject.toml)
```

模型包依赖均按 extras 惰性加载,缺失时对应清单为空而不是报错:

| extras | 模型 |
|---|---|
| `transformers` | `dinov2/{vits14,vitb14,vitl14,vitg14}[-noreg|-notpretrained]`(HuggingFace);`dinov3/*`(transformers 具备 DINOv3 类时自动启用) |
| `timm` | timm 注册表全量模型(任意输入通道数) |
| `super-gradients` | CustomizableDetector 检测家族(分割 wrapper 随 M4 分割任务跟进) |
| `ultralytics`(AGPL) | YOLOv5/6/8/11/12/26、RT-DETR |
| `rfdetr` | `rf-detr-{base,nano,small,medium,large}`(transformers `RfDetr` 类,需较新 transformers;`-notpretrained` 离线变体) |

预训练权重从 HuggingFace 下载;受限网络设置镜像:`HF_ENDPOINT=https://hf-mirror.com`。`-notpretrained` 变体离线本地构建。真实 from_pretrained 的测试默认跳过,设 `AUTOVISION_HF_TESTS=1` 开启(timm 权重下载测试同门控)。

金标准快照在 `tests/golden/`;**有意的**配方变更后删除该文件重新录制(`pytest tests/test_methods_distillationv3.py` 跑两遍)。

## 环境变量

见 `src/autovision/_env.py`(`AUTOVISION_*`,遥测默认关闭)。

## 已知边界(下一里程碑)

- albumentations 适配层未接入(变换基于 torchvision v2)
- EOMT 分割家族、DINOv2/DINOv3 方法、DenseCL/DetCon、PicoDet、D-FINE FGL/DDF 为 M4+ 范围(实现蓝图见 `docs/design/12-gap-implementation-status.md`)
- 任务 DDP 已实现(torchrun;验证经指标状态汇聚 + rank0 广播),多进程自动化测试见 `tests/test_torchrun_distributed.py`(CI 设 `AUTOVISION_DDP_TESTS=1` 门控)
- ONNX FP16 parity 门禁在无 CUDA execution provider 时跳过并告警;TensorRT 导出需自行安装 `tensorrt`
- 预训练权重清单已按 commit + LFS sha256 全量钉住(`scripts/update_weight_hashes.py` 维护);`AUTOVISION_MIRROR_URL`/`AUTOVISION_OFFLINE` 消费端为待办(见分册 12 §3)
- MLflow/W&B/TensorBoard/bitsandbytes 为可选 extras,缺失时惰性降级
- 蒸馏教师默认请求预训练权重(`teacher_pretrained=True`),离线时传 `teacher_pretrained=False`
