# AutoVision

企业级视觉模型训练框架:自监督预训练 → 蒸馏 → 微调 → 评测 → 导出。
同一套 Pydantic 严格校验驱动 Python API 与 `autovision-cli`,覆盖检测、旋转框、
实例/全景/语义分割、姿态、深度、分类等任务面。

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

## 安装

Python ≥ 3.10。模型包按 extras 惰性加载,缺失时对应清单为空而不是报错:

```bash
pip install -e .                                # 核心框架(不含第三方模型包)
pip install -e ".[transformers,timm,export]"    # 常用:HF 骨干 + timm + ONNX 导出
pip install -e ".[dev]"                         # 开发(pytest / ruff,CI 同款)
```

| extras | 能力 |
|---|---|
| `transformers` | `dinov2/*`、`dinov3/*` 骨干(HuggingFace 权重) |
| `timm` | timm 注册表全量模型(任意输入通道数) |
| `super-gradients` | CustomizableDetector 检测家族(分割 wrapper 未接入) |
| `ultralytics`(AGPL) | YOLOv5/6/8/11/12/26、RT-DETR(隔离加载) |
| `rfdetr` | `rf-detr-{base,nano,small,medium,large}`(需较新 transformers) |
| `gtr` | GTR GLA 骨干(`detr/gtrvit-s`);Triton 前向需 CUDA |
| `export` | ONNX 导出管线(onnx / onnxruntime / onnxslim) |
| `video` | 参考推理运行时的视频/流 I/O(OpenCV,文件 + RTSP/HTTP) |
| `openvino` | OpenVINO IR 导出 + 运行时后端(CPU 优先,可切 GPU/AUTO) |
| `serving` | 参考推理 serving(FastAPI + uvicorn;`/predict` 批处理 + `/metrics`) |
| `lsap` | 检测匹配器批量线性分配求解(CUDA;自动回退 scipy) |
| `tensorboard` / `mlflow` / `wandb` | 训练记录器,缺失时惰性降级 |
| `bitsandbytes` | 8-bit AdamW |

预训练权重从 HuggingFace 下载;受限网络设镜像 `HF_ENDPOINT=https://hf-mirror.com`
(框架侧开关:`AUTOVISION_MIRROR_URL` / `AUTOVISION_OFFLINE`)。`-notpretrained`
变体离线本地构建。**权重许可**:dinov2 权重 CC-BY-NC 4.0、dinov3 权重 DINOv3
License,默认均不下载——需分别设 `AUTOVISION_DINOV2_WEIGHTS=1` /
`AUTOVISION_DINOV3_WEIGHTS=1` 自证已接受对应条款;不受限的干净路径是
`-notpretrained` + 本框架自监督预训练(政策见 [NOTICE](NOTICE))。

## 快速开始

目标检测,训练 → 验证 → 预测 → 导出:

```python
import autovision

data = autovision.ensure_coco8()   # 合成冒烟数据集;换 COCO/YOLO 配置、YAML 路径或字面量 "coco8"
model = autovision.detection("rtdetr/resnet18").train(
    out="runs/det1", data=data, steps=2000, batch_size="auto",
)
model.val(data=data)          # {'map': ..., 'map_50': ...}
model.predict("image.jpg")    # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")    # 按 .onnx / .engine / .pt 后缀分发
```

等价命令行(同一套 Pydantic 校验):

```bash
autovision-cli train-object-detection out=runs/det1 data=data.yaml steps=2000
autovision-cli val     model=runs/det1/exported_models/exported_best.pt data=coco8
autovision-cli predict model=model.pt images=dir/ out=preds/ threshold=0.5
```

<details>
<summary>更多用法:自监督预训练 / 分类 / 分割 / 函数式 API / 大图切片 / 全部命令</summary>

自监督预训练与蒸馏:

```python
result = autovision.pretrain(
    out="runs/exp1",
    data="path/to/images",            # 目录或文件列表
    model="torchvision/resnet18",     # 学生骨干
    epochs=2,
    batch_size=4,
    method_args={"knn_eval": {"enabled": True, "k": 5}},   # 训练中 kNN 探针
)
print(result.exported_model_path, result.metrics)
```

分类 / 语义分割微调:

```python
autovision.train_image_classification(
    out="runs/cls1",
    data={"train": "data/train/", "val": "data/val/"},
    model="torchvision/resnet18",
)
autovision.train_semantic_segmentation(
    out="runs/seg1",
    data={"train": {"images": "images/", "masks": "masks/"}},
    num_classes=3,
    model="torchvision/resnet18",
)
```

函数式 API 与检测 handle 同源(`.train()` 是它的薄包装),另有
`train_oriented_object_detection` / `train_instance_segmentation` /
`train_panoptic_segmentation` / `train_pose_estimation` /
`train_depth_estimation`,均带同名 CLI 命令。
GTR 权重直载示例:`model_args={"checkpoint": "gtrpose_s_coco"}`。

大图切片推理(SAHI):`autovision.tasks.shared.sahi` 的 `tile_image` +
`combine_object_detection_tiles`;检测模型另有
`predict_sahi(image, tile_size, overlap, threshold, ...)`(默认 tile 取模型
输入尺寸,全图 resize 视图作全局瓦片一并推理)。

命令行全表:`pretrain`、`train-object-detection`、
`train-oriented-object-detection`、`train-image-classification`、
`train-instance-segmentation`、`train-panoptic-segmentation`、
`train-semantic-segmentation`、`train-pose-estimation`、
`train-depth-estimation`、`val`、`predict`、`export`、`benchmark`、`embed`、
`download`、`convert-data`、`plot-curves`(需 `pip install matplotlib`)、
`extract-video-frames`、`list-models`、`list-methods`、
`pseudo-label`、`analyze-errors`、`hpo`、`serve`、`data-snapshot`。

</details>

## 能力总览

| 子系统 | 能力 |
|---|---|
| 配置 / 注册 | Pydantic 严格校验 + did-you-mean 报错 + Registry;CLI 与 API 同一套校验 |
| 模型来源 | 7 个注册源:`torchvision`、`timm`、`super_gradients`、`ultralytics`(AGPL 隔离)、`dinov2`、`dinov3`、`rfdetr`;`custom` 兜底;`arch_traits()` → `weight_decay="auto"` 按架构解析;树内 `rtdetr/*` 无第三方包 |
| 数据 / 变换 | 图像、ImageFolder 分类、成对分割、COCO 关键点、深度图;mmap 文件清单(引用计数 / 原子发布 / 碰撞守卫 / 数据指纹);ViewTransform 管线 + MethodTransform 多视图协议;VOC / YOLO-pose / YOLO-seg 转换与 COCO 合并 |
| 预训练方法 | DistillationV3(全规格)、SimCLR(NT-Xent)、DINO(EMA 教师 + 中心化损失 + 多裁剪 + 动量/温度/权重衰减三调度)、DINOv2 / DINOv3、DenseCL / DetCon B/S、训练中 kNN 探针 |
| 运行时 | step 循环、预训练与任务训练双 DDP、混合精度、原子 checkpoint + resume、`on_before_optimizer_step` 钩子、自描述 checkpoint 格式 |
| 任务层 | 检测:`rtdetr/*`、`detr/*`、`dfine/*`、`picodet/*`、`rfdetr/*`;旋转框:`rtdetr-obb/*`、`gtr/*`;分割:实例 / 全景 / 语义;姿态、深度、分类;任务指标支持跨 rank 状态汇聚(DDP 验证);误差分析(FP 归因 + 面积/类别切片 + top-FP/FN 画廊)与伪标注/自训练闭环 |
| 部署 | ONNX(FP16 黑名单 + 数值对拍门禁)、TensorRT(batch-only profile、attention FP32)、benchmark(torch / ONNX / TensorRT + parity 门禁，报告携带参数量/FLOPs)、模型成本(`measure_model_cost`)、`export` CLI(导出件后缀分发)、参考推理运行时(视频/RTSP + batch,torch/ONNX/TRT 同源前后处理) |

## 模型与指标

### 检测谱系 `detr/*`(39 个已注册变体)

| 骨干族 | 变体 | 装配 |
|---|---|---|
| DINOv2 ViT(patch 14) | `vits14` / `vitb14` / `vitl14` / `vitg14`,各带 `-noreg` 与 `-notpretrained`;`vittest-notpretrained` 为离线测试变体 | 13 个;NoSTA——语义 tap 直投 `hidden_dim` |
| DINOv3 ViT(patch 16) | `vits16` / `vitb16` / `vitl16` / `vitt16` / `vitt16plus`,`-eupe` 架构别名见 `vits16` / `vitb16` / `vitt16`,各档带 `-notpretrained` | 13 个;STA,`conv_inplane = factor × patch_size` |
| DINOv3 ConvNeXt | `tiny` / `small` / `base` / `large`,`-eupe` 架构别名见 `tiny` / `small` / `base` | 7 个;末三段 stage tap,整骨干走 backbone LR |
| timm HGNetV2 + GTR 解码器装配 | `hgnetv2-b0` / `hgnetv2-b1`,冻结档位 `n` / `s`,`gtr-s` | 5 个;`n` 走蒸馏路线,`s` 走装配微调 |
| GTR GLA 骨干 | `gtrvit-s`(`gtr_tiny_spatial_swiglu` + `GTREncoder`,需 `gtr` extra) | 1 个 |

档位谱系方向冻结于 2026-10-15:`n` / `s` 已冻结为公开档位,`m` / `l` / `x` 骨干线
待定,`gtr-s` 待装配覆盖与趋势评审。编码器 / 解码器配方(隐藏维、FFN 宽、采样点、
层数)逐变体携带,不共享单一全局默认。

### 用户自持权重库(D1-a:仓库外,不可再分发)

7 个第三方训练的 COCO 检测器(DINOv3 骨干 + 第三方检测头)经合规重打包——张量
逐位不变,容器与元数据换为我方格式并携带一等许可块,逐位忠实性对拍(logits /
boxes 逐位相等)——存于**仓库之外**的用户自持库(默认
`~/Research/Model/AutoVision`,`AUTOVISION_USER_HELD_ROOT` 覆盖),每个资产为
`weights.pt` + `warm_start.pt` + `ASSET.json` 三件套:

| 资产名 | 加载 model 键 | 构建覆盖(加载必需) |
|---|---|---|
| `dinov3-vitt16` / `dinov3-vitt16plus` / `dinov3-vits16` | `detr/dinov3-{vitt16,vitt16plus,vits16}-notpretrained` | 无(变体规格配方) |
| `dinov3-convnext-tiny` / `-small` / `-base` / `-large` | `detr/dinov3-convnext-{tiny,small,base,large}` | `dec_layers=6` + `decoder_hidden_dim=256` |

ConvNeXt 检查点用 6 层 / 256 宽解码器训练,**不是**变体规格默认的 3/4 层 × 384
宽——不带覆盖会被严格加载拒绝;`user_held_model_args(<资产名>)` 一次取全所需
`model_args`(含 `backbone_pretrained=False` 与 `image_size`)。资产名与两份容器的
sha256 在 `autovision.registry.user_held_assets` 钉住(`resolve_user_held_asset`
解析、`verify_user_held_asset` 离线核验)。

两条加载路线:

```python
import autovision
from autovision.registry.user_held_assets import user_held_model_args

# 路线 A:检查点热启(名字解析 + ConvNeXt 必需的 dec_layers / 宽度覆盖)
autovision.train_object_detection(
    out="runs/d1", data="data.yaml", model="detr/dinov3-convnext-base",
    model_args=user_held_model_args("dinov3-convnext-base"),
)

# 路线 B:AVWarmStart v1(我方键布局,形状兼容子集迁移)
autovision.train_object_detection(
    out="runs/d2", data="data.yaml", model="detr/dinov3-vitt16-notpretrained",
    model_args={"warm_start": "dinov3-vitt16"},   # 或 warm_start.pt 的路径
)
```

**许可边界**(`DINOv3-License+NOASSERTION`):骨干继承 DINOv3 License 义务,检测头
无许可声明(默认保留全部权利)。**禁再分发、禁商用**;不随仓库或导出产物分发,
不注册可下载资产清单(`ASSET_MANIFEST`)。使用即视为已自行接受相应条款。

### GTR 多任务家族

Intellindust GTR(MIT)整族 26 个官方权重已在 `autovision.registry.gtr_assets`
注册(sha256 全钉住,`resolve_gtr_asset` 解析、`verify_gtr_asset` 离线核验),
文件存于用户自持库(`AUTOVISION_GTR_ROOT` 覆盖,默认 `~/Research/Model/gtr`)。
整族共享一套 GLA 骨干(`ViTAdapterSpatialSwiGLU`)+ `GTREncoder`,各任务只换
头 / 解码器。

| 任务 | model 键 | 变体 | 官方指标 | 权重资产 |
|---|---|---|---|---|
| 检测 COCO | `detr/gtr-s`(GTR 解码器装配,骨干走 timm HGNetV2-b1)、`detr/gtrvit-s`(官方 GLA 骨干 + GTREncoder,需 `gtr` extra) | s(m/l/x 权重已入库,model 变体仅 S 接入) | AP 53.6 | `gtr_{s,m,l,x}_coco` |
| 检测 Obj365 预训练 | —(作检测初始化权重) | s/m/l/x | — | `gtr_{s,m,l,x}_obj365` |
| 实例分割 COCO | `gtr/<s\|m\|l\|x>` | s/m/l/x | mask AP 45.0 / 47.7 / 49.5 / 49.8 | `gtrseg_{s,m,l,x}_coco` |
| 语义分割 Cityscapes | `gtr/<s\|m\|l\|x>` | s/m/l/x | mIoU 81.5 / 83.0 / 83.2 / 83.3 | `gtrsemseg_{s,m,l,x}_cityscapes` |
| 深度 NYU | `gtr/<s\|m\|l\|x>` | s/m/l/x | absRel 0.074 / 0.069 / 0.069 / 0.067 | `gtrdepth_{s,m,l,x}` |
| 旋转框 DOTA | `gtr/<s\|x>` | s/x | AP50 80.0 / 81.3 | `gtrobb_{s,x}_dota` |
| 姿态 COCO | `gtr/<s\|m\|l\|x>` | s/m/l/x | kp AP 70.1 / 74.1 / 74.7 / 75.5 | `gtrpose_{s,m,l,x}_coco` |

`gtr/*` 任务族用 `model_args={"checkpoint": "<注册资产名或路径>"}` 直载官方权重
后微调,不传即随机初始化。六个任务组的权重加载与前向数值对拍已打通(检测解码器
谱系金标准见 `tests/test_gtr_integration.py`;姿态全链读数与 delta 见
[docs/benchmarks.md](docs/benchmarks.md))。`detr/gtrvit-s` 的 GLA 骨干前向依赖
Triton(CUDA,`gtr` extra)。

## 目标跟踪与推理服务（A3 / F2）

参考推理运行时支持 **ByteTrack** 跟踪（卡尔曼运动模型 + 三阶段 IoU 关联，
class-aware，官方门限：新轨迹需 `score ≥ track_thresh + 0.1` 且连续两帧命中
才发布（首帧例外）、低分框 `>0.1` 只维持轨迹不新建——官方语义）:

```bash
autovision-cli predict model=exported_best.pt source=clip.mp4 out=preds/ track=true
```

JSONL 逐帧增加 `tracks`(`track_id`/`bbox`/`score`/`class_id`/`age`),渲染画面标注
`#id 类别 分数`,汇总带唯一 id 数;API 侧 `run_stream(..., track=True)`。
`age` 是轨迹创建以来的帧数;tracker 接收源视频的真实帧率,`track_buffer`
按该帧率折算丢失窗口。

参考 **serving**(roadmap F2;需要 `.[serving]`,属参考实现而非生产网关):

```bash
autovision-cli serve model=exported_best.pt port=8000
```

- `POST /predict`:`{"images": ["<base64>", ...]}`(或单张 `{"image": ...}`),
  批处理按输入顺序返回,前后处理与评测同源;
- `GET /healthz`:后端/模型/threshold;
- `GET /metrics`:Prometheus 文本(请求数/错误数/图片数/延迟 summary)。

**边界**(定位仍是训练+参考推理框架):不提供鉴权、TLS、多进程编排;漂移检测 /
A-B / 金丝雀不在范围内——`/metrics` 是它们的起点,按产品需求在其上构建。

## 数据版本化

mmap 清单的缓存键是「路径 + mtime」;做**内容寻址快照**用 `data-snapshot`
(每分片记录根路径、文件数/字节数与 `(相对路径, 大小)` 清单的 sha256;`deep=true`
再对文件内容哈希):

```bash
autovision-cli data-snapshot data=coco8 out=coco8.snap.json deep=true   # 写入
autovision-cli data-snapshot snapshot=coco8.snap.json                   # 校验(漂移非零退出)
autovision-cli data-snapshot snapshot=coco8.snap.json spec=copied.yaml  # 迁移/副本校验
```

漂移报告给出逐项差异(added / removed / resized / content_digest)与涉及文件;
摘要是「根相对」的,目录整体搬迁后可用副本 spec 校验通过。`spec` 内的相对路径按
spec 文件所在目录解析(与训练入口同一规则;快照记录该基准目录,换 CWD 校验不会
漂移);`out` 已存在时报错,需显式 `overwrite=true`;`snapshot=... deep=true` 只对
deep 快照有效——浅快照会明确报错,避免「内容校验」空转。

## 超参搜索（HPO）

随机 / 网格搜索与 ASHA（逐 rung 续训，非重训），复用与 CLI 同一套 Pydantic
覆盖面（`model_args.*` / `transform_args.*` / `trainer_args.*` 与顶层参数），
越界或拼错的键由配置校验直接报错：

```yaml
# search.yaml（点号键 → 分布；裸列表 = choice）
model_args.lr: {loguniform: [1.0e-5, 2.0e-3]}
model_args.weight_decay: {choice: [0.0, 1.0e-4, 5.0e-4]}
model_args.ema_warmup_steps: {choice: [100, 500, 2000]}
batch_size: [2, 4]
```

```bash
autovision-cli hpo space=search.yaml out=runs/hpo task=object_detection \
    data=coco8 model=rtdetr/resnet18 trials=6 budget=300 \
    objective=val_metric/map_50 strategy=random seed=0
autovision-cli hpo space=search.yaml out=runs/hpo-asha task=object_detection \
    data=coco8 model=rtdetr/resnet18 strategy=asha trials=9 budget=200 asha_max_budget=1800
```

产出 `out/report.md`（空间摘要 + 按目标排序的试验表 + 最优配置）、
`out/trials.jsonl`（逐试验机器可读记录）、`out/best.json`；已完成的试验目录
命中 `metrics.json` 即复用（中断后可续跑）。API：`autovision.hpo.run_search` +
`task_trial`（任意 `autovision.train_*` 入口）。Ray Tune 用户可用同一 trial 函数
接 `ASHAScheduler`（本实现零依赖）。

## 数据闭环（伪标注 / 误差分析）

VOC 数据可直接训练（下载 → 转换 → 训练三步；官方源较慢时自备数据集同流程）:

```bash
autovision-cli download dataset=voc2007-trainval out=datasets/       # 解包出 VOCdevkit/VOC2007
autovision-cli convert-data from=voc voc-root=datasets/VOCdevkit/VOC2007 out=voc_train.json imagesets=train
autovision-cli convert-data from=voc voc-root=datasets/VOCdevkit/VOC2007 out=voc_val.json   imagesets=val
autovision-cli train-object-detection out=runs/voc model=rtdetr/resnet18 steps=2000 \
    data='{"format": "coco", "train": {"annotations": "voc_train.json", "images": "datasets/VOCdevkit/VOC2007/JPEGImages"}, "val": {"annotations": "voc_val.json", "images": "datasets/VOCdevkit/VOC2007/JPEGImages"}}'
```

`difficult` 目标转为 COCO ignore 区（不罚不欠召回）；关键点 / 分割的 YOLO 布局
用 `from=yolo-pose | yolo-seg`。

低标注预算的两条主线。伪标注 / 自训练（教师模型 → 伪标签 → 学生 → 评测，
端到端脚本 `scripts/self_train_detect.py`）:

```bash
autovision-cli pseudo-label model=teacher.pt images=unlabeled/ out=pseudo.json data=coco.yaml threshold=0.5
python scripts/self_train_detect.py --teacher teacher.pt --images unlabeled/ \
    --val-data coco_val.yaml --out runs/round1 --student rtdetr/resnet18 --steps 2000
```

误差分析（TP/FP/FN 匹配、FP 归因 duplicate/classification/localization/background、
按类别与 COCO 面积分档切片、top-FP/FN 可视化画廊）:

```bash
autovision-cli analyze-errors model=best.pt data=coco.yaml out=report/ iou=0.5 top_k=5 threshold=0.3
```

输出 `report/report.md`（含内嵌画廊）+ `report/report.json`（逐条匹配记录）。

## 导出与部署

```python
result = autovision.train_object_detection(out="runs/det1", data="coco8", steps=2000)
model = autovision.load_task_model(result.best_exported_model_path)   # 导出件 → 任务模型
model.export_onnx("model.onnx", precision="fp16")                     # FP16 + 数值对拍门禁
model.export_tensorrt("model.engine", precision="fp16")               # 需自行安装 tensorrt

autovision.benchmark("rtdetr/resnet18", out="bench/", num_classes=2,
                     backends={"torch": {}, "onnx": {"provider": "cpu"}},
                     verify_parity=True)

autovision.measure_model_cost(model)   # 参数量 / FLOPs（含 MACs；benchmark 报告内同样携带）
```

等价命令行(导出件后缀分发 `.onnx` / `.engine`):

```bash
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.onnx precision=fp16
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.onnx dynamic_spatial=true
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.xml        # OpenVINO IR(对拍门禁;precision=fp16 可出半精度 IR)
autovision-cli export model=runs/det1/exported_models/exported_best.pt out=model.engine precision=int8 calibration_data=calib_images/
```

**动态 H/W**（`dynamic_spatial=true`）：ONNX 图的 H/W 轴动态，导出时内置"第二尺寸
parity 探针"（torch↔ONNX Runtime 在未见尺寸对拍），并要求 dynamo 导出器
（fail-closed；严格 trace 失败时静态导出会回退 legacy TorchScript）。支持的骨干线：
`rtdetr/resnet18` 等 torchvision CNN、`detr/` 的 HGNetV2 谱系（`detr/n`、
`detr/s`、`detr/hgnetv2-*`、`detr/gtr-s`）；**HF 骨干（`detr/dinov2-*`、
`detr/dinov3-*`）不支持**——上游 transformers 在输入尺寸上有 Python 级分支/守卫，
严格导出无法符号化，会以可读错误指引（导出静态 H/W 或改用可符号化骨干）。
`detr/gtrvit-s` 带 export_blocker（GLA 无 ONNX 降级）。

**int8 首层 fp32 配方**：观测到（间歇性）部分主干（如 torchvision stem 的
Conv+Relu+MaxPool 融合）没有可用的 TensorRT int8 tactic，把首层留在 fp32 是
受支持的兜底配方；构建失败时错误信息会给出可粘贴的 `nodes_to_exclude`，也可直接：

```python
from autovision.export import Int8QuantizationConfig, suggest_first_layer_nodes
nodes = suggest_first_layer_nodes("model.int8.fp32.onnx")   # 前两个 Conv 节点名
config = Int8QuantizationConfig(nodes_to_exclude=tuple(nodes))
```

视频/RTSP 参考推理运行时(`autovision.inference`,需 `.[video]`;后端换 `onnx`/`tensorrt`
时附 `artifact=` 编译产物,前后处理与评测同源):

```bash
autovision-cli predict model=exported_best.pt source=clip.mp4 out=preds/ batch_size=4
autovision-cli predict model=exported_best.pt source=rtsp://camera/live out=preds/ backend=onnx artifact=model.onnx
```

输出 `out_dir/predictions.jsonl`(逐帧框,原始像素)+ `out_dir/annotated.mp4`(渲染画面)。

## 开发

```bash
pip install -e ".[dev,export]"   # CI 同款
pytest tests                     # 1600+ 个测试(含金标准 loss 轨迹快照)
ruff check src tests scripts     # lint(规则集见 pyproject.toml)
python scripts/check_provenance.py && python scripts/check_layering.py   # CI 门禁
python scripts/check_docs_links.py && mkdocs build --strict              # 文档门禁(需 .[docs])
```

- 金标准快照在 `tests/golden/`;**有意的**配方变更后删除快照重新录制(对应测试跑两遍)。
- 真实 `from_pretrained` 的测试默认跳过,`AUTOVISION_HF_TESTS=1` 开启(timm 权重下载测试同门控)。
- 多进程 DDP 测试:`AUTOVISION_DDP_TESTS=1 pytest tests/test_torchrun_distributed.py`。
- TensorRT int8 引擎构建测试(慢,需 tensorrt + CUDA):`AUTOVISION_TRT_TESTS=1 pytest tests/test_trt_int8.py`。
- 贡献入口:[CONTRIBUTING.md](CONTRIBUTING.md)(环境 / 门禁 / 许可与溯源红线);
  发布与版本政策 [docs/release_process.md](docs/release_process.md);变更记录 [CHANGELOG.md](CHANGELOG.md)。
- 文档:模型库(许可 / sha256 / 官方指标 / 加载代码)[docs/model_zoo.md](docs/model_zoo.md)
  (由 `scripts/gen_model_zoo.py` 从注册表机器生成);as-built 架构总览
  [docs/framework_design.md](docs/framework_design.md);基准表与口径协议
  [docs/benchmarks.md](docs/benchmarks.md);环境变量全表 `src/autovision/runtime/env.py`
  (`AUTOVISION_*`,遥测默认关闭)。文档站点 = `docs/` 下 `mkdocs.yml` 导航收录的页面
  (内部分析稿经 `exclude_docs` 显式排除,不进站点)。

## 状态与已知边界

- albumentations 适配层未接入(变换基于 torchvision v2);super-gradients 分割 wrapper 未接入。
- `detr/gtr-s`、`detr/gtrvit-s` 为规划档(装配覆盖 / COCO 趋势评审后冻结);`detr/m`、`detr/l`、`detr/x` 骨干线已定方向——timm ConvNeXt(tiny→m / small→l / base→x),选型标准为可导出 + 许可干净,待实现(当前 `convnext-*` 条目为 DINOv3-ConvNeXt,用户自持权重线)。
- GTR 模型(含 `detr/gtrvit-*`)当前不可导出 ONNX/TensorRT:导出入口以明确报错拒绝(GLA 自定义 CUDA 核无 ONNX 降级,见 `GTRViTBackbone.export_blocker`);四个 GTR 任务模型的导出 verify 钩子已就位,随 G2 部署工程生效。
- GTR 部署工程 G2:Triton GLA 核已迁移(2× 提速),但 `torch.compile` 对 GLA 段负收益、CUDA Graphs 与自研核原地写接口冲突(需 op 改造)、TensorRT 部署需官方 `gla_plugin.cu`;官方训练增广(mosaic/mixup/zoom-out/color-jitter)未移植;姿态官方配方为 `precision="32-true"`(`use_amp: False`)。
- ONNX FP16 parity 门禁在无 CUDA execution provider 时跳过并告警;ONNX 导出的示例输入跟随模型设备(GPU 上训练后可直接导出)。
- TensorRT 11 已适配:该版本移除了 EXPLICIT_BATCH 与 FP16/精度约束 builder flags——fp16 引擎改为经 fp16-ONNX + strongly-typed 网络路线构建(与旧版 builder flag 路线语义等价,FP32 黑名单继续保护溢出敏感算子)。
- 蒸馏教师默认请求预训练权重(`teacher_pretrained=True`),离线时传 `teacher_pretrained=False`。
- 预训练权重清单按 commit + LFS sha256 全量钉住(`src/autovision/encoders/weight_hashes.py`)。

## 许可与溯源

Apache-2.0,见 [LICENSE](LICENSE);第三方组件与权重溯源见 [NOTICE](NOTICE)。
GTR 权重为 MIT(整族 26 个,sha256 钉住);`ultralytics` 为 AGPL-3.0,仅作
extras 隔离加载,不进核心依赖、不在模块级 import。用户自持权重库的 7 个第三方
COCO 检测器为 `DINOv3-License+NOASSERTION`,禁再分发 / 商用,边界见
[用户自持权重库](#用户自持权重库d1-a仓库外不可再分发)。
