# AutoVision

企业级视觉模型训练框架：**自监督预训练 / 蒸馏 → 微调 → 评测 → 导出**一条链走完。
Python API 与 `autovision-cli`（25 条命令）共享同一套 Pydantic 严格校验，覆盖
检测、旋转框、实例 / 全景 / 语义分割、姿态、深度、分类的完整任务面。

框架的三条差异化主线：

- **许可干净**：Apache-2.0；全仓溯源门禁与权重许可门控（受限许可权重默认
  不下载，AGPL 组件仅 extras 隔离），模型库页面对每个权重给出许可与 sha256。
- **可验证一致性**：ONNX / TensorRT 导出默认执行数值对拍门禁；官方权重加载
  与前向对拍（含 bit-equal）；金标准 loss 轨迹快照。
- **协议化基准**：[基准表](benchmarks.md)只引官方发表值、自家数字必附复现
  脚本、只增不改 + 勘误节。

## 快速开始

目标检测，训练 → 验证 → 预测 → 导出：

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

等价命令行（同一套 Pydantic 校验）：

```bash
autovision-cli train-object-detection out=runs/det1 data=data.yaml steps=2000
autovision-cli val     model=runs/det1/exported_models/exported_best.pt data=coco8
autovision-cli export  model=runs/det1/exported_models/exported_best.pt out=model.onnx precision=fp16
autovision-cli predict model=model.pt source=clip.mp4 out=preds/ batch_size=4
```

## 安装

Python ≥ 3.10；模型包按 extras 惰性加载，缺失时对应清单为空而不是报错：

```bash
pip install -e .                                # 核心框架(不含第三方模型包)
pip install -e ".[transformers,timm,export]"    # 常用: HF 骨干 + timm + ONNX 导出
pip install -e ".[video]"                       # 参考推理运行时(视频/RTSP, OpenCV)
pip install -e ".[dev]"                         # 开发(pytest / ruff, CI 同款)
```

GTR 全族（MIT）需要 `gtr` extra；完整 extras 表见仓库 README。预训练权重从
HuggingFace 下载，受限网络可设镜像；**受限许可权重默认不下载**（DINOv2
CC-BY-NC 4.0 / DINOv3 License 需显式自证接受条款），许可边界见仓库 `NOTICE`。

## 能力速览

| 子系统 | 能力 |
|---|---|
| 配置 / 注册 | Pydantic 严格校验 + did-you-mean 报错 + 注册表；CLI 与 API 同一套校验 |
| 模型层 | 7 个注册源（torchvision / timm / super_gradients / ultralytics（AGPL 隔离）/ dinov2 / dinov3 / rfdetr）+ 树内 `rtdetr/*`、`detr/*`、`dfine/*`、`picodet/*`、`rfdetr/*`、`gtr/*` |
| 数据 / 变换 | 11 类数据集；mmap 文件清单（内容指纹 / 原子发布）；检测增广全套（mosaic / mixup / copyblend / box-aware eraser）；VOC / YOLO-pose / YOLO-seg 转换 + COCO 合并 |
| 预训练方法 | DistillationV3、SimCLR、DINO、DINOv2、DINOv3、DenseCL、DetCon B/S、训练中 kNN 探针 |
| 运行时 | DDP、混合精度、原子 checkpoint + resume、NaN 捕获、JSONL / TB / MLflow / W&B、OTel（opt-in） |
| 任务层 | 8 个任务面；任务指标支持跨 rank 汇聚（DDP 验证）；误差分析（FP 归因 + top-FP/FN 画廊）+ 伪标注 / 自训练闭环；超参搜索（随机 / 网格 / ASHA + 报告） |
| 部署 | ONNX（FP16 黑名单 + 数值对拍门禁）、TensorRT、int8 QDQ、benchmark 三后端 parity、参考推理运行时（视频 / RTSP + batch，同源前后处理；可选 ByteTrack 跟踪）、参考 serving（FastAPI + /metrics） |

各模型的许可 / sha256 / 官方指标 / 一行加载代码见[模型库](model_zoo.md)。

## 文档导航

| 页面 | 内容 |
|---|---|
| [模型库](model_zoo.md) | 每模型：许可 / sha256 / 官方指标 / 加载代码（机器生成） |
| [基准](benchmarks.md) | 精度与延迟基准（不可变协议 + 勘误节） |
| [架构总览](framework_design.md) | as-built 分层契约 / 模块地图 / 质量门禁 |
| [贡献指南](contributing.md) | 开发环境 / 提交前门禁 / 许可与溯源红线 |
| [发布流程](release_process.md) | 版本政策 / 发布清单 / 权重发布政策 |
| [变更日志](changelog.md) | Keep a Changelog 格式的对外变更记录 |

## 许可

Apache-2.0。第三方组件与权重溯源见仓库根 `NOTICE`；交互式命令全表：

```bash
autovision-cli --help
autovision-cli list-models    # 当前环境可用的模型清单(按已安装 extras 过滤)
```
