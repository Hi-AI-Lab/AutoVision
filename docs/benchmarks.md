# DETR 谱系基准表（roadmap design doc 29 §4）

> **协议（D5/D6/D7，不可变）**：AP 口径 = COCO 2017 val2017 + pycocotools
> 标准；对外数字一律标注 schedule 档位与种子，关键档位 ≥2 seeds 报均值。
> 延迟 = TensorRT，batch=1 与 batch=4 两档，含 H2D 拷贝与预处理，warmup 50
> 取 p50；表头登记硬件型号，不同硬件不同表。对手数字**只引用官方仓库/论文
> 发表值**并给出处与引用日期，不复现对手；自家数字必须附复现脚本。
> **本表只增不改**：修正进文末勘误节（日期 + 原因）。

## 1. 自家精度（COCO val2017 AP）

| 模型 | 档位来源 | schedule | seeds | AP | AP50 | 权重 | 模型卡 | 复现脚本 | 日期 |
|---|---|---|---|---|---|---|---|---|---|
| detr/n | hgnetv2-b0（谱系冻结） | 12ep 蒸馏（RF-DETR-n 教师） | — | — | — | — | — | `scripts/train_spectrum_n_distill.py` | — |
| detr/s | hgnetv2-b1 + dfine_s 装配 | 装配后短程微调 | — | — | — | — | — | `scripts/finetune_spectrum_s.py` | — |

### 1.1 短程趋势冒烟（非发布口径——流程/数值健康证据，数字不进对外叙事）

| 模型 | 配置 | 步数（epoch） | AP | AP50 | 备注 | 日期 |
|---|---|---|---|---|---|---|
| detr/s | dfine_s 装配暖启 + timm 预训练骨干 + EMA warmup 100 | 2000（0.17ep） | 0.1302 | 0.2026 | val 单调升 9.6→13.0；RTX 4090D bf16 | 2026-10-15 |
| detr/gtrvit-s | **官方 GTR-S 骨干直载** + 随机 GTREncoder/decoder | 2000（0.17ep） | 0.1902 | 0.3109 | 较 detr/s 同期 **+6.0 AP**；GLA 骨干质量兑现 | 2026-10-15 |
| detr/gtrvit-s | 全随机（对照） | 500（0.04ep） | ~0.0008 | ~0.0018 | 从零训 ViT 系无信号，符合预期 | 2026-10-15 |

## 2. 自家延迟（torch eager 口径，含预处理；TRT 列待 P3）

硬件：NVIDIA GeForce RTX 4090D 24GB / torch 2.x cu / Python 3.10；p50/50 iters/warmup 20；复现 `scripts/measure_spectrum_latency_torch.py`。日期 2026-10-15。

> ⚠️ **口径警示**：本表为 **torch eager 含 CPU 预处理** 口径，与 §3 的官方协议（torch.compile + CUDA Graphs + 自研 CUDA GLA 核 + forward-only FP16）**不可直接对比**。同口径粗测：gtrvit-s forward-only FP16 bs1 在我们集成路径为 ~33ms vs GTR 官方全模型 1.225ms（~27×）——差距来源：官方用自研 chunkwise CUDA GLA 核（比 fla Triton 快 4×@1.6K tokens）+ torch.compile + CUDA Graphs + convert_to_deploy 融合。**GTR 的速度优势必须配套这三件部署工程才成立**（分册 31 G2 剩余矩阵）；TRT 部署还需其自定义 gla_plugin.cu（官方车端 DRIVE AGX Thor 2.28-8.77ms 已验证该路线）。**勘误（2026-10-15 晚）**：早前版本称 torch.compile「需 py≥3.11、本 venv 3.10 不可用」系误读——实测 `.venv`（py3.10.12）下 torch.compile 可正常编译（dynamo frames ok=1）；那条「not available in Python 3.10」警告来自 **fla 库自身的降级策略**（fla 在 <3.11 停用其内部编译装饰器，官方参考环境为 py3.11）。torch.compile 对本仓库完全可用。
>
> **部署工程迁移实测（2026-10-15，GTR-S 骨干 640px FP16 bs1，无预处理/无 decoder）**：fla Triton eager **16.25ms** → deploy（convert_to_deploy 递归 28 钩子）+ **自研 CUDA GLA 核 7.97ms（2.04×）**——核已从 GTR-main csrc 编译迁移（nvcc 13.0/torch cu130 匹配，`-std=c++20` 补丁，官方 test_gla ALL PASS，`.so` 经包 `__init__` sys.path 注入零改动接线）。另两项实测结论：① **torch.compile 默认模式对 GLA 骨干负收益**（9.62ms：自研核经 custom op + `torch.compiler` 图断裂，编译无法覆盖 GLA 段、且引入 guard/重编译开销；对 GLA 无 custom-op 断裂的路径 compile 仍可用——py3.10 不是障碍，见上勘误）；② **CUDA Graphs 当前被拒**（custom op `mutates_args={"out"}` 原地写与 graphs 冲突，需 op 接口改造）；③ **整网 half eager 与我们 decoder 的 bbox FP32 精修块（fp16 溢出保护，有意设计）冲突**——混合精度导出（TRT）才是正确部署形态，不做整网 half。

| 模型 | 精度 | batch=1 p50 (ms) | batch=4 p50 (ms) | 备注 |
|---|---|---|---|---|
| detr/n | fp32 | 19.3 | 20.6 | eager 含 CPU 预处理，TRT 口径另测 |
| detr/n | fp16 | 22.6 | 24.8 | autocast 开销为正收益前口径 |
| detr/s | fp32 | 20.0 | 20.6 | |
| detr/s | fp16 | 24.1 | 24.8 | |
| detr/gtr-s | fp32 | 21.6 | 23.4 | 装配变体（HGNetV2-b1 骨干） |
| detr/gtr-s | fp16 | 27.7 | 28.0 | |
| detr/gtrvit-s | fp32 | 29.2 | 34.2 | GLA Triton 核；fp16 慢于 fp32（核内 fp32 计算+转换） |

## 3. 对手官方数字（只引官方发表值）

**统一协议来源**：GTR 论文（arXiv:2609.26590，Intellindust-AI-Lab）将下列全部基线在**同一协议**下重测延迟——RTX 4090 / FP16 / batch=1 / torch.compile + CUDA Graphs / median CUDA-event latency / 640×640；基线 AP 引自各自原文。引用日期 2026-10-15。⚠️ 该协议与我们 §2 的 torch eager 口径不可直接对比（见 §2 表注）。

| 模型 | Params (M) | GFLOPs | 延迟 (ms) | COCO AP |
|---|---|---|---|---|
| **GTR-S**（O365 预训练） | 12.1 | 33.8 | **1.225** | **53.6** |
| GTR-M | 22.7 | 62.6 | 1.462 | 57.3 |
| GTR-L | 37.2 | 106 | 1.908 | 58.9 |
| GTR-X | 46.5 | 130.2 | 2.115 | 59.4 |
| D-FINE-S | 10 | 25 | 1.194 | 50.7 |
| D-FINE-M | 19 | 57 | 1.740 | 55.1 |
| D-FINE-L | 31 | 91 | 2.462 | 57.1 |
| D-FINE-X | 62 | 202 | 3.385 | 59.3 |
| DEIM-S | 10 | — | 1.180 | 49.0 |
| DEIM-M | 19 | — | 1.744 | 52.7 |
| DEIM-L | 31 | — | 2.460 | 54.7 |
| DEIM-X | 62 | — | 3.384 | 56.5 |
| DEIMv2-S | 10 | — | 1.687 | 50.9 |
| DEIMv2-M | 18 | — | 2.398 | 53.0 |
| DEIMv2-L | 32 | — | 2.776 | 56.0 |
| DEIMv2-X | 50 | — | 3.276 | 57.8 |
| RT-DETRv2-S | — | — | 1.291 | 48.1 |
| RT-DETRv2-M | — | — | 1.741 | 49.9 |
| RT-DETRv2-L | — | — | 2.332 | 53.4 |
| RT-DETRv2-X | — | — | 3.259 | 54.3 |
| RF-DETR-S | 32 | 60 | 1.299 | 52.9 |
| RF-DETR-M | 34 | 79 | 1.431 | 54.7 |
| RF-DETR-L | 34 | 126 | 1.968 | 56.5 |
| RF-DETR-X | 126 | 300 | 3.243 | 58.6 |
| YOLO26-S | 10 | 21 | 0.962 | 47.8 |
| YOLO26-M | 20 | 68 | 1.473 | 52.5 |
| YOLO26-L | 25 | 86 | 1.967 | 54.3 |
| YOLO26-X | 55 | 194 | 3.087 | 56.9 |
| YOLO11-S | — | — | 1.490 | 46.6 |
| YOLO11-M | — | — | 2.061 | 51.2 |
| YOLO11-L | — | — | 2.617 | 53.4 |
| YOLO11-X | — | — | 3.747 | 54.7 |

**同延迟档读数**（4090 FP16）：GTR-S 1.225ms/AP 53.6 为 ~1.2ms 档冠军（D-FINE-S 1.194/50.7、RF-DETR-S 1.299/52.9、DEIM-S 1.180/49.0）；M/L/X 档对 D-FINE 同档分别 **快 16%/22%/37%** 且 AP +2.2/+1.8/+0.1；对 YOLO26：更快一档但 GTR 同延迟 AP 高 5+。

| 模型 | AP（val2017） | 延迟（官方口径） | 出处 | 引用日期 |
|---|---|---|---|---|
| _（首批已并入上表；后续逐行补官方原始出处链接）_ | — | — | — | — |

## 4. 勘误

| 日期 | 条目 | 修正 | 原因 |
|---|---|---|---|
| — | — | — | — |
