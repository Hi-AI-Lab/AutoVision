# VTrain 实现蓝图 · 分册 03：模型层

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §6 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 6. 模型层设计

### 6.1 骨干协议（照抄基线并分层）【基线】

`_models/model_wrapper.py`（全文 246 行，VTrain 等价物直接移植）：

```python
class ForwardFeaturesOutput(TypedDict, total=False):
    features: Required[Tensor]        # (B, C, H_out, W_out)
    cls_token: NotRequired[Tensor]    # (B, C)

class ForwardPoolOutput(TypedDict, total=False):
    pooled_features: Required[Tensor] # (B, C, 1, 1)

class ArchitectureInfo(TypedDict):    # 无 enum，字符串字面量
    model_type: Literal["convolutional", "transformer", "hybrid"]
    norm_type:  Literal["batchnorm", "layernorm"]

# 方法协议（全部 @runtime_checkable，独立声明、组合使用）
ForwardFeatures.forward_features(x) -> ForwardFeaturesOutput
ForwardPool.forward_pool(x: ForwardFeaturesOutput) -> ForwardPoolOutput
FeatureDim.feature_dim() -> int
ModelGetter.get_model() -> PackageModel          # checkpoint 里存的对象
ArchitectureInfoGettable.architecture_info() -> ArchitectureInfo   # 可选：驱动 auto 决策
PatchSize.patch_size() -> int
MultiScaleFeatureDims.multiscale_feature_dims() -> list[int]
MultiScaleFeatureStrides.multiscale_feature_strides() -> list[int]
ForwardMultiScaleFeatures.forward_multiscale_features(x, layer_indices) -> list[ForwardFeaturesOutput]

# 组合协议
class ModelWrapper(ForwardFeatures, ForwardPool, FeatureDim, ModelGetter, NNModule, Protocol): ...
class MultiScaleFeatureViT(ModelWrapper, ForwardMultiScaleFeatures, MultiScaleFeatureDims, PatchSize, Protocol): ...
class MultiScaleFeatureCNN(ModelWrapper, ...多 strides 版): ...
```

配套机制：`missing_model_wrapper_attrs()` 生成「你的模型缺哪些协议成员」的诊断信息（`model_wrapper.py:229-246`）；`EmbeddingModel(wrapped_model, embed_dim)` 用 `Conv2d(C,embed_dim,1)` 头包装骨干供预训练（`_models/embedding_model.py:22-80`）；timm 架构分类靠 170 项前缀表 `_TIMM_ARCH_NAME_PREFIXES`（`_models/timm/timm.py:28-200`）。

【差异】VTrain 把协议拆成两组：预训练组（上表原样）+ 任务组（§8.5 `DetectionWrapper`）。基线的任务侧没有显式协议，而是 `TrainModel` 类级 `ClassVar` 约定（`task/train_model_cls/task_metric_args_cls/...`，`_task_models/train_model.py:47-87`）——VTrain 保留 ClassVar 钩子（好用），但为跨包分发补显式 Protocol。

### 6.2 包注册表【基线】

```python
class BasePackage(ABC):
    name: str
    @classmethod @abstractmethod
    def export_model(cls, model, out: Path, log_example=True) -> None
    @classmethod @abstractmethod
    def is_supported_model(cls, model) -> bool

class Package(BasePackage):
    @classmethod @abstractmethod
    def list_model_names(cls) -> list[str]                 # "pkg/model" 全名
    @classmethod @abstractmethod
    def get_model(cls, model_name, num_input_channels=3, model_args=None, load_weights=True)
    @classmethod @abstractmethod
    def get_model_wrapper(cls, model) -> ModelWrapper

class MultiScaleFeaturePackage(Package): ...   # wrapper 返回 MultiScale 协议
```

- 分发顺序（`package_helpers.py:34-48`）：`rfdetr → super_gradients → timm → torchvision → ultralytics → dinov2 → dinov3 → edgecrafter → custom`；custom 永远最后（last resort），只接受 `ModelWrapper` 实例。
- `parse_model_name`：`split("/", maxsplit=1)`（容纳 `ultralytics/path/to/model.pt`）；`dinov2_vit`→`dinov2` 改名兼容。
- 各包机制速览：

| 包 | 模型清单来源 | 构建方式 | 通道限制 |
|---|---|---|---|
| timm | `timm.list_models()` 全量 | `timm.create_model(name, pretrained=False, in_chans=n)`，预训练需显式 `model_args` | 任意 in_chans |
| torchvision | `list_models()` 正则过滤 3 家族 | `torchvision.models.get_model` | 仅 3 通道 |
| dinov2 | `MODELS` dict（12 个公开 + 16 隐藏别名） | vendored 代码 + OmegaConf 配置 + 自研下载器 | 任意 in_chans（patch_embed hook 裁剪/复制通道） |
| dinov3 | `MODEL_NAME_TO_INFO`（27 公开） | vendored + `torch.hub` 下载、patch 重采样 | 任意 in_chans |
| ultralytics | GITHUB_ASSETS_STEMS 合成 | `YOLO()/RTDETR()`；wrapper 手工找 backbone 边界（按层类型精确匹配，`_models/ultralytics/ultralytics.py:107`）、重新打开梯度 | 3 通道 |
| rfdetr / super_gradients / edgecrafter | 各自动态/静态清单 | 各自适配（非 nn.Module 的对象用 `self._model=[model]` 列表技巧规避注册） | 3 通道 |

- 自定义 `nn.Module`：`get_package_from_model` 找不到归属且 `fallback_custom=True` 时返回 custom 包；checkpoint 以 **pickle 整对象**保存（`_checkpoint.py:59-78`），加载 `weights_only=False`——类必须可导入。

### 6.3 DINOv2/v3 实现策略【基线→差异】

【基线】两者均为 **vendored 源码 + 权重下载**（非 transformers、非 torch.hub 模型仓）：

- `dinov2_vit/dinov2_vit_src/`（Apache-2.0 头）+ OmegaConf YAML 配置；权重 URL `dl.fbaipublicfiles.com`（reg4）与 Lightly S3（TiPSv2）；自研下载器（local-rank0 下载 + `dist.barrier()` + `weights_only=True` 严格加载，`dinov2_helper.py:24-62`）。
- `dinov3/dinov3_src/`（DINOv3 License Agreement）含 RoPE、SparseLinear、FP8 Linear 等层；`torch.hub.download_url_to_file` 下载，**无哈希校验（源码 TODO 明示）**；patch size 不匹配时 `resample_conv_weight` 重采样。
- Wrapper 语义：`forward_features` 可选 `n_blocks>1` 取中间层拼接；DINOv2 `forward_pool` 返回 **cls_token**（非平均池化）；`make_teacher()` 关闭 stochastic depth 供蒸馏；`input_normalization="none"` 的 TiPSv2 变体在前向里先**抵消**框架已做的 ImageNet 归一化（`dinov2_vit.py:118-125`，易踩坑点，VTrain 以模型元数据声明归一化代替这种隐式抵消）。
- 通道适配 hooks：`patch_embed_adjust_input_channels_hook`（checkpoint 通道多则裁剪、少则重复拼接）、`interpolate_pos_embed_hook`（bicubic 插值位置编码，`_models/_model_helpers.py:21-111`）。

【差异】VTrain 策略：优先用依赖（transformers 已含 DINOv2），能力缺口才 vendor；vendor 必须带 ORIGIN.md + 升级脚本。**权重供应链为 VTrain 硬性增强**（基线全链路零校验）：`{url, sha256, license, source}` 强制校验、失败删缓存重试一次、内网镜像源 `AUTOVISION_MIRROR_URL`、`AUTOVISION_OFFLINE=1`。包级许可标签随 `list_models()` 暴露。

### 6.4 检测任务模型协议【设计】（基线无显式协议，此为从基线实践中归纳的契约）

```python
@runtime_checkable
class DetectionWrapper(Protocol):
    def load_pretrained(self, nc: int) -> None        # 类头迁移：形状匹配复用/截断补齐(§8.4)
    def get_param_groups(self) -> list[ParamGroup]    # 四组结构(§8.3)
    def input_size(self) -> tuple[int, int]
    def raw_boxes_format(self) -> Literal["cxcywh_norm", ...]
```

---
