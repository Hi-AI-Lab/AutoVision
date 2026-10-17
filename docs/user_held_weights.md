# 用户自持权重

> 7 个第三方训练的 COCO 检测器（DINOv3 骨干 + 第三方检测头）经合规重打包后存于
> 仓库之外，**不可再分发**。资产表、许可与 sha256 见[模型库](model_zoo.md)；
> 本页讲加载路线与覆盖要求。**其中 3 个 ViT 资产为 STA-era，当前加载器不支持；
> 4 个 ConvNeXt 资产可正常加载。**

## 存储与许可

- 默认根目录 `~/Research/Model/AutoVision`，`AUTOVISION_USER_HELD_ROOT` 覆盖。
- 每个资产为 `weights.pt` + `warm_start.pt` + `ASSET.json` 三件套。
- 资产名与两份容器的 sha256 在 `autovision.registry.user_held_assets` 钉住
  （`resolve_user_held_asset` 解析、`verify_user_held_asset` 离线核验）。
- 许可：`DINOv3-License+NOASSERTION`。骨干继承 DINOv3 License 义务，检测头无
  许可声明（默认保留全部权利）。**禁再分发、禁商用**；不随仓库或导出产物分发，
  不注册可下载资产清单（`ASSET_MANIFEST`）。使用即视为已自行接受相应条款。

## 加载路线 A：检查点热启

ConvNeXt 检查点用 6 层 / 256 宽解码器训练，**不是**变体规格默认的 3/4 层 × 384
宽——不带覆盖会被严格加载拒绝。`user_held_model_args(<资产名>)` 一次取全所需
`model_args`（含 `backbone_pretrained=False` 与 `image_size`）。

```python
import autovision
from autovision.registry.user_held_assets import user_held_model_args

autovision.train_object_detection(
    out="runs/d1", data="data.yaml", model="detr/dinov3-convnext-base",
    model_args=user_held_model_args("dinov3-convnext-base"),
)
```

ConvNeXt 四档均需 `dec_layers=6` + `decoder_hidden_dim=256`，
`user_held_model_args` 已包含。`dinov3-vitt16` / `dinov3-vitt16plus` /
`dinov3-vits16` 含 `backbone.sta.*`：DEIMv2 派生 spatial prior 已删除，加载器
会抛出 `UserHeldAssetError`，需重新训练/导出 no-STA 检查点后才能使用。

## 加载路线 B：AVWarmStart v1

我方键布局，形状兼容子集迁移：

```python
import autovision

autovision.train_object_detection(
    out="runs/d2", data="data.yaml", model="detr/dinov3-convnext-base",
    model_args={
        "warm_start": "dinov3-convnext-base",   # 或 warm_start.pt 的路径
        "backbone_pretrained": False,
        "dec_layers": 6,
        "decoder_hidden_dim": 256,
        "image_size": (640, 640),
    },
)
```

## 离线核验

`verify_user_held_asset` 只对文件做 sha256 核验，不加载模型；资产名或内容不符即
抛错。加载 model 键与构建覆盖的对应关系见[模型库](model_zoo.md)。
