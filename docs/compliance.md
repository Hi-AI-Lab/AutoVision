# 企业合规报告

`autovision-cli compliance-report` 把企业评审需要的证据汇编成
`compliance.json` + `compliance.md`：

- 框架版本 / commit / Python / 平台；
- 已安装依赖的版本与许可（未知许可显式 `UNKNOWN`，不静默放过）；
- 模型文件 sha256、模型卡（许可 / 指标 / 延迟 / 训练口径）；
- 数据快照（`data-snapshot` 产物）；
- 问题清单（缺失模型卡、sha 不一致、许可待复核等）。

## 用法

```bash
autovision-cli compliance-report \
    model=runs/det/exported_models/exported_best.pt \
    snapshot=qa/data.snap.json out=compliance/
```

- `include_dependencies=false` 跳过依赖扫描（快速检查模型/数据证据）。
- `no_timestamp=true` 生成可复现报告（不写 `generated_at`）。
- `strict=true` 在存在 error 时退出码 1（适合 CI/发布门禁）。

## 判定

- `error`：模型无许可、模型卡 sha 与文件不一致、模型卡非法等；
- `warning`：缺模型卡、依赖许可未知/待复核、无数据快照；
- `info`：未提供快照等提示。

报告是**证据汇编**，不是法律意见；许可最终结论由法务确认。默认允许清单为
常见宽松许可（Apache-2.0 / MIT / BSD / ISC / MPL-2.0）；企业可在代码中
传入自定义 allowlist。

## 边界

- 不读取受限权重内容，只读文件哈希与旁车元数据；
- 不联网；
- 依赖许可来自已安装包元数据，缺失/自定义条款会显示 `UNKNOWN` 或原文。
