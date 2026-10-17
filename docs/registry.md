# 私有模型注册表与部署门禁

`autovision.registry.ModelRegistry` 是文件系统后端：不可变版本 +
dev/staging/prod + `current` 指针 + 审计日志 + 部署门禁。

## CLI

```bash
export AUTOVISION_REGISTRY_ROOT=/srv/autovision-registry

autovision-cli registry action=register file=best.pt name=det v=1.0.0
autovision-cli registry action=list
autovision-cli registry action=promote name=det v=1.0.0 to=staging gates=all
autovision-cli registry action=promote name=det v=1.0.0 to=prod gates=all
autovision-cli registry action=current name=det stage=prod
autovision-cli registry action=rollback name=det stage=prod to=0.9.0 reason="regression"
autovision-cli registry action=audit
```

- `gates=all` 运行标准门禁；`gates=none` 仅做 sha 校验与晋级（测试/紧急通道）。
- `actor=` 写入审计日志；默认读 `AUTOVISION_ACTOR`，否则 `local`。

## 门禁

| gate | 检查 |
|---|---|
| `Sha256Gate` | 晋级副本哈希与注册 manifest 一致 |
| `ModelCardGate` | 模型卡存在、`validate()` 通过、文件 sha 与卡一致 |
| `LicenseGate` | 权重许可在部署 allowlist（默认宽松许可集） |
| `BenchmarkGate` | 模型卡有正的 `coco_map` 与至少一条延迟记录 |
| `ParityGate` | metadata `parity_passed=true` 或 `parity_report` 指向通过报告 |
| `DataSnapshotGate` | metadata `data_snapshot` 指向存在的数据快照 |

注册时模型卡（`<file>.modelcard.json`）与后处理规格
（`<file>.postprocess.json`）会随产物复制，门禁读取注册表内的副本。

## Python

```python
from autovision.registry import ModelRegistry, default_gates

registry = ModelRegistry("/srv/autovision-registry")
entry = registry.register("best.pt", name="det", version="1.0.0",
                          metadata={"parity_passed": True})
registry.promote(name="det", version="1.0.0", to_stage="staging",
                 gates=default_gates())
current = registry.current(name="det", stage="staging")
print(current.path, current.sha256)
```

## 回滚与审计

- `rollback` 只移动 `current` 指针，不复制/删除不可变版本。
- `audit.jsonl` 追加 register/promote/rollback 记录（时间、actor、版本、
  stage、原因）；不修改历史。
- `current.json` 原子写入（临时文件 + rename）。

## 边界

- 文件系统后端没有多租户/远程鉴权；私有部署请放在受控目录/网络。
- 门禁读取模型卡与 metadata；缺失证据即失败，不猜。
- 签名/HSM/OCI 分发不在当前实现内，可按企业需求叠加。
