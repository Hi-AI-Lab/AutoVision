# Clean-cut 发布手册（分册 28 §7.3 / Phase 5）

状态：本地部分已完成（2026-10-14）；对外推送待配置远程后执行。

## 已完成的本地部分

1. **`v0.2-clean` tag**（main 上，commit `7993834`）：全部整改内容 +
   进行中的 fsdc/deimv2 工作。本地私有历史保留 pre-remediation 快照
   （不分发即不触发 AGPL 义务）。
2. **`v0.2-clean-publish` 孤儿分支**：单提交、树哈希与 tag 完全一致
   （`5df85263`），**不含任何 pre-remediation 历史**。这是对外发布的
   唯一对象。

## 对外发布步骤（远程就绪后）

> **推送前刷新**：若 tag 之后 main 还有新提交（如 Phase 4 收尾），
> 先重建孤儿分支再推送：`git checkout -q --orphan publish-tmp && git
> commit -q -m "VTrain v0.2-clean (publication snapshot)" && git branch
> -f v0.2-clean-publish && git switch -q main && git branch -D publish-tmp`

```bash
# 1. 配置发布远程（新仓库，不要用承载私有历史的旧远程）
git remote add publish <PUBLISH_REMOTE_URL>

# 2. 只推送孤儿分支（单提交，零 pre-remediation 历史）
git push publish v0.2-clean-publish:main

# 3. 在发布仓打同样的 tag
git tag -a v0.2-clean -m "Clean-provenance v0.2" v0.2-clean-publish && \
    git push publish v0.2-clean
```

**禁止**：`git push publish main`（main 含 14 个 pre-remediation 提交，
其中 "Baseline import: pre-remediation snapshot" 含与 lightly-train
逐字对照的历史文本——泄露即触发 AGPL 义务）。

## 发布前红线复查

- [x] `scripts/check_provenance.py` 全绿（禁词零命中，allowlist 仅余
      Phase 4 资产切换相关条目）
- [x] `scripts/check_layering.py` 全绿
- [x] `ruff check src tests` 全绿
- [x] 无二进制权重入库（`.gitignore` 挡 `*.pt/*.pth`，`weights/` 下
      yolo26n.pt 未跟踪）
- [ ] **Phase 4 收尾后**：legacy registry 三条目（license=None）被
      AVWarmStart 资产替换，allowlist 清零，§9 gate 5 完全达成

## 验收快照（2026-10-14，§9 各门）

| Gate | 状态 |
|---|---|
| 1. src 内禁词零命中 | ✅ |
| 2. 全库相似度（模式 2） | ✅ 定向验证（重移植件 diff-vs-upstream 纯净）；全量复测随 Phase 5 发布审计执行 |
| 3. 测试全绿 + golden | ✅ 1505 过 / 0 失败 |
| 4. 公开 API 兼容 + remap | ✅（alias + 三代 remap 测试在案） |
| 5. 资产 license 100% | ⏳ 新 Apache 资产 100%；legacy 3 条待 Phase 4 训练产物替换 |
| 6. 分层契约 | ✅ |
| 7. clean cut | ✅ 本地完成；远程推送待远程配置 |
