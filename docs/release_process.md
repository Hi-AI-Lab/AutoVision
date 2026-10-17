# 发布流程

本文档定义 AutoVision 的版本政策与发布清单。对应的变更记录见[变更日志](changelog.md)，
开发约定见[贡献指南](contributing.md)。

## 版本政策

- **语义化版本**（SemVer 2.0.0）：
  - `MAJOR`：公共 API 的破坏性变更（先经弃用政策过渡）；
  - `MINOR`：新能力、新任务面、新权重档位；
  - `PATCH`：向后兼容的修复。
- **版本号单一源**：`src/autovision/_version.py`。打包经 `pyproject.toml`
  动态读取（`[tool.setuptools.dynamic]`），运行时经 `autovision.__version__`
  暴露；三者不允许出现第二个书写位置。
- **发布 tag**：统一 `vX.Y.Z`。历史 tag `v0.2-clean` 为 0.2 线的干净谱系
  快照，保留原义，不重写、不移动。
- **弃用政策**：公共 API（`autovision.*` 顶层导出、CLI 命令与参数、配置
  字段）的弃用先在 MINOR 版本中标注（文档标注 + 运行时警告），至少跨一个
  MINOR 版本后才可移除；移除时在变更日志给出迁移方式。

## 发布清单（每次发布按序执行）

1. **门禁全绿**（CI 同款，逐条执行）：

   ```bash
   ruff check src tests scripts
   pytest tests -q
   python scripts/check_provenance.py
   python scripts/check_layering.py
   python scripts/check_docs_links.py
   scripts/gen_model_zoo.py --check
   mkdocs build --strict
   ```

2. **版本号**：更新 `src/autovision/_version.py`；确认打包与运行时读到的
   版本一致：

   ```bash
   pip install -e . --no-deps
   python -c "from importlib.metadata import version; import autovision; \
   assert version('autovision') == autovision.__version__"
   ```

3. **变更日志**：把 `CHANGELOG.md` 的 `[Unreleased]` 段收编为
   `[X.Y.Z] - YYYY-MM-DD`，并补一个空的 `[Unreleased]` 段。
4. **文档**：站点构建通过（`site/` 产物不进版本库）；模型库页面与注册表
   一致（`--check` 已覆盖）。
5. **发布快照**：对外仓库只承载发布快照——由当前树重建单提交快照分支，
   推送后打 `vX.Y.Z` tag；开发线历史不进入对外仓库。
6. **发布后**：核对远程 tag 指向的树与本地发布树一致；tag 推送会触发 CI 在
   打 tag 的提交上执行套件（`v*` 触发规则），其中 tag ↔ 版本一致性检查
   （`pytest tests/test_infrastructure.py -k release_tag`）必须通过；下个开发
   周期继续在 `[Unreleased]` 段记录变更。

## 权重与模型发布（政策）

- 仓库不携带权重二进制；注册表钉**身份**（sha256 / revision）而非分发
  （见[模型库](model_zoo.md)的三类资产：HF 清单 / GTR 用户自持 / 受限自持）。
- 自有训练产物如有对外发布需求，发布到 HuggingFace：
  1. 核对训练数据与权重的许可，确认允许再分发；
  2. 附 model card（训练口径、硬件、指标与复现脚本路径）；
  3. 把权重 sha256 回填到对应资产注册表；
  4. 在模型库页面登记（经 `scripts/gen_model_zoo.py` 重新生成）。
- 受限许可的权重（CC-BY-NC / 自定义 License）不允许进入公开注册表或随
  仓库 / 导出产物分发。
