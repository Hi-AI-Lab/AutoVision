# 服务面（serving + CLI + 错误体系）：设计与实施

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 状态 | as-built 设计 + 整改实施设计（待实施） |
| 日期 | 2026-11-06 |
| 审计基线 | v0.6.0 工作区；行号为该快照，实施时以符号名为锚 |
| 读者 | 框架研发、评审人、平台 / SRE |
| 关联 | [总览与路线](overview.md)、[架构总览](../framework_design.md)、[加固计划](../hardening_plan.md)（H2 系列已实施项不重复）、[已知限制](../known_limits.md) |

## 1. 范围与验收判定

范围：`src/autovision/serving/`（app/server/drift/ratelimit/metrics/
worker）、`src/autovision/cli/`（31 条命令）、`src/autovision/errors.py`、
`src/autovision/plugins.py`（抽查）。
判定：**参考实现定位内高质量，受控环境可用；对外生产需先落 2 个 P1**
（SERVE-01 状态一致性、SERVE-02 错误语义）与 SERVE-03/04/06。
攻击面加固（body/像素双上限、解压炸弹前置拒绝、请求 ID 清洗、错误
披露掩蔽、常时比较、非回环 fail-closed 绑定、多 worker 密钥防泄）
在 H2 系列已系统性落地并被测试钉住，本轮不重复。

## 2. 现状设计（as-built）

- **应用结构**：单工厂 `create_app`（`src/autovision/serving/app.py:252`）
  完成参数校验、引擎装配、A/B 权重归一、限流/drift/指标装配、路由。
  lifespan 只做逐引擎顺序 warmup（失败仅告警）。
- **并发模型**：无跨请求动态合批；`max_batch` 只约束单请求内 images
  列表；per-variant `asyncio.Semaphore(max_concurrency)`（默认 1，
  依据 TRT/OV runner 非线程安全）；解码与序列化走
  `run_in_threadpool`。
- **超时与毒化**：`queue_timeout_s` 与 `inference_timeout_s` 分离；
  超时后该 variant 被 `threading.Event` 毒化、slot 永久持有；
  `/readyz` 对单 variant 毒化保持 200、全毒化才 503（H2-3 已实施）。
- **限流/drift**：无第三方依赖、单锁原子 check-and-consume、LRU 驱逐
  有界、可注入时钟；PSI 滑窗 monitor。
- **CLI**：typer thin shell over 公共 API，`_parse_kwargs` JSON 优先
  解码，值由 API 层 Pydantic 校验（"参数同源"成立）；exit 2 = 用法
  错误、1 = 业务失败。
- **errors**：`AutoVisionError` 根 + Config 分支 +
  `ResultValidationError` 双继承；serving 内部私有异常
  （`_PayloadTooLarge`/`_ClientInputError`）不进公共层级。

## 3. 关键流程

HTTP 请求：request_context 中间件（X-Request-ID 白名单清洗 + 服务端
签发 routing_key + 访问日志）→ `/predict`：限流 → 鉴权（常时比较）→
variant 选择 → poison 短路 → Content-Length 预检 + 流式限长读体 →
JSON 解析 → 单图/批归一与上限 → 信号量 → 线程池 decode+forward →
记录数校验 → 线程池 RLE 序列化 → metrics/drift → 响应。

uvicorn 启动：transport 校验 → 非回环绑定 fail-closed（H2-1 已实施）
→ 单进程直启或 workers>1 经 `AUTOVISION_SERVE_CONFIG` 每 worker 重建。

## 4. 审计发现清单

### 4.1 P1

| 编号 | 问题 | 证据 |
|---|---|---|
| SERVE-01 | registry CLI 先变更后校验：`_reject_unknown_keys` 位于 register/promote/rollback 分支之后，typo 附加键时变更已生效再报 usage error（探针实证：register 被调用后 exit 2）；重试会产生重复审计记录或二义状态 | `src/autovision/cli/_app.py:383-430`、`:444-445` |
| SERVE-02 | 引擎/序列化边界的 `AutoVisionError` 一律映射 400：`ResultValidationError`（引擎输出契约违规）、`UnresolvedAutoError`（框架 bug）等纯服务端故障被归咎客户端——破坏重试策略、SLO 错误率区分与告警分诊 | `src/autovision/serving/app.py:1066-1093` |

### 4.2 P2

| 编号 | 问题 | 证据 |
|---|---|---|
| SERVE-03 | `serve` CLI 的 `port`/`workers` 裸 `int()` 无守卫 → 真实终端完整 traceback（同命令 `AutoVisionError` 有干净转换；TestClient 层测试掩盖了该路径） | `src/autovision/cli/_app.py:753-754` |
| SERVE-04 | 解码发生在信号量持有期内且被 `asyncio.wait_for` 与 forward 一起包住：纯解码超时也会把健康引擎永久毒化（该 variant 503 至重启），无恢复路径 | `app.py:586-606`、`:943-991` |
| SERVE-05 | lifespan warmup 无超时：引擎挂起（CUDA 停滞、NFS 工件）→ 启动永久不进入 serving 状态，编排器只能靠启动超时兜底 | `app.py:480-501` |
| SERVE-06 | 限流在反代拓扑下坍缩单桶（默认不信任 proxy 头 → 全体客户端=代理 IP 一个 key）；且 `_throttle` 与 `/predict` 内联限流共用同一 limiter 同一 per-client 桶——业务流量与 `/metrics` 抓取互相挤占 | `app.py:202-209`、`:629-650`、`:802-822` |
| SERVE-07 | workers>1 时父进程只做 `_jsonable` 序列化不做配置试运行：坏配置在每个 worker 内爆炸、uvicorn 崩溃重启循环，父进程无干净错误 | `src/autovision/serving/server.py:170-184`；`src/autovision/serving/worker.py:24-45` |

### 4.3 P3

| 编号 | 问题 | 一句话方向 |
|---|---|---|
| SERVE-08 | `_throttle` 的 429 不记录错误延迟（与 `/predict` 429 指标不对称） | 补 `_observe_error_latency` |
| SERVE-09 | `workers>1` 退出后 `AUTOVISION_SERVE_CONFIG` 残留父进程环境 | 退出时清理 |
| SERVE-10 | `/healthz` `/readyz` 无鉴权且披露 backend/模型类名/工件名/poison 状态 | loopback 默认可接受；公网部署文档化侦察面 |
| SERVE-11 | `bool()` 强转 JSON 布尔陷阱：`strict=False` 经 shell 保持字符串 → `bool("False")=True` | 布尔参数显式解析 |
| SERVE-12 | common_helpers/train_helpers 用户配置错误全抛裸 `ValueError`，未用自家异常家族 | 统一 `ConfigValidationError` |
| SERVE-13 | `get_epochs` 的 `int(epochs)` 无校验（2.5 静默截断、abc 裸崩） | 校验 + usage error |
| SERVE-14 | `get_out_dir` 多进程 overwrite 竞态：非 rank0 mkdir 后 rank0 才 rmtree | rank 同步点收敛 |
| SERVE-15 | 所有 `run_in_threadpool` 共享 anyio 默认线程池，variant 多时可饿死健康 variant | 池容量预留或专用池 |
| SERVE-16 | 默认参数组合内存偏激进（单请求解码 ~768MB 上界 + body 缓冲 ~512MB） | 文档化 sizing 指引 |
| SERVE-17 | plugins strict 回滚以 pass 起点为差集，并发注册场景可能误删 | 差集基线改为锁内快照 |

## 5. 整改实施设计

> 状态：**SERVE-01..09、11、12、13 已实施（工作区，随 0.6.1 发布）**；
> SERVE-10/14/15/16/17 登记为验收后第一批迭代项（见下）。
> 实施与设计的偏差：SERVE-02 按"5xx 分流 + 脱敏不变"实施（错误文案与
> request_id 语义不动，仅状态码与指标标签迁移）；SERVE-04 的"连续解码
> 失败 → degraded 告警"简化为计数指标 + 504（毒化语义不动，degraded
> gauge 另行排期）；SERVE-06 的 XFF 选取按标准链语义实现（客户端 =
> 从右第 N 项——每台可信代理追加一条，与旧单代理行为兼容）；
> SERVE-07 的 `validate_serve_config` 覆盖 create_app 全部纯可校验参数
> 并接受透传键（未知键在父进程即报错）。
>
> **实施后复审结论（只读代理复审 + serving/CLI 371 项回归全绿）：复审
> 判定"需修复后合入"，全部发现已闭环。** 复审驱动的追加修复：
> ① **P1-1（本轮自引入）**——`rate_limited` 延迟观测误落在放行路径
> 之外，凡启用限流的部署每笔成功请求都打 error_class 标签并双计进
> 直方图；已移入 429 分支，并新增"成功请求不得产生 rate_limited 序列"
> 的回归断言（原测试钉不住该回归）。② **P2-1 validate 覆盖缺口**——
> 受支持的弃用参数 `trust_proxy_headers` 被 workers>1 预校验误拒
> （违反弃用协议）；`threshold="0.9"` 字符串通过校验而请求期用原始值
> （每笔 500）；`rate_limit_*/backend` 值域只在 worker 内可达。已补：
> 签名收编弃用参数与透传键、严格类型检查（字符串/None 拒绝）、
> rate-limit 与 backend 白名单前置、ValueError 兜底。③ **P2-3**——
> CLI 面残留 4 处 `bool()` 强转中 kwargs 源的 3 处
> （`near_duplicate`/`include_dependencies`/`no_timestamp`）改走
> `coerce_bool_strict`（`ConfigValidationError`）；pydantic/内部字面量
> 源的 2 处非 CLI 面不属本轮。④ **P2-2**——warmup 超时后 to_thread
> 挂死线程无法取消，进程退出会被默认执行器 join 阻塞（永不返回的
> wedge 需编排器 SIGKILL），登记进 known_limits。⑤ P3——`_throttle`
> 的 429 延迟观测补齐（/metrics、/drift）、decode 504 加 Retry-After、
> `_parse_int_kwarg` 整数 whole-number 规则、ratelimit docstring 反代
> 指引、create_app docstring 收录三个新公共参数、`warmup_timeout_s=None`
> 落实"无预算"语义（对齐设计 §7）。
> 复审登记的排期项：export/analyze-errors 的未知键前置审计（P3-2，
> 需要任务面参数白名单）、degraded gauge（SERVE-04 的简化部分）、
> `hpo._as_int` 与 `_parse_int_kwarg` 的解析器收敛、model_handle 的
> `pretrained` bool 强转（内部字面量调用方，低风险）。

### 5.1 SERVE-01 registry CLI 校验前置（P1）

**设计**：把 `_reject_unknown_keys` 提到 action 分发**之前**：解析
`action` 后按动作取允许键集合（register/promote/rollback/gates/list
各自一张表），先校验后执行；未知键 → `typer.BadParameter`（exit 2）
且注册表零调用。顺带审计其余变更型命令（export/analyze-errors）的
未知键处理，缺的直接前置（加固计划 H3-6b 只覆盖了查询型 wrapper）。

**涉及文件**：`src/autovision/cli/_app.py`。

**测试计划**（`tests/test_cli_surface.py` 新增）：spy 注册表对象 +
typo 键 → registry 方法零调用 + exit 2；合法键 → 正常执行（回归）。
**验收标准**："变更生效但报 usage error" 在测试中不可再现。

### 5.2 SERVE-02 错误语义 4xx/5xx 分流（P1）

**设计**：异常 → 状态码映射表：

| 异常 | 状态码 | 响应体 |
|---|---|---|
| `_ClientInputError` | 400 | 稳定短语 + request_id（现状） |
| `_PayloadTooLarge` | 413 | 同上 |
| 限流 | 429 | 同上 |
| `TimeoutError`（推理超时） | 504 | 同上 |
| 其余 `AutoVisionError`（引擎/序列化/契约违规） | **500** | `"internal inference failure"` + request_id；原文只进服务端日志 |

**兼容性**：依赖"引擎错误返回 400"的客户端监控需要调整——
CHANGELOG 行为变更显著标注（建议随 0.7.0）；不提供兼容开关
（错误语义不应有长期双轨）。

**测试计划**（`tests/test_serving.py` 新增）：Fake engine 抛
`ResultValidationError` → 500 且响应不含内部路径；坏 base64 →
400 不变；超时 → 504。**验收标准**：`/metrics` 的 `error_class`
标签能区分 client/server 两类失败。

### 5.3 SERVE-03 serve CLI 数值参数守卫（P2）

**设计**：`port`/`workers` 解析改走显式 helper：`int()` 失败或越界
（port 1–65535、workers ≥1）→ `typer.BadParameter`（exit 2，文案含
合法范围）。**测试计划**：真实子进程 `port=abc` → exit 2 无
traceback（TestClient 之外补一条子进程级断言，补齐本轮发现的测试
盲区）。

### 5.4 SERVE-04 解码预算分离 + 毒化只看 forward（P2）

**设计**：

1. 新增 `decode_timeout_s`（默认 `min(10, inference_timeout_s / 2)`）：
   解码单独 `wait_for`；解码超时 → 504 + 计数，**不毒化**。
2. 毒化条件收敛为"forward 阶段超时或 forward 异常"（forward 仍在
   `inference_timeout_s` 内）；连续 decode 失败率超过阈值（如 50% ×
   10 次）→ 该 variant 标记 degraded 并告警（可观测，但不等于不可用）。

**涉及文件**：`src/autovision/serving/app.py`。

**测试计划**：慢解码引擎 → 504 且 `/readyz` 保持 200（修复前会毒化
503）；慢 forward → 毒化行为不变（既有 poison 测试组回归）。
**验收标准**：健康引擎不再被解码负载永久摘除。

### 5.5 SERVE-05 warmup 超时（P2）

**设计**：新增 `warmup_timeout_s: float = 60`：每引擎 warmup 包
`asyncio.wait_for`；超时 → lifespan 抛错（uvicorn 非零退出，错误含
variant 名），编排器的就绪探针自然不通过；`warmup_timeout_s=0` 表示
禁用 warmup（现状跳过语义保留）。**测试计划**：挂起的 Fake engine →
lifespan 在预算内抛错；正常引擎 → 不受影响；`tests/test_serving_server.py`
回归。

### 5.6 SERVE-06 限流拓扑正确性（P2）

**设计**：

1. 桶键按端点命名空间化：`key = f"{route}:{client}"`（`/metrics` 与
   `/predict` 天然分桶），消除端点互挤。
2. 反代支持显式化：`trust_proxy_headers` 从 bool 升级为
   `trusted_proxy_hops: int = 0`（0=不信；N=从右往左跳过 N 个代理取
   真实客户端 IP，`X-Forwarded-For` 解析 + 私有网段过滤）。
3. `src/autovision/serving/ratelimit.py` docstring 补 per-client
   collapse 语义与反代配置指引；`known_limits.md` 同步。

**测试计划**：trust=1 + `X-Forwarded-For: a, b` 两客户端 → 两桶；
`/metrics` 抓取不被 `/predict` 配额挤占；无 trust → 现状行为不变。

### 5.7 SERVE-07 workers>1 父进程预校验（P2）

**设计**：把 `create_app` 的参数校验段抽为纯函数
`validate_serve_config(**kwargs)`（无 I/O，可独立调用）；`serve()` 的
workers>1 分支在序列化前先调用它 → 干净的单条错误。worker 侧继续
走 `create_app`（校验幂等）。**测试计划**：非法 `mask_format` +
workers=2 → CLI exit 1 单条 `AutoVisionError`，无 worker 重启循环；
workers=1 路径回归。

## 6. 实施顺序

SERVE-01（一行级高价值）→ SERVE-02（语义变更，随 MINOR）→ 04 → 06
→ 05 → 03 → 07。P3 中 SERVE-11/12/13（CLI 参数卫生）建议合并一个
commit 批次处理。

## 7. 风险与回滚

- SERVE-02 改变对外状态码语义：CHANGELOG 显著标注 + 发布说明给出
  监控规则迁移指引；回滚 revert。
- SERVE-04/05 引入新默认参数（`decode_timeout_s`、
  `warmup_timeout_s`）：默认值即修复目标，存量行为依赖方可在
  构造参数显式关闭（`= None` 退化为无预算）。
- SERVE-06 的 `trusted_proxy_hops` 是破坏性参数改名（原
  `trust_proxy_headers` 为 bool）——按弃用协议：0.6.x 双参数并存 +
  deprecation warning，0.7.0 移除。
