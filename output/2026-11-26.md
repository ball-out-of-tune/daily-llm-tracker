# 🚀 vLLM & SGLang 每日更新 — 2026-11-26

> 自动生成于 2026-11-26 10:00 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：Rust Frontend（Rust 前端）**
  - 一句话通俗解释：vLLM 正在用 Rust 重写 API 服务器、请求校验、gRPC 渲染等前端组件，逐步替代部分 Python 服务层。
  - 为什么社区在关注：今天 20 个 commit 中有 5 个（#69856、#70001、#67945、#67700、#68933）都围绕 Rust 前端，说明 vLLM 正把“高性能、低延迟的请求入口”作为重点方向，同时也在补齐指标、错误恢复、gRPC 等生产级能力。
  - 对新手意味着什么：如果你会 Rust，或者想学 Rust 在 AI 基础设施中的应用，这是很好的切入点；即使只用 Python，也需要了解 Rust 前端和 Python 前端的行为差异（比如某些测试被跳过、某些字段不支持）。

- **关键词：KV Cache Offload（KV 缓存卸载）**
  - 一句话通俗解释：把 GPU 显存放不下的 KV 缓存搬到 CPU 内存、磁盘或远端，以支持更长上下文和更高并发。
  - 为什么社区在关注：今天有 3 个 KV Offload 相关 commit（#70063、#67952、#68660），包括回滚、重排队顺序修复、P2P 查找缓存修复，说明这个功能正在快速迭代、bug 较多，也说明长上下文推理是刚需。
  - 对新手意味着什么：如果你关注推理系统的内存管理、缓存调度，KV Offload 是很好的学习方向；但也要注意它还在“修 bug 阶段”，生产使用要谨慎。

- **关键词：量化与 MoE 内核适配（Quantization & MoE Kernels）**
  - 一句话通俗解释：把模型权重/激活压成 FP8、MXFP4、INT4 等低精度格式，并为 MoE（混合专家）模型选择合适的高性能内核。
  - 为什么社区在关注：今天有 MXFP4/SiTUv2（#69625）、AITER W4A16（#70009）、logits 原地计算（#69898）等，说明量化后端和 MoE 内核的兼容性、正确性、显存占用是持续热点。
  - 对新手意味着什么：学量化不能只看“精度掉了多少”，还要懂后端选择、硬件差异（ROCm vs CUDA）、显存峰值；这是推理优化的核心技能之一。

- **关键词：AMD ROCm / 异构计算**
  - 一句话通俗解释：vLLM 在 AMD MI300/MI355X 等 GPU 上的支持越来越完整，包括 CI 迁移到 DPX、AITER 内核、SiTUv2 激活模式。
  - 为什么社区在关注：今天有 #69625、#70009、#69952 三个 ROCm 相关 commit，说明 AMD 正在成为 vLLM 的一等公民，异构计算趋势明显。
  - 对新手意味着什么：如果你只有 AMD GPU，或者想进入国产/非 NVIDIA 加速卡生态，ROCm 方向机会不少；但要注意很多优化是硬件特定的。

- **关键词：健康检查与可观测性（Health & Observability）**
  - 一句话通俗解释：新增 `/health_generate` 就绪检查、`vllm:admission_rejections` 指标，让运维能判断“服务是否真的能生成”而不是“进程是否活着”。
  - 为什么社区在关注：#60505 和 #69856 都指向生产部署的痛点：K8s 里进程活着但 GPU 卡死、请求被拒绝却没人知道。
  - 对新手意味着什么：部署推理服务不只是 `vllm serve`，还要懂 liveness/readiness probe、指标监控；这是 MLOps 的基本功。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 卸载与 P2P 查找缓存**
  - **是什么**：KV Cache 是注意力机制里缓存的历史 Key/Value，长上下文时显存不够，就把它卸载到 CPU/磁盘/远端。P2P 卸载中，消费者会先“查找”某个 key 在不在生产者那里，再决定是否拉取。
  - **为什么重要**：如果查找结果（尤其是 MISS）在 fetch 之后被错误清掉，下一轮会重复探测，浪费一轮通信；更糟的是可能把“未消费的答案”丢掉，导致重复传输或死锁。今天 #68660 就是修这个。
  - **动手试试**：
    ```bash
    # 用 vLLM 的单元测试观察 P2P session 行为
    pytest -v -s tests/v1/kv_offload/tiering/p2p/test_sessions.py \
      -k "cached_miss_survives_fetch_and_load or probe_registered_before_fetch"
    ```
    如果想自己写一个小实验，可以读 `tests/v1/kv_offload/tiering/p2p/test_sessions.py` 里的 `_answer_lookup` 和 `_fetch_and_land` 辅助函数，理解“查找答案 → fetch → 下一轮查找”的状态机。
  - **预期结果**：测试通过，说明 MISS 答案在 fetch 后仍然保留，下一轮 `flush_pending_lookups()` 不会再发重复的 LookupMsg。

- **知识点 2：原地计算与显存峰值（in-place soft cap / scale）**
  - **是什么**：LLM 输出 logits 通常是一个 `[num_tokens, vocab_size]` 的大张量。如果做 soft cap 和 scale 时每次都新建张量，峰值显存会翻倍甚至更多。原地操作（`div_().tanh_().mul_()`）只保留一份。
  - **为什么重要**：在投机解码（speculative decoding）的每个 draft step 都要算 logits，vocab 262k 时一个 bf16 logits 就是 512MB 级别；多一份就可能触发 OOM 或挤占 KV cache。
  - **动手试试**：
    ```python
    import torch

    vocab, hidden, n = 262144, 16, 64
    logits = torch.randn(n, vocab, dtype=torch.bfloat16, device="cuda")
    soft_cap, scale = 30.0, 2.0

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    # 非原地：会多一份 logits
    out1 = torch.tanh(logits / soft_cap) * soft_cap * scale
    peak1 = torch.cuda.max_memory_allocated() - base

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    # 原地
    logits.div_(soft_cap).tanh_().mul_(soft_cap).mul_(scale)
    peak2 = torch.cuda.max_memory_allocated() - base

    print("non-inplace peak bytes:", peak1)
    print("inplace peak bytes:", peak2)
    ```
  - **预期结果**：原地版本的峰值显存明显更小（接近 0 额外分配），非原地版本接近一个 logits 张量的大小。对应 #69898 的测试断言 `peak < 1.5 * expected.nbytes`。

- **知识点 3：健康检查 vs 就绪检查（`/health` vs `/health_generate`）**
  - **是什么**：`/health` 只检查 API server 和 engine 进程活着；`/health_generate` 还会检查每个 engine rank 当前能不能真的跑一次生成（比如 GPU 是否卡在 collective、是否在 sleep）。
  - **为什么重要**：K8s 里如果 readiness probe 只用 `/health`，GPU 已经 hang 住的 pod 仍会被打流量，用户请求全部超时；如果 liveness probe 用 `/health_generate`，权重同步时 engine 在 sleep，会被误杀重启。
  - **动手试试**：
    ```bash
    # 启动一个最小模型服务
    vllm serve facebook/opt-125m --port 8000 &

    # 普通健康检查
    curl -s http://localhost:8000/health

    # 生成就绪检查（默认 20s 超时）
    curl -s "http://localhost:8000/health_generate?timeout_s=20" | python -m json.tool
    ```
    然后尝试在另一个终端发一个长请求占住 engine，再调用 `/health_generate` 观察返回。
  - **预期结果**：正常时 `/health_generate` 返回 200 和 `{"status": "ok", ...}`；engine 忙时可能仍返回 200（busy rank 不额外跑 GPU）；如果 engine sleep 或 probe 超时，返回 503 和 `reason` 字段（`sleeping` / `probe_timeout` 等）。

- **知识点 4：Rust 请求校验分层（serde → route → shared domain）**
  - **是什么**：今天 #68933 把 Rust 前端的请求校验统一成三层：`serde` 管 JSON 形状和类型；`vllm-server` 的 `validate.rs`/`convert.rs` 管协议级约束（如 `stream_options` 必须配 `stream`）；`vllm-text`/`vllm-chat` 管跨路由的领域规则（采样参数边界、stop 字符串数量等）。
  - **为什么重要**：如果校验散落在各处，同一个错误可能在不同路由返回不同的 `param` 或状态码，客户端很难处理；分层后“每个不变量只有一个 owner”，也方便 Rust 前端和 Python 前端行为对齐。
  - **动手试试**：
    ```bash
    # 跑 Rust 前端的校验相关测试（需要 Rust 工具链）
    cd rust
    cargo test -p vllm-chat --lib
    cargo test -p vllm-server --lib
    ```
    也可以读 `rust/AGENTS.md` 里新增的 “Request validation” 小节，对照 `rust/src/chat/src/backend/hf.rs` 的测试看 `ErrorKind::invalid_param("chat_template")` 怎么用。
  - **预期结果**：测试通过；你能说出一个请求字段（比如 `top_logprobs`）应该在哪一层校验、错误 `param` 应该叫什么。

- **知识点 5：MoE 量化后端选择与硬件差异（AITER MXFP4 / SiTUv2）**
  - **是什么**：MoE 模型的路由专家可以用不同精度跑：a4w4（4-bit 激活/4-bit 权重）、a8w4（FP8 激活）、a16w4（BF16 激活）。AMD 的 AITER 库在 gfx950（MI355X）上有 SiTUv2 FlyDSL 内核，环境变量 `VLLM_ROCM_USE_AITER_MOE_SITUV2` 选择激活精度。
  - **为什么重要**：默认 a4w4 的激活精度低于 checkpoint 训练时的 MXFP8，可能影响精度；不同硬件/构建版本支持的模式不同，设错环境变量可能静默走错路径甚至输出损坏。
  - **动手试试**：
    ```bash
    # 在 AMD ROCm 环境（gfx950）下启动 Kimi-K3 类模型时观察日志
    VLLM_ROCM_USE_AITER_MOE_SITUV2=a8w4 \
    vllm serve <model> --tensor-parallel-size 8 2>&1 | grep -i "SiTU"
    ```
    如果没有 AMD 硬件，可以读 `vllm/model_executor/layers/quantization/mxfp4.py` 里 `_use_k3_situ_aiter` 和 `__init__` 的日志分支，理解后端选择逻辑。
  - **预期结果**：日志出现类似 `Using AITER_MXFP4_BF16 for Kimi-K3 SiTU MXFP4 MoE, SiTUv2 activation mode a8w4` 的行；如果构建版本不支持，会看到回退或忽略环境变量的提示。

---

## 🔥 重要更新

- **#60505 新增 `/health_generate` 生成就绪检查**：这是生产部署的重要补丁，K8s 可以把 readiness probe 指向它，避免把流量打到 GPU 已 hang 的 pod；文档还明确提醒不要用它做 liveness probe（sleep 时会 503）。
- **#68933 Rust 前端请求校验大重构，移除 `validator` crate**：统一了校验分层和 `Error::param()`，影响所有 Rust 前端的错误返回行为，是 Rust 前端走向生产级的重要一步。
- **#70063 回滚 KV Offload 的 `on_schedule_end` 返回值改动**：说明之前 #68962 的改动引入了调度/生命周期问题，被紧急 revert；如果你在用 KV Offload 的 tiering，建议关注后续重新设计。
- **#69898 logits soft cap/scale 改为原地计算**：直接降低投机解码 draft step 的显存峰值，对长 vocab 模型（如 262k vocab）和 KV cache 紧张的场景有实际收益。

---

## 📋 逐条解读

### 1. [Rust Frontend] Count vllm:admission_rejections (#69856)

- **代码层面**：改的是 `rust/src/llm/src/admission.rs`。`Admission::new` 多接收一个 `model_name`，内部创建 `AdmissionRejectionCounters`；当请求因队列满（`QueueFull`）或 prefill backlog 满（`PrefillBacklogFull`）被拒绝时，分别给 `queued_reqs` / `queued_tokens` 计数器加一。测试里用 `AtomicU64` 给每个测试生成独立 model name，避免全局指标注册表冲突。同时 `.buildkite/test_areas/rust_frontend.yaml` 移除了对 `test_metrics_exist` 的跳过，因为 Rust 前端现在有 `vllm:admission_rejections` 指标了。
- **新手概念课堂**：想象餐厅门口有个取号机，队列满了就“拒绝取号”。以前拒绝就拒绝了，没人统计；现在门口挂了个计数器，记录“因为排队人数满拒绝了多少次”“因为后厨备菜量满拒绝了多少次”。指标（metrics）就是给运维看的计数器。
- **对你有什么影响**：如果你用 Prometheus 监控 vLLM，现在能看到 Rust 前端因 admission 控制被拒绝的请求数；如果指标突然上涨，说明队列限制太紧或流量过大。

### 2. [Bugfix][Rust Frontend] Accept "original" image detail (#70001)

- **代码层面**：改的是 `rust/src/server/src/routes/openai/chat_completions/convert.rs` 和 `responses/convert.rs`。OpenAI 请求里的 `image_url.detail` 现在用 `.map(Into::into)` 转换成 `llm_multimodal::ImageDetail`，并新增测试确认 `"original"` 被映射为 `High`、`null` 等同于省略。之前 Rust 前端可能直接拒绝或错误处理 `"original"`。
- **新手概念课堂**：图片的 `detail` 参数就像点外卖时选“小份/大份/原图”。OpenAI 协议里有 `low`、`high`、`auto`，有些客户端还会发 `original`。Rust 前端需要把这些“方言”翻译成内部统一的枚举。
- **对你有什么影响**：用 Rust 前端 + 多模态模型时，客户端发 `"detail": "original"` 不会再报错，而是按高分辨率处理。

### 3. [Security] Restrict inline request templates in Cohere serving (#69969)

- **代码层面**：改的是 Cohere 渲染相关代码和文档。当 `--tokenizer-mode cohere` 时，即使开了 `--trust-request-chat-template`，HTTP 请求里也不能再带 `chat_template` / `chat_template_kwargs.chat_template`，否则返回 400，`param` 为 `chat_template`。新增了 `tests/entrypoints/scale_out/render/test_render.py` 的端到端测试和 `tests/renderers/test_cohere.py` 的单元测试。
- **新手概念课堂**：聊天模板（chat template）是把对话消息拼成模型输入字符串的“菜谱”。允许请求自带模板，等于允许客户端往服务器里塞一段 Jinja 模板执行，有注入风险。Cohere 模式选择“只允许服务端配置的模板”，相当于餐厅只用自己的菜谱，不接受顾客自带。
- **对你有什么影响**：如果你用 Cohere tokenizer 模式并依赖请求级 `chat_template`，升级后会被拒绝；需要改用 `--chat-template` 在启动时配置。

### 4. [Core] Add a generation readiness check `/health_generate` (#60505)

- **代码层面**：新增 `vllm/entrypoints/serve/instrumentator/health.py` 中的 `/health_generate` 端点，支持 `timeout_s` 查询参数；返回 200 或 503 + JSON（`reason` 为 `engine_dead` / `sleeping` / `probe_timeout` / `probe_failed`，并带 `engine_ranks`）。文档更新了 `docs/serving/online_serving/README.md` 和 `docs/deployment/k8s.md`，给出 K8s probe 示例。CI 新增 `tests/entrypoints/serve/instrumentator/readiness/`。
- **新手概念课堂**：`/health` 像“店门开着吗”，`/health_generate` 像“厨师现在真的能炒菜吗”。K8s 的 liveness probe 决定要不要重启容器，readiness probe 决定要不要把流量导进来。用错 probe 会导致“进程活着但服务不可用”或“权重同步时被误杀”。
- **对你有什么影响**：部署到 K8s 时可以把 readiness probe 指向 `/health_generate?timeout_s=20`，并把 `timeoutSeconds` 设得比 20 大；liveness 继续用 `/health`。

### 5. [Fast Start] Add engram cpu offloading support for preload daemon (#69627)

- **代码层面**：改的是 preload daemon 和 weight cache 相关代码。非空的 pinned CPU storage 现在会被复制到 Linux memfd，通过 manifest 传递描述符；worker 端映射并注册 memfd，把 PLE 加速器视图绑定到导入的表。文档新增 “Qwen4Exp PLE offload” 小节，说明 `--engram-config '{"cpu_offload":true,"dp_shared_memory":false}'` 的用法和限制（仅支持 `ipc_cache` + 本地 GPU 共享后端，不支持 HiSparse 和 peer seeding）。
- **新手概念课堂**：preload daemon 像“提前把菜备好”的后厨；CPU offloading 是把一部分模型权重（这里是 PLE 表）放在 CPU 的 pinned 内存里，用 memfd 在进程间共享，避免重复拷贝。pinned 内存不会被操作系统换出，适合做 IPC。
- **对你有什么影响**：如果你用 `vllm preload` + `--load-format ipc_cache` 预加载 Qwen4Exp 类模型，现在可以开启 engram CPU offload，减少 GPU 显存占用；但配置必须 daemon 和 engine 一致，否则会 fallback 到磁盘加载。

### 6. [Bugfix][Multimodal] Honor Gemma 4 image resize settings (#67168)

- **代码层面**：改的是 Gemma 4 多模态处理器。现在默认使用 checkpoint 的 resize 设置；当请求传 `do_resize=False` 时，要求每个维度能被 pooled patch size（通常 48 像素）整除，且图像不超过 `max_soft_tokens`；未 resize 的视频帧固定限制 70 soft tokens。新增 `test_image_placeholder_count` 和 `test_invalid_unresized_image` 等测试，文档补充说明。
- **新手概念课堂**：多模态模型把图片切成 patch 再编码，patch 数量决定 soft token 数。如果用户说“别 resize”，图片尺寸就必须符合模型的切块规则，否则切不整齐。这就像拼图必须按格子切，不能随便裁。
- **对你有什么影响**：给 Gemma 4 发预 resize 的图片时，尺寸不对会收到明确的 `ValueError`（提示“divisible by 48”或“max_soft_tokens”），而不是静默出错。

### 7. [Perf] Decouple DeepSeek V3.2 attention sequence parallelism from ShardQ (#68074)

- **代码层面**：改的是 DeepSeek V3.2 模型代码和测试。把 `shardq_sequence_parallel` 拆成 `use_residual_sequence_parallel` 和 `residual_sp_min_tokens`（来自 `envs.VLLM_RESIDUAL_SP_MIN_TOKENS`），让 attention 的序列并行不再依赖 ShardQ 开关；测试参数化扩展到 `cudagraph_mode`、`use_shardq`、`kv_cache_dtype`、`residual_sp_min_tokens` 等组合，默认 `min_tokens` 从 64 改为 1024。
- **新手概念课堂**：序列并行是把一条长序列切给多张卡一起算 attention。以前这个优化和 ShardQ（一种量化/分片方案）绑在一起，想用序列并行就必须开 ShardQ；现在解耦了，可以独立开关。`min_tokens` 是“序列多长才值得切”的阈值。
- **对你有什么影响**：部署 DeepSeek V3.2 时，可以更灵活地组合序列并行和量化方案；默认阈值提高到 1024，短序列不会再被无谓地切分。

### 8. [ROCm][Kimi-K3] Log the SiTUv2 MoE activation mode and correct the recipe note (#69625)

- **代码层面**：改的是 `vllm/model_executor/layers/quantization/mxfp4.py` 和 Kimi-K3 recipe YAML。启动日志现在会打印 `rocm_aiter_ops.get_fused_moe_situv2_activation()` 返回的激活模式（a4w4/a8w4/a16w4）；recipe 文档重写了 AMD 段落，说明默认 a4w4、A/B 测试结果、环境变量语义和版本差异。
- **新手概念课堂**：SiTUv2 是 Kimi-K3 用的激活函数变体；AITER 是 AMD 的高性能算子库。a4w4 表示激活 4-bit、权重 4-bit。日志里打印模式，就像汽车仪表盘显示当前是“经济模式”还是“运动模式”，方便排查精度/性能问题。
- **对你有什么影响**：在 MI355X 上跑 Kimi-K3 时，能从日志确认实际走的激活精度；如果发现精度异常，可以对照 recipe 里的 A/B 数据判断是否是 a4w4 导致。

### 9. [Bugfix][KV Offload] Revert "Return pending work, events and stats from on_schedule_end (#68962)" (#70063)

- **代码层面**：回滚 #68962。`on_schedule_end` 不再返回 `StepResult`，`has_pending_push_work` 改回不依赖 step result、只取短锁检查 `manager.has_pending_work`；删除了 `test_cascade_after_the_last_store_keeps_engine_stepping` 等测试，恢复了 `_pending_finished_stores` 等结构。
- **新手概念课堂**：调度器和 KV offload manager 之间有一把“步骤锁”。之前改动让 `has_pending_push_work` 去读上一次 step 的结果，但如果 engine 空闲提前返回，锁可能永远不释放。回滚相当于“这个优化思路有死锁风险，先撤回”。
- **对你有什么影响**：如果你在用 KV Offload 的 tiering 功能，升级后行为回到旧版；之前依赖 `StepResult` 的代码需要跟着回退。

### 10. [Bugfix] Avoid importing ROCm platform in AITER W4A16 support check on non-ROCm (#70009)

- **代码层面**：改的是 `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py` 的 `is_supported_config`。先调用父类检查，不支持就直接返回；只有父类通过后才 `from vllm.platforms.rocm import on_gfx1250`，避免在非 ROCm 平台导入 ROCm 模块。
- **新手概念课堂**：这就像“先确认是不是 AMD 显卡，再加载 AMD 驱动”。以前不管什么平台都先 import ROCm 模块，在 Intel/CPU 环境可能报错或拖慢启动。
- **对你有什么影响**：非 ROCm 平台（如 Intel GPU、CPU）启动时不会再因为 AITER 检查而导入 ROCm 平台代码，减少不必要的依赖和潜在错误。

### 11. [Bugfix][Rust Frontend] Recover from per-peer ZMQ errors (#67945)

- **代码层面**：改的是 `rust/src/engine-core-client/src/coordinator/external.rs`。`socket.recv()` 遇到 `ZmqError::Codec` 或 `ZmqError::Network` 时打 `warn` 并 `continue`，而不是直接返回错误终止循环；其他错误仍然上抛。新增测试 `service_continues_after_peer_connection_error`，模拟一个 peer 重置连接后服务继续工作。
- **新手概念课堂**：ZMQ 是进程间通信库，外部 coordinator 通过 XSUB socket 发布状态。某个 peer 断线/发坏数据时，不应该让整个服务挂掉，而应该“忽略这个 peer，继续听别人说话”。
- **对你有什么影响**：Rust 前端在多 peer 场景下更健壮，单个 coordinator 连接抖动不会导致整个 engine-core-client 退出。

### 12. [Rust Frontend] Implement engine-free raw-JSON gRPC Render RPC (#65503) (#67700)

- **代码层面**：新增 `Render.Render` gRPC RPC，接受 `api = API_CHAT` 和 UTF-8 JSON `body`，返回 prompt token IDs、结构化输出约束、reasoning gate 和预处理媒体；`mm_payload` 枚举控制是否返回 inline tensors（`MM_PAYLOAD_TENSORS` / `MM_PAYLOAD_NONE`）。文档更新 `docs/serving/online_serving/renderer.md` 和 `rust/README.md`，明确“不返回 sampling/stopping 参数”的限制。
- **新手概念课堂**：Render 就是“只做模板渲染和 tokenize，不做推理”。gRPC 是比 HTTP 更高效的 RPC 协议，适合服务间调用。raw-JSON 意味着客户端不用构造强类型 protobuf 消息，直接塞 OpenAI JSON 就行。
- **对你有什么影响**：如果你在 Rust 前端做“先渲染、再决定怎么生成”的路由/调度，可以用这个 RPC；但要注意采样参数（temperature、max_tokens 等）不会从 render 响应里带回来，需要客户端自己保留。

### 13. [Doc][Misc] Make the --monokernel docstring model and hardware agnostic (#70070)

- **代码层面**：改的是 `vllm/config/kernel.py` 里 `KernelConfig.monokernel` 的 docstring，删除了具体模型/硬件列表（DeepSeek-V4.1、MI355X、Muse Spark 等），改成通用描述：“运行模型的 specialized decode（RFC #60904），等价于 `--additional-config '{"decode_backend": "latency_specialized"}'`，不支持的部署启动时拒绝”。
- **新手概念课堂**：monokernel 是把一个 decoder layer 的大部分计算塞进一个 persistent kernel 里，减少 kernel launch 开销。docstring 写死具体型号容易过时，改成“模型特定”更准确。
- **对你有什么影响**：看 `--monokernel` 帮助时不再被具体硬件列表误导；实际支持范围要看对应模型的文档。

### 14. [Bugfix][Mooncake] Reject missing draft context KV (#69870)

- **代码层面**：改的是 Mooncake connector。`MooncakeXferMetadata` 新增 `requires_draft_context_kv` 字段；worker 初始化时读取 `speculative_config.speculator_config.projects_context_kv`；`_prepare_transfer_regions` 如果发现对端需要 draft context KV 但本端没有配置同样的 drafter，就返回错误信息 “Missing draft context KV. Configure the same drafter on prefill.”。新增测试覆盖该分支。
- **新手概念课堂**：投机解码（speculative decoding）需要一个 draft 模型；如果 decode 端用了 draft 模型、prefill 端没配，KV 传输就会缺一块。这就像接力赛，前一棒没跑 draft 那段，后一棒接不上。
- **对你有什么影响**：使用 Mooncake + 投机解码时，prefill 和 decode 必须配置相同的 drafter，否则会收到明确错误而不是静默失败。

### 15. [Rust Frontend] Consolidate request validation, retire `validator`, and complete `Error::param()` (#68933)

- **代码层面**：大改 Rust 前端校验。移除 `validator` crate 依赖；在 `rust/AGENTS.md` 新增 “Request validation” 分层规范；`vllm_text::Error::kind()` 和 `vllm_chat::Error::kind()` 把所有错误变体穷举分类为 `ErrorKind::InvalidRequest { param }` 或 `ErrorKind::Internal`；HTTP handler 直接用 `Result<Json<T>, JsonRejection>`；测试从 `is_request_validation_error()` 迁移到 `kind()`。
- **新手概念课堂**：以前校验像“每个房间门口都贴一张规则”，容易重复、不一致；现在按“JSON 解析 → 路由协议 → 共享领域”三层分工，每个规则只有一个 owner。`Error::param()` 统一告诉客户端“是哪个字段错了”，方便前端高亮。
- **对你有什么影响**：Rust 前端返回的错误 `param` 更一致；如果你写客户端依赖 `param` 做字段级错误提示，行为会更可预测。Python 前端不受影响。

### 16. [Bugfix][KV Offload] Requeue CPU prefixes tail first in SimpleCPUOffload (#67952)

- **代码层面**：改的是 SimpleCPUOffload 调度逻辑。`_pending_finished_stores` 从 list 改成 dict（按 request 索引）；请求命中 CPU 前缀后，在完成时重新入队它需要的块（full attention 和最后一个 sliding window），并且排在“它运行期间其他请求存储的块”之后（tail first）。新增 Gemma3 混合注意力集成测试。
- **新手概念课堂**：CPU offload 的块要“回收再利用”。如果一个请求用了某些块，完成后这些块应该重新排队等待被淘汰或复用；顺序很重要——先排 tail（最近用的）能提高命中率。混合注意力模型（full + sliding window）需要特殊处理，因为 sliding window 只用最近一段。
- **对你有什么影响**：使用 SimpleCPUOffload + 混合注意力模型（如 Gemma3）时，前缀缓存命中后的块回收更正确，减少重复加载和缓存污染。

### 17. [Bugfix] Route DeepSeek ViT cu_seqlens through the OOT attention hook (#70044)

- **代码层面**：改的是 `vllm/models/deepseek_v4/common/vision.py`。把模块级的 `build_packed_vit_metadata` 改成 `DeepseekV4ViT` 的方法，并让 `cu_seqlens` 走 OOT（out-of-tree）attention hook，而不是直接传给 attention wrapper；`max_seqlen` 仍留在 host。新增 `numpy` 导入和 `cached` 参数控制 RoPE 表缓存。
- **新手概念课堂**：ViT 处理多张图时会把它们 pack 成一条序列，`cu_seqlens` 记录每张图的边界。OOT attention hook 是给外部注意力后端（如 FlashAttention 插件）的扩展点；把 `cu_seqlens` 走 hook，意味着外部后端可以自己处理这个元数据，而不是被写死。
- **对你有什么影响**：使用 DeepSeek V4 视觉模型 + 外部 attention 后端时，packed ViT 的元数据传递更兼容；CUDA graph capture 时的 dummy grids 不会污染 RoPE 缓存。

### 18. [Bugfix] Apply soft cap and scale in place in get_top_tokens (#69898)

- **代码层面**：改的是 `vllm/model_executor/layers/logits_processor.py` 的 `get_top_tokens`。`soft_cap` 从 `torch.tanh(logits / cap) * cap` 改为 `logits.div_(cap).tanh_().mul_(cap)`；`scale` 从 `logits = logits * scale` 改为 `logits *= scale`。新增测试 `test_get_top_tokens_holds_one_logits_sized_tensor`，断言峰值显存小于 1.5 倍 logits 大小。
- **新手概念课堂**：原地操作就是“在原来那张纸上改，不另拿一张纸”。logits 张量很大（vocab 262k × tokens），多一张就可能 OOM。`div_`、`tanh_`、`mul_` 带下划线的方法都是原地版本。
- **对你有什么影响**：投机解码 draft step 的显存峰值降低，长 vocab 模型在 KV cache 紧张时更不容易 OOM；数值结果与原来等价（测试用 gather 比较选中值）。

### 19. [ROCm][CI] Move MI300 Quantization, MoE, distributed, and batch invariance jobs to DPX (#69952)

- **代码层面**：改的是 `.buildkite/test-amd.yaml` 和 `.buildkite/test_areas/distributed.yaml`。把 MI300 的 Quantization、MoE Kernels、Distributed Comm Ops、Distributed Torchrun + Shutdown、Batch Invariance (Distributed) 等 job 的 `agent_pool` 从 `mi300_1/2/4` 改为 `mi300_dpx/mi300_dpx_2/mi300_dpx_4`，label 加上 “DPX”。
- **新手概念课堂**：CI agent pool 是跑测试的机器池；DPX 可能是新的 MI300 机器池/调度队列。迁移 job 是为了用新硬件或平衡负载。
- **对你有什么影响**：普通用户不直接感知；如果你贡献 ROCm 代码，CI 跑在 DPX 池上，排队时间和环境可能略有变化。

### 20. [Bugfix][KV Offload][P2P] Keep unconsumed lookup answers across a fetch (#68660)

- **代码层面**：改的是 P2P session 状态机。查找答案（尤其是 MISS）在 fetch 关闭当前 round 后不再被清掉；fetch 前注册但未 flush 的 key 会在下一 round 发送，其答案会被正确应用而不是当作 unknown 丢弃。新增 `test_cached_miss_survives_fetch_and_load` 和 `test_probe_registered_before_fetch_is_sent_in_next_round`。
- **新手概念课堂**：P2P 卸载里，消费者先问生产者“你有这些 key 吗”，生产者回答 HIT/MISS。一个请求可能分多轮 fetch；如果第一轮 fetch 结束时把还没用到的 MISS 答案扔了，下一轮又要重新问一遍，浪费一次往返。MISS 不需要生产者 pin 任何东西，所以保留它是安全的。
- **对你有什么影响**：使用 P2P KV offload 时，部分命中场景的查找往返次数减少，延迟更低；边界 key（fetch 前注册、下一轮才发）的答案不会丢。



## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：SWA（滑动窗口注意力）缓存治理**
  一句话通俗解释：模型只看最近一段 token 的注意力机制，它省显存，但缓存"什么时候能扔、扔多少"很容易出错。
  为什么社区在关注：今天有 3 个 commit（#52956、#52959、#53152）都在修 SWA 相关的缓存与存储逻辑，说明长上下文 + 混合注意力模型（如带滑动窗口的 draft 模型）已成主流，缓存管理是当前最密集的 bug 战场。
  对新手意味着什么：如果你在学 LLM 推理，KV cache 管理（尤其是 SWA、Mamba、HiCache 混合场景）是含金量极高的方向，真正能写出生产级代码的人不多。

- **关键词：CUDA Graph 与批大小上限（max_bs）**
  一句话通俗解释：把推理计算图"录下来"重放以省 CPU 开销，但图的大小必须提前定好。
  为什么社区在关注：#52709 把默认图上限改成按"每个 DP rank 的 running 上限"来定，说明多卡 DP 部署越来越普遍，默认参数必须跟得上。
  对新手意味着什么：性能调优不只是改 kernel，参数默认值的推导逻辑本身就是一门学问，值得读源码学习。

- **关键词：PD 分离与指标口径**
  一句话通俗解释：把"预填充（prefill）"和"解码（decode）"拆到不同机器上跑，负载均衡依赖准确的忙碌时间统计。
  为什么社区在关注：#53147、#51169 都在修指标/状态上报，说明大规模服务化后，"观测性"成为一等公民。
  对新手意味着什么：会写模型只是入门，会写"可观测、可调度"的服务才是工程能力。

- **关键词：量化 KV Cache（INT8 on Ascend）**
  一句话通俗解释：把 KV cache 用更低精度存，省显存换吞吐。
  为什么社区在关注：#49502 为昇腾 A2 NPU 新增 INT8 DSA MLA KV cache，说明国产硬件 + 稀疏注意力正在被认真适配。
  对新手意味着什么：异构硬件（NPU/AMD）适配是蓝海，懂 CUDA 之外再多懂一个平台很吃香。

- **关键词：共享内存 / 资源的生命周期管理**
  一句话通俗解释：多进程之间传大数据用共享内存，但忘了释放就会泄漏甚至崩溃。
  为什么社区在关注：#53064、#52705、#51519 都在修"请求被取消后资源没释放"的问题，说明高并发下的资源回收是稳定性关键。
  对新手意味着什么：写服务端代码，"异常路径"（失败、取消、超时）比正常路径更重要。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph 的批大小上限是怎么算出来的**
  - **是什么**：CUDA Graph 会为一批固定的 batch size 预先"录制"计算图，推理时直接重放。默认上限（max_bs）如果设小了，大 batch 就得回退到 eager 模式，慢很多。
  - **为什么重要**：在多卡数据并行（DP）下，每张卡实际只跑 `max_running_requests / dp_size` 个请求。如果还按全局上限设图，就会录一堆用不上的大图，浪费显存；设小了又跑不满。
  - **动手试试**：
    ```bash
    # 看这个 commit 的测试怎么验证的
    cd sglang
    python -m pytest test/registered/unit/server_args/test_server_args.py \
      -k "TestDecodeCudaGraphCeilingByMemoryTier" -v
    ```
    然后自己写个小脚本：
    ```python
    # 模拟 commit 里的新逻辑
    def resolve_max_bs(max_running_requests, attn_dp_size):
        return max(1, max_running_requests // attn_dp_size)
    print(resolve_max_bs(1536, 8))  # 期望 192
    print(resolve_max_bs(64, 1))    # 期望 64
    ```
  - **预期结果**：测试通过；打印出 192 和 64。你会理解"默认值 = 每 rank 能跑的上限"这个设计。

- **知识点 2：用 Triton 写 int8 量化 kernel 时，int32 偏移会溢出**
  - **是什么**：Triton kernel 里 `row_id * N` 算行偏移，如果张量元素数超过 2^31（约 21 亿），int32 会溢出变成负数，读到错误内存。
  - **为什么重要**：大模型 + 长序列很容易超过 2^31 元素。溢出不会报错，只会静默算错，是最难 debug 的一类 bug。
  - **动手试试**：
    ```python
    import torch
    # 复现 int32 溢出
    rows, cols = 83890, 25600
    print(rows * cols, rows * cols > 2**31)  # 2147584000 > 2147483648 -> True
    # 用 int32 算偏移会怎样
    import numpy as np
    idx = np.int32(rows - 1) * np.int32(cols)
    print("int32 溢出结果:", idx)   # 负数！
    print("int64 正确结果:", np.int64(rows - 1) * np.int64(cols))
    ```
  - **预期结果**：你会看到 int32 算出负数，int64 正确。再看 commit 里 `USE_INT64_OFFSETS=x.numel() >= 2**31` 就秒懂了——只在真正需要时才用 int64，不拖慢小张量。

- **知识点 3：共享内存（shared memory）的"所有权转移"与泄漏**
  - **是什么**：Linux 的 `/dev/shm` 是内存文件系统，多进程通过它零拷贝传数据。谁创建、谁释放必须明确，否则进程崩溃时资源泄漏。
  - **为什么重要**：commit #53064 修的就是"消息入队后所有权转移、分发失败时释放 SHM"。高并发下漏一个就是内存泄漏，最终 OOM。
  - **动手试试**：
    ```python
    from multiprocessing import shared_memory
    import numpy as np
    # 创建
    shm = shared_memory.SharedMemory(create=True, size=1024, name="demo_shm")
    arr = np.ndarray((256,), dtype=np.float32, buffer=shm.buf)
    arr[:] = 1.0
    print("写入完成, name =", shm.name)
    # 查看 /dev/shm 里确实多了个文件
    import os
    print(os.listdir("/dev/shm")[:5])
    # 正确清理
    shm.close()
    shm.unlink()   # 少了这步就泄漏！
    ```
  - **预期结果**：`/dev/shm` 下出现 `demo_shm`，`unlink()` 后消失。试试注释掉 `unlink()` 再跑，你会看到残留——这就是泄漏。

- **知识点 4：请求取消（abort）时的资源回收顺序**
  - **是什么**：用户中断生成、请求超时，服务器必须把该请求占用的 KV cache、多模态处理任务、共享内存全部释放。
  - **为什么重要**：commit #51519 修的是"abort 请求和 embedding 完成在同一个 tick 到达"的竞态——如果先处理完成再处理 abort，请求会"复活"进入调度器。顺序错了就是幽灵请求。
  - **动手试试**：
    ```bash
    # 跑这个 commit 新增的测试，看它如何构造"同一 tick"场景
    python -m pytest test/registered/unit/disaggregation/test_encode_receiver.py \
      -k "abort_in_the_completion_tick" -v
    ```
    读测试代码：
    ```python
    # 关键断言：abort 必须先于 tokenized 请求被处理
    new_recv_reqs, abort_reqs = receiver._process_waiting_requests([abort], None)
    # 期望 abort 单独进 abort_reqs，不混进 new_recv_reqs
    ```
  - **预期结果**：测试通过。你会理解"事件按到达顺序处理"这个原则，以及为什么不能在别处重复扫描。

- **知识点 5：LoRA 的计算精度与"无损"质量档位**
  - **是什么**：LoRA 是低秩适配器，有 fp32/fp16/bf16 存法。fp32 精度高但慢（吃不到 tensor core），fp16/bf16 快但可能溢出。
  - **为什么重要**：commit #52064 让 fp32 适配器在 `quality="high"` 时按激活的 16 位精度算（等价于同权重的 bf16 文件），但在 `exact/lossless` 时保留 fp32。这是"性能 vs 精度"的显式权衡。
  - **动手试试**：
    ```python
    import torch
    # 演示 fp32 在 bf16 激活下走 tensor core 的差异
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    w_fp32 = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    w_bf16 = w_fp32.to(torch.bfloat16)
    # fp32 权重 + bf16 激活 -> 会先 cast，慢
    y1 = x @ w_fp32.to(torch.bfloat16)   # 快
    y2 = (x.float() @ w_fp32).to(torch.bfloat16)  # 慢且可能溢出
    print("bf16 路径 finiteness:", torch.isfinite(y1).all().item())
    print("fp32 路径 finiteness:", torch.isfinite(y2).all().item())
    ```
  - **预期结果**：两条路径结果接近但速度差明显（用 `torch.cuda.synchronize()` + `time.time()` 测）。你会明白为什么"高质量档位"可以安全降精度。

---

## 🔥 重要更新

- **#52709 CUDA Graph 默认上限按每 rank 计算**：影响所有多卡 DP 部署，默认参数变了，大 batch 场景下可能不再回退 eager，性能提升明显。
- **#49502 昇腾 A2 支持 INT8 DSA MLA KV Cache**：国产 NPU 上跑 GLM-5.2 这类稀疏 MLA 模型，KV cache 显存直接减半（512 维 latent 从 1152 字节降到 656 字节），是硬件适配的重要里程碑。
- **#52959 流式会话的 SWA 及时回收**：长会话不再无限占用 SWA 池，并发会话不再因池耗尽而卡死，直接影响线上稳定性。
- **#53064 多模态共享内存泄漏修复**：分发失败时正确释放 SHM，避免高并发多模态请求下的内存泄漏。

---

## 📋 逐条解读

### 1. [Fix] hicache: store the SWA window a storage query finds missing in buffer mode (#52956)
- **代码层面**：改 `unified_radix_cache.py` 和 `storage_attachment.py`。新增 `_heals_storage_swa_gap` 标志（buffer 模式 + SWA + 非 Mamba + 单窗口时置位），并新增 `_storage_swa_gap_end_by_reqid` 字典记录"存储里 Full KV 结束的位置"。匹配结果出来后调用 `_branch_at_storage_swa_gap` 在缺口处分支，请求结束时清理记录。
- **新手概念课堂**：想象你在图书馆借书，Full KV 是"全套书"，SWA 是"最近 N 本"。存储层可能只存了全套书但缺了最近的窗口——就像借书单上写着"有全套"但最近几本被借走了。这个 commit 就是让系统在缺口处"分叉"补上，而不是假装全都命中。
- **对你有什么影响**：用 HiCache + 滑动窗口模型的用户，长前缀命中率更准，不会因为"存储说有其实没"而算出错误结果。

### 2. [Fix] cuda-graph: Size the default decode graph ceiling to the per-rank running limit (#52709)
- **代码层面**：改 `memory_hook.py` 里 `handle_gpu_memory_settings`。原来只在 `max_bs > max_running_requests` 时下调；现在只要用户没显式指定，就直接设成 `max(1, max_running_requests // attn_dp_size)`。测试新增 `test_default_ceiling_is_per_rank_running_requests`。
- **新手概念课堂**：10 个人分 2 个窗口买票，每个窗口只需准备 5 份表格，不用准备 10 份。DP 就是"窗口"，图上限就是"表格数"。
- **对你有什么影响**：多卡 DP 部署时默认配置更合理，减少显存浪费，大 batch 不再意外退回 eager 模式。

### 3. [Fix] tokenizer: Release multimodal SHM on failed dispatch and transfer ownership at enqueue (#53064)
- **代码层面**：改 `io_struct.py` 的 `sock_send`（返回 enqueue 完成句柄）、`mm_utils.py` 的共享内存创建（新增 `_create_posix_segment`，用 `resource_tracker` 注册/注销），以及分发失败路径的释放逻辑。
- **新手概念课堂**：快递员把包裹交给驿站（enqueue），交接完成前包裹归快递员管，交接后归驿站管。如果驿站拒收，快递员必须自己处理掉包裹，不能扔在门口。
- **对你有什么影响**：多模态请求失败/取消时不再泄漏共享内存，长时间运行的服务更稳定。

### 4. [Fix] scheduler: Count all non-idle time in the PD prefill busy counter (#53147)
- **代码层面**：改 `scheduler.py` 和 `role_switch.py`。新增 `_get_total_prefill_busy_us`（PD prefill 模式下读 `prefill_busy_clock`）和 `_carry_prefill_busy_us`（角色切换时延续计时）。PD prefill 模式下把 `_record_step_counters` 提前到快照前调用。
- **新手概念课堂**：厨师（prefill 引擎）的"忙碌时间"应该算所有不空闲的时间，而不是只算炒菜那几下。角色切换（从 prefill 变 decode）时，忙碌计时器要交接给新角色继续走。
- **对你有什么影响**：PD 分离部署的负载均衡更准，prefill 节点不会被低估忙碌度而被打爆。

### 5. [Feat] mem_cache: Build the Rust tree core from an empty index tensor instead of a device string (#53152)
- **代码层面**：改 `adapter.py` 和 Rust 的 `python_bindings.rs`。删掉 `parse_device` 函数和 `device: String` 字段，改成传 `empty_device_indices: Py<PyAny>`（一个空的 int64 tensor），Rust 侧从 tensor 推断设备。
- **新手概念课堂**：以前告诉 Rust "我在 cuda:1 上"，现在直接递一个"在 cuda:1 上的空盒子"，Rust 一看盒子就知道在哪。更不容易因字符串解析出错。
- **对你有什么影响**：Rust 树核心的设备处理更健壮，多卡场景下不会因设备字符串格式问题崩溃。

### 6. [Fix] scheduler: Retire sliding-window KV behind the window when a streaming session extends its own row (#52959)
- **代码层面**：改 `schedule_batch.py` 的 `maybe_evict_swa`。新增分支：extend 模式下，若请求持有 KV 且属于 streaming session，就调用 `_evict_swa` 回收窗口外的行；overlap 模式下 pre_len 要减去一个 chunk 大小（因为前一个 extend 可能还在飞行中）。
- **新手概念课堂**：录音机只保留最近 10 秒（窗口），但流式会话会一直往同一盘磁带上录。如果不主动擦掉 10 秒前的，磁带很快就满了。这个 commit 就是让录音机边录边擦。
- **对你有什么影响**：长流式会话不再耗尽 SWA 池，并发会话不会因池满而卡住。

### 7. [Feat] rust-server: Cancel a request's multimodal processing when it is aborted (#52705)
- **代码层面**：改 `message/request.rs`，新增 `MmCancellation` 结构（Arc<Mutex<CancellationState>>），提供 `cancel()`、`is_cancelled()`、`on_cancel()` 回调注册。`MmData` 增加 `cancellation` 字段。
- **新手概念课堂**：你在餐厅点了菜，中途说"不要了"，厨房要立刻停手，而不是做完再倒掉。`on_cancel` 就是"停手信号"，注册的回调会立即执行。
- **对你有什么影响**：Rust server 上取消多模态请求时，图像/视频处理会立刻停止，省 CPU 和显存。

### 8. [Fix] disaggregation: Use row_width in decode Mamba pool tests (#53296)
- **代码层面**：测试文件里把 `max_context_len=4` 改成 `row_width=4`，`max_context_len=16` 改成 `row_width=16`。只是参数名对齐。
- **新手概念课堂**：接口改名了，测试得跟着改。`max_context_len` 太含糊，`row_width` 明确表示"每行的宽度"。
- **对你有什么影响**：无直接影响，但说明代码库在清理命名，可读性提升。

### 9. [Feat] [NPU] dsa: Store the DSA MLA KV cache as INT8 on Ascend A2 (#49502)
- **代码层面**：新增 `int8` KV cache dtype，仅支持昇腾 A2 + DSA 模型（如 GLM-5.2）。每个 token 的 MLA KV 存成打包记录（INT8 latent + BF16 rope + FP32 scales），512 维 latent 每层 656 字节（BF16 是 1152 字节）。文档新增大量限制说明（拒绝 arch35、非 DSA 模型、dcp-size>1 等）。
- **新手概念课堂**：把一本精装书（BF16）压成平装书（INT8），页数不变但更薄。但只有特定书架（A2 NPU）和特定书（DSA 模型）支持这种装订。
- **对你有什么影响**：昇腾 A2 上跑 GLM-5.2 等稀疏 MLA 模型，KV cache 显存减半，能支持更长上下文或更大并发。

### 10. [Doc] blogs: Sync LMSYS SGLang blog cards (#49415)
- **代码层面**：改 `docs/index.mdx`，更新首页博客卡片的链接、图片、标题、日期。
- **新手概念课堂**：网站首页的"最新文章"轮播图更新了。
- **对你有什么影响**：无功能影响，但可以点进去看新博客（Miles 真 on-policy RL、ERNIE 5.5 Flash 低延迟等）。

### 11. [Fix] scheduler/pp: Bound dynamic chunk profiling by free KV capacity (#51481)
- **代码层面**：改 `dynamic_chunk_sizer.py`，`max_profile_tokens` 取 `min(...)` 时新增 `token_to_kv_pool_allocator.available_size()` 一项。测试新增 `kv_capacity`、`swa_capacity` 参数验证。
- **新手概念课堂**：动态分块预填充要先"试跑"测量延迟，试跑的块不能超过可用显存，否则会 OOM。以前只看上下文长度和映射行宽，忘了看实际剩余 KV 容量。
- **对你有什么影响**：PP（流水线并行）+ 动态分块场景下，profiling 阶段不再因显存不足而崩溃。

### 12. [Fix] test: Add missing moe runtime context to CPU graph fixture (#53259)
- **代码层面**：测试 `test_cpu_graph.py` 的 fixture 里给 `exec_ctx` 补上 `moe=SimpleNamespace(enable_paged_experts=False)`。
- **新手概念课堂**：测试用的假对象少了一个字段，代码访问它时 AttributeError。补上就好。
- **对你有什么影响**：无，纯测试修复。

### 13. [Fix] encoder: Abort waiting requests before draining embeddings in the same tick (#51519)
- **代码层面**：改 `encoder/receiver.py` 的 `_process_waiting_requests`，遇到 `AbortReq` 时先调 `abort_waiting_requests` 再继续；同时删掉 `scheduler.py` 里 `abort_request` 中重复的 `mm_receiver.abort_waiting_requests` 调用。
- **新手概念课堂**：同一时刻收到"取消订单"和"订单完成"两个消息，必须先处理取消，否则订单会"复活"。而且不能在两处都检查，否则会误伤后来的请求。
- **对你有什么影响**：EPD（编码器分离）部署下，取消请求不会幽灵般进入调度器，资源不浪费。

### 14. [Fix] scheduler: Emit scheduler status while idle (#51169)
- **代码层面**：改 `metrics_reporter.py` 的 `_maybe_log_idle_metrics`，空闲时也调 `scheduler_status_logger.maybe_dump`；改 `scheduler_status_logger.py` 增加 `_last_dump_was_idle` 去重（interval=0 时连续空闲只打一次）。
- **新手概念课堂**：以前只有忙的时候才打状态日志，空闲时看不到"我在待命"。现在空闲也报，但连续空闲不刷屏。
- **对你有什么影响**：监控面板上能看到空闲状态，排查"为什么没请求"时更方便。

### 15. [Doc] cookbook: Pin the Qwen3.5 H200 AgentX lanes of InferenceX #5236 to its measured run (#53280)
- **代码层面**：改 `qwen3.5.jsx`，把 `status: "in-progress"` 改成 `"verified-pr"`，补上 run URL、sha、comment ID。
- **新手概念课堂**：性能基准数据从"进行中"变成"已验证"，可以放心引用了。
- **对你有什么影响**：查 Qwen3.5 在 H200 上的 AgentX 性能数据时，能看到可复现的验证记录。

### 16. [Fix] diffusion/krea2: Prepare one latent row per requested output (#52119)
- **代码层面**：改 `krea2.py`，新增 `_output_seeds` 函数按 prompt-major 顺序生成每个输出行的种子；`num_outputs_per_prompt > 1` 时对 embedding 做 `repeat_interleave`；噪声生成按 `seeds` 逐行建 generator。
- **新手概念课堂**：一次请求生成 4 张图，以前可能只准备了 1 行 latent，现在按 4 行准备，每行用不同的种子，保证 4 张图各不相同。
- **对你有什么影响**：Krea2 模型一次生成多张图时，每张图都正确且可复现。

### 17. [Perf] diffusion: Resolve the default output quality with the request's data type (#52601)
- **代码层面**：改 `sampling_params.py` 的 `from_user_sampling_params_args`，`_adjust` 之后若发现是 IMAGE 类型且用户没显式指定 `output_compression`，就用 IMAGE 的默认质量（75）重新算，而不是沿用 VIDEO 的默认（50）。
- **新手概念课堂**：图片和视频的"默认压缩质量"不一样，以前图片误用了视频的默认值，导致 PNG 走了慢速压缩路径。现在按数据类型分别处理。
- **对你有什么影响**：生成图片时默认质量更合理，PNG 编码更快。

### 18. [Perf] diffusion/lora: Compute fp32 dynamic adapters in the activations' dtype (#52064)
- **代码层面**：改 `lora/linear.py` 的 `_lora_compute_dtype`，fp32 适配器在 `quality="high"` 且激活是 16 位时，按激活 dtype 算（等价于同权重 bf16 文件）；`exact/lossless` 仍用 fp32。改 `forward_context.py` 新增通用 `request_allows(tier)`。
- **新手概念课堂**：同一套 LoRA 权重有 fp32 和 bf16 两个文件。在高画质档位下，fp32 文件可以按 bf16 算（更快，结果几乎一样）；在无损档位下必须用 fp32（慢但精确）。
- **对你有什么影响**：用 Lightning LoRA 等 fp32 适配器时，高画质档位生成速度提升，画质无明显损失。

### 19. [Fix] moe: Honor swiglu_limit in FlashInfer TRT-LLM BF16 (#52257)
- **代码层面**：改 `flashinfer_trtllm.py`，`FlashInferTrtllmBf16MoeQuantInfo` 新增 `gemm1_clamp_limit` 字段并传给 kernel；改 `unquant.py`，在 `process_weights_after_loading` 里把 `swiglu_limit` 注册成 buffer（`persistent=False`，`_skip_weight_check=True`），forward 时读取。
- **新手概念课堂**：SwiGLU 激活值可能爆掉，`swiglu_limit` 是"上限夹子"。以前 FlashInfer TRT-LLM BF16 路径忘了传这个夹子，导致数值溢出。
- **对你有什么影响**：用 FlashInfer TRT-LLM BF16 MoE 且配置了 `swiglu_limit` 的模型，输出不再溢出/NaN。

### 20. [Fix] kernels/quantization: Use int64 row offsets in the per-token int8 quant kernel (#51196)
- **代码层面**：改 `int8_kernel.py`，Triton kernel 新增 `USE_INT64_OFFSETS` constexpr，`x.numel() >= 2**31` 时把 `row_id` 转成 int64。测试新增 `TestInt8QuantLargeInputOffsets`，用 83890×25600 的大张量（>2^31 元素）验证尾部行与"孪生行"量化结果一致。
- **新手概念课堂**：21 亿个元素用 int32 算偏移会绕回负数，读到错误内存。就像门牌号超过 9999 后从 0000 重新数，找错门。这个 commit 只在真需要时用 int64，避免拖慢小张量。
- **对你有什么影响**：超大 batch 或超长序列的 int8 量化不再静默算错，MoE 推理结果正确。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)