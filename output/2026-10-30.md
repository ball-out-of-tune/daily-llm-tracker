# 🚀 vLLM & SGLang 每日更新 — 2026-10-30

> 自动生成于 2026-10-30 07:42 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：Rust 前端（Rust Frontend）**
  - 一句话解释：vLLM 正在把原来用 Python 写的 API 前端（处理聊天模板、请求解析等）逐步迁移到 Rust，以获得更低延迟和更高并发。
  - 为什么社区在关注：今天有 3 个 commit（#64274、#62424、#64557）直接围绕 Rust 前端展开——新增 GLM-5.3 原生渲染、拒绝不支持的模板参数、修复 benchmark 计数。这说明 Rust 前端已经从“实验”走向“主力”，并且开始承担具体模型的适配工作。
  - 对新手意味着什么：如果你在学 Rust，又对 AI 基础设施感兴趣，这是一个非常好的切入点——vLLM 的 Rust 部分代码量还不大，但涉及的都是高价值的请求处理路径，容易做出有影响力的贡献。

- **关键词：异构硬件适配（XPU / ROCm / gfx950）**
  - 一句话解释：vLLM 不再只服务 NVIDIA GPU，Intel XPU、AMD ROCm（CDNA3/CDNA4）都在快速补齐功能。
  - 为什么社区在关注：今天有 5 个 commit 涉及 XPU 或 ROCm（#64285、#64065、#60867、#64567、#64610、#63885），包括 CI 测试启用、稀疏 MLA 后端修复、AITER GEMM 溢出保护、MoRIIO KV 连接器增强。
  - 对新手意味着什么：如果你手上有 AMD 显卡或 Intel Arc GPU，现在正是参与开源的好时机——很多功能刚补齐，文档和测试都还在完善，你能找到大量“good first issue”。

- **关键词：投机解码 + 结构化输出（Spec Decode + Structured Output）**
  - 一句话解释：投机解码（用一个小模型“猜”后续 token，大模型验证）正在与结构化输出（JSON Schema 约束）在流水线并行（PP）场景下做兼容。
  - 为什么社区在关注：#63219 修复了 PP 下每个请求的 draft 保留问题，说明投机解码正在从“单卡玩具”走向“生产级分布式部署”。
  - 对新手意味着什么：这是推理加速最前沿的方向之一，理解“draft model + 验证 + 回滚”的机制，是进入 LLM 推理优化领域的必修课。

- **关键词：MoE（混合专家）性能调优**
  - 一句话解释：MoE 模型（如 DeepSeek、GLM、Kimi）每次只激活部分专家，但调度和内存布局极其复杂。
  - 为什么社区在关注：#64467 把 `prepare_megamoe_inputs` 的 tile 大小从 8 行改成 32 行（1024 token 以上），带来 1.3-1.6x 加速；#61971 让 FlashMLA 直接传 int64 位置。这些都是实打实的性能收益。
  - 对新手意味着什么：MoE 是当前大模型的主流架构，理解它的 kernel 级优化（tile、block、padding）是成为性能工程师的关键。

- **关键词：Diffusion LLM（扩散语言模型）**
  - 一句话解释：用扩散模型的方式生成文本（不是逐 token 自回归，而是并行去噪），DiffusionGemma 是代表。
  - 为什么社区在关注：#51953 修复了 DiffusionGemma 的运行时 OOM，通过分块 logits 投影让显存随 tile 而非 batch 增长。这说明扩散 LLM 正在进入 vLLM 的正式支持范围。
  - 对新手意味着什么：这是一个新兴方向，相关代码和论文都不多，早进入容易建立认知优势。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：HTTP 请求重试与不完整响应体**
  - **是什么**：当服务器返回 `Content-Length: 1000` 但只发了 500 字节就断开连接时，客户端应该识别这是“传输失败”并重试，而不是把半截数据当成功。
  - **为什么重要**：在分布式推理集群里，网络抖动、节点重启都会导致响应体截断。如果不重试，用户会拿到损坏的图片/音频数据，或者莫名其妙的解析错误。vLLM 的 `connections.py` 是下载模型权重、媒体文件的核心路径，这里出 bug 影响面极大。
  - **动手试试**：
    ```python
    # 用 Python 标准库模拟一个"发一半就断"的服务器
    import socket, threading, requests

    def bad_server():
        s = socket.socket(); s.bind(("127.0.0.1", 8765)); s.listen(1)
        conn, _ = s.accept()
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n" + b"x" * 50)
        conn.close()  # 只发 50 字节就断

    threading.Thread(target=bad_server, daemon=True).start()
    try:
        r = requests.get("http://127.0.0.1:8765/", timeout=2)
        print("status:", r.status_code, "len:", len(r.content))
    except Exception as e:
        print("caught:", type(e).__name__, e)
    ```
  - **预期结果**：你会看到 `requests.exceptions.ChunkedEncodingError` 或 `ConnectionError`。vLLM 的修复就是把这个异常加入 `_is_retryable` 列表，让 `tenacity` 自动重试。理解这一点，你就掌握了“可重试错误分类”的设计模式。

- **知识点 2：Triton kernel 的 tile 大小调优（block_m）**
  - **是什么**：Triton 写 GPU kernel 时，每个 program 处理多少行数据（`block_m`）会显著影响性能。太小则启动开销大，太大则寄存器溢出/占用率高。
  - **为什么重要**：vLLM 的 `prepare_megamoe_inputs` 是 MoE 推理的热点路径。Commit #64467 发现：64-1023 token 用 8 行 tile 最优，1024+ token 用 32 行 tile 能快 1.3-1.6x。这种“按规模切换策略”的工程直觉，是性能优化的核心。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    for n in [64, 128, 1024, 4096]:
        x = torch.randn(n, device="cuda"); y = torch.empty_like(x)
        for BLOCK in [8, 32, 128]:
            grid = (triton.cdiv(n, BLOCK),)
            # 用 triton.testing.do_bench 测量
            ms = triton.testing.do_bench(lambda: copy_kernel[grid](x, y, n, BLOCK))
            print(f"n={n:5d} BLOCK={BLOCK:3d} -> {ms*1000:.2f} us")
    ```
  - **预期结果**：你会看到小 n 时 BLOCK 太大反而慢（启动开销占比高），大 n 时 BLOCK 大更快。这就是 commit 里 `block_m = 32 if num_tokens >= 1024 else 8 if num_tokens >= 64 else 1` 的实验依据。

- **知识点 3：int32 vs int64 的溢出陷阱**
  - **是什么**：GPU kernel 用 32 位整数索引时，最大只能表示约 21 亿（2^31）。当矩阵乘法的 `M*N >= 2^31` 时，索引会溢出变成负数，导致内存越界或结果错误。
  - **为什么重要**：Commit #64567 在 ROCm gfx950 上发现 AITER 的 tuned GEMM 用 int32 寻址，大矩阵会崩。Commit #61971 则反过来——把位置编码从 int32 升级到 int64，以支持更长上下文。这两个 commit 展示了同一个问题的两面：**该用 int64 的地方必须用，不该用的地方要注意溢出**。
  - **动手试试**：
    ```python
    import numpy as np
    # 模拟 int32 溢出
    m, n = 50000, 50000
    idx = np.int32(m) * np.int32(n)
    print("int32 M*N =", idx)  # 溢出！
    print("int64 M*N =", np.int64(m) * np.int64(n))
    # 正确判断
    print("unsafe:", m * n >= 1 << 31)
    ```
  - **预期结果**：你会看到 int32 结果是一个负数或错误值，而 int64 正确。vLLM 的修复就是加一个 `aiter_hgemm_unsafe = n * m >= 1 << 31 or ...` 的前置检查，不安全就回退到普通 GEMM。

- **知识点 4：投机解码中的“draft 保留”与流水线并行**
  - **是什么**：投机解码时，draft model 生成 K 个候选 token，target model 一次性验证。流水线并行（PP）下，不同 stage 处理不同请求的请求，需要按 request id 保留各自的 draft token，否则会串味。
  - **为什么重要**：Commit #63219 修复了 PP + 结构化输出场景下的 bug——之前用全局变量存 draft，PP 多请求并发时会互相覆盖，导致 JSON 输出错乱。修复方案是“per-request draft retention”，即每个请求单独保存自己的 draft。
  - **动手试试**：
    ```python
    # 模拟有 bug 的全局 draft vs 正确的 per-request draft
    class BadSpecDecode:
        draft = []  # 全局共享，bug！
        def add(self, req_id, tokens): self.draft = tokens
        def verify(self, req_id): return self.draft

    class GoodSpecDecode:
        def __init__(self): self.drafts = {}
        def add(self, req_id, tokens): self.drafts[req_id] = tokens
        def verify(self, req_id): return self.drafts[req_id]

    bad = BadSpecDecode(); bad.add("A", [1,2]); bad.add("B", [3,4])
    print("bad A:", bad.verify("A"))  # 拿到了 B 的 draft！
    good = GoodSpecDecode(); good.add("A", [1,2]); good.add("B", [3,4])
    print("good A:", good.verify("A"))  # 正确
    ```
  - **预期结果**：bad 版本 A 请求拿到 `[3,4]`，good 版本拿到 `[1,2]`。这就是“状态隔离”原则——并发场景下任何共享可变状态都是 bug 温床。

- **知识点 5：分块计算避免 OOM（tiled logits projection）**
  - **是什么**：当 batch 很大、vocab 很大时，一次性计算 `[batch, hidden] @ [hidden, vocab]` 的 logits 矩阵会瞬间吃掉几十 GB 显存。分块（tile）计算就是每次只算一小批，算完就释放。
  - **为什么重要**：Commit #51953 修复 DiffusionGemma 的 OOM，核心思路是“峰值显存随 tile 大小增长，而非随 batch 增长”。这是所有大模型推理框架的通用技巧——vLLM 的 PagedAttention、chunked prefill 都是类似思想。
  - **动手试试**：
    ```python
    import torch
    hidden, vocab, batch = 4096, 128000, 64
    W = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)

    # 方式 1：一次性算，容易 OOM
    try:
        h = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
        logits = h @ W.t()  # [64, 128000] bf16 ≈ 16MB，但中间可能更大
        print("one-shot ok, shape:", logits.shape)
    except torch.cuda.OutOfMemoryError:
        print("one-shot OOM")

    # 方式 2：分块算
    def tiled_logits(h, W, tile=8):
        outs = []
        for i in range(0, h.shape[0], tile):
            outs.append(h[i:i+tile] @ W.t())
        return torch.cat(outs, dim=0)
    h = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
    print("tiled ok, shape:", tiled_logits(h, W).shape)
    ```
  - **预期结果**：小 batch 时两者都 OK；把 `batch` 调到 512 或 `vocab` 调到 256000，one-shot 可能 OOM，tiled 依然稳定。用 `torch.cuda.max_memory_allocated()` 对比峰值显存会更直观。

---

## 🔥 重要更新

1. **#62424 [Rust Frontend] 新增 GLM-5.3 / GLM-5.3-Flash 原生聊天渲染**：这是 Rust 前端第一次完整支持一个国产大模型的复杂模板（含工具调用、推理选项），标志着 Rust 前端从“基础设施”进入“模型适配”阶段。GLM-5.3-Flash 的 recipe 也同步把默认前端切到 Rust。

2. **#63219 [Bugfix][Spec Decode] PP 下结构化输出的 per-request draft 保留**：修复了流水线并行 + 投机解码 + JSON 约束三者叠加时的严重正确性 bug，是生产级部署的关键修复。

3. **#64467 [Perf][MoE] 1024 token 以上用 32 行 tile**：GB200 上实测 1.3-1.6x 加速，MoE 推理的免费性能午餐，影响所有使用 MegaMoE 的模型（DeepSeek、GLM 等）。

4. **#64065 [Bugfix][XPU] 补齐稀疏 MLA 的 `record_logical_topk_ready` 钩子**：修复了 DeepSeek-V3.2 / GLM-5 在 Intel XPU 上首次 forward 就崩溃的问题，同时新增了专门的 CI job，说明 XPU 支持正在从“能跑”走向“稳定”。

---

## 📋 逐条解读

### 1. [Bugfix] Retry incomplete HTTP response bodies (#62674)
- **代码层面**：修改 `vllm/connections.py` 中的 `_is_retryable` 函数，把 `aiohttp.ClientPayloadError` 和 `requests.exceptions.ChunkedEncodingError` 加入“可重试”列表。同步在 `tests/test_connections.py` 新增测试，模拟服务器只发一半 body 就断开，验证 sync/async 两种客户端都会重试。
- **新手概念课堂**：这就像你点外卖，骑手说“到了”但只送了一半的菜。之前 vLLM 会把半份菜当完整订单收下（数据损坏）；现在它会识别“这单没送完”，重新下单（重试）。
- **对你有什么影响**：下载模型权重、加载远程图片/音频时，遇到网络抖动不再静默失败，而是自动重试，稳定性提升。

### 2. [Perf][DSv4.1] Bump FlashMLA to pass int64 positions to MegaAttn (#61971)
- **代码层面**：升级 FlashMLA 子模块的 git tag，删除 `flash_mla_mega_attn.py` 里所有 `positions.to(torch.int32)` 的转换，直接传 int64 的 `positions`。benchmark 脚本同步更新。
- **新手概念课堂**：位置编码（position）是告诉模型“这是第几个 token”。int32 最大约 21 亿，看起来够用，但 FlashMLA 内部某些计算会放大这个值导致溢出。int64 就像把计数器从“32 位”换成“64 位”，能表示天文数字。
- **对你有什么影响**：超长上下文（百万 token 级）场景下，位置计算不再有溢出风险，DeepSeek-V4.1 的注意力计算更稳。

### 3. [Perf][MoE] Use 32-row tiles in prepare_megamoe_inputs from 1024 tokens (#64467)
- **代码层面**：`prepare_megamoe.py` 里一行三元表达式从 `block_m = 8 if num_tokens >= 64 else 1` 改成 `block_m = 32 if num_tokens >= 1024 else 8 if num_tokens >= 64 else 1`。测试新增 `1025` 这个边界值。
- **新手概念课堂**：Triton kernel 的 `block_m` 就像搬砖时一次搬几块。小批量时搬太多反而累（启动开销），大批量时搬太少效率低（来回跑）。这个 commit 就是找到了“大批量时一次搬 32 块”的甜点。
- **对你有什么影响**：MoE 模型（DeepSeek、GLM、Kimi 等）在 batch 较大时推理速度提升 1.3-1.6x，无需改任何配置。

### 4. [XPU][CI] Enable streaming input e2e test on XPU (#64285)
- **代码层面**：`.buildkite/intel_jobs/engine_intel.yaml` 新增 `vllm/v1/engine/` 和 `test_streaming_input.py` 作为触发依赖，并追加 pytest 命令。测试文件里的 skip 条件从 `is_cuda()` 改成 `is_cuda() or is_xpu()`。
- **新手概念课堂**：CI（持续集成）就像工厂的质检线。之前 XPU 的流式输入功能没有质检，坏了也不知道；现在加上了质检工位。
- **对你有什么影响**：Intel GPU 用户使用流式输入（streaming input）时，功能回归会更早被发现，不再“发布后才知道坏了”。

### 5. [Bugfix][XPU] Add record_logical_topk_ready to XPUMLASparseImpl (#64065)
- **代码层面**：在 `xpu_mla_sparse.py` 的 `XPUMLASparseImpl` 类中新增 `record_logical_topk_ready()` 空方法（no-op）。新增 CI job 和两个测试文件，其中一个测试专门断言“这个钩子必须存在且可调用”。
- **新手概念课堂**：`MLAAttention.forward` 会无条件调用 `impl.record_logical_topk_ready()`，就像老师点名。XPU 的实现之前“没到”（没定义这个方法），一点名就报错。现在补上一个“到！”（空实现），点名通过。
- **对你有什么影响**：DeepSeek-V3.2 / GLM-5 在 Intel GPU 上不再首次 forward 就崩溃。

### 6. [Fast Start] Stop preload groups when a daemon rank exits (#63692)
- **代码层面**：修改 `vllm/entrypoints/cli/preload.py` 的进程管理逻辑：任意 rank 退出时，终止并回收其余 rank，停止健康检查端点，以非零状态退出（保留失败 rank 的退出码，信号退出用 128+N）。文档 `preload.md` 同步更新。测试新增 `test_preload_rank_failure_stops_launcher_and_siblings`，覆盖启动期、服务期、无响应兄弟三种场景。
- **新手概念课堂**：这就像团队登山，一个人掉队了，整个队伍必须停下来等他或撤退，不能假装没事继续走——否则后面会出更大的事（GPU 内存泄漏、僵尸进程）。
- **对你有什么影响**：使用 `vllm preload` 预加载模型时，单个 rank 崩溃不会再留下“半死不活”的进程组，GPU 显存能正确释放。

### 7. [Rust Frontend] Reject chat templates for native renderers (#64274)
- **代码层面**：`rust/src/chat/src/backend/hf.rs` 的 `HfChatBackend::from_resolved_model_files` 中新增检查：如果 `renderer != RendererSelection::Hf` 且用户传了 `chat_template`，直接返回 `Error::UnsupportedChatTemplateOption`。新增测试 `only_hf_renderer_accepts_chat_template`。
- **新手概念课堂**：native renderer（如 deepseek_v4、glm5_next）是用 Rust 硬编码的渲染逻辑，不读 Jinja 模板。你传 `--chat-template` 给它，就像给微波炉插 U 盘——它根本不认识这个接口。之前是静默忽略，现在明确报错并提示“用 `--tokenizer-mode hf`”。
- **对你有什么影响**：如果你用 Rust 前端 + 原生渲染器，误传 `--chat-template` 会立刻看到清晰的错误信息，而不是“为什么我的模板没生效”的困惑。

### 8. [Bugfix][Benchmark] Count tool call deltas as generated tokens in openai-responses (#62753)
- **代码层面**：`tests/benchmarks/test_responses_request_func.py` 新增多个测试，验证 `response.function_call_arguments.delta`、`response.custom_tool_call_input.delta` 等 4 种工具调用 delta 事件都会触发 TTFT/ITL 计时。这是 Python 版 benchmark 的修复。
- **新手概念课堂**：benchmark 测“首 token 时间（TTFT）”时，只数文本 delta。但强制工具调用（forced tool call）的响应里**没有文本**，只有参数 delta。之前这种请求的 TTFT 永远是无穷大/失败，现在把参数 delta 也算作“token 到达”。
- **对你有什么影响**：用 vLLM benchmark 测工具调用场景时，指标不再失真。

### 9. [Bugfix][MRV2] Fix DiffusionGemma runtime OOM via tiled logits projection (#51953)
- **代码层面**：新增 `tests/v1/worker/test_gpu_diffusion_sampler.py`（144 行），核心是让 `DiffusionSampler` 的 logits 投影按 tile 分块计算，而不是一次性算整个 batch。测试验证“采样结果与 tile 大小无关”且“峰值显存随 tile 而非 batch 增长”。
- **新手概念课堂**：DiffusionGemma 是扩散式语言模型，一次要预测整个 canvas（多个位置）的 logits。如果 canvas 长、vocab 大、batch 大，一次性算就是 `[batch*canvas, vocab]` 的巨型矩阵。分块就是“一批一批算，算完就扔”。
- **对你有什么影响**：使用 DiffusionGemma 时不再因为 batch 稍大就 OOM，可以用更大的 batch 提升吞吐。

### 10. [Bugfix] Preserve batch audio handler error status codes (#61973)
- **代码层面**：`vllm/entrypoints/launchers/run_batch.py` 的 `transcription_wrapper` 修改异常处理，保留原始 HTTP 状态码（400 / 500 / 503），而不是统一包成 500。测试新增 `test_batch_audio_preserves_handler_error_status`，参数化覆盖 3 种错误类型 × 2 种操作（transcription/translation）。
- **新手概念课堂**：批量处理音频时，如果某条音频格式错误（400），之前会被包装成“服务器内部错误（500）”，用户以为是服务挂了。现在会准确告诉用户“是你这条数据有问题（400）”。
- **对你有什么影响**：批量转录音频时，错误信息更准确，便于定位是数据问题还是服务问题。

### 11. [Qwen4Exp] Remove FlashAttention dependency and standardize QSA naming (#64110)
- **代码层面**：大规模重命名：`indexer_qsa.py` → `qsa_indexer.py`，`Qwen4ExpQSAFlashAttentionImpl` → `Qwen4ExpQSAImpl`，删除 `alibi_slopes` / `sliding_window` 等 FlashAttention 专有参数。测试同步更新。
- **新手概念课堂**：QSA（Qwen Sparse Attention）原本是“借壳”FlashAttention 的实现，就像用别人的厨房做饭。现在自己建了厨房（独立实现），不再依赖 FlashAttention 的接口，命名也统一了。
- **对你有什么影响**：Qwen4Exp 在 AMD GPU 上的 FP8 KV cache 支持更干净，不再受 FlashAttention 在 ROCm 上的限制。

### 12. [Rust Frontend] Add native GLM-5.3 / GLM-5.3-Flash chat rendering (#62424)
- **代码层面**：`rust/src/chat/` 下新增 GLM5Next 渲染器，支持工具调用和推理选项。`recipes/models/zai-org/GLM-5.3-Flash.yaml` 的 `default_frontend` 改为 `rust`，新增 `--tokenizer-mode glm5_next` 参数。文档说明图片/视频输入暂不支持 Rust 前端。
- **新手概念课堂**：聊天模板（chat template）是把 `[{role: user, content: "hi"}]` 变成模型能读的 token 序列的“翻译器”。之前 GLM-5.3 只能用 Python + Jinja 模板翻译，现在 Rust 里有了原生翻译器，速度更快、更可控。
- **对你有什么影响**：用 GLM-5.3-Flash 时设置 `VLLM_USE_RUST_FRONTEND=1` 和 `--tokenizer-mode glm5_next` 可获得更快的请求处理；但多模态请求仍需切回 Python 前端。

### 13. [Doc] Kimi-K3: move the GB300 AgentX presets to the Oct 26 points with KV-balanced decode (#64665)
- **代码层面**：`recipes/models/moonshotai/Kimi-K3.yaml` 更新 preset 描述和镜像 tag，新增 `json_args` 配置（NIXL + MooncakeStore 双连接器），移除部分 `args`（如 `--mamba-ssm-cache-dtype float32`、`--cp-kv-cache-interleave-size 1`），改为按 decode profile 动态配置。
- **新手概念课堂**：这是“调参配方”的更新——把之前针对低并发（16-32 客户端）优化的参数，改成同时覆盖低并发（延迟优先）和高并发（KV 均衡）两套 profile。就像餐厅菜单从“只做午市套餐”改成“午市/晚市两套”。
- **对你有什么影响**：部署 Kimi-K3 到 GB300 时，按 recipe 配置能同时获得低延迟和高吞吐两个场景的优化。

### 14. [KV Connector][ROCm] MoRIIO: hybrid READ for several attention groups and shared-group speculative methods (#60867)
- **代码层面**：`moriio_connector` 的 hybrid READ 模式从“只支持单个可传输 attention group”扩展到“支持多个 attention group”，因此 MTP 等共享 target attention group 的 draft 方法也能用。`_mamba_payload_index_by_layer` 重命名为 `_hybrid_payload_index_by_layer`。测试大幅扩展（804 行 diff）。
- **新手概念课堂**：MoRIIO 是 AMD 的 KV cache 传输连接器（类似 NVIDIA 的 NIXL）。之前它只能搬“一种注意力层”的 KV，现在能搬“多种注意力层”（比如 MLA + Mamba 混合），这样投机解码的 draft 模型也能复用传输通道。
- **对你有什么影响**：在 AMD GPU 上用 MoRIIO 做 P/D 分离部署时，MTP 投机解码也能正常工作，不再需要单独配置。

### 15. [ROCm][Bugfix] Skip AITER tuned GEMM when M*N >= 2^31 on gfx950 (#64567)
- **代码层面**：`vllm/model_executor/layers/utils.py` 的 `rocm_unquantized_gemm_impl` 中新增 `aiter_hgemm_unsafe` 判断：当 `n*m >= 2^31` 或 `max(n,m)*k*element_size >= 2^32` 时，跳过 AITER tuned GEMM，回退到普通 GEMM。
- **新手概念课堂**：AITER 是 AMD 的算子库，它的 gfx950 FlyDSL kernel 用 int32 寻址输出矩阵 C。当输出矩阵元素数超过 2^31 时，索引溢出，结果错乱或崩溃。这个修复就是“大矩阵绕开这个有 bug 的快速路径”。
- **对你有什么影响**：在 AMD MI350（gfx950）上跑大 batch 推理时，不再因为 AITER 溢出而崩溃或输出乱码，只是大矩阵会慢一点点。

### 16. [ROCm][CI] Update ROCm AITER nightly staging index (#64610)
- **代码层面**：`.buildkite/scripts/rocm/select_aiter_nightly_wheel.py` 的 `INDEX_URL` 从 `whl-multi-arch` 改为 `whl-multi-arch-staging`，wheel 文件名正则支持 `manylinux_2_28_x86_64` 格式。
- **新手概念课堂**：这是 CI 基础设施的维护——AMD 把 nightly wheel 的发布目录换到了 staging 子目录，文件名格式也从 `linux_x86_64` 变成 `manylinux_2_28_x86_64`（表示需要 glibc 2.28+）。脚本要跟着改，否则 CI 找不到 wheel。
- **对你有什么影响**：普通用户无感知；但如果你自己写脚本下载 AITER nightly wheel，需要更新 URL 和正则。

### 17. [ROCm][Perf][DSv4.1] Build the SWA decode metadata in one launch (#63885)
- **代码层面**：`vllm/models/deepseek_v41/amd/ops/` 下新增 `swa_decode_meta` 模块，把原本需要多次 kernel launch 的 SWA（滑动窗口注意力）decode metadata 构建合并成一次 Triton launch。测试新增 CDNA3/CDNA4 × monokernel 开关的参数化测试。
- **新手概念课堂**：每次 kernel launch 都有固定开销（几微秒），decode 阶段每步都要构建 metadata，多次 launch 累积起来很可观。合并成一次 launch，就像把“分 5 次去超市买东西”改成“一次列好清单全买回来”。
- **对你有什么影响**：DeepSeek-V4.1 在 AMD CDNA3/CDNA4 上 decode 阶段有小幅性能提升，开启 `--monokernel` 时生效。

### 18. [Bugfix][Spec Decode] Per-request draft retention for structured output under PP (#63219)
- **代码层面**：核心是让投机解码的 draft token 按 request id 存储，而不是全局共享。`test_draft_model.py` 重构出 `vllm_runner_kwargs` 辅助函数，新增 `pipeline_parallel_size` 和 `engine_kwargs` 字段。新增测试 `test_structured_output_pp.py::test_specdec_pp_structured_output`。
- **新手概念课堂**：流水线并行（PP）下，多个请求在不同 GPU stage 上同时流动。如果 draft token 存在全局变量里，请求 A 的 draft 可能被请求 B 覆盖。结构化输出（JSON Schema）对 token 顺序极其敏感，覆盖一次就输出非法 JSON。修复就是“每个请求一个抽屉”。
- **对你有什么影响**：在 PP 部署 + 投机解码 + 结构化输出（如 function calling 返回 JSON）场景下，不再出现随机 JSON 解析失败。

### 19. [EPD] Metadata-only image and video inputs for GLM-5.3-Flash (#64071)
- **代码层面**：`vllm/model_executor/models/glm4_1v.py` 的 `Glm4vMultiModalDataParser` 新增 `allow_missing_mm_embeddings` 参数。当为 True 时，只解析 `image_grid_thw` / `video_grid_thw` / `timestamps` 等元数据，不要求 `image_embeds` / `video_embeds`。新增 `test_epd_metadata.py` 验证 GLM-4V 和 GLM-5.3-Flash 共享此契约。
- **新手概念课堂**：EPD（Encoder-Prefill-Decode）分离架构中，视觉编码器（Encoder）和语言模型（Prefill/Decode）在不同进程/机器上。Prefill 进程只需要知道“有几张图、每张图多大”，不需要图片 embedding 本身。这个 commit 让 parser 支持“只解析元数据”，为 EPD 分离铺路。
- **对你有什么影响**：部署 GLM-5.3-Flash 的 EPD 分离架构时，Prefill 节点不再需要接收完整的图片 embedding，节省带宽和内存。

### 20. [Bugfix][Rust Frontend] Count tool call deltas as generated tokens in vllm-bench openai-responses (#64557)
- **代码层面**：`rust/src/bench/src/backends/openai_responses.rs` 的 `ResponsesStream::handle_event` 中，把 4 种工具调用 delta 事件加入“触发 TTFT/ITL 计时”的 match 分支。新增两个测试验证工具调用 delta 的计时和计数。
- **新手概念课堂**：这是 #62753 的 Rust 版对应修复。benchmark 工具（vllm-bench）用 Rust 重写后，需要同步 Python 版的 bug 修复。两个 commit 合起来说明：**同一逻辑在多语言实现中要保持行为一致**，这是大型项目的常见挑战。
- **对你有什么影响**：用 Rust 版 vllm-bench 测工具调用场景时，TTFT/ITL 指标准确，与 Python 版一致。

## sglang
## 💡 今日关键词

- **关键词：AMD GPU 生态适配（ROCm / AITER / gfx950）**
  - 一句话解释：SGLang 正在把大量推理优化（注意力、MoE、量化 GEMM、确定性推理）从 NVIDIA CUDA 迁移到 AMD 的 ROCm 软件栈上，AITER 是 AMD 官方的 AI 加速内核库。
  - 为什么社区在关注：今天 20 个 commit 里有 9 个带 `[AMD]` 标签，涉及注意力、量化、MoE、确定性模式、文档等。这说明 AMD Instinct（MI300/MI355）正在成为大模型推理的主流硬件选项，SGLang 在积极补齐功能与性能。
  - 对新手意味着什么：如果你只会 CUDA，现在补 ROCm/HIP 是明显的差异化优势；国内很多算力中心也在用 AMD，懂 AITER/ROCm 的工程师稀缺。

- **关键词：量化推理（FP8 / MXFP8 / MXFP4 / Quark / Compressed-Tensors）**
  - 一句话解释：把模型权重和激活从 16 位浮点压到 8 位甚至 4 位，省显存、提吞吐，但要小心精度和内核布局（layout）问题。
  - 为什么社区在关注：今天有多个量化相关的修复（Commit 10、12、6），都是"布局错了"或"配置选错了"这类隐蔽 bug。量化不只是精度问题，更是内存排布问题。
  - 对新手意味着什么：量化是推理部署最核心的技能之一；学会看 weight layout（row-major / preshuffled / block scale）能让你在排查精度异常时少走弯路。

- **关键词：投机解码（Speculative Decoding / DSpark / Verify）**
  - 一句话解释：用小模型先"猜"几个 token，再用大模型一次性验证，加速生成。
  - 为什么社区在关注：Commit 3（NPU 原生采样/验证 kernel）、Commit 5（协调验证 rank 的 replay 问题）、Commit 18（MiniMax M3.1 的 DSpark 测试）都围绕投机解码。这是当前 LLM 推理加速最火的方向之一。
  - 对新手意味着什么：投机解码涉及采样、验证、KV 一致性等复杂逻辑，是学习推理系统架构的好切入点。

- **关键词：CUDA Graph / 可中断图（Breakable CUDA Graph）与 HiCache**
  - 一句话解释：CUDA Graph 把一串 GPU 操作录制成"图"一次性提交，减少 CPU 开销；HiCache 是多级 KV 缓存（GPU→主机→存储）。
  - 为什么社区在关注：Commit 13 修复了 HiCache 与 breakable CUDA graph 配合时的事件绑定 bug，属于"性能优化与正确性边界"的经典问题。
  - 对新手意味着什么：理解 CUDA Graph 和缓存分层是进阶推理系统的必修课，也是面试高频题。

- **关键词：PD 分离（Prefill/Decode Disaggregation）与重引导（Rebootstrap）**
  - 一句话解释：把"预填充"和"解码"放在不同机器上跑，提升资源利用率；重引导是解码端在 KV 不足时重新找 prefill 端算 KV。
  - 为什么社区在关注：Commit 20 修复了 PD 重引导时 output_ids 被 pop 掉再回放的复杂逻辑，说明 PD 分离正在走向生产可用。
  - 对新手意味着什么：PD 分离是 vLLM/SGLang 等框架的核心架构，理解它需要掌握 KV cache、调度、状态机。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：量化权重的"布局（Layout）"为什么会导致精度错误**
  - **是什么**：量化后权重在内存里可能被"预打乱"（preshuffled）以便某个内核高效读取。如果选用的 runner 期望的是原始 row-major 布局，却拿到打乱过的权重，结果就会错，但不会报错。
  - **为什么重要**：Commit 12 正是修复这个问题——CT FP8 MoE 在 `_use_aiter` 为真时无条件预打乱权重，但实际 runner 可能不是 aiter，导致布局不匹配。这类 bug 极难排查，因为形状对、数值错。
  - **动手试试**：在 Python 里对比一个矩阵和它的"打乱版本"做 GEMM：
    ```python
    import torch
    a = torch.randn(8, 16)
    b = torch.randn(16, 32)
    b_shuffled = b[torch.randperm(16)]  # 模拟打乱
    print((a @ b).sum().item())
    print((a @ b_shuffled).sum().item())  # 形状对，结果不同
    ```
  - **预期结果**：两次输出数值不同，但形状完全相同——这就是"静默错误"的样子，提醒你布局必须和内核匹配。

- **知识点 2：软注意力缩放（softmax scale）必须用 layer 自己的值**
  - **是什么**：注意力计算里 `QK^T / sqrt(d)` 的那个 scale。有些模型（如 Granite）会自定义 `attention_multiplier`，不等于 `1/sqrt(head_dim)`。
  - **为什么重要**：Commit 11 修复了 AITER 后端硬编码 `self.scale = 1/sqrt(head_dim)` 的 bug，改为使用 `layer.scaling`。如果模型自定义了 scale，硬编码会导致输出偏差（不报错，但质量下降）。
  - **动手试试**：手算两种 scale 下的注意力输出差异：
    ```python
    import torch, math
    q = torch.randn(4, 8); k = torch.randn(4, 8)
    d = 8
    s1 = (q @ k.T) / math.sqrt(d)
    s2 = (q @ k.T) * (1.0 / d)  # 自定义 multiplier
    print(s1.softmax(-1).sum(-1))
    print(s2.softmax(-1).sum(-1))  # 都是 1，但分布不同
    print((s1.softmax(-1) - s2.softmax(-1)).abs().max())
    ```
  - **预期结果**：softmax 行和都为 1，但两者分布不同，说明 scale 错误会改变注意力权重。

- **知识点 3：CUDA Graph 与动态等待（Event）的配合**
  - **是什么**：CUDA Graph 把 kernel 序列固定下来，但 HiCache 加载是"动态"的（数据什么时候到不确定）。需要用 CUDA event 让图里的 kernel "等"数据到齐。
  - **为什么重要**：Commit 13 修复了 breakable CUDA graph 中 KDA 等待绑定到"捕获时"而非"运行时"的 bug——捕获时记录的是旧 event，运行时数据到了也不会等，导致读到脏数据。
  - **动手试试**：用 PyTorch 模拟"图捕获 + event 等待"的伪代码：
    ```python
    import torch
    s = torch.cuda.Stream()
    ev = torch.cuda.Event()
    x = torch.zeros(4, device="cuda")
    # 生产者：在另一流写数据后 record event
    with torch.cuda.stream(s):
        x.fill_(1.0)
        ev.record(s)
    # 消费者：等待 event 再读
    torch.cuda.current_stream().wait_event(ev)
    print(x)  # 一定是 1.0，而不是 0.0
    ```
  - **预期结果**：输出 `tensor([1., 1., 1., 1.])`。如果去掉 `wait_event`，可能读到 0.0——这就是"捕获 vs 运行时"绑定错误的本质。

- **知识点 4：投机解码中的"验证"与"奖励 token"**
  - **是什么**：draft 模型猜 K 个 token，target 模型并行验证，接受前缀匹配的部分，并在第一个不匹配处采样一个"奖励 token"（bonus token）。
  - **为什么重要**：Commit 3 在 NPU 上集成了原生 `build_out_tokens` 内核，负责把 draft tokens、correct_len、bonus 拼成最终输出。逻辑错了会丢 token 或重复 token。
  - **动手试试**：用纯 Python 模拟一遍：
    ```python
    draft = [10, 20, 30, 40]
    target = [10, 20, 99, 88]
    correct_len = 0
    for d, t in zip(draft, target):
        if d == t: correct_len += 1
        else: break
    bonus = 77
    out = draft[:correct_len] + [bonus]
    print(out)  # [10, 20, 77]
    ```
  - **预期结果**：输出 `[10, 20, 77]`，理解"接受前缀 + 奖励 token"的拼接逻辑。

- **知识点 5：PD 分离中的"重引导"与 token 计数**
  - **是什么**：解码端 KV 不够时，把已生成的 token 送回 prefill 端重新算 KV（rebootstrap）。关键是要算清楚"要重新填充多少 token"。
  - **为什么重要**：Commit 20 修复了旧逻辑"pop 掉最后一个 token 再回放"的复杂做法，改为保留 output_ids，并统一用 `_pre_alloc_fill_len` 计算（最后一个 token 的 KV 还没写，所以减 1）。计数错了会导致显存预估错误或丢 token。
  - **动手试试**：手算两种计数：
    ```python
    origin = 100  # prompt 长度
    output = 5    # 已生成 token 数
    # 普通 decode：最后一个 token KV 未写
    fill = origin + max(output - 1, 0)
    print(fill)  # 104
    ```
  - **预期结果**：输出 `104`，理解"pending token 的 KV 尚未写入"这一细节。

---

## 🔥 重要更新

- **Commit 12（AMD MoE 布局修复）**：CT FP8 权重预打乱必须和实际 runner 匹配，否则数值静默错误。这是量化部署的典型隐蔽 bug，值得所有做量化的人警惕。
- **Commit 11（AITER 软注意力 scale）**：硬编码 `1/sqrt(d)` 在自定义 `attention_multiplier` 的模型（如 Granite）上会算错，影响输出质量。
- **Commit 20（PD 重引导 token 计数）**：简化了复杂的 pop/replay 逻辑，统一用 `_pre_alloc_fill_len`，修复了潜在的 token 丢失/显存预估错误。
- **Commit 13（HiCache × Breakable CUDA Graph）**：捕获的等待必须绑定到运行时 event，否则读到脏数据——CUDA Graph 与动态数据配合的经典坑。

---

## 📋 逐条解读

### 1. [Feat] docker: Update the EFA images to EFA installer 1.51.0
- **代码层面**：修改 `docker/efa.Dockerfile`，把 AWS EFA 安装器从 1.50.0 升到 1.51.0，并改用 `--build-ngc` 最小安装（只装 libfabric、aws-ofi-nccl、hwloc），跳过 MPI 和 rdma-core。同步更新文档 `pd_disaggregation.mdx`。
- **新手概念课堂**：EFA 是 AWS 的"高速网卡"，类似 NVIDIA 的 InfiniBand。aws-ofi-nccl 是让 NCCL（GPU 通信库）走 EFA 网卡的插件。`--build-ngc` 是"只装必需组件"的安装模式，像点外卖只要主食不要赠品。
- **对你有什么影响**：在 AWS EFA 实例上跑 PD 分离的用户，镜像更小、依赖更少，启动更快。

### 2. [Perf] detokenizer: Send per-request decode inputs once and keep them in the detokenizer
- **代码层面**：修改 `beam_search/output.py` 等，把 `no_stop_trim`、`skip_special_tokens`、`spaces_between_special_tokens` 这些"每请求固定"的参数从每次输出都传，改为通过 `DetokenizeHeader` 一次性传给 detokenizer 并缓存。
- **新手概念课堂**：detokenizer 是"把 token id 变回文字"的组件。以前每个 batch 输出都重复传同样的参数（像每次寄快递都重新填一遍地址），现在只传一次并存在本地。
- **对你有什么影响**：高并发下减少进程间通信开销，吞吐略微提升，普通用户无感知。

### 3. [Feat] [NPU] dspark: Integrate native sampling and verification kernels
- **代码层面**：在 `dspark_draft_model.py` 和 `dspark_verify_window.py` 中，当检测到 NPU 时，调用 `sgl_kernel_npu` 里的原生 `markov_greedy_step_triton` 和 `build_out_tokens_triton`，替代原来的 Triton 实现。
- **新手概念课堂**：DSpark 是一种投机解码方案。NPU（如昇腾）有自己的算子库，用原生算子比通用 Triton 快，就像用专用工具代替万能工具。
- **对你有什么影响**：昇腾 NPU 用户跑 DSpark 投机解码会更快。

### 4. [Perf] [AMD] Qwen3.8: AITER FlyDSL hc_mix for hyper-connections on gfx950
- **代码层面**：新增 `hc_mix_flydsl.py`，在 gfx950（MI355）上用 AITER 的 FlyDSL 融合 GEMM epilogue 加速 Qwen3.8 的 hyper-connections（HC）混合层。有严格的条件检查（HC=4、K 是 64 倍数、R 是 32 倍数、bf16、行数 ≤128）。
- **新手概念课堂**：hyper-connections 是一种残差连接的变体。"融合 epilogue"指把 GEMM 后的加偏置/激活等操作合并进同一个 kernel，减少显存往返。
- **对你有什么影响**：AMD MI355 上跑 Qwen3.8 小 batch 推理更快（官方数据 128 行时 26us vs 42us）。

### 5. [Fix] dsv4: Keep a coordinated verify rank out of the decoder-tail idle replay
- **代码层面**：在 `deepseek_v4_replay_tail.py` 的 `plan` 里加一个判断：如果 `forward_batch.dp_spec_prefill_coordination_applied` 为真，直接返回 `False, None`，不进入 decoder-tail 的空闲 replay。
- **新手概念课堂**：DP（数据并行）下多个 rank 要协调，某个 rank 在等别人时不能乱跑 replay，否则状态不一致。像接力赛里等队友时必须原地等，不能自己先跑。
- **对你有什么影响**：DeepSeek V4 在 DP 投机解码场景下更稳定，避免偶发的输出错误。

### 6. [Perf] [AMD] dsv4.1: Wire up 32x32-block fp8 wo_a GEMM
- **代码层面**：在 `deepseek_v4_hip.py` 和 `deepseek_v4_wo_a_fp8.py` 中，新增对 32×32 block 的 fp8 wo_a GEMM 支持（V4.1 的 checkpoint 用 32 宽 block，V4 用 128），并让注意力后端识别 `wo_a_fp8_block128` 标志。
- **新手概念课堂**：量化 block 大小指"多少个元素共享一个 scale"。block 越小精度越高但开销越大。V4.1 从 128 改成 32，需要内核适配。
- **对你有什么影响**：AMD 上跑 DeepSeek V4.1 的 fp8 推理更快。

### 7. [Perf] [AMD] Qwen3.8: Enable fused GDN packing for ratio-3 heads
- **代码层面**：在 `qwen3_5.py` 中把 `_GDN_FUSED_QKVZBA_RATIOS` 在 aiter 路径下从 `(1,2,4,8)` 改为 `(1,2,3,4,8)`，让 ratio-3 的头也能走融合 kernel。
- **新手概念课堂**：GDN 是线性注意力的一种。头比例（ratio）指 value 头数 / key 头数。融合 kernel 把 split/reshape/cat 合并，省掉 `.contiguous()` 拷贝。
- **对你有什么影响**：AMD 上 Qwen3.8 的线性注意力层更快。

### 8. [Fix] [AMD] deterministic: Bypass AITER GEMM and MoE in deterministic mode
- **代码层面**：在 `unquant.py` 中新增 `_skip_aiter_unquant` 标志，在 gfx95 + 确定性推理模式下跳过 AITER 的 GEMM 和 MoE（因为 AITER 按 token 数选 kernel，结果不确定）；如果用户显式指定 aiter MoE runner 则直接报错。
- **新手概念课堂**：确定性推理指"同样输入永远同样输出"，对调试和合规很重要。AITER 的 split-K 会因 batch 大小改变求和顺序，导致结果不可复现。
- **对你有什么影响**：AMD 用户开 `--enable-deterministic-inference` 时结果可复现，但会损失一些性能。

### 9. [Test] [AMD] fp8-grid: Compare GEMM to aiter within one bf16 rounding, not bitwise
- **代码层面**：修改测试 `test_fp8_grid_producers_gfx95.py`，把"与 aiter 逐位相等"改为"在一个 bf16 舍入误差内"，因为 aiter 的求和顺序是调优实现细节，可能变。
- **新手概念课堂**：浮点加法不满足结合律，`(a+b)+c ≠ a+(b+c)`。所以不同求和顺序结果不同。"一个 bf16 舍入"指允许相对误差约 2^-8。
- **对你有什么影响**：测试更稳健，不会因为 aiter 调优而误报。

### 10. [Fix] [AMD] quark: Keep MXFP8 shared experts out of the MXFP4 fused slot
- **代码层面**：在 `quark.py` 中新增 `_NAMED_MOE_LAYER` 正则和 `_named_moe_layers` 方法，把所有"有名字的 MoE 层"都加入探测列表，而不只看 layer 0 和 draft block。修复了 dense-first 模型（layer 0 是 dense MLP）下误判 shared expert 可融合的问题。
- **新手概念课堂**：Quark 是 AMD 的量化工具。shared expert 是 MoE 里所有 token 都走的专家，有时会融合进主 MoE 槽位。但 MXFP8 和 MXFP4 精度不同，不能乱融。
- **对你有什么影响**：AMD 上跑 MiniMax-M3 等模型时，量化配置不会错配，输出更正确。

### 11. [Fix] [AMD] aiter: Use the layer's softmax scale in MHA prefill and decode
- **代码层面**：在 `aiter_backend.py` 删掉硬编码的 `self.scale = 1/sqrt(head_dim)`，改为在 prefill/decode 调用时传 `softmax_scale=layer.scaling`；`aiter_utils.py` 同步修改。测试新增了 `1/head_dim` 这种非标准 scale 的用例。
- **新手概念课堂**：见知识点 2。有些模型（Granite）用 `attention_multiplier` 自定义 scale，硬编码会算错。
- **对你有什么影响**：AMD 上跑 Granite 等自定义 scale 的模型，输出质量恢复正常。

### 12. [Fix] [AMD] MoE: Preserve CT FP8 Triton layout and reduction dimensions
- **代码层面**：在 `fused_moe_triton_kernels.py` 中 `padded_size = 0 if _use_aiter else padding_size`；在 `compressed_tensors_w8a8_fp8_moe.py` 中预打乱权重前增加 `self.runner.runner_backend.is_aiter()` 判断。新增测试 `test_compressed_tensors_fp8_moe_layout.py`。
- **新手概念课堂**：见知识点 1。预打乱是为了 aiter 内核高效读取，但如果 runner 不是 aiter，就打乱了不该打乱的权重。
- **对你有什么影响**：AMD 上 CT FP8 MoE 在非 aiter runner 下不再出错。

### 13. [BugFix][HiCache * BCG] bind captured KDA waits to live load events
- **代码层面**：新增 `hicache_graph_events.py`，在 `kda_backend.py` 的 prefill graph 路径里调用 `capture_layer_wait(layer.layer_id)`；`breakable_cuda_graph_backend.py` 增加 `enable_hicache_event_capture` 参数和 `wait_for_layer_loads` 逻辑；`prefill_cuda_graph_runner.py` 更新注释说明 Full graph 仍需 barrier。
- **新手概念课堂**：见知识点 3。CUDA Graph 捕获时记录的 event 是"旧"的，运行时数据到了新 event，必须重新绑定。
- **对你有什么影响**：HiCache + breakable CUDA graph 组合下不再读到脏 KV 数据，长上下文场景更可靠。

### 14. [Fix] humming: adapt integration to kernels 0.1.18
- **代码层面**：`pyproject.toml` 把 `humming-kernels` 从 0.1.12 升到 0.1.18；`environ.py` 新增 `HUMMING_CACHE_DIR`/`HUMMING_TMP_DIR` 重定向；`humming.py` 新增 `InputQuantizationMode` 导入、`_humming_moe_runner_core_run_fake` 假实现（供 torch.compile 追踪）。
- **新手概念课堂**：Humming 是第三方 MoE 内核库。"fake impl"是给 `torch.compile` 用的占位实现，只推形状不真算，像建筑图纸代替真盖楼。
- **对你有什么影响**：用 Humming 后端的用户升级后兼容，缓存目录统一到 `SGLANG_CACHE_DIR` 下。

### 15. [Doc] [AMD] cookbook: Update Muse Spark Open MI355X AgentX recipe
- **代码层面**：更新 `MuseSparkOpen.mdx`，镜像从 `muse-spark-open-1020` 改为 `1027`，新增 monokernel 低并发配置和 TP4 HiCache 高并发配置，hicache-size 从 253 改为 310。
- **新手概念课堂**：cookbook 是"菜谱"，给用户直接抄的部署命令。monokernel 是把多个小 kernel 合成一个大 kernel 的技术。
- **对你有什么影响**：在 MI355X 上部署 Muse Spark Open 的用户可以直接抄新配置，性能更好。

### 16. [Feat] rust-server: Report cached tokens by cache tier in native /generate meta_info
- **代码层面**：`generate.proto` 新增 `CachedTokensDetails` 消息（device/host/storage/storage_backend 字段）；`api_types.py` 新增对应 msgspec 结构和编解码函数。
- **新手概念课堂**：缓存分层指 KV cache 可能在 GPU 显存（device）、主机内存（host）、磁盘（storage）。分 tier 报告让用户知道"命中率"来自哪一层。
- **对你有什么影响**：用 Rust server 的用户能在 `/generate` 响应里看到缓存命中明细，方便调优。

### 17. [Perf] [AMD] attention: Use the AITER FAv3 varlen kernel on gfx942 as well
- **代码层面**：`aiter.py` 删掉 gfx942 的特殊分支（原来因为 ASM kernel 在 64K 序列上 hang 而改用 Triton），统一用 `aiter.flash_attn_varlen_func`，并加 `how_v3_bf16_cvt=AITER_BF16_CVT_MODE` 参数。
- **新手概念课堂**：FAv3 是 FlashAttention v3。varlen 指变长序列（不同请求长度不同）。gfx942 是 MI300X。
- **对你有什么影响**：AMD MI300X 上多模态生成的变长注意力更快，且不再 hang。

### 18. [Test] minimax-m3.1: Add nightly e2e tests on H200 and B300
- **代码层面**：`ci_install_dependency.sh` 新增 `install_msa` 函数（只在 Blackwell 上装 MiniMax MSA 稀疏注意力内核，带 flock 锁避免并发冲突）；新增 `test_minimax_m31_flash_gb300.py` 测试。
- **新手概念课堂**：MSA 是 MiniMax 的稀疏注意力。Blackwell 是 NVIDIA 新架构（B200/B300）。flock 是文件锁，防止多个 CI job 同时 `rm -rf` 同一个目录。
- **对你有什么影响**：MiniMax-M3.1 在 B300 上的部署路径有 CI 保障，质量更稳。

### 19. [Perf] dsv4: Fix amax8_varlen and mega MoE pre-dispatch grids for 1024-thread SMs
- **代码层面**：`block_amax.cuh` 新增 `kWaves = 2`，把 grid.y 从"覆盖所有 key chunk"改为"2 波常驻 CTA"，用循环 `for (first_key = by*...; first_key < num_keys; first_key += gridDim.y*...)` 处理剩余 chunk。
- **新手概念课堂**：GPU 上 CTA（线程块）按"波（wave）"调度，一波跑完再跑下一波。grid 太大反而有调度开销。"2 波"是经验调优值，在 CC 10.7 上比全宽 grid 快 56-84%。
- **对你有什么影响**：DeepSeek V4 在 Blackwell（1024 线程 SM）上 amax 和 MoE 预分发更快。

### 20. [Fix] disaggregation: Keep output ids during a PD rebootstrap
- **代码层面**：`decode.py` 删掉 `pd_rebootstrap_forced_output_id` 的 pop/replay 逻辑，删除 `_rebootstrap_prefill_len`，统一用 `_pre_alloc_fill_len`（`len(origin) + max(len(output)-1, 0)`）计算填充长度和显存预估。
- **新手概念课堂**：见知识点 5。旧逻辑"pop 最后一个 token 再回放"容易出错，新逻辑"保留所有 output_ids，最后一个 token 的 KV 当作未写"更简单一致。
- **对你有什么影响**：PD 分离下 KV 不足触发重引导时，不再丢 token 或算错显存，长生成更稳定。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)