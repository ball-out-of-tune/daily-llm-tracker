# 🚀 vLLM & SGLang 每日更新 — 2026-11-10

> 自动生成于 2026-11-10 08:48 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：EPLB（专家并行负载均衡）**
  - **一句话通俗解释**：MoE（混合专家）模型里每个专家干活多少不一样，EPLB 就是动态把"忙的专家"搬到空闲 GPU 上，让每张卡负载均衡。
  - **为什么社区在关注**：今天有两个 commit（#66660、#66399）都在优化 EPLB 的拷贝路径，说明大 MoE 模型（DeepSeek、GLM 等）在多卡部署时"负载不均"已经是核心瓶颈，大家都在抠"一次 kernel launch 搬多少数据"这种细节。
  - **对新手意味着什么**：如果你学分布式推理，MoE + 专家并行是必学方向；理解"数据搬运次数"比"搬运总量"更影响性能，是进阶的第一步。

- **关键词：ROCm / AMD GPU 适配**
  - **一句话通俗解释**：ROCm 是 AMD 的 CUDA 对标生态，AITER 是 AMD 的高性能算子库。
  - **为什么社区在关注**：今天 20 个 commit 里有 5 个（#66596、#66399、#64153、#63893、#66644）是 ROCm/AMD 相关，AITER 还专门 bump 了版本。说明 vLLM 正在把 AMD MI300/MI355 当成一等公民。
  - **对新手意味着什么**：异构计算（NVIDIA + AMD 双栈）正在成为主流，会写"平台无关"的 kernel 代码（用 Triton 而不是纯 CUDA）是稀缺技能。

- **关键词：CuTe DSL / 专用 Decode Kernel**
  - **一句话通俗解释**：CuTe 是 NVIDIA 的 C++ 模板库，用来手写极致性能的 GPU kernel；DSL 是它的 Python 封装。
  - **为什么社区在关注**：#66652 和 #63102 都在为特定硬件（GB300、B200）写"专用 decode 路径"，说明通用 kernel 已经榨不出性能了，开始"一卡一策"。
  - **对新手意味着什么**：性能优化进入"深水区"，想拿高薪得懂硬件微架构（SM 数量、显存带宽、Tensor Core 形状）。

- **关键词：Rust 前端 / 流式 Tool Call 解析**
  - **一句话通俗解释**：vLLM 正在用 Rust 重写 API 前端，今天的 #66529、#66528 让 DeepSeek 的工具调用参数可以"边生成边流式输出"。
  - **为什么社区在关注**：Agent 应用爆发，工具调用（function calling）的流式体验是刚需；Rust 重写是为了性能和内存安全。
  - **对新手意味着什么**：Rust + LLM 基础设施是新兴高价值组合，值得投入。

- **关键词：GPU-CPU 同步检测**
  - **一句话通俗解释**：GPU 和 CPU 之间如果频繁"等对方"，性能会暴跌；vLLM 新增 `VLLM_GPU_SYNC_CHECK=error` 来抓这种隐藏 bug。
  - **为什么社区在关注**：#66390 修复了 dots.ocr 的同步问题，还专门写了文档和测试，说明"隐藏同步"是推理性能的隐形杀手。
  - **对新手意味着什么**：写 GPU 代码时，`.item()`、`.cpu()`、`if tensor > 0` 都可能触发同步，学会用 profiler 抓同步是基本功。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：MoE 专家负载均衡（EPLB）与批量拷贝**
  - **是什么**：MoE 模型有几十上百个"专家"（小 FFN），每个 token 只激活其中几个。EPLB 会在运行中把"热门专家"复制到多张卡上，把"冷门专家"下线，重新分配后需要把权重从缓冲区**拷回**到模型参数里。
  - **为什么重要**：如果逐行拷贝（per-row），一次 rebalance 要发几千次 CUDA copy，卡顿明显；批量拷贝（batched）一次 launch 搞定，是"重平衡不打断服务"的关键。不懂这个，你调 MoE 服务时会看到周期性卡顿却找不到原因。
  - **动手试试**：
    ```python
    import torch
    # 模拟：把 buffer 里 6 行专家权重拷回 4 行参数（有重复、有丢弃）
    weights = torch.randint(0, 256, (4, 64), dtype=torch.uint8, device="cuda")
    buffer  = torch.randint(0, 256, (6, 64), dtype=torch.uint8, device="cuda")
    new_indices = torch.tensor([0, 1, 1, 3, 5, 5], device="cuda")  # -1 表示丢弃
    # 批量：用 index_select 一次完成
    valid = new_indices >= 0
    weights[valid] = buffer[new_indices[valid]]
    # 对比逐行写法
    for i, idx in enumerate(new_indices.tolist()):
        if idx >= 0:
            weights[i] = buffer[idx]
    print("done")
    ```
  - **预期结果**：两种写法结果一致，但用 `torch.profiler` 看会发现逐行写法产生 N 次 kernel launch，批量写法只有 1 次。这就是 #66660 的核心优化。

- **知识点 2：非连续张量的 GPU 拷贝陷阱**
  - **是什么**：PyTorch 的 `transpose`、`chunk`、切片操作产生的是"视图"（view），内存不连续；把非连续 CPU 张量拷到 GPU 会走极慢的逐元素路径。
  - **为什么重要**：#65987 修的就是 Llama-4 加载 MoE 权重时忘了 `.contiguous()`，导致模型加载慢好几倍。这是新手最容易踩的坑之一。
  - **动手试试**：
    ```python
    import torch, time
    a = torch.randn(4096, 4096)
    b = a.t()                       # 非连续视图
    c = b.contiguous()              # 连续副本
    for name, t in [("non-contig", b), ("contig", c)]:
        start = time.time()
        for _ in range(100):
            _ = t.to("cuda")
        torch.cuda.synchronize()
        print(name, f"{(time.time()-start)*1000:.1f} ms")
    ```
  - **预期结果**：`non-contig` 明显更慢（可能 2-5 倍）。记住：**大张量上 GPU 前先 `.contiguous()`**。

- **知识点 3：GPU-CPU 同步检测**
  - **是什么**：GPU 是异步执行的，但当你写 `loss.item()`、`if tensor.any()`、`print(tensor)` 时，CPU 必须等 GPU 算完，造成 pipeline 停顿。
  - **为什么重要**：#66390 用 `VLLM_GPU_SYNC_CHECK=error` 抓出了 dots.ocr 视觉编码器里的隐藏同步；#66399 也修了 AITER GDN 里的阻塞 H2D 拷贝。生产环境里这类 bug 会让吞吐掉 20%+。
  - **动手试试**：
    ```bash
    # 在 vLLM 里开启严格同步检查
    VLLM_GPU_SYNC_CHECK=error vllm serve rednote-hilab/dots.ocr \
        --trust-remote-code --max-model-len 4096 \
        --limit-mm-per-prompt '{"image": 1}'
    # 然后发一个带图片的请求，观察是否报错
    ```
  - **预期结果**：正常情况下不报错；如果你自己写模型代码时加了 `tensor.item()`，会看到 `RuntimeError: GPU<->CPU sync detected`。

- **知识点 4：流式 Tool Call 参数解析**
  - **是什么**：LLM 生成工具调用时，参数是 JSON 字符串。流式场景下 token 一个个来，解析器要能"边收边输出"合法的 JSON 片段，而不是等全部生成完。
  - **为什么重要**：#66529 让 DeepSeek 的 DSML 格式支持流式参数，Agent 应用能更早开始执行工具（比如边生成边发 HTTP 请求）。#66528 还统一了 JSON 风格（`{"a": 1}` vs `{"a":1}`），保证 roundtrip 一致。
  - **动手试试**：
    ```python
    import json
    # 模拟流式 token
    chunks = ['{"loc', 'ation":', ' "Tok', 'yo"}']
    buf = ""
    for c in chunks:
        buf += c
        try:
            obj = json.loads(buf)
            print("parsed:", obj)
        except json.JSONDecodeError:
            print("partial:", buf)
    ```
  - **预期结果**：前几次 `json.loads` 失败，最后成功解析出 `{"location": "Tokyo"}`。真实实现（Rust 的 `ArgumentsWriter`）更聪明，会增量输出字符串值。

- **知识点 5：共享内存（LDS）与批量加载**
  - **是什么**：GPU 的 Shared Memory（AMD 叫 LDS）是片上高速缓存。把数据先批量搬到 LDS，再从 LDS 分发给计算单元，比反复读显存快得多。
  - **为什么重要**：#64153 把 TP8 场景下 K2 的 X8 拷贝"批量化"，就是先攒一批再一次性写 LDS，减少同步开销。这是手写高性能 kernel 的经典套路。
  - **动手试试**（Triton 版，理解"批量 vs 逐元素"）：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def copy_kernel(src, dst, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    N = 1 << 20
    src = torch.randn(N, device="cuda")
    dst = torch.empty_like(src)
    copy_kernel[(triton.cdiv(N, 1024),)](src, dst, N, BLOCK=1024)
    assert torch.equal(src, dst); print("ok")
    ```
  - **预期结果**：打印 `ok`。把 `BLOCK` 改成 256/512/2048 用 `ncu` 或 `torch.profiler` 对比带宽，你会看到 block 大小对吞吐的影响。

---

## 🔥 重要更新

1. **#66660 [Perf][EPLB] 批量拷贝 rebalanced experts**：MoE 重平衡从"逐行拷贝"变成"一次 launch 拷一个权重"，rebalance 时的卡顿大幅降低，对 DeepSeek/GLM 这类大 MoE 服务是直接体感提升。
2. **#66257 [Security] Kimi K3 强制原生 chat encoding**：修复了一个潜在安全/正确性问题——Kimi K3 模式下如果回退到 Jinja 模板渲染，可能产生不符合模型预期的 prompt。涉及安全，值得所有部署 Kimi 的人关注。
3. **#63753 [Fast Start] 通过 D2D 传输预热编译缓存**：新节点可以从其他副本"拉"编译好的 kernel 缓存（经 Mooncake Store），冷启动时间大幅缩短——多副本部署的福音。
4. **#63102 [Perf][GLM-5.3-Flash] GB300 TP4 专用 decode**：为 GB300 单 tray（4 卡）写了一套 CuTe DSL 专用 decode kernel，64 token 以内的 decode 走快路径，长上下文/大 batch 回退。说明"硬件专用优化"路线在加速。

---

## 📋 逐条解读

### 1. [Perf][EPLB] Copy rebalanced experts back from the buffer in one launch per weight (#66660)
- **代码层面**：改 `vllm/distributed/eplb/rebalance_execute.py`，新增 `copy_rows` 函数（用 Triton 实现），把原来"每行一次 CUDA copy"改成"一个权重一次 kernel launch"；`eplb_state.py` 在 `add_model` 时预热编译这个 kernel，避免服务中途第一次 rebalance 时卡顿。
- **新手概念课堂**：想象你有 100 个快递要送到 100 个柜子，逐行拷贝 = 每送一个开一次车门；批量拷贝 = 一次装车全送。GPU 上"开一次车"（kernel launch）有固定开销，所以批量更省。
- **对你有什么影响**：跑 MoE 模型时，周期性 rebalance 的卡顿会明显减少；如果你的服务有"每隔几分钟抖一下"的现象，升级后可能消失。

### 2. [Perf][DSv4.1] CuTe DSL mHC boundary for decode batches up to 336 tokens (#66652)
- **代码层面**：在 `vllm/models/deepseek_v41/nvidia/ops/mhc.py` 新增 `mhc_boundary_weight`，把 DeepSeek V4.1 的 mHC（多头压缩）模块的"边界权重"融合进 CuTe DSL kernel；测试文件重构了正确性测试，覆盖 0/1/17/128/1024 token 等边界。
- **新手概念课堂**：CuTe DSL 就像"用乐高积木拼 GPU 指令"，能精确控制数据在寄存器/SMEM 里的摆放，比 PyTorch 算子快很多。mHC 是 DeepSeek 的一种注意力变体，"边界"指序列首尾的特殊处理。
- **对你有什么影响**：DeepSeek V4.1 在 NVIDIA 卡上 decode 更快（尤其是 batch ≤336 的对话场景）；普通用户感知为"回复更快"。

### 3. [ROCm] AITER bump to v0.1.29 (#66596)
- **代码层面**：`docker/Dockerfile.rocm_base` 和 `Dockerfile.rocm_base_gfx1250` 里 `AITER_BRANCH` 从 `v0.1.28.post1` 升到 `v0.1.29`；测试里把 CUDA graph capture 的 shape 写死成 `(256, 8192)`（避免依赖 test_cases 顺序）。
- **新手概念课堂**：AITER 是 AMD 的"cuBLAS + cuDNN + FlashAttention"合集，版本升级通常带来新 kernel 和 bug 修复。Dockerfile 里的 `ARG` 就是"构建参数"，改一行就换依赖版本。
- **对你有什么影响**：用 AMD MI300/MI355 的用户，升级 vLLM 后自动获得 AITER 新版本的性能/正确性修复；NVIDIA 用户无感。

### 4. [Fast Start] Seed compiled caches through the D2D transport (#63753)
- **代码层面**：改 `docs/features/preload.md` 文档和 `vllm preload` 的 CLI 逻辑，新增 `--warmup-cache-stores mooncake` 选项；编译缓存被打包成 content-addressed bundle（`compile_cache/<sha256>/bundle.tar`），通过 D2D（device-to-device / daemon-to-daemon）控制面传描述符，数据面从共享 Store 拉取。
- **新手概念课堂**：就像"新员工入职"，老员工（源副本）把"工作手册"（编译缓存）放到公司云盘（Mooncake Store），新员工按哈希值去下载，而不是让老员工一对一传——更快、更可靠。
- **对你有什么影响**：多副本部署时，新节点冷启动从"几分钟编译"降到"几十秒下载"；需要配置 Mooncake，不是默认开启。

### 5. [Perf][GLM-5.3-Flash] Specialized decode for GB300 TP4 (#63102)
- **代码层面**：新增 `vllm/models/glm5next/nvidia/specialized_decode/` 目录和 `decode_backend=latency_specialized` 配置项；CI 加了 B200 4 卡测试；recipe YAML 更新 GB300 验证状态和使用说明。启动时会检查一堆前置条件（SM100 家族、TP4、FULL CUDA graph、FP8 KV cache、无 LoRA 等），不满足就拒绝启动。
- **新手概念课堂**：这是"专用赛道"策略——通用 kernel 像家用轿车，什么路都能开但不够快；专用 kernel 像 F1 赛车，只在特定赛道（GB300 TP4）跑得飞快，其他场景直接拒绝。
- **对你有什么影响**：有 GB300 4 卡的用户，加一个配置项就能让短 decode（≤64 token）快很多；其他硬件不受影响。

### 6. [Core][Metrics] Collect iteration details during scheduling (#61424)
- **代码层面**：`vllm/v1/core/sched/scheduler.py` 里把原来 `scheduled_encoder_input_stats` 重构成更通用的 `IterationDetails`，在 `schedule()` 时收集（包括 encoder inputs、recomputation 算作 prefill 等），并加了开关 `enable_logging_iteration_details`。
- **新手概念课堂**：这是"可观测性"改造——把"每次调度迭代发生了什么"结构化记录，方便后续做性能分析和计费。类似给快递系统加"每单轨迹"。
- **对你有什么影响**：开启后日志/metrics 更详细，便于排查"为什么这个请求慢"；默认关闭，不影响性能。

### 7. Revert "[ZenCPU] Add zen5 CPU tests reusing the pushed zen image (#60824)" (#66737)
- **代码层面**：删除 `.buildkite/hardware_tests/zen.yaml`，从 `cpu.yaml` 里移除 zen5 相关依赖和脚本引用。
- **新手概念课堂**：Revert 就是"撤销上一个 PR"。CI 配置出问题（比如镜像拉不到、测试不稳定）时，先回滚保平安，再慢慢修。
- **对你有什么影响**：普通用户无感；贡献者如果依赖 zen5 CI，需要等重新提交。

### 8. [ROCm][Bugfix] Remove step-path GPU syncs in EPLB skip-rearrange, AITER GDN prefill metadata, Moondream3 BOS and INT4 RHT signs (#66399)
- **代码层面**：`vllm/_aiter_ops.py` 里把 GDN prefill metadata 的构建从"在 GPU 上 blocking H2D"改成"CPU 上构建 + 一次 pinned 非阻塞拷贝"；EPLB skip-rearrange 路径去掉隐式同步；Moondream3 的 BOS 处理和 INT4 RHT 符号修复。
- **新手概念课堂**：`pinned memory`（锁页内存）是 CPU 内存里"不会被换出到磁盘"的区域，GPU 可以直接 DMA 读写，比普通内存快。`async_tensor_h2d` 就是"异步拷到 GPU"，不阻塞 CPU。
- **对你有什么影响**：AMD 平台上 GDN（Gated DeltaNet）模型 prefill 更快；Moondream3 和 INT4 量化模型的正确性修复。

### 9. [Bugfix] Make fused MoE expert weights contiguous before GPU copy in Llama-4 (#65987)
- **代码层面**：`vllm/model_executor/models/llama4.py` 的 `load_moe_expert_weights` 里，在 `weight_loader` 之前对 `new_loaded_weight` 调 `.contiguous()`，且特意放在"最终切片后"而不是 transpose 后，避免拷贝过大。
- **新手概念课堂**：见知识点 2。这里还体现了"优化位置"的讲究——在最小的张量上做 contiguous，省内存省时间。
- **对你有什么影响**：Llama-4 加载速度提升（尤其 MoE 版本），启动更快。

### 10. [Rust Frontend] Stream DeepSeek DSML tool-call arguments (#66529)
- **代码层面**：`rust/src/parser/src/tool/arguments.rs` 新增 `ArgumentsWriter` 结构体，管理 JSON 语法（分隔符、键、引号、转义、重复键检测），支持字符串值"边到边转义输出"，其他类型"闭合时整体输出"。
- **新手概念课堂**：见知识点 4。`ArgumentsWriter` 就像"边听写边排版"的秘书，听到一个词就写一个词，而不是等整段话说完。
- **对你有什么影响**：用 DeepSeek 做 Agent 时，工具调用参数可以流式到达前端，用户更早看到"正在调用 XX 工具"。

### 11. [Rust Frontend] Write tool-call arguments in the template's JSON style (#66528)
- **代码层面**：`rust/src/chat/tests/roundtrip.rs` 把 `json_fmt`（格式化函数）换成 `JsonStyle`（枚举：Compact/Spaced），测试断言从 `{"location":"Tokyo"}` 改成 `{"location": "Tokyo"}`，保证解析出的参数与模板渲染的字节级一致。
- **新手概念课堂**：JSON 的 `{"a":1}` 和 `{"a": 1}` 语义相同但字节不同。Roundtrip 测试要求"渲染→解析→再渲染"完全一致，否则缓存/对比会失效。
- **对你有什么影响**：工具调用参数的格式更稳定，前端解析不用兼容多种空格风格。

### 12. [Refactor] Clean up MRv1 remaining stuff (#66661)
- **代码层面**：删除 `use_v2_model_runner=True` 参数、清理注释里对 `gpu_model_runner.py:1284` 的引用、删除 `suffix_decoding_max_spec_factor` 的冗余校验等。
- **新手概念课堂**：技术债清理——V1/V2 模型 runner 合并后，遗留的开关和注释要删掉，否则新人会困惑"这个参数还有用吗"。
- **对你有什么影响**：无功能变化；代码更干净，贡献者读代码更轻松。

### 13. [Bugfix][Model] Avoid host synchronization in dots.ocr vision encoder (#66390)
- **代码层面**：`vllm/model_executor/models/dots_ocr.py` 里去掉视觉编码器中的 `.item()`/`.cpu()` 等同步操作；新增测试 `test_dots_ocr_vision_without_host_sync`；文档加了 `VLLM_GPU_SYNC_CHECK=error` 的使用说明。
- **新手概念课堂**：见知识点 3。视觉编码器里如果为了"动态决定 grid 大小"而读 tensor 值，就会强制同步，拖慢整个 pipeline。
- **对你有什么影响**：dots.ocr 多模态推理更快；同时你多了一个排查同步问题的工具。

### 14. [Frontend] Honour --host for the run-batch Prometheus metrics server (#51859)
- **代码层面**：`vllm/entrypoints/cli/run_batch.py` 里 `start_http_server(port=args.port, addr=args.host or args.url)`，优先用 `--host`，回退到已废弃的 `--url`。
- **新手概念课堂**：CLI 参数演进——新参数 `--host` 替代旧 `--url`，但保留向后兼容（fallback）。
- **对你有什么影响**：用 `vllm run-batch` 且指定 `--host 0.0.0.0` 时，Prometheus 指标现在真的监听在所有网卡上（之前被忽略）。

### 15. [Perf][KV Connector][ROCm] MoRIIO WRITE: skip blocks the decoder prefix cache already holds (#65822)
- **代码层面**：MoRIIO KV connector 的 WRITE 模式新增 `num_skip_blocks` 参数，通知对端"前 N 个 block 我本地 prefix cache 已有，不用传"；测试覆盖 0/1/2 个 cached block 的场景。
- **新手概念课堂**：Prefix caching 是"相同前缀的请求复用 KV cache"。跨节点传输时，如果对端已有这些 block，再传就是浪费带宽。
- **对你有什么影响**：AMD 多节点 PD 分离部署时，KV 传输量减少，首 token 延迟降低。

### 16. [ROCm][Perf][DSv4.1] Mono: batch TP8 K2's X8 copies, load its push's peer bases late (#64153)
- **代码层面**：`vllm/models/deepseek_v41/amd/mono/stages/moe.py` 里把 `load_x8_rows` 的逐元素循环改成 `load_x8_rows_all` 批量路径（条件从 `tuned` 扩展到 `push_bases is not None and not GFX942`）；`layer.py` 里 `push_bases` 的启用条件从 `tp == 4` 扩到 `tp in (4, 8)`。
- **新手概念课堂**：见知识点 5。"peer bases" 是 P2P（GPU 间直接通信）的地址基址，晚加载（late load）可以让编译器更好地调度指令。
- **对你有什么影响**：DeepSeek V4.1 在 AMD TP8 部署时 MoE 阶段更快。

### 17. [ROCm][Perf][DSv4.1] Mono: tuned K1 at TP4 (#63893)
- **代码层面**：`MonoBuild` 新增 `tuned_front` 属性，TP4 下所有层的 K1（包括 DSpark draft）走 tuned 路径；测试里 `generic` 参考实现的条件从 `tp == 2` 改成 `runner.build(1, 2).tuned_front`；README 更新说明。
- **新手概念课堂**："tuned build" 是针对特定 shape/硬件手工调优的 kernel 变体，与"generic build"（通用但慢）对应。测试要求两者 bit-exact，保证优化不改变数值。
- **对你有什么影响**：DeepSeek V4.1 在 AMD TP4 部署时 attention 前端更快。

### 18. [ROCm][DSv4.1] Wire AITER MegaMoE into the V4.1 model (#66644)
- **代码层面**：`vllm/config/kernel.py` 文档更新（`aiter_mega_moe` 现在支持 V4 和 V4.1）；`vllm/models/deepseek_v41/amd/dspark.py` 和 `model.py` 引入 `finalize_mega_moe_weights`，在 `process_weights_after_loading` 里调用；`routed_experts_prefix` 根据是否用 MegaMoE 切换。
- **新手概念课堂**：MegaMoE 是 AITER 的"融合 kernel"——把 dispatch（分发 token 到专家）、GEMM（专家计算）、combine（合并结果）三步合成一个 kernel，减少中间显存读写。
- **对你有什么影响**：AMD gfx950 + DP+EP + MoRI 后端部署 DeepSeek V4.1 时，可以开 `moe_backend=aiter_mega_moe` 获得更好性能。

### 19. [Misc] Normalize filename by replacing typing with types. (#66693)
- **代码层面**：把 `vllm/entrypoints/pooling/typing.py` 重命名为 `types.py`，所有 import 同步更新（`from ...typing import` → `from ...types import`）。
- **新手概念课堂**：`typing` 是 Python 标准库模块名，项目里用自己的 `typing.py` 容易和标准库混淆（尤其做相对导入时）。改成 `types` 更清晰（虽然 `types` 也是标准库，但语义上更偏"数据结构定义"）。
- **对你有什么影响**：无功能变化；如果你有代码 import 了 `vllm.entrypoints.pooling.typing`，需要改成 `.types`（破坏性变更，但影响面小）。

### 20. [Security] Require native chat encoding in Kimi K3 mode (#66257)
- **代码层面**：`vllm/renderers/kimi_k3.py` 里，当 `tokenizer_mode="kimi_k3"` 时如果检测到会回退到 Jinja 模板渲染，直接抛 `VLLMValidationError("requires native chat encoding")`；新增测试覆盖同步渲染和 render server 两条路径。
- **新手概念课堂**：Kimi K3 有自己的原生 chat 编码方式（不是标准 Jinja 模板），如果静默回退到 Jinja，可能生成模型不认识的 prompt 格式，导致输出异常甚至安全问题（prompt injection 绕过）。这是"fail fast"原则——宁可报错也不静默降级。
- **对你有什么影响**：部署 Kimi K3 时如果配置不对，启动/首次请求会明确报错，而不是"看起来能跑但结果不对"。

## sglang
## 💡 今日关键词

- **关键词：增量流式工具调用（Incremental Streaming Tool Calls）**
  - 一句话通俗解释：大模型在调用工具（如"查天气""写文件"）时，参数可以像打字一样一个字一个字地流式返回给用户，而不是等全部生成完才一次性吐出。
  - 为什么社区在关注：Agent 场景爆发，工具调用的首字延迟（TTFT）直接影响体验；Kimi-K3 这类长参数工具调用尤其明显。这说明推理框架正从"文本流式"走向"结构化流式"。
  - 对新手意味着什么：如果你在做 Agent/RAG 应用，理解流式解析器（parser）和增量状态机是必备技能，这是 LLM 应用层最容易被忽视但最影响体验的环节。

- **关键词：参数校验与 OpenAI 兼容性（Parameter Validation）**
  - 一句话通俗解释：服务端在生成前先检查 `temperature`、`max_tokens`、`logit_bias` 等参数是否合法，不合法直接返回 HTTP 400，而不是生成到一半才崩。
  - 为什么社区在关注：SGLang 想"无缝替换 OpenAI"，兼容性细节（哪些限制该抄、哪些不该抄）成为争议点。这次明确"故意不强制 OpenAI 的 -100~100 限制"，体现工程取舍。
  - 对新手意味着什么：学会区分"协议兼容"和"语义兼容"，写 API 服务时知道哪些校验该前置、哪些该放宽。

- **关键词：异构后端适配（AMD / NPU / NVIDIA 多后端）**
  - 一句话通俗解释：同一份推理代码要能在 AMD MI355X、昇腾 NPU、NVIDIA Blackwell 等不同芯片上跑，还要压榨各自硬件特性（融合算子、FP8、Triton 配置）。
  - 为什么社区在关注：今天 20 个 commit 里至少 5 个是 AMD/NPU 相关（Qwen3-Next 融合 AR、gfx950 split-K、Ascend mixed-chunk、MI35x cookbook）。说明"一芯独大"正在被打破，异构计算成主流。
  - 对新手意味着什么：只懂 CUDA 不够了，Triton + HIP + Ascend C 的跨平台调优能力是稀缺技能。

- **关键词：CUDA Graph 与静态图捕获（Graph Capture）**
  - 一句话通俗解释：把一整串 GPU 操作"录下来"变成一张图，下次直接回放，省掉 CPU 调度开销，是低延迟推理的杀手锏。
  - 为什么社区在关注：今天有 3 个 commit 涉及 graph（full cuda graph 保留输入、DSpark 延迟 KV 写入、paged experts 无 host sync）。说明"decode 阶段极致低延迟"是当前性能主战场。
  - 对新手意味着什么：理解 CUDA Graph 的"捕获期 vs 回放期"生命周期，是读懂现代推理框架的钥匙。

- **关键词：量化内核的正确性（Quantization Kernel Correctness）**
  - 一句话通俗解释：4bit/8bit 量化能省显存、提速度，但内核里一个没加的 mask 就可能导致非法内存访问或算错。
  - 为什么社区在关注：今天同时有 MoE W4A16 越界修复和 block-fp8 GEMM 配置新增，说明量化已是生产标配，但正确性 bug 依然高发。
  - 对新手意味着什么：写 Triton kernel 时，"K 方向尾部（K tail）"是最经典的踩坑点，必须养成加 mask 的习惯。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：流式工具调用的增量解析（Streaming Tool-Call Parsing）
- **是什么**：模型输出 `{"name": "get_weather", "arguments": {"city": "北...` 时，解析器要能识别"参数还没闭合"，先不急着做完整性校验，而是把已生成的部分增量推给客户端。
- **为什么重要**：如果解析器把"参数未完成"误判为"参数缺失"，就会触发 `_check_for_unstreamed_tool_args` 的兜底逻辑，导致重复发送或报错。Commit 1 正是加了 `incomplete_tool_call_indices` 集合来标记"这个工具调用还没闭合"。
- **动手试试**：
```python
# 模拟一个极简增量 JSON 参数解析器
import json

class IncrementalArgParser:
    def __init__(self):
        self.buf = ""
        self.incomplete = set()

    def feed(self, chunk: str, tool_idx: int):
        self.buf += chunk
        try:
            json.loads(self.buf)
            self.incomplete.discard(tool_idx)
            return "COMPLETE", self.buf
        except json.JSONDecodeError:
            self.incomplete.add(tool_idx)
            return "PARTIAL", self.buf

p = IncrementalArgParser()
for c in ['{"city":', ' "Bei', 'jing"}']:
    status, buf = p.feed(c, 0)
    print(status, buf, "incomplete=", p.incomplete)
```
- **预期结果**：前两次输出 `PARTIAL` 且 `incomplete={0}`，最后一次输出 `COMPLETE` 且集合为空。你就理解了 Commit 1 里"为什么需要记录未完成索引"。

### 知识点 2：Triton 内核的 K 方向尾部掩码（K-Tail Masking）
- **是什么**：矩阵乘 `A[M,K] @ B[K,N]` 里，K 通常不是 BLOCK_SIZE_K 的整数倍，最后一个 tile 会越界读内存。必须用 `mask=k < K` 把越界部分置零。
- **为什么重要**：Commit 14 修的就是 `fused_moe_kernel_gptq_awq` 里 `b = tl.load(b_ptrs)` 少了 mask，导致非法内存访问（IMA）。这类 bug 在特定 shape 下才触发，极难复现。
- **动手试试**：
```python
import torch, triton, triton.language as tl

@triton.jit
def bad_gemm(A, B, C, M, N, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        k = k0 + offs_k
        a = tl.load(A + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.)
        # 故意不加 mask，K 不是 BLOCK 整数倍时会越界
        b = tl.load(B + k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M, N, K = 64, 64, 100   # K=100 不是 32 的倍数
A = torch.randn(M, K, device='cuda'); B = torch.randn(K, N, device='cuda')
C = torch.empty(M, N, device='cuda')
bad_gemm[(M//64,)](A, B, C, M, N, K, BLOCK=32)
torch.cuda.synchronize()
print("如果没崩，对比一下:", torch.allclose(C, A @ B, atol=1e-2))
```
- **预期结果**：把 `b = tl.load(...)` 改成带 `mask=(k[:, None] < K)` 的版本后，结果才与 `A @ B` 一致；不加 mask 可能触发 `CUDA error: an illegal memory access`。这就是 Commit 14 的修复本质。

### 知识点 3：CUDA Graph 捕获期的"输入生命周期"
- **是什么**：CUDA Graph 回放时，内核会读捕获时绑定的显存地址。如果捕获用的 dummy 输入被 Python GC 回收，回放就会读到已释放内存。
- **为什么重要**：Commit 16 加了 `self._capture_inputs[shape_key] = capture_inputs`，就是让这些输入"活到 graph reset 为止"。这是典型的"Python 引用 vs GPU 指针"陷阱。
- **动手试试**：
```python
import torch, weakref

class GraphHolder:
    def __init__(self):
        self.graphs = {}
        self.capture_inputs = {}

    def capture(self, key, x):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = x * 2
        self.graphs[key] = g
        # 关键：保留输入，否则 x 被 GC 后回放读脏数据
        self.capture_inputs[key] = x
        return y

h = GraphHolder()
x = torch.ones(4, device='cuda')
ref = weakref.ref(x)
h.capture("k", x)
del x
print("输入是否还活着:", ref() is not None)   # True
```
- **预期结果**：输出 `True`。如果你把 `self.capture_inputs[key] = x` 删掉，再 `del x` 后 `ref()` 会变 `None`，回放时可能读到垃圾数据或崩溃。

### 知识点 4：分块量化 GEMM 的 tile 配置自动调优
- **是什么**：Triton 的 block-fp8 GEMM 需要为每个 (M, N, K, GPU 型号) 选一组 `BLOCK_SIZE_M/N/K`、`num_warps`、`num_stages`。Commit 6 就是为 RTX PRO 5000 新增了一整套 JSON 配置。
- **为什么重要**：同一份 kernel，tile 选错性能差 2-3 倍。生产框架会"离线 autotune 一次，把结果存成 JSON，运行时查表"。
- **动手试试**：
```python
import torch, triton, triton.language as tl

@triton.jit
def fp8_gemm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :], mask=offs_k[None, :] < K, other=0.)
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :], mask=offs_k[:, None] < K, other=0.)
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M, N, K = 1792, 5120, 5120
A = torch.randn(M, K, device='cuda', dtype=torch.float8_e4m3fn)
B = torch.randn(K, N, device='cuda', dtype=torch.float8_e4m3fn)
C = torch.empty(M, N, device='cuda', dtype=torch.float32)

import time
for BM, BN, BK in [(16, 64, 32), (32, 64, 32), (64, 64, 32)]:
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    # warmup
    for _ in range(3): fp8_gemm[grid](A, B, C, M, N, K, BM, BN, BK)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): fp8_gemm[grid](A, B, C, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    print(f"BM={BM} BN={BN} BK={BK}: {(time.time()-t0)/20*1e3:.2f} ms")
```
- **预期结果**：不同 tile 组合耗时明显不同（通常越大 tile 在大 M 下越快，但小 M 下反而慢）。你会直观理解 Commit 6 为什么按 batch size（"1"、"2"、"4"…）分别存配置。

### 知识点 5：并行策略自动降级（Ulysses → KV Gather）
- **是什么**：序列并行（Ulysses）把 attention head 切到多卡，要求 head 数能被卡数整除。若不能整除（如 Z-Image 30 heads / 4 卡），就自动降级为 KV Gather（每卡算全部 head，只切 KV）。
- **为什么重要**：Commit 15 加了 `if self._dit_num_attention_heads() % self.sp_degree:` 的判断，避免启动时 all-to-all 直接报错。这是"优雅降级"的经典设计。
- **动手试试**：
```python
def choose_sp_strategy(num_heads: int, sp_degree: int):
    if num_heads % sp_degree == 0:
        return f"Ulysses (heads {num_heads} 可被 {sp_degree} 整除)"
    return f"KV Gather (heads {num_heads} 不能被 {sp_degree} 整除，降级)"

for heads, sp in [(32, 4), (30, 4), (30, 8), (48, 8)]:
    print(f"heads={heads}, sp={sp} -> {choose_sp_strategy(heads, sp)}")
```
- **预期结果**：
```
heads=32, sp=4 -> Ulysses (heads 32 可被 4 整除)
heads=30, sp=4 -> KV Gather (heads 30 不能被 4 整除，降级)
heads=30, sp=8 -> KV Gather (heads 30 不能被 8 整除，降级)
heads=48, sp=8 -> Ulysses (heads 48 可被 8 整除)
```
这就是 Commit 15 的核心决策逻辑。

---

## 🔥 重要更新

- **Commit 1（Kimi-K3 增量流式工具调用）**：直接影响 Agent 场景的首字延迟和工具调用体验，是"结构化流式"的重要里程碑，所有做 Agent 应用的开发者都该关注。
- **Commit 2（OpenAI 参数校验）**：明确划定了 SGLang 对 OpenAI 兼容的边界（哪些限制抄、哪些放宽），并让非法参数在生成前就返回 400，避免流式请求中途报错，是 API 稳定性的关键修复。
- **Commit 14（MoE W4A16/W8A16 越界修复）**：非法内存访问是生产环境最难排查的 bug 之一，这个修复直接消除了量化 MoE 在特定 K 形状下的崩溃风险。
- **Commit 16（CUDA Graph 保留捕获输入）**：修复了 full graph 回放时输入被 GC 导致的潜在脏读，属于"平时不出事、出事查三天"的隐性 bug，对低延迟推理稳定性至关重要。

---

## 📋 逐条解读

### 1. [Feat] kimi-k3: Stream tool-call arguments incrementally (#48775)
- **代码层面**：改 `serving_chat.py` 的 `_check_for_unstreamed_tool_args`，新增"如果该 tool_index 在 `incomplete_tool_call_indices` 里就跳过兜底检查"；`core_types.py` 的 `ToolCallItem` 加了 `source_text` 字段（`exclude=True` 不序列化）；`kimik3_detector.py` 把原来"一次性 `_parse_calls` 全解析"改成有状态的 `_stream_calls()`，用 `_tool_pos`、`_in_call`、`_argument_value` 等成员变量逐字符推进。
- **新手概念课堂**：想象你在餐厅点菜，服务员以前要等你把整句话说完才记单，现在你说"一份…宫保…鸡丁"，他边听边记，最后确认。`incomplete_tool_call_indices` 就是服务员的"这单还没记完"小本本。
- **对你有什么影响**：用 Kimi-K3 做 Agent 时，工具调用参数会像普通文本一样流式返回，客户端可以边收边展示"正在调用 get_weather(city=北…"，体验更丝滑；同时不会因为参数未闭合而误触发"参数缺失"告警。

### 2. [Fix] openai: Validate sampling bounds and preserve chat output limits (#44047)
- **代码层面**：文档 `openai_api_completions.mdx` 和 `sampling_params.mdx` 新增"参数校验"章节；代码侧对 `n`（1~128）、`max_tokens`/`max_completion_tokens`（≥0、拒绝 bool）、`temperature`（有限非负）、`top_logprobs`（≥0 且 ≤vocab_size）、`logit_bias`（有限且 |v|≤1e30）做前置校验，非法即 HTTP 400；`max_completion_tokens` 优先于 `max_tokens`。
- **新手概念课堂**：像机场安检——以前是登机口才查违禁品（生成到一半才报错），现在进安检门就查（请求进来就 400）。`logit_bias` 放宽到 1e30 是因为 sampler 内部会除以 ≥1e-6 的温度，太大才溢出 float32。
- **对你有什么影响**：传错参数会立刻收到清晰的 400 错误而不是流式响应中途断流；同时 SGLang 故意不抄 OpenAI 的 `temperature≤2`、`logit_bias∈[-100,100]` 限制，给了更大自由度。

### 3. [Perf] [AMD] Qwen3-Next: consume the fused AR+norm+quant output (#44566)
- **代码层面**：`qwen3_next.py` 新增 `_linear_accepts_fp8_tuple` 和 `_select_fused_ar_input_for_linear`，让 `in_proj_qkvz`（FP8 block-quant 线性层）直接吃 `(fp8, scale)` 元组，而 `in_proj_ba`（小门控投影，未量化）吃 bf16 旁路输出；`_forward_input_proj` 里对 tuple 输入做分流。
- **新手概念课堂**：像快递分拣中心——上游把"已打包好的 FP8 包裹"和"原始 bf16 物品"一起送下来，下游每个收货窗口（线性层）按自己能不能收 FP8 包裹来决定拿哪个。融合 AR（AllReduce）+ norm + quant 一步完成，省掉中间显存往返。
- **对你有什么影响**：在 AMD MI 系列上跑 Qwen3-Next 会更快，尤其是 decode 阶段；普通用户感知是吞吐提升、延迟下降。

### 4. [Feat] [NPU] attention: enable mixed-chunk for pure MLA on the ascend backend (#47857)
- **代码层面**：`ascend_backend.py` 的 `forward_mixed` 删掉了"MLA 且无 topk_indices 就抛 NotImplementedError"的拦截，直接复用 `forward_extend`；测试 `test_npu_ascend_backend.py` 扩展 `_check_prefix` 支持 `page_size>1` 的页粒度 gather 模拟，并新增 `prefix_token_indices` 参数。
- **新手概念课堂**：mixed-chunk 是"prefill 和 decode 混在一个 batch 里跑"，像公交车同时上客和下客。以前昇腾上 MLA 模型不支持这种混跑，现在解禁了。测试里的 page_size 模拟的是"按页取 KV"，页尾可能有垃圾数据，需要 row_index 精确索引。
- **对你有什么影响**：在昇腾 NPU 上部署 DeepSeek 类 MLA 模型时，可以开启 `--enable-mixed-chunk` 提升吞吐，不再被强制禁用。

### 5. [Fix] dspark: Write deferred draft KV only for carried requests (#49436)
- **代码层面**：把 `dspark_deferred_commit.py` 重命名为 `dspark_deferred_draft_kv.py`，文档和注释从"context-KV commit"改为"draft KV store"；`environ.py` 注释同步更新；核心逻辑改为"只为被 carry 到下一步的请求写延迟的 draft KV"，避免给已结束请求写脏数据。
- **新手概念课堂**：像接力赛——只有真正把棒子交给下一棒的选手才需要"预备起跑"（写 KV），已经退赛的选手不该占跑道。DSpark 是投机解码的 draft 模型，延迟写 KV 是为了和 verify 阶段重叠。
- **对你有什么影响**：使用 DSpark 投机解码时，长尾请求结束后的资源浪费和潜在 KV 污染被消除，稳定性提升。

### 6. [Perf] [NVIDIA] fp8: Add Triton block-fp8 GEMM configs for RTX PRO 5000 (#45263)
- **代码层面**：新增一批 JSON 配置文件，路径形如 `configs/N=1792,K=5120,device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell,dtype=fp8_w8a8,block_shape=[32, 32].json`，按 batch size（"1"、"2"、"4"…）存 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`。
- **新手概念课堂**：这是"离线调参、在线查表"——像给不同体型的顾客准备不同尺码的衣服，来一个 batch 就按尺码表直接拿，不用现场量体裁衣（autotune）。
- **对你有什么影响**：在 RTX PRO 5000（Blackwell）上跑 FP8 量化模型，GEMM 会自动命中更优 tile，吞吐提升；其他卡不受影响。

### 7. [Feat] score/embedding: Add raw embeddings, score rids, encode input_ids, and schedule-decision logging (#44152)
- **代码层面**：新增 `--disable-normalize-embedding`（关闭 Pooler 的 L2 归一化，返回原始向量）和 `--log-schedule-decisions`（记录每次 prefill 调度决策：选中/留队、running/preempted 数、KV 空闲 token、前 16 个 rid）；涉及 `field_order.py`、`fields/model.py`、`fields/observability.py`、`log_parser.py`。
- **新手概念课堂**：`disable-normalize-embedding` 像"果汁不加水"——默认 embedding 会做 L2 归一化（单位化），关掉后拿到原始浓度向量。`log-schedule-decisions` 像给调度器装"行车记录仪"，排查"为什么这个请求一直排队"时有据可查。
- **对你有什么影响**：做向量检索/相似度任务时，某些场景（如内积检索）需要原始 embedding；调优调度时可以用新日志定位饥饿请求。

### 8. [Fix] CODEOWNERS: Restore prefill-only test ownership (#49470)
- **代码层面**：`.github/CODEOWNERS` 里把 `/test/registered/` 的 owner 范围收窄为 `/test/registered/prefill_only`，避免误把整个 registered 测试目录都归给 prefill-only 团队。
- **新手概念课堂**：CODEOWNERS 是 GitHub 的"自动指派审稿人"规则，像公司门牌——以前把整层楼都挂上"prefill 组"，现在只挂 prefill 那间办公室。
- **对你有什么影响**：普通用户无感知；贡献者提 PR 时会看到更精准的 reviewer 指派。

### 9. [Fix] rust-server: Pass architectures to ModelConfig in test_rust_server_crash_dump (#49461)
- **代码层面**：测试 `test_rust_server_crash_dump.py` 里构造 `ModelConfig` 时补上 `architectures=None` 字段。
- **新手概念课堂**：Rust 结构体新增字段后，所有构造点都要补——像表格加了一列，所有填表的地方都得填（哪怕是 None）。
- **对你有什么影响**：无用户感知，属于 CI 修复，保证 Rust server 崩溃转储测试能跑通。

### 10. [Doc] [AMD] cookbook: Match the Kimi-K3.1-Pro MI35x cells to InferenceX #4928 and mount the Ionic provider for PD (#49453)
- **代码层面**：`Kimi-K3.1-Pro.mdx` 大幅精简，把原来详细的 MI355X 性能表格（16/32/48/56/92 agents 的 tok/s/GPU、TTFT 等）替换为简短描述和镜像 sha；新增 Ionic provider 挂载用于 PD 分离。
- **新手概念课堂**：cookbook 像"菜谱"，以前写得太细（每道菜几克盐），现在改成"按 InferenceX 标准菜谱来，镜像用这个 sha"——减少维护负担，避免文档和实测漂移。
- **对你有什么影响**：想在 MI35x 上部署 Kimi-K3.1-Pro 的用户，按新文档用指定镜像即可复现，不用再对着过时表格猜配置。

### 11. [Fix] [NVIDIA] jit: Build sm_107a on CC 10.7, keeping sm_100f only for modules that use CUTLASS (#49420)
- **代码层面**：`arch.py` 新增 `MODULE_USES_CUTLASS = ["deep_select_topk"]`，`get_compile_arch(kernel_name="")` 只在 CC 10.7 且模块名在列表里才走 `cutlass_supports_sm107` 分支；`loader.py`、`ninja.py`、`toolchain.py` 都把 `module_args[0]`（模块名）透传给 arch 选择。
- **新手概念课堂**：sm_107a 和 sm_100f 是两种 GPU 指令集"方言"。以前所有 kernel 在 CC 10.7 上都说 sm_100f 方言，但 CUTLASS 库只认 sm_100f，其他 kernel 用 sm_107a 更快。现在按"这个 kernel 用不用 CUTLASS"来选方言。
- **对你有什么影响**：在 CC 10.7（如 RTX PRO 5000 Blackwell）上，非 CUTLASS kernel 会用原生 sm_107a 编译，性能更好；CUTLASS 相关 kernel 保持兼容。

### 12. [Perf] [AMD] batch-invariant: Segmented split-K bf16 matmul for gfx950 (#45599)
- **代码层面**：`batch_invariant_ops.py` 新增 `_segmented_matmul_kernel`，把 K 切成 `_SEGMENT_COUNT=4` 段、每段 `_SEGMENT_BLOCK_K=64`，小 M 时每段独立 program 算 fp32 部分和再按固定顺序归约，大 M 时单 program 内累加；两种路径 bitwise 一致（batch-invariant）。
- **新手概念课堂**："batch-invariant"指同一个输入不管 batch 多大，输出 bit 完全一致——像做菜不管一次做 1 份还是 100 份，每份味道都一样。split-K 是把长 K 维度切给多个 program 并行算再合并。
- **对你有什么影响**：在 AMD gfx950（MI350X）上跑需要 batch-invariant 的场景（如 RL 训练、可复现推理），bf16 matmul 更快且结果稳定。

### 13. [Doc] cookbook: Fix the dead MiniMax-H3 "Learn more" anchors (#46052)
- **代码层面**：`minimax-h3.jsx` 里 5 处 `learnMore: "#7-feature-contracts-and-advanced-recipes"` 改成 `"#9-feature-contracts-and-advanced-recipes"`，因为文档章节编号从 7 变成了 9。
- **新手概念课堂**：锚点链接像书签，章节重排后书签页码要跟着改，否则点"了解更多"跳到空白。
- **对你有什么影响**：看 MiniMax-H3 cookbook 时"Learn more"能正常跳转。

### 14. [Fix] moe: Fix illegal memory access in the W4A16/W8A16 Triton MoE kernel (unmasked K tail) (#48556)
- **代码层面**：`fused_moe_triton_kernels.py` 的 `fused_moe_kernel_gptq_awq` 里 `b = tl.load(b_ptrs)` 改为 `b = tl.load(b_ptrs, mask=k_mask, other=k_other)`；新增测试 `test_fused_moe_gptq_awq_k_tail.py` 专门构造 K 非 BLOCK_SIZE_K 整数倍的场景。
- **新手概念课堂**：矩阵乘的 K 维度像一堵墙，BLOCK_SIZE_K 是每块砖的宽度，墙长不是砖宽整数倍时最后一块砖会"伸出墙外"读到别人的内存——必须用 mask 把伸出部分挡住。
- **对你有什么影响**：使用 GPTQ/AWQ 量化的 MoE 模型（W4A16/W8A16）时，特定 K 形状不再崩溃；这是生产环境稳定性修复。

### 15. [Fix] diffusion: Use K/V gather when auto SP cannot split the DiT's heads (Z-Image on 4 GPUs) (#49225)
- **代码层面**：`server_args.py` 的 `_adjust_parallelism` 新增分支：若 `_dit_num_attention_heads() % sp_degree != 0`，则设 `kv_gather_degree = sp_degree`、`sp_split_auto = True`，跳过 Ulysses；新增 `_dit_num_attention_heads()` 从 pipeline_config 读 head 数；测试新增 Z-Image 30 heads 在 4/8 卡下的用例。
- **新手概念课堂**：Ulysses 并行要求"head 数能被卡数整除"才能把 head 均分到各卡；不能整除就像 30 个苹果分给 4 个人，必然有人多有人少，all-to-all 会失败。KV Gather 是"每卡算全部 head，只切 KV"，不要求整除。
- **对你有什么影响**：Z-Image 在 4 卡或 8 卡上启动不再报错，自动降级到 KV Gather；3 卡（30%3==0）仍用更快的 Ulysses。

### 16. [Fix] cuda_graph: Retain capture inputs for full graph replay (#48860)
- **代码层面**：`full_cuda_graph_backend.py` 新增 `self._capture_inputs: Dict[Any, Any] = {}`，`capture_one` 里存 `self._capture_inputs[shape_key] = capture_inputs`，`cleanup` 里清空；测试用 `weakref` 验证输入在 `graph.reset()` 前不会被 GC。
- **新手概念课堂**：CUDA Graph 回放像"录像回放"，录像里的道具（输入张量）必须一直在片场，否则回放时道具被收走就穿帮（读脏内存）。Python 的 GC 不知道 GPU 还在用这块显存，所以要用 Python 引用"按住"它。
- **对你有什么影响**：使用 full CUDA graph 模式时，长时间运行不再出现偶发的输出错乱或崩溃；这是低延迟推理稳定性的关键修复。

### 17. [Fix] diffusion: Let FLUX.3 Image start with --num-gpus alone (#49224)
- **代码层面**：`server_args.py` 的 `_model_default_uses_cfg` 里把 `getattr(default_params, "guidance_scale", 0) > 1.0` 改为先取 `guidance_scale = getattr(..., None)`，再判 `is not None and > 1.0`；测试新增 `test_num_gpus_without_parallel_flags_resolves`。
- **新手概念课堂**：FLUX.3 的默认 `guidance_scale` 是 `None`（每次请求时由包动态决定），旧代码 `None > 1.0` 在 Python 3 会抛 `TypeError`，导致只传 `--num-gpus 2` 就启动失败。修复后 `None` 被视为"不使用 CFG"，走 SP 路径。
- **对你有什么影响**：FLUX.3 Image 用户只需 `--num-gpus 2` 即可启动，不用再手动指定并行参数。

### 18. [Perf] moe/paged-experts: decide block-wide and skip empty page-ins (#48605)
- **代码层面**：`paged_experts.cuh` 的 `paged_experts_decide_kernel` 从"单 warp"改为"256 线程 block-wide"：用 `extern __shared__ int s_state[]` 把 `expert_slot[E]`、`slot_expert[K]`、`slot_lastuse[K]`、`topk[T]`、`miss[T]` 等 stage 到 shared memory，`__syncthreads()` 后并行处理；victim 搜索读 shared 里的 lastuse 副本，避免每页一次全局往返；空 page-in 直接跳过。
- **新手概念课堂**：以前是"1 个收银员"处理所有决策（延迟 14.88us），现在是"8 个收银员分工"（4.36us），再多到 16 个也没用（瓶颈不在人手）。把状态搬进 shared memory 就像把账本从仓库搬到收银台旁边。
- **对你有什么影响**：MoE 模型 decode 阶段启用 paged experts 时，每步决策延迟大幅下降，CUDA Graph 回放更高效。

### 19. [Test] comfyui: Attach loaded plugin modules to their parents so mock.patch works on Python 3.10 (#49235)
- **代码层面**：`test_comfyui_h3_request.py` 和 `test_comfyui_server_mode.py` 的 `_load` 函数里加 `parent, _, name = module_name.rpartition("."); setattr(sys.modules[parent], name, module)`；`_load_plugin` 里先注册 `core` 模块再 `package.core = core`；`.agents/skills/write-sglang-test/SKILL.md` 同步更新示例。
- **新手概念课堂**：Python 3.10 的 `mock.patch("a.b.c")` 会从 `a` 开始逐层 `getattr`，如果 `b` 没被设为 `a` 的属性（只在 `sys.modules` 里），就找不到。修复就是把"户口本"（sys.modules）和"家庭关系"（parent 属性）都登记上。
- **对你有什么影响**：无用户感知；在 Python 3.10 上跑 ComfyUI 相关测试不再因 mock 失败。

### 20. [Test] gemma4: Cover the multimem.ld_reduce rounding in the AR fusion residual check (#49346)
- **代码层面**：`test_gemma4_ar_fusion.py` 里把 residual 的 `atol` 从 `3e-2` 放宽到 `5e-2`，注释说明 `multimem.ld_reduce`

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)