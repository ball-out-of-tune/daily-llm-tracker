# 🚀 vLLM & SGLang 每日更新 — 2026-10-14

> 自动生成于 2026-10-14 06:40 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：KV Cache 传输（KV Connector / P2P / Mooncake）**
  - **一句话通俗解释**：大模型推理时产生的"记忆"（KV Cache）可以在不同的 GPU/节点之间搬运，让一台机器复用另一台机器已经算好的结果。
  - **为什么社区在关注**：今天有 3 个 commit（#57139、#59625、#60712）都在动 KV 传输相关代码，说明"分布式推理 + 缓存复用"是当下最核心的工程战场。尤其是把注册 peer 的操作从调度线程挪走、支持 sleep 模式、支持 Mamba 缓存，都是在解决真实生产环境的稳定性问题。
  - **对新手意味着什么**：如果你在学分布式系统或推理优化，KV Cache 传输是一条非常值得深入的方向——它同时涉及网络编程、内存管理、调度算法，是难得的综合练兵场。

- **关键词：AMD/ROCm 与 Intel XPU 适配**
  - **一句话通俗解释**：vLLM 不只跑在英伟达 GPU 上，也在努力适配 AMD 和 Intel 的硬件。
  - **为什么社区在关注**：今天有 4 个 commit（#61378、#58555、#61230、#46685）明确针对 ROCm/XPU。这说明"异构计算"已经从口号变成工程现实，厂商在真金白银投入。
  - **对新手意味着什么**：如果你手上只有 AMD 显卡或 Intel 集成显卡，现在也能玩 vLLM 了。掌握多平台适配能力，会让你在求职时更有竞争力。

- **关键词：FP8 / NVFP4 等低精度量化**
  - **一句话通俗解释**：用更少的比特数（8 位、4 位）表示模型权重和激活值，让模型跑得更快、更省显存。
  - **为什么社区在关注**：今天有 3 个 commit（#60992、#56273、#60992）在优化 FP8/NVFP4 的编解码。低精度是推理加速的核心武器，谁先把 kernel 写得更快，谁就能在性能榜上领先。
  - **对新手意味着什么**：量化是入门推理优化的最佳切入点——你不需要重训模型，只需要理解比特表示和 kernel 编程。

- **关键词：CUDA Graph 与图捕获**
  - **一句话通俗解释**：把一串 GPU 操作"录制"成一个图，之后一键重放，省去每次调度的开销。
  - **为什么社区在关注**：今天有 3 个 commit（#60908、#59166、#56716）涉及 CUDA Graph 支持或图填充处理。图捕获是低延迟推理的关键技术。
  - **对新手意味着什么**：CUDA Graph 是"从会用 PyTorch 到会优化推理"的分水岭，值得专门学。

- **关键词：新模型接入（Qwen4Exp / DeepSeek-V4 / Kolibri 1）**
  - **一句话通俗解释**：vLLM 社区在快速支持最新的开源大模型。
  - **为什么社区在关注**：今天有 5 个 commit 涉及 Qwen4Exp、DeepSeek-V4、Kolibri 1 等新模型。模型迭代快，推理框架必须跟上。
  - **对新手意味着什么**：学会"如何给 vLLM 加一个新模型"，是参与开源贡献的绝佳入口。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：KV Cache 跨节点传输（P2P KV Connector）

- **是什么**：把一台机器算好的 KV Cache 通过 RDMA/NVLink 传给另一台机器，避免重复计算。
- **为什么重要**：在 PD 分离（Prefill/Decode 分离）架构中，Prefill 节点算完的 KV 必须传给 Decode 节点。如果传输慢，整个推理链路的延迟就被拖垮。不懂这个，你调不出好的分布式推理性能。
- **动手试试**：
  ```bash
  # 1. 先阅读 vLLM 官方文档
  # https://docs.vllm.ai/en/latest/features/nixl_connector_usage.html
  # 2. 启动一个单机 NIXL connector 示例（需要至少 2 张 GPU）
  vllm serve Qwen/Qwen3-0.6B \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
  # 3. 观察日志中是否有 "registering remote peer" 相关输出
  ```
- **预期结果**：你会看到 vLLM 启动时打印 KV connector 初始化日志，说明 P2P 传输层已就绪。如果只有 1 张卡，可以先用单元测试学习：
  ```bash
  pytest tests/v1/kv_offload/tiering/p2p/test_data_transport.py -v
  ```

### 知识点 2：FP8 e4m3 的低层解码技巧

- **是什么**：FP8 是一种用 8 位表示浮点数的格式，e4m3 表示 4 位指数 + 3 位尾数。
- **为什么重要**：在 sm_89 以下的 GPU 上，硬件不支持直接解码 FP8，必须用位运算模拟。Commit #60992 用 `shf.r.wrap` + `lop3` + `prmt` 等 PTX 指令，一次处理 4 个字节，速度飞快。不懂位运算，你就只能写慢代码。
- **动手试试**：
  ```python
  # 在 Python 里模拟 FP8 e4m3 的解码逻辑
  def decode_fp8_e4m3(code: int) -> float:
      sign = (code >> 7) & 1
      exp = (code >> 3) & 0xF
      mant = code & 0x7
      if exp == 0:
          value = (mant / 8.0) * (2 ** -6)
      elif exp == 0xF and mant == 0x7:
          return float('nan')
      else:
          value = (1 + mant / 8.0) * (2 ** (exp - 7))
      return -value if sign else value

  for c in [0x00, 0x38, 0x3C, 0x7F]:
      print(f"0x{c:02X} -> {decode_fp8_e4m3(c)}")
  ```
- **预期结果**：你会看到 `0x38` 解码为 `1.0`，`0x3C` 解码为 `1.5`，验证了你对 FP8 格式的理解。

### 知识点 3：SSE 流式响应的分块边界问题

- **是什么**：SSE（Server-Sent Events）是流式输出的格式，服务器把文本切成很多小块发过来。但网络传输可能把一次 SSE 事件切成两半。
- **为什么重要**：Commit #51172 修复了一个真实 bug——之前代码对每个 chunk 做 `.strip()`，如果 chunk 边界刚好切在空白处，就会破坏 JSON 解析。这个坑在真实生产环境非常隐蔽。
- **动手试试**：
  ```python
  import json
  # 模拟一个被切成两半的 SSE 事件
  full = 'data: {"choices":[{"text":" hello "}]}\n\n'
  # 错误做法：先 strip 再解析
  chunk1 = full[:15].strip()  # 可能破坏 JSON
  chunk2 = full[15:].strip()
  print("chunk1:", repr(chunk1))
  print("chunk2:", repr(chunk2))
  # 正确做法：先拼接再按 \n\n 切分事件
  buffer = full[:15] + full[15:]
  events = buffer.split("\n\n")
  for ev in events:
      if ev.startswith("data: "):
          print("parsed:", json.loads(ev[6:]))
  ```
- **预期结果**：你会看到 `.strip()` 会吃掉 JSON 里的空格，而按 buffer 拼接再解析则完全正确。

### 知识点 4：CUDA Graph 与图填充（Graph Padding）

- **是什么**：CUDA Graph 把一串 GPU 操作录制成图，之后重放。但图的大小是固定的，如果实际 batch 小于录制大小，多出来的位置就是"填充"。
- **为什么重要**：Commit #59166 修复了 DeepSeek-V4.1 MoE 中图填充位置没有被正确屏蔽的 bug——如果不处理，填充位置会参与计算，导致结果错误。这是 CUDA Graph 使用中最容易踩的坑。
- **动手试试**：
  ```python
  import torch
  # 模拟：录制一个 batch=4 的图，但实际只有 2 个有效 token
  x = torch.randn(4, 8, device='cuda')
  is_padding = torch.tensor([False, False, True, True], device='cuda')

  # 错误做法：填充位置也参与计算
  y_bad = x.sum(dim=1)

  # 正确做法：先屏蔽填充位置
  x_masked = x.clone()
  x_masked[is_padding] = 0
  y_good = x_masked.sum(dim=1)

  print("含填充:", y_bad)
  print("已屏蔽:", y_good)
  ```
- **预期结果**：你会看到后两个位置的输出被正确清零，说明屏蔽生效。

### 知识点 5：MoE 路由的排序与 gather/scatter

- **是什么**：MoE（混合专家）模型里，每个 token 会被路由到几个专家。为了高效计算，需要按专家把 token 分组排序。
- **为什么重要**：Commit #58897 优化了 triton_kernels 3.8 的路由构建，用 `argsort` + `bincount` 重写。理解这个流程，你才能看懂 MoE 的性能瓶颈在哪。
- **动手试试**：
  ```python
  import torch
  # 模拟 4 个 token，每个 token 选 2 个专家
  topk_ids = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2]])
  n_experts = 3
  flat = topk_ids.reshape(-1)
  # 按专家排序
  order = torch.argsort(flat, stable=True)
  # 每个专家的 token 数量
  counts = torch.bincount(flat, minlength=n_experts)
  print("排序索引:", order)
  print("每个专家 token 数:", counts)
  ```
- **预期结果**：你会看到 `order` 把所有属于专家 0 的位置排在最前，`counts` 显示每个专家分到多少 token。这就是 MoE 路由的核心逻辑。

---

## 🔥 重要更新

1. **#59625 MooncakeConnector 支持 sleep 模式**：这是生产环境的关键能力——推理服务可以在空闲时"睡觉"释放显存，需要时再唤醒。对多租户、弹性扩缩容场景意义重大。

2. **#60908 FA4 支持 ALWAYS cudagraph**：FlashAttention 4 现在可以无条件使用 CUDA Graph，意味着长序列推理的延迟会显著降低。这是性能优化的重要里程碑。

3. **#60465 新增 OpenAI Decisions API**：vLLM 开始支持 OpenAI 最新的 Decisions API，让模型输出结构化的判断结果（谓词、选择、评分）。这是 LLM 应用层的新接口范式。

4. **#61510 回滚 DeepGEMM 的 Stable ABI 迁移**：迁移到 Stable ABI 后可能引入了兼容性问题，团队果断回滚。这提醒我们：构建系统的改动风险很高，需要在 CI 里做充分验证。

---

## 📋 逐条解读

### 1. [KVConnector][P2P] Register peers off the scheduler thread (#57139)

- **代码层面**：修改了 `vllm/v1/kv_offload/tiering/p2p/data/nixl.py` 和对应测试。核心是把 `add_remote_peer` 改成异步版本 `add_remote_peer_async`，把 O(num_blocks) 的注册工作从调度线程挪到 worker 线程。同时增加了"注册晚于移除"的保护——如果 peer 已经被移除，晚到的注册会被丢弃并释放句柄。
- **新手概念课堂**：调度线程就像餐厅的前台，如果前台去后厨炒菜（做耗时操作），点单就会卡住。这个 commit 就是把"炒菜"交给后厨（worker 线程），前台继续接单。
- **对你有什么影响**：在 PD 分离场景下，KV 传输的注册不再阻塞调度，整体吞吐会更稳定。普通用户可能感觉不到，但在高并发下延迟抖动会明显减小。

### 2. [Perf][Model][Qwen4Exp] QSA: decode four FP8 e4m3 bytes per register below sm_89 (#60992)

- **代码层面**：修改 `vllm/models/qwen4_exp/nvidia/ops/qsa.py` 中的 `_maybe_dequant` 函数。用内联 PTX 汇编（`tl.inline_asm_elementwise`）替代原来的标量位运算，一次处理 4 个 FP8 字节。
- **新手概念课堂**：这就像原来一个人一次搬一块砖，现在一次搬四块。PTX 汇编是 GPU 的"汇编语言"，能精确控制每个寄存器操作。
- **对你有什么影响**：在老 GPU（sm_89 以下）上跑 Qwen4Exp 模型时，FP8 解码会更快。如果你用的是 H100 等新卡，不受影响。

### 3. [Bugfix][Benchmark] Preserve SSE output across transport chunk boundaries (#51172)

- **代码层面**：删除 `vllm/benchmarks/lib/endpoint_request_func.py` 中三处 `chunk_bytes = chunk_bytes.strip()`。新增了一个参数化测试，验证任意切分点下解析结果都一致。
- **新手概念课堂**：这就像你把一封信撕成两半再拼起来，如果撕在字中间，拼起来就错了。SSE 流式响应也是同理，必须在拼接后再解析。
- **对你有什么影响**：跑 benchmark 时，如果模型输出包含前后空格或多字节字符，之前可能被误判为失败，现在会正确统计。

### 4. [Bugfix] Bump tokenspeed-mla to 2.0.2 to fix nanobind conflict with Triton 3.9 (#61125)

- **代码层面**：把 `requirements/cuda.txt` 里 `tokenspeed-mla` 从 `0.2.15` 升到 `2.0.2`。
- **新手概念课堂**：nanobind 是 Python 的 C++ 绑定库，Triton 3.9 和旧版 tokenspeed-mla 用的 nanobind 版本冲突，升级就解决了。
- **对你有什么影响**：如果你装 vLLM 时遇到 nanobind 相关的报错，升级依赖后就能解决。

### 5. [KV Connector] Support sleep mode with MooncakeConnector over RDMA (#59625)

- **代码层面**：文档更新、测试脚本增加 `SLEEP_WAKE=1` 环境变量、`test_accuracy.py` 增加 sleep/wake 调用。核心是让 MooncakeConnector 在 sleep 模式下也能正确工作。
- **新手概念课堂**：sleep 模式就像电脑的休眠——不用的服务暂时释放显存，需要时再唤醒。RDMA 是远程直接内存访问，绕过 CPU 直接读写对端内存。
- **对你有什么影响**：部署多模型服务时，可以共享 GPU 显存，降低成本。

### 6. [Bugfix] Keep force-killing the process tree when a child exits mid-walk (#59984)

- **代码层面**：`vllm/utils/system_utils.py` 中 `kill_process_tree` 函数，把 `contextlib.suppress(ProcessLookupError)` 扩展为同时捕获 `psutil.NoSuchProcess`。
- **新手概念课堂**：杀进程树时，如果某个子进程在你遍历过程中已经退出，psutil 会抛 `NoSuchProcess`。之前只捕获了 `ProcessLookupError`，会漏掉这种情况导致崩溃。
- **对你有什么影响**：vLLM 关闭时更稳定，不会因为子进程提前退出而崩溃。

### 7. [ROCm][Perf] Enqueue the default chain first in CSA/HCA at small batch sizes for DeepSeek-V4 (#61378)

- **代码层面**：`vllm/models/deepseek_v4/amd/rocm.py` 新增 `_default_first_token_threshold()` 函数，在 gfx950 上小于等于 24 token 时优先入队默认链。
- **新手概念课堂**：GPU 上有多个计算流（stream），任务入队顺序会影响并行效率。小 batch 时先跑长任务能缩短尾部延迟。
- **对你有什么影响**：在 AMD MI355 等 gfx950 卡上跑 DeepSeek-V4，小 batch 场景延迟会降低。

### 8. [Perf][MoE] Cheaper triton_kernels 3.8 routing build in the OAI Triton MoE (#58897)

- **代码层面**：`vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` 新增 `_tk38_routing_data` 函数，用 `argsort` + `bincount` 替代原来的路由构建方式。
- **新手概念课堂**：MoE 路由就是把 token 按专家分组。原来用复杂逻辑，现在用排序+计数，更简单也更快。
- **对你有什么影响**：跑 GPT-OSS 等 MoE 模型时，路由开销更小，端到端吞吐提升。

### 9. [ROCm][MLA] Route gfx942 DCP causal multi-token verify to Triton split-KV decode (#58555)

- **代码层面**：`vllm/platforms/rocm.py` 和 MLA 相关代码，新增 `supports_triton` 参数，让 gfx942 上的多 token 验证走 Triton split-KV 路径。
- **新手概念课堂**：MLA（Multi-head Latent Attention）是 DeepSeek 的核心注意力机制。DCP 是分布式上下文并行。gfx942 是 AMD MI300 的架构代号。
- **对你有什么影响**：在 MI300 上跑 DeepSeek 系列模型，多 token 验证会更快。

### 10. [Quantization] Support packed NVFP4 Qwen4Exp PLE embeddings (#56273)

- **代码层面**：新增 `Qwen4ExpPLENvFp4EmbeddingMethod` 类，支持 NVFP4 打包格式的 PLE（Per-Layer Embedding）加载。
- **新手概念课堂**：NVFP4 是 NVIDIA 的 4 位浮点格式，两个 4 位值打包进一个字节。PLE 是 Qwen4Exp 特有的逐层嵌入。
- **对你有什么影响**：Qwen4Exp 模型可以用 NVFP4 量化，显存占用降低 4 倍。

### 11. [MRV2] Keep warmup out of the EPLB load stats (#56716)

- **代码层面**：`vllm/distributed/eplb/eplb_state.py` 调整 `step()` 顺序，`vllm/v1/worker/gpu/eplb_utils.py` 新增 `suppress()` 上下文管理器。
- **新手概念课堂**：EPLB 是专家负载均衡。warmup 阶段只是热身，不应该计入真实的负载统计，否则会误导均衡策略。
- **对你有什么影响**：MoE 模型的专家负载均衡更准确，长期运行更稳定。

### 12. [ROCm] Bump ROCm base images to AITER v1.25.0 (#61230)

- **代码层面**：`docker/Dockerfile.rock_base` 和 `Dockerfile.rocm_base` 中 AITER 分支从 `v0.1.24.post1` 升到 `v0.1.25`。
- **新手概念课堂**：AITER 是 AMD 的推理加速库，类似 NVIDIA 的 cuBLAS/cuDNN。
- **对你有什么影响**：用 ROCm 官方 Docker 镜像时，会自动获得最新的 AITER 优化。

### 13. [Feat] Add text-only OpenAI Decisions API MVP (#60465)

- **代码层面**：新增 `docs/serving/online_serving/decisions.md` 文档，以及对应的 API 实现。支持 predicate（谓词）、choice（选择）、score（评分）三种问题类型。
- **新手概念课堂**：Decisions API 是 OpenAI 推出的新接口，让模型对输入做结构化判断，而不是自由生成文本。比如"这条评论是否包含损坏物品？"返回 true/false。
- **对你有什么影响**：做分类、审核、打分类应用时，可以直接调用 `/v1/decisions`，不用自己写 prompt 解析。

### 14. [Bugfix] Fix DSpark graph-padding handling in the DeepSeek-V4.1 MoE fused top-k (#59166)

- **代码层面**：`vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` 增加 `_get_padding_mask` 处理，测试新增 `use_padding` 参数化。
- **新手概念课堂**：CUDA Graph 录制时 batch 大小是固定的，实际推理时不足的部分要"填充"。如果填充位置参与计算，结果就错了。
- **对你有什么影响**：DeepSeek-V4.1 在 CUDA Graph 模式下推理结果更准确。

### 15. [Attention] Set ALWAYS cudagraph support for FA4 (#60908)

- **代码层面**：`vllm/v1/attention/backends/flash_attn.py` 中 `_cudagraph_support` 判断从 `== 3` 改为 `in (3, 4)`。
- **新手概念课堂**：FA4 是 FlashAttention 第 4 代，性能和功能都更强。ALWAYS 表示任何情况下都支持 CUDA Graph。
- **对你有什么影响**：用 FA4 时，长序列推理的延迟会明显降低。

### 16. [Mooncake] Allow Mamba cache mode `none` for capacity-only workers (#60712)

- **代码层面**：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py` 放宽校验，允许纯容量 worker 使用 `mamba_cache_mode="none"`。
- **新手概念课堂**：Mamba 是状态空间模型，缓存机制和 Transformer 不同。`none` 模式表示不缓存。
- **对你有什么影响**：混合架构（Attention + Mamba）模型在 Mooncake 上的部署更灵活。

### 17. [Perf][Qwen4Exp] Add SM121 TP=2 merged QSA projection plans (#61437)

- **代码层面**：`vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` 新增 `(7296, 2560)` 形状的 GEMM 配置，针对 SM121（Blackwell）。
- **新手概念课堂**：GEMM 是矩阵乘法，是 Transformer 的核心运算。不同硬件、不同形状需要不同的 kernel 配置。
- **对你有什么影响**：在 Blackwell 架构 GPU 上跑 Qwen4Exp，TP=2 时性能更好。

### 18. Revert "[Build] DeepGEMM migration to torch Stable ABI (#59482)" (#61510)

- **代码层面**：回滚 `cmake/external_projects/deepgemm.cmake` 和 `tools/install_deepgemm.sh` 中的 DeepGEMM 版本，从 Stable ABI 版本回退到 `TORCH_LIBRARY` 版本。
- **新手概念课堂**：Stable ABI 是 PyTorch 的稳定二进制接口，理论上跨版本兼容。但实际迁移中可能遇到问题，所以回滚。
- **对你有什么影响**：暂时不影响普通用户，但说明构建系统改动需要谨慎。

### 19. Fix write-after-write hazard in xpu KV RoPE triton kernel (#46685)

- **代码层面**：`vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py` 中，把完整 KV 拷贝改为只拷贝 NoPE 部分，RoPE 部分由旋转写入独占。
- **新手概念课堂**：写后写冒险（WAW hazard）是指同一内存地址被写两次，如果顺序不对，结果就错了。这里通过 mask 避免重叠写。
- **对你有什么影响**：Intel XPU 上跑 DeepSeek-V4 时，RoPE 计算结果更准确。

### 20. [Model] Kolibri 1 via the Transformers modeling backend (#60827)

- **代码层面**：`docs/models/supported_models.md` 和 `tests/models/registry.py` 新增 `Kolibri1ForCausalLM` 支持，依赖 transformers >= 5.20.0。
- **新手概念课堂**：Kolibri 1 是 Aleph-Alpha 推出的新模型。vLLM 通过 Transformers 后端接入，不需要写专门的 kernel。
- **对你有什么影响**：可以直接用 vLLM 部署 Kolibri 1 模型。

## sglang
## 💡 今日关键词

- **关键词：sgl-router 处理器统一渲染（Processor-Model Parity）**
  - **一句话通俗解释**：把"聊天消息 → 模型输入 token"这一步从 Python 引擎里搬到 Rust 写的路由器里，让路由和引擎对同一个请求渲染出一模一样的 token。
  - **为什么社区在关注**：Kimi-K3、GLM-5.3、DeepSeek-V4 接连被接入，说明社区正在把"请求预处理"从推理引擎中解耦出来，做成可复用的独立组件。这是推理服务走向"前后端分离"的信号。
  - **对新手意味着什么**：如果你在学 Rust + LLM 服务，这是一个绝佳的切入点——写一个模型渲染器，就能直接服务真实用户请求，而不是只改 Python 脚本。

- **关键词：异构硬件适配（XPU / NPU / MLU / ROCm）**
  - **一句话通俗解释**：让同一份代码能跑在 Intel XPU、华为 NPU、寒武纪 MLU、AMD ROCm 等非 NVIDIA 显卡上。
  - **为什么社区在关注**：今天有 6 个 commit 直接与硬件适配相关（XPU、NPU、MLU、AMD/ROCm），说明 SGLang 正在摆脱"CUDA 独占"，走向多硬件后端。
  - **对新手意味着什么**：如果你手头只有 AMD 显卡或国产加速卡，现在正是参与贡献的好时机——很多路径刚打通，bug 和优化空间都很大。

- **关键词：投机解码（Speculative Decoding）跨平台落地**
  - **一句话通俗解释**：用小模型"猜"几个 token，再用大模型一次性验证，从而加速生成。
  - **为什么社区在关注**：DSPARK 在 XPU 上启用、EAGLE topk=1 在 HIP 上启用、NPU 上修 libdevice 问题——投机解码正在从"只有 CUDA 能用"变成"全平台可用"。
  - **对新手意味着什么**：投机解码是当前推理加速最热门的方向之一，理解它的原理（草稿-验证两阶段）能让你在面试和项目中都加分。

- **关键词：CUDA Graph 与图捕获（Graph Capture / Breakable Graph）**
  - **一句话通俗解释**：把一串 GPU 操作"录下来"变成一张图，之后重复播放，省掉每次启动 kernel 的开销。
  - **为什么社区在关注**：DSV4.1 的 commit 专门处理"图捕获内部不能做 host 同步"的问题，说明图捕获已经深入到了模型内部的 MoE 通信逻辑。
  - **对新手意味着什么**：CUDA Graph 是高性能推理的必修课，但坑很多（比如图里不能做 `.item()`）。今天的 commit 就是一个真实案例。

- **关键词：Diffusion 模型性能优化（去同步、缓存启动）**
  - **一句话通俗解释**：让图像/视频生成模型跑得更快，主要手段是减少 CPU-GPU 之间的同步等待。
  - **为什么社区在关注**：今天有 3 个 diffusion 相关 commit（cache-dit、fast_launch、Z-Image 去同步），说明多模态生成正在成为 SGLang 的重要战场。
  - **对新手意味着什么**：Diffusion 推理优化是一个相对蓝海的领域，比 LLM 推理竞争小，但需求在快速增长。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：CUDA Graph 里不能做 host 同步

- **是什么**：CUDA Graph 是把一串 GPU 操作录制下来重复播放的技术。但录制时如果代码里有"把 GPU 数据拷回 CPU"的操作（比如 `.item()`、`.cpu()`），就会破坏图的可重放性。
- **为什么重要**：在 MoE 模型里，每个 rank 需要知道其他 rank 处理了多少 token，这通常需要 host 同步。如果这个同步发生在图捕获期间，图就会捕获到"过时的"值，重放时出错。Commit 13 就是在解决这个问题——在图内用 `torch.full_like` 和 `torch.clamp` 替代 host 拷贝。
- **动手试试**：
  ```python
  import torch

  # 错误示范：图内做 host 同步
  def bad_fn(x):
      n = x.sum().item()  # 这会触发 host 同步
      return x + n

  # 正确示范：全程在 device 上
  def good_fn(x):
      n = x.sum()  # 保持 tensor
      return x + n

  g = torch.cuda.CUDAGraph()
  x = torch.randn(4, device="cuda")
  # 先 warmup
  for _ in range(3):
      good_fn(x)
  with torch.cuda.graph(g):
      y = good_fn(x)
  print("图捕获成功，输出:", y)
  ```
- **预期结果**：`good_fn` 能成功被图捕获；如果你把 `good_fn` 换成 `bad_fn`，PyTorch 会报错或产生错误结果。你会理解为什么 commit 13 要把 `torch.tensor(counts)` 换成 `torch.full_like`。

---

### 知识点 2：投机解码的"草稿-验证"两阶段

- **是什么**：用小模型（draft）快速生成 K 个候选 token，再用大模型（target）一次前向验证这 K 个 token 中哪些是对的。
- **为什么重要**：这是当前 LLM 推理加速最有效的手段之一，通常能提速 2-3 倍。但实现细节很多：topk=1 和 topk>1 的路径不同，CUDA 和 HIP 的 kernel 也不同（见 Commit 20）。
- **动手试试**：
  ```python
  # 用 SGLang 启动一个带 EAGLE 投机解码的服务
  # 假设你有一个 target 模型和对应的 EAGLE draft 模型
  python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path lmsys/sglang-EAGLE-Llama-3.1-8B-Instruct \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 6 \
      --port 30000

  # 另开终端测试
  curl http://localhost:30000/generate \
      -H "Content-Type: application/json" \
      -d '{"text": "The capital of France is", "max_new_tokens": 20}'
  ```
- **预期结果**：你会看到输出速度比不开投机解码时快。观察日志里的 "accept length"（平均接受的 draft token 数），通常在 2-4 之间。

---

### 知识点 3：Triton kernel 的启动开销与缓存

- **是什么**：Triton 是一个用 Python 写 GPU kernel 的框架。每次调用 kernel 时，Triton 需要做参数检查、编译缓存查找等，对于小 kernel 来说这些开销可能比计算本身还大。
- **为什么重要**：Commit 17 引入了 `replay_launches` 装饰器，把"相同签名"的 kernel 启动参数缓存下来，下次直接重放，跳过检查。这在 diffusion 模型里的小 kernel（如 SiLU、LayerNorm）上能省不少时间。
- **动手试试**：
  ```python
  import torch
  import triton
  import triton.language as tl
  import time

  @triton.jit
  def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      mask = offs < n
      x = tl.load(x_ptr + offs, mask=mask)
      y = tl.load(y_ptr + offs, mask=mask)
      tl.store(out_ptr + offs, x + y, mask=mask)

  n = 1024
  x = torch.randn(n, device="cuda")
  y = torch.randn(n, device="cuda")
  out = torch.empty_like(x)

  # 测量 1000 次调用的时间
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(1000):
      add_kernel[(1,)](x, y, out, n, BLOCK=1024)
  torch.cuda.synchronize()
  print(f"1000 次调用耗时: {(time.perf_counter()-t0)*1000:.2f} ms")
  ```
- **预期结果**：你会看到每次调用大约几十微秒，其中很大一部分是启动开销。如果 kernel 本身只算 1 微秒，那启动开销占了 90%+。这就是为什么需要 `replay_launches` 这样的优化。

---

### 知识点 4：多硬件后端的条件导入

- **是什么**：同一份 Python 代码在不同硬件上需要导入不同的库。比如 NPU 上要用 `triton.language.extra.cann.libdevice`，CUDA 上要用 `triton.language.extra.libdevice`。
- **为什么重要**：Commit 7 就是一个典型案例——NPU 上 `libdevice` 的导入路径不同，导致 Markov 采样 kernel 崩溃。这类问题在跨平台开发中非常常见，处理方式通常是运行时判断设备类型再条件导入。
- **动手试试**：
  ```python
  # 模拟跨平台条件导入的模式
  def get_device_type():
      # 实际代码里会检查 torch.version.hip、torch_npu 等
      return "cuda"  # 或 "npu"、"xpu"

  device = get_device_type()

  if device == "npu":
      # 假装导入 CANN 版本
      libdevice = "cann_libdevice"
  else:
      libdevice = "cuda_libdevice"

  print(f"在 {device} 上使用 {libdevice}")

  # 检查你当前环境支持哪些后端
  import torch
  print(f"CUDA 可用: {torch.cuda.is_available()}")
  print(f"HIP 版本: {torch.version.hip}")
  try:
      import torch_npu
      print("NPU 可用")
  except ImportError:
      print("NPU 不可用")
  ```
- **预期结果**：你会看到当前环境的硬件信息。理解"同一份代码在不同设备上走不同分支"是跨平台开发的基本功。

---

### 知识点 5：Diffusion 模型的 CFG 并行

- **是什么**：CFG（Classifier-Free Guidance）是 diffusion 模型的一种技术，每个去噪步骤需要跑两次前向（条件 + 无条件）。CFG 并行就是把这两次前向分到不同 GPU 上同时跑。
- **为什么重要**：Commit 15 修复了一个 bug——在 CFG 并行下，cache-dit 会错误地把相邻两次调用当成"条件/无条件"配对，导致缓存失效。这说明并行策略和缓存策略之间有微妙的交互。
- **动手试试**：
  ```python
  # 用 SGLang 的 diffusion 功能生成一张图
  # 需要先安装 diffusion 相关依赖
  python -m sglang.multimodal_gen.launch_server \
      --model-path Wan-AI/Wan2.2-T2V-A14B \
      --port 30000

  # 另开终端
  curl http://localhost:30000/v1/images/generations \
      -H "Content-Type: application/json" \
      -d '{"prompt": "A cat sitting on a beach", "num_inference_steps": 20}'
  ```
- **预期结果**：你会得到一张生成的图片。如果开启 CFG 并行（多卡），观察日志里 cache-dit 的命中率——修复前可能命中率异常高（错误配对），修复后会回归正常。

---

## 🔥 重要更新

1. **Commit 1 + Commit 9：sgl-router 接入 Kimi-K3 和 GLM-5.3** — 这是 sgl-router 走向"通用模型渲染器"的关键一步，意味着路由器可以独立完成聊天模板渲染和 tokenization，不再依赖引擎。对部署架构影响深远。

2. **Commit 5：Qwen3.8 的 Prefill Context Parallel 端到端支持** — 长上下文推理是当前热点，CP（Context Parallel）能把长序列切分到多卡上处理。这个 commit 让 Qwen3.8 支持了 CP+TP 组共享，是长文本场景的重要基础设施。

3. **Commit 6：PD 分离架构下预填充服务器返回路由专家** — MoE 模型的"路由专家"信息对后续推理很重要，这个 commit 让 PD 分离架构下的预填充服务器能把路由专家信息传回。对 MoE 部署有直接影响。

4. **Commit 13：DSV4.1 在可断图捕获下推导晚层 DP 几何** — 这是 CUDA Graph 与 MoE 通信交互的典型案例，修复了图捕获期间的 host 同步问题。对理解"图捕获的边界条件"很有价值。

5. **Commit 17：Diffusion 小 Triton kernel 的启动重放优化** — 引入 `replay_launches` 装饰器，把 kernel 启动参数按签名缓存。这是一个通用的性能优化模式，可以推广到其他小 kernel 场景。

---

## 📋 逐条解读

### 1. [Feat] sgl-router: Render Kimi-K3 through sglang-processor

- **代码层面**：主要改动在 `experimental/sgl-router/` 下，包括 `chat_formatter.rs`、`README.md` 和 parity 测试。核心是给 Kimi-K3 模型添加了一个原生的 Rust 渲染器，替代之前依赖 dynamo-render 的方案。同时更新了文档，说明 Kimi-K3 现在和 DeepSeek-V4、GLM-5.3 一样，属于"fixture-verified"（用固定测试用例验证过）的渲染器。
- **新手概念课堂**：想象你在餐厅点菜。以前是"服务员（路由器）把订单原样传给厨房（引擎），厨房自己翻译成菜谱"。现在服务员学会了直接翻译菜谱，厨房拿到就能做。这样服务员可以提前检查订单是否合理，也能做更智能的调度（比如把相似的订单放一起做）。
- **对你有什么影响**：如果你用 sgl-router 部署 Kimi-K3，聊天请求会更快（省去引擎端渲染），而且路由决策更准确（因为路由器知道 token 数量）。普通用户感知不到，但服务延迟会降低。

---

### 2. [AMD] diffusion: expect the residual-gate fast path to be off on ROCm in test_modulate

- **代码层面**：只改了一个测试文件 `test_modulate.py`。添加了 `CUDA_NATIVE_ONLY` 标记，让那些依赖 CUDA 专属 fast path 的测试在 ROCm 上跳过，并断言 `can_use_residual_gate_add_cuda` 在 HIP 上返回 False。
- **新手概念课堂**：这就像考试时，老师发现某道题只有用特定计算器才能做，于是给用其他计算器的学生标注"这题你们跳过"。测试代码需要知道"哪些功能在当前硬件上不可用"，否则会误报失败。
- **对你有什么影响**：如果你在 AMD 显卡上跑测试，不会再看到这些"假失败"。但这也说明 residual-gate 的 bit-exact fast path 目前还是 CUDA 独占的，AMD 用户享受不到这个优化。

---

### 3. Revert "[XPU] Support Gemma-3 bidirectional image attention in XPUAttentionBackend"

- **代码层面**：这是一个 revert（回滚）操作，把 Commit 4 的改动撤销了。删除了 `_bidirectional_image_attn_args` 函数、`bidirectional_block_ids` 属性，以及 `forward_extend` 里的相关逻辑。
- **新手概念课堂**：Revert 就像"撤销上一步操作"。在开源项目里，如果一个 PR 合并后发现有问题（比如破坏了下游功能、性能回退），就会 revert 它，等修好再重新提交。
- **对你有什么影响**：如果你在 XPU 上用 Gemma-3 处理图像，双向注意力可能暂时不可用。但注意 Commit 4 和 Commit 3 是成对的——Commit 4 是原始提交，Commit 3 是回滚。这说明这个功能可能还需要打磨。

---

### 4. [XPU] Support Gemma-3 bidirectional image attention in XPUAttentionBackend

- **代码层面**：在 `xpu_backend.py` 里添加了 `_bidirectional_image_attn_args` 函数，用于处理 Gemma-3 图像 token 的双向注意力。核心逻辑是：如果 KV 长度不超过滑动窗口，就丢弃窗口限制，让图像 token 可以双向关注。
- **新手概念课堂**：通常语言模型是"因果注意力"——每个 token 只能看到前面的 token。但图像 token 之间应该能互相看到（就像你看一张图，能同时看到左上角和右下角）。这个 commit 就是让 XPU 后端支持这种"双向注意力"。
- **对你有什么影响**：在 Intel XPU 上用 Gemma-3 做多模态任务时，图像理解质量会更好。但注意这个 commit 后来被 revert 了（见 Commit 3），所以可能还有问题。

---

### 5. [Qwen3.8 CP 4/4] Prefill context parallel E2E support of Qwen3.8

- **代码层面**：主要改动在 `cp_tp_group_sharing_hook.py`，把 Qwen3.8（代码里叫 `Qwen4ExpForConditionalGeneration`）加入了支持 CP-TP 组共享的模型列表。同时添加了一系列限制条件（不支持 EP、不支持 waterfill、不支持 PP 等）。
- **新手概念课堂**：Context Parallel（CP）是把长序列切成几段，每张卡处理一段。TP（Tensor Parallel）是把模型权重切到多卡。CP-TP 组共享意味着这两种并行方式可以复用同一组通信组，减少通信开销。
- **对你有什么影响**：如果你用 Qwen3.8 处理超长文本（比如 100K token），现在可以用 CP 加速预填充阶段。但要注意很多功能还不支持（比如 MoE 的 EP 并行），部署时需要仔细看文档。

---

### 6. [Disagg] Return prompt routed experts from the PD prefill server

- **代码层面**：在 `prefill.py` 里加了两行代码：当请求完成 KV 传输后，如果 `req.return_routed_experts` 为真，就调用 `_maybe_collect_routed_experts` 收集路由专家信息。
- **新手概念课堂**：MoE 模型里，每个 token 会被路由到几个"专家"（子网络）。PD 分离架构下，预填充服务器算完 KV 后要把结果传给解码服务器。这个 commit 让预填充服务器顺便把"这个 token 用了哪些专家"也传过去。
- **对你有什么影响**：如果你在做 MoE 模型的 PD 分离部署，解码服务器现在能拿到路由专家信息，可能用于更精细的调度或分析。普通用户感知不到。

---

### 7. [NPU][Bugfix] use CANN libdevice for DSpark Markov sampling

- **代码层面**：在 `markov_sample.py` 里，把 `from triton.language.extra import libdevice` 改成了条件导入——NPU 上用 `triton.language.extra.cann.libdevice`，其他平台用原来的。
- **新手概念课堂**：`libdevice` 是一组数学函数库（比如 `exp`、`log`、`sin`）。不同硬件厂商会提供自己的实现。就像不同品牌的手机充电器接口不同，你需要用对应的线。
- **对你有什么影响**：如果你在华为 NPU 上用 DSPARK 投机解码，之前会崩溃，现在能正常工作了。这是 NPU 生态完善的一小步。

---

### 8. [AMD][DI][CI] nightly mi355x disagg: stage packed logs under RUNNER_TEMP, not /tmp

- **代码层面**：只改了一行 CI 配置，把临时目录从 `/tmp` 改成 `${RUNNER_TEMP:-/tmp}`。
- **新手概念课堂**：CI（持续集成）是在服务器上自动跑测试。`RUNNER_TEMP` 是 GitHub Actions 提供的临时目录环境变量。用 `/tmp` 的问题是多个 job 可能冲突，用 `RUNNER_TEMP` 更安全。
- **对你有什么影响**：这只影响 SGLang 自己的 CI，普通用户无感知。但如果你在写 CI 脚本，这是一个好习惯。

---

### 9. [sgl-router] Render GLM-5.3 and GLM-5.3-Flash through sglang-processor

- **代码层面**：和 Commit 1 类似，但针对 GLM-5.3。更新了 `SKILL.md` 文档，添加了 GLM-5.3 的渲染器实现，包括模板检测、工具调用处理、schema 类型别名等。
- **新手概念课堂**：GLM-5.3 用的是 Jinja 模板（一种文本模板语言），而 Kimi-K3 用的是原生编码器。这个 commit 展示了如何为不同类型的模型适配 sgl-router。
- **对你有什么影响**：用 sgl-router 部署 GLM-5.3 时，聊天请求会走路由器渲染，速度更快。同时支持了工具调用（`tool_choice`）和推理控制。

---

### 10. [Test] dsv4: Fix Hopper indexer replay test after DeepSelect decode

- **代码层面**：在测试文件里加了一行 `full.use_deep_select_decode = False`，强制关闭 DeepSelect 解码路径，让测试专注于 paged select_decode 的输出契约。
- **新手概念课堂**：DeepSelect 是一种新的解码策略，但它和旧的 paged select 行为不同。测试需要明确"我在测哪个路径"，否则会混淆。
- **对你有什么影响**：只影响测试，普通用户无感知。但这说明 DeepSelect 已经成为一个独立的、需要单独测试的代码路径。

---

### 11. DSPARK enable for XPU

- **代码层面**：三处改动：(1) 文档更新，把 XPU 的投机解码状态从"未实现"改成"支持"；(2) `dsv4_attn_metadata_kernels.py` 里把 `start_loc` 的计算从 `tl.sum` 改成 `tl.max`（更高效）；(3) `speculative_hook.py` 里允许 XPU 使用 DSPARK；(4) `xpu_backend.py` 里用 `spec_info.num_tokens_per_req` 替代全局的 `speculative_num_draft_tokens`。
- **新手概念课堂**：DSPARK 是 SGLang 的一种投机解码算法。之前只支持 CUDA 和 NPU，现在 XPU 也能用了。`num_tokens_per_req` 是"每个请求的 token 数"，在投机解码里这个值可能和全局配置不同。
- **对你有什么影响**：如果你用 Intel XPU，现在可以开启 DSPARK 投机解码来加速生成。但要注意，XPU 的投机解码可能还有一些边界情况需要测试。

---

### 12. [Doc] cookbook: Refresh SemiAnalysis AgentX launches to InferenceX main

- **代码层面**：更新了文档里的代码片段（JSX 文件），同步 SemiAnalysis 的 InferenceX 仓库最新配置。主要是删除了过时的 PR 引用和环境变量。
- **新手概念课堂**：Cookbook 是"菜谱"，告诉用户怎么启动服务。这些配置来自 SemiAnalysis 的基准测试，是经过验证的最佳实践。
- **对你有什么影响**：如果你参考 cookbook 部署 Qwen3.5 等模型，会拿到更新的启动参数。但这是文档更新，不影响代码功能。

---

### 13. [DSV4.1] Derive the late-layer DP geometry from the captured shape under the breakable prefill graph

- **代码层面**：修改了 `decoder_swa_bounded_replay.py` 和 `deepseek_v4.py`。核心是给 `late_layer_dp_counts` 加了 `in_graph` 参数，在图捕获时用 `[local_late_rows] * len(early)` 替代跨 rank 通信。同时把 `torch.tensor(counts)` 换成 `torch.full_like` 或 `gpu.new_tensor`，避免 host 拷贝。
- **新手概念课堂**：MoE 模型的"晚层"需要知道每个 DP rank 处理了多少 token，才能正确做 all-gather。通常这需要跨 rank 通信。但在图捕获时，所有 rank 的形状是统一的，所以可以直接用本地值推断。
- **对你有什么影响**：如果你用 DSV4.1 开启可断图捕获（breakable graph），之前可能会崩溃或结果错误，现在能正常工作了。这是长上下文 + MoE + 图捕获三个技术栈的交汇点。

---

### 14. [MLU][CI] Bootstrap MLU runner workflows

- **代码层面**：新增了 `.github/workflows/mlu-ci-reliability-report.yml` 和相关的 Python 脚本，用于在寒武纪 MLU 上跑 CI 并生成可靠性报告。
- **新手概念课堂**：MLU 是寒武纪的 AI 加速卡。Bootstrap 意思是"从零搭建"。这个 commit 让 SGLang 的 CI 能跑在 MLU 上，确保代码不会破坏 MLU 支持。
- **对你有什么影响**：如果你用寒武纪 MLU，SGLang 的代码质量会更有保障。但普通用户无感知。

---

### 15. [diffusion] cache-dit: respect CFG parallel for dual-transformer models

- **代码层面**：在 `cache_dit_integration.py` 里给 `enable_cache_on_dual_transformer` 加了 `has_separate_cfg` 参数，并把它传给 cache-dit 的适配器。在 `denoising.py` 里传入运行时值。
- **新手概念课堂**：CFG 并行下，每个 rank 只跑条件或无条件的其中一次前向。但 cache-dit 默认假设"相邻两次调用是条件/无条件配对"。如果并行方式变了，这个假设就不成立，缓存会失效甚至出错。
- **对你有什么影响**：如果你用 Wan2.2 等多 transformer 的 diffusion 模型并开启 CFG 并行，之前可能生成质量下降，现在修复了。生成速度也可能更快（缓存命中率正常）。

---

### 16. [XPU] Use the shared fused decode-metadata kernel in the intel_xpu backend

- **代码层面**：在 `xpu_backend.py` 里，把原来分散的 `cache_seqlens_int32`、`cu_seqlens_k`、`page_table` 更新逻辑，替换成调用共享的 `_set_decode_page_metadata`（一个融合的 Triton kernel）。
- **新手概念课堂**：原来需要启动 3-4 个 kernel 分别更新不同的元数据，现在合并成 1 个 kernel。就像原来要跑 4 个窗口分别办手续，现在一个窗口全搞定。
- **对你有什么影响**：在 Intel XPU 上，解码阶段的元数据准备会更快，整体推理延迟降低。这是"减少 kernel 启动次数"这一优化思路的典型案例。

---

### 17. [diffusion] perf: replay small Triton wrappers' launches per operand signature

- **代码层面**：新增了 `fast_launch.py`，提供 `replay_launches` 装饰器。它把 kernel 的启动参数按"操作数签名"（dtype、shape 等）缓存，下次遇到相同签名直接重放。应用到 `silu_mul_bitexact.py` 和 `layernorm_modulate_triton.py`。
- **新手概念课堂**：Triton kernel 每次启动都要做参数检查、编译缓存查找。对于小 kernel，这些开销可能比计算还大。`replay_launches` 就像"记住上次怎么启动的，这次直接照做"。
- **对你有什么影响**：如果你在跑 diffusion 模型（比如视频生成），小 kernel 的启动开销会降低，整体生成速度提升。这是一个通用的优化模式，可以推广到其他场景。

---

### 18. [Agents] Add a rule to prefer sgl-eval for accuracy evals

- **代码层面**：新增了 `.claude/rules/prefer-sgl-eval.md`，更新了 skill 文档，把准确率评估工具从 `run_eval` 改成 `sgl-eval`。
- **新手概念课堂**：`sgl-eval` 是一个独立的评估工具，通过 HTTP API 调用服务，而不是在服务进程内跑。这样更接近真实使用场景，也更容易并行。
- **对你有什么影响**：如果你用 SGLang 做模型评估，建议改用 `sgl-eval`。旧的 `run_eval` 已标记为 deprecated，未来可能移除。

---

### 19. [diffusion] perf: replace padded rows without a host sync for Z-Image

- **代码层面**：修改了 `zimage.py` 的 `_replace_padding_with_token` 方法。原来用布尔掩码赋值（`tensor[pad_mask] = pad_token`），这会触发 host 同步。现在改成 `torch.where`，全程在 device 上。
- **新手概念课堂**：布尔掩码赋值看起来简单，但 PyTorch 需要把 mask 的形状信息同步回 CPU 才能决定怎么分配内存。`torch.where` 是纯 device 操作，不需要同步。
- **对你有什么影响**：Z-Image 模型的生成速度会略有提升（减少了一次 CPU-GPU 同步）。虽然单次同步只有几十微秒，但在 diffusion 的几十步迭代中会累积。

---

### 20. [AMD] Use the preallocated EAGLE topk=1 draft chain on HIP

- **代码层面**：在 `eagle_worker_v2.py` 里，把条件从 `_is_cuda` 改成 `(_is_cuda or _is_hip)`，让 HIP（AMD）也能使用预分配的 topk=1 草稿链。
- **新手概念课堂**：EAGLE 的 topk=1 模式下，草稿 token 是一条链（每个 token 依赖前一个）。预分配内存可以避免每步都分配新 tensor。之前这个优化只有 CUDA 能用，现在 AMD 也能用了。
- **对你有什么影响**：如果你在 AMD 显卡上用 EAGLE topk=1 投机解码，现在会更快（减少了内存分配开销）。这是 AMD 生态追赶 CUDA 的一个小进步。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)