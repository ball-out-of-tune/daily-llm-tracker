# 🚀 vLLM & SGLang 每日更新 — 2026-11-25

> 自动生成于 2026-11-25 10:25 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：KV Cache 规划简化**
  - 一句话通俗解释：把"给模型算需要多少显存放历史对话缓存"这件事，从"按字节数算"改成"直接指定块数"，逻辑更直白。
  - 为什么社区在关注：`get_kv_cache_config_from_groups` 这类底层 API 被大量测试和 benchmark 依赖，接口简化说明 vLLM 在稳定内部抽象，为后续 KV 卸载、稀疏注意力等新特性铺路。
  - 对新手意味着什么：如果你在写自定义 attention 或 KV 管理代码，现在可以少算一层"字节→块"的换算，调试时更不容易算错。

- **关键词：多模态编码器数据并行（MM Encoder DP）**
  - 一句话通俗解释：把一张大图/一段视频的视觉编码任务拆给多张 GPU 并行做，而不是每张卡都重复算一遍。
  - 为什么社区在关注：今天有 HY4、Nemotron VL、Kimi-K3 三个多模态相关 commit，说明多模态推理是当下最热的落地场景。
  - 对新手意味着什么：多模态 + 分布式是稀缺技能，学会 `mm_encoder_tp_mode="data"` 的配置和原理，能直接用在生产部署里。

- **关键词：Tekken 分词器字节回退（Byte Fallback）**
  - 一句话通俗解释：Mistral 的 Tekken 分词器遇到 emoji、生僻字时会把字符拆成原始字节，解码时要正确拼回去。
  - 为什么社区在关注：分词器 bug 会直接导致输出乱码/丢字，这类"小 bug 大影响"的问题社区修得很积极。
  - 对新手意味着什么：分词器是 LLM 的第一道关，理解 `skip_special_tokens` 和字节回退，能帮你排查"模型输出乱码"这类玄学问题。

- **关键词：Skinny GEMM 调优（SM100/SM103）**
  - 一句话通俗解释：针对"矩阵很瘦"（行数很少、列很多）的 GEMM 场景，手写 kernel 配置表，比 cuBLAS 还快。
  - 为什么社区在关注：decode 阶段每步只算几个 token，GEMM 非常"瘦"，这是推理延迟的关键瓶颈，Blackwell（B200/B300）上新硬件需要新调优。
  - 对新手意味着什么：性能优化不只是"换更快的库"，而是理解 shape 特征后做针对性调参，这是高级工程师的核心能力。

- **关键词：XPU / Intel GPU 支持**
  - 一句话通俗解释：vLLM 正在把 CUDA 上的功能（LoRA、engine 测试、Triton MLA）逐步移植到 Intel GPU。
  - 为什么社区在关注：今天有 4 个 XPU 相关 commit，说明异构计算（不只 NVIDIA）正在成为 vLLM 的一等公民。
  - 对新手意味着什么：如果你手上有 Intel Arc / BMG 显卡，现在可以真正跑 vLLM 了；异构后端开发也是热门方向。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 规划从"字节"到"块数"**
  - **是什么**：vLLM 给 KV cache 分配显存时，原来要传 `available_memory`（字节数），现在直接传 `num_blocks`（块数）。一块就是固定大小的缓存单元。
  - **为什么重要**：算错字节数会导致显存不够或浪费；直接指定块数让测试和 benchmark 更可控，也避免"页对齐"带来的隐蔽 bug。
  - **动手试试**：
    ```python
    from vllm.v1.kv_cache_interface import get_kv_cache_config_from_groups
    # 旧写法（已废弃）: get_kv_cache_config_from_groups(cfg, groups, available_memory=...)
    # 新写法：
    config = get_kv_cache_config_from_groups(vllm_config, groups, num_blocks=16)
    print(config.num_blocks)  # 应该输出 16
    ```
  - **预期结果**：看到 `num_blocks == 16`，说明你理解了"块"是 KV cache 的最小分配单位，而不是字节。

- **知识点 2：多模态编码器的数据并行（MM Encoder DP）**
  - **是什么**：一张图切成多个 patch，按 patch 数量把不同的图片/视频分给不同 GPU 编码，最后 all-gather 拼回来。
  - **为什么重要**：多模态推理中视觉编码器往往比 LLM 还慢，DP 能让多卡真正并行，而不是每卡重复算。
  - **动手试试**：
    ```bash
    # 用 HY4 模型开启多模态编码器数据并行
    vllm serve <hy4-model-path> \
      --mm-encoder-tp-mode data \
      --tensor-parallel-size 2
    ```
    然后发一个带图片的请求，观察日志里每个 rank 是否只处理了部分图片。
  - **预期结果**：日志显示不同 rank 处理不同图片的 patch，最终输出与单卡一致，但吞吐提升。

- **知识点 3：Tekken 分词器的字节回退**
  - **是什么**：Tekken 词表里 emoji/生僻字没有独立 token，会被编码成多个字节 token；解码时 `skip_special_tokens=False` 曾导致这些字节被错误处理。
  - **为什么重要**：分词器 bug 会让输出乱码，尤其影响多语言和 emoji 场景。
  - **动手试试**：
    ```python
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Base-2407")
    text = "Hi 🤖 数学 ok"
    ids = tok(text, add_special_tokens=False).input_ids
    # 逐 token 解码，模拟增量解码
    for i in range(len(ids)):
        print(repr(tok.decode(ids[:i+1], skip_special_tokens=False)))
    ```
  - **预期结果**：emoji 和中文能正确拼回，不出现 `�` 替换字符；这就是 commit 修复的场景。

- **知识点 4：Skinny GEMM 与 decode 阶段性能**
  - **是什么**：decode 时每步只计算几个 token（M 很小），但权重矩阵 N×K 很大，GEMM 变成"瘦长"形状，标准 cuBLAS 不是最优。
  - **为什么重要**：decode 是生成式推理的主要耗时，优化 skinny GEMM 能直接降低首 token 后延迟（TPOT）。
  - **动手试试**：
    ```python
    import torch
    # 模拟 decode 的 skinny GEMM: M=1, N=2048, K=2048
    x = torch.randn(1, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    # 用 torch profiler 看 aten::mm 耗时
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        y = x @ w
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    ```
  - **预期结果**：看到 `aten::mm` 的 CUDA 耗时；对比不同 M（1/2/4/8）下的耗时，理解为什么需要针对小 M 调优。

- **知识点 5：XPU 上的 LoRA dense GEMM 路径**
  - **是什么**：单 adapter 的 prefill 批次里，所有 token 用同一个 LoRA，可以把 BGMV（逐 token 分组矩阵乘）换成普通 dense GEMM，更快。
  - **为什么重要**：LoRA 推理常被 BGMV 拖慢，识别"单 adapter"场景走 GEMM 是经典优化。
  - **动手试试**：
    ```python
    import torch
    from vllm.lora.ops.xpu_ops import lora_linear
    # 注意：需要 XPU 环境
    y = torch.zeros(8, 384, dtype=torch.float16, device="xpu:0")
    x = torch.randn(8, 512, dtype=torch.float16, device="xpu:0")
    lora_a = [torch.randn(1, 1, 16, 512, dtype=torch.float16, device="xpu:0")]
    lora_b = [torch.randn(1, 1, 384, 16, dtype=torch.float16, device="xpu:0")]
    indices = torch.zeros(8, dtype=torch.long, device="xpu:0")  # 全 0 = 单 adapter
    lora_linear(y, x, lora_a, lora_b, indices, [384], 0.5, True)
    ```
  - **预期结果**：用 `torch.profiler` 能看到 `aten::mm` 而非 BGMV 相关 kernel，说明走了 dense GEMM 快路径。

---

## 🔥 重要更新

- **#69814 [Core] Simplify KV cache planning**：KV cache 底层 API 从"字节数"改为"块数"，影响所有自定义 attention/KV 管理代码，是今天最底层的接口变更。
- **#69676 [Model] Enable data-parallel multimodal encoder for HY4**：多模态编码器 DP 又添一员，配合文档更新，说明这是官方主推的优化路径。
- **#69736 [Perf] SM100/SM103 skinny GEMM plans**：为 B200/B300 手写 GEMM 配置表，decode 延迟有望显著下降，是硬件适配 + 性能优化的双重信号。
- **#67755/#67814 [Bugfix][Tokenizer] Tekken byte fallback**：修复 emoji/多语言乱码，影响所有 Mistral 系模型，普通用户能直接感知。

---

## 📋 逐条解读

### 1. [Kernel] Bump FlashKDA to a32181e (codegen opts + 132→9 variant pruning)
- **代码层面**：`cmake/external_projects/flashkda.cmake` 里把 FlashKDA 的 git tag 从 `9d9e928` 升到 `a32181e`；`csrc/flashkda_registration.cpp` 里把 `cu_seqlens` 参数从可选位置挪到必选位置（参数顺序变了）；`kda.py` 两处调用改成关键字传参以匹配新签名。
- **新手概念课堂**：FlashKDA 是 KDA（一种线性注意力变体）的 CUDA kernel 库。"variant pruning" 就像把 132 种不同的"做菜配方"精简到 9 种——代码生成（codegen）时只编译真正需要的变体，编译更快、二进制更小。
- **对你有什么影响**：使用 GLM5-next / Kimi-K3 等带 KDA 的模型时，编译时间更短，kernel 性能可能更好；普通用户感知不到，但自定义编译时会更省时。

### 2. [Bugfix][MRV2] Align CUDA graph capture input_ids and inputs_embeds with execute_model
- **代码层面**：`tests/v1/cudagraph/test_cudagraph_manager.py` 新增大量参数化测试，验证 CUDA graph capture 阶段选择 `input_ids` 还是 `inputs_embeds` 的逻辑必须和 `execute_model` 一致；核心修复在 `cudagraph_utils` 里（diff 被截断）。
- **新手概念课堂**：CUDA graph 是"把一串 GPU 操作录下来，之后一键重放"的技术，能省掉 CPU 调度开销。但如果录制时用的输入类型和实际执行时不一样（比如录制用 input_ids，执行用 embeds），重放就会出错。
- **对你有什么影响**：使用 MRV2（多模态/embedding 输入）时，CUDA graph 不再因输入类型不一致而崩溃或算错，长上下文/多模态推理更稳。

### 3. [Model] Enable data-parallel multimodal encoder for HY4
- **代码层面**：`vllm/models/hy_v4/common/vision.py` 新增 `run_dp_sharded_hyv4_vision_tower` 函数，用 `get_load_balance_assignment` 按 patch 数把图片分给各 rank，再 `tensor_model_parallel_all_gather` 拼回；`docs/configuration/optimization.md` 更新支持模型列表。
- **新手概念课堂**：想象一个视觉编码器是"洗菜工"，一堆图片是"菜"。以前每张卡都把所有菜洗一遍（重复劳动），现在按菜量分给不同卡洗，最后汇总——这就是 data parallel。
- **对你有什么影响**：部署 HY4 多模态模型时加 `--mm-encoder-tp-mode data`，多卡吞吐更高，尤其适合高并发图片/视频请求。

### 4. [Bugfix][Multimodal] Nemotron 3.5 Super VL: size video frames from the image processor config
- **代码层面**：`tests/models/multimodal/test_nano_nemotron_vl.py` 新增测试，验证视频帧尺寸优先读 `preprocessor_config.json` 里的 `video_target_num_patches` / `video_maintain_aspect_ratio`，vision_config 次之；修复代码在 processor 里（diff 截断）。
- **新手概念课堂**：视频帧尺寸决定"一帧切成多少 patch"，切错了会导致 token 数不对、显存爆炸或画面变形。优先级就像"用户自定义 > 模型默认"。
- **对你有什么影响**：用 Nemotron VL 处理视频时，不同尺寸视频不再被错误地统一缩放，画面比例和显存都更合理。

### 5. [Model] Qwen4Exp: stage CPU-offloaded PLE rows on the host when UVA is unavailable
- **代码层面**：`vllm/models/qwen4_exp/nvidia/ple_layer.py` 新增 `Qwen4ExpPLEHostMemoryEmbedding` 和 `stage_host_rows`，把原来叫 `disk_offload_modules` 的字段改名为 `host_staged_modules`；`docs/features/engram.md` 说明无 UVA 时走 host 内存 staging。
- **新手概念课堂**：UVA（统一虚拟地址）让 GPU 直接访问主机内存。没有 UVA（比如 NVIDIA 机密计算）时，就得先把这一步要用的行"搬"到 GPU 上——像没有直达电梯，只能人工搬货。
- **对你有什么影响**：在机密计算或老 GPU 上跑 Qwen4Exp 的 PLE（位置编码嵌入）不再直接报错，而是自动降级到 host staging，兼容性更好。

### 6. [XPU][LoRA] Use dense GEMM for single-adapter prefill batches
- **代码层面**：`vllm/lora/ops/xpu_ops.py` 新增 `lora_linear`，检测到单 adapter 场景时走 `aten::mm` 而非 BGMV；`tests/lora/test_punica_xpu_ops.py` 新增 profiler 测试验证走了 GEMM。
- **新手概念课堂**：BGMV 是"逐 token 查表做矩阵乘"，适合多 adapter 混合；单 adapter 时所有 token 用同一套权重，直接一次大 GEMM 更快，就像"每个人都点同一道菜"时直接批量做。
- **对你有什么影响**：Intel GPU 上跑单 LoRA 的 prefill 会更快，多 LoRA 场景不受影响。

### 7. [Bugfix][Tokenizer] Route skip_special_tokens=False through Tekken byte fallback
- **代码层面**：`vllm/tokenizers/mistral.py` 里 `convert_ids_to_tokens` 重构，`skip_special_tokens=False` 时也走字节回退逻辑；`_vocab_dict` 构建改用 `id_to_piece`；`serving.py` 里把 bytes 类型的 token 解码成 str。
- **新手概念课堂**：Tekken 词表里 emoji 被拆成多个"字节 token"，解码时要把这些字节拼回 UTF-8 字符。`skip_special_tokens=False` 以前绕过了这个拼接，导致乱码。
- **对你有什么影响**：Mistral 系模型输出 emoji、中文不再乱码，`/tokenize` 接口返回的 token 字符串也不再是 bytes。

### 8. [KV Offload] Share replicated MLA KV per host
- **代码层面**：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/` 里新增 `config.replicated_layout` 判断，多节点 MLA 复制场景下只让部分 rank 做写入（writer），其他 rank 标记完成；测试新增 `test_multi_node_mla_replication_is_independent_of_backend`。
- **新手概念课堂**：MLA（多头潜在注意力）在某些并行配置下 KV 是复制的，多个 rank 存了同样的数据。如果每个 rank 都往主机内存写一份，就浪费了——现在按 host 共享一份。
- **对你有什么影响**：多节点 MLA 部署 + KV 卸载时，主机内存占用降低，卸载效率提升。

### 9. [KV Offload] Return pending work, events and stats from on_schedule_end
- **代码层面**：`OffloadingConnectorScheduler` 的 `on_schedule_end` 现在返回 `StepResult(pending=..., events=..., stats=...)`；`has_pending_push_work` 改为读上次 `StepResult` 而非直接进 manager。
- **新手概念课堂**：调度器每步结束要告诉引擎"还有没有活没干完"。以前靠"进 manager 查一下"，现在直接把结果带回来，避免锁竞争和空闲时误开 step region。
- **对你有什么影响**：KV 卸载场景下引擎调度更稳，不会因为空闲时误持锁而卡住。

### 10. [Core] Simplify KV cache planning
- **代码层面**：`get_kv_cache_config_from_groups` 的参数从 `available_memory`（字节）改为 `num_blocks`（块数），大量测试从 `N * bytes_per_block` 改为直接传块数。
- **新手概念课堂**：就像订酒店从"给我多少平米"改成"给我几间房"——块是固定大小的单元，直接说块数更清晰，不用再算"每块多少字节"。
- **对你有什么影响**：自定义 attention/KV 管理代码需要适配新签名；普通用户无感知，但底层更稳。

### 11. [XPU][CI] Enable v1/engine tests on XPU
- **代码层面**：`.buildkite/intel_jobs/engine_intel.yaml` 把 engine 测试超时从 30 分钟提到 60 分钟，加 `--num-shards=4` 并行，deselect 几个 CUDA-only 测试；`test_async_llm.py` 的 skip 条件从 `is_cuda()` 改为 `is_cuda() or is_xpu()`。
- **新手概念课堂**：CI 是"自动跑测试的流水线"。把测试分片（sharding）就像把一份大作业分给 4 个人同时做，缩短总时间。
- **对你有什么影响**：Intel GPU 上的 v1 engine 测试正式启用，说明 XPU 支持进入"可持续维护"阶段。

### 12. [Bugfix][KV Connector][Mooncake] Keep extensible KV with RDMA-capable peermem allocations
- **代码层面**：`mooncake_connector.py` 删除了"extensible KV + peermem 直接报错"的逻辑，改为检查 `supports_gpudirect_rdma_with_vmm`；`vmm_driver.py` 新增该方法，查询 `_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED`。
- **新手概念课堂**：peermem 是 NVIDIA 让 RDMA 网卡直接读写 GPU 显存的驱动。VMM（虚拟内存管理）分配的显存以前 peermem 不支持，现在新硬件支持了，所以从"一刀切禁止"改为"查一下支不支持"。
- **对你有什么影响**：用 Mooncake + extensible KV cache + 新 NVIDIA 硬件时，不再被强制要求关 peermem，配置更灵活。

### 13. [Bugfix][EC] Let Mooncake consumers encode requests without EC params locally
- **代码层面**：`mooncake/scheduler.py` 里 `ensure_cache_available`、`update_state_after_free`、`request_finished` 都加了 `getattr(request, "ec_transfer_params", None)` 判断——没有 EC 参数的请求不等 push，本地编码。
- **新手概念课堂**：EC（Encoder Cache）连接器原本假设所有请求都要等 producer 推送编码结果。但如果请求没带 `ec_transfer_params`，它根本不需要等，应该直接本地算。
- **对你有什么影响**：混用 EC 和非 EC 请求时，非 EC 请求不再被卡住等待超时，延迟降低。

### 14. [Bugfix][Frontend] Account for PCP size in DP supervisor device_ids
- **代码层面**：`dp_supervisor.py` 的 `_build_device_ids` 把每个 DP 子进程需要的设备数从 `TP * PP` 改为 `TP * PP * PCP`；测试新增 `test_dp_supervisor_device_ids_accounts_for_pcp`。
- **新手概念课堂**：PCP（Prefill Context Parallel）是把长 prompt 的 prefill 切给多卡算。DP supervisor 给每个 DP 副本分 GPU 时，忘了算 PCP 占的卡，导致分错。
- **对你有什么影响**：同时用 DP + PCP 时，GPU 分配正确，不会出现两个副本抢同一张卡。

### 15. [Rust Frontend] Enforce --max-num-queued-reqs and --max-num-queued-tokens
- **代码层面**：`rust/src/cmd/src/cli.rs` 新增 `max_num_queued_reqs` 和 `max_num_queued_tokens` 参数，支持 `32k` 这种人类可读格式（`parse_human_readable_int`）；新增 `AdmissionLimits` 结构传给 server。
- **新手概念课堂**：这是"排队限流"——最多允许多少请求在排队/运行、最多允许多少 prompt token 在 prefill。超了直接返回 503，让客户端换实例重试。
- **对你有什么影响**：Rust 前端部署时可以精确控制过载保护，避免雪崩；Python 前端早就有，现在 Rust 对齐了。

### 16. [Bugfix][Bench] Fix KV cache planning for pipeline forwards
- **代码层面**：`vllm/benchmarks/forward/pipeline.py` 的 `ForwardPipelineExecutor` 重写 `collective_rpc`（用 `all_gather_object` 收集所有 rank 结果）、`determine_available_memory`（取所有 rank 最小值）、`initialize_from_config`（每个 rank 用自己的 config）。
- **新手概念课堂**：流水线并行（PP）下，每个 rank 只跑模型的一部分，显存需求不同。benchmark 要模拟真实情况，必须让每个 rank 拿到自己的 KV cache 配置，而不是所有人共用一份。
- **对你有什么影响**：用 vLLM benchmark 测 PP 模型时，KV cache 规划更准，benchmark 结果更可信。

### 17. [Kernel][XPU] Tensor-descriptor Q loads in Triton MLA decode attention
- **代码层面**：`vllm/v1/attention/ops/triton_decode_attention.py` 里 Q 的加载改用 tensor descriptor（TMA 风格），新增大量测试覆盖不同 layout（contiguous / row_stride_unaligned / base_unaligned）和 fp8 场景。
- **新手概念课堂**：Tensor descriptor 是硬件加速的"批量搬运工"，能一次性把一块内存搬到共享内存，比逐元素加载快。但要求内存对齐，所以测试里专门测了"不对齐"的情况。
- **对你有什么影响**：Intel GPU 上 MLA decode attention 更快；如果 Q 内存不对齐，会自动回退到安全路径，不会崩。

### 18. [Bugfix][Kimi-K3] Render image and text parts in request order
- **代码层面**：`tests/models/multimodal/processing/test_kimi_k3.py` 新增测试，验证多图 + 文本交错时渲染顺序与请求一致；修复代码在 Kimi-K3 的 processor 里（diff 截断），新增 `PromptUpdateMismatchError` 检查。
- **新手概念课堂**：多模态请求里图文顺序很重要（"先看这张图再看那张"），如果渲染时把图片都挪到前面，模型理解就错了。
- **对你有什么影响**：Kimi-K3 处理交错图文时顺序正确，多图问答不再"张冠李戴"。

### 19. [Perf] SM100/SM103 skinny GEMM plans for Qwen3, Qwen3.5/3.6 MoE and Qwen3.8 decode projections
- **代码层面**：`skinny_gemm_plans.py` 新增 `SM100_SKINNY_GEMM_PLANS` 字典，按 `(N, K)` shape 和 batch size 给出最优的 `SkinnyGemmConfig`（block size、k_unroll、vector_width 等），覆盖 Qwen3 系列各种 TP 配置。
- **新手概念课堂**：就像给不同体型的衣服准备不同的裁剪方案——不同的 (N, K) 和 M，最优的 tile 大小、展开次数不同，手写配置表比自动调优更稳。
- **对你有什么影响**：B200/B300 上跑 Qwen3 系列 decode 更快，官方称比 cuBLAS 快至少 10%（部分 dense 条目 5-10%）。

### 20. [Bugfix] Align AFMoE BF16 router precision with Transformers
- **代码层面**：`vllm/model_executor/models/afmoe.py` 里 router gate 的 dtype 从固定 `float32` 改为：BF16 默认 dtype 且无量化（或 modelopt_fp4）时用 `bfloat16`，否则 `float32`；forward 里 `router_logits.float()` 再算；`lm_head` 补上 `prefix`。
- **新手概念课堂**：MoE 的 router 决定"每个 token 走哪些专家"，精度不一致会导致选专家结果和 Transformers 不同，输出就对不上。对齐精度是为了"和参考实现一致"。
- **对你有什么影响**：AFMoE 模型输出与 HuggingFace Transformers 对齐，评测分数不再因精度差异漂移；BF16 下还可能更快。

## sglang
## 💡 今日关键词

- **关键词：EPLB（专家并行负载均衡）在线重定位**
  - 一句话通俗解释：MoE 模型里有很多"专家"子网络，EPLB 就是根据实时流量把专家在 GPU 之间搬家，让每张卡都不闲着。
  - 为什么社区在关注：Kimi-K3 这类超大 MoE 模型必须靠专家并行才能放下，而"能在线搬专家"意味着服务不中断就能调负载，是推理框架走向生产级的关键能力。
  - 对新手意味着什么：如果你在学 MoE 推理，理解"专家放置/复制/重定位"是核心技能，很多框架（SGLang、vLLM、DeepSpeed-MoE）都在这个方向卷。

- **关键词：DCP（Decode Context Parallel）top-k 合并**
  - 一句话通俗解释：长上下文推理时把 KV cache 切到多张卡上，每张卡选出自己认为最重要的 top-k token，再跨卡合并成全局 top-k。
  - 为什么社区在关注：DSA（DeepSeek Sparse Attention）这类稀疏注意力是长上下文降本的关键，而 DCP 组从 8/16 扩展到 2/4，说明小规模部署也能享受这个优化。
  - 对新手意味着什么：稀疏注意力 + 上下文并行是长上下文推理的两大支柱，值得深入。

- **关键词：Triton kernel 特化（specialization）与重编译**
  - 一句话通俗解释：Triton 默认会为每个不同的输入形状/参数值生成一份编译好的 kernel，形状一变就重新编译，服务时会造成卡顿。
  - 为什么社区在关注：今天有两个 commit（#52747、#47278）都在治"形状变化导致运行时编译"的病，这是生产环境的真实痛点。
  - 对新手意味着什么：写 Triton kernel 时，`do_not_specialize`、`BLOCK_K` 的选取是必须掌握的工程技巧。

- **关键词：异构硬件适配（AMD ROCm / 昇腾 NPU）**
  - 一句话通俗解释：同一份代码要在 NVIDIA、AMD、华为昇腾上都能跑好。
  - 为什么社区在关注：今天有 4 个 commit 涉及 AMD Docker/Triton/gfx950 kernel 和 NPU 文档/测试，说明 SGLang 在认真做多硬件后端。
  - 对新手意味着什么：只会 CUDA 已经不够了，ROCm/HIP、CANN 都是加分项，异构算子优化是稀缺人才方向。

- **关键词：PD 分离（Prefill/Decode Disaggregation）的并发正确性**
  - 一句话通俗解释：把"预填充"和"解码"放到不同机器上跑，中间要传 KV cache，状态机并发控制极易出错。
  - 为什么社区在关注：今天有两个 commit（#52976 加锁、#52654 迟到 abort）都在修状态机的竞态，说明这套机制在生产中被大量使用、也容易踩坑。
  - 对新手意味着什么：分布式状态机的 read-modify-write、终态保护是经典并发问题，值得当作案例学习。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：MoE 专家并行中的 EPLB 与"权重槽位"**
  - **是什么**：MoE 层有 N 个"逻辑专家"，每张 GPU 只放一部分。EPLB 会根据负载把专家从一个 GPU 搬到另一个 GPU。有些框架允许"冗余专家"（同一专家放多份）来应对热点，但 Kimi-K3 的实现是"一个逻辑专家一个权重槽"，所以只能搬、不能复制。
  - **为什么重要**：如果你不理解"逻辑专家 vs 本地专家 vs 冗余专家"的区别，配置 `--ep-num-redundant-experts` 时会直接报错或 OOM；理解 `get_moe_weights()` 为什么需要过滤 `correction_bias` 和全局专家，才能正确实现自定义模型的 EPLB 支持。
  - **动手试试**：
    ```bash
    # 在一个支持 EPLB 的 MoE 模型上启动服务，观察专家重定位日志
    python -m sglang.launch_server \
        --model-path <your-moe-model> \
        --tp 8 --ep 8 \
        --enable-eplb \
        --eplb-algorithm deepseek \
        --log-level info
    # 另开终端压测，观察日志里是否有 expert relocation 相关打印
    python -m sglang.bench_serving --backend sglang --num-prompts 200 \
        --dataset-name random --random-input 1024 --random-output 256
    ```
  - **预期结果**：日志中出现专家重定位/负载均衡相关的信息；如果模型不支持冗余专家（如 Kimi-K3），配置 `--ep-num-redundant-experts` 会得到明确的 `ValueError` 而不是静默失败。

- **知识点 2：Triton kernel 的 specialization 与 `do_not_specialize`**
  - **是什么**：Triton 编译 kernel 时，默认会把标量参数的值当作常量参与特化（比如 `num_pages=17` 和 `num_pages=18` 是两个 kernel）。如果这个值每批次都变，就会在服务时不断重编译。
  - **为什么重要**：线上服务最怕"运行时编译"导致的延迟毛刺。`do_not_specialize=["num_pages"]` 让 Triton 把该参数当运行时值处理，形状变化不再触发重编译；`BLOCK_K` 用 `topk` 而不是实际 `k` 来定，也是为了让短上下文复用同一份 kernel。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    # 对照组：n 参与特化
    @triton.jit
    def add_kernel_nospec(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    for n in [100, 101, 102, 103]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)
        add_kernel[(triton.cdiv(n, 128),)](x, y, out, n, BLOCK=128)
    print("cache size after varied n:", len(add_kernel.cache))
    ```
  - **预期结果**：`add_kernel.cache` 会随 `n` 变化增长（说明发生了特化重编译）；给 `n` 加上 `do_not_specialize=["n"]` 后，cache 只保留一份，说明你理解了 specialization 的代价。

- **知识点 3：PD 分离状态机的并发正确性（锁 + 终态保护）**
  - **是什么**：`update_status` 是典型的 read-modify-write：读当前状态 → 判断 → 写新状态。当 bootstrap 线程和 scheduler 线程同时改同一个 room 的状态时会丢更新。修法是加锁 + 明确"Failed 是终态、不能被 max() 提升"。
  - **为什么重要**：KV cache 传输是有状态的（Bootstrapping → Transferring → Success/Failed），状态错了会导致请求永久挂起或误报失败。`max(current, status)` 这种写法在枚举上"看起来对"，但 Failed 排序最低，会被静默覆盖——这是非常隐蔽的 bug。
  - **动手试试**：
    ```python
    import threading, time
    from enum import IntEnum

    class KVPoll(IntEnum):
        Failed = 0
        Bootstrapping = 1
        WaitingForInput = 2
        Transferring = 3
        Success = 4

    # 有 bug 的版本：无锁 read-modify-write
    status = {}
    def update_bad(room, s):
        cur = status.get(room)
        if cur is None:
            status[room] = s; return
        if s == KVPoll.Failed:
            status[room] = KVPoll.Failed; return
        if cur == KVPoll.Failed:
            return
        status[room] = max(cur, s)

    # 并发场景：一个线程报 Failed，另一个线程报 Success
    def worker(fn, s):
        for _ in range(100000):
            fn(1, s)

    t1 = threading.Thread(target=worker, args=(update_bad, KVPoll.Failed))
    t2 = threading.Thread(target=worker, args=(update_bad, KVPoll.Success))
    t1.start(); t2.start(); t1.join(); t2.join()
    print("bad final status:", status[1])  # 可能被 Success 覆盖
    ```
  - **预期结果**：无锁版本最终状态可能是 `Success`（Failed 被覆盖），说明你复现了竞态；把 `update_bad` 改成用 `threading.Lock()` 包住整个 read-modify-write，最终状态稳定为 `Failed`。

- **知识点 4：CUDA kernel 里的内存访问合并（coalesced load/store）**
  - **是什么**：HiSparse 的 swap kernel 原来每个 lane 是"load 一对 → 立刻 store 一对"，改成"先把 2 对都 load 进寄存器，再统一 store"。这样每个 lane 一次发出更多内存请求，隐藏了 host 往返延迟。
  - **为什么重要**：GPU 内存带宽再高，也要靠足够的 in-flight 请求来打满。load/store 交替会形成依赖链，让流水线空转。这个技巧在 KV cache 搬运、embedding gather 等场景通用。
  - **动手试试**：
    ```python
    import torch, time

    N = 64 * 1024 * 1024  # 64M 个 int64
    src = torch.randint(0, 2**63, (N,), dtype=torch.int64, device="cuda")
    dst = torch.empty_like(src)

    def bench(fn, iters=20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    # 基线：PyTorch copy_（底层是高度优化的 memcpy）
    t = bench(lambda: dst.copy_(src))
    print(f"copy_: {N*8/t/1e9:.1f} GB/s")
    ```
  - **预期结果**：你会得到一个接近显存带宽上限的 GB/s 数字（如 H100 上 ~2-3 TB/s）；这告诉你"写 kernel 时要向 memcpy 的吞吐看齐"，进而理解为什么 HiSparse 要把 load 和 store 分离以增加 in-flight 请求。

- **知识点 5：Docker 镜像里的依赖钉版本与"补丁式"修复**
  - **是什么**：ROCm Dockerfile 里 torch 2.11 钉死了 `triton==3.8.0+git4cff872c`，但这个版本有个 bug（masked async load 不写 `other` 值，导致 NaN）。修法是：装一个包含修复的 AMD 构建，然后**直接改 torch 的 METADATA 文件**把它的 pin 指向新版本。
  - **为什么重要**：真实世界的依赖冲突往往不能靠"升级 torch"解决（会连带一堆东西），只能做"外科手术式"修补。理解 `importlib.metadata` 和 wheel 的 METADATA 结构，是排障硬技能。
  - **动手试试**：
    ```bash
    # 查看某个已安装包的依赖声明
    python -c "import importlib.metadata as m; print(m.requires('torch')[:5])"
    # 找到 METADATA 文件位置
    python -c "import importlib.metadata as m, pathlib; print(pathlib.Path(m.distribution('torch')._path)/'METADATA')"
    # 查看 triton 的版本
    python -c "import triton; print(triton.__version__)"
    ```
  - **预期结果**：你能看到 `Requires-Dist: triton==...` 这一行，理解 Dockerfile 里那条 `re.subn(r'^Requires-Dist: triton==[^\s;]+', ...)` 到底在改什么——这就是"不改 torch wheel，只改它的依赖声明"的 hack。

---

## 🔥 重要更新

1. **#53050 Kimi-K3 支持在线 EPLB 专家重定位**：从"启动时静态放置"升级到"运行时动态搬专家"，这是超大 MoE 模型上线生产的关键一步，同时明确不支持冗余专家，避免用户配置踩坑。
2. **#52976 + #52654 PD 分离状态机的两个并发修复**：一个给 `update_status` 加锁防丢更新，一个防止迟到的 abort 把已完成的 Success 改成 Failed。这类 bug 会导致请求挂起或误报，影响面大。
3. **#52747 + #47278 Triton kernel 特化治理**：一个用 `do_not_specialize` 消除按 batch 形状的运行时重编译，一个给 gfx950 的 64-head fp8 稀疏 MLA 加 split-K decode plan，直接关系到线上延迟稳定性。
4. **#52641 AMD Docker 换 Triton 构建修 masked load 进 LDS 的 bug**：gfx950 上 AITER 的 fp4 GEMM 会出 NaN，属于"不修就没法用"的硬件级问题，AMD 用户必看。

---

## 📋 逐条解读

### 1. [Feat] kimi-k3: Support online EPLB expert relocation (#53050)
- **代码层面**：改 `python/sglang/srt/models/kimi_k3.py`。删掉了"只要开 EPLB 就报错"的检查，只保留"不支持冗余专家"的检查；新增 `get_moe_weights()` 方法收集每层 MoE 的权重张量（排除 `correction_bias` 和非本地专家），并在 `__init__` 里用 `LazyValue` 延迟构建 `routed_experts_weights_of_layer` 字典，供 EPLB 重定位时按层取权重。
- **新手概念课堂**：把 MoE 想象成一家有 64 个专科医生的医院，8 个诊室（GPU）。EPLB 就是"根据今天哪个科室病人多，把医生临时调到那个诊室"。Kimi-K3 的规则是"每个医生只有一个工位"，所以能调岗但不能分身（无冗余专家）。
- **对你有什么影响**：Kimi-K3 用户现在可以在不重启服务的情况下让专家负载均衡；但如果你配了 `--ep-num-redundant-experts`，会得到明确报错，不再静默出错。

### 2. [Doc] cookbook: Update Ling-3.2-air MTP benchmarks (#53082)
- **代码层面**：改 `docs/cookbook/autoregressive/InclusionAI/Ling-3.2-air.mdx` 和 `docs/src/snippets/configs/inclusionAI/ling-3.2-air-benchmarks.jsx`。把 GB300 上 MTP 的吞吐从 169.1 tok/s 更新到 355.2 tok/s，TPOT 从 5.60ms 降到 2.72ms，并说明 Blackwell 上 MTP draft 默认走 `trtllm_mla` + `--page-size 64` 以启用 CUDA graph。
- **新手概念课堂**：MTP（Multi-Token Prediction）是"一次猜多个 token"的投机解码；之前在小 batch 下反而更慢，现在通过 attention backend 和 page size 的调优，在 GB300 上反超了标准解码。
- **对你有什么影响**：如果你在 Blackwell 上用 Ling-3.2-air，可以放心开 MTP 了；文档也提醒了并发上限默认 48，需要按需调 `--max-running-requests`。

### 3. [Fix] [AMD] docker: Use a ROCm 10.0 Triton build that fixes masked loads into LDS (#52641)
- **代码层面**：改 `docker/rocm.Dockerfile`。新增 `ROCM1000_TRITON_OVERRIDE_VERSION="3.8.0+git669b31ac"`，在装完 SDK 后强制重装这个 Triton，并用一段 Python 脚本直接改写 torch 的 `METADATA` 把 `Requires-Dist: triton==...` 指向新版本；同时更新 `TRITON_COMMIT_DEFAULT`。
- **新手概念课堂**：masked load 就像"只从货架上取指定几件商品，其余位置放一个默认占位符"。这个 Triton bug 是"占位符没放"，导致读到旧数据（stale LDS），在 gfx950 上让 fp4 GEMM 输出 NaN。
- **对你有什么影响**：用 ROCm 10.0 镜像的 AMD 用户，之前跑 AITER fp4 模型可能出 NaN，升级镜像即可修复；不需要手动改环境。

### 4. [Perf] [NVIDIA] DSA: Enable DCP top-k merge for DCP groups of 2 and 4 (#53024)
- **代码层面**：改 `dcp_topk_merge.cuh` 和 `dsa_dcp_sharded.py`。kernel 的 `static_assert` 从"只支持 8/16 ranks"扩展到"2/4/8/16"；把 `pending` 掩码从 `unsigned` 改为 `uint64_t` 并引入 `low_mask()` 辅助；Python 侧新增 `_DCP_MERGE_CAND_K_BY_WORLD = {2: 1280, 4: 640}`，按 DCP 组大小选候选 K。
- **新手概念课堂**：DCP 是"把长上下文的 KV cache 切成几份放不同卡"。每张卡先选自己看到的 top-k token，再跨卡合并成全局 top-k。组越小，每张卡看到的候选越多，所以 2 卡组要给 1280 个候选而不是 384。
- **对你有什么影响**：TP=2 或 TP=4 部署 DSA 模型时，长上下文 decode 也能享受 top-k 合并优化，不再只有 8/16 卡才能开。

### 5. [Doc] cookbook: Refresh SemiAnalysis AgentX launches to InferenceX main ad39ddd2 (#53052)
- **代码层面**：改 `docs/src/snippets/agentx/...` 下的 JSX 配置。同步 SemiAnalysis InferenceX 主分支到 `ad39ddd2`，更新 PR 列表（新增 #5236/#5235/#5194，移除 #4853），并删掉了一组重复的 Mooncake 环境变量配置。
- **新手概念课堂**：这是"文档即配置"的典型——cookbook 里的启动命令是从上游仓库自动生成的，保证用户看到的命令和 benchmark 实际跑的一致。
- **对你有什么影响**：用 AgentX 配方部署 MiniMax-M3.1 等模型的用户，拿到的启动命令是最新的、经过验证的。

### 6. [Doc] [NPU] docs: use Ascend Docker Runtime for the container commands (#52988)
- **代码层面**：改 `docs/docs/hardware-platforms/ascend-npus/` 下的多个 mdx。把原来一长串 `--device=/dev/davinci0..15` 换成 `--env ASCEND_VISIBLE_DEVICES=0-15`，依赖 Ascend Docker Runtime；FAQ 新增第 9 节，给出"没有 runtime 时"的等价手动挂载命令。
- **新手概念课堂**：Ascend Docker Runtime 相当于 NVIDIA Container Toolkit 的昇腾版，负责把 NPU 设备节点和驱动自动注入容器，省去手写十几个 `--device`。
- **对你有什么影响**：昇腾用户启动容器的命令大幅简化；没有 root 权限装不了 runtime 的，可以照 FAQ 里的长命令手动挂载。

### 7. [Perf] Diffusion: Fuse Anima LayerNorm and timestep modulation during graph replay (#52897)
- **代码层面**：改 `python/sglang/multimodal_gen/runtime/models/dits/anima.py`。把 `forward` 拆成"融合路径"和 `_modulate` 原始路径；在 breakable CUDA graph 内调用 `layernorm_modulate_bitexact`，用 `BitExactFusionGate` 做逐签名校验，失败则回退到原始计算。
- **新手概念课堂**：LayerNorm 后接 `x * (1+scale) + shift` 是 DiT 里最常见的组合。融合就是把两步合成一个 kernel，省一次显存读写；`bitexact` 门控保证融合结果和原实现逐位一致，避免画质漂移。
- **对你有什么影响**：跑 Anima 扩散模型时，CUDA graph replay 阶段更快，且输出和原来数值一致（不会因为融合引入肉眼可见的差异）。

### 8. [Perf] moe/humming: Reduce host overhead in prepare_buffers (#51885)
- **代码层面**：改 `python/sglang/srt/layers/moe/moe_runner/humming.py`。删掉 `_workspace_shapes` 和 `make_workspaces` 两个方法，把逻辑合并进 `prepare_buffers`，避免重复调用 `get_buffer_metas` 和重复计算 shape/字节数；缓冲区反向分配，让最终结果落在 workspace 0。
- **新手概念课堂**：MoE runner 每次前向都要准备一堆临时 buffer（中间结果、workspace）。这些 Python 层的 shape 计算和字典操作在 CPU 上，虽然单次很快，但每层每 token 都做就变成 host 瓶颈。
- **对你有什么影响**：MoE 模型高并发时 CPU 开销降低，GPU 利用率更稳，吞吐略有提升。

### 9. [Fix] hunyuan_v4_flash: Stop IndexPool kernels from compiling per batch shape (#52747)
- **代码层面**：改 `python/sglang/kernels/ops/attention/dsa/hy4_flash_indexpool.py`。给多个 Triton kernel 加 `do_not_specialize`（`pages_stride`、`num_pages`、`K`、`R`、`pos_stride_a`）和 `do_not_specialize_on_alignment`；`_expand_kernel` 的 `BLOCK_K` 从 `next_power_of_2(max(k,16))` 改为 `next_power_of_2(topk // 4)`，让短上下文复用同一 kernel。
- **新手概念课堂**：Triton 默认把标量参数的值当编译期常量做特化。batch 一变，`num_pages` 就变，于是服务过程中不停编译新 kernel，造成延迟毛刺。`do_not_specialize` 告诉编译器"这个值运行时再读"。
- **对你有什么影响**：Hunyuan-V4-Flash 长上下文服务不再因为 batch 形状变化而周期性卡顿，首 token 延迟更稳定。

### 10. [Perf] kimi-k3: Extend deferred-finalize push AR+RMSNorm window to 2 MB (M <= 292) (#52776)
- **代码层面**：改 `python/sglang/srt/layers/communication/k3_ar_fusion.py`。新增 `_FINALIZE_PUSH_MAX_BYTES = 2MB`，`finalize_push_fits` 从用 `_PUSH_MAX_BYTES`（512KB）改为用它，使延迟 finalize 的 push all-reduce 能覆盖到 M=292（NORM_DIM=3584）；测试新增 M=80/160/256/288 的用例。
- **新手概念课堂**：MoE 的 finalize（把各专家结果按权重加回）和 all-reduce 是两个独立步骤，各读写一次显存。延迟 finalize 把它们融成一个 kernel，省一次写+读。窗口越大，能享受这个优化的 batch 越多。
- **对你有什么影响**：Kimi-K3 在中等 batch（M 到 292）下 all-reduce + RMSNorm 更快，且不额外占对称内存（复用已分配的 2MB 槽）。

### 11. [Refactor] models: Fix silent failures and prune dead code in srt models and layers (2/4) (#50804)
- **代码层面**：改 `python/sglang/srt/configs/` 和 `models/` 下大量文件。删除未使用的 logger、`DBRX_PRETRAINED_CONFIG_ARCHIVE_MAP`、`deepseek_ocr.py` 里重复定义的 `DictOutput`（改为从 `deepseekvl2` 导入）、无意义的 `conversation = conversation` 等；把一些静默失败改成显式报错。
- **新手概念课堂**：这类"清理型 PR"看起来无聊，但价值很高：重复定义的类会导致 `isinstance` 判断失效；静默失败（try/except pass）会让 bug 藏几个月。代码整洁度直接决定排障速度。
- **对你有什么影响**：普通用户感知不到，但开发者遇到问题时错误信息更清晰，模型加载失败不再"无声无息"。

### 12. [Perf] HiSparse: optimize the swap in copy kerne (#51506)
- **代码层面**：改 `python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh`。`transfer_item_warp` 从"每 lane 循环 load→store"改为"每 1KB chunk 先 load 2 对到寄存器数组，再统一 store"，用 `#pragma unroll` 展开，并加注释提醒 ptxas 可能把 `.nc` load 沉到 store 之后、需查 SASS。
- **新手概念课堂**：GPU 靠"同时发出很多内存请求"来打满带宽。load 后立刻 store 会形成依赖，lane 只能等。先把多个 load 发出去再统一 store，相当于"一次点 10 个菜再一起吃"，隐藏了往返延迟。
- **对你有什么影响**：HiSparse 的 KV cache swap 更快，长上下文/大 batch 下首 token 延迟降低。

### 13. [Feat] sgl-router: Reject gateway-only flags with their replacement (#50643)
- **代码层面**：新增 `sgl-router/src/config/gateway_flags.rs`（453 行），定义 `Migration` 枚举（`Use`/`Omit`/`Unsupported`）和 `GATEWAY_ONLY_FLAGS` 表，把 sgl-model-gateway 独有、sgl-router 不认识的 flag（如 `--pd-disaggregation`、`--json-log`）在启动时拦截，并给出替代方案或"不支持"说明。
- **新手概念课堂**：sgl-router 是从 sgl-model-gateway 演进/分叉来的，用户很容易照旧文档传老 flag，clap 只会报"unexpected argument"，让人一头雾水。这个 PR 相当于"迁移向导"。
- **对你有什么影响**：从旧 gateway 迁到 sgl-router 的用户，启动报错会直接告诉你"用 X 代替"或"这个功能不支持"，不用翻文档猜。

### 14. [Fix] disaggregation: Make update_status's read-modify-write atomic (#52976)
- **代码层面**：改 `python/sglang/srt/disaggregation/common/conn.py`。给 `CommonKVManager` 加 `_status_lock`，把 `update_status` 整个 read-modify-write 包进 `with self._status_lock:`；把 `_notify_bootstrap` 挪到锁外避免持锁回调。逻辑上明确"Failed 是终态，max() 不能覆盖它"。
- **新手概念课堂**：两个线程同时"读-改-写"同一个字典键，后写的会覆盖先写的（lost update）。加锁是最直接的修法；但要注意别在锁里做可能阻塞的回调，否则死锁。
- **对你有什么影响**：PD 分离部署下，请求状态不会再因为并发更新而错乱，减少"请求卡住不结束"或"误报失败"。

### 15. [Test] openai: Check malformed requests to /generate and OpenAI routes get a 4xx (#50753)
- **代码层面**：新增 `test/registered/openai_server/validation/test_malformed_requests.py`（424 行）。从 `CompletionRequest`/`ChatCompletionRequest`/`GenerateRequest` 的 pydantic schema 自动生成畸形请求（每个字段给每种非法 JSON 类型、Literal 越界值、数值边界外值），逐一发请求断言 4xx，并在每个 case 后发一个合法请求确认服务仍可用、最后 `/health` 通过。
- **新手概念课堂**：这是"基于 schema 的模糊测试"（schema-based fuzzing）。手写用例容易漏，从类型注解自动推导能覆盖所有字段和嵌套对象。
- **对你有什么影响**：普通用户无感知，但意味着 API 对畸形输入更健壮——你写错参数会得到 4xx 而不是 500 或连接断开。

### 16. [Perf] [AMD] dsa: gfx950 split-K decode plan for 64-head fp8 sparse MLA (#47278)
- **代码层面**：改 `python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py`。新增 `_DECODE_MAX_SPLITS = 16`（实测 1-14 行时 split 计划比 BLOCK_H=16 慢 1.05-2.5x，16 split 才打平）；`_load_kv_group`/`_load_kv_tail` 增加 `GW` 和 `CACHE_MOD` 参数；新增 `_no_async_copy_scope_is_effective()` 检查 Triton knob 是否真的生效。
- **新手概念课堂**：split-K 是把一次大的 reduce 拆成多份并行做再合并，适合 K 维很大但行数很少（decode 阶段 batch 小）的场景。但 split 太多会引入额外同步开销，所以要按实测数据设上限。
- **对你有什么影响**：AMD MI355X 上跑 64-head fp8 稀疏 MLA 的 decode 更快，且避免了"以为关了 async copy 其实没关"的隐蔽性能问题。

### 17. [Fix] [AMD] docker: Pin apache-tvm-ffi==0.1.11 in ROCm image for xgrammar 0.2.9 (#52873)
- **代码层面**：改 `docker/rocm.Dockerfile`。把 `apache-tvm-ffi` 从 git commit 改为 `==0.1.11`；把 `git clone && ... && ...` 链式命令改成 `;` 分隔以支持中途 cherry-pick；给 TileLang 的 TVM 子模块 cherry-pick 两个上游 commit（#32/#33），因为 0.1.11 要求派生对象显式声明 slots；最后 import 检查加上 `xgrammar`。
- **新手概念课堂**：`apache-tvm-ffi` 是 TVM 的 FFI 层，xgrammar 0.2.9 依赖它的新接口；版本不匹配会 import 失败或运行时报错。cherry-pick 是"只把某个修复 commit 摘过来"而不升级整个依赖。
- **对你有什么影响**：用 ROCm 镜像 + xgrammar 做结构化输出的用户，不再遇到 TileLang/xgrammar 的兼容性崩溃。

### 18. [Fix] pd: Keep a drained transfer Success when an abort lands late (#52654)
- **代码层面**：改 `disaggregation/common/conn.py` 的 `abort()`：如果当前状态已是 `Success`、staging 无未完成项、且不是"延迟分配且 entry 还在"的情况，就直接 return，不把 Success 改成 Failed。同时改 `mooncake/conn.py` 的 `transfer_worker`：把 `request_status.get(room)` 只读一次（避免两次读之间被 scheduler 清掉），并用 `pop(room, None)` 替代 `if in: pop`。
- **新手概念课堂**：这是"迟到事件"问题——请求已经成功完成了，abort 信号才到，如果不加保护就会把成功改成失败。另外"读两次共享状态"是经典 TOCTOU（check-then-act）bug。
- **对你有什么影响**：PD 分离下偶发的"明明传完了却报失败"会减少；用户取消请求不会误伤已完成的传输。

### 19. [Refactor] router: align Generate forwarding with the rust server API (#45808)
- **代码层面**：改 `sgl-router/Cargo.toml`/`Cargo.lock`/`README.md`/`REFERENCE.md` 及源码。移除 `sglang-runtime-v1-types` 依赖，`--grpc-port` 从服务 `sglang.runtime.v1.SglangService`（每个 RPC 按 HTTP 路由转发）改为只服务 `sglang.api.v1.SglangService/Generate`（单条文本/ token 序列），且只转发给原生 Rust engine。
- **新手概念课堂**：这是 API 收敛——原来 router 的 gRPC 是"HTTP 路由的镜像"，现在改成"只暴露一个 Generate RPC"，减少维护面、语义更清晰。代价是原来依赖其他 RPC 的客户端要改。
- **对你有什么影响**：用 `--grpc-port` 的 gRPC 客户端，现在只能调 `Generate`，且后端必须是 Rust engine；如果你之前用其他 RPC，需要迁移。

### 20. [Test] [NPU] test: Lower the DSV4-Pro W4A8 L2 hit-rate GSP input to 28k to fit the prefill KV pool (#52970)
- **代码层面**：改 `test/registered/npu/accuracy/deepseek_v4_pro/..._gpqa.py`。把 L2 hit-rate 测试的 `input_len` 从 32000 降到 28000，system prompt 从 28800 降到 25200，注释说明 prefill 端在 DSPARK draft 模型常驻后每 rank KV pool 只剩 ~31.5k token，32k prompt 会被 400 拒绝导致 warmup 失败。
- **新手概念课堂**：KV pool 是有限的显存预算，draft 模型（投机解码用）也要占显存，挤占后能容纳的上下文就变短。测试输入必须落在预算内，否则测的是"报错路径"而不是"hit rate"。
- **对你有什么影响**：普通用户无感知；但如果你在昇腾上做 1P1D + 投机解码，要意识到 draft 模型会吃掉 prefill 的 KV 预算，长 prompt 可能被拒。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)