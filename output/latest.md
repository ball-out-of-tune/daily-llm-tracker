# 🚀 vLLM & SGLang 每日更新 — 2026-10-17

> 自动生成于 2026-10-17 06:08 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：MLA / CuTe DSL 后端选择**
  - 一句话通俗解释：MLA（Multi-head Latent Attention）是 DeepSeek 系模型用的省显存注意力机制；CuTe DSL 是 NVIDIA 提供的用 Python 写 GPU kernel 的 DSL，现在 vLLM 允许在 MLA prefill 阶段选择用它。
  - 为什么社区在关注：说明 vLLM 正在把"后端可插拔"做到注意力层——不同 GPU、不同负载可以选不同 kernel 实现，这是性能工程走向精细化的信号。
  - 对新手意味着什么：如果你在学 GPU 编程，CuTe/CUTLASS 生态是值得投入的方向；同时"可插拔后端"是理解 vLLM 架构的好切口。

- **关键词：MoRIIO / AMD ROCm 异构 KV 传输**
  - 一句话通俗解释：MoRIIO 是 AMD 平台上的 KV cache 连接器，用于 P/D 分离（prefill/decode 分离）场景下跨 GPU 传 KV 缓存。
  - 为什么社区在关注：这次加了 hybrid WRITE 传输和 DSpark 支持，还专门加了 MI355 的 CI 流水线——说明 AMD 平台从"能跑"进入"生产级特性对齐"阶段。
  - 对新手意味着什么：异构计算（NVIDIA + AMD 混合）正在成为真实需求，懂 ROCm 生态是差异化竞争力。

- **关键词：Sampler 性能优化（split top-p pipeline）**
  - 一句话通俗解释：采样器里 top-p 的"分块并行"实现被优化，同时修了一个跨 block 的 tie 计数 bug。
  - 为什么社区在关注：采样是每个 token 都要跑的 hot path，小 batch 场景（RL、agent）下采样开销占比很高。
  - 对新手意味着什么：Triton kernel 优化 + 数值正确性（tie-breaking）是高性能推理的必修课。

- **关键词：KV Cache 规划可插拔（KVCacheConfigBuilder）**
  - 一句话通俗解释：把"怎么给模型分配 KV cache 块"这件事抽象成可替换的 Builder，平台/模型可以自定义策略。
  - 为什么社区在关注：KV cache 规划直接影响显存利用率和吞吐，是各家硬件厂商差异化竞争的关键点。
  - 对新手意味着什么：理解"策略模式 + 平台钩子"是读懂 vLLM 核心代码的钥匙。

- **关键词：SM90 上的 Triton 编译器坑（wgmma + ptxas）**
  - 一句话通俗解释：Hopper（SM90）上 Triton 生成的 wgmma 指令在某些配置下会被 ptxas 错误编译，产生 NaN/垃圾值。
  - 为什么社区在关注：一天内两个 commit（Mamba2 chunk scan、LoRA expand）都在绕这个坑——说明这是当前 Triton + Hopper 的共性痛点。
  - 对新手意味着什么：写 Triton kernel 时"数值验证"比"跑通"重要得多，autotune 的每个 config 都要测。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：Triton autotune 配置的正确性验证（以 Mamba2 chunk scan NaN bug 为例）**
  - **是什么**：Triton 的 `@triton.autotune` 会尝试多个 `triton.Config`，只按速度挑最快的。但如果某个 config 被编译器错误编译，产出的结果可能是 NaN——autotune 不会检查正确性。
  - **为什么重要**：Commit 15 里 sm90 上"pipeline wgmma + cb 在寄存器"的 config 会写 NaN 到整行。如果你只测默认 config 就上线，线上会遇到随机 NaN。正确做法是：强制每个 config 都跑一遍并比对。
  - **动手试试**：
    ```python
    # 思路：monkeypatch autotune 的 configs 列表，逐个跑并 assert 有限值
    from triton.runtime.errors import OutOfResources
    import torch

    ref = None
    for config in list(my_kernel.configs):
        my_kernel.configs = [config]  # 强制只用这一个
        try:
            out = run_kernel(...)
        except OutOfResources:
            continue  # 显存不够的 config 跳过
        assert torch.isfinite(out).all(), f"NaN with {config}"
        if ref is None:
            ref = out
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    ```
  - **预期结果**：你会看到某个 config 报 `non-finite output`，这就是 bug；修复后所有 config 输出一致。

- **知识点 2：Top-p 采样的 tie-breaking 与跨 block 状态传递**
  - **是什么**：top-p 采样要保留累积概率 ≤ p 的 token，边界上概率相等的 token（tie）需要确定性地"取前 N 个"。分块并行 kernel 里，每个 block 都要知道前面 block 已经保留了多少个 tie。
  - **为什么重要**：Commit 8 的 bug 是每个 8192 大小的 block 都从 0 重新计数，导致保留的 tie 数量爆炸。这会导致采样分布错误——用户看到的是"模型行为诡异"而不是报错，极难排查。
  - **动手试试**：
    ```python
    import torch
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
    from vllm.v1.sample.ops.topk_topp import apply_top_k_top_p_pytorch

    batch, vocab, num_ties = 64, 151936, 4096
    tie_idx = torch.arange(num_ties) * 37  # 分散到多个 block
    logits = torch.full((batch, vocab), -8.0)
    logits[:, tie_idx] = 0.0
    logits[:, -64:] = 4.0
    p = torch.full((batch,), 0.75)

    kept = apply_top_k_top_p_triton(logits.clone(), None, p) != float("-inf")
    expected = apply_top_k_top_p_pytorch(logits.clone(), None, p) != float("-inf")
    assert torch.equal(kept.sum(-1), expected.sum(-1))
    ```
  - **预期结果**：修复前 `kept.sum(-1)` 会远大于 `expected.sum(-1)`；修复后两者完全相等。

- **知识点 3：KV cache 的 block size 与 hybrid 模型对齐**
  - **是什么**：vLLM 把 KV cache 切成固定大小的 block。混合模型（attention + Mamba/GDN）里，attention 的 block 和 SSM state 的 page 大小必须能对齐，否则 prefix cache 无法复用。
  - **为什么重要**：Commit 7 的 bug 是 CPU 平台在 engine core 重跑 config 时把"为对齐而调大的 block size"覆盖回默认值，导致 prefix cache 一个 block 都命中不了——性能静默退化 10 倍。
  - **动手试试**：
    ```python
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model="...", enable_prefix_caching=True, max_num_batched_tokens=2048)
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    num_tokens = block_size + block_size // 2  # 故意停在 block 中间
    prompt = TokensPrompt(prompt_token_ids=list(range(num_tokens)))
    sp = SamplingParams(max_tokens=1)

    hits = [llm.generate(prompt, sp)[0].num_cached_tokens for _ in range(2)]
    print(hits)  # 期望 [0, block_size]
    ```
  - **预期结果**：第二次请求的 `num_cached_tokens` 应该等于 `block_size`（最后一个完整 block）；如果是 0，说明对齐 bug 还在。

- **知识点 4：PIL 图像透明通道与 RGB PNG 的 tRNS chunk**
  - **是什么**：PNG 有两种透明方式：RGBA（每像素 alpha）和 RGB + tRNS chunk（指定某个颜色为透明色）。PIL 打开 RGB+tRNS 的图时，`mode` 仍是 `"RGB"`，但 `info["transparency"]` 存在。
  - **为什么重要**：Commit 19 修的就是这个——之前只处理 `mode == "RGBA"`，导致 RGB+tRNS 的透明色被当成实色渲染，模型看到的图和用户看到的图不一致。
  - **动手试试**：
    ```python
    from PIL import Image
    import numpy as np
    from vllm.multimodal.media import ImageMediaIO

    img = Image.new("RGB", (2, 1), (255, 0, 0))
    img.putpixel((1, 0), (0, 0, 255))
    img.save("/tmp/t.png", transparency=(255, 0, 0))  # 红色透明

    with Image.open("/tmp/t.png") as src:
        print("mode:", src.mode, "transparency:", src.info.get("transparency"))

    io = ImageMediaIO(rgba_background_color=(12, 34, 56))
    out = np.asarray(io.load_file("/tmp/t.png"))
    print(out)  # 红色像素应被替换为 (12, 34, 56)
    ```
  - **预期结果**：`mode` 是 `"RGB"` 但 `transparency` 存在；加载后红色像素变成背景色，蓝色像素保留。

- **知识点 5：detokenize 与 truncation 的顺序陷阱**
  - **是什么**：当用户传 token IDs 且要求返回文本（`needs_detokenization=True`），同时设置了 `truncate_prompt_tokens`，必须先截断 token 再 detokenize，否则返回的文本和实际送进引擎的 token 不一致。
  - **为什么重要**：Commit 2 的 bug 是顺序反了——先 detokenize 整个 prompt 再截断，导致 echo 回去的 prompt 文本比实际 token 长。这在 RL 训练里会让 reward 计算错位。
  - **动手试试**：
    ```python
    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.params import TokenizeParams

    renderer = HfRenderer(...)  # 用任意小模型
    params = TokenizeParams(
        max_total_tokens=100,
        truncate_prompt_tokens=4,
        truncation_side="left",
        needs_detokenization=True,
    )
    result = renderer.tokenize_prompts(
        [{"prompt_token_ids": list(range(100, 110))}], params
    )[0]
    print(result["prompt_token_ids"])  # 期望 [106,107,108,109]
    print(result["prompt"])            # 期望 "106,107,108,109" 而不是 10 个 token
    ```
  - **预期结果**：`prompt` 字段解码出的 token 数应等于 `prompt_token_ids` 长度（4），而非原始 10。

---

## 🔥 重要更新

- **#62046 + #61996（Sampler）**：split top-p pipeline 修了跨 block tie 计数 bug 并做了性能优化——采样是每 token 必经路径，这两个改动直接影响所有用户的生成质量和延迟。
- **#61718 + #61906（SM90 Triton 编译坑）**：一天内两个独立 bug 都指向 ptxas < 13.3 对 wgmma 的误编译，Mamba2 和 LoRA 用户都可能遇到 NaN/垃圾输出，建议 SM90 用户升级 ptxas 或关注这两个 workaround。
- **#61805（CPU hybrid block size）**：CPU 平台混合模型的 prefix cache 之前静默失效，这个修复能让 CPU 推理的缓存命中率恢复正常。
- **#53558（KVCacheConfigBuilder）**：KV cache 规划变成可插拔，是架构级改动，未来各硬件厂商会通过这个钩子做平台特化——值得关注后续生态。

---

## 📋 逐条解读

### 1. [Attention] Add CuTe DSL backend selection for MLA context prefill (#62069)

- **代码层面**：在 `vllm/v1/attention/backends/mla/prefill/` 下新增 `trtllm_ragged.py` 里 `is_cute_dsl_mla_prefill_supported()` 的检查，测试文件 `test_mla_backends.py` 新增 `trtllm_ragged_backend` 参数（`"trtllm-gen"` / `"cute-dsl"`）和 `use_prefill_query_quantization` 开关，覆盖 chunked context 场景。
- **新手概念课堂**：想象 MLA prefill 是一道菜，"trtllm-gen" 是现成的半成品，"cute-dsl" 是你自己用 CuTe 写的食谱。现在厨房允许你选——如果 CuTe 版本在你的 GPU 上更快或支持 fp8 query 量化，就选它。
- **对你有什么影响**：用 DeepSeek 类模型做长上下文 prefill 的用户，可能在某些 GPU 上获得更好的性能或更低的显存占用；普通用户无感知。

### 2. [Bugfix][Frontend] Detokenize echoed prompts after truncation (#61876)

- **代码层面**：`vllm/renderers/base.py` 的 `_tokenize_singleton_prompt` 和异步版本里，把 `params.apply_post_tokenization()` 移到 detokenize 之前，确保截断先发生。测试新增 `test_detokenization_matches_truncated_tokens`。
- **新手概念课堂**：就像你寄快递时先称重再贴标签——如果先贴标签（detokenize）再称重（truncate），标签上的重量就和实际不符。这里"标签"是返回给用户的 prompt 文本，"重量"是实际送进引擎的 token。
- **对你有什么影响**：使用 token IDs 输入 + truncation + 需要返回文本的场景，echo 回去的 prompt 现在和实际推理内容一致了。RL 训练场景尤其重要。

### 3. [Kernel] Add FP32 router GEMM support for Step-5 Preview and Muse Spark Open (#62126)

- **代码层面**：`csrc/fp32_router_gemm.cu` 新增 `(E=352, H=4096)` 和 `(E=256, H=8192)` 两组 shape 的模板实例化，针对 B300 加了 `kBlockSize=256` 的分支；`fp32_router_gemm_entry.cu` 的 `fp32_router_gemm_supported()` 和 `dispatchFp32RouterGemm` 同步支持。
- **新手概念课堂**：MoE 模型的 router 层是个小矩阵乘法（token × hidden × experts）。FP32 router GEMM 是专门为这个"小但频繁"的乘法写的 kernel。新模型 Step-5 和 Muse Spark 的 router 形状不同，所以要加新的"模具"。
- **对你有什么影响**：跑 Step-5 或 Muse Spark 模型的用户，router 层会走优化 kernel 而不是通用 GEMM，小 batch 场景更快。

### 4. [Feature][MoRIIO] Support hybrid WRITE transfers with DSpark (#60544)

- **代码层面**：`.buildkite/test_areas/misc.yaml` 新增 MI355 上的 MoRIIO 单测和 WRITE P/D E2E 流水线；`docs/features/moriio_connector_usage.md` 新增 hybrid attention + Mamba/KDA 模型（含 DSpark）的 WRITE 模式说明；`vllm/distributed/kv_transfer/kv_connector/v1/moriio/` 下实现 hybrid WRITE 路径。
- **新手概念课堂**：P/D 分离就像"做饭"和"上菜"分两个厨房。WRITE 模式是 prefill 厨房主动把做好的菜（KV cache）端到 decode 厨房。之前只会端"普通菜"（纯 attention），现在能端"混合菜"（attention + Mamba 状态），DSpark 这种带草稿模型的也能端。
- **对你有什么影响**：在 AMD MI355 上部署混合模型 + P/D 分离的用户，现在可以用 WRITE 模式了；纯 attention 用户无变化。

### 5. [Perf][Sampler] Speed up the small-batch split top-p pipeline (#61996)

- **代码层面**：`vllm/v1/sample/ops/topk_topp_triton.py` 里优化 split top-p 路径，新增 `has_topp_only_rows` 参数——如果 batch 里没有"只有 top-p 没有 top-k"的行，就跳过 split pipeline。测试新增 `test_skip_split_without_topp_only_rows` 和大 vocab 场景。
- **新手概念课堂**：split top-p 是"把大词表切块并行算 top-p"。但如果所有行都同时有 top-k，就不需要切块——top-k 已经先把候选缩小了。这个优化就是"能不分块就不分块"。
- **对你有什么影响**：小 batch（RL rollout、agent 单请求）场景下采样延迟会降低，尤其是 top-k + top-p 同时开启时。

### 6. [Bugfix] Remove extra args from CPU's compute slot mapping kernel (#61992)

- **代码层面**：`vllm/utils/cpu_triton_utils.py` 的 `_compute_slot_mapping_kernel_impl` 删掉了 `block_size`、`BLOCKS_PER_KV_BLOCK` 等多余参数，只保留 `KV_CACHE_BLOCK_SIZE` 传给 C++ 实现。测试新增 `test_cpu_slot_mapping_override_forwards_kernel_arguments`。
- **新手概念课堂**：slot mapping 是把"逻辑 token 位置"翻译成"物理 KV cache 槽位"的映射表。CPU 上的实现之前多传了几个参数，C++ 端不认，导致 kernel 调用失败或行为异常。
- **对你有什么影响**：CPU 推理（包括 ARM CPU）的用户，slot mapping 计算恢复正常，不再因为参数不匹配而报错。

### 7. [CPU] Keep the hybrid block size across the engine core's config rerun (#61805)

- **代码层面**：`vllm/platforms/cpu.py` 把 block size 的默认值设置从 `check_and_update_config` 移到 `update_block_size_for_backend`，避免 `VllmConfig.__post_init__` 重跑时把 hybrid 对齐后的 block size 覆盖回 128。测试新增 `test_prefix_cache_hit_reaches_last_full_block`。
- **新手概念课堂**：混合模型（attention + Mamba）的 block size 需要"凑整"到能同时容纳两种 state。但 engine core 启动时会重跑 config 检查，之前会把凑整结果冲掉——就像你整理好的书桌被室友又翻乱了。
- **对你有什么影响**：CPU 上跑混合模型 + prefix caching 的用户，缓存命中率会从"几乎为 0"恢复到正常水平，长 prompt 重复请求的延迟显著降低。

### 8. [Bugfix][Sampler] Carry the boundary tie count across blocks in the split top-p mask kernel (#62046)

- **代码层面**：`vllm/v1/sample/ops/topk_topp_triton.py` 的 `_topp_sb_mask_kernel` 里加了一行 `num_preceding += tl.sum(dmask.to(tl.float32))`，把当前 block 的 tie 计数累加到下一个 block。测试新增 `test_topp_ties_across_blocks`。
- **新手概念课堂**：top-p 边界上可能有多个概率相同的 token，需要"取前 N 个"。分块计算时，每个 block 都要知道前面 block 已经取了多少个 tie。之前每个 block 都从 0 开始数，导致 tie 被重复保留——就像接力赛每棒都从起点重新跑。
- **对你有什么影响**：top-p 采样在 tie 多的情况下（低精度 logits、量化模型）结果更正确，生成分布不再偏移。

### 9. [Bugfix][KV Offload] Allow explicit SimpleCPUOffload block counts (#61674)

- **代码层面**：`SimpleCPUOffloadConnector` 新增 `num_cpu_blocks` 配置项，scheduler 和 worker 都用这个固定值而不是自动按字节数算。`docs/features/kv_offloading_usage.md` 新增说明，测试新增 `test_connector_explicit_cpu_blocks_align_uneven_stages`。
- **新手概念课堂**：KV offload 是把不常用的 KV cache 挪到 CPU 内存。之前每个 worker 按自己的显存/内存自动算块数，PP 不同 stage 的 KV 大小不同就会对不齐。现在允许手动指定统一块数——就像给不同大小的行李箱规定统一的收纳盒数量。
- **对你有什么影响**：用 `SimpleCPUOffloadConnector` + PP 且各 stage KV 大小不均的用户，现在可以手动对齐，避免初始化时的 collective 和潜在的不匹配错误。

### 10. [FlashInfer] Migrate autotune warmup to flashinfer.autotune_v2 managed cache (#55671)

- **代码层面**：`vllm/model_executor/warmup/kernel_warmup.py` 迁移到 FlashInfer 的 `autotune_v2` API，缓存路径从 `$VLLM_CACHE_ROOT/flashinfer_autotune_cache/...` 改为 `~/.cache/flashinfer/autotune/`（可通过 `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` 覆盖）。`docs/usage/security.md` 更新缓存路径表。
- **新手概念课堂**：FlashInfer 的 kernel 需要"试跑"找最快配置（autotune），结果要缓存下来避免每次启动都试。v2 版把缓存管理交给 FlashInfer 自己管，vLLM 不再自己拼路径。
- **对你有什么影响**：升级后第一次启动会重新 autotune（慢一点），之后启动更快；旧缓存目录可以删了。

### 11. [RL][Frontend] Return weight-version spans on OpenAI completions and chat (#61347)

- **代码层面**：`/v1/completions` 和 `/v1/chat/completions` 新增 `return_weight_versions: true` 参数，返回每个 choice 的 `weight_versions` 列表（`{version, start, end}`）。流式请求在最后一个 chunk 返回。`docs/training/async_rl.md` 新增说明。
- **新手概念课堂**：RL 训练时模型权重会中途更新。weight-version span 告诉你"这段输出是哪一版权重生成的"——就像视频的"拍摄日期水印"，方便你判断哪些数据该保留、哪些该丢弃。
- **对你有什么影响**：做 async RL 的用户可以在 OpenAI 兼容 API 上直接拿到版本信息，不用再走 `/inference/v1/generate`；普通推理用户无感知。

### 12. [RL] Support per-parameter source layouts for NCCL M2N (#60934)

- **代码层面**：`vllm/distributed/weight_transfer/nccl_m2n/` 重构，用 `nccl4py` 直接创建 communicator 传给 `nccl.m2n`，不再依赖 `VLLM_NCCL_SO_PATH`。每个参数可以有自己的 source mesh 和 placement。`docs/training/weight_transfer/m2n.md` 大幅更新。
- **新手概念课堂**：M2N 是"多对多"权重传输——训练侧多个 rank 把权重分片发给推理侧多个 rank。之前所有参数共享一套 mesh 布局，现在每个参数可以有自己的"发货地址"——就像快递公司允许每个包裹走不同路线。
- **对你有什么影响**：用 NCCL M2N 做 RL 权重同步的用户，配置更灵活（不同参数不同 TP 布局），且不再需要手动指定 NCCL 库路径。

### 13. [Chore] Use upstream Transformers InklingProcessor (#61883)

- **代码层面**：`vllm/models/inkling/common/mm_preprocess.py` 删掉 vLLM 自己维护的 `InklingImageProcessor` / `InklingAudioFeatureExtractor`，改用 `transformers.InklingProcessor`。新增 `_rescale_image` 辅助函数处理长边缩放。
- **新手概念课堂**：模型的多模态预处理器（图像/音频怎么切成 token）之前 vLLM 自己写了一份，现在上游 Transformers 有了官方实现，就"认祖归宗"——减少维护负担，也保证和 HF 行为一致。
- **对你有什么影响**：用 Inkling 模型的用户，预处理行为和 HF 完全对齐，减少"HF 能跑 vLLM 跑不了"的问题。

### 14. [ROCm] attn res + sigmoid_mul + conv fusions (#52968)

- **代码层面**：新增 `tests/models/kimi_k3/test_amd_kda_fused_conv1d.py`，ROCm 上启用 AITER 的 `causal_conv1d_split_qkv_triton_tile_fn` 融合 kernel，把 attention residual、sigmoid_mul、conv1d 三个操作融合。
- **新手概念课堂**：融合 kernel 就像"洗切炒"一条龙——本来要三次进出厨房（三次读写显存），现在一次搞定。在 Kimi-K3 的 KDA 层里，这三个操作经常连续出现，融合后显存带宽压力大减。
- **对你有什么影响**：AMD GPU 上跑 Kimi-K3 的用户，prefill 阶段会更快；NVIDIA 用户无变化。

### 15. [Bugfix] Fix NaN output of the Mamba2 chunk scan kernel on sm90 (#61718)

- **代码层面**：`vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py` 新增 `_drop_racy_wgmma_configs()`，过滤掉 sm90 上 `num_warps % 4 == 0` 且 `BLOCK_SIZE_M % 64 == 0` 的 config。测试新增 `test_chunk_scan_fwd_every_config_matches` 逐个 config 验证。
- **新手概念课堂**：Hopper 的 wgmma 指令在 Triton 里有个已知 bug——当 `cb`（chunk 边界矩阵）放在寄存器而不是共享内存时，pipelining 会写错数据，产生 NaN。这个修复直接把这些"有毒"的 config 从 autotune 列表里删掉。
- **对你有什么影响**：SM90（H100/H200）上跑 Mamba2 模型的用户，不再遇到随机 NaN 输出；性能可能略降（少了几个快 config），但正确性有保障。

### 16. [Bugfix][LoRA] Fix garbage/NaN lora_expand output on SM90 with fp32 i… (#61906)

- **代码层面**：`vllm/lora/ops/triton_ops/lora_expand_op.py` 里，当 `CAST_TYPE=True` 且平台是 SM90 时，强制 `NUM_STAGES = 1`，禁用 pipelining。
- **新手概念课堂**：和 Commit 15 同源——SM90 上 ptxas < 13.3 会错误编译"pipelined wgmma + 寄存器 A 操作数"。LoRA expand 的 fp32 输入路径正好踩中，所以直接把 pipelining 关掉（`NUM_STAGES=1`）绕过。
- **对你有什么影响**：SM90 上用 LoRA + fp32 输入的用户，输出不再出现垃圾值/NaN；性能略降但正确。

### 17. [DP] Engines own wave wake-ups; drop the front-end FIRST_REQ (#59559)

- **代码层面**：`rust/src/engine-core-client/src/client.rs` 删掉 frontend 发 `FIRST_REQ` 的逻辑，`coordinator/external.rs` 删掉 `CoordinatorWakeupMessage` 和 `CoordinatorCommand` 相关代码。改成"收到请求的 engine 自己唤醒其他 rank"。
- **新手概念课堂**：DP（数据并行）下多个 engine 协同工作，空闲时都暂停。之前是前端发现"第一个请求来了"就通知 coordinator 唤醒大家。现在改成"谁先收到请求谁负责叫醒大家"——去掉了前端这个中间人，减少一层通信。
- **对你有什么影响**：DP 部署下首个请求的延迟可能略降；Rust frontend 用户行为更简洁。

### 18. [Core][Feat] Pluggable KVCacheConfigBuilder for platform/model-specific KV cache planning (#53558)

- **代码层面**：新增 `vllm/v1/core/kv_cache_config_builder.py` 定义 `KVCacheConfigBuilder` 接口和 `get_kv_cache_config_builder()`，`vllm/v1/core/kv_cache_planning.py` 的 `DefaultKVCacheConfigBuilder` 作为默认实现。平台可通过 `Platform.get_kv_cache_config_builder_cls()` 钩子替换，模型可通过 `SupportsKVCacheGrouping` 协议自定义分组。
- **新手概念课堂**：KV cache 规划就像"给书架分配格子"——哪些层共享一个格子、每个格子多大、怎么对齐。之前是硬编码一套算法，现在允许平台/模型"自带书架设计图"。
- **对你有什么影响**：普通用户无感知；但未来新硬件（如某些 NPU）或特殊模型（如混合专家 + 特殊 attention）会有更适配的 KV 规划，显存利用率更高。

### 19. [Bugfix][Multimodal] Honor RGB PNG transparency when loading images (#61986)

- **代码层面**：`vllm/multimodal/media/image.py`（推测）的 `ImageMediaIO` 加载逻辑，除了 `mode == "RGBA"` 外，还检查 `info.get("transparency")`，用 `convert("RGBA")` 后合成背景色。测试新增 `test_image_media_io_rgb_png_transparency` 和 hasher 的对应 case。
- **新手概念课堂**：PNG 有两种透明：RGBA（每像素 alpha）和 RGB+tRNS（指定某颜色透明）。PIL 打开后者时 `mode` 还是 `"RGB"`，但 `info["transparency"]` 存在。之前 vLLM 只认 RGBA，导致 RGB+tRNS 的透明色被当实色。
- **对你有什么影响**：上传带透明色的 RGB PNG（常见于图标、logo）时，模型现在看到的图和用户看到的一致，不再有"透明区域变成实色"的偏差。

### 20. [ROCm][Perf][Kimi-K3] Use AITER Gluon chunk_kda for KDA prefill (#59552)

- **代码层面**：`vllm/_aiter_ops.py` 新增 `is_kda_gluon_prefill_available()`，`vllm/model_executor/layers/mamba/ops/kda.py`（推测）把 `use_fused_chunk` 布尔参数改成 `backend` 字符串（`"triton"` / `"fused"` / `"gluon"`）。测试新增 `test_gluon_chunk_matches_triton`。
- **新手概念课堂**：KDA（Kimi Delta Attention）是 Kimi-K3 的线性注意力变体。Gluon 是 Triton 的新前端，AITER 用 Gluon 写了 `chunk_kda` kernel，在 AMD GPU 上比 Triton 版本更快。
- **对你有什么影响**：AMD GPU 上跑 Kimi-K3 的用户，prefill 阶段会走 Gluon kernel，速度提升；NVIDIA 用户仍走 Triton/fused 路径。

## sglang
## 💡 今日关键词

- **关键词：EPLB（专家并行负载均衡）**
  - 一句话通俗解释：MoE 模型里有很多"专家"小网络，不同专家被请求的次数不一样，EPLB 就是把热门的专家搬到空闲的 GPU 上，让每张卡负载更均匀。
  - 为什么社区在关注：今天有两个 commit（#44847、#44816）都在修 MegaMoE 的 EPLB 相关问题，说明大模型 MoE 推理的负载均衡已经从"能跑"进入"跑得稳、跑得准"的深水区。
  - 对新手意味着什么：如果你在学分布式推理，MoE + 负载均衡是当前最值钱的方向之一，理解"参数跟着专家走"是入门关键。

- **关键词：HiCache / 分层 KV 缓存**
  - 一句话通俗解释：把 KV Cache 从 GPU 显存分层放到 CPU 内存甚至磁盘上，像电脑的 L1/L2/L3 缓存一样，用容量换命中率。
  - 为什么社区在关注：今天有别名简化（#44849）和容量规划文档（#44790）两个 commit，说明这个功能正在从"专家玩具"变成"普通用户能配好"的成熟特性。
  - 对新手意味着什么：KV Cache 管理是 LLM 推理优化的核心战场，HiCache 是理解"显存不够怎么办"的最佳案例。

- **关键词：Rust 前端 / Python 后端混合架构**
  - 一句话通俗解释：HTTP 接入层用 Rust 写（快、省内存），真正的调度和模型执行还是 Python（生态好），两边通过共享内存和环形队列通信。
  - 为什么社区在关注：今天有 3 个 commit（#44764、#44766、#44785）都在给 Rust 前端补齐 Python 端的 API 能力，说明这个架构正在快速产品化。
  - 对新手意味着什么：Rust + Python 混合是 AI Infra 的新趋势，学一点 Rust 和 FFI/共享内存通信会让你在求职中很有竞争力。

- **关键词：量化推理（NVFP4 / FP8）**
  - 一句话通俗解释：把模型权重和激活从 16 位浮点压到 4 位或 8 位，显存占用和带宽大幅下降，速度提升。
  - 为什么社区在关注：#41745 把 FlashInfer 的 4over6 量化配置显式化，说明量化后端正在从"靠环境变量调"走向"代码里写死、可复现"。
  - 对新手意味着什么：量化是推理部署的必备技能，理解"scale 怎么算、舍入误差从哪来"比会调 API 更重要。

- **关键词：投机解码 + PD 分离**
  - 一句话通俗解释：投机解码用小模型"猜"多个 token 再让大模型验证；PD 分离是把"预填充"和"解码"放到不同机器上各干各的。
  - 为什么社区在关注：#44848 宣布 DeepSeek-V4.1 的 PD 分离可以和 DSpark 投机解码组合使用了，这是之前被认为"不能组合"的能力。
  - 对新手意味着什么：这两项技术是当前推理加速的两大支柱，理解它们的组合约束（CP=1、静态验证模式等）是进阶必修课。

## 🧪 重要知识点 & 动手实验

- **知识点 1：MoE 的 EPLB 与"参数跟着专家走"（分布式方向）**
  - **是什么**：MoE 层有 N 个专家，每个 token 只激活其中几个。EPLB 在运行时会重新分配"哪个专家放在哪张卡上"，所以专家对应的权重、量化 scale 都必须一起搬。
  - **为什么重要**：如果只搬权重不搬 scale，重新平衡后数值就错了，模型输出会悄悄劣化——这是最隐蔽的 bug 之一。Commit #44847 就是把 `mega_w13_weight_scale_utccp` 等注册成 `Parameter`，让 EPLB 的搬运逻辑能识别它们。
  - **动手试试**：
    ```python
    import torch, torch.nn as nn
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 4), requires_grad=False)
            # 错误示范：普通属性，EPLB 搬不走
            self.scale_bad = torch.randn(4)
            # 正确示范：注册为 Parameter
            self.register_parameter("scale_good",
                nn.Parameter(torch.randn(4), requires_grad=False))
    e = Expert()
    print("named_parameters:", [n for n, _ in e.named_parameters()])
    print("普通属性在 state_dict 里吗:", "scale_bad" in e.state_dict())
    ```
  - **预期结果**：你会看到 `scale_good` 出现在 `named_parameters` 和 `state_dict` 里，而 `scale_bad` 不在——这就是为什么 EPLB 搬不动它。理解这一点，你就懂了 commit #44847 的动机。

- **知识点 2：HiCache 主机池容量规划（缓存方向）**
  - **是什么**：HiCache 把 GPU 上的 KV 缓存"溢出"到 CPU 内存（host pool）。当请求命中存储层（L3）时，数据要先落到 host pool 再进 GPU。
  - **为什么重要**：如果 host pool 不比 GPU pool 大，存储命中就没有落脚空间，会被直接丢弃并重新计算 prompt——你花了钱买 SSD 却一点没加速。Commit #44790 专门写了这段文档，并给了监控指标 `sglang:storage_prefetch_unfulfilled_tokens_total{reason="host_capacity"}`。
  - **动手试试**：启动一个带 HiCache 的服务，观察启动日志里的 host pool token 数：
    ```bash
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct \
      --enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0
    ```
    然后查看指标：
    ```bash
    curl -s localhost:30000/metrics | grep storage_prefetch_unfulfilled
    ```
  - **预期结果**：启动日志会打印 GPU pool 和 host pool 的 token 容量，`--hicache-ratio 2` 意味着 host 是 GPU 的两倍。如果指标持续增长，说明你的 host pool 太小了，需要调大 ratio 或 size。

- **知识点 3：显式量化配置 vs 环境变量（量化方向）**
  - **是什么**：以前 FlashInfer 的 4over6 量化行为靠环境变量 `FLASHINFER_NVFP4_4OVER6` 控制，现在改成在代码里显式传 `NVFP44Over6Config(e4m3_max=448, err_mode="MSE", ...)`。
  - **为什么重要**：环境变量是"隐式全局状态"，多线程/多进程时极易互相污染。Commit #41745 还加了一个带引用计数的 `contextmanager`，保证并发量化时环境变量只设置一次、用完恢复——这是并发编程的经典模式。
  - **动手试试**：写一个最小复现，感受环境变量的"全局污染"：
    ```python
    import os, threading
    def worker(name, val):
        os.environ["MY_QUANT_MODE"] = val
        import time; time.sleep(0.01)
        print(name, "读到:", os.environ.get("MY_QUANT_MODE"))
    t1 = threading.Thread(target=worker, args=("A", "fast"))
    t2 = threading.Thread(target=worker, args=("B", "exact"))
    t1.start(); t2.start(); t1.join(); t2.join()
    ```
  - **预期结果**：你很可能看到两个线程读到同一个值（后写的覆盖先写的），这就是"隐式全局状态"的坑。对比 commit 里的 `_nvfp4_weight_quantization_scope()` 用锁 + 引用计数解决它，你就理解了这次重构的价值。

- **知识点 4：Rust 前端如何"假装"是 Python 服务（混合架构方向）**
  - **是什么**：Rust 写的 HTTP 服务器要对外表现得和 Python 的 `/server_info` 一模一样，所以在 Python 侧启动时把 `server_args`、`launch_command` 等序列化成 JSON，通过共享内存传给 Rust。
  - **为什么重要**：前端替换最怕"行为不一致"，客户端/路由器依赖某个字段，Rust 端漏了就会挂。Commit #44764 还特意把 `enable_http2` 强制设为 `False`，因为 Rust 监听器只支持 HTTP/1。
  - **动手试试**：启动服务后对比 `/server_info` 的字段：
    ```bash
    curl -s localhost:30000/server_info | python -m json.tool | head -40
    ```
    重点看 `enable_http2`、`max_total_num_tokens`、`launch_command` 这几个字段。
  - **预期结果**：你会看到一份完整的 JSON，包含启动参数和运行时信息。理解"哪些字段是启动时固定的、哪些是运行中变化的"，你就懂了 commit 里 `_launch_server_info` 为什么只挑一部分字段传过去。

- **知识点 5：Triton kernel 的 `do_not_specialize_on_alignment`（性能优化方向）**
  - **是什么**：Triton 编译 kernel 时会根据指针是否 16 字节对齐生成不同版本。Commit #44592 给 `_copy_pages_kernel` 加上 `do_not_specialize_on_alignment=["src_pages", "dst_pages"]`，告诉编译器"别为对齐与否生成两份"。
  - **为什么重要**：分块搬运 page id 时，偏移量可能不对齐，如果每次不对齐都触发一次 JIT 编译，就会在服务中途卡顿（"mid-serving compile"）。这是一个非常实战的性能陷阱。
  - **动手试试**：写一个简单 Triton kernel，分别传对齐和不对齐的切片，观察编译次数（需要装 `triton`）：
    ```python
    import torch, triton, triton.language as tl
    @triton.jit
    def k(x, y, N: tl.constexpr):
        i = tl.program_id(0)
        tl.store(y + i, tl.load(x + i))
    x = torch.arange(64, device="cuda"); y = torch.empty_like(x)
    k[(1,)](x, y, 64)            # 对齐
    k[(1,)](x[1:], y[1:], 63)    # 不对齐，可能触发新编译
    ```
  - **预期结果**：如果你打开 `TRITON_ALWAYS_COMPILE=1` 或查看缓存目录，会看到不同对齐方式产生不同编译产物。加上 `do_not_specialize_on_alignment` 后只剩一份——这就是 commit #44592 省下的"服务中途编译"风险。

## 🔥 重要更新

- **#44403 DSv4.1 解码器 SWA 有界重放默认开启**：这是性能上的大变化，DeepSeek-V4.1 用户不用再加 flag 就能享受更快的 prefill，但要注意它和 prompt logprobs、完整 prefill CUDA graph 不兼容。
- **#44848 PD 分离 + DSpark 投机解码可以组合了**：之前文档明确写"不能组合"，现在有了明确的兼容矩阵（Mooncake/NIXL、静态验证、CP=1），对做大规模部署的团队影响很大。
- **#44766 / #44764 Rust 前端补齐 `/flush_cache` 和 `/server_info`**：Rust 前端正在从"实验性"变成"可替代 Python 前端"，这会影响所有依赖这些 API 的网关和监控系统。
- **#44785 `--gc-threshold` 全进程生效 + 冻结所有 tokenizer worker**：这是一个容易被忽视但影响长稳的修复，多 tokenizer 场景下 GC 停顿会拖慢首 token 延迟。

## 📋 逐条解读

### 1. [Test] attention: Fix replicated attention test missing full_attention_interval (#44602)
- **代码层面**：只改了一个测试文件 `test_replicated_attention_parallel_groups.py`，在构造模型配置时补上 `full_attention_interval=4`。
- **新手概念课堂**：这就像做菜时菜谱漏写了一味调料——测试里的"模型配置"少了这个参数，模型结构和真实部署不一致，测试就测了个寂寞。`full_attention_interval` 控制"每几层放一个完整注意力层"，混合注意力模型（如 Qwen 系列）靠它决定层结构。
- **对你有什么影响**：普通用户无感，但如果你在跑单元测试，这个修复让测试真正覆盖了"复制注意力并行组"的场景，避免假绿。

### 2. [Fix] megamoe: Support EPLB rebalance on SM100 (#44847)
- **代码层面**：`mega_moe.py` 里把 `w13_sf_utccp`、`w2_sf_utccp` 两个量化 scale 注册为 `Parameter`；`modelopt_quant.py` 里把 `mega_l1_alphas`、`mega_l2_alphas`、`mega_l2_act_scales` 也注册为 `Parameter`。
- **新手概念课堂**：EPLB 重新平衡专家时，是"按参数搬家"的——只有注册成 `Parameter` 的东西才会被搬。之前 scale 是普通属性，专家搬走了但 scale 留在原地，数值就错位了。这就像搬家只搬了家具没搬钥匙。
- **对你有什么影响**：在 SM100（Blackwell）上开 EPLB 的 MegaMoE 用户，之前可能遇到输出质量悄悄下降，现在修好了。

### 3. [Feat] server-args: Add --enable-hicache alias for --enable-hierarchical-cache (#44849)
- **代码层面**：给 `enable_hierarchical_cache` 字段加了 `aliases=["--enable-hicache"]`；`server_args_config_parser.py` 新增 `_config_keys()`，让 YAML 配置里也能用任意长选项拼写（`enable-hicache` 或 `enable_hierarchical_cache` 都认）。
- **新手概念课堂**：别名就像一个人的"小名"和"大名"，命令行和配置文件都能用。更妙的是 `_config_keys` 让 YAML 里的 key 也支持别名，避免"命令行能写、配置文件不认"的割裂。
- **对你有什么影响**：以后可以少打几个字 `--enable-hicache`，配置文件里写 `enable-hicache: true` 也能生效。

### 4. [Doc] cookbook: Allow PD with DSpark for DeepSeek-V4.1 (#44848)
- **代码层面**：只改文档和前端配置 snippet，把"PD 和投机解码不能组合"改成"可以组合，但有约束：Mooncake/NIXL 传输、静态验证模式、decode 侧 CP=1/DCP=1"，并删掉了 `incompatibleSpeculativeAlgorithms: ["DSPARK"]`。
- **新手概念课堂**：PD 分离是把"读 prompt"（prefill）和"逐字生成"（decode）拆到不同机器；DSpark 是投机解码算法。以前两者组合会出问题，现在加了启动时校验，不合法组合直接拒绝启动。
- **对你有什么影响**：部署 DeepSeek-V4.1 时，你可以同时开 PD 分离和 DSpark 提速，但要注意 decode 侧不能开上下文并行。

### 5. [Fix] gateway: Drop unused npyz dev-dependency to unbreak builds after zip yank (#44720)
- **代码层面**：删掉 `sgl-model-gateway/Cargo.toml` 里的 `npyz` 依赖，删掉 `generate_vision_golden.py` 脚本和 CI 里生成 golden fixture 的步骤，`.gitignore` 也清理了。
- **新手概念课堂**：`zip` crate 被上游 yank（撤回）了，导致依赖它的 `npyz` 编译失败，进而整个 Rust 网关构建挂掉。修复方式是"既然这个依赖只用于一个已经不需要的测试脚本，那就整个删掉"。
- **对你有什么影响**：如果你在本地编译 `sgl-model-gateway`，之前可能因为 yank 失败，现在能正常构建了。

### 6. [Refactor] nvfp4_online: Use explicit FlashInfer 4over6 config (#41745)
- **代码层面**：`nvfp4_online.py` 新增 `_nvfp4_weight_quantization_scope()` 上下文管理器（带锁 + 引用计数），把环境变量 `FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH` 的开关收敛到一处；量化时显式构造 `NVFP44Over6Config(e4m3_max=448, err_mode="MSE", err_use_fast_math=True)`，不再从环境变量读 `e4m3_max`。
- **新手概念课堂**：环境变量是"全局共享的白板"，多线程同时改会互相踩。引用计数就像会议室的使用登记：第一个人进去开灯，最后一个人出来关灯。这样并发量化时不会有人中途把灯关了。
- **对你有什么影响**：用 NVFP4 在线量化的用户，量化结果更可复现，不再受外部环境变量影响。

### 7. [Doc] hicache: Explain HiCache host pool sizing with a storage backend (#44790)
- **代码层面**：纯文档，在 `hicache_best_practices.mdx` 加了一节 "Host Pool Sizing"，给出 `--hicache-ratio 2 --hicache-size 0` 推荐配置和监控指标。
- **新手概念课堂**：host pool 是"中转仓库"，存储层命中的 KV 要先在这里落脚才能进 GPU。如果中转仓库比 GPU 还小，货到了没地方放就被扔了（重新计算 prompt）。
- **对你有什么影响**：按文档把 `--hicache-ratio` 设成 2 以上，能显著提高存储层命中率，减少重复 prefill。

### 8. [Fix] unified_cache: Resize test allocator before allocating backed pages (#44843)
- **代码层面**：测试文件里在 `finalize_backing()` 之后、`alloc()` 之前加了一行 `allocator.resize(...)`，把"向下增长"分配器的边界推进到已后备的范围内。
- **新手概念课堂**：统一内存分配器有个"边界指针"，测试里先缩小了池子但没告诉分配器，导致分配时越界。这就像先划定了停车位范围，但没更新"下一个空位"的指针。
- **对你有什么影响**：只影响测试，但说明统一内存的分配契约（sizing contract）被更严格地验证了。

### 9. [Perf] DSv4.1: Enable decoder bounded replay by default (#44403)
- **代码层面**：文档和配置 snippet 把 `--enable-decoder-swa-bounded-replay` 从"opt-in"改成"默认开"，新增 `--no-enable-decoder-swa-bounded-replay` 关闭；CUDA 核心里给 `CandidateBlockTableConfig` 加了模板参数 `kWords`。
- **新手概念课堂**：SWA（滑动窗口注意力）的有界重放 = 早期层处理全部 token，晚期层只处理每个请求最后 128 个 token，晚期层的 KV 从第 20 层共享。这就像写文章时前面详细写、结尾只写摘要，但摘要引用前面的内容。
- **对你有什么影响**：DeepSeek-V4.1 用户默认就能享受更快的 prefill，但如果你需要 prompt logprobs 或完整 prefill CUDA graph，要显式加 `--no-enable-decoder-swa-bounded-replay`。

### 10. [Fix] [NPU] attention: Use FIA TND for MLA prefix attention and add Gemma norm fallback (#40131)
- **代码层面**：`ascend_backend.py` 里把 MLA 的 `npu_ring_mla` 换成 `npu_fused_infer_attention_score_v2`（TND 布局），删掉 `ringmla_mask` 相关代码，并加了 Gemma RMSNorm 的回退实现。
- **新手概念课堂**：TND 布局是"Token-Number-Dim"的变长序列布局，适合变长 batch；FIA 是昇腾 NPU 的融合注意力算子。换算子就像把手工炒菜换成料理机，一步到位但要求食材切法统一。
- **对你有什么影响**：在昇腾 NPU 上跑 MLA 模型（如 DeepSeek）的用户，prefix attention 性能和兼容性更好，Gemma 系列模型也能跑了。

### 11. [Fix] megamoe: Respect request-pool limits in decode validation (#44816)
- **代码层面**：`mega_moe_hook.py` 里 `mega_moe_decode_tokens_per_rank()` 新增逻辑：如果设了 `max_running_requests`，用 `ceil_align(max_running_requests // attn_dp_size, tp_size // attn_dp_size)` 和 CUDA graph 的 max_bs 取较小值；报错信息也从 "cuda graph max bs" 改成 "largest decode graph bucket"，并提示可以调 `--max-running-requests`。
- **新手概念课堂**：之前校验只看 CUDA graph 的最大 batch size，但实际请求池可能更小，导致"校验通过但运行时 token 预算不够"。这就像按最大载客量买车险，但实际只允许坐一半人。
- **对你有什么影响**：DP16、每 rank 256 请求的部署，之前会被误判为需要 4096 的预算，现在正确算出是 1536，不用再盲目调大环境变量。

### 12. [Feat] rust-server: Report Python's launch-time /server_info fields from the Rust frontend (#44764)
- **代码层面**：`scheduler.py` 启动时调 `rust_server.set_startup_time()`；新增 `_launch_server_info()` 把 `server_args`、`launch_command`、`kv_events` 等序列化成 JSON 传给 Rust；`internal_state.py` 增加 `rust_server_startup_time` 上报。
- **新手概念课堂**：Rust 前端要"扮演"Python 服务，就得把 Python 侧启动时才知道的信息（比如解析后的参数、启动命令）提前打包传过去。`enable_http2` 被强制设 False 是因为 Rust 监听器只支持 HTTP/1。
- **对你有什么影响**：用 Rust 前端的用户，`/server_info` 返回的字段和 Python 版一致了，网关/监控不会因为字段缺失而报错。

### 13. [Feat] sglang-server: Serve /flush_cache from the Rust frontend (#44766)
- **代码层面**：`flush_wrapper.py` 不再直接持有 `ipc_channels`，改成注入一个 `send_output` 回调；`scheduler.py` 新增 `send_control_output()` 统一处理控制请求的回复路径（Rust 走 egress ring，Python 走 zmq）。
- **新手概念课堂**：`/flush_cache` 是"清空 KV 缓存"的控制接口。之前 Rust 前端不能处理它，现在把"回复走哪条路"抽象成一个函数，Rust 和 Python 各走各的通道。
- **对你有什么影响**：用 Rust 前端的用户可以直接调 `/flush_cache` 了，不用再绕回 Python 服务。

### 14. [Fix] srt: Apply --gc-threshold in every server process and freeze every tokenizer worker (#44785)
- **代码层面**：`http_server.py` 里 `_wait_and_warmup` 新增 `post_warmup_callback` 参数，多 tokenizer 场景下每个 worker 都会在 warmup 后调 `freeze_gc(..., gc_threshold)`；`detokenizer_manager.py` 的 `handle_freeze_gc_req` 也把 `gc_threshold` 传下去。
- **新手概念课堂**：Python 的 GC（垃圾回收）在 gen2 全量回收时会停顿，冻结 GC 就是把"自动扫地"改成"手动扫地"，避免服务中途卡顿。之前只有主进程生效，多 tokenizer worker 没冻结，长稳跑会掉性能。
- **对你有什么影响**：多 tokenizer 部署的用户，首 token 延迟更稳定，不会因为某个 worker 触发 GC 而抖动。

### 15. [Feat] profiler: Add the forward pass id to detailed step span labels (#44775)
- **代码层面**：`model_runner.py` 把 `self.forward_pass_id` 传给 `build_step_span_name()`；`profile_utils.py` 在 detailed annotations 开启时给 span 名加上 `forward_id=N`。
- **新手概念课堂**：连续两次 forward 的 shape 可能完全一样（比如都是 batch=2、seq_len=[10,20]），在 profiler 里分不清谁是谁。加一个自增 ID 就像给每个快递包裹贴单号，形状相同也能对上。
- **对你有什么影响**：用 profiler 分析性能时，能把 trace 里的 span 和专家分布记录器的 forward ID 对应起来，定位更准。

### 16. [Perf] unified-memory: Prewarm the compaction move at startup and copy whole page envelopes by page id (#44592)
- **代码层面**：`copy_pages.py` 的 Triton kernel 加 `do_not_specialize_on_alignment`；新增环境变量 `SGLANG_COMPACTION_MAX_PAGES_PER_MOVE=4096`；`scheduler.py` 在 `init_memory_pools()` 后调 `prewarm_compaction()`；各 unified allocator 实现 `prewarm_compaction()`。
- **新手概念课堂**：统一内存池满了要"压缩"（把零散页搬到一起）。第一次压缩如果发生在服务中途，会触发 Triton JIT 编译 + 显存分配，造成卡顿。预热就是在启动时先跑一遍，把编译和内存都准备好。
- **对你有什么影响**：开统一内存的用户，服务运行中不会因为第一次压缩而突然卡一下，尾延迟更稳。

### 17. [Feat] dsv4.1: Support encoder SWA bounded replay with CUDA DP attention (#43813)
- **代码层面**：`deepseek_v4_hook.py` 新增 `validate_deepseek_v41_encoder_replay_budget()`，把 encoder replay 的预算校验从"不支持 DP attention"改成"要求 max_running_requests > 0 且 chunked_prefill_size >= 128"；`dp_attn.py` 引入 `folded_replay_rows`。
- **新手概念课堂**：之前 encoder SWA replay 和 DP attention 是互斥的，现在通过"折叠重放行"（folded replay rows）让两者共存。约束是 chunked prefill 至少 128 token，否则重放的窗口不够。
- **对你有什么影响**：在 CUDA 上用 DP attention 跑 DeepSeek-V4.1 的用户，现在可以开 encoder SWA replay 提速 prefill 了。

### 18. [Perf] DSv4.1: Skip discarded KV rows during window commit (#44783)
- **代码层面**：`request_window_copy.py` 新增 `_commit_packed_rows` Triton kernel 和 `commit_packed_rows()` 函数，用 `keep` mask 跳过被丢弃的行，不读 KV 也不写共享 sink 行。
- **新手概念课堂**：窗口提交时有些行是"被丢弃的"（比如被截断的 token），之前还是老老实实读一遍、写一遍。现在先看 mask，不保留的行直接跳过——就像整理行李时先看清单，不要的东西根本不拿出来。
- **对你有什么影响**：DeepSeek-V4.1 长上下文场景下，窗口提交的 kernel 时间下降，整体 decode 更快。

### 19. [Fix] DSA: skip-topk pipeline handoff for owed outputs (#43832)
- **代码层面**：`deepseek_v2.py` 里判断"是否有行"从 `hidden_states.shape[0] != 0` 改成 `proxy_tensors["hidden_states"].shape[0] != 0`，`topk_indices` 的空张量也从 `hidden_states.new_empty` 改成 `proxy_tensors["hidden_states"].new_empty`。
- **新手概念课堂**：流水线并行（PP）时，某个 stage 可能"欠着"输出（owed output），此时 `hidden_states` 是空但 `proxy_tensors` 里有真实数据。用错变量就像查快递时看了错误的单号。
- **对你有什么影响**：用 PP + DSA（DeepSeek Sparse Attention）的用户，skip-topk 层的 handoff 不再误判为空，避免流水线卡住或数据错位。

### 20. [Fix] engram: Avoid blocking history copies during PD admission (#44787)
- **代码层面**：`ngram_embedding_manager.py` 里把 `fill_ids = origin_input_ids + output_ids` 改成先切尾部再拼接（`tail = (origin_input_ids[-(n1+1):] + output_ids)[-(n1+1):]`），并把 `torch.tensor(...)` 换成 `async_h2d(...)` 做 pinned 异步拷贝。
- **新手概念课堂**：`origin_input_ids` 可能有 10 万 token，先拼再切会复制整个列表，非常慢；先切尾部只复制最后 n1+1 个。`async_h2d` 用 pinned memory 做异步拷贝，不阻塞 CPU。
- **对你有什么影响**：PD 分离场景下，engram 的 admission 阶段不再因为拷贝长历史而阻塞，首 token 延迟更稳定。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)