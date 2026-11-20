# 🚀 vLLM & SGLang 每日更新 — 2026-11-20

> 自动生成于 2026-11-20 09:08 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：异构硬件适配（ROCm/XPU/sm_100）**
  - 一句话通俗解释：同一份 vLLM 代码要能在 NVIDIA、AMD、Intel 等不同 GPU 上高效跑起来，需要针对每类硬件的特性单独调优。
  - 为什么社区在关注：今天有 4 个 commit（#65911 ROCm、#68624 XPU、#68842 sm_100、#61620 量化）都在做硬件专属优化，说明 vLLM 已从"只支持 NVIDIA"走向"多硬件后端并行发展"。
  - 对新手意味着什么：如果你熟悉 CUDA，学一点 HIP（AMD）或 SYCL（Intel）会让你在推理框架岗位上有明显优势；异构计算是未来几年的确定性方向。

- **关键词：量化（NVFP4 → MXFP4、E8M0 缩放字节、Quark INT8）**
  - 一句话通俗解释：把模型权重从 16 位浮点压到 4/8 位整数或低精度浮点，省显存、提速度，但要想办法保证精度。
  - 为什么社区在关注：今天有 3 个量化相关 commit（#65256、#61620、#58672），且涉及"在线量化"（加载时转换）这种高级玩法，说明量化正从"离线预量化"走向"运行时灵活转换"。
  - 对新手意味着什么：量化是 LLM 推理落地最核心的省钱手段，学会看 `quant_config`、理解 block-wise scale 是入门必备。

- **关键词：KV Cache / 前缀缓存（Prefix Caching）**
  - 一句话通俗解释：把多个请求共享的 prompt 前缀对应的 KV 缓存复用起来，避免重复计算。
  - 为什么社区在关注：今天 #66234 移除 EAGLE block dropping、#67994 修复流水线并行下的 Mamba block size，说明混合架构（Mamba + Attention）下的缓存一致性是当前热点难点。
  - 对新手意味着什么：KV cache 管理是推理引擎性能的核心，理解 block、hash、lookahead 这些概念能让你读懂 vLLM 最核心的调度代码。

- **关键词：多模态安全与正确性（ICNS 拒绝、M-RoPE 位置）**
  - 一句话通俗解释：多模态输入（图片、视频）要防恶意 payload，同时视频裁剪后位置编码不能算错。
  - 为什么社区在关注：#68245 是安全修复（拒绝 ICNS 解码器），#68270 是正确性修复（Qwen3-VL 视频裁剪后 M-RoPE），说明多模态从"能跑"进入"跑对、跑安全"阶段。
  - 对新手意味着什么：多模态是模型能力扩展的主战场，但坑很多（格式、位置编码、裁剪），做这块要格外注意边界情况。

- **关键词：RL 训练与推理协同（Direct Reload 默认化）**
  - 一句话通俗解释：强化学习训练时，训练进程要不断把新权重推给推理引擎，默认改成"原地直接重载"更快。
  - 为什么社区在关注：#68041 把 `VLLM_RELOAD_MODE` 默认从 layerwise 改成 auto(直接)，说明 RLHF/RLVR 场景已成为 vLLM 的重要用户群。
  - 对新手意味着什么：如果你做 RL 训练框架，理解推理引擎的权重更新机制是打通训练-推理闭环的关键。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：前缀缓存（Prefix Caching）与块对齐
- **是什么**：把 prompt 按固定 token 块（如 16 个 token 一块）切分并计算哈希，相同前缀的块直接复用 KV cache，不重复做 attention 计算。
- **为什么重要**：在多轮对话、few-shot、RAG 场景下，大量请求共享长系统提示，前缀缓存能把首 token 延迟降低数倍。不懂块对齐（block alignment）就会踩坑：比如 Mamba 层和 Attention 层的 block size 不一致时，缓存命中会错位——这正是 #67994 修的问题。
- **动手试试**：
  ```bash
  # 启动一个带前缀缓存的服务
  vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-prefix-caching \
    --max-model-len 4096

  # 用同一个长前缀连续请求两次，观察 num_cached_tokens
  python -c "
  from openai import OpenAI
  c = OpenAI(base_url='http://localhost:8000/v1', api_key='x')
  prefix = 'You are a helpful assistant. ' * 50
  for i in range(2):
      r = c.completions.create(model='Qwen/Qwen2.5-0.5B-Instruct',
          prompt=prefix + f'Question {i}: hello', max_tokens=8)
      print('cached tokens:', r.usage.prompt_tokens_details.cached_tokens)
  "
  ```
- **预期结果**：第二次请求的 `cached_tokens` 明显大于 0（接近前缀长度），说明前缀被复用；第一次为 0。

### 知识点 2：量化中的 scale 字节保真（E8M0）
- **是什么**：MXFP8/MXFP4 这类格式用 E8M0（8 位指数、0 位尾数）存缩放因子，本质是一个"只有指数"的浮点。加载时如果把它当成普通 uint8 做数值转换，字节会被破坏。
- **为什么重要**：量化模型加载时最常见的 bug 就是"权重加载了但精度崩了"，往往就是 scale 被错误地 cast。不懂字节容器（byte container）的概念，你会以为 `torch.uint8` 就是普通整数，结果一 `copy_` 就把 E8M0 的位模式改写了。
- **动手试试**：
  ```python
  import torch
  # 模拟 E8M0 缩放因子：用 uint8 存原始字节，再 view 成 e8m0
  raw = torch.tensor([127, 128, 129], dtype=torch.uint8)
  e8m0 = raw.view(torch.float8_e8m0fnu)
  print(e8m0)          # 看到的是 2^0, 2^1, 2^2 附近的值
  # 错误做法：raw.to(torch.float32) 会得到 127.0, 128.0，完全不是 scale
  print(raw.to(torch.float32))
  ```
- **预期结果**：`e8m0` 打印出接近 1、2、4 的缩放值，而 `raw.to(float32)` 打印 127、128、129——这就是为什么 #65256 要专门保证"加载到 uint8 参数时保留 E8M0 字节"。

### 知识点 3：MoE 的空输入与路由
- **是什么**：MoE（混合专家）层里，每个 token 会被路由到 top-k 个专家。当某个 rank 上没有任何 token（空输入）时，路由张量应该是形状 `(0, k)` 的空张量，而不是报错。
- **为什么重要**：在专家并行（EP）或数据并行（DP）下，某些 rank 完全可能拿到 0 个 token。如果 `fused_topk` 不处理空输入，整个 batch 就会崩——这正是 #58672 新增测试要保护的场景。
- **动手试试**：
  ```python
  import torch
  from vllm.model_executor.layers.fused_moe.fused_topk import fused_topk
  hidden = torch.empty((0, 128), dtype=torch.bfloat16, device='cuda')
  gating = torch.empty((0, 2), dtype=torch.bfloat16, device='cuda')
  w, ids, idx = fused_topk(hidden, gating, topk=1, renormalize=True)
  print(w.shape, ids.shape, idx.shape)  # 期望 (0,1) (0,1) (0,1)
  ```
- **预期结果**：三个张量形状都是 `(0, 1)`，dtype 分别是 float32、int32/int64、int32，不抛异常。

### 知识点 4：Triton kernel 的 launch 配置按硬件分表
- **是什么**：Triton kernel 启动时要指定 `BLOCK_N`、`num_warps`、split-K 数等参数。不同 GPU 架构（sm_90 Hopper、sm_100 Blackwell、sm_120）最优参数完全不同。
- **为什么重要**：同一份 kernel 在小 batch 下用错 split-K 配置，性能可能差 2-3 倍。#68842 就是给 sm_100 单独写了 `_select_sm100_config` 查表函数。不懂这个，你调 kernel 只能靠玄学。
- **动手试试**：
  ```python
  # 查看你的 GPU 算力，决定走哪张配置表
  import torch
  print(torch.cuda.get_device_capability())  # 例如 (9, 0) = Hopper, (10, 0) = B200
  # 在 vllm/models/qwen4_exp/nvidia/ops/qsa.py 里找到 _select_config，
  # 对照你的 capability 看命中了哪个分支
  ```
- **预期结果**：输出类似 `(9, 0)`，你能在代码里定位到 `_is_sm90()` 分支，理解为什么同一 kernel 在不同卡上要走不同表。

### 知识点 5：多模态位置编码（M-RoPE）
- **是什么**：Qwen3-VL 用 M-RoPE（多模态旋转位置编码），给每个 token 三个位置维度（时间 t、高 h、宽 w），图片/视频 token 的 h/w 按网格递增。
- **为什么重要**：视频裁剪（prune）后，被裁掉的帧 token 不再存在，但后续 token 的位置必须重新计算，否则模型会"以为"视频还在。这是 #68270 修的核心问题——位置算错会导致生成乱码。
- **动手试试**：
  ```python
  # 简化理解：一个 2x3 的视觉网格，spatial_merge_size=1
  grid_h, grid_w = 2, 3
  positions = []
  base = 10  # 假设前面有 10 个文本 token
  for i in range(grid_h):
      for j in range(grid_w):
          positions.append((base, base + i, base + j))
  print(positions)
  # 文本 token 的 t/h/w 三个分量都等于其线性位置
  ```
- **预期结果**：视觉 token 的 h 分量只在换行时 +1，w 分量在每列 +1，t 分量固定；理解这个规律后，你就能看懂 `_build_prompt` 里 `add_grid` 的逻辑。

---

## 🔥 重要更新

- **#68041 RL 权重更新默认走 Direct Reload**：`VLLM_RELOAD_MODE` 默认改为 `auto`，配合 `--weight-transfer-config` 时原地重载权重，RL 训练的权重同步延迟显著降低。这是 RL 场景的体验级改进。
- **#66234 移除 EAGLE block dropping，默认 lookahead hashes**：简化了 KV cache 在投机解码下的复杂度，删掉了一个容易出错的旧机制，prefix caching 行为更可预测。
- **#61620 在线 NVFP4 → MXFP4 转换**：让 NVFP4 检查点可以在 MXFP4 硬件上跑，扩大了量化模型的可用硬件范围，是"一份权重多处跑"的重要一步。
- **#68245 安全修复：拒绝 ICNS 图像**：ICNS 解码器无法在加载前拿到可靠尺寸，存在资源耗尽风险，直接从允许列表移除，属于安全加固。

---

## 📋 逐条解读

### 1. [ROCm][Perf] W4A16: serve 6-8 decode rows with the HIP skinny GEMM on gfx1151 (#65911)
- **代码层面**：`CMakeLists.txt` 新增 3 个源文件 `skinny_gemms_int4_n6/n7/n8.cu`；`skinny_gemms_int4.cuh` 新增 `__HIP__GFX1151__` 宏和 `INT4_SKIP_WIDE_N_OFF_GFX1151`，在非 gfx1151 架构上编译期直接跳过 N>5 的 kernel body，避免无谓编译。
- **新手概念课堂**：把 kernel 按输入宽度 N 拆成多个编译单元，就像把一本厚字典拆成 8 本分册——查哪本编哪本，编译时间从"编一整本"变成"编一册"。"skinny GEMM"指矩阵乘的 N 很小（这里 6-8 行），常规 GEMM 会浪费大量算力，需要专门的瘦长 kernel。
- **对你有什么影响**：用 AMD gfx1151（如 Ryzen AI Max）跑 W4A16 量化模型做解码时，6-8 行的 batch 吞吐会提升；其他硬件无感知。

### 2. [Bugfix][Pooling] Let chunked embeddings truncate above max_model_len (#68852)
- **代码层面**：`vllm/entrypoints/pooling/base/serving.py` 删掉 `self.max_model_len` 的硬校验，改为在 chunked 处理场景下用 `max_embed_len` 作为截断上限；测试里错误消息格式也更新为包含具体数值。
- **新手概念课堂**：chunked processing 是把超长文本切成多块分别编码再合并，所以"单次输入长度"可以超过 `max_model_len`——就像把一本 1000 页的书分 10 次读，每次只读 100 页，不需要一次装下整本书。
- **对你有什么影响**：用 embedding 服务处理超长文档时，`truncate_prompt_tokens` 可以设得比 `max_model_len` 大（只要开了 chunked processing），不再被 400 错误拦住。

### 3. [Perf][Startup] Move atomic_writer out of weight_utils so the model-info cache save stays light (#57311)
- **代码层面**：把 `atomic_writer` 从 `weight_utils` 移到独立模块；新增一个在**全新解释器**里跑的测试，断言"导入 registry 并保存 model-info 缓存"不会顺带导入 `model_loader` 和 `attention` 层。
- **新手概念课堂**：Python 的 `import` 是有副作用的——导入一个模块会执行它顶层的所有 import。如果 `weight_utils` 顶层 import 了重型模块，那么只是想存个 JSON 缓存的 API server 进程也会被迫加载整个模型加载器，启动变慢、内存变大。这叫"导入边界"问题。
- **对你有什么影响**：API server 冷启动更快，尤其是只做路由不做推理的进程，内存占用更小。

### 4. [Refactor][MoE] Migrate Quark static INT8 to the MoE oracle (#58672)
- **代码层面**：`tests/kernels/moe/test_fused_topk.py` 新增空输入测试；`test_moe.py` 引入 `MoEPrepareAndFinalizeNoDPEPModular` 和 `kInt8StaticTensorSym`，把 Quark INT8 的 MoE 路径统一到"oracle"（参考实现）测试框架下。
- **新手概念课堂**：MoE oracle 是一套"标准答案"式的测试基础设施——所有量化后端（FP8、INT8、NVFP4）都跑同一组输入，结果与参考实现对比。把 Quark INT8 迁进来，意味着它不再有"自己的特殊测试"，而是纳入统一质量门禁。
- **对你有什么影响**：用 Quark INT8 量化的 MoE 模型（如 DeepSeek 系列）会有更严格的回归测试保护，出 bug 的概率降低。

### 5. [Recipe] Inkling 2 Small: serve from nightly and add Hopper (H200) (#68864)
- **代码层面**：新增 `recipes/models/thinkingmachines/Inkling-2-Small-MXFP8.yaml`，声明 H200 已验证（`h200: verified`）、最低 vLLM 版本 0.35.0、需要 nightly 镜像，并配置了 DSpark 投机解码（7 个 draft token）。
- **新手概念课堂**：Recipe 是 vLLM 的"部署说明书"——用 YAML 描述某个模型在哪些硬件上怎么跑、需要什么参数。`nightly_required: true` 表示该模型架构还没进稳定版，必须用每日构建。
- **对你有什么影响**：想部署 Thinking Machines 的 Inkling 2 Small 时，直接照抄这个 YAML 就能在 H200 上跑起来，不用自己摸索 TP 配置和投机解码参数。

### 6. [Perf][Mamba] Export ShortConv (LFM2/LFM3) prefill checkpoints (#68673)
- **代码层面**：新增 `tests/kernels/mamba/test_short_conv_checkpoint.py`，测试 `ShortConv` 层在 prefill 阶段导出 checkpoint 元数据（`MambaPrefillCheckpointMetadata`）的正确性，覆盖 decode + prefill 混合 batch。
- **新手概念课堂**：Mamba 类模型（LFM2/LFM3）用卷积状态（conv state）代替 attention 的 KV cache。prefill checkpoint 是把 prefill 过程中的中间状态存下来，让后续 decode 能接着算——类似游戏存档。
- **对你有什么影响**：用 LFM2/LFM3 这类混合架构模型时，长 prompt 的 prefill 可以分段做并复用中间状态，首 token 延迟降低。

### 7. [Refactor][KV Cache] Remove EAGLE block dropping and default to lookahead hashes (#66234)
- **代码层面**：删除 `use_eagle_block_drop()` 相关逻辑；`SpeculativeConfig` 新增 `num_hash_lookahead_tokens` 字段；文档更新说明 EAGLE/MTP 不再需要"draft-attention proof tokens"才能复用 checkpoint。
- **新手概念课堂**：投机解码（EAGLE）会一次生成多个 draft token，这些 token 的 KV 缓存要不要提前算？旧方案是"先丢掉，验证通过再补"，新方案是"用 lookahead hash 提前算好"。后者更简单、更不容易错。
- **对你有什么影响**：开投机解码 + 前缀缓存时，行为更稳定，不再有"某些 checkpoint 不能复用"的隐藏限制。

### 8. [Core] Resolve the Mamba block size from the KV cache specs of all pipeline stages (#67994)
- **代码层面**：`vllm/v1/core/kv_cache_planning.py` 新增 `resolve_mamba_block_size()`，从所有 pipeline stage 的 KV cache spec 里取 Mamba block size；新增测试验证"只有 Mamba 层的 PP stage 也能用对 block size 做前缀缓存"。
- **新手概念课堂**：流水线并行（PP）把模型层切到不同 GPU 上。如果 stage 0 只有 Mamba 层、stage 1 才有 attention 层，stage 0 自己算不出 block size，必须看全局——就像接力赛，每棒要知道全程的格子线在哪。
- **对你有什么影响**：用 PP 部署混合架构模型（如 Granite 4.0）时，前缀缓存能正确命中，不会因为 block size 不一致而失效。

### 9. [Bugfix][Quantization] Load quantized Qwen3-Next MoE gate from INC/AutoRound checkpoints (#68120)
- **代码层面**：INC 配置新增 `checkpoint_unquantized_modules` 集合，通过读 safetensors 元数据判断某个 linear 是 `weight`（未量化）还是 `qweight`（量化），并支持 `WeightsMapper` 前缀重映射。
- **新手概念课堂**：Qwen3-Next 的 MoE gate 在 INC/AutoRound 检查点里可能是 BF16 未量化的，但周围层是量化的。如果统一按量化层加载，就会找不到 `qweight` 而报错——需要"逐模块看元数据"决定加载路径。
- **对你有什么影响**：用 Intel Gaudi 或 AutoRound 量化的 Qwen3-Next 模型能正常加载了，之前会报 key 缺失。

### 10. [Model][LoRA] Serve native Transformers GTE rerankers and their LoRA adapters (#68622)
- **代码层面**：新增 `GteForSequenceClassification` 架构支持，注册到 scoring 模型表；新增测试 `test_gte_native_lora.py`，验证 Transformers 原生 GTE 的 LoRA（作用于 split 的 q/k/v/gate/up）在 vLLM 融合层（qkv_proj/up_gate_proj）上结果一致。
- **新手概念课堂**：Transformers 原生 GTE 把 Q/K/V 分成三个独立矩阵，vLLM 为了性能把它们合并成一个 `qkv_proj`。LoRA 适配器是按原生命名训练的，加载到融合层时必须"切片拼装"——拼错位置结果就全错。
- **对你有什么影响**：可以直接用 `alibaba-nlp-community/gte-multilingual-reranker-base` 及其 LoRA 适配器做重排序，不用再依赖 `trust_remote_code`。

### 11. [Bugfix] Apply --enable-force-include-usage to streaming generate (#67249)
- **代码层面**：Rust 服务端 `generate.rs` 在 `generate_chunk_stream` 里把 `force_include_usage` 合并进 `include_usage` 和 `include_continuous_usage`；测试扩展到 `/inference/v1/generate` 路由。
- **新手概念课堂**：OpenAI 兼容 API 里 `stream_options.include_usage=true` 才会在流式响应末尾返回 token 用量。`--enable-force-include-usage` 是服务端强制开启，但之前漏了 `/inference/v1/generate` 这个新路由。
- **对你有什么影响**：用 Rust 版 vLLM server 的 `/inference/v1/generate` 流式接口时，能正确拿到 usage 统计了。

### 12. [Quantization] Keep E8M0 scale bytes when loading into uint8 params (#65256)
- **代码层面**：`test_load_plan.py` 引入 `_BYTE_CONTAINERS`，测试里用 `torch.randint(256).view(dtype)` 生成字节容器张量；LoadPlan 对 `router.bias` 这类 FP32 参数改为"直接 cast"而非走特殊路径。
- **新手概念课堂**：`torch.uint8` 在量化里常被当作"原始字节容器"——里面装的可能是 E8M0、E4M3 等格式。如果你对它做 `to(float32)`，就把位模式当整数解释了，scale 全错。正确做法是 `view()` 成目标 dtype。
- **对你有什么影响**：加载 MXFP8/MXFP4 量化模型时，缩放因子不再被悄悄破坏，精度更可靠。

### 13. [Bugfix] Fingerprint skipped multimodal components in the weight cache key (#66396)
- **代码层面**：`weight_cache` 的 key 计算加入 `multimodal_config` 相关字段（`language_model_only`、`mm_encoder_only`、`limit_mm_per_prompt` 的零值项）；新增测试验证"daemon 用 `--language-model-only` 启动时，需要视觉塔的引擎会 fallback 到磁盘加载"。
- **新手概念课堂**：权重缓存（weight cache）让多个引擎进程共享一份已加载的权重。但如果 daemon 只加载了语言模型（跳过视觉塔），而引擎需要视觉塔，两者权重就不一致——必须用 fingerprint 检测出来并回退。
- **对你有什么影响**：多模态模型用 weight cache 时不会因为 daemon 配置不同而拿到缺组件的权重，避免奇怪的推理错误。

### 14. [Bugfix][Pooling] Don't run chunked post-processing for /pooling (#67791)
- **代码层面**：`embed/io_processor.py` 把判断条件从 `not self.enable_chunked_processing` 改为 `ctx.chunked_embedding_metadata is None`；新增测试验证 `/pooling` 和 `/v1/embeddings` 在 chunked server 上结果一致。
- **新手概念课堂**：`/pooling` 和 `/v1/embeddings` 共用同一个 IO processor，但 `/pooling` 不做分块。如果按"服务端是否开启 chunked"判断，就会对 `/pooling` 错误地跑分块后处理——应该按"本次请求是否有分块元数据"判断。
- **对你有什么影响**：在开了 chunked processing 的服务器上调 `/pooling` 接口不再报错或返回错误结果。

### 15. [Security] Reject ICNS image inputs before decoding (#68245)
- **代码层面**：`vllm/multimodal/image.py` 从允许的 Pillow 格式列表里移除 `"ICNS"`，注释说明"解码器在加载前无法给出可靠尺寸"；新增测试验证 bytes/file/base64 三种入口都拒绝 ICNS。
- **新手概念课堂**：ICNS 是 macOS 图标格式，Pillow 的解码器在真正解码前拿不到图像尺寸，攻击者可以构造一个"声明很小、实际巨大"的文件耗尽内存。安全原则是"解码前先验证"，验证不了就拒绝。
- **对你有什么影响**：上传 ICNS 图片会被拒绝（返回 "Failed to load image"），其他常见格式（PNG/JPEG/WebP）不受影响。

### 16. [XPU] quick fix for _compact_sampling_mask_xpu (#68624)
- **代码层面**：`vllm/v1/worker/gpu/sample/output.py` 在 `_compact_sampling_mask_xpu` 开头加 `if logits.dtype != torch.float32: logits = logits.to(torch.float32)`。
- **新手概念课堂**：Intel XPU 的 `compact_sampling_mask` 算子只接受 float32 输入，但上游可能传 bf16/fp16。加一行 dtype 转换是最小修复——"快速修复"（quick fix）通常意味着没时间做完整重构。
- **对你有什么影响**：在 Intel GPU 上用结构化输出（如 JSON schema 约束）采样时不再因 dtype 不匹配而崩溃。

### 17. [Perf][Qwen4Exp] QSA: sm_100 split-K launch configs for small batches (#68842)
- **代码层面**：`qsa.py` 新增 `_is_sm100()` 缓存函数和 `_select_sm100_config()` 查表函数，按 `base_programs`（行数 × KV head 数）和是否 fp8 返回 `(BLOCK_N, target_splits, num_warps)`；`_select_config` 里插入 sm_100 分支。
- **新手概念课堂**：split-K 是把一次矩阵乘的 K 维度切给多个线程块并行算再求和。小 batch 时并行度不够，需要更多 split 来填满 GPU；大 batch 时 split 太多反而增加归约开销。不同架构的 SM 数量和缓存不同，最优 split 数也不同。
- **对你有什么影响**：在 B200/GB200 上跑 Qwen4Exp 的小 batch 解码，QSA attention 延迟降低（具体幅度取决于 batch 大小和是否 fp8）。

### 18. [Bugfix][Multimodal] Fix Qwen3-VL M-RoPE input positions after a pruned video (#68270)
- **代码层面**：新增测试 `test_qwen3_vl_mrope.py` 里的 `_build_prompt` 辅助函数，构造"文本 + 视频（部分帧被裁剪）+ 文本"的 token 布局，验证 M-RoPE 位置在裁剪后正确重算；修复逻辑在 Qwen3-VL 的 mrope 计算路径。
- **新手概念课堂**：M-RoPE 给每个 token 三个位置（t/h/w）。视频裁剪后，被裁掉的帧 token 不存在了，但后续 token 的 t 位置不能简单按"剩余 token 数"算——必须按"原本占了多少时间步"算，否则模型会误判视频时长。
- **对你有什么影响**：用 Qwen3-VL 处理带视频裁剪（EVS/VidCom2）的请求时，生成质量恢复正常，不再出现位置错乱导致的胡言乱语。

### 19. [Feat][Quantization] Online linear NVFP4 to MXFP4 (#61620)
- **代码层面**：`online.md` 文档新增"Serialized ModelOpt NVFP4 linear → any online linear method"支持行；测试 `test_online.py` 把 `source_format` 参数扩展到 `modelopt_nvfp4`，`projection` 增加 `merged_unequal` 场景；核心逻辑在 online quantization 的权重加载重量化路径。
- **新手概念课堂**：在线量化是"加载检查点时顺便转成另一种量化格式"。NVFP4（NVIDIA 的 4 位浮点）和 MXFP4（Open Compute 的 4 位浮点）scale 粒度不同，转换时要重新计算 scale 并处理不等长的 merged 投影。
- **对你有什么影响**：手里只有 NVFP4 检查点、但硬件更适合 MXFP4 时，可以直接 `--quantization mxfp4` 转换，不用离线重新量化；文档也提醒转换可能掉精度。

### 20. [RL][Model Loader] Reload directly by default with weight updates (VLLM_RELOAD_MODE=auto) (#68041)
- **代码层面**：`docs/training/direct_reload.md` 重写，说明 `VLLM_RELOAD_MODE=auto` 成为默认；`Refused Configurations` 列表重构（LoRA 在 auto 模式下回退 layerwise，而不是报错）；核心逻辑在 model loader 的 reload 路径选择。
- **新手概念课堂**：RL 训练时，训练进程不断产生新权重，推理引擎要"热更新"。layerwise 是逐层替换（省内存但慢），direct 是原地整体替换（快但要求每层都完整）。auto 模式 = "能 direct 就 direct，不能就自动降级 layerwise 并打日志"。
- **对你有什么影响**：跑 RLHF/RLVR 时，权重同步默认更快；如果用了 LoRA 或 torchao 等不支持的配置，会自动降级而不是直接报错，体验更平滑。

## sglang
## 💡 今日关键词

- **关键词：NPU/Ascend 生态适配**
  - 一句话通俗解释：把原本只在 NVIDIA GPU（CUDA）上跑的推理代码，改造到华为昇腾 NPU 上也能跑。
  - 为什么社区在关注：今天 20 个 commit 里有 4 个直接和 NPU 相关（Kimi K3.1、PD KV checksum、DeepSeek-V4.1、checksum 设备分派），说明国产算力平台正在从"能跑"走向"跑得好、跑得稳"。
  - 对新手意味着什么：如果你学大模型推理，NPU 是一个增量很大的方向，懂 CUDA 又愿意学 AscendC/CANN 的人非常稀缺。

- **关键词：AMD ROCm/gfx950 性能调优**
  - 一句话通俗解释：针对 AMD 的 MI350X 等新卡，微调 Triton kernel 的启动参数让矩阵乘法更快。
  - 为什么社区在关注：AMD commit 覆盖 batch-invariant、NIXL、AITER、FP8 MoE，说明 AMD 已经从"能跑通"进入"逐 kernel 抠性能"阶段。
  - 对新手意味着什么：异构计算不是口号，Triton 这种跨平台 kernel 语言是你切入 AMD 生态最省力的入口。

- **关键词：投机解码（Speculative Decoding）的稳定性**
  - 一句话通俗解释：用小模型"猜"几个 token，大模型再验证，加速生成；今天修的是它在 DP attention 下的死锁问题。
  - 为什么社区在关注：EAGLE 相关两个 commit 都是 fix，说明投机解码在复杂并行组合（DP attention + CUDA graph）下仍有边界 bug。
  - 对新手意味着什么：加速技术不是"开了就快"，理解并行模式之间的交互是进阶必修课。

- **关键词：量化（FP8/INT4/GPTQ）跨硬件落地**
  - 一句话通俗解释：把模型权重压成低精度数字，省显存、提速度。
  - 为什么社区在关注：今天有 AMD FP8 MoE 的 `is_shuffled` 标记修复、Intel CPU 上 GPTQ INT4 支持 GELU 激活，说明量化正在向更多硬件、更多激活函数扩展。
  - 对新手意味着什么：量化不只是"调个参数"，权重布局（shuffle/interleave）和激活函数的匹配是真正的坑。

- **关键词：可观测性与接口抽象（watermark / profiler / logprob）**
  - 一句话通俗解释：给采样加水印算法抽象层、修 profiler 停止逻辑、新增逐 token 的 logprob 行接口。
  - 为什么社区在关注：这些是"产品化"信号——推理框架开始重视可扩展接口和运维健壮性，而不只是吞吐。
  - 对新手意味着什么：学会写"算法无关的接口 + 适配器"是工程能力的分水岭。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：跨平台 kernel 分派（device dispatch）**
  - **是什么**：同一份 Python 代码，根据当前设备是 CUDA 还是 NPU，动态 import 不同的底层实现。比如 Commit 5 里 `checksum.py` 判断 `expected_device == "npu"` 就 import `sgl_kernel_npu` 的版本，否则用 `sglang.kernels` 的 CUDA 版本。
  - **为什么重要**：如果不做分派，NPU 上会 import 一个 CUDA-only 的 kernel 直接报错；反过来 CUDA 上误用 NPU 包也会崩。这是多硬件框架的"地基"。
  - **动手试试**：
    ```python
    import torch

    def adler32_dispatch(tensor):
        dev = tensor.device.type
        if dev == "npu":
            from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum
        elif dev == "cuda":
            from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
        else:
            raise RuntimeError(f"unsupported device: {dev}")
        return adler32_strided_checksum

    # 在你机器上打印当前可用设备
    print("cuda:", torch.cuda.is_available())
    try:
        import torch_npu  # noqa
        print("npu:", torch.npu.is_available())
    except ImportError:
        print("npu: torch_npu not installed")
    ```
  - **预期结果**：你会看到本机支持哪些设备；理解"先探测设备、再懒加载 kernel"的模式。这就是 Commit 5 的核心思路。

- **知识点 2：Triton kernel 的 launch 参数调优**
  - **是什么**：Triton 编译 kernel 时需要指定 `BLOCK_SIZE_M/N/K`、`num_warps`、`num_stages` 等参数，不同矩阵形状（M 小还是大）最优参数完全不同。
  - **为什么重要**：Commit 2 就是把 `M <= 32` 和 `M <= 64` 的分支从"通用路径"提到最前面，并配了新的 tile 参数 `(32,128,4,3)` 和 `(64,128,8,3)`，在 MI350X 上小 M 场景更快。选错参数可能慢 2 倍。
  - **动手试试**：
    ```python
    # 一个最小 Triton matmul 骨架，观察不同 BLOCK_SIZE 的影响
    import torch, triton, triton.language as tl

    @triton.jit
    def matmul_kernel(A, B, C, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(A + offs_m[:, None] * K + (k + offs_k)[None, :],
                        mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.)
            b = tl.load(B + (k + offs_k)[:, None] * N + offs_n[None, :],
                        mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.)
            acc += tl.dot(a, b)
        tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def run(M, N, K, BM, BN, BK, warps=4, stages=3):
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        matmul_kernel[grid](A, B, C, M, N, K, BM, BN, BK, num_warps=warps, num_stages=stages)
        return C

    import time
    for BM, BN in [(32, 64), (32, 128), (64, 128)]:
        run(32, 4096, 4096, BM, BN, 64)  # warmup
        t0 = time.time()
        for _ in range(20): run(32, 4096, 4096, BM, BN, 64)
        print(f"M=32 BLOCK=({BM},{BN}) -> {(time.time()-t0)/20*1000:.2f} ms")
    ```
  - **预期结果**：不同 BLOCK 组合耗时不同，你能亲眼看到"为什么 Commit 2 要给小 M 单独配置"。这就是 autotune 的本质。

- **知识点 3：投机解码 + DP attention 的死锁成因**
  - **是什么**：DP attention 把不同请求分到不同 rank 并行算 attention；MoE 层需要所有 rank 一起做 all-to-all/all-reduce 集合通信。如果某个 rank 因为"没拿到 DSA seed"退回 eager 模式，而其他 rank 还在走 CUDA graph，集合通信的参与方就对不上了 → 死锁。
  - **为什么重要**：Commit 14 加的条件 `not get_parallel().enable_dp_attention` 就是"DP attention 开启时禁止单 rank 回退"，Commit 3 则把旧字段 `enable_dp_attention` 换成新字段 `attn_dp_enabled`（字段重命名后的兼容修复）。
  - **动手试试**：
    ```python
    # 用两个进程模拟集合通信"参与方不一致"会卡住
    # save as deadlock_demo.py
    import os, torch, torch.distributed as dist

    def main():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29500",
                                rank=int(os.environ["RANK"]), world_size=2)
        rank = dist.get_rank()
        # rank 0 假装"回退到 eager"：跳过 all_reduce
        if rank == 0:
            print("rank0: skipping all_reduce (simulating eager fallback)", flush=True)
        else:
            print("rank1: entering all_reduce, will hang...", flush=True)
            dist.all_reduce(torch.ones(1))  # 永远等不到 rank0
        dist.destroy_process_group()

    if __name__ == "__main__":
        main()
    ```
    运行：
    ```bash
    RANK=0 python deadlock_demo.py & RANK=1 python deadlock_demo.py
    ```
  - **预期结果**：rank1 卡住不返回（Ctrl+C 结束）。你就直观理解了 Commit 14 说的"rank-local eager fallback can deadlock draft MoE collectives"。

- **知识点 4：量化权重的"shuffle 标记"为什么不能丢**
  - **是什么**：AITER 在 AMD 上会调用 `shuffle_weight` 把权重按特定 layout 重排，并在 tensor 上打一个 `is_shuffled = True` 标记。后续 kernel 靠这个标记判断"权重已经是重排过的，可以直接用"。
  - **为什么重要**：Commit 11 的 bug 是：把 tensor 包成 `nn.Parameter` 时，Python 的自定义属性 `is_shuffled` 被丢掉了，于是 AITER 的 gfx950 FP8 CK MoE 拒绝"未标记"的权重 → 报错。修复只有两行，但定位极难。
  - **动手试试**：
    ```python
    import torch, torch.nn as nn

    t = torch.randn(4, 4)
    t.is_shuffled = True
    print("before Parameter:", hasattr(t, "is_shuffled"))  # True

    p = nn.Parameter(t, requires_grad=False)
    print("after Parameter:", hasattr(p, "is_shuffled"))   # False ← 坑在这里

    # 修复方式：重新打标记
    p.is_shuffled = True
    print("after re-tag:", hasattr(p, "is_shuffled"))      # True
    ```
  - **预期结果**：亲眼看到 `nn.Parameter` 包装会丢失自定义属性。这是 PyTorch 元编程里非常经典的坑。

- **知识点 5：CPU 上的 INT4 MoE 与激活函数匹配**
  - **是什么**：Commit 12 让 Intel CPU 的 GPTQ INT4 MoE 支持 GELU 激活（之前只支持 SiLU）。做法是把 `CPUActMethod activation` 参数一路传进 kernel，在 fused store 阶段按激活函数分支计算。
  - **为什么重要**：Gemma 系列用 GELU，如果 kernel 硬编码 SiLU，模型输出就全错（但不会报错，更难查）。"静默错误"比崩溃更可怕。
  - **动手试试**：
    ```python
    import torch

    def silu_mul(gate, up):
        return torch.nn.functional.silu(gate) * up

    def gelu_mul(gate, up):
        return torch.nn.functional.gelu(gate) * up

    g = torch.randn(4); u = torch.randn(4)
    print("silu_mul:", silu_mul(g, u))
    print("gelu_mul:", gelu_mul(g, u))
    # 对比差异：如果用错激活，数值会系统性偏移
    print("diff:", (silu_mul(g, u) - gelu_mul(g, u)).abs().max())
    ```
  - **预期结果**：看到两种激活输出明显不同。你就理解了为什么 Commit 12 要专门为 INT4 加 GELU 支持——Gemma4 需要它。

---

## 🔥 重要更新

- **Commit 19 [NPU] DSV4.1 支持 DeepSeek-V4.1 on Ascend A5**：单 commit 近 7000 行 diff，新增 7 个 DSL 算子和 compressor 封装，是今天体量最大的硬件适配，标志着昇腾对新旗舰模型的支持节奏在加快。
- **Commit 10 [Feat] logprob: token_ids_logprob_rows**：新增"逐输入位置指定要打 logprob 的 token id"接口，对做 RLHF/蒸馏/评测的人非常实用——以前只能对固定 token 集合打 logprob。
- **Commit 17 [Refactor] watermark 算法中立接口**：把 Gumbel 水印挪到 `watermarking/gumbel/` 子包后面，新增 `--watermark-algorithm` 参数。这是"接口先行、算法后插"的教科书式重构。
- **Commit 14 [Fix] 投机解码 DP attention 死锁**：修复了一个能导致服务挂死的并发 bug，生产环境用户应当关注。

---

## 📋 逐条解读

### 1. [Feat] [NPU] kimi-k3.1: Add model support and route DSA attention path on Ascend
- **代码层面**：改了 3 处。`fused_sigmoid_gating_recurrent.py` 把 PTX 内联汇编开关从 `not _is_hip` 收紧为 `is_cuda() and not _is_hip`（NPU 走通用 add）；`triton_symm_mem_ag.py` 在非 CUDA 平台上直接禁用 multimem all-gather；`ascend_backend.py` 因为 Kimi K3.1 的 96 q heads / 32 kv heads 分组数=3 不是 2 的幂，把 q heads 复制补齐到 128 再切片。
- **新手概念课堂**：NPU 的 `npu_sparse_flash_attention` 算子要求 GQA 分组数是 2 的幂（像要求"每排座位必须坐 2/4/8 人"），Kimi 是 3 人一组，只好"多搬几把椅子凑成 4 人组，算完再把多出来的椅子撤掉"。
- **对你有什么影响**：在昇腾上跑 Kimi K3.1 从"跑不起来"变成"能跑"，但补齐 head 会带来少量算力浪费（128/96 ≈ 1.33 倍 attention 计算量）。

### 2. [Perf] [AMD] batch-invariant: Faster small-M launch for the segmented bf16 matmul on gfx950
- **代码层面**：`_segmented_matmul_launch()` 把 `M <= 32` / `M <= 64` 的判断提到 `N < 1024` 之后、`N <= 8192` 之前，并换成新 tile 参数；`_matmul_segmented_triton()` 里 reduce block 从固定 4096 改为 `1024 if M <= 64 else 4096`。测试里补了新的 launch 参数组合。
- **新手概念课堂**：batch-invariant 是指"同样的输入，不管 batch 多大，输出 bit 级一致"——这对可复现性和评测很重要。代价是要固定 kernel 的分块策略，牺牲一点灵活性换确定性。
- **对你有什么影响**：AMD MI350X 上小 batch 的 batch-invariant 推理会更快，普通用户感知是"小请求延迟降低"。

### 3. [Fix] speculative: Read the current DP attention field in EAGLE draft
- **代码层面**：`eagle_worker_v2.py` 一行改动，`get_parallel().enable_dp_attention` → `get_parallel().attn_dp_enabled`。
- **新手概念课堂**：这是字段重命名后的"漏网之鱼"。旧字段被废弃但没删，读旧字段会拿到默认值 False，导致本应禁用的 CUDA graph 被误启用。
- **对你有什么影响**：修复了 DP attention + EAGLE 组合下偶发的错误行为，属于稳定性修复。

### 4. [Fix] muse-spark: Honor decrypted config in architecture probe
- **代码层面**：`muse_spark_hook.py` 里给 `get_config()` 增加 `_configuration_file=cfg.decrypted_config_file` 参数，让架构探测读取解密后的配置文件而不是原始加密文件。
- **新手概念课堂**：有些模型仓库的 `config.json` 是加密的，需要先解密到临时文件再读。如果探测阶段读了加密原文，就解析不出 architecture，后续逻辑全错。
- **对你有什么影响**：使用 muse-spark 加密 checkpoint 的用户现在能正确识别模型架构。

### 5. [Feat] [NPU] disaggregation: Support PD KV checksum
- **代码层面**：`checksum.py` 把设备断言从"必须 CUDA"放宽为"CUDA 或 NPU"，并根据设备类型懒加载 `sgl_kernel_npu` 或 `sglang.kernels` 的 `adler32_strided_checksum`；新增 NPU 集成测试文件。
- **新手概念课堂**：PD 分离（Prefill/Decode disaggregation）时，KV cache 要从 prefill 节点传到 decode 节点，checksum 用来校验"传过去的 KV 没坏"。以前只有 CUDA 实现，NPU 上跑 PD 分离就没法校验。
- **对你有什么影响**：昇腾上做 PD 分离部署时，KV 传输的完整性校验能力补齐了。

### 6. [Fix] [AMD] nixl: Bind AITER's HIP runtime before UCX hooks dlopen
- **代码层面**：新增 `pin_aiter_hip_runtime()` 函数，在创建 NIXL agent 之前强制 import 并调用 AITER 的 `load_hip_runtime()`，且在 `conn.py` 和 `hicache_nixl.py` 两处调用。
- **新手概念课堂**：UCX 会 hook `dlopen` 来跟踪内存注册。AITER 加载 HIP runtime 时靠 soname 查找，被 hook 干扰后会加载第二份 HIP runtime，两份 LLVM 冲突直接 abort。修复就是"在 UCX 装 hook 之前先把 AITER 的 runtime 绑好"。
- **对你有什么影响**：AMD 上用 NIXL 做 KV 传输不再随机崩溃。这类"动态链接顺序"bug 极难复现，属于典型的环境问题。

### 7. [Fix] diffusion: Keep FA4 for Blackwell sub-tile attention in the eager breaks of a graph replay
- **代码层面**：`sdpa.py` 的 `_use_cudnn_sdpa()` 条件从 `not is_in_breakable_cuda_graph()` 改为 `not is_in_breakable_cuda_graph() and get_current_replay_token() is None`，即"graph replay 期间的 eager break 也算在 graph 内"。文档和测试同步更新。
- **新手概念课堂**：breakable CUDA graph 允许图里"打断"去跑 eager 代码。之前判断逻辑没考虑"replay 期间的 break 仍属于图上下文"，导致这部分误用了 cuDNN 而不是更快的 FA4。
- **对你有什么影响**：Blackwell 上 diffusion 模型（如视频生成）的 attention 在 graph replay 场景下性能更好。

### 8. [Fix] diffusion: stop the worker profiler even if the HTTP profiler fails
- **代码层面**：`http_server.py` 的 `stop_profile()` 重构为三段 try/finally：HTTP server profiler、scheduler profiler 各自独立 try，`finally` 里统一清理状态。host 失败返回 500，worker 失败返回 200 + warning。
- **新手概念课堂**：原来所有逻辑在一个大 try 里，HTTP profiler 一挂，后面的 worker profiler 就不会停，状态也不会清，profiler 永远"开着"持续拖慢服务。
- **对你有什么影响**：profiling 结束后不会残留"僵尸 profiler"，服务性能恢复正常。

### 9. [Fix] [AMD] unquant: Keep CPU inputs off the AITER GEMM path
- **代码层面**：`unquant.py` 的 `apply()` 里给 AITER 分支加条件 `and x.is_cuda`。
- **新手概念课堂**：AITER GEMM 只接受 GPU tensor，CPU tensor 传进去会直接崩。这个 bug 在"部分层跑 CPU、部分跑 GPU"的混合场景下触发。
- **对你有什么影响**：AMD 上做 CPU offload 或混合推理不再崩溃，自动回退到 `F.linear`。

### 10. [Feat] logprob: Add token_ids_logprob_rows
- **代码层面**：新增 `_RaggedRows` dataclass（拼接存储变长行 + 长度列表，一次 `.tolist()` 拷贝后切片），`LogitsMetadata.token_ids_logprobs` 类型扩展为 `List[Optional[Union[List[int], TokenIdRows]]]`，文档新增参数说明。
- **新手概念课堂**：原来 `token_ids_logprob` 是"每个请求一组 token id"，新接口是"每个输入位置一组 token id"——比如你想在输入的第 5 个位置查 token 100 和 200 的 logprob，第 7 个位置查 token 300 的 logprob。`_RaggedRows` 是处理变长行的经典优化（避免逐行 H2D 拷贝）。
- **对你有什么影响**：做 RLHF 奖励建模、蒸馏、token 级评测时，可以精确指定每个位置要打分的 token，不用再全词表计算。

### 11. [Fix] [AMD] fp8-moe: Mark the AITER-shuffled per-tensor FP8 MoE weights is_shuffled
- **代码层面**：`fp8.py` 的 `process_weights_hip_scale_padding()` 里，在 `nn.Parameter` 包装后补两行 `layer.w13_weight.is_shuffled = True` / `layer.w2_weight.is_shuffled = True`。
- **新手概念课堂**：见知识点 4。`shuffle_weight` 给 tensor 打标记，`nn.Parameter` 包装丢标记，AITER kernel 拒绝未标记权重。
- **对你有什么影响**：AMD gfx950 上跑 per-tensor FP8 MoE 不再报错。

### 12. [Fix] [Intel] gemma4: Enable GPTQ INT4 inference on CPU
- **代码层面**：`gemm.h` / `moe.cpp` / `moe_int4.cpp` 三处：`fused_experts_int4_w4a8_kernel_impl` 新增 `CPUActMethod activation` 参数；`fused_experts_cpu` 的激活校验从"INT8/INT4 都只支持 SiLU"改为"INT8 只支持 SiLU，INT4 支持 SiLU 或 GELU"；kernel 内部按 activation 分支计算。
- **新手概念课堂**：见知识点 5。Gemma 系列用 GELU，之前 INT4 MoE 硬编码 SiLU 会静默输出错误结果。
- **对你有什么影响**：Intel CPU 上跑 Gemma4 GPTQ INT4 从"结果错误"变成"结果正确"。

### 13. [Doc] cookbook: Align Hy4 installation with the Docker guide
- **代码层面**：纯文档。`Hy4.mdx` 的安装段落改成引用官方 install 指南 + Docker 标签页，`Hy4-flash.mdx` 加 `tag: NEW`，Usage examples 补了示例输出和说明。
- **新手概念课堂**：文档里"复制粘贴就能跑"的命令如果过时，新用户第一步就卡住。这次是把散落的安装说明统一到单一来源（install 指南），避免多处维护不同步。
- **对你有什么影响**：照着 cookbook 部署 Hy4 更不容易踩坑。

### 14. [Fix] speculative: Prevent draft deadlock under DP attention when the DSA seed is missing
- **代码层面**：`eagle_worker_v2.py` 的 `draft()` 里，在"是否禁用 decode CUDA graph"的条件中增加 `and not get_parallel().enable_dp_attention`。
- **新手概念课堂**：见知识点 3。DP attention 下各 rank 必须同步走集合通信，单 rank 回退 eager 会导致其他 rank 永久等待。
- **对你有什么影响**：DP attention + EAGLE 投机解码组合不再偶发挂死。这是生产环境的关键修复。

### 15. [Perf] [NVIDIA] fused_moe: Add H20-3e configs for GLM-5.3-Flash TP8/TP4
- **代码层面**：新增 JSON 配置文件 `E=288,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json`，为不同 M（1~64+）指定 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages、USE_TMA。
- **新手概念课堂**：这是 Triton MoE kernel 的"调参表"——不同 M（token 数）下最优 tile 配置不同，预先 benchmark 好存成 JSON，运行时按 M 查表。`USE_TMA` 是 Hopper/Blackwell 的 Tensor Memory Accelerator，能大幅提速数据搬运。
- **对你有什么影响**：H20-3e 上跑 GLM-5.3-Flash TP8/TP4 的 MoE 层更快。

### 16. [Perf] [NVIDIA] dspark: Stage draft DP-MoE token counts through persistent buffers
- **代码层面**：`dspark_draft.py` 删除每次调用都 `torch.tensor(...).to(device)` 的 `_make_num_token_non_padded()`，改为在 `__init__` 里预分配 `_dp_meta_host` / `_dp_meta_device` 持久 buffer，用 slot 轮转 + `_dp_meta_copied` 列表管理异步拷贝。
- **新手概念课堂**：每次 forward 都新建 pinned tensor 再 H2D 拷贝，会产生大量小分配和同步开销。持久 buffer + 多 slot 轮转是 CUDA 编程的标准优化手法（类似 double/triple buffering）。
- **对你有什么影响**：dspark 投机解码的 draft 阶段 CPU 开销降低，小 batch 场景延迟改善。

### 17. [Refactor] sampling: Add an algorithm-neutral watermark interface and move Gumbel behind it
- **代码层面**：新增 `WATERMARK_ALGORITHM_CHOICES = ["gumbel"]` 和 `--watermark-algorithm` 参数；把 `watermarking/detector.py` 挪到 `watermarking/gumbel/detector.py`；新增 `sampling/watermarking/runtime.py` 作为算法适配层；文档同步更新 import 路径。
- **新手概念课堂**：这是"策略模式"的典型应用——上层只认 `--watermark-algorithm` 名字，具体算法通过 adapter 注册。以后加 Kirchenbauer 等其他水印算法只需新增子包 + 注册，不动上层。
- **对你有什么影响**：目前行为不变（默认还是 gumbel），但为未来多算法铺路。注意：直接 import `watermarking.detector` 的外部代码需要改成 `watermarking.gumbel.detector`。

### 18. [Doc] docs: Fix indentation, spelling and formatting issues
- **代码层面**：纯文档。修了 markdown 列表缩进、代码块前空行、`DDR→AICORE` → `DDR→AI Core` 拼写、Python 列表末尾逗号、目录树缩进等。
- **新手概念课堂**：MDX 对缩进敏感，列表项后紧跟代码块如果没空行，渲染会出错。这类"文档 lint"看似琐碎，但直接影响文档可读性和渲染正确性。
- **对你有什么影响**：Ascend 开发文档显示更规范。

### 19. [Feat] [NPU] DSV4.1: Support DeepSeek-V4.1 on Ascend A5
- **代码层面**：新增 `ascend_dsv4_backend.py` 的 `_ds41_ops()`（functools.cache 缓存 7 个 DSL 算子：`compressor_v2`、`mixed_quant_sparse_flash_mla_dsl` 等）；`row_argmax.py` 去掉 `CUDA_ONLY` capability 并在 NPU 上分派到 `sgl_kernel_npu.operators.sample.div_argmax`；引入 `RaggedVerifyMode` 支持 ragged verify。
- **新手概念课堂**：DSL（Domain Specific Language）算子是指用特定框架（这里是 sgl-kernel-npu 的 DSL）写的 kernel，比手写 AscendC 更易维护。`functools.cache` 保证算子只 import 一次，避免重复加载开销。
- **对你有什么影响**：昇腾 A5 上可以跑 DeepSeek-V4.1 完整推理（含投机解码的 ragged verify 路径）。

### 20. [Fix] [NVIDIA] qwen3.5: Give text-only Qwen3.5 checkpoints the Qwen3 MoE family overrides
- **代码层面**：`qwen3_moe.py` 的架构列表新增 `"Qwen3_5MoeForCausalLM"` 和 `"Qwen3_5ForCausalLM"`（纯文本版本），测试新增 `test_qwen3_moe_family_covers_text_only_qwen3_5`。
- **新手概念课堂**：模型家族 override 是按 architecture 名匹配的。Qwen3.5 有多模态版（`ForConditionalGeneration`）和纯文本版（`ForCausalLM`），之前只列了多模态版，导致纯文本版（如 Qwen3.8-2.4T-A95B）拿不到 SM100 上的 `flashinfer_trtllm` MoE runner 规则，ModelOpt NVFP4 权重被错误准备。
- **对你有什么影响**：纯文本 Qwen3.5 在 SM100（Blackwell）上跑 NVFP4 MoE 不再出现"权重准备和 kernel 不匹配"的问题。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)