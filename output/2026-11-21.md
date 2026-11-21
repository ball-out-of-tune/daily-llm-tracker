# 🚀 vLLM & SGLang 每日更新 — 2026-11-21

> 自动生成于 2026-11-21 08:44 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：安全加固（Security Hardening）**
  - 一句话通俗解释：给服务器加"限流阀门"，防止恶意用户用超大请求把服务打挂。
  - 为什么社区在关注：今天 20 个 commit 里有 3 个是安全相关（#53473、#68249、#53131），说明 vLLM 开始把"面向公网部署"当成正经需求，DoS 防护、整数溢出这类"老坑"正在被系统性地清理。
  - 对新手意味着什么：如果你以后要部署推理服务，理解"输入上限/资源预算"是必备技能，别以为模型跑通就完事了。

- **关键词：ROCm / AMD 生态适配**
  - 一句话通俗解释：让 vLLM 在 AMD GPU 上也能跑得又快又稳。
  - 为什么社区在关注：今天有 5 个 ROCm 相关 commit（#61214、#66586、#65935、#63725、#68012、#60839），从 MoE kernel 到 attention warmup 到 bugfix 全覆盖，说明 AMD 正在成为真正的"第二平台"。
  - 对新手意味着什么：只学 CUDA 已经不够了，HIP/ROCm 的岗位在增长，异构编程是加分项。

- **关键词：多模态 Encoder 加速（CUDA Graphs for Vision/Audio）**
  - 一句话通俗解释：把图像/音频编码器的执行路径"录下来"反复重放，省掉每次启动开销。
  - 为什么社区在关注：#68831 把 CUDA Graph 从视觉扩展到 Qwen3-Omni 的音频塔，说明多模态推理的性能瓶颈正在从 LLM 主干转移到 encoder。
  - 对新手意味着什么：想搞推理优化，多模态 encoder 是个还没被卷烂的方向。

- **关键词：量化后端多样化（MXFP8 / FP8 / Compressed-Tensors）**
  - 一句话通俗解释：同一份模型权重可以用不同"压缩格式"存，选对格式能省一半显存。
  - 为什么社区在关注：#61214 新增 AITER Triton MXFP8 MoE 后端，#66586 修 ROCm 上 FP8 KV scale 的 bug，说明低精度格式（MXFP8/MXFP4/FP8）正在快速铺开。
  - 对新手意味着什么：量化不再是"调个 flag"，要理解 block-wise scale、fnuz 格式差异这些细节。

- **关键词：分布式模型加载与通信正确性**
  - 一句话通俗解释：多机多卡时，怎么让每台机器高效读权重、怎么让 all-reduce 不出错。
  - 为什么社区在关注：#68351 修了多节点预取权重的 bug，#60839 修了 AITER 自定义 all-reduce 对 int8 的崩溃，说明大集群部署的坑还很多。
  - 对新手意味着什么：分布式不是"加 --tensor-parallel-size 就完事"，page cache、通信 dtype 都是真问题。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：请求级资源预算（Request-wide Budget）——防止 DoS 的通用模式

- **是什么**：一次 HTTP 请求可能带很多条 prompt、很多张图、很多段音频。如果不限制"这次请求总共能消耗多少资源"，攻击者发一个超大 batch 就能把显存/内存打爆。vLLM 现在用 `VLLM_MAX_COMPLETION_PROMPTS`、`VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`、`VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB` 这类环境变量给"整个请求"设上限。
- **为什么重要**：真实部署中，一个恶意请求可以让整个服务 OOM 重启，影响所有用户。不懂这个，你的服务上线第一天就可能被扫。
- **动手试试**：
  ```bash
  # 启动一个带 prompt 上限的 embedding 服务
  VLLM_MAX_COMPLETION_PROMPTS=8 \
  vllm serve BAAI/bge-small-en-v1.5 --task embed --port 8000

  # 另开一个终端，发 9 条 prompt，应该被拒绝
  python - <<'PY'
  import requests
  r = requests.post("http://localhost:8000/v1/embeddings", json={
      "model": "BAAI/bge-small-en-v1.5",
      "input": ["hello"] * 9,
  })
  print(r.status_code, r.text[:300])
  PY
  ```
- **预期结果**：返回 4xx 错误，提示超过 `VLLM_MAX_COMPLETION_PROMPTS` 限制；改成 8 条则返回 200。你就理解了"请求级限流"是怎么落地的。

---

### 知识点 2：CUDA Graph 与多模态 Encoder 的"录制-重放"

- **是什么**：CUDA Graph 是把一段 GPU 计算序列"录"成一个静态图，之后每次用不同输入重放，省掉 kernel launch 的 CPU 开销。vLLM 之前只对 LLM 主干做，现在扩展到视觉/音频 encoder（#68831、#68739）。
- **为什么重要**：多模态场景下，encoder 的 kernel 数量可能比 LLM 主干还多，launch 开销占比高。开启后吞吐能明显提升，但要注意"动态 shape"（如图片数量变化）如何处理——vLLM 用 token budget 分桶。
- **动手试试**：
  ```bash
  # 对比 eager vs cudagraph encoder 的日志/吞吐
  vllm serve Qwen/Qwen3-ASR-0.6B \
    --compilation-config '{"cudagraph_mm_encoder": false}' --port 8001
  # 另开终端跑一次推理，记录吞吐
  # 再换成 true 重启，对比
  ```
  或者直接跑 vLLM 的测试：
  ```bash
  pytest tests/models/multimodal/generation/test_audio_encoder_cudagraph.py -v
  ```
- **预期结果**：测试会对比 eager 和 cudagraph 两种模式的 logprobs，要求"接近"（`check_logprobs_close`），说明 graph 化不改变数值结果，只是加速。

---

### 知识点 3：整数溢出与 int32 → int64 指针算术

- **是什么**：CUDA kernel 里 `blockIdx.x * hidden_size` 如果都是 int32，当 token 数 × hidden_size 超过 2^31（约 21 亿）时会溢出成负数，导致越界读写。#53131 把这类乘法提升到 int64。
- **为什么重要**：大模型 + 大 batch 时，`num_tokens * hidden_size` 很容易逼近 2^31。溢出不是崩溃，是"静默写错内存"，最难查的 bug。
- **动手试试**：
  ```python
  # 在 Python 里模拟 int32 溢出（numpy）
  import numpy as np
  a = np.int32(2**30)
  b = np.int32(4)
  print("int32:", a * b)          # 溢出成 0
  print("int64:", np.int64(a) * np.int64(b))  # 正确
  ```
  再读一下 `csrc/layernorm_kernels.cu` 里 `const int64_t token_idx = blockIdx.x;` 这一行。
- **预期结果**：看到 int32 乘法溢出成 0 或负数，你就明白为什么 kernel 里要把索引提升到 int64。

---

### 知识点 4：量化 KV Cache 与 fnuz 格式的 scale 修正

- **是什么**：FP8 有两种格式：e4m3（NVIDIA）和 e4m3fnuz（AMD ROCm）。fnuz 的表示范围不同，同一份 scale 在 fnuz 上需要乘一个修正系数（通常 2 倍）才能对齐。#66586 修的就是 compressed-tensors 在 ROCm 上没乘这个系数的问题。
- **为什么重要**：不修的话，AMD 上 FP8 KV cache 的数值会偏，生成质量下降但不报错——典型的"静默精度 bug"。
- **动手试试**：
  ```python
  # 观察 e4m3 与 e4m3fnuz 的表示范围差异
  import torch
  x = torch.tensor([448.0, 0.5, 1e-3])
  print("e4m3:", x.to(torch.float8_e4m3fn))
  # fnuz 在 PyTorch 里没有直接 dtype，但可以理解：最大值约 240 而非 448
  ```
  然后读 `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` 里 `process_weights_after_loading` 新增的 fnuz 分支。
- **预期结果**：理解为什么同一个 scale 值在两种格式下语义不同，以及"幂等性"（第二次调用不能再乘一次）为什么重要——测试里专门验证了 `data_ptr` 不变。

---

### 知识点 5：多节点 page cache 预取——"每台机器都要读全量"

- **是什么**：加载大模型时，vLLM 会先把 checkpoint 文件读一遍"预热"到操作系统 page cache，加速后续加载。多机场景下，page cache 是**每台机器独立**的，所以每个节点都必须读全量文件；#68351 之前错误地按全局 rank 切分，导致某些节点只读了部分文件。
- **为什么重要**：多机部署时，如果某节点没预热到全部文件，加载会变慢甚至抖动，但不会报错——又是一个"性能静默 bug"。
- **动手试试**：
  ```bash
  # 观察 page cache 效果（Linux）
  # 先清缓存（需要 root，谨慎）
  # sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  # 第一次加载计时
  time python -c "from vllm import LLM; LLM('facebook/opt-125m')"
  # 第二次加载计时（page cache 命中，应更快）
  time python -c "from vllm import LLM; LLM('facebook/opt-125m')"
  ```
  再跑 vLLM 的单元测试看切分逻辑：
  ```bash
  pytest tests/model_executor/model_loader/test_prefetch_checkpoints.py -v
  ```
- **预期结果**：第二次加载明显更快；测试断言"每个 node 都拿到了全部 FILES"，理解"按节点切分而非按全局 rank 切分"的设计。

---

## 🔥 重要更新

1. **#53473 安全：pooling/scoring 路由加 prompt 数量上限** —— 之前只有 `/v1/completions` 有防 fan-out 限制，现在 `/v1/embeddings`、`/pooling`、`/classify`、`/score`、`/rerank` 全部加上，公网部署必须关注。
2. **#68249 安全：批量请求共享媒体字节预算** —— 之前每个子请求独立算预算，攻击者可以用一个大 batch 绕过；现在整个请求共享一份预算，是 DoS 防护的关键一环。
3. **#68831 性能：Qwen3-Omni / Qwen3-ASR 音频塔支持 Encoder CUDA Graph** —— 多模态推理性能优化的新战场，音频场景吞吐有望明显提升。
4. **#61214 ROCm：新增 AITER Triton MXFP8 MoE 后端（gfx942）** —— MI325X 上 MXFP8 专家不再被反量化成 BF16，权重显存从 196 GiB 降到 109 GiB，是 AMD 大模型部署的重要一步。

---

## 📋 逐条解读

### 1. fix(security): enforce prompt-count bound on pooling and scoring routes (#53473)

- **代码层面**：在 `vllm/entrypoints/pooling/` 下的 `pooling/protocol.py`、`scoring/protocol.py`、`classify/protocol.py` 里，给请求模型的 validator 加上 `VLLM_MAX_COMPLETION_PROMPTS` 检查；新增测试 `test_pooling_prompt_limit.py`；文档 `docs/models/pooling_models/README.md` 补充说明。
- **新手概念课堂**：这就像餐厅规定"一桌最多点 20 道菜"——不是不让你吃，是防止有人一次点 10000 道把厨房堵死。`VLLM_MAX_COMPLETION_PROMPTS` 就是那个"每桌上限"。
- **对你有什么影响**：如果你用 embedding/rerank 接口批量送数据，超过 1024 条（默认值）会被拒，需要拆批或调大环境变量。

---

### 2. [Security] Share media byte budgets across batched requests (#68249)

- **代码层面**：在 `vllm/renderers/chat_utils/item_tracker.py` 引入 `MediaRequestBudgets` 和 `use_media_request_budgets` 上下文管理器；`PoolingIOProcessor` 等批量处理路径在进入时建立共享预算，所有子请求从同一预算扣减；测试 `test_io_processor_unit.py` 验证"只共享本请求的预算"。
- **新手概念课堂**：好比一个旅行团共用一个"行李总重量额度"，不是每人独立额度。之前每人 20kg，10 人团能带 200kg；现在全团共享 20kg，堵住了批量绕过。
- **对你有什么影响**：批量发多图/多音频的请求，总字节数受限于 `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` / `VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB`，需要合理拆批。

---

### 3. [Security] Widen int32 pointer arithmetic to int64 in CUDA kernels (#53131)

- **代码层面**：`csrc/layernorm_kernels.cu`、`csrc/layernorm_quant_kernels.cu` 等文件里，把 `blockIdx.x * hidden_size`、`input_stride` 等索引计算从 `int` 改为 `int64_t`，避免大 batch 下溢出。
- **新手概念课堂**：int32 像一个只能数到 21 亿的计数器，超过就"绕回负数"。int64 能数到 9 百亿亿，够用了。CUDA kernel 里所有"元素下标 × 步长"的乘法都要警惕。
- **对你有什么影响**：普通用户感知不到，但在超大 batch、超长序列场景下避免了静默的内存越界（可能导致乱码输出或崩溃）。

---

### 4. [Bugfix][Multimodal] Convert PIL video frames to RGB like PIL images (#69089)

- **代码层面**：`vllm/multimodal/parse.py` 新增 `_convert_pil_frames`，在 `_get_video_with_metadata` 里对 list/tuple 形式的 PIL 帧做 `convert_image_mode(frame, "RGB")`；测试 `test_parse.py` 覆盖 L/LA/RGBA/P 四种模式。
- **新手概念课堂**：PIL 图片有"调色板模式（P）""灰度（L）""带透明（RGBA）"等，模型通常只吃 3 通道 RGB。之前图片会转，视频帧忘了转，导致通道数不一致报错。
- **对你有什么影响**：传视频帧列表做多模态推理时，非 RGB 的帧不再报错，行为与传单张图片一致。

---

### 5. [Bugfix][Multimodal] Size rope_2d DP vision outputs from the spatial grid (#68739)

- **代码层面**：`vllm/model_executor/models/vision.py` 里 `run_dp_sharded_mrope_vision_model` 对 `rope_2d` 类型，输出长度从"t×h×w"改为按"空间网格 + 时间池化"计算（如 Kimi-K2.5 的 MoonViT 会把 t 帧池化成 1 帧）；测试新增 `SimpleTemporalPoolVisionModel` 验证。
- **新手概念课堂**：视频有"时间维度"，有些视觉塔会把多帧"平均"成一帧（temporal pooling）。如果还按"帧数 × 高 × 宽"算输出长度，就会多分配内存、对不齐。
- **对你有什么影响**：使用 Kimi-K2.5 等带时间池化的视觉模型做视频推理时，DP（数据并行）切分不再出错。

---

### 6. [Perf][Multimodal] Encoder CUDA graphs for the Qwen3-Omni / Qwen3-ASR audio tower (#68831)

- **代码层面**：`vllm/v1/cudagraph/` 下的 encoder graph manager 扩展到音频塔；`Qwen3ASRForConditionalGeneration`、`Qwen3OmniMoeThinkerForConditionalGeneration` 注册 graph 支持；文档 `docs/design/cuda_graphs_multimodal.md` 新增音频表格；测试 `test_audio_encoder_cudagraph.py` 对比 eager 与 graph 的 logprobs。
- **新手概念课堂**：CUDA Graph 像"录制宏"——把一串 GPU 操作录下来，下次一键重放，省掉每次"按按钮"（kernel launch）的开销。音频塔的 conv 前端仍在 eager 跑（因为形状变化大），transformer 层和输出投影走 graph。
- **对你有什么影响**：Qwen3-ASR / Qwen3-Omni 音频推理吞吐提升；注意 token budget 最小是 104（一个 attention window，约 8 秒音频），太短的音频可能享受不到 graph 收益。

---

### 7. [Bugfix] Honor batch invariance in manual FlashInfer norm fusion (#68600)

- **代码层面**：`vllm/models/common/ops/fused_allreduce_rms_norm.py` 里手动融合的 all-reduce + RMSNorm 路径，在 `VLLM_BATCH_INVARIANT=1` 时改用与 batch size 无关的规约顺序；新增测试 `test_fused_allreduce_rms_norm_batch_invariant.py`，验证"单条 vs 大批量"结果 bit-exact。
- **新手概念课堂**：Batch Invariance 指"同样一条输入，不管跟谁一起 batch，输出必须完全一样"。GPU 的并行规约顺序会随 batch 大小变，导致浮点结果有微小差异。对需要可复现的评测/审计场景很关键。
- **对你有什么影响**：开启 `VLLM_BATCH_INVARIANT=1` 后，TP + FlashInfer 融合路径也能保证 batch 无关性，代价是可能略慢。

---

### 8. [Attention][FlashInfer] DiffusionGemma: mixed causal/bidirectional attention on TRTLLM-Gen (#62516)

- **代码层面**：`vllm/v1/attention/backends/flashinfer.py` 新增 `_fill_mixed_causal_metadata`，支持"同一 batch 里有的请求 causal、有的 bidirectional"；SM100/SM103 上文本-only 场景默认启用；测试 `test_mixed_causal_attn.py` 覆盖 bool/int32 两种 causal dtype 和 CUDA graph padding。
- **新手概念课堂**：Causal 是"只能看前面"（生成模型），bidirectional 是"前后都能看"（编码/扩散模型）。DiffusionGemma 需要混合模式，就像一条流水线上有的工位只能看上游、有的能看全图。
- **对你有什么影响**：Blackwell（SM100）上跑 DiffusionGemma 文本任务会自动走这个后端，性能更好；其他 GPU 仍走旧路径。

---

### 9. [Bugfix][DSv4.1] Read the DSpark draft's native mega MoE flag from its layers (#69023)

- **代码层面**：`vllm/models/deepseek_v41/nvidia/specialized_decode/runtime/draft.py` 里，判断条件从 `backbone.use_native_mega_moe` 改为 `backbone.layers[0].ffn.use_native_mega_moe`。
- **新手概念课堂**：模型对象上的属性可能"过期"——`backbone.use_native_mega_moe` 是构造时的快照，而真正决定行为的是每一层 FFN 的配置。就像查"公司是否远程办公"要看每个部门的实际政策，而不是总部的初始公告。
- **对你有什么影响**：DSv4.1 的 latency_specialized 解码后端在 mega MoE 配置下不再误报错。

---

### 10. [HiSparse] Support HY4 Flash (#69026)

- **代码层面**：`vllm/model_executor/layers/attention/sparse_mla_attention.py` 里，当 `sliding_window is not None` 时不再把 `vllm_config` 传给 index group builder（即滑动窗口层不参与 HiSparse）；`vllm/models/hy_v4_flash/nvidia/model.py` 放宽 KV transfer 限制，只允许 `HiSparseConnector`；测试验证滑动窗口层 `hisparse_cache is None`。
- **新手概念课堂**：HiSparse 是一种"稀疏 KV 缓存"技术，只对全注意力层有意义。滑动窗口层的 KV 由滑动窗口管理器管，HiSparse 去接管会"抢错地盘"导致首次写入失败。
- **对你有什么影响**：HY4 Flash 模型在 gfx942 上可以用 HiSparse + 滑动窗口共存，不再启动即崩。

---

### 11. [Feature] Encoder-only attention hooks for model-owned layouts (#67798)

- **代码层面**：新增 `EncoderOnlyAttention(attn_backend=...)` 允许模型显式指定 encoder 注意力后端；`FlexAttentionMetadataBuilder._make_metadata` 开放 hook；新增 `register_model_config` 让插件注册模型配置；测试 `test_encoder_only_attention_hooks.py` 覆盖 backend 选择、警告日志、metadata 构建。
- **新手概念课堂**：以前 vLLM 的 attention 后端是"全局选一个"，现在允许"模型自己说我这个 encoder 要用 FlexAttention"。就像公司统一采购电脑，但设计部门可以申请特定型号。
- **对你有什么影响**：模型插件作者（如自定义视觉塔）可以精确控制 encoder 注意力实现，不用改 vLLM 核心。

---

### 12. [Benchmark] Add decisions backends to vllm bench serve (#68760)

- **代码层面**：`vllm/benchmarks/lib/endpoint_request_func.py` 新增 `vllm-systemone` 和 `openai-decisions` 两个 backend；文档 `docs/benchmarking/cli.md` 新增 "Decisions Benchmark" 章节；测试 `test_decisions_request_func.py` 验证请求 schema。
- **新手概念课堂**：`vllm bench serve` 是压测工具，"backend"决定它怎么构造请求。新增的 decisions backend 用于压测"结构化决策"接口（一次请求问多个 yes/no/score 问题）。
- **对你有什么影响**：想压测 `/v1/systemone` 或 `/v1/decisions` 接口，现在有官方支持，不用自己写脚本。

---

### 13. [Bugfix][ROCm] Apply the e4m3fnuz KV-scale factor for compressed-tensors (#66586)

- **代码层面**：`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` 的 `CompressedTensorsKVCacheMethod.process_weights_after_loading` 里，检测 `current_platform.is_fp8_fnuz()`，对 k/v scale 乘 2（fnuz 范围是 e4m3 的一半），并保证幂等（第二次调用不重复乘）；测试覆盖 fnuz/非 fnuz、有/无 q_scale 四种组合。
- **新手概念课堂**：e4m3fnuz 是 AMD 的 FP8 格式，最大表示值约 240，而 NVIDIA e4m3 是 448。同样一个 scale 值，在 fnuz 上需要乘 2 才能表达同样的实际范围。就像"摄氏度和华氏度"的换算系数。
- **对你有什么影响**：AMD GPU 上用 compressed-tensors 的 FP8 KV cache，数值精度恢复正常，不再悄悄偏移。

---

### 14. [ROCm][MoE] Add AITER Triton MXFP8 MoE backend for gfx942 (#61214)

- **代码层面**：新增 `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp8_moe.py`（`AiterTritonMxfp8ExpertsMonolithic`）；oracle 在 gfx942 + MXFP8 时自动选它；文档 `moe_kernel_features.md` 和 `online.md` 更新；`recipes/models/tencent/Hy4-preview.yaml` 更新 MI325X 部署说明。
- **新手概念课堂**：MXFP8 是"每 32 个元素共享一个指数"的 8 位浮点格式，比 FP8 per-tensor 更省显存。之前 gfx942 不支持，只能反量化成 BF16，权重占 196 GiB；现在原生跑，只要 109 GiB。
- **对你有什么影响**：AMD MI325X 上跑 MXFP8 大模型，显存占用大降，吞吐提升 13-23%，GSM8K 精度不变。

---

### 15. [Bugfix][ROCm] benchmark_moe: keep HIP_VISIBLE_DEVICES for Ray (#65935)

- **代码层面**：`benchmarks/kernels/benchmark_moe.py` 里，不再把 `HIP_VISIBLE_DEVICES` 改写成 `ROCR_VISIBLE_DEVICES`，而是删掉 `CUDA_VISIBLE_DEVICES`（因为 vLLM import 时把 HIP 复制过去了），让 Ray 自己按 worker 设置 `HIP_VISIBLE_DEVICES`。
- **新手概念课堂**：Ray 用 `HIP_VISIBLE_DEVICES` 给每个 worker 分配 GPU。如果代码提前把它改名成 `ROCR_VISIBLE_DEVICES`，Ray 就"找不到"该给谁分卡了。就像快递员按门牌号送货，你把门牌号改了，他就迷路。
- **对你有什么影响**：ROCm 上用 Ray 跑 MoE benchmark，多 worker 能正确分到不同 GPU，不再全部挤在 0 号卡。

---

### 16. [Bugfix][Model Loader] Prefetch the whole checkpoint into every node's page cache (#68351)

- **代码层面**：`vllm/model_executor/model_loader/weight_utils.py` 的 `_prefetch_all_checkpoints`，把"按全局 rank 切分文件"改为"按节点内 local rank 切分"，保证每个节点都读全量文件；新增测试 `test_prefetch_checkpoints.py` 验证 1/2/4 节点 × 1-4 rank 的组合。
- **新手概念课堂**：page cache 是"每台机器自己的内存缓存"。两台机器不会共享缓存，所以每台都得把文件读一遍。之前按全局 rank 切，节点 A 可能只读了 1/4 文件，剩下 3/4 没预热，加载就慢。
- **对你有什么影响**：多机部署大模型时，权重加载时间更稳定，不再出现"某些节点特别慢"。

---

### 17. [ROCm][Perf][Muse Spark] Run the pooler and unpooler through the default ROCm GEMM dispatch on gfx942 and gfx950 (#63725)

- **代码层面**：`vllm/models/muse_spark/amd/model.py` 新增 `_pooling_uses_default_gemm()`，在 CDNA3/CDNA4（gfx942/gfx950）上 pooler/unpooler 用 `UnquantizedLinearMethod`（走 AITER 默认 GEMM），其他 CDNA 版本仍用 `MuseSparkTorchLinearMethod`；测试参数化 4 个 CDNA 版本。
- **新手概念课堂**：pooler 是把 hidden state 压成 embedding 的小线性层。在支持 AITER 的 GPU 上，用默认 GEMM 调度比强制 PyTorch linear 更快。就像"本地有高速公路就走高速，没有就走国道"。
- **对你有什么影响**：Muse Spark 模型在 MI300X/MI325X/MI355X 上 embedding 阶段更快。

---

### 18. [ROCm] Warm the AITER FA extend kernels from the attention backend (#68012)

- **代码层面**：`vllm/v1/attention/backends/rocm_aiter_fa.py` 新增 `AiterFlashAttentionExtendKernel`，实现 `get_warmup_keys` / `compile_many`，在 `enable_jit_warmup` 时注册到 `JitWarmupRegistry`；滑动窗口层跳过 warmup；测试验证注册、跳过、调用次数。
- **新手概念课堂**：AITER 的 FA kernel 是 JIT 编译的，第一次调用要编译几百毫秒。Warmup 就是"服务启动时先把各种形状的 kernel 编译好"，避免第一个用户等半天。就像餐厅开门前先把常用菜备好。
- **对你有什么影响**：ROCm 上 AITER FA 的首 token 延迟（TTFT）更稳定，不会因为形状没预热而抖动。

---

### 19. [ROCm][Bugfix] Keep non-float all-reduces off AITER custom all-reduce (#60839)

- **代码层面**：`vllm/distributed/device_communicators/aiter_custom_all_reduce.py` 的 `should_custom_ar` 里，dtype 不是 float32/float16/bfloat16 时直接返回 False；测试用 int8 tensor 验证走 fallback 路径且结果正确。
- **新手概念课堂**：AITER 的自定义 all-reduce kernel 只支持浮点。int8（比如 FP8 权重的原始字节）传进去会崩。加个 dtype 白名单，非浮点就走通用 all-reduce。
- **对你有什么影响**：ROCm 上跑 FP8 量化模型时，权重同步不再因为 dtype 是 int8 而崩溃。

---

### 20. [Doc] Make CUDA 13 the default in the GPU install guide (#67572)

- **代码层面**：`docs/getting_started/installation/gpu.cuda.inc.md`、`docs/contributing/README.md`、`docs/contributing/ci/nightly_builds.md` 里把默认 CUDA 版本从 12.9 改为 13.0（cu130），安装命令改为用 `wheels.vllm.ai/<version>/<variant>` 索引。
- **新手概念课堂**：CUDA 版本决定 PyTorch 和 vLLM 的二进制兼容性。装错版本会报 "undefined symbol" 或 "no kernel image"。文档现在明确"默认 cu130"，并给出查所有变体的 URL。
- **对你有什么影响**：新装 vLLM 默认拉 CUDA 13.0 wheel；如果你的驱动太老（不支持 CUDA 13），需要按文档手动选 cu128 等变体。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：Day-0 模型支持（Day-0 Support）**
  - 一句话解释：模型官方发布当天，推理框架就能跑起来，不用等社区慢慢适配。
  - 为什么社区在关注：Commit 1 里 Inkling-2 系列刚发布，SGLang 就同时覆盖 NVIDIA、AMD、TPU 三种硬件，说明"模型-框架-硬件"三方协同的节奏越来越快，推理框架的竞争已进入"首发即支持"阶段。
  - 对新手意味着什么：如果你在学模型部署，光会写模型不够，能快速把新模型跑通在多硬件上才是稀缺能力；关注各家 cookbook 的写法是很好的入门路径。

- **关键词：采样惩罚（Sampling Penalty）架构重构**
  - 一句话解释：控制模型"别重复说同一个词"的那套机制（frequency/presence/repetition penalty），今天被大改了两轮。
  - 为什么社区在关注：Commit 2 把惩罚历史从"batch 级"搬到"每请求独立池"，Commit 17 把散落的 penaltylib 合并成一个模块。这说明采样层正在为"PD 分离、请求重调度、dLLM"等新场景让路。
  - 对新手意味着什么：采样不只是"softmax 之后取最大值"，它和调度、KV 管理、请求生命周期深度耦合，是理解推理系统的好切入点。

- **关键词：扩散模型（Diffusion）内存自治**
  - 一句话解释：图像/视频生成模型开始自己规划"哪些组件放显存、哪些放内存、哪些流式加载"。
  - 为什么社区在关注：Commit 5 引入基于 warmup 校准的组件驻留规划，Commit 9 给 AMD 写 Triton 卷积核复现 MIOpen 数值顺序，Commit 12 修 fp16 LoRA 溢出。扩散推理正从"能跑"走向"在消费级卡上跑得好"。
  - 对新手意味着什么：多模态生成是当下最热的方向之一，显存规划和数值精度是绕不开的两座山。

- **关键词：异构硬件适配（AMD / TPU / Rust Server）**
  - 一句话解释：同一份代码要跑在 NVIDIA、AMD、Google TPU 上，还要有 Rust 写的高性能服务端。
  - 为什么社区在关注：Commit 1、9、14 都涉及多后端；Commit 15 给 Rust server 加 peer token 鉴权。说明 SGLang 正在从"研究工具"变成"生产级多硬件平台"。
  - 对新手意味着什么：只学 CUDA 已经不够，Triton、HIP、跨平台抽象层是加分项。

- **关键词：缓存一致性（Cache Coherence）**
  - 一句话解释：缓存里"我以为数据在"和"数据真的在"必须一致，否则会读到脏数据或永远不备份。
  - 为什么社区在关注：Commit 19 修 HiCache 换后端后"存在性信念"没清空导致永不备份，Commit 20 把 HiSparse 所有释放路径统一。缓存系统的 bug 往往最隐蔽、最难查。
  - 对新手意味着什么：学缓存系统要先学"状态机思维"——每个缓存条目有哪些状态、状态迁移由谁触发。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：采样惩罚的"每请求独立池"（Commit 2 + 17）**
  - **是什么**：模型生成时，惩罚机制需要记住"这个请求已经输出过哪些 token"。以前这份历史挂在 batch 上，现在每个请求自己持有一个 `penalty_index`，指向调度器里的 `PenaltyStatusPool`。
  - **为什么重要**：在 PD 分离（Prefill/Decode 分开部署）和请求被"撤回重调度"（retract）时，batch 会重组，如果历史挂在 batch 上就会丢。挂在请求上，请求被 retract 再 re-prefill 时历史还在，惩罚才正确。不懂这个，你会在长对话里看到模型突然开始重复。
  - **动手试试**：
    ```bash
    # 起一个带 repetition_penalty 的服务
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000
    ```
    ```python
    import requests
    # 故意让模型重复，观察惩罚是否生效
    payload = {
        "text": "The the the the the the",
        "sampling_params": {"max_new_tokens": 20, "repetition_penalty": 1.5}
    }
    print(requests.post("http://localhost:30000/generate", json=payload).json())
    # 再把 repetition_penalty 改成 1.0 对比
    ```
  - **预期结果**：penalty=1.5 时输出明显更多样，1.0 时容易复读。改完后你能说清"惩罚历史存在哪、什么时候被重置"。

- **知识点 2：JIT 编译的"架构族"目标（Commit 14，sm_100f）**
  - **是什么**：NVIDIA Blackwell 有 sm_100、sm_103 等具体架构，也有"族"目标 sm_100f。编译成族目标，一份二进制能跑整个家族，代价是放弃某些架构专属指令。
  - **为什么重要**：数据中心里 GPU 型号混杂（B200/GB200/B300），如果每个型号单独编译，缓存命中率低、启动慢。用族目标能"一次编译，多处运行"。这是一个典型的"通用性 vs 极致性能"权衡。
  - **动手试试**：
    ```bash
    # 查看当前 GPU 架构
    nvidia-smi --query-gpu=name,compute_cap --format=csv
    # 在 SGLang 里切换 JIT 架构策略（需支持该环境变量）
    export SGLANG_JIT_CUDA_ARCH_POLICY=family   # 或 specific
    python -c "from sglang.kernels.jit.utils.arch import resolve_compile_arch; print(resolve_compile_arch())"
    ```
  - **预期结果**：`specific` 打印类似 `sm_100`，`family` 在 Blackwell + CUDA≥12.9 下打印 `sm_100f`。你就理解了"编译目标"不是唯一值。

- **知识点 3：CUDA 缓存分配器的"保留 vs 已分配"（Commit 6）**
  - **是什么**：`torch.cuda.memory_reserved()` 是 PyTorch 向驱动申请的总量，`memory_allocated()` 是真正给张量用的量，差值就是"空闲但没还"的缓存块。Commit 6 删掉了"每处理一张图就检查并 empty_cache"的逻辑。
  - **为什么重要**：`empty_cache()` 会同步设备、释放所有空闲块，非常慢。每次预处理都调用会让 Qwen-VL 吞吐暴跌。但完全不释放又会让 reserved 只增不减。正确做法是"在正确时机释放"，而不是"每步都检查"。
  - **动手试试**：
    ```python
    import torch
    x = torch.randn(1024, 1024, device="cuda")
    print("allocated:", torch.cuda.memory_allocated() / 2**20, "MB")
    print("reserved :", torch.cuda.memory_reserved() / 2**20, "MB")
    del x
    print("after del allocated:", torch.cuda.memory_allocated() / 2**20, "MB")
    print("after del reserved :", torch.cuda.memory_reserved() / 2**20, "MB")  # 没降！
    torch.cuda.empty_cache()
    print("after empty reserved:", torch.cuda.memory_reserved() / 2**20, "MB")  # 降了
    ```
  - **预期结果**：`del` 后 allocated 降为 0 但 reserved 不变，`empty_cache()` 后 reserved 才降。你就明白为什么"频繁 empty_cache"是性能杀手。

- **知识点 4：LoRA 在 autocast 下的 dtype 陷阱（Commit 12）**
  - **是什么**：fp16 的 LoRA 权重遇到 bf16 激活时，如果直接 `x.to(fp16) @ A.to(fp16)`，DiT 里超过 65504 的激活会溢出成 inf，输出变 NaN（黑图）。修法是"在激活的 dtype 里算"。
  - **为什么重要**：混合精度是标配，但"谁的 dtype 说了算"很容易搞错。fp16 的动态范围只有 ±65504，bf16 有 ±3.4e38，扩散模型的激活值经常超 fp16 范围。
  - **动手试试**：
    ```python
    import torch
    big = torch.tensor([70000.0], dtype=torch.float32)
    print("fp16:", big.to(torch.float16))   # inf
    print("bf16:", big.to(torch.bfloat16))  # 70016 左右，没溢出
    # 模拟 LoRA：x(fp32大值) @ A(fp16)
    x = torch.full((1, 4), 70000.0)
    A = torch.randn(4, 4, dtype=torch.float16)
    print("错法:", (x.to(torch.float16) @ A))  # 可能 nan
    print("对法:", (x.to(torch.bfloat16) @ A.to(torch.bfloat16)))
    ```
  - **预期结果**：看到 fp16 溢出成 inf/nan，bf16 正常。你就理解了 Commit 12 的 `_lora_compute_dtype` 在防什么。

- **知识点 5：缓存"存在性信念"与换后端的一致性（Commit 19）**
  - **是什么**：HiCache 会缓存"某页数据在存储后端存在"的判断（`storage_existence_cache`），避免重复备份。换后端时如果不清空，旧后端的"存在"信念会让新后端永远收不到这页数据。
  - **为什么重要**：这是典型的"缓存失效"bug——状态描述的是旧世界，但代码以为还是新世界。类似问题在 CDN、DNS、数据库连接池里都会出现。
  - **动手试试**：
    ```python
    # 用最小例子模拟"信念缓存"的坑
    existence = {}          # page_id -> bool
    backend = "A"
    def backup(page):
        if existence.get(page): return "skip (以为已存在)"
        existence[page] = True
        return f"backed up to {backend}"
    print(backup(1))        # backed up to A
    backend = "B"           # 换后端，但没清 existence
    print(backup(1))        # skip —— bug!
    existence.clear()       # 修复：换后端时清空
    print(backup(1))        # backed up to B
    ```
  - **预期结果**：第二行输出 "skip"，说明信念缓存导致新后端漏备份；清空后正常。对应 Commit 19 的一行 `cache.storage_existence_cache.clear()`。

---

## 🔥 重要更新

- **Commit 2 + 17（采样惩罚重构）**：惩罚历史从 batch 迁到每请求池，并合并 penaltylib。这是为 PD 分离、请求 retract、dLLM 等新场景打地基，影响所有用 penalty 的线上服务。
- **Commit 5（扩散组件驻留规划）**：基于 warmup 校准自动决定哪些组件常驻显存，直接决定消费级卡能不能跑大扩散模型，是 2/4 系列的核心一步。
- **Commit 14（sm_100f 族编译）**：Blackwell 多型号混部场景下，一份 JIT 产物服务整个架构族，显著提升缓存命中率和启动速度。
- **Commit 15（Rust server peer token 鉴权）**：`/internal/peer` 端点加 token，只有本次启动的 rank 知道，堵住内部端点被外部调用的安全漏洞。

---

## 📋 逐条解读

### 1. [Doc] cookbook: Add the Inkling-2 model introductions and hardware tables (#52042)
- **代码层面**：改的是 `docs/cookbook/.../Inkling-2-Mini.mdx` 和 `Inkling-2-Small.mdx`。加了硬件支持表格（B300/GB300、B200/GB200、H200、RTX 5090、DGX Spark、MI350X、TPU7x）、day-0 支持公告、模型介绍（Mini 26B 总参 5.4B 激活，Small 594B 总参 17B 激活、1M 上下文）。还把 AMD 镜像从 `radixark/...` 改成 `lmsysorg/...`。
- **新手概念课堂**：MoE（混合专家）就像一家大公司有 128 个部门，每个 token 只找最相关的 8 个部门干活，所以"总人数很多但每次只动用一小部分"。这就是"总参 594B、激活 17B"的意思——显存要装下所有人，但算力只花在少数人身上。
- **对你有什么影响**：想部署 Inkling-2 的用户现在有一张"哪块卡用哪个镜像"的对照表，不用再猜。

### 2. [Feat] sampling: Count penalty history right after sampling in a per-request pool (#52035)
- **代码层面**：`schedule_batch.py` 删掉 `Req.penalty_cumulated_len`，新增 `Req.penalty_index`；新增 `PenaltyStatusPool`（在 `penalties.py` 里）；`decode.py` 的 PD 分离路径在构造 fake prefill batch 时调用 `penalty_status_pool.alloc/reset_history`；`dllm/algorithm/base.py` 显式拒绝 dLLM 使用 penalty。
- **新手概念课堂**：把"惩罚历史"想象成每个用户的"已说词汇本"。以前词汇本放在"当前这批人共用的桌子上"，换一批人桌子就清了；现在每人自带一本，被叫去别的房间（retract）再回来，本子还在。
- **对你有什么影响**：长对话、PD 分离部署下，repetition_penalty 的行为更稳定，不会因为调度重组而"失忆"。

### 3. [Test] disaggregation: Run the Kimi-Linear PD+DCP prefill at hybrid checkpoint level 3 (#51979)
- **代码层面**：测试文件 `test_kimi_linear_pd_dcp4.py` 的 PREFILL_ARGS 加了 `--hybrid-cache-checkpoints-level 3`。
- **新手概念课堂**：Mamba 类模型的"状态"像游戏存档。默认只存"关卡结束"的存档（level 2），但测试会重发同一 prompt 期望"从中间继续"，所以需要每章都存（level 3）。
- **对你有什么影响**：纯测试改动，普通用户无感；但说明混合缓存（attention + Mamba）的 checkpoint 粒度是可配置的。

### 4. [Test] layers: Expect Kimi QKV on the KDA sharding group (#51974)
- **代码层面**：`test_repeated_batched_parallel_groups.py` 里简化了 `size` 的计算，去掉 `QKVParallelLinear` 的特殊分支——Kimi 的 KDA 所有投影（含 QKV）都在同一 sharding group 上。
- **新手概念课堂**：张量并行像把一个大矩阵切成几块分给几张卡。以前 QKV 可能切法和别的层不一样，现在统一了，测试期望也随之更新。
- **对你有什么影响**：测试对齐实现，无用户可见变化。

### 5. [Feat] diffusion: Plan component residency from calibrated warmup records (2/4) (#37917)
- **代码层面**：`auto_residency.py` 从空壳变成完整规划器：引入 `optimize_placement`、`PlacementOption`、VRAM 预留比例（measured 5%、extrapolated 10%、最小 4GiB、上限 20%）、pinned H2D 带宽估计（24 GiB/s）等，根据 warmup 校准记录决定每个组件是 RESIDENT / COMPONENT_OFFLOAD / LAYERWISE_OFFLOAD。
- **新手概念课堂**：像给行李箱打包——先量好每件衣服多大（校准），再决定哪些随身带（常驻显存）、哪些托运（offload 到内存）、哪些压缩袋抽真空（逐层流式）。预留空间是"给机场安检留的余量"。
- **对你有什么影响**：扩散模型在显存有限的卡上能自动选最优放置策略，减少 OOM 和手动调参。

### 6. [Perf] qwen-vl: Stop checking the CUDA cache after every image preprocess (#51867)
- **代码层面**：`qwen_vl.py` 删掉 `_release_preprocess_cache()` 及常量 `_PREPROCESS_CACHE_RELEASE_BYTES`，两处调用点移除；对应单测整段删除。
- **新手概念课堂**：`empty_cache()` 像"每次做完一道菜就把整个厨房擦一遍"——干净但极慢。正确做法是"做完一桌菜再擦"。
- **对你有什么影响**：Qwen-VL 多图预处理吞吐提升，延迟降低；显存 reserved 可能比以前高一点，但这是正常缓存行为。

### 7. [Fix] scheduler: Hold a capped chunk's unfinished rest in the prefill budget (#50955)
- **代码层面**：`schedule_policy.py` 的 `_update_prefill_budget` 加 `commit_unfinished` 参数，`_add_chunked_req` 在"chunk 被截断且设了 chunked_req_limit"时传 True；`prefill_budget.py` 的 `reserve` 在 `commit_unfinished` 时把"未完成部分 + 延迟的 max_new_tokens"计入 `total_offset`；`SWAPrefillBudget` 对纯 SWA 池特殊处理。
- **新手概念课堂**：预算是"这轮 prefill 还能塞多少 token"。被截断的请求还剩一大截，如果不提前占预算，后面 decode 时 KV 就不够，会把前面的请求挤掉。这叫"预留"。
- **对你有什么影响**：开启 chunked prefill + chunk limit 时，长请求更不容易被中途打断或 OOM。

### 8. [Fix] diffusion: Fall back from K/V-gather for a cross-attention key mask (#51620)
- **代码层面**：`attention/layer.py` 的 `_kv_gather_unsupported_reason` 新增 `query_len` 参数，当 `attn_mask.shape[1] != query_len`（cross-attention 的 key mask 长度等于 key 数而非 query 数）时返回不支持原因，走 Ulysses 回退路径；测试补了对应用例。
- **新手概念课堂**：K/V-gather 的并行方式假设"mask 每一行对应一个 query"。cross-attention 里 mask 描述的是"哪些 key 可见"，长度对不上，硬用会算错。
- **对你有什么影响**：带 cross-attention mask 的扩散模型在 SP（序列并行）下不再算错，auto 模式自动回退。

### 9. [Perf] [AMD] diffusion: Run BigVGAN vocoder's convolutions in MIOpen's summation orders from Triton (#51934)
- **代码层面**：新增 `kernels/ops/diffusion/conv/` 目录，`conv1d_fp64acc_triton.py` 用 Triton 实现"FP32 输入、FP64 累加、按 MIOpen naive solver 顺序求和"的 conv1d，`depthwise_conv1d_triton.py` 实现 depthwise conv 和转置 depthwise；在 `__init__.py` 注册三个 kernel（仅 HIP）。
- **新手概念课堂**：浮点加法不满足结合律，`(a+b)+c ≠ a+(b+c)`。要"复现"另一个库的结果，必须用同样的累加顺序。MIOpen 那个 kernel 慢（4-5ms/次），Triton 版只要几十微秒，且结果逐位一致。
- **对你有什么影响**：AMD 上跑 BigVGAN 语音合成快很多，且输出与参考实现数值一致。

### 10. [Fix] diffusion/media: Keep request URLs out of download errors (#51962)
- **代码层面**：`vision.py` 新增 `_read_media_source` 包装器，把 `requests.RequestException` 重新分类为 Timeout/TooManyRedirects/HTTPError/ConnectionError，只保留 HTTP 状态码，丢弃原始 URL 和 prepared request；`raise ... from None` 切断异常链；`load_video` 拆出 `_read_video_source`。
- **新手概念课堂**：异常对象会"抱着"原始请求（含 URL、可能含签名 token）。直接抛给用户等于泄露。就像快递单上印了你的完整地址，退货时要把地址撕掉。
- **对你有什么影响**：错误信息更安全，不会把用户请求的 URL 暴露在日志或 API 响应里。

### 11. [Fix] diffusion: Keep request content out of automatic media filenames (#51971)
- **代码层面**：`sampling_params.py` 删掉基于 prompt 哈希/时间戳生成文件名的逻辑，`kandinsky6_sr.py` 删掉基于源视频名的 `_set_output_file_name` 覆盖；文档说明默认文件名与 prompt/输入媒体名无关，需要可预测名请用 `--output-file-name`。
- **新手概念课堂**：文件名里带 prompt 哈希，等于把用户输入"编码"进文件名，可能被下游工具记录、泄露。安全原则：默认不把用户内容写进任何可观测标识。
- **对你有什么影响**：默认输出文件名变成随机/固定格式，不再反映 prompt；需要可预测名请显式指定。

### 12. [Fix] diffusion: Compute fp16 LoRA deltas in the activations' dtype (#51494)
- **代码层面**：`lora/linear.py` 新增 `_lora_compute_dtype`：fp16 adapter + 非 fp16 激活时，用 autocast dtype 或激活 dtype 计算；`_add_lora_delta_in_gemm` 要求 target/rows/b 三者 dtype 一致才 `addmm_`；`fused_lora_delta` 的 zero 张量和 matmul 都按该 dtype。
- **新手概念课堂**：fp16 的"量程"只有 ±65504，bf16 有 ±3.4e38。DiT 激活经常超过 6 万，转 fp16 就变 inf，再算就 NaN，图就黑了。
- **对你有什么影响**：Wan 等扩散模型用 fp16 LoRA 不再出黑图/NaN。

### 13. [Fix] scheduler: Handle repeated and retried memory release and resume requests (#51398)
- **代码层面**：`weight_updater.py` 的 `release_memory_occupation` 改为"只处理尚未释放的 tag，每个 tag 释放成功后才加入 `offload_tags`"；`resume_memory_occupation` 改为"只处理已释放的 tag，每个 tag resume 成功后才 discard"。测试补了重复/重试场景。
- **新手概念课堂**：以前是"先全部标记为已释放，再逐个释放"，中途失败就状态错乱。现在像"每做完一件事才打一个勾"，可重入、可重试。
- **对你有什么影响**：RL 训练等场景反复 release/resume 显存时更稳定，不会因为重复请求报错或状态错乱。

### 14. [Feat] [NVIDIA] jit: Let a module build for its arch family (sm_100f) instead of the arch-specific target (#51929)
- **代码层面**：`jit/utils/arch.py` 新增 `CudaArchPolicy`、`get_family_compile_arch`、`resolve_compile_arch`；`compile/cache.py` 的 hash 纳入 arch 策略；新增环境变量 `SGLANG_JIT_CUDA_ARCH_POLICY`（specific/family）。
- **新手概念课堂**：sm_100f 是"Blackwell 家族通用目标"，像"Windows 通用安装包"vs"某型号专用驱动"。通用包能跑所有 Blackwell，但不能用某些专属指令（比如更大的静态共享内存）。
- **对你有什么影响**：多型号 Blackwell 混部时 JIT 缓存命中率提升、启动更快；需要极致性能可切回 `specific`。

### 15. [Fix] rust-server: Serve /internal/peer only to the launch's own listeners (#51946)
- **代码层面**：`rust_server/server.py` 的 `gather_server_endpoints` 给每个 listener 生成 `secrets.token_urlsafe(32)` 的 `peer_token`；Rust 侧 `pyerr_to_status` 新增 `PyPermissionError → PERMISSION_DENIED` 映射；`/internal/peer` 校验 token。
- **新手概念课堂**：内部端点像"员工通道"，以前谁找到门都能进，现在每个 rank 发一张一次性工牌，只有本次启动的兄弟 rank 知道工牌号。
- **对你有什么影响**：内部端点不再被外部随意调用，安全性提升；正常多 rank 通信无感。

### 16. [Perf] kimi-k3: Order shared-expert side-stream frees on the current stream (#51900)
- **代码层面**：`kimi_k3.py` 的 `_forward_unfused` 移除 `shared_input.record_stream(self.alt_stream)`，新增 `shared_alt_output` 和 `release_shared_output()`，在非 fine-grained overlap 路径下把释放排到 current stream 上，避免 side stream 上的释放与主 stream 使用竞争。
- **新手概念课堂**：多 CUDA stream 像多条流水线，A 线用完的零件不能马上回收，因为 B 线可能还在用。`record_stream` 是"告诉分配器这条线也在用"，但顺序错了会提前释放。
- **对你有什么影响**：Kimi-K3 共享专家路径减少不必要的同步，性能提升且避免潜在 use-after-free。

### 17. [Refactor] sampling: Merge penaltylib into one penalties module (#52002)
- **代码层面**：新建 `sampling/penalties.py`（603 行），把原来分散在 penaltylib 里的 `BatchedPenalizerOrchestrator`、各类 `_BatchedPenalizer` 合并；`apply_scaling_penalties` 用 `torch.compile` 加速。
- **新手概念课堂**：把散落在多个抽屉里的工具收进一个工具箱，找起来快、改起来不容易漏。`torch.compile` 是把 Python 循环编译成融合 kernel，减少 kernel launch 开销。
- **对你有什么影响**：代码更易维护；采样惩罚路径可能有小幅性能提升。

### 18. [Doc] diffusion/storage: Prefer workload credentials for S3 uploads (#51987)
- **代码层面**：文档改为推荐"给 serving workload 分配 IAM role，不设 `SGLANG_S3_ACCESS_KEY_ID/SECRET`，让 Boto3 自动获取临时凭证"；MinIO 等才需显式 endpoint 和 key；说明返回的 URL 不是 presigned。
- **新手概念课堂**：IAM role 像"员工刷工牌进仓库"，凭证自动轮换；硬编码 access key 像"把钥匙焊在门上"，泄露就完蛋。
- **对你有什么影响**：按文档部署更安全；私有 bucket 需要自己处理下载鉴权。

### 19. [Fix] hicache: Clear storage existence beliefs when a new backend is attached (#51254)
- **代码层面**：`storage_attachment.py` 的 `attach` 在成功挂载新后端后调用 `cache.storage_existence_cache.clear()`；测试新增"detach 后 attach 新后端，确认会重新备份"的用例。
- **新手概念课堂**：见知识点 5——"我以为数据在旧仓库"的信念不清空，新仓库永远收不到货。
- **对你有什么影响**：切换 HiCache 存储后端后，数据会正确备份到新后端，不会出现"缓存命中但存储里没有"的诡异现象。

### 20. [Refactor] hisparse: Route every allocator free through one path that releases the HiSparse mapping (#51252)
- **代码层面**：`allocator/hisparse.py` 抽出 `HiSparseAllocatorBase` 基类，统一 `available_size`、`alloc_logical_only`、`alloc_decode`、`_extend_device_buffer` 等；所有 free 走同一路径，确保释放时同时清理 `full_to_hisparse_device_index_mapping`。
- **新手概念课堂**：HiSparse 有两套索引——逻辑索引和物理设备槽位，靠一张映射表对应。如果 free 时只还了逻辑索引没清映射，下次分配会拿到"看起来空闲但映射还指着旧数据"的槽，读到脏数据。
- **对你有什么影响**：HiSparse 长跑更稳定，减少内存泄漏和脏读；代码结构更清晰，便于后续扩展。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)