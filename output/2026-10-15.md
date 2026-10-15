# 🚀 vLLM & SGLang 每日更新 — 2026-10-15

> 自动生成于 2026-10-15 06:23 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：KV Offload（KV 缓存卸载）**
  - **一句话解释**：把 GPU 显存里放不下的 KV 缓存搬到 CPU 内存或远端节点，需要时再拉回来，相当于给显存"外挂硬盘"。
  - **为什么社区在关注**：今天有 3 个 commit（#58168、#59544、#56836）都在改 KV Offload，包括线程模型、内存 pin、P2P 超时。说明长上下文推理的显存瓶颈已经成为核心痛点，社区在把它从"能跑"打磨到"生产可用"。
  - **对新手意味着什么**：如果你想学推理系统优化，KV 缓存管理是当下最热的方向之一，涉及内存管理、并发、网络传输，是绝佳的练手项目。

- **关键词：ROCm / AMD GPU 适配**
  - **一句话解释**：让 vLLM 在 AMD 的 GPU（MI300/MI355 等）上跑得又快又稳。
  - **为什么社区在关注**：今天有 5 个 commit 涉及 ROCm（#58297、#59596、#58151、#57773、#61697、#61720），从 MLA 注意力、MoE 量化到 CI 流水线全覆盖。这说明 AMD 正在成为 NVIDIA 之外的真实选择，异构计算不是口号。
  - **对新手意味着什么**：如果你只会 CUDA，现在补 ROCm/HIP 知识是差异化竞争力；AMD 生态人才缺口大，机会多。

- **关键词：推测解码（Speculative Decoding）**
  - **一句话解释**：用小模型"猜"接下来几个 token，再用大模型一次性验证，猜对了就白赚速度。
  - **为什么社区在关注**：今天 #61701 直接删掉了 Medusa 和 MLP speculator 两种旧方案，说明社区在"做减法"，把资源集中到 EAGLE 等更有效的方案上。
  - **对新手意味着什么**：推测解码是推理加速的"必修课"，但要注意——老教程里的 Medusa 已经过时了，学之前先看仓库当前支持什么。

- **关键词：torch.compile / CUDA Graph**
  - **一句话解释**：把 PyTorch 模型编译成高效代码，并把固定形状的计算图"录制"下来重复播放，省去每次的调度开销。
  - **为什么社区在关注**：今天 #61572 给 EmbeddingGemma2 开启了这两个功能。这说明"新模型接入 vLLM 后默认没开编译"是常见遗漏，性能提升空间很大。
  - **对新手意味着什么**：学会给模型加 `@support_torch_compile` 装饰器，是参与 vLLM 模型适配的最低门槛任务。

- **关键词：自定义算子注册（Custom Op）**
  - **一句话解释**：把一段计算封装成 PyTorch 认识的"官方算子"，这样编译器和 CUDA Graph 才能正确追踪它。
  - **为什么社区在关注**：今天 #57331 把 int8 scaled_mm 注册成自定义算子，修复了编译时的问题。这是 torch.compile 兼容性的常见坑。
  - **对新手意味着什么**：如果你写的 kernel 在 `torch.compile` 下报错或结果不对，八成是没注册成 custom op。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：KV 缓存卸载与线程模型

- **是什么**：GPU 显存有限，长对话的 KV 缓存放不下。KV Offload 把不常用的块搬到 CPU 内存或别的机器，需要时再取回。今天 #58168 引入了一个"轮询线程"，专门服务 P2P（点对点）传输，不用等引擎每一步结束。

- **为什么重要**：如果 P2P 传输只能等引擎 step 边界，一个慢节点可能让整个集群卡几秒。不懂这个，你调优时会发现"GPU 利用率明明不高，为什么吞吐上不去"——瓶颈在网络等待上。

- **动手试试**：
```bash
# 查看 KV offload 相关配置文档
python -c "
import vllm
print([x for x in dir(vllm) if 'offload' in x.lower()])
"
# 阅读源码中新增的线程模型注释
grep -n "tier_poll_interval_s" docs/features/kv_offloading_usage.md
```

- **预期结果**：你会看到文档里解释了"哪些 tier 需要 off-thread 轮询"以及 `tier_poll_interval_s=0` 会禁用线程。理解"锁保护 + 多线程访问"的设计，你就懂了分布式缓存的核心难点。

---

### 知识点 2：自定义算子注册（torch.compile 兼容性）

- **是什么**：`torch.compile` 需要知道每个操作的"形状规则"才能优化。自定义 kernel 如果不注册，编译器会报错或走慢路径。今天 #57331 用 `direct_register_custom_op` 把 int8 矩阵乘法注册成了 `torch.ops.vllm.int8_scaled_mm_func`。

- **为什么重要**：这是量化推理的常见坑。你写了个超快的 int8 kernel，结果一开 `torch.compile` 就崩，或者性能反而下降——就是因为没注册。

- **动手试试**：
```python
import torch
from vllm.utils.torch_utils import direct_register_custom_op

def my_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

def my_add_fake(x, y):
    return torch.empty_like(x)

direct_register_custom_op("my_add", my_add, fake_impl=my_add_fake)

x = torch.randn(4, device="cuda")
y = torch.randn(4, device="cuda")
out = torch.ops.vllm.my_add(x, y)
print(out)
```

- **预期结果**：成功输出 `x+y` 的结果。关键点是 `fake_impl`——它告诉编译器"输出形状是什么"，不需要真的算。掌握了这个，你就能让自己的 kernel 在 `torch.compile` 下正常工作。

---

### 知识点 3：Cross-Encoder 的 token_type_ids 对齐

- **是什么**：Cross-Encoder（重排序模型）需要区分"查询"和"文档"两段文本。`token_type_ids` 就是标记每段归属的数组。今天 #58143 修复了截断+填充时这个数组错位的问题。

- **为什么重要**：如果 `token_type_ids` 和 `input_ids` 错位，模型会以为"查询段是空的"，重排序结果完全错误——而且不会报错，只是效果变差，极难排查。

- **动手试试**：
```python
# 模拟一个 cross-encoder 的 token 序列
# 查询 3 个 token（type 0），文档 2 个 token（type 1），填充到 8
token_type_ids = [0, 0, 0, 1, 1]
pad_token_type_id = 0  # 来自 tokenizer.pad_token_type_id
padded = token_type_ids + [pad_token_type_id] * (8 - len(token_type_ids))
print("Padded:", padded)  # [0,0,0,1,1,0,0,0]

# 压缩回边界
def compress(ids):
    doc_start = ids.index(1) if 1 in ids else len(ids)
    doc_end = doc_start + sum(1 for x in ids[doc_start:] if x == 1)
    return doc_start, doc_end

print("Boundary:", compress(padded))  # (3, 5)
```

- **预期结果**：输出 `(3, 5)`，表示文档从索引 3 到 5。如果填充时用了 `token_type_ids[-1]`（旧代码），会得到 `(3, 8)`，把填充也算进文档，模型就错了。

---

### 知识点 4：MXFP4 量化与 MoE 权重布局

- **是什么**：MXFP4 是一种 4-bit 浮点量化格式，能把模型压到 1/4 大小。今天 #59596 让 AMD 的 MoRI 框架支持 MXFP4 的 MoE 分发。

- **为什么重要**：MoE 模型（如 DeepSeek、GPT-OSS）专家数量多，权重巨大。不量化根本放不下；量化后如果权重布局不对，kernel 会算错或崩溃。

- **动手试试**：
```python
import torch
# 模拟 MXFP4 量化的核心思想：把 float32 压到 4-bit + 共享指数
def fake_mxfp4_quant(w, block_size=32):
    # 每 block_size 个元素共享一个 scale
    flat = w.flatten()
    n_blocks = flat.numel() // block_size
    reshaped = flat[:n_blocks * block_size].view(n_blocks, block_size)
    scale = reshaped.abs().max(dim=1, keepdim=True).values / 7.0
    quantized = (reshaped / scale).round().clamp(-8, 7).to(torch.int8)
    return quantized, scale

w = torch.randn(64, 128)
q, s = fake_mxfp4_quant(w)
print(f"原始: {w.numel() * 4} bytes")
print(f"量化后: {q.numel()} bytes + {s.numel() * 4} bytes scale")
print(f"压缩比: {(w.numel() * 4) / (q.numel() + s.numel() * 4):.1f}x")
```

- **预期结果**：看到约 7-8 倍的压缩比。理解"block 共享 scale"是 MXFP4 的核心，也是为什么需要专门的 kernel 来还原。

---

### 知识点 5：P2P 查找超时与降级

- **是什么**：分布式 KV 缓存里，节点 A 向节点 B 发"你有没有这个 key"的请求。如果 B 挂了或网络卡了，A 不能永远等。今天 #56836 给这个等待加了超时，超时后降级为"本地重新计算"。

- **为什么重要**：没有超时，一个故障节点会让整个集群的请求全部挂起。这是分布式系统的"熔断"思想。

- **动手试试**：
```python
import time

class FakeP2PLookup:
    def __init__(self, timeout_s=2.0):
        self.timeout_s = timeout_s
        self.start_time = None
    
    def lookup(self, peer_alive):
        if self.start_time is None:
            self.start_time = time.monotonic()
        elapsed = time.monotonic() - self.start_time
        if not peer_alive and elapsed >= self.timeout_s:
            return "MISS"  # 降级为本地计算
        if peer_alive:
            return "HIT"
        return "RETRY"

lookup = FakeP2PLookup(timeout_s=0.5)
# 模拟对端不响应
for i in range(5):
    result = lookup.lookup(peer_alive=False)
    print(f"第 {i+1} 次: {result}")
    time.sleep(0.2)
```

- **预期结果**：前几次返回 `RETRY`，超过 0.5 秒后返回 `MISS`。这就是"有界等待"——超过阈值就放弃，避免无限阻塞。

---

## 🔥 重要更新

1. **#61701 删除 Medusa 和 MLP speculator 支持**：这是破坏性变更。如果你还在用 Medusa 做推测解码，升级后会直接报错。社区在集中资源到 EAGLE 方案，老方案正式退役。

2. **#61712 删除旧版 Ray compiled-graph executor**：另一个破坏性清理。`VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 环境变量不再需要，Ray 依赖也从 `ray[cgraph]` 简化为 `ray`。使用 Ray 多机部署的用户需要检查配置。

3. **#58168 KV Offload 线程模型重构**：引入了独立的 tier 轮询线程，P2P 传输不再阻塞引擎 step。对长上下文、多机部署场景是显著的延迟改善。

4. **#57331 int8 scaled_mm 注册为自定义算子**：修复了量化模型在 `torch.compile` 下的兼容性问题，对使用 int8 量化的用户是重要 bugfix。

---

## 📋 逐条解读

### 1. [Bugfix][Multimodal] Recover PyNvVideoCodec decoder slots after failures

- **代码层面**：修改了 `tests/multimodal/media/test_video.py` 和 `test_gpu_ipc_memory.py`，新增了 `get_pynvvideocodec_decoder_gpu_memory_bytes()` 函数，并让 `merge_kwargs` 不再允许请求级参数覆盖 `max_width`/`max_height`。核心是让解码器的显存预算根据分辨率动态计算。

- **新手概念课堂**：视频解码器就像"停车位"——每个解码器占一块 GPU 显存。如果按固定大小分配，高分辨率视频会爆显存，低分辨率又浪费。现在改成"按车的大小分配车位"。

- **对你有什么影响**：处理视频的多模态推理更稳定，高分辨率视频不会因为显存估算错误而崩溃。普通用户感知不到，但视频推理的可靠性提升了。

---

### 2. [KV Offload] Expose OffloadingManager.lock()/unlock() and service the tiering control plane off the scheduler thread

- **代码层面**：在 `docs/features/kv_offloading_usage.md` 新增"Threading model"章节，引入 `tier_poll_interval_s` 配置项（默认 0.001 秒）。新增测试验证 `on_schedule_end` 是每步最后一个 manager 调用。

- **新手概念课堂**：想象一个餐厅，厨师（调度器）每做完一道菜（一个 step）才能去前台（P2P tier）处理点单。现在雇了个服务员（轮询线程），厨师做菜时服务员也能接单，不用等。

- **对你有什么影响**：多机 KV 缓存场景下，P2P 传输延迟大幅降低。如果你的集群有节点间 KV 共享，升级后吞吐会提升。

---

### 3. [Bugfix][Benchmark] Fix probe latency and duration in bench serve

- **代码层面**：`vllm/benchmarks/serve.py` 中把 session 创建抽成 `make_session(limit)` 函数，探针请求用独立的 `make_session(1)`，不再和主请求抢连接池。

- **新手概念课堂**：主请求占满了连接池（比如 100 个并发），探针请求挤不进去，测出来的延迟是"排队时间"而不是真实延迟。给探针单独开一条通道，才能测准。

- **对你有什么影响**：跑 benchmark 时，probe latency 和 duration 数据更准确。如果你用 benchmark 做容量规划，之前的数字可能偏悲观。

---

### 4. [KV Offload] Pin only each worker's own offload slots

- **代码层面**：`vllm/v1/kv_offload/cpu/gpu_worker.py` 的 `pin_mmap_region` 新增 `worker_slot_only` 参数。每个 worker 只 pin（锁定）自己负责的内存槽位，不再 pin 整个共享区域。

- **新手概念课堂**：pin 内存就像"把书钉在桌上不让它被搬走"。之前每个 worker 都把整张桌子钉死，现在只钉自己那本书。多 worker 场景下内存浪费大幅减少。

- **对你有什么影响**：多 worker 部署时 CPU 内存占用降低，避免因为重复 pin 导致 OOM。

---

### 5. [Perf][KV Connector][Mooncake] Send the KV layout once per peer

- **代码层面**：`vllm/v1/kv_connector/mooncake/` 中引入 `layout_id` 机制。首次传输发送完整 layout（146 层的信息），后续只发 `layout_id`，对端缓存后复用。测试显示后续消息体积不到首次的 20%。

- **新手概念课堂**：就像寄快递，第一次要填完整地址（省市区街道门牌），之后寄同一个地址只需写"同上"加个编号。

- **对你有什么影响**：Mooncake KV 连接器的网络开销大幅降低，多轮对话场景下延迟改善明显。

---

### 6. [Bugfix][KV Offload] Bound unanswered P2P lookup waits

- **代码层面**：`vllm/v1/kv_offload/tiering/p2p/session/client.py` 新增 `_LOOKUP_TIMEOUT_S` 超时机制。超时后 session 被清理，查找返回 `MISS`，降级为本地重新计算。

- **新手概念课堂**：打电话找人，响了 10 秒没人接就挂掉自己想办法，而不是一直等到对方接。这就是"有界等待"。

- **对你有什么影响**：一个节点故障不会拖垮整个集群。P2P KV 共享的鲁棒性提升。

---

### 7. [XPU][MLA] Integrate XPU MLA support into the existing FLASH_ATTN_MLA backend

- **代码层面**：`vllm/_xpu_ops.py` 中修正了 `flash_attn_varlen_func` 的输出形状（用 `v.shape[-1]` 而非 `q.shape[-1]`），测试中为 XPU 添加了 FLASH_ATTN prefill backend。

- **新手概念课堂**：MLA（Multi-head Latent Attention）是 DeepSeek 的注意力变体，Q 和 V 的维度不一样。之前代码假设它们一样，在 XPU 上就出错了。修正后 Intel GPU 也能跑 MLA。

- **对你有什么影响**：使用 Intel GPU 跑 DeepSeek 类模型时，MLA 注意力可以正常工作。

---

### 8. [Model] Enable torch.compile and CUDA graphs for EmbeddingGemma2

- **代码层面**：`vllm/model_executor/models/embedding_gemma2.py` 给 `EmbeddingGemma2TextModel` 加了 `@support_torch_compile` 装饰器。

- **新手概念课堂**：这个装饰器就像给模型贴了个"可编译"标签，告诉 vLLM"这个模型可以用 torch.compile 加速"。不加的话，模型只能走慢速的 eager 模式。

- **对你有什么影响**：用 EmbeddingGemma2 做 embedding 推理时速度提升，延迟降低。

---

### 9. [Feature][Model Loader] Apply model-owned MTP shard selection to RunAI ModelStreamer

- **代码层面**：`vllm/model_executor/model_loader/runai_streamer_loader.py` 现在会调用模型的 `is_unused_checkpoint_weight` 钩子，跳过只包含无用权重的 safetensors 文件。文档新增说明章节。

- **新手概念课堂**：MTP（Multi-Token Prediction）模型有些权重是训练用的，推理时不需要。之前 RunAI 加载器会傻乎乎地把它们也下载下来，浪费带宽和时间。

- **对你有什么影响**：从对象存储加载 MTP 模型时更快，省带宽。前提是模型实现了 `is_unused_checkpoint_weight`。

---

### 10. [ROCm][DI] Enable MoRI FP4 dispatch for mxfp4 w4a4 moe

- **代码层面**：`tests/kernels/moe/test_mori_moe.py` 新增 `_mxfp4_aiter_weights` 辅助函数，用 `dynamic_mxfp4_quant` 和 `convert_gpt_oss_weight_to_mxfp4_moe_kernel_format` 把权重转成 AITER 的 W4A4 布局。

- **新手概念课堂**：W4A4 指权重和激活都是 4-bit。MoE 模型有几十个专家，每个专家的权重布局都要符合 kernel 的期望，否则算出来的结果是错的。

- **对你有什么影响**：AMD GPU 上跑 MXFP4 量化的 MoE 模型（如 GPT-OSS）时，性能和正确性都有保障。

---

### 11. [ROCm][DSv4.1] Enable DSpark adaptive verification

- **代码层面**：`.buildkite/test_areas/kernels.yaml` 新增测试依赖，`tests/kernels/attention/test_rocm_triton_attn_dsv4.py` 新增 `test_dsv41_adaptive_mla_swa_metadata_graph_replay` 测试。

- **新手概念课堂**：DSpark 是 DeepSeek V4.1 的推测解码方案，"adaptive verification"指根据情况动态决定验证多少个草稿 token。这个 commit 让它在 ROCm 上也能用。

- **对你有什么影响**：AMD GPU 上跑 DeepSeek V4.1 时，推测解码可以开启，吞吐提升。

---

### 12. [ROCm][Perf] Enable DPA+ETP prefill compute/comm dual-batch-overlap for DeepSeek-V4

- **代码层面**：新增 `tests/kernels/moe/test_naive_dp_ep_dbo.py`，`vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` 中引入 `_comm_region()` 上下文管理器，实现计算和通信的双批次重叠。

- **新手概念课堂**：DPA（数据并行注意力）+ ETP（专家张量并行）时，GPU 一边算一边等通信。双批次重叠就是"算 A 批次时通信 B 批次的数据"，把等待时间填满。

- **对你有什么影响**：DeepSeek-V4 在 AMD 多卡上 prefill 阶段吞吐提升，尤其是专家并行场景。

---

### 13. [Bugfix] Preserve padding token types for cross encoders

- **代码层面**：`vllm/entrypoints/pooling/scoring/io_processor.py` 中 `_apply_post_tokenization_to_token_type_ids` 改用 `tokenizer.pad_token_type_id` 而非 `token_type_ids[-1]`。`compress_token_type_ids` 现在返回 `(start, end)` 元组并校验合法性。

- **新手概念课堂**：填充 token 的 type 应该是 tokenizer 定义的"填充类型"（通常是 0），而不是"最后一个真实 token 的类型"。用错了会让模型把填充部分当成文档内容。

- **对你有什么影响**：使用 cross-encoder 重排序模型（如 BGE-reranker）时，截断+填充的长文本评分更准确。

---

### 14. [Bugfix][ROCm][Disagg] Allow MoRIIO HMA per-layer block_len

- **代码层面**：`vllm/v1/kv_connector/moriio/` 中允许 HMA（混合内存分配）模式下每层有不同的 `block_len`（字节数），但 `block_size`（token 数）必须一致。新增测试验证。

- **新手概念课堂**：HMA 模式下，不同层的 KV cache 大小可能不同（比如 MLA 层和普通注意力层）。之前代码要求所有层字节数一样，现在放宽了，但 token 数仍要一致。

- **对你有什么影响**：GLM-5.2、MiniMax 等混合架构模型在 AMD 分离式部署时不再报错。

---

### 15. [Bugfix][Kernel] Register int8 scaled_mm as a custom op

- **代码层面**：`vllm/model_executor/kernels/linear/scaled_mm/triton.py` 中把 `triton_scaled_mm` 调用替换为 `torch.ops.vllm.int8_scaled_mm_func`，并用 `direct_register_custom_op` 注册。

- **新手概念课堂**：`torch.compile` 需要知道算子的"假实现"（fake impl）来推断输出形状。没注册的算子会导致编译失败或走 fallback 慢路径。

- **对你有什么影响**：int8 量化模型开启 `torch.compile` 后不再报错，性能提升。

---

### 16. [ROCm][CI] Add AMD job for DeepSeek-V4.1 Mono Decode Numerics

- **代码层面**：`.buildkite/test-amd.yaml` 新增一个 MI355 上的 CI job，跑 `test_dsv41_mono_numerics.py` 和 `test_dsv41_mono_layers.py`。

- **新手概念课堂**：CI（持续集成）就是"每次改代码自动跑测试"。新增这个 job 意味着 DeepSeek-V4.1 在 AMD 上的数值正确性有了持续保障。

- **对你有什么影响**：普通用户无感知，但 AMD 用户遇到数值问题的概率降低。

---

### 17. [Core] Remove the legacy Ray compiled-graph executor

- **代码层面**：删除 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` 环境变量相关代码，清理 `ray_communicator.py`，文档中 `ray[cgraph]` 改为 `ray`。大量测试文件更新。

- **新手概念课堂**：这是"技术债清理"。旧版 Ray 执行器用编译图（compiled graph）方式跑，新版用更简单的方式。旧代码维护成本高，直接删掉。

- **对你有什么影响**：**破坏性变更**。如果你设置了 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`，升级后这个变量会被忽略。Ray 依赖安装更简单。

---

### 18. [Bugfix][PCP] Align MoE routing padding with gathered tokens

- **代码层面**：`vllm/v1/worker/gpu/pcp_manager.py` 中新增 `cp_interleave` 属性，`get_draft_input_buffers` 返回元组 `(inputs, num_tokens)`，`prepare_inputs_to_capture` 改名为 `prepare_inputs_for_capture_or_profile`。

- **新手概念课堂**：PCP（Prefill Context Parallelism）把长序列切成多段并行处理。MoE 路由时，padding 的 token 数必须和实际 gather 的 token 数对齐，否则路由到错误的专家。

- **对你有什么影响**：使用 PCP + MoE 模型时，推测解码的草稿 token 路由正确，输出质量提升。

---

### 19. [Spec Decode] Remove Medusa and MLP speculator support

- **代码层面**：删除 `vllm/model_executor/models/medusa.py` 和 `mlp_speculator.py`，清理相关配置、测试、CI 依赖。`mergify.yml` 和 `issue_autolabel.yml` 中的 Medusa 关键词也移除。

- **新手概念课堂**：Medusa 是最早的推测解码方案之一（在模型头上加多个"预测头"），但效果不如后来的 EAGLE。社区决定集中维护 EAGLE，删掉老方案减少维护负担。

- **对你有什么影响**：**破坏性变更**。如果你用 Medusa 或 MLP speculator，升级后无法使用。需要迁移到 EAGLE 方案。

---

### 20. [ROCm] Bump AITER to v0.1.25 and relax the DSv4.1 mono MoE tolerance

- **代码层面**：`docker/Dockerfile.rocm_base` 和 `Dockerfile.rock_base` 中 AITER 版本从 `v0.1.24.post1` 升到 `v0.1.25`。`test_dsv41_mono_numerics.py` 中数值容差从 `cos>0.9999, rel<0.015` 放宽到 `cos>0.9998, rel<0.02`，并加了 TODO 注释说明原因。

- **新手概念课堂**：AITER 是 AMD 的 AI 算子库，类似 NVIDIA 的 cuDNN。升级版本会带来新 kernel 和性能改进，但有时数值精度会有微小变化，所以测试容差要相应调整。

- **对你有什么影响**：AMD 用户升级后获得 AITER 新功能。DeepSeek-V4.1 的 MoE 数值测试容差放宽，是因为 AITER 的 FP8 kernel 用了不同的 scale 规则（已在注释中说明，等上游修复后会收紧）。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：Scheduler 请求生命周期管理**
  - **一句话通俗解释**：调度器要精准地跟踪"每个请求占了多少内存资源、什么时候释放"，就像餐厅领班要清楚每桌客人占了几号桌、什么时候走人。
  - **为什么社区在关注**：今天有 3 个 commit（#44401、#44351、#43502）都在修调度器的请求计数/释放逻辑，说明大规模推理服务的资源泄漏和 OOM 是真实痛点。
  - **对新手意味着什么**：调度器是推理引擎的心脏，理解"请求-资源"的映射关系，比学任何花哨的 kernel 都更基础、更值钱。

- **关键词：AMD/ROCm 全面适配**
  - **一句话通俗解释**：SGLang 正在把原本只跑在 NVIDIA CUDA 上的功能，一个个搬到 AMD GPU 上。
  - **为什么社区在关注**：今天 6 个 commit（#44311、#43643、#43642、#43770、#44368、#35435）与 AMD 相关，从 kernel 到文档到 CPU 集合通信，说明异构计算已是硬需求。
  - **对新手意味着什么**：只懂 CUDA 已经不够了，ROCm/HIP 是新的加分项，而且 AMD 生态文档少、坑多，正是新手可以贡献的地方。

- **关键词：Bit-exact 融合 kernel**
  - **一句话通俗解释**：把多个小算子合并成一个大 kernel，但结果必须和原来"一个比特都不差"。
  - **为什么社区在关注**：commit #38650 花大量篇幅论证 conv bias 融合的数值精度，说明性能优化不能牺牲可复现性。
  - **对新手意味着什么**：学会用 `torch.equal` 验证融合前后结果，是进入 kernel 优化领域的敲门砖。

- **关键词：CUDA Graph / Breakable Graph**
  - **一句话通俗解释**：把一串 GPU 操作"录下来"重放，省掉每次的启动开销；"breakable"是允许中途打断去跑动态逻辑。
  - **为什么社区在关注**：commit #44326、#43642 都在让更多路径支持 graph，这是低延迟推理的关键技术。
  - **对新手意味着什么**：CUDA Graph 是推理性能的"必修课"，理解它的限制（静态形状、不能动态分配）能帮你避开大量坑。

- **关键词：ComfyUI 集成 / 扩散模型**
  - **一句话通俗解释**：让 SGLang 的扩散模型能力接入 ComfyUI 这个流行的可视化工作流工具。
  - **为什么社区在关注**：commit #44058、#44088 显示 SGLang 在往"易用性"方向走，不只是给工程师用。
  - **对新手意味着什么**：AI 应用层的集成能力（API、插件、UI）是快速做出产品的捷径。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：调度器的 req_to_token 池与"持行"（held row）概念

- **是什么**：每个请求在 `req_to_token_pool` 里占一行（row），用来记录 token 到 KV cache 的映射。一个正在被 chunked prefill 处理的请求，它的行会跨多个调度步骤保留。
- **为什么重要**：如果调度器重复计算"这个请求已占的行"，就会误判可用空间，导致 OOM 或者拒绝本可以接纳的请求。commit #44401 修的就是这个。
- **动手试试**：
```python
# 用 Python 模拟一个简化版的调度器计数逻辑
class Pool:
    def __init__(self, total):
        self.total = total
        self.used = 0
    def available_size(self):
        return self.total - self.used

class Req:
    def __init__(self, holds_kv):
        self.holds_kv = holds_kv  # 是否已经持有行

pool = Pool(total=10)
can_run = [Req(holds_kv=True), Req(holds_kv=False)]

# 错误做法：只数 can_run 长度
wrong = len(can_run) >= pool.available_size()
# 正确做法：已持行的请求不需要新行，要把它们"还回去"
num_held = sum(r.holds_kv for r in can_run)
correct = len(can_run) >= pool.available_size() + num_held
print(f"错误判断: {wrong}, 正确判断: {correct}")
```
- **预期结果**：你会看到错误判断可能拒绝一个实际能跑进来的请求，而正确判断会接纳它。这就理解了"held row"为什么要从可用空间里加回来。

### 知识点 2：Bit-exact 融合 kernel 与数值精度

- **是什么**：把 `conv2d` 后的 bias 加法融合进卷积 kernel，但结果必须和 PyTorch 原生"卷积+加法"一模一样。
- **为什么重要**：融合能省一次内存读写（H200 上 1.5 TB/s vs 4.1 TB/s），但如果改变了数值，模型输出就会漂移，测试会挂。
- **动手试试**：
```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(1, 8, 32, 32, device="cuda", dtype=torch.bfloat16)
w = torch.randn(16, 8, 3, 3, device="cuda", dtype=torch.bfloat16)
b = torch.randn(16, device="cuda", dtype=torch.bfloat16)

# 原生路径：conv 不带 bias，再手动加
conv = F.conv2d(x, w, bias=None, padding=1)
native = conv + b.view(1, -1, 1, 1)

# 融合路径（模拟）：理论上应该 bit-exact
fused = F.conv2d(x, w, bias=b, padding=1)

print("bit-exact?", torch.equal(native, fused))
print("max diff:", (native.float() - fused.float()).abs().max().item())
```
- **预期结果**：在 CUDA 上，`torch.equal` 可能是 `False`（因为 cuDNN 的 bias 加法顺序不同），这正是 commit #38650 要解决的问题——它自己写 kernel 保证和 aten 完全一致。

### 知识点 3：CUDA Graph 的"静态形状"约束

- **是什么**：CUDA Graph 把 GPU 操作录制成一张图，重放时不能改变 tensor 形状或地址。
- **为什么重要**：commit #44326 里，TRT-LLM 的 prefill 在 graph 里跑时，需要把 `seq_lens` 从"全上下文"改成"tail-relative"，因为 graph 里形状是固定的。
- **动手试试**：
```python
import torch

# 模拟：动态形状 vs 静态形状
def dynamic_op(x):
    # 每次调用形状可能不同
    return x * 2 + x.sum()

# CUDA Graph 要求输入地址固定
static_input = torch.randn(4, 8, device="cuda")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = static_input * 2 + static_input.sum()

# 重放时只能改 static_input 的值，不能改形状
static_input.copy_(torch.randn(4, 8, device="cuda"))
g.replay()
print("graph output shape:", static_output.shape)

# 如果尝试用不同形状，会失败
try:
    other = torch.randn(8, 8, device="cuda")
    # 这行不会报错，但 graph 不会用到 other
    print("注意：graph 只认 static_input 的地址")
except Exception as e:
    print("error:", e)
```
- **预期结果**：你会理解为什么 graph 里的 tensor 必须预分配、形状固定，这就是为什么 commit #44326 要把 `topk_result` 按 `x_meta.shape[0]` 预分配。

### 知识点 4：ReduceOp 的身份比较（`is` vs `==`）

- **是什么**：PyTorch 的 `ReduceOp.SUM` 是一个枚举值，`==` 会调用 `__eq__`，而 `is` 比较的是对象身份。
- **为什么重要**：commit #43770 指出，Dynamo（torch.compile 的编译器）在 torch 2.14 之前无法 trace `RedOpType.__eq__`，用 `==` 会打断 graph。
- **动手试试**：
```python
import torch.distributed as dist

op = dist.ReduceOp.SUM
print("op == SUM:", op == dist.ReduceOp.SUM)   # True，但会触发 __eq__
print("op is SUM:", op is dist.ReduceOp.SUM)   # True，且不触发 __eq__

# 在 torch.compile 下，== 可能导致 graph break
import torch
@torch.compile
def f(x, op):
    if op == dist.ReduceOp.SUM:  # 可能 break
        return x + 1
    return x

@torch.compile
def g(x, op):
    if op is dist.ReduceOp.SUM:  # 不会 break
        return x + 1
    return x

x = torch.randn(4, device="cuda")
print("f:", f(x, op))
print("g:", g(x, op))
```
- **预期结果**：在支持的 torch 版本上，`f` 可能触发 graph break（用 `TORCH_LOGS=graph_breaks` 能看到），而 `g` 不会。这就是"用 `is` 代替 `==`"的原因。

### 知识点 5：扩散模型的参考图 resize 策略

- **是什么**：扩散模型在编码参考图前，需要把图片缩放到合适尺寸。不同策略（diffusers/max/match）会影响生成质量。
- **为什么重要**：commit #44377 文档化了这个选项，因为 LoRA 训练时用的 resize 策略必须和推理时一致，否则效果会崩。
- **动手试试**：
```python
from PIL import Image
import numpy as np

# 模拟三种 resize 策略
def resize_diffusers(img, target_short=2048):
    """短边缩放到 target，小图也放大"""
    w, h = img.size
    scale = target_short / min(w, h)
    return img.resize((int(w*scale), int(h*scale)))

def resize_max(img, target_short=2048):
    """只缩小，不放大"""
    w, h = img.size
    if min(w, h) <= target_short:
        return img
    scale = target_short / min(w, h)
    return img.resize((int(w*scale), int(h*scale)))

def resize_match(img, target_area=2048*2048):
    """按面积匹配，不放大"""
    w, h = img.size
    area = w * h
    if area <= target_area:
        return img
    scale = (target_area / area) ** 0.5
    return img.resize((int(w*scale), int(h*scale)))

# 测试
small = Image.new("RGB", (512, 512))
large = Image.new("RGB", (4096, 4096))

for name, fn in [("diffusers", resize_diffusers), ("max", resize_max), ("match", resize_match)]:
    print(f"{name}: small->{fn(small).size}, large->{fn(large).size}")
```
- **预期结果**：你会看到 `diffusers` 会把 512x512 放大到 2048x2048，而 `max` 和 `match` 保持小图不变。这解释了为什么用不同策略会导致质量差异。

---

## 🔥 重要更新

- **#44401 + #44351 调度器请求计数修复**：这两个 commit 一起修了调度器在请求准入时的资源计数 bug，直接影响高并发下的 OOM 和吞吐，是生产环境的稳定性关键。
- **#38650 Wan VAE 融合 kernel**：把 conv bias 和 RMSNorm+SiLU 融合，Wan 2.1 VAE 解码提速约 10%，且保证 bit-exact，是扩散模型性能优化的典型案例。
- **#44326 TRT-LLM decoder bounded replay**：让 DeepSeek-V4.1 的 TRT-LLM 后端支持 breakable CUDA graph，进一步压低解码延迟。
- **#44058 ComfyUI 集成大改**：SGLDiffusion 的选项分组、原生模型加载、模型感知 preflight，让 ComfyUI 用户更容易用上 SGLang 的扩散能力。

---

## 📋 逐条解读

### 1. [diffusion] doc: document reference resizing, LoRA auto mode and image profiling (#44377)

- **代码层面**：只改文档，涉及 `docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`、`docs/docs/sglang-diffusion/api/cli.mdx`、`docs/docs/sglang-diffusion/profiling.mdx`。新增了 `reference_resize_mode` 的三种取值说明、`--lora-merge-mode auto` 的细节，以及 image API 请求级别的 profiling 用法。
- **新手概念课堂**：就像买家电要读说明书，这个 commit 是给"参考图缩放"和"LoRA 合并模式"补说明书。`reference_resize_mode` 决定参考图怎么缩放，`lora-merge-mode auto` 决定 LoRA 是静态合并还是动态应用。
- **对你有什么影响**：如果你用 MiniMax-H3 或 LoRA，现在能查到官方推荐的参数组合，不用瞎试了。Image API 也支持单请求 profiling，调试性能更方便。

### 2. [diffusion] perf: fuse the Wan VAE conv bias epilogue and tile the RMSNorm+SiLU kernel (#38650)

- **代码层面**：新增 `conv_bias_epilogue_triton.py`，把 conv 的 bias 加法融合进一个 Triton kernel；同时优化 `wan_rmsnorm_silu` 的 tiling。核心是避免 PyTorch 在 channels_last 输出上做非向量化的 broadcast add。
- **新手概念课堂**：想象你在搬砖，PyTorch 原生路径是"先把砖搬到一个地方，再搬 bias 到另一个地方"，而融合 kernel 是"一趟搬完"。关键是搬完后砖的排列必须一模一样（bit-exact）。
- **对你有什么影响**：用 Wan 2.1 做视频生成时，VAE 解码会快约 10%，而且输出和之前完全一致，不用担心质量变化。

### 3. [diffusion] fix: fix the load duration missing from subclassed server tests (#40657)

- **代码层面**：修改 `test_server_common.py` 的 `_validate_and_record` 签名，把 `load_time_ms` 参数改成从 `ctx` 对象读取；同时更新多个测试文件调用点。
- **新手概念课堂**：就像考试时老师要记你的"入场时间"，之前这个时间是通过参数传的，容易漏传；现在改成从"你的准考证"（ctx）上直接读，不会漏。
- **对你有什么影响**：普通用户无感，但如果你在写扩散模型的性能测试，这个改动让测试更健壮，不会因为漏传参数而丢失加载时间数据。

### 4. [Fix] scheduler: Count a request's held req_to_token row once at admission (#44401)

- **代码层面**：在 `scheduler.py` 的 `get_num_allocatable_reqs` 新增 `num_held_rows` 参数，把已持行请求的行数加回可用空间；`dllm/mixin/scheduler.py` 同步修改。
- **新手概念课堂**：餐厅有 10 张桌子，其中 2 桌客人已经坐下但还没点菜（持行），你在算"还能接几桌"时，不能把这 2 桌算成"已占用新桌"，因为它们本来就有桌。
- **对你有什么影响**：高并发下，调度器不会再因为误判空间不足而拒绝请求，吞吐更稳定，OOM 风险降低。

### 5. [CPU]: Add group-aware CPU SHM collective kernels (#35435)

- **代码层面**：在 `csrc/cpu/interface.cpp` 和 `shm.cpp` 里给所有集合通信函数（allreduce、allgather、reduce_scatter、alltoall）加上 `handle` 参数，支持按进程组通信。
- **新手概念课堂**：原来大家一起开会（全局通信），现在支持"小组讨论"（组内通信）。`handle` 就是小组编号。
- **对你有什么影响**：在 CPU 上做张量并行或专家并行时，可以只和同组进程通信，减少不必要的全局同步，提升效率。

### 6. [Test] [AMD] diffusion: Expect CUDA-only fast paths to be off on ROCm in kernel tests (#44311)

- **代码层面**：在多个测试文件里加 `@pytest.mark.skipif(not is_cuda(), ...)`，让 CUDA 专属的 kernel 测试在 ROCm 上跳过；同时把 ROCm 上应该走 fallback 的断言改成验证 fallback。
- **新手概念课堂**：就像"这道菜只有川菜馆有，粤菜馆不卖"，测试要明确哪些功能是 CUDA 专属的，在 AMD 上不要误判为 bug。
- **对你有什么影响**：如果你在 AMD GPU 上跑 SGLang 测试，不会再看到一堆"假失败"，测试结果更可信。

### 7. [Fix] DSv4.1: Size the encoder-replay SWA pool by the DSpark draft's stages, and keep more prefix tails (#43502)

- **代码层面**：新增 `dspark_num_stages` 字段，在 `pool_configurator.py` 里根据 draft 层数调整 SWA 池大小和 prefix tails 数量。
- **新手概念课堂**：SWA（滑动窗口注意力）的缓存池就像停车位，原来按"总层数"估算，现在发现只有 draft 层才需要车位，所以可以多留一些"尾巴"给前缀复用。
- **对你有什么影响**：用 DeepSeek-V4.1 + DSpark 做长上下文推理时，前缀缓存命中率更高，重复前缀不用重新计算。

### 8. [Fix] dsv41: Bound MoE routing to live SWA graph-tail rows (#44347)

- **代码层面**：在 `deepseek_v4_backend.py` 里，进入 late layer tail 时把 `global_num_token_non_padded` 和 `num_token_non_padded` 都 clamp 到真实行数，退出时恢复。
- **新手概念课堂**：CUDA graph 里 tensor 是"填满"的（有 padding），但 MoE 路由只能看真实行，不能把 padding 也算进去，否则会路由到不存在的 token。
- **对你有什么影响**：DeepSeek-V4.1 在 prefill CUDA graph 下跑 MoE 时，路由更准确，不会因为 padding 行导致计算错误。

### 9. [Feat] sgl-router: Add --dp-rank-policy round_robin for --dp-aware (#43511)

- **代码层面**：在 sgl-router 的 CLI 和配置里新增 `--dp-rank-policy`，支持 `round_robin` 和默认的 `least_in_flight`；在 PD 模式下，decode rank 选择也走这个策略。
- **新手概念课堂**：路由到哪个 DP rank，原来只看"谁最闲"，现在可以选"轮流来"，适合请求量均匀的场景。
- **对你有什么影响**：如果你的负载比较均匀，用 `round_robin` 可以避免某些 rank 被反复选中，负载更均衡。

### 10. [diffusion] comfyui: grouped SGLDOptions, native model loader and model-aware preflight (#44058)

- **代码层面**：重构 ComfyUI 插件，把 `SGLDiffusion Options` 的选项分组（Parallel/Attention/Quantization/Cache/Execution/Memory），新增 `SGLDiffusion Native Model Loader` 节点，加载前做模型感知的 preflight 检查。
- **新手概念课堂**：原来选项是一长串，现在像手机设置一样分组折叠；新增的 loader 能直接读 SGLang 原生的模型目录，不用先转成 ComfyUI 格式。
- **对你有什么影响**：用 ComfyUI + SGLang 时，配置更清晰，加载原生模型更方便，错误提示更早（preflight 阶段就报错，不用等 worker 启动）。

### 11. [Fix] scheduler: Drop finished requests from the running batch before every admission (#44351)

- **代码层面**：把 `filter_batch()` 从"只在 prefill-only 时调用"改成"每一步都调用"，并检查 batch size 是否变小来决定是否清 `batch_is_full`。
- **新手概念课堂**：就像餐厅每来一桌新客人前，先看看有没有老客人已经吃完走了，把桌子腾出来，而不是等到"只做点菜不做上菜"时才清理。
- **对你有什么影响**：overlap 模式下，完成的请求能及时从 running batch 移除，`/v1/loads` 的 `num_running_reqs` 更准确，准入判断更准。

### 12. [PD] Send DSA indexer state only past decode_prefix_len (#44140)

- **代码层面**：在 `decode.py` 和 `prefill.py` 里，DSA 状态的 payload 从"全量 KV pages"改成"只发 decode_prefix_len 之后的 suffix pages"，用 `suffix_page_indices` 实现。
- **新手概念课堂**：PD 分离下，decode 端已经有前缀的 indexer 行了，prefill 端不用重复发，只发"新增部分"就行，省带宽。
- **对你有什么影响**：PD 分离 + DeepSeek-V4 的 DSA 索引器场景下，KV 传输量减少，首 token 延迟降低。

### 13. [diffusion] comfyui: return several server-mode images as one IMAGE batch (#44088)

- **代码层面**：在 ComfyUI 插件里新增 `num_images` 参数（1-10），`_decode_image_batch` 把服务器返回的多张图拼成一个 IMAGE batch；每张图用 `seed + i`。
- **新手概念课堂**：原来一次请求只能拿一张图，现在可以一次拿多张，像"连拍"一样，而且种子递增保证可复现。
- **对你有什么影响**：在 ComfyUI 里做 server 模式生成时，可以一次出多张图，方便挑选或做 batch 处理。

### 14. [Doc][AMD] Update v41 amd cookbook 1015 (#44368)

- **代码层面**：更新 DeepSeek-V4.1 的 AMD cookbook，包括 Docker 镜像版本、TP2/TP4 的并发范围、新增环境变量 `SGLANG_OPT_HIP_FLYDSL_MXFP8=1`、调整 `--swa-prefix-tails` 等参数表。
- **新手概念课堂**：这是"AMD 玩家的攻略更新"，告诉你在不同并发下该怎么配参数。
- **对你有什么影响**：如果你在 AMD MI300X 上部署 DeepSeek-V4.1，直接照抄新参数表，性能更好。

### 15. [Perf] [AMD] dsa: Enable the gfx950 fused DSA indexer for block-FP8 wq_b/wk (#43643)

- **代码层面**：在 `fused_decode.py` 里，当 `wq_b` 和 `wk` 是 block-FP8 时，不再预合并权重，而是让 node 1 调用层自己的 GEMM；同时更新 `_derived_weight_cache_error` 逻辑。
- **新手概念课堂**：原来把两个 bf16 权重拼成一个大权重做一次 GEMV，但 FP8 权重不能这么拼，所以改成"各算各的"，但仍在同一个 kernel 里。
- **对你有什么影响**：AMD gfx950 上用 GLM-5.3 等 block-FP8 模型时，DSA 索引器的 decode 更快，而且在线权重更新会被正确拒绝（不会用旧权重）。

### 16. [Doc] cookbook: Add Step 5 Preview cookbook (#44357)

- **代码层面**：新增 `Step-5-Preview.mdx`，介绍 StepFun 的 600B MoE 模型部署，包括 B300/H200 的 TP4/EP8 配置、MegaMoE、FlashInfer TRT-LLM MoE 等。
- **新手概念课堂**：这是"新模型上手指南"，告诉你 600B 模型怎么在有限 GPU 上跑起来。
- **对你有什么影响**：如果你想试 Step 5 Preview，直接照 cookbook 配，不用从零摸索。

### 17. [DeepSeek-V4.1] DeepSelect cluster top-k for long rows and bf16 index logits (#44051)

- **代码层面**：重写 DeepSelect 的 cluster top-k kernel，支持长行（多 CTA 协作）、bf16 索引 logits、NaN 时写 0 而不是 0x3F3F3F3F，并优化 histogram pivot 查找。
- **新手概念课堂**：top-k 选择就像"从一堆候选人里选前 k 名"，长行（很多 token）需要多个 CTA 一起选，最后合并。
- **对你有什么影响**：DeepSeek-V4.1 在长上下文下，索引器 top-k 更快，且 NaN 处理更安全（不会越界）。

### 18. [Perf] [AMD] dsa: Run DSA prefill under breakable CUDA graphs on HIP (#43642)

- **代码层面**：在 `dsa_indexer.py` 和 `dsa_prefill_cuda_graph.py` 里，让 HIP 上的 DSA prefill 支持 breakable CUDA graph，处理 aiter 融合 RMSNorm+quant 返回的 tuple。
- **新手概念课堂**：breakable graph 就像"录像时允许暂停去接个电话"，DSA prefill 里有些动态逻辑需要在 graph 中间跑。
- **对你有什么影响**：AMD GPU 上 DeepSeek-V4 的 prefill 延迟更低，因为 graph 减少了 kernel 启动开销。

### 19. [Fix] [AMD] diffusion: Compare ReduceOp by identity so the compiled TP all-reduce stays one graph (#43770)

- **代码层面**：在 `group_coordinator.py` 里把 `op == torch.distributed.ReduceOp.SUM` 改成 `op is torch.distributed.ReduceOp.SUM`。
- **新手概念课堂**：`==` 会调用 `__eq__` 方法，Dynamo 无法 trace 它，会导致 graph break；`is` 只是比较对象身份，不触发方法调用。
- **对你有什么影响**：AMD 上用 torch.compile 做 TP all-reduce 时，graph 不会被打断，性能更好。

### 20. [DSv4.1] Support TRT-LLM decoder bounded replay with BCG (#44326)

- **代码层面**：在 `deepseek_v4_hook.py` 里允许 TRT-LLM + `--cuda-graph-backend-prefill breakable`；在 `deepseek_v4_trtllm_backend.py` 里让 prefill 在 BCG break 时用 tail-relative 的 `seq_lens`。
- **新手概念课堂**：BCG（breakable CUDA graph）允许在 graph 中间跑动态逻辑，TRT-LLM 的 prefill 需要根据 tail 调整 seq_lens，才能正确做 SWA masking。
- **对你有什么影响**：DeepSeek-V4.1 用 TRT-LLM 后端时，decoder bounded replay 可用，解码延迟进一步降低，且不再要求 prefill graph 完全关闭。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)