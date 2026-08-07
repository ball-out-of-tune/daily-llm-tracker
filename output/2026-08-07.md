# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 15:32 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词 1：异构计算加速（AMD ROCm / Intel XPU）**
  - **一句话通俗解释**：vLLM 不再只支持 NVIDIA GPU，正在大力适配 AMD 和 Intel 的硬件，让更多人在不同显卡上跑大模型。
  - **为什么社区在关注**：今天的 20 个 commits 中有 5 个直接涉及 AMD/Intel 平台适配。这说明大模型推理正在从"NVIDIA 独占"走向"多平台共存"，各大云厂商和硬件厂商都在争夺推理市场。
  - **对新手意味着什么**：如果你只有 AMD 或 Intel 电脑，现在也可以尝试跑 vLLM 了。学习时不用非买 NVIDIA 卡，降低了入门门槛。

- **关键词 2：KV Cache 管理与优化**
  - **一句话通俗解释**：KV Cache 是大模型推理时保存中间计算结果的高速缓存，管理好它就能大幅提升推理速度和吞吐量。
  - **为什么社区在关注**：今天有 6 个 commits 涉及 KV transfer/offload/prefix caching 相关的 bugfix 和优化。这说明 KV Cache 是当前推理性能优化的核心战场，大家都在解决分布式场景下的缓存一致性和效率问题。
  - **对新手意味着什么**：理解 KV Cache 是理解 vLLM 性能优化的关键。建议花时间弄懂"什么是 KV Cache""什么是 prefix caching"，这是面试和实战的热点。

- **关键词 3：量化（Quantization）与低精度推理**
  - **一句话通俗解释**：把模型权重从 16 位/32 位压缩到 8 位甚至 4 位，用更少的内存跑更大的模型。
  - **为什么社区在关注**：今天有 4 个 commits 涉及量化（FP8、INT8、NVFP4）的 bugfix 和配置修复。量化是部署大模型的刚需——没有它，很多模型在消费级显卡上根本跑不起来。
  - **对新手意味着什么**：学会理解量化配置（如 `--quantization fp8`）和不同量化格式的差异，是实际部署模型时绕不开的技能。

- **关键词 4：分布式推理与多机多卡**
  - **一句话通俗解释**：把一个大模型拆到多张 GPU 或多台机器上协同推理，解决单卡显存不够的问题。
  - **为什么社区在关注**：多个 commits 涉及 NCCL 通信、EPLB（专家并行负载均衡）、MoRIIO 传输等分布式组件。随着模型越做越大，分布式推理从"可选"变成"必选"。
  - **对新手意味着什么**：分布式概念（TP/PP/EP/DP）是进阶必学。不过建议先把单机跑通，再逐步学习多卡。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：端口分配中的死锁（Livelock）问题

- **一句话解释**：程序在找可用端口时，如果逻辑有缺陷，可能陷入无限循环——每次都找到一个"看似可用"但实际被保留的端口，永远跳不出去。
- **为什么重要**：vLLM 在分布式推理时需要为多个进程分配端口。如果端口分配卡死，整个服务就起不来。这个 bug 影响的是 `get_open_port()` 函数——当环境变量 `VLLM_PORT` 恰好落在 DP（数据并行）保留端口区间内时，函数会无限循环。
- **动手试试**：
  1. 在你本地安装的 vLLM 中找到 `vllm/utils/network_utils.py` 里的 `get_open_port` 函数。
  2. 设置环境变量 `VLLM_DP_MASTER_PORT=5680` 和 `VLLM_PORT=5682`（5682 在 5680-5690 的保留区间内）。
  3. 调用 `get_open_port()` 并设置超时（比如 5 秒），观察是否卡住。
  4. 修复前：函数会一直循环，永远不返回。修复后：函数会跳过保留区间，返回一个区间外的可用端口。
- **预期结果**：你能亲眼看到"死循环"现象，然后理解修复逻辑——在检查端口可用性时，还要检查该端口是否在保留区间内。

### 知识点 2：Prefix Caching（前缀缓存）

- **一句话解释**：当多个请求共享相同的前缀（比如同一个 system prompt 或对话历史）时，只计算一次前缀的 KV Cache，后续请求直接复用，大幅减少重复计算。
- **为什么重要**：这是 vLLM 提高吞吐量的核心机制之一。今天的 commits 中有两个专门修复 prefix caching 在分布式场景下的 bug，说明这个功能虽然强大但实现复杂。
- **动手试试**：
  1. 启动 vLLM 服务时加上 `--enable-prefix-caching` 参数。
  2. 发送两个共享相同前缀的请求（比如都以 "The capital of France is" 开头）。
  3. 观察第二个请求的处理时间——如果 prefix caching 生效，第二个请求会明显更快。
  4. 再对比不开启 `--enable-prefix-caching` 时两个请求的处理时间差异。
- **预期结果**：开启后第二个请求的 TTFT（首 token 延迟）显著降低，说明前缀被缓存并复用了。

---

## 🔥 重要更新

1. **修复 `get_open_port()` 端口分配死锁**（Commit 1）：分布式部署时端口分配卡死会导致服务无法启动，这个修复直接关系到 vLLM 的可用性。
2. **Qwen3.5 模型支持 AMD ROCm**（Commit 4）：让 Qwen3.5 在 AMD 显卡上跑起来，是国产模型 + AMD 硬件组合的重要一步。
3. **磁盘卸载支持**（Commit 16）：KV Cache 不仅能放 GPU 显存和 CPU 内存，现在还能放到磁盘上，极大扩展了可服务的模型规模。
4. **修复 NVML 重复初始化性能问题**（Commit 11）：每次检查设备能力都重新初始化 NVML，在每层注意力计算中都会被调用，修复后推理性能有明显提升。

---

## 📋 逐条解读

### 1. [Bugfix] Fix get_open_port() livelock on DP-reserved ports
- **代码层面**：修改了 `vllm/utils/network_utils.py` 中的 `get_open_port()` 函数，使其在分配端口时跳过 DP 保留区间（`VLLM_DP_MASTER_PORT` 到 +10 的范围）。同时增加了带超时的测试，防止测试本身卡死。
- **新手概念课堂**：想象你在停车场找车位。普通找法是一个个看有没有空位。但如果某个区域被标记为"预留"，你每次看到一个空位走进去才发现不能用，然后出来重新找——如果整个停车场都是预留位，你就会永远转圈。修复就是：提前知道哪些区域是预留的，直接跳过。
- **对你有什么影响**：如果你使用多卡分布式部署（如 `--tensor-parallel-size 8`），不会再遇到服务启动时卡死的问题。

### 2. [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier
- **代码层面**：修改了 `moriio_connector.py`，为 MoRIIO 的 KV 读取增加了逐层完成屏障，并修复了 CUDA Graph 模式下屏障无法触发的问题（增加警告而非静默降级）。
- **新手概念课堂**：想象一条流水线，每个工位完成自己的任务后要通知下一个工位可以开始了。如果某个工位没通知，下一个工位就可能拿到半成品。这里的"屏障"就是确保每一层都真正读完了 KV 再继续。
- **对你有什么影响**：使用 MoRIIO 做 KV 传输的高并发场景下，推理结果的准确性有保障，不会因为读取未完成而算错。

### 3. [PD][NixlPush][Bugfix] Fix prefix caching
- **代码层面**：修复了 NixlPush 模式下 prefix caching 的裁剪逻辑——当部分前缀命中时，只传输"未计算"的尾部块，而不是错误地传输整个序列。同时增加了针对 Mamba 混合模型的测试。
- **新手概念课堂**：你在图书馆借书，发现前 10 页有人已经抄好了（缓存命中），你只需要抄第 11 页开始的内容。但之前的 bug 是：它把前 10 页的内容又抄了一遍，还抄到了错误的位置。
- **对你有什么影响**：使用 PD（Prefill-Decode）分离架构时，prefix caching 能正确工作，不会浪费带宽或产生错误结果。

### 4. [Model] Enable Qwen3.8 for AMD Rocm
- **代码层面**：在 `Qwen3_5ForCausalLMBase` 中增加了 `SupportsMRoPE` 支持，并实现了 `get_mrope_input_positions` 方法，使 Qwen3.5 模型能在 AMD ROCm 平台上运行。
- **新手概念课堂**：MRoPE 是一种位置编码方式，不同硬件平台可能需要不同的实现方式。这就像同一本书，在不同出版社有不同的排版，但内容是一样的。
- **对你有什么影响**：AMD 显卡用户现在可以运行 Qwen3.5 模型了。

### 5. [Bugfix] Skip fetching revision for model when model and weights_model are different
- **代码层面**：修改了 `vllm/config/model.py` 中的 `__post_init__` 逻辑——当 `model_weights` 与 `model` 不同（比如模型来自 A 仓库、权重来自 B 仓库的 GGUF 文件）时，不再尝试从 HuggingFace 解析 revision，避免不必要的网络请求和潜在错误。
- **新手概念课堂**：revision 是 HuggingFace 上模型的版本号。如果你用 A 模型的配置 + B 模型的权重，那么 B 的 revision 才是你真正需要的。之前的代码会去 A 找 revision，找错了地方。
- **对你有什么影响**：使用 GGUF 等外部权重文件时，启动速度更快，不会因为多余的网络请求而报错。

### 6. Fix ROCm architecture import on non-ROCm platforms
- **代码层面**：在 `mxfp4.py` 和 `oracle/mxfp4.py` 中，将 `from vllm.platforms.rocm import on_gfx1250` 改为先检查 `current_platform.is_rocm()` 再导入。避免在非 ROCm 平台上导入 ROCm 专属模块导致报错。
- **新手概念课堂**：就像你在 Windows 电脑上安装了 Mac 专属的驱动程序，虽然不会用，但安装过程可能报错。修复就是：先检查是不是 Mac，不是就不装。
- **对你有什么影响**：非 AMD 用户不会再因为导入 ROCm 模块而遇到 ImportError。

### 7. feat: extended EPLB support for Mistral Large 3 and additional MoE backends
- **代码层面**：增加了新的测试文件 `test_eplb_quant_scale_consistency.py`，验证 EPLB（专家并行负载均衡）在重排专家权重后，量化相关的 scale 和 alpha 参数能正确同步。扩展了 EPLB 对 Mistral Large 3 和更多 MoE 后端的支持。
- **新手概念课堂**：EPLB 就像餐厅里动态调整服务员负责的桌子数量，让每个服务员工作量均衡。但如果调整桌子后，每张桌子对应的"小费记录"（量化参数）没跟着搬，就会出错。
- **对你有什么影响**：使用 Mistral Large 3 或特定 MoE 后端时，推理结果更稳定。

### 8. [XPU] quick fix online quantization UT break
- **代码层面**：在 `test_online.py` 中将硬编码的 `device="cuda"` 改为 `DEVICE = current_platform.device_type`，并在 NVFP4 的跳过条件中增加了 XPU 判断。
- **新手概念课堂**：测试代码里写死了"只能在 CUDA 上跑"，但 Intel XPU 也可以跑部分测试。修复就是让测试自动识别当前平台。
- **对你有什么影响**：Intel 显卡用户可以运行更多量化相关的测试。

### 9. [Misc] Add and enable Triton kernel unit tests on XPU
- **代码层面**：在多个测试文件中将 `device = torch.device("cuda:0")` 改为根据平台动态选择设备，并调整了跳过条件以支持 XPU。
- **新手概念课堂**：测试代码从"只认 NVIDIA"变成"认识所有平台"，让 Intel 用户也能验证 Triton kernel 的正确性。
- **对你有什么影响**：Intel 用户可以更放心地使用 vLLM 的 Triton kernel。

### 10. [PD][PushConnector] Record last activity of remotes to allow clean up of stale ones
- **代码层面**：在 `base_worker.py` 中增加了 `_engine_last_active` 记录机制，每次推送 KV 时刷新对应远端引擎的活动时间，并在握手时清理超过 TTL 的过期引擎。
- **新手概念课堂**：就像社交软件上"最近活跃时间"——如果某个好友超过 30 天没上线，系统就把他从好友列表移除。这里同理，如果某个远端引擎超过 TTL 没活动，就清理掉它的连接资源。
- **对你有什么影响**：分布式部署中，当某个节点缩容或重启后，不会残留僵尸连接占用资源。

### 11. [Bugfix][Platform] Stop re-initializing NVML on every device-capability check
- **代码层面**：移除了 `has_device_capability` 中的 `with_nvml_context` 包装，因为 `get_device_capability` 内部已经维护了 NVML 上下文。避免了每次调用都执行 `nvmlInit()`/`nvmlShutdown()` 对。
- **新手概念课堂**：想象你每次要查一个电话号码，都重新打开电话簿、查完再合上。修复就是：电话簿一直打开着，直接查就行。这个函数在每层注意力计算中都会被调用，所以性能提升显著。
- **对你有什么影响**：推理速度有可感知的提升，尤其是在层数多的模型中。

### 12. [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16
- **代码层面**：修改了 `oracle/int8.py` 中的 `make_int8_moe_quant_config`——当 `per_act_token_quant=True` 且没有提供 activation scale 时，不再错误地构建成 W8A16 配置，而是正确构建 W8A8 动态量化配置。同时将 assert 改为更友好的 ValueError。
- **新手概念课堂**：W8A8 和 W8A16 的区别是：权重都是 8 位，但激活值（中间计算结果）一个是 8 位、一个是 16 位。之前的 bug 是：明明想要 W8A8 动态量化，却配成了 W8A16。
- **对你有什么影响**：使用动态 INT8 量化的 MoE 模型时，推理速度和显存占用符合预期。

### 13. [Refactor] Remove kernel dead code
- **代码层面**：删除了 `cpu_attn_fp8.hpp` 中未使用的 `fp8e5m2_to_float_scalar` 函数，以及 `libtorch_stable/cache_kernels.cu` 中未使用的 `copy_blocks_kernel` 等 CUDA kernel。
- **新手概念课堂**：代码清理就像整理房间——把不用的东西扔掉，房间更整洁，找东西也更快。虽然不影响功能，但让代码库更易维护。
- **对你有什么影响**：无直接用户感知，但代码库更健康。

### 14. Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel
- **代码层面**：新增了 DeepSeek-V4 在 AMD 上的 NVFP4 量化支持，包括新增测试配置文件（`DeepSeek-V4-Flash-NVFP4.yaml` 等）和 CI 流水线步骤。
- **新手概念课堂**：NVFP4 是一种 4 位浮点量化格式。AMD 用"模拟 kernel"（emulation kernel）来跑原本为 NVIDIA 设计的量化格式——就像用翻译软件读外文书。
- **对你有什么影响**：AMD 用户可以在 MI355 等显卡上运行 DeepSeek-V4 模型。

### 15. [ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes
- **代码层面**：在 `test_block_fp8.py` 中，将参考实现改为使用生产级的 `per_token_group_quant_fp8` 函数，并增加了 `silu_fp32` 选项来匹配不同 kernel 的精度行为，同时放宽了大规模 K 形状下的误差容忍度。
- **新手概念课堂**：测试就像考试，之前的"标准答案"（参考实现）不够精确，导致 AMD 显卡上算出的结果和标准答案差距略大、被判"不合格"。修复就是让标准答案更接近真实 kernel 的行为。
- **对你有什么影响**：AMD 显卡上运行 FP8 量化的 MoE 模型，CI 测试更稳定。

### 16. [Feat][Core] Add disk offloading support to SimpleCPUOffloadConnector
- **代码层面**：在 `simple_cpu_offload_connector.py` 中新增 `kv_offload_backend` 配置，支持 `"cpu"` 和 `"disk"` 两种后端。磁盘后端支持配置 `disk_path`、`disk_capacity_bytes`、`disk_buffer_slots` 等参数。
- **新手概念课堂**：之前 KV Cache 只能放 GPU 显存或 CPU 内存。现在可以放到磁盘上——就像把不常用的书从书桌（显存）移到书架（内存）再移到储藏室（磁盘）。容量更大但速度更慢。
- **对你有什么影响**：超长上下文或超大模型可以在显存不足时，把部分 KV Cache 卸载到磁盘，扩展可服务的最大序列长度。

### 17. [rl] Stateful Trainer Send: NCCL + Sparse NCCL [3/N]
- **代码层面**：重构了 RL 训练器向推理引擎发送权重的 API。将原来的 `NCCLWeightTransferEngine.trainer_init` + `trainer_send_weights` 模式改为新的 `WeightTransferTrainerFactory.trainer_init` + `ModuleSource` 模式，简化了调用方式。
- **新手概念课堂**：RLHF 训练中，训练器需要不断更新模型权重并同步给推理引擎。旧 API 需要手动处理很多细节，新 API 封装得更好，用起来更简单。
- **对你有什么影响**：如果你在做 RLHF，新的 API 更简洁易用。

### 18. [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path
- **代码层面**：新增了 `tests/models/kimi_k3/` 测试目录，为 ROCm 平台的 Kimi-K3 模型的 latent MoE up-projection 增加了分片逻辑的单元测试。
- **新手概念课堂**：Kimi-K3 是 MoE 模型，up-projection 是其中一步。在 ROCm 上，为了性能需要把这一步分片到多卡上算，但分片逻辑容易出错。测试就是验证分片后结果和不分片一致。
- **对你有什么影响**：AMD 多卡用户运行 Kimi-K3 时，性能和正确性有保障。

### 19. [Bugfix] Fix Mamba all-mode CPU offload boundary alignment
- **代码层面**：在 `offloading/scheduler.py` 中，将 `resolve_mamba_align_size` 的检查条件从仅 `"align"` 模式扩展到 `"align"` 和 `"all"` 两种模式，并增加了 `mamba_cache_mode="all"` 的测试。
- **新手概念课堂**：Mamba 模型的 KV Cache 和 Transformer 不一样——它只有一个"状态"，而不是每个 token 都有 KV。在 CPU offload 时，如果边界对齐不对，可能把包含当前 token 的状态也加载进来，导致重复计算。
- **对你有什么影响**：使用 Mamba 模型 + CPU offload 时，推理结果更准确。

### 20. [Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance
- **代码层面**：修改了 `test_encoder_runner.py`，新增测试验证 encoder-only 实例（EPD 架构中的编码器节点）只执行 `execute_mm_encoder` 而不执行 `get_mm_embeddings` 中的 gather 步骤——因为 encoder 实例不运行语言模型，gather 出来的 embedding 没有人消费，反而会因 cache miss 导致崩溃。
- **新手概念课堂**：EPD（Encoder-Prefill-Decode）架构中，编码器节点只负责把图片/视频转成 embedding，不负责生成文本。之前的代码会让编码器节点也执行"收集 embedding"的步骤，但这个节点根本没有语言模型来用这些 embedding，反而报错。
- **对你有什么影响**：使用 EPD 架构处理多模态输入时，编码器节点不再崩溃。

## sglang
## 💡 今日关键词

- **CUDA Graph 与可中断图（Breakable CUDA Graph）**：把模型计算预编译成一张"快照图"，执行时跳过 Python 解释器直接跑 GPU，大幅降低调度开销。今日多个 commit 在解决它与流水线并行、张量并行的兼容性问题。
- **流水线并行（Pipeline Parallelism, PP）**：把大模型按层切分到多张 GPU 上，每张卡只负责一部分层。今日的修复表明 PP 已进入实用阶段，但边界条件（如 KV cache 索引）仍在持续打磨。
- **扩散模型（Diffusion）推理优化**：SGLang 正在将文生图/文生视频的推理性能推向极致，包括 CUDA Graph、注意力后端抽象、checkpoint 加载加速等。这是目前社区最活跃的领域之一。
- **量化推理（W4A16 NVFP4）**：用 4-bit 权重 + 16-bit 激活来压缩模型内存占用。今日修复了 MoE（混合专家）模型在该量化方案下的后端选择问题。
- **健康检查与部署运维**：将"进程存活"和"服务就绪"两个概念分离，提供不同的探针端点，这是大规模部署的标配能力。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：流水线并行中的"层偏移"陷阱

- **一句话解释**：当模型被切分到多张 GPU 上时，第 0 层可能不在当前 GPU 上，所以不能假设 `layer 0` 一定存在。
- **为什么重要**：今日 commit #9 修复的正是这个问题——在 PP 模式下，KV cache 的 `v_head_dim` 查询用了 `layer 0` 导致 IndexError。这类 bug 在 PP 推广后会越来越常见。
- **动手试试**：
  1. 用 `torch.distributed` 初始化一个 2 卡环境，将模型按层切分（例如 `model.layers[0:12]` 放 rank 0，`model.layers[12:24]` 放 rank 1）。
  2. 在 rank 1 上尝试访问 `model.layers[0]`，观察报错。
  3. 改用 `model.layers[model.start_layer]` 或记录每张卡的起始层索引，验证能正确访问。
- **预期结果**：你会看到 `IndexError: list index out of range` 或类似错误，然后通过正确的层索引解决问题。这说明你理解了 PP 中"每张卡只拥有部分层"的核心约束。

### 知识点 2：CUDA Graph 捕获时的"非法操作"

- **一句话解释**：CUDA Graph 捕获期间，所有操作必须是可图化的（graph-safe），例如不能在捕获中做 CPU 到 GPU 的同步拷贝。
- **为什么重要**：今日 commit #2 专门处理了 LTX-2 模型中在 CUDA Graph 捕获区域内构建 RoPE 坐标的问题——那段代码用了 `torch.tensor(list, device=cuda)`（未固定内存的 H2D 拷贝），这在图捕获中是违法的。
- **动手试试**：
  1. 写一个简单的 PyTorch 模型，在 `forward` 中做 `torch.tensor([1,2,3], device='cuda')`。
  2. 用 `torch.cuda.graph()` 上下文捕获该模型的前向传播。
  3. 观察捕获是否报错，然后改为在捕获前预分配张量，再在捕获中只做 `copy_` 操作。
- **预期结果**：第一次捕获会报错（`Capture was not successful` 或类似信息），修改后捕获成功。这说明你理解了 CUDA Graph 对操作类型的严格限制。

---

## 🔥 重要更新

1. **LTX-2 可中断 CUDA Graph 加速（commit #2）**：H200 上端到端延迟从 10.75 秒降到 6.90 秒（1.56 倍加速），这是扩散模型推理性能的重大提升。
2. **Nemotron W4A16 NVFP4 MoE 后端修复（commit #5）**：修复了 MoE 模型在 4-bit 量化下可能选错推理后端的问题，并增加了防护性检查。
3. **流水线并行 KV cache 索引修复（commit #9）**：修复了 PP 模式下 Triton 注意力后端的 IndexError，这是 PP 走向成熟的重要一步。
4. **健康检查端点拆分（commit #10）**：新增 `/liveness` 和 `/health` 分离，为 Kubernetes 部署提供标准探针方案。

---

## 📋 逐条解读

### 1. 移除不再使用的 HiMambaRadixTree
- **代码层面**：删除了 `hi_mamba_radix_cache.py` 整个文件（约 2178 行），这是一个用于 Mamba 模型的缓存数据结构。
- **新手概念课堂**：Radix Tree（基数树）是一种高效存储和检索前缀的数据结构，类似字典的树形版本。Mamba 是一种状态空间模型。这个类曾经用于混合缓存，现在被废弃了。
- **对你有什么影响**：无直接影响，这是清理死代码。但说明项目在持续演进，旧的缓存策略正在被更简单的方案替代。

### 2. LTX-2 启用可中断 CUDA Graph
- **代码层面**：修改了 `breakable_cuda_graph/runner.py` 和 LTX-2 的 denoising 阶段。增加了诊断日志（签名不匹配时输出差异字段），并将最大分段数从 128 提升到 512。
- **新手概念课堂**：可中断 CUDA Graph 是把一个长计算图切成多个"分段"，每个分段可以独立重放。LTX-2 有 48 个 Dual-Tower 块，每个块有 6 个注意力断点，所以需要更大的分段上限。
- **对你有什么影响**：如果你使用 LTX-2 文生视频模型，延迟会显著降低（1.56 倍加速）。

### 3. 调度器 RPC 超时显式化
- **代码层面**：新增 `--scheduler-rpc-timeout` 参数，默认不设置（避免长任务被误杀）。同时修复了服务器关闭时视频任务未正确取消的问题。
- **新手概念课堂**：RPC（远程过程调用）是不同进程间的函数调用。调度器 RPC 是 HTTP 服务器向调度器发送请求的通道。如果任务排队很久，默认超时可能导致任务失败。
- **对你有什么影响**：长视频生成任务不再因为传输层超时而失败。服务器优雅关闭时，正在生成的视频任务会被正确取消。

### 4. 修复 VAE 快速路径测试
- **代码层面**：修改了测试文件，适应新的 `use_vae_fast_path` 上下文管理器 API。之前通过 `gate.enabled` 属性切换，现在改用上下文管理器。
- **新手概念课堂**：VAE（变分自编码器）是扩散模型中负责图像编解码的组件。"快速路径"是指用融合算子（fused kernels）替换多个独立算子，提高速度。上下文管理器是 Python 中 `with` 语句使用的对象。
- **对你有什么影响**：这是测试代码，但反映了 API 设计趋势——用上下文管理器控制优化开关比手动设置属性更安全。

### 5. 修复 Nemotron W4A16 NVFP4 MoE 后端
- **代码层面**：在 `overrides.py` 中检测是否包含 W4A16_NVFP4 量化的 MoE 层，如果是则强制使用 `marlin` 后端并要求禁用 `moe-a2a-backend`。同时修复了 Marlin 权重 padding 后的维度计算。
- **新手概念课堂**：MoE（混合专家）模型有多个"专家"子网络，每次只激活其中几个。NVFP4 是 NVIDIA 的 4-bit 浮点格式。Marlin 是一种高效的矩阵乘法内核。
- **对你有什么影响**：如果你使用 Nemotron 模型的 4-bit 量化版本，推理不再崩溃或产生错误结果。

### 6. 加速 TP 和 FSDP checkpoint 加载
- **代码层面**：重构了 FSDP 加载流程，支持直接从 rank-local checkpoint 加载（跳过 safetensors 迭代器），并优化了参数名映射。
- **新手概念课堂**：FSDP（完全分片数据并行）把模型参数切分到多张 GPU。Checkpoint 是模型权重的存档。之前加载时需要逐参数处理，现在可以批量快速加载。
- **对你有什么影响**：多卡部署时模型加载时间大幅缩短，特别是大模型（如 30B+）从几分钟降到几十秒。

### 7. 分布式初始化前绑定设备
- **代码层面**：将设备绑定（`set_device`）移到分布式初始化之前，并抽象为 `current_platform.set_device` 接口，支持 CUDA、NPU、MPS 等平台。
- **新手概念课堂**：分布式训练中，每个进程必须先绑定到特定 GPU，再初始化通信组。之前是先初始化再绑定，可能导致设备错乱。
- **对你有什么影响**：多卡推理/训练时，每张卡能正确分配到对应的 GPU，避免"所有进程都跑在 GPU 0"的问题。

### 8. 支持 TP 下的可中断 CUDA Graph
- **代码层面**：在 CUDA Graph 捕获时进入 TP 组的 `graph_capture` 上下文，确保自定义 all-reduce 在图捕获期间使用正确的路径。
- **新手概念课堂**：TP（张量并行）把单个算子的计算切分到多张 GPU。CUDA Graph 捕获时，通信操作（如 all-reduce）必须也是图安全的。
- **对你有什么影响**：TP 模式下扩散模型也能使用 CUDA Graph 加速了。

### 9. 修复 Triton 后端的 PP 索引错误
- **代码层面**：将 `get_value_buffer(0)` 改为 `get_value_buffer(start_layer)`，在多处（`triton_backend.py`、`memory_pool.py`）修复。
- **新手概念课堂**：KV cache 是按层存储的。PP 模式下，每张卡只存部分层的 KV cache。用 `0` 索引会导致访问不存在的第 0 层。
- **对你有什么影响**：使用 PP + Triton 注意力后端的组合不再崩溃。

### 10. 健康检查端点拆分
- **代码层面**：新增 `/liveness` 端点（进程存活），`/health` 在 warmup 期间返回 503，完成后返回 200。
- **新手概念课堂**：Kubernetes 有 liveness（进程是否活着）、readiness（是否准备好接收流量）、startup（启动是否完成）三种探针。之前 SGLang 只有一个 `/health`，无法区分"还在启动"和"已经挂了"。
- **对你有什么影响**：部署到 Kubernetes 时，探针配置更标准，不会因为启动慢而被误杀。

### 11. 支持 LingBot-Video MoE 30B 模型
- **代码层面**：新增 `lingbot_video_moe.py` 配置文件，定义了模型架构参数（48 层、128 专家、每 token 激活 8 个专家），并注册到模型列表。
- **新手概念课堂**：LingBot-Video 是新的文生视频模型，MoE 版本有 30B 参数但推理时只激活部分参数。`patch_size=(1,2,2)` 表示视频的空间和时间维度的下采样倍率。
- **对你有什么影响**：可以直接用 SGLang 运行 LingBot-Video MoE 30B 模型生成视频。

### 12. Ring 注意力作为后端能力
- **代码层面**：新增 `supports_ring_rotation()` 类方法，FlashAttention 和 SageAttention 返回 True，其他后端默认 False。Ring 注意力只在支持的后端上启用。
- **新手概念课堂**：Ring Attention（环形注意力）是把长序列切分到多张 GPU 上，像接力棒一样循环传递计算。它需要内核能输出 softmax 的 LSE（log-sum-exp）才能合并结果。
- **对你有什么影响**：使用不兼容的注意力后端时，Ring 注意力会被自动禁用，避免错误结果。

### 13. Qwen 的 masked varlen 元数据改为 host 端构建
- **代码层面**：在 `qwen_image.py` 中，如果已有 `txt_seq_lens`，则直接用它在 CPU 上构建 varlen 元数据（`build_varlen_mask_meta_from_ranges`），避免每步去 GPU 上做 nonzero 操作。
- **新手概念课堂**：Varlen（变长）元数据描述每个序列的起始和结束位置。之前通过 `mask.nonzero()` 在 GPU 上计算，每次都要设备同步，很慢。
- **对你有什么影响**：Qwen 图像模型的每个 denoising 步骤都更快了。

### 14. SWA chunk-cap 测试迁移到注册套件
- **代码层面**：将 `test_schedule_policy.py` 中的 SWA 相关测试迁移到 `test_prefill_adder.py`，并调整了导入。
- **新手概念课堂**：SWA（滑动窗口注意力）只关注最近的 token。chunk-cap 是调度器为避免死锁而设计的"逃生舱"逻辑。
- **对你有什么影响**：纯测试代码调整，无用户可见影响。

### 15. 昇腾 NPU 版本升级推荐
- **代码层面**：文档更新，推荐版本从 `v0.5.13.post1-cann9.0.0-a3` 升级到 `cann9.0.0-a3-v0.5.16`。
- **新手概念课堂**：昇腾（Ascend）是华为的 AI 芯片。CANN 是其软件栈。镜像标签格式从 `v版本-cann版本-型号` 改为 `cann版本-型号-v版本`。
- **对你有什么影响**：使用昇腾 NPU 的用户应该升级到新版本以获取最新修复。

### 16. Z-Image 的 bit-exact 融合 QK-Norm
- **代码层面**：新增 Triton 内核 `_qk_rmsnorm_native_kernel`，精确复现 eager 模式的数值计算顺序（包括 bf16 舍入点），实现 bit-exact 对齐。
- **新手概念课堂**：QK-Norm 是注意力机制中对 Query 和 Key 做归一化的操作。bit-exact 意味着融合内核和原始实现产生完全相同的输出（`torch.equal` 为 True）。
- **对你有什么影响**：Z-Image 模型推理速度提升 6.4%，且输出与之前的实现完全一致。

### 17. 修复 prefill CP 图溢出
- **代码层面**：新增 `required_local_tokens()` 和 `select_replay_bucket()` 方法，在 zigzag 布局下计算实际需要的本地 token 数，并选择最小的满足要求的捕获桶。
- **新手概念课堂**：CP（上下文并行）把序列切分到多张 GPU。Zigzag 是一种负载均衡的切分方式。CUDA Graph 是预分配的，如果实际输入超过捕获时的尺寸就会溢出。
- **对你有什么影响**：长序列 prefill 不再因为图尺寸不足而崩溃。

### 18. FlashInfer 的 DSV4 mHC 融合
- **代码层面**：新增 `SGLANG_OPT_USE_FLASHINFER_MHC` 环境变量（默认关闭），实现了 FlashInfer 的 `mhc_pre_big_fuse` 与 DeepGEMM 的 `tf32_hc_prenorm_gemm` 融合。
- **新手概念课堂**：mHC（multi-head Compression）是 DeepSeek-V4 的注意力压缩机制。融合是指把多个算子合并成一个，减少内存访问和 kernel 启动开销。
- **对你有什么影响**：这是可选优化，默认关闭。有兴趣的用户可以设置环境变量开启，观察性能变化。

### 19. 修复 masked-path 复制的 SP 守卫
- **代码层面**：将 `NotImplementedError` 的条件从"有复制 token"改为"有复制 token 且处于 SP 模式且 SP 大小 > 1"。
- **新手概念课堂**：SP（序列并行）下，序列被切分到多张卡，复制的前缀/后缀会被重复计算导致错误。但单卡时，mask 已经描述了完整序列，复制计数无意义但也不会有问题。
- **对你有什么影响**：单卡运行时不误报错误。

### 20. 修复 DSpark 分数模拟接受
- **代码层面**：将 `_sample_simulated_acc_len` 改名为 `sample_simulated_acc_len`（公开 API），并在 DSpark 验证器中每次调用该函数而不是缓存固定值。
- **新手概念课堂**：DSpark 是推测解码的一种实现。`simulate_acc_len` 是基准测试用的模拟接受长度，之前缓存了固定值导致模拟不准确。
- **对你有什么影响**：使用 DSpark 做基准测试时，模拟结果更准确。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)