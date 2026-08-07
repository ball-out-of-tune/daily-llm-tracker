# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 15:29 UTC | AI 解读: ✅ 含代码解读

## vllm
# vLLM 24小时重要 Commits 解读

## 🔥 重要更新

1. **网络端口分配死锁修复 (Commit 1)** — 修复了 `get_open_port()` 在数据并行保留端口范围内无限循环的严重 bug，影响所有分布式部署的稳定性。
2. **MoRIIO KV传输屏障机制 (Commit 2)** — 修复了 KV 跨节点传输中每层读取完成屏障的时序问题，防止高并发下精度退化。
3. **NixlPush 前缀缓存修复 (Commit 3)** — 修复了分布式前缀缓存命中时块 ID 对齐错误，直接影响推理性能。
4. **NVML 初始化性能修复 (Commit 11)** — 消除了每次设备能力检查都重新初始化 NVML 的性能开销，对每步推理调用该检查的注意力层有显著加速。

---

## 📋 逐条解读

### 1. [Bugfix] Fix get_open_port() livelock on DP-reserved ports

- **代码层面**：修改了 `vllm/utils/network_utils.py` 中 `get_open_port()` 和 `get_open_ports_list()` 函数。修复了当 `VLLM_PORT` 环境变量落在数据并行（DP）保留端口范围内时，函数会无限循环尝试获取端口的问题。新增了 `_call_with_timeout` 测试辅助函数，防止测试挂起。
- **新手概念课堂**：端口就像公寓的门牌号。数据并行（DP）就像多个住户共享一个楼层，系统预先保留了一些门牌号（端口）给 DP 使用。原来的代码在分配门牌号时，如果发现目标门牌号被保留，就会一直尝试下一个，但如果所有候选都被保留，就会陷入死循环（livelock）——就像在走廊里来回跑却永远找不到空房间。
- **对你有什么影响**：如果你使用多 GPU 分布式推理（如张量并行），这个修复防止了服务启动时可能出现的永久卡死。

---

### 2. [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`。将 `_recving_transfers` 的数据结构从 `{"req": [DoneStatus()]}` 改为 `{"req": {"layer0": DoneStatus()}}`，实现了按层跟踪读取完成状态。新增了 `requires_piecewise_for_cudagraph` 方法，并在 READ 模式下检查 CUDA graph 模式，如果使用完整 CUDA graph 则发出警告。
- **新手概念课堂**：KV 传输就像图书馆间的图书调拨。MoRIIO 是跨节点传输 KV 缓存的机制。想象多本书（多层 KV）需要从 A 图书馆调到 B 图书馆，原来的代码只跟踪"这批书是否到齐"，现在改为跟踪"每一本书是否到齐"。CUDA graph 就像预编排的舞台剧，所有动作必须按剧本走，无法中途插入新动作——所以需要 PIECEWISE（分段）模式来允许中途等待。
- **对你有什么影响**：使用 MoRIIO 做 KV 传输的分布式推理在高并发下结果更准确，不会因为读取未完成就继续计算。

---

### 3. [PD][NixlPush][Bugfix] Fix prefix caching

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` 下的 push 连接器。`_apply_prefix_caching` 函数新增了 `local_physical_per_logical` 参数，修复了前缀缓存命中时本地和远程物理块映射不一致的问题。新增了 `TestPushPrefixCaching` 测试类，验证部分前缀命中时只传输尾部未计算的块。
- **新手概念课堂**：前缀缓存就像你写论文时的参考文献。如果两个请求有相同的前缀（比如相同的开头提示词），系统可以复用之前计算过的结果。Push 模式是主动把结果推送给对方。原来的 bug 就像你引用了参考文献的"前半部分"，却把"后半部分"的内容也当作引用贴了上去——现在修复为只推送对方真正缺失的那部分。
- **对你有什么影响**：使用 PD（Prefill-Decode）分离部署时，前缀缓存命中率提高，推理延迟降低，不会因为缓存错位导致结果错误。

---

### 4. [Model] Enable Qwen3.8 for AMD Rocm

- **代码层面**：修改了 `vllm/model_executor/models/qwen3_5.py`。为 `Qwen3_5ForCausalLMBase` 添加了 `SupportsMRoPE` 支持，新增了 `get_mrope_input_positions` 方法，返回一个 3 维的 position tensor（形状为 `(3, seq_len)`），用于多模态旋转位置编码。
- **新手概念课堂**：MRoPE（Multimodal Rotary Position Embedding）就像给每个词分配一个"时间戳"和"空间坐标"。普通 RoPE 只给一维位置（词在句子中的位置），MRoPE 给三维位置（比如文本位置、图像位置、音频位置），让模型理解不同模态的相对位置关系。AMD ROCm 是 AMD GPU 的计算平台，相当于 NVIDIA 的 CUDA。
- **对你有什么影响**：如果你在 AMD GPU 上运行 Qwen3.8 多模态模型，现在可以正确支持多模态输入的位置编码。

---

### 5. [Bugfix] Skip fetching revision for model when model and weights_model are different

- **代码层面**：修改了 `vllm/config/model.py` 中的 `ModelConfig.__post_init__`。新增了 `weights_from_model` 和 `config_from_model` 两个布尔变量，只有当权重来源和配置来源都与模型名一致时才解析 revision。新增了对应测试，验证当 `model_weights` 与 `model` 不同时不会调用 `resolve_revision`。
- **新手概念课堂**：Hugging Face 的 revision 就像 GitHub 的 commit hash，标识模型的具体版本。当你说"我要用 Qwen/Qwen3-0.6B 模型，但权重来自 unsloth/Qwen3-0.6B-GGUF"时，系统不需要去查 Qwen 仓库的最新版本——因为权重不是从那里来的。原来的代码会多此一举地去查询，浪费时间和网络请求。
- **对你有什么影响**：使用不同来源的权重文件（如 GGUF 量化版）时，启动速度更快，不会因为不必要的网络请求而延迟。

---

### 6. Fix ROCm architecture import on non-ROCm platforms

- **代码层面**：修改了 `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` 和 `vllm/model_executor/layers/quantization/mxfp4.py`。将 `from vllm.platforms.rocm import on_gfx1250` 的顶层导入改为条件导入，仅在 `current_platform.is_rocm()` 为真时才导入。`is_gfx1250` 变量默认设为 `False`。
- **新手概念课堂**：`on_gfx1250` 是检查 GPU 是否为 AMD GFX1250 架构的函数。原来的代码在所有平台上都尝试导入这个函数，就像在 Windows 电脑上强制安装 Mac 驱动——会报错。现在改为"只有在 AMD 平台上才检查 AMD 驱动"。
- **对你有什么影响**：在非 AMD GPU（如 NVIDIA、Intel）上使用 MXFP4 量化 MoE 模型时，不会因为导入错误而崩溃。

---

### 7. feat: extended EPLB support for Mistral Large 3

- **代码层面**：新增了 `tests/distributed/test_eplb_quant_scale_consistency.py` 测试文件，验证 EPLB（Expert Parallel Load Balancing）重排后量化相关的缩放因子和 alpha 参数保持一致性。测试覆盖了 NVFP4 量化方法在 EPLB 重排后的正确性。
- **新手概念课堂**：EPLB（专家并行负载均衡）就像餐厅里动态调整服务员的工作量。MoE（Mixture of Experts）模型有多个"专家"网络，EPLB 会根据实际负载动态重排专家在 GPU 间的分布。量化缩放因子就像每个菜品的"盐量"——重排后必须跟着专家一起移动，否则菜就咸淡不均了。
- **对你有什么影响**：使用 Mistral Large 3 配合 NVFP4 量化和 EPLB 时，推理结果更准确，不会因为量化参数错位导致输出异常。

---

### 8. [XPU] quick fix online quantization UT break

- **代码层面**：修改了 `tests/quantization/test_online.py`。将硬编码的 `device="cuda"` 改为 `device=DEVICE`（根据平台动态选择 `xpu:0` 或 `cuda:0`）。修复了 NVFP4 测试在 XPU 平台上的跳过条件逻辑。
- **新手概念课堂**：XPU 是 Intel 的加速器平台（类似 NVIDIA 的 CUDA）。原来的测试代码硬编码使用 CUDA，就像测试说明书只写了"在 Mac 上运行"，现在改为"在 Mac 或 Windows 上运行"。
- **对你有什么影响**：在 Intel XPU 上运行量化测试时不会再报错。

---

### 9. [Misc] Add and enable Triton kernel unit tests on XPU

- **代码层面**：修改了多个测试文件，包括 `tests/kernels/core/test_fused_rms_norm_gated.py`、`tests/kernels/quantization/test_block_int8.py` 等。将硬编码的 CUDA 设备改为根据平台选择，添加了 XPU 平台的跳过条件和设备选择逻辑。
- **新手概念课堂**：Triton 是一种 GPU 编程语言，类似 Python 但能直接控制 GPU 底层。这些测试验证 Triton 编写的内核（kernel）在不同硬件上是否正确。就像同一道菜谱（Triton 代码）要在不同品牌的烤箱（CUDA/XPU）上测试是否都能烤出好蛋糕。
- **对你有什么影响**：在 Intel XPU 上使用 Triton 内核的模型（如 RMSNorm、INT8 量化）现在有测试保障，功能更可靠。

---

### 10. [PD][PushConnector] Record last activity of remotes

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`。新增了 `_engine_last_active` 字典记录每个远端引擎的最后活跃时间，在推送 KV 时刷新对应引擎的活跃时间。新增了 `_eviction_worker` 测试辅助函数，验证过期引擎会被清理。
- **新手概念课堂**：这就像聊天软件的"在线状态"。如果某个远端节点长时间没有通信，系统就认为它已经下线，清理掉它的连接资源。原来的代码只在建立连接时记录时间，导致活跃的推送不会刷新时间，可能误杀还在工作的节点。
- **对你有什么影响**：PD 分离部署中，解码节点（D）扩缩容后，过期的连接会被正确清理，不会泄漏资源。

---

### 11. [Bugfix][Platform] Stop re-initializing NVML on every device-capability check

- **代码层面**：修改了 `vllm/platforms/cuda.py` 中 `has_device_capability` 的实现。移除了多余的 `with_nvml_context` 包装，直接使用缓存的 `get_device_capability` 结果。新增测试验证 NVML 只初始化一次。
- **新手概念课堂**：NVML 是 NVIDIA 的管理库，用来查询 GPU 信息。初始化 NVML 就像打开一个数据库连接——很耗时。原来的代码每次查询 GPU 能力都重新打开连接，就像每次查一个电话号码都要重新拨号而不是用通讯录。这个修复让 `triton_reshape_and_cache_flash` 等每步都调用的函数不再重复初始化。
- **对你有什么影响**：使用 FP8 或 BF16 KV 缓存时，推理速度提升，因为减少了大量重复的 NVML 初始化开销。

---

### 12. [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config

- **代码层面**：修改了 `vllm/model_executor/layers/fused_moe/oracle/int8.py` 中的 `make_int8_moe_quant_config`。将断言改为显式检查，新增 `scales_absent` 变量，当没有激活缩放且 `per_act_token_quant=True` 时，不再错误地构建 W8A16 配置。
- **新手概念课堂**：W8A8 和 W8A16 是量化方案。W8A8 表示权重和激活都用 8 位整数，W8A16 表示权重 8 位但激活 16 位。原来的 bug 就像你要求"所有配料都切成小块"（W8A8），但厨师却把部分配料切成了大块（W8A16），导致口感不一致。
- **对你有什么影响**：使用动态 INT8 量化的 MoE 模型时，配置正确，不会出现性能下降或精度损失。

---

### 13. [Refactor] Remove kernel dead code

- **代码层面**：删除了 `csrc/cpu/cpu_attn_fp8.hpp` 中的 `fp8e5m2_to_float_scalar` 函数，以及 `csrc/libtorch_stable/cache_kernels.cu` 中的 `copy_blocks_kernel` 和 `copy_blocks_mla_kernel` 等未使用的 CUDA 内核。
- **新手概念课堂**：死代码就像房间角落里没人用的旧家具——占地方但没人碰。这些是参考实现或已废弃的内核，删除它们让代码更整洁，编译更快。
- **对你有什么影响**：编译时间略有减少，代码库更干净，没有功能影响。

---

### 14. Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel

- **代码层面**：新增了多个测试配置文件，包括 `DeepSeek-V4-Flash-NVFP4.yaml` 和 `DeepSeek-V4-Pro-NVFP4.yaml`，用于在 AMD MI355 GPU 上测试 DeepSeek-V4 的 NVFP4 量化版本。新增了 `.buildkite/test-amd.yaml` 中的 CI 步骤。
- **新手概念课堂**：NVFP4 是 NVIDIA 的 4 位浮点格式。AMD 的 Quark 是 AMD 的量化工具，可以生成类似格式的模型。"emulation kernel" 是在 AMD 上模拟 NVIDIA 行为的核函数，就像用翻译软件让只会中文的人听懂英文。
- **对你有什么影响**：在 AMD GPU 上可以运行 DeepSeek-V4 的 NVFP4 量化版本，显存占用更低。

---

### 15. [ROCm][CI] Loosen block-FP8 fused MoE test tolerance

- **代码层面**：修改了 `tests/kernels/moe/test_block_fp8.py`。将参考实现中的 `native_per_token_group_quant_fp8` 替换为生产用的 `per_token_group_quant_fp8`，新增 `silu_fp32` 参数区分两种精度路径，并为大 K 形状放宽了测试容差。
- **新手概念课堂**：测试容差就像考试评分标准。如果标准太严格，微小的浮点舍入误差也会导致"不及格"。FP8 是 8 位浮点，精度有限，在计算大量数据时误差会累积。这个修改让参考实现更接近真实生产代码，避免误报。
- **对你有什么影响**：AMD GPU 上运行 FP8 MoE 模型时，CI 测试更稳定，不会因为精度问题导致测试失败。

---

### 16. [Feat][Core] Add disk offloading support

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`。新增 `kv_offload_backend` 配置选项，支持 `"cpu"` 和 `"disk"` 两种后端。新增 `disk_path`、`disk_capacity_bytes`、`disk_buffer_slots`、`use_page_cache` 等磁盘相关配置。
- **新手概念课堂**：KV 缓存就像模型推理时的"工作记忆"。CPU 卸载是把工作记忆从 GPU 搬到内存，磁盘卸载是进一步搬到硬盘。硬盘就像仓库——容量大但速度慢。这个功能让你在 KV 缓存超大时可以用磁盘作为最后的存储层。
- **对你有什么影响**：超长上下文推理时，KV 缓存可以卸载到磁盘，突破内存限制。

---

### 17. [rl] Stateful Trainer Send: NCCL + Sparse NCCL

- **代码层面**：大幅重构了 `examples/rl/rlhf_async_new_apis.py`。将原来的 `NCCLWeightTransferEngine.trainer_init` 和 `trainer_send_weights` 替换为新的 `WeightTransferTrainerFactory.trainer_init` 和 `ModuleSource` API。新增了 `RayVLLMWeightSyncClient` 用于与 Ray 上的 vLLM 实例同步。
- **新手概念课堂**：NCCL 是 NVIDIA 的 GPU 通信库，让多 GPU 之间高效交换数据。在强化学习（RL）中，训练器需要把模型权重发送给推理引擎。这个重构让 API 更简洁，就像把原来需要手动配置的"快递流程"封装成"一键下单"。
- **对你有什么影响**：使用 RL 训练时，权重同步代码更简洁，支持更多传输后端。

---

### 18. [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection

- **代码层面**：新增了 `tests/models/kimi_k3/test_amd_latent_moe_runner.py` 测试文件，验证 ROCm 平台上 `ROCmLatentMoERunner` 的尾部计算与复制的 up-projection 结果一致。测试覆盖了多 TP 大小下的数值精度。
- **新手概念课堂**：Kimi-K3 是 Kimi 的第三代模型，使用了 Latent MoE（潜在专家混合）。"Shard" 是把计算分片到多个 GPU 上。这个测试确保分片后的结果与不分片时一致——就像把一个大蛋糕切成 8 块分给 8 个人，每人吃到的味道必须和整块蛋糕一样。
- **对你有什么影响**：在 AMD GPU 上运行 Kimi-K3 时，多 GPU 分片推理结果更精确。

---

### 19. [Bugfix] Fix Mamba all-mode CPU offload boundary alignment

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` 中的 `resolve_mamba_align_size`。将 Mamba 缓存模式从仅 `"align"` 扩展为 `"align"` 和 `"all"` 两种模式都进行边界对齐。测试新增了 `mamba_cache_mode` 参数化。
- **新手概念课堂**：Mamba 是一种状态空间模型，它的"状态"就像一条河流的水位——每个时间点只有一个状态值。CPU 卸载时，如果提示词恰好在一个块的边界上，系统不能加载那个边界处的缓存状态，因为那个状态已经包含了最后一个 token 的信息，而那个 token 需要重新计算。
- **对你有什么影响**：使用 Mamba 模型配合 CPU 卸载时，即使使用 `"all"` 缓存模式，边界处理也正确，不会出现精度问题。

---

### 20. [Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance

- **代码层面**：修改了 `vllm/v1/worker/gpu/mm/encoder_runner.py`。新增了 `execute_mm_encoder` 方法，让编码器实例只执行编码和发布，不执行 embedding 收集。修改了调度器逻辑，当连接器已有缓存项时跳过编码器输入调度。新增了对应测试。
- **新手概念课堂**：EPD（Encoder-Prefill-Decode）分离部署中，编码器实例专门处理多模态输入（如图像）。原来的代码让编码器也尝试收集 embedding——就像让打字员去装订文件，不仅多余还会出错。现在编码器只做自己该做的事。
- **对你有什么影响**：使用 EPD 分离部署多模态模型时，编码器实例不会崩溃，重复的图片输入不会被重复编码，节省计算资源。

---

## 💡 今日关键词

1. **KV Cache (键值缓存)**：Transformer 模型推理时缓存中间计算结果（Key 和 Value 向量）的技术，避免重复计算，是推理加速的核心机制。

2. **MoE (Mixture of Experts, 专家混合)**：一种模型架构，包含多个"专家"子网络，每次推理只激活部分专家，在保持模型能力的同时大幅降低计算量。

3. **CUDA Graph**：NVIDIA 提供的一种 GPU 执行优化技术，将一系列 GPU 操作预编译成一张"执行图"，减少调度开销，提升推理速度。

## sglang
## 🔥 重要更新

1. **LTX-2 视频生成性能大幅提升（1.56x）**：通过启用 breakable CUDA graph，H200 上端到端生成时间从 10.75 秒降至 6.90 秒，这是视频生成效率的重大突破。
2. **新增 LingBot-Video MoE 30B 模型支持**：为视频生成增加了新的 MoE 架构模型，扩展了模型生态。
3. **修复 Pipeline Parallelism 下的 IndexError**：解决了 Triton 注意力后端在流水线并行场景下访问不存在的 layer 0 缓冲区导致的崩溃，提升了多 GPU 部署的稳定性。
4. **Z-Image 融合 QK-Norm 优化**：通过 Triton 内核实现 bit-exact 的 QK-Norm 融合，在 H200 上端到端性能提升 6.4%，且输出与原始路径完全一致。

---

## 📋 逐条解读

### 1. 移除不再使用的 HiMambaRadixTree
- **代码层面**：删除了 `hi_mamba_radix_cache.py` 文件（约 2178 行），这是一个基于 Mamba 架构的缓存树实现。从导入列表看，它依赖了 `MambaRadixCache`、`LRUList` 等类，但已不再被任何代码引用。
- **新手概念课堂**：Radix Tree（基数树）就像图书馆的索引系统，按前缀共享来组织缓存。想象你要找"苹果"和"苹果汁"两本书，它们共享"苹果"这个前缀，基数树就把公共前缀存一次。HiMambaRadixTree 是专门为 Mamba 模型设计的这种索引，但后来有了更好的替代方案，所以这个"旧索引卡"被扔掉了。
- **对你有什么影响**：无感知，这是内部清理，减少维护成本。

### 2. [diffusion] 为 LTX-2 启用可断点 CUDA 图
- **代码层面**：将 `max_segments` 默认值从 128 提高到 512（因为 LTX-2 的 48 个 block 会产生约 289 个断点）。同时修改了 `denoising.py`，在启用 breakable CUDA graph 时，将 RoPE 坐标构建移到 CUDA 图捕获区域之外（因为 `torch.tensor(list, device=cuda)` 是未锁页的 H2D 拷贝，在 CUDA 图捕获中非法）。
- **新手概念课堂**：CUDA Graph 就像把一系列 GPU 操作"录制成视频"，之后可以反复播放而无需重新编排。但录制时不能有"临时插入"的操作（比如从 CPU 拷贝数据）。Breakable CUDA Graph 允许把长视频切成多个片段，在片段间可以插入其他操作。
- **对你有什么影响**：LTX-2 视频生成速度提升 1.56 倍，生成体验更流畅。

### 3. [diffusion] 使调度器 RPC 截止时间显式化
- **代码层面**：新增 `--scheduler-rpc-timeout` 参数（默认不设置）。在 `http_server.py` 的 shutdown 流程中，先调用 `shutdown_video_jobs()` 取消所有视频生成任务，再取消 broker 任务并等待其完成。`video_api.py` 中新增了 `_VIDEO_JOB_TASKS` 字典来跟踪所有视频任务，shutdown 时统一取消。
- **新手概念课堂**：RPC（远程过程调用）就像你打电话给远处的朋友让他帮你做事。Deadline（截止时间）就是"如果 30 秒内没回应就挂断"。原来这个电话没有超时限制，视频生成可能永远等下去。现在你可以选择设置超时，但默认不设，因为视频生成本来就需要很长时间。
- **对你有什么影响**：服务器关闭时更干净，不会留下挂起的视频任务；新增的 `--scheduler-rpc-timeout` 参数让部署者可以控制请求的最长等待时间。

### 4. 修复 VAE 快速路径测试
- **代码层面**：修改测试文件 `test_autoencoder_kl_fastpath.py`。原来直接操作 `gate.enabled` 属性来切换快速路径，现在改为使用 `use_vae_fast_path(opt, True)` 上下文管理器。同时取消了"gate 未启用"的断言，改为检查包装器类是否已安装。
- **新手概念课堂**：上下文管理器（`with` 语句）就像"临时借东西，用完自动还"。原来测试是手动开/关开关（`gate.enabled = True/False`），容易忘记关；现在用 `with use_vae_fast_path(...)`，进入时自动开，退出时自动关，更安全。
- **对你有什么影响**：无直接用户影响，但测试更可靠，能更好地保证 VAE 快速路径的正确性。

### 5. 修复 Nemotron W4A16 NVFP4 MoE 后端
- **代码层面**：在 `overrides.py` 中新增检测：如果模型有 W4A16_NVFP4 量化的 MoE 层，则强制要求 `--moe-a2a-backend=none` 和 `--moe-runner-backend=marlin`（auto 时自动切换）。在 `marlin_utils_fp4.py` 中，修复了 padding 后的维度计算：`padded_size_k = weight.size(0) * 16`，并在需要时对输入做 padding。
- **新手概念课堂**：MoE（混合专家）就像一家公司有多个专业部门，每个请求只派给最擅长的几个部门处理。W4A16 量化意味着权重用 4-bit 存储（更省内存），但计算时用 16-bit。NVFP4 是 NVIDIA 的 4-bit 浮点格式。这个修复解决的是：当分片大小不对齐时（比如 N=928 需要 padding 到 960），旧代码没有正确计算 padding 后的维度导致崩溃。
- **对你有什么影响**：使用 Nemotron 模型且启用 W4A16 NVFP4 量化时，不再崩溃，且自动选择正确的后端。

### 6. [diffusion] 加速 TP 和 FSDP 检查点加载
- **代码层面**：重构了 `fsdp_load.py`，移除了 `_get_param_for_weight_loading` 辅助函数，新增 `rank_local_checkpoint` 导入。当不使用 FSDP 且无预处理时，走更快的路径：直接加载 safetensors 并广播，避免逐参数处理。
- **新手概念课堂**：FSDP（Fully Sharded Data Parallel）就像把一本大书拆成多份，每个 GPU 只拿一部分。检查点加载就是"把书重新拼回来"。原来的加载方式像逐页检查每页是否完整，现在改为"如果不需要拆分就直接整本复制"，快得多。
- **对你有什么影响**：多 GPU 部署时模型加载更快，启动时间缩短。

### 7. [diffusion] 在分布式初始化前绑定设备
- **代码层面**：将 `current_platform.set_device(device)` 移到 `init_distributed_environment` 之前，这样每个 rank 在分布式初始化前就绑定到正确的加速器。删除了原来在初始化后单独设置 CUDA/NPU 设备的代码。MPS 平台的 `set_device` 是空操作（pass）。
- **新手概念课堂**：分布式训练就像多个工人协作，每个工人需要先知道自己在哪个工位（设备）工作。原来先开会（初始化分布式环境）再分配工位，可能导致混乱；现在先分配工位再开会，更合理。
- **对你有什么影响**：多 GPU/多 NPU 部署更稳定，避免设备绑定错误。

### 8. [diffusion] 支持 TP 下的可断点 CUDA 图
- **代码层面**：新增 `_tp_graph_capture` 上下文管理器，在 CUDA 图捕获时进入 TP 组的 `graph_capture` 上下文。修改了 `group_coordinator.py`，确保 custom all-reduce 在 CUDA 图捕获时使用图路径（而非 eager 路径），避免重放时因未映射的 peer IPC 地址而崩溃。
- **新手概念课堂**：TP（Tensor Parallelism）就像把一个大矩阵运算拆成多块，每个 GPU 算一块，最后合并。CUDA 图录制时，TP 需要协调多个 GPU 的通信。原来录制时用了错误的通信路径（eager 而非 graph），导致"重播"时找不到对方地址而崩溃。
- **对你有什么影响**：TP 部署下也能使用 breakable CUDA graph 加速，性能提升。

### 9. 修复 Pipeline Parallelism 下的 Triton 后端 IndexError
- **代码层面**：在 `triton_backend.py` 和 `memory_pool.py` 中，将 `get_value_buffer(0)` 改为 `get_value_buffer(start_layer)`。在 PP 场景下，每个 stage 只持有部分 layer 的 KV 缓冲区，layer 0 可能不在当前 stage 中。
- **新手概念课堂**：Pipeline Parallelism（流水线并行）就像工厂流水线，每个工位（GPU）只负责一部分工序（layer）。原来代码总是问"第 0 道工序的产品在哪"，但在流水线上第 0 道工序可能在其他工位，导致找不到（IndexError）。现在改为问"我这个工位负责的第一道工序在哪"。
- **对你有什么影响**：使用 Pipeline Parallelism 部署时不再崩溃，多 GPU 扩展更可靠。

### 10. [diffusion] 健康检查端点分离
- **代码层面**：新增 `/liveness` 端点（始终返回 200 表示进程存活）。`/health` 和 `/health_generate` 在 warmup 完成前返回 503，完成后返回 200。如果 warmup 失败，服务器直接终止而不是报告就绪。更新了文档和 K8s 部署示例。
- **新手概念课堂**：K8s 探针就像给服务器做体检。Liveness 探针检查"人还活着吗"（进程在跑），Readiness 探针检查"能干活了吗"（模型加载完、编译完成）。原来只有一个 `/health`，K8s 无法区分"还在启动"和"已经死了"，现在分开后更精确。
- **对你有什么影响**：K8s 部署更稳定，不会在 warmup 期间被误杀，也不会过早地路由流量到未就绪的实例。

### 11. [diffusion] 支持 LingBot-Video MoE 30B T2V 模型
- **代码层面**：新增 `lingbot_video_moe.py` 配置文件，定义 `LingBotVideoMoEArchConfig`，包含 MoE 参数（128 个专家、每 token 选 8 个专家）、patch size (1,2,2)、48 层深度等。在 `dits/__init__.py` 中注册该配置。
- **新手概念课堂**：MoE（混合专家）模型就像一家咨询公司有 128 个专家，每个问题只咨询最擅长的 8 个专家（`num_experts_per_tok=8`），既保证质量又节省计算。T2V 表示 Text-to-Video（文本生成视频）。
- **对你有什么影响**：可以使用 LingBot-Video 30B MoE 模型进行文本生成视频。

### 12. [diffusion] 将 Ring Admission 作为后端能力
- **代码层面**：在 `AttentionBackend` 基类中新增 `supports_ring_rotation()` 方法（默认返回 False）。FlashAttention 和 SageAttention 后端重写为返回 True。在 `layer.py` 中，只有支持 ring rotation 的后端才允许启用 ring attention。
- **新手概念课堂**：Ring Attention（环状注意力）就像几个人围成一圈传纸条，每个人只和左右邻居通信，最终所有人都获得完整信息。但某些注意力内核（kernel）不支持这种"环形传递"的中间结果合并，所以需要声明"我能支持环形传递"。
- **对你有什么影响**：使用不支持 ring attention 的后端时会自动禁用该功能，避免静默错误。

### 13. [diffusion] 在主机端构建 Qwen 的 Masked Varlen 元数据
- **代码层面**：在 `qwen_image.py` 中，当 `txt_seq_lens` 可用时，使用 `build_varlen_mask_meta_from_ranges` 在主机端直接构建 varlen 元数据，避免每次 denoising step 都调用 GPU 上的 `nonzero`（需要设备同步）。新增了测试验证主机端构建与 mask-based 构建结果一致。
- **新手概念课堂**：Varlen（变长）元数据描述"每个序列从哪里开始、到哪里结束"。原来每次去噪都要在 GPU 上数一下"哪些位置是有效 token"（nonzero），这需要 GPU 和 CPU 同步，很慢。现在直接在 CPU 上算好，省去了 GPU 同步。
- **对你有什么影响**：Qwen 图像生成速度提升，因为减少了 GPU 同步等待。

### 14. 将 SWA chunk-cap hatch 测试移入注册套件
- **代码层面**：从 `test/manual/test_schedule_policy.py` 中删除了 `TestSwaChunkCapHatch` 测试类（约 50 行），在 `test/registered/unit/managers/test_prefill_adder.py` 中新增了对应测试（约 56 行）。测试内容不变，只是移动了位置。
- **新手概念课堂**：测试套件分类就像把玩具分类放好：手动测试（manual）需要人工运行，注册测试（registered）会自动在 CI 中运行。把重要的 SWA 测试移到注册套件，意味着每次代码提交都会自动检查这个功能。
- **对你有什么影响**：SWA（滑动窗口注意力）的调度策略有了更可靠的自动测试保障。

### 15. [NPU] 升级 Ascend NPU 推荐版本
- **代码层面**：将文档中推荐的 SGLang 镜像从 `v0.5.13.post1-cann9.0.0-a3` 升级到 `cann9.0.0-a3-v0.5.16`（A3 和 A2 平台都更新）。
- **新手概念课堂**：Docker 镜像版本号就像软件版本号。`v0.5.13.post1` 是旧版本，`v0.5.16` 是新版本。CANN 是华为昇腾的软件栈，类似 CUDA 对 NVIDIA 的作用。
- **对你有什么影响**：使用昇腾 NPU 的用户应该升级到新镜像，获得 bug 修复和新功能。

### 16. [diffusion] Z-Image bit-exact 融合 QK-Norm
- **代码层面**：新增 Triton 内核 `_qk_rmsnorm_native_kernel`，通过精确模拟 aten 的 reduce 顺序（先按 8 元素组串行累加，再做 shfl-down butterfly 合并），实现与 eager 路径 bit-exact 的 RMSNorm。注释中详细描述了如何匹配 aten 的向量化加载和归约顺序。
- **新手概念课堂**：RMSNorm 就像对一组数据做"归一化"：先算平方和，再开根号，最后除以根号值。bit-exact 意味着融合后的结果和原来的每一步计算都完全一致（二进制级别）。这很难，因为浮点运算的顺序会影响结果。
- **对你有什么影响**：Z-Image 模型在 H200 Turbo 上 1024px 端到端性能提升 6.4%，且输出与原始路径完全一致。

### 17. 修复 Prefill CP 图溢出（更大的 bucket 搜索）
- **代码层面**：在 `bcg.py` 中新增 `required_local_tokens()` 方法（计算 zigzag 布局下每个 rank 需要的对齐后的 token 数）和 `select_replay_bucket()` 方法（选择最小的、能容纳所需 local tokens 的捕获 bucket）。修改了 bucket 选择逻辑，考虑 CP 布局的实际需求。
- **新手概念课堂**：CP（Context Parallelism）就像把一篇文章分成多段，每段给不同的人看。Zigzag 布局是一种特定的分段方式（来回折叠）。Bucket 是预先录制好的 CUDA 图，按 token 数分档。原来选 bucket 只看总 token 数，现在还要看每个 rank 实际分到的 token 数是否超出 bucket 容量。
- **对你有什么影响**：使用 Prefill CP + CUDA 图时，不再因 bucket 容量不足而溢出崩溃。

### 18. 为 DeepSeek-V4 添加 FlashInfer MHC 融合
- **代码层面**：新增 `SGLANG_OPT_USE_FLASHINFER_MHC` 环境变量（默认 False）。新增 `_flashinfer_hc_pre` 函数，使用 FlashInfer 的 `mhc_pre_big_fuse` 实现 MHC pre 融合，并自动选择 split-K 数量（1/2/4/8/16）。
- **新手概念课堂**：MHC（Multi-Head Compression）是 DeepSeek 模型中的多头压缩机制，用于减少 KV cache 大小。FlashInfer 是一个高性能注意力库。这个 PR 提供了另一种实现选择，但默认关闭，需要用户显式开启。
- **对你有什么影响**：使用 DeepSeek-V4 且想尝试 FlashInfer 的 MHC 融合时，可以设置 `SGLANG_OPT_USE_FLASHINFER_MHC=1` 开启。

### 19. [diffusion] 将 masked-path 复制保护限定到 SP 运行
- **代码层面**：修改 `layer.py` 中 USPAttention 的 masked path 保护逻辑。原来只要检测到 replicated prefix/suffix 就抛 `NotImplementedError`，现在增加了条件：仅在 SP world size > 1 且未跳过 SP 时才拒绝。单 rank 时允许调用。
- **新手概念课堂**：SP（Sequence Parallelism）下，序列被切成多段分给不同 GPU。如果某段 token 被复制到多个 rank（replicated），每个 rank 都会处理它，导致结果重复计算而错误。但在单 GPU 下，mask 描述的就是完整序列，复制计数没有意义，所以不需要拒绝。
- **对你有什么影响**：单 GPU 下使用 masked attention 且带有 replicated 参数时不再报错。

### 20. 修复 DSpark 模拟接受中的小数问题
- **代码层面**：将 `_sample_simulated_acc_len` 重命名为 `sample_simulated_acc_len`（公开 API）。在 `dspark_verify.py` 中，原来用 `round(min(max(self._simulate_acc_len - 1.0, 0.0), float(self.gamma)))` 直接取整，现在改为调用 `sample_simulated_acc_len`，支持小数（如 3.5）的随机采样。
- **新手概念课堂**：DSpark 是推测解码（Speculative Decoding）的一种实现。模拟接受长度（simulate_acc_len）是测试用的参数，模拟"模型预测被接受的长度"。原来如果设置 3.5，会被四舍五入成 3 或 4（固定值）；现在 3.5 意味着 50% 概率取 3，50% 概率取 4，更真实。
- **对你有什么影响**：使用 DSpark 且设置小数模拟接受长度时，行为更符合预期。

---

## 💡 今日关键词

1. **CUDA Graph**：把 GPU 上的一系列操作"录制成视频"，之后可以反复"播放"而无需重新编排，大幅减少内核启动开销。就像把做饭的每个步骤录下来，之后一键播放，不用每次重新想下一步做什么。

2. **MoE（Mixture of Experts）**：混合专家架构，模型包含多个"专家"子网络，每个输入只激活最相关的少数几个专家。就像一家公司有 128 个专业顾问，每个问题只咨询最擅长的 8 位，既保证专业度又节省成本。

3. **PP/TP/SP（Pipeline/Tensor/Sequence Parallelism）**：三种多 GPU 并行策略。PP 像工厂流水线（每个 GPU 做一道工序），TP 像团队分工（每个人算矩阵的一部分），SP 像分章节阅读（每个人读文章的一段）。它们让大模型能够跨多 GPU 运行。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)