# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 15:39 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词 1：AMD ROCm 适配加速**
  - **一句话通俗解释**：让 vLLM 在 AMD 显卡（如 MI355、gfx950）上也能高效运行，而不是只能用在 NVIDIA 显卡上。
  - **为什么社区在关注**：今天有 5+ 个 commit 涉及 AMD 平台（Qwen3.8 支持、NVFP4 量化、MoE 优化、CI 测试），说明 AMD 在 AI 推理市场正在快速崛起，vLLM 社区正在投入大量资源适配。
  - **对新手意味着什么**：如果你只有 AMD 显卡，或者想学习异构计算，现在是最好的入场时机。AMD 相关的 bug 和新特性非常多，贡献门槛相对较低。

- **关键词 2：KV Cache 传输与卸载**
  - **一句话通俗解释**：KV Cache 是 LLM 推理时保存中间状态的内存区域，多个 GPU 之间传输、以及卸载到 CPU/磁盘来省显存。
  - **为什么社区在关注**：今天有 5 个 commit 涉及 KV 传输（MoRIIO、NixlPush、CPU Offload、磁盘 Offload），说明随着模型越来越大、推理集群越来越复杂，KV 管理成为性能瓶颈的核心。
  - **对新手意味着什么**：KV Cache 是 vLLM 最核心的优化领域之一，理解它等于理解了 vLLM 的"血管系统"。学习成本高但回报极大。

- **关键词 3：量化与精度权衡**
  - **一句话通俗解释**：把模型权重从 16 位/32 位压缩到 8 位/4 位来省显存加快速度，但会损失一点精度。
  - **为什么社区在关注**：多个 commit 涉及 FP8、INT8、NVFP4 量化的 bug 修复和测试优化，说明量化是生产环境中的刚需，但精度问题仍然困扰着开发者。
  - **对新手意味着什么**：量化是 AI 推理的"必修课"，每个框架都在做。学会量化 = 掌握生产环境部署的核心技能。

- **关键词 4：前缀缓存（Prefix Caching）**
  - **一句话通俗解释**：如果多个请求有相同的开头（比如相同的 system prompt），只计算一次，后面直接复用。
  - **为什么社区在关注**：多个 commit 修复前缀缓存在分布式场景下的 bug（NixlPush、Mamba），说明前缀缓存在多 GPU 环境下容易出错，但收益巨大。
  - **对新手意味着什么**：前缀缓存是 vLLM 高吞吐的秘密武器，理解它 = 理解为什么 vLLM 比裸 PyTorch 快那么多。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：端口分配中的"活锁"（Livelock）
- **一句话解释**：程序在循环里反复尝试同一个操作但永远无法成功，CPU 空转但任务一直不完成——就像一个人反复按电梯按钮但电梯永远不来。
- **为什么重要**：`get_open_port()` 是 vLLM 分布式部署时分配通信端口的关键函数。如果它活锁，整个集群启动会卡死。这个 bug 在 DP（数据并行）预留端口范围内触发，影响多卡部署。
- **动手试试**：
  1. 设置环境变量 `VLLM_DP_MASTER_PORT=5680` 和 `VLLM_PORT=5682`（5682 在 [5680, 5690) 区间内）
  2. 调用 `vllm.utils.network_utils.get_open_port()`，观察修复前它会无限循环
  3. 应用修复后，它应该返回一个不在 5680-5690 范围内的端口
- **预期结果**：修复前程序卡死（可以用 `timeout 5 python -c "..."` 验证），修复后立刻返回。你掌握了"活锁"和"端口预留"的概念。

### 知识点 2：前缀缓存对齐（Prefix Cache Alignment）
- **一句话解释**：当多个 GPU 共享 KV Cache 时，缓存命中的位置必须对齐到块边界，否则缓存数据错位导致推理结果错误。
- **为什么重要**：在分布式推理中，前缀缓存不是简单的"命中就复用"，还要考虑块对齐、物理/逻辑块映射等问题。今天的 NixlPush 和 Mamba 修复都是围绕这个。
- **动手试试**：
  1. 用 `state-spaces/mamba-1.4b-hf` 模型，设置 `mamba_cache_mode="align"` 和 `"all"` 两种模式
  2. 构造一个 prompt 长度恰好等于块大小整数倍的请求（如 block_size=16，prompt 长度 = 16 或 32）
  3. 对比两种模式下输出是否一致
- **预期结果**：修复前 `"all"` 模式在边界处会输出错误结果（因为加载了包含最后一个 token 的缓存状态），修复后两种模式输出一致。你理解了"边界对齐"在缓存系统中的重要性。

---

## 🔥 重要更新

1. **`[Bugfix] Fix get_open_port() livelock on DP-reserved ports`** — 修复了分布式部署时端口分配可能死循环的问题，影响所有使用 DP 并行的大规模部署。
2. **`[PD][NixlPush][Bugfix] Fix prefix caching`** — 修复了 Push 模式下前缀缓存的块对齐问题，这是分布式 KV 传输中最难调试的 bug 类型。
3. **`[Model] Enable Qwen3.8 for AMD Rocm`** — Qwen3.8 正式支持 AMD 平台，AMD 用户终于可以跑最新的 Qwen 模型。
4. **`[Bugfix][Platform] Stop re-initializing NVML on every device-capability check`** — 性能修复：每次检查 GPU 能力不再重新初始化 NVML，减少推理时的额外开销。

---

## 📋 逐条解读

### 1. [Bugfix] Fix get_open_port() livelock on DP-reserved ports
- **代码层面**：修改了 `vllm/utils/network_utils.py` 中的 `get_open_port()` 函数。修复前，当 `VLLM_PORT` 恰好落在 DP 预留的端口范围内时，函数会无限循环尝试获取同一批被预留的端口。修复后，函数会跳过预留范围。
- **新手概念课堂**：想象你要找一个空房间（端口），但有一排房间（DP 端口范围）被标记为"已预留"。修复前的代码只会在这排预留房间中反复敲门（循环），永远找不到空房。修复后的代码会跳过这排房间，去别处找。
- **对你有什么影响**：如果你用多卡 DP 并行部署，之前可能遇到启动卡死的问题。修复后启动会顺畅很多。

### 2. [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier
- **代码层面**：修改了 `moriio_connector.py`，在 READ 模式下增加了一个"每层读取完成屏障"（per-layer barrier），确保每一层的 KV 数据都读取完成后再继续。同时增加了 CUDA Graph 模式兼容性警告。
- **新手概念课堂**：想象你在搬家具（KV 数据），每搬完一层楼（layer）需要确认所有箱子都到了才能继续搬下一层。之前的代码是"搬完所有层再确认"，容易出错。现在改成"每层都确认"。
- **对你有什么影响**：使用 MoRIIO 做 KV 传输的用户在高并发下精度更稳定，不会出现数据未就绪就继续计算的问题。

### 3. [PD][NixlPush][Bugfix] Fix prefix caching
- **代码层面**：修改了 `_apply_prefix_caching` 函数签名，增加了 `local_physical_per_logical` 参数，修复了 Push 模式下前缀缓存只保留"未计算块"而不是"全部块"的问题。
- **新手概念课堂**：前缀缓存就像抄作业——如果两个同学的前半部分作业一样，第二个同学只需要写后半部分。但这个 bug 让第二个同学把前半部分也重写了（或者写错位置）。修复后，Push 模式下只传输需要计算的后半部分。
- **对你有什么影响**：使用 NixlPush 做 PD 分离部署的用户，前缀缓存命中率提升，推理速度更快，显存占用更少。

### 4. [Model] Enable Qwen3.8 for AMD Rocm
- **代码层面**：在 `qwen3_5.py` 中为 Qwen3.8 模型添加了 `SupportsMRoPE` 支持，并实现了 `get_mrope_input_positions` 方法（多模态旋转位置编码的输入位置计算）。
- **新手概念课堂**：MRoPE（Multimodal Rotary Position Embedding）是 Qwen3.8 用来同时处理文本、图片、视频位置信息的技术。AMD 平台上之前不支持这个特性，现在补上了。
- **对你有什么影响**：AMD 显卡用户现在可以跑 Qwen3.8 多模态模型了。

### 5. [Bugfix] Skip fetching revision for model when model and weights_model are different
- **代码层面**：修改了 `vllm/config/model.py`，当 `model_weights` 指向的仓库与 `model` 不同时，不再尝试解析 model 的 revision（版本号）。因为 weights 来自不同仓库，model 的 revision 没有意义。
- **新手概念课堂**：`model` 是模型架构的仓库，`model_weights` 是权重的仓库。比如架构来自 `Qwen/Qwen3-0.6B`，但权重来自 `unsloth/Qwen3-0.6B-GGUF`。之前代码会去查 `Qwen/Qwen3-0.6B` 的最新版本号，但权重根本不在那里，查了也白查。
- **对你有什么影响**：使用 GGUF 或其他第三方权重格式的用户，启动速度会更快（少了一次网络请求）。

### 6. Fix ROCm architecture import on non-ROCm platforms
- **代码层面**：修改了 `mxfp4.py` 和 `oracle/mxfp4.py`，将 `from vllm.platforms.rocm import on_gfx1250` 改为条件导入——只有当前平台是 ROCm 时才导入。
- **新手概念课堂**：`on_gfx1250()` 是检查 GPU 是否是 AMD 的 gfx1250 架构。之前不管什么平台都尝试导入这个函数，在非 AMD 平台上会报错。现在改成"是 AMD 才检查"。
- **对你有什么影响**：NVIDIA 用户不会再遇到奇怪的导入错误，MXFP4 量化在非 AMD 平台更稳定。

### 7. feat: extended EPLB support for Mistral Large 3 and additional MoE backends
- **代码层面**：新增了 `test_eplb_quant_scale_consistency.py` 测试文件，验证 EPLB（Expert Parallel Load Balancing，专家并行负载均衡）重排后量化参数的一致性。
- **新手概念课堂**：EPLB 是 MoE（混合专家）模型中的负载均衡技术——把计算量大的专家（expert）分配到更多 GPU 上。重排（rearrange）后，每个专家对应的量化缩放因子也要跟着移动，否则数值就乱了。
- **对你有什么影响**：使用 Mistral Large 3 或其他 MoE 模型的用户，EPLB 和量化可以同时使用，不会出现精度问题。

### 8. [XPU] quick fix online quantization UT break
- **代码层面**：修改了 `tests/quantization/test_online.py`，将硬编码的 `device="cuda"` 改为 `device=DEVICE`（根据平台动态选择），并修复了 NVFP4 在 XPU 上的跳过条件。
- **新手概念课堂**：测试代码之前写死了"只能在 CUDA 上跑"，现在改成"根据平台自动选择设备"。就像把"只在北京开的分店"改成"全国各地都有分店"。
- **对你有什么影响**：Intel XPU 用户可以跑在线量化测试了。

### 9. [Misc] Add and enable Triton kernel unit tests on XPU
- **代码层面**：修改了多个内核测试文件（`test_fused_rms_norm_gated.py`、`test_block_int8.py`、`test_int8_kernel.py`），将设备从 `cuda:0` 改为动态选择，并添加了 XPU 的跳过/启用逻辑。
- **新手概念课堂**：Triton 是一个 GPU 编程语言，可以同时编译到 CUDA 和 XPU。之前测试只验证 CUDA 路径，现在 XPU 路径也被覆盖了。
- **对你有什么影响**：Intel XPU 用户对 Triton 内核的稳定性更有信心。

### 10. [PD][PushConnector] Record last activity of remotes to allow clean up of stale ones
- **代码层面**：在 `base_worker.py` 中增加了 `_engine_last_active` 字典记录每个远端引擎的最后活跃时间，并在 `_ensure_handshake` 中增加清理逻辑，删除超过 TTL 的过期引擎。
- **新手概念课堂**：想象一个会议室（远端引擎），如果没人用超过 30 分钟，就自动释放。之前没有这个机制，导致 D 引擎扩容/缩容后，旧引擎的连接残留，浪费资源。
- **对你有什么影响**：PD 分离部署中，D 引擎动态扩缩容时不会残留僵尸连接，资源利用更高效。

### 11. [Bugfix][Platform] Stop re-initializing NVML on every device-capability check
- **代码层面**：修复了 `has_device_capability` 方法——之前每次调用都重新初始化 NVML（NVIDIA 管理库），导致大量不必要的开销。修复后只初始化一次，后续直接读缓存。
- **新手概念课堂**：NVML 是 NVIDIA 的"体检中心"，每次检查 GPU 能力都要"挂号"（初始化）和"缴费"（关闭）。之前每个 attention 层每步都挂号一次，效率极低。现在只挂号一次，后面直接看报告。
- **对你有什么影响**：推理性能提升，特别是 FP8/BF16 KV Cache 下，之前每个 step 都有大量 NVML 开销。

### 12. [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16
- **代码层面**：修改了 `oracle/int8.py` 中的 `make_int8_moe_quant_config`，修复了动态 INT8 量化下，当 `per_act_token_quant=True` 且没有 scale 时，错误地构建为 W8A16 配置的问题。
- **新手概念课堂**：W8A8 表示权重 8 位、激活 8 位；W8A16 表示权重 8 位、激活 16 位。动态量化下，激活的 scale 是运行时计算的，所以没有预定义的 scale。之前的代码没识别这种情况，错误地走了 W8A16 路径。
- **对你有什么影响**：使用动态 INT8 量化的 MoE 模型，显存占用和速度都更优。

### 13. [Refactor] Remove kernel dead code
- **代码层面**：删除了 `cpu_attn_fp8.hpp` 中的 `fp8e5m2_to_float_scalar` 函数和 `cache_kernels.cu` 中的 `copy_blocks_kernel` 等未使用的内核代码。
- **新手概念课堂**："死代码"是写了但没人调用的代码。删除它们让代码库更干净，编译更快，也减少维护负担。
- **对你有什么影响**：编译时间略微缩短，代码库更易维护。

### 14. Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel
- **代码层面**：新增了 DeepSeek-V4 在 AMD 平台的 NVFP4 量化支持，包括新的测试配置（`DeepSeek-V4-Flash-NVFP4.yaml`）和 CI 测试任务。
- **新手概念课堂**：NVFP4 是 NVIDIA 的 4 位浮点格式，AMD 通过"模拟内核"（emulation kernel）来兼容它。就像用软件模拟器在 AMD 上跑只支持 NVIDIA 的游戏。
- **对你有什么影响**：AMD 用户终于可以跑 DeepSeek-V4 的量化版本了。

### 15. [ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes
- **代码层面**：修改了 `test_block_fp8.py`，增加了 `silu_fp32` 参数区分两种不同精度的参考实现，并放宽了大 K 形状的测试容差。
- **新手概念课堂**：FP8 精度有限，大矩阵乘法时误差会累积。测试之前用统一的容差，导致大 K 时失败。现在区分了两种计算路径（一种在 fp32 下计算 SiLU+量化，一种在 bf16 下），各自匹配容差。
- **对你有什么影响**：AMD ROCm 平台的 CI 测试更稳定，不会因为精度问题误报失败。

### 16. [Feat][Core] Add disk offloading support to SimpleCPUOffloadConnector
- **代码层面**：修改了 `simple_cpu_offload_connector.py`，新增 `kv_offload_backend` 配置，支持 `"cpu"` 和 `"disk"` 两种后端。Disk 模式将 KV Cache 卸载到磁盘（通过 `disk_path` 指定路径），并支持 `use_page_cache` 选项。
- **新手概念课堂**：之前 KV Cache 只能卸载到 CPU 内存（RAM）。现在可以卸载到磁盘（SSD），虽然慢但容量更大。就像把不常用的东西从书桌（RAM）搬到仓库（磁盘），书桌腾出空间放常用的。
- **对你有什么影响**：显存极小的用户可以把 KV Cache 卸载到 SSD，虽然速度慢但至少能跑起来。

### 17. [rl] Stateful Trainer Send: NCCL + Sparse NCCL
- **代码层面**：重构了 RL（强化学习）训练器的权重传输 API，新增 `WeightTransferTrainerFactory`、`RayVLLMWeightSyncClient` 等抽象，支持 NCCL 和稀疏 NCCL 两种传输后端。
- **新手概念课堂**：RL 训练中，训练器（trainer）需要频繁把更新的权重发送给推理引擎（vLLM）。之前的代码是"手动档"（直接操作 NCCL 组），现在改成"自动档"（工厂模式 + 客户端抽象）。
- **对你有什么影响**：RL 训练和推理的权重同步更稳定，API 更简洁。

### 18. [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path
- **代码层面**：新增 `ROCmLatentMoERunner` 和 `KimiRoutedOutputTransform`，在 AMD 平台实现 Kimi-K3 的分片潜在 MoE 上投影（up-projection）优化。
- **新手概念课堂**：MoE 模型的"上投影"是把专家输出的低维向量映射回高维。AMD 平台上之前是复制（replicate）每个专家的上投影，现在改成"分片"（shard）——每个 GPU 只算一部分，然后合并。
- **对你有什么影响**：AMD 用户跑 Kimi-K3 时显存占用降低，速度提升。

### 19. [Bugfix] Fix Mamba all-mode CPU offload boundary alignment
- **代码层面**：修改了 `offloading/scheduler.py` 中的 `resolve_mamba_align_size`，将 Mamba 的边界对齐从仅 `"align"` 模式扩展到 `"align"` 和 `"all"` 两种模式。
- **新手概念课堂**：Mamba 是状态空间模型，它的"缓存"是一个点状态（point state），不像 Transformer 那样有完整的 token 序列。当 prompt 长度恰好等于块边界时，缓存中已经包含了最后一个 token 的状态，不能再加载它（否则重复计算）。
- **对你有什么影响**：使用 Mamba 模型 + CPU offload 的用户，在边界位置不会出现精度问题。

### 20. [Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance
- **代码层面**：修改了 `encoder_runner.py` 和调度器逻辑，当 EPD（Encoder-Prefill-Decode）架构中 encoder 实例已经通过 connector 获取到多模态嵌入时，跳过 gather 步骤（不再重新编码）。
- **新手概念课堂**：EPD 架构中，encoder 实例专门处理视觉/音频输入，生成嵌入向量。之前即使 connector 里已经有缓存，encoder 还是会重新编码一遍（浪费算力），而且可能因为找不到本地缓存而崩溃。
- **对你有什么影响**：EPD 部署中，多模态请求的重复图像/视频不会重复编码，速度更快，也不会崩溃。

## sglang
## 💡 今日关键词

- **CUDA Graph 与性能优化**：通过将 GPU 操作预编译为静态图，大幅减少内核启动开销。社区在持续优化 SGLang Diffusion 的生成性能，如 LTX-2 模型通过可中断 CUDA Graph 实现了 1.56 倍加速，Z-Image 的融合 QK-Norm 也带来了 6.4% 的端到端提升。这说明在 LLM 推理之外，多模态生成（尤其是视频生成）的性能优化已成为重要战场。对新手而言，理解 CUDA Graph 是深入 GPU 高性能计算的关键一步，也是未来优化工作的常见切入点。

- **Pipeline Parallelism (PP) 与分布式推理**：将模型切分到多个 GPU 上，每个 GPU 负责一部分层。多个 commit 都在修复 PP 场景下的 bug（如第 9 个 commit 修复了 PP 下 Triton attention 的 IndexError），说明 PP 作为大规模模型推理的核心技术，其工程成熟度仍在提升。对新手来说，理解 PP 是学习大规模分布式推理的必经之路。

- **Diffusion 模型工程化**：SGLang 正在将扩散模型（图像/视频生成）从研究原型推向生产级服务。今天有大量 commit 涉及 Diffusion 的模型支持（LingBot-Video MoE）、健康检查、RPC 超时、Checkpoint 加载优化等。这说明多模态生成模型的部署和工程化正在成为热点，对新手意味着学习 Diffusion 模型的服务化部署是一个新兴且有前景的方向。

- **量化与 MoE 优化**：低精度量化（如 NVFP4）和混合专家模型（MoE）是提升推理效率的关键。第 5 个 commit 修复了 Nemotron W4A16 NVFP4 MoE 后端的 bug，第 11 个 commit 为 LingBot-Video MoE 模型提供了支持。社区正在为更多模型和硬件平台适配这些技术，说明高效推理是持续的核心诉求。

- **健康检查与运维**：第 10 个 commit 引入了 `/liveness` 和 `/health` 分离的端点，第 3 个 commit 增加了 RPC 超时控制。这表明 SGLang 正在向生产级部署迈进，关注服务可用性和可运维性。对新手而言，学习这些运维实践有助于理解生产环境中的服务部署要求。

## 🧪 重要知识点 & 动手实验

### 知识点 1：CUDA Graph 捕获与重放

- **一句话解释**：CUDA Graph 将一系列 GPU 内核启动预先录制为一张图，然后在推理时一次性重放，从而消除内核启动的开销。

- **为什么重要**：在深度学习推理中，GPU 内核启动的开销（通常几微秒）在大量小操作时会被放大。CUDA Graph 通过将数百个内核启动合并为一次图启动，可以显著降低延迟，尤其适合 Diffusion 这类需要多步迭代的模型。

- **动手试试**：
  1. 安装 PyTorch 和 CUDA 环境，创建一个简单的模型（如一个包含多个线性层的 MLP）。
  2. 使用 `torch.cuda.graph` 上下文管理器捕获模型的推理过程。
  3. 对比普通推理和 CUDA Graph 推理的耗时（使用 `torch.cuda.Event` 计时）。
  4. 尝试修改输入张量的形状或值，观察 CUDA Graph 重放时是否出错。

- **预期结果**：你会看到 CUDA Graph 推理的耗时显著低于普通推理（尤其是当模型包含大量小操作时）。如果修改了输入形状，你可能会遇到错误，这说明了 CUDA Graph 对输入形状的静态要求。

### 知识点 2：Pipeline Parallelism (PP)

- **一句话解释**：将一个大模型按层切分成多个阶段，每个 GPU 负责一个阶段，数据像流水线一样依次经过各个阶段。

- **为什么重要**：当模型过大无法放入单张 GPU 时，PP 是常用的分布式训练/推理策略。它允许更大的模型规模，但引入了跨 GPU 的通信开销和负载均衡问题。

- **动手试试**：
  1. 使用 PyTorch 的 `torch.distributed.pipeline.sync.Pipe` 模块，将一个包含 4 个线性层的模型切分为 2 个阶段，分布在 2 个 GPU 上（如果没有多 GPU，可以用 CPU 模拟）。
  2. 输入一批数据，观察模型输出是否正确。
  3. 尝试将模型的 `start_layer` 设置为非 0 的值（模拟 PP 中某个阶段只负责部分层），复现第 9 个 commit 中修复的 IndexError 问题。
  4. 修改代码，使用 `start_layer` 而不是硬编码的 0 来访问 KV 缓存，观察问题是否解决。

- **预期结果**：你会看到在 PP 场景下，`get_value_buffer(0)` 会抛出 IndexError，因为第 0 层不在当前阶段的 KV 缓存中。改用 `get_value_buffer(start_layer)` 后问题解决。

## 🔥 重要更新

1. **LTX-2 可中断 CUDA Graph 加速（Commit 2）**：将 H200 上两阶段端到端推理从 10.75 秒降至 6.90 秒，1.56 倍加速，是 Diffusion 性能优化的重要突破。

2. **Nemotron W4A16 NVFP4 MoE 后端修复（Commit 5）**：修复了量化 MoE 层在特定配置下的错误，并强制使用 Marlin 后端，对使用 NVFP4 量化模型的用户至关重要。

3. **Pipeline Parallelism 下 Triton attention 的 IndexError 修复（Commit 9）**：修复了 PP 场景下 KV 缓存访问的 bug，对使用 PP 部署大模型的用户有直接影响。

4. **Diffusion 服务健康检查分离（Commit 10）**：引入 `/liveness` 和 `/health` 分离端点，使 Kubernetes 等编排系统能更精确地判断服务状态，是生产级部署的重要改进。

## 📋 逐条解读

### 1. Remove the HiMambaRadixTree that is no longer in use (删除不再使用的 HiMambaRadixTree)
- **代码层面**：删除了 `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` 整个文件（约 2178 行）。这是一个用于管理 Mamba 模型 KV 缓存的旧数据结构，已被更通用的 `MambaRadixCache` 取代。
- **新手概念课堂**：想象一个图书馆的索引系统。旧系统（HiMambaRadixTree）是为特定类型的书（Mamba 模型）设计的，但后来发现通用系统（MambaRadixCache）也能处理，而且更简洁，所以把旧系统拆掉了。
- **对你有什么影响**：这是内部清理，普通用户无感知。但如果你在代码中引用了这个模块，需要更新。

### 2. Enable breakable CUDA graph for LTX-2 (为 LTX-2 启用可中断 CUDA Graph)
- **代码层面**：修改了 `breakable_cuda_graph/runner.py`，将最大段数从 128 提升到 512（因为 LTX-2 双塔结构的注意力断点更多）；同时在 `denoising.py` 中，将 RoPE 坐标的构建移到 CUDA Graph 捕获区域之外，避免在捕获期间执行非法的 H2D 拷贝。
- **新手概念课堂**：CUDA Graph 就像录制一段视频（预编译 GPU 操作），录制完成后可以无限重放。但录制期间不能有"即兴表演"（如从 CPU 拷贝数据到 GPU），所以需要把所有准备工作提前做好。可中断 CUDA Graph 则允许在录制中插入"断点"，方便某些操作在重放时动态执行。
- **对你有什么影响**：如果你使用 LTX-2 模型生成视频，会感受到明显的速度提升（1.56 倍）。

### 3. Make scheduler rpc deadlines explicit (使调度器 RPC 超时显式化)
- **代码层面**：新增 `--scheduler-rpc-timeout` 参数，允许用户设置调度器 RPC 的端到端超时；同时在 HTTP 服务器关闭时，增加了对视频生成任务的取消和清理（`shutdown_video_jobs`）。
- **新手概念课堂**：RPC（远程过程调用）就像你打电话给朋友让他帮你做事。如果朋友一直不回应（如排队等待），你可能需要设置一个"最长等待时间"（超时），避免无限等待。这个 commit 让这个超时时间可以由用户控制，而不是默认无限等待。
- **对你有什么影响**：如果你部署了 SGLang Diffusion 服务，可以设置合理的超时时间，避免长时间运行的任务被意外中断，或者在需要时强制限制请求时长。

### 4. Fix vae fast path test after the gate refactor (修复 VAE 快速路径测试)
- **代码层面**：更新了 `test_autoencoder_kl_fastpath.py` 测试文件，适配新的 `use_vae_fast_path` 上下文管理器接口，不再直接操作 `gate.enabled` 属性。
- **新手概念课堂**：VAE（变分自编码器）是扩散模型中负责图像编解码的组件。"快速路径"是一种优化手段，但保证优化后的结果与原始结果一致非常重要。测试就是用来验证这一点的。
- **对你有什么影响**：这是测试代码的维护，普通用户无感知。

### 5. Fix Nemotron W4A16 NVFP4 MoE backend (修复 Nemotron W4A16 NVFP4 MoE 后端)
- **代码层面**：在 `overrides.py` 中，检测到 W4A16_NVFP4 量化的 MoE 层时，强制要求使用 `--moe-a2a-backend=none` 和 `--moe-runner-backend=marlin`；在 `marlin_utils_fp4.py` 中，修复了 Marlin 权重 padding 后维度不匹配的问题，增加了对 `padded_size_k` 的处理。
- **新手概念课堂**：MoE（混合专家）模型将输入路由到多个"专家"子网络。NVFP4 是一种 4-bit 浮点量化格式，能大幅减少显存占用。Marlin 是一种高效的矩阵乘法内核。这个 commit 确保这些技术能正确组合使用。
- **对你有什么影响**：如果你使用 Nemotron 模型的 NVFP4 量化版本，这个修复能避免运行时错误，并确保推理结果正确。

### 6. Speed up tp and fsdp checkpoint loading (加速 TP 和 FSDP 检查点加载)
- **代码层面**：优化了 `fsdp_load.py` 中的检查点加载逻辑，增加了 `rank_local_checkpoint` 的使用，减少了对 safetensors 格式的依赖，并简化了部分参数加载流程。
- **新手概念课堂**：FSDP（Fully Sharded Data Parallel）将模型参数分片到多个 GPU 上。检查点（Checkpoint）是模型参数的持久化快照。加载检查点时，如果每个 GPU 只加载自己需要的分片（rank-local），而不是加载全部再分发，速度会快很多。
- **对你有什么影响**：如果你使用多 GPU 加载大型 Diffusion 模型，会感受到检查点加载时间明显缩短。

### 7. Bind each rank to accelerator before distributed init (在分布式初始化前绑定设备)
- **代码层面**：在 `parallel_state.py` 中，将 `current_platform.set_device(device)` 提前到 `init_distributed_environment` 之前；在 `gpu_worker.py` 中，使用 `current_platform.set_device` 替代 `torch.get_device_module().set_device`。
- **新手概念课堂**：在分布式训练/推理中，每个进程（rank）需要绑定到特定的 GPU 设备。如果绑定太晚，可能会导致设备分配错误。提前绑定可以确保后续操作都在正确的设备上执行。
- **对你有什么影响**：修复了多 GPU 环境下可能出现的设备分配错误，对使用多 GPU 部署的用户更稳定。

### 8. Enable BCG with TP (启用 BCG 与 TP 组合)
- **代码层面**：在 `breakable_cuda_graph/runner.py` 中，新增 `_tp_graph_capture` 上下文管理器，在 CUDA Graph 捕获期间进入 TP 组的图捕获上下文；在 `group_coordinator.py` 中，优化了 `graph_capture` 上下文，确保自定义 all-reduce 在捕获时使用图路径。
- **新手概念课堂**：TP（张量并行）将模型权重切分到多个 GPU 上，每个 GPU 计算一部分。BCG（可中断 CUDA Graph）需要在捕获期间确保所有通信操作也符合图捕获的要求。这个 commit 让两者可以同时使用。
- **对你有什么影响**：如果你使用 TP 和 BCG 的组合，这个修复能避免图重放时的错误，并提高性能。

### 9. Fix IndexError in Triton backend with pipeline parallelism (修复 PP 下 Triton 后端的 IndexError)
- **代码层面**：在 `triton_backend.py` 和 `memory_pool.py` 中，将 `get_value_buffer(0)` 改为 `get_value_buffer(start_layer)`，避免在 PP 场景下访问不存在的第 0 层。
- **新手概念课堂**：在 PP 中，每个 GPU 只负责模型的一部分层。假设模型有 12 层，GPU 0 负责 0-5 层，GPU 1 负责 6-11 层。GPU 1 的 KV 缓存中没有第 0 层的数据，所以访问 `get_value_buffer(0)` 会出错，应该访问 `get_value_buffer(6)`（即 `start_layer`）。
- **对你有什么影响**：如果你使用 PP 部署模型，这个修复能避免运行时崩溃。

### 10. Gate /health on warmup completion and add liveness endpoint (健康检查与存活检查分离)
- **代码层面**：新增 `/liveness` 端点（始终返回 200），并将 `/health` 和 `/health_generate` 与服务器预热完成状态绑定（预热中返回 503，完成后返回 200）。
- **新手概念课堂**：在 Kubernetes 等编排系统中，有两种探针：存活探针（Liveness Probe）用于判断进程是否还活着，就绪探针（Readiness Probe）用于判断服务是否可以接收流量。以前 `/health` 同时承担两个角色，预热时间长时会导致误判。现在分离后，`/liveness` 管存活，`/health` 管就绪。
- **对你有什么影响**：如果你在 Kubernetes 上部署 SGLang Diffusion，可以更准确地配置探针，避免预热期间服务被误杀。

### 11. Support LingBot-Video MoE 30B T2V (支持 LingBot-Video MoE 30B 文生视频模型)
- **代码层面**：新增 `lingbot_video_moe.py` 配置文件，定义了 LingBot-Video MoE 模型的架构参数（如 48 层、128 个专家、每个 token 激活 8 个专家等），并在 `__init__.py` 中注册。
- **新手概念课堂**：MoE 模型包含多个"专家"子网络，每个 token 只激活其中一小部分，从而在保持模型容量的同时降低计算量。LingBot-Video 是一个视频生成模型，支持文生视频（T2V）。
- **对你有什么影响**：如果你使用 LingBot-Video MoE 模型，现在可以在 SGLang 中直接加载和推理。

### 12. Make ring admission a backend capability (将 Ring Attention 支持声明为后端能力)
- **代码层面**：在 `AttentionBackend` 基类中新增 `supports_ring_rotation` 方法（默认返回 False），并在 FlashAttention 和 SageAttention 后端中重写为 True。在 `layer.py` 中，根据后端能力决定是否允许 Ring Attention。
- **新手概念课堂**：Ring Attention 是一种处理超长序列的注意力机制，将序列分片到多个 GPU 上，通过环形通信合并结果。但并非所有注意力后端都支持，所以需要显式声明能力。
- **对你有什么影响**：避免了使用不支持 Ring Attention 的后端时出现错误，提高了系统的健壮性。

### 13. Build Qwen's masked varlen metadata host-side (在 CPU 侧构建 Qwen 的掩码变长元数据)
- **代码层面**：在 `qwen_image.py` 中，当 `txt_seq_lens` 可用时，直接在 CPU 侧构建 varlen 注意力元数据（`cu_seqlens`、`indices` 等），避免在 GPU 上使用 `nonzero` 操作（会触发设备同步）。
- **新手概念课堂**：变长注意力（Varlen Attention）需要知道每个序列的长度和位置信息。以前这些信息通过 GPU 上的 `nonzero` 操作计算，会触发 GPU 和 CPU 的同步（很慢）。现在直接从已有的 `txt_seq_lens` 在 CPU 上计算，避免同步。
- **对你有什么影响**：如果你使用 Qwen 图像生成模型，会感受到每一步去噪的延迟降低。

### 14. Move SWA chunk-cap hatch tests into the registered suite (将 SWA 块上限测试移到注册测试套件)
- **代码层面**：将 `test/manual/test_schedule_policy.py` 中的 SWA 块上限相关测试移到 `test/registered/unit/managers/test_prefill_adder.py` 中，使其在 CI 中自动运行。
- **新手概念课堂**：SWA（Sliding Window Attention）是一种只关注最近 token 的注意力机制，节省显存。"块上限"（Chunk Cap）是一种调度策略，防止内存不足。测试移动意味着这些测试现在会自动在 CI 中运行，确保功能不被破坏。
- **对你有什么影响**：这是测试基础设施的改进，普通用户无感知，但提高了代码质量保证。

### 15. Upgrade recommended sglang version on Ascend NPU (更新 Ascend NPU 推荐版本)
- **代码层面**：将 Ascend NPU 文档中推荐的 SGLang 版本从 `v0.5.13.post1` 更新到 `v0.5.16`，并更新了对应的 Docker 镜像标签。
- **新手概念课堂**：Ascend NPU 是华为的 AI 芯片。SGLang 提供了针对该平台的 Docker 镜像，版本更新意味着包含了更多 bug 修复和性能优化。
- **对你有什么影响**：如果你在华为 Ascend NPU 上使用 SGLang，建议升级到新版本以获得更好的体验。

### 16. Z-Image bit-exact fused qk-norm (Z-Image 位精确融合 QK-Norm)
- **代码层面**：在 `zimage_native_norm.py` 中新增 `_qk_rmsnorm_native_kernel` Triton 内核，实现了与 eager 路径位精确（bit-exact）匹配的 QK-Norm 计算，并进行了融合优化。
- **新手概念课堂**：QK-Norm 是注意力机制中在计算 Q 和 K 的点积之前对它们做归一化（RMSNorm）的操作。"位精确"意味着优化后的结果与原始结果在二进制层面完全相同，这是为了保证模型输出的一致性。"融合"意味着将多个操作合并为一个内核，减少内存读写。
- **对你有什么影响**：如果你使用 Z-Image 模型，会感受到端到端推理速度提升（6.4%），同时保证输出与原始版本完全一致。

### 17. Fix prefill CP graph overflow with larger bucket search (修复预填充 CP 图溢出)
- **代码层面**：在 `bcg.py` 中，新增 `required_local_tokens` 方法，根据 zigzag CP 布局计算所需的本地 token 数；新增 `select_replay_bucket` 方法，在重放时选择最小的、能容纳所需本地 token 的捕获桶。
- **新手概念课堂**：CP（Context Parallelism）将长序列的上下文切分到多个 GPU 上。Zigzag 布局是一种特定的切分方式。CUDA Graph 需要预分配内存（桶），如果实际请求需要的 token 数超过预分配的大小，就会溢出。这个 commit 让系统在重放时能选择更大的桶。
- **对你有什么影响**：修复了预填充阶段 CUDA Graph 可能溢出的问题，提高了长序列推理的稳定性。

### 18. Add flashinfer mHC fusion for DSV4 (为 DSV4 添加 FlashInfer mHC 融合)
- **代码层面**：在 `deepseek_v4.py` 中新增 `_flashinfer_hc_pre` 函数，使用 FlashInfer 的 `mhc_pre_big_fuse` 内核替代原有的 TileLang 实现，并支持 split-K 优化。新增环境变量 `SGLANG_OPT_USE_FLASHINFER_MHC` 控制开关。
- **新手概念课堂**：mHC（Multi-Head Companion）是 DeepSeek-V4 模型中的一种注意力机制。FlashInfer 是一个高性能 GPU 内核库。这个 commit 为 mHC 的预处理阶段提供了新的高性能实现，并允许用户通过环境变量选择。
- **对你有什么影响**：如果你使用 DeepSeek-V4 模型，可以通过设置环境变量 `SGLANG_OPT_USE_FLASHINFER_MHC=1` 来尝试新的加速实现。

### 19. Scope the masked-path replicated guard to SP runs (将掩码路径的复制保护限定在 SP 场景)
- **代码层面**：在 `layer.py` 中，将"掩码路径不支持复制前缀/后缀"的保护条件从"总是生效"改为"仅在序列并行（SP）且世界大小大于 1 时生效"。
- **新手概念课堂**：在 SP 中，序列被切分到多个 GPU 上，每个 GPU 只处理一部分。"复制前缀/后缀"是指在每个 GPU 上都复制相同的前缀/后缀 token。在 SP 下这会导致数据重复，所以需要拒绝；但在单 GPU 上，掩码已经描述了完整序列，复制计数没有意义，所以可以忽略。
- **对你有什么影响**：修复了单 GPU 场景下使用掩码注意力时的误报错误。

### 20. Fix fractional simulated acceptance in DSpark (修复 DSpark 中的分数模拟接受率)
- **代码层面**：在 `dspark_verify.py` 中，将 `_simulated_correct_len` 的返回值从 `round()` 后的整数改为调用 `sample_simulated_acc_len` 动态采样；在 `spec_utils.py` 中，将 `_sample_simulated_acc_len` 重命名为 `sample_simulated_acc_len` 并导出。
- **新手概念课堂**：DSpark 是一种投机解码（Speculative Decoding）算法，用一个草稿模型生成多个候选 token，再用目标模型验证。"模拟接受率"是用于基准测试的开关，模拟草稿模型的接受长度。以前这个值只能是整数（通过 `round()`），现在可以支持分数（如 2.5），通过采样实现。
- **对你有什么影响**：如果你使用 DSpark 进行基准测试，现在可以设置更精细的模拟接受率，获得更准确的性能评估。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)