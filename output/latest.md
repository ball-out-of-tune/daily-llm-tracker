# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 15:15 UTC | AI 解读: ✅ 含代码解读

## vllm
# vLLM 仓库 24 小时重要 Commits 解读

## 🔥 重要更新

1. **KV 传输可靠性大幅提升**：多个 commit 修复了 KV 传输中的端口死锁、前缀缓存错误、远程引擎清理等问题，这些直接影响多卡/多节点推理的稳定性。
2. **AMD ROCm 平台支持显著增强**：多个 commit 为 AMD GPU 添加了 Qwen3.5、DeepSeek-V4、Kimi-K3 等模型的优化和测试，AMD 用户可用模型范围扩大。
3. **量化与 MoE 精度修复**：修复了 INT8 量化配置错误、FP8 测试精度问题，并扩展了 EPLB 对更多 MoE 后端的支持，提升模型量化后的准确性。
4. **磁盘卸载新功能**：为 CPU 卸载连接器新增了磁盘后端，允许将 KV 缓存卸载到磁盘，极大扩展了长上下文场景下的内存容量。

---

## 📋 逐条解读

### 1. [Bugfix] Fix get_open_port() livelock on DP-reserved ports and cover get_open_ports_list

- **代码层面**：修改了 `vllm/utils/network_utils.py` 中的 `get_open_port()` 函数，修复了当 `VLLM_PORT` 环境变量设置的值落在数据并行（DP）预留端口范围内时，函数会无限循环的问题。现在会跳过预留端口范围，并增加了超时保护测试。
- **新手概念课堂**：端口就像公寓的门牌号，每个程序要通信必须独占一个。数据并行时多个进程需要一组连续的端口，如果主端口配置恰好落在预留范围内，就像你要住 5682 房间但 5680-5690 被整层包了，程序会一直找空房却找不到，陷入死循环。修复后程序会聪明地跳过整层楼。
- **对你有什么影响**：如果你使用多卡数据并行，且自定义了端口号，之前可能遇到启动卡死的问题，现在会自动避开冲突。

---

### 2. [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier in wait_for_layer_load

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`，为 MoRIIO 的 KV 读取增加了按层（per-layer）的完成屏障，并添加了 CUDA Graph 模式兼容性警告。当使用完整 CUDA Graph 时，会警告用户需要切换为 PIECEWISE 模式。
- **新手概念课堂**：CUDA Graph 就像把一系列 GPU 操作"录制"成一段视频，之后可以快速"重放"。但"重放"时无法在中间插入新指令。MoRIIO 需要在每层加载完 KV 后做一次检查（屏障），这就像在视频中间插入暂停点，完整 CUDA Graph 不支持，所以需要分段模式（PIECEWISE）。
- **对你有什么影响**：如果你使用 MoRIIO 做 KV 传输且启用完整 CUDA Graph，会收到警告，建议切换为 PIECEWISE 模式以保证准确性。

---

### 3. [PD][NixlPush][Bugfix] Fix prefix caching

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/nixl_push_connector.py`，修复了前缀缓存命中时，推送方（P）只应写入未计算的部分（尾部）到接收方（D）的槽位，而不是错误地写入整个序列。增加了 `local_physical_per_logical` 参数参与对齐计算。
- **新手概念课堂**：前缀缓存就像你做笔记时，如果别人已经记了前 10 页，你只需要从第 11 页开始写。之前代码错误地从第 1 页开始写，把别人已有的内容覆盖了。修复后只写新内容（尾部），避免覆盖已有缓存。
- **对你有什么影响**：使用 NixlPush 做 KV 传输时，前缀缓存命中率更高，重复内容的计算量减少，推理速度提升。

---

### 4. [Model] Enable Qwen3.8 for AMD Rocm

- **代码层面**：在 `vllm/model_executor/models/qwen3_5.py` 中为 Qwen3.5 模型添加了 `SupportsMRoPE` 支持声明，并实现了 `get_mrope_input_positions` 方法，返回 3 维的位置编码（适用于多模态输入）。
- **新手概念课堂**：MRoPE 是多模态旋转位置编码，它告诉模型"这个 token 在序列中的什么位置"。对于图片、音频等多模态输入，需要 3 个维度的位置信息（比如高度、宽度、时间）。AMD 之前不支持这个特性，现在补齐了。
- **对你有什么影响**：AMD ROCm 用户现在可以运行 Qwen3.5 系列多模态模型了。

---

### 5. [Bugfix] Skip fetching revision for model when model and weights_model are different

- **代码层面**：修改了 `vllm/config/model.py` 中的 `__post_init__`，增加了判断条件：如果权重来源（`model_weights`）与模型 ID 不同，或配置路径（`hf_config_path`）与模型 ID 不同，则跳过解析模型版本（revision）。避免对不存在的仓库发起网络请求。
- **新手概念课堂**：Hugging Face 上的模型像 GitHub 仓库，每个版本有唯一 commit ID（revision）。当你指定 `model=Qwen/Qwen3-0.6B` 但 `model_weights=unsloth/Qwen3-0.6B-GGUF` 时，权重和模型来自不同仓库，去 Qwen 仓库查版本号没有意义，反而会报错。现在代码聪明地识别了这种情况。
- **对你有什么影响**：使用 GGUF 等外部权重文件时，不再因版本解析失败而报错，启动更顺畅。

---

### 6. Fix ROCm architecture import on non-ROCm platforms

- **代码层面**：修改了 `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` 和 `vllm/model_executor/layers/quantization/mxfp4.py`，将 `from vllm.platforms.rocm import on_gfx1250` 的顶层导入改为条件导入，只有当前平台是 ROCm 时才导入，避免在非 AMD 平台导入不存在的模块。
- **新手概念课堂**：`on_gfx1250()` 是检查 GPU 是否为 AMD 特定型号（gfx1250）的函数。之前代码在导入时就调用它，但如果是 NVIDIA 或 Intel 平台，这个模块根本不存在，就像在 Windows 上运行只有 Mac 才有的软件，直接报错。修复后先检查系统类型再决定是否导入。
- **对你有什么影响**：NVIDIA/Intel 用户不会再遇到因 ROCm 相关导入导致的崩溃。

---

### 7. feat: extended EPLB support for Mistral Large 3 and additional MoE backends

- **代码层面**：新增了 `tests/distributed/test_eplb_quant_scale_consistency.py` 测试文件，验证 EPLB（Expert Parallel Load Balancing，专家并行负载均衡）对量化参数（如 NVFP4 的 scale 和 alpha）的重排一致性。确保专家重排后，量化参数也跟着正确重排。
- **新手概念课堂**：EPLB 就像餐厅里动态调整服务员（专家）负责的餐桌，以平衡工作量。但每个服务员有自己的"工具包"（量化参数）。这个 commit 确保当服务员被调去别的餐桌时，他的工具包也一起带走，不会拿错别人的。
- **对你有什么影响**：使用 Mistral Large 3 且启用 EPLB 时，量化模型的输出更准确，不会因参数错位导致结果异常。

---

### 8. [XPU] quick fix online quantization UT break

- **代码层面**：修改了 `tests/quantization/test_online.py` 测试文件，将硬编码的 `device="cuda"` 替换为 `DEVICE = current_platform.device_type`，并在 NVFP4 测试的跳过条件中增加 XPU 平台判断。
- **新手概念课堂**：测试代码之前把所有操作都指定在 NVIDIA GPU（cuda）上运行，就像只给一个品牌的电脑写测试。现在改为"当前平台是什么就用什么设备"，兼容 Intel 的 XPU（类似 GPU 的加速卡）。
- **对你有什么影响**：Intel 数据中心 GPU 用户在运行量化测试时不再报错，CI 流程更顺畅。

---

### 9. [Misc] Add and enable Triton kernel unit tests on XPU

- **代码层面**：修改了多个测试文件（如 `test_fused_rms_norm_gated.py`、`test_block_int8.py`、`test_int8_kernel.py`），将设备从硬编码的 `"cuda:0"` 改为根据平台动态选择，并调整了跳过条件以支持 XPU。
- **新手概念课堂**：Triton 是一种 GPU 编程语言，类似 CUDA 但更易用。测试之前只针对 NVIDIA GPU 编写，现在扩展到了 Intel 的 XPU 加速器，就像给同一款游戏增加了对更多显卡的支持。
- **对你有什么影响**：Intel 平台用户现在可以运行这些内核测试，确保 Triton 内核在 XPU 上正常工作。

---

### 10. [PD][PushConnector] Record last activity of remotes to allow clean up of stale ones

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`，新增 `_engine_last_active` 字典记录每个远程引擎的最后活跃时间，并实现 `_evict_stale_engines` 方法，在推送 KV 时检查并清理超过 TTL（生存时间）的过期引擎连接。
- **新手概念课堂**：想象一个办公室，每个远程连接像一位同事。如果某位同事很久不露面（超过 TTL），就认为他离职了，需要清理他的工位（连接资源），否则工位越占越多，新同事没法坐。这个 commit 实现了自动清理机制。
- **对你有什么影响**：长时间运行的推理集群中，动态扩缩容后不会积累僵尸连接，内存和资源占用更稳定。

---

### 11. [Bugfix][Platform] Stop re-initializing NVML on every device-capability check

- **代码层面**：修改了 `vllm/platforms/cuda.py`，移除了 `has_device_capability` 中多余的 `with_nvml_context` 包装，避免每次检查设备能力时都调用 `nvmlInit()`/`nvmlShutdown()`。测试验证了重复调用只初始化一次 NVML。
- **新手概念课堂**：NVML 是 NVIDIA 的硬件管理库，初始化它就像启动一个"硬件监控服务"。之前每次检查 GPU 算力都要重启这个服务，非常耗时。现在只启动一次，之后直接查缓存结果，就像查字典而不是每次重新编字典。
- **对你有什么影响**：推理性能提升，特别是使用 FP8 量化时，因为每个注意力层每步都要检查设备能力，之前会频繁调用 NVML 导致性能下降（issue #50381）。

---

### 12. [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16

- **代码层面**：修改了 `vllm/model_executor/layers/fused_moe/oracle/int8.py`，将断言改为显式错误抛出，并修正了条件判断：`if scales_absent and not per_act_token_quant:` 才返回 W8A16 配置，否则应该走 W8A8 路径。
- **新手概念课堂**：W8A8 和 W8A16 是两种量化方案：W8A8 表示权重和激活都用 8 位整数，速度快但精度稍低；W8A16 只量化权重，激活保持 16 位，精度高但慢。之前配置逻辑有误，把动态量化误判为 W8A16，现在修正为正确的 W8A8。
- **对你有什么影响**：使用动态 INT8 量化的 MoE 模型时，推理速度更快，因为正确启用了 W8A8 优化路径。

---

### 13. [Refactor] Remove kernel dead code

- **代码层面**：删除了 `csrc/cpu/cpu_attn_fp8.hpp` 中未使用的标量反量化函数 `fp8e5m2_to_float_scalar`，以及 `csrc/libtorch_stable/cache_kernels.cu` 中未使用的 `copy_blocks_kernel` 和 `copy_blocks_mla_kernel` 等死代码。
- **新手概念课堂**：死代码就像衣柜里永远不穿的衣服，占地方但没用。这些是之前版本遗留的旧内核函数，已经不被任何代码调用，删除后编译更快、二进制更小。
- **对你有什么影响**：编译时间略微缩短，安装包体积减小，无功能影响。

---

### 14. Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel

- **代码层面**：新增了 `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml` 和 `DeepSeek-V4-Pro-NVFP4.yaml` 测试配置，并在 `.buildkite/test-amd.yaml` 中添加了在 8xMI355 GPU 上运行的 CI 测试步骤。为 AMD 平台提供了 DeepSeek-V4 的 NVFP4 量化支持。
- **新手概念课堂**：NVFP4 是一种 4 位浮点量化格式，能把模型压缩到很小。AMD 的 Quark 工具链可以将模型量化为 NVFP4，但这个格式需要特殊的 GPU 指令支持。这个 commit 让 vLLM 在 AMD 上通过模拟（emulation）方式支持这种格式。
- **对你有什么影响**：AMD 用户可以在 MI355 等 GPU 上运行 DeepSeek-V4 的 NVFP4 量化版本，大幅降低显存需求。

---

### 15. [ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes

- **代码层面**：修改了 `tests/kernels/moe/test_block_fp8.py`，将参考实现中的 `native_per_token_group_quant_fp8` 替换为生产环境的 `per_token_group_quant_fp8`，并增加了 `silu_fp32` 参数来匹配不同内核的精度路径，同时放宽了大 K 形状的测试容差。
- **新手概念课堂**：FP8 是一种 8 位浮点格式，精度有限。在数学计算中，不同实现方式（比如先算 SiLU 再量化，还是先量化再算）会有微小差异（约 0.06-0.1%）。测试之前要求结果完全一致，现在允许微小误差，就像考试从"必须满分"放宽到"95 分以上即可"。
- **对你有什么影响**：AMD 平台上的 FP8 MoE 测试不再因微小精度差异而误报失败，CI 更稳定。

---

### 16. [Feat][Core] Add disk offloading support to SimpleCPUOffloadConnector

- **代码层面**：在 `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` 中新增了 `kv_offload_backend` 配置项，支持 `"cpu"` 和 `"disk"` 两种后端。磁盘后端需要设置 `disk_path`，并支持 `disk_capacity_bytes`、`disk_buffer_slots`、`use_page_cache` 等参数。
- **新手概念课堂**：KV 缓存是推理时保存的中间结果，占显存。之前只能卸载到 CPU 内存（RAM），现在可以卸载到硬盘。就像你把不常用的文件从桌面（显存）移到内存（RAM），现在还能移到硬盘（磁盘），容量更大但速度更慢。
- **对你有什么影响**：超长上下文场景下，KV 缓存可以卸载到磁盘，极大扩展了可处理的序列长度，不再受限于显存和内存大小。

---

### 17. [rl] Stateful Trainer Send: NCCL + Sparse NCCL [3/N]

- **代码层面**：重构了 `vllm/distributed/weight_transfer/` 模块，将旧的 `NCCLWeightTransferEngine` 拆分为 `WeightTransferTrainerFactory` 和 `RayVLLMWeightSyncClient`，并更新了 `examples/rl/rlhf_async_new_apis.py` 示例代码，使用新的 API 进行权重传输。
- **新手概念课堂**：在强化学习（RL）中，训练器（Trainer）需要不断把更新后的模型权重发给推理引擎。这就像教练（Trainer）不断把新的战术手册发给球员（推理引擎）。旧的 API 是"一次性广播"，新的 API 支持"有状态传输"，可以增量更新，节省带宽。
- **对你有什么影响**：RL 训练场景下，权重同步更高效，支持更大的模型和更频繁的更新。

---

### 18. [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path

- **代码层面**：新增了 `vllm/models/kimi_k3/amd/latent_moe_runner.py` 和对应的测试文件 `tests/models/kimi_k3/test_amd_latent_moe_runner.py`，为 AMD ROCm 平台实现了 Kimi-K3 模型的潜变量 MoE（Latent MoE）上投影分片优化。
- **新手概念课堂**：潜变量 MoE 是一种高效的专家混合架构，先用一个小网络压缩信息，再路由到专家。上投影（up-projection）是把压缩后的信息还原为完整维度。AMD 之前只能复制所有专家（浪费显存），现在可以像 NVIDIA 一样分片处理。
- **对你有什么影响**：AMD 用户运行 Kimi-K3 模型时，显存占用更低，推理速度更快。

---

### 19. [Bugfix] Fix Mamba all-mode CPU offload boundary alignment

- **代码层面**：修改了 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`，将 `resolve_mamba_align_size` 中的条件从 `mamba_cache_mode == "align"` 扩展为包含 `"all"` 模式。同时更新测试，增加 `mamba_cache_mode="all"` 参数化测试，并使用 float32 确保精确比较。
- **新手概念课堂**：Mamba 是一种状态空间模型，每个 token 只保留一个"状态"而不是像 Transformer 那样保留所有历史。CPU 卸载时，缓存是按块（block）存储的。当提示词恰好落在块边界时，边界上的状态已经包含了最后一个 token 的信息，加载它会重复计算。修复后所有模式都正确处理这个边界情况。
- **对你有什么影响**：使用 Mamba 模型且启用 CPU 卸载时，输出结果更准确，不会因边界处理错误导致内容异常。

---

### 20. [Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance

- **代码层面**：修改了 `vllm/v1/worker/gpu/mm/encoder_runner.py`，为编码器专用实例（encoder-only instance）跳过多模态嵌入的 gather 操作。新增测试验证编码器实例只编码并发布结果，不执行语言模型相关的嵌入聚合。
- **新手概念课堂**：EPD（Encoder-Prefill-Decode）架构把编码器、预填充、解码分成不同实例。编码器实例只负责把图片转成向量（embedding），不生成文本。之前代码错误地让编码器实例也尝试聚合嵌入（gather），但编码器没有语言模型，就像让只负责拍照的同事去写文章，当然会失败。
- **对你有什么影响**：使用 EPD 架构处理多模态输入时，编码器实例不再崩溃，整体推理流程更稳定。

---

## 💡 今日关键词

1. **KV Cache（键值缓存）**：Transformer 推理时保存的中间计算结果，避免重复计算，是推理引擎中最占用显存的部分。
2. **MoE（Mixture of Experts，专家混合）**：一种模型架构，将网络分成多个"专家"子网络，每次推理只激活少数专家，在保持模型能力的同时降低计算量。
3. **量化（Quantization）**：将模型权重从高精度（如 FP16）转换为低精度（如 INT8、FP8、NVFP4），以减小模型大小和加快推理速度，但会带来微小精度损失。

## sglang
## 🔥 重要更新

1. **LTX-2 视频生成性能大幅提升** (Commit 2)：通过启用 breakable CUDA graph，H200 端到端耗时从 10.75 秒降到 6.90 秒，提升 1.56 倍，对视频生成用户是立竿见影的体验改善。
2. **新增 LingBot-Video MoE 30B 模型支持** (Commit 11)：支持新的视频生成模型架构，扩展了 SGLang Diffusion 的模型生态。
3. **健康检查端点重构** (Commit 10)：区分了 `/liveness`（进程存活）和 `/health`（推理就绪），让 Kubernetes 部署更可靠，避免在长 warmup 期间误杀服务。
4. **Nemotron W4A16 NVFP4 MoE 后端修复** (Commit 5)：修复了量化 MoE 层的兼容性问题，并自动选择正确的后端，避免用户手动配置错误。

---

## 📋 逐条解读

### 1. 移除不再使用的 HiMambaRadixTree 缓存
- **代码层面**：删除了 `hi_mamba_radix_cache.py` 整个文件（约 2178 行）。这是一个专为 HiMamba 模型设计的基数树缓存实现，现在已被更通用的混合缓存替代。
- **新手概念课堂**：基数树（Radix Tree）像图书馆的索引卡片——按前缀共享来组织，让相同前缀的请求能复用已计算的 KV 缓存。HiMamba 版本是它的一个特殊变体，现在"退休"了。
- **对你有什么影响**：无直接感知，代码更简洁，维护成本降低。

### 2. [diffusion] 为 LTX-2 启用 breakable CUDA graph
- **代码层面**：在 `breakable_cuda_graph/runner.py` 中把最大段数从 128 调到 512（因为 LTX-2 每个 block 有 6 个 attention 断点），并新增了签名不匹配时的诊断日志。在 `denoising.py` 中，把 RoPE 坐标构建移到 CUDA graph 捕获区域之外，因为 `torch.tensor(list, device=cuda)` 是非 pinned 的 H2D 拷贝，在捕获中非法。
- **新手概念课堂**：CUDA graph 像"录好的舞蹈视频"——把一系列 GPU 操作录下来，之后直接"播放"而不用重新编排，省去大量调度开销。Breakable 版本允许中途"暂停/恢复"，适合 diffusion 这种迭代结构。
- **对你有什么影响**：LTX-2 视频生成速度提升 1.56 倍，等待时间明显缩短。

### 3. [diffusion] 让 scheduler RPC 超时显式化
- **代码层面**：新增 `--scheduler-rpc-timeout` 参数，默认不设超时（避免长视频任务被误杀）。同时修复了 shutdown 流程：先取消所有视频任务，再取消 broker 任务并等待其完成。
- **新手概念课堂**：RPC（远程过程调用）像"打电话给另一个服务办事"。超时就像"电话等待上限"——如果对方太久没回应就挂断。之前这个上限是隐式的，现在可以显式配置。
- **对你有什么影响**：长视频任务不会被传输层误杀；部署时可按需设置请求截止时间。

### 4. 修复 VAE 快速路径测试
- **代码层面**：测试从直接操作 `gate.enabled` 属性改为使用 `use_vae_fast_path(opt, True)` 上下文管理器。同时增加了对 wrapper 层是否真正安装的断言。
- **新手概念课堂**：VAE（变分自编码器）是图像生成中的"压缩/解压器"。Fast path 是它的"快车道"——用融合算子加速，但结果与慢速路径略有差异。上下文管理器像"临时通行证"——进入时开启快车道，退出时自动恢复。
- **对你有什么影响**：测试更健壮，确保快速路径不会悄悄破坏精度。

### 5. 修复 Nemotron W4A16 NVFP4 MoE 后端
- **代码层面**：在 `overrides.py` 中检测模型是否有 W4A16_NVFP4 量化的 MoE 层，若有则强制要求 `--moe-a2a-backend=none` 和 `--moe-runner-backend=marlin`（否则报错）。在 `marlin_utils_fp4.py` 中修复了 weight 维度 padding 后的尺寸计算，用 `padded_size_k/n` 代替逻辑尺寸，避免越界。
- **新手概念课堂**：MoE（混合专家）像"多个专业顾问团队"，每次只让最相关的几个专家回答问题。NVFP4 是 NVIDIA 的 4-bit 浮点量化格式，能省显存但需要特定内核支持。Marlin 是高效的量化矩阵乘法内核。
- **对你有什么影响**：Nemotron 模型在 W4A16 NVFP4 量化下能正确加载和推理，不再因后端不匹配而报错。

### 6. [diffusion] 加速 TP 和 FSDP checkpoint 加载
- **代码层面**：在 `fsdp_load.py` 中，当满足条件（使用 FSDP、有权重目录、无预处理函数、非 bitsandbytes 量化）时，跳过 safetensors 迭代器逐权重加载，直接使用 `rank_local_checkpoint` 的分布式 checkpoint 格式。这避免了每个 rank 都读取全部权重再切分的开销。
- **新手概念课堂**：FSDP（Fully Sharded Data Parallel）像"把大书拆成多册分给不同人保管"。Checkpoint 是"书签"。之前每个保管员都要把整本书读一遍再找自己的部分，现在直接只读自己那册，省时省力。
- **对你有什么影响**：多 GPU 部署时模型加载时间显著缩短。

### 7. [diffusion] 在分布式初始化前绑定设备
- **代码层面**：把 `current_platform.set_device(device)` 移到 `init_distributed_environment` 之前调用，确保每个 rank 在初始化通信组之前就绑定了正确的加速器。同时为 MPS 平台添加了 `set_device` 的 no-op 实现。
- **新手概念课堂**：分布式训练像"多个工人协作"，每个工人需要先明确"自己在哪台机器、用哪个 GPU"，然后才能开始互相通信。之前这个"认领 GPU"的步骤太晚了，可能导致通信初始化时用错设备。
- **对你有什么影响**：多 GPU 分布式推理更稳定，避免设备绑定错误。

### 8. [diffusion] 修复 TP 下的 breakable CUDA graph
- **代码层面**：在 `runner.py` 中新增 `_tp_graph_capture` 上下文管理器，在捕获 CUDA graph 时进入 TP 组的 graph capture 上下文。在 `group_coordinator.py` 中，`graph_capture` 现在在捕获模式下使用 custom all-reduce 的 graph 路径（而非 eager 路径），因为 eager 路径在回放时会因未映射的 peer IPC 地址而崩溃。
- **新手概念课堂**：TP（张量并行）像"把一个大矩阵切成几块分给不同 GPU 算"。Custom all-reduce 是 GPU 间"汇总结果"的高效方式。在 CUDA graph 里，必须用 graph 版本，否则"回放"时找不到之前记录的通信地址。
- **对你有什么影响**：TP 模式下也能享受 CUDA graph 加速，不再崩溃。

### 9. 修复 Triton 注意力后端在流水线并行下的 IndexError
- **代码层面**：在 `triton_backend.py` 和 `memory_pool.py` 中，把 `get_value_buffer(0)` 改为 `get_value_buffer(start_layer)`。在流水线并行（PP）下，当前 stage 可能不包含 layer 0，所以索引 0 会越界。
- **新手概念课堂**：流水线并行像"工厂流水线"——不同 GPU 负责不同层的计算。之前代码硬编码"取第 0 层的 buffer"，但流水线中某个 GPU 可能只负责第 10-20 层，第 0 层根本不在它手里。
- **对你有什么影响**：流水线并行 + Triton 注意力后端的组合不再崩溃。

### 10. [diffusion] 健康检查端点重构
- **代码层面**：新增 `/liveness` 端点（HTTP 服务存活即返回 200），`/health` 和 `/health_generate` 现在仅在 warmup 完成后返回 200，否则返回 503。warmup 失败则直接终止服务器。
- **新手概念课堂**：Kubernetes 有"存活探针"（检查进程是否活着）和"就绪探针"（检查是否准备好接收流量）。之前 `/health` 在 warmup 期间也返回 200，导致流量过早涌入；现在分开后，存活探针用 `/liveness`，就绪探针用 `/health`。
- **对你有什么影响**：K8s 部署更稳定，不会在模型加载期间把请求发给未就绪的服务。

### 11. [diffusion] 支持 LingBot-Video MoE 30B 文生视频模型
- **代码层面**：新增 `lingbot_video_moe.py` 配置文件，定义了模型架构参数（48 层、128 专家、top-8 路由等），并注册到模型配置中心。同时实现了对应的模型代码。
- **新手概念课堂**：MoE（混合专家）模型像"一个由 128 个专家组成的委员会"，每个 token 只让最相关的 8 个专家处理。这样能用更少的计算量获得更大的模型容量。
- **对你有什么影响**：可以直接用 SGLang 部署 LingBot-Video MoE 30B 模型做文生视频。

### 12. [diffusion] 将 ring admission 变为后端能力
- **代码层面**：在 `AttentionBackend` 基类中新增 `supports_ring_rotation()` 方法，默认返回 `False`。FlashAttention 和 SageAttention 后端重写为返回 `True`。在 `layer.py` 中，只有支持该能力的后端才允许启用 ring attention。
- **新手概念课堂**：Ring attention 像"环形接力"——多个 GPU 围成一圈，每个只处理序列的一部分，然后传给下一个。但接力需要每个 GPU 能算出"当前最优值"（softmax LSE），不是所有注意力内核都能做到。
- **对你有什么影响**：ring attention 只在支持的内核上启用，避免静默错误。

### 13. [diffusion] 在 host 端构建 Qwen 的 masked varlen 元数据
- **代码层面**：在 `qwen_image.py` 中，当 `txt_seq_lens` 可用且合法时，用 `build_varlen_mask_meta_from_ranges` 在 host 端直接构建 varlen 元数据，避免每个 denoising 步骤都做 GPU 上的 nonzero 操作和设备同步。
- **新手概念课堂**：Varlen 元数据像"每个句子的长度清单"，告诉注意力内核每行该看多长。之前每次都要在 GPU 上"数一遍"（nonzero），现在直接从已知的 `txt_seq_lens` 算出来，省去 GPU 往返。
- **对你有什么影响**：Qwen 图像生成速度提升，尤其是长序列场景。

### 14. 将 SWA chunk-cap 测试移入注册套件
- **代码层面**：把 `test_schedule_policy.py` 中手写的 `_swa_adder` 测试移到 `test_prefill_adder.py` 的注册测试套件中，使用更真实的测试环境。
- **新手概念课堂**：SWA（滑动窗口注意力）像"只看最近 N 个字的阅读方式"。Chunk-cap 是"如果请求太大，池子永远装不下，就直接放行"的机制，防止死锁。
- **对你有什么影响**：测试覆盖更完整，调度策略的边界情况更可靠。

### 15. [NPU] 升级 Ascend NPU 推荐版本
- **代码层面**：文档中的 Docker 镜像标签从 `v0.5.13.post1-cann9.0.0-a3` 更新为 `cann9.0.0-a3-v0.5.16`。
- **新手概念课堂**：Docker 镜像标签像"软件版本号"。新标签对应更新的 SGLang 版本（v0.5.16），包含更多 bug 修复和功能。
- **对你有什么影响**：Ascend NPU 用户应使用新镜像以获取最新修复。

### 16. [diffusion] Z-Image 位精确融合 qk-norm
- **代码层面**：新增 `_qk_rmsnorm_native_kernel` Triton 内核，逐位复现 eager 路径的 RMSNorm 计算顺序（包括 bf16 舍入时机和 lane 累加顺序），确保 `torch.equal` 级一致。在 H200 Turbo 1024px 下端到端提速 6.4%。
- **新手概念课堂**：RMSNorm 像"给向量做标准化"——让每个向量的长度变成 1。融合（fused）意味着把多个操作合并成一个 GPU 内核，减少内存读写。位精确意味着"结果和原来一模一样，只是更快"。
- **对你有什么影响**：Z-Image 生成速度提升，且输出与未优化版本完全一致。

### 17. 修复 prefill CP graph 溢出
- **代码层面**：在 `bcg.py` 中为 zigzag CP 策略新增 `required_local_tokens` 方法，计算实际需要的 CP-local 行数（考虑 zigzag 布局的 rank 间分配），并用 `select_replay_bucket` 搜索最小的、能容纳所需 local tokens 的捕获 bucket。
- **新手概念课堂**：CP（上下文并行）像"把长文章分成几段给不同人读"。Zigzag 布局是"交替分配"——第 1 段给 rank 0，第 2 段给 rank 1，第 3 段给 rank 0…… 之前选 bucket 只看全局 token 数，可能低估了某个 rank 实际需要的内存导致溢出。
- **对你有什么影响**：长序列 prefill 不再因 CP graph 溢出而崩溃。

### 18. 为 DSV4 添加 FlashInfer mHC 融合
- **代码层面**：新增 `SGLANG_OPT_USE_FLASHINFER_MHC` 环境变量（默认关闭）。实现了 `_flashinfer_hc_pre` 函数，用 FlashInfer 的 `mhc_pre_big_fuse` 替代 TileLang 版本，并自动选择最优 split-K 数。
- **新手概念课堂**：mHC（multi-head Compression）是 DeepSeek-V4 中的注意力压缩机制。Split-K 像"把一个大矩阵乘法拆成 K 份并行算，最后再合并"，能提高 GPU 利用率。
- **对你有什么影响**：DSV4 用户可通过环境变量启用 FlashInfer 加速路径。

### 19. [diffusion] 将 masked-path 复制保护限定到 SP 场景
- **代码层面**：在 `layer.py` 中，`raise NotImplementedError` 的条件从"有 replicated 前缀/后缀"改为"有 replicated 前缀/后缀 **且** 不是有效跳过 SP **且** SP world size > 1"。单 rank 下 replicated 计数无意义，不应报错。
- **新手概念课堂**：SP（序列并行）下，序列被切分到多个 GPU，如果某些 token 被"复制"到每个 rank，会导致注意力计算重复。但单 GPU 时没有切分，复制计数没意义，不该报错。
- **对你有什么影响**：单 GPU 使用 masked attention + replicated 参数不再误报错误。

### 20. 修复 DSpark 模拟接受率的分数问题
- **代码层面**：在 `dspark_verify.py` 中，`_simulated_correct_len` 不再缓存标量值，而是每次调用 `sample_simulated_acc_len`（支持浮点模拟值，如 2.5），并填充到 buffer。同时把 `_sample_simulated_acc_len` 重命名为 `sample_simulated_acc_len` 并导出。
- **新手概念课堂**：模拟接受率是"假装模型接受了 N 个 draft token"的基准测试工具。之前 `int(round(2.5))` 会变成 2 或 3，导致模拟不精确；现在直接用浮点数采样，更真实。
- **对你有什么影响**：DSpark 基准测试的模拟结果更准确。

---

## 💡 今日关键词

1. **CUDA Graph**：把一串 GPU 操作"录制成视频"，之后直接"播放"以省去调度开销。Breakable 版本允许中途暂停/恢复，适合 diffusion 这种迭代结构。

2. **MoE（Mixture of Experts）**：由多个"专家"子网络组成的模型，每个 token 只激活少数几个专家，用更少计算量获得更大模型容量。常用于大规模语言模型和视频生成模型。

3. **FSDP（Fully Sharded Data Parallel）**：把模型权重"分片"到多个 GPU，每个 GPU 只持有部分权重，训练/推理时按需通信获取。比简单的数据并行更省显存。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)