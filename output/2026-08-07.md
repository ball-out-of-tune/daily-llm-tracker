# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 15:15 UTC | AI 解读: ✅ 含代码解读

## vllm
## 🔥 重要更新

1. **MoRIIO KV 传输屏障修复**（Commit 2）：解决了 KV 传输中每层读取完成屏障的问题，直接影响分布式推理的准确性。
2. **NixlPush 前缀缓存修复**（Commit 3）：修复了前缀缓存（Prefix Caching）的关键 bug，影响推理性能。
3. **端口分配死锁修复**（Commit 1）：修复了端口分配时可能无限循环的问题，避免服务启动卡死。
4. **NVML 重复初始化性能问题**（Commit 11）：消除了每次设备能力检查时重复初始化 NVML 的开销，提升性能。

---

## 📋 逐条解读

### 1. [Bugfix] Fix get_open_port() livelock on DP-reserved ports and cover get_open_ports_list (#50965)

- **代码层面**：修改了 `vllm/utils/network_utils.py` 中的 `get_open_port()` 函数，修复了当 `VLLM_PORT` 环境变量设置在数据并行（DP）保留端口范围内时，函数会无限循环的问题。现在会跳过保留端口范围。
- **新手概念课堂**：端口就像公寓的门牌号。如果两个程序想用同一个门牌号就会冲突。这里的问题是，分配端口的函数（像物业管理员）被要求分配一个"已经被预留"的门牌号，导致它一直尝试、一直失败、一直重试，陷入死循环。修复后，管理员会直接跳过这些预留的门牌号。
- **对你有什么影响**：修复了在某些配置下服务启动可能卡死的问题。

### 2. [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier in wait_for_layer_load (#48534)

- **代码层面**：修改了 `moriio_connector.py`，为每层 KV 读取添加了完成屏障（barrier），确保所有层的数据读取完成后才继续执行。同时增加了对 CUDA 图模式的警告。
- **新手概念课堂**：想象你在做一道多层蛋糕。每层蛋糕都要等下面一层烤好才能放上去。这里的"屏障"就是一个检查点，确保所有层的原料都准备好了再开始组装。如果不加屏障，可能会出现"底层还没烤好，顶层就已经放上去"的错误。
- **对你有什么影响**：修复了 MoRIIO KV 传输模式下的潜在数据不一致问题，提高分布式推理的准确性。

### 3. [PD][NixlPush][Bugfix] Fix prefix caching (#48758)

- **代码层面**：修改了 `_apply_prefix_caching` 函数签名，增加了 `local_physical_per_logical` 参数，并修复了前缀缓存命中时块对齐的逻辑。同时为 `_logical_to_kernel_block_ids` 增加了 `ratio` 参数。
- **新手概念课堂**：前缀缓存就像你写论文时的"参考文献"——如果两篇文章开头一样，就不用重新写一遍开头了。这里修复的是"如何准确地复用之前计算好的部分"的逻辑，确保复用的部分和实际需要的部分完全对齐。
- **对你有什么影响**：修复了前缀缓存可能导致的错误结果，提升了推理性能和准确性。

### 4. [Model] Enable Qwen3.8 for AMD Rocm (#50068)

- **代码层面**：在 `qwen3_5.py` 中为 Qwen3.5 模型添加了 `SupportsMRoPE` 支持，并实现了 `get_mrope_input_positions` 方法，返回扩展为 3 维的位置张量。
- **新手概念课堂**：MRoPE（Multi-modal Rotary Position Embedding）是一种位置编码方式，就像给每个词标记"它在句子中的位置"。多模态模型（能同时处理文字和图片的模型）需要更复杂的位置标记。这次更新让 Qwen3.5 在 AMD 显卡上也能正确理解多模态输入的位置信息。
- **对你有什么影响**：AMD 显卡用户现在可以运行 Qwen3.5 多模态模型了。

### 5. [Bugfix] Skip fetching revision for model when model and weights_model are different (#51260)

- **代码层面**：修改了 `vllm/config/model.py` 中 `__post_init__` 的逻辑，增加 `weights_from_model` 和 `config_from_model` 两个判断条件，只有当模型、权重和配置都来自同一个仓库时才解析 revision。
- **新手概念课堂**：Hugging Face 的模型仓库就像一个仓库，模型文件（权重）可能存放在不同的仓库中。revision 就像仓库的"版本号"。如果模型和权重来自不同仓库，就不能用模型的版本号去查权重的版本。这次修复就是避免这种"张冠李戴"的错误。
- **对你有什么影响**：修复了使用不同仓库的模型和权重时可能出现的错误。

### 6. Fix ROCm architecture import on non-ROCm platforms (#51357)

- **代码层面**：修改了 `mxfp4.py` 和 `oracle/mxfp4.py`，将 `from vllm.platforms.rocm import on_gfx1250` 的导入改为条件导入，只有在当前平台是 ROCm 时才导入并调用。
- **新手概念课堂**：就像你家里的电器说明书，只有在你买的是这个品牌时才需要看。这里的问题是，在非 AMD 显卡（如 NVIDIA）上，代码也会尝试导入 AMD 特有的函数，导致报错。修复后，只有 AMD 显卡才加载这些代码。
- **对你有什么影响**：修复了非 AMD 平台上运行时的导入错误。

### 7. feat: extended EPLB support for Mistral Large 3 and additional MoE backends (#48355)

- **代码层面**：新增了 `test_eplb_quant_scale_consistency.py` 测试文件，验证 EPLB（Expert Parallel Load Balancing）对量化参数的重排一致性。涉及 NVFP4 量化方法。
- **新手概念课堂**：EPLB 就像"专家分工"——在大模型中，不同的"专家"（子网络）处理不同的任务。为了负载均衡，专家会被重新排列。这个测试确保排列后，量化参数（用于压缩模型大小的数据）也跟着正确排列，否则推理结果会出错。
- **对你有什么影响**：扩展了 EPLB 对更多模型和量化后端的支持，提升性能和准确性。

### 8. [XPU] quick fix online quantization UT break (#51365)

- **代码层面**：修改了 `tests/quantization/test_online.py`，将硬编码的 `device="cuda"` 改为 `device=DEVICE`，其中 `DEVICE = current_platform.device_type`，并修复了 NVFP4 在 XPU 上的跳过条件。
- **新手概念课堂**：测试代码就像质检员。以前质检员只检查"CUDA 品牌"的产品，现在改为检查"当前平台"的产品，这样在 Intel 的 XPU 上也能进行同样的质检。
- **对你有什么影响**：修复了 Intel XPU 上在线量化测试失败的问题。

### 9. [Misc] Add and enable Triton kernel unit tests on XPU (#45694)

- **代码层面**：修改了多个测试文件（`test_fused_rms_norm_gated.py`、`test_block_int8.py`、`test_int8_kernel.py`），将设备从硬编码的 `cuda:0` 改为根据平台动态选择，并添加了 XPU 支持条件。
- **新手概念课堂**：Triton 是一种 GPU 编程语言，就像用高级语言写底层代码。这些测试确保 Triton 写的内核在 Intel XPU 上也能正确运行。以前这些测试只在 NVIDIA GPU 上跑，现在 Intel 也能跑了。
- **对你有什么影响**：Intel XPU 用户现在有更多测试保障，内核稳定性提升。

### 10. [PD][PushConnector] Record last activity of remotes to allow clean up of stale ones (#50234)

- **代码层面**：在 `base_worker.py` 中增加了 `_engine_last_active` 字典，记录每个远程引擎的最后活动时间，并添加了 `_evict_stale_engines` 和 `_cleanup_remote_engine` 方法，用于清理超时未活动的远程引擎。
- **新手概念课堂**：就像图书馆的"借书超时"管理。如果一本书（远程引擎）很久没人借（没有活动），图书馆就会把它收回去（清理掉），腾出空间给新书。这里记录"最后借阅时间"来判定是否超时。
- **对你有什么影响**：避免了远程引擎资源泄漏，提升分布式推理的稳定性。

### 11. [Bugfix][Platform] Stop re-initializing NVML on every device-capability check (fixes #50381) (#50393)

- **代码层面**：修改了 `vllm/platforms/cuda.py`，移除了 `has_device_capability` 中不必要的 `with_nvml_context` 包装，因为 `get_device_capability` 已经自带 NVML 上下文。测试验证了 NVML 只初始化一次。
- **新手概念课堂**：NVML 是 NVIDIA 显卡的"体检中心"。每次检查显卡能力都要去体检一次，但体检中心开门关门很耗时。修复后，先去一次拿到结果存起来，之后直接查结果，不用反复开门关门。
- **对你有什么影响**：显著减少了重复调用时的开销，特别是对每个注意力层、每个步骤都调用的情况。

### 12. [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16 (#50833)

- **代码层面**：修改了 `oracle/int8.py` 中的 `make_int8_moe_quant_config`，将 `assert` 改为显式的 `raise ValueError`，并修复了条件判断：当 `scales_absent` 且 `per_act_token_quant` 为真时，不再错误地构建 W8A16 配置。
- **新手概念课堂**：W8A8 和 W8A16 是两种量化方案，分别代表"权重 8 位、激活 8 位"和"权重 8 位、激活 16 位"。后者精度更高但更慢。这次修复确保动态量化时不会错误使用精度较低的方案。
- **对你有什么影响**：修复了动态 INT8 量化 MoE 模型时精度下降的问题。

### 13. [Refactor] Remove kernel dead code (#51051)

- **代码层面**：删除了 `cpu_attn_fp8.hpp` 中未使用的 `fp8e5m2_to_float_scalar` 函数，以及 `cache_kernels.cu` 中未使用的 `copy_blocks_kernel` 和 `copy_blocks_mla_kernel` 内核。
- **新手概念课堂**：就像清理房间里的旧家具——这些代码从未被使用，但占据空间、增加维护成本。删除它们让代码更干净、更容易维护。
- **对你有什么影响**：代码更简洁，编译时间可能略有减少，无功能变化。

### 14. Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel (#47972)

- **代码层面**：新增了 DeepSeek-V4 在 AMD 上的 NVFP4 量化支持，包括新的测试配置文件（`DeepSeek-V4-Flash-NVFP4.yaml`、`DeepSeek-V4-Pro-NVFP4.yaml`）和 CI 测试步骤。
- **新手概念课堂**：NVFP4 是一种 4 位浮点量化格式，能大幅压缩模型大小。DeepSeek-V4 是大型模型，支持 NVFP4 意味着在 AMD 显卡上能以更少显存运行这个大模型。
- **对你有什么影响**：AMD 用户现在可以运行 DeepSeek-V4 模型，且占用更少显存。

### 15. [ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes (#48847)

- **代码层面**：修改了 `test_block_fp8.py`，将参考实现中的 `native_per_token_group_quant_fp8` 替换为生产环境的 `per_token_group_quant_fp8`，并增加了 `silu_fp32` 参数来控制 SiLU 激活函数的精度。
- **新手概念课堂**：测试中的"容差"就像考试及格线。FP8 是一种 8 位浮点数，精度较低。当计算量很大（大 K 形状）时，误差会累积。这次调整了测试的"及格线"，并让参考实现更贴近实际生产代码，避免误报失败。
- **对你有什么影响**：修复了 AMD 显卡上 MoE 测试误报失败的问题。

### 16. [Feat][Core] Add disk offloading support to SimpleCPUOffloadConnector (#49644)

- **代码层面**：在 `simple_cpu_offload_connector.py` 中新增了 `kv_offload_backend` 配置项，支持 `"cpu"` 和 `"disk"` 两种后端。新增了 `disk_path`、`disk_capacity_bytes`、`disk_buffer_slots`、`use_page_cache` 等配置。
- **新手概念课堂**：CPU 卸载就像把不常用的物品从书桌（GPU 显存）搬到储物柜（CPU 内存）。现在支持搬到"地下室"（磁盘）了。磁盘比内存慢但容量更大，适合存放很少用到的数据。
- **对你有什么影响**：现在可以将 KV 缓存卸载到磁盘，支持更大模型的推理。

### 17. [rl] Stateful Trainer Send: NCCL + Sparse NCCL [3/N] (#50902)

- **代码层面**：重构了 RL（强化学习）训练器的权重传输 API，将 `NCCLWeightTransferEngine.trainer_init` 改为 `WeightTransferTrainerFactory.trainer_init`，新增 `RayVLLMWeightSyncClient` 和 `ModuleSource` 等抽象。
- **新手概念课堂**：NCCL 是 GPU 之间通信的"高速公路"。在强化学习中，训练器需要把模型权重发送给推理引擎。这次重构让代码更模块化，就像把"寄快递"的流程标准化，支持不同的快递公司（NCCL 和 Sparse NCCL）。
- **对你有什么影响**：强化学习训练流程更稳定、更易扩展。

### 18. [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path (#51253)

- **代码层面**：新增了 `ROCmLatentMoERunner` 类，实现了 Kimi-K3 模型的 latent MoE 上投影分片功能，并添加了相应的测试文件。
- **新手概念课堂**：MoE（专家混合）模型像一家公司，不同"专家"处理不同任务。latent MoE 是其中一种实现。这次更新让 AMD 显卡上能更高效地并行处理这些"专家"。
- **对你有什么影响**：AMD 显卡上运行 Kimi-K3 模型的性能提升。

### 19. [Bugfix] Fix Mamba all-mode CPU offload boundary alignment (#51100)

- **代码层面**：修改了 `offloading/scheduler.py` 中的 `resolve_mamba_align_size`，将 Mamba 缓存模式从仅 `"align"` 扩展到 `"align"` 和 `"all"` 两种模式都需要对齐。
- **新手概念课堂**：Mamba 是一种状态空间模型，它的缓存就像"快照"。当 CPU 卸载缓存时，需要确保快照的边界正确。这次修复确保在"all"模式下（保存所有状态）也能正确对齐边界，避免加载错误的状态。
- **对你有什么影响**：修复了 Mamba 模型在 CPU 卸载时的潜在错误。

### 20. [Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance (#51222)

- **代码层面**：修改了 `encoder_runner.py`，新增 `execute_mm_encoder` 方法，让编码器实例只编码并发布结果，不再执行 `gather` 操作。测试验证了编码器实例不会尝试收集不存在的嵌入。
- **新手概念课堂**：在多模态模型中，编码器负责处理图片/视频，语言模型负责生成文字。以前编码器处理完图片后还会尝试"收集"结果，但编码器本身没有语言模型，收集会报错。这次修复让编码器"干完自己的活就下班"。
- **对你有什么影响**：修复了 EPD（Encoder-Producer-Decoder）架构下编码器实例崩溃的问题。

---

## 💡 今日关键词

1. **前缀缓存（Prefix Caching）**：像写论文时复用已写好的开头段落，避免重复计算相同的前缀内容，大幅提升推理速度。
2. **量化（Quantization）**：把模型中的高精度数字（如 32 位浮点）压缩成低精度（如 8 位整数），像把高清照片压缩成 JPEG，减小体积但可能损失一些细节。
3. **KV 传输（KV Transfer）**：在多 GPU 分布式推理中，把注意力机制的键值缓存（KV Cache）从一个 GPU 传到另一个 GPU，像接力赛中传递接力棒。

## sglang
# SGLang 仓库 24 小时重要 Commits 解读

---

## 🔥 重要更新

1. **LTX-2 可中断 CUDA 图加速 (Commit 2)**：H200 两阶段端到端耗时从 10.75 秒降至 6.90 秒（1.56 倍加速），这是视频生成性能的重大突破。
2. **新增 LingBot-Video MoE 30B 模型支持 (Commit 11)**：为视频生成领域引入了新的 MoE 架构模型，扩展了 SGLang Diffusion 的模型生态。
3. **Z-Image 融合 QK-Norm 优化 (Commit 16)**：H200 Turbo 1024px 端到端性能提升 6.4%，且保证与原始实现逐位一致（bit-exact）。
4. **Nemotron W4A16 NVFP4 MoE 后端修复 (Commit 5)**：修复了量化 MoE 层在特定 GPU 上的兼容性问题，并自动选择正确的后端。

---

## 📋 逐条解读

### 1. 移除不再使用的 HiMambaRadixTree

- **代码层面**：直接删除了整个 `hi_mamba_radix_cache.py` 文件（约 2178 行）。这是一个专门为 HiMamba 模型设计的缓存树实现，现在已被废弃。
- **新手概念课堂**：想象你有一个专门存放某种特殊文件格式的旧书架，现在这种格式已经没人用了，书架占地方还积灰，干脆扔掉。代码删除同理——死代码（dead code）会增加维护负担和混淆。
- **对你有什么影响**：无直接影响，但代码库更干净了，编译和加载可能略微加快。

---

### 2. [diffusion] 为 LTX-2 启用可中断 CUDA 图（H200 两阶段端到端 10.75s -> 6.90s，1.56 倍）

- **代码层面**：修改了 `breakable_cuda_graph/runner.py`，将最大段数从 128 提升到 512（因为 LTX-2 的双塔结构每个 block 有 6 个注意力断点）；同时修改了 LTX-2 的 denoising 阶段，将 RoPE 坐标构建移到 CUDA 图捕获区域之外（避免未锁定的 H2D 拷贝导致捕获失败）。还新增了签名未命中时的诊断日志。
- **新手概念课堂**：CUDA 图就像**预录的舞蹈视频**——先把所有动作录好，之后每次播放都直接放视频而不是重新编排。但录制时有些动作（如从 CPU 拷贝数据到 GPU）是"违禁"的，必须提前准备好。可中断 CUDA 图进一步允许在录制中"暂停"和"恢复"，适合结构复杂的模型。
- **对你有什么影响**：如果你用 LTX-2 生成视频，速度提升明显（约 1.56 倍），等待时间大幅缩短。

---

### 3. [diffusion] 让调度器 RPC 超时显式化

- **代码层面**：新增 `--scheduler-rpc-timeout` 参数，用于设置调度器 RPC 的端到端截止时间（默认不设限）。同时改进了 HTTP 服务器关闭流程：新增 `shutdown_video_jobs()` 来取消所有正在运行的视频任务，并正确处理 broker 任务的取消。
- **新手概念课堂**：RPC（远程过程调用）就像**打电话给另一个部门**。超时就是"如果对方 30 秒不接电话就挂断"。默认不设超时意味着"等多久都行"，适合长视频任务，但如果你需要服务有界响应时间，可以设置超时。
- **对你有什么影响**：长视频任务不会被传输层误杀；服务关闭时视频任务会被正确取消，不会残留僵尸进程。

---

### 4. 修复 VAE 快速路径测试（gate 重构后）

- **代码层面**：更新了 `test_autoencoder_kl_fastpath.py` 测试，适配新的 `use_vae_fast_path()` 上下文管理器 API（替代旧的 `gate.enabled` 属性）。测试现在检查 wrapper 是否安装，并使用 `with` 语句来切换快速路径。
- **新手概念课堂**：`with` 语句就像**进出房间自动开关灯**——进入时打开，退出时自动关闭。旧 API 需要手动开关（容易忘记关），新 API 保证无论是否出错都会恢复原状。
- **对你有什么影响**：测试更健壮，防止回归 bug。

---

### 5. 修复 Nemotron W4A16 NVFP4 MoE 后端

- **代码层面**：在 `overrides.py` 中检测 `W4A16_NVFP4` 量化的 MoE 层，强制要求 `--moe-a2a-backend=none` 和 `--moe-runner-backend=marlin`（否则报错）。在 `marlin_utils_fp4.py` 中修复了权重 padding 后的维度恢复逻辑（`padded_size_k`/`padded_size_n`），确保输入正确 padding 到物理 tile 尺寸。
- **新手概念课堂**：MoE（专家混合）就像**一个公司有多个专家小组**，每个 token 只让最相关的几个专家处理。量化是把模型"压缩"以节省内存，但压缩后的格式需要特定的"解码器"（后端）才能高效运行。这个修复确保选对了解码器。
- **对你有什么影响**：使用 Nemotron 模型且开启 W4A16 NVFP4 量化时，不再崩溃或产生错误结果，且会自动选择正确的后端。

---

### 6. [diffusion] 加速 TP 和 FSDP 检查点加载

- **代码层面**：重构了 `fsdp_load.py` 的加载逻辑。新增 `rank_local_checkpoint` 路径：当不使用 FSDP 且没有预处理回调时，直接从磁盘加载 rank 本地的检查点分片（跳过 FSDP 的完整状态字典构建）。同时删除了未使用的 `_get_param_for_weight_loading` 函数。
- **新手概念课堂**：FSDP（完全分片数据并行）就像**把一本大书拆成多本小册子分给不同的人**。加载时旧方法需要先"拼回完整书"再分发，新方法直接读自己的分册，省去了拼接步骤，速度自然更快。
- **对你有什么影响**：多 GPU 部署时，模型加载时间显著缩短，启动更快。

---

### 7. [diffusion] 修复：分布式初始化前将每个 rank 绑定到对应加速器

- **代码层面**：在 `parallel_state.py` 中将 `current_platform.set_device(device)` 提前到 `init_distributed_environment()` 之前，确保在初始化分布式环境时当前设备已正确设置。同时在 `gpu_worker.py` 中使用 `current_platform.set_device()` 替代 `torch.get_device_module().set_device()`。`MPS` 平台新增了 `set_device` 空实现（MPS 不需要设置设备）。
- **新手概念课堂**：想象一个**多人在线游戏**——每个玩家（rank）需要先选好自己的角色（绑定 GPU），才能加入游戏大厅（初始化分布式环境）。如果先加入大厅再选角色，可能会选错或冲突。
- **对你有什么影响**：多 GPU 训练/推理时，设备绑定更可靠，避免"使用了错误的 GPU"这类问题。

---

### 8. [diffusion] 修复：TP 下启用可中断 CUDA 图

- **代码层面**：在 `breakable_cuda_graph/runner.py` 中新增 `_tp_graph_capture()` 上下文管理器，在捕获 CUDA 图时正确进入 TP 组的 `graph_capture` 上下文。同时修复了 `group_coordinator.py` 中 `graph_capture` 对自定义 all-reduce 的处理（区分 eager 路径和 graph 路径）。
- **新手概念课堂**：TP（张量并行）就像**多人合唱**——每个人唱一个声部，合起来才是完整歌曲。录制 CUDA 图时，需要确保所有"歌手"同步录制，否则回放时会乱套。
- **对你有什么影响**：TP 模式下也能用可中断 CUDA 图加速了，之前会报错或回退到 eager 模式。

---

### 9. 修复 Triton 后端在流水线并行下的 IndexError

- **代码层面**：在 `triton_backend.py` 和 `memory_pool.py` 中，将 `get_value_buffer(0)` 改为 `get_value_buffer(start_layer)`。在流水线并行（PP）中，当前阶段可能不包含第 0 层，所以必须使用 `start_layer` 来获取正确的 KV 缓存。
- **新手概念课堂**：流水线并行就像**工厂流水线**——每个工位只处理自己负责的工序。如果工位 A 负责第 1-10 层，工位 B 负责第 11-20 层，那么 B 去拿"第 0 层"的缓存就像去拿别人的工作台，肯定拿不到。用 `start_layer` 就是"从我负责的第一层开始拿"。
- **对你有什么影响**：使用流水线并行（多 GPU 分层部署）时不再崩溃。

---

### 10. [diffusion] 新增 /liveness 端点，/health 与 /health_generate 在预热完成前返回 503

- **代码层面**：在 `http_server.py` 中新增 `/liveness` 端点（始终返回 200，表示进程活着）；`/health` 和 `/health_generate` 在服务器预热完成前返回 503，完成后返回 200。更新了文档说明三种端点的适用场景（liveness 探针 vs 就绪探针）。
- **新手概念课堂**：Kubernetes 探针就像**体检**——liveness 探针检查"人还活着吗"（进程是否在跑），readiness 探针检查"能干活了吗"（模型加载完没）。以前两个混在一起，现在分开了，更符合 K8s 的最佳实践。
- **对你有什么影响**：部署到 K8s 时，探针配置更准确，不会因为预热时间长被误杀，也不会过早把流量打进来。

---

### 11. [diffusion] 模型：支持 LingBot-Video MoE 30B T2V

- **代码层面**：新增 `lingbot_video_moe.py` 配置文件，定义了 `LingBotVideoMoEArchConfig`（48 层、128 专家、每 token 选 8 个专家、hidden_size=2048 等），并注册到模型配置的 `__init__.py` 中。同时新增了对应的模型实现文件。
- **新手概念课堂**：MoE（专家混合）模型就像**一个大型咨询公司**——每个问题（token）只找最擅长的几个专家（专家层）处理，而不是让所有 100 个专家都参与。这样计算量不变，但参数量可以非常大，效果更好。
- **对你有什么影响**：可以直接用 SGLang 跑 LingBot-Video MoE 30B 模型做文生视频了。

---

### 12. [diffusion] 让环形注意力准入成为后端能力

- **代码层面**：在 `AttentionBackend` 基类中新增 `supports_ring_rotation()` 方法（默认返回 `False`），`FlashAttentionBackend` 和 `SageAttentionBackend` 覆写为返回 `True`。在 `attention/layer.py` 中根据该能力决定是否允许使用环形注意力。
- **新手概念课堂**：环形注意力（Ring Attention）就像**接力赛**——不同 GPU 各持有一段序列，按环形互相传递并合并结果。但不是所有注意力内核都支持这种"接力"，所以需要明确声明"我能跑接力"。
- **对你有什么影响**：只有支持环形注意力的后端才会启用该功能，避免选错后端导致错误。

---

### 13. [diffusion] 在主机端构建 Qwen 的掩码变长元数据（性能优化）

- **代码层面**：在 `qwen_image.py` 中，当 `txt_seq_lens` 可用且合法时，直接调用 `build_varlen_mask_meta_from_ranges()` 在主机端（CPU）构建变长注意力元数据，避免每次 denoising 步骤都做 GPU 非零运算和设备同步。新增了单元测试验证主机端构建与掩码构建结果一致。
- **新手概念课堂**：GPU 就像**高级厨师**，CPU 就像**助手**。以前每次做菜（denoising 步骤）都要让高级厨师花时间切菜（GPU 非零运算），现在助手提前把菜切好（CPU 构建元数据），高级厨师只需专注炒菜（注意力计算），效率更高。
- **对你有什么影响**：Qwen 图像生成速度提升，因为减少了 GPU 上的重复计算和同步开销。

---

### 14. 将 SWA chunk-cap hatch 测试移入注册测试套件

- **代码层面**：从 `test/manual/test_schedule_policy.py` 中删除了 SWA chunk-cap hatch 相关的 3 个测试类，迁移到 `test/registered/unit/managers/test_prefill_adder.py` 中（新增约 50 行测试代码）。
- **新手概念课堂**：测试套件就像**健身房的不同区域**——手动测试是"自由训练区"（需要自己安排），注册测试是"团课区"（自动排课，CI 会自动跑）。把重要测试移到自动跑的区域，防止回归。
- **对你有什么影响**：这些测试现在会在 CI 中自动运行，SWA 相关的调度逻辑有更好的保障。

---

### 15. [NPU] [文档] 升级 Ascend NPU 推荐 SGLang 版本

- **代码层面**：纯文档变更，将推荐的 Docker 镜像标签从 `v0.5.13.post1-cann9.0.0-a3` 等更新为 `cann9.0.0-a3-v0.5.16` 等新版标签。
- **新手概念课堂**：Docker 镜像标签就像**软件版本号**——`v0.5.16` 比 `v0.5.13.post1` 更新，包含更多 bug 修复和新功能。升级版本就像更新手机系统。
- **对你有什么影响**：使用 Ascend NPU 的用户应该更新到新版本镜像，以获得更好的稳定性和性能。

---

### 16. [diffusion] Z-Image 逐位精确的融合 QK-Norm（H200 Turbo 1024px 端到端 -6.4%）

- **代码层面**：新增 `_qk_rmsnorm_native_kernel` Triton 内核，通过精确模拟 aten 的 reduce 流程（8 元素串行累加 + shfl-down 蝶形归约），实现与 eager 路径**逐位一致**（`torch.equal`）的 RMSNorm。同时在 `zimage_native_norm.py` 中新增了对应的融合实现。
- **新手概念课堂**：浮点数运算的**顺序影响结果**——`(a+b)+c` 可能不等于 `a+(b+c)`。这个内核通过精确复现 PyTorch 的运算顺序（先串行加 8 个，再蝶形归约），保证结果和原来一模一样，但速度更快。就像用更快的路线到达目的地，但每一步都踩在原来的脚印上。
- **对你有什么影响**：Z-Image 生成图片速度提升 6.4%，且输出结果与之前完全一致（不会有任何数值差异）。

---

### 17. 修复预填充 CP 图溢出（更大的 bucket 搜索）

- **代码层面**：在 `bcg.py` 中新增 `required_local_tokens()` 方法（计算 zigzag 布局下 CP 所需的本地 token 数）和 `select_replay_bucket()` 方法（选择能容纳所需本地 token 的最小捕获 bucket）。修复了 CP 图选择时可能选到过小 bucket 导致溢出的问题。
- **新手概念课堂**：CP（上下文并行）就像**把一篇文章分给多个人读**——每个人读一段。zigzag 布局是特殊的分配方式。bucket 是预录好的 CUDA 图大小。以前可能选了一个太小的"录音"（bucket），导致"播放"时内存溢出，现在会先检查"录音"是否足够大。
- **对你有什么影响**：使用 CP 且开启 CUDA 图时，不再出现图溢出导致的崩溃。

---

### 18. 为 DeepSeek-V4 添加 FlashInfer mHC 融合

- **代码层面**：新增 `SGLANG_OPT_USE_FLASHINFER_MHC` 环境变量（默认关闭）。新增 `_flashinfer_hc_pre()` 函数，使用 FlashInfer 的 `mhc_pre_big_fuse` 内核替代原 TileLang 实现。包含 split-K 数量的自动选择逻辑（根据 SM 数和 hidden size 计算）。
- **新手概念课堂**：mHC 是 DeepSeek-V4 的一种特殊注意力结构。FlashInfer 是一个高性能注意力库，它提供**预融合内核**——把多个操作合并成一个 GPU 内核，减少内存读写。就像把"买菜、洗菜、切菜"合并成一步"买净菜"。
- **对你有什么影响**：DeepSeek-V4 用户可以通过设置环境变量 `SGLANG_OPT_USE_FLASHINFER_MHC=1` 尝试新的融合内核，可能获得性能提升。

---

### 19. [diffusion] 修复：将掩码路径的复制保护限定在 SP 运行

- **代码层面**：在 `attention/layer.py` 中，将复制前缀/后缀的拒绝逻辑从"只要存在复制就拒绝"改为"存在复制**且**在 SP 模式下（`get_sequence_parallel_world_size() > 1`）才拒绝"。单 rank 时不再拒绝，因为掩码已经描述了完整序列。
- **新手概念课堂**：SP（序列并行）就像**多人接力读一本长书**，每个人读一部分。如果有"复制"（同一段内容给多人读），会导致内容重复，所以需要拒绝。但只有一个人的时候，不存在"重复"问题，所以可以放行。就像一个人读书，重复读一段也没关系。
- **对你有什么影响**：单 GPU 运行 Qwen 等模型时，带复制前缀的掩码路径不再报错。

---

### 20. 修复 DSpark 中的分数模拟接受率

- **代码层面**：在 `dspark_verify.py` 中，将 `_simulated_correct_len` 从"一次性计算固定值"改为"每次调用时通过 `sample_simulated_acc_len()` 重新采样"。同时将 `_sample_simulated_acc_len` 重命名为 `sample_simulated_acc_len`（去掉下划线前缀，表示公开 API），并更新了所有调用点。
- **新手概念课堂**：模拟接受率是**测试用的假数据**——模拟"模型接受了多少候选 token"。以前这个值是固定的（比如总是接受 3 个），现在改为每次采样（可能这次 2 个下次 4 个），更接近真实情况。就像以前考试模拟总是用同一套题，现在每次随机抽题。
- **对你有什么影响**：DSpark 投机解码的基准测试结果更真实、更稳定，不会再出现"分数"（如 2.5 个 token）这种不合理的值。

---

## 💡 今日关键词

1. **CUDA 图 (CUDA Graph)**：一种 GPU 优化技术，将一系列 GPU 操作"录制"成图结构，之后可以快速"回放"，省去重复的调度开销。就像预先编排好的舞蹈，每次表演直接放录像而不是重新排练。

2. **MoE (Mixture of Experts)**：一种模型架构，包含多个"专家"子网络，每次推理只激活其中一小部分（如 8/128），在增加模型容量的同时保持计算量可控。就像大型医院，每次就诊只找最相关的几个科室。

3. **流水线并行 (Pipeline Parallelism, PP)**：一种分布式训练/推理策略，将模型的不同层分配给不同 GPU，数据像流水线一样逐层传递。就像工厂流水线，每个工位只处理自己负责的工序。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)