# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 16:03 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **AMD ROCm 生态加速成熟**：今天的 commits 中有大量针对 AMD GPU 的适配工作（WSL2 内存锁定、Qwen3.5 模型支持、Kimi-K3 性能优化、DeepSeek-V4 NVFP4 量化等）。这说明 vLLM 正在从 NVIDIA 独占走向真正的多硬件平台支持。
- **KV Cache 传输与卸载成为热点**：MoRIIO 屏障修复、NixlPush 前缀缓存修复、CPU/磁盘卸载支持等多个 commits 都在解决 KV Cache 在分布式和异构存储间的传输问题。这说明 KV Cache 管理是当前 LLM 推理性能优化的关键战场。
- **量化精度与性能的平衡**：多个 commits 涉及 FP8/INT8/NVFP4 量化（MXFP4 导入修复、INT8 MoE 配置修复、块 FP8 测试容差调整）。社区正在为不同硬件和模型寻找最优的量化方案。
- **分布式训练与推理的融合**：NCCL 权重传输、EPLB 专家并行负载均衡等 commits 表明，vLLM 正在将训练和推理的分布式能力统一起来。

## 🧪 重要知识点 & 动手实验

### 知识点 1：WSL2 下的 GPU 内存锁定（Pinned Memory）
- **是什么**：pinned memory（锁定内存）是 CPU 上不会被操作系统换页的内存，GPU 可以直接通过 DMA 访问它，无需 CPU 参与拷贝。WSL2 的某些旧内核不支持此功能。
- **为什么重要**：如果 GPU 无法使用 pinned memory，数据拷贝会变慢，影响推理性能。vLLM 现在会在 WSL2 内核版本过低时自动禁用该功能并警告用户。
- **动手试试**：
```bash
# 检查你的 WSL2 内核版本
uname -r
# 如果版本低于 4.19.121，运行
wsl --update
```
- **预期结果**：更新后内核版本提升，vLLM 不再打印 `pin_memory=False` 的警告，性能可能提升。

### 知识点 2：端口分配与预留（get_open_port）
- **是什么**：`get_open_port()` 是 vLLM 用来找一个空闲 TCP 端口的函数。数据并行（DP）模式会预留一段端口范围，如果用户指定的端口恰好落在这个范围内，函数会陷入死循环。
- **为什么重要**：分布式推理中，每个进程需要独立的端口进行通信。端口冲突或死循环会导致服务无法启动。
- **动手试试**：
```python
# 模拟问题场景
import os
os.environ["VLLM_DP_MASTER_PORT"] = "5680"
os.environ["VLLM_PORT"] = "5682"  # 在预留范围内
from vllm.utils.network_utils import get_open_port
# 修复前会卡死，修复后返回范围外的端口
print(get_open_port())
```
- **预期结果**：修复后的代码会返回一个不在 [5680, 5690) 范围内的端口（如 5682 会被跳过）。

### 知识点 3：前缀缓存（Prefix Caching）与 KV 传输
- **是什么**：前缀缓存是指多个请求共享相同的前缀 token 时，复用这些 token 的 KV Cache，避免重复计算。在分布式场景下，需要将前缀的 KV Cache 从 Producer 传输到 Consumer。
- **为什么重要**：前缀缓存可以大幅降低首 token 延迟。但如果传输逻辑有 bug（如 NixlPush 中的前缀缓存修复），可能导致缓存命中错误或数据错乱。
- **动手试试**：
```python
# 查看 vLLM 中前缀缓存相关配置
from vllm.config import KVTransferConfig
config = KVTransferConfig(kv_connector="NixlPush", kv_role="kv_producer")
print(config)
```
- **预期结果**：你会看到 KV 连接器的配置项，包括传输模式、角色等。

### 知识点 4：W8A8 vs W8A16 量化
- **是什么**：W8A8 表示权重和激活都用 8 位整数表示，W8A16 表示权重 8 位、激活 16 位。前者计算更快但精度更低。
- **为什么重要**：如果配置错误（如把动态 W8A8 误配成 W8A16），会导致精度下降或性能损失。这个 commit 修复了动态 INT8 MoE 配置被错误构建为 W8A16 的问题。
- **动手试试**：
```python
# 检查一个模型的量化配置
from vllm.model_executor.layers.fused_moe.oracle.int8 import make_int8_moe_quant_config
# 尝试传入只有 a1_scale 的情况
try:
    make_int8_moe_quant_config(a1_scale=torch.ones(1), a2_scale=None)
except ValueError as e:
    print(f"正确报错: {e}")
```
- **预期结果**：修复后，当 a1_scale 和 a2_scale 不同时出现时会抛出 ValueError，而不是静默构建错误的配置。

### 知识点 5：CPU/磁盘 KV Cache 卸载
- **是什么**：当 GPU 显存不足时，可以将 KV Cache 卸载到 CPU 内存甚至磁盘上。`SimpleCPUOffloadConnector` 现在支持 `kv_offload_backend="disk"` 选项。
- **为什么重要**：这允许在显存有限的机器上运行更大的模型或更长的上下文，但会牺牲一些性能。
- **动手试试**：
```python
# 配置磁盘卸载
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import SimpleCPUOffloadConnector
config = {
    "kv_offload_backend": "disk",
    "disk_path": "/tmp/kv_cache",
    "disk_capacity_bytes": 10 * 1024**3,  # 10GB
}
# 如果忘记设置 disk_path 会怎样？
try:
    bad_config = {"kv_offload_backend": "disk"}
    # 会抛出 ValueError
except ValueError as e:
    print(f"正确报错: {e}")
```
- **预期结果**：配置 `disk` 后端时必须提供 `disk_path`，否则会抛出清晰的错误提示。

## 🔥 重要更新

1. **Commit 1: [ROCm] WSL2 内存锁定支持** - 解决了 AMD GPU 用户在 WSL2 下性能受限的问题，自动检测内核版本并给出清晰警告。
2. **Commit 12: NVML 重复初始化修复** - 消除了每个 attention 层每次计算都会重复初始化 NVML 的性能瓶颈，对 NVIDIA GPU 用户有直接性能提升。
3. **Commit 4: NixlPush 前缀缓存修复** - 修复了分布式 KV 传输中前缀缓存命中时的数据错乱问题，对使用多节点部署的用户至关重要。
4. **Commit 17: CPU 卸载连接器支持磁盘** - 扩展了 KV Cache 卸载能力，让显存不足的用户有更多选择。

## 📋 逐条解读

### 1. [ROCm] Enable pinned memory on supported WSL2 kernels (#50126)
- **代码层面**：修改了 `vllm/platforms/rocm.py`，新增 `is_pin_memory_available()` 方法，通过检查 WSL2 内核版本决定是否启用 pinned memory。同时修复了 CI 脚本中 docker manifest 命令的引号问题。
- **新手概念课堂**：pinned memory 就像你办公桌上专门放重要文件的固定抽屉，不用每次找文件时都翻整个房间。GPU 可以直接从这个抽屉拿文件，不用等 CPU 传。但 WSL2 的某些旧版本不支持这个"抽屉"。
- **对你有什么影响**：如果你在 WSL2 上使用 AMD GPU 跑 vLLM，低版本内核会收到警告提示你更新。更新后性能可能提升。

### 2. [Bugfix] Fix get_open_port() livelock on DP-reserved ports (#50965)
- **代码层面**：修复了 `vllm/utils/network_utils.py` 中 `get_open_port()` 的死循环问题。当 `VLLM_PORT` 环境变量落在 DP 预留端口范围内时，函数会永远找不到空闲端口。新增了测试用例确保返回的端口不在预留范围内。
- **新手概念课堂**：想象你在电影院选座位，DP 模式已经预留了第 10-20 排。如果你指定的座位恰好是第 15 排，系统会一直说"这个座位被预留了"，然后继续找下一个，但每次找到的都在预留区，就卡死了。修复后系统会直接跳过整个预留区域。
- **对你有什么影响**：如果你设置了 `VLLM_PORT` 环境变量且恰好落在 DP 预留范围内，之前服务会卡住，现在会正常工作。

### 3. [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier (#48534)
- **代码层面**：修改了 `moriio_connector.py`，为 READ 模式添加了逐层的 KV 读取完成屏障。当使用完整 CUDA graph 时，屏障无法触发，现在会发出警告而不是静默强制切换模式。
- **新手概念课堂**：CUDA graph 就像一个预编排的舞蹈，所有动作都提前编排好。但 KV 读取屏障需要在舞蹈中途插入一个"等一下，确认数据到了"的动作，这在预编排的舞蹈中做不到。所以要么用分段式舞蹈（PIECEWISE），要么接受可能出错的风险。
- **对你有什么影响**：如果你使用 MoRIIO 连接器且开启了完整 CUDA graph，会收到警告，提示你改用 PIECEWISE 模式以保证准确性。

### 4. [PD][NixlPush][Bugfix] Fix prefix caching (#48758)
- **代码层面**：修复了 NixlPush 连接器中前缀缓存的部分命中问题。当消费者只预分配了未计算的块时，生产者应该只写入序列的尾部，而不是整个序列。新增了 `TestPushPrefixCaching` 测试类验证这一行为。
- **新手概念课堂**：前缀缓存就像你写论文时，如果两个同学的前半部分一样，你只需要让第二个同学从不同处开始写。但这里的问题是，传输数据时不能把共同的前半部分也传过去覆盖掉对方已经写好的内容。
- **对你有什么影响**：使用 NixlPush 进行多节点 KV 传输时，前缀缓存命中率会提高，首 token 延迟降低。

### 5. [Model] Enable Qwen3.8 for AMD Rocm (#50068)
- **代码层面**：在 `qwen3_5.py` 中为 Qwen3.5 模型添加了 `SupportsMRoPE` 支持，实现了 `get_mrope_input_positions()` 方法，使模型能在 AMD ROCm 平台上运行。
- **新手概念课堂**：MRoPE（Multi-modal Rotary Position Embedding）是一种位置编码，让模型知道每个 token 在序列中的位置。不同硬件平台可能需要不同的实现方式。
- **对你有什么影响**：AMD GPU 用户可以运行 Qwen3.5 系列模型了。

### 6. [Bugfix] Skip fetching revision for model when model and weights_model are different (#51260)
- **代码层面**：修改了 `vllm/config/model.py`，当 `model_weights` 或 `hf_config_path` 与 `model` 不同时，不再尝试解析模型 revision，避免不必要的网络请求。
- **新手概念课堂**：revision 是 HuggingFace 上模型的一个版本快照。当你指定不同的权重仓库时（如 GGUF 量化版），再去获取原始模型的 revision 没有意义，因为权重来自另一个仓库。
- **对你有什么影响**：使用 `model_weights` 参数加载不同仓库的权重时，启动速度会更快，不会因为无意义的 revision 解析而卡顿。

### 7. Fix ROCm architecture import on non-ROCm platforms (#51357)
- **代码层面**：修复了 `mxfp4.py` 和 `fused_moe/oracle/mxfp4.py` 中无条件导入 `vllm.platforms.rocm` 的问题。现在会先检查 `current_platform.is_rocm()` 再导入。
- **新手概念课堂**：这就像你家的门锁说明书，以前不管谁进门都要先看 AMD 专用说明书，但 Intel 用户根本不需要。现在会先确认是不是 AMD 用户再给对应说明书。
- **对你有什么影响**：非 ROCm 平台（如 NVIDIA、Intel）在使用 MXFP4 量化时不会再因为误导入 ROCm 模块而报错。

### 8. feat: extended EPLB support for Mistral Large 3 and additional MoE backends (#48355)
- **代码层面**：扩展了 EPLB（Expert Parallel Load Balancing）对 Mistral Large 3 模型和更多 MoE 后端的支持。新增了测试文件验证量化方法在 EPLB 重排后的一致性。
- **新手概念课堂**：EPLB 是让 MoE 模型中不同"专家"（expert）的负载均衡的技术。就像餐厅里，如果某个厨师特别忙，就把一些菜品分给其他厨师。量化则像是把菜谱从精装版改成简装版以节省空间。
- **对你有什么影响**：使用 Mistral Large 3 且开启 EPLB 时，负载均衡效果更好，推理吞吐量可能提升。

### 9. [XPU] quick fix online quantization UT break (#51365)
- **代码层面**：修复了 `tests/quantization/test_online.py` 中的测试，将硬编码的 `device="cuda"` 改为使用 `current_platform.device_type`，并调整了 NVFP4 的跳过条件以支持 XPU。
- **新手概念课堂**：测试代码以前假设所有 GPU 都是 NVIDIA，现在改为动态获取当前平台类型。就像测试脚本以前只支持 Windows，现在也能在 Mac 和 Linux 上跑了。
- **对你有什么影响**：Intel XPU 用户可以运行在线量化测试了。

### 10. [Misc] Add and enable Triton kernel unit tests on XPU (#45694)
- **代码层面**：修改了多个 Triton 内核测试文件，将 `cuda:0` 替换为平台相关的 `DEVICE` 变量，并为 INT8 内核测试添加了 XPU 支持条件。
- **新手概念课堂**：Triton 是一种 GPU 编程语言，类似 CUDA 但更高级。以前这些内核测试只在 NVIDIA GPU 上跑，现在 Intel XPU 也能跑同样的测试了。
- **对你有什么影响**：Intel XPU 用户可以获得更好的内核测试覆盖，问题发现更早。

### 11. [PD][PushConnector] Record last activity of remotes to allow clean up of stale ones (#50234)
- **代码层面**：在 `base_worker.py` 中为 push 模式添加了 `_engine_last_active` 记录，当远程引擎超过 TTL 未活动时会被清理。新增测试验证活动推送会刷新时间戳，过期引擎会被驱逐。
- **新手概念课堂**：这就像分布式系统中的"心跳检测"。每个节点定期发送"我还活着"的信号，如果一段时间没收到，就认为节点挂了并清理相关资源。
- **对你有什么影响**：长时间运行的分布式部署中，不再使用的远程引擎会被自动清理，避免资源泄漏。

### 12. [Bugfix][Platform] Stop re-initializing NVML on every device-capability check (#50393)
- **代码层面**：修复了 `vllm/platforms/cuda.py` 中 `has_device_capability` 方法重复初始化 NVML 的问题。之前每次调用都会执行 `nvmlInit()`/`nvmlShutdown()`，现在只读取缓存的 `get_device_capability()` 结果。
- **新手概念课堂**：NVML 是 NVIDIA 的 GPU 管理库。以前每次检查 GPU 算力都要重新打开和关闭这个库，就像每次查电话号码都要重新翻一遍电话簿。现在查一次记下来，之后直接看笔记。
- **对你有什么影响**：使用 FP8 或 BF16 KV Cache 时，每个 attention 层每步都会检查 GPU 能力，修复后性能显著提升。

### 13. [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16 (#50833)
- **代码层面**：修复了 `fused_moe/oracle/int8.py` 中动态 INT8 MoE 配置被错误构建为 W8A16 的问题。现在当 `per_act_token_quant=True` 且没有 scale 时会正确构建 W8A8 配置。
- **新手概念课堂**：W8A8 和 W8A16 是两种量化方案。W8A8 更快但精度略低，W8A16 更慢但更精确。之前代码在动态量化场景下错误地选择了 W8A16，导致性能下降。
- **对你有什么影响**：使用动态 INT8 量化的 MoE 模型推理速度会提升。

### 14. [Refactor] Remove kernel dead code (#51051)
- **代码层面**：删除了 `csrc/cpu/cpu_attn_fp8.hpp` 和 `csrc/libtorch_stable/cache_kernels.cu` 中的未使用代码，包括 `fp8e5m2_to_float_scalar` 函数和 `copy_blocks_kernel` 等。
- **新手概念课堂**：dead code 就像你衣柜里很久没穿的衣服，占地方但没用。删掉它们可以让代码更干净、编译更快、维护更容易。
- **对你有什么影响**：代码库更简洁，编译时间可能略有缩短。

### 15. Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel (#47972)
- **代码层面**：为 DeepSeek-V4 在 AMD GPU 上添加了 NVFP4 量化支持。新增了测试配置文件和 CI 步骤，使用 AITER 内核模拟 NVFP4 量化。
- **新手概念课堂**：NVFP4 是 NVIDIA 的一种 4 位浮点格式。AMD GPU 没有原生支持，所以用模拟内核来近似实现。就像用软件模拟器在 PC 上玩 Switch 游戏。
- **对你有什么影响**：AMD GPU 用户现在可以运行 DeepSeek-V4 并享受 NVFP4 量化的内存节省。

### 16. [ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes (#48847)
- **代码层面**：调整了 `tests/kernels/moe/test_block_fp8.py` 的测试容差，并为大 K 形状添加了 `silu_fp32` 参数来匹配不同内核的精度差异。
- **新手概念课堂**：测试容差就是允许的误差范围。不同内核实现（如 Triton 和 fused_experts）在计算 SiLU 激活函数时精度略有不同，测试需要容忍这种差异。
- **对你有什么影响**：AMD GPU 上的块 FP8 MoE 测试不再因微小精度差异而误报失败。

### 17. [Feat][Core] Add disk offloading support to SimpleCPUOffloadConnector (#49644)
- **代码层面**：为 `SimpleCPUOffloadConnector` 添加了 `kv_offload_backend="disk"` 选项，支持将 KV Cache 卸载到磁盘。新增了 `disk_path`、`disk_capacity_bytes` 等配置项。
- **新手概念课堂**：当 GPU 和 CPU 内存都不够用时，可以把 KV Cache 存到硬盘上。硬盘最慢但容量最大，适合处理超长上下文。
- **对你有什么影响**：显存和内存都不足的用户可以通过磁盘卸载运行更长上下文的模型，但性能会有所下降。

### 18. [rl] Stateful Trainer Send: NCCL + Sparse NCCL [3/N] (#50902)
- **代码层面**：重构了 RL 训练器的权重传输 API。新增了 `WeightTransferTrainerFactory` 和 `RayVLLMWeightSyncClient` 等抽象，简化了训练器向推理引擎发送权重的流程。
- **新手概念课堂**：在 RLHF 训练中，训练器需要不断更新推理引擎的权重。以前这个流程很复杂，现在封装成了更简单的 API，就像把复杂的遥控器换成了几个大按钮。
- **对你有什么影响**：使用 vLLM 做 RL 训练时，权重同步代码更简洁，更容易上手。

### 19. [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path (#51253)
- **代码层面**：为 ROCm 平台添加了 Kimi-K3 模型的 latent MoE up-projection 分片实现。新增了 `ROCmLatentMoERunner` 和测试文件验证分片与复制的 up-projection 结果一致。
- **新手概念课堂**：MoE 模型中的 up-projection 层将 latent 空间映射回原始空间。分片实现可以将这个计算分散到多个 GPU 上，但需要确保结果与不分片时完全一致。
- **对你有什么影响**：AMD GPU 用户运行 Kimi-K3 时，up-projection 计算速度提升。

### 20. [Bugfix] Fix Mamba all-mode CPU offload boundary alignment (#51100)
- **代码层面**：修复了 `offloading/scheduler.py` 中 Mamba 模型在 "all" cache 模式下的边界对齐问题。之前只有 "align" 模式会进行对齐，现在 "all" 模式也需要。
- **新手概念课堂**：Mamba 模型的状态与普通 Transformer 不同，它只有一个状态值。当处理到块边界时，需要特殊处理避免重复计算。对齐操作让缓存块与模型状态边界一致。
- **对你有什么影响**：使用 Mamba 模型并开启 CPU 卸载时，边界处理更准确，避免输出错误。

## sglang
## 💡 今日关键词

- **关键词：CUDA Graph 可断点化（Breakable CUDA Graph）**
  - 一句话通俗解释：把一次完整的模型推理拆成多个小段，每段分别用 CUDA Graph 录制和重放，避免整体录制时因动态形状而失效。
  - 为什么社区在关注：今天多个 commit（SANA、LTX-2、Z-Image）都在为不同扩散模型启用这一技术，说明它已成为扩散模型推理性能优化的核心手段，且能带来 20%-50% 的端到端加速。
  - 对新手意味着什么：学习 CUDA Graph 时，不要只停留在"整体录制"的层面，理解"分段录制 + 动态形状处理"是进阶的关键；这也是未来模型推理优化的主流方向之一。

- **关键词：扩散模型（Diffusion Models）推理优化**
  - 一句话通俗解释：扩散模型（如 SANA、LTX-2、Z-Image、LingBot）是生成图像/视频的 AI 模型，优化其推理速度是当前热点。
  - 为什么社区在关注：今天的 20 个 commit 中超过一半与扩散模型相关，涵盖 CUDA Graph、加载加速、健康检查、新模型适配（LingBot MoE 30B）、注意力后端能力抽象等。这说明 SGLang 正在从纯 LLM 推理框架扩展为多模态生成推理平台。
  - 对新手意味着什么：如果你对"用 AI 生成图片/视频"感兴趣，现在正是学习扩散模型推理优化的好时机，SGLang Diffusion 子项目提供了完整的工程实践案例。

- **关键词：张量并行（Tensor Parallelism, TP）与 CUDA Graph 的兼容**
  - 一句话通俗解释：在多 GPU 上做张量并行时，各 GPU 之间需要通信；CUDA Graph 录制时如果包含通信操作，可能导致重放时出错。
  - 为什么社区在关注：Commit #33421 专门修复了 TP 下 CUDA Graph 捕获的问题，说明分布式推理与 CUDA Graph 的结合是工程上的难点，也是大规模部署的刚需。
  - 对新手意味着什么：理解"通信与计算重叠"、"图捕获上下文"是分布式推理进阶的必修课；学会在 TP 环境中调试 CUDA Graph 是重要的实战技能。

- **关键词：MoE（Mixture of Experts）量化与推理后端**
  - 一句话通俗解释：MoE 模型（如 DeepSeek）有多个"专家"子网络，量化（如 NVFP4）可以压缩模型大小，但需要专门的推理后端（如 Marlin）来加速。
  - 为什么社区在关注：Commit #33543 修复了 Nemotron W4A16 NVFP4 MoE 后端，Commit #33616 为 DeepSeek-V4 添加了 FlashInfer 融合核。说明 MoE + 量化 + 专用核的"组合拳"是当前大模型部署的硬核技术。
  - 对新手意味着什么：理解 MoE 的稀疏计算模式、量化格式（NVFP4）与内核选择（Marlin vs Triton）之间的关系，是走向高性能推理工程师的必经之路。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：CUDA Graph 可断点化（Breakable CUDA Graph）
- **是什么**：把模型前向传播拆成多个"断点"，每个断点之间用 CUDA Graph 录制，从而允许在断点处插入动态操作（如不同的序列长度），同时保留图加速。
- **为什么重要**：整体 CUDA Graph 录制要求所有输入形状固定，这在扩散模型（每步去噪的 latent 可能变化）中不适用。可断点化让 CUDA Graph 在动态场景下也能用，是性能提升的关键。
- **动手试试**：
  ```python
  # 观察 SGLang 中可断点化 CUDA Graph 的配置
  # 运行一个扩散模型并启用 breakable CUDA graph
  python -m sglang.multimodal_gen.launch --model efficient-large-model/sana1.5_1.6b_1024px_diffusers --enable-breakable-cuda-graph
  # 查看日志中的 "Diffusion BCG" 相关输出，了解分段和捕获情况
  ```
- **预期结果**：你会看到日志显示捕获了多个图段，端到端延迟显著下降（如 SANA 的 -26%）。掌握后你就能理解为什么"动态形状 + CUDA Graph"不是矛盾，而是可以通过分段解决。

### 知识点 2：张量并行下的 CUDA Graph 捕获（TP + CUDA Graph）
- **是什么**：在 TP 环境中，每个 GPU 负责模型的一部分，需要跨 GPU 通信（如 all-reduce）。CUDA Graph 捕获时，通信操作必须使用"图路径"而非"即时路径"，否则重放时出错。
- **为什么重要**：如果你想在多 GPU 上部署模型并启用 CUDA Graph，必须正确处理通信操作的捕获上下文。否则会出现"捕获成功、重放崩溃"的诡异问题。
- **动手试试**：
  ```bash
  # 在 TP=2 环境下启用 CUDA Graph
  python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --tp 2 --cuda-graph-max-bs 4
  # 观察是否正常启动和推理
  ```
- **预期结果**：如果没有正确的图捕获上下文，你可能会看到 CUDA 错误（如 "illegal memory access"）或结果错误。修复后（如 commit #33421 所示），TP + CUDA Graph 能正常工作，且性能提升明显。

### 知识点 3：MoE 量化推理后端（NVFP4 + Marlin）
- **是什么**：NVFP4 是一种 4-bit 浮点量化格式；Marlin 是一种针对稀疏/量化矩阵乘法优化的 GPU 内核。MoE 模型的专家层用 NVFP4 量化后，需要 Marlin 后端来高效计算。
- **为什么重要**：如果选错后端（如用 Triton 而非 Marlin），推理速度会大幅下降甚至出错。Commit #33543 专门修复了这个问题，说明后端选择是部署 MoE 量化模型的关键决策。
- **动手试试**：
  ```bash
  # 启动一个 NVFP4 量化的 MoE 模型，指定 Marlin 后端
  python -m sglang.launch_server --model-path nvidia/Nemotron-H-4B --quantization modelopt_mixed --moe-runner-backend marlin --moe-a2a-backend none
  # 对比使用默认后端时的性能差异
  ```
- **预期结果**：使用 Marlin 后端时，推理速度明显快于默认后端，且结果正确。如果强制使用不兼容的后端，会出现报错（如 commit 中新增的 ValueError）。

### 知识点 4：varlen 注意力元数据的主机端构建（Host-side Metadata）
- **是什么**：在注意力计算中，需要为变长序列构建 cu_seqlens、indices 等元数据。传统方法在 GPU 上用 nonzero 操作构建，会引入设备同步；新方法直接在主机端（CPU）用 Python 构建，避免同步开销。
- **为什么重要**：在扩散模型的每一步去噪中，都要重新构建注意力元数据，GPU 同步会成为性能瓶颈。主机端构建能显著降低每步延迟。
- **动手试试**：
  ```python
  import torch
  from sglang.multimodal_gen.runtime.layers.attention.layer import (
      build_varlen_mask_meta,
      build_varlen_mask_meta_from_ranges,
  )
  
  # 构造一个 mask 和对应的 ranges
  txt_len, img_len = 7, 5
  txt_seq_lens = [3, 7, 0]
  mask = torch.zeros(len(txt_seq_lens), txt_len + img_len, dtype=torch.bool)
  for row, n in enumerate(txt_seq_lens):
      mask[row, :n] = True
      mask[row, txt_len:] = True
  
  ref = build_varlen_mask_meta(mask)
  host = build_varlen_mask_meta_from_ranges(
      [[(0, n), (txt_len, txt_len + img_len)] for n in txt_seq_lens],
      max_seqlen=txt_len + img_len,
      device=mask.device,
  )
  
  # 验证两者结果一致
  for key in ("cu_seqlens", "indices", "inv_indices"):
      assert torch.equal(host[key], ref[key]), f"{key} mismatch!"
  print("Host-side and GPU-side metadata match!")
  ```
- **预期结果**：脚本输出 "Host-side and GPU-side metadata match!"，证明两种构建方式结果完全一致。掌握后你就理解了为什么"避免 GPU 同步"是性能优化的关键。

### 知识点 5：健康检查与存活探针（Health vs Liveness）
- **是什么**：`/health` 表示服务"就绪"（可接收推理请求），`/liveness` 表示进程"存活"（没有崩溃）。两者分离，避免在长启动过程中被误杀。
- **为什么重要**：在 Kubernetes 部署中，如果只用 `/health` 做存活探针，服务启动时需要几分钟加载模型，探针会误判为"不健康"并重启容器，导致永远无法启动完成。
- **动手试试**：
  ```bash
  # 启动 SGLang Diffusion 服务（warmup 模式开启）
  python -m sglang.multimodal_gen.launch_server --model-path comfy-org/ideogram-4 --warmup-mode server
  
  # 在另一个终端观察探针状态
  curl -s -o /dev/null -w "%{http_code}" http://localhost:30010/liveness   # 立即返回 200
  curl -s -o /dev/null -w "%{http_code}" http://localhost:30010/health     # 可能返回 503（warmup 中）
  sleep 60
  curl -s -o /dev/null -w "%{http_code}" http://localhost:30010/health     # 最终返回 200
  ```
- **预期结果**：`/liveness` 始终返回 200，`/health` 在 warmup 期间返回 503，完成后返回 200。掌握后你就能正确配置 K8s 探针，避免服务被误杀。

---

## 🔥 重要更新

1. **扩散模型全面启用可断点 CUDA Graph**（Commit #33989、#33885、#33886）：SANA、LTX-2、Z-Image 三个模型都获得了显著的端到端加速（26%、1.56x、6.4%），说明这项技术已成熟，是扩散模型性能优化的标配。

2. **TP 环境下 CUDA Graph 捕获修复**（Commit #33421）：解决了多 GPU 并行 + CUDA Graph 的兼容性问题，让大规模分布式部署也能享受图加速，是生产环境的关键修复。

3. **MoE 量化后端修复**（Commit #33543）：Nemotron W4A16 NVFP4 MoE 模型的推理后端得到了修正，并增加了参数校验（如必须使用 `--moe-a2a-backend=none`），避免用户误配置。

4. **健康检查与存活探针分离**（Commit #33787）：新增 `/liveness` 端点，`/health` 在 warmup 期间返回 503。这对 Kubernetes 部署至关重要，避免了长启动服务被探针误杀的问题。

---

## 📋 逐条解读

### 1. [diffusion] Enable breakable CUDA graph for SANA (H200 1024px e2e -26%, bit-exact)
- **代码层面**：去掉了 `denoising.py` 和 `denoising_dmd.py` 中硬编码的 `mask_strategy` 参数（原本是 `None` 的 3D 列表），并在 `server_args.py` 中将 SANA 模型加入 `BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS` 集合。
- **新手概念课堂**：CUDA Graph 就像把一段舞蹈录制成视频，播放时不需要演员重新跳。但录制要求每个动作完全一样，如果中间有人临时改动作就失效了。"可断点"就是把舞蹈拆成多段录制，每段之间可以插入不同的动作。SANA 模型之前因为动态 mask 无法整体录制，现在去掉这个动态参数后就能分段录制了。
- **对你有什么影响**：如果你用 SANA 生成图片，速度会提升约 26%，且生成结果与之前完全一致（bit-exact）。

### 2. Remove the HiMambaRadixTree that is no longer in use
- **代码层面**：直接删除了一个 2178 行的文件 `hi_mamba_radix_cache.py`，这个文件是 Mamba 模型的缓存管理实现，但已不再被使用。
- **新手概念课堂**：代码库就像仓库，时间久了会有一些不再使用的旧设备占地方。删除无用代码（dead code）是保持仓库整洁的重要工作，但需要确认没有其他代码依赖它。
- **对你有什么影响**：普通用户无感知，但代码库更简洁，维护成本降低。

### 3. [diffusion] Enable breakable CUDA graph for LTX-2 (H200 two-stage e2e 10.75 s -> 6.90 s)
- **代码层面**：将 `max_segments` 默认值从 128 提升到 512（因为 LTX-2 有 48 个块，每个块有 6 个注意力断点，共约 289 段）。同时在 LTX-2 的 denoising 逻辑中，将 RoPE 坐标构建移到了 CUDA Graph 捕获区域之外（因为捕获区域不允许 H2D 拷贝）。
- **新手概念课堂**：RoPE 是给 Transformer 注入位置信息的方法，需要根据序列长度动态计算坐标。CUDA Graph 捕获时不能做"从 CPU 拷贝数据到 GPU"的操作，所以要把这类动态计算放在捕获之前完成。
- **对你有什么影响**：LTX-2 视频生成速度提升 1.56 倍（10.75 秒 → 6.90 秒），且结果完全相同。

### 4. [diffusion] feat: make scheduler rpc deadlines explicit
- **代码层面**：新增 `--scheduler-rpc-timeout` 参数，默认不设置（避免长视频任务被超时中断）；在 HTTP 服务器关闭时，主动取消所有视频生成任务（`shutdown_video_jobs`），并等待 broker 任务结束。
- **新手概念课堂**：RPC（远程过程调用）就像打电话，需要设置超时避免一直等待。但视频生成任务可能很长，默认超时会误杀任务。现在把超时设为可选，只在需要时设置。
- **对你有什么影响**：长视频生成任务不会被意外中断；服务器关闭时，未完成的视频任务会被正确取消，不会留下僵尸进程。

### 5. Fix vae fast path test after the gate refactor
- **代码层面**：更新了 VAE（变分自编码器）快速路径的单元测试，改为使用 `use_vae_fast_path(opt, True)` 上下文管理器来切换快速路径，而不是直接操作内部的 `gate.enabled` 属性。
- **新手概念课堂**：VAE 是扩散模型中负责"压缩/解压"图像的部分。快速路径（fast path）是经过优化的 CUDA 内核。测试代码需要跟随实现的变化而更新。
- **对你有什么影响**：无用户可见影响，但保证了测试的可靠性。

### 6. Fix Nemotron W4A16 NVFP4 MoE backend
- **代码层面**：在 `overrides.py` 中新增了对 W4A16_NVFP4 MoE 层的检测，强制要求 `--moe-a2a-backend=none` 和 `--moe-runner-backend=marlin`（否则报错）；在 `marlin_utils_fp4.py` 中处理了 TP 分片后维度不对齐的问题（通过 padding 输入）。
- **新手概念课堂**：NVFP4 是一种 4-bit 浮点格式，Marlin 是专用 GPU 内核。当模型过大需要切分到多 GPU 时，每个 GPU 上的权重维度可能不是内核要求的对齐大小，需要 padding（补零）到正确尺寸。
- **对你有什么影响**：如果你使用 Nemotron 模型的 NVFP4 量化版本，推理结果会更准确，且避免因后端选择不当而崩溃。

### 7. [diffusion] UX: speed up tp and fsdp checkpoint loading
- **代码层面**：重构了 FSDP 模型加载流程。当不需要预处理且非 bitsandbytes 量化时，直接使用 `rank_local_checkpoint` 加载本地分片，而不是从完整权重迭代器中逐层读取，大幅减少磁盘 I/O 和内存占用。
- **新手概念课堂**：FSDP（完全分片数据并行）会把模型权重切分到多 GPU。加载时，原本需要每张卡读取全部权重再丢弃不属于自己的部分，现在可以直接读取自己的分片，就像"只下载自己需要的文件"而不是"下载整个压缩包再解压"。
- **对你有什么影响**：多 GPU 部署时，模型加载时间显著缩短（尤其是大模型），启动更快。

### 8. [diffusion] fix: bind each rank to accelerator before distributed init
- **代码层面**：将设备绑定操作（`current_platform.set_device(device)`）提前到 `init_distributed_environment` 之前，并统一通过平台抽象接口（`current_platform.set_device`）处理不同硬件（CUDA/NPU/MPS）。
- **新手概念课堂**：分布式初始化时，每个进程需要先"绑定"到自己的 GPU，再进行通信组的建立。如果顺序反了，可能导致进程使用了错误的 GPU。
- **对你有什么影响**：多 GPU 启动时更稳定，尤其是混合硬件环境（如部分 NPU 部分 CUDA）下不会出错。

### 9. [diffusion] fix: enable bcg with tp
- **代码层面**：在 CUDA Graph 捕获时，如果启用了 TP，则先进入 TP 组的 `graph_capture` 上下文（`_tp_graph_capture`），确保 TP 通信操作（如 all-reduce）使用图路径而非即时路径。
- **新手概念课堂**：TP 中的 all-reduce 有两种实现：即时执行和图内执行。图内执行需要特殊的缓冲区注册。如果不进入正确的上下文，图重放时 all-reduce 会访问未映射的内存地址，导致崩溃。
- **对你有什么影响**：TP + CUDA Graph 现在可以同时使用，多 GPU 部署也能享受图加速。

### 10. Fix IndexError in Triton backend with pipeline parallelism
- **代码层面**：在 `triton_backend.py` 和 `memory_pool.py` 中，将获取 `v_head_dim` 时硬编码的 `layer 0` 改为 `start_layer`（流水线并行中当前阶段的第一层）。
- **新手概念课堂**：流水线并行（PP）把模型分成多段，每张 GPU 只负责一段。之前代码假设每张卡都有第 0 层，但 PP 中第 2 张卡可能从第 20 层开始，访问第 0 层就出错了。
- **对你有什么影响**：使用 PP + Triton 注意力后端时，不再出现 IndexError，推理正常。

### 11. [diffusion] feat: gate /health and /health_generate on warmup completion and add liveness endpoint
- **代码层面**：新增 `/liveness` 端点（始终返回 200），`/health` 在 warmup 完成前返回 503，完成后返回 200。更新了文档，说明了 K8s 探针配置方式。
- **新手概念课堂**：K8s 探针有三种：liveness（进程是否存活）、readiness（是否可接收流量）、startup（启动是否完成）。之前 `/health` 同时承担了 liveness 和 readiness 的职责，导致长启动时被误杀。
- **对你有什么影响**：K8s 部署更稳定，服务启动期间不会被误杀，且能正确感知就绪状态。

### 12. [diffusion] model: support lingbot-video moe 30b t2v
- **代码层面**：新增 `LingBotVideoMoEConfig` 配置类（48 层、128 专家、每 token 激活 8 个专家），并注册到模型配置工厂。这是对 LingBot 视频生成 MoE 模型的支持。
- **新手概念课堂**：MoE（混合专家）模型有多个"专家"子网络，每个 token 只激活少数专家，计算量大大降低。LingBot 是一个视频生成模型，30B 参数但推理时只用到一部分。
- **对你有什么影响**：你可以用 SGLang 部署 LingBot 30B 视频生成模型，体验 MoE 架构的高效推理。

### 13. [diffusion] feat: make ring admission a backend capability
- **代码层面**：在 `AttentionBackend` 基类中新增 `supports_ring_rotation()` 方法（默认返回 False），FlashAttention 和 SageAttention 后端返回 True。Ring Attention 只在支持的后端上启用。
- **新手概念课堂**：Ring Attention 是一种处理超长序列的分布式注意力方法，把序列分到多 GPU 上环形传递。但它的在线 softmax 合并需要内核暴露 LSE（log-sum-exp），不是所有后端都支持。
- **对你有什么影响**：Ring Attention 只会在合适的后端上启用，避免因后端不支持导致错误。

### 14. [diffusion] perf: build qwen's masked varlen metadata host-side
- **代码层面**：在 Qwen-Image 模型中，当 `txt_seq_lens` 可用时，直接在主机端用 `build_varlen_mask_meta_from_ranges` 构建注意力元数据，避免 GPU 上的 nonzero 操作和设备同步。
- **新手概念课堂**：GPU 同步是指 CPU 等待 GPU 完成计算，非常耗时。之前每步去噪都要在 GPU 上算 mask 的非零元素位置，现在在 CPU 上直接计算，省去了同步。
- **对你有什么影响**：Qwen-Image 生成速度提升（减少每步延迟），且结果与之前完全一致。

### 15. Move SWA chunk-cap hatch tests into the registered suite
- **代码层面**：将 SWA（滑动窗口注意力）相关的 `PrefillAdder` 测试从 `test/manual/` 移动到 `test/registered/unit/managers/test_prefill_adder.py`，并添加了更完整的测试用例。
- **新手概念课堂**：SWA 是一种只关注最近 token 的注意力机制。测试移动意味着这些测试现在会在 CI 中自动运行，防止回归。
- **对你有什么影响**：无用户可见影响，但代码质量更有保障。

### 16. [NPU] [DOC] Upgrade recommendeded sglang version on Ascend NPU
- **代码层面**：更新文档中的 Docker 镜像标签，从 `v0.5.13.post1-cann9.0.0-a3` 升级到 `cann9.0.0-a3-v0.5.16`。
- **新手概念课堂**：Ascend NPU 是华为的 AI 芯片，SGLang 支持在其上运行。镜像标签格式从 `v版本-cann版本` 改为 `cann版本-v版本`，更清晰地标识 CANN（NPU 计算库）版本。
- **对你有什么影响**：如果你在 Ascend NPU 上部署，使用新标签能获得更新的功能和修复。

### 17. [diffusion] Z-Image bit-exact fused qk-norm (H200 Turbo 1024px e2e -6.4%)
- **代码层面**：新增 Triton 内核 `_qk_rmsnorm_native_kernel`，用融合方式计算 QK 归一化（RMSNorm），并精确模拟了 eager 路径的浮点运算顺序，保证 bit-exact。
- **新手概念课堂**：RMSNorm 是 Transformer 中常用的归一化方法。融合（fused）是指把多个操作合并到一个内核中，减少内核启动开销。bit-exact 是指结果与原始实现逐位相同，这对于保证模型输出一致性很重要。
- **对你有什么影响**：Z-Image 生成速度提升 6.4%，且结果与之前完全一致。

### 18. Fix prefill CP graph overflow with larger bucket search
- **代码层面**：在 CP（上下文并行）的 CUDA Graph 管理中，新增 `required_local_tokens` 方法计算 zigzag 布局下每个 rank 所需的本地 token 数，并在选择重放 bucket 时考虑这个约束，防止溢出。
- **新手概念课堂**：CP 是把序列分到多 GPU 上并行处理。zigzag 布局是一种负载均衡的切分方式。CUDA Graph 的 bucket 是预分配的固定大小缓冲区，如果实际需求超过 bucket 容量就会溢出。
- **对你有什么影响**：长序列的 prefill 在 CP + CUDA Graph 下不再崩溃，结果正确。

### 19. feat: Add flashinfer mHC fusion for DSV4
- **代码层面**：为 DeepSeek-V4 新增 FlashInfer 的 mHC（multi-head Compression）融合实现，通过环境变量 `SGLANG_OPT_USE_FLASHINFER_MHC` 控制（默认关闭）。实现了自动选择 split-K 数量，并调用 `mhc_pre_big_fuse` 内核。
- **新手概念课堂**：mHC 是 DeepSeek 模型中用于压缩 KV cache 的技术。FlashInfer 是一个高性能注意力库，提供了融合内核。split-K 是把矩阵乘法按 K 维度切分到多个 GPU 块上并行计算。
- **对你有什么影响**：如果你使用 DeepSeek-V4 并开启该选项，推理速度可能进一步提升（取决于硬件配置）。

### 20. [diffusion] fix: scope the masked-path replicated guard to sp runs
- **代码层面**：将 `USPAttention` 中拒绝复制前缀/后缀的检查限定在"序列并行（SP）且世界大小 > 1"的条件下，单 rank 时允许复制参数。
- **新手概念课堂**：USPAttention 是支持序列并行的注意力层。在 SP 下，序列被切分到多 GPU，复制的前缀/后缀会在每张卡上重复，导致结果错误。但在单卡上，mask 已经描述了完整序列，复制参数没有意义但也不应报错。
- **对你有什么影响**：单卡运行时不再出现误报的 NotImplementedError，推理正常。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)