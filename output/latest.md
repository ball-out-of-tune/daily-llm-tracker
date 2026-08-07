# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 15:04 UTC | AI 解读: ✅

## vllm
## 🔥 重要更新（如果有的话）

本周没有对普通用户影响特别巨大的“重磅更新”，但有几个值得关注的趋势：**AMD ROCm 支持持续增强**（多个模型和优化适配）、**Model Runner V2 加速推进**（多个功能补齐与优化）、以及 **KV 传输/卸载相关 Bug 修复**（分布式场景稳定性提升）。

---

## 📋 逐条解读

### 1. [Bugfix] Fix get_open_port() livelock on DP-reserved ports and cover get_open_ports_list
- **做了什么**：修复了在 DP（数据并行）预留端口场景下，获取可用端口时可能陷入死循环的问题，并补充了相关测试。
- **涉及概念**：`get_open_port()` 是用于在分布式训练/推理中动态获取空闲端口（网络通信入口）的工具函数。**比喻**：就像多人合租时抢唯一空闲的卫生间，如果锁坏了，大家会一直敲门死等。
- **影响**：使用数据并行（DP）部署 vLLM 的用户，启动时更稳定，不会再卡在端口分配上。

### 2. [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier in wait_for_layer_load
- **做了什么**：修复了 KV 传输组件 MoRIIO 中，等待每层数据加载完成时的同步屏障（barrier）逻辑，确保每一层都真正读完才继续。
- **涉及概念**：KV-transfer 用于多机多卡间传输 KV Cache（注意力缓存）。**比喻**：流水线上每个工位（层）必须确认自己手里的零件（数据）拿全了，才能开始组装下一步。
- **影响**：使用 MoRIIO 做 KV 传输的分布式推理用户，多卡间的数据一致性更强，减少因“没读完就开工”导致的错误。

### 3. [PD][NixlPush][Bugfix] Fix prefix caching
- **做了什么**：修复了 PD 分离架构下，NixlPush 连接器中前缀缓存（prefix caching）失效的问题。
- **涉及概念**：PD 分离（Prefill/Decode 分离）是把“预填充”和“生成”阶段放到不同实例上。前缀缓存是复用相同提示词前缀的计算结果。**比喻**：做菜时，如果客人点的前菜一样，就不用每桌都重新炒一遍。
- **影响**：使用 PD 分离 + NixlPush 的用户，前缀缓存重新生效，相同前缀请求的响应速度会明显提升。

### 4. [Model] Enable Qwen3.8 for AMD Rocm
- **做了什么**：让 Qwen3.8 模型在 AMD ROCm 平台上可以正常运行。
- **涉及概念**：ROCm 是 AMD 的 GPU 计算平台，相当于 NVIDIA 的 CUDA。**比喻**：原来只给 NVIDIA 显卡写的游戏，现在也适配了 AMD 显卡。
- **影响**：AMD GPU 用户现在可以直接跑 Qwen3.8 模型，无需等待额外适配。

### 5. [Bugfix] Skip fetching revision for model when model and weights_model are different
- **做了什么**：当模型与权重来源模型不同时，跳过获取模型版本（revision）的操作，避免不必要的报错。
- **涉及概念**：revision 是模型仓库（如 HuggingFace）中某个版本的标识。**比喻**：你点了一份“牛肉面”，但指定用“另一家店的牛肉”，那就没必要去查“牛肉面店”的菜单版本。
- **影响**：使用自定义模型+外部权重的用户，加载过程更顺畅，减少无意义的报错。

### 6. Fix ROCm architecture import on non-ROCm platforms
- **做了什么**：修复了在非 ROCm 平台上导入 ROCm 相关架构代码时报错的问题。
- **涉及概念**：架构（architecture）检测用于确定硬件平台类型。**比喻**：修好了“在 Windows 电脑上误装 Mac 驱动”导致的报错。
- **影响**：非 AMD 平台（如 NVIDIA/Intel）用户，升级后不会再遇到因 ROCm 导入导致的意外崩溃。

### 7. feat: extended EPLB support for Mistral Large 3 and additional MoE backends
- **做了什么**：为 Mistral Large 3 模型和更多 MoE（混合专家）后端扩展了 EPLB（Expert Parallel Load Balancing，专家并行负载均衡）支持。
- **涉及概念**：MoE 模型把任务分给多个“专家”子网络，EPLB 负责让各专家工作量均衡。**比喻**：客服中心有多个专家坐席，调度系统要保证每个专家接的电话数量差不多。
- **影响**：使用 Mistral Large 3 或其他 MoE 模型的用户，多卡推理时负载更均衡，吞吐量更稳定。

### 8. [XPU] quick fix online quantization UT break
- **做了什么**：修复了 XPU（Intel GPU）平台上在线量化单元测试失败的问题。
- **涉及概念**：量化（Quantization）是把模型权重从高精度（如 FP16）压缩到低精度（如 INT8），以节省显存和加速。**比喻**：把高清照片压缩成小尺寸缩略图，省空间但细节略损失。
- **影响**：Intel GPU 用户使用在线量化功能时，测试和功能不再报错。

### 9. [Misc] Add and enable Triton kernel unit tests on XPU
- **做了什么**：为 Intel XPU 平台添加并启用了 Triton 内核的单元测试。
- **涉及概念**：Triton 是一种 GPU 编程语言/编译器，用于编写高性能内核。**比喻**：给新修的发动机（Triton 内核）增加了质检环节（单元测试）。
- **影响**：对普通用户影响较小，主要是 Intel GPU 平台的质量保障提升。

### 10. [PD][PushConnector] Record last activity of remotes to allow clean up of stale ones
- **做了什么**：在 PushConnector 中记录远端节点的最后活动时间，从而可以清理长期不活跃的过期节点。
- **涉及概念**：PushConnector 是 PD 分离架构中负责推送数据的组件。**比喻**：群聊中记录每个成员最后发言时间，把长期潜水的人移出群聊，保持群活跃。
- **影响**：分布式部署中，失效节点能被自动清理，系统资源利用更合理。

### 11. [Bugfix][Platform] Stop re-initializing NVML on every device-capability check
- **做了什么**：修复了每次检查设备能力时都重复初始化 NVML（NVIDIA 管理库）的问题，改为只初始化一次。
- **涉及概念**：NVML 是 NVIDIA 提供的 GPU 状态查询接口。**比喻**：每次问路都重新打开地图 App，现在改为打开一次，之后直接查。
- **影响**：启动和运行时的额外开销减少，尤其对频繁检查设备能力的场景有明显提速。

### 12. [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16
- **做了什么**：修复了动态 INT8 W8A8 量化的 MoE 模型配置被错误构建为 W8A16 的问题。
- **涉及概念**：W8A8 表示权重和激活都用 8 位整数，W8A16 表示权重 8 位、激活 16 位。**比喻**：原本想用“8 位精度”存储，结果系统错误地用了“8 位权重+16 位激活”的方案，导致显存占用和速度不符合预期。
- **影响**：使用动态 INT8 量化的 MoE 模型用户，显存占用和推理速度会符合预期，不再“名不副实”。

### 13. [Refactor] Remove kernel dead code
- **做了什么**：删除了内核中不再使用的死代码。
- **涉及概念**：死代码是永远不会被执行到的代码。**比喻**：清理仓库里堆了十年没用的旧机器，腾出空间。
- **影响**：代码更干净，编译时间可能略减，对普通用户无直接感知。

### 14. Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel
- **做了什么**：支持 DeepSeek-V4 模型在 AMD 平台上使用 Quark NVFP4 量化格式（通过模拟内核实现）。
- **涉及概念**：NVFP4 是 NVIDIA 的 4 位浮点格式，AMD 通过模拟方式兼容。**比喻**：AMD 显卡上装了个“翻译器”，能读懂 NVIDIA 专属的压缩格式。
- **影响**：AMD 用户现在可以运行 DeepSeek-V4 并使用 4 位量化，显存占用大幅降低。

### 15. [ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes
- **做了什么**：放宽了 ROCm 平台上 block-FP8 融合 MoE 测试在 K 值较大时的误差容忍度。
- **涉及概念**：K 是矩阵乘法中的维度大小。FP8 是 8 位浮点格式，精度较低，大 K 时误差更明显。**比喻**：考试题目变难了（大 K），评分标准适当放宽，避免误判。
- **影响**：AMD 平台上某些大矩阵场景的测试不再误报失败，CI 更

## sglang
## 🔥 重要更新（如果有的话）

- **CUDA PyTorch 升级到 2.13**（#28836）：这是影响面最大的变更，涉及整个 CUDA 技术栈的升级，可能影响所有使用 NVIDIA GPU 的用户，建议关注官方发布说明。
- **LTX-2 视频生成性能提升 1.56 倍**（#33885）：通过启用可中断 CUDA 图，将端到端延迟从 10.75 秒降至 6.90 秒，对使用该模型的用户是显著的体验提升。

## 📋 逐条解读

### 1. [Remove the HiMambaRadixTree that is no longer in use]
- **做了什么**：删除了一段不再使用的代码（HiMambaRadixTree 数据结构）。
- **涉及概念**：RadixTree（基数树）是一种高效存储和检索前缀的树结构，常用于缓存管理。这里相当于清理仓库里一个废弃的旧工具。
- **影响**：无直接用户影响，属于代码清理，让项目更整洁。

### 2. [[diffusion] Enable breakable CUDA graph for LTX-2]
- **做了什么**：为 LTX-2 视频生成模型启用了“可中断 CUDA 图”功能，将端到端时间从 10.75 秒降至 6.90 秒。
- **涉及概念**：CUDA 图（CUDA Graph）是将一系列 GPU 操作打包成一张图一次性执行，减少调度开销。“可中断”意味着可以在中途打断并继续，增加了灵活性。
- **影响**：使用 LTX-2 生成视频的用户会感受到明显的速度提升（约 1.56 倍）。

### 3. [[diffusion] feat: make scheduler rpc deadlines explicit]
- **做了什么**：让调度器的 RPC（远程过程调用）超时时间变得明确可配置。
- **涉及概念**：RPC 是不同计算节点之间通信的方式；“deadline”指等待响应的最长时间。以前可能是隐式或写死的，现在可以显式设置。
- **影响**：对分布式部署的用户更友好，可以按需调整超时策略，避免任务卡死。

### 4. [Fix vae fast path test after the gate refactor]
- **做了什么**：修复了在“门控重构”之后 VAE 快速路径测试失败的问题。
- **涉及概念**：VAE（变分自编码器）是图像/视频生成中负责压缩和解压潜空间的组件；“门控”指根据条件决定走哪条代码路径的逻辑。
- **影响**：保证测试通过，确保代码质量，对用户无直接感知。

### 5. [Fix Nemotron W4A16 NVFP4 MoE backend]
- **做了什么**：修复了 Nemotron 模型在 W4A16 NVFP4 混合精度下的 MoE（混合专家）后端问题。
- **涉及概念**：W4A16 指权重 4-bit、激活 16-bit 的量化方式；NVFP4 是 NVIDIA 的 4-bit 浮点格式；MoE 是混合专家模型，将任务分给不同“专家”子网络。
- **影响**：使用 Nemotron 模型且开启低比特量化的用户会得到更正确的推理结果。

### 6. [[diffusion] UX: speed up tp and fsdp checkpoint loading]
- **做了什么**：加速了张量并行（TP）和 FSDP 检查点（模型存档）的加载速度。
- **涉及概念**：TP 是将模型切分到多张 GPU 上并行计算；FSDP 是 PyTorch 的一种分布式训练/加载方案；检查点就是模型训练好的“存档文件”。
- **影响**：分布式加载模型时等待时间更短，用户体验更流畅。

### 7. [[diffusion] fix: bind each rank to accelerator before distributed init]
- **做了什么**：在分布式初始化之前，先将每个进程（rank）绑定到对应的加速器（GPU）上。
- **涉及概念**：分布式训练中每个进程需要明确绑定到哪块 GPU；“绑定”可以避免资源冲突和错误分配。
- **影响**：修复了多卡环境下可能出现的设备分配错误，让分布式运行更稳定。

### 8. [[diffusion] fix: enable bcg with tp]
- **做了什么**：修复了在张量并行（TP）模式下启用 BCG（一种通信优化技术）的问题。
- **涉及概念**：BCG 可能指某种通信组（Byte-based Communication Group）优化；TP 模式需要特定的通信模式，之前两者冲突。
- **影响**：TP 用户现在可以同时享受 BCG 带来的通信效率提升。

### 9. [Fix IndexError in Triton backend with pipeline parallelism]
- **做了什么**：修复了 Triton 后端在流水线并行（PP）模式下出现的索引越界错误（IndexError）。
- **涉及概念**：Triton 是一种 GPU 编程语言；流水线并行是将模型的不同层分配到不同设备上，像流水线一样接力计算。
- **影响**：使用 PP 模式的用户在特定场景下不再遇到崩溃。

### 10. [[diffusion] feat: gate /health and /health_generate on warmup completion and add liveness endpoint]
- **做了什么**：让 `/health` 和 `/health_generate` 接口在模型预热完成前返回“未就绪”，并新增了一个 `/liveness` 接口用于纯存活检查。
- **涉及概念**：预热（warmup）是模型启动后先跑一些假数据让 GPU 达到稳定状态；“健康检查”接口用于监控服务状态。
- **影响**：运维人员可以更准确地区分“服务活着但没准备好”和“服务挂了”，便于自动化调度。

### 11. [[diffusion] model: support lingbot-video moe 30b t2v]
- **做了什么**：新增了对 lingbot-video moe 30b 文生视频（t2v）模型的支持。
- **涉及概念**：文生视频（text-to-video）指根据文字描述生成视频；MoE 30b 指该模型有约 300 亿参数且使用混合专家结构。
- **影响**：用户现在可以直接使用这个新模型进行文生视频任务。

### 12. [[diffusion] feat: make ring admission a backend capability]
- **做了什么**：将“环形准入”（ring admission）功能改为由后端（backend）声明的能力。
- **涉及概念**：“环形”可能指环形通信拓扑；“准入”指是否允许某个请求进入；“后端能力”指某个后端能做什么的声明。
- **影响**：架构更清晰，不同后端可以按需启用该功能，减少不必要的限制。

### 13. [[diffusion] perf: build qwen's masked varlen metadata host-side]
- **做了什么**：将 Qwen 模型的“掩码变长元数据”改为在 CPU（host）侧构建，而不是 GPU 侧。
- **涉及概念**：变长（varlen）指输入长度不固定；掩码（masked）指忽略某些位置的注意力；“元数据”是描述数据的数据。在 CPU 构建可以减少 GPU 等待。
- **影响**：Qwen 模型推理性能有所提升，因为 GPU 不再需要花时间构建这些元数据。

### 14. [Move SWA chunk-cap hatch tests into the registered suite]
- **做了什么**：将 SWA（滑动窗口注意力）的 chunk-cap 相关测试移入正式注册的测试套件中。
- **涉及概念**：SWA 是只关注最近窗口的注意力机制；chunk-cap 指对块大小的限制；“测试套件”是一组自动化测试的集合。
- **影响**：保证该功能被持续测试，减少回归风险。

### 15. [[NPU] [DOC] Upgrade recommendeded sglang version on Ascend NPU]
- **做了什么**：更新了昇腾 NPU 上推荐的 sglang 版本号。
- **涉及概念**：NPU 是华为的 AI 芯片；版本推荐是告诉用户哪个版本最稳定。
- **影响**：使用昇腾 NPU 的用户应参考新推荐的版本进行升级。

### 16. [[diffusion] Z-Image bit-exact fused qk-norm]
- **做了什么**：为 Z-Image 模型实现了“位精确”的融合 QK-Norm 操作，端到端性能提升 6.4%。
- **涉及概念**：QK-Norm 是对注意力中的 Query 和 Key 做归一化；“融合”指将多个操作合并成一个；“位精确”指结果与原始实现完全一致。
- **影响**：Z-Image 用户获得性能提升，且数值结果不变，无精度损失。

### 17. [Fix prefill CP graph overflow with larger bucket search]
- **做了什么**：通过扩大桶（bucket）搜索范围，修复了预填充（prefill）阶段在上下文并行（CP）下图溢出（overflow）的问题。
- **涉及概念**：预填充是生成第一个 token 前的处理阶段；CP 是并行处理长上下文的方式；“桶”是缓存分配的粒度。
- **影响**：长上下文场景下不再因图溢出而报错，稳定性提升。

### 18. [[CI] Share VLM engines and prune launch matrices on the per-commit H100/H200 suites]
- **做了什么**：在 CI（持续集成）中共享视觉语言模型（VLM）引擎，

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)