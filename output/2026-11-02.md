# 🚀 vLLM & SGLang 每日更新 — 2026-11-02

> 自动生成于 2026-11-02 08:14 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：ROCm / AMD 生态适配**
  - 一句话通俗解释：ROCm 是 AMD 对标 NVIDIA CUDA 的 GPU 计算平台，vLLM 正在大量为 MI300/MI355 等 AMD 卡做专门优化。
  - 为什么社区在关注：今天 20 个 commit 里有 6 个带 `[ROCm]` 标签（Muse Spark、Kimi-K3、DSv4.1、RDNA W4A16、K3 量化），说明 vLLM 正从"能跑"走向"跑得快"，异构计算（NVIDIA + AMD + 其他）已成主流。
  - 对新手意味着什么：如果你只会 CUDA，补一下 HIP/ROCm 编程模型（HIP 基本是 CUDA 的方言），能直接参与 vLLM 最活跃的优化前线；AMD 的招聘和开源贡献机会都在涨。

- **关键词：量化（FP8 / MXFP8 / W4A16 / FP4）**
  - 一句话通俗解释：把模型权重和激活从 16 位浮点压缩到 8 位甚至 4 位，用少量精度换大幅显存和带宽节省。
  - 为什么社区在关注：MiniMax-M3 的 fused norm+MXFP8、RDNA 的 W4A16 GEMM、Kimi-K3 的 FP4 BMM 都在同一天出现——量化已从"可选加速"变成"部署标配"。
  - 对新手意味着什么：理解 E4M3/E5M2、per-block scale、group size 这些概念，是做推理优化的入门钥匙；建议从读懂一个 Triton 量化 kernel 开始。

- **关键词：Speculative Decoding（投机解码）**
  - 一句话通俗解释：先用小模型"猜"几个 token，再用大模型一次性验证，猜对了就白赚速度。
  - 为什么社区在关注：Commit 17 修了 V2 rejection sampler 的 logprob_token_ids 返回问题，Commit 2/5/12 都在为 speculative decode 做专用 kernel（K2、monokernel）。
  - 对新手意味着什么：这是当前 LLM 推理加速最热的方向之一，涉及采样、拒绝采样、KV cache 管理，是很好的系统+算法交叉学习点。

- **关键词：Kernel 融合 / Monokernel**
  - 一句话通俗解释：把多个小 GPU 算子合并成一个大的"超级 kernel"，减少启动开销和显存往返。
  - 为什么社区在关注：Muse Spark 的 FlyDSL monokernel、MiniMax-M3 的 fused norm+quant、DSv4.1 的 mono layer，说明大家已经卷到"逐 kernel 抠性能"的阶段。
  - 对新手意味着什么：想深入 GPU 性能优化，必须学 Triton / CUDA / FlyDSL 这类 DSL，并理解 roofline、occupancy、bank conflict。

- **关键词：可扩展 KV Cache / 显存管理**
  - 一句话通俗解释：KV cache 是推理时缓存历史 token 的"草稿纸"，新的可扩展设计让显存按需增长而非启动时一次性占满。
  - 为什么社区在关注：Commit 11 是今天最大的核心重构，改了 `CacheConfig` 的默认 `gpu_memory_utilization` 语义，影响所有用户的启动行为。
  - 对新手意味着什么：这是理解 vLLM 架构的最佳切口——PagedAttention、block manager、prefix caching 都围绕它展开。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：int32 溢出——为什么大矩阵的行偏移要用 int64**
  - **是什么**：GPU kernel 里用 32 位整数计算元素下标，当 `行号 × 列数` 超过 2^31（约 21 亿）时就会溢出变成负数，读到错误内存。
  - **为什么重要**：大模型的长序列 prefill 很容易触发（比如 M=2^31/256+128 行），不处理就是静默的数值错误，比崩溃更难查。
  - **动手试试**：
    ```python
    import numpy as np
    M, K = 2**31 // 256 + 128, 256
    idx = np.arange(M, dtype=np.int32)
    offs = idx * K          # 溢出！
    print(offs[-1], "vs", int(idx[-1]) * K)   # 负数 vs 正确值
    # 修复：计算前转 int64
    offs64 = idx.astype(np.int64) * K
    print(offs64[-1] == int(idx[-1]) * K)
    ```
  - **预期结果**：你会看到 int32 版最后一个偏移是负数，int64 版正确；这正对应 Commit 7 里 `pid_m.to(tl.int64)` 的修复。

- **知识点 2：RMSNorm + 量化融合是怎么省时间的**
  - **是什么**：RMSNorm（归一化）和 FP8 量化本来是两步，融合后一次读显存、一次写显存，省掉中间结果的往返。
  - **为什么重要**：LLM 每层都有 norm，显存带宽是推理瓶颈，融合能直接提升吞吐；但融合后要保证数值和参考实现一致（Commit 4 专门写了 bitwise 对比测试）。
  - **动手试试**：
    ```python
    import torch
    x = torch.randn(1024, 6144, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(6144, device="cuda", dtype=torch.bfloat16) * 0.2
    # 参考：先 norm 再量化
    def rmsnorm(x, w, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    y = rmsnorm(x, w)
    y_q = (y / y.abs().amax(-1, keepdim=True) * 448).to(torch.float8_e4m3fn)
    print(y_q.dtype, y_q.shape)
    ```
  - **预期结果**：得到 FP8 张量；理解"norm 输出范围"和"量化 scale"的关系，就能读懂 Commit 4 里 `gemma_rmsnorm_mxfp8` 的测试逻辑。

- **知识点 3：LoRA 的 `__eq__` 和 `__hash__` 为什么必须一起改**
  - **是什么**：Python 对象放进 dict/set 时用 hash 定位、用 eq 判等；两者不一致会导致"两个相等的对象被当成不同 key"。
  - **为什么重要**：Commit 14 的 bug 就是 hash 只看 name、eq 也只看 name，结果同名不同 id 的 LoRA 被错误合并，导致 KeyError 崩溃。
  - **动手试试**：
    ```python
    class Bad:
        def __init__(self, name, id): self.name, self.id = name, id
        def __eq__(self, o): return self.name == o.name
        def __hash__(self): return hash(self.name)
    class Good(Bad):
        def __eq__(self, o): return (self.name, self.id) == (o.name, o.id)
        def __hash__(self): return hash((self.name, self.id))
    print(len({Bad("a",1), Bad("a",2)}), len({Good("a",1), Good("a",2)}))
    ```
  - **预期结果**：Bad 输出 1（错误合并），Good 输出 2（正确区分）；这就是 Commit 14 修复的本质。

- **知识点 4：ZMQ ROUTER/DEALER 与端口自动分配**
  - **是什么**：ZMQ 是消息队列库，ROUTER 服务端 + DEALER 客户端是经典的一对多通信模式；端口填 0 让操作系统自动分配空闲端口。
  - **为什么重要**：Commit 9 把硬编码端口 5601 改成自动分配，解决了多实例同机部署时端口冲突的痛点——这是分布式系统常见的"配置即 bug"问题。
  - **动手试试**：
    ```python
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    print("bound port:", port)   # OS 分配的端口
    sock.close(); ctx.term()
    ```
  - **预期结果**：每次运行得到不同的随机端口；理解后就能看懂 Commit 9 里 `peer_port` 通过 `ec_transfer_params` 广播给消费者的设计。

- **知识点 5：投机解码里的"拒绝采样"是什么**
  - **是什么**：小模型给出候选 token 和概率，大模型验证时按概率比决定接受或拒绝，保证最终分布和大模型单独采样一致。
  - **为什么重要**：Commit 17 修的是拒绝采样器在只请求 `logprob_token_ids`（不请求 top-k logprobs）时返回错误分数的问题，直接影响 API 返回给用户的 logprob 正确性。
  - **动手试试**：
    ```python
    import torch
    # 简化版拒绝采样：p=目标分布, q=草稿分布
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.4, 0.4, 0.2])
    tok = torch.multinomial(q, 1).item()
    accept_prob = min(1.0, (p[tok] / q[tok]).item())
    print("draft token", tok, "accept prob", accept_prob)
    # 若拒绝，从 max(p-q,0) 归一化后重采样
    ```
  - **预期结果**：多次运行后统计接受率应接近 `sum(min(p,q))`；理解这个机制，就能读懂 Commit 17 里 `logprobs_dims` 参数为什么和 top-k 不同。

---

## 🔥 重要更新

- **Commit 11 [Core] Extensible KV cache**：核心架构级重构，`CacheConfig.gpu_memory_utilization` 默认语义从 0.9 变为"未设置即 1.0 按需扩展"，影响所有用户的启动显存行为和测试代码，是今天影响面最大的变更。
- **Commit 5 [ROCm][Perf][Muse Spark] FlyDSL decode monokernel**：为 MI355X 上的 DSpark verify 步骤实现持久化 monokernel，是 AMD 平台上投机解码性能的重要突破，标志着 vLLM 在非 NVIDIA 平台的 kernel 级优化进入深水区。
- **Commit 8 [Perf][MiniMax-M3] Sequence parallelism + GEMM-RS**：把 GEMM、ReduceScatter、AllReduce 融合，并支持直接消费已量化的激活，是 TP + MoE 场景下通信-计算重叠的典型优化。
- **Commit 14 [Bugfix][LoRA] KeyError crash**：修复同名 LoRA 重载时在途请求崩溃的生产级 bug，涉及 Python 对象相等性语义，是很好的"小改动大影响"案例。

---

## 📋 逐条解读

### 1. [Bugfix][ROCm] Muse Spark: merge the two process_weights_after_loading hooks
- **代码层面**：`vllm/models/muse_spark/amd/model.py` 里同一个类定义了两个同名方法 `process_weights_after_loading`，Python 中后者会覆盖前者，导致前半段逻辑（materialize 残差常量）永远不执行。修复方式是删掉第二个 `def`，把两段逻辑合并到一个方法体内。
- **新手概念课堂**：就像你在通讯录里给同一个人存了两个号码，后存的把先存的覆盖了——Python 类里同名方法也是"后定义者胜"。这是典型的"复制粘贴导致的方法重复定义"陷阱。
- **对你有什么影响**：使用 Muse Spark 在 AMD GPU 上推理时，残差路径常量和 monokernel 挂载现在都会正确执行，避免潜在的数值错误或性能退化。

### 2. [ROCm][Perf][Kimi-K3] Specialized decode K2: sentinel all-reduces, pre-AR loads, tagged x rows
- **代码层面**：大量改动在 `tests/models/kimi_k3/amd_specialized_decode/` 和 mono kernel 相关代码。测试里把 KDA 输入从"单请求 S 个 slot"改成"R 个请求 × L 个 slot"的二维布局（`st_idx` 变成 `(R, L)`，`num_acc` 变成每请求不同值），并为 W_UK/W_UV 打开 AITER 的 MXFP4 BMM。核心是让 K2 专用 decode kernel 支持多请求批处理、预取 all-reduce 结果、给 x 行打 tag。
- **新手概念课堂**：把"排队买奶茶"想象成 batch——以前一次只服务一个人（单请求），现在一次服务 R 个人，每人点 L 杯（speculative tokens）。"sentinel all-reduce" 是用一个哨兵值标记通信完成，避免轮询等待；"pre-AR loads" 是在 all-reduce 结果到达前就预加载数据，属于典型的通信-计算重叠。
- **对你有什么影响**：Kimi-K3 在 AMD MI 系列卡上的投机解码吞吐会提升；普通用户感知为"同样硬件下生成更快"。

### 3. [Model] Enable --mm-encoder-tp-mode data for MiMo-V2.6
- **代码层面**：`tests/models/multimodal/test_mimo_v2_omni.py` 新增测试，验证 `MiMoVisionAttention` 在 `mm_encoder_tp_mode="data"` 下的行为；通过 monkeypatch 伪造 `get_multimodal_config` 和 TP rank/world_size，检查视觉编码器按数据并行切分时注意力输出是否正确。同时在 `.buildkite/test-amd.yaml` 里加了该测试。
- **新手概念课堂**：多模态模型的视觉编码器（ViT）参数量大，`tp_mode=data` 表示"每张卡都跑完整 ViT，但各处理不同图片"，区别于 `tp` 模式（把 ViT 层切开放到多卡）。就像餐厅：data 并行是"每个厨师都会做全菜，各做各的订单"，tensor 并行是"一个厨师只切菜、一个只炒"。
- **对你有什么影响**：部署 MiMo-V2.6 多模态模型时可以用 `--mm-encoder-tp-mode data` 提升视觉编码吞吐，尤其在图片多、TP 组大的场景。

### 4. [Perf][MiniMax-M3] (NVIDIA-only) Fused norm+quant MXFP8
- **代码层面**：新增 `vllm/models/minimax_m3/nvidia/ops/norm_mxfp8.py` 实现 `gemma_rmsnorm_mxfp8`，把 Gemma 风格 RMSNorm（可带 residual add）和 MXFP8 量化融合成一个 Triton kernel。测试 `test_norm_mxfp8.py` 用 FlashInfer 的 norm + 参考量化做 bitwise 对比，覆盖 strided 输入、特殊值（0、1e-40、inf、448）等边界。
- **新手概念课堂**：MXFP8 是"每 32 个元素共享一个 2 的幂次 scale"的 FP8 格式，scale 用 UE8M0 编码（只有指数没有尾数）。融合的意义：norm 输出本来要写回显存再读出来量化，融合后省一次往返——就像洗菜和切菜在同一水槽完成，不用端来端去。
- **对你有什么影响**：MiniMax-M3 在 NVIDIA GPU 上（尤其配合 FP8 推理）吞吐提升；测试里对 inf/NaN 的处理保证了数值稳定性。

### 5. [ROCm][Perf][Muse Spark] FlyDSL decode monokernel for DSpark verify steps on MI355X
- **代码层面**：新增 `tests/models/test_muse_spark_rocm_mono.py`，`pyproject.toml` 加 typos 白名单（TPE/shs/shw/opf），recipe 文档加 `--monokernel` 用法。核心是 FlyDSL 写的持久化 kernel：1-2 个请求每层一次 launch，3-16 个请求三次 launch，≤4 个请求把 verify attention 也融进去。
- **新手概念课堂**：FlyDSL 是 AMD 的 kernel DSL（类似 Triton）。"monokernel" 是把 decode 一层的所有算子（attention、MoE、norm 等）塞进一个持久化 kernel，减少 kernel launch 开销和中间显存读写。就像把"洗菜→切菜→炒菜"三个工人合并成一个流水线工人，不用来回交接。
- **对你有什么影响**：MI355X 上 Muse Spark 的 DSpark 投机解码（1-16 请求）显著加速，需手动加 `--monokernel` 并配合 TP8 + EP。

### 6. [Model] Remove redundant Muse Spark configuration overrides
- **代码层面**：`vllm/models/muse_spark/nvidia/model.py` 里删掉一堆 `getattr(config, "muse_spark_xxx", default)` 的配置开关，统一用局部变量 `use_tep_optimizations`（条件：replicated_tokens + CUDA + capability 100）。`fp32_router_input` 直接等于 `torch.backends.cuda.matmul.allow_tf32`。
- **新手概念课堂**：这是"配置腐化"清理——曾经为了灰度上线加了 N 个开关，验证稳定后应该删掉，否则代码里到处是 `getattr` 分支，难读难测。就像装修时临时拉的电线，完工后要收进墙里。
- **对你有什么影响**：行为不变（默认值就是原来生效的值），但代码更简洁；如果你之前手动设过 `muse_spark_tep_*` 这些非公开配置，现在会被忽略。

### 7. [Bugfix][ROCm] Use 64-bit row offsets in the RDNA hybrid W4A16 GEMM
- **代码层面**：`rdna_hybrid_w4a16.py` 的 Triton kernel 里 `offs_m = pid_m * BLOCK_M + ...` 改成 `pid_m.to(tl.int64) * BLOCK_M + ...`。测试新增两个用例：`M*N > 2^31`（输出偏移溢出）和 `M*K > 2^31`（激活偏移溢出），并只校验最后 64 行（因为前面的行太大不好参考）。
- **新手概念课堂**：见知识点 1。int32 最大约 21 亿，大矩阵的行偏移一乘就爆。GPU 编程里"用 int32 算下标"是性能优化（寄存器少、指令快），但必须确认范围；一旦溢出就是静默错误。
- **对你有什么影响**：RDNA（gfx11/gfx12，即消费级 AMD 卡）上跑大 batch prefill 的 W4A16 量化模型不再出现数值错误。

### 8. [Perf][MiniMax-M3] Sequence parallelism and GEMM-RS for TP + MegaMoE
- **代码层面**：`benchmark_gemm_rs_ar.py` 加 `minimax_m3` 投影配置；`test_gemm_rs_ar.py` 加"输入已是 QuantizedActivation（MXFP8）时跳过重复量化"的测试；`cute_dsl/gemm_rs_ar.py` 引入 `QuantizedActivation` 和 `as_quantized_activation`，让 GEMM 直接消费上游产出的量化激活。
- **新手概念课堂**：GEMM-RS-AR = 矩阵乘（GEMM）→ ReduceScatter（把结果按 TP 切分）→ AllReduce（跨 TP 组求和），融合成一个通信原语。序列并行（SP）是把长序列按 token 切到多卡，配合 TP 减少激活显存。就像工厂流水线：GEMM 是加工，RS 是把成品分给各仓库，AR 是汇总库存，融合后不用中间搬运。
- **对你有什么影响**：MiniMax-M3 在 TP4 + MegaMoE 部署下吞吐提升；对开发者来说，`QuantizedActivation` 这个抽象值得学习——它让"量化格式"在算子间传递而不重复量化。

### 9. [Bugfix][EC Connector] Allocate producer control ports at startup
- **代码层面**：`docs/features/ec_cpu_connector.md` 更新协议说明：默认用 `get_ip()` + OS 分配端口，通过 `ec_transfer_params.peer_port` 广播；`VLLM_EC_SIDE_CHANNEL_PORT` 默认改为 0（自动分配），非 0 时用 `base_port + data_parallel_index`。测试删掉硬编码 `get_open_port()`，新增 `test_zmq_servers_allocate_distinct_reachable_ports`。
- **新手概念课堂**：见知识点 4。硬编码端口是分布式部署的经典坑——两个实例同机就冲突。OS 自动分配（bind port 0）+ 通过控制面广播实际端口，是"配置即 bug"的正确解法。
- **对你有什么影响**：EC（编码器缓存）连接器多实例部署更省心；如果你之前设了固定端口，注意新的 `base_port + dp_index` 语义。

### 10. [Bugfix][Tool Parser] Parse bare FunctionGemma arguments and stream them as valid JSON
- **代码层面**：新增 `tests/entrypoints/tool_parsers/test_functiongemma_tool_parser.py`，用 `structured_outputs.regex` 把模型输出钉死，只测 serving + parsing。测试覆盖非流式和流式，流式时按 `index` 累积 delta 的 `arguments` 字符串再 `json.loads`。
- **新手概念课堂**：FunctionGemma 是 Google 的函数调用格式，模型可能输出"裸参数"（不带 JSON 包装）。流式输出时 JSON 是分片到达的，前端必须拼接完整才能解析——就像拼图，中途看每一片都不像图。
- **对你有什么影响**：用 FunctionGemma 模型做工具调用时，流式和非流式都能拿到合法 JSON 参数，不再解析失败。

### 11. [Core] Extensible KV cache
- **代码层面**：`CacheConfig.gpu_memory_utilization` 从类属性（默认 0.9）改为 `DEFAULT_GPU_MEMORY_UTILIZATION` 常量，未设置时语义变成"1.0 按需扩展"。大量测试从 `CacheConfig.gpu_memory_utilization` 改为 `CacheConfig.DEFAULT_GPU_MEMORY_UTILIZATION`，并用 `or` 处理 None。`tests/conftest.py` 注释说明：可扩展 KV cache 不再需要启动时预留显存。
- **新手概念课堂**：旧设计是"启动时按 gpu_memory_utilization 一次性划走显存"，新设计是"按需增长"。就像租房：以前一次性租下整层楼（可能空着），现在按需续租。好处是启动更灵活、多进程共存更容易；代价是峰值显存管理更复杂。
- **对你有什么影响**：不显式设 `gpu_memory_utilization` 时行为变化（不再是 0.9），ROCm 测试的显存等待逻辑也相应调整；这是需要关注的行为变更。

### 12. [ROCm][Perf][DSv4.1] Mono: batch TP4 K2's X8 copies, load its push's peer bases late
- **代码层面**：`deepseek_v41/amd/mono/layer.py` 在 TP4 时把 `push_bases` 存进 `cb`，并作为 lambda 延迟传入 `run_ffn`；`stages/moe.py` 的 `load_x8_rows` 新增 `x8all` 分支：TP4 且非 GFX942 且 `live` 是 int 时，用 `live * quads` 的批量加载替代按 TILE 循环。测试新增 `test_mono_decode_k2_tp4_no_scratch`，用 `llvm-objdump` 检查编译产物里没有 `scratch_` 符号（即寄存器没溢出到 scratch）。
- **新手概念课堂**："peer bases 延迟加载"是把跨 rank 的基地址计算推迟到真正需要时，减少寄存器压力；"no scratch" 指 GPU 寄存器不够时编译器会把变量溢出到 scratch 内存（很慢），检查汇编里没有 scratch 符号说明 kernel 很"干净"。
- **对你有什么影响**：DSv4.1 在 TP4 + MI355X 上的 mono decode 更快且无寄存器溢出；测试方法（反汇编查 scratch）值得学习。

### 13. [Doc] Move the Ascend content of the recipes to vllm-ascend overlays
- **代码层面**：`recipes/README.md` 和 `.agents/skills/model-recipes/SKILL.md` 更新说明：Ascend NPU 内容从主仓库 recipes 移到 `vllm-ascend/recipes/overlays/`，构建站点时合并。`Qwen3-30B-A3B.yaml` 删掉 `ascend_910c` 硬件、`ascend_w8a8` variant 等整段内容。
- **新手概念课堂**：这是"仓库职责分离"——主仓库专注通用 recipe，厂商特定硬件（Ascend）放到对应厂商仓库，避免主仓库被硬件细节淹没。就像菜谱书只写通用做法，地方特色菜放到地方分册。
- **对你有什么影响**：在 vLLM 主仓库找不到 Ascend 部署配置了，要去 vllm-ascend 仓库找；对非 Ascend 用户无影响。

### 14. [Bugfix][LoRA] Fix KeyError crash when a LoRA name is reloaded with requests in flight
- **代码层面**：`vllm/lora/request.py` 的 `LoRARequest.__eq__` 从只比 `lora_name` 改为比 `(lora_name, lora_int_id)`，`__hash__` 同步改为 `hash((lora_name, lora_int_id))`。测试 `test_same_lora_name_with_different_ids` 用同一个 name "pet" 但 id 1/2 加载两个不同 LoRA（meowing/woofing），同一 batch 里各请求用自己的 adapter。
- **新手概念课堂**：见知识点 3。同名 LoRA 卸载后重载会拿到新 id，但在途请求还引用旧 id；如果 eq/hash 只看 name，引擎会把两个 adapter 当成一个，旧请求找不到就 KeyError。这是"对象身份 vs 对象内容"的经典混淆。
- **对你有什么影响**：热更新 LoRA（同名替换）时不再崩溃，在途请求继续用旧 adapter 完成，新请求用新 adapter。

### 15. [Bugfix][LoRA] respect `attention_k_eq_v=True` QKK LoRA layers
- **代码层面**：`vllm/lora/model_manager.py` 的 `_register_packed_modules` 新增 `module` 参数，优先读 `getattr(module, "lora_packed_modules", None)` 覆盖全局 `packed_modules_mapping`。测试 `test_packed_loras_layer_override` 设置 `lora_packed_modules = ["gate_proj", "gate_proj"]`，验证 packed LoRA 的两个分片都用 gate_proj 的权重。
- **新手概念课堂**：`k_eq_v` 是 Gemma 类模型的优化（K 和 V 共享投影），对应 LoRA 要打包成 `[q, k, k]` 而不是默认的 `[q, k, v]`。这是"层级别覆盖全局配置"的设计模式——就像公司有统一着装规定，但某个部门可以申请例外。
- **对你有什么影响**：Gemma4 等 `k_eq_v=True` 的模型加载 LoRA 时不再用错投影权重，输出正确。

### 16. [Bugfix][Frontend] Fix tool_reference results on DeepSeek V4.1 and /v1/messages
- **代码层面**：`tests/entrypoints/anthropic/test_anthropic_messages_conversion.py` 新增测试：`tool_result` 里含 `tool_reference` 项时，转换后仍是一条 tool 消息，且 references 排在最前（`[tool_reference, tool_reference, text]`）。`tests/entrypoints/scale_out/render/test_render.py` 新增 DeepSeek V4.1 的 render server 测试。
- **新手概念课堂**：`tool_reference` 是"工具搜索结果引用"——模型搜索工具后返回"找到了 get_weather 和 get_time"，需要作为 tool 消息回填给对话历史。顺序很重要：模板要求"一个 tool_call 对应一个 tool 结果"，references 放前面才能正确配对。
- **对你有什么影响**：用 DeepSeek V4.1 的 tool search 功能或 Anthropic `/v1/messages` 接口时，工具引用结果能正确渲染，不再错位。

### 17. [Bugfix][Spec Decode] V2 rejection sampler: return logprob_token_ids scores
- **代码层面**：`test_gpu_batch_shard.py` 把 `logprobs_dims` 参数化（top-k only、token_ids only、混合）；`test_gpu_rejection_sampler_chunking.py` 把 `max_num_logprobs` 改成 `logprobs_dims` 元组，新增 `test_logprob_token_ids_without_topk_logprobs`。核心是 rejection sampler 在只请求 `logprob_token_ids`（投机解码批次常见）时也返回正确分数。
- **新手概念课堂**：`logprob_token_ids` 是"只返回指定 token 的 logprob"，`top-k logprobs` 是"返回 top-k 的 logprob"，两者维度不同。以前 sampler 只处理 top-k，只请求 token_ids 时分数错误。就像点餐：你只要"这道菜的价格"，系统却只准备了"最贵三道菜的价格表"。
- **对你有什么影响**：投机解码 + 请求 `logprob_token_ids` 时返回的分数正确；API 用户做置信度分析时不再拿到错误值。

### 18. [Bugfix][ROCm] Fix Kimi K3 online quantization with padded projections
- **代码层面**：`kimi_k3/amd/kda.py` 新增 `_PaddedKimiGDNMergedColumnParallelLinear`，构造时记录 `padding_size`，用 `set_weight_attrs` 把 `weight_loader_numel` 设为去掉 padding 后的元素数（排除 padding 行参与加载完成计数），并在 `weight_loader`/`weight_loader_v2` 后调用 `_zero_padding` 重新清零（因为在线量化会在 `__init__` 之后重新 materialize 权重）。
- **新手概念课堂**：padding 是为了对齐硬件要求（比如 TP 切分后每 rank 行数要 128 对齐），但 padding 行不能算进"权重加载完成"计数，否则会一直等不到完成。在线量化（online quant）在加载后重新生成权重，会把之前清零的 padding 又填上垃圾值，所以要在 loader 里再清一次。
- **对你有什么影响**：Kimi-K3 在 ROCm 上用在线量化 + padded 投影时不再出现数值错误或加载挂起。

### 19. [Bugfix][Mooncake] Let decodes reuse divergent local hybrid prefix hits
- **代码层面**：`test_mooncake_connector_hybrid_mamba.py` 新增 `test_hybrid_gdn_remote_prefill_keeps_local_attention_prefix`：缓存 4 个 block 的 attention prefix 但驱逐 GDN 状态后，`kv_consumer` 角色只拉 2 个 block（保留本地 attention prefix，只拉 suffix + GDN state），`kv_both` 拉 6 个（走完整 reconcile）。`make_hybrid_gdn_kv_cache_config` 加 `num_blocks` 和 `mamba_cache_mode` 参数。
- **新手概念课堂**：混合模型（attention + Mamba/GDN）的 KV cache 有两部分：attention 的 block 和 recurrent state。远程 prefill 时，本地已有 attention prefix 但 GDN 状态被驱逐，以前会整个放弃本地命中；现在 consumer 角色可以"保留本地 attention prefix，只拉 GDN state 和 suffix"，省传输。
- **对你有什么影响**：Mooncake 连接器 + 混合 Mamba 模型的 decode 阶段能复用本地 prefix 命中，减少跨节点传输，降低延迟。

### 20. [Responses] Normalize OpenResponses input items to match openai-python SDK requirements
- **代码层面**：`test_harmony_utils.py` 里 reasoning item 无 content / content=None / content=[] 时，返回值从 `None` 改为"空 content 消息"（`msg.content[0].text == ""`）；多个测试文件加 `pytestmark = pytest.mark.skip_global_cleanup`。核心是让 Responses API 的输入项规范化，匹配 openai-python SDK 的模型定义。
- **新手概念课堂**：OpenAI 的 Responses API 用 Pydantic 模型定义输入，SDK 要求某些字段不能是 None（比如 reasoning 的 content 必须是列表）。vLLM 以前返回 None，SDK 客户端会校验失败。规范化就是"把 None 补成空列表/空字符串"，让客户端能正常解析。
- **对你有什么影响**：用 openai-python SDK 调 vLLM 的 `/v1/responses` 接口时，reasoning 项不再因为 None 而报校验错误；对直接发 HTTP 的用户无感。

## sglang
## 💡 今日关键词

- **关键词：调度器状态不变量（Scheduler Invariant）**
  - 一句话通俗解释：给调度器加一套"账本核对"机制，确保每个请求只存在于一个队列里，不会"分身"或"失踪"。
  - 为什么社区在关注：推理服务越来越复杂（PD 分离、HiSparse、chunked prefill、投机解码），请求会在多个队列间流转，状态 bug 极难排查。加不变量检查说明 SGLang 正在从"能跑"走向"可证明正确"。
  - 对新手意味着什么：学会写"断言式调试"和状态机思维，是进入大模型推理框架核心开发的敲门砖。

- **关键词：XPU/NPU 后端适配（异构计算）**
  - 一句话通俗解释：让 SGLang 不只在 NVIDIA GPU 上跑，还能在 Intel XPU、昇腾 NPU、AMD MI355X 上跑。
  - 为什么社区在关注：今天有 3 个 commit 直接涉及 XPU/NPU/AMD，说明"一卡难求"的现实下，多硬件后端是刚需，厂商也在积极贡献代码。
  - 对新手意味着什么：懂 CUDA 之外再懂一个后端（如 SYCL/昇腾），在国产化和云厂商岗位非常吃香。

- **关键词：Kernel 融合（Kernel Fusion）**
  - 一句话通俗解释：把好几个小算子合并成一个 GPU 内核，减少启动开销和显存读写。
  - 为什么社区在关注：DeepSeek V4、MoE、MLA 等大模型的 decode 阶段是延迟敏感区，融合 kernel 是提升吞吐最直接的手段。
  - 对新手意味着什么：Triton/CUDA 编程能力是高性能推理岗位的硬通货，从"看懂一个融合 kernel"开始入门。

- **关键词：显存/资源回滚（Rollback & Leak Safety）**
  - 一句话通俗解释：分配资源失败时，把已经占用的资源还回去，避免内存泄漏。
  - 为什么社区在关注：HiCache、host memory pool 这类复杂内存管理一旦异常路径处理不好，服务会越跑越慢最后 OOM。
  - 对新手意味着什么：异常安全（exception safety）是工程素养的体现，写代码时多想一步 "如果这里抛异常了怎么办"。

- **关键词：OOT 插件注册（Out-of-Tree Plugin）**
  - 一句话通俗解释：允许第三方在不改主仓库代码的情况下，注册自己的注意力后端。
  - 为什么社区在关注：这是框架走向生态化的标志，硬件厂商可以自己维护插件，不用等上游合并。
  - 对新手意味着什么：理解"注册表模式 + 工厂函数"这种设计模式，是读懂大型框架源码的关键。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：调度器不变量检查（Invariant Checking）**
  - **是什么**：在程序运行的关键时刻，主动检查"每个请求是否只在一个队列里"，一旦违反就报错。
  - **为什么重要**：请求在 waiting / running / chunked / grammar 等多个队列间流转，如果同一个请求被两个队列同时持有，会导致重复计算、显存泄漏甚至结果错乱。没有检查时，这类 bug 可能表现为"偶发的输出乱码"，极难定位。
  - **动手试试**：用 Python 写一个最小化的"请求队列不变量检查器"：
    ```python
    from collections import Counter

    class QueueBookkeeper:
        def __init__(self):
            self.queues = {}

        def register(self, name, items):
            self.queues[name] = list(items)

        def check_no_duplicates(self):
            all_ids = []
            for name, items in self.queues.items():
                all_ids.extend(items)
            dup = [k for k, v in Counter(all_ids).items() if v > 1]
            if dup:
                raise AssertionError(f"请求 {dup} 同时存在于多个队列！")
            return True

    bk = QueueBookkeeper()
    bk.register("waiting", ["req1", "req2"])
    bk.register("running", ["req3"])
    print(bk.check_no_duplicates())  # True

    bk.register("running", ["req2"])  # 人为制造 bug
    bk.check_no_duplicates()          # 抛出 AssertionError
    ```
  - **预期结果**：第一次打印 `True`，第二次抛出 `AssertionError: 请求 ['req2'] 同时存在于多个队列！`。你会理解为什么 SGLang 要在 scheduler 里加 `request_lists()` 和 `init_invariant_checker()`。

- **知识点 2：GPU 上的网格生成（Meshgrid on GPU）**
  - **是什么**：3D 模型导出时需要把空间切成密集网格点，原来用 NumPy 在 CPU 上算完再拷到 GPU，现在直接在 GPU 上生成。
  - **为什么重要**：CPU→GPU 的数据拷贝是性能瓶颈，尤其网格点数量随分辨率立方增长（resolution=512 时是 512³ ≈ 1.3 亿个点）。GPU 端生成省掉一次大拷贝。
  - **动手试试**：
    ```python
    import torch, time

    res = 256
    # 方式一：CPU 生成再拷贝
    t0 = time.time()
    x = torch.linspace(-1, 1, res)
    xs, ys, zs = torch.meshgrid(x, x, x, indexing="ij")
    grid_cpu = torch.stack((xs, ys, zs), dim=-1).reshape(-1, 3)
    grid_gpu = grid_cpu.cuda()
    torch.cuda.synchronize()
    print("CPU->GPU:", time.time() - t0)

    # 方式二：GPU 上直接生成
    t0 = time.time()
    x = torch.linspace(-1, 1, res, device="cuda")
    xs, ys, zs = torch.meshgrid(x, x, x, indexing="ij")
    grid_gpu2 = torch.stack((xs, ys, zs), dim=-1).reshape(-1, 3)
    torch.cuda.synchronize()
    print("GPU only:", time.time() - t0)
    print("结果一致:", torch.allclose(grid_gpu, grid_gpu2))
    ```
  - **预期结果**：GPU only 明显更快（分辨率越高差距越大），且两者结果一致。这正是 Commit 4 做的事。

- **知识点 3：Triton 内核中的前缀和陷阱（Stale Prefix Read）**
  - **是什么**：GPU 内核里多个线程块并行计算前缀和时，一个线程块读到的可能是另一个线程块"还没写完"的旧值。
  - **为什么重要**：这是典型的竞态条件（race condition），在普通测试中可能通过，但在 CUDA Graph 重放或高并发下随机出错，是内核开发最阴险的 bug 之一。
  - **动手试试**：理解修复思路——不用"先写后读"，而是"本地重算"：
    ```python
    import torch

    # 模拟：expert 的起始偏移 = 前面所有 expert 的 token 数之和
    tokens_per_expert = torch.tensor([3, 0, 5, 2, 7])

    # 危险写法：先算 cumsum 存回，再由别的线程读（可能读到旧值）
    # 安全写法：每个 expert 自己用 masked sum 算前缀
    starts = []
    for cur_expert in range(len(tokens_per_expert)):
        prefix = torch.where(
            torch.arange(len(tokens_per_expert)) < cur_expert,
            tokens_per_expert, 0
        ).sum()
        starts.append(prefix.item())
    print("expert 起始偏移:", starts)  # [0, 3, 3, 8, 10]

    # 对照官方 cumsum 写法
    ref = (tokens_per_expert.cumsum(0) - tokens_per_expert).tolist()
    assert starts == ref
    print("与 cumsum 参考一致 ✓")
    ```
  - **预期结果**：输出 `[0, 3, 3, 8, 10]`，与 cumsum 结果一致。你会理解 Commit 15 为什么把 `tl.load(expert_start_loc + cur_expert)` 换成 `tl.sum(tl.where(...))`。

- **知识点 4：异常安全的资源回滚（Exception-Safe Rollback）**
  - **是什么**：分配资源（显存、host 内存槽位）的过程中如果中途抛异常，要确保已经分配的部分被释放。
  - **为什么重要**：HiCache 一次要同时分配主池 + 索引池 + 草稿池，如果第三个池分配失败，前两个池的槽位不还回去就是泄漏，长时间运行必然 OOM。
  - **动手试试**：
    ```python
    class FakePool:
        def __init__(self):
            self.free_slots = list(range(10))
            self.allocated = []

        def alloc(self, n, fail=False):
            if fail:
                raise RuntimeError("模拟分配失败")
            got = [self.free_slots.pop(0) for _ in range(n)]
            self.allocated.append(got)
            return got

        def free(self, slots):
            self.free_slots.extend(slots)
            self.allocated.remove(slots)

    def allocate_multi(pool, sizes):
        done = []
        try:
            for n in sizes:
                done.append((n, pool.alloc(n, fail=(n == 99))))
            return done
        except BaseException:
            # 回滚已分配的
            for n, slots in reversed(done):
                pool.free(slots)
            raise

    pool = FakePool()
    try:
        allocate_multi(pool, [2, 3, 99])
    except RuntimeError as e:
        print("捕获异常:", e)
    print("剩余槽位数:", len(pool.free_slots))  # 应该是 10，全部回滚
    assert len(pool.free_slots) == 10
    print("回滚成功 ✓")
    ```
  - **预期结果**：打印异常信息，剩余槽位数为 10，说明 2+3 个已分配槽位都被正确归还。对应 Commit 14 的 `rollback()` 逻辑。

- **知识点 5：注册表 + 工厂模式（Registry & Factory）**
  - **是什么**：用一个字典把"名字 → 构造函数"存起来，运行时按名字查找并创建对象，而不是写死 `if/else`。
  - **为什么重要**：SGLang 有几十种注意力后端（flashinfer、triton、fa3、ascend、intel_xpu...），如果每加一个就改主逻辑，代码会爆炸。注册表让插件可以"自己挂上去"。
  - **动手试试**：
    ```python
    BACKENDS = {}

    def register(name):
        def deco(factory):
            BACKENDS[name] = factory
            return factory
        return deco

    class FlashBackend:
        def __init__(self, runner): self.runner = runner
    class TritonBackend:
        def __init__(self, runner): self.runner = runner

    @register("flashinfer")
    def make_flash(runner): return FlashBackend(runner)

    @register("triton")
    def make_triton(runner): return TritonBackend(runner)

    def create_backend(name, runner):
        if name not in BACKENDS:
            raise ValueError(f"未知后端 {name}")
        return BACKENDS[name](runner)

    b = create_backend("triton", runner="dummy")
    print(type(b).__name__)  # TritonBackend
    print("已注册后端:", list(BACKENDS))
    ```
  - **预期结果**：输出 `TritonBackend` 和 `['flashinfer', 'triton']`。这正是 Commit 19 中 `register_oot_linear_attention_backend` 的核心思想。

---

## 🔥 重要更新

- **Commit 2 + 17（调度器重构）**：给调度器加了"请求不变量检查"并把 PD prefill 的 inflight 队列重命名为 transfer 队列。这是两个大型重构，说明 SGLang 正在系统性治理调度器的状态正确性，对稳定性和可维护性影响深远。
- **Commit 10（XPU 全量 prefill graph）**：Intel XPU 支持把整个 prefill 前向捕获成一张 graph，按 token 长度分桶。这是 XPU 性能优化的重要里程碑，也标志着 Intel 在推理框架上的持续投入。
- **Commit 13（BRIA Fibo 2.0 模型支持）**：新增一个完整的扩散 transformer 模型 + 文本条件注入，说明 SGLang 的多模态生成能力在快速扩张，不再只是 LLM 推理框架。
- **Commit 8（DeepSeek V4 decode 三合一融合 kernel）**：把 mHC post、combine、RMSNorm 三个算子融合成一个 CUDA kernel，直接优化 DeepSeek V4 的 decode 延迟，对生产部署有实际收益。

---

## 📋 逐条解读

### 1. [Chore] owner: update (#47875)
- **代码层面**：只改了 `.github/CODEOWNERS` 文件。把 `.github/workflows` 单独指派给 `@bingxche @michaelzhang-ai`，给 `distributed` 目录加了 `@hubertlu-tw`，新增 `monokernels` 目录的负责人。
- **新手概念课堂**：CODEOWNERS 就像"值日表"——GitHub 会根据它自动把 PR 指派给对应模块的负责人 review。谁负责哪块代码，一目了然。
- **对你有什么影响**：普通用户无感知。但如果你提 PR，知道该找谁 review 能加速合并。

### 2. [Refactor] scheduler: List every request holder and check no request is in two (#43901)
- **代码层面**：在 `scheduler.py`、`decode.py`、`hisparse_coordinator.py` 里新增 `request_lists()` 方法，把 waiting / grammar / chunked / running / prefill_bootstrap / prefill_transfer / decode_transfer 等所有持有请求的队列列出来。然后在 `init_invariant_checker` 里注册 `get_request_lists`，检查同一个请求不会同时出现在两个队列里。
- **新手概念课堂**：想象一个快递分拣中心，包裹可能出现在"待分拣""运输中""异常件"等多个区域。如果同一个包裹同时出现在两个区域，说明系统出 bug 了。这个 commit 就是给分拣中心加了个"盘点员"，定期核对每个包裹只能在一个区域。
- **对你有什么影响**：普通用户感知不到，但服务会更稳定，减少"请求卡死""重复输出"这类诡异问题。

### 3. [Test] linear: Expect replicated rows to skip TP symmetric memory (#47837)
- **代码层面**：修改 `test_linear_parallel_groups.py` 的断言，让 `replicated` 组的分配器期望 `disabled=True`（即跳过 TP 对称内存）。同时在 XPU graph runner 里加了个 `logging` import。
- **新手概念课堂**：TP（张量并行）会把权重切分到多张卡上，但有些层（如某些 MLP）的权重是"复制"到每张卡的。复制层不需要对称内存分配（对称内存是给需要跨卡通信的切分层用的），跳过它能省显存。
- **对你有什么影响**：用 TP 部署模型时显存占用可能略微下降，尤其是含大量复制层的模型。

### 4. [Perf] Hunyuan3D: Speed up shape export with GPU-side grid generation (#46905)
- **代码层面**：`hunyuan3d_vae.py` 里把原来用 `generate_dense_grid_points`（NumPy CPU 实现）生成网格点再 `.to(device)` 拷贝到 GPU 的流程，改成直接在 GPU 上用 `torch.linspace` + `torch.meshgrid` 生成。新增了一个单测验证 GPU 结果与 NumPy 参考一致。
- **新手概念课堂**：就像做蛋糕，原来是在厨房（CPU）做好面糊再端到烤箱（GPU），现在直接在烤箱里和面。省掉了搬运时间，分辨率越高省得越多。
- **对你有什么影响**：用 Hunyuan3D 生成 3D 模型时，导出速度更快，尤其是高分辨率网格。

### 5. [Fix] [Intel] gemma4: Bidirectional image attention on the intel_xpu backend (#46816)
- **代码层面**：`xpu_backend.py` 里给 `_bidirectional_image_attn_args` 加了 `sliding_layers_only` 参数，让全注意力层（无滑动窗口）保持因果注意力，只有滑动窗口层对图像 token 做双向注意力。`gemma4_mm.py` 和 `gemma4_diffusion.py` 相应调整调用。文档补充了 Intel XPU 上的行为说明。
- **新手概念课堂**：图像 token 之间应该"互相看见"（双向注意力），但文本 token 只能"看前面"（因果注意力）。Gemma 4 的设计是：滑动窗口层对图像双向，全注意力层保持因果。这个 commit 让 XPU 后端也遵守这个规则，和 HuggingFace 对齐。
- **对你有什么影响**：在 Intel GPU 上跑 Gemma 4 多模态时，图像理解质量提升，和 HF 结果更一致。

### 6. [Fix] PP: record received proxy tensors on the forward stream (#47614)
- **代码层面**：`scheduler_pp_mixin.py` 里在 `run_batch` 之前，对收到的 `pp_proxy_tensors` 里的每个非 CPU tensor 调用 `record_stream(self.forward_stream)`。
- **新手概念课堂**：流水线并行（PP）中，不同 stage 之间传张量。PyTorch 的显存分配器不知道这个张量还在被 forward stream 使用，可能提前回收导致数据被覆盖。`record_stream` 就是告诉分配器："这个张量还在用，别急着回收。"
- **对你有什么影响**：用 PP 部署时，减少偶发的数值错误或崩溃。

### 7. [Test] diffusion: align warmup and LoRA fixtures with runtime contracts (#47835)
- **代码层面**：修改 `test_cfg_parallel_warmup.py` 和 `test_fused_lora_compose.py` 两个测试文件。warmup 测试现在会验证 LTX2 的 halving stage 能接受 warmup 分辨率；LoRA 测试改用真实保存的 safetensors 文件而不是 mock。
- **新手概念课堂**：测试也要"与时俱进"。如果运行时逻辑变了（比如 warmup 分辨率规则改了），测试的 mock 数据也得跟着改，否则测试过了但实际是错的。
- **对你有什么影响**：普通用户无感知，但说明 LTX2 视频生成的 warmup 分辨率规则有调整。

### 8. [Perf] dsv41: Fuse decode mHC post, combine and RMSNorm into one kernel (#46088)
- **代码层面**：新增 `mhc_post_combine_norm_decode.cuh` CUDA 内核文件，把 DeepSeek V4 decode 阶段的 mHC post 处理、combine 和 RMSNorm 三个操作融合成一个 kernel，每个 CTA 处理一行，支持可选的 MXFP8 量化。
- **新手概念课堂**：三个算子分开跑，每次都要读写显存（约 5120 维隐藏层 × 多份），融合后只读写一次，省掉两次显存往返。就像做菜时切菜、炒菜、装盘分开做要洗三次锅，融合后一口锅搞定。
- **对你有什么影响**：跑 DeepSeek V4 decode 时延迟降低、吞吐提升。

### 9. [Fix] attention: Use ModelRunner query width for DSpark FlashAttention drafts (#46478)
- **代码层面**：`flashattention_backend.py` 里不再用 `resolve_num_tokens_per_req` 静态计算 draft 的 query 宽度，改用 `model_runner.decode_num_tokens_per_req(...)`，让 ModelRunner 根据实际配置（如 `sample_from_anchor`）决定。新增大量测试。
- **新手概念课堂**：投机解码时，draft 模型每步生成多少 token 不是固定的，取决于算法配置。原来写死一个公式算，遇到特殊配置（如 DSpark 的 anchor 采样）就算错了。
- **对你有什么影响**：用 DSpark 投机解码 + FlashAttention 时，draft 阶段不再出错，加速效果正确生效。

### 10. [Feat] [XPU] graph: Add full prefill graph (#30980)
- **代码层面**：XPU 后端新增 `full` prefill graph 后端，按 token 长度分桶捕获整个 prefill 前向为一张 `torch.xpu.XPUGraph`。支持通过 `--cuda-graph-config` 配置 `full_prefill_max_req` 控制请求槽位数。文档大幅更新。
- **新手概念课堂**：CUDA Graph 就像"录制宏"——把一串 GPU 操作录下来，之后一键重放，省掉每次启动 kernel 的开销。prefill 阶段计算量大但形状固定（按桶），非常适合录 graph。
- **对你有什么影响**：在 Intel GPU 上跑推理，prefill 阶段延迟降低，尤其适合固定长度输入的批处理场景。

### 11. [Feat] [NPU] attention: support mixed-chunk on the ascend attention backend (#47367)
- **代码层面**：`ascend_backend.py` 的 `forward_mixed` 方法大幅简化，不再单独实现一套逻辑，而是直接复用 `forward_extend`（因为 mixed batch 本质就是变长 extend，decode 行是 1-token extend）。只保留 MLA 无稀疏注意力时的 NotImplementedError。
- **新手概念课堂**：mixed-chunk 是指一个 batch 里既有 prefill 又有 decode 请求。原来 NPU 后端为它写了一套独立代码，现在发现"其实和 extend 是一回事"，直接复用，代码更简洁、bug 更少。
- **对你有什么影响**：在昇腾 NPU 上用 mixed-chunk 时支持场景更广（之前不支持 FIA=0 且 head_dim>128 的情况，现在支持了）。

### 12. [Doc] [AMD] cookbook: Verify MiniMax-M3.1-Flash High-Throughput on MI355X (#47844)
- **代码层面**：更新 MiniMax-M3.1 的 cookbook 文档和配置。MI355X 高吞吐配置新增 `--hicache-size 180`、`--hicache-write-policy write_back` 和 `HSA_USERPTR_FOR_PAGED_MEM=0` 环境变量，去掉 fp8 draft KV cache。标记为 `verified: true`。
- **新手概念课堂**：`HSA_USERPTR_FOR_PAGED_MEM=0` 是个 workaround——AMD 上 HiCache 的 host 内存池如果用默认的 pageable 内存，第一次写入会触发 HIP 非法内存访问。这个环境变量让它改用 `hipHostMalloc` 分配。
- **对你有什么影响**：在 MI355X 上部署 MiniMax-M3.1 有了经过验证的配置，实测 90,168 tokens/s per GPU。

### 13. [Feat] diffusion/model: add the BRIA Fibo 2.0 transformer and text conditioning (#46580)
- **代码层面**：新增 `bria_fibo2.py` 配置文件和一整套模型实现。Fibo 2.0 基于 Z-Image transformer，加了 Perceiver resampler（把文本特征压成 gist token）、5 个注入块（各自 cross-attend 文本）、per-token adaLN 调制（用于图像编辑）。
- **新手概念课堂**：这是个图像生成/编辑模型。Perceiver resampler 像"摘要器"，把长文本压成固定数量的关键 token；注入块像"多次提醒"，在生成过程中反复参考文本；per-token adaLN 让每个 token 根据噪声/干净状态有不同的调制参数，实现"编辑"能力。
- **对你有什么影响**：SGLang 现在支持 BRIA Fibo 2.0 图像生成/编辑模型。

### 14. [Fix] cache: Roll back host slots when allocation raises (#46606)
- **代码层面**：`hybrid_cache_controller.py` 里给 `resolve_host_transfers` 包了 try/except，失败时 `free(host_indices)` 回滚。`group.py` 的 `rollback()` 改成从后往前逐个释放，且清理异常不掩盖原始异常。
- **新手概念课堂**：见知识点 4。分配多个池的槽位时，如果第三个池失败，前两个池的槽位必须还回去，否则泄漏。回滚时还要注意：清理本身也可能失败，但不能让清理异常掩盖原始错误。
- **对你有什么影响**：长时间运行 HiCache 服务时，减少 host 内存泄漏导致的 OOM。

### 15. [Fix] moe: Avoid stale prefix reads in ep_scatter (#44974)
- **代码层面**：`ep_moe_kernels.py` 的 `_fwd_kernel_ep_scatter_1` 里，不再"先 cumsum 存回全局内存再读回"，改成每个 program 用 `tl.sum(tl.where(...))` 在本地算前缀。新增回归测试，包括 CUDA Graph 重放场景。
- **新手概念课堂**：见知识点 3。GPU 上不同线程块的写入顺序不保证，一个线程块写完另一个可能还没看到。本地重算避免了跨线程块的"写后读"依赖。
- **对你有什么影响**：MoE 模型在 expert 并行（EP）时，输出更稳定，减少偶发的 token 错乱。

### 16. [Perf] moe: Enable deferred finalize for NVFP4 on the routed TRT-LLM runner (#42654)
- **代码层面**：`fused_moe_triton/layer.py` 新增 `supports_routed_deferred_finalize` 判断；`flashinfer_trtllm.py` 里即使 `defer_finalize=True` 也分配 `symm_output` 缓冲区（绕过 flashinfer#6077 的 autotune 问题）。
- **新手概念课堂**：deferred finalize 是指 MoE 计算完后不立即做最终缩放/累加，而是延迟到下一步。routed kernel 之前不支持，现在支持了，但需要传一个"被忽略的缓冲区"来骗过 autotune 的查表逻辑（上游 bug 的 workaround）。
- **对你有什么影响**：用 NVFP4 量化 + TRT-LLM routed MoE 时，性能提升。

### 17. [Refactor] scheduler: Move deferred responses into an object and rename the PD prefill inflight queue to the transfer queue (#43902)
- **代码层面**：把延迟响应（deferred responses）封装成一个对象；把 `disagg_prefill_inflight_queue` 重命名为 `disagg_prefill_transfer_queue`，相关方法 `process_disagg_prefill_inflight_queue` 改为 `process_disagg_prefill_transfer_queue`。大量注释同步更新。
- **新手概念课堂**：命名即文档。"inflight" 有歧义（可能指"正在计算"或"正在传输"），改成 "transfer" 明确表示"KV 正在传输中"。把散落的延迟响应字段封装成对象，方便管理和测试。
- **对你有什么影响**：普通用户无感知，但代码可读性提升，后续 bug 修复更容易。

### 18. [Fix] rust-server: Give each attention-DP listener its own frontend cores (#47599)
- **代码层面**：`scheduler.py` 里新增 `gather_frontend_cpu_share` 集合通信；`rust_server/config.py` 新增 `_frontend_cpu_key`（用 hostname + CPU affinity 标识进程）和 `_frontend_cpu_share`（计算本 rank 在同 CPU mask 的 peers 中的索引和总数），`_partition_cores` 支持 `cpu_share` 参数。
- **新手概念课堂**：attention DP 时每个 DP rank 都起一个 rust server listener，如果它们共享同一个 CPU mask，会互相抢核。这个 commit 让同 mask 的 listener 把核平均分掉，各用各的。
- **对你有什么影响**：用 Rust server + attention DP 部署时，前端不再因抢 CPU 核而卡顿。

### 19. [Feat] attention: Add OOT linear-attention backend registration (#47234)
- **代码层面**：`attention_registry.py` 新增 `OOT_LINEAR_ATTENTION_BACKENDS` 字典和 `register_oot_linear_attention_backend` 函数，`_create_linear_attention_backend` 在 out-of-tree 平台上查找注册的工厂函数，找不到则用默认。文档补充用法。
- **新手概念课堂**：见知识点 5。这是插件化设计——硬件厂商可以自己写一个 `GDNAttnBackend` 子类，注册进来，不用改 SGLang 主仓库。
- **对你有什么影响**：第三方硬件平台可以更方便地接入 SGLang，生态更开放。

### 20. [Perf] mla: Use masked K concatenation for padded head counts (#47743)
- **代码层面**：`forward_mha.py` 的 `_concat_and_cast_mha_k` 去掉了 `next_power_of_2(self.num_local_heads) == self.num_local_heads` 这个条件，允许 head 数不是 2 的幂（如 Kimi-K3 在 TP8 下 12 个 local head）也走 Triton 融合路径，因为 Triton helper 本身会 mask 掉 padding 的 head。
- **新手概念课堂**：原来要求 head 数必须是 2 的幂才敢用融合 kernel，怕越界。后来发现 Triton 实现里已经做了 mask，多出来的 head 会被忽略，所以这个限制可以去掉。
- **对你有什么影响**：Kimi-K3 等 head 数不是 2 的幂的模型，在 MLA + MHA 路径下也能享受融合 kernel 的加速。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)