# 🚀 vLLM & SGLang 每日更新 — 2026-11-28

> 自动生成于 2026-11-28 09:24 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：MoE 量化与专家并行（Expert Parallelism）**
  - **为什么社区在关注**：今天有多条 commit（#69889、#70018）围绕 MoE 的量化与 EP 权重加载展开。MoE 是 DeepSeek、Qwen 等大模型的标配结构，如何让 int4/int8 专家在 CPU/GPU 上正确加载、跳过非本 rank 专家，是部署大模型的关键工程问题。
  - **对新手意味着什么**：如果你在学大模型推理，"分布式权重加载 + 量化"是绕不开的组合技能。可以从读懂 `expert_map` 这个映射表开始。

- **关键词：SM100 / tcgen05 手写 CUDA kernel**
  - **为什么社区在关注**：#70329、#70160 两条 commit 都在为 NVIDIA 最新 Blackwell（SM100）架构手写 tcgen05 MMA kernel，涉及 TMA、mbarrier、TMEM、PDL 等底层机制。这说明 vLLM 正在为下一代 GPU 提前做极致性能优化。
  - **对新手意味着什么**：CUDA 高性能编程（尤其是 Hopper/Blackwell 的异步流水线）是稀缺技能。不必立刻上手，但值得了解 TMA、mbarrier 这些名词。

- **关键词：ROCm / AMD 异构适配**
  - **为什么社区在关注**：#64752、#68806、#69603 三条 commit 都是 ROCm 相关（HSA 工具库、gfx942 回退移除、CI 精度测试）。AMD MI300/MI355 正在成为 NVIDIA 之外的现实选择。
  - **对新手意味着什么**：异构计算（CUDA + ROCm）正成为主流，掌握 HIP/ROCm 基础会让你在求职时更有竞争力。

- **关键词：Rust 前端**
  - **为什么社区在关注**：#70077 让 Rust 前端开始支持 `/v1/systemone` 路由。vLLM 正在把 Python 前端逐步迁移到 Rust，追求更低延迟和更少内存占用。
  - **对新手意味着什么**：Rust 在基础设施领域持续升温，关注 vLLM 的 Rust 前端是了解"Python→Rust 迁移"真实案例的好窗口。

- **关键词：CI 与构建加速**
  - **为什么社区在关注**：#66438 优化镜像构建（sccache、arm64 缓存）、#70344 调整 XPU 测试资源。大项目的 CI 成本极高，构建加速是持续投入的方向。
  - **对新手意味着什么**：学会读 CI 配置（Buildkite YAML、Dockerfile）是参与开源项目的实用技能。

## 🧪 重要知识点 & 动手实验

- **知识点 1：MoE 的专家映射（expert_map）与 EP 权重加载**
  - **是什么**：MoE 模型有几十上百个"专家"，专家并行（EP）会把专家分散到不同 GPU 上。每个 GPU 只加载属于自己的专家，`expert_map` 就是"全局专家 ID → 本地槽位"的映射表，非本地专家记为 -1。
  - **为什么重要**：如果加载权重时不先检查 `expert_id == -1`，就会去查一个不存在的参数，直接报错或加载错误权重。Commit #70018 修的就是这个坑：在 EP requantization 时要先跳过非本地专家。
  - **动手试试**：
    ```python
    import torch
    # 模拟 8 个全局专家，2 个 rank，每个 rank 4 个本地专家
    num_global, num_ranks = 8, 2
    local_per_rank = num_global // num_ranks
    expert_map = torch.full((num_global,), -1, dtype=torch.int64)
    rank = 0
    start = rank * local_per_rank
    expert_map[start:start + local_per_rank] = torch.arange(local_per_rank)

    print("expert_map:", expert_map.tolist())
    for gid in range(num_global):
        lid = expert_map[gid].item()
        if lid == -1:
            print(f"global expert {gid}: 跳过（非本地）")
        else:
            print(f"global expert {gid}: 加载到本地槽位 {lid}")
    ```
  - **预期结果**：你会看到前 4 个专家被加载到槽位 0-3，后 4 个被跳过。理解这个逻辑后，再读 `routed_experts.py` 的 `weight_loader` 就不难了。

- **知识点 2：量化日志概率（logprobs）的类型校验**
  - **是什么**：OpenAI 兼容接口里 `logprobs` 参数表示"返回几个 token 的对数概率"，必须是整数。但 Python 里 `True` 是 `int` 的子类，`1.5` 也可能被误接受。
  - **为什么重要**：前端参数校验不严会导致下游出现难以排查的 bug。Commit #69334 显式拒绝 `bool` 和非整数的 `float`，并给出清晰的 400 错误。
  - **动手试试**：
    ```python
    def check_logprobs(field_name, value):
        # 复现 commit 里的判断逻辑
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not value.is_integer())
        ):
            raise ValueError(f"`{field_name}` must be an integer.")
        return True

    for v in [2, 2.0, 1.5, True, False, "2", None]:
        try:
            check_logprobs("logprobs", v)
            print(f"{v!r:>6} -> 通过")
        except ValueError as e:
            print(f"{v!r:>6} -> 拒绝: {e}")
    ```
  - **预期结果**：`2` 和 `2.0` 通过，`1.5`、`True`、`False`、`"2"` 被拒绝。注意 `True` 被拒绝是因为 `isinstance(True, bool)` 先命中——这是一个典型的 Python 陷阱。

- **知识点 3：Triton 与 batch invariance（批不变性）**
  - **是什么**：batch invariance 指"同一个请求，无论和谁拼成一个 batch，输出都完全一致"。某些 attention 后端（如 FA4）在不同 batch 大小下会有数值差异，所以 Commit #70234 在开启 `VLLM_BATCH_INVARIANT` 时强制走 Triton 后端。
  - **为什么重要**：在 RL 训练、可复现评测、A/B 测试中，输出必须可复现。不懂 batch invariance 会以为"模型不稳定"，其实是 kernel 的归约顺序变了。
  - **动手试试**：
    ```python
    import torch

    # 演示：不同归约顺序导致浮点结果不同
    x = torch.randn(1000, dtype=torch.float32)
    s1 = x.sum()
    s2 = x.reshape(10, 100).sum(dim=1).sum()  # 分组归约
    print(f"整体 sum: {s1.item():.10f}")
    print(f"分组 sum: {s2.item():.10f}")
    print(f"差值: {abs(s1 - s2).item():.3e}")
    ```
  - **预期结果**：两个和非常接近但不完全相等（差值约 1e-5 量级）。这说明浮点归约顺序会影响结果——batch invariance 就是要把这种差异消除或控制住。

- **知识点 4：mbarrier 与 TMA（异步内存拷贝）**
  - **是什么**：在 Hopper/Blackwell GPU 上，TMA（Tensor Memory Accelerator）可以异步把数据从全局内存搬到共享内存，mbarrier 是"进度条"——数据搬完就翻转状态，warp 等待它。
  - **为什么重要**：Commit #70277 把 `cute_utils.mbarrier` 从 nvvm dialect 改成内联 PTX，因为前者生成的指令语义不够精确。手写 kernel 的性能瓶颈往往就在这些异步原语的正确使用上。
  - **动手试试**（概念验证，无需 GPU）：
    ```python
    import threading, time

    barrier = threading.Barrier(2)

    def producer():
        time.sleep(0.2)
        print("producer: 数据已写入共享内存")
        barrier.wait()  # 相当于 mbarrier arrive

    def consumer():
        barrier.wait()  # 等待 producer 完成
        print("consumer: 检测到 barrier 翻转，开始读取")

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start(); t2.start(); t1.join(); t2.join()
    ```
  - **预期结果**：producer 先打印，consumer 后打印。这就是 mbarrier 的"等待-翻转"语义。真实 GPU 上，TMA 完成时硬件自动翻转 mbarrier，无需线程调度。

- **知识点 5：Docker 构建缓存与依赖约束**
  - **是什么**：Commit #67274 在安装 KV connector 时用 `--override` 文件锁定 OpenTelemetry 版本，防止第三方包把 vLLM 依赖的版本降级。Commit #66438 用 sccache 和 registry cache 加速镜像构建。
  - **为什么重要**：`pip install` 的依赖解析是"贪心"的，后安装的包可能覆盖先安装的版本，导致运行时报 `ImportError` 或行为不一致。约束文件（constraints/overrides）是生产环境的必备实践。
  - **动手试试**：
    ```bash
    # 创建一个虚拟环境，演示 constraints 的作用
    python -m venv /tmp/demo-venv && source /tmp/demo-venv/bin/activate

    # 不加约束：安装一个会升级依赖的包
    pip install "requests==2.31.0"
    pip install "some-package-that-needs-newer-urllib3" 2>/dev/null || true

    # 加约束：锁定版本
    echo "urllib3==2.0.7" > /tmp/constraints.txt
    pip install -c /tmp/constraints.txt "requests==2.31.0"
    pip show urllib3 | grep Version
    ```
  - **预期结果**：加约束后 `urllib3` 版本被锁定为 2.0.7，不会因后续安装被意外升级。这就是 Dockerfile 里 `-c /tmp/kv-constraints.txt` 和 `--override` 的作用。

## 🔥 重要更新

- **#70018 [Bugfix][MoE] EP requantization 跳过非本地专家**：修复了专家并行下权重加载可能查错参数的 bug，对多卡部署 MoE 模型影响直接，属于"不修就炸"的级别。
- **#70329 / #70160 [Perf][Qwen4Exp] SM100 tcgen05 kernel**：为 Blackwell 架构新增融合 kernel（HC up-gate、down+inject+SiLU），是面向下一代 GPU 的深度性能优化，代表 vLLM 的前沿方向。
- **#69334 [Bugfix][Frontend] 拒绝非整数 logprobs**：修复了 `True` 被当作 `1`、`1.5` 被静默接受的参数校验漏洞，影响所有 OpenAI 兼容接口用户。
- **#70077 [Rust Frontend] 支持 /v1/systemone**：Rust 前端功能持续补齐，是前端迁移路线图上的重要一步。

## 📋 逐条解读

### 1. [Perf][Model] Export Bailing V2.5 linear attention prefill checkpoints (#69244)
- **代码层面**：在 `tests/kernels/attention/test_lightning_attn.py` 新增测试 `test_bailing_prefill_exports_state_at_checkpoint`，验证 `linear_attention_prefill_and_mix` 在指定 offset 处导出 prefill 中间状态（checkpoint），并检查 decode 行、未预留的 null block 不受影响。
- **新手概念课堂**：想象你在读一本很长的书（prefill），读到第 512 页时夹一个书签（checkpoint）。之后可以从书签处继续读，而不必从头再来。线性注意力（linear attention）的中间状态就是那个"书签"。
- **对你有什么影响**：普通用户无感，但做长上下文推理或 prefix caching 的开发者会受益于更精确的状态复用。

### 2. [Perf][Qwen4Exp] HC up-gate kernel restructure, early PDL trigger, SM100 down from M = 9 (#70329)
- **代码层面**：重写 `csrc/qwen4_exp/hc_up_gate_mix.cu` 的 warp 分工：warp 0 专门做 w_up 的 TMA、warp 1 做 lowrank 的 TMA、warp 2 做 tcgen05.mma 发射、warp 4 负责 TMEM 分配释放；新增 `elect_lane()`、`arrive_expect_tx()` 等内联 PTX 辅助函数；去掉 `cudaTypedefs.h` 依赖。
- **新手概念课堂**：这就像一条流水线工厂重新排班——原来一个人干两件事（TMA + 发指令），现在拆成专人专岗，减少等待。PDL（Programmatic Dependent Launch）是"提前按启动键"，让下一个 kernel 在上一个还没完全结束时就开始准备。
- **对你有什么影响**：使用 Qwen4Exp 模型的用户在 SM100 GPU 上会获得更高吞吐，普通用户无感知。

### 3. [CPU] Support DeepSeekV3 routing in x86 int4/int8 MoE experts (#69889)
- **代码层面**：在 `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py` 中，为 `CPUExpertsInt4`、`CPUExpertsInt8`、`ZenCPUExpertsInt8` 等类的 `_supports_routing_method` 增加 `RoutingMethodType.DeepSeekV3`；注释说明路由由 `CPURouter` 在 `apply()` 之前完成，所以专家层不必关心打分方式。
- **新手概念课堂**：DeepSeekV3 的路由是"sigmoid + bias + grouped top-k"。以前 CPU 专家层说"我不认识这种路由"，现在说"路由是上层的事，我只管算，来者不拒"——职责分离。
- **对你有什么影响**：在 x86 CPU 上跑 DeepSeekV3 或 LFM2.5 这类模型的 int4/int8 量化版，不再因路由类型不匹配而报错。

### 4. [Bugfix][MoE] Skip non-local experts before parameter lookup during EP requantization (#70018)
- **代码层面**：在 `routed_experts.py` 抽出 `_uses_global_input_scale()` 辅助函数；在 `load_weights` 中先拿到 `quant_method`，在遍历权重前跳过 `expert_id == -1` 的非本地专家，避免去查不存在的参数。新增 4 卡测试 `test_modelopt_nvfp4_mxfp4_ep_loads_local_experts`。
- **新手概念课堂**：EP 就像 8 个人分 100 张卡片，每人只拿自己那份。以前有人不看名单就去抽屉里找"第 99 号卡片"，抽屉里根本没有，就崩了。现在先查名单（expert_map），不是自己的直接跳过。
- **对你有什么影响**：多卡 + 专家并行 + 在线量化（mxfp4/nvfp4）的部署不再因加载阶段崩溃，属于稳定性修复。

### 5. [Kernel][Perf] Fused GDN MTP: support 16-token verify windows (#70253)
- **代码层面**：`csrc/gdn/fused_gdn_decode.h` 把 `kMaxMtpTokens` 从 9 改为 16；`fused_gdn_decode_16bit_kernel.cu` 把原来多个 `__shared__` 数组重构为一个 `TokenSmem` 结构体数组，改用 `extern __shared__` 动态共享内存，并加 `static_assert` 保证 16 token 不超过 48KB。
- **新手概念课堂**：MTP（Multi-Token Prediction）是"一次猜多个 token"，验证窗口越大越省时间。共享内存（shared memory）是 GPU 上的"高速白板"，但容量有限（通常 48KB），所以要把数据结构打包得更紧凑。
- **对你有什么影响**：使用投机解码（spec decode）的用户在 GDN 模型上可获得更长的验证窗口，吞吐提升。

### 6. [Bugfix][Frontend] Reject non-integer logprobs values in CompletionRequest (#69334)
- **代码层面**：`vllm/entrypoints/openai/completion/protocol.py` 的 `check_logprobs` 增加判断：`bool` 一律拒绝，`float` 必须 `is_integer()`；同步修正测试里的 `True`/`False` 为 `1`/`0`。
- **新手概念课堂**：Python 里 `True == 1` 且 `isinstance(True, int)` 为真，所以 `logprobs=True` 以前会被当成 `1` 静默接受。这是经典的"布尔是整数的子类"陷阱。
- **对你有什么影响**：如果你误传了 `logprobs=True` 或 `1.5`，现在会收到清晰的 400 错误，而不是奇怪的下游行为。

### 7. [Bugfix][RL] Destroy the previous NCCL communicator when the weight-transfer engine is re-initialized (#64335)
- **代码层面**：`NCCLWeightTransferEngine` 在 `init_weight_transfer_engine` 时先销毁旧的 communicator；测试新增 `stale_master_port` 参数模拟"trainer 崩溃后重连"场景；文档补充说明。
- **新手概念课堂**：NCCL communicator 像一条专用电话线，重新初始化时如果不挂断旧线，新线可能连不上或资源泄漏。RL 训练中 trainer 重启是常态，必须能干净地重连。
- **对你有什么影响**：做 RLHF / 在线权重更新的用户，trainer 重启后不再需要手动重启推理服务。

### 8. [XPU][CI] Run Intel V1 test on 24+ GiB GPUs (#70344)
- **代码层面**：`.buildkite/intel_jobs/test-intel.yaml` 把 Intel GPU 测试从 `intel_gpu` 设备切到 `intel_gpu_k8s`，使用 `b70-1` 资源模板，要求 24GB+ 显存；测试命令改为直接 `pytest` 并支持分片。
- **新手概念课堂**：CI 配置就是"自动化测试的排班表"。原来跑在 16GB 卡上可能 OOM，现在换到 24GB+ 的 B70 卡，测试更稳定。
- **对你有什么影响**：Intel GPU 用户遇到的 bug 能更早被发现和修复，间接提升 XPU 支持质量。

### 9. [Rust Frontend] Serve /v1/systemone with next-token reads (#70077)
- **代码层面**：`rust/src/server/src/routes.rs` 新增 `/v1/systemone` 路由；`middleware/offload.rs` 把 `/v1/decisions` 和 `/v1/systemone` 加入 offload 路径；新增 `serde-json-fmt` 依赖；CI 移除 "not systemone" 的过滤。
- **新手概念课堂**：Rust 前端是 vLLM 的"新大门"，用 Rust 重写 HTTP 层以降低延迟。`/v1/systemone` 是一个决策/结构化输出接口，"next-token reads" 意味着它按 token 流式读取。
- **对你有什么影响**：使用 Rust 前端的用户现在可以用 `/v1/systemone`，功能与 Python 前端对齐。

### 10. [Refactor][Weight Transfer] Expose transport hooks on sparse NCCL engines (#70341)
- **代码层面**：`sparse_nccl_engine.py` 把 `worker_init_process_group`、`torch.cuda.current_stream`、`open_trainer_endpoint` 抽成可覆写的方法（`_init_process_group`、`_current_stream`、`_open_trainer_endpoint`），方便子类替换传输实现。
- **新手概念课堂**：这是"模板方法模式"——把可变的部分做成钩子（hook），子类只需覆写钩子就能换传输层（比如从 CUDA 换成别的设备）。`Any` 返回类型是为了让子类可以返回非 CUDA 对象。
- **对你有什么影响**：普通用户无感，但做自定义权重传输（如跨厂商 GPU）的开发者可以更容易扩展。

### 11. [ROCm] Skip RTLD_GLOBAL libtorch_cpu.so promotion when HSA_TOOLS_LIB is set (#64752)
- **代码层面**：`vllm/env_override.py` 的 `_maybe_promote_torch_symbols_for_rocm` 在检测到 `HSA_TOOLS_LIB` 环境变量时提前返回，不执行 `ctypes.CDLL(..., RTLD_GLOBAL)`；新增测试 `test_rocm_rtld_global_skip.py` 验证三种场景。
- **新手概念课堂**：`RTLD_GLOBAL` 是把动态库的符号"广播"到全局，让其他库也能用。但 ROCm 的 HSA 工具库（如 rocprof）依赖 `OnLoad` 回调，如果 libtorch_cpu.so 先以 GLOBAL 方式加载，会把 libhsa-runtime64 提前拉进全局作用域，导致工具的回调不触发——相当于"抢跑"破坏了别人的初始化顺序。
- **对你有什么影响**：在 AMD GPU 上用 rocprof 等性能分析工具时，不再因 vLLM 的库加载顺序而失效。

### 12. [Bugfix][ROCm][MiniMax-M3] Drop the gfx942 native QK-norm/RoPE fallback (#68806)
- **代码层面**：删除 `vllm/models/minimax_m3/amd/model.py` 中 gfx942 专用的 QK-norm/RoPE 融合回退路径，以及对应测试 `test_gfx942_native_qknorm_rope_writes_main_and_index_caches`；统一走通用实现。
- **新手概念课堂**：gfx942 是 MI300 的架构代号。"回退路径"是"老卡专用补丁"，维护成本高且容易和新代码不一致。删掉它意味着通用实现已经足够好，不需要特判。
- **对你有什么影响**：在 MI300 上跑 MiniMax-M3 的代码路径更统一，减少"只在某张卡上出问题"的诡异 bug。

### 13. [−0.2 min][CI] Speed up image builds: arm64 CPU registry cache + sccache, uncapped Rust relink (#66438)
- **代码层面**：`image_build_cpu.sh` 增加 `x86|arm64` 参数，arm64 用独立的 `arm64-cpu` 缓存标签；启用 `USE_SCCACHE=1`；Rust 重链接不再限制并行度。
- **新手概念课堂**：sccache 是"编译结果缓存"，同一个源文件编译过一次就复用，像"做过的菜放冰箱"。registry cache 是把 Docker 层缓存推到镜像仓库，跨机器复用。
- **对你有什么影响**：CI 构建时间缩短，间接加快 PR 合并速度；自己构建镜像时也可以学这套缓存策略。

### 14. [Perf] Fuse SiLU-and-mul with group-128 FP8 input quant for the CUTLASS block-FP8 linear (#69778)
- **代码层面**：新增融合路径，把 `SiluAndMul` 激活和 group-128 FP8 动态量化合并成一个 kernel，直接喂给 `CutlassFp8BlockScaledMMKernel`；测试覆盖 M=1,2,7,64,512 并检查 scale 的列主序布局（`stride == (1, num_tokens)`）。
- **新手概念课堂**：原来流程是"算 SiLU → 乘 → 转 FP8 → 做 GEMM"，中间结果要写回显存再读出来。融合后一步到位，省两次显存往返。注意 scale 的布局：CUTLASS block kernel 读列主序 scale，M>1 时行主序会算错——这是测试特意覆盖的坑。
- **对你有什么影响**：使用 FP8 量化的模型推理吞吐提升，尤其是 MLP 层。

### 15. [Bugfix][Transformers backend] Apply checkpoint converters to the renamed weights (#69995)
- **代码层面**：`transformers/base.py` 的 `load_weights` 调整顺序：先做权重重命名（renaming），再应用 checkpoint converters（如 `Split` 拆分融合的 GQA QKV）；测试从直接调 `_drop_ignored_unexpected` 改为走完整 `Base.load_weights` 路径。
- **新手概念课堂**：checkpoint converter 是"权重格式转换器"，比如把 HuggingFace 融合的 `[q|k|v]` 大矩阵拆成三个。如果先拆分再重命名，名字对不上就找不到目标参数——顺序错了就全乱套。
- **对你有什么影响**：使用 Transformers backend 加载 Sarvam-30B 等融合 QKV 的模型时，权重能正确拆分和加载。

### 16. [Bugfix][Attention] Use Triton for Gemma 4 batch invariance (#70234)
- **代码层面**：`vllm/model_executor/models/config.py` 中，Gemma4 选择 FA4 的条件加上 `not envs.VLLM_BATCH_INVARIANT`；日志文案从 "FA4 not available" 改为 "FA4 is unavailable for this configuration"。
- **新手概念课堂**：FA4（FlashAttention 4）速度快但不同 batch 大小下数值不完全一致。开启 batch invariance 时，宁可牺牲一点速度也要保证"同输入同输出"，所以强制走 Triton 后端。
- **对你有什么影响**：开启 `VLLM_BATCH_INVARIANT=1` 跑 Gemma 4 时，输出可复现，适合评测和 RL。

### 17. [ROCm][CI] Bound the logprob shift in the per-token-head KV accuracy test; run all prompts for int4 (#69603)
- **代码层面**：`test_per_token_kv_cache.py` 引入 `check_logprobs_shift` 和按 dtype 的 `LOGPROB_SHIFT_BOUNDS`（int4: 0.3/2.0，int8: 0.065/0.45，fp8: 0.105/1.0）；关闭 V1 多进程保证请求分组可复现；int4 因精度太低不检查 top-k 命中，只检查 shift 边界。
- **新手概念课堂**：量化 KV cache 会引入误差，测试不能要求"完全相等"，而要设"误差上限"。int4 只有 16 个量化级别，误差天然比 int8/fp8 大，所以边界更宽。
- **对你有什么影响**：使用 per-token-head KV 量化的用户，CI 能更准确地捕捉精度回归，而不是因过严的断言频繁误报。

### 18. [Bugfix][Docker] Keep vLLM's OpenTelemetry versions when installing KV connectors (#67274)
- **代码层面**：`docker/Dockerfile` 安装 KV connector 时挂载 `requirements/test/kv_connectors_cuda.overrides` 并通过 `--override` 传入；同时保留 `${UV_OVERRIDE}` 以不覆盖 NCCL 的 override 文件。
- **新手概念课堂**：`--override` 是 uv 的"强制指定版本"机制，优先级高于普通依赖解析。lmcache 等 KV connector 可能把 OpenTelemetry 降到不兼容版本，override 文件就是"钉死版本"的钉子。
- **对你有什么影响**：使用官方 Docker 镜像 + KV connector 时，OpenTelemetry 相关功能（tracing/metrics）不会因版本冲突而失效。

### 19. [Perf][Qwen4Exp] Add SM100 tcgen05 fused HC down+inject+SiLU kernel (#70160)
- **代码层面**：新增 `csrc/qwen4_exp/hc_down_silu.cu`（918 行），实现融合的 down GEMM + inject + SiLU：`acc = xn @ w_down^T`，前 320 列做 `silu(acc/HC)` 得 lowrank，后 4 列直接输出 inj_out；支持 SPLITK > 1（decode）和 SPLITK == 1（prefill 持久化 2-SM）两种模式；CMakeLists 加入新源文件。
- **新手概念课堂**：这是把三个操作（down 投影、注入、SiLU 激活）塞进一个 kernel，减少显存往返。split-K 是"把 K 维度切给多个 CTA 并行算，最后用 DSMEM 归约"，适合 M 小（decode）的场景。
- **对你有什么影响**：Qwen4Exp 模型在 Blackwell GPU 上 decode 和 prefill 都更快。

### 20. [Bugfix][CuTe DSL] Use inline PTX in cute_utils.mbarrier (#70277)
- **代码层面**：`vllm/cute_utils/mbarrier.py` 把 `nvvm.mbarrier_txn` 调用改为 `llvm.inline_asm` 直接写 PTX（`mbarrier.arrive.{order}.{space}.shared::{space}.b64`），并加 `_validate` 检查 space/order 参数；非 relaxed 序时加 `~{memory}` clobber。
- **新手概念课堂**：CuTe DSL 的 nvvm dialect 生成的 mbarrier 指令语义不够精确（比如内存序处理），直接写 PTX 可以精确控制。`~{memory}` clobber 告诉编译器"这条指令会改内存，别乱重排"。
- **对你有什么影响**：使用 CuTe DSL 写自定义 kernel 的开发者，mbarrier 行为更可预测，减少难查的同步 bug。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：AMD MI355X / gfx950 深度优化**
  - 一句话通俗解释：AMD 的旗舰 AI 显卡（MI355X）和它的 GPU 架构代号（gfx950）正在被 SGLang 针对性打磨，包括 MoE 路由、量化、attention tile 尺寸等。
  - 为什么社区在关注：连续多个 commit（1、3、19）都在为 AMD 平台做专用 kernel，说明 SGLang 正在把 AMD 从"能跑"推进到"跑得好"，异构计算（NVIDIA + AMD + NPU）已经是硬需求。
  - 对新手意味着什么：如果你只熟悉 CUDA，现在正是学习 HIP/ROCm、Triton 多后端适配的好时机——这类"跨硬件调优"岗位会越来越多。

- **关键词：MoE（混合专家）推理优化**
  - 一句话通俗解释：MoE 模型每层有很多"专家"小网络，每个 token 只激活其中几个，省算力但路由逻辑复杂。
  - 为什么社区在关注：commit 1、3、4、19 都在改 MoE 的 router、sorting、量化，说明大模型（GLM、Qwen3.5、Kimi）普遍 MoE 化，推理引擎必须把"选专家"这一步做到极致快。
  - 对新手意味着什么：理解 top-k 路由、专家并行、排序融合是进入大模型推理优化领域的敲门砖。

- **关键词：确定性推理（Deterministic Inference）**
  - 一句话通俗解释：同样的输入，每次跑出来必须一模一样的结果，不能因为 GPU 调度顺序不同而"抖动"。
  - 为什么社区在关注：commit 4、9、15 都在围绕"batch-invariant"和"确定性 cut"做文章，说明生产环境（尤其是 RL 训练、评测、合规场景）对可复现性的要求越来越高。
  - 对新手意味着什么：学会区分"浮点累加顺序"和"算法逻辑"对结果的影响，是写出可复现系统的关键。

- **关键词：NPU（昇腾）生态适配**
  - 一句话通俗解释：华为昇腾 AI 芯片（NPU）上的 SGLang 支持，包括 attention、KV cache、DSA 等。
  - 为什么社区在关注：commit 2、10、13、16、20 大量涉及 NPU，说明国产算力平台正在被认真对待，不再是"边角料"。
  - 对新手意味着什么：如果你在国内做 AI 基础设施，昇腾/CANN 相关经验会非常抢手。

- **关键词：KV Cache 分层与卸载（HiCache / Layer Split）**
  - 一句话通俗解释：把 KV cache 按层拆开、放到不同存储层级（显存/内存/SSD），缓解显存不够用的问题。
  - 为什么社区在关注：commit 7、10、16 都在改 storage backup、layer split、prefetch，说明长上下文推理的瓶颈已经从算力转向"显存墙"。
  - 对新手意味着什么：理解 KV cache 的生命周期（prefill→decode→淘汰→备份）是做大模型 serving 的基本功。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：MoE 路由的"一次启动"融合（One-launch Router）

- **是什么**：MoE 模型每个 token 要经过"算 logits → softmax → 选 top-k → 归一化 → 写 ids/weights"这一串操作。传统做法是多个 kernel 依次启动，每次启动都有开销。"One-launch"就是把这串操作塞进一个 GPU kernel 里一次做完。
- **为什么重要**：decode 阶段 batch 很小（1~16 个 token），kernel 启动开销占比极高。commit 1 里 GLM-5.3-Flash 就是靠这个把 router 从多次启动压成一次。不懂这个，你写的推理代码在小 batch 下会慢得离谱。
- **动手试试**：用 Python 模拟"多次启动 vs 一次启动"的开销差异（真实 GPU 上更明显）：

```python
import time

def many_launches(n):
    total = 0
    for _ in range(n):
        total += sum(range(100))  # 模拟每次 kernel 启动的固定开销
    return total

def one_launch(n):
    return sum(sum(range(100)) for _ in range(n))

n = 10000
t0 = time.perf_counter(); many_launches(n); t1 = time.perf_counter()
t2 = time.perf_counter(); one_launch(n);   t3 = time.perf_counter()
print(f"多次启动: {(t1-t0)*1000:.2f} ms")
print(f"一次启动: {(t3-t2)*1000:.2f} ms")
```

- **预期结果**：一次启动明显更快。在真实 GPU 上，这个差距会从"毫秒"变成"微秒级 × 启动次数"，小 batch decode 时能差 2~5 倍。

### 知识点 2：MXFP4 量化与 E2M1 浮点格式

- **是什么**：MXFP4 是一种 4-bit 浮点量化格式，每个数只有 4 个 bit（1 符号 + 2 指数 + 1 尾数），叫 E2M1。它比 INT4 能表示更大的动态范围，适合权重量化。
- **为什么重要**：commit 3 里 `_f32_to_e2m1` 就是在做"把 FP32 舍入到 E2M1 网格"的操作。量化做得好，模型能塞进更小显存、跑得更快；做得差，精度崩盘。不懂舍入规则（round-to-nearest-even）就会写出有偏的量化器。
- **动手试试**：用 numpy 手动实现一个简化版 E2M1 舍入，观察它和普通 round 的区别：

```python
import numpy as np

def f32_to_e2m1_sim(x):
    # 简化版：E2M1 可表示值约为 {0, 0.5, 1, 1.5, 2, 3, 4, 6}（含符号）
    grid = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    sign = np.sign(x)
    ax = np.abs(x)
    ax = np.clip(ax, 0, 6)  # 饱和到 6
    # 找最近的网格点（round-to-nearest-even 的简化）
    idx = np.abs(grid[None, :] - ax[:, None]).argmin(axis=1)
    return sign * grid[idx]

x = np.array([0.1, 0.7, 1.2, 2.6, 5.9, 7.0, -3.2])
print("原始:", x)
print("E2M1:", f32_to_e2m1_sim(x))
```

- **预期结果**：你会看到 0.1→0、0.7→0.5、1.2→1、2.6→3、5.9→6、7.0→6（饱和）、-3.2→-3。理解"饱和"和"最近网格"是量化器的核心。

### 知识点 3：确定性推理与 batch-invariant

- **是什么**：同样的输入，不管 batch 里有多少条、GPU 怎么调度，输出必须完全一致。commit 4 的做法是：在确定性模式下，让 MoE 路由函数"eager 执行"（不走 torch.compile），因为 Inductor 会按测速结果选不同的 kernel 配置，导致浮点累加顺序变化。
- **为什么重要**：RL 训练、模型评测、合规审计都要求可复现。不懂这个，你会遇到"同样的 prompt 两次跑出不同答案"的诡异 bug。
- **动手试试**：观察浮点累加顺序对结果的影响：

```python
import random

def sum_in_order(xs):
    s = 0.0
    for x in xs:
        s += x
    return s

xs = [0.1] * 10 + [1e-16] * 10
random.seed(0); random.shuffle(xs)
a = sum_in_order(xs)
random.seed(1); random.shuffle(xs)
b = sum_in_order(xs)
print(f"顺序 A: {a!r}")
print(f"顺序 B: {b!r}")
print(f"相等? {a == b}")
```

- **预期结果**：你会看到两个值可能不相等（浮点加法不满足结合律）。这就是为什么"确定性推理"需要固定累加顺序。

### 知识点 4：KV Cache 分层存储与 Prefetch

- **是什么**：KV cache 是 attention 计算时缓存的 key/value 张量。长上下文时它非常大，放不下显存就要"卸载"到 CPU 内存甚至 SSD，需要时再"预取"（prefetch）回来。
- **为什么重要**：commit 7、9、10、16 都在改这套机制。commit 9 特别有意思：它把"最短 prefill 优先"的 backlog 计算和 `/v1/loads` 的负载报告统一成一个函数 `prefill_load_tokens`，避免两处逻辑不一致。
- **动手试试**：用 Python 模拟一个简单的 LRU KV cache 淘汰 + prefetch 逻辑：

```python
from collections import OrderedDict

class SimpleKVCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return "hit"
        self.misses += 1
        if len(self.cache) >= self.capacity:
            evicted, _ = self.cache.popitem(last=False)
            print(f"  淘汰: {evicted}")
        self.cache[key] = True
        return "miss"

kv = SimpleKVCache(capacity=3)
for k in ["a", "b", "c", "d", "a", "e"]:
    print(f"访问 {k}: {kv.access(k)}")
print(f"命中率: {kv.hits}/{kv.hits+kv.misses}")
```

- **预期结果**：你会看到 `d` 进来时淘汰 `a`，之后访问 `a` 又 miss。这就是 KV cache 淘汰策略的基本形态，真实系统会在此基础上加"分层"（显存→内存→SSD）。

### 知识点 5：TLX 后端与多 kernel 后端抽象

- **是什么**：TLX（Triton Low-Level Extensions）是 Triton 的扩展，允许你显式控制 warp group、异步内存搬运、barrier 等底层细节。commit 19 把 TLX kernel 从独立的 `tlx_ops/` 目录搬进按算子分组的 `ops/<group>/tlx_<op>.py`，并新增 `KernelBackend.TLX`。
- **为什么重要**：SGLang 的 kernel 系统支持多种后端（Triton、CUDA、CuteDSL、Helion、TLX…），同一个算子可以有多个实现，运行时按硬件选最优。不懂这个抽象，你就无法理解为什么一个 `ops.gemm` 能对应 5 种实现。
- **动手试试**：用 Python 模拟一个"多后端注册 + 优先级选择"的简化系统：

```python
from enum import IntEnum

class Backend(IntEnum):
    TRITON = 1
    CUDA = 2
    TLX = 3

registry = {}

def register(op, backend, fn, priority):
    registry.setdefault(op, []).append((priority, backend, fn))

def call(op, *args):
    candidates = sorted(registry[op], reverse=True)
    for prio, backend, fn in candidates:
        try:
            return fn(*args), backend.name
        except NotImplementedError:
            continue
    raise RuntimeError(f"no impl for {op}")

register("gemm", Backend.TRITON, lambda x: x * 2, priority=10)
register("gemm", Backend.TLX,   lambda x: x * 3, priority=20)

print(call("gemm", 5))  # TLX 优先级更高
```

- **预期结果**：输出 `(15, 'TLX')`，说明系统按优先级选了 TLX 后端。真实 SGLang 还会检查硬件是否支持该后端。

---

## 🔥 重要更新

1. **Commit 4：确定性推理下 MoE 路由改为 eager 执行** — 这是"可复现推理"的关键修复，RL 训练和评测场景直接受益，避免"同输入不同输出"。
2. **Commit 1 + 3：AMD GLM-5.3-Flash 的 MoE router 和 MXFP4 量化** — AMD 平台推理性能的重要推进，小 batch decode 场景提速明显。
3. **Commit 16：NPU 上 Kimi-K3.1 支持 DSA cache layer split 和 prefetch** — 国产算力平台长上下文推理的关键能力补齐。
4. **Commit 9：统一 prefill 负载定义** — 调度策略和 `/v1/loads` 负载报告用同一个函数，避免"调度器认为的负载"和"监控看到的负载"不一致，是运维友好性的重要改进。

---

## 📋 逐条解读

### 1. [Perf] [AMD] GLM-5.3-Flash: One-launch MoE router and narrower KDA V tile for MTP verify

- **代码层面**：改了两个文件。`fused_sigmoid_gating_recurrent.py` 里给 AMD gfx950 加了一个新的 launch 配置：当 KDA attention 在 verify 阶段、head 数是 16、K/V 维度 128 时，按 token 数选不同的 V tile 宽度（4/8/16），让 grid 更宽、并行度更高。`smallm_router_gfx950/__init__.py` 里把原来只支持 Qwen3.5 的 one-launch MoE router 扩展成同时支持 GLM-5.x（sigmoid + noaux_tc top-8），新增 `GlmArgs` 结构和 3 种变体配置。
- **新手概念课堂**：想象一个餐厅，MoE router 是"点菜员"，每个 token 都要点 8 个菜（专家）。传统做法是点菜员跑 5 趟（5 个 kernel 启动），现在改成一次点完（one-launch）。"V tile 宽度"就像切菜的砧板大小——batch 小的时候用小砧板（BV=4）能让更多厨师同时干活，batch 大了换大砧板（BV=16）。
- **对你有什么影响**：如果你在 AMD MI355X 上跑 GLM-5.3-Flash，decode 和 MTP verify 会更快；其他平台无感知。

### 2. [Fix] [NPU] ci: Fix Indexer test fixtures and move NPU unit tests to extra-a

- **代码层面**：把一批 NPU 单元测试的 CI 注册从 `register_cpu_ci`（base-a-test-cpu）改成 `register_npu_ci`（extra-a-test-npu），并在 `test_kimi_k31_prefill_cp.py` 里给 `backend` 补了一个新属性 `use_decode_indexer_request_shard = False`，因为 commit 13 新增了这个字段，测试 fixture 不补就会 AttributeError。
- **新手概念课堂**：CI 测试分组就像学校把考试分到不同考场——NPU 测试以前被误分到 CPU 考场，现在挪到 NPU 专用考场。fixture 补字段就像考试前检查文具，新加的字段（铅笔）没带就会挂。
- **对你有什么影响**：纯 CI 改动，普通用户无感知；但如果你自己写 NPU 测试，记得新字段要初始化。

### 3. [Perf] [AMD] GLM-5.3-Flash: Small-M MoE kernel and sort-fused MXFP4 quant for decode

- **代码层面**：`moe_sorting_small.py` 新增了 `_f32_to_e2m1`（FP32→E2M1 舍入）和 `_mx_quant_row`（按 32 个元素一组算 e8m0 scale，再量化到 MXFP4/MXFP8）。这是把"排序"和"量化"融合进一个 kernel，decode 小 batch 时少启动一次。
- **新手概念课堂**：MXFP4 就像把每个数字压缩成"4 个 bit 的缩写"——原来 32 bit 的浮点数，现在 1 个符号 + 2 个指数 + 1 个尾数。e8m0 scale 是每 32 个数共享一个"放大倍数"，像一箱货物共用一个标签。
- **对你有什么影响**：AMD 平台上 GLM-5.3-Flash 的 decode 显存占用更低、速度更快；量化精度由 E2M1 的饱和（最大 6.0）保证。

### 4. [Fix] deterministic: Run the MoE routing functions eagerly in batch-invariant mode

- **代码层面**：`topk.py` 新增装饰器 `_compile_unless_batch_invariant`，包住 `grouped_topk_gpu` 和 `biased_topk_impl`。逻辑是：如果开了 batch-invariant 模式，就走 `torch.compiler.disable` 的 eager 版本；否则走 `torch.compile`。原因是 Inductor 会按测速选不同 kernel 配置，导致浮点累加顺序变化，进而选出不同的专家。
- **新手概念课堂**：torch.compile 像"自动配菜机器人"，它会试几种切法选最快的，但不同切法做出来的菜味道（浮点结果）可能微差。确定性推理要求"每次味道一模一样"，所以只能让厨师手工切（eager）。
- **对你有什么影响**：开 `--enable-deterministic-inference` 时，MoE 路由结果可复现；代价是可能略慢（少了 compile 优化）。

### 5. [Perf] [NVIDIA] inkling2: Fuse SM120 block attn_res, loop controller, and RMSNorm

- **代码层面**：新增 `norm_ctrl_sm120.cuh`（1313 行 CUDA 头文件），把 attention residual、循环控制器、RMSNorm 三个操作融合成一个 kernel，专门针对 SM120（Blackwell 消费级架构）。用了 `ld.global.nc.L1::no_allocate.L2::256B.v8.b32` 这种底层 PTX 指令做向量化加载。
- **新手概念课堂**：融合 kernel 就像把"洗菜→切菜→炒菜"三个工序合并到一个流水线上，省去工序之间的搬运（显存读写）。`v8.b32` 是一次搬 8 个 32 位整数，像用大卡车代替小推车。
- **对你有什么影响**：SM120 显卡（如 RTX 50 系列）上跑 inkling2 模型会更快；其他架构无感知。

### 6. [Perf] [NVIDIA] inkling2: Size SM12x SWA decode/verify tiles to fit one SM wave

- **代码层面**：`inkling2_decode_attention_sm90.py` 新增 `_sm12x_swa_launch_config`，根据 batch、head 数、qlen、span 等参数算出 block_n 和 splits，目标是让 CTA 总数刚好塞进一个 SM wave（不浪费调度周期）。只对 sliding window attention（SWA）生效。
- **新手概念课堂**：SM wave 就像"一班公交车能坐多少人"——如果 CTA 数不是 SM 数的整数倍，最后一班车会空一半座位。这个函数就是在调"每班车拉多少人"，让座位坐满。
- **对你有什么影响**：SM12x 上 SWA decode 性能提升，尤其是小 batch 场景。

### 7. [Feat] load-report: Carry the storage backup token counters in /v1/loads

- **代码层面**：`LoadSnapshot` 新增 `total_backuped_tokens` 和 `total_backup_dropped_tokens` 两个字段；`MetricsCollector` 里 `log_backuped_tokens` 和 `log_backup_dropped_tokens` 除了打 Prometheus counter，还累加到普通整数变量；`load_inquirer.py` 把它们塞进 `/v1/loads` 返回。
- **新手概念课堂**：Prometheus counter 像"只增不减的计数器"，但 router 需要"两次采样之间的差值"来做负载均衡。所以额外存一份普通整数，方便直接读。
- **对你有什么影响**：用 `/v1/loads` 做路由的集群，现在能看到"备份到存储的 token 数"和"备份被丢弃的 token 数"，便于诊断存储瓶颈。

### 8. [Fix] rl: One tensor weight payload per engine rank under pipeline parallelism

- **代码层面**：`engine.py` 里 `_serialize_tensors_per_rank` 改名为 `serialize_tensors_per_rank`（公开 API），循环次数从 `tp_size` 改成 `tp_size * pp_size`。原因是 PP（流水线并行）下每个 stage 都要有自己的权重副本，原来只按 TP 数序列化会漏掉 PP 的 rank。文档也同步更新。
- **新手概念课堂**：TP 是"同一层切给多张卡"，PP 是"不同层放不同卡"。原来只给 TP 的每张卡发一份权重，PP 的第 2 个 stage 就没权重可用——就像给每个车间发图纸，但只发了第一车间的。
- **对你有什么影响**：用 PP + RL 训练时，`update_weights_from_tensor` 不再报错或静默丢权重。

### 9. [Fix] scheduler: Share one prefill-load definition between the shortest-prefill-first backlog and /v1/loads

- **代码层面**：`mem_cache/common.py` 新增 `prefill_load_tokens(seqlen, matched, storage_span)` 函数，计算"还需要 prefill 多少 token"（扣掉已匹配前缀和存储命中）。`schedule_policy.py` 的 `_update_shortest_prefill_load` 和 `load_inquirer.py` 的 `get_num_waiting_uncached_tokens` 都改成调用它，保证两边算法一致。
- **新手概念课堂**：以前"调度器算的 backlog"和"监控报的负载"是两套代码，可能算出不同数字——就像财务和销售各记一本账。现在统一成一个函数，两边永远对得上。
- **对你有什么影响**：`/v1/loads` 报的负载和调度器实际用的负载一致，router 做负载均衡更准。

### 10. [Feat] [NPU] hicache: Use kv_exchange for MHA KV and Mamba state transfers

- **代码层面**：`npu/utils.py` 删掉了 `os.environ.setdefault("SGLANG_HICACHE_HOST_MEM_BACKEND", "acl")`，因为新路径不再需要；`pool_host/mamba.py` 里 NPU 分支从"try/except 导入"改成"直接导入" `transfer_kv_all_layer_direct_lf_pf` 等，说明这些 kernel 已经稳定。
- **新手概念课堂**：HiCache 是"KV cache 的分层缓存"，`kv_exchange` 是新的传输通道（类似快递公司换了一家）。Mamba state 是 Mamba 模型的"记忆状态"，也要跟着搬。
- **对你有什么影响**：NPU 上跑 MHA/Mamba 混合模型时，KV cache 卸载/加载更稳、更快；不再依赖环境变量开关。

### 11. [Fix] diffusion: Keep MiniMax-H3's Qwen3-VL vision tower independent of the encoder TP width

- **代码层面**：`MiniMaxH3Qwen3VLConfig` 新增 `vision_replicated: bool = True`；`qwen3vl.py` 的 `Qwen3VL` 构造函数新增 `vision_replicated` 参数；`minimax_h3_qwen3vl.py` 里如果 `vision_replicated` 且 TP>1，就把 `deepstack_merger_list` 也加入 `layer_names`（流式加载）。图像 pipeline 配置里设 `vision_replicated=False`（图像走 TP 切分），视频走 `True`（视觉塔每 rank 复制一份）。
- **新手概念课堂**：TP 切分像"把一张大饼切成几份分给几个人"，但视觉塔的某些层（如 merger）切了会改变数值。视频发布要求"和单卡数学一致"，所以干脆每个 rank 复制一份，不切。
- **对你有什么影响**：MiniMax-H3 视频生成在 TP>1 时视觉输出和单卡一致；图像生成仍走 TP 切分。

### 12. [Fix] diffusion/loader: Fix direct GPU loading of VAEs with non-persistent buffers

- **代码层面**：`vae_loader.py` 新增 `_init_vae_with_meta_parameters` 上下文管理器，用 `register_module_parameter_registration_hook` 在参数注册时把 storage 换成 meta（占位），避免构造函数里的 buffer 值被误丢。同时新增 `mmap_safetensors_to_device` 路径，支持 B200 上 MiniMax-H3 和 Wan2.1 480p VAE 直接从 CPU mmap 加载到 GPU。
- **新手概念课堂**：non-persistent buffer 是"不存进 state_dict 的临时张量"（如位置编码）。直接 GPU 加载时如果先建 meta 参数再填值，这些 buffer 的初始值会被覆盖。这个 hook 就是"在参数注册那一刻拦截，只丢参数不丢 buffer"。
- **对你有什么影响**：B200 上 VAE 加载更快、显存峰值更低；其他设备行为不变。

### 13. [Feat] [NPU] dsa: Shard indexer requests in Decode and Verify

- **代码层面**：`ascend_backend.py` 新增 `use_decode_indexer_request_shard = parallel.attn_tp_size > 1`；`dsa_npu_indexer.py` 的 `_IndexerQueryShard` 新增 `request_rows: Optional[slice]` 字段，`take`/`gather` 支持"按 request 切分"（而不是按 token 切分），新增 `_build_decode_indexer_shard` 按 batch 均分给各 TP rank。
- **新手概念课堂**：DSA（DeepSeek Sparse Attention）的 indexer 要算 top-k。原来按 token 切分给各 rank，但 decode 时每个 request 只有 1 个 token，切了等于没切。现在按 request 切——rank 0 处理前一半 request，rank 1 处理后一半，all-gather 时补齐 padding。
- **对你有什么影响**：NPU 上 DSA 模型 decode/verify 时 TP 并行效率更高，小 batch 也能利用多卡。

### 14. [Fix] rust-server: Always send output_ids on /generate frames, empty without tokens

- **代码层面**：proto 文件 `GenerateResponse.output_ids` 的注释从"Omitted while empty"改成"Set on every frame"；Rust 侧 `completions.rs` 新增测试 `zero_token_choice_reports_empty_token_ids`，确保零 token 的 choice 返回 `"token_ids": []` 而不是省略字段。
- **新手概念课堂**：JSON 里"字段不存在"和"字段是空数组"对客户端是两种行为——前者会让 `if 'token_ids' in resp` 为 False，后者为 True。Python 版 SGLang 一直返回空数组，Rust 版要对齐。
- **对你有什么影响**：用 Rust server 的 OpenAI 兼容接口时，客户端解析 `token_ids` 不会再因为字段缺失而报错。

### 15. [Fix] scheduler: Keep deterministic mamba prefill cuts on the kernel chunk grid

- **代码层面**：`schedule_policy.py` 重构 `mamba_prefill_chunk_grid`，新增 `deterministic_cuts_on_grid(tree_cache)` 判断"确定性推理 + 支持 mamba"；`PrefillAdder.__init__` 里 `strict_chunk_grid` 改为 `windows_retire_behind_checkpoints or deterministic_cuts_on_grid`，确保确定性模式下 prefill 切分点落在 kernel chunk 网格上。
- **新手概念课堂**：Mamba 的 kernel 按固定 chunk 大小处理序列，如果 prefill 切分点不在 chunk 边界上，kernel 会从切分点重新开始算，导致结果和"整段算"不同。确定性推理要求两者一致，所以强制切在网格上。
- **对你有什么影响**：开确定性推理 + Mamba 模型时，prefill 结果可复现；代价是可能少用一点 prefill 预算。

### 16. [Feat] [NPU] kimi-k3.1: Support DSA cache layer splitting and prefetch

- **代码层面**：`cp_tp_group_sharing_hook.py` 里把 `enable-dsa-cache-layer-split` 从"不支持"列表里排除（仅 KimiK31 + NPU）；`dsa_cache_layer_split.py` 新增 `NPUDSATokenToKVPool` 支持，`get_kv_size_bytes` 把 scratch 存储也算进内存报告；`ascend/conn.py` 的 `send_kvcache` 在 DSA layer split 时走 generic 路径。
- **新手概念课堂**：DSA cache layer split 是"把 KV cache 按层拆到不同 rank"，每层只存一部分。prefetch 是"提前把要用的层拉回来"。Kimi-K3.1 是 MoE + DSA 模型，层数多，拆分后显存压力小。
- **对你有什么影响**：NPU 上跑 Kimi-K3.1 长上下文时，KV cache 可以按层拆分 + 预取，显存利用率更高。

### 17. [Feat] metrics: Export load-scaled shortest-prefill-first backlog, boundary and budget gauges

- **代码层面**：`scheduler.py` 把 `policy.shortest_prefill_load_scaling` 传给 `metrics_reporter`；`metrics_reporter.py` 新增 `_update_shortest_prefill_load_stats`，计算 backlog、class boundary、token budget 三个指标；`metrics_collector.py` 的 `SchedulerStats` 新增对应字段。
- **新手概念课堂**：shortest-prefill-first 是调度策略——优先处理"prefill 工作量最小"的请求。backlog 是"排队总工作量"，boundary 是"分档边界"，budget 是"这一轮能给多少 token"。这三个指标暴露到 Prometheus，运维就能看到调度器在想什么。
- **对你有什么影响**：用 Prometheus 监控时，能看到 `sglang:shortest_prefill_backlog_tokens` 等新指标，便于调优。

### 18. [Fix] test: Pin Python mode in the native gRPC multi-tokenizer rejection test

- **代码层面**：`test_server_args.py` 的 `test_native_grpc_rejects_multi_tokenizer` 在 `assertRaises` 外面包了 `envs.SGLANG_RUST_SERVER.override(False)`，强制走 Python 模式。因为 Rust server 现在接受 multi-tokenizer，测试不 pin 就会失败。
- **新手概念课堂**：测试要"钉住环境"——同一个测试在不同默认配置下行为可能不同。`override` 就像给测试戴上"固定眼镜"，不管全局设置是什么，都按 False 跑。
- **对你有什么影响**：纯测试改动，普通用户无感知。

### 19. [Feat] [AMD] kernels: Add a TLX kernel backend and move the TLX kernels into their operator groups

- **代码层面**：`.pre-commit-config.yaml` 的 exclude 从 `tlx_ops/` 改成按算子列出 `ops/gemm/tlx_mm.py` 等；`kernels/README.md` 新增 TLX 后端章节和 kernel 来源表（上游路径 + commit）；新增 `KernelBackend.TLX` 枚举，TLX kernel 从独立目录搬进 `ops/<group>/tlx_<op>.py`。
- **新手概念课堂**：TLX 是 Triton 的"低层扩展"，能显式控制 warp group、异步拷贝、barrier。以前 TLX kernel 放在独立目录，现在按算子分组——就像图书馆从"按出版社排"改成"按学科排"，找起来更方便。
- **对你有什么影响**：AMD gfx950 上部分算子（GEMM、RMSNorm、MoE）可以用 TLX 实现，性能可能更好；调用方需要显式 opt-in（TLX 不在默认优先级里）。

### 20. [Test] [NPU] qwen3.6-27b: fix the 16k perf OOM and the GPQA W8A8 accuracy drift

- **代码层面**：CI workflow 给 16k 性能测试加了 `partitions: '{"size":2,"arr":[0,1]}'`（分 2 卡跑）；GPQA 精度测试删掉 `HCCL_DETERMINISTIC`（观察漂移是否复现），加 `GDN_ATTN_BACKEND_TRITON=1`；16k 性能测试加 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（缓解 OOM），`--mem-fraction-static` 从 0.70 降到 0.63。
- **新手概念课堂**：`expandable_segments` 是 PyTorch 的显存分配策略，允许"碎片合并"，缓解 OOM。`mem-fraction-static` 是"静态预留多少显存"，降下来给 KV cache 留更多空间。`HCCL_DETERMINISTIC` 是昇腾通信库的确定性开关，关掉可能更快但结果可能漂移。
- **对你有什么影响**：纯测试/CI 改动，普通用户无感知；但如果你在昇腾上跑 qwen3.6-27b，可以参考这些参数调优。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)