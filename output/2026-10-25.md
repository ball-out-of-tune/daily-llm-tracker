# 🚀 vLLM & SGLang 每日更新 — 2026-10-25

> 自动生成于 2026-10-25 06:47 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：KV Offload / KV Connector**
  - **一句话通俗解释**：把 GPU 显存放不下的 KV Cache（大模型"记忆"）挪到 CPU 内存或磁盘上，需要时再搬回来，让长上下文请求跑得下。
  - **为什么社区在关注**：今天 20 个 commit 里有 4 个（#61019、#63266、#61791、#63534）都在改 KV 连接器/卸载，说明长上下文 + 大模型让显存成为最紧的瓶颈，社区正把"显存不够用"当成头等大事。
  - **对新手意味着什么**：如果你在学推理系统，KV Cache 管理（分页、前缀复用、卸载、对齐）是当前最热、最容易做出贡献的方向之一。

- **关键词：MRV2 / DBO（微批重叠）**
  - **一句话通俗解释**：把一次前向计算拆成两个"微批"交替跑，一个在算时另一个在等通信，从而把 GPU 空转时间填满。
  - **为什么社区在关注**：今天有 2 个 bugfix（#63476、#63372）专治 DBO 的报错定位和路由专家行丢失，说明这个特性已经进生产，稳定性问题开始集中暴露。
  - **对新手意味着什么**：并发编程 + GPU 流同步是硬核技能，能读懂 `dbo_yield()` 这种协程式交错，你就超过了大多数只会调 API 的人。

- **关键词：AMD ROCm / MI325X / RDNA3**
  - **一句话通俗解释**：vLLM 正在给 AMD 的各种 GPU 写专属 kernel 和调优配方。
  - **为什么社区在关注**：今天有 3 个 commit（#63520、#62713、#60413）是 AMD 相关，包括给 RDNA3 手写 MXFP4 GEMV kernel，说明"非 NVIDIA 硬件"已从"能跑"进入"跑得好"阶段。
  - **对新手意味着什么**：CUDA/HIP 双栈能力是稀缺技能，异构计算岗位正在增加。

- **关键词：量化（ModelOpt / MXFP4 / FP8）**
  - **一句话通俗解释**：把模型权重从 16 位压到 8 位、4 位，省显存、提速度，精度尽量不掉。
  - **为什么社区在关注**：#63001 支持混合精度 draft 加载、#60413 新增 W4A8 kernel，量化格式越来越碎片化，加载器必须更聪明。
  - **对新手意味着什么**：理解 `quant_algo`、per-tensor/per-channel/per-group 缩放，是部署大模型的必备常识。

- **关键词：投机解码（Spec Decode / EAGLE / DFlash / DSpark）**
  - **一句话通俗解释**：让小模型先"猜"几个 token，大模型一次性验证，猜对就白赚速度。
  - **为什么社区在关注**：#61019、#63001 都围绕 EAGLE/MTP draft 的 KV 对齐和量化加载，投机解码已是高吞吐服务的标配。
  - **对新手意味着什么**：draft 模型、验证、接受率这些概念，是理解现代推理加速的关键。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 卸载与"部分尾部"对齐（来自 #61019、#63266）**
  - **是什么**：KV Cache 是模型生成时缓存的注意力键值；卸载就是把暂时不用的部分搬到 CPU，需要时再搬回 GPU。但"搬回来"必须对齐到 block/chunk 边界，否则数据会错位。
  - **为什么重要**：不对齐会导致模型输出错乱（commit #63266 的测试就是验证"冷启动预测必须保持不变"）。不懂对齐规则，你会遇到"开了 prefix caching 结果答案变了"的诡异 bug。
  - **动手试试**：
    ```bash
    # 用一个小模型开启 CPU offload + prefix caching，对比两次相同请求的输出
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --enable-prefix-caching \
      --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both"}'
    # 然后连续发两次相同的长 prompt，观察是否命中缓存
    curl localhost:8000/v1/completions -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"<长文本>","max_tokens":8}'
    ```
  - **预期结果**：第二次请求的 TTFT（首 token 延迟）明显降低，且输出与第一次一致；如果不对齐，输出会漂移。

- **知识点 2：DBO 微批交错与错误定位（来自 #63476、#63372）**
  - **是什么**：DBO 把一次 batch 拆成两个 micro-batch，用 `dbo_yield()` 让出执行权，实现"计算-通信重叠"。两个 micro-batch 在同一个线程组里协作。
  - **为什么重要**：如果某个 micro-batch 挂了，另一个还停在 yield 点，错误会被掩盖成下游 KeyError，极难排查。#63476 改成"报告第一个失败"而不是最小 ID。
  - **动手试试**：
    ```python
    # 简化版：用两个线程模拟 micro-batch 交错
    import threading
    e1, e2 = threading.Event(), threading.Event()
    trace = []
    def mb0():
        trace.append("mb0-enter"); e1.set(); e2.wait(); trace.append("mb0-exit")
    def mb1():
        e1.wait(); trace.append("mb1-enter"); e2.set(); trace.append("mb1-exit")
    t0, t1 = threading.Thread(target=mb0), threading.Thread(target=mb1)
    t0.start(); t1.start(); t0.join(); t1.join()
    print(trace)  # 观察交错顺序
    ```
  - **预期结果**：输出类似 `['mb0-enter', 'mb1-enter', 'mb0-exit', 'mb1-exit']`，理解"交错"就是两个执行流在 yield 点互相让路。

- **知识点 3：量化配置校验——在启动时就拒绝不兼容组合（来自 #63496）**
  - **是什么**：LoRA（低秩适配器）和 MegaMoE（超大专家混合 kernel）不兼容，新代码在配置校验阶段直接报错，而不是等到模型加载后才崩。
  - **为什么重要**："早失败"是工程最佳实践。晚失败会浪费几分钟加载时间，还可能留下半初始化的 GPU 状态。
  - **动手试试**：
    ```python
    from vllm.config.lora import LoRAConfig
    from vllm.config.kernel import KernelConfig
    lora = LoRAConfig(max_lora_rank=16)
    kernel = KernelConfig(moe_backend="deep_gemm_mega_moe")
    try:
        lora.verify_with_kernel_config(kernel)
    except ValueError as e:
        print("捕获到预期错误:", e)
    ```
  - **预期结果**：打印出 `LoRA is not supported with moe_backend='deep_gemm_mega_moe'...`，理解配置校验器的作用。

- **知识点 4：用 `torch.accelerator` 写硬件无关代码（来自 #63457）**
  - **是什么**：把 `torch.cuda.synchronize()` 换成 `torch.accelerator.synchronize()`，同一份代码在 CUDA 和 ROCm 上都能跑。
  - **为什么重要**：硬编码 `torch.cuda.*` 会让 AMD 用户直接崩。这是"一次编写、多硬件运行"的基本功。
  - **动手试试**：
    ```python
    import torch
    print("accelerator available:", torch.accelerator.is_available())
    print("device type:", torch.accelerator.current_accelerator())
    x = torch.randn(1024, 1024, device="cuda")
    torch.accelerator.synchronize()  # 替代 torch.cuda.synchronize()
    print("sync ok")
    ```
  - **预期结果**：在 NVIDIA 上打印 `cuda`，在 AMD 上打印 `hip`，代码无需改动。

- **知识点 5：投机解码 draft 模型的量化加载（来自 #63001）**
  - **是什么**：DFlash/DSpark 这类 draft 模型可能是混合精度（某些层 FP8、某些层 BF16），加载器需要把 checkpoint 里的模块名映射到 vLLM 内部前缀。
  - **为什么重要**：名字映射错了，量化层就建不出来，模型直接加载失败。这是"checkpoint 格式碎片化"时代的典型问题。
  - **动手试试**：
    ```bash
    # 查看一个 ModelOpt 混合精度 checkpoint 的 config.json 里 quant_algo
    python -c "
    import json
    cfg = json.load(open('config.json'))  # 换成你的模型路径
    print(cfg.get('quantization_config', {}).get('quantization', {}).get('quant_algo'))
    "
    ```
  - **预期结果**：看到 `MIXED_PRECISION` 或 `FP8` 等值，理解 vLLM 是如何据此选择 `modelopt_mixed` 后端的。

---

## 🔥 重要更新

- **#61019（KV Connector 部分尾部 + EAGLE/MTP）**：修复了开启投机解码 + KV 卸载时，部分尾部 block 存错哈希位置的严重 bug，直接影响长上下文 + 投机解码组合的正确性。
- **#63372（DBO 路由专家行丢失）**：修复了 DBO 微批下 routed-expert 行被覆盖的问题，这是数据正确性 bug，对 MoE 模型影响巨大。
- **#63496（LoRA + MegaMoE 校验）**：把不兼容组合从"运行时报错"提前到"配置校验时报错"，是用户体验和工程规范的改进。
- **#60413（RDNA3 MXFP4 W4A8 GEMV）**：手写 HIP kernel 给消费级 AMD GPU 提速，标志着 vLLM 对非数据中心硬件的支持进入深水区。

---

## 📋 逐条解读

### 1. [Bugfix][KV Connector] Serve and store OffloadingConnector partial tails with EAGLE/MTP groups (#61019)
- **代码层面**：改 `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` 和 `OffloadingConnectorScheduler` 相关逻辑。核心是新增 `_partial_tail_key_tokens` 逻辑：当 EAGLE/MTP draft 组和 attention 组共享 KV 时，尾部 block 的哈希位置要按 `cache_hit_alignment_tokens` 对齐，而不是简单按 block 大小。
- **新手概念课堂**：想象图书馆书架（KV Cache），每个格子（block）有编号。EAGLE 组和 attention 组共用一个书架，但两组的"格子边界"定义不同。以前按 A 组的边界存，B 组去取就错位了。现在按"两组都能对齐"的边界存。
- **对你有什么影响**：如果你用 Qwen3-Next MTP 或 EAGLE 投机解码 + CPU offload，以前可能遇到"命中缓存但输出乱掉"，现在修好了。

### 2. [Bugfix][MRV2] Report the first failed DBO microbatch (#63476)
- **代码层面**：`vllm/v1/worker/gpu/ubatch_utils.py` 里 `failed = min(errors)` 改成 `failed = next(iter(errors))`。因为 `errors` 是 dict，`min` 取的是最小 ID，但真正先失败的可能是另一个 micro-batch。
- **新手概念课堂**：两个工人同时干活，A 先摔倒了，B 后来也摔了。以前报告"B 摔了"（因为 B 编号小），现在报告"第一个摔的 A"，方便排查根因。
- **对你有什么影响**：DBO 出问题时错误信息更准确，不用再猜是哪个 micro-batch 先崩。

### 3. [Doc][ROCm] Align Kimi-K3 MI355X AgentX recipe with InferenceX #4393 (#63520)
- **代码层面**：只改 `recipes/models/moonshotai/Kimi-K3.yaml` 文档。更新 MI355X 的调优配方：DCP8、resident/offload 两种 profile、量化策略调整。
- **新手概念课堂**：这是"菜谱"——告诉你在这块 GPU 上跑这个模型该用什么参数。就像游戏里的"推荐配置"。
- **对你有什么影响**：在 MI355X 上部署 Kimi-K3 时照抄配方即可，不用自己摸索。

### 4. [Doc] MiniMax-M3: InferenceX AgentX lanes as tuned presets (#63534)
- **代码层面**：改 `recipes/models/MiniMaxAI/MiniMax-M3.yaml`，新增 `presets` 字段，把 H200/B200 上不同并发点的参数打包成"预设"。
- **新手概念课堂**：像手机相机的"人像模式""夜景模式"，一键切换整套参数。
- **对你有什么影响**：在 H200/B200 上跑 MiniMax-M3 可以直接选 preset，不用手调 `--max-num-seqs`、`--max-cudagraph-capture-size`。

### 5. [Bugfix][MRV2] Keep routed-expert rows of every DBO microbatch (#63372)
- **代码层面**：`test_gpu_ubatch_slicing.py` 新增 `RoutedExpertsCapturer` 测试，验证两个 micro-batch 的路由专家行都被正确写入 `device_buffer`，不会互相覆盖。
- **新手概念课堂**：两个 micro-batch 共用一个"记录本"（capturer buffer），以前后写的会覆盖先写的，现在每行独立记录。
- **对你有什么影响**：MoE 模型开 DBO 时，路由统计/专家负载数据不再丢失。

### 6. [Bugfix][KV Offload] Reject misaligned chunked-local CPU suffix hits (#63266)
- **代码层面**：`docs/features/kv_offloading_usage.md` 加文档，`tests/v1/simple_kv_offload/test_integration.py` 新增 Llama4（chunked-local attention）测试。核心逻辑：如果 GPU 本地前缀没对齐到 attention chunk 边界，就拒绝 CPU 后缀复用，改为重算。
- **新手概念课堂**：chunked-local attention 把序列切成固定大小的"块"处理。如果缓存的前缀停在块中间，后面的块就接不上，只能重算。就像拼图，你手里有一半，但缺的那块在中间，只能重新拼。
- **对你有什么影响**：Llama4 等用 chunked-local attention 的模型，开 CPU offload 时输出更可靠（可能牺牲一点缓存命中率）。

### 7. [Refactor][RL] Share NCCL trainer rendezvous initialization (#63460)
- **代码层面**：新增 `vllm/distributed/weight_transfer/nccl_common.py` 的 `run_trainer_rendezvous`，把"worker 线程初始化 + trainer 线程初始化 + 等待对方"的握手逻辑抽出来复用。
- **新手概念课堂**：两个人在电话里约见面，必须同时在线才能接通。这个函数就是"同时拨号并等对方接"的通用流程。
- **对你有什么影响**：RL 训练（如 RLHF）的权重传输更稳定，错误能正确传播而不是死锁。

### 8. [Hardware] Use accelerator APIs in weight transfer (#63457)
- **代码层面**：`ipc_engine.py`、`nccl_engine.py`、`packed_tensor.py`、`sharded_rdt_engine.py` 里 `torch.cuda.*` 全换成 `torch.accelerator.*`。
- **新手概念课堂**：把"只能插 NVIDIA 插座"的插头改成"万能插头"，AMD 也能插。
- **对你有什么影响**：AMD GPU 上做权重传输不再报错。

### 9. [Kernel] Update DeepGEMM commit (#63253)
- **代码层面**：`cmake/external_projects/deepgemm.cmake` 里 `_DEEPGEMM_UPSTREAM_TAG` 从 `727f847...` 升到 `0820030...`，引入 fp4_fp4_mega_moe、RMSNorm 融合、MX MegaMoE 对齐修复。
- **新手概念课堂**：DeepGEMM 是专门给 MoE 模型做矩阵乘的加速库，升级就像给赛车换新引擎。
- **对你有什么影响**：MoE 模型（如 DeepSeek）推理更快，FP4 量化路径更稳。

### 10. [LoRA] Reject incompatible MegaMoE backends during config validation (#63496)
- **代码层面**：`vllm/config/lora.py` 新增 `verify_with_kernel_config`，`vllm/config/vllm.py` 在 `__post_init__` 里调用它；同时删掉 `vllm/models/step5/nvidia/model.py` 里 Step5 专属的重复检查。
- **新手概念课堂**：以前是"进厨房才发现没食材"，现在是"买菜前就检查冰箱"。早失败，省时间。
- **对你有什么影响**：同时开 LoRA 和 MegaMoE 会在启动时立刻报错，而不是跑一半崩。

### 11. [Perf][MRV2] Parallelize staged writes across 1024-element chunks (#62698)
- **代码层面**：`vllm/v1/worker/gpu/buffer_utils.py` 新增 `_WRITE_BLOCK_SIZE = 1024` 和 `_staged_write_max_len`，把一次大批量写入拆成多个 1024 元素的 chunk 并行拷贝。
- **新手概念课堂**：搬 2500 本书，以前一个人搬（串行），现在分成 3 摞同时搬（并行）。
- **对你有什么影响**：block table 更新更快，长序列推理的调度开销降低。

### 12. [Deprecation] Deprecate LM Format Enforcer structured outputs backend (#63423)
- **代码层面**：`vllm/config/structured_outputs.py` 里当 `backend == "lm-format-enforcer"` 时打 warning；文档加迁移指南。
- **新手概念课堂**：结构化输出就是强制模型输出合法 JSON/正则。LM Format Enforcer 是旧引擎，要换 XGrammar 或 Guidance。注意：正则语法不同（Python vs Rust 风格）。
- **对你有什么影响**：v0.33 还能用但会警告，v0.34 要显式 opt-in，v0.35 移除。现在就该迁移。

### 13. [Sampling] Add adaptive verification support to batch-sharded sampling (#58466)
- **代码层面**：`BatchSharder` 新增 `adaptive_verification_enabled` 参数，`num_admitted_draft_tokens` 从 GPU 读回 host 后再分片，保证分片计划与"只调度已接受 draft"的 batch 一致。
- **新手概念课堂**：投机解码时，draft 数量是"申请的"，实际"接受的"可能更少。分片采样必须按"实际接受"来分，否则 TP 各 rank 分配的请求不一致。
- **对你有什么影响**：投机解码 + TP 分片采样时结果更准确。

### 14. [GLM 5.3 Perf] Skip scale calculation for `concat_and_cache_mla` (#62102)
- **代码层面**：`csrc/cache_kernels.cu` 里 `concat_and_cache_mla_kernel` 加模板参数 `skip_scale`，当 scale=1.0 时走 `scaled_convert(..., 1.0f)` 分支，省一次乘法。
- **新手概念课堂**：乘以 1 等于没乘，编译器不一定能优化掉（因为 scale 是运行时值），所以手动加分支跳过。
- **对你有什么影响**：GLM 5.3 的 TPOT（每 token 输出时间）提升 1.2%~3.0%。

### 15. [Profiling] Add support for profile_prefix payload in HTTP /start_profile (#51974)
- **代码层面**：`/start_profile` 接口新增 `profiler_kwargs` 字段，透传给 PyTorch profiler（如 `record_shapes`、`with_stack`）。
- **新手概念课堂**：以前只能设"trace 文件名前缀"，现在能设"记录哪些细节"，像从"只录视频"升级到"录视频+录音频+录字幕"。
- **对你有什么影响**：性能分析更细，能定位到具体算子形状和调用栈。

### 16. [Kernel] Bump up FlashInfer to 0.7.2.post1 (#63466)
- **代码层面**：`requirements/cuda.txt` 里 `flashinfer-python==0.7.2` 改为 `0.7.2.post1`。
- **新手概念课堂**：FlashInfer 是高性能 attention kernel 库，打个补丁版本修 bug。
- **对你有什么影响**：attention 相关 bug 修复，无需改代码。

### 17. [Bugfix][Quantization][Spec Decode] Load ModelOpt mixed-precision DFlash and DSpark drafts (#63001)
- **代码层面**：`model_arch_config_convertor.py` 新增 `MIXED_PRECISION` 归一化；`qwen3_dflash.py` 和 `utils.py` 做模块名映射；文档更新。
- **新手概念课堂**：混合精度 checkpoint 里每层格式不同，加载器要"按名字对号入座"，名字对不上就建不出量化层。
- **对你有什么影响**：ModelOpt 导出的混合精度 draft 模型现在能正确加载。

### 18. [UMBP][3/10] Add shared lookup, transfer, hybrid, and observability (#61791)
- **代码层面**：`tests/v1/kv_connector/umbp_test_utils.py` 大幅扩展，新增 `cancel`、`take_evicted_keys`、`_hybrid_kv_cache_config`（attention + mamba 混合）等。UMBP 是一个统一的 KV 管理后端抽象，这是 10 个 PR 中的第 3 个。
- **新手概念课堂**：UMBP 像"万能遥控器"，把不同 KV 后端（CPU、磁盘、远程）统一成一套接口。
- **对你有什么影响**：未来 KV 卸载配置会更统一，混用不同后端更容易。

### 19. [DeepSeek][ROCm] V4.1-Flash: enable --monokernel on MI325X (#62713)
- **代码层面**：`recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml` 给 MI325X 加 `--monokernel` 参数，隐藏 pip 安装 tab（因为 0.31.0 wheel 不支持该 flag）。
- **新手概念课堂**：monokernel 是把多个小 kernel 合并成一个大 kernel，减少启动开销。gfx942（MI325X）上 TP4、≤42 行 decode 时启用。
- **对你有什么影响**：MI325X 上跑 DeepSeek-V4.1-Flash 更快，但必须用 nightly 镜像。

### 20. [Perf][ROCm][Kernel] Add W4A8 int8-dot MXFP4 GEMV for RDNA3/RDNA3.5 (#60413)
- **代码层面**：新增 `csrc/rocm/mxfp4_w4a8_gemv_rdna3.cu`（254 行 HIP kernel），`CMakeLists.txt` 加入编译。激活量化到 int8（每 32 元素一组），E2M1 权重码映射为 int8（2*e2m1），用 `v_dot4_i32_iu8` 指令做点积。
- **新手概念课堂**：W4A8 = 权重 4 位、激活 8 位。int8 点积指令是 AMD GPU 的"乘法加速器"，一次算 4 个。MXFP4 是 OCP 标准的 4 位浮点格式。
- **对你有什么影响**：RDNA3 消费级显卡（如 RX 7000 系列）跑小 batch decode 时更快，不用改 checkpoint 布局。

## sglang
## 💡 今日关键词

- **关键词：KV Cache 压缩与稀疏注意力（QSA / DSA / MiniMax-Sparse）**
  - **一句话通俗解释**：把长上下文里"不重要的 token"打包压缩或直接跳过，只让 GPU 算真正相关的部分，从而让 100K+ 上下文跑得动。
  - **为什么社区在关注**：今天有 3 个 commit（#45752、#46179、#46135）都在优化稀疏/压缩注意力的 kernel 和重叠调度，说明"长上下文推理成本"是当前推理引擎最烧钱的地方，大家都在抢这块性能。
  - **对新手意味着什么**：如果你想进推理框架方向，Triton/CUDA kernel 编写 + 注意力机制理解是硬通货，建议从读懂一个 attention kernel 开始。

- **关键词：投机解码的并行布局（Speculative Decoding + DP Attention）**
  - **一句话通俗解释**：用小模型"打草稿"、大模型"批改"，现在连草稿模型也能跨多张卡并行跑了。
  - **为什么社区在关注**：#46330 和 #46290 两个 commit 同时在做 DFlash 的 DP attention 支持和"从模型声明解析并行布局"的重构，说明投机解码正从"单卡玩具"走向"生产级多卡部署"。
  - **对新手意味着什么**：理解 TP/DP/PP/CP 这些并行维度的组合方式是分布式推理的入门必修课。

- **关键词：Kernel 融合（AllReduce + Residual + RMSNorm 三合一）**
  - **一句话通俗解释**：本来要跑三趟 GPU 的操作，现在塞进一个 kernel 里一次做完，省掉两次显存往返。
  - **为什么社区在关注**：#46137 把 TP all-reduce、残差加、RMSNorm 融到"push plane"上，这是典型的"通信-计算重叠"优化，是当前大模型推理提速的主流手段。
  - **对新手意味着什么**：学会看 Nsight 的 timeline，理解"内存带宽才是瓶颈"这个认知，比会写 CUDA 更重要。

- **关键词：服务健壮性 / Fail-Closed（故障即拒绝服务）**
  - **一句话通俗解释**：worker 挂了、OOM 了，健康检查立刻返回 503，而不是假装还活着继续接请求。
  - **为什么社区在关注**：#46025 给 diffusion 服务加了本地监控和 fail-closed 语义，说明推理服务正在从"能跑"走向"能上生产"，可观测性和故障语义变成刚需。
  - **对新手意味着什么**：K8s liveness/readiness probe 的区别、优雅降级，是 MLOps 方向的基本功。

- **关键词：外部缓存链接器 / 分层 KV 缓存（HiCache / Unified Cache）**
  - **一句话通俗解释**：把 KV cache 存到 GPU 之外（CPU 内存、远端存储），下次请求命中就不用重算。
  - **为什么社区在关注**：#40595 支持"树外"（out-of-tree）外部链接器，说明缓存生态开始开放插件化，第三方存储厂商可以自己接入。
  - **对新手意味着什么**：prefix caching 是降本增效最直接的手段，值得深入理解 radix tree 的匹配逻辑。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：Triton Kernel 与"按行 gather"（来自 Commit 1）

- **是什么**：Triton 是 OpenAI 出的 Python 级 GPU 编程语言，你写类似 NumPy 的代码，它帮你编译成高效 GPU kernel。Commit 1 用 Triton 写了一个 kernel，把压缩后的 KV 块按序列顺序"打包"到一个连续张量里。
- **为什么重要**：传统做法是用 Python 循环 + 多次 `torch.index_select`，每次都是一次 kernel launch，几百个序列就是几百次 launch，launch 开销比计算还大。合并成"一次 launch 处理所有序列"是性能优化的经典套路。不懂这个，你会写出"逻辑正确但慢 10 倍"的代码。
- **动手试试**：
  ```python
  # 先装 triton（一般随 torch 自带）
  import torch, triton, triton.language as tl

  @triton.jit
  def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      mask = offs < n
      x = tl.load(x_ptr + offs, mask=mask)
      y = tl.load(y_ptr + offs, mask=mask)
      tl.store(out_ptr + offs, x + y, mask=mask)

  n = 100000
  x = torch.randn(n, device="cuda")
  y = torch.randn(n, device="cuda")
  out = torch.empty_like(x)
  BLOCK = 1024
  grid = (triton.cdiv(n, BLOCK),)
  add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
  print(torch.allclose(out, x + y))  # True
  ```
- **预期结果**：打印 `True`。然后你可以试试把 `BLOCK` 改成 256 / 2048，用 `torch.cuda.Event` 计时，观察 block size 对性能的影响——这就是 Commit 1 里 `BLOCK_B=64` 调参的动机。

---

### 知识点 2：投机解码中的"草稿模型"与并行维度（来自 Commit 3、4）

- **是什么**：投机解码 = 小模型快速猜 N 个 token，大模型一次性并行验证，接受多少个算多少个。DFlash 是一种"线性块验证"的变体。而 TP（张量并行）是把一层矩阵切开放多卡，DP（数据并行）是把不同请求分给不同卡。
- **为什么重要**：DFlash 之前不支持 DP attention，意味着多卡部署时草稿模型只能单卡跑，成为瓶颈。Commit 3 打开了这个限制，但要求 `--enable-dp-lm-head`（让 target 的词表分片和草稿副本对齐）。不懂这些约束，你会遇到"启动就报错"或"结果错乱"。
- **动手试试**：
  ```bash
  # 单卡先跑通最基础的投机解码
  python3 -m sglang.launch_server \
      --model-path Qwen/Qwen3-4B \
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path lmsys/sglang-EAGLE-Qwen3-4B \
      --port 30000

  # 另一个终端，测吞吐
  python3 -m sglang.bench_serving --backend sglang \
      --dataset-name random --num-prompts 50 \
      --random-input-len 512 --random-output-len 128
  ```
- **预期结果**：bench 输出里能看到 `Accept length`（平均接受长度）> 1，说明投机生效。对比不加 `--speculative-*` 参数时的 output throughput，通常能提升 1.5–2.5 倍。

---

### 知识点 3：通信-计算融合（AllReduce + RMSNorm）（来自 Commit 5）

- **是什么**：Transformer 每层结尾都要做 all-reduce（把各卡的 partial 结果加起来）、加残差、做 RMSNorm。这三步各自都要读写一遍显存。融合 kernel 让数据在寄存器/SMEM 里走完三步，只读写一次。
- **为什么重要**：大模型 decode 阶段是 memory-bound（显存带宽瓶颈），不是 compute-bound。减少显存往返 = 直接提速。Commit 5 的 `MoeFinalizeAllReduceParams` 新增了 `add_residual` 和 `residual_out` 字段，就是让 finalize kernel 顺手把残差也加了。
- **动手试试**：
  ```python
  import torch, time

  N = 4096
  x = torch.randn(N, N, device="cuda")
  w = torch.randn(N, device="cuda")

  # 分三步做
  def unfused(x, w):
      r = x + x            # residual add
      mean = r.pow(2).mean(-1, keepdim=True)
      return r * torch.rsqrt(mean + 1e-6) * w

  # 用 torch.compile 融合
  fused = torch.compile(unfused)

  for fn in (unfused, fused):
      for _ in range(3): fn(x, w)  # warmup
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      for _ in range(100): fn(x, w)
      torch.cuda.synchronize()
      print(fn.__name__ if hasattr(fn,'__name__') else 'compiled',
            (time.perf_counter()-t0)*10, "ms/iter")
  ```
- **预期结果**：`torch.compile` 版本通常快 20–40%。用 `nsys profile python your_script.py` 打开 timeline，能看到 unfused 版本有 3 个 kernel，compiled 版本被融合成 1–2 个——这就是 Commit 5 手工做的事。

---

### 知识点 4：健康检查与 Fail-Closed 语义（来自 Commit 13）

- **是什么**：`/liveness` 回答"进程还活着吗"，`/health` 回答"我能接活吗"。Fail-closed 意思是：一旦内部组件（worker、broker、路由循环）出问题，`/health` 立刻返回 503，让负载均衡器把流量摘走。
- **为什么重要**：如果不 fail-closed，一个半死不活的实例会持续接收请求然后超时，造成雪崩。Commit 13 还特别说明：多卡 OOM 必须杀掉 worker，因为其他 rank 可能卡在 collective 里，继续跑会死锁。
- **动手试试**：
  ```python
  # 用 FastAPI 模拟一个带健康检查的服务
  from fastapi import FastAPI, Response
  import threading, time

  app = FastAPI()
  state = {"worker_alive": True, "warmup_done": False}

  @app.get("/liveness")
  def liveness(): return {"ok": True}

  @app.get("/health")
  def health(response: Response):
      if not state["warmup_done"] or not state["worker_alive"]:
          response.status_code = 503
          return {"ready": False}
      return {"ready": True}

  @app.post("/kill_worker")
  def kill(): state["worker_alive"] = False; return {"killed": True}

  @app.post("/warmup_done")
  def warm(): state["warmup_done"] = True; return {"done": True}

  # uvicorn 启动后测试：
  # curl -i localhost:8000/health          -> 503
  # curl -X POST localhost:8000/warmup_done
  # curl -i localhost:8000/health          -> 200
  # curl -X POST localhost:8000/kill_worker
  # curl -i localhost:8000/health          -> 503, 但 /liveness 仍 200
  ```
- **预期结果**：`/liveness` 始终 200，`/health` 随状态在 503/200 之间切换。这就是 K8s 里 liveness probe 和 readiness probe 应该分别挂哪个端点的原因。

---

### 知识点 5：prefix caching 与 Radix Tree（来自 Commit 9、10）

- **是什么**：Radix tree（基数树）把请求的 token 序列按公共前缀组织成一棵树。新请求来了，从根往下匹配，匹配到的部分直接复用 KV cache，不用重算。
- **为什么重要**：多轮对话、few-shot prompt、系统提示词都是大量重复前缀，prefix caching 能省 50%+ 的 prefill 计算。Commit 10 修了一个 EAGLE 场景下"边界 token 没算进去"的 bug——这种 off-by-one 在树匹配里极难发现。
- **动手试试**：
  ```python
  # 用 sglang 自带服务测 prefix caching 收益
  # 启动服务（默认开启 radix cache）
  # python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000

  import time, requests
  long_prefix = "You are a helpful assistant. " * 200  # 大量重复前缀

  def ask(q):
      t0 = time.time()
      r = requests.post("http://localhost:30000/generate",
          json={"text": long_prefix + q, "sampling_params": {"max_new_tokens": 8}})
      return time.time() - t0

  print("first:", ask("Q1"))   # 冷启动，慢
  print("second:", ask("Q2"))  # 命中前缀，快
  ```
- **预期结果**：第二次请求明显更快（可能快 2–5 倍，取决于前缀长度）。如果看服务日志，会看到 `#cached-token` 之类的统计，说明命中了 radix tree。

---

## 🔥 重要更新

1. **#46330 DFlash 支持 CUDA DP attention** —— 投机解码终于能在多卡 DP 部署下跑，草稿模型不再拖后腿，对高吞吐线上服务影响大。
2. **#46137 AllReduce+Residual+RMSNorm 三合一融合** —— 每层都执行的路径，融合后省两次显存往返，TP 大模型 decode 阶段直接受益。
3. **#46025 Diffusion 服务 fail-closed** —— worker 挂了、OOM 了立刻 503，生产环境稳定性大幅提升，避免"半死不活"的实例拖垮整个集群。
4. **#46135 MiniMax-M3 稀疏注意力 Hopper 加速** —— 支持 bf16→fp8 的 cast store，长上下文稀疏注意力在 Hopper 上提速明显。

---

## 📋 逐条解读

### 1. [Perf] QSA: Optimize the prefill indexer gather (#45752)
- **代码层面**：在 `qsa/kernel.py` 新增 `_gather_qsa_compressed_keys_kernel` Triton kernel 和 `gather_qsa_compressed_keys` 包装函数。原来可能每个序列单独 gather，现在一个 kernel launch 处理所有序列：grid 的 y 维度是序列号，x 维度是块号，用 `cu_blocks`（累积块数）定位每个序列在输出里的行区间，用 `token_slot_table` 查每个块的第一个 token 槽位再除以压缩比得到压缩块索引。
- **新手概念课堂**：想象你有 100 个学生的作业要收，每个学生交 3–10 本。笨办法是老师一个个走过去收（100 次 kernel launch），聪明办法是让所有人同时把作业扔进一个大箱子，箱子按学号分好格子（一次 launch）。Triton 的 `program_id(0/1)` 就是"格子坐标"。
- **对你有什么影响**：QSA 压缩注意力的 prefill 阶段更快，长上下文首 token 延迟（TTFT）降低。用户感知就是"提问后等的时间变短了"。

### 2. [Perf] dsa: Overlap CP KV gather with indexer computation (#46179)
- **代码层面**：`dsa_indexer.py` 和 `dsa_indexer_kpool.py` 的 `forward_cuda` 新增 `use_cuda_graph` 参数，把 CUDA graph 检测条件从"是否在 graph 里"改成"是否启用 graph 且在 graph 里"。新增 `topk_output_width` 属性（普通 indexer 是 `index_topk`，kpool 版本是 `index_topk + index_kpool - 1`）。`dsa_backend.py` 里 FP8 路径去掉了 `q_rope is not None` 的断言。
- **新手概念课堂**：CP（Context Parallel）是把一条超长序列切成几段分给不同卡，每张卡算自己那段，但需要把 KV 互相 gather 过来。Overlap 就是"一边 gather 一边算 indexer"，像洗衣机洗衣服的同时烧水——两件事不互相等。
- **对你有什么影响**：启用 DSA（DeepSeek Sparse Attention）的长上下文场景，prefill 更快。`use_cuda_graph` 参数化让同一份代码能在 graph 和 eager 两种模式下复用，减少维护成本。

### 3. [Feat] spec: Support CUDA attention DP for DFlash and DFlash2 (#46330)
- **代码层面**：`speculative_hook.py` 的 `_handle_dflash` 里，原来"非 NPU 一律拒绝 DP attention"改成"CUDA 上允许，但必须满足：`--enable-dp-lm-head`、`attn_cp_size == 1`、`SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=0`"。文档 `speculative_decoding.mdx` 新增了 DP attention 章节和示例命令。
- **新手概念课堂**：草稿模型（draft）要借用大模型（target）的 lm_head 来算概率，DP 之后每张卡只持有词表的一部分，所以必须开 `--enable-dp-lm-head` 让草稿副本和 target 的分片对齐——就像两个同事合作写报告，得先约定好谁负责哪几章。
- **对你有什么影响**：以前 DFlash 只能单卡跑，现在 4 卡 TP4 可以配 DP2/TP2，草稿模型也能并行，整体吞吐提升。但注意 Domino 和 LiLiCorr 两种 head 还不支持。

### 4. [Refactor] spec: Resolve draft parallel placement from model declarations (#46290)
- **代码层面**：新增 `configs/draft_parallel.py`，定义 `DraftTPExecution` 枚举（`SINGLE_GROUP` / `TARGET_LAYOUT`）、`DraftParallelSpec`、`DraftParallelLayout` 三个结构。`tp_worker.py` 的 `alloc_memory_pool` 新增 `draft_parallel_layout` 关键字参数，`scheduler.py` 把 draft worker 的布局传进去。核心思想：并行布局不再靠猜，而是由模型自己声明。
- **新手概念课堂**：以前是"调度器猜草稿模型该用哪个 TP group"，现在是"草稿模型说：我需要这个 group"。就像装修时不再让工头猜你要什么风格，而是设计师直接给图纸。
- **对你有什么影响**：普通用户无感，但这是为 Commit 3 铺路的架构重构。对开发者来说，新增草稿模型时不用改调度器，只需声明布局，扩展性更好。

### 5. [Feat] layer-boundary: Fuse TP all-reduce, residual add and RMSNorm (#46137)
- **代码层面**：`all_reduce_fusion.cuh` 的 `MoeFinalizeAllReduceParams` 新增 `add_residual`、`residual_out`、`norm_weight_offset` 字段（后者为 1.0 表示 Gemma RMSNorm）。`RowClusterTrait` 新增 `kVecs` 模板参数，允许每个线程处理多个 16B 向量（减少 block 大小）。新增 `plain_vec` 函数处理"无 finalize"场景。
- **新手概念课堂**：All-reduce 是"把各卡的 partial 结果加起来"，残差是"加回原始输入"，RMSNorm 是"归一化"。三步都读写显存，融合后数据留在寄存器里走完三步——就像做菜时切菜、腌肉、下锅在同一个砧板上完成，不用来回搬。
- **对你有什么影响**：TP 部署的大模型（尤其是 MoE）每层 decode 都更快。`norm_weight_offset` 支持 Gemma 系模型（权重 +1 的变体），扩大了适用范围。

### 6. [Perf] minimax-sparse: Speed up MiniMax-M3 sparse attention on Hopper (#46135)
- **代码层面**：`fused_store_kv_index.cuh` 的 `StoreTrait` 新增 `kCastFp8` 模板参数。当开启时，输入是 bf16（2 倍字节），用 `__nv_cvt_bfloat16raw2_to_fp8x2` 在 store 时直接转成 FP8 写入 cache，省掉一次单独的量化 kernel。`kWorkerSize` 按输入字节数算（`kInHeadBytes`）。
- **新手概念课堂**：FP8 是一种 8 位浮点格式，显存占用是 bf16 的一半。传统做法是"先转 FP8 到临时张量，再拷贝到 cache"，现在"边拷贝边转换"，像边翻译边打字，不用先写草稿。
- **对你有什么影响**：MiniMax-M3 在 Hopper（H100/H200）上跑稀疏注意力更快，KV cache 显存占用更低，能塞更长的上下文。

### 7. [Fix] sgl-router: Bump tokio for RUSTSEC-2026-0386 (#46260)
- **代码层面**：`Cargo.lock` 里把多个依赖的 `windows-sys` 从 0.61.2 降到 0.52/0.59/0.60，同时升级 tokio 相关版本。这是安全公告 RUSTSEC-2026-0386 的修复。
- **新手概念课堂**：`Cargo.lock` 是 Rust 的"依赖版本锁文件"，保证每次构建用完全相同的版本。安全公告（RUSTSEC）是 Rust 官方的漏洞库，类似 Python 的 CVE。
- **对你有什么影响**：sgl-router 用户升级后不再有该安全漏洞。普通推理用户无感，但生产环境应该跟进。

### 8. [Fix] chat-template: Pass return_dict=False to apply_chat_template(tokenize=True) for transformers 5 (#44020)
- **代码层面**：在 `speed_bench.py`、`ollama/serving.py`、`chat_encoding.py`、`voxtral.py` 等所有调用 `apply_chat_template(..., tokenize=True)` 的地方加上 `return_dict=False`。因为 transformers 5 里 `tokenize=True` 默认可能返回 dict（含 `input_ids`、`attention_mask`），而 sglang 期望直接拿 list。
- **新手概念课堂**：这是典型的"上游库改了默认行为"引发的兼容性问题。就像快递公司改了包装规格，所有下游都得跟着改。`return_dict=False` 明确说"我只要 list，别给我 dict"。
- **对你有什么影响**：用 transformers 5 + 自定义 chat template 的用户不再报 `TypeError: unhashable type: 'dict'` 之类的错。测试里还加了断言，验证 prompt_len 和模板化后的 token 数一致。

### 9. [Feat] unified-cache: Support out-of-tree external linkers (#40595)
- **代码层面**：`memory.py` 的 `unified_cache_external_linker_backend` 新增 `dynamic` 选项，新增 `unified_cache_external_linker_config` 字段（JSON）。`dynamic` 模式下通过 `linker`（类名）+ `linker_module_path`（Python 模块）+ `linker_extra_config` 动态导入第三方 linker。文档从"undocumented"移到正式文档。
- **新手概念课堂**：这是"插件化"设计。核心代码不再硬编码支持哪些存储后端，而是开放一个接口，第三方（比如某家存储厂商）自己写个类注册进来。类似 VS Code 的扩展机制。
- **对你有什么影响**：如果你公司自研了 KV 存储系统，现在可以写个 linker 类直接接进 sglang，不用改 sglang 源码。普通用户如果只用 mooncake/mori，行为不变。

### 10. [Fix] hicache: Count the EAGLE bigram boundary token in PD decode restores (#45076)
- **代码层面**：`decode_hicache_mixin.py` 新增 `_bigram_boundary_tokens` 函数，返回 `tree_cache.tree_core.is_eagle`（0 或 1）。在 `_try_hicache_queue_load_back` 里，`match_kv_cache` 的 `max_prefix_len` 从 `pm.decode_prefix_len` 改成 `pm.decode_prefix_len + _bigram_boundary_tokens(...)`。
- **新手概念课堂**：EAGLE 用 bigram（二元组）做树结构：N 个 bigram 需要 N+1 个 token id。所以"前缀长度"和"token 数量"差 1。这个 off-by-one 在 PD（Prefill-Decode 分离）场景下会导致最后一个 token 没被 restore，输出错乱。
- **对你有什么影响**：用 EAGLE + PD 分离 + HiCache 的用户，decode 阶段恢复缓存后不再丢 token，输出正确性修复。测试里给 mock 加了 `tree_core=SimpleNamespace(is_eagle=False)` 适配。

### 11. [Fix] kimi-k3: Avoid vision attention child-name collisions (#45110)
- **代码层面**：`kimi_k3_vl.py` 里 `self.attention_backend_impls` 从 `nn.ModuleDict` 改成普通 `dict[str, nn.Module]`。因为 `nn.ModuleDict` 会把 key 注册成子模块名，如果 backend 名字恰好是 `"cuda"`、`"forward"` 这类和 nn.Module API 冲突的名字，就会出问题。
- **新手概念课堂**：`nn.ModuleDict` 是 PyTorch 的"带名字的模块容器"，但它会把这些名字当成属性。如果名字叫 `forward`，就会覆盖 `nn.Module.forward` 方法——就像你给变量起名叫 `list`，把内置函数覆盖了。
- **对你有什么影响**：Kimi-K3 视觉模型在某些 attention backend 名字下启动不再崩溃。测试新增了 `test_kimi_k3_vision_backend_label_can_overlap_module_name` 专门验证这个场景。

### 12. [Fix] diffusion/zimage: Stop serving a previous request's RoPE tables (#46044)
- **代码层面**：`zimage.py` 里 `_get_rope_cache` 的缓存 key 从 `(data_ptr, shape, dtype, device)` 改成直接用源张量的 Python 对象身份（`cached[0][0] is cos`）。新增 `_get_unified_freqs_cis` 把 image 和 caption 的 RoPE 表 concat 后按"源张量对"缓存。注释明确说："data_ptr key hit a freed table's address"——旧张量释放后新张量可能复用同一地址，导致缓存误命中。
- **新手概念课堂**：`data_ptr()` 是张量在显存里的地址。张量被 GC 后地址会被复用，如果只按地址做 key，新张量可能"继承"旧张量的缓存——就像你用旧门牌号找人，结果住进去的是另一个人。用对象身份（`is`）比较就安全了。
- **对你有什么影响**：Z-Image 模型连续处理不同尺寸/不同 prompt 的请求时，不再出现"用了上一个请求的 RoPE 表导致图像错乱"的 bug。

### 13. [Fix] diffusion: Fail closed after local worker loss or multi-rank OOM (#46025)
- **代码层面**：`http_server.py` 引入 `psutil`，新增本地监控线程每秒检查 worker 存活、broker 存活、disaggregation 路由循环存活。`orchestrator.py` 新增 `is_alive` 属性。多卡 forward 时 allocator OOM 直接终止 worker（因为其他 rank 可能卡在 collective 里）。`/health` 文档更新为"warmup 完成 + worker 存活 + broker/路由循环运行"才返回 200。
- **新手概念课堂**：Fail-closed = "不确定安全就拒绝"。对比 fail-open = "不确定就放行"。分布式系统里，一个卡住的 rank 会让所有 rank 卡在 all-reduce 里，所以必须 fail-closed 杀掉整个 worker，让上层重新调度。
- **对你有什么影响**：Diffusion 服务生产部署更稳。K8s 里 `/health` 返回 503 时流量会被摘走，不会打到坏实例上。但注意：standalone disaggregation router 不拥有远端 worker，检测不到它们。

### 14. [Perf] diffusion: Remove per-step host syncs and uploads in video models (#46047)
- **代码层面**：`mova_dual_tower.py` 新增 `rope_cos_sin_cache` 函数，把 `(cos, sin)` 元组转成 FlashInfer 需要的 `[seqlen, head_dim]` 格式，且只做一次。`forward` 签名从接收 `(cos, sin)` 元组改成接收已构造好的 `cos_sin_cache` 张量。原来每次 forward 都要 `.to(dtype).to(device).squeeze(0).contiguous()`，这些操作会触发 host-device 同步。
- **新手概念课堂**：GPU 是异步执行的，但 `.to(device)` 如果涉及 CPU→GPU 拷贝，会强制同步等待。视频模型每步都做这种操作，几十步下来浪费大量时间。预计算一次、之后复用，是消除同步的标准手法。
- **对你有什么影响**：视频生成模型（MOVA 等）每步去噪更快，整体生成时间缩短。用户感知是"出片更快了"。

### 15. [Fix] diffusion/openai: Bound job lists and complete cursor pagination (#46156)
- **代码层面**：`mesh_api.py` 和 video API 的 `list` 端点，`limit` 从 `Optional[int]` 改成 `int = Query(20, ge=1, le=100)`，`order` 从 `Optional[str]` 改成 `Literal["asc", "desc"]`。响应新增 `first_id`、`last_id`、`has_more` 字段。文档示例从"轮询 list 找 id"改成"直接 `retrieve(video_id)`"。
- **新手概念课堂**：Cursor pagination（游标分页）比 offset 分页更稳：offset 分页在数据变动时会漏/重，游标分页用"上一页最后一个 id"作为锚点，不会错位。`has_more` 告诉客户端"还有没有下一页"，避免多一次空请求。
- **对你有什么影响**：调 `/v1/videos` 或 `/v1/meshes` 时默认只返回 20 条，避免一次拉几千条撑爆内存。用 `after=last_id` 翻页。旧代码如果不传 `limit` 会拿到 20 条而非全部，需要适配。

### 16. [Refactor] diffusion: Share segmented row-parallel weight loading (#46264)
- **代码层面**：`linear.py` 的 `_load_tensor_parallel_weight` 新增 `input_sizes` 和 `tp_size` 参数。当权重是"多个独立 shard 的输入拼接"时（比如 attention 输出 + MLP 输出 concat 后接一个 RowParallelLinear），先按 `input_sizes` 把 checkpoint 权重 split 成段，每段再按 TP rank 切，最后 cat 回去。`RowParallelLinear.__init__` 新增 `input_sizes` 参数并校验 `input_is_parallel=True` 且 `sum(input_sizes) == input_size`。
- **新手概念课堂**：普通 RowParallelLinear 的输入是一整块，按 TP 切一刀就行。但"拼接输入"的每一段宽度不同，不能一刀切——得按段分别切，再拼起来。就像切一根香肠和切一根由不同馅料拼成的肉卷，切法不一样。
- **对你有什么影响**：Diffusion 模型开发者新增模型时不用自己写 weight loader，传 `input_sizes=[attn_dim, mlp_dim]` 即可。文档 `support_new_models.mdx` 新增了说明。

### 17. [Feat] rust-server: Report weight versions and media token counts in generate meta_info (#46205)
- **代码层面**：`generate.proto` 的 `GenerateMetaInfo` 新增 `image_tokens`、`audio_tokens`、`video_tokens` 三个 optional uint32 字段。`api_types.py` 的 Python 侧同步新增字段及 encode/decode 逻辑。`output_streamer.py` 新增 `_scheduler_decode_latency` 等辅助函数。
- **新手概念课堂**：`meta_info` 是响应里附带的元数据（不是生成的文本本身），包含 token 数、耗时、权重版本等。新增媒体 token 数让客户端能知道"这张图占了多少 token"，便于计费和调试。
- **对你有什么影响**：多模态请求的响应里能直接看到图片/音频/视频各消耗多少 token，方便成本核算。Rust server 用户升级后可用。

### 18. [Fix] [AMD] custom-allreduce: Stop registering oversized inputs in the deterministic all-reduce (#34543)
- **代码层面**：`custom_all_reduce.py` 的 `_all_reduce_impl` 里，AMD 确定性路径把 `if inp_size < self.max_size` 改成 `<=`，超出的走新增的 `_deterministic_all_reduce_oversized`：把输入按 `max_size` 切成多块，每块用 `deterministic_all_reduce_unreg` 处理。注释解释：每次 `register_buffer` 永久消耗一个 `rank_data` slot，大 `chunked_prefill_size` 会每步注册一次，最终 "Rank data buffer is overflowed"。
- **新手概念课堂**：`register_buffer` 是"把这块显存登记到通信库的固定槽位"，槽位有限。超大输入不能每次都登记，否则槽位耗尽。分块处理是"用固定大小的篮子分多次搬"，结果不变因为 all-reduce 是逐元素的。
- **对你有什么影响**：AMD GPU 上用确定性 all-reduce + 大 chunked prefill 的用户，不再因为槽位溢出崩溃。测试新增了 `OVERSIZED_MAX_SIZE = 256KB` 的小 buffer 场景专门验证。

### 19. [Fix] diffusion: Complete Blackwell routing test fixture (#46294)
- **代码层面**：`test_dynamic_cudnn_sdpa_routing.py` 里 `_uses_cudnn` 的 mock `SimpleNamespace` 新增 `_is_sm103=False` 字段。因为 `_use_cudnn_sdpa` 的实现新增了对 `_is_sm103`（Blackwell 的另一个 SM 版本）的判断，测试 fixture 不完整会 `AttributeError`。
- **新手概念课堂**：这是"测试 fixture 跟实现脱节"的典型。实现加了新字段，mock 没跟上。`SimpleNamespace` 是 Python 的"万能对象"，可以随便挂属性，但不会自动补全——所以实现加字段时测试得手动加。
- **对你有什么影响**：纯测试修复，不影响运行时。但提醒我们：mock 对象要尽量贴近真实接口，否则实现演进时测试会假失败。

### 20. [Chore] docker: Select released wheels and the torch index in the cu134 image (#46136)
- **代码层面**：`Dockerfile.cu134` 新增 `TORCH_INDEX_URL`、`TORCH_VERSION`、`TORCHVISION_VERSION`、`TORCHAUDIO_VERSION` 参数（默认指向 nightly），新增 `SGL_WHEELS_SOURCE`（`source` 或 `pypi`）。`pypi` 模式下用 `pip download` 拉发布版 wheel，并用 Python 脚本校验 wheel 的 `Requires-Dist: torch==X` 和 `TORCH_VERSION` 匹配，不匹配直接 `sys.exit` 报错。
- **新手概念课堂**：Docker 的 `ARG` 是构建时参数，`--build-arg` 传入。BuildKit 会跳过不需要的 stage（`cu134_wheels_source` vs `cu134_wheels_pypi`），所以选 pyp

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)