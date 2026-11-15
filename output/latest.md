# 🚀 vLLM & SGLang 每日更新 — 2026-11-15

> 自动生成于 2026-11-15 08:25 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：惰性导入（Lazy Import）**
  - **一句话解释**：不要一上来就把所有东西都加载完，用到哪个再加载哪个。
  - **为什么社区在关注**：vLLM 的模型模块越来越大，导入时附带初始化 CUDA 会让 fork 出来的子进程用不了 GPU。今天 Muse Spark 的 bugfix 就是典型例子——"导入即初始化 CUDA"是分布式推理里非常隐蔽的坑。
  - **对新手意味着什么**：写 Python 包时，`__init__.py` 里不要写重活。学会用 `__getattr__` + `importlib.import_module` 做懒加载，是进阶 Python 工程师的必修课。

- **关键词：异构硬件适配（ROCm / XPU / AITER）**
  - **一句话解释**：同一份代码要跑在 NVIDIA、AMD、Intel 等不同厂商的 GPU 上。
  - **为什么社区在关注**：今天 20 个 commit 里有 5 个跟 ROCm/XPU 有关（SDMA 传输、FlyDSL MXFP8、XPU MLA、XPU MXFP8 flatten），说明 vLLM 正在认真做"非 NVIDIA 一等公民"。
  - **对新手意味着什么**：只会 CUDA 已经不够了。了解 HIP/oneAPI 的基本差异、以及"kernel 抽象层"（如 ops 注册表）的设计，是未来两三年的加分项。

- **关键词：量化（FP8 / MXFP8 / FP4）**
  - **一句话解释**：用更少的 bit 表示权重和激活，省显存、提吞吐，但要处理精度。
  - **为什么社区在关注**：今天有 FP8 dummy 权重填充、ROCm MXFP8 GEMM、TRT-LLM FP8 block-scale MoE 三个 commit，量化已经是推理引擎的主战场。
  - **对新手意味着什么**：想进推理方向，一定要搞懂"per-tensor / per-channel / per-block / MX"这些量化粒度，以及 scale 是怎么存的。

- **关键词：流水线并行 & 通信重叠（PP / DBO）**
  - **一句话解释**：把模型切到多张卡上，让"通信"和"计算"同时进行，别让 GPU 闲着。
  - **为什么社区在关注**：SDMA handoff（ROCm 上用拷贝引擎做 PP 传输）和 DBO 支持纯 TP 两个 commit，都是在榨干多卡带宽。
  - **对新手意味着什么**：分布式推理不是"跑起来就行"，性能瓶颈常在通信。理解 NCCL、copy engine、all-reduce overlap 是高级岗位的敲门砖。

- **关键词：多模态设备端归一化（mm_device_do_normalize）**
  - **一句话解释**：图片预处理（减均值除方差）从 CPU 挪到 GPU 上做，省 CPU 时间。
  - **为什么社区在关注**：今天 LLaVA-OneVision 和 Qwen3-Omni 两个 commit 都在加这个能力，说明多模态 serving 的 CPU 预处理已经成为瓶颈。
  - **对新手意味着什么**：多模态不只是"能读图"，预处理流水线的工程优化空间巨大。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：Python 惰性导入（Lazy Import）**
  - **是什么**：模块被 `import` 时不真正执行重量级代码，等到第一次访问属性时才加载。今天 Commit 1 用 `__getattr__` 实现了这一点。
  - **为什么重要**：某些库（如 FlashInfer）在导入时会查询 GPU 并初始化 CUDA 上下文。如果父进程在 fork 之前初始化了 CUDA，子进程就无法再用 CUDA（经典 fork + CUDA 崩溃）。分布式推理框架里这非常致命。
  - **动手试试**：
    ```python
    # lazy_mod.py
    import importlib

    _MODULES = {"Heavy": "heavy_impl"}

    def __getattr__(name):
        if name not in _MODULES:
            raise AttributeError(name)
        mod = importlib.import_module(f".{_MODULES[name]}", __name__)
        return getattr(mod, name)
    ```
    然后 `import lazy_mod`，观察 `heavy_impl` 是否被加载（用 `sys.modules` 检查）。
  - **预期结果**：`import lazy_mod` 后 `'lazy_mod.heavy_impl' not in sys.modules`；访问 `lazy_mod.Heavy` 后才出现。说明你掌握了"导入即副作用"的规避方法。

- **知识点 2：FP8 权重的取值范围与 dummy 权重**
  - **是什么**：FP8 E4M3 的最大值是 448，最小值正常数是 2⁻⁶。今天 Commit 9 把 dummy 权重从"±1"改成"随机填满整个 FP8 编码空间"。
  - **为什么重要**：做性能基准测试时，如果权重全是很小的数，GPU 功耗上不去，你会测出"假快"的结果。真实 checkpoint 的权重分布是宽的，dummy 权重必须模拟这一点，否则在功耗封顶的 GPU 上基准失真。
  - **动手试试**：
    ```python
    import torch
    from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
    p = torch.empty(1 << 16, dtype=torch.float8_e4m3fn)
    initialize_single_dummy_weight(p, name="weight")
    print(p.view(torch.uint8).unique().numel())   # 期望 > 240
    print(p.float().abs().max())                   # 期望接近 448
    ```
  - **预期结果**：字节编码种类超过 240 种，最大绝对值接近 FP8 上限。说明你理解了"dummy 权重也要模拟真实分布"。

- **知识点 3：N-D 张量展平后再做 GEMM**
  - **是什么**：线性层输入可能是 `[B, T, K]` 这种三维形状，但 GEMM 只接受二维。Commit 5 在 XPU MXFP8 里先 `reshape(-1, K)`，算完再 reshape 回 `[B, T, N]`。
  - **为什么重要**：很多 kernel 假设输入是 2D。如果你忘了展平，要么报错，要么拿到错误的形状。反过来，如果展平后忘了恢复，下游 shape 全乱。
  - **动手试试**：
    ```python
    import torch
    x = torch.randn(2, 5, 64)          # [B=2, T=5, K=64]
    W = torch.randn(128, 64)           # [N=128, K=64]
    out_shape = x.shape[:-1] + (W.shape[0],)   # [2,5,128]
    y = (x.reshape(-1, 64) @ W.t()).reshape(out_shape)
    print(y.shape)                     # torch.Size([2, 5, 128])
    ```
  - **预期结果**：输出形状是 `[2, 5, 128]`。说明你掌握了"展平-计算-恢复"的通用套路。

- **知识点 4：内存对齐与 slot mapping**
  - **是什么**：某些 kernel（如 DeepSeek-V4 的 CuTe DSL）要求输入指针 64 字节对齐。Commit 13 通过 `round_up(num_tokens, 16)` 给每行 padding，使 int64 的每行起始地址 128 字节对齐。
  - **为什么重要**：不对齐会直接触发 kernel 报错或悄悄性能下降。这在手写 kernel 或跨后端移植时是必踩的坑。
  - **动手试试**：
    ```python
    import torch
    from vllm.utils.math_utils import round_up
    num_groups, num_tokens = 3, 509
    buf = torch.zeros(num_groups, round_up(num_tokens, 16), dtype=torch.int64)
    rows = buf[:, :num_tokens]
    for r in rows:
        print(r.data_ptr() % 64)   # 期望全为 0
    ```
  - **预期结果**：每行 `data_ptr() % 64 == 0`。说明你理解了"行 stride 对齐"的技巧。

- **知识点 5：`n` 参数的语义（一次请求生成多个候选）**
  - **是什么**：OpenAI API 里 `n>1` 表示一次请求返回 n 个独立采样结果。Commit 4 让 Rust 前端 `/inference/v1/generate` 支持了这个参数。
  - **为什么重要**：很多应用（如 best-of-N、投票、RL 采样）依赖 `n>1`。实现上要正确处理"多个流合并成一个响应"和"usage 累加"。
  - **动手试试**：
    ```bash
    # 先起一个 vllm serve（任意小模型）
    curl http://localhost:8000/inference/v1/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Hello", "n": 3, "max_tokens": 8}'
    ```
  - **预期结果**：返回 3 个不同的 `choices`，`usage.completion_tokens` 是 3 份之和。说明你理解了 `n` 参数在 token-in-token-out 接口里的实现方式。

---

## 🔥 重要更新

- **Commit 10（DBO 支持纯 TP）**：DBO 之前只能在 DP+EP 下用，现在纯 TP 也能让 all-reduce 和计算重叠。这是分布式推理性能的一大步，影响所有多卡 TP 用户。
- **Commit 8（ROCm DeepSeek-V4.1 用 AITER FlyDSL MXFP8）**：AMD MI355X 上 DeepSeek 的 MXFP8 GEMM 换成 AITER 调优 kernel，是 AMD 生态追平 NVIDIA 的关键性能补丁。
- **Commit 9（FP8 dummy 权重填满范围）**：看似小改动，但直接修正了"用 dummy 权重跑 benchmark 结果偏乐观"的系统性问题，所有做性能评测的人都会受益。
- **Commit 20（移除 OpenPangu-VL）**：模型支持列表在做减法，说明 vLLM 在收缩维护面、聚焦主流模型。

---

## 📋 逐条解读

### 1. [Bugfix][Muse Spark] Load the model classes lazily so importing the package doesn't initialize CUDA (#67691)
- **代码层面**：改的是 `vllm/models/muse_spark/__init__.py`。原来在模块顶层直接 `from .nvidia.model import ...`，现在改成定义 `_MODULES` 字典 + `__getattr__`，等真正访问 `MuseSparkForCausalLM` 时才根据 `current_platform.is_rocm()` 决定从 `amd` 还是 `nvidia` 子模块导入。
- **新手概念课堂**：想象你搬家，`__init__.py` 就是"搬家清单"。以前的做法是搬家第一天把所有箱子都拆开摆好（导入即执行），现在改成"清单上写着东西在哪，用到再去拿"（惰性加载）。好处是：有些箱子（NVIDIA 的 FlashInfer）一拆开就会"占住客厅"（初始化 CUDA），后面想再分家（fork 子进程）就分不了了。
- **对你有什么影响**：如果你在 Muse Spark 模型上做多进程推理，以前可能遇到"fork 后 CUDA 不可用"的诡异错误，现在不会了。普通用户感知不到，但分布式场景下更稳。

### 2. [ROCm][PP] Stage hand-off on SDMA copy engines (--pp-sdma-handoff) (#65658)
- **代码层面**：新增 `--pp-sdma-handoff` 开关；`create_pp_transport` 在 ROCm 上 `auto` 模式默认走 NCCL；测试文件 `test_pp_transport.py` 扩展支持 gfx950（MI355X）。核心思路是把 PP 各 stage 之间的张量传递从 NCCL 换成 SDMA 拷贝引擎。
- **新手概念课堂**：流水线并行像工厂流水线，工位之间要传半成品。NCCL 是"专门的快递员"（用 SM 计算单元干活），SDMA 是"工厂自带的传送带"（独立的拷贝硬件）。用传送带传东西，工人（SM）就能专心干活，不被打断。
- **对你有什么影响**：ROCm 用户开 PP 时，如果硬件支持（gfx950 + xGMI），可以开这个开关减少 SM 占用、提升吞吐；不支持的硬件会自动回退到 NCCL，不会挂。

### 3. [Bugfix][KV Offload] Report only written blocks from fs tier store jobs (#63524)
- **代码层面**：改 `csrc/fs_io.cpp`。`_store_block` 新增返回值 `kAlreadyPresent = -1` 表示"文件已存在，没写"；`batch_store_block` 统计真正写入的块数并作为返回值（以前返回 `None`）。
- **新手概念课堂**：KV cache offload 到文件系统时，块是按内容哈希命名的。如果两个请求算出同一个块，第二个就不用重写。以前不管写没写都算"传输了"，导致带宽统计虚高。现在区分"真写了"和"已存在"。
- **对你有什么影响**：KV offload 的监控指标更准了。如果你在调带宽或算成本，数字不再被重复块污染。

### 4. [Rust Frontend] Support n > 1 on /inference/v1/generate (#66878)
- **代码层面**：`rust/src/server/src/routes/inference/generate.rs`。把 `NonZeroU32::MIN` 硬编码改成 `prepared.n`，用 `merge_choice_chunk_streams` 把 n 个流合并；`validate_n_limit` 校验 `n` 不超过 `max_n_sequences`。
- **新手概念课堂**：`n` 就像"让模型同时答几遍"。以前 Rust 前端只答一遍，现在能答 n 遍并打包返回。合并流就像把 n 条并行的收据按顺序拼成一张账单。
- **对你有什么影响**：用 Rust 前端（性能更高的新前端）时，`n=3` 这种请求终于能用了，不用再退回 Python 前端。

### 5. [XPU] Flatten N-D input in XPU MXFP8 linear kernel (#66494)
- **代码层面**：`vllm/model_executor/kernels/linear/mxfp8/xpu.py`。`apply_weights` 里先算 `out_shape = x.shape[:-1] + (N,)`，把输入 `reshape(-1, K)` 再量化，GEMM 后 `reshape(out_shape)`。
- **新手概念课堂**：GEMM 只会算二维矩阵乘法。三维输入就像"一摞纸"，得先摊平成一长条再算，算完再叠回原来的摞。Intel XPU 的 MXFP8 kernel 之前只接受二维，三维输入会出错。
- **对你有什么影响**：Intel GPU 用户跑带 batch 维的 MXFP8 模型（如某些 MoE）不再报 shape 错误。

### 6. [Bugfix][Audio] Treat zero decode limits as disabled (#67611)
- **代码层面**：`vllm/multimodal/media/audio.py`（未截全但可推断）。`max_duration_s=0` 或 `max_decode_bytes=0` 现在被解释为"不限制"，而不是"限制为 0 秒/0 字节"。测试覆盖 soundfile / pyav / torchcodec 三个后端。
- **新手概念课堂**：配置里的 `0` 有两种可能语义："零"或"关闭"。以前 vLLM 理解成"零"，导致音频全被拒。现在理解成"关闭"，符合大多数配置系统的惯例。
- **对你有什么影响**：如果你把 `VLLM_MAX_AUDIO_DECODE_BYTES=0` 想表达"不限"，现在终于按你期望的工作了。

### 7. [Rust Frontend] Bump llm-multimodal version (#67670)
- **代码层面**：`rust/Cargo.lock`。`llm-multimodal` 从 1.14.0 升到 1.16.0，仓库地址从 `smg-project/llm-multimodal` 迁到 `smg-project/smg`；顺带引入 `multiversion`、`syn 3.0`，`tokio` 从 1.52.3 升到 1.54.1。
- **新手概念课堂**：`Cargo.lock` 就是 Rust 的"依赖版本快照"。升级就像手机系统更新——新功能、修 bug，但也可能引入不兼容。
- **对你有什么影响**：Rust 前端多模态能力更新（可能支持更多图像格式或更快 resize），普通用户感知不到，但用 Rust 前端跑多模态会更稳。

### 8. [Perf][ROCm][DSv4.1] DeepSeek-V4.1-Flash: dense MXFP8 GEMMs and wo_a on AITER's tuned FlyDSL kernels, and a faster MXFP8 activation quant (MI355X) (#67377)
- **代码层面**：`vllm/model_executor/kernels/linear/mxfp8/rocm.py` 等。DeepSeek-V4.1 的 `wq_b`、`wo_b`、`wqa` 等线性层，如果 AITER 有针对当前 GPU 调优过的 32x32 分块配置，就把权重原地 preshuffle 并走 AITER FlyDSL kernel，否则回退 Triton。
- **新手概念课堂**：AITER 是 AMD 的"kernel 商店"，里面有针对不同矩阵尺寸手工调优的 GEMM。就像点外卖时选"这家店的招牌菜"——同样的食材（矩阵），大厨（调优过的 kernel）做得更快。
- **对你有什么影响**：AMD MI355X 用户跑 DeepSeek-V4.1 会明显更快（dense 层也能吃到 MXFP8 加速）；其他 GPU 或未调优尺寸自动走 Triton，不受影响。

### 9. [Model Loader] Fill dummy FP8 weights with random codes across the range (#67693)
- **代码层面**：`vllm/model_executor/model_loader/weight_utils.py`。删掉 `_DUMMY_FP8_RANGE = 1.0`，新增 `_fill_random_fp8_codes`，随机填满 FP8 编码空间（跳过 NaN/Inf）。测试从"值不为 0"改成"字节种类 > 240 且最大值接近 FP8 上限"。
- **新手概念课堂**：以前 dummy 权重都填 ±1，就像测汽车油耗时只在平地上开——数据好看但不真实。真实权重分布很宽，功耗高。现在 dummy 权重模拟真实分布，基准测试才有参考价值。
- **对你有什么影响**：如果你用 `--load-format dummy` 做性能测试，测得的速度会更接近真实模型（可能比以前慢一点，但那才是真相）。

### 10. [Core][DBO] Dual batch overlap under pure tensor parallelism (split PR of M3.1) (#63483)
- **代码层面**：新增 `--dbo-schedule` 参数，可选 `dispatch-combine-overlap`（原 MoE all-to-all 重叠）或 `all-reduce-overlap`（新，纯 TP 下让 all-reduce 与计算重叠）。`GroupCoordinator._all_reduce_out_place` 里加 yield 点；`maybe_build_ubatch_runner` 在纯 TP 场景也构建 runner。
- **新手概念课堂**：DBO 就是"双缓冲"——把一批请求拆成两半，A 半在算的时候 B 半在通信，轮流来。以前只在 MoE 的 all-to-all 上做，现在纯 TP 的 all-reduce 也能重叠。
- **对你有什么影响**：单机多卡纯 TP 用户开 `--enable-dbo` 就能白嫖性能提升（尤其 decode 阶段 all-reduce 频繁时）。多模态模型在 `all-reduce-overlap` 下也支持了。

### 11. [Bugfix][Core] Keep dummy-run requests within max_model_len (#67545)
- **代码层面**：`vllm/v1/worker/gpu/model_runner.py` 的 `_dummy_run` 里加一行 `num_tokens = min(num_tokens, num_reqs * self.max_model_len)`。原因是 auto-fit 会调低 `max_model_len`，但 `max_num_batched_tokens` 可能还是很大，dummy run 会构造超长请求，导致 block table 越界读。
- **新手概念课堂**：block table 是"每个请求的 KV cache 地址簿"，只够登记 `max_model_len` 个 token。dummy run 如果塞超过这个数，就像往只有 32 页的笔记本里写第 33 页——写到别人的纸上了。
- **对你有什么影响**：开启 auto-fit 或显式设置大 `max_num_batched_tokens` 时，启动阶段不再出现诡异的 attention 越界错误。

### 12. [XPU][MLA] Register XPUFlashAttnMLABackend for FLASH_ATTN_MLA in XPU (#64255)
- **代码层面**：`vllm/platforms/xpu.py`。用 `register_backend(AttentionBackendEnum.FLASH_ATTN_MLA, "....XPUFlashAttnMLABackend")` 注册，`get_attn_backend_cls` 里返回 `AttentionBackendEnum.FLASH_ATTN_MLA.get_path()` 而不是硬编码字符串。
- **新手概念课堂**：`register_backend` 就像给"FLASH_ATTN_MLA"这个名字在 Intel 平台上挂了个别名，指向 XPU 专用实现。以前只有平台选择逻辑里能拿到 XPU 版本，现在任何地方 `get_class()` 都能拿到。
- **对你有什么影响**：Intel GPU 上跑 MLA 模型（如 DeepSeek）时，backend 解析更一致，不会出现"某处拿到通用类、某处拿到 XPU 类"的混乱。

### 13. [Bugfix][MRV2] Allow non 8-bytes aligned --max-num-batched-tokens for DeepSeek-v4 (#66401)
- **代码层面**：`vllm/v1/worker/gpu/block_table.py` 新增 `make_slot_mappings`，用 `round_up(num_tokens, 16)` 分配行 stride，再切片 `[:, :num_tokens]`。这样每行起始地址 128 字节对齐，满足 DeepSeek-V4 CuTe DSL compressor 的 64 字节要求。
- **新手概念课堂**：内存对齐就像停车位——车（数据）必须停在画好的格子里。有些 kernel 要求"车头必须对着特定方向"，不对就报错。给每行多留一点空间，就能保证每行都对齐。
- **对你有什么影响**：`--max-num-batched-tokens` 设成 509 这种"不整"的数时，DeepSeek-V4 不再崩溃。

### 14. [Transformers backend] Serve Qwen2 with the Transformers modeling backend (#66821)
- **代码层面**：大量 `.buildkite/*.yaml` 里删掉 `vllm/model_executor/models/qwen2.py` 的触发路径，说明 Qwen2 现在走 Transformers backend（HF 原生实现）而不是 vLLM 手写实现。CI 不再因 qwen2.py 变更触发。
- **新手概念课堂**：vLLM 支持两种模型实现——"vLLM 手写版"（性能极致优化）和"Transformers 版"（直接用 HF 代码，兼容性好）。Qwen2 迁到后者，说明手写版维护成本大于收益。
- **对你有什么影响**：Qwen2 用户可能感受到细微的性能差异（Transformers 版通常稍慢但更通用），但模型兼容性更广、维护更省心。

### 15. [MM] Support mm_device_do_normalize for LLaVA-OneVision (#67505)
- **代码层面**：`vllm/model_executor/models/llava_onevision.py` 加 `input_norm` 参数，用 `build_mm_input_norm` 构建；processor 在 `mm_device_do_normalize=True` 时输出 uint8 原始像素而非归一化后的 float。测试对比 CPU 归一化和 GPU 归一化结果。
- **新手概念课堂**：图片预处理（减均值除方差）以前在 CPU 上做，占 CPU 时间。现在把原始 uint8 图直接传 GPU，用 GPU kernel 做归一化。就像把"洗菜"从厨房搬到工厂流水线。
- **对你有什么影响**：LLaVA-OneVision 多模态 serving 的 CPU 瓶颈减轻，吞吐提升；显存占用略增（传 uint8 而非 float 反而更省带宽）。

### 16. [Chore] Update agent skill for vLLM (#67273)
- **代码层面**：`.agents/skills/pr-checklist/SKILL.md`。更新 PR 审查清单：强调"diff 要小"、"优先默认值/自动选择而非新开关"、"benchmark 要贴完整命令和原始输出"、"bugfix 要展示 main 上失败、PR 上成功的同一命令"。
- **新手概念课堂**：这是给 AI agent（如 Claude Code）看的"PR 审查指南"。相当于把资深 reviewer 的经验写成 checklist，让 AI 帮忙审 PR 时有章可循。
- **对你有什么影响**：如果你用 AI 辅助提 PR，它会按更严格的标准要求你（比如不能只贴总结表，要贴原始日志）。对贡献者来说门槛更高但质量更好。

### 17. [MM] Support mm_device_do_normalize for Qwen3-Omni (#67426)
- **代码层面**：`vllm/model_executor/models/qwen3_omni_moe_thinker.py` 的 vision encoder 加 `input_norm` 参数，`forward` 里 `self.input_norm(x.to(device), self.dtype)` 替代原来的 `.to(dtype)`。测试加 Qwen3-Omni 的 PPL 测试（需 120GB+ GPU）。
- **新手概念课堂**：和 Commit 15 同一个思路，但 Qwen3-Omni 是多模态 MoE 模型，vision encoder 更复杂。`IdentityInputNorm` 作为默认值保证不开启时行为不变。
- **对你有什么影响**：Qwen3-Omni 多模态推理的 CPU 预处理开销降低，大模型 serving 更流畅。

### 18. [Perf] Allow EPLB with modular TRT-LLM FP8 block-scale MoE (#67591)
- **代码层面**：`vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py` 的 `_supports_parallel_config` 里删掉 `and not moe_parallel_config.enable_eplb`，即不再禁止 EPLB。
- **新手概念课堂**：EPLB（Expert Parallel Load Balancer）是 MoE 的"专家负载均衡器"——哪个专家太忙就挪几个到闲的 GPU 上。以前 TRT-LLM FP8 MoE 不支持 EPLB（可能因为 kernel 假设专家位置固定），现在支持了。
- **对你有什么影响**：用 TRT-LLM FP8 block-scale MoE + EPLB 的部署终于能跑了，MoE 推理的 GPU 利用率更均衡。

### 19. [Bugfix][AuxOutput] Size the routed-experts row cache from one source (#67599)
- **代码层面**：`vllm/distributed/aux_output_connector/connector.py` 的 `retained_objects` 函数统一计算缓存大小：从 `max_bytes`、KV cache 块大小、routed experts 每 token 字节数推导，不再各处独立估算。构造函数 `AuxOutputSchedulerConnector` 新增参数。
- **新手概念课堂**：缓存大小以前是"各处各算各的"，容易不一致（比如 scheduler 算 6 个对象、worker 算 12 个）。现在统一从 `max_bytes` 推导，就像家里预算统一由一个人管，不会出现"老公以为有 1000、老婆以为有 500"。
- **对你有什么影响**：用 aux output（routed experts 捕获）功能时，缓存不会因大小不一致而 OOM 或浪费。

### 20. [Model] Remove OpenPangu-VL (#67632)
- **代码层面**：删掉 `vllm/model_executor/models/openpangu_vl.py`（未截全）、文档里的模型条目、示例脚本里的 `load_openpangu_vl` / `run_openpangu_vl` 函数。
- **新手概念课堂**：模型支持列表就像餐厅菜单，卖不动的菜要下架——维护成本高（要跟着 HF 更新、要测试）、用户少。OpenPangu-VL 是盘古系列的 VL 模型，社区使用量不足以支撑维护。
- **对你有什么影响**：如果你在用 `FreedomIntelligence/openPangu-VL-7B`，需要回退到旧版 vLLM 或改用其他 VL 模型。普通用户无感。

## sglang
## 💡 今日关键词

- **关键词：GQA（分组查询注意力）的边界 bug**
  - 一句话通俗解释：多个 Query 头共享一个 KV 头，当"共享组"大小不是 16 的整数倍时，Triton kernel 会算错下标、读到别人的 KV 头。
  - 为什么社区在关注：GLM-4-32B 这类模型直接崩了，说明大模型推理框架的底层 kernel 里仍有"只在特定形状下才正确"的隐性假设，这是推理框架走向生产化的必经之战。
  - 对新手意味着什么：如果你在学 GPU kernel / Triton，这类"tile 边界 + 取模"问题是绝佳练手题，也是面试高频考点。

- **关键词：投机解码的拒绝采样（Rejection Sampling）支持"缩减词表"**
  - 一句话通俗解释：草稿模型只认识 32000 个词，目标模型认识 128256 个词，现在可以用 `d2t` 映射把草稿概率"摊"到目标词表上再验证。
  - 为什么社区在关注：投机解码是当下推理加速的头号手段，EAGLE3 常用小词表草稿模型，打通这一步等于解放了一大类加速配置。
  - 对新手意味着什么：投机解码 + 概率采样是"推理加速"方向的核心知识，值得深入。

- **关键词：统一鉴权表（auth_levels.json）**
  - 一句话通俗解释：把每个 HTTP 路由该要 API key 还是 admin key，写进一张 JSON 表，Python 服务器和 Rust 服务器都读它。
  - 为什么社区在关注：SGLang 正在把服务端往 Rust 迁移，两套实现必须行为一致，否则会出现"Python 拒绝、Rust 放行"的安全漏洞。
  - 对新手意味着什么：这是"多语言双实现一致性"的经典工程范式，做后端/基础设施的人必学。

- **关键词：Diffusion 推理的性能打磨（in-place 写回、kernel 融合、warp 级并行）**
  - 一句话通俗解释：视频/图像生成模型里，能原地写就不拷贝、能一个 kernel 干完就别拆两个、能一个 warp 跑四行就别跑一行。
  - 为什么社区在关注：今天 20 个 commit 里有 6 个是 diffusion 相关，说明多模态生成正在成为 SGLang 的第二增长曲线。
  - 对新手意味着什么：CUDA/Triton 性能优化在生成式 AI 里极其吃香，且门槛比训练侧低。

- **关键词：int32 溢出（int64 加宽）**
  - 一句话通俗解释：偏移量先乘后转 int64 是没用的——int32 乘法早就溢出了，必须先把操作数转成 int64 再乘。
  - 为什么社区在关注：长上下文、大 state 池越来越常见，这类"静默溢出"bug 极难定位。
  - 对新手意味着什么：写 kernel 时永远问自己"这个乘法会不会溢出"。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：GQA 的 tile 边界与取模（Commit 3）

- **是什么**：GPU kernel 把工作切成"瓦片（tile）"并行处理。GQA 里一个 KV 头对应 `kv_group_num` 个 Q 头，如果这个数不是 tile 大小（16）的整数倍，最后一块 tile 是"半满"的，必须从组起点重新编号，否则会读到隔壁 KV 头的内存。
- **为什么重要**：不懂这个，你会写出"在 H_Q=32 时正确、H_Q=48 时静默算错"的 kernel。推理框架里这种 bug 表现为"某些模型输出乱码"，排查成本极高。
- **动手试试**：用纯 Python 模拟 tile 编号逻辑，直观感受差异。

```python
import math

BLOCK_H = 16

def old_head_ids(cur_head_id, kv_group_num):
    # 旧逻辑：全局按 tile 编号
    return [cur_head_id * BLOCK_H + i for i in range(BLOCK_H)]

def new_head_ids(cur_head_id, kv_group_num, cur_kv_head):
    # 新逻辑：组内 tile 编号，从组起点算
    tiles_per_group = math.ceil(kv_group_num / BLOCK_H)
    tile_in_group = cur_head_id % tiles_per_group
    return [cur_kv_head * kv_group_num + tile_in_group * BLOCK_H + i
            for i in range(BLOCK_H)]

# GLM-4-32B: 48 个 Q 头，2 个 KV 头 -> 每组 24 个头
H_Q, H_KV = 48, 2
g = H_Q // H_KV  # 24
print("每组头数:", g)
for kv_head in range(H_KV):
    for tile in range(math.ceil(g / BLOCK_H)):
        old = old_head_ids(kv_head * math.ceil(g / BLOCK_H) + tile, g)
        new = new_head_ids(kv_head * math.ceil(g / BLOCK_H) + tile, g, kv_head)
        print(f"KV头{kv_head} tile{tile}: 旧={old[:4]}... 新={new[:4]}...")
```

- **预期结果**：你会看到旧逻辑在 KV 头 1 的 tile 上算出的 head id 跑到了 KV 头 0 的区间（比如 KV 头 1 应该从 24 开始，旧逻辑却给了 16），新逻辑则严格落在 `[24, 48)` 内。这就是"读错 KV 头"的根源。

---

### 知识点 2：int32 溢出与"先加宽再相乘"（Commit 17）

- **是什么**：Triton/NumPy 里 `(a * b).to(int64)` 会先用 int32 算 `a*b`，溢出后再转 int64——溢出已经发生了。正确写法是 `a.to(int64) * b`。
- **为什么重要**：KV cache、state 池的偏移量经常是 `batch * heads * dim * dim`，很容易超过 2^31（约 21 亿）。一旦溢出，kernel 会写到完全不相干的内存，表现为随机崩溃或数据污染。
- **动手试试**：

```python
import numpy as np

H, V, K = 64, 128, 128
bos = np.int32(300_000)   # 30 万 token 的 batch 偏移

# 错误写法：先乘后转
bad = np.int32(bos * H * V * K).astype(np.int64)
# 正确写法：先转再乘
good = int(bos) * H * V * K

print("int32 乘法结果:", np.int32(bos * H * V * K))
print("错误写法得到:", bad)
print("正确结果    :", good)
print("是否相等:", bad == good)
```

- **预期结果**：`bad` 是一个很小的数甚至负数（因为 int32 回绕），`good` 是约 3.1e11。`是否相等: False`。这就是为什么 kernel 里必须 `bos.to(tl.int64)` 之后再乘。

---

### 知识点 3：拒绝采样如何保证输出分布不变（Commit 2）

- **是什么**：投机解码里草稿模型先猜 k 个 token，目标模型再验证。拒绝采样（rejection sampling）用"接受概率 = min(1, p_target/p_draft)"筛掉草稿的偏采样，被拒后用残差分布 `max(0, p_target - p_draft)` 重采样，数学上保证最终分布严格等于目标模型分布。
- **为什么重要**：不懂这个，你会以为"草稿模型不准就会让输出变差"——实际上拒绝采样保证无损，只影响速度不影响质量。而"缩减词表"支持意味着草稿分布要先通过 `d2t` 映射散射到目标词表。
- **动手试试**：用 numpy 验证拒绝采样确实无偏。

```python
import numpy as np
rng = np.random.default_rng(0)

V = 6
p_target = np.array([0.05, 0.1, 0.5, 0.2, 0.1, 0.05])
p_draft  = np.array([0.3,  0.3, 0.1, 0.1, 0.1, 0.1])

def rejection_sample(n):
    out = np.zeros(V)
    for _ in range(n):
        x = rng.choice(V, p=p_draft)
        accept = min(1.0, p_target[x] / p_draft[x])
        if rng.random() < accept:
            out[x] += 1
        else:
            residual = np.maximum(p_target - p_draft, 0)
            residual /= residual.sum()
            out[rng.choice(V, p=residual)] += 1
    return out / n

emp = rejection_sample(200_000)
print("目标分布:", p_target)
print("经验分布:", np.round(emp, 3))
print("最大误差:", np.abs(emp - p_target).max())
```

- **预期结果**：经验分布与目标分布几乎一致（最大误差 < 0.01）。把 `accept` 改成永远 1.0 再跑一次，你会看到分布明显偏向草稿模型——这就是"没有拒绝采样就没有正确性保证"。

---

### 知识点 4：原地写回避免拷贝（Commit 11 & 12）

- **是什么**：SDPA 的 varlen 路径原来对每个 segment 调 `forward()` 得到新 tensor，再 `copy_` 回去；改成调 `forward_into()` 直接写进输出切片。Commit 12 更进一步，把一个"门控残差 + RMSNorm + 调制"三步合成一个 Triton kernel，原地更新。
- **为什么重要**：视频生成模型 attention 的中间 tensor 动辄几百 MB，一次多余拷贝就是几十微秒到毫秒；kernel 融合还能省掉中间结果的显存往返。这是 diffusion 推理提速的主要手段。
- **动手试试**：用 PyTorch 量化"拷贝 vs 原地"的差别。

```python
import torch, time

N = 4096
a = torch.randn(N, N, device="cuda")
out = torch.empty_like(a)

def with_copy():
    for _ in range(20):
        seg = a * 1.0001          # 模拟 forward 产生新 tensor
        out.copy_(seg)            # 再拷回去

def in_place():
    for _ in range(20):
        out.copy_(a)              # 直接写目标

for fn in (with_copy, in_place):
    torch.cuda.synchronize(); t = time.perf_counter()
    fn(); torch.cuda.synchronize()
    print(f"{fn.__name__}: {(time.perf_counter()-t)*1000:.2f} ms")
```

- **预期结果**：`in_place` 明显更快，且 `torch.cuda.max_memory_allocated()` 更低。放到真实 varlen attention 里，segment 越多差距越大。

---

### 知识点 5：分布式广播前先校验 manifest（Commit 15）

- **是什么**：`receive_weights_from_distributed` 原来一边 `zip(names, dtypes, shapes)` 一边发起异步 `broadcast`。如果第 3 个 shape 非法，前 2 个 broadcast 已经发出去了，对端会永远等下去（hang）。修复方案：先校验长度相等、先把所有 buffer 分配好，再统一发起 broadcast。
- **为什么重要**：分布式系统里"半途失败"比"直接失败"危险得多——前者会 hang 住整个集群。这是分布式编程的第一原则：**要么全做，要么全不做**。
- **动手试试**：用 `multiprocessing` 模拟"对端等待"。

```python
import multiprocessing as mp, time

def worker(q):
    q.put("ready")
    time.sleep(5)   # 模拟等待一个永远不会来的 broadcast
    q.put("done")

if __name__ == "__main__":
    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start()
    print(q.get())          # ready
    p.terminate()           # 模拟主进程放弃
    print("主进程已退出，子进程被强制终止 —— 真实场景里这就是 hang")
```

- **预期结果**：你会直观看到"对端在等、发起方已放弃"的状态。真实分布式里若没有超时机制，就是永久 hang。修复的关键就是在发起任何异步操作前完成全部校验。

---

## 🔥 重要更新

1. **Commit 3（GQA tile 修复）**：直接修复 GLM-4-32B 在 Triton decode 下的崩溃/乱码，影响所有 `kv_group_num > 16 且非 16 倍数`的模型，属于正确性级别的关键修复。
2. **Commit 2（拒绝采样支持缩减词表）**：解锁 EAGLE3 小词表草稿模型 + 拒绝采样的组合，是投机解码能力边界的重要扩展，同时更新了文档与 HIP 自动开关逻辑。
3. **Commit 10（统一鉴权表 auth_levels.json）**：Python 与 Rust 双服务端共读一张表，是服务端 Rust 化进程中"行为一致性"的里程碑，也顺手修了 Commit 7 的 wrong-method 鉴权绕过。
4. **Commit 18（DeepSeek-V4.1 DP 空闲尾 replay）**：修复 DP 场景下"活跃 rank 被卷进 idle tail replay"的问题，涉及 TBO（two-batch overlap）与 CUDA graph 兼容性投票，属于大规模分布式推理的稳定性修复。

---

## 📋 逐条解读

### 1. [Doc] diffusion: Update the CausalWanSelfAttention Args block after #44753
- **代码层面**：只改 `causal_wanvideo.py` 里 `forward()` 的 docstring，把过时的参数说明（`x`、`seq_lens`、`grid_sizes`）换成实际的 `q/k/v`、`freqs_cis`、`block_mask`、`kv_cache`、`current_start`、`cache_start`。零逻辑变更。
- **新手概念课堂**：docstring 就像菜谱上的"食材清单"。函数签名改了但菜谱没改，后来的人照着做就会放错料。文档债是最便宜也最容易被忽视的债。
- **对你有什么影响**：普通用户无感；但如果你在读源码理解 Causal Wan 的 attention 接口，现在文档和代码对得上了。

### 2. [Fix] spec: Support a reduced draft vocabulary in rejection sampling
- **代码层面**：`eagle_worker_v2.py` 新增 `draft_probs_vocab_size`，把草稿概率通过 `d2t` 映射散射到目标词表；`eagle_draft_cuda_graph_runner.py` 的 `draft_probs` buffer 宽度改为跟随这个值（CUDA graph replay 时形状必须匹配）；`speculative_hook.py` 更新 HIP 自动开启拒绝采样的注释（ROCm 上仍未验证，故仍不自动开）；文档补充参数说明与 PD 分离场景限制。
- **新手概念课堂**：草稿模型像"实习生先写草稿"，目标模型像"主编审稿"。实习生只认识 32000 个词，主编认识 128256 个——`d2t` 就是一张"实习生词汇 → 主编词汇"的翻译表。翻译后概率会摊薄，接受率下降，但输出分布不变。
- **对你有什么影响**：用 EAGLE3 + 小词表草稿模型时，现在可以显式加 `--speculative-use-rejection-sampling` 而不崩；但 PD 分离的 decode server 上仍不支持。

### 3. [Fix] attention: Triton decode reads wrong KV head for GQA groups > 16 not divisible by 16
- **代码层面**：`decode_attention.py` 的 `_grouped_head_tiles` 改为"每个 KV 头占整数个 tile"，并在 `_fwd_grouped_kernel_stage1` 里当 `kv_group_num % BLOCK_H != 0` 时按组内 tile 重新计算 `cur_head` 和 mask。新增测试覆盖 (48,2)、(40,2)、(16,2)、(32,2)、(64,2) 五组配置。
- **新手概念课堂**：想象 24 个人分成 2 组（每组 12 人），但班车每辆坐 16 人。旧逻辑按"第几辆车"编号，第二组的人被安排上了第一组的车；新逻辑按"组内第几辆"编号，从本组起点数。
- **对你有什么影响**：GLM-4-32B 等 GQA 组大小非 16 倍数的模型，decode 输出从"乱码/崩溃"恢复正常。

### 4. [Fix] diffusion: Build an offloaded text encoder's RoPE table on the GPU
- **代码层面**：`text_encoder_loader.py` 新增 `_build_rope_caches_on()`，对 `use_precomputed_cache` 的 `RotaryEmbedding` 模块在目标 device 上重算 `cos_sin_cache`；Qwen3 文本编码器若从 CPU 起加载则调用它。原因是 CPU 与 CUDA 的 `cos/sin` 某些项舍入不同，会导致 offload 与 resident 输出不一致。
- **新手概念课堂**：RoPE 是给 token 位置"打时间戳"的旋转编码。CPU 和 GPU 算三角函数时最后一位可能差 1 ulp，就像两个计算器算 sin(30°) 一个显示 0.5 一个显示 0.4999999——单看没事，叠 40 层就放大成可见差异。
- **对你有什么影响**：开了 text encoder offload 的 diffusion 用户，生成结果与不 offload 时保持一致（bit-level 对齐）。

### 5. [Fix] granitemoe: Declare packed_modules_mapping
- **代码层面**：`granitemoe.py` 的 `GraniteMoeForCausalLM` 新增 `packed_modules_mapping = {"qkv_proj": ["q_proj","k_proj","v_proj"], "input_linear": ["input_linear"]}`。
- **新手概念课堂**：推理框架把 q/k/v 三个矩阵"打包"成一个 `qkv_proj` 大矩阵以提速。但量化配置（比如 compressed-tensors）是按 `q_proj` 这种原始名字写的——没有映射表，量化器找不到目标，直接报错或静默跳过。
- **对你有什么影响**：GraniteMoE 模型配 per-projection 量化方案现在能正确加载。

### 6. [Feat] glm5.3flash: Allow PD prefill CP with native MTP or without speculative decoding
- **代码层面**：`cp_tp_group_sharing_hook.py` 移除 GLM5Next 在 PD 场景下"必须 DFLASH"的限制；测试重构出 `speculative_args()` 钩子，新增 `TestGlm53CpBcgMtpPD`（MTP 版本）继承 DFlash 测试。
- **新手概念课堂**：PD 分离 = 把"读题"（prefill）和"答题"（decode）拆到不同机器。CP = 上下文并行，把长 prompt 切给多卡算。以前 prefill 侧开 CP 只允许配 DFLASH 投机，现在 MTP 或不投机也行——因为 decode 侧只是普通 TP，不关心 prefill 用什么投机算法。
- **对你有什么影响**：GLM-5.3-Flash 的 PD 部署配置自由度大幅提升。

### 7. [Fix] auth: Key-check wrong methods by route and allowlist `/internal/peer` messages
- **代码层面**：Python 侧 `_get_auth_level_from_app_and_scope` 在 `Match.PARTIAL`（路径匹配但方法不对）时也查鉴权等级，从而在 405 之前先做 403 检查；Rust 侧新增 `is_admin_force()`，并给 `/internal/peer` 的转发消息加白名单校验。
- **新手概念课堂**：以前 `POST /start_profile`（只允许 GET）会先返回 405"方法不对"，攻击者由此可探测路由存在性；现在先用该路由的鉴权等级拦一道 403，信息不泄露。
- **对你有什么影响**：开了 API key 的部署更安全；正常调用无感知。

### 8. [Fix] observability: Snapshot STANDALONE metrics before settlement
- **代码层面**：`metrics_reporter.py` 的投机算法集合加入 `SpeculativeAlgorithm.STANDALONE`；测试同步加入。
- **新手概念课堂**：STANDALONE 是一种独立投机算法。metrics 需要在请求"结算"（settle）前快照，否则被撤回（retraction）的请求统计会丢。漏了这个枚举值，STANDALONE 用户的吞吐指标会偏低。
- **对你有什么影响**：用 STANDALONE 投机时，Prometheus 指标现在准确了。

### 9. [Feat] grpc: Advertise api.v1 owning-runtime instance identity
- **代码层面**：proto 的 `GetServerInfoResponse` 新增 `string instance_id = 16`；Python `api_types.py` 的 encode/decode 同步；Rust crate `sglang-api-types` 版本 0.1.0 → 0.2.0。
- **新手概念课堂**：`instance_id` 是"这个 Rust 运行时进程的出生证明"（UUIDv4），进程重启就换新。它和 `PortArgs.instance_id`（IPC 命名空间）不是一回事——注释里特意强调"不是健康检查、不是集群 epoch"。
- **对你有什么影响**：外部编排器可以据此判断"对端是不是同一个运行时实例"，从而安全地做重连/重启决策。

### 10. [Refactor] auth: Declare every HTTP route's auth level in one table
- **代码层面**：新增 `python/sglang/srt/utils/auth_levels.json`，Python 与 Rust 服务端都从这张表读取路由鉴权等级；CI 缓存 key、Rust workspace 触发条件、文档全部同步加入该文件。
- **新手概念课堂**：就像公司把"哪些门需要刷卡、哪些需要主管卡"写进一张总表贴在门口，而不是每个保安各记一套。双实现读同一份配置 = 天然一致。
- **对你有什么影响**：Python 与 Rust 服务端的鉴权行为不再漂移；改路由权限只需改一个 JSON。

### 11. [Perf] diffusion: Write varlen SDPA segments in place when the backend can
- **代码层面**：`sdpa.py` 的 `forward_varlen` 对非 MPS 设备优先调 `self.forward_into(q,k,v,out_slice,None)`，返回 False 才回退到 `forward + copy_`。新增测试用假实现验证"长 segment 走 into、短 segment 回退 forward"，以及 FA4 路径 bit-identical。
- **新手概念课堂**：`forward_into` 就像"把汤直接盛进你给的碗"，`forward` 是"盛进新碗再倒进你的碗"。少一次倒腾，省一次显存分配 + 拷贝。
- **对你有什么影响**：视频生成 varlen attention 更快，显存峰值更低，输出 bit 级不变。

### 12. [Perf] diffusion/minimax-h3: Fuse the attention gated residual with the norm2 modulation
- **代码层面**：`rmsnorm_scale_shift_triton.py` 新增 `_indexed_gate_rmsnorm_scale_shift_kernel`，把"门控残差 `x += gate*other`（原地写回）"和"RMSNorm + indexed scale/shift"合成一个 kernel；`__init__.py` 注册 `indexed_gate_rmsnorm_scale_shift_` 与 `can_use_*`。
- **新手概念课堂**：原本是"炒菜 → 装盘 → 再下锅炒"三步，现在一口锅炒完。中间结果不落地，省两次全局内存往返。注意 kernel 里刻意用 `round_bf16_to_fp32` 复现原两步的舍入，保证数值一致。
- **对你有什么影响**：MiniMax-H3 视频生成 attention 块更快，数值结果不变。

### 13. [Test] test_utils: Add ADMIN_LAUNCH_ARGS_FOR_TEST and admin_post
- **代码层面**：`test_utils.py` 新增 `ADMIN_LAUNCH_ARGS_FOR_TEST` 元组和 `admin_post()` 辅助函数；`pause_generation_kit.py`、`test_elastic_scale.py` 等改用它们，避免手写 headers/args 重复。
- **新手概念课堂**：测试代码也需要 DRY。把"启动参数 + 请求头"封装成常量/函数，以后改 admin key 只改一处。
- **对你有什么影响**：普通用户无感；贡献测试的开发者少写样板代码。

### 14. [Perf] hicache: Reuse the host eviction heap across one write_back eviction walk
- **代码层面**：`full.py` 新增 `_host_eviction_heap` 成员，在一次 eviction walk 内复用堆；新增 `on_host_leaf_added()` 在 walk 进行中把新叶子 push 进堆；`drive_host_eviction` 改为惰性建堆 + try/finally 清理（session radix cache 开启且 evict_device 进行中则保留堆）。
- **新手概念课堂**：以前每次驱逐都"重新把所有候选节点堆一遍"（O(n) heapify），现在一次 walk 只建一次堆，walk 中新产生的叶子直接 push。就像整理衣柜时不用每次都把所有衣服倒出来重排。
- **对你有什么影响**：hicache 的 host 侧驱逐更快，长会话高并发下 CPU 开销下降。

### 15. [Fix] weights: Reject mismatched manifests before distributed broadcasts
- **代码层面**：`weight_updater.py` 的 `receive_weights_from_distributed` 先校验 `len(names)==len(dtypes)==len(shapes)`，再把"分配 buffer"和"发起 broadcast"分成两个循环——先全部分配，再统一异步广播。
- **新手概念课堂**：`zip` 遇到长度不等会静默截断，导致"少收一个权重"；而边分配边广播，一旦中途失败，已发出的 broadcast 会让对端永久等待。分布式铁律：**校验在前，动作在后；要么全发，要么全不发**。
- **对你有什么影响**：权重更新遇到非法 manifest 会立刻报清晰错误，而不是集群 hang 死。

### 16. [Perf] attention: Avoid host synchronization in GDN MIS state resets
- **代码层面**：`gdn_backend.py` 的 `_forward_extend_mis` 把 `conv_states[cache_indices] = 0`（高级索引，会触发 host 同步）改成 `index_fill_(0, cache_indices.to(int64), 0)`（纯 device 操作）。测试用 `torch.cuda.set_sync_debug_mode("error")` 验证无同步，并覆盖 strided 池、重复索引、空索引。
- **新手概念课堂**：`tensor[idx] = 0` 在 PyTorch 里可能触发"把索引搬到 CPU 再决定怎么写"的同步点，GPU 流水线被打断。`index_fill_` 是原生 kernel，全程留在 GPU 上。
- **对你有什么影响**：GDN（Gated DeltaNet）多请求混合推理时，每个 step 少一次 GPU 空等，吞吐提升。

### 17. [Fix] attention: Widen GDN chunk-state offsets to int64 before multiplying
- **代码层面**：`chunk_delta_h.py`（CUDA 与 XPU 两份）把 `bos`、`boh` 先 `.to(tl.int64)` 再参与乘法；`p_track_base` 改为 `i_n.to(tl.int64) * stride_track_state + ...`。测试新增超大偏移场景。
- **新手概念课堂**：`(a*b).to(int64)` 是"先在小杯子里搅，搅溢出了再倒进大杯子"。必须先把原料倒进大杯子再搅。int32 上限约 21 亿，`batch * heads * V * K` 很容易破。
- **对你有什么影响**：长上下文 / 大 state 池下 GDN 不再静默写错内存。

### 18. [Fix] deepseek-v4.1: Keep active DP decode ranks out of idle tail replay
- **代码层面**：`forward_batch_info.py` 新增 `global_late_layer_tail_graph_compatible`，通过 all-gather 收集各 rank 的"是否 tail-graph 兼容"投票并取 `all()`；`deepseek_v4_replay_tail.py` 的 `scattered_dp_counts` 增加该参数并返回三元组；`two_batch_overlap.py` 的 filter 同步补字段；`deepseek_v4.py` 移除旧的 `late_dp_counts` 传参。
- **新手概念课堂**：DP（数据并行）下各 rank 步调要一致。某些 rank 这步"空闲"（idle），框架会做 tail replay 补齐。但如果某个 rank 其实在活跃 decode，把它卷进 idle replay 就会算错。新逻辑：所有 rank 投票，只要有一个不兼容就整体不 replay。
- **对你有什么影响**：DeepSeek-V4.1 在 DP + TBO + CUDA graph 组合下 decode 输出稳定，不再偶发错乱。

### 19. [Perf] [NVIDIA] gemma4: write o_proj/down_proj partials straight into the AR-fusion symmetric buffer
- **代码层面**：`fp8_kernel.py` 的 `prepare_block_fp8_matmul_inputs` / `w8a8_block_fp8_matmul_deepgemm` 新增 `output` 参数（调用方提供存储）；`gemma4_ar_fusion.py` 新增 `partial_out()`，`_stage()` 检测 `data_ptr` 相同则跳过 copy；`compressed_tensors.py` 新增 `apply_into()` 让 linear 直接写调用方 buffer。
- **新手概念课堂**：AllReduce 融合需要把各卡的 partial 结果放进一块"对称缓冲区"。以前是 GEMM 算到临时 tensor，再拷进对称 buffer；现在 GEMM 直接写进去，省一次拷贝。
- **对你有什么影响**：Gemma4 在 NVIDIA 上开启 AR fusion 时，o_proj/down_proj 更快，显存占用更低。

### 20. [Perf] diffusion: Run head_dim-64 rounded NeoX QK-norm+RoPE four rows per warp
- **代码层面**：`qknorm_rope.cuh` 新增 `fused_qknorm_rope_quarter_warp` kernel：head_dim=64、rounded NeoX、half-width cache 场景下，8 lane 处理一行、一个 warp 处理 4 行（原来是 32 lane 一行）。逐 lane 复现原 warp kernel 的 xor butterfly 归约顺序，保证 bit 级一致。
- **新手概念课堂**：一个 warp 有 32 个线程。head_dim=64 时，原方案 32 线程处理一行有点"人多活少"；新方案 8 线程一行、4 行并行，warp 利用率更高。难点是要精确复现原归约顺序（16→8→4 跨 lane，2→1 在寄存器），否则浮点结果会变。
- **对你有什么影响**：diffusion 模型的 QK-norm+RoPE 融合 kernel 更快，输出 bit 级不变。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)