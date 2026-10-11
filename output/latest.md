# 🚀 vLLM & SGLang 每日更新 — 2026-10-11

> 自动生成于 2026-10-11 07:47 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：显存快照（Memory Snapshot）时机**
  - **一句话通俗解释**：在加载模型权重前后记录 GPU 显存用了多少，用来决定 KV Cache 能开多大。顺序错了，算出来的可用显存就不准。
  - **为什么社区在关注**：vLLM 的 "Fast Start" 系列提交连续出现，说明社区非常在意**启动时间**和**显存利用率**这两个直接影响成本的指标。
  - **对新手意味着什么**：理解"显存账本"是入门 LLM 推理部署的第一课。学会看 `MemorySnapshot`、`empty_cache`、`gc.collect` 这些工具，你就能自己排查 OOM 问题。

- **关键词：异步 RL / 权重版本追踪（Weight Version）**
  - **一句话通俗解释**：在强化学习里，模型一边生成数据一边更新权重，需要知道"哪些 token 是旧权重生成的，哪些是新权重生成的"。
  - **为什么社区在关注**：RLHF/RLVR 是当前大模型训练的主流范式，vLLM 正在把推理引擎改造成支持"权重热更新"的基础设施。
  - **对新手意味着什么**：如果你对 RL 训练感兴趣，这是理解"推理-训练解耦"架构的绝佳切入点。

- **关键词：KV Cache 多组管理（KV Cache Groups）**
  - **一句话通俗解释**：一个模型可能同时有多种注意力模式（比如滑动窗口 + 全注意力），每种模式需要独立的 KV 缓存块，管理起来很复杂。
  - **为什么社区在关注**：混合注意力模型（如 Gemma、K-EXAONE）越来越多，KV Cache 管理从"单组"变成"多组"是必然趋势。
  - **对新手意味着什么**：这是理解现代推理引擎调度器（Scheduler）的核心难点之一。

- **关键词：跨进程权重传输（Packed Tensor / IPC）**
  - **一句话通俗解释**：把模型权重从一个进程（比如训练进程）打包传给另一个进程（推理进程），中间不能丢数据、不能变形。
  - **为什么社区在关注**：分布式推理、RL 权重同步、模型预加载都依赖这个底层能力。
  - **对新手意味着什么**：0-dim 张量、dtype、shape 这些 PyTorch 基础功，在真实工程里就是 bug 的来源。

- **关键词：废弃代码清理（Deprecation & Dead Code）**
  - **一句话通俗解释**：把过时的参数、没人调用的函数删掉，让代码库保持干净。
  - **为什么社区在关注**：vLLM 迭代速度极快，定期清理是保持可维护性的必要手段。
  - **对新手意味着什么**：学会读 deprecation warning，能帮你避免在新项目里用已经要删的 API。


## 🧪 重要知识点 & 动手实验

- **知识点 1：GPU 显存快照与 KV Cache 分配**
  - **是什么**：vLLM 启动时会先加载模型权重，然后"拍一张"显存快照，用总显存减去已用显存，算出还剩多少可以给 KV Cache。
  - **为什么重要**：如果快照拍早了（权重还没加载完），算出的可用显存会偏大，运行时就会 OOM；拍晚了又会浪费显存。Commit 1 就是修复这个顺序问题。
  - **动手试试**：
    ```python
    import torch, gc

    # 模拟加载权重前后的显存变化
    def snapshot():
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1024**3  # GB

    print("加载前:", snapshot())
    dummy_weights = [torch.randn(1024, 1024, device="cuda") for _ in range(100)]
    print("加载后:", snapshot())

    # 释放并清理缓存
    del dummy_weights
    print("释放后:", snapshot())
    ```
  - **预期结果**：你会看到"加载后"的显存明显增加，"释放后"如果不调用 `empty_cache`，显存不会立刻归还。这就是为什么 vLLM 要在关键节点调用 `gc.collect()` + `torch.accelerator.empty_cache()`。

- **知识点 2：0-dim 张量与张量打包传输**
  - **是什么**：0-dim 张量就是"标量张量"，比如 `torch.tensor(1.5)`，它的 shape 是 `()`。打包传输时如果处理不当，会丢失形状信息。
  - **为什么重要**：量化模型里经常有 `scale`、`zero_point` 这样的标量参数，传输时丢了形状就会报错或算错。
  - **动手试试**：
    ```python
    import torch

    # 对比不同 view 顺序的结果
    scalar = torch.tensor(1.5, dtype=torch.float32)
    matrix = torch.randn(2, 3, dtype=torch.float32)

    # 错误顺序：先 view(uint8) 再 view(-1)
    try:
        bad = scalar.contiguous().view(torch.uint8).view(-1)
        print("错误顺序 shape:", bad.shape)
    except Exception as e:
        print("错误顺序报错:", e)

    # 正确顺序：先 view(-1) 再 view(uint8)
    good = scalar.contiguous().view(-1).view(torch.uint8)
    print("正确顺序 shape:", good.shape)
    ```
  - **预期结果**：你会发现 0-dim 张量直接 `view(torch.uint8)` 会报错，必须先 `view(-1)` 变成 1-dim。这就是 Commit 8 修复的核心。

- **知识点 3：PyTorch 缓存分配器与显存回收**
  - **是什么**：PyTorch 不会把释放的显存立刻还给 GPU，而是放进"缓存池"里备用。`torch.cuda.empty_cache()` 才会真正归还。
  - **为什么重要**：在权重加载、量化后处理等阶段，如果不主动清理缓存，后续的 CUDA 驱动级分配（比如 IPC 共享内存）可能失败。
  - **动手试试**：
    ```python
    import torch, gc

    x = torch.randn(1000, 1000, device="cuda")
    print("分配后 reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
    del x
    print("删除后 reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
    gc.collect()
    torch.cuda.empty_cache()
    print("清理后 reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
    ```
  - **预期结果**：删除张量后 `reserved` 不变，调用 `empty_cache()` 后才会下降。理解这一点能帮你排查"明明删了张量还 OOM"的问题。

- **知识点 4：滑动窗口注意力与 RoPE 的应用范围**
  - **是什么**：滑动窗口注意力只看最近的 N 个 token，全注意力看所有 token。RoPE（旋转位置编码）是给 token 加位置信息的方法，但不是所有层都需要加。
  - **为什么重要**：混合注意力模型（部分层滑动窗口、部分层全注意力）越来越多，搞错哪些层加 RoPE 会导致模型输出完全错误。
  - **动手试试**：
    ```python
    # 模拟一个混合注意力配置
    layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
    sliding_windows = [4096, None, 4096]

    for i, (lt, sw) in enumerate(zip(layer_types, sliding_windows)):
        use_rope = bool(sw) or lt == "full_attention"
        print(f"层 {i}: 类型={lt}, 窗口={sw}, 是否用 RoPE={use_rope}")
    ```
  - **预期结果**：你会看到滑动窗口层和全注意力层的 RoPE 使用逻辑不同。Commit 14 修复的就是 MTP 层应该强制用 RoPE 的逻辑。

- **知识点 5：异步 RL 中的权重版本追踪**
  - **是什么**：在异步 RL 中，推理引擎可能在生成到一半时收到新权重，需要记录"第 0-57 个 token 是 v7 权重生成的，第 57-128 个是 v8 权重生成的"。
  - **为什么重要**：训练时需要知道每个 token 对应的策略版本，才能正确计算重要性采样比率。搞错了会导致训练不稳定。
  - **动手试试**：
    ```python
    # 模拟权重版本追踪
    class WeightVersionTracker:
        def __init__(self):
            self.spans = []
            self.current_version = "default"
            self.current_start = 0

        def update_version(self, new_version, current_token_count):
            if self.current_version != new_version:
                self.spans.append({
                    "version": self.current_version,
                    "start": self.current_start,
                    "end": current_token_count,
                })
                self.current_version = new_version
                self.current_start = current_token_count

        def finalize(self, total_tokens):
            self.spans.append({
                "version": self.current_version,
                "start": self.current_start,
                "end": total_tokens,
            })
            return self.spans

    tracker = WeightVersionTracker()
    tracker.update_version("v7", 0)
    tracker.update_version("v8", 57)  # 第 57 个 token 时切换权重
    print(tracker.finalize(128))
    ```
  - **预期结果**：你会得到 `[{'version': 'v7', 'start': 0, 'end': 57}, {'version': 'v8', 'start': 57, 'end': 128}]`，这正是 Commit 7 文档里展示的格式。


## 🔥 重要更新

- **Commit 1 & 2（显存管理修复）**：这两个提交都在修复显存快照和缓存回收的时机问题，直接影响 vLLM 启动时的 KV Cache 分配准确性。如果你遇到过"启动时显示显存够但运行时 OOM"，这两个修复可能就是你需要的。

- **Commit 7（权重版本追踪）**：这是异步 RL 基础设施的重要一步。现在每次请求的最终输出都会带上 `weight_versions`，训练侧可以精确知道每个 token 对应的策略版本。对做 RLHF 的团队来说，这是刚需功能。

- **Commit 6（KV Cache 多组预留修复）**：修复了投机解码 + 混合注意力模型下的死锁问题。如果你在用 EAGLE、Medusa 等投机解码方案，并且模型有滑动窗口注意力，这个修复很关键。

- **Commit 11（FlashInfer 自动调优预加载）**：把 FlashInfer 的 autotune 过程提前到预加载阶段，能显著减少首次启动时间。对追求"秒级启动"的生产环境很有价值。


## 📋 逐条解读

### 1. [Fast Start][Bugfix] Snapshot GPU memory after external weights are ready (#60966)

- **代码层面**：修改了 `vllm/v1/worker/gpu_worker.py` 的 `init_device` 方法。原来是在获取外部权重之前就拍显存快照，现在改成了先获取外部权重，再拍快照。这样算出的 `init_snapshot` 才包含外部权重占用的显存。
- **新手概念课堂**：想象你要租房子，先量了一下房间面积，然后搬进来一堆家具，再问"还能放多少东西"——这显然不对。正确做法是先搬完家具再量剩余空间。显存快照也是同理。
- **对你有什么影响**：如果你使用外部模型加载器（比如从共享内存或 IPC 加载权重），启动时 KV Cache 的分配会更准确，减少 OOM 风险。

### 2. [Bugfix] Reclaim cached GPU memory during weight loading (#57516)

- **代码层面**：在 `vllm/model_executor/model_loader/base_loader.py` 的 `load_model` 方法中，权重加载后和权重后处理（量化等）后各加了一次 `gc.collect()` + `torch.accelerator.empty_cache()`。
- **新手概念课堂**：PyTorch 的显存分配器像一个"二手市场"——你退租的房子不会立刻还给房东，而是挂在市场上等下一个租客。`empty_cache()` 就是强制清空市场，把房子还给房东。
- **对你有什么影响**：加载大模型时显存峰值会降低，尤其是在线量化场景下，能避免"加载完权重后没显存做后处理"的尴尬。

### 3. [Bugfix][Multimodal] Fix per-video cache option reuse (#55203)

- **代码层面**：修改了 `tests/models/multimodal/processing/test_qwen3_vl.py` 和相关的多模态处理代码。修复了多视频请求中，每个视频的 `fps`/`num_frames` 参数在缓存复用时被错误覆盖的问题。
- **新手概念课堂**：你去餐厅点了两碗面，一碗加辣一碗不加辣。如果服务员把"加辣"这个选项记在了整张订单上，而不是每碗面上，就会出错。多视频处理也是同理，每个视频的参数要独立记录。
- **对你有什么影响**：如果你用 Qwen3-VL 等模型处理多视频请求，每个视频的帧率、帧数参数现在会被正确处理，不会串味。

### 4. [Perf][MiniMax-M3] MSA prefill attends with the E4M3 query; bump fmha_sm100 to 968061a (#60684)

- **代码层面**：更新了 `cmake/external_projects/fmha_sm100.cmake` 中的 FlashInfer 子模块版本，并新增了多个 Python 文件（`_jit_cache.py`、`msa_warmup.py`、`q8kv4_prefill_adapter.py` 等）。核心变化是 MSA prefill 阶段用 E4M3 格式的 query 做注意力计算。
- **新手概念课堂**：E4M3 是一种 8 位浮点数格式（4 位指数 + 3 位尾数），比 BF16 更省显存和带宽。用更低的精度做注意力计算，就像用更小的集装箱运货——装得少但跑得快。
- **对你有什么影响**：如果你在 H100/B200 等 SM100 架构 GPU 上跑 MiniMax-M3，prefill 阶段会更快、更省显存。普通用户可能感知不到，但吞吐量会有提升。

### 5. [MRV2] Detatch weight-reload logic from MRV1 (#60937)

- **代码层面**：把权重重载逻辑从 `gpu_model_runner.py` 中抽离出来，放到独立的 `vllm/model_executor/model_loader/reload/inplace.py` 模块。测试代码也相应调整，用 `Worker.update_config` 替代 `model_runner.update_config`。
- **新手概念课堂**：这就像把厨房里的"切菜"功能单独搬到一个料理台上，而不是和"炒菜"挤在一起。代码模块化后，维护和测试都更容易。
- **对你有什么影响**：普通用户无感知，但如果你在开发自定义权重加载逻辑，新架构会更清晰。

### 6. [Bugfix][KV Connector] Fix speculative block reservations across KV cache groups (#59880)

- **代码层面**：修改了 KV Connector 的块预留逻辑，新增了多组 KV Cache 的测试用例。修复了投机解码场景下，异步加载请求没有为"提升（promotion）"预留足够块的问题。
- **新手概念课堂**：想象停车场有多个区域（对应多个 KV Cache 组），一辆车要停进来，不仅要看总空位，还要看它要去的那个区域有没有空位。原来只检查了总空位，导致某些区域满了但车还是被放进来，最后卡死。
- **对你有什么影响**：如果你用投机解码 + 混合注意力模型 + KV Connector，不会再遇到"请求卡在 WAITING 状态永远不执行"的死锁问题。

### 7. [RL][Frontend] Return token-level weight-version spans on final outputs (#59582)

- **代码层面**：在最终输出中新增 `weight_versions` 字段，记录每个 token 范围对应的权重版本。新增了 `/update_weight_version` HTTP 接口，文档也相应更新。
- **新手概念课堂**：就像给每段录音打上时间戳和"录制设备版本"标签。训练时回放这些录音，就知道哪段是旧设备录的、哪段是新设备录的。
- **对你有什么影响**：如果你在做异步 RL 训练，现在可以精确知道每个 token 对应的策略版本，训练稳定性会提升。

### 8. [Bugfix] Support 0-dim tensors in packed weight transfer (#52235)

- **代码层面**：修改了 `vllm/distributed/weight_transfer/packed_tensor.py`，把 `view(dtype).view(-1)` 改成 `view(-1).view(dtype)`，并支持 0-dim 张量的打包和解包。
- **新手概念课堂**：0-dim 张量就像一个"没有形状"的数字。你想把它塞进一个方形盒子里，得先把它压扁成一条线（`view(-1)`），再重新塑形。
- **对你有什么影响**：量化模型里的标量参数（如 scale）现在能正确传输，不会再报 shape 错误。

### 9. [Deprecation] Remove scheduled deprecated items (#60923)

- **代码层面**：删除了 `--dataset` 参数、`wait_for_inflight_requests` 参数等已标记废弃的 API。涉及 `vllm/benchmarks/throughput.py`、`vllm/engine/protocol.py` 等多个文件。
- **新手概念课堂**：就像手机系统更新后，旧版本的 App 可能不再兼容。vLLM 也会定期清理过时 API，保持代码库健康。
- **对你有什么影响**：如果你还在用 `--dataset` 参数跑 benchmark，需要改成 `--dataset-name` + `--dataset-path`。用 `wait_for_inflight_requests` 的代码要改成 `mode="wait"`。

### 10. [Bugfix][Frontend] Forward --enable-log-outputs to the Anthropic handler (#60982)

- **代码层面**：在 `vllm/entrypoints/anthropic/serving.py` 的 `__init__` 中新增 `enable_log_outputs` 和 `enable_log_deltas` 参数，并在 `api_router.py` 中传入。
- **新手概念课堂**：就像你设置了"打印日志"开关，但某个子模块没接收到这个设置。这个修复就是把开关信号正确传递到 Anthropic 处理器。
- **对你有什么影响**：如果你用 Anthropic API 兼容接口，并且开启了 `--enable-log-outputs`，现在日志会正常输出。

### 11. [Fast Start] Preload the FlashInfer autotune table on the weight cache daemon (#60085)

- **代码层面**：在 `vllm preload` 命令中新增了 FlashInfer autotune 预加载功能，并更新了文档。支持 TP/EP/PP/DP 各种并行配置。
- **新手概念课堂**：FlashInfer 有多种注意力实现，需要跑一遍 benchmark 选最快的。这个过程很耗时，就像新电脑第一次开机要跑分。预加载就是提前把跑分做完，下次开机直接用结果。
- **对你有什么影响**：首次启动时间显著缩短，尤其是在大 MoE 模型上。如果你用 `vllm preload`，启动会更快。

### 12. [Refactor] Remove dead code multiple places (#60120)

- **代码层面**：删除了 `cpu_activation_lut_bf16`、`SupportsLoRAMultiModalModel`、`prepare_int8_moe_layer_for_cpu` 等无人调用的函数和类。
- **新手概念课堂**：就像打扫房间，把不用的旧家具扔掉，房间更整洁。代码库也一样，删掉死代码能减少维护负担。
- **对你有什么影响**：普通用户无感知。如果你在代码里引用了这些被删的函数，需要更新。

### 13. [Compile] Fix sampler compile warning (#60680)

- **代码层面**：调整了 `csrc/libtorch_stable/sampler.cu` 中 `#ifdef USE_ROCM` 的位置，把 ROCm 专用的 topK 实现正确包裹起来。
- **新手概念课堂**：就像给不同国家的电器配不同的插头。`#ifdef` 是编译时的"条件开关"，确保 ROCm 代码只在 AMD GPU 上编译。
- **对你有什么影响**：编译 vLLM 时不会再看到相关警告，ROCm 用户和 CUDA 用户各用各的代码路径。

### 14. [Bugfix][Model] K-EXAONE: restore loading of pre-2.0 configs and apply RoPE to the MTP layer (#60187)

- **代码层面**：修改了 `vllm/model_executor/models/exaone_moe.py`，重构了滑动窗口和 RoPE 的判定逻辑，支持旧版配置（只有标量 `sliding_window`），并确保 MTP 层强制使用 RoPE。
- **新手概念课堂**：旧版模型配置就像老式收音机，只有"开关"和"音量"两个旋钮。新版配置像智能音箱，有各种参数。这个修复就是让 vLLM 同时支持两种"收音机"。
- **对你有什么影响**：如果你用 K-EXAONE 旧版模型，现在能正常加载了；MTP 层的 RoPE 也正确应用，输出质量会更好。

### 15. [BUG] Fix KV offload shared-memory slot selection using worker rank (#58497)

- **代码层面**：修改了 KV offload 的共享内存槽位选择逻辑，从使用 `device_index % world_size` 改成使用 `worker_rank`。测试也相应更新。
- **新手概念课堂**：就像 4 个人合租一个公寓，每人有一个储物柜。原来用"房间号 % 4"来分配柜子，但房间号可能不连续（比如 5、6、7、8），导致分配错乱。改成用"第几个租客"来分配就对了。
- **对你有什么影响**：如果你用 KV offload 功能，共享内存槽位分配更正确，不会再出现 rank 冲突。

### 16. [Bugfix][Responses API] Complete cache-write token accounting (#60973)

- **代码层面**：在 Responses API 的上下文中新增 `num_cache_creation_tokens` 字段，并在多轮对话中累加。涉及 `test_parsable_context_unit.py` 和 `test_context.py` 等测试文件。
- **新手概念课堂**：就像记账时只记了"花了多少钱"，忘了记"存了多少钱"。Cache-write token 就是"存进 KV Cache 的 token"，需要单独统计。
- **对你有什么影响**：如果你用 Responses API，现在能看到完整的 cache-write token 统计，方便做成本核算和性能分析。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：调度器（Scheduler）与 KV 池的"精打细算"**
  - **一句话通俗解释**：推理引擎的"调度员"要决定每一步把哪些请求放进 GPU 跑，同时不能超出显存里 KV 缓存的空间。
  - **为什么社区在关注**：今天有 4 个 commit（#43290、#43298、#43479 等）都在修调度器的边界情况，说明大模型推理的瓶颈已经从"算得慢"转向"调度得乱"——显存够不够、请求公平不公平、多卡之间同步不一致，都是实际生产里最痛的问题。
  - **对新手意味着什么**：如果你在学 LLM 推理框架，调度器是"最容易被忽视但最影响吞吐"的模块，深入读 `scheduler.py` 比追新模型更有性价比。

- **关键词：torch.compile 与 CUDA Graph 的"可追踪性"**
  - **一句话通俗解释**：PyTorch 的编译器想把整个模型"录制"成一个静态图来加速，但有些操作（比如通信、自定义 kernel）录不进去，就会"图断裂"。
  - **为什么社区在关注**：今天有 5 个 diffusion 相关 commit（#43634、#43577、#43586、#43632、#43633）都在处理"哪些操作该在编译图里、哪些该 eager 执行"，说明 diffusion 模型的编译优化已经成了性能主战场。
  - **对新手意味着什么**：`torch.compiler.is_compiling()` 这个判断正在成为新代码的标配，学会写"对编译器友好"的代码是未来必备技能。

- **关键词：Attention 后端与稀疏注意力（Sparse Attention）**
  - **一句话通俗解释**：不同的注意力实现（FlashAttention、稀疏注意力、Sliding Window）适配不同的硬件和场景。
  - **为什么社区在关注**：Qwen 的稀疏注意力（QSA）、AMD HIP Graph、Helion KDA 都在今天出现，说明"一个模型多种后端"正在成为常态。
  - **对新手意味着什么**：不要只学一种 attention 实现，理解"后端可插拔"的抽象设计比背 kernel 更重要。

- **关键词：多模态生成（Diffusion）工程化**
  - **一句话通俗解释**：视频/图像生成模型的部署比 LLM 更复杂——有任务队列、云存储、ComfyUI 集成、CUDA Graph 重放。
  - **为什么社区在关注**：今天一半的 commit 都是 diffusion 相关，说明 sglang 正在从"LLM 推理"扩展到"全模态推理"。
  - **对新手意味着什么**：如果你对 AIGC 部署感兴趣，现在正是入局的好时机，diffusion 的工程化程度远不如 LLM 成熟。

- **关键词：数值精度与内存布局（KV Cache 的物理存储）**
  - **一句话通俗解释**：KV Cache 在显存里怎么摆（HND 还是 NHD、BF16 还是 FP8），直接决定 kernel 快不快。
  - **为什么社区在关注**：#43237（BF16 融合写入 HND）、#43332（修复 CuTe DSL 的 state layout）都在处理这个底层问题。
  - **对新手意味着什么**：理解内存 layout 和 stride 是写高性能 kernel 的基本功，`stride()` 这个 API 值得吃透。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：调度器的"混合步骤"与 KV 池容量检查

- **是什么**：LLM 推理时，一个 step 里可以同时处理"新请求的 prefill"和"老请求的 decode"，这叫 mixed step。但如果 prefill 占用了太多 KV 空间，decode 就没地方放了。
- **为什么重要**：如果不检查，decode 时会直接 OOM 或崩溃。这在长上下文 + 高并发场景下非常常见。
- **动手试试**：
```python
# 模拟一个简化版的容量检查逻辑
class FakeKVPool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.used = 0
    def has_room_for(self, n_tokens):
        return self.used + n_tokens <= self.capacity

pool = FakeKVPool(capacity=100)
pool.used = 90  # prefill 已用 90

# 模拟 commit #43290 的逻辑
decode_rows = 20
if pool.has_room_for(decode_rows):
    print("decode 可以执行")
else:
    print("decode 被 hold back，等下一个 step")
```
- **预期结果**：输出"decode 被 hold back"。这解释了为什么 commit 里要加 `running_batch.check_decode_mem()` 这个判断。

### 知识点 2：torch.compile 的"图断裂"与自定义算子注册

- **是什么**：`torch.compile` 会把 Python 代码追踪成静态图，但如果遇到它不认识的操作（比如自定义 CUDA kernel），就会"断裂"回到 eager 模式，性能大幅下降。
- **为什么重要**：commit #43577 就是把一个 JIT kernel 用 `register_custom_op` 包成"不透明算子"，让编译器把它当成一个黑盒整体处理，避免断裂。
- **动手试试**：
```python
import torch

# 一个普通的函数
def my_add(x, y):
    return x + y

# 用 torch.compile 编译
compiled = torch.compile(my_add)
x = torch.randn(1000, 1000)
print(compiled(x, x).sum())

# 查看图断裂情况
import torch._dynamo as dynamo
dynamo.config.verbose = True
```
- **预期结果**：你会看到编译器打印出它追踪的图，如果用了 `torch.compiler.is_compiling()` 判断，就能看到哪些分支被跳过。

### 知识点 3：KV Cache 的内存布局与 stride

- **是什么**：KV Cache 在显存里的排列方式。HND 表示 [Head, Num_tokens, Dim]，NHD 表示 [Num_tokens, Head, Dim]。不同 kernel 对 layout 的偏好不同。
- **为什么重要**：commit #43237 让 kernel 支持任意 stride，这样就能直接写入 HND 布局的物理存储，避免额外的 reshape/copy。
- **动手试试**：
```python
import torch

# 创建一个 [num_blocks, block_size, num_heads, head_dim] 的 KV cache
cache = torch.randn(10, 16, 8, 64)
print("默认 stride:", cache.stride())  # (8192, 512, 64, 1)

# 转成 HND 视图
hnd_view = cache.permute(0, 2, 1, 3)
print("HND 视图 stride:", hnd_view.stride())  # (8192, 64, 512, 1)

# 关键：stride 告诉你相邻元素在内存里隔多远
# commit #43237 就是把这些 stride 传给 kernel，让它能直接写
```
- **预期结果**：你会看到 `stride()` 返回的元组，理解为什么 kernel 需要知道这些数字才能正确寻址。

### 知识点 4：多卡一致性——为什么不能用时间做判断

- **是什么**：在多张 GPU 上跑同一个模型时，每张卡上的调度逻辑必须完全一致，否则会出现"数据不一致"的 bug。
- **为什么重要**：commit #43479 把"每 0.5 秒刷新一次"改成"每 32 次调度刷新一次"，因为时间在不同卡上可能不同步，但调度次数是同步的。
- **动手试试**：
```python
# 反例：用时间做判断，不同机器可能不一致
import time
last_refresh = time.monotonic()
for i in range(100):
    now = time.monotonic()
    if now - last_refresh > 0.5:
        print(f"Step {i}: 刷新（基于时间）")
        last_refresh = now

# 正例：用计数器，跨机器一致
counter = 0
for i in range(100):
    if counter % 32 == 0:
        print(f"Step {i}: 刷新（基于计数）")
    counter += 1
```
- **预期结果**：时间版本在不同机器上刷新点不同，计数版本完全一致。

### 知识点 5：稀疏注意力的"打包"操作

- **是什么**：把分散在 KV 池里的 K/V 张量，按照请求的顺序"打包"成一个连续的张量，方便后续的 attention 计算。
- **为什么重要**：commit #41933 把"打包"和"dtype 转换"融合成一个 Triton kernel，减少了内存读写。
- **动手试试**：
```python
import torch

# 模拟：3 个请求，各自的 KV 分散在池子里
pool = torch.randn(100, 8, 64)  # [num_slots, num_heads, head_dim]
req_to_token = torch.tensor([
    [0, 5, 10],   # 请求 0 用的 slot
    [1, 6, 11],   # 请求 1
    [2, 7, 12],   # 请求 2
])

# 打包：按请求顺序 gather
packed = torch.stack([pool[req_to_token[i]] for i in range(3)])
print("打包后形状:", packed.shape)  # [3, 3, 8, 64]

# 融合版：一个 kernel 同时 gather + cast
packed_bf16 = packed.to(torch.bfloat16)
print("dtype:", packed_bf16.dtype)
```
- **预期结果**：理解 gather 操作怎么把"指针"（slot 索引）变成"连续数据"，以及为什么融合能省一次显存往返。

---

## 🔥 重要更新

1. **#43290 + #43298：调度器修复混合步骤的 KV 池容量问题**
   这是生产环境最容易踩的坑——prefill 和 decode 抢显存。修复后，装不下的 decode 请求会被"hold back"到下一个 step，而不是直接崩溃。

2. **#43237：BF16 融合写入原生 HND KV Cache**
   性能优化，省掉一次数据搬运。对长上下文推理的吞吐提升明显。

3. **#41933：QSA 稀疏注意力的 prefill KV 打包融合**
   把 gather + dtype 转换合成一个 Triton kernel，是"kernel fusion"的经典案例，值得学习。

4. **#43084：视频任务输出被删除后不再报告"完成"**
   生产级 bug 修复——如果云存储上传失败且没有持久化路径，任务应该报 failed 而不是 completed，否则用户下载会 404。

---

## 📋 逐条解读

### 1. [Scheduler] Hold back decode rows on a mixed step when they do not fit the KV pool (#43290)

- **代码层面**：修改 `scheduler.py` 的 `_get_new_batch_prefill_raw` 函数。原本只要 `running_batch` 非空就把它混进新 batch，现在多加一个条件 `running_batch.check_decode_mem()`——如果 decode 行装不下 KV 池，就不混进来。
- **新手概念课堂**：把 GPU 显存想象成停车场。prefill 是"大卡车"（一次占很多车位），decode 是"小轿车"（持续占车位）。如果大卡车进来后车位不够了，小轿车就得在外面等一下，而不是硬挤进去撞车。
- **对你有什么影响**：长上下文 + 高并发场景下，服务更稳定，不会因为一个混合步骤就 OOM 崩溃。

### 2. [KV canary] Expect sweep counters on the target runner only (#43286)

- **代码层面**：给 `KernelRunCounterHealthChecker` 加了一个 `is_draft_runner` 参数。draft runner（草稿模型）不会触发 sweep 操作，所以健康检查不应该期待它。
- **新手概念课堂**：健康检查像"考勤打卡"。draft runner 这个员工不上"夜班"（sweep），所以你不能因为它没打卡就认为它旷工。commit 就是给考勤系统加上"部门"字段。
- **对你有什么影响**：使用投机解码（speculative decoding）时，不会因为误报而重启服务。

### 3. [Test] Do not hang scripted-runtime teardown when the server died (#43285)

- **代码层面**：`http_server.py` 的 `shutdown` 方法，发送 `Shutdown` 消息时加了 `flags=zmq.NOBLOCK`。原来是阻塞发送，如果对端已死就会永远卡住。
- **新手概念课堂**：就像打电话，如果对方已经挂断，你还在"喂喂喂"等回应就会一直等。加 `NOBLOCK` 就是"响一声没人接就挂"。
- **对你有什么影响**：测试框架不会因为服务器崩溃而卡死，CI 更可靠。

### 4. [Scheduler] Requeue preempted requests when admission admits nothing (#43298)

- **代码层面**：`scheduler.py` 里，当 `can_run_list` 为空（没有任何请求被接纳）时，把 `preempt_list` 里的请求重新加回队列。之前这些请求被"抢占"释放了资源，但因为没有新请求进来，它们就"消失"了。
- **新手概念课堂**：像餐厅排队，你为了让 VIP 进来而暂时离席，结果 VIP 没来，你却回不到队伍里了。commit 就是让服务员把你重新排回去。
- **对你有什么影响**：高优先级请求调度时，低优先级请求不会被"饿死"。

### 5. [qwen 3.8 next] Fuse QSA prefill KV gathering and dtype conversion (#41933)

- **代码层面**：新增一个 Triton kernel `_pack_qsa_prefill_kv`，把原来两步（gather K/V + dtype 转换）合成一步。用二分查找在 `cu_k` 里定位每个 token 属于哪个请求。
- **新手概念课堂**：像"打包快递"——原来要先把货物从仓库搬到打包台，再换箱子；现在直接在仓库里换好箱子搬走，少一趟。
- **对你有什么影响**：Qwen 稀疏注意力模型的 prefill 阶段会更快，尤其是长 prompt。

### 6. [Spec] Allow Qwen3.5 backends to opt into packed target verification (#43037)

- **代码层面**：`qwen3_5.py` 里，判断是否使用 fused proj conv 时，从"仅 decode 模式"扩展到"decode 或 packed verify 模式"。同时重命名 `use_fused_decode_proj_conv` → `use_fused_proj_conv`。
- **新手概念课堂**：投机解码时，target 模型要"验证"draft 模型猜的多个 token。这个 commit 让验证阶段也能用上融合 kernel。
- **对你有什么影响**：Qwen3.5 + 投机解码的吞吐会提升。

### 7. [AMD]Qwen add Sparse attention to capturing hip graphs (#43383)

- **代码层面**：`eagle_worker_v2.py` 里，把 `QwenSparseAttnBackend` 加入支持 CUDA Graph（AMD 上是 HIP Graph）捕获的后端列表。
- **新手概念课堂**：CUDA Graph 像"录制宏"——把一串 GPU 操作录下来，下次直接重放，省去 CPU 调度开销。但不是所有后端都能被录制，这个 commit 就是让 Qwen 稀疏注意力也能被录。
- **对你有什么影响**：在 AMD GPU 上跑 Qwen 稀疏注意力会更快。

### 8. [KDA] Support Helion MTP verification and harden prefill, decode (#42857)

- **代码层面**：新增 `kda_verify.py`，用 Helion（一个 kernel DSL）实现 KDA 模型的 MTP（Multi-Token Prediction）验证。支持 per-candidate 的 state snapshot。
- **新手概念课堂**：MTP 像"一次猜多个词"，每个猜测都要独立验证。这个 kernel 让验证过程能在 GPU 上高效并行。
- **对你有什么影响**：KDA 模型的投机解码会更快。

### 9. [GDN] Fix CuTe DSL decode to read the [N, HV, V, K] state layout (#43332)

- **代码层面**：`cutedsl_gdn.py` 里，把 `k_write = flat_idx // TILE_V_SMALL` 改成 `k_write = flat_idx % TILE_K`。这是修一个索引计算 bug——原来的写法假设了错误的 layout。
- **新手概念课堂**：像 Excel 表格，原来按"行优先"填数据，但实际表格是"列优先"的，填进去就错位了。commit 就是修正填充顺序。
- **对你有什么影响**：使用 GDN 模型的 decode 结果会正确（之前可能有静默错误）。

### 10. [diffusion] Don't report video jobs completed after their outputs are deleted (#43084)

- **代码层面**：`video_api.py` 里，先尝试上传所有视频到云存储，如果上传失败且没有持久化路径，直接抛错让任务标记为 failed。同时 `VideoResponse` 加了 `urls` 字段支持多输出。
- **新手概念课堂**：像外卖平台——如果餐没送到，不能显示"已送达"。之前代码"假装成功"，用户去下载就 404。
- **对你有什么影响**：视频生成任务的 API 返回更诚实，客户端能正确处理失败。

### 11. [Perf] Fuse BF16 writes to native HND KV cache storage (#43237)

- **代码层面**：`cache_ops.py` 的 `reshape_and_cache_flash` kernel 加了 6 个 stride 参数（key/value 各自的 token/head stride）。这样 kernel 能直接写入任意 layout 的物理存储，不用先转成 NHD。
- **新手概念课堂**：像搬家——原来必须把所有家具摆成一种姿势才能装车，现在告诉搬家公司每件家具的尺寸和朝向，他们能直接装。
- **对你有什么影响**：使用 HND layout 的模型（很多新模型默认）KV 写入更快。

### 12. [Scheduler] Keep waiting-prefix LRU refresh consistent across TP ranks (#43479)

- **代码层面**：`schedule_policy.py` 里，把 `WAITING_PREFIX_REFRESH_INTERVAL_S = 0.5`（秒）改成 `WAITING_PREFIX_REFRESH_INTERVAL = 32`（调度次数）。用计数器代替时间戳。
- **新手概念课堂**：多卡同步时，"每 0.5 秒"在不同卡上可能因为时钟漂移而不一致，但"每 32 次调度"是确定性的。
- **对你有什么影响**：张量并行（TP）场景下的前缀缓存行为更一致，避免某些卡上缓存被意外驱逐。

### 13. [diffusion] comfyui: consolidate executor conditioning-cache fixes (#43492)

- **代码层面**：`adapter.py` 和 `base.py` 里，把 `cached_extra_keys` 抽成类属性，并新增 `_hash_value` 函数对 conditioning 做递归哈希。这样缓存 key 更准确，避免"命中错误缓存"。
- **新手概念课堂**：像给文件做"指纹"——两个文件内容不同但名字相同，用指纹就能区分。commit 就是给 conditioning 数据做指纹。
- **对你有什么影响**：ComfyUI 集成时，不同 prompt 不会错误复用缓存，生成结果更准确。

### 14. [diffusion] comfyui: consolidate FluxAdapter correctness fixes (#43490)

- **代码层面**：`flux.py` 里，`pack` 方法加了 `ref_latents` 和 `control` 参数检查（不支持就报错），`y=None` 时零填充，`_pack_latents` 返回 padded 尺寸。
- **新手概念课堂**：像"点菜"——如果顾客点了菜单上没有的（Kontext 参考图），服务员应该明确说"没有"，而不是默默上一道别的菜。
- **对你有什么影响**：用 ComfyUI 跑 Flux 工作流时，不支持的节点会明确报错，而不是静默产生错误结果。

### 15. [1/N][DSv41] Enable DP attention with MegaMoE (#43228)

- **代码层面**：`mega_moe.py` 里，`check_mega_moe_shapes` 放宽了 `intermediate` 的对齐要求（从 512 到 128），并新增 `vision_topk` 分支处理 V4.1 的视觉路由。
- **新手概念课堂**：MoE 像"专家会诊"——每个 token 选几个专家处理。MegaMoE 是把多个专家融合成一个大 kernel。这个 commit 让 DeepSeek V4.1 的视觉 token 也能正确路由。
- **对你有什么影响**：DeepSeek V4.1 多模态模型在 DP attention 下能正常工作。

### 16. [diffusion] perf: run the attention section eager under torch.compile for MiniMax-H3 (#43634)

- **代码层面**：`minimax_h3.py` 里，把 attention 部分抽成 `_attend` 方法，用 `torch.compiler.disable` 装饰成 `_attend_eager`。编译时走 eager，未编译时走原路径。
- **新手概念课堂**：像"录视频"——有些场景（比如快速运动）录下来会糊，不如现场直播。attention 里的融合 kernel 就是这种"录不下来"的场景。
- **对你有什么影响**：MiniMax-H3 的 torch.compile 优化能真正生效，而不是因为图断裂而变慢。

### 17. [diffusion] fix: replay Qwen-Image-2.1 BCG graphs across prompt lengths (#43633)

- **代码层面**：新增 `qwen_image21.py` padder，把不同长度的 prompt K/V 左填充到统一的 bucket 长度，这样不同 prompt 长度能共享同一个 CUDA Graph 签名。
- **新手概念课堂**：像"停车位标准化"——不管你的车多长，都停在同样大的车位里，停车场就能统一规划。
- **对你有什么影响**：Qwen-Image-2.1 在 ComfyUI 里用不同长度 prompt 时，CUDA Graph 能正确重放，不会因为长度不同而失效。

### 18. [diffusion] fix: Qwen-Image BCG prompt padding misses every graph and attends the pad (#43632)

- **代码层面**：`qwen_image.py` 里，把 `freqs_cis` 的 padding 扩展到 `freqs_complex`，并把 `txt_seq_lens` 设为 `None`（让 mask 单独决定有效长度）。
- **新手概念课堂**：像"考试"——之前用"答题卡长度"判断学生答了多少题，现在改成看"哪些格子被填了"。更准确。
- **对你有什么影响**：Qwen-Image 的 CUDA Graph padding 不再"attend pad"（把填充数据也当真实数据算），生成质量更稳定。

### 19. [diffusion] perf: keep Cosmos3's fused kernels in the graph (#43577)

- **代码层面**：`layernorm.py` 里，用 `register_custom_op` 把 `fused_inplace_qknorm` 注册成不透明算子。同时移除 Cosmos3 里多处 `not torch.compiler.is_compiling()` 判断。
- **新手概念课堂**：像"黑盒"——告诉编译器"这个函数你别管内部，当成一个整体就行"，编译器就不会因为它内部复杂而放弃优化。
- **对你有什么影响**：Cosmos3 视频生成模型的编译优化更彻底，推理更快。

### 20. [diffusion] perf: keep Qwen-Image-2.1 and Ulysses graphs whole (#43586)

- **代码层面**：`layer.py`、`usp.py`、`qwen_image21.py` 里，多处加 `not torch.compiler.is_compiling()` 判断，让编译时走可追踪的 NCCL all-to-all，而不是不可追踪的 IPC 传输。
- **新手概念课堂**：像"换交通工具"——编译图里只能坐"公交车"（NCCL），不能坐"私家车"（IPC），因为私家车没法被"录制"。
- **对你有什么影响**：Qwen-Image-2.1 和 Ulysses 并行在 torch.compile 下能保持完整图，性能提升。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)