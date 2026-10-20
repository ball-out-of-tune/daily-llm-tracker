# 🚀 vLLM & SGLang 每日更新 — 2026-10-20

> 自动生成于 2026-10-20 06:33 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：元数据复用（Metadata Reuse）**
  - **一句话通俗解释**：训练端和推理端同步权重时，参数清单（名字、形状、类型）只发一次，之后每轮发个空包就行，省掉大量重复传输。
  - **为什么社区在关注**：RL 训练场景下，一个 30B MoE 模型有近 19000 个参数，每轮同步都重发清单是纯浪费。这说明社区开始认真优化"训练-推理协同"这种长链路场景的细粒度开销。
  - **对新手意味着什么**：如果你在做 RLHF/RLVR 训练框架，这类"协议层优化"往往是性价比最高的性能提升点，值得深入理解通信协议设计。

- **关键词：可插拔 KV Cache 数据类型（Pluggable KV Cache Dtypes）**
  - **一句话通俗解释**：以前 `--kv-cache-dtype` 只能选固定几种（fp8、int8 等），现在可以通过注册机制自定义新的 KV 缓存数据类型。
  - **为什么社区在关注**：量化是推理降本的核心手段，KV Cache 占显存大头。把 dtype 做成注册表意味着硬件厂商/研究团队可以自己接入新的量化格式，不用改核心代码。
  - **对新手意味着什么**：这是"开放扩展点"设计的典型案例，学习如何设计注册表模式（registry pattern）对做框架开发很有帮助。

- **关键词：异构硬件适配（AMD ROCm / Intel XPU / ARM）**
  - **一句话通俗解释**：今天的 commits 里同时出现了 MI300X/MI325X（AMD）、XPU（Intel）、aarch64（ARM）的适配和测试修复。
  - **为什么社区在关注**：vLLM 不再只是"NVIDIA 推理引擎"，正在变成跨硬件平台。DeepSeek-V4.1 在 gfx942 上的 mono decode 内核、NIXL 在 XPU 上的 block size 兼容，都是实打实的平台工程。
  - **对新手意味着什么**：如果你在学异构计算或想进 AMD/Intel 的 AI 软件栈团队，vLLM 的多平台适配代码是最好的学习材料。

- **关键词：投机解码的采样策略（Speculative Decoding Sampling）**
  - **一句话通俗解释**：DSpark 投机解码从 `greedy`（贪心）改成 `probabilistic`（概率采样），生成质量更接近原始模型分布。
  - **为什么社区在关注**：投机解码的"草稿模型"如果用贪心采样，接受率可能高但分布有偏；概率采样能保证输出分布正确性。这是推理正确性 vs 速度的经典权衡。
  - **对新手意味着什么**：投机解码是当前推理加速最热的方向之一，理解 draft sampling 和 rejection sampling 的区别是入门必修课。

- **关键词：睡眠模式状态恢复（Sleep Mode State Restoration）**
  - **一句话通俗解释**：GPU 进入 level-2 深度睡眠后显存内容被清空，唤醒时必须把 LoRA 相关的索引映射等"非权重状态"也恢复回来，否则推理结果错误。
  - **为什么社区在关注**：RL 训练中频繁 sleep/wake 是常态（训练和推理抢 GPU），这类"唤醒后状态不一致"的 bug 非常隐蔽，测试很难覆盖。
  - **对新手意味着什么**：状态管理是分布式系统的永恒难题，学会区分"可重建状态"和"必须保存状态"是重要工程能力。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：注册表模式（Registry Pattern）——以 KV Cache dtype 为例**
  - **是什么**：用一个装饰器 `@register_kv_cache_dtype("test-c8")` 把自定义类注册到一个全局字典里，运行时根据字符串名字查表创建实例。就像给手机装 App，系统不需要预装所有 App，只要有一个"应用商店"登记表。
  - **为什么重要**：框架如果硬编码所有支持的类型，每加一个新格式都要改核心代码、重新测试全量。注册表让扩展变成"插件"，第三方可以独立开发。不懂这个模式，你读 vLLM 代码时会到处看到 `@register_xxx` 却不知道在干嘛。
  - **动手试试**：
    ```python
    # 最小注册表模式演示
    REGISTRY = {}

    def register(name):
        def decorator(cls):
            REGISTRY[name] = cls
            return cls
        return decorator

    @register("fp8")
    class FP8Handler:
        def torch_dtype(self):
            return "float8_e4m3fn"

    @register("my-custom")
    class MyHandler:
        def torch_dtype(self):
            return "int4"

    # 运行时按名字创建
    name = "my-custom"
    handler = REGISTRY[name]()
    print(handler.torch_dtype())  # 输出: int4
    print(list(REGISTRY.keys()))  # 输出: ['fp8', 'my-custom']
    ```
  - **预期结果**：你能看到注册表里有两个条目，按名字能取到对应类。然后去读 `vllm/config/cache.py` 里的 `register_kv_cache_dtype`，会发现结构几乎一样。

- **知识点 2：NCCL 权重传输的"清单一次发"协议**
  - **是什么**：RL 训练时，训练进程把权重广播给推理进程，需要先告诉对方"我要发哪些参数、什么形状、什么类型"。这个清单（metadata）很大，但每轮都一样。优化后：第一轮发完整清单，后续发空字典表示"沿用上次"。
  - **为什么重要**：如果你自己写分布式训练-推理同步代码，每轮重发 metadata 会浪费大量 CPU 序列化和网络带宽。更关键的是——**如果参数变了但你没重发清单，NCCL 会挂死或加载错误权重**，这是极难调试的 bug。
  - **动手试试**：
    ```python
    # 模拟 metadata 复用逻辑（纯 Python，无需 GPU）
    class WeightSyncWorker:
        def __init__(self):
            self._saved_metadata = None

        def update_weights(self, update_info):
            names = update_info.get("names", [])
            if names:  # 非空：更新并保存
                self._saved_metadata = update_info
                print(f"[worker] 收到完整清单，共 {len(names)} 个参数")
            elif self._saved_metadata is not None:
                print(f"[worker] 空清单，复用上次 {len(self._saved_metadata['names'])} 个参数")
            else:
                print("[worker] 空清单且无缓存，什么也不做")

    w = WeightSyncWorker()
    w.update_weights({"names": ["a", "b", "c"], "shapes": [[4], [8], [16]]})
    w.update_weights({})  # 复用
    w.update_weights({"names": ["a", "b"], "shapes": [[4], [8]]})  # 参数变了，重发
    w.update_weights({})  # 复用新的
    ```
  - **预期结果**：输出依次显示"完整清单 3 个"→"复用 3 个"→"完整清单 2 个"→"复用 2 个"。理解后去看 commit 1 的 `docs/training/weight_transfer/nccl.md` 表格，会一目了然。

- **知识点 3：`replace_parameter` 保持对象身份——为什么 PyTorch Parameter 对象不能随便换**
  - **是什么**：量化后需要把原来的 `nn.Parameter` 替换成量化后的 tensor。旧实现会创建一个新 Parameter 对象；新实现默认**原地更新原 Parameter 的 `.data`**，保持对象身份不变。
  - **为什么重要**：CUDA Graph 捕获时记录的是**内存地址**和**对象引用**。如果你换了一个新 Parameter 对象，之前捕获的 graph 可能指向旧地址，导致推理结果错误或崩溃。这是"性能优化代码"和"正确性"的经典冲突。
  - **动手试试**：
    ```python
    import torch

    layer = torch.nn.Linear(4, 4)
    original = layer.weight
    print(f"原对象 id: {id(original)}, data_ptr: {original.data_ptr()}")

    # 模拟"原地替换"：保持对象，只换数据
    new_data = torch.ones(4, 4)
    original.data = new_data
    print(f"替换后 id: {id(layer.weight)}, data_ptr: {layer.weight.data_ptr()}")
    print(f"对象是否相同: {layer.weight is original}")  # True

    # 模拟"新建对象"：会破坏引用
    layer.weight = torch.nn.Parameter(torch.zeros(4, 4))
    print(f"新建后对象是否相同: {layer.weight is original}")  # False
    ```
  - **预期结果**：第一段 `id` 相同、`is` 为 True；第二段 `is` 为 False。去读 commit 3 的测试 `test_replace_parameter_keeps_object_by_default`，会看到它断言 `layer.weight is original`。

- **知识点 4：投机解码的贪心 vs 概率采样**
  - **是什么**：投机解码先用小模型（draft）快速生成 K 个候选 token，再用大模型验证。draft 阶段如果永远选概率最高的 token（greedy），候选质量高但分布有偏；如果按概率分布采样（probabilistic），候选更"多样"，验证阶段通过拒绝采样修正，最终输出分布严格等于大模型分布。
  - **为什么重要**：greedy draft 在 temperature > 0 的场景下会破坏输出多样性，用户会发现"模型变得死板"。概率采样 + 拒绝采样是保证**数学上无损**加速的关键。
  - **动手试试**：
    ```python
    import torch

    logits = torch.tensor([2.0, 1.0, 0.5, 0.1])
    probs = torch.softmax(logits, dim=-1)
    print("概率分布:", probs)

    # greedy：永远选最大
    print("greedy 选择:", logits.argmax().item())

    # probabilistic：按分布采样
    torch.manual_seed(0)
    samples = [torch.multinomial(probs, 1).item() for _ in range(20)]
    print("probabilistic 20 次采样:", samples)
    ```
  - **预期结果**：greedy 永远是 0；probabilistic 大部分是 0 但偶尔出现 1、2。这解释了为什么 commit 4 把 `draft_sample_method` 从 `greedy` 改成 `probabilistic`——为了输出分布正确性。

- **知识点 5：睡眠模式下的状态分类——哪些必须保存，哪些可以重建**
  - **是什么**：GPU level-2 睡眠会释放显存，唤醒后权重从磁盘重新加载。但有些"非权重"状态（比如 LoRA 的 `sharded_to_full_mapping_gpu` 索引映射）也是运行时分配的，如果不恢复就会出错。
  - **为什么重要**：分布式系统里状态分三类：①可从持久化存储重建（权重）②可从配置重建（buffer 大小）③**必须显式保存/恢复**（运行时计算的索引、随机种子状态）。第三类最容易漏，漏了就是隐蔽 bug。
  - **动手试试**：
    ```python
    # 模拟一个"睡眠/唤醒"状态机
    class ModelState:
        def __init__(self):
            self.weights = torch.ones(4)          # 可从磁盘重建
            self.lora_mapping = torch.arange(4)   # 运行时计算，必须保存
            self._saved = {}

        def sleep(self, level=2):
            if level == 2:
                self._saved["lora_mapping"] = self.lora_mapping.clone()
                self.lora_mapping = None  # 模拟显存被释放
                self.weights = None

        def wake_up(self, tags):
            if "weights" in tags:
                self.weights = torch.ones(4)  # 从磁盘重建
            if "lora" in tags and "lora_mapping" in self._saved:
                self.lora_mapping = self._saved["lora_mapping"]  # 恢复
                del self._saved["lora_mapping"]

    m = ModelState()
    m.sleep(level=2)
    print("睡眠后 lora_mapping:", m.lora_mapping)  # None
    m.wake_up(tags=["weights", "lora"])
    print("唤醒后 lora_mapping:", m.lora_mapping)  # tensor([0,1,2,3])
    ```
  - **预期结果**：睡眠后映射为 None，唤醒后恢复。对照 commit 2 的测试 `_lora_logits_mapping_correct`，它就是在验证唤醒后映射内容正确。

---

## 🔥 重要更新

1. **可插拔 KV Cache 数据类型（#52670）**：`--kv-cache-dtype` 从硬编码枚举变成注册表，第三方可以自定义量化格式。这是量化生态开放的关键一步，影响所有做 KV Cache 压缩的团队。

2. **`replace_parameter` 默认保持对象身份（#62326）**：修复了量化后 Parameter 对象被替换导致 CUDA Graph 失效的隐患，涉及 788 行改动，影响所有量化路径的稳定性。

3. **NCCL 权重元数据只发一次（#35911）**：RL 训练场景下每轮同步省掉近 19000 条参数清单的传输，是训练-推理协同的性能优化，同时文档明确了"参数变更必须重发清单"的契约。

4. **run-batch 响应增量写入（#52505）**：批处理任务不再等全部完成才写文件，中途失败也能保留已完成部分，支持断点续跑。对跑大规模离线推理的用户是实用改进。

---

## 📋 逐条解读

### 1. [RL] Send NCCL weight metadata once and reuse it (#35911)
- **代码层面**：改的是 `docs/training/weight_transfer/nccl.md` 文档和 `tests/distributed/test_weight_transfer.py` 测试。核心逻辑在 `NCCLTrainerWeightTransferEngine.send_weights`：它比较当前 `metadata()` 和上次成功发送的清单，相同则发空列表，worker 端收到空列表就复用上次保存的清单。文档新增了一张三行表格说明"发什么 → 加载什么"。
- **新手概念课堂**：就像快递员送货，第一次给你一张"货物清单"（名字、尺寸、类型），后面每次送同样的货，你只要说"跟上次一样"就行，不用再念一遍 19000 个名字。但如果货物变了（加了一件、改了尺寸），必须重新给清单，否则快递员按旧清单装车会装错。
- **对你有什么影响**：普通用户无感知。做 RL 训练的用户如果自己构造 `update_info`（比如通过 HTTP 直连），需要遵守新契约：参数不变时发 `{}`，变了必须发完整清单，否则 NCCL 会挂死。

### 2. [Bugfix][LoRA] Restore LoRA state on wake-up from level-2 sleep (#62358)
- **代码层面**：改的是 `vllm/v1/worker/lora_model_runner_mixin.py`，引入 `get_mem_allocator_instance` 和 `nullcontext`，在睡眠/唤醒流程中保存和恢复 LoRA 的 `sharded_to_full_mapping_gpu` 等状态。测试 `test_sleep_mode.py` 把检查函数从"映射存在"改成"映射内容正确"（逐元素比对）。
- **新手概念课堂**：深度睡眠（level-2）像把电脑内存条拔了，唤醒后要重新插回去。权重可以从硬盘（磁盘）重新加载，但 LoRA 的索引映射是运行时算出来的，硬盘上没有，必须提前备份到"安全区"（保存的 buffer），唤醒再拷回来。
- **对你有什么影响**：用 LoRA + 睡眠模式的用户，唤醒后推理结果不再出现乱码或错误。之前这个 bug 在 TP>1 时才触发，单卡用户可能没遇到。

### 3. [Quantization] replace_parameter keeps the parameter object by default (#62326)
- **代码层面**：改的是 `vllm/model_executor/utils.py` 的 `replace_parameter` 函数，以及 `tests/model_executor/test_utils.py`。旧版有 `prefer_copy` 参数决定是否复用存储；新版默认**原地更新原 Parameter 的 `.data`**，保持对象身份，删掉了 `prefer_copy` 参数。测试新增了"同布局拷贝、新 stride、新 shape、新 dtype"四种场景。
- **新手概念课堂**：Parameter 对象就像一个人的身份证。CUDA Graph 捕获时记的是"身份证号"（对象引用）和"住址"（内存地址）。如果你换了个新身份证（新 Parameter 对象），graph 还拿着旧身份证去找人，就找不到了。原地更新 `.data` 相当于"同一个人搬了家但身份证不变"，graph 通过身份证能找到新住址。
- **对你有什么影响**：量化模型（FP8、INT8 等）的推理稳定性提升，尤其是开了 CUDA Graph 的场景。如果你自己写量化 kernel 并调用 `replace_parameter`，注意它现在默认保持对象身份，行为变了。

### 4. [Docs] Use probabilistic DSpark sampling in Kimi-K3 GB300 recipes (#62610)
- **代码层面**：改的是 `recipes/models/moonshotai/Kimi-K3.yaml`，把 `--speculative-config` 里的 `"draft_sample_method":"greedy"` 改成 `"probabilistic"`。一行 YAML 改动。
- **新手概念课堂**：投机解码像"实习生先写草稿，老板再审核"。实习生如果永远写"最可能的答案"（greedy），草稿通过率高但答案死板；如果按概率写（probabilistic），草稿更多样，老板审核时用拒绝采样修正，最终输出和老板自己写的一样。greedy 在 temperature>0 时会破坏多样性。
- **对你有什么影响**：用 Kimi-K3 GB300 配方跑投机解码的用户，输出质量更接近原始模型分布，不会因为投机解码变得"过度确定"。

### 5. [Bugfix][NIXL] Use XPU-compatible block size in sleep test (#62578)
- **代码层面**：改的是 `tests/v1/kv_connector/unit/test_nixl_connector.py`，把硬编码的 `block_size=16` 换成 `maybe_update_block_size(16)`，让测试在 Intel XPU 平台上用兼容的 block size。
- **新手概念课堂**：不同硬件对内存对齐的要求不同。NVIDIA GPU 可能喜欢 16 的 block size，Intel XPU 可能需要 32 或别的值。测试代码硬编码 16 在 XPU 上会失败，用 `maybe_update_block_size` 让平台自己决定。
- **对你有什么影响**：普通用户无感知。在 Intel XPU 上跑 NIXL KV Connector 测试的开发者不再遇到假失败。

### 6. [Bugfix][KV Connector][Mooncake] Wait for all peers before completing failed receives (#62282)
- **代码层面**：改的是 Mooncake connector 的接收逻辑和 `tests/v1/kv_connector/unit/test_mooncake_connector.py`。核心修复：当某个 peer 失败时，不能立即把请求标记为完成，必须等所有 peer 都回复（成功或失败），否则 KV block 可能被提前释放并被其他请求覆写（write-after-free）。新增两个测试验证"部分/重复回复不释放他人目标"和"过期 peer 的 finish 不搁置待处理接收"。
- **新手概念课堂**：就像团队协作，A 说"我做完了"，B 还没说话。你不能因为 A 完成就把整个项目的资源释放掉，因为 B 可能还在用。必须等所有人都表态（完成或失败），才能统一释放。
- **对你有什么影响**：用 Mooncake KV Connector 做 PD 分离的用户，失败场景下不再出现 KV block 被覆写导致的输出错误。

### 7. [Doc] Add Kimi-K3 GB300 Dynamo disaggregated profile to the recipe selector (#62563)
- **代码层面**：改的是 `recipes/models/moonshotai/Kimi-K3.yaml`，新增 `gb300` 硬件配置块，包含 TP8 + DCP8、DSpark K4 draft、Mooncake/NIXL 连接器、大量环境变量（`VLLM_USE_DIRECT_DCP_A2A`、`NCCL_MNNVL_ENABLE` 等）和 prefill 节点的 `deep_gemm_mega_moe` 后端配置。
- **新手概念课堂**：这是"配方"（recipe），像菜谱一样告诉你在这个硬件上跑这个模型要加什么参数。GB300 是 NVIDIA 的新一代机架，有 MNNVL（多节点 NVLink）互连，所以环境变量里一堆 `NCCL_MNNVL_ENABLE`、`NCCL_NVLS_ENABLE`。
- **对你有什么影响**：有 GB300 硬件的用户可以直接抄配方，不用自己调参数。没硬件的用户也能从中学到大规模部署的配置思路。

### 8. [Frontend] Write run-batch responses incrementally instead of at the end (#52505)
- **代码层面**：改的是 `vllm/entrypoints/launchers/run_batch.py` 和 `tests/entrypoints/launchers/test_run_batch.py`（1257 行测试改动）。新增 `batch_output_writer`、`BatchProgressTracker` 等组件，响应按输入顺序实时写入输出文件，不再等全部完成。文档说明"输出文件在开始时被截断，失败时保留已完成部分，上传仅在成功时发生"。
- **新手概念课堂**：以前是"攒够一筐鸡蛋再卖"，现在是"下一个蛋卖一个"。好处是中途失败不白干，坏处是失败时文件不完整，需要检查退出码。
- **对你有什么影响**：跑大规模离线批处理的用户，进程被杀后不用从头再来，可以看输出文件最后一行是哪个请求，从下一行续跑。

### 9. [Perf][Qwen4Exp] Internal prefill checkpoints for the PLE short conv (#61601)
- **代码层面**：新增 `vllm/model_executor/layers/mamba/checkpoint.py`（`MambaPrefillCheckpointExporter`），以及 `tests/kernels/mamba/test_checkpoint_exporter.py`（174 行测试）。为 Qwen4Exp 的 PLE（Per-Layer Embedding）短卷积在 prefill 阶段保存内部 checkpoint，供投机解码的 draft 模型复用。
- **新手概念课堂**：prefill 阶段要处理整个 prompt，计算量大。如果 draft 模型需要中间层的某些状态（比如短卷积的中间结果），与其重算不如存下来（checkpoint）给 draft 用。这是"用显存换计算"的经典权衡。
- **对你有什么影响**：用 Qwen4Exp + 投机解码的用户，prefill 阶段可能更快。普通用户无感知。

### 10. [Feature][Quantization] Gather MiMo fused-QKV FP8 rows exactly by default (#59053)
- **代码层面**：改的是 `vllm/model_executor/models/mimo_v2.py` 的 `_shard_fp8_qkv_proj`，以及 `tests/models/quantization/test_mimo_v2_qkv_shard.py`。新增 `_fused_qkv_kv_chunk_rows`、`_split_qkv` 辅助函数。核心变化：FP8 QKV 权重按行 gather 时默认走"精确 gather"路径（纯行置换，bit 级一致），而不是之前的近似路径。
- **新手概念课堂**：FP8 量化权重按行切分给不同 TP rank 时，如果切分边界和量化 block 边界不对齐，需要"重新 gather"行。精确 gather 保证结果和 checkpoint 逐 bit 一致，不会因为重新量化引入误差。
- **对你有什么影响**：用 MiMo-V2 模型 + FP8 量化的用户，TP 切分后输出精度更高（和单卡一致）。

### 11. [Misc] Fix pre-commit failure on main from conflicts between pluggable KV cache and Step5 (#62593)
- **代码层面**：改的是 `vllm/models/step5/nvidia/sparse_attention.py`，把 `is_quantized_kv_cache` 的 import 从 `vllm.utils.torch_utils` 移到 `vllm.v1.kv_cache_interface`（配合 commit 13 的迁移），并给 `alibi_slopes` 类型加 `list[float]`、给 `dtype` 加 `cast`。
- **新手概念课堂**：这是两个 PR 合并冲突的"善后"。commit 13 把函数挪了位置，commit 11 是修因挪位置导致的 import 失败。大型项目里这种"合并后修复"很常见。
- **对你有什么影响**：无感知，纯 CI 修复。

### 12. [Bugfix][DCP] Support strided LSE in direct A2A reduction (#60373)
- **代码层面**：改的是 `csrc/attention/dcp_utils/dcp_direct_a2a_lse_reduce.cu` CUDA kernel，新增 `lse_head_stride` 参数，允许 LSE tensor 的 head 维度有 stride（非连续）。删掉了"LSE 必须 packed heads"的断言。测试新增 `lse_layout` 函数模拟 FlashAttention MLA 返回的 `[H,T]` 转置布局。
- **新手概念课堂**：tensor 在内存里可以是"连续"的（元素挨着元素），也可以是"带 stride"的（每隔几个位置取一个）。以前 kernel 假设 LSE 是连续的，但 FlashAttention MLA 返回的是转置视图，不连续。新 kernel 支持 stride，就不用先拷贝成连续的了。
- **对你有什么影响**：用 DCP（Decode Context Parallel）+ FlashAttention MLA 的用户，LSE 归约不再需要额外的拷贝开销，且支持更多 attention backend 的布局。

### 13. [Quantization][Feature] Pluggable KV Cache Data Types (#52670)
- **代码层面**：改的是 `vllm/config/cache.py`（新增 `register_kv_cache_dtype`、`KV_CACHE_DTYPES`、`get_kv_cache_dtype_handler`）和 `vllm/v1/kv_cache_interface.py`（新增 `KVQuantMode`、`get_kv_quant_mode`）。`is_quantized_kv_cache` 从 `torch_utils` 迁移到 `kv_cache_interface`。新增 `tests/config/test_kv_cache_dtype.py`（144 行测试）演示如何注册自定义 dtype。
- **新手概念课堂**：以前 `--kv-cache-dtype` 是 if-else 硬编码的，加一个新格式要改核心代码。现在用注册表，第三方写个类 + 装饰器就接入了。就像手机从"预装应用"变成"应用商店"。
- **对你有什么影响**：做 KV Cache 量化的研究团队可以自己接入新格式（比如 FP4、NF4），不用 fork vLLM。普通用户还是用原来的 fp8/int8，行为不变。

### 14. [CI/Build] Drop CUDA 12.9 from the release pipeline (#62575)
- **代码层面**：改的是 `.buildkite/release-pipeline.yaml`，删掉所有 CUDA 12.9 的 wheel 构建和镜像构建步骤（x86_64 和 aarch64），以及 `CUDA_ARCH_X86_CU129`、`CUDA_ARCH_AARCH64_CU129` 环境变量。
- **新手概念课堂**：CUDA 版本像操作系统版本，12.9 是过渡版本，社区决定统一到 CUDA 13。就像 Windows 停止支持 Win7，大家都升 Win10。
- **对你有什么影响**：如果你还在用 CUDA 12.9 环境，需要升级到 CUDA 13 才能用最新的 vLLM release wheel。或者从源码编译。

### 15. [ROCm][Perf][DSv4.1] Mono decode on MI300X and MI325X (gfx942) (#60731)
- **代码层面**：新增大量 AMD gfx942 专用内核和测试（`tests/kernels/models/amd/test_dsv41_gfx942_*.py`，9510 行 diff），涉及 `vllm/models/deepseek_v41/amd/`、`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`、`vllm/v1/worker/gpu/spec_decode/dspark/`。Buildkite 新增 MI300 单卡和 4 卡测试步骤。
- **新手概念课堂**：DeepSeek-V4.1 的 mono decode（单请求解码）在 AMD MI300X 上需要专门的 kernel 优化（topk、candidate logits、sparse indexer、fused markov 等）。"mono"指 batch size=1 的延迟敏感场景。
- **对你有什么影响**：用 AMD MI300X/MI325X 跑 DeepSeek-V4.1 的用户，单请求解码延迟降低。NVIDIA 用户无感知。

### 16. [Chore] Port more vendored configs and processors to upstream Transformers (#60823)
- **代码层面**：改的是 `tests/models/registry.py`、`tests/config/base_model_arch_groundtruth.json` 等（4367 行 diff），把 vLLM 里 vendored（内嵌）的模型配置和 processor 代码迁移到上游 HuggingFace Transformers，删掉对应的 `trust_remote_code=True` 和版本限制。
- **新手概念课堂**：vLLM 以前为了支持新模型，会把模型的 config 代码"抄"一份到自己仓库（vendored）。现在这些代码被上游 Transformers 接收了，vLLM 就可以删掉自己的副本，直接用上游的。就像你以前自己维护一个工具库的 fork，现在上游合并了你的 PR，你就可以删 fork 了。
- **对你有什么影响**：加载这些模型（Cohere2Moe、Cosmos3Edge、FireRedASR2 等）不再需要 `trust_remote_code=True`，更安全、更少版本冲突。

### 17. [Doc] Document DeepSeek-V4.1-Flash B300 Dynamo disaggregated profile (#62561)
- **代码层面**：改的是 `recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`，新增 B300/GB300 上 Dynamo 分离式部署的详细文档（约 80 行），包括 prefill/decode 引擎数量、`--max-num-seqs`、NIXL 后端（LIBFABRIC vs UCX）、Engram 配置、CUDA Graph 捕获尺寸等。
- **新手概念课堂**：Dynamo 是 NVIDIA 的推理调度框架，分离式部署（disaggregated）指 prefill 和 decode 跑在不同 GPU 池上，中间用 NIXL 传 KV Cache。B300 用 EFA（AWS 的网络）走 LIBFABRIC，GB300 用 MNNVL 走 UCX。
- **对你有什么影响**：有 B300/GB300 的用户可以直接抄配置。学习者能从中看到大规模部署的完整参数组合。

### 18. [Bugfix][KV Connector][NIXL] Shard a GQA draft's KV by head under an MLA target (#58471)
- **代码层面**：改的是 NIXL connector 的 `_resolve_head_sharded_draft_kv_heads` 逻辑和 `tests/v1/kv_connector/unit/test_nixl_connector.py`（590 行测试）。修复：当 target 模型是 MLA（Multi-head Latent Attention）而 draft 模型是 GQA 时，draft 的 KV 需要按 head 切分（SPLIT 区域），而不是像 MLA 那样复制（REPLICATE 区域）。
- **新手概念课堂**：MLA 的 KV 是所有 head 共享的（一个 latent 向量），所以 TP 时每个 rank 复制一份。GQA 的 KV 是按 head 分组的，TP 时每个 rank 拿一部分 head。混合场景下（MLA target + GQA draft），draft 的 KV 必须按 GQA 规则切分，不能按 MLA 规则复制。
- **对你有什么影响**：用 MLA 模型（如 DeepSeek）+ GQA draft 做投机解码 + NIXL PD 分离的用户，draft 的 KV 传输正确了，不再出现输出错误。

### 19. [Bugfix][AuxOutput] Send routed-experts block hashes only for computed or scheduled blocks (#62489)
- **代码层面**：改的是 `vllm/distributed/aux_output_connector/` 的 scheduler 和 worker 逻辑，以及 `tests/distributed/aux_output_connector/test_store.py`（200 行测试）。修复：只发送"已计算或已调度"的 block 的 routed-experts hash，不再 pin 未调度的 prompt block。
- **新手概念课堂**：routed-experts 是 MoE 模型每个 token 选了哪些 expert 的记录。aux output connector 把这些记录传给外部（比如做分析）。如果请求的 prompt 有 16 个 block 但只调度了 4 个，不应该把另外 12 个的 hash 也发出去并 pin 住（占内存）。
- **对你有什么影响**：用 aux output connector 做 MoE 路由分析的用户，显存占用降低，不再因为 pin 了未调度的 block 导致 OOM。

### 20. [Rust][Benchmark] Add openai-responses backend to vllm-bench (#59489)
- **代码层面**：新增 `rust/src/bench/src/backends/openai_responses.rs`（384 行），在 `backends/mod.rs` 的 `Backend` 枚举里加 `OpenAIResponses` 变体。支持 `/v1/responses` 端点的流式压测，reasoning delta 计入 TTFT 和 ITL。
- **新手概念课堂**：OpenAI 的 Responses API 是新的对话接口（区别于 Chat Completions），支持 reasoning 模型的"思考过程"流式输出。vllm-bench 现在能压测这个端点，并且把 reasoning token 的时间也算进首 token 延迟（TTFT）和 token 间延迟（ITL）。
- **对你有什么影响**：用 Rust 版 vllm-bench 压测 `/v1/responses` 端点的用户，现在可以直接跑。之前只能压 `/v1/completions` 和 `/v1/chat/completions`。

## sglang
## 💡 今日关键词

- **关键词：ROCm/AMD 适配（ROCm-aware tests）**
  - 一句话解释：让测试代码在 AMD GPU（ROCm 平台）上也能正确跳过或运行，而不是在 CUDA 专用功能上直接崩掉。
  - 为什么社区在关注：今天 20 个 commit 里有 3 个（#44599、#44421、#43335）专门做 AMD/ROCm 适配，说明 SGLang 正在把 AMD 当作一等公民平台，异构计算（NVIDIA + AMD + Intel）成为主流现实。
  - 对新手意味着什么：如果你只会写 CUDA，现在补上 HIP/ROCm 知识会立刻有溢价；跨平台测试意识（`skipUnless`、`current_platform.is_rocm()`）是进入大厂推理框架团队的敲门砖。

- **关键词：DeepSeek-V4 深度优化（DSV4 kernel/graph tuning）**
  - 一句话解释：围绕 DeepSeek-V4/V4.1 模型，在注意力后端、CUDA Graph、MoE 专家放置、SM100 专用 kernel 上做的一连串性能与正确性修补。
  - 为什么社区在关注：今天有 6 个 commit（#45301、#45251、#45299、#44632、#41630、#43838）都指向 DSV4，说明这个模型是当前推理优化的主战场，且已进入"挤牙膏式微优化"阶段（从 8 行提到 40 行、TP4 扩到 TP2）。
  - 对新手意味着什么：想学推理优化，DeepSeek 系列是最佳练手对象——代码公开、优化点密集、社区讨论多，跟着读这些 PR 比看教科书有效十倍。

- **关键词：CUDA Graph 与捕获期不变性（capture-time invariants）**
  - 一句话解释：CUDA Graph 会把一段 GPU 计算"录制"下来反复重放，录制期间分配的内存地址必须固定，不能中途扩容。
  - 为什么社区在关注：#45301 专门加了一个 `q_pad_captured` 标志防止 buffer 在捕获后增长，#45033 修了 XPU 上捕获查询 API 的兼容性——说明 graph 捕获的边界条件是最容易出隐蔽 bug 的地方。
  - 对新手意味着什么：CUDA Graph 是推理加速的必修课，但"捕获后不能改形状"这条铁律踩一次坑记一辈子。学会用断言保护这类不变量，是工程成熟度的标志。

- **关键词：批不变性（batch invariance）**
  - 一句话解释：同一个请求，无论和其他请求怎么拼 batch，输出结果必须逐比特一致。
  - 为什么社区在关注：#42087 为 Qwen3.8 修了一批 batch-invariant kernel，还新增了 `aten::mm.out` 的确定性实现——这是 RL 训练、可复现评测的刚需，正在从"可选"变成"默认要求"。
  - 对新手意味着什么：如果你做推理服务，客户问"为什么同样输入两次输出不一样"，batch invariance 就是答案。理解它需要懂浮点归约顺序，是深入 GPU 编程的好入口。

- **关键词：HiCache 存储后端生命周期（storage backend lifecycle）**
  - 一句话解释：给多级缓存（L3 后端）定义一个明确的"所有内存池都注册完了"回调，让远程缓存管理器能一次性完成注册。
  - 为什么社区在关注：#44030 引入 `on_host_pools_registered()` 钩子，配合文档说明注册顺序——说明 HiCache 生态在扩张（Mooncake、HF3FS、MORI…），需要标准化接口。
  - 对新手意味着什么：分布式缓存是长上下文推理的命脉。学会设计"生命周期钩子"这种模式，比背 API 更有价值。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：跨平台测试的条件跳过（pytest skip 机制）**
  - **是什么**：用 `pytest.importorskip` / `unittest.skipUnless` 在依赖不存在或平台不支持时优雅跳过测试，而不是让整个 CI 红掉。
  - **为什么重要**：SGLang 同时支持 CUDA/ROCm/XPU/CPU，一个测试文件在四种平台上跑。不懂条件跳过，你的 PR 会在别人的 CI 上莫名其妙挂掉；更糟的是，你可能会用 `try/except` 吞掉真正的 bug。
  - **动手试试**：
    ```python
    # demo_skip.py
    import importlib.util
    import unittest
    import pytest

    _requires_flashinfer = unittest.skipUnless(
        importlib.util.find_spec("flashinfer"),
        "flashinfer is CUDA-only",
    )

    class TestDemo(unittest.TestCase):
        @_requires_flashinfer
        def test_needs_flashinfer(self):
            import flashinfer  # noqa
            self.assertTrue(True)

        def test_always_runs(self):
            self.assertEqual(1 + 1, 2)

    # 另一个文件风格：模块级 importorskip
    # gemm = pytest.importorskip("flashinfer.prims_ts.gemm", reason="CUDA-only")
    ```
    运行：`python -m pytest demo_skip.py -v`
  - **预期结果**：在无 flashinfer 的机器上，`test_needs_flashinfer` 显示 `SKIPPED`，`test_always_runs` 显示 `PASSED`，整体退出码为 0。看到 `s` 和 `.` 就说明你掌握了。

- **知识点 2：CUDA Graph 捕获期的不变量保护**
  - **是什么**：CUDA Graph 捕获时，所有 tensor 地址被固化；捕获后若某 buffer 扩容，重放时会写到旧地址导致数据错乱。解法是用一个布尔标志在捕获后禁止扩容。
  - **为什么重要**：这是 CUDA Graph 最容易踩的坑之一。SGLang 的 `q_pad_buffer` 是"按需增长"的 scratch buffer，一旦被 graph 捕获就必须冻结。不懂这个，你写的 graph 优化会在长序列下静默出错（输出 garbage 而非报错），极难排查。
  - **动手试试**：
    ```python
    import torch

    class Meta:
        def __init__(self):
            self.buf = None
            self.captured = False

    def get_buf(meta, rows, cols):
        if meta.buf is None or meta.buf.shape[0] < rows:
            assert not meta.captured, "buffer grew after capture!"
            meta.buf = torch.zeros(rows, cols, device="cuda")
        if torch.cuda.is_current_stream_capturing():
            meta.captured = True
        return meta.buf[:rows]

    meta = Meta()
    get_buf(meta, 8, 16)          # 首次分配
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        get_buf(meta, 8, 16)      # 捕获，meta.captured=True
    try:
        get_buf(meta, 16, 16)     # 试图扩容 → AssertionError
    except AssertionError as e:
        print("caught:", e)
    ```
  - **预期结果**：打印 `caught: buffer grew after capture!`。看到这条断言，你就理解了为什么 SGLang 要新增 `q_pad_captured` 字段。

- **知识点 3：批不变性与确定性矩阵乘**
  - **是什么**：PyTorch 默认的 `mm` 在不同 batch 形状下会选不同 kernel、不同归约顺序，导致浮点结果有微小差异。batch-invariant 模式强制走"persistent matmul"（固定切分策略），保证逐比特一致。
  - **为什么重要**：RLHF 训练、可复现评测、审计场景都要求"同输入同输出"。SGLang 通过 `torch.library.impl` 劫持 `aten::mm`，还补了 `aten::mm.out`（带 out 参数的重载），因为很多框架内部走的是 out 版本。
  - **动手试试**：
    ```python
    import torch
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    torch.manual_seed(0)
    a = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    b = torch.randn(128, 256, device="cuda", dtype=torch.float32)

    r1 = a @ b
    r2 = torch.cat([a[:2] @ b, a[2:] @ b], dim=0)
    print("default equal:", torch.equal(r1, r2))  # 可能是 False

    enable_batch_invariant_mode()
    r3 = a @ b
    r4 = torch.cat([a[:2] @ b, a[2:] @ b], dim=0)
    print("invariant equal:", torch.equal(r3, r4))  # 应为 True
    ```
  - **预期结果**：第一行可能打印 `False`（取决于 GPU 和形状），第二行打印 `True`。看到 `True` 你就明白了为什么 Qwen3.8 的 PR 要同时改 elementwise kernel 的 `num_warps`——kernel 配置本身也会破坏不变性。

- **知识点 4：MoE 专家位置映射与"旁路"**
  - **是什么**：MoE 模型在大规模并行下会把逻辑专家映射到不同物理设备（EP/专家重排）。但 draft 模型（投机解码的草稿模型）的专家放置独立于 target 模型，不能共用同一份映射表。
  - **为什么重要**：SGLang 的 `FusedMoE` 新增 `uses_expert_location` 参数，让 draft MoE 显式声明"我不参与全局专家重排"。不懂这个，投机解码 + EP 组合时会加载错权重，表现为输出乱码而非崩溃——最毒的 bug 类型。
  - **动手试试**：
    ```python
    # 概念演示：用一个假 metadata 验证旁路逻辑
    class FakeMeta:
        def logical_to_all_physical(self, layer_id, eid):
            return [eid * 2, eid * 2 + 1]  # 假装重排

    def local_experts(layer, eid, meta):
        if meta is None or not layer.uses_expert_location:
            physical = [eid]
        else:
            physical = meta.logical_to_all_physical(layer.layer_id, eid)
        return physical

    class L:
        layer_id = 0
        uses_expert_location = False

    print(local_experts(L(), 3, FakeMeta()))  # [3]，不走重排
    L.uses_expert_location = True
    print(local_experts(L(), 3, FakeMeta()))  # [6, 7]，走重排
    ```
  - **预期结果**：先打印 `[3]` 再打印 `[6, 7]`。理解了这个开关，你就看懂了 #45299 为什么要同时改 `mega_moe.py`、`mega_moe_flydsl.py`、`moe_wna16.py` 三处。

- **知识点 5：存储后端的生命周期钩子设计**
  - **是什么**：定义 `on_host_pools_registered()` 抽象方法，在"初始一批内存池注册完毕"时恰好触发一次；后续动态注册的池不再触发。
  - **为什么重要**：远程缓存管理器（如 Mooncake）需要一次性拿到完整的池规格才能注册，否则要反复 amend。SGLang 用"基类默认触发 + 子类声明 `_registers_host_pools_after_super_attach = True` 接管"的方式处理继承链——这是模板方法模式的经典应用。
  - **动手试试**：
    ```python
    class StorageBackend:
        def register_mem_pool_host(self, pool): print("anchor:", pool)
        def register_mem_host_pool_v2(self, pool, name): print("v2:", name)
        def on_host_pools_registered(self): print(">>> burst done")

    class Controller:
        _registers_host_pools_after_super_attach = False
        def __init__(self, backend): self.backend = backend
        def attach(self):
            self.backend.register_mem_pool_host("anchor")
            if not self._registers_host_pools_after_super_attach:
                self.backend.on_host_pools_registered()

    class HybridController(Controller):
        _registers_host_pools_after_super_attach = True
        def attach(self):
            super().attach()
            for name in ("mamba", "indexer", "swa"):
                self.backend.register_mem_host_pool_v2(name, name)
            self.backend.on_host_pools_registered()

    print("--- base ---"); Controller(StorageBackend()).attach()
    print("--- hybrid ---"); HybridController(StorageBackend()).attach()
    ```
  - **预期结果**：base 在 anchor 后立即打印 `>>> burst done`；hybrid 在三个 v2 池之后才打印，且只打印一次。这就完全复现了 #44030 的设计意图。

---

## 🔥 重要更新

- **#45301 [Fix] deepseek-v4.1: 把 padded query buffer 锁在 decoder tail graph 内**：新增 `q_pad_captured` 标志，一旦 buffer 被 CUDA Graph 捕获就禁止扩容。这是防止 graph 重放写出垃圾数据的"保险丝"，影响所有启用 breakable CUDA graph 的 DSV4 用户。
- **#44632 [Perf] DeepSeek-V4.1: SM100 swap-AB decode 上限从 8 行提到 40 行**：把 DSpark verify 的可用行数提升 5 倍，覆盖 TP4、bs≤8 的场景。这是实打实的吞吐提升，且测试中把 `[0,1,31,...]` 长度表复制 5 份来覆盖新行数。
- **#43335 [Fix] AMD aiter: 修复非滑动窗口模型在 vectorized_5d KV 布局下的 decode 输出乱码**：引入 `decode_uses_page_table` 统一判断"哪些 decode kernel 读 2D page table"，修复了 dense 模型 + page_size>1 时拿到 flat kv_indices 的严重正确性 bug。
- **#45299 [Fix] deepseek-v4: 让 draft MoE 不参与 target 的专家放置**：新增 `uses_expert_location` 开关，避免投机解码草稿模型误用 target 的全局专家映射，修复了 EP + 投机解码组合下的静默错误。

---

## 📋 逐条解读

### 1. [Test] [AMD] diffusion: Make multimodal_gen unit tests ROCm-aware (#44599)
- **代码层面**：改了 `test_flux3_image_dit_quant.py`、`test_flux3_prims_ts_config.py`、`test_kandinsky6_layerwise_offload.py` 等测试文件。核心改动：用 `pytest.importorskip("flashinfer.prims_ts.gemm")` 和 `unittest.skipUnless(importlib.util.find_spec("flashinfer"), ...)` 把依赖 CUDA-only flashinfer 的测试标记为可跳过；并加 `pytestmark = pytest.mark.usefixtures("single_process_model_parallel")` 保证构造 attention 层时 SP group size 已就绪。
- **新手概念课堂**：这就像考试前检查"考场有没有配这个仪器"——没有就贴个"本场免考"的标签，而不是让考生（CI）直接晕倒。`importorskip` 是 pytest 提供的"依赖不存在就跳过整个模块"的开关。
- **对你有什么影响**：AMD 用户跑 pytest 不再被一堆 flashinfer 相关失败淹没；CI 在不同硬件上都能保持绿色。

### 2. [Feat] entrypoints: add a chat encoding adapter registry (#45300)
- **代码层面**：新增文件 `chat_encoding_adapters.py`，定义 `ChatEncodingAdapter` / `ChatEncodingOutputAdapter` 两个 `Protocol`，以及 `ChatOutputHooks`（`msgspec.Struct`，携带 `tool_call_parsing_active`、`history_tool_calls_count`、`tool_call_id` 三个回调）。核心思想：未注册的 encoding spec 继续走原有 serving 分支，只有被选中时才 import 具体 adapter。
- **新手概念课堂**：这是"注册表 + 延迟加载"模式。像餐厅菜单：菜单上列了 100 道菜（Protocol 定义接口），但厨房只在你点单时才准备食材（import 具体实现），避免开门就备齐所有菜。
- **对你有什么影响**：未来接入新的 chat template 编码方式（比如某家厂商的私有格式）只需注册一个 adapter，不用改核心 serving 代码；启动开销也更低。

### 3. [Feat] hicache: Notify storage backends when the initial host-pool registration burst completes (#44030)
- **代码层面**：在 `HiCacheStorage` 基类新增 `on_host_pools_registered()` 空实现；`cache_controller.py` 的 `attach_storage_backend` 在 `register_mem_pool_host` 后调用它，但用类属性 `_registers_host_pools_after_super_attach` 让子类接管；`hybrid_cache_controller.py` 设该标志为 `True`，在注册完所有 v2 池后自己调用。文档 `hicache_design.mdx` 补充了注册顺序说明。
- **新手概念课堂**：这是模板方法模式。基类说"我负责在 super() 返回时敲钟"，子类说"别急，我后面还有几个池要注册，钟由我来敲"。钟只能敲一次（恰好一次），否则远程管理器会重复注册。
- **对你有什么影响**：使用 Mooncake/HF3FS 等远程缓存后端时，注册更可靠；未来动态挂载 sidecar 池不会误触发重新注册。

### 4. [Test] [AMD] diffusion: Skip the Kandinsky6 FA dispatch test on ROCm (#44421)
- **代码层面**：`test_kandinsky6_bf16_dtype.py` 中，在已有的 SM12.x skip 旁边新增 `if backend == "FA" and current_platform.is_rocm(): pytest.skip(...)`，因为 ROCm 上 FA 会被解析成 SDPA。
- **新手概念课堂**：`AttentionBackendEnum.FA` 是"请求用 FlashAttention"，但平台可能把它降级成 SDPA（PyTorch 内置注意力）。测试断言的是"FA 行为"，在 ROCm 上实际跑的是 SDPA，自然对不上——跳过比改断言更诚实。
- **对你有什么影响**：AMD 上跑 diffusion 测试不再出现"假失败"，维护者能一眼看出哪些是平台限制。

### 5. [Doc] [AMD] cookbook: Fix DeepSeek-V4 PD image and c1 capacity (#45297)
- **代码层面**：改 `DeepSeek-V4.mdx` 和 `deepseek-v4.jsx`。三处实质变更：(1) Docker 镜像从 `v0.5.21-rocm724-mi35x-20261001` 升到 `v0.5.22-rocm720-mi35x-20261019`；(2) c1 覆盖配置的 `--max-running-requests` 从 2 改成 8，`--cuda-graph-bs-decode` 从 `1 2` 扩到 `1..8`；(3) 删掉了对 InferenceX PR #3664 的外部引用。
- **新手概念课堂**：`--max-running-requests` 是"同时允许多少请求在跑"，投机解码下要按 `N*2` 预留（因为 draft 也要占位）。c1 场景把 2 改成 8 是发现"2 个槽位不够覆盖 graph 预热"，属于实测调参。
- **对你有什么影响**：按 cookbook 部署 MI355X 多机 PD 的用户，c1 场景吞吐会更好；镜像升级带来新 kernel 支持。

### 6. [Fix] deepseek-v4.1: Keep the padded query buffer inside decoder tail graphs (#45301)
- **代码层面**：`DSV4Metadata` 新增 `q_pad_captured: bool`；`deepseek_v4.py` 抽出 `_zero_padded_q()` 方法，分配/复用 `q_pad_buffer` 时若捕获后需扩容则 `assert` 失败；`deepseek_v4_backend_hip_radix.py` 把 `q_pad_captured` 加入 `BREAKABLE_CUDA_GRAPH_KEEP_FIELDS`，保证 graph 重放时该标志也被保留。
- **新手概念课堂**：CUDA Graph 像录像机，录的时候把每个 tensor 的"家庭住址"写死了。如果之后 buffer 搬家（扩容到新地址），重放时还往老地址写——数据就丢了，而且不报错。`q_pad_captured` 就是"搬家禁令"。
- **对你有什么影响**：开启 breakable CUDA graph 的 DSV4 用户在长序列/变长 batch 下不再出现偶发乱码；代价是 buffer 一旦捕获就不再增长，内存占用更可预测。

### 7. [Perf] deepseek-v4.1: Score a request's verify rows as one request in the full-scan indexer (#45251)
- **代码层面**：`init_forward_metadata_indexer` 新增 `rows_per_request` 参数；`make_forward_metadata_from_raw_verify` 里 `rows_per_request = 1 if is_ragged else num_draft_tokens`；`PagedIndexerMetadata` 新增同名字段，并把它纳入 `use_jit_indexer` 的判定条件（`rows_per_request == 1` 才允许走 JIT metadata）。
- **新手概念课堂**：投机解码的 verify 阶段，一个请求会产生多行（draft token 数）。索引器原本把每行当独立请求处理，浪费调度；现在告诉它"这 N 行属于同一个请求"，可以合并扫描。JIT metadata 路径不支持这种合并，所以要显式禁用。
- **对你有什么影响**：投机解码 + 低压缩比 indexer 场景下 verify 阶段更快；`is_ragged` 分支保持旧行为，不影响非投机路径。

### 8. [Fix] observability: Log scheduler stats once per attention CP group (#45319)
- **代码层面**：`metrics_collector.py` 的 `init_new` 里，`is_stats_logging_rank` 从 `parallel.attn_tp_rank == 0` 改为 `parallel.attn_tp_rank == 0 and parallel.attn_cp_rank == 0`。
- **新手概念课堂**：TP（张量并行）和 CP（上下文并行）是两个正交维度。原来只判断 TP rank 0，导致每个 CP 组都各打一份日志——N 个 CP 组就 N 份重复。加上 CP rank 0 后，全局只有一份。
- **对你有什么影响**：开启 context parallel 的大规模部署，日志量骤降，指标不再重复计数。

### 9. [Fix] deepseek-v4: Keep the draft MoE out of the target's expert placement (#45299)
- **代码层面**：`FusedMoE.__init__` 新增 `uses_expert_location: bool = True` 参数并保存；`weight_loader` 和 `moe_wna16._local_expert_ids` 在该标志为 False 时跳过 `get_global_expert_location_metadata()`；`mega_moe.py` / `mega_moe_flydsl.py` 在 `uses_expert_location` 为 False 时传 `expert_location_dispatch_info=None`；`deepseek_v2.py` 给 draft 模型的 MoE 层设该标志为 False。
- **新手概念课堂**：MoE 的"逻辑专家"和"物理专家"是两回事，EP 重排会改变映射。draft 模型是独立训练/加载的，它的专家编号跟 target 没关系——硬套 target 的映射表等于让两个人共用一张身份证。
- **对你有什么影响**：投机解码 + EP 部署不再出现"draft 输出正常但 verify 全 reject"的诡异现象。

### 10. [Perf] DeepSeek-V4.1: Raise the SM100 swap-AB decode cap to 40 rows (#44632)
- **代码层面**：`decode_attention_sm100.py` 新增 `MAX_ROWS = 40`（原 `MAX_BATCH = 8` 保留但不再用于行数判断），`can_use_swapab_attention` 和 `swapab_attention` 的断言都改用 `MAX_ROWS`。测试 `test_swapab_attention.py` 把长度/专家表复制 5 份以覆盖最多 40 行，并新增 `test_rows_up_to_cap` 测 20 行和 40 行。
- **新手概念课堂**：swap-AB 是一种把矩阵乘的 A/B 操作数交换以适配 SM100 新硬件布局的 kernel 技巧。行数上限 8 是保守估计，实测到 ~48 行才被 FlashMLA 反超，所以设 40 留安全边际。
- **对你有什么影响**：DSpark verify 在 TP4、bs≤8 时能一次处理更多行，减少 kernel launch 次数，decode 吞吐提升。

### 11. [Feat] simulator: Add baseline collection and ML latency predictor training tools (#44299)
- **代码层面**：新增 `.github/workflows/_pr-test-simulator-gpu.yml`（GPU 版 simulator CI），CPU workflow 里加跑 `test_train_latency_model.py`，`_pr-test-check-changes.yml` 把新 workflow 纳入 simulator 变更检测。主体是 `tools/sglang-simulator/` 下的基线采集与 ML 延迟预测器训练工具。
- **新手概念课堂**：simulator 是"不真跑模型、只算账"的性能预测器。传统 simulator 用解析公式估延迟，现在引入 ML 模型（用真实 GPU 采集的 baseline 训练）来预测——更准，但需要采集工具和训练流程。
- **对你有什么影响**：做容量规划/集群选型时，可以用 simulator 快速评估不同并行策略的端到端延迟，不必真起集群。

### 12. [Fix] deepseek-v4: Refresh attn_sink after loading weights (#41630)
- **代码层面**：`_local_attn_sink` 里把切片索引存到 `self._attn_sink_local_slice`；新增 `refresh_attn_sink_cache()`，从 `self.attn_sink.data[slice]` 拷贝到 `_attn_sink_local`；`post_load_weights` 里对每个 self_attn 调用它。
- **新手概念课堂**：`attn_sink`（注意力汇点）是某些模型用来"吸收"多余注意力权重的可学习参数。加载权重前如果已经构建过本地缓存（`_attn_sink_local`），加载后缓存就过期了——必须刷新，否则用的是随机初始化的 sink。
- **对你有什么影响**：修复了"权重加载后输出质量下降"的隐蔽 bug，尤其是先 warmup 再 load weights 的部署流程。

### 13. [Fix] [Intel] spec: Fix XPU nightly dflash stop_regex and DSpark capture check (#45033)
- **代码层面**：两处。`schedule_batch.py` 的 `_locate_str_stop_finished_len` 新增 `full_window_matched` 参数并返回 `Optional[int]`；`_check_str_based_finish` 在正则匹配但完整窗口不匹配时（`new_accepted_len > 1`）走 `full_window_matched=False` 分支，允许"中间 token 匹配、末尾不匹配"的 end-anchored 正则。`dspark_draft_sampler.py` 用 `torch.xpu.is_current_stream_capturing()` 替代 `torch.cuda.*`（XPU 上会抛异常）。
- **新手概念课堂**：end-anchored 正则（如 `foo$`）在投机解码多 token 接受时，可能"第 3 个 token 后匹配"但"第 5 个 token 后不匹配"。原逻辑只看最后一个 token，会漏判。新逻辑回溯中间位置。
- **对你有什么影响**：XPU 上 dflash 的 stop_regex 停止条件更准确；DSpark 在 XPU 上不再因捕获查询 API 报错而崩。

### 14. [Perf] DeepSeek-V4.1: Run the small-batch decode fusions at TP2 (#43838)
- **代码层面**：`wo_a_fused.cuh` 的 `WoATrait` 从 `template <int kMaxTokens>` 扩为 `template <int kMaxTokens, int kGroups = 2>`，`kGroups` 允许 2（TP4）或 4（TP2）；`kNOut = kGroups * kGroupN`，`kClusters = kNOut / kNTile`。注释说明 TP2 会启动 64 clusters / 512 CTAs，约 1.7 波，可能限制收益。
- **新手概念课堂**：WO-A 融合 kernel 把 rope + 部分矩阵乘 + 量化三步合一。TP4 时每个 rank 负责 2 个 o_groups（2048 列），TP2 时负责 4 个（4096 列）。用模板参数把 group 数编译期化，避免运行时分支。
- **对你有什么影响**：TP2 部署的 DSV4 用户在小 batch decode 时也能享受融合 kernel 加速；TP4 行为不变（默认 `kGroups=2`）。

### 15. [Fix] test-utils: Allow expected server exits in canary tests (#45172)
- **代码层面**：`CustomTestCase` 新增类属性 `expect_server_exit = False`；`_callTestMethod` 里若为 True 则跳过重试逻辑；`test_self_e2e_perturb_raise.py` 的 `TestPerturbRaiseMha` 设 `expect_server_exit = True`。
- **新手概念课堂**：SGLang 测试框架默认失败重试（`SGLANG_TEST_MAX_RETRY`），但故障注入测试（故意让 server 崩）重试会复用已死的 server，永远失败。`expect_server_exit` 是"我知道它会死，别重试"的声明。
- **对你有什么影响**：kv-canary 的故障注入测试不再因重试逻辑而假红，CI 更稳定。

### 16. [Fix] test: Add logical_forward_mode to the KDA prefill-graph stand-in batch (#45218)
- **代码层面**：`test_kda_prefill_graph.py` 的 `_batch()` 构造的 `SimpleNamespace` 新增 `logical_forward_mode=ForwardMode.EXTEND`。
- **新手概念课堂**：`forward_mode` 是"物理上怎么跑"（比如 CUDA graph 捕获时可能标记为特殊模式），`logical_forward_mode` 是"逻辑上是什么阶段"（prefill/extend vs decode）。KDA prefill graph 代码新增了对 `logical_forward_mode` 的读取，测试的假 batch 得补上这个字段。
- **对你有什么影响**：纯测试修复，保证 KDA prefill graph 测试在代码演进后仍能跑通。

### 17. docs: promote SGLang Summit 2026 in README (#45271)
- **代码层面**：`README.md` 在 "Get Started" 上方插入一段 blockquote，宣传 2026 年 11 月 12-13 日旧金山峰会。
- **新手概念课堂**：无技术概念，纯社区运营。
- **对你有什么影响**：想线下认识 SGLang 核心开发者的，这是个机会。

### 18. [Fix] qwen3.8: Fix batch-invariant kernels for Qwen3.8 Flash-Next (#42087)
- **代码层面**：三处。`elementwise.py` 的 `_launch_fused_gate_sigmoid_mul` 在 `is_batch_invariant_mode_enabled()` 时也把 `num_warps` 压到 ≤8（原来只在 `num_tokens >= 1024` 时）。`batch_invariant_ops.py` 新增 `mm_out_batch_invariant` 并注册 `aten::mm.out`。`qsa/kernel.py` 的 `qsa_fast_topk` 在 batch-invariant + 非 HIP 时改用 flashinfer 的 `top_k_ragged_transform(deterministic=True, tie_break=TopKTieBreak.SMALL)`。
- **新手概念课堂**：batch invariance 不只靠 matmul，任何"按 batch size 选 kernel 配置"的地方都会破坏它。elementwise kernel 的 `num_warps` 影响归约树形状；top-k 的 tie-break 策略影响相等值的选取顺序。全部要固定。
- **对你有什么影响**：Qwen3.8 Flash-Next 在 batch-invariant 模式下输出可复现，适合 RL 训练和审计场景。

### 19. [Fix] [AMD] aiter: Fix garbage decode output for non-sliding-window models on the vectorized_5d KV layout (#43335)
- **代码层面**：`aiter_backend.py` 新增 `self.decode_uses_page_table = self.use_triton_unified_attention or self.kv_cache_is_vectorized_5d`，把原来所有 `if not self.use_triton_unified_attention` / `if self.use_triton_unified_attention` 的判断统一替换为 `decode_uses_page_table` 的正反。新增测试 `test_aiter_vectorized_5d_decode.py` 跑 GSM8K。
- **新手概念课堂**：KV cache 有两种寻址方式：flat（一维 per-token 索引 + indptr）和 page table（二维 `[bs, max_blocks]`）。`pa_decode_gluon`（SHUFFLE 5D 布局的 decode kernel）只认 page table。原来只有 unified attention 才建 page table，dense 非滑动窗口模型在 5D 布局下拿到 flat 索引 → 读错地址 → 输出乱码。
- **对你有什么影响**：AMD + vectorized_5d KV 布局 + dense 模型的用户，decode 输出恢复正常；这是严重正确性修复。

### 20. [Perf] [XPU] fp8: Optimize FP8 scale layouts (#40052)
- **代码层面**：`compressed_tensors_w8a8_fp8.py` 和 `fp8.py` 在 XPU 上调用新函数 `prepare_xpu_block_scale_for_scaled_mm`，把 `weight_scale` 重排成 scaled_mm 期望的布局（用 `set_` 原地替换 data），避免每次 forward 做 contiguous/copy。把原先散落在 `fp8.py` 里的内联重排逻辑收敛到统一函数。
- **新手概念课堂**：FP8 量化权重带 scale（缩放因子），scaled_mm 对 scale 的内存布局有要求（比如按 block 转置连续）。检查点存的是逻辑布局，加载后重排一次，比每次 forward 都转置便宜得多。
- **对你有什么影响**：Intel XPU 上 FP8 推理的每 token 延迟下降；代码更易维护（逻辑集中在一个函数）。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)