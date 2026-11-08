# 🚀 vLLM & SGLang 每日更新 — 2026-11-08

> 自动生成于 2026-11-08 07:55 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：MRV1 退场（Model Runner V1 被删除）**
  - 一句话解释：vLLM 彻底删掉了旧版模型执行器 V1 的代码，只保留 V2 架构。
  - 为什么社区在关注：这是引擎核心的大扫除，说明 vLLM 已全面转向新架构（V2 worker/runner），旧代码路径不再维护，CI 配置里大量 `MRV1` 相关测试项被移除。这代表着"技术债清理 + 架构收敛"的趋势。
  - 对新手意味着什么：如果你在网上看到老教程提到 `cpu_model_runner.py`、`gpu_model_runner.py` 这类文件，说明教程已过时。学习时应以当前主干代码为准，别被旧博客带偏。

- **关键词：ROCm / AMD GPU 适配**
  - 一句话解释：今天有 5 个以上 commit 都在改 AMD ROCm 相关的 CI、内核、workspace 和测试。
  - 为什么社区在关注：AMD MI300/MI355X 正在成为 NVIDIA 之外的重要推理硬件，vLLM 团队在系统性地补齐 ROCm 的 CI 稳定性（GPU 锁、AITER 开关、缓存策略）。说明异构计算已是主流。
  - 对新手意味着什么：如果你在学 GPU 推理，除了 CUDA，了解 ROCm/HIP 生态会大大增加你的竞争力，尤其是算子移植、AITER 这类 AMD 专属库。

- **关键词：KV Connector / 卸载缓存的生命周期**
  - 一句话解释：KV 缓存可以卸载到 CPU 或外部存储，今天新增了"睡眠/暂停时不丢弃卸载缓存"的选项。
  - 为什么社区在关注：RL 训练、多轮对话、服务化部署都希望缓存能跨请求、跨暂停复用，减少重复计算。这是"推理服务化"趋势的体现。
  - 对新手意味着什么：理解 KV cache 的层次（GPU → CPU → 外部存储）是做大模型推理优化的必修课，学会用 `reset_connector=False` 能省下大量重算时间。

- **关键词：CUDA Graph 复用与预算细分**
  - 一句话解释：多模态编码器的 CUDA Graph 现在支持跨不同 token 预算共享缓冲区，预算表也从 2 的幂改成更细的网格。
  - 为什么社区在关注：CUDA Graph 是降低推理延迟的关键技术，但每档预算都单独分配显存太浪费。共享缓冲区把显存占用从"总和"降到"最大值"，这是性能优化的典型思路。
  - 对新手意味着什么：CUDA Graph 是高性能推理的核心技能，理解"捕获-重放"和"缓冲区复用"能让你看懂很多推理框架的优化手段。

- **关键词：模块化导入 / 启动加速**
  - 一句话解释：今天有 commit 专门避免 `vllm.v1.structured_output` 导入 `vllm.parser`，减少启动时的依赖链。
  - 为什么社区在关注：vLLM 启动时间直接影响用户体验和 CI 速度，社区在系统性地做"懒加载"和"依赖解耦"。
  - 对新手意味着什么：学会用 `sys.modules` 检查和延迟导入（`if TYPE_CHECKING`）是写大型 Python 项目的必备技能。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph 的缓冲区共享（来自 Commit 8）**
  - **是什么**：CUDA Graph 是把一段 GPU 计算录制成"录像带"，之后直接重放，省去每次启动 kernel 的开销。但每档 token 预算都录一盘带子，就要各自分配显存。共享缓冲区就是让所有录像带都用同一块大画布，只取前面一段来用。
  - **为什么重要**：不共享的话，25 档预算就要 25 份显存，模型直接 OOM。共享后显存只占最大预算的 1 倍。不懂这个，你调 `encoder_cudagraph_token_budgets` 时很容易把显存撑爆。
  - **动手试试**：
    ```bash
    # 先看看预算生成逻辑（纯 CPU 可跑，不需要 GPU）
    python -c "
    from vllm.v1.worker.gpu.cudagraph_utils import EncoderCudaGraphManager
    print(EncoderCudaGraphManager._generate_budgets(64, 16384))
    "
    ```
  - **预期结果**：你会看到类似 `[64, 80, 96, 112, ..., 12288, 16384]` 的 25 档细分网格，而不是旧的 `[64, 128, 256, ..., 16384]` 的 2 的幂。这验证了文档里说的"四分之一倍频程步进"。

- **知识点 2：Python 导入副作用检测（来自 Commit 1 和 Commit 20）**
  - **是什么**：用 `sys.meta_path` 插入一个自定义 Finder，在模块被导入时"抓现行"。Commit 1 用它检测 `nixl` 是否被意外导入，Commit 20 用 `sys.modules[name] = None` 让导入直接报错。
  - **为什么重要**：大型项目里一个意外导入可能拖慢启动、引入重依赖、甚至导致环境不兼容。这是"导入卫生"的工程实践。
  - **动手试试**：
    ```python
    import sys, importlib

    class Spy:
        def find_spec(self, name, path, target=None):
            if name.startswith("json"):
                print(f"⚠️ 有人导入了 {name}")
            return None

    sys.meta_path.insert(0, Spy())
    import json  # 会触发打印
    ```
  - **预期结果**：运行后看到 `⚠️ 有人导入了 json`。把 `json` 换成 `nixl`，你就复现了 Commit 1 的测试思路。

- **知识点 3：张量并行下的 QKV 切分陷阱（来自 Commit 4）**
  - **是什么**：大模型在多卡上做张量并行（TP）时，Q/K/V 投影会被切分到不同卡。`QKVParallelLinear` 输出的是"本地分片"，而 `ColumnParallelLinear` 输出的是"全量后切分"，两者的内存布局不同，切分方式也必须不同。
  - **为什么重要**：用错切分函数会把 Q、K、V 的数据"搅乱"，模型输出变成乱码但不会报错——这是最可怕的 bug 类型。Commit 4 就是修这个：把 `split_qkv()` 换成 `chunk(3, dim=-1)`。
  - **动手试试**：
    ```python
    import torch
    # 模拟 QKVParallelLinear 的本地输出：[S, B, 3*local_heads*head_dim]
    x = torch.arange(24).reshape(1, 1, 24)  # 假设 3 组，每组 8 维
    q, k, v = x.chunk(3, dim=-1)  # 正确的本地切分
    print("q:", q.flatten().tolist())
    print("k:", k.flatten().tolist())
    print("v:", v.flatten().tolist())
    ```
  - **预期结果**：q 是 `[0..7]`，k 是 `[8..15]`，v 是 `[16..23]`，干净利落。如果你用"先拼接再按 head 交错切"的错误方式，就会得到交错的数据。

- **知识点 4：KV 缓存卸载与睡眠模式（来自 Commit 7）**
  - **是什么**：`llm.sleep(level=1)` 会释放 GPU 上的 KV 缓存。默认情况下，连卸载到 CPU 的 KV 缓存也会被清掉。新参数 `reset_connector=False` 让你保留这部分缓存。
  - **为什么重要**：在 RL 训练中，模型权重更新后需要重新推理，但很多请求的 KV 缓存其实可以复用。保留卸载缓存能省下大量 prefill 时间。但要注意：缓存哈希不包含权重，权重变了就必须清缓存或加 `cache_salt`。
  - **动手试试**：
    ```python
    from vllm import LLM
    llm = LLM(model="facebook/opt-125m", enable_prefix_caching=True)
    # 第一次推理
    out1 = llm.generate(["Hello, my name is"])
    # 睡眠但保留卸载缓存
    llm.sleep(level=1, reset_connector=False)
    llm.wake_up()
    # 第二次推理应能命中缓存
    out2 = llm.generate(["Hello, my name is"])
    print("缓存命中的请求应更快返回")
    ```
  - **预期结果**：第二次推理的 prefill 时间明显缩短（可通过日志或计时观察）。如果权重已变，记得不要用这个参数。

- **知识点 5：AITER 与精度模拟（来自 Commit 11）**
  - **是什么**：AITER 是 AMD 的算子库。它的 RMSNorm 只做一次舍入，而 HuggingFace 参考实现做两次，导致 bf16 下数值有微小差异。解决方案是开启 Inductor 的 `emulate_precision_casts`，让周围的算子模拟 HF 的舍入行为。
  - **为什么重要**：跨硬件、跨库的数值对齐是推理正确性的基础。PPL（困惑度）测试对数值差异很敏感，不懂这个你会以为"模型坏了"，其实只是舍入不同。
  - **动手试试**：
    ```bash
    # 在 ROCm 机器上跑（需要 AITER）
    VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 \
    pytest tests/models/multimodal/generation_ppl_test/test_qwen.py -v
    ```
  - **预期结果**：开启 `emulate_precision_casts` 后 PPL 测试通过；关闭后可能出现 PPL 偏差超标。这让你直观理解"数值对齐"的重要性。

---

## 🔥 重要更新

1. **Commit 10：删除 Model Runner V1**——这是引擎架构的大手术，旧代码路径彻底移除，CI 配置大规模瘦身。任何依赖 MRV1 的定制代码都会失效，需要迁移到 V2。
2. **Commit 7：KV Connector 缓存跨睡眠保留**——对 RL 训练和服务化部署影响巨大，新增 `reset_connector=False` 参数，能显著减少重复 prefill 计算。
3. **Commit 8：多模态编码器 CUDA Graph 缓冲区共享**——显存占用从"所有预算之和"降到"最大预算"，让更细的预算网格变得可行，直接提升多模态推理性能。
4. **Commit 9：FA 滑动窗口在 batch invariance 下保持激活**——修复了一个隐蔽的正确性 bug：滑动窗口请求单独跑和混批跑结果不一致。对需要确定性输出的场景（如评测、RL）至关重要。

---

## 📋 逐条解读

### 1. [Test][EC Connector] Add the no-NIXL import test the docs refer to (#56508)
- **代码层面**：新增 `tests/v1/ec_connector/unit/test_no_nixl_imports.py`，用子进程 + 自定义 `MetaPathFinder` 检测导入 `cpu/connector.py` 等模块时是否意外拉入 `nixl` 或 P2P 传输模块。同时更新了文档措辞。
- **新手概念课堂**：这就像给厨房装了个"食材监控"——你做"清汤面"（gate-off 路径）时，监控会记录你有没有偷偷拿"辣椒"（nixl）。即使你拿了又放回去（导入失败被吞掉），监控也知道。目的是保证"不用 NIXL 的用户不需要装 NIXL"。
- **对你有什么影响**：普通用户无感知。但如果你在无 NIXL 环境部署 EC Connector，这个测试保证了它不会因为缺包而崩溃。

### 2. [Bugfix][Qwen2.5-Omni] Size the video encoder budget from the processor's video limits (#66190)
- **代码层面**：新增 `tests/models/multimodal/processing/test_qwen2_5_omni.py`，验证视频 token 预算与 `max_model_len` 对齐、FPS 采样视频不超预算、dummy 视频填满预算。
- **新手概念课堂**：视频编码器需要预先"预留座位"（token 预算）。以前预留的座位数算错了，导致要么浪费（空座），要么超载（OOM）。现在按处理器的真实视频上限来算，就像按实际乘客数订票。
- **对你有什么影响**：用 Qwen2.5-Omni 处理长视频时，不再因为预算算错而报错或浪费显存。

### 3. [ROCm][CI] Wait for DPX GPU lock to avoid collisions (#66189)
- **代码层面**：`.buildkite/scripts/hardware_ci/run-amd-test.sh` 里把 `flock -n`（非阻塞，抢不到就失败）改成 `flock -w 600`（最多等 600 秒），返回码从 1 改成 134。
- **新手概念课堂**：以前是"抢不到停车位就走人"，现在是"排队等 10 分钟"。CI 任务多时，GPU 冲突从"直接失败"变成"排队等待"，减少误报。
- **对你有什么影响**：普通用户无感知，但 AMD CI 的红灯会变少，PR 合并更顺畅。

### 4. [Bugfix][Model] Fix dots_ocr vision tower QKV split for TP > 1 (#64543)
- **代码层面**：`vllm/model_executor/models/dots_ocr.py` 中，把 `Qwen2VisionAttention.split_qkv()` 换成 `x.chunk(3, dim=-1)` 并手动 reshape。因为 `self.qkv` 是 `QKVParallelLinear`，输出已是本地分片。
- **新手概念课堂**：想象把一条三色绳子分给三个人。`split_qkv` 以为绳子是"红红红绿绿绿蓝蓝蓝"（全量布局），实际是"红绿蓝红绿蓝"（本地分片）。用错方法就会把颜色搅乱。`chunk(3)` 是按段切，符合本地分片的布局。
- **对你有什么影响**：dots.ocr 在 TP=2 及以上多卡部署时输出正确，不再乱码。

### 5. [Recipe] Kimi-K3.1-Pro: add MI355X concurrency-1 preset (#66343)
- **代码层面**：更新 `recipes/models/moonshotai/Kimi-K3.1-Pro.yaml`，新增 MI355X 并发 1 的预设，调整 speculative decoding token 数（并发 48 用 3 个，并发 1 用 7 个），补充 FP32 KDA 状态、`--mamba-ssm-cache-dtype float32` 等参数。
- **新手概念课堂**：这是"菜谱"——针对特定硬件（MI355X）和特定场景（低并发）调好的参数组合。就像游戏里的"推荐配置"，照着抄就能跑出最佳性能。
- **对你有什么影响**：在 MI355X 上部署 Kimi-K3.1-Pro 时，可以直接用预设，不用自己调参。

### 6. [Bugfix][ROCm] Reserve kpool decode indexer logits in workspace during profiling (#65092)
- **代码层面**：新增测试 `test_kpool_profiling_reserves_decode_logits_workspace`，验证 profiling 阶段为 decode 阶段的 logits 预留 workspace，避免后续 decode 时 workspace 增长导致锁冲突。
- **新手概念课堂**：workspace 是 GPU 上的"临时工位"。profiling 时只按 prefill 需求订工位，结果 decode 时发现工位不够，临时扩建会撞上"工位已锁定"。修复方法是 profiling 时按最坏情况（decode）订工位。
- **对你有什么影响**：ROCm 上使用 GLM5Next 的 kpool 稀疏索引器时，decode 阶段不再因 workspace 不足而报错。

### 7. [Core][KVConnector] Keep the offloaded KV tier across pause/sleep (#57810)
- **代码层面**：新增 `reset_connector` 参数贯穿 `sleep`、`release_kv_cache_memory`、`reset_prefix_cache` 和 HTTP 端点。文档补充了使用说明和 `cache_salt` 的注意事项。同时 `vllm-proto` 版本从 0.6.0 升到 0.7.0。
- **新手概念课堂**：KV 缓存像"草稿纸"。睡眠时默认把草稿纸全扔了（包括锁在柜子里的）。新参数让你说"柜子里的别扔"，醒来后还能接着用。但注意：如果模型权重变了，旧草稿就废了，要么扔掉，要么给新请求打上 `cache_salt` 标记。
- **对你有什么影响**：RL 训练和服务化部署中，暂停后恢复能复用卸载缓存，显著减少重算。用错（权重变了还保留）会导致错误输出。

### 8. [Multimodal] Refactor encoder cuda graph to support share graph buffer reuse (#65084)
- **代码层面**：`EncoderCudaGraphManager` 新增 `_shared_input_buffers` / `_shared_output_buffers`，预算生成从 2 的幂改为细分网格（≤1024 用四分之一倍频程，以上用半倍频程），固定尺寸 item 的路径用精确倍数预算。新增 `TestSharedCaptureBuffers` 和 `TestSharedBufferFallback` 测试。
- **新手概念课堂**：以前每档预算都单独买一块画布，25 档就买 25 块。现在买一块最大的，所有档都在这块画布上"取前面一段"来画。画布不兼容的（形状不同）就单独买，保证兼容性。
- **对你有什么影响**：多模态模型开 `cudagraph_mm_encoder` 时显存占用大幅降低，可以开更细的预算网格，命中率更高、延迟更低。

### 9. [Bugfix][Attention] Keep FA sliding window active under batch invariance (#65627)
- **代码层面**：`vllm/v1/attention/backends/flash_attn.py` 修复滑动窗口在 batch invariance 模式下被错误关闭的问题。新增测试对比"单独跑"和"混批跑"的 logprobs 是否逐位一致。
- **新手概念课堂**：滑动窗口是"只看最近 N 个词"的注意力。以前在混批时，FA3 会误以为"反正都被覆盖了，不用管窗口"，结果单独跑和混批跑结果不同。修复后两种模式结果逐位一致。
- **对你有什么影响**：需要确定性输出的场景（评测、RL、可复现实验）不再因批大小不同而结果漂移。

### 10. [Core] Remove Model Runner V1 (#63373)
- **代码层面**：大规模删除 `cpu_model_runner.py`、`gpu_model_runner.py` 等 MRV1 文件，清理 CI 配置中的 MRV1 测试项和依赖项。涉及 5 万+ 行变更。
- **新手概念课堂**：这就像拆掉老房子的旧电路，只保留新电路。旧电路（MRV1）曾支撑早期功能，但维护成本高、和新架构冲突。拆掉后代码更干净，但任何还接在旧电路上的"私接电线"（定制代码）都会断电。
- **对你有什么影响**：如果你有基于 MRV1 的 fork 或插件，需要迁移到 MRV2。普通用户无感知，但启动可能更快、代码更易读。

### 11. [ROCm][CI] Fix the test_qwen.py failure when AITER is on (#63166)
- **代码层面**：新增 `tests/models/multimodal/generation_ppl_test/conftest.py`，用 fixture 参数化 AITER 开关（off/on/on+rmsnorm off）。`test_qwen.py` 在 AITER 开启时注入 `emulate_precision_casts: True`。
- **新手概念课堂**：AITER 的 RMSNorm 只舍入一次，HF 舍入两次，bf16 下差之毫厘。`emulate_precision_casts` 让周围算子"假装"多舍入一次，和 HF 对齐。就像两个厨师做菜，一个放盐一次，一个分两次放，最后味道微差，统一成"分两次放"就一致了。
- **对你有什么影响**：ROCm + AITER 环境下 Qwen 多模态的 PPL 测试通过，数值更接近 HF 参考。

### 12. [Bugfix][Step5][Spec Decode] Skip VllmConfig re-validation for the MTP draft config (#66000)
- **代码层面**：`vllm/config/model.py` 扩展 `is_submodel_config` 的语义，`vllm/config/vllm.py` 在 `__post_init__` 中跳过 draft config 的重复校验。新增 `tests/models/step5/test_mtp_config.py` 验证 PP=1/2 下配置视图不触发 `NotImplementedError` 也不降级 CUDA graph 模式。
- **新手概念课堂**：draft 模型（草稿模型）的配置是"借用"目标模型的配置视图。以前每次创建视图都会重新校验一遍，就像每次借书都要重新办借书证，还会把书的位置搞乱（CUDA graph 模式被降级）。现在直接复用已验证的配置。
- **对你有什么影响**：Step5 模型用 MTP 投机解码 + 流水线并行时不再报错，CUDA graph 配置不被意外降级。

### 13. [ROCm][CI] Make Zen5 CPU tests optional (#66297)
- **代码层面**：`.buildkite/hardware_tests/zen.yaml` 两个 step 加 `optional: true`。
- **新手概念课堂**：Zen5 CPU 测试从"必须通过"变成"可选通过"。就像选修课，挂了不影响毕业，但成绩单上还是会有记录。
- **对你有什么影响**：Zen5 硬件不可用时 CI 不会因此红灯，PR 合并更顺畅。

### 14. [Chore] Use upstream Transformers MiniMaxM3VLConfig (#61813)
- **代码层面**：删除 `vllm/transformers_utils/configs/minimax_m3.py` 中的自定义配置类，改用 `transformers.MiniMaxM3VLTextConfig`。测试中 `sparse_attention_config` 字典访问改为直接的 `index_n_heads` / `index_block_size` 属性。
- **新手概念课堂**：以前 vLLM 自己"手搓"了一个 MiniMax-M3 的配置类，现在上游 Transformers 官方支持了，就删掉自己的，用官方的。就像以前自己写工具函数，现在标准库有了，就用标准库，减少维护负担。
- **对你有什么影响**：MiniMax-M3 的配置解析更标准、更易维护。如果你有自定义代码访问 `sparse_attention_config`，需要改成属性访问。

### 15. [Bugfix][MRV2] Use promised max_query_len for encoder-only FULL CUDA graphs (#65275)
- **代码层面**：新增 `tests/v1/cudagraph/test_encoder_only_full_cudagraph.py`，验证 encoder-only（非因果）注意力在 FULL CUDA Graph 下，单 prompt 重放结果与 eager 一致。涉及 Qwen3、Gemma3、DeepseekV2 三种架构。
- **新手概念课堂**：CUDA Graph 捕获时用"多请求 dummy batch"录带子，重放时可能只有"单请求"。以前用实际 query_len 而不是承诺的 max_query_len，导致非因果注意力算错。修复后用承诺值，保证一致性。
- **对你有什么影响**：使用 encoder-only 注意力的模型（如某些 embedding 模型）开 FULL CUDA Graph 时输出正确。

### 16. [Build][MiniMax-M3] Bump fmha_sm100 to 51f4f5c (#66228)
- **代码层面**：`cmake/external_projects/fmha_sm100.cmake` 中 `GIT_TAG` 从 `25c1447...` 更新到 `51f4f5c...`。
- **新手概念课堂**：这是"升级依赖库版本"。fmha_sm100 是 SM100（Blackwell）架构的 fused multi-head attention 库，升级到新 commit 以获取 bug 修复或性能提升。
- **对你有什么影响**：在 SM100 GPU 上使用 MiniMax-M3 时，attention 计算可能更快或更稳定。

### 17. [ROCm][CI] Cache only a ROCm stack the PR's CI pushed itself (#65957)
- **代码层面**：`.buildkite/scripts/rocm/build.sh` 新增 `rocm-$STACK-stack-built` 元数据键，只在 base 模式自己 push 了 stack 时才记录。`run_ci_command.py` 相应调整 `ROCM_STACK_BUILT_KEY`。新增 `.github/workflows/ci-command-tests.yml` 和单元测试。
- **新手概念课堂**：以前 CI 缓存可能"误认"别的 PR 推的镜像为自己的，导致用错版本。现在只有自己推的才记录，就像"只认自己做的菜，不认别人放在你冰箱里的"。
- **对你有什么影响**：ROCm CI 缓存更可靠，不会因为别的 PR 的镜像污染而构建失败。

### 18. [CPU][Refactor] Move CPU ops wrappers to vllm/_cpu_ops.py (#66100)
- **代码层面**：把 CPU 算子包装从 `vllm/_custom_ops.py` 移到新的 `vllm/_cpu_ops.py`，更新所有引用（benchmark、setup.py、CI 依赖列表等）。
- **新手概念课堂**：把"CPU 专用工具箱"从"通用工具箱"里分出来。以前所有算子都堆在一个大文件里，现在 CPU 算子独立成文件，导入时不会连带拉入 GPU 依赖。
- **对你有什么影响**：CPU 用户导入更轻量，代码结构更清晰。如果你有代码 `from vllm._custom_ops import cpu_attention_with_kv_cache`，需要改成 `from vllm._cpu_ops import ...`。

### 19. [Refactor] Remove dead code kernels (#65743)
- **代码层面**：删除 `is_aiter_found_and_supported_on_rdna4()`、`make_host_fp32w()`、`make_workspaces()`、`get_layer_input_schema()` 等未被调用的函数。
- **新手概念课堂**：大扫除，扔掉"以为以后会用但一直没用"的东西。死代码会增加阅读负担、误导新手、增加维护成本。
- **对你有什么影响**：如果你有代码调用了这些被删的函数（虽然它们本就是死代码），需要移除。普通用户无感知。

### 20. [Perf][Startup] Don't import vllm.parser from vllm.v1.structured_output (#65589)
- **代码层面**：`vllm/reasoning/abs_reasoning_parsers.py` 把 `ReasoningEnd` 的导入移到 `TYPE_CHECKING` 下，新增 `reasoning_end_token_ids` 和 `implicit_reasoning_end_token_ids` 属性。`tests/standalone_tests/lazy_imports.py` 新增 `fastapi`、`starlette` 和 `vllm.v1.structured_output` 的导入检查。
- **新手概念课堂**：`vllm.parser` 会拉入 `fastapi` 等前端依赖，但引擎核心（structured_output）根本不需要它们。把导入移到 `TYPE_CHECKING`（只在类型检查时导入，运行时跳过），就像"只在需要看图纸时才打开图纸柜，平时不打开"。
- **对你有什么影响**：vLLM 启动更快，引擎核心进程不再加载前端依赖。如果你在无 FastAPI 的纯引擎环境部署，不再因缺包而报错。

## sglang
## 💡 今日关键词

- **关键词：AMD/ROCm 生态适配（gfx950 / MI350X / MI355X）**
  - **一句话解释**：今天有 5 个 commit 专门为 AMD 的 MI350X/MI355X（gfx950 架构）做 kernel 和模型支持，包括 MiniMax-M3.1-Flash、Qwen3 的 FP8 融合、Quark MXFP4 量化布局、aiter 注意力后端的 q 量化修复。
  - **为什么社区在关注**：AMD 正在从"能跑"走向"跑得快"，社区开始针对 gfx950 做专门的 kernel 调优（比如 smallm FP8 GEMM 的形状表、1x128 分块量化的融合），说明异构计算已经是主流，不再是 NVIDIA 一家独大。
  - **对新手意味着什么**：如果你只会写 CUDA，现在补上 HIP/ROCm 的知识会非常有价值。AMD 的 kernel 生态还比较年轻，很多优化机会（比如今天这个把量化融进 RMSNorm/SiLU 的 Triton kernel）等着人去做。

- **关键词：MoE / 大模型量化（MXFP4、FP8、BF16 路由）**
  - **一句话解释**：DeepSeek V4 的 commit 修复了"未量化的 BF16/F16/F32 checkpoint 被错误地送进 MXFP4 专家路径"的问题；Quark MXFP4 修复了 scale 布局；Megamoe 适配 FlashInfer 0.7.3 的枚举配置。
  - **为什么社区在关注**：MoE 模型越来越大，量化格式五花八门（FP8、MXFP4、NVFP4），"一个 checkpoint 该走哪条路径"成了真正的工程难题，走错路径就是精度崩坏或直接报错。
  - **对新手意味着什么**：量化不只是"把 float 变 int8"，而是涉及数据布局（column-major vs row-major）、scale 粒度（per-tensor / per-block / 1x128）、以及框架间的格式约定。这块是当前最缺人的方向之一。

- **关键词：投机解码 / 稀疏注意力 / Top-K 融合 kernel**
  - **一句话解释**：MiniMax 的 verify top-k、DSA（DeepSeek Sparse Attention）的 packed prefill top-k、DCP8 的 fused indexer top-k，今天有 3 个 commit 都在优化"从海量 key 里挑 top-k"这个操作。
  - **为什么社区在关注**：长上下文 + 稀疏注意力 + 投机解码是大模型推理的三个热点，而它们共同的瓶颈都是"top-k 选择"。把 top-k 融进 kernel、用直方图 + Lamport 通信做分布式 top-k，是性能优化的前沿。
  - **对新手意味着什么**：想学 GPU 性能优化，top-k/直方图/多卡通信融合是最好的练手项目——既有算法（近似 vs 精确），又有系统（多卡同步），还有硬件细节（shared memory、PTX）。

- **关键词：扩散模型（Diffusion）推理优化**
  - **一句话解释**：今天有 4 个 commit 在优化扩散模型：FA3 单序列走 batch kernel、LongCat 复用 timestep 条件、Qwen-Image 2.1 打包 BF16 gate/up 投影、Flux3 让 TP/SP rank 对 sequential CFG 达成一致。
  - **为什么社区在关注**：文生图/视频模型的推理成本极高，社区在用"复用中间结果"、"打包权重减少 GEMM 次数"、"多卡决策一致性"等手段抠性能。
  - **对新手意味着什么**：扩散模型的推理优化和 LLM 有相似之处（kernel 融合、权重打包），但也有独特问题（CFG 双分支、timestep 条件、多卡采样一致性），是很好的交叉学习方向。

- **关键词：调度器 / 批处理正确性（Scheduler Refactor）**
  - **一句话解释**：scheduler 重构，让一个 step 的结果只在"仍在 batch 里的请求行"上提交，避免已 abort 的请求被错误提交。
  - **为什么社区在关注**：推理框架的正确性 bug 往往藏在并发/异步/overlap 的边角，比如"请求已经 abort 了但结果晚一拍才到"。这类重构说明框架在向更严谨的状态管理演进。
  - **对新手意味着什么**：分布式推理框架最难的不是 kernel，而是状态机。想参与框架开发，先理解"请求生命周期"和"异步结果处理"。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：量化格式的数据布局陷阱（column-major scale）**
  - **是什么**：量化时，除了把权重压成 FP4/FP8，还需要存"缩放因子"（scale）。scale 的内存布局（行优先还是列优先）必须和下游 kernel 的期望一致，否则数值全错。
  - **为什么重要**：Commit 14 修复的就是这个——aiter 的 MXFP4 量化 kernel 期望 scale 是列优先的 `(N/32, M)` 转置视图，而 fake tensor 实现写成了 `(M, N/32)`，导致 `torch.compile` 的 `assert_size_stride` 直接报错。这种 bug 在 CPU 上测不出来，只有上 GPU 才炸。
  - **动手试试**：
    ```python
    import torch
    # 模拟 aiter 期望的 column-major scale
    M, N = 37, 2080
    scale_correct = torch.empty(((N + 31) // 32, M), dtype=torch.uint8).T
    print("shape:", scale_correct.shape)   # (37, 65)
    print("stride:", scale_correct.stride())  # (1, 37)  <- 列优先！

    # 错误写法
    scale_wrong = torch.empty((M, (N + 31) // 32), dtype=torch.uint8)
    print("wrong stride:", scale_wrong.stride())  # (65, 1) 行优先
    ```
  - **预期结果**：你会看到 `(1, 37)` 的 stride，说明第 0 维（行）在内存里是连续的——这就是 column-major。理解这一点，你就能看懂为什么 Inductor 会 assert 失败。

- **知识点 2：MoE 专家路径的路由检测（dtype 决定走哪条路）**
  - **是什么**：大模型的 checkpoint 可能是 BF16、FP8、MXFP4 等不同格式，框架需要读 safetensors 的 header 判断"这个模型的专家权重是什么类型"，然后选择对应的计算路径。
  - **为什么重要**：Commit 2 修复了一个真实 bug——训练导出的 BF16 DeepSeek-V4 没有 `quantization_config`，框架就回退到环境变量默认值（假设是 FP4），结果把 BF16 权重送进 MXFP4 路径，精度直接崩。检测函数必须显式处理 BF16/F16/F32。
  - **动手试试**：
    ```python
    import json, struct, os, tempfile

    def make_fake_ckpt(dtype):
        d = tempfile.mkdtemp()
        header = json.dumps({
            "model.layers.0.mlp.experts.0.w1.weight": {"dtype": dtype, "shape": [2048, 4096]}
        }).encode()
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        return d

    # 读 header 判断 dtype
    for dt in ["BF16", "F16", "F32", "U8", "F8_E4M3"]:
        path = make_fake_ckpt(dt)
        with open(os.path.join(path, "model.safetensors"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        print(dt, "->", list(hdr.values())[0]["dtype"])
    ```
  - **预期结果**：你会看到 safetensors 文件开头就是 8 字节长度 + JSON header，框架就是靠这个"偷看"权重类型的。理解了这点，你就能自己写模型格式探测工具。

- **知识点 3：FA3 的 batch kernel vs varlen kernel**
  - **是什么**：FlashAttention 有两种调用方式——`flash_attn_func`（batch 模式，每个序列等长）和 `flash_attn_varlen_func`（变长模式，用 cu_seqlens 记录每条序列的起止）。varlen 更通用但有额外的索引开销。
  - **为什么重要**：Commit 10 发现"只有一条活跃序列时，varlen 的索引开销是纯浪费"，于是直接走 batch kernel，快 4%。这是一个经典的"识别退化情况"优化。
  - **动手试试**：
    ```python
    import torch
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    B, S, H, D = 1, 1024, 8, 64
    q, k, v = (torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    # varlen 方式：一条序列，cu_seqlens = [0, S]
    cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    out_varlen = flash_attn_varlen_func(q[0], k[0], v[0], cu, cu, S, S)

    # batch 方式
    out_batch = flash_attn_func(q, k, v)[0]

    print("max diff:", (out_varlen - out_batch).abs().max().item())
    ```
  - **预期结果**：两者数值几乎一致（可能差 1e-3 量级），但 batch 方式少了 cu_seqlens 的索引计算。如果你用 `torch.cuda.Event` 计时，会发现 batch 略快。

- **知识点 4：分布式 top-k 的 Lamport 通信模式**
  - **是什么**：多卡做 top-k 时，每个 rank 只持有部分 key，需要交换信息才能确定全局 top-k。Lamport 方式是指"先写自己的消息到共享缓冲区，然后轮询对方的缓冲区"，用简单的内存读写代替复杂的 collective 通信。
  - **为什么重要**：Commit 19 用 Lamport 模式做 DCP8 的 fused top-k——768 字节的消息，384 个线程各轮询一个 16 字节向量，比 NCCL all-gather 更轻量。这是"用共享内存 + 轮询替代通信原语"的经典技巧。
  - **动手试试**（单机模拟 Lamport 交换）：
    ```python
    import torch

    # 模拟两个 rank 的 Lamport 交换：rank0 写 buffer，rank1 轮询
    buffer = torch.zeros(8, dtype=torch.int32, device="cuda")

    # rank0: 写入消息（用非零值表示"已写好"）
    buffer[0:4] = torch.tensor([1, 2, 3, 4], device="cuda")

    # rank1: 轮询直到所有字非零
    while not torch.all(buffer[0:4] != 0):
        pass
    print("received:", buffer[0:4].tolist())
    ```
  - **预期结果**：你会看到 rank1 等到 rank0 写完才读到数据。真实 kernel 里用 `__threadfence` / `.release` / `.acquire` 保证内存序，这里用 PyTorch 简化演示了"轮询等待"的思想。

- **知识点 5：调度器的"提交只在存活行"原则**
  - **是什么**：推理框架里，一个 batch 的 forward 是异步的，结果晚一拍才回来。如果这期间某个请求被 abort 了，结果就不能再提交到它身上。
  - **为什么重要**：Commit 3 的重构把 `commit_decode(batch)` 改成 `commit_decode(reqs, forward_iter)`，只对"仍在 batch 里的请求"提交。这是并发正确性的核心——"结果属于发起它的那个时刻的请求，而不是现在 batch 里的请求"。
  - **动手试试**（伪代码理解）：
    ```python
    # 旧逻辑：遍历 batch 里的所有请求
    for req in batch.reqs:
        if req.aborted: continue
        commit(req, result)   # bug: result 可能是 abort 前发起的

    # 新逻辑：只提交"发起 forward 时还在 batch 里"的请求
    for req in reqs_at_forward_time:
        if req.aborted: continue
        commit(req, result)
    ```
  - **预期结果**：理解"时间快照"的概念——`reqs` 是 forward 发起时刻的请求列表，`batch.reqs` 是当前时刻的列表，两者可能不同。这是异步系统的通用设计模式。

---

## 🔥 重要更新

- **Commit 3（scheduler 重构）**：把"提交 step 结果"的范围从"当前 batch 的所有请求"收窄到"发起 forward 时仍在 batch 的请求"，修复了 abort 请求被错误提交的并发 bug。这是框架正确性的核心重构，影响所有推理场景。
- **Commit 2（DeepSeek V4 BF16 路由修复）**：未量化的 BF16/F16/F32 checkpoint 不再被误送进 MXFP4 专家路径，修复了训练导出模型直接精度崩坏的问题。
- **Commit 6（MiniMax-M3.1-Flash 支持 MI350X/MI355X）**：AMD gfx950 上正式支持 MiniMax-M3.1-Flash，同时删除了 aiter 的 asm block-scale GEMM（约 5000 行），说明 AMD 路径在收敛到更统一的实现。
- **Commit 19（DCP8 fused top-k）**：新增 1912 行的 CUDA kernel，用直方图 + Lamport 通信做 8 卡 decode 的全局 top-k，是今天技术含量最高的性能优化。

---

## 📋 逐条解读

### 1. [Perf] [AMD] minimax: decode indexer CP for DSpark verify on the NVFP4 index cache (#46694)
- **代码层面**：改的是 `verify_topk_gfx950.py`，在 `_split_plan` 里新增 `WORLD`/`RANK` 参数，用 `_local_range` 计算"本 rank 负责的 block 范围"（`l * WORLD + RANK`），让 top-k 的候选 block 按 rank 切分。文档字符串从"最多 16 个 draft 行"更新为"最多 16，CP 下 32"。
- **新手概念课堂**：想象 8 个人一起从 1000 本书里挑最好的 10 本。如果每个人独立挑，最后还要合并；更聪明的做法是"按页码分工"——rank 0 只看第 0、8、16… 页，rank 1 看第 1、9、17… 页，这样每个 block 只被一个人看，合并时不会重复。
- **对你有什么影响**：AMD 用户跑 MiniMax 投机解码时，多卡（CP）场景下 top-k 更快、更省内存。普通用户感知不到，但长上下文 + 投机解码的吞吐会提升。

### 2. [Fix] deepseek_v4: route unquantized BF16/F16/F32 checkpoints off the MXFP4 expert path (#44813)
- **代码层面**：改的是 `configs/deepseek_v4.py` 的 `try_detect_fp4_experts`，在 `F8_E4M3 -> False` 之后新增 `BF16/F16/F32 -> False` 分支。新增测试文件 `test_deepseek_v4_fp4_detect.py`，用伪造的 safetensors header 验证各 dtype 的检测结果。
- **新手概念课堂**：就像快递分拣——包裹上写着"易碎"就走特殊通道，写着"普通"就走普通通道。如果包裹上什么都没写，你不能默认它是易碎品（否则普通包裹被当易碎品处理，反而出事）。这里"没写"= 没有 `quantization_config`，不能默认走 FP4。
- **对你有什么影响**：如果你用自己训练/导出的 BF16 DeepSeek-V4 模型，以前可能直接报错或精度崩坏，现在能正确走未量化路径。修复了一个真实的用户痛点。

### 3. [Refactor] scheduler: Commit a step's result only on the rows of requests still in the batch (#49019)
- **代码层面**：改的是 `beam_search/coordinator.py` 和 `managers/scheduler.py`。`commit_decode(batch)` 改为 `commit_decode(reqs, forward_iter)`，遍历 `reqs`（forward 发起时的请求）而非 `batch.reqs`（当前 batch）。删除了 `newly_finished` 返回值，新增 `still_in_batch` 回调传给 `SchedulerBatchResultProcessor`。
- **新手概念课堂**：就像餐厅点单——你点了菜（发起 forward），菜做好时（结果返回）你已经走了（abort），服务员不应该把菜端到你原来的位置（错误的请求）。正确做法是"菜只端给还在座位上的人"。
- **对你有什么影响**：高并发场景下，abort 请求不会收到"幽灵结果"，避免了状态错乱、重复输出等问题。对普通用户是稳定性提升。

### 4. [Fix] test: Keep a passed precision baseline row pointing at its own tensors (#49128)
- **代码层面**：改的是 `test/precision_baseline_store.py` 的 `push_run`。原来 `skip_tensors = tensors_already_present and not force`，现在加了条件 `meta["pass_label"] == "failed"`——只有"失败"的 push 才允许跳过 tensor 上传，因为"通过"的 push 会被 fetch 当作基线，必须上传自己的 tensor。新增 `ci_runner` 字段记录 runner 名。
- **新手概念课堂**：就像考试存档——如果上次考试不及格，这次补考及格了，你不能说"反正卷子还在，直接用上次的"（上次是错的答案）。及格的那次必须用自己的卷子存档。
- **对你有什么影响**：CI 精度回归测试的基线更可靠，不会因为"复用了失败 push 的 tensor"而把错误数值当成基线。对普通用户无感，对维护者很重要。

### 5. [Test] kv_canary: Ignore EOS in the PD fixture for the SWA model (#49028)
- **代码层面**：改的是 `test/kv_canary/pd_fixture.py`，在 `send_parallel_short_requests` 里加 `ignore_eos=self.model_mode == "swa"`。注释说明 greedy gemma-4-E2B-it 在这些 prompt 上可能第一个 token 就 EOS，导致 D（decode）没有 forward 可验证 KV 传输。
- **新手概念课堂**：就像测试快递——你寄了一个包裹，收件人一看到就签收了（EOS），你根本没机会检查包裹内容（decode forward）。所以测试时要"强制收件人多看几眼"（ignore_eos）。
- **对你有什么影响**：纯测试修复，保证 KV cache 传输测试在 SWA 模型上真的能跑到 decode 阶段。

### 6. [Feat] [AMD] minimax: Support MiniMax-M3.1-Flash on MI350X / MI355X (gfx950) (#46693)
- **代码层面**：涉及 `hipcc.py`（简化 launch 的 grid 处理）、删除 `aiter_asm_blockscale.py`（133 行）、以及大量 minimax 相关文件。核心是让 MiniMax-M3.1-Flash 的 kernel 在 gfx950 上能跑，同时清理了不再需要的 aiter asm GEMM 路径。
- **新手概念课堂**：就像给一个新游戏做主机适配——不只是"能启动"，还要针对这台主机的显卡特性（gfx950 的 MFMA 指令、shared memory 大小）调优。删掉的 asm GEMM 说明"有了更好的替代方案，旧轮子可以扔了"。
- **对你有什么影响**：AMD MI350X/MI355X 用户可以直接跑 MiniMax-M3.1-Flash 了。这是 AMD 生态逐渐成熟的一个信号。

### 7. Revert "[Perf] [NVIDIA] unquant: Autotune low-M BF16 GEMMs via FlashInfer mm_bf16 (#43703)" (#49118)
- **代码层面**：回滚了之前的 FlashInfer `mm_bf16` 自动调优。恢复了一堆 `_splitk_tactic`、`_run_splitk_dense`、`_BF16_SPLITK_TUNED_TACTICS`（GB300 TP16 的调优表）等代码，`bf16_gemm_backend` 的帮助文本也改回去了。
- **新手概念课堂**：就像"新菜谱试了一周，发现还不如老菜谱稳"，于是撤回。Revert 不代表新方案没价值，而是"当前版本不够稳定/收益不够大，先回滚，以后再上"。
- **对你有什么影响**：NVIDIA 用户如果用 `--bf16-gemm-backend cutedsl` 且遇到问题，这次回滚可能修复了你的问题。低 M 的 BF16 GEMM 回到更保守的路径。

### 8. [Test] kv-canary: Serve the EAGLE canary fixture in bf16 to avoid an fp16 overflow (#49097)
- **代码层面**：改的是 `test/kv_canary/mode_config.py`，给 EAGLE canary 配置加 `--dtype bfloat16`。注释说明 fp16 checkpoint 在重复 `</s>` 的 draft 上会溢出成 NaN，触发 verify-logits 的 NaN assert。
- **新手概念课堂**：fp16 的最大值约 65504，bf16 的最大值约 3.4e38。就像用小杯子（fp16）装水，水多了就溢出；换大杯子（bf16）就不会。测试里用 bf16 避免"杯子太小"的假阳性。
- **对你有什么影响**：纯测试修复，避免 EAGLE canary 测试因 fp16 溢出而误报。

### 9. [Fix] megamoe: Accept FlashInfer 0.7.3 knob cache and enum config fields in startup autotune (#49096)
- **代码层面**：改的是 `flashinfer_megamoe_autotune.py`。`_NATIVE_CACHE_VERSION` 从 1 改为 2（匹配 FlashInfer 0.7.3 的缓存格式），`_geometry_key` 新增 `enum.Enum` 处理（用 `.name` 而非值，保证跨进程稳定）。
- **新手概念课堂**：就像 API 升级——对方把返回格式从 v1 改成 v2，你这边必须同步更新，否则解析失败。枚举字段不能直接 JSON 序列化，要用 `.name` 转成字符串。
- **对你有什么影响**：升级到 FlashInfer 0.7.3 的用户，MegaMoE 的启动自动调优不再报错。

### 10. [Perf] diffusion: Run a single packed FA3 sequence through the batch kernel (#47696)
- **代码层面**：改的是 `flash_attn.py` 的 `forward_varlen`。检测到 `cu_seqlens == (0, query.shape[0])`（只有一条序列）时，走 `flash_attn_varlen_func` 的 batch 模式（传 `cu_seqlens_q=None`），并把 padding 行清零。新增单测 `TestFlashAttentionSingleSequence`。
- **新手概念课堂**：就像寄快递——如果只有一个包裹，走"批量寄送"流程（填一堆表单）反而更慢，直接走"单件寄送"更快。代码里识别"只有一条序列"这个退化情况，切换到更快的路径。
- **对你有什么影响**：扩散模型（比如 H3 的 H200 形状）推理快 4%。普通用户感知不到，但生成速度略有提升。

### 11. [Perf] longcat-image: Reuse timestep conditioning for serial CFG (#48688)
- **代码层面**：改的是 `longcat_image.py`。新增 `_longcat_modulation` 函数，用 `modulation_cache` 缓存 `linear(silu(emb))` 的结果。`_LongCatAdaLayerNormZero.forward` 新增 `modulation_cache` 参数，serial CFG（先跑 conditional 再跑 unconditional）时复用 timestep 条件。
- **新手概念课堂**：CFG（classifier-free guidance）要跑两遍模型——一遍带条件、一遍不带。但 timestep 的 embedding 和调制参数（shift/scale/gate）在两遍里是一样的，没必要算两次。就像做两道菜都要切同样的葱，切一次就够了。
- **对你有什么影响**：LongCat-Image 生成图片时，serial CFG 场景下更快。普通用户感知为"出图稍快"。

### 12. [Perf] qwen-image21: Pack BF16 gate/up projections on SM120 (#47090)
- **代码层面**：改的是 `qwen_image21.py` 的 `pack_gate_up_weights`。在 SM120（Blackwell）上，如果 gate/up 都是未量化的 BF16、无 bias、TP=1、形状是 `(12288, 4096)`，就把两个权重拼成一个 `[2F, C]` 的连续 tensor，一次 GEMM 算完。新增 `fused_packed_silu_mul_bitexact` 做打包的 SiLU-mul。
- **新手概念课堂**：就像两个快递合并成一个包裹——gate 和 up 是两个独立的 GEMM，合并成一个更大的 GEMM 能减少 kernel 启动开销、提高 GPU 利用率。但要注意"合并后舍入可能变"，所以只在特定形状（1024/4096 token）启用。
- **对你有什么影响**：SM120（比如 RTX 5090）用户跑 Qwen-Image 2.1 生成图片时，FFN 部分更快。文档里也加了说明。

### 13. [Perf] dsa: Run the packed prefill top-k on CUDA (#45891)
- **代码层面**：改的是 `dsa_topk_backend.py` 和 `dsa_backend.py`。`packed_extend_topk` 的判断从"只在 ROCm/Kimi-K3.1 启用"扩展为"默认 top-k 模式下所有 PAGED extend 都启用"，但 prefill-CP shard 需要显式 `packed_extend_topk` 才走。新增 `prefill_cp_sharded` 元数据字段。
- **新手概念课堂**：就像"新算法先在部分场景试用，验证稳定后推广到所有场景"。packed top-k 之前在 ROCm 上验证过，现在推广到 CUDA 的默认路径，但 CP shard 这种特殊场景仍保守。
- **对你有什么影响**：DSA（DeepSeek Sparse Attention）的 prefill 在 CUDA 上更快，尤其是用 trtllm + fp8 KV cache 的场景。

### 14. [Fix] [AMD] quark: Match aiter's column-major scale layout in the MXFP4 quant fake (#47908)
- **代码层面**：改的是 `quark_w4a4_mxfp4.py` 的 `_aiter_dynamic_mxfp4_quant_fake`。`blockscale` 从 `(M, (N+31)//32)` 改为 `((N+31)//32, M).T`，即列优先视图。新增测试验证 `scale.stride() == (1, m)`。
- **新手概念课堂**：就像书架——你可以按"每层从左到右"排（行优先），也可以按"每列从上到下"排（列优先）。aiter 的 kernel 假设是列优先，fake tensor 必须报告同样的 stride，否则 `torch.compile` 的 Inductor 会 assert 失败。
- **对你有什么影响**：AMD 用户用 Quark MXFP4 + `torch.compile` 时不再报 stride 错误。修复了一个真实的编译期 bug。

### 15. [Perf] [AMD] qwen3: Fuse the 1x128 FP8 input quant into RMSNorm and SiLU-and-mul for block-FP8 linears on gfx950 (#48043)
- **代码层面**：新增 `silu_mul_fp8_group_quant.py`（Triton kernel），在 `smallm_fp8_blockscale_gfx950/__init__.py` 新增 `prefers_bf16_input` 和 `_BEATS_PREQUANTIZED_MAX_M` 表。当 M 很小（比如 ≤4）时，让 kernel 直接吃 bf16 输入（自己量化），比"上游先量化好"更快。
- **新手概念课堂**：就像做菜——"先切好所有菜再炒"（上游量化）vs"边切边炒"（融合量化）。小批量时，融合更快（省了一次内存往返）；大批量时，分开更快（量化 kernel 能充分利用并行）。这里用小 M 的形状表来决策。
- **对你有什么影响**：AMD gfx950 用户跑 Qwen3 的 block-FP8 线性层时，小 batch 场景更快。这是"融合 vs 分离"的经典权衡。

### 16. [Fix] diffusion/flux3: Make TP and SP ranks agree on sequential CFG (#46430)
- **代码层面**：改的是 `flux3_image.py` 的 `resolve_sequential_cfg`。在决定"是否用 sequential CFG"前，用 `all_reduce(MIN)` 让所有 TP/SP rank 的 `free` 内存达成一致。新增 `_sequential_cfg_worker` 测试，用 2 卡模拟 TP/SP 场景。
- **新手概念课堂**：就像 8 个人投票决定"走哪条路"——如果每个人看自己的地图（本地显存）投票，可能 4 人选 A 路、4 人选 B 路，队伍就散了。正确做法是"先同步信息（all_reduce），再统一决策"。
- **对你有什么影响**：Flux3 在多卡（TP/SP）场景下，sequential CFG 的决策一致，避免"有的卡跑 packed CFG、有的卡跑 sequential CFG"导致 collective 死锁。

### 17. [Perf] [NVIDIA] deepseek-v4.1: Fuse BF16 prefill indexer score epilogue (#45807)
- **代码层面**：新增 `index_score_reduce.py`（Triton kernel），把"32 个 head 的 score 加权求和"融进一个 kernel。用 4 个交错累加器（`acc0..acc3`）精确匹配 Torch `Reduce.cuh` 的 `vt0=4` 累加顺序，保证 bit-exact。
- **新手概念课堂**：就像"32 个人报数求和"——如果按 1、2、3… 顺序加，浮点误差会累积；如果 4 个人一组交错加（0+4+8…, 1+5+9…），误差分布更均匀，且和 PyTorch 的累加顺序一致，结果 bit-exact。
- **对你有什么影响**：DeepSeek-V4.1 的 prefill indexer 在 NVIDIA 上更快，且数值和原来完全一致（bit-exact），不会引入精度回归。

### 18. [Fix] [AMD] aiter: Quantize q with its own scale in the fp8 prefills over a gathered cache (#48434)
- **代码层面**：改的是 `aiter_backend.py` 和 `aiter_utils.py`。新增 `_quantize_q_fp8(q)` 方法，返回 `(q_fp8, q_scale)`。原来用 `layer.k_scale` 作为 q 的 descale（错误），现在用 q 自己的 scale。
- **新手概念课堂**：就像"用自己的尺子量自己的身高"——q 和 k 的量化 scale 是独立的，不能拿 k 的 scale 去反量化 q。用错 scale 会导致注意力分数整体偏移，输出错误。
- **对你有什么影响**：AMD 用户用 fp8 KV cache + gathered cache 的 prefill 场景，输出正确了。修复了一个数值正确性 bug。

### 19. [Perf] [NVIDIA] dsa/dcp: Fused DCP8 indexer top-k for decode (#48895)
- **代码层面**：新增 1912 行的 `dcp_fused_topk.cuh`。核心流程：采样直方图 → 过滤候选 → 精细直方图 → Lamport 交换 768 字节消息 → 决策 → 输出。失败的 certificate 走精确 fallback（3 轮 all-gather 直方图）。
- **新手概念课堂**：就像 8 个人各自有一堆卡片，要找出全局最大的 10 张。做法是：每个人先粗筛（直方图），把"可能是前 10"的卡片编号广播给所有人（Lamport 交换），每个人独立算出全局 top-10（决策一致），最后各自输出自己负责的卡片。
- **对你有什么影响**：DeepSeek Sparse Attention 在 8 卡 decode 场景下，top-k 选择更快。这是今天技术含量最高的 kernel，涉及直方图、Lamport 通信、精确 fallback。

### 20. [Fix] mnnvl: Match FlashInfer's arch-specific CuTe DSL AR profiles (#46929)
- **代码层面**：改的是 `flashinfer_mnnvl_cutedsl.py` 的 `_config_for_shape`。新增 `compute_capability` 参数，用 `default_config.find(...)` 按"形状 + 架构"查找 profile，而不是只按形状。日志里也打印 `sm_XX`。
- **新手概念课堂**：就像买鞋——同样的尺码（形状），但不同品牌（架构）的鞋楦不同。FlashInfer 的 profile 是"形状 + 架构"绑定的，只按形状匹配可能拿到"给别的架构调的参数"，性能不对甚至报错。
- **对你有什么影响**：MNNVL（多节点 NVLink）用户用 CuTe DSL 的 AllReduce 融合时，profile 匹配更准确，避免"用了别的架构的调优参数"。

---

*以上解读基于各 commit 的 diff 和提交说明，部分被截断的 diff 细节（如 Commit 6 的 5000+ 行删除、Commit 19 的 2763 行 kernel）按可见部分解读。*

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)