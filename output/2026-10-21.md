# 🚀 vLLM & SGLang 每日更新 — 2026-10-21

> 自动生成于 2026-10-21 06:31 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：多模态缓存竞态（Multimodal Cache Race）**
  - 一句话通俗解释：当多个请求同时渲染图片/视频并读写同一份缓存时，如果"失效"发生在"渲染"中途，缓存可能被写回旧数据——就像你在擦黑板时别人又写上了字。
  - 为什么社区在关注：vLLM 正把多模态（图像、视频、音频）当作一等公民，缓存一致性从"单线程离线"转向"高并发在线服务"，竞态问题开始成为 bug 主战场。
  - 对新手意味着什么：并发编程 + 缓存一致性是 AI 推理框架的核心难点，掌握 asyncio/threading 同步原语（Event、Lock）能让你在框架层脱颖而出。

- **关键词：异构平台适配（XPU / ROCm / CPU / GB200）**
  - 一句话通俗解释：同一份 vLLM 代码要跑在 Intel XPU、AMD ROCm、CPU AMX、NVIDIA Blackwell 上，每个平台后端能力不同，需要"分流路由"。
  - 为什么社区在关注：今天 20 个 commit 里有 6 个直接跟平台相关（XPU 注意力后端、ROCm 编译加速、CPU AMX 调度、GB200/GB300 配置、AITER 升级），说明"一套代码多硬件"是刚需。
  - 对新手意味着什么：懂一点 C++/CUDA/HIP、懂 CMake 构建、懂后端抽象层（backend registry），在国产/异构芯片公司非常吃香。

- **关键词：编译期并行化（Build Parallelism）**
  - 一句话通俗解释：把一个巨大的 .cu 文件拆成十几个小文件，让编译器的多个核心同时干活，7.6 倍加速。
  - 为什么社区在关注：AI 框架编译动辄 30 分钟起，开发者体验（DX）正在成为框架竞争力的一部分。
  - 对新手意味着什么：CMake、编译单元（TU）拆分、链接期优化是"看不见但很值钱"的技能。

- **关键词：MoE 路由专家捕获（Routed Expert Capture）**
  - 一句话通俗解释：MoE 模型每个 token 只激活少数专家，捕获"哪个 token 走了哪个专家"对分析、调试、投机解码都很重要。
  - 为什么社区在关注：MoE 已成为大模型主流架构（DeepSeek、Kimi、Qwen），可观测性需求暴涨。
  - 对新手意味着什么：懂 MoE 原理 + 分布式 gather/scatter，是做推理优化的敲门砖。

- **关键词：类型检查与测试基建（MyPy / Test Infra）**
  - 一句话通俗解释：给 Python 测试代码加类型注解，让 mypy 静态检查抓出 `None` 解引用这类低级错误。
  - 为什么社区在关注：今天有 3 个 commit 专门修 mypy 错误，说明 vLLM 在从"能跑"走向"工程化"。
  - 对新手意味着什么：写测试时养成加 `assert x is not None` 和类型注解的习惯，能少踩一半的坑。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：缓存失效与渲染的竞态（asyncio + threading 混合同步）**
  - **是什么**：异步代码（asyncio）和线程（threading）混用时，需要用 `asyncio.Event` 和 `threading.Event` 桥接两个世界，保证"失效操作"等到"正在进行的渲染"完成后再执行。
  - **为什么重要**：多模态服务里，一张图处理可能耗时几百毫秒，如果此时缓存被清空，渲染结果会写回一个"已失效"的槽位，导致后续请求读到脏数据。不懂会踩"偶发输出错误图片描述"的坑。
  - **动手试试**：
    ```python
    import asyncio, threading, time

    cache = {"value": "old"}
    render_started = asyncio.Event()
    render_done = threading.Event()

    def slow_render():
        # 模拟线程中耗时渲染
        time.sleep(0.5)
        cache["value"] = "new"
        render_done.set()

    async def main():
        loop = asyncio.get_running_loop()
        # 启动渲染线程
        loop.run_in_executor(None, slow_render)
        render_started.set()
        # 失效方等待渲染完成再清缓存
        await loop.run_in_executor(None, render_done.wait)
        cache["value"] = None
        print("cache after invalidation:", cache)

    asyncio.run(main())
    ```
  - **预期结果**：输出 `cache after invalidation: {'value': None}`。若把 `render_done.wait()` 去掉，可能看到 `{'value': 'new'}`（渲染结果覆盖了失效），这就是竞态。

- **知识点 2：注意力后端的"复合路由"（Triton/FlashAttention Composite）**
  - **是什么**：同一个模型里，有的 batch 需要"多模态前缀双向掩码"（只能走 Triton），有的 batch 是普通因果注意力（可以走更快的 FlashAttention）。复合后端按 batch 动态选路。
  - **为什么重要**：一刀切用 Triton 慢，一刀切用 FA 又会算错多模态。选错后端 = 要么性能腰斩，要么结果错误。
  - **动手试试**：
    ```python
    import torch
    from vllm.platforms import current_platform

    # 查看当前平台支持的注意力后端
    print("platform:", current_platform.device_name)
    print("is_xpu:", current_platform.is_xpu())
    print("is_rocm:", current_platform.is_rocm())

    # 伪代码：模拟选择逻辑
    use_mm_prefix = True
    head_size = 256
    if use_mm_prefix and current_platform.is_xpu():
        print("→ 走 Triton/FlashAttention 复合后端")
    else:
        print("→ 走单一后端")
    ```
  - **预期结果**：在 XPU 上看到"复合后端"提示；在其他平台看到单一后端。理解"能力探测 → 路由选择"的框架设计模式。

- **知识点 3：CMake 编译单元拆分（Translation Unit Splitting）**
  - **是什么**：把 `attention.cu`（一个巨型文件）拆成 12 个按 (dtype, kv_dtype, head_size) 组合命名的 .cu 文件，每个文件只实例化一种 kernel，让 `ninja` 多核并行编译。
  - **为什么重要**：CUDA/HIP 的模板实例化是编译热点，单文件串行编译 10 分钟。拆分后 7.6 倍加速。同时用 CMake 脚本在配置期校验"每个实例化文件都被 dispatch 引用"，防止死代码。
  - **动手试试**：
    ```bash
    # 观察 ninja 并行度（在 vllm 源码根目录）
    cd /path/to/vllm
    # 只 build ROCm extension 目标
    cmake -B build -DVLLM_TARGET_DEVICE=rocm 2>&1 | tail -20
    # 看生成的目标文件数量
    ls build/csrc/rocm/*.o 2>/dev/null | wc -l
    # 用 -j 控制并行度对比耗时
    time ninja -C build -j1 _rocm_C
    time ninja -C build -j8 _rocm_C
    ```
  - **预期结果**：`-j8` 明显快于 `-j1`；`.o` 文件数量与拆分后的 .cu 数量对应。理解"编译期并行"和"链接期合并"的权衡。

- **知识点 4：MoE 路由行捕获的平台钩子（Platform Hook for Routed Expert Rows）**
  - **是什么**：在 MoE 层，不同平台对 router 输入做了不同的 padding/shard，导致"token → expert"的映射行布局不同。新增 `get_routed_experts_local_rows` 钩子让平台自定义返回本 rank 的路由行。
  - **为什么重要**：调试 MoE 负载均衡、做专家级 profile、实现投机解码都需要精确知道"哪个 token 走了哪个专家"。默认 DP/TP 恢复逻辑在自定义平台会错位。
  - **动手试试**：
    ```python
    import torch
    from vllm.platforms.interface import Platform

    class MyPlatform(Platform):
        @classmethod
        def get_routed_experts_local_rows(cls, ids, context, dp_rank, tp_size):
            # 自定义：直接返回前 N 行，跳过默认 gather
            return ids[:3].to(torch.int32)

    topk = torch.tensor([[7, 8], [9, 10], [11, 12], [13, 14]], dtype=torch.int32)
    rows = MyPlatform.get_routed_experts_local_rows(topk, None, 0, 2)
    print(rows)  # 期望: [[7,8],[9,10],[11,12]]
    ```
  - **预期结果**：输出前 3 行，验证钩子可以绕过默认的 TP gather + trim 逻辑。

- **知识点 5：协作式 TopK 的精确溢出恢复（Exact Recovery on Candidate Overflow）**
  - **是什么**：GPU 上做 TopK 时，先用 10-bit 直方图"粗筛"出阈值 bin，如果该 bin 内候选数超过容量（`kCoarseTieCapacity`），需要从溢出缓冲里精确恢复，而不是丢弃。
  - **为什么重要**：采样温度、top-p、投机解码都依赖 TopK 精确性。溢出时静默丢弃会导致"相同输入不同输出"的诡异 bug。
  - **动手试试**：
    ```python
    import torch
    from vllm import _custom_ops  # 假设已编译

    logits = torch.randn(1, 4096, device="cuda")
    values, indices = torch.topk(logits, k=2048)
    # 构造大量并列值，触发阈值 bin 溢出
    logits[0, :3000] = 0.0
    try:
        v, i = _custom_ops.cooperative_topk(logits, 2048)
        print("topk ok, sum:", v.sum().item())
    except RuntimeError as e:
        print("溢出保护触发:", e)
    ```
  - **预期结果**：在大量并列值时仍能返回精确的 2048 个索引（含并列），而不是报错或返回不足。

---

## 🔥 重要更新

1. **[ROCm][Build] Compile paged attention kernels in parallel (7.6x faster)** —— AMD 用户构建时间从 ~10 分钟降到 ~1.3 分钟，是今天对开发者体验影响最大的改动。
2. **[Bugfix] Include KV cache quantization settings in IPC weight cache key** —— 修复了"预加载权重缓存命中错误 KV 量化配置"的隐蔽 bug，可能影响所有用 `vllm preload` + FP8 KV 的用户。
3. **[Model] Add Nemotron Parse 3.5 serving and MTP support** —— 新增 NVIDIA 文档解析模型支持，含 MTP（多 token 预测），对做 OCR/文档理解的新手是开箱即用的新玩具。
4. **[Bugfix] Make cooperative TopK exact on candidate overflow** —— 修复 TopK 在并列值溢出时结果不精确，影响采样正确性，属于"静默错误"类高危 bug。

---

## 📋 逐条解读

### 1. [Bugfix] Fix multimodal sender-cache invalidation race with queued renders (#62164)
- **代码层面**：`tests/multimodal/test_cache.py` 新增 `test_cache_drift_invalidation_waits_for_render_without_delaying_outputs`，用 `asyncio.Event` 标记"渲染已进入"，用 `threading.Event` 阻塞"失效方"，验证失效操作会等待已排队的渲染完成，同时不阻塞输出流水线。核心改动在 `vllm/multimodal/cache.py`（diff 截断部分）的失效路径。
- **新手概念课堂**：想象图书馆的"归还-上架"流程：读者还书时如果管理员正在把这本书放到书架，你得等他放完再贴"已下架"标签，否则书架上会出现一本"已下架但还在"的书。
- **对你有什么影响**：多模态在线服务（图像/视频输入）在并发高时不会再偶发"描述错图"，普通用户可能感知不到，但服务稳定性提升。

### 2. [Bugfix][KV Offload] Capture event metadata for pending promotions (#59855)
- **代码层面**：`vllm/v1/kv_offload/` 的 offloading scheduler 在 `_maximal_prefix_lookup` 返回 pending 时，提前捕获事件元数据（block_size、ownership 等），避免后续 inventory 变化导致事件丢失。测试覆盖 `never_seen/evicted/retained/secondary_only` 四种历史 × 成功/失败两种结果。
- **新手概念课堂**：KV Offload 像"内存不够时把书暂存到仓库"；pending promotion 是"仓库里的书正在搬回书架"。如果搬运途中仓库清单变了，你得先记下"这本书原本属于哪个批次"，否则搬回来时标签丢了。
- **对你有什么影响**：使用 KV 卸载（CPU/二级存储）的长上下文服务，事件追踪更可靠，监控面板不再丢事件。

### 3. [Model] Add Nemotron Parse 3.5 serving and MTP support (#61651)
- **代码层面**：新增 `RadioMBartPrefixForCausalLM` 模型类，注册到 `docs/models/supported_models.md`；`examples/generate/multimodal/vision_language_offline.py` 新增 `run_nemotron_parse_35`，演示 `<pad>` + `<predict_bbox>` 等特殊 token 的用法；测试文件 `test_nemotron_parse.py` 更新。
- **新手概念课堂**：Nemotron Parse 是"文档解析模型"，输入一张扫描件，输出带 bbox 的 Markdown。MTP（Multi-Token Prediction）是"一次预测多个 token"，像打字时输入法一次联想出整个短语。
- **对你有什么影响**：可以直接 `vllm serve nvidia/Nemotron-3.5-Parse` 做 PDF/图片转结构化文本，无需自己拼 prompt。

### 4. [Perf] Optimize Molmo2 pooling preparation (#57101)
- **代码层面**：`vllm/model_executor/models/molmo2.py` 新增 `_prepare_molmo2_pooling`，把"按 token_pooling 索引 gather + 掩码平均"从 Python 循环改为向量化张量操作，并支持 `masked_average` 开关。测试验证 batch 索引保持、无效组丢弃。
- **新手概念课堂**：原来像"一个一个捡豆子"，现在像"用筛子一次筛完"。pooling 是把图像 patch 特征聚合成少量 token 的步骤，是视觉编码器的最后一公里。
- **对你有什么影响**：Molmo2 多图/视频输入的预处理更快，GPU 利用率更高。

### 5. [Doc] Kimi-K3: DCP selector and TOKENSPEED_MLA draft backend on GB200/GB300 (#62798)
- **代码层面**：`recipes/models/moonshotai/Kimi-K3.yaml` 新增 GB200/GB300 的 `--speculative-config`（draft 用 `TOKENSPEED_MLA` 后端），新增 `decode_context_parallel` 选择器（`sizes: gb200/gb300: [8, 16]`，`match_tp: true`），更新 guide 文档。
- **新手概念课堂**：DCP（Decode Context Parallel）是把长上下文的 KV cache 切到多卡上，像"把一本厚词典拆成几册分给几个人查"。`match_tp: true` 表示 DCP 大小必须等于 TP 大小。
- **对你有什么影响**：在 GB200/GB300 上部署 Kimi-K3 长上下文推理，按 recipe 选 DCP8/DCP16 即可，不用自己算兼容性。

### 6. [CPU] Improve long-context MTP attention scheduling on AMX (#59082)
- **代码层面**：`csrc/cpu/cpu_attn_impl.hpp` 重构 `AttentionWorkItemGroup`，新增 `kv_head_idx`、`q_head_start`、`q_head_num`、`reduction_id` 字段，把"按 KV 头 + Q 头分块"的调度粒度做细，配合 AMX 的 tile 大小；benchmark 传 `kv_cache_dtype`。
- **新手概念课堂**：AMX 是 Intel CPU 的矩阵加速指令。长上下文 MTP 时，注意力计算像"分块搬砖"，块切得不好（太大/太小）都会浪费 AMX 的算力。
- **对你有什么影响**：CPU 上跑长上下文 + 投机解码更快，尤其 Intel Sapphire Rapids 及以上。

### 7. [ROCm][Build] Compile paged attention kernels in parallel (7.6x faster attention build) (#62475)
- **代码层面**：`CMakeLists.txt` 把 `csrc/rocm/attention.cu` 拆成 `attention_dispatch.cu` + 12 个 `attention_paged_attn_h{64,128}_{f16,bf16}_{auto,fp8_f16,fp8_fp8}.cu`；新增 CMake 脚本在配置期扫描每个实例化文件里的 `ATTN_ENTRY(...)`，若 dispatch 未引用则 `FATAL_ERROR`。
- **新手概念课堂**：像把一本 1000 页的菜谱拆成 12 本小册子，12 个厨师同时做，而不是 1 个厨师从头做到尾。CMake 的静态校验像"上菜前检查每道菜是否在菜单上"。
- **对你有什么影响**：AMD 用户 `pip install` 或本地编译 vLLM 的时间大幅缩短，CI 也更快。

### 8. [XPU][CI] Skip test_deep_sleep_lora_tp2 on non-CUDA platforms (#62679)
- **代码层面**：`tests/basic_correctness/memory/sleep_mode/test_sleep_mode.py` 给 `test_deep_sleep_lora_tp2` 加 `@pytest.mark.skipif(not current_platform.is_cuda_alike(), ...)`。
- **新手概念课堂**：LoRA 的 state 在 CUDA/ROCm 上放在 runtime pool，XPU 上没有这个机制，测试必然失败。跳过不是"掩盖 bug"，而是"承认平台差异"。
- **对你有什么影响**：XPU CI 不再红，普通用户无感。

### 9. [Bugfix] Include KV cache quantization settings in IPC weight cache key (#61758)
- **代码层面**：`WeightCacheKey.from_model_config` 新增 `cache_config` 参数，把 `kv_cache_dtype` 和 `kv_cache_dtype_skip_layers` 纳入 hash；daemon 侧 `cache_config` 改名为 `cache_key`；测试补 `CacheConfig()`。
- **新手概念课堂**：IPC 权重缓存是"进程间共享已加载的模型权重"。如果两个进程一个用 FP8 KV、一个用 auto KV，权重本身相同但 KV 量化不同，缓存 key 不区分就会串味——像两个人口味不同却共用一个饭盒。
- **对你有什么影响**：用 `vllm preload` + 不同 KV 量化配置时不再加载错权重，避免"输出乱码"类诡异问题。

### 10. [Docs] Add @TheEpicDolphin as codeowner for spec decode (#62742)
- **代码层面**：`.github/CODEOWNERS` 新增 `/vllm/v1/worker/gpu/spec_decode` 的 owner。
- **新手概念课堂**：CODEOWNERS 是"这块代码谁负责 review"，像小区楼栋的楼长。
- **对你有什么影响**：无直接功能影响，但说明投机解码模块在扩张，PR review 会更专业。

### 11. [XPU][Attention] Route multimodal prefix masks through the Triton/FlashAttention composite (#62014)
- **代码层面**：`vllm/platforms/xpu.py` 的 `get_attn_backend_cls` 在 `use_mm_prefix` 且 `dtype != float32` 时返回 `TRITON_FLASH_ATTN` 复合后端；`flash_attn.py` 的 `supports_head_size` 在 XPU 上放宽到 512；`model_executor/models/config.py` 对 Gemma4 异构 head_dim 在 XPU 上不强制 TRITON_ATTN。
- **新手概念课堂**：多模态前缀掩码是"图像 token 之间可以互相看"（双向），普通因果注意力是"只能看前面"。Triton 支持双向，FlashAttention 更快但只支持因果，复合后端按 batch 切换。
- **对你有什么影响**：XPU 上跑 Gemma4 等多模态模型，纯文本 batch 走 FA 更快，多模态 batch 走 Triton 更准。

### 12. [Feature][MoE] Add platform hook for routed-expert capture rows (#62624)
- **代码层面**：`vllm/platforms/interface.py` 新增 `get_routed_experts_local_rows(ids, context, dp_rank, tp_size)` 默认返回 `None`；`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` 在 `capture` 时优先调用该钩子，`None` 时走原 DP/TP gather 逻辑。
- **新手概念课堂**：MoE 的 router 输出是"每个 token 选哪几个专家"。不同平台对 router 输入做了 padding，导致行号对不上。钩子让平台"自己报数"，避免框架猜错。
- **对你有什么影响**：用 MoE 模型 + 自定义硬件时，专家捕获（用于调试/投机解码）更准确。

### 13. [ROCm][Bugfix] Fix wrong output of the QK-norm + RoPE + KV-cache fusion on ROCM_ATTN (#62403)
- **代码层面**：`tests/compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py` 新增 `test_qk_norm_rope_kvcache_fusion_interleaves_every_layer`，验证 3 层模型全部设置 `_use_interleaved_v_cache=True`；`test_rope_kvcache_fusion.py` 参数化 `interleaved_v`。
- **新手概念课堂**：QK-norm + RoPE + KV-cache 融合 kernel 把 V 以"交错"格式写入缓存。如果只有第一层注册了交错模式，后面层仍按非交错读，就会读到错位的 V——像一群人约定"从左到右"排队，结果有人"从右到左"。
- **对你有什么影响**：ROCm + AITER + 融合优化开启时，输出不再错乱。

### 14. [XPU][CI] Add kaldi-native-fbank to XPU test requirements (#62616)
- **代码层面**：`requirements/test/xpu.in` 和 `.txt` 新增 `kaldi-native-fbank`。
- **新手概念课堂**：这是语音特征提取库（fbank），FireRedASR2 测试需要。
- **对你有什么影响**：XPU 上跑语音识别测试不再因缺依赖失败。

### 15. [MyPy] Fix mypy errors in KV connector tests (kv-B) (#55901)
- **代码层面**：`tests/v1/kv_connector/` 下多个测试文件加 `assert output.prompt_token_ids is not None`、`assert out0.kv_transfer_params is not None`，引入 `_as_mock` 辅助函数。
- **新手概念课堂**：mypy 是 Python 静态类型检查器。`Optional[T]` 类型不 `assert is not None` 就直接 `.attr`，mypy 会报错，运行时也可能 `AttributeError`。
- **对你有什么影响**：无功能变化，但测试代码更健壮，未来重构不易踩 `None` 坑。

### 16. [Frontend] DiffusionGemma joint reads on /v1/systemone and /v1/decisions, images on /v1/systemone (#62437)
- **代码层面**：`docs/models/structured_decisions_models.md` 新增 "Joint reads" 章节；`examples/features/structured_diffusion/structured_server.py` 去掉 `--max-logprobs 32`；新增 `tests/entrypoints/generate/structured_decisions/test_joint_e2e.py`。
- **新手概念课堂**：扩散模型（DiffusionGemma）一次去噪可以读多个问题，像"一张答题卡上同时涂多个选项"。`joint: true` 让多个问题共享一个 canvas，答案可能互相影响。
- **对你有什么影响**：结构化决策 API 支持批量问题联合读取，吞吐更高；图片输入也支持了。

### 17. [ROCm] Bump AITER to v1.26.0 (#62486)
- **代码层面**：`docker/Dockerfile.rocm_base` 和 `rocm_72_base` 的 `AITER_BRANCH` 从 v0.1.25 升到 v0.1.26；`test_rocm_aiter_moe.py` 新增 `swigluoai` 激活的参考实现。
- **新手概念课堂**：AITER 是 AMD 的 AI 算子库，类比 NVIDIA 的 cuBLAS/cuDNN。`swigluoai` 是 SwiGLU 的一个变体（带 clamp）。
- **对你有什么影响**：ROCm 用户获得新算子支持，Mi350 MXFP4 W4A16 精度测试通过。

### 18. [MyPy][5/N] Fix mypy errors in hard tests/ dirs and enforce follow-imports=silent (#62142)
- **代码层面**：`tests/entrypoints/` 下大量测试去掉冗余 `# type: ignore`，加 `assert not isinstance(response, ErrorResponse)`，`chat_template_content_format` 加 `# type: ignore[misc]`；`mypy.ini` 启用 `follow-imports=silent`。
- **新手概念课堂**：`follow-imports=silent` 表示"检查当前文件，但不深入检查 import 进来的模块"，避免测试目录被第三方库类型噪音淹没。
- **对你有什么影响**：无功能变化，但 vLLM 的 CI 类型检查更严格，贡献代码时要注意类型注解。

### 19. [MyPy] Fix mypy errors in easy tests/v1/kv_connector files (kv-A) (#55542)
- **代码层面**：`test_spec_decode_acceptance.py` 用 `vllm.tokenizers.get_tokenizer` 替代 `transformers.AutoTokenizer`，加 `cast(list[str], ...)`；多个测试加 `assert vllm_config.kv_transfer_config is not None`；`_make_vllm_config` 的 `kv_role` 类型改为 `KVRole`。
- **新手概念课堂**：`cast` 是"我保证这是这个类型"的显式声明，比 `# type: ignore` 更精确。`KVRole` 是 Literal 类型，比 `str` 更安全。
- **对你有什么影响**：无功能变化，KV connector 测试代码质量提升。

### 20. [Bugfix] Make cooperative TopK exact on candidate overflow (#59645)
- **代码层面**：`csrc/cooperative_topk.cu` 把 `kTieWsPerRow` 统一为 `ct::kCoarseTieCapacity`（= kMaxTopK）；`cooperative_topk.cuh` 引入 `ExactRadixTraits`，`dsmem_hist_reduce` 改为模板参数 `NumBins` 并支持 > CTA 大小的直方图，`tie_buffer` 容量从 `kMaxTies` 改为 `kCoarseTieCapacity`。
- **新手概念课堂**：TopK 先"粗筛"（10-bit 直方图），阈值 bin 里可能有大量并列值。原来只保留 `kMaxTies` 个，超了就丢；现在保留 `kMaxTopK` 个，再用精确基数排序恢复。像考试排名先按分数段粗分，同分段再精确排。
- **对你有什么影响**：采样（top-k/top-p）在大量并列 logits 时结果更精确，避免"相同输入不同输出"的非确定性 bug。

## sglang
## 💡 今日关键词

- **关键词：AMD/ROCm 深度优化**
  - **一句话通俗解释**：越来越多 commit 专门针对 AMD 的 MI300/MI350（gfx950）做 kernel 级调优，比如 small-M FP8 GEMM、extend attention tile、mHC boundary tile。
  - **为什么社区在关注**：这说明 SGLang 正把 AMD 当作一等公民来对待，不再只是"能在 AMD 上跑"，而是"在 AMD 上跑得最快"。异构计算（NVIDIA + AMD + NPU + TPU）正在成为推理框架的标配。
  - **对新手意味着什么**：如果你会写 Triton/HIP kernel，或者熟悉 ROCm 生态，这是巨大的机会——大量性能优化 PR 需要人手，而且门槛比 CUDA 生态略低，竞争也小。

- **关键词：统一缓存 / 外部 KV 存储（Unified Cache / Direct Linker）**
  - **一句话通俗解释**：把 KV cache 从 GPU 显存"溢出"到主机内存或外部存储（如 Mooncake），让长上下文、多轮对话的会话能复用历史 KV。
  - **为什么社区在关注**：Agent 场景（coding agent、多轮工具调用）上下文越来越长，GPU 显存根本不够。Commit 1、11 都在扩展 unified cache 支持更多模型结构（packed draft layers、FULL+MAMBA MLA）。
  - **对新手意味着什么**：KV cache 管理是推理系统的核心难点，学会 HiCache、Mooncake、paged attention 这些概念，是进入大模型推理工程的关键敲门砖。

- **关键词：量化（FP8 / MXFP4 / W4A8 / NVFP4）**
  - **一句话通俗解释**：把模型权重和激活从 16 位浮点压到 8 位甚至 4 位，省显存、提吞吐，但要小心数值精度。
  - **为什么社区在关注**：Commit 2、3、17 都在处理量化细节——MoE 激活量化、block-scale GEMM、gate/up 顺序对齐。量化已经从"能跑"进入"每个细节都要对"的阶段。
  - **对新手意味着什么**：量化是性价比最高的推理加速手段。搞懂 per-tensor / per-block / per-channel scale 的区别，以及量化 kernel 的布局要求，能让你在面试和实战中都占优势。

- **关键词：投机解码（Speculative Decoding）**
  - **一句话通俗解释**：用小模型先"猜"几个 token，大模型再一次性验证，猜对了就省时间。
  - **为什么社区在关注**：Commit 7 修复了 verify 后 KV 搬移的目标地址问题，Commit 14 修复了 NPU 上 draft 模型的 all_reduce group。投机解码是当前提升 decode 吞吐的主流手段，但正确性 bug 很隐蔽。
  - **对新手意味着什么**：投机解码涉及 KV cache、调度、并行通信的交叉，是理解推理系统全链路的好切入点。

- **关键词：Mamba / 混合状态空间模型**
  - **一句话通俗解释**：Mamba 用状态空间模型（SSM）替代 attention，线性复杂度，适合超长序列；但它的"状态"需要像 KV cache 一样管理。
  - **为什么社区在关注**：Commit 6、16 都在处理 Mamba 状态的 checkpoint 追踪和高带宽拷贝。Kimi-K3、GLM 等新模型开始混合使用 Mamba 和 attention。
  - **对新手意味着什么**：Mamba 是 attention 之外的新范式，理解它的状态管理（ping-pong buffer、checkpoint）是前沿技能。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：Triton Kernel 的分块（Tiling）与 persistent grid

- **是什么**：GPU kernel 把大矩阵切成小块（tile），每个 thread block 处理一块。Persistent grid 是指启动固定数量的 block，让它们循环处理所有工作项，而不是一个工作项启动一个 block。
- **为什么重要**：Commit 16（mamba state copy）和 Commit 18（mHC boundary tile）都用到了这个技巧。不懂 tiling，你写的 kernel 要么寄存器溢出，要么访存效率极低；不懂 persistent grid，小 batch 场景下启动开销会吃掉所有收益。
- **动手试试**：
```python
import torch, triton, triton.language as tl

@triton.jit
def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)

n = 1024 * 1024
src = torch.randn(n, device="cuda")
dst = torch.empty_like(src)
grid = (triton.cdiv(n, 1024),)
copy_kernel[grid](src, dst, n, BLOCK=1024)
print(torch.allclose(src, dst))  # True
```
  然后试着把 `BLOCK` 改成 128、256、4096，用 `torch.cuda.Event` 测时间，观察性能变化。
- **预期结果**：你会看到 BLOCK 太小（128）时启动 block 太多、开销大；BLOCK 太大（4096）时寄存器压力大、可能变慢。中间某个值最快——这就是 tiling 的权衡。

### 知识点 2：KV Cache 的分页（Paged Attention）与外部存储

- **是什么**：把 KV cache 切成固定大小的"页"（page），像操作系统管理内存一样按需分配，而不是给每个请求预留最大长度的连续空间。
- **为什么重要**：Commit 1、11 都在扩展 unified cache 的 pool 组装逻辑。不懂 paged KV，你就无法理解为什么 `req_to_token` 是一个二维表、为什么 `out_cache_loc` 是离散的 slot 索引。
- **动手试试**：
```python
# 模拟 paged KV：page_size=4，每个请求的 token 映射到物理 slot
page_size = 4
# req_to_token: [num_reqs, max_tokens] -> 物理 slot 索引
req_to_token = torch.full((2, 16), -1, dtype=torch.int32)
# 请求 0 有 5 个 token，分配 slot 0-3 和 8-9
req_to_token[0, :5] = torch.tensor([0, 1, 2, 3, 8])
# 请求 1 有 3 个 token，分配 slot 4-6
req_to_token[1, :3] = torch.tensor([4, 5, 6])
print(req_to_token)
# 取出请求 0 第 4 个 token 的物理 slot
print(req_to_token[0, 4].item())  # 8
```
- **预期结果**：你会看到同一个请求的 token 在物理上不连续，但通过 `req_to_token` 表可以任意寻址。这就是 paged attention 的核心：逻辑连续，物理离散。

### 知识点 3：量化中的 per-tensor / per-block scale

- **是什么**：量化需要 scale 把浮点数映射到整数。Per-tensor 是整个张量一个 scale；per-block 是每 128 个元素一个 scale（如 FP8 block-scale GEMM）。
- **为什么重要**：Commit 3 的 small-M FP8 block-scale GEMM 要求 `w_scale.shape == (N // 128, K // 128)`，Commit 17 修复了 MoE gate/up 的 global scale 顺序。scale 的形状和顺序错了，结果会静默错误（不报错但精度崩）。
- **动手试试**：
```python
import torch
# 模拟 per-block 量化：128x128 的块共享一个 scale
def quantize_block(x, block=128):
    M, N = x.shape
    x_blocks = x.view(M // block, block, N // block, block)
    scales = x_blocks.abs().amax(dim=(1, 3)) / 127.0
    q = (x_blocks / scales[:, None, :, None]).round().clamp(-127, 127)
    return q.view(M, N).to(torch.int8), scales

x = torch.randn(256, 256)
q, s = quantize_block(x)
print(q.shape, s.shape)  # torch.Size([256, 256]) torch.Size([2, 2])
# 反量化
deq = (q.view(2, 128, 2, 128) * s[:, None, :, None]).view(256, 256)
print((deq - x).abs().max())  # 误差应该很小
```
- **预期结果**：你会看到 scale 的形状是 `(M/128, N/128)`，反量化后误差在量化步长范围内。如果 scale 顺序搞反（比如 gate/up 互换），误差会集中在某些通道上。

### 知识点 4：投机解码的 verify 窗口

- **是什么**：投机解码时，draft 模型生成 K 个候选 token，target 模型一次 forward 验证这 K 个 token，接受连续正确的前缀。verify 窗口就是这 K 个 token 在 `out_cache_loc` 中的位置。
- **为什么重要**：Commit 7 修复了一个隐蔽 bug——verify 完成后，scheduler 可能已经把已完成请求的 `req_to_token` 行分配给了新请求，如果此时再读 `req_to_token` 搬 KV，就会搬到错误的地方。正确做法是从 verify 窗口（`out_cache_loc`）读目标地址。
- **动手试试**：
```python
import torch
# 模拟：batch=2, 每个请求 verify 窗口 4 个 token
bs, window = 2, 4
out_cache_loc = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
num_correct_drafts = torch.tensor([2, 1])  # 请求 0 接受 2 个 draft，请求 1 接受 1 个

# 从窗口读目标地址（正确做法）
window_view = out_cache_loc.view(bs, -1)
tgt = torch.stack([window_view[i, :num_correct_drafts[i] + 1] for i in range(bs)])
print(tgt)  # tensor([[10, 11, 12], [20, 21]])

# 错误做法：从 req_to_token 读，可能已被新请求覆盖
req_to_token = torch.tensor([[10, 11, 12, 13, 99, 99], [20, 21, 22, 23, 99, 99]])
seq_lens = torch.tensor([0, 0])
# 如果此时 req_to_token 被改写，结果就错了
```
- **预期结果**：你会看到从窗口读的地址是确定的，而从 `req_to_token` 读依赖调度状态。这就是为什么 Commit 7 要改成从窗口读——消除对调度时序的依赖。

### 知识点 5：MoE 的 expert parallelism 与 expert map

- **是什么**：MoE 模型有很多 expert，expert parallelism（EP）把 expert 分散到不同 GPU 上。每个 GPU 只持有部分 expert 的权重，需要一张 expert map 把全局 expert id 映射到本地 id。
- **为什么重要**：Commit 12 的测试验证了 gfx1250 W4A8 在 EP 下的 expert map 构造：`build_ep_expert_map(num_experts=4, num_local_experts=2, ep_rank=1)` 返回 `[-1, -1, 0, 1, -1]`，表示全局 expert 0/1 不在本 rank（-1），全局 expert 2/3 映射到本地 0/1，末尾 -1 是 padding。
- **动手试试**：
```python
import torch
def build_ep_expert_map(num_experts, num_local_experts, ep_rank, device="cpu"):
    num_ep = num_experts // num_local_experts
    start = ep_rank * num_local_experts
    expert_map = torch.full((num_experts + 1,), -1, dtype=torch.int32, device=device)
    for i in range(num_local_experts):
        expert_map[start + i] = i
    return expert_map

print(build_ep_expert_map(4, 2, 0).tolist())  # [0, 1, -1, -1, -1]
print(build_ep_expert_map(4, 2, 1).tolist())  # [-1, -1, 0, 1, -1]
```
- **预期结果**：你会看到每个 rank 只认识自己负责的 expert，其他都是 -1。路由时如果 token 被分到 -1 的 expert，就需要 all-to-all 发送到对应 rank。

---

## 🔥 重要更新

- **Commit 11（unified_cache 支持 FULL + MAMBA MLA，Kimi-K3）**：这是功能性的重大扩展，让 unified cache 能服务混合架构模型。Kimi-K3 这类新模型用户直接受益，也是外部 KV 存储走向通用的关键一步。
- **Commit 7（spec：verify 后 KV 搬移目标从 verify 窗口取）**：修复了一个会导致 KV 写错 slot 的正确性 bug，且新增了针对性测试。投机解码用户必看，这类 bug 极难排查。
- **Commit 3（AMD small-M FP8 block-scale GEMM 融合激活量化）**：新增了一个完整的 HIP kernel + Python 封装，是学习"如何为特定硬件写融合 kernel"的优秀范例。
- **Commit 13（移除 tokenspeed_mla 后端）**：删除了一个 attention backend 及其依赖，说明项目在收敛技术栈。使用 `--attention-backend tokenspeed_mla` 的用户需要迁移到 `cutedsl_mla`。

---

## 📋 逐条解读

### 1. [Fix] cache: Accept packed draft layer mappings in the UMBP linker (#44929)
- **代码层面**：改的是 `umbp_direct_linker.py` 里的 `_ordered_layers` 和 `_range_items`。原来假设 `layer_mapping` 的值是单个整数（一个逻辑层对应一个 buffer），现在允许值是列表（一个逻辑层对应多个 packed buffer）。同时去掉了"buffer 数量必须等于层数"的硬性检查，改成检查映射是否构成连续双射。
- **新手概念课堂**：想象你有 10 个抽屉（buffer）和 10 个标签（logical layer），原来要求一一对应。现在有些标签可以贴到多个抽屉上（比如 draft 模型的 packed layers），只要每个抽屉都被贴到、不重复贴就行。
- **对你有什么影响**：使用 UMBP 外部缓存 + draft 模型（投机解码）的用户，之前可能遇到 "buffer 数量不匹配" 的报错，现在能正常工作了。

### 2. [Fix] [AMD] dsv41: Set moe act to fp8 by default (#45480)
- **代码层面**：`aiter.py` 的 `AiterMoeQuantInfo` 新增 `fp8_activations` 字段；`fp8.py` 在构造 quant info 时根据 `is_fp4_expert` 和 layer 属性设置它；`deepseek_v4.py` 在 `_use_aiter` 且模型是 `deepseek_v41` 时打开该开关。kernel 侧在 gfx95 上若 `fp8_activations` 为真，会跳过 small-M MXFP4 kernel，改走 aiter 的 A8W4 kernel（激活 FP8、权重 4bit）。
- **新手概念课堂**：MoE 的 expert 权重是 4bit（MXFP4），但激活如果也用 4bit 精度损失大。折中方案是权重 4bit、激活 8bit（W4A8），就像用低分辨率存照片、但显示时先放大到中等分辨率再处理。
- **对你有什么影响**：在 AMD MI300/MI350 上跑 DeepSeek-V4.1 的用户，MoE 层默认走 FP8 激活，精度和速度都更好。其他模型不受影响。

### 3. [Perf] [AMD] qwen3.5: Fuse activation quant into small-M FP8 block-scale GEMM on gfx950 (#42229)
- **代码层面**：新增 `smallm_fp8_blockscale_gfx950/` 目录，包含一个 HIP kernel（`smallm_fp8_blockscale_gemm.hip`）和 Python 封装（`__init__.py`）。封装里用 `ctypes` 加载 HIP 模块，按 `(N, K)` 形状查表选择 tile 配置，并检查一堆前置条件（dtype、stride、对齐、是否 shuffled 权重、是否在编译中）。不满足就返回 `None` 走 fallback。
- **新手概念课堂**：FP8 block-scale GEMM 需要先把 bf16 激活量化成 FP8，再和 FP8 权重做矩阵乘。传统做法是"先量化再 GEMM"两个 kernel；融合后在一个 kernel 里完成，省一次显存往返。就像做菜时"切菜+下锅"合并成一个动作。
- **对你有什么影响**：Qwen3.5 在 gfx950 上小 batch（M ≤ 32）推理更快。可用 `SGLANG_ROCM_SMALLM_FP8_BS=0` 关闭。

### 4. [Doc] cookbook: Add GLM-5.3 AgentX section and refresh with eight open InferenceX PRs (#45513)
- **代码层面**：纯文档改动。`GLM-5.3.mdx` 新增 "Agentic Long-Context Deployment" 章节，引入 `AgentX` 组件和 `agentx` 数据；`qwen3.5.jsx` 更新了 InferenceX 的 PR 引用和 SHA。
- **新手概念课堂**：AgentX 是 SemiAnalysis InferenceX 项目里针对 agent 负载（长 prompt、多轮、前缀复用）的部署配置集合。文档把这些配置整理成可复制的启动命令。
- **对你有什么影响**：想部署 GLM-5.3 做 coding agent 的用户，可以直接从文档复制完整启动命令，不用自己摸索 router、KV offload、镜像的组合。

### 5. [Perf] [AMD] attention: Restore the gfx950 extend tile for head_dim <= 128 (#45099)
- **代码层面**：`extend_attention.py` 的 `_get_block_sizes_for_extend_attention` 里，原来 gfx95 只在 `128 < Lq <= 256` 时用 `(BLOCK_M, BLOCK_N) = (128, 64)`，现在放宽到 `Lq <= 256`（即 head_dim ≤ 128 也适用）。
- **新手概念课堂**：extend attention（prefill）时每个 workgroup 要读整个 prefix 的 KV。BLOCK_M 是 query 方向的分块，BLOCK_M 越大，每个 workgroup 处理的 query 越多，读 KV 的次数越少。就像一次搬 128 块砖比一次搬 64 块更省往返。
- **对你有什么影响**：MI350X 上 head_dim 64/128 的模型 prefill 更快。数值结果不变（只是分块方式变了，归约顺序没变）。

### 6. [Fix] mamba: Keep prefill rows' checkpoint tracking through mixed chunked prefill (#45435)
- **代码层面**：`schedule_batch.py` 的 `ReqKvInfo` 新增 `mamba_untracked_decode_batch_idx` 字段；`mix_with_running` 在合并 batch 前保存 prefill 行的 track 状态，合并后把 decode 行的 track 状态设为 -1（不追踪）；新增 `mamba_lazy_undo_prealloc_at_boundary` 方法，把 decode batch 在混合 forward 前预分配的 track slot 归还。
- **新手概念课堂**：Mamba 的状态需要定期"存档"（checkpoint），以便回滚。混合 batch（prefill + decode 一起跑）时，decode 行的存档逻辑和 prefill 不同——decode 的存档在混合 forward 里不会写，所以要提前把预分配的 slot 还回去，否则会泄漏。
- **对你有什么影响**：使用 Mamba 混合模型 + chunked prefill 的用户，之前可能遇到状态泄漏或 checkpoint 错乱，现在修复了。

### 7. [Fix] spec: Take post-verify KV move targets from the verify window (#45491)
- **代码层面**：`spec_utils.py` 的 `move_accept_tokens_to_target_kvcache` 不再从 `batch.req_to_token_pool.req_to_token` 读目标地址，而是从 `batch.out_cache_loc.view(bs, -1)`（verify 窗口）读。新增测试文件验证这个行为。
- **新手概念课堂**：verify 完成后要搬 KV，目标地址是"接受的那几个 token 在 target KV cache 里的位置"。这些位置在 verify 窗口里就有，不需要再查 `req_to_token`。查 `req_to_token` 有风险：scheduler 可能在 verify 的 read-done 事件后把已完成请求的行分配给新请求。
- **对你有什么影响**：投机解码用户，之前偶发的"KV 写到错误 slot"bug 被修复。这类 bug 表现为生成结果偶发乱码，很难复现。

### 8. [Fix] disaggregation: Keep a requeued optimistic prefill request's multimodal inputs (#45440)
- **代码层面**：`prefill.py` 的 `optimistic_release_and_requeue` 不再调用 `_release_aborted_request`（会丢弃多模态输入），改为调用 `tree_cache.finish(..., CacheRequestOutcome.ABORT)`。新增测试验证 requeue 后 `multimodal_inputs` 还在。
- **新手概念课堂**：optimistic prefill 是 PD 分离（prefill-decode disaggregation）里的重试机制。请求 prefill 失败后被 requeue 重试，如果此时把图片输入丢了，重试时就会用占位符 id 去查 embedding，直接崩溃。
- **对你有什么影响**：PD 分离 + 多模态（图片/视频）用户，之前重试时可能崩溃，现在修复了。

### 9. [Fix] rust-server: Drop the fixed 64-item multimodal cap in favor of --limit-mm-data-per-request (#45432)
- **代码层面**：`driver.rs` 删除 `MAX_ITEMS_PER_REQUEST = 64` 常量和对应检查；`config.py` 把 `limit_mm_data_per_request` 从 dict 改成 list of pairs（保持配置顺序，决定多模态超限时的报错顺序）；README 更新说明。
- **新手概念课堂**：Rust 多模态处理器原来硬编码"一个请求最多 64 个媒体项"，但 Python 前端没有这个限制，导致行为不一致。现在统一由 `--limit-mm-data-per-request` 控制，按模态分别限制。
- **对你有什么影响**：用 Rust server 处理多图请求的用户，不再被 64 项硬限制卡住；字节预算（1.25 GiB）仍然生效。

### 10. [Refactor] sgl-router: Move experimental/sgl-router to top-level sgl-router (#45414)
- **代码层面**：把 `experimental/sgl-router/` 整个目录移到顶层 `sgl-router/`；更新所有 CI workflow、Dockerfile、脚本里的路径引用；workflow 名称从 "Nightly Experimental sgl-router Docker Image" 改为 "Nightly sgl-router Docker Image"。
- **新手概念课堂**：这是"毕业"——一个实验性项目成熟到可以独立了。就像大学里的实验室项目变成正式产品线。
- **对你有什么影响**：如果你在脚本里硬编码了 `experimental/sgl-router` 路径，需要改成 `sgl-router`。用官方 Docker 镜像的用户不受影响（镜像名会更新）。

### 11. [Feat] unified_cache: Support FULL + MAMBA MLA models (Kimi-K3) in the direct external linker (#44977)
- **代码层面**：`cache_init_params.py` 新增 `sidecar_draft_device_pools` 字段（draft KV pool 共享 target 的 slot 但分开存储）；`hybrid_pool_assembler.py` 新增 `build_direct_linker_pool_group`；`linker_pool_assembler.py` 给 `_mha_device_pool_entry` 加 `indices_from_pool` 参数，给 `_mamba_device_pool_entry` 加 unified memory 和 int8 checkpoint 的不支持检查。
- **新手概念课堂**：Kimi-K3 这类模型同时有 FULL attention 层和 MAMBA 层，还有 MLA（multi-head latent attention）。unified cache 要把这些不同结构的 KV pool 组装成一个统一的视图，对外部 linker 暴露。sidecar 就像"副驾驶"——draft 模型有自己的 KV pool，但索引指向 target 的 slot。
- **对你有什么影响**：用 Kimi-K3 + 外部 KV 存储（Mooncake 等）的用户，现在可以启用 unified cache 了。混合 Mamba 模型暂不支持 unified memory 和 int8 checkpoint。

### 12. [Test] [AMD] moe: Expect gfx1250 W4A8 layers to build with expert parallelism (#45476)
- **代码层面**：测试从"期望抛 ValueError"改为"期望正常构建"，并验证 `build_ep_expert_map(num_experts=4, num_local_experts=2, ep_rank=1)` 返回 `[-1, -1, 0, 1, -1]`。
- **新手概念课堂**：之前 gfx1250 W4A8 的 kernel 内部用 raw logits 路由，无法处理"只持有部分 expert"的 EP 场景，所以直接报错。现在 kernel 支持通过 expert map 把全局路由映射到本地 shard，所以测试改为验证正常构建。
- **对你有什么影响**：gfx1250（AMD 新卡）上跑 W4A8 MoE + EP 的用户，之前会报 "expert parallelism" 错误，现在能正常构建了。

### 13. [Chore] attention: Remove the tokenspeed_mla backend and dependency (#44803)
- **代码层面**：删除 `tokenspeed_mla` attention backend 的所有代码、文档、配置引用；Kimi-K2.5 部署 snippet 里 B300/GB300 从 `--attention-backend tokenspeed_mla` 改为 `--attention-backend cutedsl_mla`；Kimi-K3 文档删除 tokenspeed 相关说明。
- **新手概念课堂**：tokenspeed_mla 是一个只在 Blackwell（SM100/SM12x）上可用、且强制 FP8 KV 的 MLA backend。项目决定收敛到 cutedsl_mla，减少维护负担。就像手机厂商停掉一个销量不好的型号。
- **对你有什么影响**：使用 `--attention-backend tokenspeed_mla` 的用户必须迁移到 `cutedsl_mla`。B300/GB300 上的 Kimi 用户按文档更新命令即可。`cutedsl_mla` 支持 fp8 和 bf16 KV，更灵活。

### 14. [Fix] [NPU] dspark: Fix attn_tp > 1 draft wo_b all_reduce group (#40399)
- **代码层面**：`ascend_dsv4_backend.py` 在构造 sparse metadata 时，如果 query 被 padding 到 attn_tp 的倍数，把 `ori_sparse_indices` 的尾部用 -1 填充到相同行数；`deepseek_v4.py` 给 `wo_b` 的 `ColumnParallelLinear` 加 `use_dp_attention_reduce=is_dp_attention_enabled()`；`dspark_worker_v2.py` 在 NPU 上跳过某个 DP attention 分支。
- **新手概念课堂**：NPU 上 `attn_tp > 1` 时，query 会被 padding 到 tp_size 的倍数，但 `seq_lens` 不会。如果 sparse indices 不 padding，形状对不上，kernel 会越界或算错。-1 是"跳过"标记。
- **对你有什么影响**：NPU（昇腾）上跑 DeepSeek-V4 + DSPARK 投机解码 + attn_tp > 1 的用户，之前可能崩溃或结果错误，现在修复了。

### 15. [Doc] cookbook: add TPU7x recipe for GLM-5.3-Flash (#45326)
- **代码层面**：纯文档。`GLM-5.3-Flash.mdx` 新增 TPU7x 安装说明（从源码装 sglang-torchtpu）和 "Serve on TPU7x" 章节；benchmarks 新增 TPU7x low-latency 数据（TTFT 363ms、TPOT 2.47ms、794 tok/s/GPU）。
- **新手概念课堂**：TPU7x 是 Google 的 TPU，SGLang 通过 `sglang-torchtpu` 项目支持。TPU 上的 DSA（DeepSeek Sparse Attention）走 Pallas 写的 sparse MLA kernel，MTP 走实验性的 chain spec worker。
- **对你有什么影响**：有 TPU7x 的用户现在有官方 recipe 了。注意 TPU7x 只有 Low Latency 策略，且需要从源码安装（v0.1.0-tpu 镜像太旧）。

### 16. [Perf] mamba: Copy tracked decode states at HBM bandwidth in one persistent launch (#45213)
- **代码层面**：`mamba_state_scatter_triton.py` 的 `track_mamba_state_if_needed_kernel` 从"每个 batch element 一个 block"改成 persistent grid：每个 program 读整个 mask，在 kernel 内 compact 出被追踪的行，然后按 `(row, layer, chunk)` 工作项分给 grid。新增 `_copy_track_chunk` 辅助函数，用 `tl.static_range` 循环拷贝 chunk。
- **新手概念课堂**：原来每个 batch element 启动一个 block，batch 小的时候大量 block 空转（mask 为 False）。Persistent grid 让固定数量的 block 循环处理所有有效工作项，小 batch 时只启动少量 block，大 batch 时充分利用。就像快递分拣：原来每个包裹一个分拣员，现在固定几个分拣员循环处理所有包裹。
- **对你有什么影响**：Mamba 模型 decode 时状态拷贝更快，尤其小 batch 场景。没有追踪行时只读一次 mask 向量就退出，开销极低。

### 17. [Fix] [NVIDIA] modelopt-nvfp4: Keep MoE global scales in gate/up order (#45289)
- **代码层面**：`fused_moe_triton/layer.py` 的 `_weight_loader_impl` 里，原来 ModelOptNvFp4 也走"reorder per-tensor scales"逻辑（因为 FlashInfer 假设 w31 格式），现在改为只有 Fp8MoEMethod、UnquantizedFusedMoEMethod、CompressedTensorsMxInt4MoE 走这个逻辑。`modelopt_quant.py` 的 `_compute_gemm1_alphas` 去掉 `up_first` 参数，固定 col 0 = gate、col 1 = up；删除 `_compute_trtllm_gated_act_params` 函数。
- **新手概念课堂**：MoE 的 w13 权重把 gate 和 up 两个投影拼在一起。ModelOpt NVFP4 的 `_load_w13` 已经重排了权重和 block scale，但 per-tensor global scale 保持逻辑顺序 [gate, up]。如果 loader 再重排一次，gate 和 up 的 scale 就互换了，结果静默错误。
- **对你有什么影响**：用 ModelOpt NVFP4 量化 + FlashInfer TRT-LLM MoE 的用户，之前可能遇到精度下降（gate/up scale 互换），现在修复了。

### 18. [Perf] [AMD] dsv41: faster mHC boundary Triton tile for 96+ rows, bitwise unchanged (#45124)
- **代码层面**：`mhc_boundary_hip.py` 新增 `_HC_BOUNDARY_WIDE_MIN_M = 96` 和 `_HC_BOUNDARY_WIDE` 配置（BLOCK_M=32, num_warps=4, num_stages=2, kpack=2, matrix_instr_nonkdim=16, enable_fp_fusion=False）；`_hc_boundary_partials` 根据行数选择 wide tile 或原 tile；kernel 里去掉四个 weight tile 的预取，改成在循环内按需加载（减少寄存器压力）。
- **新手概念课堂**：mHC（multi-head compression？）的 boundary kernel 在行数多时，32 行的 tile 比 16 行的 tile 更高效（每个 program 处理更多行，减少启动开销）。`enable_fp_fusion=False` 和 `-ffp-contract=off` 保证两种 tile 的浮点结果逐位相同（bitwise unchanged），这样切换 tile 不会改变数值。
- **对你有什么影响**：DeepSeek-V4.1 在 MI355X 上 96+ 行的 mHC boundary 计算更快，结果完全不变。可用 `SGLANG_HIP_MHC_BOUNDARY_WIDE_TILE` 环境变量控制。

### 19. [Doc] layer-boundary/moe: Correct guides and comments that drifted from the code (#44896)
- **代码层面**：纯文档。`expert_parallelism.mdx` 更新 EP 框架说明：新增 `FusedOpPool`，说明 fused function 优先于 permute 路径，LoRA 时跳过 fused function；`layer_boundary.mdx` 补充 `facts.py`、`row_transport.py`、`residual/ihc.py`、`residual/gated.py`、`residual/post_norm.py`、`residual/attn_bank.py`、`residual/short_conv.py` 等文件的说明。
- **新手概念课堂**：文档"漂移"是指代码改了但文档没跟上。这次是反方向——文档落后于代码，现在补齐。`FusedOpPool` 是新的 fused 函数注册表，按 (all-to-all backend, runner backend) 对注册。
- **对你有什么影响**：读文档学 EP 框架或 layer boundary 的新手，能看到和代码一致的说明，不会被过时信息误导。

### 20. [Refactor] moe: Derive routing inputs from the layer and active forward (#44994)
- **代码层面**：这个 commit 很大（diff 被截断），涉及 CI（Python 3.10 → 3.11 wheel）、AOT kernel（`ggml_moe_get_block_size` 不再限定 CUDA、`moe.cuh` 和 `moe_vec.cuh` 处理 `exp_idx < 0` 时写 0 而不是直接返回）以及 MoE 路由逻辑的重构。核心是让 routing 输入从 layer 和当前 forward 状态推导，而不是显式传参。
- **新手概念课堂**：MoE 路由需要知道 top_k、num_experts、expert map 等信息。原来这些参数在调用链里显式传递，容易漏传或传错。重构后从 layer 对象和 forward 上下文自动推导，减少参数传递的样板代码。
- **对你有什么影响**：普通用户感知不到。但 `exp_idx < 0` 时写 0（而不是跳过）修复了一个潜在 bug——之前 padding 的 expert 位置可能保留垃圾值，现在明确写 0。CI 升级到 Python 3.11 意味着 wheel 兼容性变化。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)