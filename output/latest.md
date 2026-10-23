# 🚀 vLLM & SGLang 每日更新 — 2026-10-23

> 自动生成于 2026-10-23 06:27 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：多模态缓存身份（Cache Identity）**
  - 一句话解释：同一张图片在不同裁剪模式下，系统要能识别出"这是不同的输入"，不能混用缓存。
  - 为什么社区在关注：DeepSeek-VL2、Pixtral、MiniCPM-OCR 等多个多模态模型同时出现 bugfix，说明多模态推理正在大规模落地，缓存复用成了性能与正确性的核心战场。
  - 对新手意味着什么：如果你学多模态，别只盯着模型结构，**输入预处理 + 缓存键设计**才是工程落地最容易翻车的地方。

- **关键词：ROCm / AMD 适配**
  - 一句话解释：把原本跑在 NVIDIA CUDA 上的内核，移植到 AMD GPU 的 ROCm 平台。
  - 为什么社区在关注：今天有 4 个 commit 直接和 ROCm 相关（Muse Spark、GDN Triton 移植、MXFP4 indexer、MiniMax-M3），说明异构计算已经是刚需，AMD 不再是"顺便支持"。
  - 对新手意味着什么：会写 Triton kernel、懂 GPU 内存层次的人，在 AMD 生态里机会正在变多。

- **关键词：量化与 Batch Invariance（批不变性）**
  - 一句话解释：不管一次处理 1 条还是 100 条请求，同一条输入算出来的结果应该完全一样，不能因为 batch 大小变化而抖动。
  - 为什么社区在关注：FP8 MoE、CPU FP8、INT8 padding 多个 commit 都涉及量化，而 batch invariance 是可复现推理和测试的基石。
  - 对新手意味着什么：量化不只是"换成低精度"，**数值一致性**才是生产环境的硬要求。

- **关键词：CPU 后端与 NUMA 亲和性**
  - 一句话解释：让每个进程跑在离它内存最近的 CPU 核上，避免跨节点访问内存变慢。
  - 为什么社区在关注：CPU 推理（DP NUMA、GELU-tanh MoE、FP8 checkpoint）连续出现，说明"没有 GPU 也要能跑大模型"的需求在增长。
  - 对新手意味着什么：懂操作系统、NUMA、OpenMP 的人，在 CPU 推理优化方向有独特优势。

- **关键词：分布式超时与共享内存广播**
  - 一句话解释：多进程之间通信时，如果对方卡住，要有超时机制及时报错，而不是无限等待。
  - 为什么社区在关注：`cpu_all_reduce_` 超时透传、shm_broadcast 忙轮询修复，都是"分布式系统稳定性"的典型问题。
  - 对新手意味着什么：分布式不是只会 `all_reduce` 就行，**超时、内存泄漏、僵尸进程**才是真实世界的坑。


## 🧪 重要知识点 & 动手实验

- **知识点 1：多模态输入的缓存键设计（Cache Key）**
  - **是什么**：多模态模型处理图片时，会先把图片编码成 embedding 并缓存起来；缓存需要一个"键"来区分不同输入。如果键设计得太粗（比如只用图片哈希），不同的裁剪模式就会撞车。
  - **为什么重要**：撞车会导致调度器拿到错误数量的 embedding，直接报 `finish_reason="error"`。这是多模态推理最隐蔽的 bug 之一——单条测试过，批量就挂。
  - **动手试试**：
    ```bash
    # 复现 DeepSeek-VL2 的缓存身份测试（需要有 GPU 和模型权重）
    pytest tests/models/multimodal/processing/test_deepseek_vl2.py -v
    ```
    或者读一下 commit 1 里新增的 `test_deepseek_vl2.py`，重点看 `test_image_cache_identity_depends_on_cropping` 这个测试函数：它用同一张图、不同 `use_uuid` 参数，断言缓存键必须不同。
  - **预期结果**：你会看到测试通过，说明"裁剪模式不同 → 缓存键不同"这条不变量被守住了。如果手动把缓存键改成只用图片哈希，测试会失败。

- **知识点 2：FP8 量化的 batch invariance**
  - **是什么**：FP8 是一种 8 位浮点格式，能省一半显存。但量化时如果按 batch 动态算 scale，batch 大小一变结果就变，这叫"批不变性"被破坏。
  - **为什么重要**：生产环境里 batch 大小是动态的（取决于并发请求数）。如果结果随 batch 抖动，就无法做确定性测试、无法复现 bug、也无法保证 A/B 对比公平。
  - **动手试试**：
    ```python
    # 简化版：感受 per-tensor vs per-token 量化的区别
    import torch
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    # per-tensor: 整个张量一个 scale
    scale_tensor = x.abs().max() / 448.0
    # per-token: 每行一个 scale
    scale_token = x.abs().amax(dim=-1, keepdim=True) / 448.0
    print("per-tensor scale:", scale_tensor.item())
    print("per-token scales:", scale_token.flatten().tolist())
    # 观察：per-token 的 scale 随行变化，per-tensor 是全局一个值
    ```
    然后去看 commit 8 里新增的 `test_batched_mm_per_tensor_weight_per_token_act`，理解"权重 per-tensor + 激活 per-token"这个组合为什么要特殊处理。
  - **预期结果**：你能说清楚为什么 per-tensor 权重 scale 在 per-token 激活下需要广播而不是索引，以及为什么错误的索引会读到别的专家的 scale。

- **知识点 3：NUMA 亲和性与 DP rank 绑定**
  - **是什么**：NUMA 是"非统一内存访问"——多路 CPU 服务器上，每个 CPU 插槽有自己的内存。进程跑在哪个核上，访问哪块内存，速度差好几倍。
  - **为什么重要**：数据并行（DP）时，每个 rank 应该绑定到自己插槽的核和内存。绑错了，性能可能掉 30% 以上，而且很难排查。
  - **动手试试**：
    ```bash
    # 在 Linux 多路服务器上查看 NUMA 拓扑
    numactl --hardware
    lscpu | grep NUMA
    # 查看某个进程的 NUMA 绑定
    taskset -cp <pid>
    ```
    然后读 commit 5 里的 `test_omp_manager_binds_each_dp_rank_to_its_assigned_node`，看它如何用 mock 构造 2 节点 × 4 物理核 × 2 逻辑 CPU 的拓扑，断言每个 DP rank 拿到不相交的核。
  - **预期结果**：你能解释"为什么 DP rank 0 和 rank 1 不能共用同一批核"，以及 `get_visible_memory_node([2, 3])` 为什么返回 `[2, 3]` 而不是全部节点。

- **知识点 4：分布式 all-reduce 的超时透传**
  - **是什么**：`all_reduce` 是所有进程把数据汇总再分发的操作。如果某个进程挂了，其他进程会一直等。超时机制让等待有上限。
  - **为什么重要**：没有超时，一个卡死的 worker 会让整个集群挂起，运维只能 kill 进程。超时透传意味着上层配置的 `cpu_distributed_timeout` 能真正传到内核层。
  - **动手试试**：
    ```python
    # 最小复现：两个进程，一个不参与 all_reduce，看超时是否生效
    # 参考 commit 14 的 tests/distributed/test_cpu_all_reduce.py
    import torch.distributed as dist
    from datetime import timedelta
    # 伪代码：
    # rank 0: dist.all_reduce(tensor, timeout=timedelta(seconds=1))
    # rank 1: sleep(5) 然后退出
    # 预期：rank 0 在 ~1 秒内抛 RuntimeError("Timed out")
    ```
    直接跑：`pytest tests/distributed/test_cpu_all_reduce.py::test_cpu_all_reduce_timeout -v`
  - **预期结果**：测试在 3 秒内结束（而不是等 5 秒），并捕获到 `Timed out` 异常。说明超时参数真的传到了底层。

- **知识点 5：忙轮询下的 ZeroMQ 通知队列内存增长**
  - **是什么**：ZeroMQ 的 `CONFLATE` 模式只保留最后一条消息，但如果 reader 从不 `poll()`，libzmq 内部仍会为每条通知排一个命令，内存无限涨。
  - **为什么重要**：这是典型的"看起来没问题、跑久了 OOM"的 bug。忙轮询（busy loop）是低延迟系统的常见模式，但必须定期"排空"通知队列。
  - **动手试试**：
    ```python
    # 读 commit 9 的测试 test_busy_reader_consumes_write_notifications
    # 核心断言：
    # 1. writer 发一条消息后，notify_socket.poll() 返回 True
    # 2. reader.dequeue() 拿到消息后，notify_socket.poll() 返回 False
    # 说明通知被消费掉了，不会累积
    pytest tests/distributed/test_shm_broadcast.py::test_busy_reader_consumes_write_notifications -v
    ```
    然后看 `record_read()` 里新增的 `SHM_NOTIFY_POLL_INTERVAL_S = 2.0` 逻辑：每 2 秒 poll 一次，用 `poll(timeout=0)` + `recv(NOBLOCK)` 排空。
  - **预期结果**：测试通过，且你能解释"为什么 `recv()` 单独调用不够，必须配合 `poll()`"——因为 `poll()` 才会让 libzmq 处理内部命令队列。


## 🔥 重要更新

1. **Commit 8（FP8 MoE batch invariance）**：这是数值正确性的核心修复。如果你的线上服务开了 `VLLM_BATCH_INVARIANT`，之前 FP8 MoE 专家层的结果会随 batch 大小漂移，现在修好了。对做可复现推理和回归测试的团队影响最大。

2. **Commit 13（MiniCPM-OCR 1.0 支持）**：新增一个完整的多模态 OCR 模型支持，含 635 行测试。做文档解析、票据识别的团队可以直接用，不用自己写 processor。

3. **Commit 17（ROCm Triton 版 GDN MTP decode kernel）**：把 CUDA 上的融合 kernel 移植到 Triton，AMD 用户终于能用上 GDN 的 MTP 加速。这是"一份代码跑两个平台"的典型范例。

4. **Commit 10（权重缓存 daemon 的 GPU 显存与 warmup 挂起修复）**：Fast Start 功能之前会让 daemon 占额外显存、warm-cache 时挂死。如果你在用 weight cache 加速冷启动，务必升级。


## 📋 逐条解读

### 1. [Bugfix] Fix DeepSeek-VL2 cache identity across cropping modes
- **代码层面**：新增两个测试文件 `test_deepseek_vl2.py`（生成）和 `test_deepseek_vl2.py`（处理）。核心是让图片缓存的键包含"裁剪模式"信息，而不是只看图片内容哈希。测试用 `(1,3)` 和 `(3,1)` 两种图片数量组合，断言第二次请求不会因为缓存撞车而报错。
- **新手概念课堂**：就像图书馆借书，如果两本不同的书贴了同一个索书号，管理员就会拿错。缓存键就是"索书号"，裁剪模式不同 = 书不同，索书号必须不同。
- **对你有什么影响**：用 DeepSeek-VL2 做多图输入时，之前可能间歇性报 `finish_reason="error"`，现在稳定了。

### 2. [MM][Mistral] Use real-valued RoPE in the compiled Pixtral ViT
- **代码层面**：`pixtral.py` 里把复数版 RoPE（`_apply_rotary_emb_vit_complex`）拆出一个实数版（`_apply_rotary_emb_vit_real`），因为 `torch.compile` 对复数运算支持不好。新增测试 `test_real_rotary_emb_vit_matches_complex` 断言两者数值一致。
- **新手概念课堂**：RoPE（旋转位置编码）可以用复数乘法表示，但编译器对复数支持差。实数版就是把 `a+bi` 拆成两个 float 分别算，数学等价但能编译。
- **对你有什么影响**：Pixtral 视觉编码器在 `torch.compile` 下能正常编译了，推理更快，且结果和 eager 模式一致。

### 3. [Frontend] Share non-streaming chat message assembly with batch derender
- **代码层面**：新增 `message_assembly.py`，把"组装 chat message 字段"的逻辑抽成公共函数 `assemble_chat_message_fields`。测试覆盖了 tool_calls 在 `enable_auto_tools=False`、`tool_choice="none"` 等情况下应被丢弃。
- **新手概念课堂**：就像做菜时把"切菜"这个步骤抽成独立工序，流式和非流式两条产线都用同一套切法，避免一边切丝一边切块。
- **对你有什么影响**：OpenAI 兼容接口的 chat 响应在不同路径下行为一致了，工具调用不会在某些路径下"漏出来"。

### 4. [Bugfix][Pooling] Validate late-interaction mode and use count in the Python API
- **代码层面**：`pooling_params.py` 给 `LateInteractionParams` 加了 `verify()` 方法，校验 `mode` 只能是 `cache_query` 或 `score_doc`，且 `cache_query` 模式下 `query_uses` 必须是正整数。之前 `None` 会让 runner 的 assert 崩溃，非正数会让缓存提前弹出。
- **新手概念课堂**：就像点外卖必须选"堂食"或"外带"，选了个"随便"系统就懵了。参数校验就是提前拦住"随便"。
- **对你有什么影响**：用 Python API 做 late-interaction 检索时，传错参数会立刻报清晰的错，而不是在 worker 里神秘崩溃。

### 5. [Bugfix][CPU] Fix DP NUMA affinity regression
- **代码层面**：`cpu_resource_utils.py` 的 `get_visible_memory_node` 支持传入 `assigned_ids`，让每个 DP rank 只看到分配给自己的 NUMA 节点。之前所有 rank 都看到全部节点，导致绑核重叠。新增 8 个测试覆盖环境变量、assigned_ids、sim 模式等组合。
- **新手概念课堂**：NUMA 节点就像公司里的两个茶水间。DP rank 0 应该去 A 茶水间，rank 1 去 B。之前大家都去 A，挤爆了。
- **对你有什么影响**：CPU 上跑数据并行推理，性能回归被修复，多路服务器上能跑满带宽。

### 6. [Bugfix] Don't enable MM embedding inputs on KV consumers in language-model-only mode
- **代码层面**：`vllm.py` 的 `_resolve_mm_embedding_inputs` 增加判断：KV consumer 在 `language_model_only=True` 且 `enable_mm_embeds=False` 时，不再强制打开 `allow_missing_mm_embeddings`。测试用参数化覆盖 4 种角色组合。
- **新手概念课堂**：P/D 分离（Prefill/Decode 分离）里，Decode 节点只做文本生成，不需要图像编码器。之前系统"好心"给它开了多模态入口，结果它根本没有处理能力。
- **对你有什么影响**：纯文本的 P/D 分离部署不会再因为多模态配置而报错或浪费资源。

### 7. [Security][CI/Build] Bump uv to 0.13.4 in CI scripts for eMudhra CA distrust
- **代码层面**：两个 CI 脚本里 `UV_VERSION` 从 `0.11.28` / `0.11.14` 升到 `0.13.4`，安装方式改成从固定版本 URL 下载。
- **新手概念课堂**：CA 证书就像身份证。eMudhra 这个"身份证颁发机构"被 distrust 了，旧版 uv 还用它的证书，就会连不上。升级 uv 就是换发新身份证。
- **对你有什么影响**：CI 构建不再因为证书问题失败；如果你本地用旧 uv，建议也升级。

### 8. [Bugfix] Make online FP8 MoE experts batch-invariant under VLLM_BATCH_INVARIANT
- **代码层面**：MoE 的 Triton kernel 里，当"权重 per-tensor scale + 激活 per-token scale"时，权重 scale 需要沿 N 维广播，而不是按专家索引。之前索引错了会读到别的专家的 scale。新增测试 `test_batched_mm_per_tensor_weight_per_token_act` 覆盖 `[E]` 和 `[E,1,1]` 两种 scale 形状。
- **新手概念课堂**：8 个专家各有各的"音量旋钮"（scale）。如果激活是逐 token 调音量，权重的旋钮应该是"全局一个"，结果代码去按 token 索引了专家旋钮，放错了声音。
- **对你有什么影响**：开 `VLLM_BATCH_INVARIANT` 时，FP8 MoE 模型的结果不再随 batch 大小变化，回归测试终于稳定了。

### 9. [Bugfix] shm_broadcast: poll the notify socket from busy readers
- **代码层面**：`shm_broadcast.py` 的 `record_read()` 里新增逻辑：每 2 秒 `poll(timeout=0)` 一次 notify socket，如果有消息就 `recv(NOBLOCK)` 排空。新增 `SHM_NOTIFY_POLL_INTERVAL_S = 2.0` 常量。
- **新手概念课堂**：ZeroMQ 的通知像快递柜的取件短信。你一直不取件，短信会一直堆在柜子里（内存涨）。定期"取件"（poll + recv）才能清空。
- **对你有什么影响**：长时间运行的高吞吐服务，共享内存广播不再缓慢泄漏内存。

### 10. [Fast Start] Fix weight cache daemon GPU memory and warm-cache warmup hang
- **代码层面**：`weight_cache/daemon.py` 里 daemon 现在会设置自己的 `parallel_config.rank`（之前都是 0，导致 torch.compile 缓存串号）；新增 `get_memory` 命令只上报"被 export 的权重字节数"，不把 daemon 私有分配算进去；修复 warm-cache warmup 挂起。
- **新手概念课堂**：daemon 就像后厨的备菜员。之前所有备菜员都用 0 号柜子（rank 0），拿错菜；而且把"自己吃的零食"也算进"给客人的菜量"（显存预算）。
- **对你有什么影响**：用 weight cache 加速冷启动时，不再额外吃显存，warmup 不再卡死。

### 11. [Bugfix][ROCm] Muse Spark: apply output multiplier and softcap in compute_logits_local
- **代码层面**：`muse_spark/amd/model.py` 新增 `compute_logits_local` 方法，把 `_apply_output_transform`（输出乘数 + tanh softcap）抽出来，让 local 路径也走同样的变换。新增测试断言 `compute_logits_local(h) == compute_logits(h)`。
- **新手概念课堂**：logits 是模型的"原始打分"，输出乘数和 softcap 是"调音台"。之前 local 路径绕过了调音台，声音不对。
- **对你有什么影响**：ROCm 上跑 Muse Spark 做投机解码（local logits）时，采样结果和标准路径一致了。

### 12. [Test][Pooling] Cover explicit right truncation for the JinaVL score token
- **代码层面**：`test_io_processor_unit.py` 的参数化增加 `truncation_side="right"` 场景，验证显式指定截断方向时，JinaVL 的 score token 仍然保留在截断后的 prompt 里。
- **新手概念课堂**：JinaVL 打分时会在 prompt 末尾加一个特殊 token。如果从右边截断，可能把这个 token 截掉，分数就废了。测试确保"截断后 token 还在"。
- **对你有什么影响**：用 JinaVL 做 rerank 时，长文本截断不会导致打分异常。

### 13. [model] Support MiniCPM-OCR 1.0
- **代码层面**：新增 `minicpm_ocr.py` 模型实现，继承 MiniCPM-V 4.6 但覆盖图像几何逻辑：`"4x"` patch-grid 对齐（merge factor 2 而非 4），以及 placeholder 数量从 `target_sizes` 读取而非重算。新增 635 行测试 `test_minicpm_ocr.py`。
- **新手概念课堂**：OCR 模型对图像分辨率特别敏感。MiniCPM-OCR 用 "4x" 下采样（2×2 合并），和 V 4.6 的 "16x"（4×4 合并）不同，切图数量算法必须跟着改。
- **对你有什么影响**：可以直接用 vLLM 跑 MiniCPM-OCR 做文档识别，不用自己写 processor。

### 14. [BugFix] Forward the configured cpu distributed timeout to `cpu_all_reduce_`
- **代码层面**：`parallel.py` 的 `sync_dp_state` 从 `@staticmethod` 改成实例方法，把 `self.cpu_distributed_timeout` 传给 `cpu_all_reduce_`。`parallel_state.py` 里 `cpu_all_reduce_` 增加 `timeout` 参数。新增测试用 2 进程、rank 1 故意不参与，断言 rank 0 在 3 秒内超时。
- **新手概念课堂**：all-reduce 就像开会等所有人到齐。之前没有"等多久就散会"的规则，一个人不来大家永远等。现在能配置"等 1 秒就报错"。
- **对你有什么影响**：CPU 分布式推理时，某个 worker 挂了不会再让整个集群无限挂起。

### 15. [CPU] Add GELU-tanh MoE, FP8 checkpoint support and Gemma-4 INT8 TP=4 padding
- **代码层面**：新增 `csrc/cpu/fp8_quant.cpp`（AVX512 优化的 FP8 量化内核），CPU MoE 支持 GELU-tanh 激活，支持直接加载 FP8 checkpoint，Gemma-4 INT8 在 TP=4 时做 padding 对齐。CMake 里把新文件加进编译列表。
- **新手概念课堂**：CPU 没有 Tensor Core，只能靠 AVX512 向量指令一条算 16 个 float。FP8 量化在 CPU 上要手写 SIMD 内核，不能直接调 cuBLAS。
- **对你有什么影响**：纯 CPU 部署 Gemma-4 / MoE 模型时，能用 FP8 省内存、用 GELU-tanh 匹配模型原始激活。

### 16. [Perf] Watermarking: skip greedy-fallback argmax when no request is greedy
- **代码层面**：`gpu_sampler.py` 的 `_sample_random` 里，先检查 `sampling_states.any_greedy(idx_mapping_np)`，如果没有任何请求是 greedy（temperature=0），就跳过 argmax 计算。测试用 `temperatures=(0.0, 1.0)` 和 `(1.0, 1.0)` 两种组合验证只有 greedy 行走 argmax。
- **新手概念课堂**：水印采样本来就要算 gumbel，如果 batch 里没人要 greedy，再算一遍 argmax 就是白干。就像做菜时没人点清蒸，就不用烧那锅水。
- **对你有什么影响**：纯采样的水印推理（temperature>0）会快一点，尤其是大 batch。

### 17. [ROCm][Perf] Triton port of the fused GDN MTP decode kernel
- **代码层面**：新增 `fused_gdn_decode_post_conv_mtp_triton`（Triton 实现），测试参数化 `impl=["cuda", "triton"]`，在 ROCm 上自动选 Triton 版。CUDA 版仍走 `ops.fused_gdn_decode_post_conv_mtp`。
- **新手概念课堂**：GDN（Gated DeltaNet）是线性注意力的一种，MTP（多 token 预测）是投机解码。融合 kernel 把卷积后处理、门控、状态更新打包成一个 kernel，减少显存往返。Triton 版让 AMD 也能跑。
- **对你有什么影响**：AMD GPU 上跑 GDN 模型的投机解码，吞吐提升，不用再 fallback 到慢路径。

### 18. [ROCm][Perf] Apply opt-in paged MXFP4 sparse indexer dense path for DeepSeek-V4
- **代码层面**：新增 `rocm_paged_mxfp4_indexer.py`，实现 DeepSeek-V4 的 MXFP4 稀疏 indexer 的 K 写入（pooling + aiter 的 norm/RoPE/MXFP4 缓存融合）和 dense scoring 路径。测试覆盖 64 head × 128 dim、ratio 4、page 64 的配置。
- **新手概念课堂**：DeepSeek-V4 用"稀疏 indexer"先粗筛 token，再 dense attention 精算。MXFP4 是 4 位浮点，把 K cache 压到 1/8。AMD 上要用 aiter 库做融合算子。
- **对你有什么影响**：AMD CDNA4（MI355 等）上跑 DeepSeek-V4，长上下文推理显存和速度都改善。

### 19. [Bugfix][ROCm][MiniMax-M3] Reject IndexCache overrides that miss text_config
- **代码层面**：`minimax_m3` 的 config 校验里，如果用户把 `use_index_cache` 等参数放在顶层 `--hf-overrides` 而不是 `text_config` 下，直接报错并给出正确写法。新增 `test_index_cache_overrides.py` 验证错误信息。
- **新手概念课堂**：MiniMax-M3 是多模态模型，文本配置在 `text_config` 子对象里。参数放错层级就像把"房间号"写在"街道名"栏，系统找不到。
- **对你有什么影响**：配置 IndexCache 时如果写错层级，会立刻看到清晰的报错和修正示例，不用猜。

### 20. [Doc] Kimi-K3: document the GB300 disaggregated specialized-decode points
- **代码层面**：`Kimi-K3.yaml` 配方里，GB300 的 specialized-decode 配置从 `bfloat16` 改成 `float32` SSM state、`--cp-kv-cache-interleave-size 1`、关闭 FlashInfer autotune；decode 侧加 `latency_specialized` 后端、`max-num-seqs 12`、显式 `cudagraph_capture_sizes` 列表。
- **新手概念课堂**：投机解码（K4 draft）在低并发（c16-c32）时，KV cache 的 interleave 如果被 NIXL 的 P/D 传输改掉，spec-verify 的 causal mask 会用错值。配方把这些参数钉死，保证 specialized decode 路径正确。
- **对你有什么影响**：在 GB300 上按配方部署 Kimi-K3 的 P/D 分离 + 投机解码，能复现官方调优的性能点，不用自己试参数。

## sglang
## 💡 今日关键词

- **关键词：DSv4.1 / DeepSeek V4.1 生态优化**
  - 一句话解释：今天一半以上的 commit 都在给 DeepSeek V4.1 模型"打补丁提速"，涉及 KV cache、MoE 路由、mHC 残差合并、engram 哈希等。
  - 为什么社区在关注：说明 sglang 已经把主力优化资源押在 DeepSeek V4.1 系列上，这也是当前开源大模型推理最卷的战场。谁把 V4.1 在自家硬件上跑得最快，谁就赢得用户。
  - 对新手意味着什么：如果你在学推理框架，DeepSeek 系模型的 kernel 优化是绝佳的"源码教材"——每个 commit 都带有清晰的性能动机和微基准。

- **关键词：Fused Kernel（融合算子）**
  - 一句话解释：把原本要跑好几个 GPU kernel 的操作（比如 RMSNorm + RoPE + 量化 + 存 KV cache）合并成一个 kernel，减少启动开销和显存往返。
  - 为什么社区在关注：今天的 commit 里大量出现 "fused k-norm-rope store"、"fuse pre-expert RMSNorm with NVFP4 quantization"、"MegaGate routing"，说明融合算子已是性能优化的主战场。
  - 对新手意味着什么：想进推理框架团队，Triton / CUDA 融合算子编写是硬通货技能，而且需求量大、门槛高、薪资好。

- **关键词：Breakable CUDA Graph / Prefill Graph**
  - 一句话解释：CUDA Graph 把一串 GPU 操作录制成"宏"，回放时省掉 CPU 调度开销；"breakable" 指可以在图中间插入动态逻辑（如 MoE 路由）。
  - 为什么社区在关注：Commit 3、5、17 都在调整"什么操作能进图、什么必须留在图外"，这是推理引擎从"能跑"到"跑满"的关键分水岭。
  - 对新手意味着什么：CUDA Graph 是高性能推理的必修课，建议从 PyTorch 的 `torch.cuda.CUDAGraph` 入手做小实验。

- **关键词：量化（MXFP8 / NVFP4 / FP8）**
  - 一句话解释：把模型权重和激活从 bf16/fp16 压到 8bit 甚至 4bit，显存占用减半甚至减到 1/4，算力吞吐翻倍。
  - 为什么社区在关注：Commit 3（MXFP8）、Commit 10（NVFP4）、Commit 7（FP8 KV cache）都在量化上做文章，说明"低精度推理"已经从可选项变成必选项。
  - 对新手意味着什么：量化是部署侧最实用的技能之一，学会写一个简单的 int8 量化 kernel 就能让你在面试中脱颖而出。

- **关键词：多模态 / 扩散模型推理**
  - 一句话解释：sglang 不再只是 LLM 推理框架，还在扩展图像/视频生成（diffusion）和视觉语言模型（VLM）。
  - 为什么社区在关注：Commit 1（Cosmos3）、4（Lanczos resize）、9（ComfyUI 插件）、15（FLUX.3 分离式部署）说明 sglang 正在向"全模态推理平台"演进。
  - 对新手意味着什么：如果你对 AIGC 感兴趣，sglang 的 diffusion 模块是学习"如何把生成模型部署成服务"的绝佳入口。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph 与"可断图"（Breakable CUDA Graph）**
  - **是什么**：CUDA Graph 是把一连串 GPU 操作录制下来、一次性提交的技术，能省掉每个 kernel 的 CPU launch 开销。但有些操作（比如 MoE 的专家路由）需要根据运行时数据动态决定，所以需要"可断图"——在图中间留出可以跳出执行动态逻辑的"断点"。
  - **为什么重要**：在高并发推理中，CPU launch 开销能占到总延迟的 20%~40%。不懂 CUDA Graph，你的推理速度永远上不去。而且一旦图里混入了不该进图的操作，就会出错（今天的 Commit 5 就是在修这个 bug）。
  - **动手试试**：
    ```python
    import torch

    # 一个简单的 MLP
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(1024, 4096).cuda()
            self.fc2 = torch.nn.Linear(4096, 1024).cuda()
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = MLP()
    x = torch.randn(32, 1024, device="cuda")

    # 预热（CUDA Graph 要求先跑一遍）
    for _ in range(3):
        model(x)

    # 录制图
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_graph = model(x)

    # 回放图
    g.replay()
    print("CUDA Graph 输出:", y_graph.shape)
    ```
  - **预期结果**：你会看到 `CUDA Graph 输出: torch.Size([32, 1024])`。用 `nvidia-smi` 或 `torch.profiler` 对比图回放与普通调用的耗时，能明显看到图回放更快。

- **知识点 2：Triton 写一个融合算子（以 RMSNorm + 量化为灵感）**
  - **是什么**：Triton 是 OpenAI 开源的 GPU kernel 编写语言，用 Python 语法写高性能 kernel。今天的 Commit 4（Lanczos resize）和 Commit 13（索引映射）都是 Triton 写的。
  - **为什么重要**：融合算子能把"读数据→计算→写数据"三步压缩成一步，减少显存带宽浪费。在显存带宽成为瓶颈的今天（尤其是大模型推理），融合是最大的性能杠杆。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def fused_norm_scale(x_ptr, w_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        w = tl.load(w_ptr + offs, mask=offs < N, other=0.0)
        # 简单归一化 + 加权（示意）
        mean = tl.sum(x, axis=0) / N
        out = (x - mean) * w
        tl.store(out_ptr + offs, out, mask=offs < N)

    N = 1024
    x = torch.randn(N, device="cuda")
    w = torch.randn(N, device="cuda")
    out = torch.empty_like(x)
    fused_norm_scale[(1,)](x, w, out, N=N, BLOCK=1024)
    print(out[:5])
    ```
  - **预期结果**：你会看到归一化后的前 5 个值。把这个 kernel 和 PyTorch 原生实现对比，用 `torch.profiler` 看 kernel 数量——融合版本只有 1 个 kernel。

- **知识点 3：FP8 / 低精度量化的基本概念**
  - **是什么**：FP8 是 8bit 浮点数（E4M3 或 E5M2 格式），比 bf16 少一半显存，算力翻倍。今天的 Commit 7 用 FP8 存 KV cache，Commit 10 用 NVFP4（4bit）做量化。
  - **为什么重要**：KV cache 在大模型长上下文推理中能占到显存的 50% 以上。用 FP8 存 KV cache 等于把可用上下文长度翻倍，这是生产环境的刚需。
  - **动手试试**：
    ```python
    import torch

    # 模拟 FP8 量化的基本流程（E4M3 格式）
    def fp8_quantize(x, scale):
        # 缩放后截断到 FP8 范围 [-448, 448]（E4M3）
        q = torch.clamp(x / scale, -448, 448)
        return q.to(torch.float8_e4m3fn)

    x = torch.randn(1000, device="cuda") * 10
    scale = x.abs().max() / 448
    q = fp8_quantize(x, scale)
    deq = q.to(torch.float32) * scale

    print("原始 dtype:", x.dtype, "量化后 dtype:", q.dtype)
    print("相对误差:", ((deq - x).abs() / (x.abs() + 1e-6)).mean().item())
    ```
  - **预期结果**：你会看到 dtype 从 `float32` 变成 `float8_e4m3fn`，相对误差在 1%~5% 左右（取决于动态范围）。这说明 FP8 量化在精度损失可接受的前提下，显存减半。

- **知识点 4：KV Cache 与分页注意力（PagedAttention）**
  - **是什么**：KV cache 是 Transformer 推理时缓存的历史 key/value，避免重复计算。PagedAttention 像操作系统管理内存页一样管理 KV cache，把不连续的物理块映射成连续的逻辑序列。
  - **为什么重要**：今天的 Commit 7（token-interleaved V4.1 KV cache）、Commit 11（HiCache 融合 DFLASH KV）都在改 KV cache 的组织方式。不懂 KV cache 就无法理解 vLLM/sglang 的核心优势。
  - **动手试试**：
    ```python
    # 用 vLLM 或 sglang 观察 KV cache 占用
    # 这里用 sglang 的简单启动命令
    # pip install sglang[all]
    # 启动：
    # python -m sglang.launch_server --model-path Qwen/Qwen2-0.5B --port 30000
    # 然后在另一个终端：
    # curl http://localhost:30000/get_server_info
    ```
  - **预期结果**：`get_server_info` 会返回 `max_total_num_tokens`（KV cache 能容纳的总 token 数）。你可以试试改 `--mem-fraction-static` 参数，观察这个数字的变化——这就是 KV cache 显存分配的直观体现。

- **知识点 5：多模态扩散模型的分离式部署（Disaggregated Serving）**
  - **是什么**：把扩散模型的"文本编码器 / 图像编码器 / 去噪网络 / VAE 解码器"拆成独立的服务进程，各自部署在不同机器上，通过 RPC 传输中间张量。
  - **为什么重要**：Commit 15 给 FLUX.3 加了这套能力。不同阶段的显存需求差异巨大（编码器可能只要 2GB，DiT 要 40GB），拆开部署能大幅提高 GPU 利用率。
  - **动手试试**：
    ```bash
    # 参考 Commit 15 的文档，启动 FLUX.3 分离式部署
    # 先启动 head
    sglang serve --model-path black-forest-labs/FLUX.3-Image \
      --disagg-role server --disagg-server-addr 0.0.0.0:9000

    # 再启动 denoiser worker（另一台机器或另一个进程）
    sglang serve --model-path black-forest-labs/FLUX.3-Image \
      --disagg-role denoiser \
      --disagg-server-addr <head-ip>:9000 \
      --disagg-p2p-hostname <worker-ip>
    ```
  - **预期结果**：你会在 head 的日志里看到 worker 注册成功，随后通过标准图像 API（`/v1/images/generations`）发起请求，就能得到完整生成的图片。这说明"拆分部署但接口不变"的目标达成了。

---

## 🔥 重要更新

1. **Commit 7：sgl-flash-mla 1.0.0 集成，V4.1 KV cache 用 token-interleaved 布局** —— 这是 DeepSeek V4.1 推理性能的核心升级，直接影响长上下文场景的吞吐和延迟，生产环境值得重点关注。

2. **Commit 16：DeepGEMM MegaGate 路由集成** —— 把 MoE 的专家路由从"先算 topk 再 dispatch"变成"fused 的 mega gate"，是 MoE 推理的关键提速点，也是 v4.1 系列性能优化的重要一环。

3. **Commit 15：FLUX.3 分离式编码部署** —— 标志着 sglang 从"LLM 推理框架"向"全模态推理平台"迈进，对 AIGC 服务化部署有直接指导意义。

4. **Commit 11：HiCache 支持 DFLASH KV 融合到统一 hybrid SWA 池** —— 层级缓存（HiCache）是长上下文推理的核心机制，这次升级让 DFLASH 草稿模型也能享受 HiCache 带来的显存红利。

---

## 📋 逐条解读

### 1. [Feat] [CPU] model: Support Cosmos3 series models on CPU (#45478)

- **代码层面**：改的是 `video_sparse_attn.py` 的 import 逻辑。原来是"直接 try import `vsa`，失败就设为 None"，现在改成"先判断 `torch.cuda.is_available()`，只有在 CUDA 可用时才尝试 import"。这样在纯 CPU 环境下就不会因为找不到 CUDA 版本的 `vsa` 库而报错。
- **新手概念课堂**：这就像你去餐厅点菜，原来不管有没有厨师都先叫"上菜"，现在先看看厨房里有没有人——没有就不叫了。`vsa`（video sparse attention）是一个依赖 CUDA 的库，CPU 环境本来就用不了，所以干脆别 import。
- **对你有什么影响**：如果你在纯 CPU 机器上跑 sglang 的 multimodal 模块，以前可能会看到一堆 ImportError 警告，现在干净了。对 GPU 用户无影响。

### 2. [Fix] [XPU] runtime: Fix XPU rotary embeddings and tokenizer padding (#41213)

- **代码层面**：两处改动。第一处在 `rotary_embedding/base.py` 的 `forward_xpu` 函数里，加了一个分支：当 `head_size != rotary_dim`（部分 RoPE）时，回退到 `forward_native`。第二处在 `patch_tokenizer.py`，新增 `_PadPaddingSideShim`，给某些老版本 tokenizer 的 `_pad()` 方法补上 `padding_side` 参数支持。
- **新手概念课堂**：RoPE（旋转位置编码）是 Transformer 里表示"位置"的技术。XPU（Intel GPU）对 RoPE 的支持不完整，有些 kernel 只支持"完整旋转"，遇到"部分旋转"就出错。tokenizer 的 `_pad` 是给序列补 padding 的函数，老版本没 `padding_side` 参数，新版 API 调用会报错——这个 shim 就是"翻译官"，把新调用翻译成老接口。
- **对你有什么影响**：用 Intel GPU 跑 sglang 的用户，以前可能遇到 RoPE 报错或 tokenizer 崩溃，现在修好了。普通 NVIDIA 用户无感。

### 3. [Perf] dsv4: Let the fused MXFP8 producers run inside the breakable prefill graph (#45980)

- **代码层面**：改的是 `deepseek_v4_mhc.py` 的 `fused_mxfp8_rows` 函数。原来判断条件是"不在 breakable CUDA graph 里才用 fused MXFP8"，现在改成"在 prefill graph scope 里就用"。同时 import 从 `is_in_breakable_cuda_graph` 换成了 `is_in_prefill_graph_scope`。
- **新手概念课堂**：MXFP8 是一种 8bit 量化格式（微软提出）。"fused producers" 指把量化操作和其他操作融合的 kernel。原来因为怕和 CUDA Graph 冲突，prefill 阶段不用；现在发现可以安全地用，就打开了——等于 prefill 阶段也享受量化加速。
- **对你有什么影响**：用 DeepSeek V4 做 prefill（长 prompt 输入）时，速度会更快。这是纯性能提升，无 API 变化。

### 4. [Perf] diffusion/cosmos3: optimize v2v preprocess performance (#45398)

- **代码层面**：新增文件 `lanczos.py`，用 Triton 实现了一个 GPU 版的 Lanczos 图像缩放 kernel（含可选裁剪和归一化）。在 `mm/__init__.py` 里注册了 `mm.lanczos_resize` 这个 kernel 算子。
- **新手概念课堂**：Lanczos 是一种高质量的图像缩放算法（比双线性插值清晰）。原来在 CPU 上用 PIL 做，慢；现在用 Triton 在 GPU 上做，快。这是"把预处理搬到 GPU"的经典优化。
- **对你有什么影响**：用 Cosmos3 做视频到视频（v2v）生成时，预处理阶段（缩图）会明显变快。如果你在跑 ComfyUI 或类似工作流，感知会更明显。

### 5. [Fix] dsv41: Keep MegaGate out of prefill graphs by gating on the forward mode (#45969)

- **代码层面**：改的是 `mega_gate.py` 的 `should_use_mega_gate` 函数。原来是"看 `is_extend_in_batch` 标志"，现在改成"看 `forward_mode` 是不是 decode 或 target_verify"。原因是 prefill graph capture 时会把 `is_extend_in_batch` 清掉，导致误判。
- **新手概念课堂**：MegaGate 是 DeepGEMM 里的 MoE 路由融合算子。它只适合 decode（逐 token 生成）阶段，不适合 prefill（批量输入）阶段。原来的判断逻辑有 bug——prefill graph 捕获时会篡改标志位，导致 MegaGate 错误地跑在 prefill 里。现在改成直接看 forward mode，更可靠。
- **对你有什么影响**：用 DeepSeek V4.1 做 prefill 时，不会再因为 MegaGate 误入而出错或变慢。是稳定性修复。

### 6. [Perf] dsv41: Drop the MegaGate padded-row copy on sgl-deep-gemm 0.2.3.post2 (#45933)

- **代码层面**：升级 `sgl-deep-gemm` 从 0.2.3 到 0.2.3.post2。在 `mega_gate.py` 里删掉了 `torch.where(valid_token_mask, hidden_states, 0)` 这一行——新版本库内部已经处理了 padding 行的清零。
- **新手概念课堂**：MegaGate 处理 MoE 时，如果 batch 里有 padding 行（凑齐 batch size 的空位），需要把这些行清零。原来 sglang 自己在外面做一次 copy，现在新版库内部做了，外面就不用重复做——省一次显存读写。
- **对你有什么影响**：用 DeepSeek V4.1 + MegaGate 时，decode 阶段会更快一点点。需要升级 `sgl-deep-gemm` 依赖。

### 7. [Perf] dsv41: Serve token-interleaved V4.1 KV caches with sgl-flash-mla (#45893)

- **代码层面**：这是今天最大的 commit 之一。升级 `sgl-flash-mla` 到 1.0.0，改了多个 CUDA kernel（`c1.cuh`、`c2.cuh`、`flashmla_sched_meta.cuh`），支持"token-interleaved"（token 交错）布局的 KV cache。还修了 scheduler 里一个"读越界"的 bug。
- **新手概念课堂**：KV cache 的布局（layout）决定了 GPU 读取效率。"token-interleaved" 是把多个 token 的 KV 交错存放，相比"block-interleaved"更适合 FlashMLA 的访问模式。这就像图书馆书架——书怎么摆，决定了你找书的速度。
- **对你有什么影响**：用 DeepSeek V4.1 做长上下文推理时，KV cache 读写更快，吞吐更高。这是纯性能提升，但需要升级 `sgl-flash-mla` 依赖。

### 8. [Perf] dspark: Combine the draft's mHC streams from bf16 without an fp32 copy (#45964)

- **代码层面**：改的是 `deepseek_v4_dspark.py` 的 `_forward_impl`。原来调用 `mhc_combine(x.flatten(1).float(), ...)`，现在去掉 `.float()`，直接传 bf16。因为 kernel 内部加载时本来就会转 fp32。
- **新手概念课堂**：mHC（multi-head compression）是一种残差合并机制。原来代码先把 bf16 转成 fp32 再传进 kernel，但 kernel 内部本来就会转——等于转两次。去掉外层转换，省一次显存 copy。
- **对你有什么影响**：用 dspark（DeepSeek 的推测解码）时，draft 模型前向更快。是纯性能优化。

### 9. [Fix] comfyui: Fix GGUF detection, import guard and VRAM accounting in SGLDiffusion plugin (#43536)

- **代码层面**：改的是 ComfyUI 插件。三处：1）`__init__.py` 里 import 失败时现在会打日志（以前静默吞掉）；2）删掉了 `_h3_detect_companion` 函数（GGUF 检测逻辑重构）；3）修正了 VRAM 显存计算逻辑。
- **新手概念课堂**：ComfyUI 是流行的 Stable Diffusion 图形界面。GGUF 是量化模型格式（llama.cpp 生态）。插件原来在 import 失败时静默返回空节点列表，用户根本不知道为啥节点没了——现在会打日志告诉你缺什么依赖。
- **对你有什么影响**：如果你用 ComfyUI + SGLDiffusion 插件，装错依赖时现在能看到明确报错，而不是"节点消失"的灵异现象。

### 10. [Perf] [NVIDIA] muse-spark-open: Fuse pre-expert RMSNorm with static-scale NVFP4 quantization (#44554)

- **代码层面**：改的是 `norm_fp4_quant.cuh`。新增了一个模板参数 `kStaticScale`，在 kernel 里加了"静态缩放"分支——用固定的 `global_scale` 而不是 per-token 动态缩放。同时新增了 `output_scale_static` 函数。
- **新手概念课堂**：NVFP4 是 NVIDIA 的 4bit 量化格式。量化需要"缩放因子"（scale）——可以是 per-token 动态算（准但慢），也可以是全局静态（快但略损精度）。这个 commit 让 kernel 支持两种模式，静态模式省掉 per-token 的 amax 归约。
- **对你有什么影响**：用 muse-spark-open 模型时，MoE 前的量化更快。是纯性能优化，精度损失在可接受范围。

### 11. [Feat] HiCache: Support fused DFLASH KV in unified hybrid SWA pools (#45700)

- **代码层面**：改的是 `hybrid_pool_assembler.py` 和 `kv_cache_builder.py`。新增 `validate_fused_hicache_draft` 函数，验证"DFLASH 草稿模型的 KV 能否融合到目标模型的统一 hybrid SWA 池里"。加了多重检查：必须是 DFLASH 算法、必须是 hybrid SWA、必须无 Mamba、必须 L2-only 存储。
- **新手概念课堂**：HiCache 是 sglang 的层级缓存（GPU + CPU + 磁盘）。DFLASH 是 DeepSeek 的推测解码算法。SWA 是滑动窗口注意力。这个 commit 让 DFLASH 草稿模型的 KV 也能进 HiCache——等于给推测解码也加上了"缓存加速"。
- **对你有什么影响**：用 DeepSeek V4.1 + DFLASH 推测解码 + HiCache 时，显存利用率更高，长上下文场景吞吐更好。功能需要显式开启。

### 12. [Perf] dsv4: Write V4.1 K rows with one warp per token in the fused k-norm-rope store (#45934)

- **代码层面**：改的是 `main_norm_rope.cuh`。新增了一个 kernel `fused_k_norm_rope_flashmla_v41`，用"一个 warp 处理一个 token"的方式写 K 行。同时抽出了 `v41_rope_pair` 函数，把 RoPE 的 fma 运算固定下来，保证两个 kernel 输出 bitwise 一致。
- **新手概念课堂**：K 是 attention 里的 key 向量。"k-norm-rope store" 是把 RMSNorm + RoPE + 存 KV cache 三个操作融合成一个 kernel。原来是一个 block 处理多个 token，现在改成"一个 warp 一个 token"，更适合 V4.1 的 head_dim=512 布局。
- **对你有什么影响**：DeepSeek V4.1 的 prefill 阶段（token 数 > 2048 时）会更快。是纯性能优化，输出 bitwise 不变。

### 13. [Perf] dsv4: Map V4.1 prefill index selections to pool slots in one kernel (#45938)

- **代码层面**：改的是 `fp4_indexer.py` 和 `scoring.py`。新增 Triton kernel `_flat_selection_slots_kernel`，把原来"用 PyTorch 的 `torch.where` + 高级索引"做的映射，改成一个 fused kernel 一次搞定。
- **新手概念课堂**：V4.1 的 prefill 阶段要选 topk 的 KV 位置，然后把这些"逻辑位置"映射到"物理池槽位"。原来用 PyTorch 操作分好几步（clamp、gather、where），现在一个 Triton kernel 全干完——减少 kernel 数量和中间张量。
- **对你有什么影响**：DeepSeek V4.1 的 prefill 索引阶段更快。是纯性能优化。

### 14. [Perf] dsv4: Combine the final mHC streams from bf16 without an fp32 copy (#45940)

- **代码层面**：改的是 `mhc.py` 和 `deepseek_v4.py`。和 Commit 8 类似——去掉 `.float()` 调用，直接传 bf16 给 `mhc_combine`。同时在 kernel 里加了 `assert x_flat.stride(1) == 1` 确保内存连续。
- **新手概念课堂**：mHC 残差合并的输入是 bf16，但 kernel 内部会转 fp32 计算。原来外层代码多做了一次 `.float()` 转换——纯浪费。去掉后省一次显存 copy。
- **对你有什么影响**：DeepSeek V4 的最后一层残差合并更快。是纯性能优化。

### 15. [Feat] diffusion/disaggregation: split FLUX.3 prompt and image encoders (#45309)

- **代码层面**：这是今天第二大 commit。新增 `prompt_encoder` 和 `image_encoder` 两个 disagg role，把 FLUX.3 的文本编码器和图像编码器拆成独立服务。同时更新了大量文档（`FLUX.3.mdx`、`cli.mdx`、`disaggregation.mdx`）。
- **新手概念课堂**：扩散模型推理分几步：文本编码 → 图像编码（如果是图生图）→ 去噪（DiT）→ VAE 解码。原来这些步骤在一个进程里，现在可以拆到不同机器——就像工厂流水线，每个工位独立。
- **对你有什么影响**：部署 FLUX.3 时，你可以把编码器放在小 GPU 上、DiT 放在大 GPU 上，资源利用率更高。API 不变，只是部署方式更灵活。

### 16. [Perf] dsv41: Integrate DeepGEMM MegaGate routing (#43262)

- **代码层面**：改的是 `mega_moe_pre_dispatch.cuh`。新增模板参数 `kCopyTopk`，当 MegaGate 已经算好 topk 时，pre-dispatch kernel 就跳过 topk 的 copy。同时注册了新的 MoE kernel 算子。
- **新手概念课堂**：MoE 推理分两步：1）路由（算每个 token 去哪些专家）；2）dispatch（把 token 发给对应专家）。MegaGate 把路由融合进了 DeepGEMM，算完 topk 直接传给 dispatch——不用再单独 copy 一次 topk 索引和权重。
- **对你有什么影响**：DeepSeek V4.1 的 MoE 推理更快。是纯性能优化。

### 17. [Perf] [NVIDIA] dspark: Move the host work between DSv4.1 decode graphs into the graphs for the megakernel path (#45450)

- **代码层面**：改的是 `engram_hash.py` 和 `cache_locs.py` 等。给 engram commit history kernel 加了 `live_rows` 和 `gate` 参数，支持在 CUDA graph 内跳过"非活跃行"和"不提交的 step"。同时给 `assign_extend_cache_locs_uniform` 加了 `positions_out`、`anchor_src` 等参数，支持在图内完成更多工作。
- **新手概念课堂**：CUDA graph 回放时，所有操作都会执行——哪怕某些行是 padding、某些 step 不需要提交。这个 commit 让 kernel 自己判断"这行要不要处理"，把原来在 host 上做的判断搬进 GPU，减少 host-device 同步。
- **对你有什么影响**：DeepSeek V4.1 decode 时，host 和 GPU 之间的同步更少，延迟更低。是纯性能优化。

### 18. [Refactor] unified-cache: Factor match validators and match completion into helpers (#45714)

- **代码层面**：改的是 `unified_tree_core.py` 和 `unified_radix_cache.py`。把 `match_prefix` 里的"创建 validators"逻辑抽成 `_create_match_validators` 函数，把"finalize match result"抽成 `_finalize_match_result` 函数。纯重构，无行为变化。
- **新手概念课堂**：这是典型的"提取函数"重构——把一大段代码里重复的逻辑抽出来，方便复用和测试。validators 是"判断节点是否匹配"的谓词函数，finalizers 是"匹配完后做收尾"的函数。
- **对你有什么影响**：无用户感知变化。对开发者来说代码更好读、更好改。

### 19. [Perf] request_logger: skip per-token watermark redaction walk (#44692)

- **代码层面**：改的是 `watermarking/core.py`。把"水印密钥字段名"和"容器字段名"抽成 frozenset 常量。在 `redact_watermark_secrets` 里加了一个快速路径：如果 list/tuple 里全是标量（int/float/str/bool/bytes/None），直接返回，不递归——因为密钥只可能藏在 dict key 或 struct 字段里。
- **新手概念课堂**：水印（watermark）是给生成文本打标记的技术，密钥必须脱敏才能写日志。原来每处理一个 token 都要递归遍历整个结构——慢。现在发现"纯标量列表"就直接跳过，因为里面不可能有密钥。
- **对你有什么影响**：开启水印 + 请求日志时，日志写入更快，对推理延迟影响更小。是纯性能优化。

### 20. [Perf] dsv4: Optionally stage V4.1 engram rows from the shared host table on the scheduler (#45921)

- **代码层面**：改的是 `environ.py` 和 `engram.py`。新增环境变量 `SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING`（默认关闭）。在 `engram.py` 里新增 `hash_windows` 方法，把哈希窗口计算从"必须用 CUDA kernel"改成"可以在 CPU 上算"——配合 scheduler 的 host 端 staging。
- **新手概念课堂**：engram 是 DeepSeek V4 的一种"记忆"机制，需要算 n-gram 哈希。原来必须在 GPU 上算，现在可以选择在 scheduler 的 CPU 上预先算好、再传给 GPU——适合大 prefill batch（GPU 算不过来时）。
- **对你有什么影响**：用 DeepSeek V4.1 做大 batch prefill 时，可以通过环境变量开启 host staging 来提速。默认关闭，需要显式开启。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)