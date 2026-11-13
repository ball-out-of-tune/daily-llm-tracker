# 🚀 vLLM & SGLang 每日更新 — 2026-11-13

> 自动生成于 2026-11-13 08:43 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：容错与请求恢复（Fault Tolerance）**
  - **一句话通俗解释**：当推理服务器的一部分挂了，不是简单报错，而是能把没算完的请求"捡回来"继续算。
  - **为什么社区在关注**：大模型服务动辄几十张卡，硬件故障是常态而非例外。vLLM 新增 `request_failure_policy="recompute"` 说明社区正在把"高可用"从附加功能变成核心能力。
  - **对新手意味着什么**：如果你在学分布式系统，这是绝佳的实战案例——如何设计"可恢复的推理状态"。

- **关键词：Rust 前端（Rust Frontend）**
  - **一句话通俗解释**：vLLM 正在用 Rust 重写 API 服务层，以换取更低延迟和更高吞吐。
  - **为什么社区在关注**：今天有 3 个 commit（#65994、#67359、#67257）都在推进 Rust 前端，包括新增 gRPC Render 服务、text 输出模式、mock 引擎 logprobs。这说明 Rust 重写已进入功能补齐阶段。
  - **对新手意味着什么**：Rust + 系统编程正在成为 AI Infra 的新门槛，值得投入学习。

- **关键词：多硬件后端适配（XPU / ROCm / CPU / Arm）**
  - **一句话通俗解释**：vLLM 不再只服务 NVIDIA GPU，Intel XPU、AMD ROCm、Arm CPU 都在被认真对待。
  - **为什么社区在关注**：今天有 5 个 commit（#67383、#66981、#67379、#67050、#65191）涉及非 CUDA 平台。异构计算已成既定事实。
  - **对新手意味着什么**：不要只盯着 CUDA，懂多后端适配的人才是稀缺资源。

- **关键词：KV Cache 架构重构**
  - **一句话通俗解释**：KV Cache（推理时缓存的历史键值对）的分配策略正在从"规格描述"中剥离出来，变成独立模块。
  - **为什么社区在关注**：#61452 是一个 RFC 的第二部分，影响面极大（3 万行 diff），说明 KV Cache 管理正在走向更清晰的抽象。
  - **对新手意味着什么**：理解 KV Cache 是理解 LLM 推理性能的钥匙。

- **关键词：融合算子（Fused Kernels）**
  - **一句话通俗解释**：把多个小计算步骤合并成一个 GPU 内核，减少内存往返。
  - **为什么社区在关注**：#67235 把 RMSNorm + RoPE + 残差加法融合进一个算子，说明性能优化已经深入到"每一次内存读写"级别。
  - **对新手意味着什么**：想学 GPU 编程，融合算子是必练的基本功。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：容错推理与请求恢复（Fault Tolerance + Recompute）**
  - **是什么**：当推理引擎某个 rank 崩溃后，未被完成的请求不直接失败，而是把 prompt + 已生成 token 重新喂给恢复后的引擎，重算 KV Cache 后继续生成。
  - **为什么重要**：在生产环境中，一次故障可能导致数百个用户请求失败。`recompute` 策略能显著提升用户体验。不懂这个机制，你可能误以为"故障=请求全丢"。
  - **动手试试**：
    ```bash
    # 启动带容错配置的 vLLM 服务
    vllm serve Qwen/Qwen3-0.6B \
      --fault-tolerance-config '{"request_failure_policy": "recompute"}'
    
    # 另一个终端：发起一个长流式请求
    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello, my name is","max_tokens":100,"stream":true}'
    ```
    然后在服务运行中触发一次恢复请求（参考 `docs/serving/expert_parallel_deployment.md`）。
  - **预期结果**：流式输出在恢复期间短暂停顿，之后继续输出，最终完成 100 个 token，而不是报错中断。

- **知识点 2：CUDA Graph 回退机制（Eager Fallback）**
  - **是什么**：CUDA Graph 是把一串 GPU 操作录制成"图"来加速重放。但如果重放时的输入尺寸超过录制时的缓冲区，就必须回退到普通 eager 模式。
  - **为什么重要**：如果不做这个检查，会导致内存越界或静默错误（commit 标记为 `[Security]`）。这是典型的"性能优化引入的边界 bug"。
  - **动手试试**：
    ```python
    import torch
    from vllm.v1.worker.gpu.encoder_cudagraph import EncoderCudaGraphManager
    
    # 模拟：捕获缓冲区 33 个元素，重放需要 67 个
    input_buffers = {"cu_seqlens": torch.zeros(33, dtype=torch.int32)}
    replay_values = {"cu_seqlens": torch.arange(67, dtype=torch.int32)}
    
    exceeds = EncoderCudaGraphManager._replay_exceeds_capture(input_buffers, replay_values)
    print(f"需要回退到 eager: {exceeds}")  # 应为 True
    ```
  - **预期结果**：输出 `True`，说明检测逻辑正确识别了缓冲区溢出。

- **知识点 3：编译范围去重（Compile Ranges Deduplication）**
  - **是什么**：vLLM 的 `torch.compile` 会针对不同序列长度编译不同图。`compile_ranges_endpoints` 定义这些长度点，重复值会导致重复编译。
  - **为什么重要**：重复编译浪费启动时间和内存。这是一个典型的"配置幂等性"问题——同一个配置调用两次应该得到相同结果。
  - **动手试试**：
    ```python
    from vllm.config import CompilationConfig, SchedulerConfig, VllmConfig
    
    cc = CompilationConfig(compile_ranges_endpoints=[32, 8])
    vc = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192, max_model_len=8192, is_encoder_decoder=False
        ),
        compilation_config=cc,
    )
    cc.pass_config.fuse_rope_kvcache = True
    vc._set_compile_ranges()
    vc._set_compile_ranges()  # 调用两次
    print(cc.compile_ranges_endpoints)  # 应为 [8, 32, 256, 8192]，无重复
    ```
  - **预期结果**：输出 `[8, 32, 256, 8192]`，没有重复的 8 或 32。

- **知识点 4：M-RoPE 与 PCP 位置编码**
  - **是什么**：M-RoPE（Multimodal RoPE）是多模态模型（如 Qwen-VL）用的位置编码，需要为文本、图像、视频分别分配位置。PCP（Pipeline Context Parallelism）把长序列切分到多个 rank 上。
  - **为什么重要**：切分后，每个 rank 看到的局部 token 索引与全局不一致。如果位置准备错误，模型会"看到"错误的位置信息，输出乱码。
  - **动手试试**：
    ```python
    import torch
    from vllm.v1.worker.gpu.mm.rope import prepare_positions
    
    # 模拟 PCP 片段：请求 0 有 4 个 token，被切成 [0,1] 和 [2,3] 两段
    idx_mapping = [0, 0]
    query_start_loc = [0, 2, 4]
    input_positions = [0, 1, 3, 4]
    prefill_lens = [8, 4]
    
    pos = prepare_positions(idx_mapping, query_start_loc, input_positions, prefill_lens)
    print(pos)  # 应包含全局位置 [110, 111, 113, 114] 等
    ```
  - **预期结果**：位置张量中的值对应全局 token 索引，而非局部索引。

- **知识点 5：INT8 W4A8 量化（Arm CPU）**
  - **是什么**：W4A8 表示权重用 4-bit 存储、激活用 8-bit 计算。这是量化的一种极端压缩方式，适合内存受限场景。
  - **为什么重要**：在 Arm CPU 上跑 Whisper 这类模型，W4A8 能大幅降低内存占用。commit 新增了 smoke test，说明这条路径已可用。
  - **动手试试**：
    ```bash
    # 在 Arm CPU 上运行（需要 Arm 架构机器）
    pytest -x -v -s tests/quantization/test_cpu_int8_w4a8.py
    ```
    或手动测试：
    ```python
    from vllm import LLM, SamplingParams
    llm = LLM(model="Arm/whisper-large-v3-quantized.w4a8", dtype="bfloat16", enforce_eager=True)
    # 传入音频做转写
    ```
  - **预期结果**：模型能正确转写音频（测试断言输出含 "the first words"），说明 W4A8 量化精度可用。

---

## 🔥 重要更新

- **#50265 容错请求恢复**：新增 `request_failure_policy="recompute"`，故障后未完成请求可重算 KV Cache 继续生成。这是生产级高可用的关键一步。
- **#61452 KV Cache 分配策略重构**：3 万行 diff，把分配策略从 `KVCacheSpecs` 中剥离。影响所有 KV Cache 相关代码，是架构级重构。
- **#60112 CUDA Graph 回退安全修复**：标记为 `[Security]`，修复 encoder CUDA graph 重放缓冲区溢出问题，避免潜在内存越界。
- **#65994 Rust gRPC Render 服务**：Rust 前端新增 `vllm.Render` 服务，把 chat 请求渲染成 token IDs，是 Rust 重写的重要里程碑。

---

## 📋 逐条解读

### 1. [Feature] add the option to continue serving unfinished requests after the recovery (#50265)
- **代码层面**：修改 `docs/serving/expert_parallel_deployment.md` 和 `tests/v1/fault_tolerance/test_fault_tolerance_e2e.py`。新增配置项 `request_failure_policy`，默认 `"fail"`（直接失败），可设为 `"recompute"`（重算恢复）。测试新增流式请求在故障后恢复的端到端验证。
- **新手概念课堂**：就像下载文件时断网了，默认行为是"重新下载"，而 `recompute` 是"从断点续传"——把已下载的部分保留，只补缺失的。
- **对你有什么影响**：如果你在生产环境部署 vLLM，开启 `recompute` 能显著减少故障时的请求失败率。但注意：死掉的引擎上的请求无法恢复。

### 2. [Rust Frontend][gRPC] Add the Render service with RenderChat (#65994)
- **代码层面**：新增 `rust/proto/render.proto`，在 `rust/proto/build.rs` 中注册新 proto，Rust 前端新增 `Render` gRPC 服务。`RenderChat` 把 chat 请求渲染成 prompt token IDs + 结构化输出约束等设置。
- **新手概念课堂**：就像餐厅的"点单台"和"厨房"分离——Render 服务负责把用户的话翻译成厨房能懂的"食材清单"（token IDs），厨房（Inference）只管做菜。
- **对你有什么影响**：如果你用 gRPC 调用 vLLM，现在可以分离"渲染"和"推理"两步，便于客户端做更精细的控制。

### 3. [CPU][Perf] Split oversized final attention spans (#66855)
- **代码层面**：修改 `csrc/cpu/cpu_attn_impl.hpp` 的 `AttentionScheduler`，在最后一个线程分区时，即使 `group == 1` 也把剩余任务作为可共享任务发射。新增测试用例 `decode_final_partition_remainder`。
- **新手概念课堂**：就像分披萨，最后一块太大一个人吃不完，现在允许把它再切小分给多人——避免最后一个线程"撑死"而其他线程"饿着"。
- **对你有什么影响**：CPU 上跑长序列 decode 时，负载更均衡，延迟更低。

### 4. [Security] Fall back to eager when encoder CUDA graph replay exceeds capture (#60112)
- **代码层面**：新增 `EncoderCudaGraphManager._replay_exceeds_capture()` 方法，比较重放张量的首维与捕获缓冲区首维。若超出则回退到 eager 模式。新增 3 个单元测试。
- **新手概念课堂**：就像你按"标准尺寸"做了个模具（CUDA Graph capture），但要浇注的零件比模具大——必须换回手工制作（eager），否则会溢出。
- **对你有什么影响**：修复了一个潜在的内存越界安全问题，多模态模型（如视觉 encoder）在高负载下更稳定。

### 5. [Bugfix] Let SIGTERM during engine startup stop the API server (#67096)
- **代码层面**：`vllm/utils/network_utils.py` 的 `zmq_socket_ctx` 在捕获 `KeyboardInterrupt` 后加了 `raise`，不再吞掉异常。新增测试验证异常传播。
- **新手概念课堂**：就像你按了电梯"停止"按钮，但电梯系统把信号吞了继续运行——现在信号能正确传递到顶层，电梯会停。
- **对你有什么影响**：启动过程中 `Ctrl+C` 或 `SIGTERM` 现在能真正终止服务，不会卡住。

### 6. [XPU][CI] Use platform dist backend in dist_init test fixture (#67383)
- **代码层面**：`tests/conftest.py` 中 `dist_init` fixture 把硬编码的 `"gloo" if is_cpu else "nccl"` 改为 `current_platform.dist_backend`。
- **新手概念课堂**：就像原来只准备了"汽油车"和"柴油车"两种加油枪，现在改成"按车型自动选"——XPU 也能用自己的通信后端了。
- **对你有什么影响**：在 Intel XPU 上跑测试不再因后端不匹配而失败。

### 7. [Bugfix] Fix M-RoPE position preparation with PCP (#67002)
- **代码层面**：新增 `tests/v1/worker/test_gpu_mrope_pcp.py`，修复 `vllm/v1/worker/gpu/mm/rope.py` 中 `prepare_positions` 在 PCP 切分下的位置计算，确保使用全局 token 索引。
- **新手概念课堂**：就像一本书被撕成几页分给多人读，每个人读的时候页码必须按原书编号，而不是"我手里这页的第 1 行"。
- **对你有什么影响**：多模态模型 + 长上下文并行（PCP）时，位置编码正确，输出不再错乱。

### 8. [Test][UMBP][10/10] Add lazy offload and token restore coverage (#61798)
- **代码层面**：修改 `.buildkite/test-amd.yaml` 和 `misc.yaml`，把 UMBP 测试从逐个文件列举改为通配符 `test_umbp_*.py`，并新增 `python3 -c "import mori"` 前置检查。新增 lazy offload 和 token restore 测试。
- **新手概念课堂**：就像原来只检查"语文、数学、英语"三科，现在改成"所有科目"——不会漏掉新加的科目。
- **对你有什么影响**：AMD MI355 上的 UMBP（统一内存块池）测试覆盖更全，KV Cache 卸载/恢复更可靠。

### 9. [Bugfix] Avoid duplicate values in `compile_ranges_endpoints` for compilation config (#67286)
- **代码层面**：`vllm/config/vllm.py` 的 `_set_compile_ranges` 中，`sorted(...)` 改为 `sorted(set(...))`。新增幂等性测试。
- **新手概念课堂**：就像购物清单上"苹果"写了两次，现在自动去重——避免重复编译同一尺寸的图。
- **对你有什么影响**：启动更快，内存占用更少，配置可安全重复调用。

### 10. [XPU][Test] Make test_shardq device-agnostic (#66981)
- **代码层面**：`tests/v1/attention/test_shardq.py` 中把硬编码的 `"cuda"` 改为 `torch.device(current_platform.device_type, rank)`，并传入 `backend=current_platform.dist_backend`。
- **新手概念课堂**：就像原来只支持"USB-A 接口"，现在改成"通用接口"——XPU、CUDA 都能插。
- **对你有什么影响**：ShardQ（注意力分片）测试可在 Intel XPU 上运行，验证跨平台正确性。

### 11. docs: typo in Elastic EP error and ASR comment (#67346)
- **代码层面**：`vllm/entrypoints/cli/serve.py` 修复 "with with" 重复词；`cohere_asr.py` 修复 "the the" 重复词。
- **新手概念课堂**：就像简历上的错别字——不影响功能，但影响专业度。
- **对你有什么影响**：日志和注释更清晰，无功能变化。

### 12. [Kernel][GLM-5.3-Flash] Fused in-place q/k RMSNorm + RoPE and residual add inside RMSNorm (#67235)
- **代码层面**：`vllm/models/glm5next/common/multimodal.py` 中，把原来的 `fused_q_kv_rmsnorm` + `ApplyRotaryEmb` 两步替换为单个 `ops.fused_qk_norm_rope` 调用，原地修改 qkv 张量。
- **新手概念课堂**：就像原来要"洗菜→切菜→炒菜"三道工序，现在一个"智能锅"全搞定——减少数据在内存中的搬运次数。
- **对你有什么影响**：GLM-5.3-Flash 模型推理更快，显存占用更低。

### 13. [XPU] reduce memory footprint for test_sagemaker_invocation_validation (#67379)
- **代码层面**：`tests/entrypoints/serve/sagemaker/test_sagemaker_invocation_validation.py` 中，把 `generate_server` fixture 从 module 级改为 class 级，embedding_server 从 module 级改为 function 级，减少同时驻留的模型实例。
- **新手概念课堂**：就像测试时不要同时开 5 个浏览器窗口——用完一个关一个，省内存。
- **对你有什么影响**：XPU 上跑这些测试不再因内存不足而失败。

### 14. [Core][KV Cache] Move KV cache allocation policy out of KVCacheSpecs (RFC #42449 part 2) (#61452)
- **代码层面**：大规模重构，把 `FullAttentionSpec` 等规格类中的分配策略逻辑剥离到 `single_type_kv_cache_manager.py` 中的 Manager 类。`AttentionLayerBase.get_kv_cache_spec()` 改为 `get_kv_cache_config()`。影响 benchmarks、tests、core 多处。
- **新手概念课堂**：就像原来"菜单"上写着"这道菜用哪个锅做"，现在菜单只写"这是什么菜"，"用哪个锅"由厨房（Manager）决定——职责分离。
- **对你有什么影响**：KV Cache 管理更灵活，未来支持新缓存类型（如混合注意力）更容易。但如果你有自定义 KV Cache 代码，需要适配新接口。

### 15. [Rust Frontend] Support output_mode="text" on the generate route (#67359)
- **代码层面**：`rust/src/server/src/routes/inference/generate.rs` 中，`generate` 路由支持 `output_mode="text"`，需要 tokenizer。新增 `ChatLogProbs` 类型转换，流式输出时同时返回 text 和 token IDs。
- **新手概念课堂**：就像原来只给你"食材编号"（token IDs），现在可以直接给你"做好的菜"（text）——更方便但需要厨房有"翻译官"（tokenizer）。
- **对你有什么影响**：Rust 前端用户可以直接拿文本输出，不用自己解码 token。

### 16. [CPU][Test] Add INT8 W4A8 smoke tests for Arm CPUs (#65191)
- **代码层面**：新增 `tests/quantization/test_cpu_int8_w4a8.py`，测试 `Arm/whisper-large-v3-quantized.w4a8` 模型在 Arm CPU 上的转写能力。CI 脚本 `run-cpu-test-arm.sh` 新增该测试。
- **新手概念课堂**：就像给手机装了个"压缩包"版的语言模型——体积小 4 倍，但还能听懂话。
- **对你有什么影响**：Arm 服务器/边缘设备上跑 Whisper 有了官方验证的量化方案。

### 17. [Bugfix][Model Loader] Handle merged-linear padding for Qwen4Exp and Kimi KDA (#66784)
- **代码层面**：`vllm/model_executor/model_loader/reload/direct.py` 新增 `record_checkpoint_layout`，处理 `MergedColumnParallelLinear` 的 padding。新增测试验证 padded 参数在 reload 后保持未填充状态且地址不变。
- **新手概念课堂**：就像行李箱有"预留空间"（padding），重新打包时不能把预留空间也塞满——否则下次装东西会溢出。
- **对你有什么影响**：Qwen4Exp 和 Kimi KDA 模型热重载（direct reload）时权重正确，不会因 padding 错位导致输出异常。

### 18. [Recipe][ROCm] MiMo-V2.6: use ROCM_AITER_UNIFIED_ATTN and add encoder-parallel opt-in (#67050)
- **代码层面**：`recipes/models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml` 中，为 MI300X/MI325X 添加 `--attention-backend ROCM_AITER_UNIFIED_ATTN --block-size 16`，新增 `encoder_parallel` 可选特性（`--mm-encoder-tp-mode data`）。
- **新手概念课堂**：就像给 AMD 显卡换了个"专用加速器"（AITER 统一注意力后端），比默认的 DiffKV 更快。
- **对你有什么影响**：在 AMD GPU 上部署 MiMo-V2.6 性能更好，视觉 encoder 可选择数据并行而非张量并行。

### 19. [Security][Model] Preserve Gemma audio frame dimensions and unified token counts (#47652)
- **代码层面**：`vllm/model_executor/models/gemma3n_mm.py` 中修复 `batch_audio_features` 对单帧音频的处理，保留帧维度。新增测试 `test_single_frame_audio_keeps_engine_usable` 验证单帧音频后引擎仍可用。
- **新手概念课堂**：就像处理一张"1 像素高"的图片，如果代码假设"至少 2 像素"，就会崩溃——现在正确处理了边界情况。
- **对你有什么影响**：Gemma 模型处理极短音频（如 161 采样点）不再导致引擎崩溃。

### 20. [Rust Frontend] Return requested logprobs from the mock engine (#67257)
- **代码层面**：`rust/src/engine-core-client/src/protocol/dtype.rs` 中，`NumpyDtype` 的 `Display` 实现改为：native 字节序输出纯类型名（如 `f4`），非 native 输出带前缀（如 `<f4`）。`imp.rs` 中把 `msgpack_zero_copy_threshold` 移到 protocol 模块。Mock 引擎现在返回请求的 logprobs。
- **新手概念课堂**：就像 Python 和 NumPy 对"数据类型"的写法不同——Python 要 `torch.float32`，NumPy 要 `f4`。现在 Rust 序列化时按接收方习惯输出。
- **对你有什么影响**：Rust 前端测试更真实（mock 引擎返回 logprobs），跨语言数据交换更兼容。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：异构硬件适配（AMD / NPU / Blackwell）**
  - 一句话通俗解释：同一份推理框架代码要跑在 NVIDIA、AMD、华为昇腾等不同芯片上，每个芯片的"脾气"不同，需要分别修 bug、调性能。
  - 为什么社区在关注：今天 20 个 commit 里至少 7 个是 AMD/NPU/Blackwell 专属修复（#48773、#50096、#50004、#50107、#48350、#50101、#50258），说明 SGLang 正在从"英伟达专属"变成"多芯片通用推理框架"，异构计算已是硬需求。
  - 对新手意味着什么：如果你只会写 CUDA，建议补一下 HIP/ROCm 和 AscendC 的基本概念；"跨后端兼容"是当下最值钱的工程能力之一。

- **关键词：量化与精度对齐（FP8 / block-scale / 位精确）**
  - 一句话通俗解释：把模型权重和激活从 16 位浮点压到 8 位甚至 4 位，省显存、提速度，但必须保证数值误差可控、结果可复现。
  - 为什么社区在关注：#42566（block-FP8 TP 切分）、#48350（Blackwell swap-AB 布局）、#50101（RoPE 舍入位精确）都在处理"量化后数值必须和参考实现逐位一致"这个硬骨头。
  - 对新手意味着什么：量化不只是"调个参数"，要懂浮点表示、舍入模式、块缩放（block scaling），这是推理优化的核心技能。

- **关键词：前缀缓存与状态复用（HiCache / Mamba checkpoint）**
  - 一句话通俗解释：多轮对话中，前面已经算过的内容别重算，把 KV 缓存和循环状态（Mamba 的隐状态）存下来复用。
  - 为什么社区在关注：#49886（只在分叉点和请求结束存 Mamba 状态）、#49054（NPU 上支持 Mooncake L3 存储）、#50102（跨层复用 int64 偏移）都是围绕"缓存命中率"和"存储层级"做文章。
  - 对新手意味着什么：推理系统的性能瓶颈常常不在算力而在"重复计算"，理解缓存层级（L1 显存 / L2 主机内存 / L3 远端存储）是系统优化的入门必修课。

- **关键词：调度器正确性（chunked prefill / 预算耗尽）**
  - 一句话通俗解释：长请求要分块预填充，如果这一轮显存或 token 预算不够，就得"挂起"等下一轮，而不是硬塞。
  - 为什么社区在关注：#44823 修的是调度器在预算耗尽时仍把请求塞进 batch 导致崩溃/错算的问题，这类 bug 隐蔽但影响所有长上下文用户。
  - 对新手意味着什么：调度器是推理框架的"大脑"，读懂 schedule_policy 比调 kernel 更能建立系统观。

- **关键词：开发者体验与工具链（ComfyUI 转换器 / 依赖升级 / 死代码清理）**
  - 一句话通俗解释：让用户更容易上手（工作流一键转换）、让依赖更稳定（版本升级）、让代码更干净（删死代码）。
  - 为什么社区在关注：#48998 加了离线工作流转换器、#50016 升依赖、#49189 大扫除，说明项目已过"能跑就行"阶段，开始重视工程质量和生态。
  - 对新手意味着什么：参与开源不一定要写 kernel，写工具、写文档、修 typo 都是被欢迎的贡献方式。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：GPU 架构特化编译（`__CUDA_ARCH__` 条件编译）

- **是什么**：CUDA 代码里可以用 `#if __CUDA_ARCH__ >= 1000` 判断当前编译目标是不是 Blackwell（SM100）及更新架构，从而对不同代际 GPU 走不同代码路径。
- **为什么重要**：新架构的编译器（nvcc/ptxas）可能把浮点表达式"融合"成 FMA，导致舍入结果和参考实现不一致。Commit #50101 就是把判断从 `==1030 || >=1200` 扩展到 `>=1000`，因为 SM100 也有同样的"乘法被合约"问题。不懂这个，你写的 kernel 在 B200 上结果会莫名其妙对不上。
- **动手试试**：写一个最小 CUDA 文件，用 `nvcc -arch=sm_100` 和 `-arch=sm_90` 分别编译，看 PTX 里 `fma.rn` 和 `mul.rn`+`add.rn` 的区别。
  ```bash
  cat > t.cu <<'EOF'
  __global__ void k(float* o, float a, float b, float c) {
  #if __CUDA_ARCH__ >= 1000
    o[0] = __fmul_rn(a,b) + __fmul_rn(c,1.0f); // 强制两次舍入
  #else
    o[0] = a*b + c;
  #endif
  }
  EOF
  nvcc -arch=sm_100 -ptx t.cu -o t.ptx && grep -E "fma|mul|add" t.ptx
  ```
- **预期结果**：SM100 分支出现两次 `mul.rn` + `add.rn`，默认分支可能出现 `fma.rn`。看到差异说明你理解了"编译器合约"这个坑。

### 知识点 2：块缩放量化（Block-wise FP8 Scaling）

- **是什么**：FP8 只有 8 位，动态范围小，所以把权重按 128×128 的小块分别存一个缩放因子（scale），每块用自己的 scale 解量化，精度比全张量一个 scale 高很多。
- **为什么重要**：Commit #42566 处理的是"张量并行（TP）切分时，切分线正好落在某个 128 宽的块中间"——这时两个 rank 各自只拿到半块，但 scale 是整块共享的，必须重新精化 scale 网格（refine to finer grid）才能正确加载。不懂块缩放，你会在 TP>1 时遇到精度暴跌或加载报错。
- **动手试试**：用 PyTorch 模拟块量化与解量化，观察"切分线在块中间"时的误差。
  ```python
  import torch
  w = torch.randn(256, 256)
  bs = 128
  # 按块量化到 int8 再反量化
  def q(t):
      s = t.abs().amax(dim=(1,2), keepdim=True) / 127
      return (t/s).round().clamp(-127,127) * s
  wq_full = torch.cat([q(w[i:i+bs, j:j+bs]) for i in range(0,256,bs) for j in range(0,256,bs)], 0)
  # 模拟 TP=2 切分：每 rank 拿 128 列，但块边界在 128 处刚好对齐 vs 不对齐
  left, right = w[:, :128], w[:, 128:]
  print("对齐误差:", (q(left)-left).abs().mean().item())
  print("非对齐误差:", (q(w[:, 64:192])-w[:, 64:192]).abs().mean().item())
  ```
- **预期结果**：非对齐切分的误差明显更大，说明为什么需要"scale 精化"。

### 知识点 3：Mamba 状态检查点与前缀树（Prefix Tree Checkpointing）

- **是什么**：Transformer 的 KV 缓存可以按 token 存进前缀树（Radix Tree）复用；但 Mamba 这类状态空间模型每层只有一个"隐状态"，不能按 token 存，只能在"分叉点"和"请求结束"时存快照。
- **为什么重要**：Commit #49886 把默认策略从"每个 chunk 都存"改成"只在分叉点和请求结束时存"（level 2），大幅减少状态副本数量。多轮对话场景下，每轮都在上一轮末尾续写，只需在轮末存一次即可复用。不懂这个，你会以为 Mamba 模型没法做前缀缓存。
- **动手试试**：用 dict 模拟一棵前缀树，统计"每 chunk 存"vs"只在分叉点存"的状态数量差异。
  ```python
  # 模拟：3 个请求共享前缀 "hello world"，各自续写不同后缀
  reqs = ["hello world foo", "hello world bar", "hello world baz"]
  def chunks(s, n=4): return [s[i:i+n] for i in range(0,len(s),n)]
  every = set()
  for r in reqs:
      for i in range(1, len(chunks(r))+1):
          every.add("".join(chunks(r)[:i]))
  fork_only = set()
  for r in reqs:
      fork_only.add(r)  # 只在请求结束存
  fork_only.add("hello world")  # 分叉点
  print("每chunk存:", len(every), "只在分叉点存:", len(fork_only))
  ```
- **预期结果**：`每chunk存` 数量远大于 `只在分叉点存`，直观理解为什么 level 2 更省内存。

### 知识点 4：调度器"预算耗尽时挂起"（Parking a Chunked Request）

- **是什么**：chunked prefill 把长 prompt 切成小块逐轮喂入；如果这一轮剩余的 KV 显存或 token 配额不够放下一个对齐的块，就把这个请求"挂起"（park），本轮不调度它，等下一轮再试，而不是强行塞入。
- **为什么重要**：Commit #44823 修的就是"预算不够还硬塞"导致的越界/错算。同时它保证挂起时"不把其他请求排在它前面"，避免 batch 里混入一个不会执行的请求。不懂调度器，你调 `--chunked-prefill-size` 时遇到 OOM 或结果错乱会毫无头绪。
- **动手试试**：写一个极简调度模拟，观察"硬塞"vs"挂起"两种策略。
  ```python
  budget = 10
  reqs = [("A", 6), ("B", 5), ("C", 4)]  # (名字, 需要token)
  def schedule(hard=False):
      used, out = 0, []
      for name, need in reqs:
          if used + need <= budget:
              used += need; out.append(name)
          elif hard:
              out.append(name + "(硬塞!)")  # 越界
      return out, used
  print("硬塞:", schedule(True))
  print("挂起:", schedule(False))
  ```
- **预期结果**：硬塞策略 `used` 超过 budget，挂起策略 `used <= budget` 且下一轮还能继续。

### 知识点 5：CPU offload 与设备一致性检查（`b.is_cuda`）

- **是什么**：张量可以放在 GPU（`is_cuda=True`）或 CPU（`is_cuda=False`）上；CPU offload 是把不常用的权重挪到主机内存省显存，但 kernel 只能读 GPU 内存。
- **为什么重要**：Commit #49380 修的是 warp router 只检查了"连续 + 16 字节对齐"却漏了"是否在 GPU 上"，导致 CPU offload 的 bias 被传给 kernel 后崩溃或读到垃圾。这是个经典陷阱：**指针地址对齐不代表设备正确**。
- **动手试试**：在 Python 里复现这个检查逻辑。
  ```python
  import torch
  def bad_check(b): return b is None or (b.is_contiguous() and b.data_ptr()%16==0)
  def good_check(b): return b is None or (b.is_cuda and b.is_contiguous() and b.data_ptr()%16==0)
  cpu = torch.zeros(8); gpu = torch.zeros(8, device="cuda") if torch.cuda.is_available() else None
  print("CPU bias 通过旧检查?", bad_check(cpu), "通过新检查?", good_check(cpu))
  if gpu is not None: print("GPU bias 通过新检查?", good_check(gpu))
  ```
- **预期结果**：CPU bias 通过旧检查（bug！）但不通过新检查；GPU bias 通过新检查。理解"防御性检查要覆盖设备维度"。

---

## 🔥 重要更新

1. **#44823 调度器修复**：chunked prefill 在预算耗尽时正确挂起，避免 batch 混入不执行的请求。影响所有长上下文/高并发场景，是最贴近"线上事故"的一类修复。
2. **#42566 block-FP8 TP 切分加载**：让 TP>1 时能正确加载"切分线穿过缩放块"的量化权重，直接决定大模型多卡部署能否跑起来。
3. **#49886 Mamba 状态检查点策略**：默认改为"只在分叉点和请求结束存"，混合模型（Mamba+Transformer）多轮对话显存占用大幅下降。
4. **#48350 Blackwell swap-AB MoE**：在 SM100/SM103 上启用新的权重布局，MoE 推理性能提升，代表新硬件适配的持续投入。

---

## 📋 逐条解读

### 1. [Fix] [AMD] topk-v2: Fix wrong picks on split-launch short rows (#48773)
- **代码层面**：改 `topk_impl.cuh` 的 `TopKStreaming` 结构。在 ROCm（AMD）平台上，当需要"精化并列值"（tie refinement）时，改调用 `refine_ties_cold` 而不是 `refine_ties`。原因是 hipcc 编译器在 gfx950 上把内联版本编译错了。同时新增一个只在 HIP 上跑的测试，覆盖"短行 + 页表变换"的组合。
- **新手概念课堂**：`topk` 是 MoE 路由的核心——从 256 个专家里挑分数最高的 8 个。当多个分数完全相同（tie）时需要额外逻辑打破平局。"split-launch"是把一个 kernel 拆成两段发射，短行会走特殊路径。AMD 编译器（hipcc）和 nvcc 是两套编译器，同一份 C++ 可能一个编对、一个编错。
- **对你有什么影响**：用 AMD GPU 跑 DeepSeek 系列模型时，之前可能选错专家导致输出质量下降，现在修好了。NVIDIA 用户无感知。

### 2. [Fix] quantization: Load block-FP8 MoE TP shards that split checkpoint blocks (#42566)
- **代码层面**：在 `deep_gemm.py` 的 `DeepGemmMoeQuantInfo` 里新增 `w2_block_shape` 字段，`scale_recipes` 增加 `down` 参数，让"下投影"（down projection）的 GEMM 使用比 checkpoint 更细的缩放块。文档里同步更新了 Qwen3.8 的说明。
- **新手概念课堂**：MoE 层有两个 GEMM：w13（上投影，gate+up）和 w2（下投影）。量化时按 128 宽的块存 scale。TP=2 时 640 维的 MTP 专家被切成 320，正好切在块中间——两个 rank 各拿半块但共享一个 scale，必须重新算 scale 网格才能正确解量化。
- **对你有什么影响**：用 TP>1 部署带 block-FP8 量化的 MoE 模型（如 Qwen3.8-Flash-Next）不再加载失败或精度异常。

### 3. [Fix] [NPU] test: Set TRITON_CACHE_DIR to /tmp/triton_cache (#50258)
- **代码层面**：在 NPU 性能测试脚本的环境变量字典里加了一行 `"TRITON_CACHE_DIR": "/tmp/triton_cache"`。
- **新手概念课堂**：Triton 是 OpenAI 的 kernel 编译 DSL，编译产物默认缓存在 `~/.triton/cache`。CI 容器里 HOME 可能不可写或每次重置，导致重复编译拖慢测试。显式指定 `/tmp` 下的路径解决这个问题。
- **对你有什么影响**：纯 CI 修复，普通用户无感知。但如果你自己跑 Triton 测试遇到"cache 权限"报错，可以照抄这招。

### 4. [Feat] [NPU] attention: Support FP8 KV for DSA prefill context parallelism (#50096)
- **代码层面**：在 `ascend_backend.py` 里抽出 `_forward_fp8_sparse` 方法，当 KV cache 是 FP8 且走 DSA（DeepSeek Sparse Attention）上下文并行预填充时，调用 `npu_kv_quant_sparse_flash_attention` 做量化稀疏注意力。原来这段逻辑内联在 `forward_sparse` 里，现在复用。
- **新手概念课堂**：FP8 KV cache 是把 KV 缓存压成 8 位省显存；DSA 是 DeepSeek 的稀疏注意力，只算 top-k 相关的 token；context parallelism（CP）是把超长序列切到多张卡上并行算。三者叠加需要专门的融合算子。
- **对你有什么影响**：在昇腾 NPU 上跑 DeepSeek-V4 类模型、长上下文预填充时，可以用 FP8 KV 省显存且不损失功能。

### 5. [Perf] kda: Reuse int64 prefill offsets across layers (#50102)
- **代码层面**：`kda_backend.py` 在 `init_forward_metadata` 里把 `query_start_loc` 转成 int64 存进 metadata，每层复用；`kda_flashinfer_prefill.py` 的 `extend` 新增 `query_start_loc_i64` 参数直接用它。新增 `wants_int64_query_start_loc` 标志位，只有需要 int64 的 kernel 才触发转换。
- **新手概念课堂**：`query_start_loc` 记录每个请求在拼接 batch 里的起始位置（如 [0, 5, 12, 20]）。FlashInfer 内部按张量身份缓存 host 端内容，但每次调用都把 int32 转成新的 int64 张量，导致缓存永远不命中。每层转一次 → 每 forward 转一次，省下几十次小拷贝。
- **对你有什么影响**：KDA（Kimi 的线性注意力）模型预填充略快，尤其层数多时收益累积。

### 6. [Fix] comfyui: Refresh the reviewed MiniMax-H3 source hash after #49826 (#50219)
- **代码层面**：`h3.py` 里的 `SOURCE_SHA256` 常量从旧值改成新值。
- **新手概念课堂**：ComfyUI 插件对上游 MiniMax-H3 实现打了 monkey patch，为防止上游代码悄悄变化导致 patch 错位，用 SHA256 锁死源码。上游 PR #49826 改了代码，所以这里同步更新"已审核"的哈希。
- **对你有什么影响**：用 ComfyUI + SGLDiffusion 跑 H3 模型时不会因哈希不匹配而拒绝加载。

### 7. [Perf] [NVIDIA] kimi-k3: Bound q6 preparation by server concurrency (#49376)
- **代码层面**：`binding.py` 新增 `_kimi_build_widths`，在特定条件下（KimiK3 架构、DSPARK 投机解码、TP=8、无 DP/CP、全图 decode、对齐=1）把 megakernel 的"准备宽度"从 `max_rows` 截断到 `request_limit * tokens`。其他情况保持全量准备，且所有 rank 必须一致（`megakernel_agree`）。
- **新手概念课堂**：megakernel 是"把多个小 kernel 融合成一个大 kernel"的技术，但需要提前为每种 batch 大小准备编译产物。q6 指每请求 6 个 token。如果服务器并发上限是 256，那宽度超过 256×6 的 bucket 永远不会被用到，白准备浪费启动时间。
- **对你有什么影响**：Kimi-K3 在 TP=8 部署时启动更快，运行时无变化。

### 8. [Fix] moe/router: Reject CPU-offloaded bias in warp router (#49380)
- **代码层面**：`bias_layout_ok` 增加 `b.is_cuda` 检查。测试新增 `test_cpu_offloaded_bias_falls_back`，验证 pinned CPU bias 会回退到 Triton router。
- **新手概念课堂**：见知识点 5。warp router 是一个手写的高性能路由 kernel，只接受 GPU 上的连续、对齐 bias；不满足就回退到通用 Triton 实现。之前漏了设备检查，CPU offload 场景下会崩。
- **对你有什么影响**：开启 `--cpu-offload` 跑 MoE 模型时不再崩溃，自动走慢但正确的路径。

### 9. [Feat] mem_cache: Checkpoint hybrid Mamba state only at fork points and request ends (#49886)
- **代码层面**：新增 `--hybrid-cache-checkpoints-level` 参数（默认 2），`--mamba-max-states-per-path` 标记为废弃但兼容。新增 `mamba_path_state_cap` 派生字段。文档更新说明 level 2 和 level 3 的区别。
- **新手概念课堂**：见知识点 3。混合模型（Transformer + Mamba）的 KV 可以按 token 存树，但 Mamba 隐状态只能按"点"存快照。level 2 只在分叉点和请求结束存，level 3 每个 chunk 都存（兼容旧行为）。
- **对你有什么影响**：多轮对话/agent 场景下显存占用显著下降；如果依赖"从任意 chunk 恢复"的高级用法，显式设 `--hybrid-cache-checkpoints-level 3`。

### 10. [Perf] flashinfer_cutedsl: Use swap-AB MoE on Blackwell (#48350)
- **代码层面**：删除 `interleave_w13_halves`（Python 端手动重排），改为把 `swap_ab=True` 传给 CuteDSL wrapper，由 kernel 内部处理布局。新增 `use_cutedsl_moe_swap_ab` 按设备能力（SM100/SM103）判断，用 `lru_cache` 缓存结果。`modelopt_quant.py` 里权重加载的 interleave 粒度从 64 行改为 16 行（Blackwell）或 64 行（旧架构）。
- **新手概念课堂**：MoE 的 w13 权重有 [Up, Gate] 两半，SwiGLU 需要交错排列。旧方案在 Python 里 cat 一堆切片（慢且占内存），新方案让 kernel 直接按 swap-AB 布局读——A/B 矩阵互换是 GEMM 优化的经典手法，能更好地利用 Blackwell 的 Tensor Core。
- **对你有什么影响**：B200/GB200 上跑 NVFP4 MoE 模型更快，权重加载更省内存。

### 11. [Fix] [NPU] moe: Add ascend_ops backend checks across MoE models (#50004)
- **代码层面**：在 bailing_moe、deepseek_v2、dots3、exaone_moe 等多个模型的 `__init__`/`forward` 里，把 `is_deepep()` 的判断扩展成 `is_deepep() or is_ascend_ops()`。共享专家的 `parallel_group` 也从 `"tp"` 改为 `"replicated"`。
- **新手概念课堂**：`get_moe_a2a_backend()` 返回 MoE 的 all-to-all 通信后端。DeepEP 是 NVIDIA 的，AscendOps 是华为的。之前代码只认 DeepEP，导致 NPU 上 MoE 层的专家分发逻辑走错分支。`parallel_group="replicated"` 表示共享专家每卡都存一份副本，不需要 TP 切分。
- **对你有什么影响**：在昇腾 NPU 上跑这些 MoE 模型（Bailing、DeepSeek-V2、Dots3、Exaone）时 MoE 计算正确。

### 12. [Feat] [NPU] HiCache: Support Mooncake L3 for DeepSeek-V4 and GLM-5.2 (#49054)
- **代码层面**：`c128_sidecar_component.py` 新增 L3 存储驻留逻辑：`block_pages()`、`_group_tokens()`、`_block_endpoint_keys()` 等方法，用 `heapq` 做淘汰，新增 `PreparePrefetchResult`、`FreeComponentHostSlot` 等动作。核心是"用块边界哈希而非节点尾部哈希作为存储 key"。
- **新手概念课堂**：HiCache 是三级缓存（GPU 显存 L1 / 主机内存 L2 / 远端存储 L3）。C128 是 DeepSeek-V4 的压缩 KV 格式，一个 C128 page 覆盖 128×kernel_page_size 个 token，比普通 KV page 粗得多。存到 L3 时必须用"块结束位置"的哈希做 key，否则恢复时找不到。
- **对你有什么影响**：NPU 上部署 DeepSeek-V4/GLM-5.2 时，可以把 KV 缓存下沉到 Mooncake 远端存储，支持更大规模的并发和更长的上下文。

### 13. [Test] disaggregation: Give the fake draft pool follows_target_swa_ids (#49699)
- **代码层面**：测试里给 `SimpleNamespace` 加了一个 `follows_target_swa_ids=False` 属性。
- **新手概念课堂**：PD 分离（prefill/decode 分离）场景下，draft 模型（投机解码的小模型）的滑动窗口（SWA）ID 是否跟随 target 模型，是个配置项。测试用的 fake 对象之前缺这个属性导致 `AttributeError`。
- **对你有什么影响**：纯测试修复，普通用户无感知。

### 14. [Chore] deps: Bump sgl-deep-gemm to 0.2.8 and sgl-flash-mla to 1.0.5 (#50016)
- **代码层面**：`pyproject.toml` 和 `Dockerfile.cu134` 里的版本号从 `0.2.7`→`0.2.8`、`1.0.4`→`1.0.5`。
- **新手概念课堂**：`sgl-deep-gemm` 是 DeepGEMM 的 SGLang 封装（FP8 GEMM 库），`sgl-flash-mla` 是 MLA 注意力的快速实现。版本升级通常带 bug 修复和性能改进。
- **对你有什么影响**：`pip install -e .` 后自动拿到新版；用 Docker 镜像的重新拉取即可。

### 15. [Fix] scheduler: Park a chunked prefill continuation when its chunk or KV budget is spent (#44823)
- **代码层面**：`schedule_policy.py` 的 `_add_chunked_req` 在"对齐后剩余 token <= 0 且剩余预算 < 一个 chunk"时直接 `return req`（挂起）。`scheduler.py` 新增 `chunk_parked` 标志，挂起时本轮不再接纳新请求排在它前面。`prefill.py` 的 `process_prefill_chunk` 用 `computed = req.extend_end > req.prefix_len` 判断是否真的算了新 KV，没算就不 checkpoint、不发送。
- **新手概念课堂**：见知识点 4。`kv_shard_granule` 是 KV 分片粒度（比如 128 token 对齐），如果剩余显存只够 50 token，不够一个对齐步，就得等下一轮释放显存后再来。之前代码"用完整 chunk 硬塞"会越界。
- **对你有什么影响**：长上下文 + 高并发 + PD 分离场景下不再出现"batch 里有请求但没执行"导致的挂起或结果错乱。

### 16. [Fix] [NPU] kimi-k3: Skip the CuTe DSL TGV tactic registration on NPU (#50107)
- **代码层面**：`kimi_k3.py` 里把 `if not _is_hip` 改成 `if not (_is_hip or _is_npu)`。
- **新手概念课堂**：CuTe DSL 是 NVIDIA 的 C++ 模板库（cutlass 的一部分），依赖 `cuda.bindings`，在 AMD 和华为 NPU 上都不存在。TGV 是某种 GEMM tactic（策略）。之前只排除了 AMD，漏了 NPU，导致 import 报错。
- **对你有什么影响**：昇腾 NPU 上加载 Kimi-K3 模型不再因 import 失败而崩溃。

### 17. [Feat] comfyui: Add an offline workflow converter and reject weight patches on SGLD models (#48998)
- **代码层面**：新增 `convert/` 子包和 `convert` CLI（`python -m ...convert my_workflow.json`）。转换规则：`UNETLoader`→`SGLDUNETLoader`、`LoraLoaderModelOnly`→`SGLDiffusion LoRA Loader`，并生成 Markdown 报告列出需要手动处理的部分。`__init__.py` 里把"不在 ComfyUI 环境"的 ImportError 降级为 debug 日志。
- **新手概念课堂**：ComfyUI 是节点式工作流工具，用户拖节点连线。SGLD 是 SGLang Diffusion 的节点。两者的 loader 节点不兼容，之前用户得手动替换。转换器自动做替换并提示"哪些节点无法自动转换"（如 SageAttention patch、TeaCache 等）。
- **对你有什么影响**：从 ComfyUI 迁移到 SGLang Diffusion 时，一条命令转换工作流，不用逐个节点手改。

### 18. [Fix] diffusion: Align Cosmos3 request sizes to the 16-pixel latent grid (#50114)
- **代码层面**：`cosmos3.py` 新增 `COSMOS3_SIZE_MULTIPLE = 16` 和 `_align_resolution` 方法，在 `_adjust_visual_fields` 和 `prepare_synthetic_warmup_request_for_queue` 里调用，把宽高向下取整到 16 的倍数并打 warning。
- **新手概念课堂**：VAE（变分自编码器）把图像压缩到潜空间，Cosmos3 的 Wan VAE 下采样 16 倍，所以输入宽高必须是 16 的倍数，否则 `Cosmos3LatentPreparationStage` 直接拒绝。用户传 1000×1000 会被拒，自动调成 992×992 更友好。
- **对你有什么影响**：用 Cosmos3 生成图片/视频时，任意尺寸请求都能跑，日志会提示实际使用的尺寸。

### 19. [Fix] diffusion: Keep fused QKNorm+RoPE rounding exact on SM100 (#50101)
- **代码层面**：`qknorm_rope.cuh` 里三处 `#if __CUDA_ARCH__ == 1030 || >= 1200` 改成 `>= 1000`，覆盖 SM100。测试文件更新注释并新增 B200 的 CI 注册。
- **新手概念课堂**：见知识点 1。QKNorm 是注意力前的归一化，RoPE 是旋转位置编码，两者融合成一个 kernel 省一次读写。融合后 `x*cos + y*sin` 会被 nvcc 合约成 FMA，但参考实现是先舍入两次乘法再相加，结果差 1 ulp。SM100 也有这个问题，所以扩展条件。
- **对你有什么影响**：在 B200 上跑 diffusion 模型时，QKNorm+RoPE 的输出与参考实现逐位一致，避免生成质量细微漂移。

### 20. [Refactor] cleanup: Prune dead code and env vars, fix silent failures found by the code-cleanup skill (#49189)
- **代码层面**：大范围清理——更新 AMD 调优文档里的命令（`bench_latency`→`benchmark.one_batch`）、把 `sgl-router.Dockerfile` 的默认端口从 8090 改成 30000 并删掉 config 环境变量、修正 LongCat 文档里的参数名（`--fp8-gemm-runner-backend`→`--fp8-gemm-backend`）、Ming-Image 文档澄清 `num_layers` 用法等。还修复了一些静默失败（`except: pass` 之类）。
- **新手概念课堂**："code-cleanup skill" 是团队内部的自动化审查工具，专门找死代码、过期文档、静默吞异常的地方。静默失败是最危险的——程序不报错但结果错了，排查起来极其痛苦。
- **对你有什么影响**：文档更准确（少踩"参数名写错"的坑），Docker 镜像默认行为变了（router 用户注意端口从 8090 变 30000）。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)