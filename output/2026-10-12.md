# 🚀 vLLM & SGLang 每日更新 — 2026-10-12

> 自动生成于 2026-10-12 07:56 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：token_ids 透传**
  - **一句话通俗解释**：聊天接口返回结果时，除了给人看的文字，还要把模型真正生成的“数字编号”也一并返回，方便上层系统精确比对。
  - **为什么社区在关注**：这说明 vLLM 正在从“能跑”走向“可观测、可校验”，前端和推理后端的一致性验证越来越被重视。分布式部署（disaggregated）场景下，token 级别的对齐是刚需。
  - **对新手意味着什么**：如果你在做 LLM 应用或评测系统，学会关注 token 级别数据而不是只看文本，是进阶的第一步。可以尝试用 `return_token_ids=True` 做端到端一致性校验。

- **关键词：AITER MegaMoE + 共享专家融合**
  - **一句话通俗解释**：在 AMD ROCm 平台上，把 DeepSeek 系列 MoE 模型里“所有 token 都要走”的共享专家，和“按需路由”的专家合并到一个大 kernel 里跑，省一次通信和启动开销。
  - **为什么社区在关注**：AMD GPU 正在被认真对待，MoE 推理优化从 NVIDIA 独享走向跨平台。共享专家融合是 MoE 推理的关键性能点。
  - **对新手意味着什么**：如果你在学 CUDA/ROCm kernel 优化，MoE 融合是一个非常有价值的方向，ROCm 生态人才缺口大。

- **关键词：投机解码元数据复用**
  - **一句话通俗解释**：投机解码（draft model 猜、target model 验）时，有些中间元数据不用每次重建，可以原地刷新，省 CPU 和显存带宽。
  - **为什么社区在关注**：投机解码是当前提升吞吐的主流手段，但元数据构建常成为瓶颈。这类优化说明社区在“抠”每一毫秒。
  - **对新手意味着什么**：想深入推理加速，投机解码是必修课。理解 metadata builder 的构建/复用机制，是读懂 vLLM v1 架构的关键。

- **关键词：XPU 生态适配**
  - **一句话通俗解释**：Intel GPU（XPU）上的图捕获、量化库版本、KV connector 测试等都在被逐步打通。
  - **为什么社区在关注**：异构计算不再是口号，Intel、AMD、NVIDIA 三足鼎立，vLLM 作为推理框架必须多平台可用。
  - **对新手意味着什么**：如果你手头只有 Intel Arc 或集成显卡，现在也能跑 vLLM 了。多平台适配经验在求职时是加分项。

- **关键词：P2P KV 卸载安全加固**
  - **一句话通俗解释**：分布式 KV 缓存点对点传输时，攻击者可以伪造大量 peer 地址耗尽 ZMQ socket，现在加了 peer 数量上限、握手超时和空闲回收。
  - **为什么社区在关注**：LLM 服务开始暴露在公网，安全不再是“以后再说”。这是 vLLM 安全文档中新增的一节。
  - **对新手意味着什么**：部署 LLM 服务时，除了性能，安全配置（环境变量、反向代理）也要懂一点，否则可能被一个请求打崩。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：Token ID 透传与端到端一致性校验**
  - **是什么**：聊天接口除了返回文本，还能返回模型生成的 token 数字 ID 列表，用于精确比对不同部署方式的结果是否一致。
  - **为什么重要**：在分布式推理（prefill/decode 分离）中，文本可能因解码差异看起来一样，但 token 序列不同。不懂这个，你会误判“结果一致”，实际底层已经跑偏。
  - **动手试试**：
    ```python
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        extra_body={"return_token_ids": True},
    )
    print(resp.choices[0].token_ids)
    ```
  - **预期结果**：你会看到一串整数（如 `[9707, 0, ...]`），而不是只有文本。用同一个 prompt 跑两次，token_ids 应该完全一致（贪心解码下）。

- **知识点 2：MoE 共享专家融合（ROCm/AITER）**
  - **是什么**：MoE 模型里有两类专家——共享专家（所有 token 都走）和路由专家（按 token 选择）。融合就是把共享专家的计算塞进路由专家的大 kernel 里，一次搞定。
  - **为什么重要**：不融合的话，共享专家要单独跑一次 GEMM + 通信，延迟高。融合后省一次 all-to-all，吞吐提升明显。不懂 MoE 结构，你调参时根本不知道瓶颈在哪。
  - **动手试试**：
    ```bash
    # 在 ROCm 机器上
    export VLLM_ROCM_AITER_MEGA_MOE_FUSE_SHARED_EXPERTS=True
    export MORI_SHMEM_HEAP_SIZE=16G
    vllm serve deepseek-ai/DeepSeek-V4-Flash --tensor-parallel-size 8
    ```
  - **预期结果**：启动日志里会看到 `AITER MegaMoE: MORI_SHMEM_HEAP_SIZE=16G`，推理吞吐比不开启融合时更高（可用 benchmark 脚本对比）。

- **知识点 3：投机解码元数据原地刷新**
  - **是什么**：投机解码时，draft model 每轮都要构建注意力元数据（哪些 token 可见、窗口多大）。原地刷新就是复用同一块显存，只更新变化的部分。
  - **为什么重要**：元数据构建涉及大量 CPU 计算和 H2D 拷贝，是投机解码的隐藏瓶颈。不懂这个，你会以为“投机解码就是加个 draft model”，实际性能可能不升反降。
  - **动手试试**：
    ```bash
    # 跑 vLLM 的投机解码测试
    pytest tests/v1/attention/test_triton_attn_metadata_builder.py -v
    ```
  - **预期结果**：测试会验证“刷新后的元数据”和“重新构建的元数据”完全一致。你会看到 `torch.testing.assert_close` 通过。

- **知识点 4：CUDA IPC Handle 的 ABI 兼容性**
  - **是什么**：多进程共享 GPU 显存时，需要用 `cudaIpcMemHandle_t` 这个 64 字节的不透明结构传递指针。新版 cuda-bindings 不再暴露 `.reserved` 字段，得用 `getPtr()` 拿原始字节。
  - **为什么重要**：ABI 变化是跨版本升级的经典坑。不懂这个，你的多进程推理代码在新版 CUDA 上会直接崩，而且报错信息很晦涩。
  - **动手试试**：
    ```python
    import ctypes
    from cuda import cudart
    handle = cudart.cudaIpcMemHandle_t()
    raw = ctypes.string_at(handle.getPtr(), 64)
    print(len(raw))  # 应该是 64
    ```
  - **预期结果**：输出 `64`。如果你用旧代码 `bytes(handle.reserved)`，在新版上会报 `AttributeError`。

- **知识点 5：P2P KV 卸载的资源限制**
  - **是什么**：分布式 KV 缓存传输时，每个 peer 连接占一个 ZMQ socket。不加限制的话，攻击者可以伪造大量 peer 地址把 socket 耗尽。
  - **为什么重要**：这是真实的安全漏洞。不懂这个，你部署的 LLM 服务可能被一个恶意请求打崩 EngineCore。
  - **动手试试**：
    ```bash
    export VLLM_P2P_MAX_PEERS=16
    export VLLM_P2P_HANDSHAKE_TIMEOUT_S=10
    export VLLM_P2P_IDLE_TIMEOUT_S=60
    # 启动带 P2P 卸载的 vLLM
    vllm serve <model> --kv-transfer-config '{"kv_connector":"OffloadingConnector",...}'
    ```
  - **预期结果**：超过 16 个 peer 时，新连接会被拒绝，日志里会看到 peer 上限警告，而不是直接崩溃。

---

## 🔥 重要更新

- **#61065 保留 chat derender 响应中的 token IDs**：分布式部署下 token 级别一致性校验的关键修复，影响所有用 `return_token_ids` 的场景。
- **#61004 ROCm 上融合共享专家到 AITER MegaMoE**：AMD GPU 上 DeepSeek 系列 MoE 推理的重大性能优化，需要 16G 共享内存。
- **#51504 修复 P2P ZMQ socket 耗尽安全漏洞**：新增 peer 数量上限、握手超时、空闲回收，公网部署必看。
- **#61060 RL 场景下释放权重传输通信器而不停止推理**：RLHF 训练与推理共存的场景，现在可以动态释放通信资源。

---

## 📋 逐条解读

### 1. [Bugfix][Frontend] Preserve token IDs in chat derender responses (#61065)

- **代码层面**：修改 `vllm/renderers/online_derenderer.py` 中的 `_derender_chat` 函数，新增 `return_token_ids` 判断逻辑。当请求要求返回 token IDs 且没有 parser 或 `include_reasoning` 为真时，把 token IDs 塞进响应。测试文件里加了 `assert d["token_ids"] == c["token_ids"]` 做一致性校验。
- **新手概念课堂**：想象你去餐厅点菜，服务员给你端上来的菜（文本）和厨房实际做的菜（token IDs）可能因为摆盘（解码）看起来一样，但食材不同。这个 commit 就是让服务员把“食材清单”也给你，方便你核对。
- **对你有什么影响**：如果你用 vLLM 做分布式推理，现在可以精确比对 prefill/decode 分离前后的 token 序列，不用担心“看起来一样但实际不同”。

### 2. [ROCm][DSv4] Fuse the shared expert into AITER MegaMoE (#61004)

- **代码层面**：在 `vllm/envs.py` 新增 `VLLM_ROCM_AITER_MEGA_MOE_FUSE_SHARED_EXPERTS` 环境变量。`vllm/distributed/device_communicators/all2all.py` 里根据该变量把 `MORI_SHMEM_HEAP_SIZE` 从 8G 调到 16G。测试文件新增 `test_shared_expert_route_appended` 验证共享专家路由追加逻辑。
- **新手概念课堂**：MoE 模型像一个大公司，共享专家是“所有员工都要参加的晨会”，路由专家是“各部门自己的会议”。融合就是把晨会塞进部门会议里一起开，省一次集合时间。
- **对你有什么影响**：如果你在 AMD GPU 上跑 DeepSeek V4，开启这个环境变量后吞吐会提升，但需要至少 16G 共享内存，小显存机器要注意。

### 3. [Spec Decode] Skip metadata build for dflash/dspark when all model backends support it (#60692)

- **代码层面**：新增 `tests/v1/attention/test_triton_attn_metadata_builder.py`，修改 `tests/v1/attention/test_deepseek_v4_swa_visible.py` 新增 `test_dspark_noncausal_draft_decode_update_matches_build`。核心是让 DSpark draft graph 原地刷新非因果 SWA 元数据，而不是重建。
- **新手概念课堂**：投机解码像“草稿纸+正式答卷”。以前每轮都要重新画草稿纸的格子（元数据），现在格子可以复用，只擦掉内容重写。
- **对你有什么影响**：用投机解码时，CPU 开销降低，吞吐更高。普通用户可能感知不到，但服务端 QPS 会提升。

### 4. [XPU] bump up auto-round-lib to 0.16.0 (#60976)

- **代码层面**：`requirements/xpu.txt` 里 `auto_round_lib` 从 0.15.0 升到 0.16.0。`tests/quantization/test_auto_round.py` 删除了 `XPU_EAGER_ONLY_MODELS` 和 `_runner_kwargs`，因为新版库修复了 XPU graph capture 问题。
- **新手概念课堂**：量化库升级就像手机系统更新，以前某些 App（模型）在 XPU 上跑图模式会闪退，现在修好了，不用再强制“省电模式”（eager mode）。
- **对你有什么影响**：Intel GPU 用户跑量化模型时，可以用图模式加速，不用再降级到 eager 模式。

### 5. [Bugfix][Distributed] Own SHM out-of-band buffers before releasing read slots (#61000)

- **代码层面**：`tests/distributed/test_shm_broadcast.py` 里重构了 `distributed_run`，用 `get_open_port()` 动态分配端口，修复超时检测逻辑。测试用例把大 tensor 单独 enqueue，避免 ring buffer 被覆盖。
- **新手概念课堂**：共享内存像传阅文件，以前传阅完就还回去，但别人可能还没看完。现在改成“先复制一份自己留着”，确保数据不被覆盖。
- **对你有什么影响**：分布式推理时，大 tensor 广播更可靠，不会出现“数据被覆盖导致结果错误”的诡异 bug。

### 6. [Bugfix][GLM-5.3-Flash] Restore MTP index sharing across draft iterations (#61079)

- **代码层面**：`vllm/config/speculative.py` 的 `hf_config_override` 里，对 `glm5_next` 模型新增 `index_share_for_mtp_iteration` 字段的传递。测试新增 `test_glm5_next_mtp_override_promotes_index_share`。
- **新手概念课堂**：MTP（多 token 预测）像接力赛，以前每棒都要重新确认接力区（index），现在接力区可以共享，省时间。
- **对你有什么影响**：用 GLM-5.3-Flash 做投机解码时，MTP 迭代更高效，生成速度更快。

### 7. [Bugfix][Frontend] Honor endpoint plugin route overrides (#58839)

- **代码层面**：`docs/design/endpoint_plugins.md` 更新文档，说明插件路由会覆盖核心路由。测试新增 `_ShadowEndpointPlugin` 和 `_RouterEndpointPlugin`，验证覆盖行为。
- **新手概念课堂**：插件像手机壳，以前手机壳不能遮住按键（核心路由），现在可以了，但你要小心别把重要按键遮住。
- **对你有什么影响**：如果你写 vLLM 插件，现在可以覆盖 `/version` 等核心路由，但文档提醒你要命名空间隔离，避免冲突。

### 8. [XPU][CI] Enable XPU graph in graph tests (#60784)

- **代码层面**：`tests/v1/cudagraph/test_breakable_cudagraph.py` 里把 `torch.cuda` 相关调用替换为 `DEVICE_TYPE`，XPU 上通过 `_torch_cuda_wrapper` 路由到 `torch.xpu`。
- **新手概念课堂**：图捕获像录制宏，以前只能在 NVIDIA 显卡上录，现在 Intel 显卡也能录了。
- **对你有什么影响**：Intel GPU 用户现在可以用 CUDA Graph 加速推理，延迟更低。

### 9. [XPU] limit FA kernel block size to multipleOf 64 (#57297)

- **代码层面**：`.buildkite/intel_jobs/test-intel.yaml` 里调整测试命令。`tests/v1/kv_connector/unit/` 下多个测试文件引入 `maybe_update_block_size`，把 block size 限制为 64 的倍数。
- **新手概念课堂**：FlashAttention kernel 像停车场，车位大小必须是 64 的倍数，否则停不进去。这个 commit 就是强制车位对齐。
- **对你有什么影响**：Intel GPU 上跑 FlashAttention 时，block size 必须设成 64 的倍数，否则会报错。

### 10. [Bugfix][RecoverSSM] Correct aligned state indices at exact block boundaries (#59962)

- **代码层面**：`tests/models/kimi_k3/test_kda.py` 里把 `accepted` 参数化，修正 `final_block` 的计算逻辑。`tests/v1/worker/test_mamba_hybrid_model_state.py` 新增多组参数化测试。
- **新手概念课堂**：SSM 状态像记账本，以前在“整页边界”处会记错页码，现在修正了。
- **对你有什么影响**：用 Kimi K3 或 Mamba 混合模型时，投机解码在 block 边界处的结果更准确。

### 11. [Bugfix] Read Ling quantization_config from top-level config for Bailing V3 VL (#59026)

- **代码层面**：`vllm/model_executor/models/bailing_moe_v3.py` 里，当 `config.quantization_config` 为 None 时，从 `vllm_config.model_config.hf_config.quantization_config` 读取。
- **新手概念课堂**：量化配置像说明书，以前只看小盒子（子配置），现在小盒子没有就看大盒子（顶层配置）。
- **对你有什么影响**：Bailing V3 VL 模型的量化配置现在能正确读取，FP8 量化不会失效。

### 12. [Fix] MiniMax: fix CUDA IPC handle exchange on cuda-bindings >= 13.4 (#60917)

- **代码层面**：`vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py` 里新增 `_ipc_handle_to_bytes` 和 `_ipc_handle_from_bytes`，用 `ctypes.string_at(handle.getPtr(), 64)` 替代 `bytes(handle.reserved)`。
- **新手概念课堂**：CUDA IPC handle 像身份证，以前可以直接看“姓名”字段，现在只能看“整张身份证的字节”，得自己解析。
- **对你有什么影响**：升级 cuda-bindings 到 13.4+ 后，MiniMax 模型的多进程推理不会崩。

### 13. [Bugfix][MRV2] Fix FlashInfer attention and ReplaySSM warmups on MRV2 (#61013)

- **代码层面**：新增 `vllm/model_executor/warmup/attention_warmup.py`，把 `mixed_batch_attention_warmup` 抽出来，MRV2 用 `run_mixed_prefill_decode_warmup`，MRV1 用 `_dummy_run`。新增测试 `tests/model_executor/test_attention_warmup.py`。
- **新手概念课堂**：预热像运动员赛前热身，MRV2 的热身动作和 MRV1 不同，以前用错了动作，现在分开了。
- **对你有什么影响**：用 MRV2（V2 模型运行器）时，FlashInfer 注意力和 ReplaySSM 的预热正确，首次推理不会出错。

### 14. [Bugfix][GLM-5.3-Flash] Honor `--mamba-ssm-cache-dtype` for the KDA recurrent state (#59147)

- **代码层面**：`vllm/models/glm5next/common/kda.py` 和 `model.py` 里，`kda_state_dtype` 新增 `mamba_ssm_cache_dtype` 参数。新增测试 `tests/models/glm5next/test_kda_state_dtype.py`。
- **新手概念课堂**：KDA 循环状态像笔记本，以前只能用一种纸（float32），现在可以选 bfloat16 的纸，省空间。
- **对你有什么影响**：用 GLM-5.3-Flash 时，`--mamba-ssm-cache-dtype bfloat16` 能生效，显存占用降低。

### 15. [MRV2] Switch CI tests still running on MRV1 unnecessarily (#61014)

- **代码层面**：`.buildkite/test-amd.yaml`、`.buildkite/test_areas/quantization.yaml` 等文件里删除 `unset VLLM_USE_V2_MODEL_RUNNER` 和 `VLLM_USE_V2_MODEL_RUNNER: "0"`。
- **新手概念课堂**：CI 像考试，以前有些科目还在用旧版教材（MRV1），现在统一用新版（MRV2）。
- **对你有什么影响**：普通用户无感知，但说明 MRV2 已经足够稳定，社区在全面迁移。

### 16. [Bugfix] Respect the configured offline chat template (#61061)

- **代码层面**：`vllm/entrypoints/common/offline.py` 里 `_preprocess_chat` 新增 `self.chat_template` 作为默认值。测试 `tests/entrypoints/multimodal/llm/test_mm_processor_kwargs.py` 里给 mock LLM 加 `chat_template = None`。
- **新手概念课堂**：离线推理像写信，以前用默认信纸（模板），现在可以用你配置的信纸。
- **对你有什么影响**：用 `LLM` 类做离线推理时，`chat_template` 参数能正确传递，不会静默忽略。

### 17. [Security] Fix P2P ZMQ socket exhaustion via unbounded peer sessions (#51504)

- **代码层面**：`docs/usage/security.md` 新增 P2P 安全章节。`vllm/envs.py` 新增 `VLLM_P2P_MAX_PEERS`、`VLLM_P2P_HANDSHAKE_TIMEOUT_S`、`VLLM_P2P_IDLE_TIMEOUT_S`。测试 `tests/v1/kv_offload/tiering/p2p/test_manager.py` 新增 idle 超时测试。
- **新手概念课堂**：ZMQ socket 像电话线，以前谁都能打进来占线，现在限制最多 128 条线，打不通就挂断。
- **对你有什么影响**：公网部署带 P2P 卸载的 vLLM 时，建议把 `VLLM_P2P_MAX_PEERS` 调低到 16-32，防止被恶意请求打崩。

### 18. [Sleep] Retain draft-private weights with existing parameter selector (#61046)

- **代码层面**：`docs/features/sleep_mode.md` 更新文档，说明 `sleep_preserve_parameter_names` 现在支持 `draft.*` 模式。`tests/v1/worker/test_gpu_worker.py` 新增 `test_sleep_preservation_restores_private_draft_not_shared_target`。
- **新手概念课堂**：睡眠模式像冬眠，以前只能保留“共享的脂肪”（目标模型权重），现在也能保留“私有的脂肪”（draft 模型权重）。
- **对你有什么影响**：RLHF 训练时，draft 模型权重不会被误删，唤醒后不用重新加载。

### 19. [Perf][GLM-5.3-Flash] Shard long-context indexer prefill rows across TP (#54951)

- **代码层面**：新增 `tests/v1/attention/test_indexer_tp_row_shard.py`，引入 `PrefillRowShard` 类，把 prefill 行按 TP rank 分片，用 `all_gatherv` 重组。
- **新手概念课堂**：长上下文 prefill 像大扫除，以前一个人扫全部，现在 8 个人各扫一片，最后拼起来。
- **对你有什么影响**：GLM-5.3-Flash 在长上下文场景下，TP 并行效率更高，prefill 速度更快。

### 20. [RL] Release weight-transfer communicators without stopping inference (#61060)

- **代码层面**：`rust/src/engine-core-client/src/client.rs` 新增 `shutdown_weight_transfer_engine` 方法。`rust/src/server/src/routes.rs` 新增 `/shutdown_weight_transfer_engine` 路由。测试更新调用次数从 10 到 11。
- **新手概念课堂**：权重传输通信器像临时租的会议室，以前退租要停掉整个公司（推理），现在可以只退会议室，公司继续营业。
- **对你有什么影响**：RL 训练和推理共存时，可以动态释放权重传输资源，不用重启推理服务。

## sglang
## 💡 今日关键词

- **关键词：DCP（Decode Context Parallel，解码上下文并行）**
  - **一句话通俗解释**：把长上下文推理时每张卡要扫描的 KV Cache 按 token 位置切分到多张卡上，每张卡只算一部分，最后一合并。这样单卡显存压力小，能支持更长的上下文。
  - **为什么社区在关注**：今天 Intel XPU 也支持 DCP 了（Commit 1），说明长上下文推理不再只是 NVIDIA 的专利，异构平台正在补齐关键能力。
  - **对新手意味着什么**：长上下文推理是当前 LLM 落地的核心痛点，掌握 KV 切分和通信合并逻辑，是进入推理框架优化的好切入口。

- **关键词：HiCache 与 MTP 状态保存**
  - **一句话通俗解释**：HiCache 是 SGLang 的多级 KV 缓存系统，MTP 是"一次预测多个 token"的投机解码。今天多个 commit 都在修 HiCache 对 MTP 草稿层和 Mamba 状态的保存问题。
  - **为什么社区在关注**：投机解码 + 多级缓存是提升吞吐的关键组合，但两者状态管理极其复杂，容易出现"缓存错位"导致结果错误。
  - **对新手意味着什么**：缓存一致性是分布式系统的经典难题，学懂这套机制，对理解任何带状态的服务系统都有帮助。

- **关键词：PD 分离（Prefill/Decode Disaggregation）**
  - **一句话通俗解释**：把"读 prompt"和"生成 token"两个阶段放到不同机器上跑，各自用最适合的硬件，提升整体效率。
  - **为什么社区在关注**：今天 Commit 6、19 都在修 PD 场景下的 abort、retract 语义，说明 PD 已经进入"细节打磨期"，是生产部署的主流方案。
  - **对新手意味着什么**：PD 是当前大模型推理服务的标准架构，理解它的失败恢复、请求生命周期管理，是后端工程师的必备技能。

- **关键词：异构平台适配（XPU / AMD / ROCm）**
  - **一句话通俗解释**：让 SGLang 不只在 NVIDIA GPU 上跑，也能在 Intel XPU、AMD GPU 上跑。
  - **为什么社区在关注**：今天有 XPU DCP、AMD MORI EP V2、XPU Triton kernel 等多个适配 commit，说明"去 NVIDIA 化"是真实需求。
  - **对新手意味着什么**：如果你会写跨平台 kernel 或熟悉 Triton/ROCm，这是非常稀缺的技能。

- **关键词：调度器公平性与缓存刷新语义**
  - **一句话通俗解释**：调度器要决定"谁先跑、谁等着"，如果逻辑有 bug，请求会被永久卡住；权重更新后必须清缓存，否则新权重配旧 KV 结果就错了。
  - **为什么社区在关注**：Commit 15、16 都在修这类"边界条件"，这类 bug 平时不显，压力大时才会暴露。
  - **对新手意味着什么**：调度和状态机是系统稳定性的核心，值得深入阅读源码。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：Decode Context Parallel（DCP）——把 KV Cache 切到多张卡

- **是什么**：正常推理时，每张 GPU 都要保存整个上下文的 KV Cache。DCP 把这个 Cache 按 token 位置切成 N 份，每张卡只存 1/N，解码时各卡算自己那部分注意力，再通过 All-to-All 或 AllGather+ReduceScatter 合并结果。
- **为什么重要**：不做 DCP，长上下文（比如 128K token）的 KV Cache 会直接把显存撑爆。不懂 DCP，你就无法理解为什么有些框架能跑 1M 上下文。
- **动手试试**：
  ```bash
  # 在 Intel XPU 上启动带 DCP 的服务（需 2 张卡）
  python -m sglang.launch_server \
      --model-path Qwen/Qwen3.5-4B \
      --device xpu \
      --attention-backend intel_xpu \
      --tp 2 --dcp-size 2 \
      --disable-radix-cache \
      --mem-fraction-static 0.60
  ```
  注意 `--dcp-size 2` 表示把 KV 切成 2 份，配合 `--tp 2` 使用。
- **预期结果**：服务能正常启动并处理长 prompt。如果启动时报 `UR_RESULT_ERROR_OUT_OF_RESOURCES`，说明显存不够，需要调低 `--mem-fraction-static`。

### 知识点 2：多级缓存（HiCache）中的"键命名空间"问题

- **是什么**：HiCache 会把 KV 缓存从 GPU 显存卸载到主机内存甚至磁盘。为了区分不同模型、不同 TP rank 的缓存，每份缓存都要加一个"后缀"作为命名空间。
- **为什么重要**：Commit 2 修的就是一个经典 bug：MLA 模型的 KV 是共享的（不加 TP rank 后缀），但 Mamba 状态其实是按 TP rank 分片的。如果不区分，rank 0 的 Mamba 状态会覆盖 rank 1 的，导致结果错误。
- **动手试试**：
  ```python
  # 模拟一下命名空间逻辑
  class CacheKey:
      def __init__(self, model_name, tp_rank, tp_size, is_mla):
          self.tp_suffix = f"_{model_name}_{tp_rank}_{tp_size}"
          self.config_suffix = f"_{model_name}" if is_mla else self.tp_suffix

      def get_key(self, key, pool_name=None):
          # Mamba 状态即使 MLA 模型也要按 TP rank 区分
          if pool_name == "MAMBA":
              return key + self.tp_suffix
          return key + self.config_suffix

  c = CacheKey("deepseek", 0, 2, is_mla=True)
  print(c.get_key("prefix1"))              # MLA KV: 共享
  print(c.get_key("prefix1", "MAMBA"))     # Mamba: 加 rank 后缀
  ```
- **预期结果**：你会看到 MLA KV 的 key 不带 rank，而 Mamba 的 key 带 rank。这解释了为什么 Commit 2 要单独处理 `PoolName.MAMBA`。

### 知识点 3：调度器的 `batch_is_full` 闩锁（Latch）问题

- **是什么**：调度器用一个布尔标志 `batch_is_full` 表示"当前批次满了，别再往里加请求"。如果这个标志被错误地永久置为 True，新请求就永远进不来。
- **为什么重要**：Commit 16 修的就是：当 running batch 为空时，标志没被清除，导致请求永远排队。这类"状态机死锁"是分布式系统最隐蔽的 bug。
- **动手试试**：
  ```python
  # 用伪代码理解这个 bug
  class Scheduler:
      def __init__(self):
          self.running_batch = []
          self.batch_is_full = False

      def try_admit(self, new_req):
          if self.batch_is_full:
              return "rejected"
          if len(self.running_batch) >= 2:
              self.batch_is_full = True
              return "rejected"
          self.running_batch.append(new_req)
          return "accepted"

      def finish_batch(self):
          self.running_batch = []
          # BUG: 忘了清 batch_is_full

  s = Scheduler()
  s.try_admit("r1"); s.try_admit("r2"); s.try_admit("r3")  # 第三个被拒
  s.finish_batch()
  print(s.try_admit("r4"))  # 修复前：rejected，修复后：accepted
  ```
  修复方法就是在 `finish_batch` 里加 `self.batch_is_full = False`。
- **预期结果**：修复后 `r4` 能被接受。这个实验让你体会"空批次必须复位标志"的重要性。

### 知识点 4：权重更新后必须刷新缓存

- **是什么**：RL 训练中，模型权重会不断更新。如果更新权重后不清空旧的 KV 缓存，新权重会配上旧 KV，输出就是错的。
- **为什么重要**：Commit 15 把"缓存刷新失败"从 `assert` 崩溃改成了"返回失败结果"。因为刷新失败通常是因为还有请求在跑，这时不应该直接崩，而应该告诉调用方"更新失败，请先清空请求"。
- **动手试试**：
  ```python
  # 对比两种错误处理风格
  def old_style(flush_ok):
      assert flush_ok, "Cache flush failed"  # 直接崩，调用方无法处理

  def new_style(flush_ok):
      if not flush_ok:
          return False, "缓存未刷新，因为还有请求在跑，请先撤回"
      return True, "成功"

  # 旧风格：调用方拿到的是崩溃，无法优雅降级
  # 新风格：调用方可以根据返回值决定是否重试
  print(new_style(False))
  ```
- **预期结果**：你会看到新风格返回 `(False, 错误信息)`，调用方可以据此决定"先撤回请求再重试"。这就是"把 assert 改成返回值"的工程价值。

### 知识点 5：Triton kernel 的跨平台写法（CUDA / XPU）

- **是什么**：Triton 是一种写 GPU kernel 的高级语言。以前很多代码写死了 `torch.cuda.device`，只能跑在 NVIDIA 卡上。现在改成 `torch.get_device_module(x.device).device(x.device)`，就能同时支持 CUDA 和 XPU。
- **为什么重要**：Commit 10 用这种方式让 FLUX.2 的 SwiGLU kernel 在 Intel XPU 上也能跑。这是"一次编写，多平台运行"的典型模式。
- **动手试试**：
  ```python
  import torch

  # 旧写法：写死 CUDA
  # with torch.cuda.device(x.device):
  #     kernel[...](...)

  # 新写法：根据设备类型动态获取模块
  def get_device_module(device):
      if device.type == "cuda":
          return torch.cuda
      elif device.type == "xpu":
          return torch.xpu
      else:
          raise ValueError(f"Unsupported device: {device}")

  # 用法
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"设备模块: {get_device_module(device)}")
  ```
- **预期结果**：在 CUDA 机器上输出 `torch.cuda` 模块，在 XPU 机器上输出 `torch.xpu` 模块。这就是 Commit 10 的核心改动模式。

---

## 🔥 重要更新

- **Commit 1（XPU DCP 支持）**：Intel XPU 首次支持 Decode Context Parallel，意味着长上下文推理在异构平台上成为可能，是硬件生态多样化的标志性事件。
- **Commit 4（HiCache 保存全部 MTP KV 和循环状态）**：修复了投机解码 + 多级缓存组合下的状态丢失问题，对生产环境的正确性至关重要。
- **Commit 13（FlashInfer 升级到 0.7.1）**：依赖升级，包含 SM120 Sage 后端等新特性，影响所有使用 FlashInfer 的用户。
- **Commit 19（PD 场景下撤回已传输的解码请求）**：修复了 PD 分离模式下暂停/恢复时的 KV 所有权问题，是生产部署稳定性的关键修复。

---

## 📋 逐条解读

### 1. [XPU] Support decode context parallelism (DCP) on Intel XPU

- **代码层面**：主要改的是 `attention_hook.py` 里的参数校验逻辑，以及文档 `dcp.mdx`。核心是让 `--dcp-comm-backend` 在 XPU 上也能解析为 `a2a`，并新增了对 MHA（非 MLA）模型在 XPU 上的 DCP 支持。
- **新手概念课堂**：DCP 就像"把一本厚书撕成几份，几个人各读一份，最后对答案"。以前只有 NVIDIA 的卡能这么干，现在 Intel 的卡也能了。
- **对你有什么影响**：如果你用 Intel GPU 跑长上下文模型，现在可以用 `--dcp-size` 来降低单卡显存压力了。文档里还给了完整的启动命令示例。

### 2. Keep NIXL Mamba checkpoint keys separate by TP rank

- **代码层面**：修改 `hicache_nixl.py` 中的 `_get_suffixed_key` 和 `_get_component_key`，当 `pool_name == PoolName.MAMBA` 时，强制加上 TP rank 后缀，即使模型是 MLA（MLA 的 KV 本来是共享的）。
- **新手概念课堂**：想象一个共享储物柜，MLA 模型的 KV 是"大家共用一把钥匙"，但 Mamba 状态是"每个人有自己的抽屉"。如果不给抽屉编号，大家的衣服就混在一起了。
- **对你有什么影响**：如果你用 MLA 模型 + Mamba 层 + NIXL 存储 + 多 TP rank，之前可能会遇到"结果莫名其妙不对"的问题，这个 commit 修复了它。

### 3. Capture KV host transfer buffers

- **代码层面**：重构 `MHATokenToKVPoolHost` 类，把设备端的 K/V buffer 捕获成元组 `device_k_buffers` / `device_v_buffers`，并新增 `_initialize_transfer_views` 方法。同时把 `device_pool` 从可选改为必需。
- **新手概念课堂**：以前每次传输都要去"问"设备池要 buffer，现在提前"拍照存档"，传输时直接用存档，速度更快。
- **对你有什么影响**：HiCache 的 host-to-device 传输会更快、更稳定，尤其是多 MTP 草稿池的场景。

### 4. [HiCache] Preserve all MTP KV and recurrent state

- **代码层面**：改了好几个文件，核心是让 HiCache 能正确处理 MTP（Multi-Token Prediction）的所有草稿层 KV 和循环状态。包括在 `model_config.py` 里修正 `full_attention_layer_ids` 的计算，以及新增 `mtp_draft_mamba_pools` 参数。
- **新手概念课堂**：MTP 就像"一次猜好几个字"，每个猜测都有自己的一套"草稿纸"（KV 和状态）。HiCache 要保存这些草稿纸，不然恢复时就得重新猜。
- **对你有什么影响**：用投机解码 + HiCache 时，之前可能丢失部分草稿状态导致加速失效，现在能完整保存了。

### 5. [HiCache] Support Mamba in buffer-only host memory mode

- **代码层面**：修改 `schedule_policy.py` 里的 `_mamba_gap_budget_for_req`，新增 `include_host_hit` 参数，用于在准入时预留 Mamba 状态槽位。同时修改 `pipeline.py` 支持 buffer-only 模式。
- **新手概念课堂**：Mamba 是一种"循环状态"模型，每个请求都要占一个"状态槽"。以前在纯主机内存模式下，调度器不知道要预留这些槽，导致请求被错误接纳后崩溃。
- **对你有什么影响**：如果你用 Mamba 模型 + buffer-only 主机内存模式，之前可能启动就崩，现在能正常跑了。

### 6. [PD] Poison the deferred abort ack when a Mooncake send fails

- **代码层面**：在 `conn.py` 里新增 `_poison_unsettled_send` 方法，当 Mooncake 传输失败时，把这个 room 标记为"已中毒"，防止后续的延迟 abort ack 被错误触发。
- **新手概念课堂**：Mooncake 是 PD 分离中的传输层。如果一次传输失败了，但"确认消息"还按计划发出，接收方会以为传输成功了，这就是"中毒"。所以要提前把这条路封死。
- **对你有什么影响**：PD 分离场景下，传输失败时不会再出现"假成功"导致的后续错误，稳定性提升。

### 7. [Perf] Route BF16 mixed-batch decode tails through decode attention

- **代码层面**：修改 `trtllm_mha_backend.py`，当 batch 是"混合形状"（前面是 prefill，尾部是单 token decode）时，把尾部单独走 decode attention kernel，而不是统一走 context attention。
- **新手概念课堂**：想象一个餐厅，前面几桌在点大餐（prefill），最后几桌只想加个甜点（decode）。以前都按大餐流程走，很慢；现在给甜点单独开个快速通道。
- **对你有什么影响**：混合 batch 场景下（比如连续对话），解码速度会更快。

### 8. [Fix] Link cuda_ipc JIT against CUDA driver stubs dir

- **代码层面**：在 `all_reduce.py` 里，把 `extra_ldflags=["-lcuda"]` 改成 `extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"]`，显式指定 CUDA driver stub 库的路径。
- **新手概念课堂**：编译时链接器要找到 `libcuda.so`，但有些系统里它不在默认路径。这个改动就是告诉编译器"去这个目录找"。
- **对你有什么影响**：在某些 CUDA 版本或系统配置下，之前 JIT 编译会失败，现在能成功了。

### 9. docs: clarify prefill CP terminology and simplify wording

- **代码层面**：纯文档修改，重写了 `prefill_cp.mdx` 里关于 Prefill CP、TP、DCP 的对比说明。
- **新手概念课堂**：Prefill CP 是"把 prompt 切成几段并行处理"，TP 是"把模型权重切开"，DCP 是"把解码时的 KV 切开"。三者容易混淆，文档现在讲清楚了。
- **对你有什么影响**：看文档时不会再被这几个概念绕晕了。

### 10. [diffusion][xpu] FLUX.2: enable the bit-exact Triton SwiGLU on XPU

- **代码层面**：把 `silu_mul_bitexact.py` 和 `flux_2.py` 里的 `x.is_cuda` 改成 `(x.is_cuda or x.is_xpu)`，把 `torch.cuda.device` 改成 `torch.get_device_module(x.device).device`。
- **新手概念课堂**：SwiGLU 是 Transformer 里的一个激活函数。以前这个 kernel 只认 NVIDIA 卡，现在 Intel 卡也能用了。
- **对你有什么影响**：在 Intel XPU 上跑 FLUX.2 图像生成模型，现在能用上这个优化 kernel，速度更快。

### 11. [Fix] Adapt to transformers 5.18 / 5.19 changes

- **代码层面**：多个文件的小改动，主要是适配 transformers 库的 API 变化。比如 `_commit_hash` 属性被移除、Dinov2 的层命名改了、`layer_types` 的校验逻辑变了。
- **新手概念课堂**：transformers 是 HuggingFace 的模型库，它升级时经常改 API。SGLang 要跟着改，不然就报错。
- **对你有什么影响**：如果你用最新版 transformers，之前可能加载模型失败，现在能正常加载了。

### 12. [XPU CI] Optimize PR CI per-file timeouts and fail hanging tests fast

- **代码层面**：在 `ci_utils.py` 里新增 `timeout_overrides` 参数，允许为特定测试文件单独设置超时时间。在 `run_suite.py` 里解析 `--timeout-override` 命令行参数。
- **新手概念课堂**：CI 是"自动测试流水线"。有些测试会卡死，以前要等很久才超时，现在可以给它们单独设个短超时，快速失败后重试。
- **对你有什么影响**：如果你给 SGLang 提 PR，CI 会跑得更快，不会因为某个测试卡死而等半天。

### 13. [Deps] Bump FlashInfer to 0.7.1

- **代码层面**：把 `Dockerfile`、`pyproject.toml`、`engine.py` 里的 FlashInfer 版本从 `0.7.0.post1` 改成 `0.7.1`。
- **新手概念课堂**：FlashInfer 是一个高性能注意力 kernel 库。升级版本就像给车换个新引擎，性能和功能都会提升。
- **对你有什么影响**：安装 SGLang 时会自动装 FlashInfer 0.7.1，包含 SM120 Sage 后端等新特性。

### 14. docs: sync LMSYS SGLang blog cards

- **代码层面**：纯文档修改，更新 `docs/index.mdx` 里的博客卡片链接和标题。
- **新手概念课堂**：这是自动同步脚本干的活，把 LMSYS 官网的最新博客同步到 SGLang 文档首页。
- **对你有什么影响**：文档首页能看到最新的 SGLang 相关博客。

### 15. [RL] Return a failed weight update when the cache flush is refused

- **代码层面**：把 `flush_cache_after_weight_update` 从返回 `None` 改成返回 `(success, message)` 元组。当缓存刷新失败时（因为还有请求在跑），返回 `False` 和错误信息，而不是直接 `assert` 崩溃。
- **新手概念课堂**：RL 训练时权重会不断更新。更新后必须清缓存，但如果还有请求在跑，清缓存会失败。以前直接崩，现在优雅返回错误。
- **对你有什么影响**：RL 训练中，如果缓存刷新失败，你能拿到明确的错误信息，知道要先撤回请求再重试，而不是看到崩溃日志。

### 16. [Fix] scheduler: stop latching batch_is_full when nothing can run

- **代码层面**：在 `scheduler.py` 的 `_get_new_batch_prefill_raw` 里，新增"如果 running batch 为空，就清除 `batch_is_full` 标志"的逻辑。同时把之前只在 hierarchical cache 场景下才做的"检查 can_run_list"逻辑推广到所有场景。
- **新手概念课堂**：`batch_is_full` 就像"餐厅满座"的牌子。如果餐厅空了但牌子还挂着"满座"，新客人就永远进不来。这个 commit 就是在餐厅空了时把牌子摘掉。
- **对你有什么影响**：修复了请求被永久卡在队列里的 bug，提升了调度器的鲁棒性。

### 17. [Test] Re-anchor the VLM flashinfer throughput floor

- **代码层面**：把 `test_vlm_serving_flashinfer.py` 里的 `output_throughput` 阈值从 5940 降到 4510。
- **新手概念课堂**：性能测试有"及格线"，如果实际性能低于及格线就报错。这次是把及格线调低了，可能是因为硬件或环境变化。
- **对你有什么影响**：CI 不会因为性能波动而误报失败。

### 18. [DSV4.1] Reduce eager Engram extend preparation

- **代码层面**：修改 `engram_hash.py` 里的 Triton kernel，新增 `ROWS_FROM_STARTS` 模式，用二分查找在 kernel 内解析行索引，而不是在 Python 侧用 `repeat_interleave` 物化一个大张量。
- **新手概念课堂**：Engram 是 DeepSeek V4.1 里的一个组件。以前准备输入数据时要生成一个很大的索引张量，很慢；现在在 kernel 里用二分查找实时算，省了内存和带宽。
- **对你有什么影响**：DeepSeek V4.1 的 extend 阶段准备会更快。

### 19. [PD] Retract transferred decode requests on a retract pause

- **代码层面**：在 `scheduler.py` 的 `pause_generation` 里，新增逻辑：PD decode 模式下，把 waiting queue 里"已经持有 KV"的请求也加入 retract 列表。同时在 `is_fully_idle` 里，如果 waiting queue 里有持有 KV 的请求，就报告"不空闲"。
- **新手概念课堂**：PD decode 模式下，有些请求已经拿到了传输过来的 KV，但还在等运行槽位。暂停时如果不撤回它们，它们的 KV 会被错误释放。
- **对你有什么影响**：PD 分离模式下暂停/恢复更安全，不会出现 KV 被错误释放导致的崩溃。

### 20. [AMD][MORI] Add MORI EP V2 support

- **代码层面**：新增 `_resolve_mori_ep_v2` 函数，根据环境变量 `SGLANG_MORI_EP_V2`、模型类型、`ep_size`、dtype 等条件决定是否使用 MORI EPv2。同时新增 `SGLANG_MORI_RECV_BOUND` 环境变量控制接收缓冲区大小。
- **新手概念课堂**：MORI 是 AMD 的通信库，EP 是"专家并行"（MoE 模型里把不同专家放到不同卡上）。EPv2 是新版本，性能更好。
- **对你有什么影响**：如果你在 AMD GPU 上跑 DeepSeek-V4 等 MoE 模型，现在可以用 MORI EPv2 获得更好性能。文档里详细说明了自动选择的规则。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)