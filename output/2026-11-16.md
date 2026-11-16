# 🚀 vLLM & SGLang 每日更新 — 2026-11-16

> 自动生成于 2026-11-16 08:33 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：Rust 前端（Rust Frontend）**
  - 一句话通俗解释：vLLM 正在用 Rust 重写 API 服务器/请求处理层，逐步替代 Python 前端，以换取更低延迟和更高并发。
  - 为什么社区在关注：今天 20 个 commit 里有 6 个（#67871、#67873、#67870、#63912、#67869、#63749、#61788）都在给 Rust 前端补功能、修 bug，说明 Rust 前端正从"实验性"走向"功能对齐 Python"的阶段。
  - 对新手意味着什么：如果你会一点 Rust，现在正是参与 vLLM 核心路径开发的好时机；即使不会，也应该开始关注 `VLLM_USE_RUST_FRONTEND` 这个开关，因为它会影响未来的部署方式。

- **关键词：序列并行 × 流水线并行（SP + PP）**
  - 一句话通俗解释：以前"序列并行"只能在没有"流水线并行"时用，现在两者可以叠加，让大模型在多机多卡上切得更细、跑得更快。
  - 为什么社区在关注：#66866、#65947、#66632 三个 commit 连续打通 Qwen3-Next/Qwen3.5/DeepSeek-V2/V3 的 SP+PP 组合，说明大模型分布式推理正在往"多维并行"演进。
  - 对新手意味着什么：想学分布式推理，SP、TP、PP、EP 这几个缩写的组合关系是必修课，现在正是理解它们如何交互的好时机。

- **关键词：多模态预处理（Multimodal Preprocessing）**
  - 一句话通俗解释：把图片/视频/音频变成模型能吃的 token 和 embedding 的那套流程。
  - 为什么社区在关注：#63912（透明图合成白底）、#67744（LLaVA-OneVision 不等帧数）、#67237（GLM-5.3-Flash 强制 Rust 预处理）、#67870（多模态缓存 key 域名标记）——多模态正确性和缓存一致性是当前高频 bug 区。
  - 对新手意味着什么：多模态是 vLLM 增长最快的方向之一，理解"预处理 → 缓存 key → 引擎"这条链路能让你快速定位大部分 VLM 问题。

- **关键词：KV Cache 卸载与分层（KV Offload / Tiering）**
  - 一句话通俗解释：把 GPU 显存放不下的 KV cache 挪到 CPU 内存、磁盘或远端，按需换入换出。
  - 为什么社区在关注：#67788（tiering 指标字节数）、#67258（SharedOffloadRegion 支持自定义 mmap 路径）说明 KV offload 正从"能跑"走向"可观测、可配置"。
  - 对新手意味着什么：长上下文推理的成本大头在 KV cache，学这块能直接帮你在生产环境省钱。

- **关键词：专用解码内核（Specialized Decode Kernel）**
  - 一句话通俗解释：针对特定模型、特定 GPU 手写的极致优化推理内核，而不是通用路径。
  - 为什么社区在关注：#66324 给 MiniMax-M3.1 在 GB300 上做了 CuTe DSL 专用解码，最高 1.45x 加速；#60801 给 Intel XPU 的 Mamba2 加了 tensor descriptor 路径。
  - 对新手意味着什么：性能优化的天花板越来越依赖"模型 × 硬件"的定制化，学 CUDA/Triton/CuTe DSL 的回报在上升。

## 🧪 重要知识点 & 动手实验

- **知识点 1：多模态缓存 key 的"域名标记"（Domain-tagging）**
  - **是什么**：给每个多模态输入的缓存 key 加前缀，比如客户端传的 `uuid` 变成 `uuid:xxx`，内容哈希变成 `content:xxx`。
  - **为什么重要**：如果不加前缀，客户端可以故意传一个和别的图片内容哈希一模一样的 `uuid`，从而"撞"到别人的缓存条目——既是正确性 bug，也是安全隐患。这是典型的"命名空间污染"问题。
  - **动手试试**：
    ```python
    # 模拟 commit #67870 的逻辑
    def cache_key(uuid, content_hash):
        return f"uuid:{uuid}" if uuid else f"content:{content_hash}"

    # 攻击场景：客户端 uuid 恰好等于另一张图的内容哈希
    victim_hash = "abc123"
    attacker_uuid = "abc123"
    print(cache_key(attacker_uuid, "deadbeef"))  # uuid:abc123
    print(cache_key(None, victim_hash))          # content:abc123
    # 两者不再相等，攻击失效
    ```
  - **预期结果**：你会看到两个 key 不再相同，理解"加前缀 = 隔离命名空间"这个通用设计模式。

- **知识点 2：序列并行 + 流水线并行的边界处理**
  - **是什么**：序列并行（SP）把一条序列的 token 切到多张卡上算；流水线并行（PP）把模型层切成多段放到不同卡上。两者叠加时，PP 阶段之间传的 hidden_states 必须是"完整 token、TP 复制"的形态，所以 SP 分片后要在 PP 边界 `all_gather` 拼回去。
  - **为什么重要**：不懂这个边界，SP+PP 一起开就会得到错误结果或直接 assert 失败。commit #66866/#65947 就是在补这个 `all_gather` + `split` 逻辑。
  - **动手试试**：
    ```python
    import torch
    import torch.distributed as dist

    # 单进程模拟：假设 2 个 TP rank 各持有半条序列的 hidden_states
    hidden_size = 8
    full_tokens = 6
    # rank0 和 rank1 各持有 3 个 token
    shard0 = torch.randn(3, hidden_size)
    shard1 = torch.randn(3, hidden_size)

    # PP 边界：all_gather 拼回完整 token
    packed = torch.cat([shard0, shard1], dim=0)
    print("packed shape:", packed.shape)  # [6, 8]
    # 注意：真实实现里 all_gather 是沿 dim=0 收集，然后截断到 full_num_tokens
    ```
  - **预期结果**：理解"SP 分片 → PP 边界 all_gather → 下一 stage 重新分片"这个往返流程；真实代码见 `vllm/model_executor/models/qwen3_next.py` 的 `forward`。

- **知识点 3：透明图片的 alpha 合成策略**
  - **是什么**：PNG 图片带 alpha 通道（透明度），模型通常不吃 alpha，所以要把透明像素"合成"到一个背景色上。Python 版 vLLM 默认合成到白色，Rust 前端之前没做，导致同一张图两边结果不一致。
  - **为什么重要**：多模态模型对像素值极其敏感，透明区域合成到黑还是白，可能让输出完全不同。这是"跨实现一致性"的经典坑。
  - **动手试试**：
    ```python
    from PIL import Image
    import numpy as np

    # 一张全透明的 4x4 图
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    # 合成到白底
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white, img)
    print(np.array(composited)[0, 0])  # [255 255 255 255]
    # 合成到黑底
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    print(np.array(Image.alpha_composite(black, img))[0, 0])  # [0 0 0 255]
    ```
  - **预期结果**：你会看到同一张透明图在不同背景下得到完全不同的像素值，理解 commit #63912 为什么要统一为"合成白底"（Kimi-K3 例外，因为它自己处理 alpha）。

- **知识点 4：分层 KV 卸载的字节数统计**
  - **是什么**：KV offload 有多个 tier（GPU→CPU→磁盘/远端），每个 transfer job 完成后要上报"传了多少字节"。commit #67788 修的是：之前用"完成的 key 数 × chunk size"估算，现在优先用 job 实际报告的 `transfer_bytes`。
  - **为什么重要**：指标不准会导致容量规划错误——你以为只写了 16 字节，实际写了 32 字节，磁盘很快就满了。
  - **动手试试**：
    ```python
    # 模拟 metrics.py 的逻辑
    def compute_transfer_bytes(completed_key_count, chunk_size, reported_bytes):
        if reported_bytes is not None:
            return reported_bytes
        return completed_key_count * chunk_size

    print(compute_transfer_bytes(2, 16, None))   # 32（回退估算）
    print(compute_transfer_bytes(2, 16, 0))      # 0（实际没传）
    print(compute_transfer_bytes(2, 16, 16))     # 16（部分成功）
    ```
  - **预期结果**：理解"优先用实测值、回退到估算值"这个指标上报模式，以及 `None` vs `0` 的语义区别（`None` = 不知道，`0` = 确实为 0）。

- **知识点 5：NCCL 打包加载的流顺序（Stream Ordering）**
  - **是什么**：commit #65487 修的是"打包 NCCL 广播"里，consumer 端的 unpack 操作可能和调用者 stream 上的写入操作乱序——需要显式插入 event 等待，保证 unpack 在调用者 stream 完成后才执行。
  - **为什么重要**：GPU 是异步执行的，不同 stream 之间没有顺序保证。不插入 event 等待，就会出现"数据还没写完就开始读"的竞态，表现为偶发的权重损坏——极难 debug。
  - **动手试试**：
    ```python
    import torch

    if torch.cuda.is_available():
        a = torch.zeros(1024, device="cuda")
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        with torch.cuda.stream(s1):
            a.fill_(1.0)
        # 错误做法：直接切到 s2 读，可能读到 0
        # 正确做法：s2 等待 s1
        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            b = a.clone()
        torch.cuda.synchronize()
        print(b.min().item())  # 应为 1.0
    ```
  - **预期结果**：注释掉 `s2.wait_stream(s1)` 后多跑几次可能看到 0；加上后稳定为 1.0。这就是 `torch.Event` + `stream.wait_event` 的用途。

## 🔥 重要更新

- **#66324 [Perf][MiniMax-M3.1] GB300 专用解码最高 1.45x**：这是今天性能收益最大的 commit，用 CuTe DSL 为 MiniMax-M3.1-Flash 在 GB300 上手写了专用 decode 内核，并附带 drafter 加速。如果你在跑这个模型，升级后 decode 吞吐会明显提升。
- **#66866 / #65947 SP + PP 打通 Qwen3-Next/Qwen3.5/DeepSeek-V2/V3**：以前开 SP 就必须关 PP，现在可以叠加，意味着长序列 + 大模型的多机部署组合更灵活，是分布式推理能力的重要补齐。
- **#67237 GLM-5.3-Flash 强制 Rust 前端做多模态**：Python 多模态预处理器被移除，Python LLM/API server 会直接拒绝多模态初始化。这是一个"强制迁移"信号，用这个模型的用户必须切到 `VLLM_USE_RUST_FRONTEND=1`。
- **#65487 修复打包 NCCL 加载的流顺序竞态**：这类 bug 表现为偶发的权重损坏，极难复现和定位，修复后大规模权重加载的稳定性会显著提升。

## 📋 逐条解读

### 1. [Rust Frontend] Support --hf-token (#67871)
- **代码层面**：在 `rust/src/chat/src/backend/mod.rs` 的 `LoadModelBackendsOptions` 里新增 `hf_token: Option<HfToken>` 字段，并在 `rust/src/cmd/src/cli.rs` 的 `RenderArgs`/`SharedRuntimeArgs` 里把 CLI 的 `--hf-token` 一路传递到 `hf.rs` 的 `load_model_backends`。简单说：给 Rust 前端补上了"下载私有模型时用的 HuggingFace token"这个参数。
- **新手概念课堂**：`--hf-token` 就像你去私人仓库取快递时出示的会员卡。没有它，公开模型随便下，私有/受限模型（比如 gated repo）就下不了。Rust 前端之前缺这张卡，所以只能下公开模型。
- **对你有什么影响**：如果你用 Rust 前端加载需要登录的模型（如 Llama 系列），现在可以直接 `vllm-rs serve --hf-token xxx`，不再需要靠环境变量 `HF_TOKEN` 兜底。

### 2. [Rust Frontend] Support --default-mm-loras (#67873)
- **代码层面**：在 `rust/src/cmd/src/cli.rs` 新增 `default_mm_loras: DefaultMmLoras` 字段，CLI 参数 `--default-mm-loras` 接收一个 JSON（如 `{"audio": "org/speech-lora"}`），并通过 `python_args` 转发给引擎。核心逻辑：每个模态可以配一个默认 LoRA，请求没指定 adapter 但带了该模态时自动使用。
- **新手概念课堂**：LoRA 是"模型的小补丁"，只训练少量参数就能让模型学会新技能。`--default-mm-loras` 就是"按模态自动贴补丁"——你说的是音频，系统就自动贴上音频专用的 LoRA，不用每次请求手动指定。
- **对你有什么影响**：做多模态服务时，可以按模态预挂 LoRA，客户端代码更简洁；前提是开了 `--enable-lora`。

### 3. [Rust Frontend] Domain-tag multimodal cache keys like Python (#67870)
- **代码层面**：`rust/src/chat/src/multimodal.rs` 里把 `identifier` 和 `mm_hash` 从"直接用 uuid 或 hash"改成 `uuid:{uuid}` / `content:{hash}` 格式。测试同步更新（`uuid:image-1` 等）。这是一次安全 + 一致性修复。
- **新手概念课堂**：想象两个人在同一个储物柜系统里存东西，一个人用"身份证号"当柜号，另一个人用"物品指纹"当柜号。如果身份证号恰好等于别人的物品指纹，就会拿错东西。加前缀 `uuid:` / `content:` 就是给两类编号划清"行政区"。
- **对你有什么影响**：普通用户感知不到，但如果你在客户端传自定义 `uuid`，现在它不会意外命中别人的缓存条目了——多模态结果更可靠。

### 4. [Bugfix][Rust Frontend] Composite transparent images over white like Python (#63912)
- **代码层面**：`rust/src/chat/src/multimodal.rs` 的 `fetch_media` 里新增 `TransparencyPolicy`：默认 `CompositeOverWhite`，除非图片处理器自己处理 alpha（如 Kimi-K3，用 `KeepAlpha`）。同时新增两个测试验证透明像素变白、以及策略跟随处理器。
- **新手概念课堂**：透明 PNG 的像素是 RGBA 四通道，模型通常只吃 RGB。把透明像素"贴"到白纸上，就是 alpha 合成。Python 版一直这么做，Rust 版之前漏了，导致同一张图两边结果不同。
- **对你有什么影响**：用 Rust 前端跑 VLM 时，带透明背景的图片（logo、图标、截图）现在和 Python 前端结果一致了，不再出现"换个前端输出就变"的诡异现象。

### 5. [Rust Frontend] Return 500 for engine errors on non-streaming generate (#67869)
- **代码层面**：`rust/src/server/src/routes/inference/generate.rs` 的 `collect_generate` 里，把 `FinishReason::Error(Some(error))` 的匹配改成 `FinishReason::Error(error)`，无论有没有具体错误信息都返回 `engine_error`。新增测试断言非流式请求遇到引擎错误返回 HTTP 500 + `InternalServerError`。
- **新手概念课堂**：以前引擎出错时，非流式请求会返回一个"看起来成功、但 finish_reason 是 error"的响应——客户端容易误以为拿到了正常结果。现在改成直接 500，让错误"响亮地失败"（fail loud），符合 HTTP 语义。
- **对你有什么影响**：客户端错误处理要检查 HTTP 状态码了；如果你之前依赖 `finish_reason == "error"` 判断，现在要改成看 5xx。

### 6. [Bugfix][Rust Frontend] Return 400 for media the request got wrong (#63749)
- **代码层面**：`rust/src/chat/src/error.rs` 新增 `UnsupportedInputAudioFormat` 和 `Media(MediaConnectorError)` 变体；`is_bad_request` 里根据 `MediaFault::Client` 判断是否 400。新增测试验证：请求本身有问题（如 data URL 格式错）→ 400；媒体服务器超时/解码器内部失败 → 500。
- **新手概念课堂**：HTTP 400 是"你的请求有问题"，500 是"服务器自己出问题了"。以前所有媒体错误都混成 500，客户端无法区分"我该改请求"还是"该重试"。现在按"谁的锅"分流。
- **对你有什么影响**：调用多模态 API 时，拿到 400 说明你的图片/音频格式或 URL 有问题，改请求即可；拿到 500 才是服务端问题，可以重试。

### 7. [Bugfix][ROCm][MoRIIO] Slice WRITE heads by each layer's head count (#62474)
- **代码层面**：`tests/v1/kv_connector/unit/moriio/test_moriio_hetero_tp_write.py` 里给 `Layer` 加了 `heads` 和 `draft` 字段，新增 `_ArchConfig`/`_model_config` 辅助类模拟"每层 KV head 数不同"的模型（如 Gemma4：sliding 层 4 个 head，full 层 8 个）。核心修复是 WRITE 阶段按每层自己的 head 数切片，而不是用模型全局最大值。
- **新手概念课堂**：KV head 是注意力里"存 key/value 的头"。有些模型不同层 head 数不一样（混合注意力），如果统一按最大数切片，少 head 的层会读到越界数据。就像给不同尺码的人发统一大号衣服，小个子会穿成麻袋。
- **对你有什么影响**：在 ROCm + MoRIIO KV 连接器 + 混合 head 模型（Gemma4 类）的组合下，KV 传输不再出错；普通用户感知不到，但跑异构 TP 的部署会受益。

### 8. [Bugfix][KV Offload] Report transferred bytes in tiering metrics (#67788)
- **代码层面**：`vllm/v1/kv_offload/tiering/metrics.py` 的 `_observe_finished_job_stats` 里，优先使用 `JobResult.transfer_bytes`（如果非 None），否则回退到 `completed_key_count * primary_chunk_size`。测试新增参数化用例覆盖 `None/0/16/32` 四种情况。
- **新手概念课堂**：指标上报有两种值——"我猜的"（key 数 × 块大小）和"实测的"（job 自己报告的字节数）。猜的会在部分成功、压缩、对齐等场景下失真。修复后优先用实测值。
- **对你有什么影响**：如果你用 Prometheus 监控 KV offload 的读写字节数，数字会变准；容量规划更可靠。

### 9. [Bugfix][MRV2] Apply EVS-recomputed M-RoPE positions on every PP stage (#66632)
- **代码层面**：新增 `vllm/v1/worker/gpu/pp_utils.py` 到 CI 依赖；`tests/models/multimodal/generation/test_qwen2_5_vl.py` 新增 PP=1 vs PP=2 的 prompt logprobs 对比测试。核心修复：EVS（视频采样）在第一个 PP stage 重算的 M-RoPE 位置，要传给后续所有 PP stage，否则后续 stage 用旧位置导致 logprobs 偏差约 0.2。
- **新手概念课堂**：M-RoPE 是多模态的位置编码（时间+高+宽三个维度）。视频抽帧后帧数变了，位置要重算。PP 把模型切段后，只有第一段跑视觉编码器，它算出的新位置必须"接力"传给后面的段——否则前后段对"第几帧"的理解不一致。
- **对你有什么影响**：用 Qwen2.5-VL + 视频 + PP 部署时，输出质量不再因 PP 而下降；这是多模态 + 分布式交叉的典型 bug。

### 10. [Bugfix][Multimodal] Fix LLaVA-OneVision embeddings for unequal video frame counts (#67744)
- **代码层面**：`vllm/model_executor/models/llava_onevision.py` 的 `_process_video_pixels` 里，把 `image_newline` 从 `[None, None, :]` 改成 `[None, :]`，reshape 从 `(1, num_frame*..., -1)` 改成 `(num_frame*..., -1)`，cat 的 dim 从 1 改成 0。新增测试 `test_llava_onevision_mixed_frame_count_videos` 验证不等帧数视频的 token 数正确。
- **新手概念课堂**：一个请求里有两个视频，帧数不同（2 帧和 3 帧）。拼 embedding 时如果维度对不齐，就会广播错或丢 token。修复是把每个视频的 embedding 当成"一维序列"拼接，而不是硬塞进 batch 维度。
- **对你有什么影响**：一次请求传多个不同长度视频给 LLaVA-OneVision 时，不再丢 token、不再输出 NaN logprobs。

### 11. [Rust Frontend] Pass pulled object storage model dir to the Rust frontend (#61788)
- **代码层面**：`tests/entrypoints/launchers/api_server/test_api_server_process_manager.py` 新增测试，验证当模型来自 `s3://bucket/model` 时，Python 侧先把 config/tokenizer 拉到本地临时目录，然后把该本地目录作为 `model_tag` 传给 Rust 前端；`served_model_name` 保持用户指定值。
- **新手概念课堂**：Rust 前端不会读 S3，所以 Python 侧要"先下好菜再交给厨师"——把对象存储里的模型文件拉到本地目录，再把目录路径告诉 Rust。这是"胶水层"的典型职责。
- **对你有什么影响**：用 `s3://`/`gs://` 等对象存储路径启动 Rust 前端时，现在能正常工作了；`--served-model-name` 也正确透传。

### 12. [Bugfix][Distributed] Order packed NCCL loads after the caller stream (#65487)
- **代码层面**：`tests/distributed/test_packed_tensor.py` 重构了 roundtrip 测试，引入 `torch.cuda._sleep(SLEEP_CYCLES)` 和 `torch.Event` 来检测"unpack 时调用者 stream 的写入是否已完成"。核心修复是在 consumer 端 unpack 前插入 event 等待，保证顺序。
- **新手概念课堂**：CUDA 有多个 stream（流水线），同一 stream 内有序，跨 stream 无序。打包广播的 producer 在 stream A 写，consumer 在 stream B 读，如果不等 A 完成就读，会读到旧数据。`torch.Event` 就是"信号灯"：A 完成后亮灯，B 看到灯才走。
- **对你有什么影响**：大规模权重加载（如多机 TP 广播）时偶发的"权重损坏/NaN"问题会减少；这类 bug 以前极难复现。

### 13. [Kernel][XPU] Mamba2 SSD: tensor-descriptor path for _chunk_scan_fwd_kernel and _chunk_state_fwd_kernel (#60801)
- **代码层面**：新增 Intel BMG GPU 的 CI job（`.buildkite/intel_jobs/kernels_intel.yaml`），`tests/kernels/mamba/test_mamba_ssm_ssd.py` 新增 `ssd_td` 测试。核心是给 Mamba2 的 SSD（state space dual）kernel 加了 tensor descriptor 路径——用硬件 TMA（Tensor Memory Accelerator）替代手动地址计算，减少寄存器压力和地址计算开销。
- **新手概念课堂**：Tensor descriptor 就像"快递单"——你告诉硬件"从这块内存按这个形状取数据"，硬件自己搬运，不用你一条条算地址。Intel GPU 的 TMA 类似 NVIDIA 的 TMA，能显著提升大块数据搬运效率。
- **对你有什么影响**：在 Intel Arc/Flex GPU 上跑 Mamba2 模型（如 Codestral Mamba）会更快；普通 NVIDIA 用户无感知。

### 14. [Fast Start] Add weights mirror support for daemon2daemon (#64641)
- **代码层面**：`docs/features/preload.md` 大幅扩写，新增"Seeding a replica from another"章节；`vllm/model_executor/model_loader/weight_cache/` 相关代码支持 `peer_cache` 加载格式——第二个副本的 daemon 直接从已加载的副本 daemon 拉取"后处理、TP 分片后"的权重，而不是重读 checkpoint。CI 新增跨主机 D2D seeding 测试。
- **新手概念课堂**：Fast Start 是"预热"机制——先把模型加载到共享内存，后续实例秒级启动。新功能是"副本间镜像"：A 副本已经加载好了，B 副本不用再读磁盘，直接从 A 的内存里拷，省掉 checkpoint 解析和 TP 切分的时间。
- **对你有什么影响**：多副本部署时，第二个及以后的副本启动时间大幅缩短；`--load-format peer_cache` 是新的可选值，`vllm preload` 不接受它（daemon 必须从磁盘或 peer 加载）。

### 15. [Multimodal] Require Rust preprocessing for GLM-5.3-Flash (#67237)
- **代码层面**：`docs/features/multimodal_inputs.md` 新增 GLM-5.3-Flash 章节；`tests/models/glm5next/test_epd_metadata.py` 新增测试验证 Python LLM 在 `VLLM_USE_RUST_FRONTEND=0` 时抛 `ValueError("requires the Rust frontend")`。Python 多模态预处理器被移除，模型从 `mm_processing.md` 的 Python 支持表中删除。
- **新手概念课堂**：这是"强制迁移"——某个模型的多模态预处理只在 Rust 前端实现了，Python 路径直接删掉。就像某个 App 只出 64 位版，32 位设备直接装不上。
- **对你有什么影响**：用 GLM-5.3-Flash 做多模态必须 `VLLM_USE_RUST_FRONTEND=1` 或 `vllm-rs serve`；纯文本推理可以设 `limit_mm_per_prompt={"image": 0, "video": 0}` 继续用 Python。

### 16. [Model] Support sequence parallelism with pipeline parallelism for Qwen3-Next, Qwen3.5 and DeepSeek-V2/V3 (#66866)
- **代码层面**：`deepseek_v2.py`、`qwen3_5.py`、`qwen3_next.py` 里删掉 `parallel_config.pipeline_parallel_size == 1` 的限制条件；在 PP 非最后 rank 的 `forward` 里新增：如果用了 SP，先把 `hidden_states` 和 `residual` 拼起来 `all_gather` 到完整 token，再 split 回去传给下一 stage。
- **新手概念课堂**：SP 把序列切碎分给多卡，PP 把层切段分给多卡。两者叠加时，PP 段与段之间传的数据必须是"完整序列"（因为下一段要重新切），所以要在边界"拼回去再传"。就像接力赛，交接棒时必须完整，下一棒再自己决定怎么握。
- **对你有什么影响**：Qwen3-Next/Qwen3.5/DeepSeek-V2/V3 现在可以同时开 SP 和 PP，长序列 + 大模型的多机部署更灵活、吞吐更高。

### 17. [Perf][MiniMax-M3.1] MiniMax-M3.1-Flash Specialized Decode: CuTe DSL Optimized GB300 (up to 1.45x) (#66324)
- **代码层面**：`vllm/models/minimax_m3/common/dspark.py` 的 `MiniMaxM3DSparkForCausalLM.__init__` 里，检测到 `decode_backend == "latency_specialized"` 且是 CUDA 时，调用 `attach_drafter` 挂载专用 drafter kernel；`nvidia/model.py` 里用 `specialized_decode_config` 调整 `fused_finalize_max_tokens`。新增 `vllm/models/minimax_m3/nvidia/specialized_decode/` 目录（大量新代码）。
- **新手概念课堂**：CuTe DSL 是 NVIDIA 的 C++ 模板库，用来手写极致优化的 GPU kernel。这个 commit 相当于给 MiniMax-M3.1 在最新 GB300 上"定制了一套赛车引擎"，decode 阶段（逐 token 生成）最高快 1.45 倍。
- **对你有什么影响**：在 GB300 上跑 MiniMax-M3.1-Flash 并开启 `decode_backend=latency_specialized`，decode 吞吐显著提升；其他硬件/模型无影响。

### 18. [DeepSeek-V3.2] Support sequence parallelism with pipeline parallelism (#65947)
- **代码层面**：`vllm/models/deepseek_v32/attention.py` 和 `nvidia/model.py` 里删掉 `pipeline_parallel_size == 1` 限制；PP 非最后 rank 的 `forward` 里，SP 模式下用 `sp_all_gather(hidden_states + residual)` 拼回完整 token 再返回，非 SP 模式保持原来的 `all_reduce`。
- **新手概念课堂**：和第 16 条是同一类修复，但针对 DeepSeek-V3.2 的稀疏注意力（DSA）架构。DSA 的 indexer 也需要正确的完整 token 才能工作，所以边界处理更复杂。
- **对你有什么影响**：DeepSeek-V3.2 可以同时开 SP + PP，长上下文推理的多机效率提升。

### 19. [KV Offload] Add mmap_path to SharedOffloadRegion (#67258)
- **代码层面**：`vllm/v1/kv_offload/cpu/shared_offload_region.py` 的 `SharedOffloadRegion` 新增 `mmap_path` 参数；如果指定，所有 segment 文件都创建在该路径下（`.1`、`.2` 后缀），否则用默认的 `/dev/shm/vllm_offload_{engine_id}.mmap`。空闲空间检查也随之改为在 `mmap_path` 所在目录进行。测试覆盖路径放置、清理、相对路径解析。
- **新手概念课堂**：`/dev/shm` 是内存文件系统，容量有限。以前 KV offload 的共享区域只能放 `/dev/shm`，现在可以指定到任意路径（比如大容量 NVMe 盘）。就像以前只能把行李放头顶行李架，现在可以托运。
- **对你有什么影响**：KV offload 的共享内存区域可以放到更大容量的磁盘上，突破 `/dev/shm` 的容量限制；`mmap_path` 是新的可选参数。

### 20. [Perf][Hy4] Enable sink-aware ShardQ prefill (#67490)
- **代码层面**：`.buildkite/test_areas/distributed.yaml` 把 ShardQ 测试扩展到 Hy4（新增 `vllm/models/hy_v4/nvidia/attention.py`、`flashmla_sparse.py` 依赖和 `tests/models/hy_v4/test_shardq_sinks.py`）；删除 `docs/models/hy4.md`（内容合并到别处）。核心是让 Hy4 的 prefill 支持"sink-aware"的 ShardQ——attention sink（注意力汇）是某些模型（如 StreamingLLM 类）的特殊 token，需要在分片 Q 计算时特殊处理。
- **新手概念课堂**：Attention sink 是序列开头几个"永远被关注"的 token，用来稳定长序列注意力。ShardQ 是把 Q 矩阵分片计算以省显存。两者结合时，sink token 不能被随便切分，否则注意力权重算错。
- **对你有什么影响**：Hy4 模型在 H100 上开 ShardQ prefill 时，长序列 prefill 更快且结果正确；普通用户感知不到，但跑 Hy4 的部署会受益。

## sglang
## 💡 今日关键词

- **关键词：sgl-router 统一入口**
  - 一句话通俗解释：SGLang 把原来叫 `sgl-model-gateway` 的模型网关改名为 `sgl-router`，并让所有用户（文档、cookbook 模板、故障排查指南）都改用这一个入口，旧的 `python -m sglang_router.launch_router` 写法被替换成 `sgl-router` 命令。
  - 为什么社区在关注：这说明 SGLang 正在收敛对外接口，减少“一个东西三个名字”的混乱。项目从“能跑”走向“好用”，是工程成熟度的信号。
  - 对新手意味着什么：如果你在学部署 SGLang，记住 `sgl-router` 这个命令就够了；看到老教程里的 `sglang_router.launch_router` 就知道它已经过时。学一个入口比学三个入口省时间。

- **关键词：AMD ROCm 性能调优**
  - 一句话通俗解释：今天有两个 commit 专门针对 AMD GPU（gfx1250 架构、aiter 后端）做 kernel 启动参数调优，让 DeepSeek-V4 的稀疏 prefill 和 GDN 线性注意力在 AMD 卡上跑得更快。
  - 为什么社区在关注：AMD MI300X/MI355X 正在成为 NVIDIA 之外的真实选择，SGLang 在认真做异构适配，而不是“能跑就行”。
  - 对新手意味着什么：如果你只学 CUDA，会错过一大块市场。ROCm/HIP 的调优思路（tile 大小、num_warps、num_stages）和 CUDA 是相通的，学一套能迁移。

- **关键词：投机解码 + 线性注意力验证**
  - 一句话通俗解释：投机解码（speculative decoding）让一个小模型先“猜”几个 token，大模型再批量验证；今天新增了用 AMD aiter 的 FlyDSL kernel 来做这个验证步骤。
  - 为什么社区在关注：投机解码是当前推理加速最有效的杠杆之一，而线性注意力模型（GDN/KDA）的验证路径和普通 Transformer 不一样，需要专门 kernel。
  - 对新手意味着什么：这是“算法 + 系统”交叉的典型场景。只懂算法不懂 kernel，或只懂 kernel 不懂算法，都做不了这类优化。

- **关键词：调度策略（LPM）的工程化**
  - 一句话通俗解释：LPM（最长前缀匹配）调度在等待队列很长时会变慢，今天加了一个“固定批次 + 有界窗口”的策略，只在队列超过 128 时按固定节奏检查前缀复用机会。
  - 为什么社区在关注：调度器是推理服务的“大脑”，一个策略改动直接影响吞吐和延迟。而且这个改动强调“确定性节奏”，是为了多 rank 之间保持一致。
  - 对新手意味着什么：调度不是简单的“先来先服务”，背后有大量启发式和权衡。想学推理系统，调度器是最值得读的模块之一。

- **关键词：多模态 / 扩散模型的尺寸对齐**
  - 一句话通俗解释：今天有三个 commit 都在处理“用户请求的图片尺寸不是 16 的倍数”的问题，自动向下对齐并打 warning，避免 DiT/VAE 报错。
  - 为什么社区在关注：扩散模型服务正在从“demo”走向“产品”，产品就要处理用户乱传参数。这类“防御性编程”是成熟服务的标志。
  - 对新手意味着什么：写模型服务时，永远不要假设用户输入合法。对齐、裁剪、warning 这些“脏活”才是真实工程。

## 🧪 重要知识点 & 动手实验

- **知识点 1：Triton kernel 的启动配置（block size / num_warps / num_stages）**
  - **是什么**：Triton 写 GPU kernel 时，你要告诉它“每个程序块处理多大的数据”（BLOCK_H/BLOCK_K）、“用多少个 warp 并行”（num_warps）、“软件流水线几级”（num_stages）。这些参数不改变计算结果，只改变调度方式，但性能可能差 2 倍。
  - **为什么重要**：Commit 2 里 gfx1250 上把 `(BLOCK_H=16, num_warps=8, num_stages=None)` 改成 `(64, 4, 2)`，实测 geomean 加速约 2.2x。不懂这些参数，你写的 kernel 可能只有别人一半速度，还找不到原因。而且这些参数是**架构相关**的，NVIDIA 上调好的配置搬到 AMD 上可能更慢。
  - **动手试试**：
    ```python
    # 在一个有 GPU 的机器上跑（需要 triton）
    import torch, triton, triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1 << 20
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    import time
    for BLOCK, warps in [(256, 4), (1024, 4), (1024, 8), (4096, 8)]:
        # warmup
        add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK, num_warps=warps)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK, num_warps=warps)
        torch.cuda.synchronize()
        print(f"BLOCK={BLOCK:5d} warps={warps} -> {(time.perf_counter()-t0)/100*1e6:.1f} us")
    ```
  - **预期结果**：你会看到不同配置下耗时明显不同（可能差 2-3 倍），而且不是“越大越好”。这让你直观理解 Commit 2 为什么值得单独发一个 PR，还配了 254 行测试来保证“所有配置结果一致”。

- **知识点 2：投机解码的“验证”步骤为什么需要专门 kernel**
  - **是什么**：投机解码里，draft 模型一次猜 k 个 token，target 模型要一次性验证这 k 个 token 是否可接受。对于线性注意力（GDN/KDA）模型，验证需要更新 recurrent state（卷积状态 + SSM 状态），这个更新是顺序依赖的，不能简单并行。
  - **为什么重要**：Commit 6 给 GDN 加了 aiter 后端做 linear-chain target verify，Commit 4 把 KDA 的 CuTe DSL MTP verify kernel 扩展到 SM90（并加了 `USE_PACKED_FMA` 开关，因为 packed f32x2 指令是 SM100+ 才有的）。不懂验证步骤的特殊性，你会以为“投机解码就是多跑几次模型”，完全错过性能关键点。
  - **动手试试**：
    ```python
    # 用 SGLang 起一个带投机解码的服务，对比开/关的吞吐
    # 先装好 sglang，然后：
    # 终端 1：起服务（以 Qwen 为例，具体模型路径按你本地改）
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-7B-Instruct \
      --speculative-algorithm EAGLE \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 4 \
      --port 30000

    # 终端 2：跑 benchmark
    python -m sglang.bench_serving \
      --backend sglang \
      --dataset-name random \
      --num-prompts 100 \
      --random-input-len 512 \
      --random-output-len 256 \
      --port 30000
    ```
  - **预期结果**：开启投机解码后 output throughput 通常明显提升（具体倍数取决于模型和 draft 质量）。如果你在 AMD 上，可以加 `--linear-attn-verify-backend aiter`（需 `SGLANG_USE_AITER=1`）对比 Commit 6 的效果。你会理解为什么“验证 kernel”是投机解码的性能瓶颈之一。

- **知识点 3：量化加载时的“稠密层 vs 量化层”判定**
  - **是什么**：一个量化 checkpoint 里，不是所有层都被量化了（比如 GDN 的某些 projection 可能保持 bf16）。加载时如果搞错了，要么把 bf16 权重当 GPTQ 解，要么把 GPTQ 权重当 bf16 读，直接报错或精度崩掉。Commit 17 通过读 checkpoint 的 tensor manifest（哪些模块有 `.qweight` 等）来动态判定，而不是靠硬编码规则。
  - **为什么重要**：量化模型的加载逻辑是“看起来简单、踩坑无数”的典型。Commit 17 的 diff 里光是 `_canonicalize_gptq_module_prefix` 就处理了 `model.language_model.`、`language_model.model.` 等多种前缀，还要“不混淆 MTP”。不懂这些，你换个模型就加载失败。
  - **动手试试**：
    ```python
    # 下载一个 GPTQ 模型，看看它的 checkpoint index 长什么样
    # 以 Qwen2.5-7B-Instruct-GPTQ-Int4 为例（需 huggingface-cli login）
    from huggingface_hub import hf_hub_download
    import json

    idx_path = hf_hub_download(
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "model.safetensors.index.json"
    )
    with open(idx_path) as f:
        index = json.load(f)

    # 统计哪些模块有 qweight / scales / qzeros
    from collections import Counter
    kinds = Counter()
    for name in index["weight_map"]:
        if name.endswith(".qweight"): kinds["qweight"] += 1
        elif name.endswith(".scales"): kinds["scales"] += 1
        elif name.endswith(".weight"): kinds["dense_weight"] += 1
    print(kinds)
    # 打印几个“稠密”模块名，观察它们和量化模块名的区别
    dense = [n for n in index["weight_map"] if n.endswith(".weight") and "qweight" not in n]
    print(dense[:10])
    ```
  - **预期结果**：你会看到 checkpoint 里同时存在 `.qweight`（量化）和 `.weight`（稠密）两类 tensor，且稠密层通常集中在 `lm_head`、`embed_tokens`、某些 norm 或 gate。这让你直观理解 Commit 17 为什么要做“动态排除”，以及为什么硬编码规则会挂。

- **知识点 4：调度器的“有界窗口”与确定性**
  - **是什么**：LPM 调度在队列很长时要排序所有请求，开销大。Commit 16 的策略是：队列 ≤128 时照旧；>128 时只看前 64 个，每 32 个“合格 pass”检查一次，且检查节奏是确定性的（不依赖本地时钟），保证多 rank 看到同样的队列时做出同样的决策。
  - **为什么重要**：分布式推理里，如果每个 rank 的调度决策不一致，会导致 batch 组成不同、通信错位、甚至死锁。所以“确定性”比“最优”更重要。Commit 16 还加了一个 gate：只有“受威胁的前缀复用 ≥ 1.5 倍 KV 容量”时才触发，避免无谓开销。
  - **动手试试**：
    ```python
    # 读 Commit 16 新增的 lpm_opportunity.py，它只有 100 行左右，非常适合精读
    # 本地 clone 后：
    # sed -n '1,101p' python/sglang/srt/managers/lpm_opportunity.py
    #
    # 然后自己跑一下它的核心函数（不依赖 GPU）：
    from types import SimpleNamespace
    from sglang.srt.managers.lpm_opportunity import competition_opportunity

    def make_req(prefix, suffix, ns="default"):
        return SimpleNamespace(
            origin_input_ids=list(prefix) + list(suffix),
            output_ids=[],
            extra_key=ns,
            cache_salt=None,
        )

    # 构造 3 个共享长前缀的请求
    shared = list(range(5000))  # 5000 tokens 共享前缀
    reqs = [make_req(shared, [i]*100) for i in range(3)]
    print(competition_opportunity(reqs, capacity=2000))   # 小容量 -> 应该 allowed
    print(competition_opportunity(reqs, capacity=100000)) # 大容量 -> 应该拒绝
    ```
  - **预期结果**：小 KV 容量时返回 `allowed=True`（因为共享前缀放不下，复用有价值），大容量时返回 `allowed=False`（放得下，不需要特殊调度）。你会理解“启发式 gate”是怎么用简单代码表达复杂权衡的。

- **知识点 5：扩散模型服务的输入防御（尺寸对齐）**
  - **是什么**：DiT 模型把 latent 切成 2x2 patch，VAE 下采样 8 倍，所以输入宽高必须是 16 的倍数。Commit 18/19 在 `_adjust_visual_fields` 里自动向下取整到 16 的倍数，并打 warning；Commit 20 则修复了 `load_image` 默认把 RGBA 转 RGB 导致 alpha 丢失的问题。
  - **为什么重要**：这是“模型能跑”和“服务能用”的分界线。用户传 513x769 的图，模型直接报错是灾难；自动对齐到 512x768 并告知用户，才是产品。Commit 20 更隐蔽：alpha 通道丢了不报错，但生成结果全错。
  - **动手试试**：
    ```python
    # 复现 Commit 20 的 bug 和修复（不需要 GPU）
    from PIL import Image
    import numpy as np
    from diffusers.image_processor import VaeImageProcessor

    # 造一张左半边透明的 RGBA 图
    pixels = np.zeros((64, 96, 4), dtype=np.uint8)
    pixels[..., :3] = (30, 140, 220)
    pixels[:, 48:, 3] = 255
    img = Image.fromarray(pixels, "RGBA")

    # 错误做法：先转 RGB 再交给 VAE
    wrong = img.convert("RGB")
    # 正确做法：保持 RGBA
    right = img

    proc = VaeImageProcessor(vae_scale_factor=16)
    out_wrong = proc.preprocess(wrong, 64, 96)
    out_right = proc.preprocess(right, 64, 96)
    print("wrong shape:", out_wrong.shape)  # [1, 3, H, W]
    print("right shape:", out_right.shape)  # [1, 4, H, W]
    # 看 alpha 通道是否还在
    print("alpha present:", out_right.shape[1] == 4)
    ```
  - **预期结果**：错误路径得到 3 通道，alpha 丢失；正确路径得到 4 通道，左半边 alpha=0。你会理解为什么 Commit 20 要给 `load_image` 传 `convert_method=lambda image: image.convert("RGBA")`——因为默认的 `load_image` 会偷偷转 RGB。

## 🔥 重要更新

- **Commit 1（sgl-router 统一入口）**：影响面最广，所有文档、cookbook 模板、故障排查指南都改了命令写法。老用户会突然发现 `python -m sglang_router.launch_router` 的教程“过时了”，新用户则少踩一个“到底该用哪个命令”的坑。
- **Commit 6（AMD GDN aiter 验证后端）**：给 AMD 上的投机解码加了一条新路径，且明确限制了“不能和 ReplaySSM 同时用”。对 AMD 用户是实打实的性能更新，也说明 SGLang 的 AMD 支持在从“能跑”走向“调优”。
- **Commit 16（LPM 有界窗口策略）**：调度器核心逻辑改动，直接影响长队列场景的吞吐。而且它强调“确定性节奏”，是分布式调度设计的教科书式案例，值得精读。
- **Commit 17（GPTQ 稠密/量化动态判定）**：修的是量化模型加载的“隐形炸弹”，diff 里大量前缀归一化和 manifest 逻辑说明这个坑很深。对做量化部署的人影响很大。

## 📋 逐条解读

### 1. [Refactor] gateway: Keep sgl-model-gateway as source only and move every user to sgl-router (#50671)

- **代码层面**：改的是文档和模板，不是运行时代码。`.agents/skills/cookbook-add-model/references/authoring-reference.md` 里把“SGLang Model Gateway”改成“sgl-router”；`config.jsx.tmpl` 里把 `python3 -m sglang_router.launch_router --pd-disaggregation --prefill ... --decode ...` 换成 `sgl-router --model-id {{MODEL_NAME}} --no-tokenizer --worker-urls http://... http://...`；`replay-trace-profile.md` 里删掉了 `--enable-trace --otlp-traces-endpoint`（因为 sgl-router 不再自己导出 span，只转发 `traceparent`）；CODEOWNERS 里删掉了 `grpc_server.py` 和 `grpc` 目录的 owner。
- **新手概念课堂**：把 `sgl-model-gateway` 想成“公司前台”，以前前台有三个不同的工牌（`sgl-model-gateway`、`sglang_router`、`sgl-router`），现在统一成一张工牌 `sgl-router`。前台本身还是那个前台，只是名字和用法统一了。`--no-tokenizer` 的意思是“router 不需要自己加载 tokenizer，省内存”，`--worker-urls` 是“告诉 router 后端 worker 在哪”。
- **对你有什么影响**：如果你照着老教程敲 `python -m sglang_router.launch_router`，可能还能跑但已经不被推荐；新教程一律用 `sgl-router`。PD 分离部署时，`--model-id` 必须和 worker 的 `--served-model-name` 一致，否则 router 找不到模型——这是模板里新加的注释强调的点。

### 2. [Perf] [AMD] dsv4: Retune sparse paged-prefill launch config for gfx1250 (#41382)

- **代码层面**：改 `python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py` 里的 `_sparse_attn_v4_paged_prefill_triton`。新增模块级 `_is_gfx1250 = is_gfx1250_supported()`，在 kernel 启动前根据架构选参数：gfx1250 用 `block_h=64, block_k=16, num_warps=4, num_stages=2`，其他架构保持 `block_h=16, block_k=16或32, num_warps=8, num_stages=None`。同时新增 254 行测试 `test_dsv4_paged_prefill_launch_config.py`，验证“所有配置结果数值一致”。
- **新手概念课堂**：Triton kernel 的启动参数就像“餐厅翻台策略”。`BLOCK_H=64` 意思是“一个服务员一次服务 64 桌”（等于 `index_n_heads`，一个 CTA 覆盖一个 token 的所有 head），`num_stages=2` 是“提前把下一批菜备好”（软件流水线）。gfx1250 上原来的配置是“一个服务员只服务 16 桌、8 个人一起干、不提前备菜”，对这个 gather-bound（访存密集）的 kernel 来说太浪费。调优后实测 T=128/512/2048 分别加速 2.8x/2.1x/2.3x。
- **对你有什么影响**：如果你在 gfx1250（AMD 新卡）上跑 DeepSeek-V4 的 prefill，会明显变快。其他卡不受影响。测试的存在意味着以后有人改这些常数，CI 会拦住“数值变了”的情况——这是好的工程实践。

### 3. [Fix] [NPU] CI: Update NPU failure analyzer coverage (#44736)

- **代码层面**：改 CI workflow。`_npu-pr-test-stage.yml` 和 `_npu-single-node-test-stage.yml` 各加了一个 step：`Upload test logs for failure analysis`，用 `actions/upload-artifact@v4` 把 `/tmp/test-logs/*.log` 上传，保留 7 天。新增 `ci-failure-monitor-diffusion-npu.yml`（117 行），是一个 reusable workflow，把 CI 失败和“覆盖率推荐”交叉分析。
- **新手概念课堂**：CI 失败分析就像“考试后老师分析错题”。以前 NPU 测试挂了，日志散落在各个 job 里，要人工翻；现在统一上传 artifact，再由一个专门的 workflow 自动交叉分析。`if: always()` 意思是“不管测试成功失败都上传”，`if-no-files-found: ignore` 是“没日志也别报错”。
- **对你有什么影响**：普通用户无感。但如果你给 SGLang 提 PR 且改了 NPU 相关代码，失败时维护者能更快定位问题，你的 PR 可能更快被 review。

### 4. [Perf] [NVIDIA] KDA: Extend the CuTe DSL MTP verify kernel to SM90 (#44612)

- **代码层面**：改 `python/sglang/kernels/ops/attention/kda_decode_mtp.py`。给 `kda_decode_mtp_kernel` 加了一个 `USE_PACKED_FMA: cutlass.Constexpr[bool]` 参数。在 state 更新和输出累加的两处循环里，原来无条件用 `cute.arch.mul_packed_f32x2` / `fma_packed_f32x2`（packed f32x2 指令），现在用 `if cutlass.const_expr(USE_PACKED_FMA)` 分支：SM100+ 走 packed，SM90 走标量逐元素乘加。
- **新手概念课堂**：`packed f32x2` 是 GPU 的一条指令同时算两个 float32，像“一双筷子夹两块肉”，SM100+ 支持；SM90 只有“一根筷子”，硬用 packed 指令会编译失败或走模拟。`cutlass.const_expr` 是编译期分支，不满足条件的分支根本不会生成代码，所以不会拖慢 SM100。
- **对你有什么影响**：如果你在 H100（SM90）上跑 KDA 模型的 MTP（multi-token prediction）验证，以前这个 kernel 可能用不了或走 fallback，现在能用了。SM100（Blackwell）用户不受影响，仍然走 packed 快路径。

### 5. [Feat] [NVIDIA] deepseek-v4.1: Support prefill CP with BCG in PD serving (2/N) (#40333)

- **代码层面**：改 `deepseek_v4_backend.py` 和 `cp/bcg/__init__.py`。在 `refresh_for_breakable_cuda_graph_replay_` 里，原来对每个字段直接 `dst_val.copy_(src_val)`，现在先检查 shape 是否一致，不一致时只允许 `_CP_GLOBAL_FIELDS` 里的字段（CP 全局字段），并用新函数 `_copy_tensor_allowing_storage_alias` 处理“storage 别名”和“短前缀 + pad”的情况。`bcg/__init__.py` 里给 `(INTERLEAVE, DSV4)` 组合开了 BCG 支持，`get_cp_bcg_input_type` 加了 `attention_backend` 参数。
- **新手概念课堂**：BCG（Breakable CUDA Graph）是“可打断的 CUDA Graph”。CUDA Graph 把一串 kernel 录下来一次重放，快但“不能中途改参数”；BCG 允许在图的某些点“打断”，插入动态逻辑。PD 分离 + CP（context parallel）场景下，prefill 的序列长度每次不同，需要 BCG 来兼顾图和动态性。`storage alias` 是指两个 tensor 共享同一块显存，直接 copy 会“自己拷自己”出问题，所以要 `clone()`。
- **对你有什么影响**：如果你用 DeepSeek-V4.1 做 PD 分离 + prefill context parallel 部署，现在支持 BCG 了，prefill 阶段能享受 CUDA Graph 的加速。普通单机用户无感。

### 6. [Perf] [AMD] gdn: Add an aiter backend for linear-chain target verify (#48118)

- **代码层面**：改三处。`attention_hook.py` 里加校验：`--enable-linear-replayssm-spec` 不能和 `--linear-attn-verify-backend aiter` 同时用（因为 ReplaySSM 有自己的 kernel）。`exec_.py` 里给 `linear_attn_verify_backend` 的 choices 加 `"aiter"`。`gdn_backend.py` 里加 `verify_kernel_is_aiter` 标志和 `verify_conv_update` 属性，新增分支：当 `verify_backend.is_aiter()` 时，要求 `is_hip() and SGLANG_USE_AITER`，否则报错；然后 import `causal_conv1d_update_sglang_flydsl` 和 `AiterGDNKernel`，把 `forward_extend` 里的 `causal_conv1d_update` 改成 `self.kernel_dispatcher.verify_conv_update`。新增 `kernels/gdn_aiter.py`。
- **新手概念课堂**：GDN（Gated Delta Network）是线性注意力的一种。投机解码的“target verify”要一次性验证多个 draft token，需要更新卷积状态和 SSM 状态。aiter 是 AMD 的 kernel 库，FlyDSL 是它的一种 DSL。`kernel_dispatcher` 是“调度员”，根据配置决定用哪个 kernel 实现，`verify_conv_update` 就是“验证时用哪个卷积更新函数”。
- **对你有什么影响**：AMD 用户可以用 `--linear-attn-verify-backend aiter` 让 GDN 模型的投机验证走 aiter 的 FlyDSL kernel，可能更快。但如果你同时开了 `--enable-linear-replayssm-spec`，会直接报错——这是有意的，避免两个 kernel 打架。

### 7. [Doc] [AMD] cookbook: Add the Qwen3.5 MI300X FP8 agentic recipe (#50402)

- **代码层面**：改 `docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx`。新增 3.3.2 节“Agentic Long-Context on MI300X (FP8, MTP)”，给出完整启动命令：`SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 SGLANG_ENABLE_SPEC_V2=1 sglang serve --model-path Qwen/Qwen3.5-397B-A17B-FP8 --tp 8 --attention-backend aiter --page-size 16 ... --speculative-algorithm EAGLE ...`。还更新了 FP8 KV Cache 的 Warning，说明 3.3.2 也用了 `--kv-cache-dtype fp8_e4m3`。
- **新手概念课堂**：Cookbook 就是“菜谱”，告诉你“什么硬件 + 什么模型 + 什么参数”能跑出好效果。`SGLANG_USE_AITER_UNIFIED_ATTN=1` 让 MTP target verify 走 AITER 的 `unified_attention`，需要 `--page-size 16`；不开的话走 Triton extend kernel，没有 split-KV，在长上下文 agentic 场景下 decode 会很慢。`--max-running-requests` 是并发数的 2 倍，因为“一个 session 树可能有多个 request 在飞”。
- **对你有什么影响**：如果你有 8 张 MI300X 想跑 Qwen3.5-397B-FP8 做 agentic 场景，直接抄这个命令。注意 `--kv-cache-dtype fp8_e4m3` 有精度风险（文档明确警告），只在内存不够时用。

### 8. [Feat] hisparse: Add the HiCache backend data plane (#49473)

- **代码层面**：改 `dsa_indexer_metadata.py`、`dsa_backend.py`、`hisparse_memory_pool.py`。`DSAIndexerMetadata` 和 `DSAMetadata` 各加一个 `indexer_page_table: Optional[torch.Tensor]` 字段，`get_page_table_64` 优先返回它。`hisparse_memory_pool.py` 新增 `HiSparseHiCacheBuffer` 类（约 160 行），用 `load_cache_to_device_buffer_mla` 做 host→device 的 cache 搬运。
- **新手概念课堂**：HiSparse 是“分层稀疏注意力”，HiCache 是“分层 KV cache”（GPU 显存 + CPU 内存 + 可能还有 SSD）。`indexer_page_table` 是“索引页表”，记录每个 token 的 KV 存在哪一页。为什么要单独一个 `indexer_page_table`？因为 HiSparse 的 index 页可能比原始 attention KV 页“活得更久”，不能共用一张表，否则会读到已释放的页。
- **对你有什么影响**：如果你用 DeepSeek 系列 + HiSparse + HiCache 做超长上下文推理，这个改动让 cache 分层搬运更正确。普通用户无感，但长上下文场景下能避免“cache 错乱导致的输出异常”。

### 9. [Feat] [NPU] DSpark: Support multi-stream execution in draft layers (#49294)

- **代码层面**：改 `deepseek_v4_dspark.py`。把 `torch.cuda.current_stream()` 换成 `torch.get_device_module().current_stream()`，`torch.cuda.stream(stream_kv)` 换成 `torch.get_device_module().stream(stream_kv)`，`torch.cuda.Stream()` 换成 `torch.get_device_module().Stream()`。multi-stream 的开关条件从 `envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()` 改成 `(... or envs.SGLANG_NPU_USE_MULTI_STREAM.get())`，可用性判断从 `torch.cuda.is_available()` 改成 `(_is_npu or torch.cuda.is_available())`。
- **新手概念课堂**：`torch.cuda.*` 是 NVIDIA 专用 API，NPU（昇腾）上要用 `torch_npu` 提供的 `torch.get_device_module()` 来抽象。多流（multi-stream）就是“GPU 上开多条流水线”，一条算 KV projection，一条算别的，互相 overlap。DSpark 是 DeepSeek 的 draft 模型，draft 层里 KV 计算和主计算可以并行。
- **对你有什么影响**：昇腾 NPU 用户跑 DSpark 投机解码时，draft 层可以用多流加速了。NVIDIA 用户行为不变（`get_device_module()` 在 CUDA 上就是 `torch.cuda`）。这是一个“用抽象 API 换跨平台兼容”的典型重构。

### 10. [Test] [NPU] llada2-mini: Run FDFO performance case with 32 prompts at 6K/3K (#50105)

- **代码层面**：改 `test_npu_llada2_mini_fdfo.py`。benchmark 参数从 `num_prompts=128, random_input_len=3584, random_output_len=1024, max_concurrency=16` 改成 `num_prompts=32, random_input_len=6144, random_output_len=3072, max_concurrency=8`。删掉了 `p99_ttfts` 的收集和断言。
- **新手概念课堂**：FDFO（First-Done-First-Out）是扩散语言模型（dLLM）的调度策略，优先处理“扩散步数先完成”的请求。测试参数从“多而短”改成“少而长”，是为了更贴近 agentic 长上下文场景。删掉 p99 断言是因为“32 个请求的 p99 统计意义不大”，容易 flaky。
- **对你有什么影响**：普通用户无感。但如果你关注 dLLM 调度，这个测试参数变化说明团队在往“长上下文”场景倾斜。

### 11. [Perf] [NVIDIA] mega_moe: Allow DeepGEMM weight localization on SM 10.0 and 10.3 (#50608)

- **代码层面**：改 `mega_moe.py` 的 `_is_mega_moe_localization_enabled`。原来只允许 `_device_sm == 107`，现在允许 `_device_sm in [100, 103, 107]`。
- **新手概念课堂**：`_device_sm` 是 GPU 的 compute capability（SM 版本）。SM 10.0 是 B200，10.3 是 B300（或类似），10.7 是另一款 Blackwell。DeepGEMM 是 DeepSeek 开源的 GEMM 库，“weight localization”是把权重放到更近的存储层级（比如 SMEM）以减少访存。原来只在 10.7 上开，现在 10.0/10.3 也开。
- **对你有什么影响**：如果你在 B200/B300 上跑 MegaMoE（DeepSeek 的 MoE 后端），现在能享受 weight localization 的性能收益了。前提是 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE=1`。

### 12. [Fix] disaggregation: Release multimodal features when PD prefill finishes a request (#44576)

- **代码层面**：改 `disaggregation/prefill.py` 和 `scheduler.py`。`scheduler.py` 里把 `_maybe_clear_mm_inputs` 的逻辑抽成独立的 `_release_mm_inputs(req)` 方法（处理 session / abort / 非 session 三种情况），`_maybe_clear_mm_inputs` 只负责遍历 batch 调它。`prefill.py` 的 `process_disagg_prefill_transfer_queue` 里，在 `maybe_release_metadata_buffer` 之后加了一行 `self._release_mm_inputs(req)`。新增测试 `test_prefill_complete_integration.py` 里的 `test_...`（50 行）。
- **新手概念课堂**：PD 分离（Prefill-Decode disaggregation）里，prefill 节点算完 KV 后要把 KV 传给 decode 节点。多模态请求的 image/video features 占大量内存，如果 prefill 完成后不释放，会一直占着。以前只在 scheduler 的 batch 清理时释放，但 PD 分离的 prefill 完成路径不经过那里，所以泄漏。现在显式在 prefill 完成时释放。
- **对你有什么影响**：如果你用 PD 分离 + 多模态模型（图片/视频输入）跑长服务，以前可能内存缓慢泄漏，现在修了。session 请求（多轮对话）的 mm_inputs 会保留，abort 时会释放，逻辑正确。

### 13. [Doc] diffusion: Clarify H100 baseline calibration limits (#49998)

- **代码层面

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)