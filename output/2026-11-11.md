# 🚀 vLLM & SGLang 每日更新 — 2026-11-11

> 自动生成于 2026-11-11 08:21 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：投机解码（Speculative Decoding）与自适应验证**
  - 一句话解释：让一个小模型"抢跑"猜出若干 token，大模型只做一次并行验证，猜对就白赚速度。
  - 为什么社区在关注：今天有 3 个 commit（#62769、#66764、#66433）都在打磨 DSpark / adaptive verification，说明"推理加速"已从"能不能跑"进入"多卡下如何保持一致、如何融合 kernel"的深水区。
  - 对新手意味着什么：投机解码是当前 LLM 推理最热的方向之一，懂"草稿-验证-接受率"这套逻辑，就能读懂大量推理框架的核心代码。

- **关键词：AMD ROCm 生态适配**
  - 一句话解释：让 vLLM 在 AMD 的 GPU（MI300/MI355 等）上也能跑得又快又对。
  - 为什么社区在关注：今天至少 5 个 commit 带 `[ROCm]`/`[XPU]` 前缀（#64152、#66876、#63838、#66758、#66924），说明异构计算已不是"顺便支持"，而是主力战场。
  - 对新手意味着什么：只学 CUDA 已经不够，理解"平台抽象层（platform layer）+ 后端分派"是跨硬件部署的关键技能。

- **关键词：量化格式多样化（MXFP4 / FP8 / LUT）**
  - 一句话解释：用 4bit、8bit 甚至查表方式压缩模型权重，省显存、提吞吐。
  - 为什么社区在关注：#61530 支持 MXFP4 打包的 Engram 表，#66876 放开 lut_b_moe 在线量化，量化格式正在"百花齐放"。
  - 对新手意味着什么：量化不再是"INT8 一刀切"，学会读 `quantization_config` 和反量化数学，是模型部署的硬技能。

- **关键词：前端协议细节的正确性（role / echo / 日志 / 错误码）**
  - 一句话解释：API 返回的 role、echo 内容、错误是 400 还是 501，这些"小事"决定用户体验。
  - 为什么社区在关注：#63357、#66848、#65029、#66853 全是前端行为修正，说明 vLLM 正从"能推理"走向"生产级 OpenAI 兼容"。
  - 对新手意味着什么：写服务端不只是算得快，协议细节（HTTP 状态码、流式 chunk 结构）才是工程功力的体现。

- **关键词：流水线并行（PP）与数值精度**
  - 一句话解释：把模型按层切到多张卡上，卡与卡之间传张量；传的时候 dtype 不对就会悄悄丢精度。
  - 为什么社区在关注：#65755 修复 FP32 残差在 PP 边界被截断，说明大规模多卡训练的"数值正确性"开始被认真对待。
  - 对新手意味着什么：分布式不只是"能通信"，dtype 转换、buffer 复用这些细节才是 bug 温床。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：流式解析中的"前缀缓冲"（Commit 1）**
  - **是什么**：流式输出时，解析器看到一个 `<` 不能立刻判断它是普通文本还是 `<function_calls>` 标签的开头，必须"先扣住"这段文本，等后续字符确认。
  - **为什么重要**：如果不缓冲，用户会看到 `<p>hello` 被误判成工具调用，或者标签被拆成两半吐出去。所有流式 JSON/工具解析器都有这个坑。
  - **动手试试**：
    ```python
    def stream_parse(chunks, start_tag="<function_calls>"):
        buf = ""
        for ch in chunks:
            buf += ch
            if start_tag.startswith(buf):
                continue          # 还可能是标签前缀，扣住
            if buf.startswith(start_tag):
                yield ("TOOL", buf)  # 确认是工具调用
            else:
                yield ("TEXT", buf)  # 确认是普通文本，一次性放出
            buf = ""
        if buf:
            yield ("TEXT", buf)      # 流结束，剩余文本放出

    print(list(stream_parse(list("<p>hello"))))
    print(list(stream_parse(list("<function_calls>x"))))
    ```
  - **预期结果**：第一行输出 `[('TEXT', '<p>hello')]`，第二行输出 `[('TOOL', '<function_calls>x')]`。你能解释"为什么 `<` 要单独扣住"就掌握了。

- **知识点 2：torch.compile 的字节码钩子与"禁止改 buffer"（Commit 2）**
  - **是什么**：CUDA Graph 会把一整段前向计算录制成一张"图"反复回放；如果图里偷偷改了模型 buffer 的值，回放结果就错了，所以 vLLM 用字节码钩子静态检查 `self._buffers.update(...)`。
  - **为什么重要**：这是"性能优化 vs 正确性"的经典冲突。很多新手写自定义层时随手 `self.register_buffer` 后又在 forward 里改它，一开 CUDA Graph 就静默出错。
  - **动手试试**：
    ```python
    import torch, torch.nn as nn

    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.ones(2))
        def forward(self, x):
            self.b = self.b * 2      # 在 forward 里改 buffer！
            return x

    m = Bad()
    try:
        torch.compile(m, backend="eager", fullgraph=True)(torch.ones(2))
        print("没报错，但 CUDA Graph 下结果会错")
    except Exception as e:
        print("编译期就发现了:", type(e).__name__)
    ```
  - **预期结果**：你会看到编译/图捕获阶段就报错或警告。记住结论：**forward 里不要改 buffer，改用局部变量或 `self.b.copy_()` 之外的显式状态**。

- **知识点 3：FP32 残差在 PP 边界的 dtype 收窄（Commit 5）**
  - **是什么**：模型里 residual stream 常是 FP32，但跨卡 buffer 可能是 BF16；`copy_` 到 BF16 buffer 会把 FP32 尾数截掉，且不报错。
  - **为什么重要**：单卡测不出，多卡 PP 才暴露，属于"最难查的一类 bug"。今天新增的 `check_received_tensors` 就是专门防这个。
  - **动手试试**：
    ```python
    import torch
    a = torch.tensor([1.0000001], dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.bfloat16)
    b.copy_(a)
    print("FP32:", a.item(), "-> BF16:", b.item(), "相等?", a.item() == b.item())
    # 判断是否"收窄"：用 torch.finfo 比较尾数位
    print("fp32 mantissa bits:", torch.finfo(torch.float32).nmant)
    print("bf16 mantissa bits:", torch.finfo(torch.bfloat16).nmant)
    ```
  - **预期结果**：`相等? False`，且 bf16 尾数位（7）远少于 fp32（23）。你就明白了为什么"同宽不同精度"也算 lossy。

- **知识点 4：MXFP4 打包与反量化（Commit 6）**
  - **是什么**：MXFP4 把两个 4bit 浮点数（E2M1）塞进一个字节，低半字节是偶数列；配合每 32 个元素一个 ue8m0 指数缩放。解包时按 4bit 查表还原。
  - **为什么重要**：这是当前最省显存的格式之一（DeepSeek-V4.1 的表从 200GB 降到 ~104GB）。不懂打包顺序，读 checkpoint 会全错。
  - **动手试试**：
    ```python
    import torch
    # 模拟一个字节 = 两个 nibble，低 nibble 是偶数列
    packed = torch.tensor([0x21], dtype=torch.uint8)  # 高=2, 低=1
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten().long()
    # E2M1 查找表（正数部分）
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = torch.where(nibbles & 8 != 0, -1.0, 1.0)
    values = sign * lut[nibbles & 7]
    print("nibbles:", nibbles.tolist(), "-> values:", values.tolist())
    ```
  - **预期结果**：输出 `nibbles: [1, 2] -> values: [0.5, 1.0]`。这就是 Commit 6 测试里 `_reference` 函数的核心逻辑。

- **知识点 5：HTTP 错误码语义 —— 400 vs 501（Commit 8）**
  - **是什么**：客户端传了非法 role（如大写 `SYSTEM`）是"你请求错了"（400），不是"服务器没实现"（501）。vLLM 把 `NotImplementedError` 改成 `ValueError` 就是为了让映射正确。
  - **为什么重要**：错误码错了，客户端 SDK 的重试逻辑、监控告警全都会误判。这是 API 设计的常识题，但真实项目里天天踩。
  - **动手试试**：
    ```python
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.exception_handler(ValueError)
    async def ve(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(NotImplementedError)
    async def nie(request: Request, exc: NotImplementedError):
        return JSONResponse(status_code=501, content={"error": str(exc)})

    @app.get("/bad-role")
    def bad_role():
        raise ValueError("Invalid role: SYSTEM")

    print(TestClient(app).get("/bad-role").status_code)  # 400
    ```
  - **预期结果**：打印 `400`。把 `raise ValueError` 改成 `raise NotImplementedError` 会变成 `501`，你就理解了 Commit 8 的动机。

---

## 🔥 重要更新

1. **#61530 DSV4.1 支持 MXFP4 打包 Engram 表**：把 200GB 的巨型嵌入表压到 ~104GB，且兼容 CPU offload / DP 分片，直接决定这个模型能不能在单机多卡上跑起来。
2. **#62769 跨 TP rank 同步自适应 DSpark 决策**：多卡投机解码时各 rank 必须对"本步接受几个 token"达成一致，否则 batch 布局错位、结果全错——这是分布式投机解码的正确性基石。
3. **#65755 修复 PP 边界 FP32 残差截断**：新增 `check_received_tensors` 主动检测 dtype 收窄、缺张量、形状广播，把"静默精度损失"变成显式告警。
4. **#61501 Mamba 可选 FlashInfer SSD prefill scan**：新增 `--mamba-prefill-backend`，在 SM100/SM103 上启用融合扫描，Mamba 类模型的 prefill 性能再上一档。

---

## 📋 逐条解读

### 1. [Bugfix][Tool Parser] Olmo3: stream plain text that starts with "<"
- **代码层面**：改 `vllm/tool_parsers/olmo3_tool_parser.py` 的 `extract_tool_calls_streaming`。原来只要当前文本不以 `<` 开头就直接放行，现在先判断 `"<function_calls>".startswith(current_text)`（可能是标签前缀就扣住），确认不是标签后，若之前扣过文本则一次性补发 `current_text`。测试扩展到 `<p>hello`、`<b>bold</b>` 等用例。
- **新手概念课堂**：像快递分拣——看到包裹上第一个字是"退"字，不能立刻扔进退货区，得等看到"退货"还是"退款"再决定。
- **对你有什么影响**：用 Olmo3 做流式对话时，普通 HTML 文本不会再被吞掉或被误判成工具调用。

### 2. [torch.compile] Only flag `_buffers.update` in the cudagraph buffer assignment check
- **代码层面**：`vllm/compilation/wrapper.py` 的 `bytecode_hook` 引入 `dis` 模块反汇编字节码，只精确拦截 `_buffers.update(...)` 这类"改 buffer"的调用，不再误伤普通的 `dict.update`（如 `kwargs["scale"]=2`）。新增 `test_cudagraph_buffer_assignment_check` 参数化测试。
- **新手概念课堂**：CUDA Graph 像"录播"——把一段操作录下来反复播。如果你在"录"的时候偷偷改了道具（buffer），"播"的时候道具还是旧的，结果就对不上。
- **对你有什么影响**：自定义层里用 `kwargs.update(...)` 不再被误报错，但 `self._buffers.update` 仍会被拦——写自定义模型时注意别在 forward 里改 buffer。

### 3. [Frontend] Log outputs for batched chat completions
- **代码层面**：`OpenAIServingChatBatch` 支持 `enable_log_outputs`，批量聊天时为每个子请求（`{response.id}_0`、`_1`…）调用 `request_logger.log_inputs/log_outputs`。测试验证子请求 ID 拼接正确。
- **新手概念课堂**：批量请求像"团购"，以前只记一张总账单，现在给每个团员单独开发票，方便审计。
- **对你有什么影响**：开启 `--enable-log-outputs` 后，批量 chat 的每条输出都能被日志系统追踪，便于合规审计和调试。

### 4. [XPU][CI] Skip LMCacheMPConnector tests when lmcache is not installed
- **代码层面**：`tests/v1/engine/test_engine_args.py` 新增 `requires_lmcache = pytest.mark.skipif(importlib.util.find_spec("lmcache") is None, ...)`，给相关参数化用例和 `test_extensible_kv_cache_off_with_lmcache_mp` 打上标记。
- **新手概念课堂**：像考试前检查"这道题需要的计算器你带了吗"，没带就跳过这题而不是判零分。
- **对你有什么影响**：在没装 lmcache 的平台（如 Intel XPU）跑 CI 不再因 import 失败而红，本地开发也更友好。

### 5. [Bugfix][Step5] Fix FP32 residual truncation at pipeline boundaries
- **代码层面**：`vllm/models/step5/nvidia/model.py` 移除 `make_empty_intermediate_tensors_factory` 的误用；`vllm/v1/worker/gpu/pp_utils.py` 新增 `check_received_tensors`，检测 dtype 收窄、缺张量、per-token 形状广播，并给出 WARNING/ValueError。测试覆盖 6 种 dtype 组合。
- **新手概念课堂**：像接力赛传棒，上一棒手里是"高精度秒表"，下一棒只有"秒表"（BF16），交接时精度就丢了——所以要检查"下一棒的表够不够准"。
- **对你有什么影响**：跑 Step5 多卡 PP 时，如果某个 stage 的 buffer dtype 比收到的窄，会看到明确警告而不是悄悄算错。

### 6. [Model][DSV4.1] Support MXFP4-packed Engram embedding tables
- **代码层面**：`vllm/models/.../engram` 相关代码支持 `quantization_config.engram_embed_quant == "mxfp4"`，读取 packed uint8 权重 + ue8m0 block-32 scale，反量化时按 nibble 拆分（偶数列在低 nibble）。文档 `docs/features/engram.md` 补充说明；DP 分片测试新增 packed 路径。
- **新手概念课堂**：像把两件衣服叠进一个压缩袋（一个字节装两个 4bit 数），取出来时要按"先左后右"的顺序拆。
- **对你有什么影响**：`nvidia/DeepSeek-V4.1-Flash-NVFP4` 这类 checkpoint 现在可以直接加载，显存占用从 ~200GB 降到 ~104GB，单机可行性大幅提升。

### 7. [Bugfix][Frontend] Honor --response-role and echo in chat derender
- **代码层面**：`message_assembly.py` 新增 `get_echo_content`，兼容 Harmony `Message` 对象（gpt-oss 用）；`OnlineDerenderer` 支持 `--response-role` 和 `echo`，流式时把 echo 内容拼到首个 chunk 的 `content` 开头。文档同步更新。
- **新手概念课堂**：像"回音壁"——你喊一句，它把你说的话原样贴回来再回答。以前 derender 忘了这个开关，导致 role 和 echo 行为与 `/v1/chat/completions` 不一致。
- **对你有什么影响**：用 `/derender` 做离线解析时，输出和在线 chat 完全对齐，`echo: true` 也能正常工作。

### 8. [Bugfix][Tokenizers] Raise ValueError on unknown message role in deepseek_v41_encoding
- **代码层面**：`deepseek_v41_encoding.py` 的 `render_message` 把 `raise NotImplementedError(f"Unknown role: {role}")` 改成 `raise ValueError(f"Invalid role: {role}")`。测试扩展到 v41 模块。
- **新手概念课堂**：HTTP 400 = "你请求错了"，501 = "我还没实现"。传错 role 是客户端问题，应该回 400 让用户改，而不是 501 让人以为服务器坏了。
- **对你有什么影响**：给 DeepSeek-V4.1 发非法 role 时，现在收到 400 而不是 501，SDK 重试逻辑不会误触发。

### 9. [Bugfix][Frontend] Report the line of a run-batch request rejected by request validators
- **代码层面**：`run_batch.py` 的 `validate_batch` 捕获异常从 `pydantic.ValidationError` 扩展到 `(pydantic.ValidationError, VLLMValidationError)`，保证 request 模型自身 validator 抛的错也带 `line N:` 前缀。
- **新手概念课堂**：像批改作业，以前只在"格式错"时标行号，现在"内容不合规"也标行号，方便定位。
- **对你有什么影响**：用 `vllm run-batch` 跑大批量请求时，报错信息会精确指出第几行，不用再肉眼数 JSONL。

### 10. [XPU] fix test_dots_ocr.py::test_dots_ocr_vision_without_host_sync
- **代码层面**：测试从硬编码 `.cuda()` 改为 `current_platform.device_type`，`torch.cuda.set_sync_debug_mode` 改为按平台分派（仅 CUDA-like 才启用），CPU 上直接 skip。
- **新手概念课堂**：像写"通用遥控器"，以前只认"索尼电视"（CUDA），现在按平台选"红外/蓝牙"（device_module）。
- **对你有什么影响**：DotsOCR 的视觉测试在 XPU/CPU 上也能跑（或正确 skip），不再因 `torch.cuda` 不存在而崩。

### 11. [KV Offload] Warn when secondary-tier KV is tied to the parallel config
- **代码层面**：`vllm/v1/kv_offload/tiering/spec.py` 在构造 tiering spec 时，若存在 secondary tiers 且布局与并行配置绑定（非 canonical、非 replicated MLA），打 WARNING 提示"换 TP 大小后缓存不可复用"。
- **新手概念课堂**：像把家具按"当前户型"定制，搬家（换 TP）就装不进去。警告你"这是定制款，别指望通用"。
- **对你有什么影响**：配二级 KV offload 时，如果布局和 TP/DP 绑定，日志会提前警告，避免换配置后缓存失效却无感知。

### 12. [Perf][Muse Spark] Fuse DSpark draft norms and Markov sampling on NVIDIA
- **代码层面**：新增 `vllm/models/muse_spark/nvidia/specialized_decode/runtime/markov_chain.py`，把 draft 的 norm 和 Markov 采样融合成 Triton kernel；`DSparkSpeculator` 在 SM100 上走融合路径。新增 `test_muse_spark_markov.py` 验证"每步 confidence 读的是该步实际采样的 token 的 embedding"。
- **新手概念课堂**：像把"洗菜→切菜→下锅"三个动作合成一个流水线，减少来回搬运（kernel launch 和显存读写）。
- **对你有什么影响**：Muse Spark 在 NVIDIA SM100 上投机解码吞吐提升，且自适应验证路径数学正确性有测试兜底。

### 13. [Perf][Mamba] Opt-in FlashInfer SSD prefill scan; per-step ReplaySSM seed
- **代码层面**：`MambaConfig` 新增 `prefill_backend`（triton/flashinfer），`--mamba-prefill-backend` CLI；FlashInfer 仅在 SM100/SM103 允许，否则报错。`ReplaySSM` 增加 per-step 随机种子 `_replayssm_rand_seed(s)`。
- **新手概念课堂**：SSD scan 像"沿时间轴滚动累加"，FlashInfer 把它做成融合 kernel；per-step seed 像"每步重新洗牌"，保证可复现。
- **对你有什么影响**：Mamba 模型 prefill 可手动开 FlashInfer 加速；MTP 场景下 ReplaySSM 的随机性按步隔离，调试更可复现。

### 14. [Multimodal] Add keyframe-aware sampling to PyNvVideoCodec backend
- **代码层面**：`vllm/multimodal/media/video.py` 新增 `_sample_nearest_keyframes`，`keyframe_sampling="nearest"` 把均匀采样位置替换成最近的关键帧（去重、保序，关键帧不足则回退均匀）。`resolve_video_backend_kwargs` 拒绝非 pynvvideocodec 后端使用该参数。
- **新手概念课堂**：视频关键帧像"书的章节标题"，均匀采样可能落在两章之间的任意页，关键帧采样则对齐到章节起点，信息更完整。
- **对你有什么影响**：用 `--media-io-kwargs '{"video": {"backend": "pynvvideocodec", "keyframe_sampling": "nearest"}}'` 可提升视频理解质量，尤其是镜头切换多的素材。

### 15. [Recipe] Kimi-K3.1-Pro: add B300 AgentX presets, align with InferenceX#4871
- **代码层面**：`recipes/models/moonshotai/Kimi-K3.1-Pro.yaml` 新增 `b300_agentx` 预设（TP8+DCP8、FP8 KV、MXFP4 indexer、host KV offload、low-latency decode），环境变量对齐 InferenceX 基准。
- **新手概念课堂**：像"官方推荐配置单"，告诉你在 B300 上跑 Kimi-K3.1-Pro 该开哪些开关、设什么环境变量。
- **对你有什么影响**：想复现 InferenceX 的 AgentX 基准结果，直接抄这个 YAML 即可，不用自己摸索参数组合。

### 16. [ROCm][Perf][DSv4.1] Mono: tuned K1 at TP8
- **代码层面**：`vllm/models/deepseek_v41/amd/mono/layer.py` 的 `tuned_front` 条件从 `self.tp == 4` 放宽到 `self.tp in (4, 8)`，让 TP8 下每层 K1 也走 tuned 编译路径。
- **新手概念课堂**：像给"4 车道"专门铺的柏油路，现在"8 车道"也能用，不用再走普通水泥路。
- **对你有什么影响**：在 ROCm gfx950 上用 TP8 跑 DeepSeek-V4.1 Mono，K1 层性能提升，且与 generic 路径 bit-exact 对拍。

### 17. [Bugfix][Spec Decode] Synchronize adaptive DSpark decisions across TP ranks
- **代码层面**：`adaptive_verification.py` 在决策"本步接受几个 draft token"时用 `get_tp_group().broadcast_object` 同步各 rank 的 batch 布局；新增多进程 gloo 测试 `tp_consensus_worker` 验证各 rank 布局一致。
- **新手概念课堂**：像合唱团，每个人自己数拍子会乱，必须听指挥（rank 0）统一口令。
- **对你有什么影响**：TP 多卡跑 DSpark 自适应验证时，不会再因各 rank 决策不一致导致 batch 错位、输出乱码。

### 18. [ROCm][Bugfix] Allow lut_b_moe online quantization on ROCm
- **代码层面**：`vllm/platforms/rocm.py` 的 `supported_quantization` 列表加入 `"lut_b_moe"`。
- **新手概念课堂**：像"白名单"——以前 ROCm 平台不认这个量化方案，现在加进允许列表，加载时不再报"不支持"。
- **对你有什么影响**：在 AMD GPU 上用 lut_b_moe 在线量化的 MoE 模型可以正常加载运行。

### 19. [ROCm][Kimi-K3] Keep the AttnRes block bank in a persistent workspace
- **代码层面**：`vllm/models/kimi_k3/amd/linear.py` 把每步按 token 数分配的 AttnRes block bank 改成从 `workspace` 管理器取固定大小 bank（按 `max_num_batched_tokens` 分配），DBO 下按 ubatch id 分槽。新增测试验证"增长/收缩/重复"三种 step 都复用同一块。
- **新手概念课堂**：像从"每次吃饭现买锅"改成"家里常备一口大锅"，减少反复分配释放（显存碎片和开销）。
- **对你有什么影响**：Kimi-K3 在 ROCm 上长序列推理时显存更稳，不再随 token 数波动反复分配。

### 20. [Doc][Recipe] Muse Spark Open: DSpark + specialized decode mode; gate Kimi-K3.1 low-latency decode off AMD
- **代码层面**：`Muse-Spark-Open-816B-A42B.yaml` 新增 `dspark_specialized` 特性（NVIDIA GB300 / AMD MI355X 支持，需 nightly 镜像），补充 `notes` 提示镜像版本；同时把 Kimi-K3.1 的 low-latency decode 在 AMD 上标记为不支持。
- **新手概念课堂**：像"菜单上标注'本店特色菜，需提前一天预订'"，告诉你哪些硬件+镜像组合才能点这道菜。
- **对你有什么影响**：想试 Muse Spark 的 specialized decode，按 recipe 选对镜像和硬件即可；在 AMD 上跑 Kimi-K3.1 时不会误开 low-latency decode 导致启动失败。

## sglang
## 💡 今日关键词

- **关键词：NPU/Ascend 适配**
  - **一句话解释**：让 SGLang 在华为昇腾（Ascend）NPU 上也能跑起来，包括权重缓存、启动权重加载等。
  - **为什么社区在关注**：今天有两个 commit（#40141、#41495）专门给 NPU 加功能，说明 SGLang 正在从"CUDA 专属"走向"多硬件后端"。国产 AI 芯片生态正在被认真对待。
  - **对新手意味着什么**：如果你在学异构计算或国产芯片适配，这是一个正在快速扩张的方向，相关岗位和开源贡献机会都在增加。

- **关键词：Pipeline Parallelism（流水线并行）约束**
  - **一句话解释**：把一个大模型切成多段，放在不同 GPU 上"接力"计算。今天有多个 commit 在收紧它的参数校验。
  - **为什么社区在关注**：#49756 禁止 `--scheduler-recv-interval > 1` 配合 PP，#49722 修复 PP 下的 abort 竞态，#48382 修复 PP 下 MoRI DSA 状态配对。说明 PP 是当前生产部署的重点，但边界情况多，社区在密集补洞。
  - **对新手意味着什么**：分布式推理是 LLM 工程的核心技能，PP 是其中最容易出 bug 的部分，值得深入理解。

- **关键词：输入校验与错误返回（400 vs 500）**
  - **一句话解释**：把用户传错的参数在入口就拦下来，返回明确的 400 错误，而不是让它在 GPU 上炸掉。
  - **为什么社区在关注**：#49758 拒绝 `logprob_start_len < -1`，#49414 把 `top_logprobs` 限制在 `[0, vocab_size]`。这些都是"一个坏参数杀死整个调度器"的典型事故。
  - **对新手意味着什么**：写服务端代码时，**参数校验是最便宜、最有价值的防御**。一个 `if` 能省掉一次线上事故。

- **关键词：投机解码（Speculative Decoding）性能调优**
  - **一句话解释**：用小模型"猜"几个 token，大模型一次性验证，加速生成。#44115 给 FlashInfer 的 verify 计划传了 `uniform_q_len` 提示。
  - **为什么社区在关注**：推理加速是 LLM 落地的刚需，投机解码是当前最主流的加速手段之一，每一个 kernel 级别的优化都能带来实打实的吞吐提升。
  - **对新手意味着什么**：想进推理优化方向，投机解码 + CUDA kernel 是黄金组合技能。

- **关键词：多模态与 KV 事件**
  - **一句话解释**：图片/视频输入的处理，以及把缓存块的事件（谁存了什么）报告给路由器做调度。
  - **为什么社区在关注**：#48385 修复多模态占位 token 被覆盖的 bug，#45737 让 KV 事件携带多模态路由标签，#49673 支持 HEIF/BMP/TIFF 图片。多模态正在从"能用"走向"生产可用"。
  - **对新手意味着什么**：多模态推理是增长最快的应用方向，理解 token 对齐、占位符、缓存键这些细节很有价值。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：请求参数校验——在入口拦住坏参数（来自 #49758、#49414）

- **是什么**：在请求进入调度器之前，检查参数是否合法（比如 `logprob_start_len` 不能小于 -1，`top_logprobs` 不能超过词表大小），不合法就返回 400。
- **为什么重要**：如果不校验，坏参数会一路传到 GPU kernel，触发 `assert` 或越界，导致整个调度器崩溃——一个用户的错误请求会拖垮所有用户。这是典型的"输入验证是最便宜的防御"。
- **动手试试**：启动一个 SGLang 服务，然后发一个非法请求：
  ```bash
  # 启动服务（假设已安装 sglang）
  python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

  # 另一个终端：发一个 top_logprobs 超大的请求
  curl -s http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "messages": [{"role": "user", "content": "Hi"}],
      "logprobs": true,
      "top_logprobs": 999999
    }' | python -m json.tool
  ```
- **预期结果**：你会看到返回 `400` 状态码和类似 `top_logprobs must be in [0, 151936], got 999999.` 的错误信息，而不是服务崩溃或 500。这说明校验生效了。

---

### 知识点 2：Pipeline Parallelism 的参数互斥约束（来自 #49756）

- **是什么**：PP（流水线并行）把模型按层切到多张卡上，每张卡只算一部分。`--scheduler-recv-interval` 控制调度器多久收一次请求。当它 > 1 时，各 stage 的节奏会错开，导致 PP 各 stage 不同步。
- **为什么重要**：这类"参数组合不兼容"的问题，如果不显式拒绝，会在运行时表现为难以调试的挂起或错误结果。显式 `assert` 是最佳实践。
- **动手试试**：用 Python 直接调用校验函数（不需要 GPU）：
  ```python
  from types import SimpleNamespace
  from sglang.srt.arg_groups.validation_hook import check_pipeline_parallel_compat

  cfg = SimpleNamespace(
      disable_overlap_schedule=True,
      enable_multi_layer_eagle=False,
      disaggregation_mode="prefill",
      min_free_slots_delay=None,
      scheduler_recv_interval=2,   # 故意设成 2
      # ... 其他字段按需补齐
  )
  try:
      check_pipeline_parallel_compat(cfg)
  except AssertionError as e:
      print("被拒绝了：", e)
  ```
- **预期结果**：输出 `被拒绝了： --scheduler-recv-interval > 1 is not supported with pipeline parallelism; use --scheduler-recv-interval 1`。你会理解"参数校验"在框架里长什么样。

---

### 知识点 3：多进程张量零拷贝传输（IPC）（来自 #41495）

- **是什么**：权重缓存的 daemon 进程要把 GPU 上的权重张量"零拷贝"地交给推理进程。CUDA 用 `_share_cuda_`，NPU 用 `torch_npu` 注册的 reducer。如果平台不支持，就退化成 CPU 拷贝（慢但能跑）。
- **为什么重要**：零拷贝能省掉大量显存带宽和启动时间。理解 IPC 机制，你就理解了"进程间共享 GPU 内存"的底层原理。
- **动手试试**：在 CUDA 机器上验证 PyTorch 的张量归约函数名：
  ```python
  import torch
  import torch.multiprocessing.reductions as r

  t = torch.randn(4, device="cuda")
  rebuild_fn, args = r.reduce_tensor(t)
  print("reducer name:", getattr(rebuild_fn, "__name__", rebuild_fn))
  ```
- **预期结果**：CUDA 上会打印类似 `rebuild_cuda_tensor`；在 NPU 上（需 `import torch_npu`）应看到 `rebuild_npu_tensor`。如果看到的是别的名字，说明零拷贝不可用，会退化成 CPU 拷贝——这正是 #41495 里那段探测代码在做的事。

---

### 知识点 4：投机解码的 verify 计划与 `uniform_q_len`（来自 #44115）

- **是什么**：EAGLE 投机解码里，每个请求的 verify 阶段有固定数量的 query 行（`num_tokens_per_req`）。把这个"统一长度"作为提示传给 FlashInfer 的 `plan()`，kernel 就能选更优的 tile 大小。
- **为什么重要**：CUDA graph 捕获时 kernel 的 tile 是固定的，每次 replay 必须传同样的提示，否则结果不一致。这是"图捕获 + 动态参数"的经典陷阱。
- **动手试试**：理解逻辑即可，实际需要 FlashInfer + GPU。可以读这段纯 Python 逻辑：
  ```python
  # 模拟 _verify_plan_uniform_q_len 的决策
  def verify_plan_uniform_q_len(spec_info):
      if spec_info is None or spec_info.spec_input_type != "EAGLE_VERIFY":
          return None
      if spec_info.num_tokens_per_req <= 0:
          return None
      return int(spec_info.num_tokens_per_req)

  class SI:
      spec_input_type = "EAGLE_VERIFY"
      num_tokens_per_req = 5

  print(verify_plan_uniform_q_len(SI()))   # 5
  SI.num_tokens_per_req = -1
  print(verify_plan_uniform_q_len(SI()))   # None（未设置，保持原行为）
  ```
- **预期结果**：输出 `5` 和 `None`。你会理解"默认值 -1 表示未设置，要回退到旧行为"这种约定。

---

### 知识点 5：多模态占位 token 的对齐（来自 #48385）

- **是什么**：多模态输入里，图片/视频/音频在 `input_ids` 里用特殊的占位 token（如 `im_token_id`）表示。如果后续处理覆盖了这些位置，模型就会"看不到"图片。
- **为什么重要**：token 对齐是多模态推理最容易出错的地方。一个位置偏移就会让模型输出乱码或忽略图像。
- **动手试试**：用 numpy 模拟 `pad_mm_tokens` 的核心思想——按模态分组，只在对应位置填占位符：
  ```python
  import numpy as np

  input_ids = np.array([1, 2, 3, 4, 5, 6])
  # 假设位置 1-2 是图片，位置 4 是音频
  im_mask = np.array([False, True, True, False, False, False])
  audio_mask = np.array([False, False, False, False, True, False])

  im_token_id = 100
  audio_token_id = 200

  input_ids[im_mask] = im_token_id
  input_ids[audio_mask] = audio_token_id
  print(input_ids)
  ```
- **预期结果**：输出 `[  1 100 100   4 200   6]`。图片和音频的占位符只出现在对应位置，其他 token 不受影响——这就是 #48385 修复的核心。

---

## 🔥 重要更新

- **#49756 拒绝 `--scheduler-recv-interval > 1` 配合 PP**：把原本只在特定 invariant-check 下才报的错，提前到参数校验阶段。生产环境配 PP 的人不会再踩"运行时才挂"的坑。
- **#49414 把 `top_logprobs` 限制在 `[0, vocab_size]`**：之前传超大值会让 `topk` 直接杀死调度器，现在返回 400。这是一个真实的线上稳定性修复。
- **#48385 修复多模态占位 token 被覆盖**：影响所有图片/视频/音频输入的正确性，属于"静默错误"类 bug，修复价值高。
- **#41495 NPU 权重缓存支持**：让昇腾平台也能用零拷贝权重共享，是 NPU 生态走向生产可用的重要一步。

---

## 📋 逐条解读

### 1. [Feat] sgl-router: Add production API compatibility (#49727)
- **代码层面**：主要改 `sgl-router/README.md`、`REFERENCE.md`、`monitoring/README.md` 和 `buckets_reorg.rs`。新增 `ReadyPools` 结构体，把"可服务的引擎"和"解码器"分开管理；文档里补充了 `/server_info`、`/model_info`、`/health_generate` 等端点，以及不支持端点返回 `501 unsupported_endpoint` 的说明。
- **新手概念课堂**：路由器就像公司前台——它不干活，但决定把访客（请求）分给哪个员工（worker）。"生产 API 兼容"意味着前台学会了更多"标准话术"，同时明确告诉访客"这个业务我们不办"（501），而不是让人干等。
- **对你有什么影响**：如果你用 sgl-router 做网关，现在能更清楚地知道哪些端点是转发给 worker 的、哪些是明确不支持的，排障更容易。

### 2. [Fix] tokenizer_manager: Reject logprob_start_len below -1 with a 400 (#49758)
- **代码层面**：在 `tokenizer_manager.py` 新增 `_validate_logprob_start_len`，在 `_validate_one_request` 里调用。当 `return_logprob=True` 且 `logprob_start_len < -1` 或不是 int 时抛 `ValueError`（最终变成 400）。测试文件新增 8 个用例覆盖边界。
- **新手概念课堂**：`logprob_start_len` 决定"从第几个 token 开始返回对数概率"。`-1` 表示只返回输出 token；更小的负数会从 prompt 末尾往前切，导致调度器里的 `assert` 失败。就像点外卖时填"从第 -5 道菜开始上"，厨房直接懵了。
- **对你有什么影响**：传错参数会得到清晰的 400 错误，而不是服务崩溃。

### 3. [Fix] server_args: Reject --scheduler-recv-interval above 1 with pipeline parallelism (#49756)
- **代码层面**：在 `validation_hook.py::check_pipeline_parallel_compat` 新增 `assert cfg.scheduler_recv_interval <= 1`；删掉 `scheduler_pp_mixin.py` 里原来只在 invariant-check 下才触发的检查；同步更新文档和单测。
- **新手概念课堂**：PP 的 stage 0 每轮都把请求往后传，但后面的 stage 只在"recv skipper 放行"的轮次才收。如果 interval > 1，各 stage 就像接力赛里有人跑两圈才交棒，队伍就散了。
- **对你有什么影响**：配 PP 时如果误设 interval > 1，启动阶段就会报错并提示正确用法，不用等到运行时才发现。

### 4. [Chore] sgl-router: Remove a local design-doc path from the crate docs (#49755)
- **代码层面**：删掉 `sgl-router/src/lib.rs` 顶部注释里指向某个开发者本地路径（`~/.claude/projects/-Users-kangyan-zhou-...`）的设计文档链接。
- **新手概念课堂**：这就像代码注释里留了"详见我电脑 D 盘的文档"——别人根本打不开。Rust 的 `//!` 是 crate 级文档，会出现在 `cargo doc` 里，不能放本地路径。
- **对你有什么影响**：无功能影响，但体现了"公开代码不能引用私有路径"的工程规范。

### 5. [Feat] [NPU] startup-weight-load: Support overlap mode on NPU (#40141)
- **代码层面**：`startup_weight_load.py` 的 `StartupWeightLoadOptions` 新增 `is_npu` 字段；`_get_unsupported_reason` 里把"仅 CUDA"改成"CUDA 或 NPU"，device 允许 `"npu"`。测试同步更新。
- **新手概念课堂**："启动权重加载重叠"指在加载权重的同时做 CUDA graph 捕获等准备工作，缩短启动时间。原来只支持 CUDA，现在 NPU 也能用。
- **对你有什么影响**：昇腾用户启动大模型时，启动时间有望缩短。

### 6. [Feat] [NPU] weight-cache: Add NPU weight cache support (#41495)
- **代码层面**：`daemon.py` 的 `load()` 允许 `is_npu()` 平台通过；`transport.py::choose_daemon_transport_backend` 新增 NPU 分支：先 `import torch_npu` 注册 reducer，探测 `reduce_tensor` 是否返回 `rebuild_npu_tensor`，不是则警告并退化为 CPU 拷贝，最后返回 `TorchIpcTransportBackend`。
- **新手概念课堂**：权重缓存 = 把加载好的权重存在共享内存里，多个进程直接"引用"而不复制。零拷贝需要平台提供"共享句柄"能力。NPU 上如果 HDK/CANN 版本不够，就退化成普通拷贝——像共享单车坏了就改走路，慢但能到。
- **对你有什么影响**：昇腾用户可以用 `--weight-cache-mode` 加速多实例启动；版本不够时会看到明确警告而不是神秘崩溃。

### 7. [Refactor] scheduler: Edit request lists only through move and the retirer (#49677)
- **代码层面**：大范围重构。`grammar_manager.py` 用 `move([req], self.grammar_queue)` 替代 `join`；`decode.py` 删掉 `pending_reqs` 列表，改用 `DecodeRequest.awaiting_prefill_dp_rank` 标志；`_abort_retracted` 用列表推导替代 `take`。核心思想：请求列表只能通过 `move` 和 `request_retirer` 修改，杜绝散落的 `append`/`remove`。
- **新手概念课堂**：这就像公司规定"所有物资进出必须走仓库管理系统"，不允许部门自己偷偷搬。好处是任何时刻都能审计"东西在哪"，不会出现"两个地方都以为自己在管"的竞态。
- **对你有什么影响**：内部重构，普通用户无感知；但长期看能减少调度器的诡异 bug。

### 8. [Feat] [AMD] tokenizer: Install fastokens in the ROCm image (#47923)
- **代码层面**：`docker/rocm.Dockerfile` 在每个 ROCm 分支的 `srt_extras` 里加 `fastokens`；`pyproject_other.toml` 新增 `fastokens = ["fastokens>=0.3.2"]` extra，并把它加进 `all_hip*`。
- **新手概念课堂**：`fastokens` 是一个快速分词库（tokenizer）。ROCm 镜像之前只装 `srt_hip*` 而不是 `all`，所以漏掉了这个 extra——就像套餐里少配了一道菜，得单独点。
- **对你有什么影响**：用 AMD ROCm 镜像的用户，分词速度会更快，且不需要手动装。

### 9. [Fix] [AMD] monokernels: Pin sgl-amd-mono at 7113024f for DSpark on current SGLang (#49428)
- **代码层面**：`pyproject_other.toml` 里 `monokernels_hip` 的 git commit 从 `e52803f9...` 改为 `7113024f...`。
- **新手概念课堂**：monokernel 是为特定 GPU 架构手写的融合 kernel。pin 到某个 commit 是为了保证"DSpark 特性"和"当前 SGLang 版本"能对上——像给插件锁定兼容的主程序版本。
- **对你有什么影响**：AMD gfx950 用户启用 `--enable-amd-monokernel` 时能正常用 DSpark。

### 10. [Fix] logprobs: Bound top_logprobs to [0,vocab_size] (#49414)
- **代码层面**：`serving_base.py` 新增 `_top_logprobs_error(field, value)`，检查 `0 <= value <= vocab_size`；`serving_chat.py` 校验 `request.top_logprobs`，`serving_completions.py` 校验 `request.logprobs`。
- **新手概念课堂**：`top_logprobs` 是"返回概率最高的前 N 个 token"。N 超过词表大小毫无意义，而且会让 `topk` kernel 申请超大缓冲区，直接 OOM 或断言失败。就像问"给我前 100 万名员工的名单"，但公司只有 1000 人。
- **对你有什么影响**：传超大值会得到 400 和明确提示，不会拖垮服务。

### 11. [Fix] offload: Keep DSA indexer parameters resident on GPU under CPU offload (#48316)
- **代码层面**：`offloader.py::_get_resident_parameter_ids` 新增：如果模块有 `indexer` 子模块，把它的参数 id 加入 resident 集合，不参与 CPU offload。测试新增 `test_indexer_runs_outside_offloaded_forward`。
- **新手概念课堂**：CPU offload 是把暂时不用的权重挪到 CPU 省显存。但 DSA indexer 在 breakable CUDA graph 里是"eager break"——它在 offloaded forward 返回之后才被调用，此时权重已经被挪走了，就会报错。所以必须让它"常驻"。
- **对你有什么影响**：用 CPU offload + DSA 模型的用户，不会再遇到 indexer 找不到权重的崩溃。

### 12. [Perf] [NVIDIA] flashinfer: Pass uniform_q_len to the target-verify plan for EAGLE speculative decoding (#44115)
- **代码层面**：`flashinfer_backend.py` 新增 `_verify_plan_uniform_q_len(spec_info)`；`fast_prefill_plan` 新增 `uniform_q_len` 参数并传给 `plan()`；`call_begin_forward` 里把该提示加入 `paged_plan_kwargs`。另外把 mask buffer 的断言从 `<= numel` 改成 `+8 <= numel`（SM90 fp8-KV kernel 会多读 8 字节）。
- **新手概念课堂**：EAGLE verify 时每个请求的 query 行数固定，告诉 FlashInfer 这个"统一长度"，它就能选更优的 kernel tile。CUDA graph 捕获时 tile 固定，所以每次 replay 必须传同样的提示——像给裁缝报"统一尺码"，他才能批量裁剪。
- **对你有什么影响**：用 EAGLE 投机解码 + FlashInfer 的用户，verify 阶段吞吐提升。

### 13. [Fix] scheduler: Apply pending chunked-prefill aborts in the PP PD prefill loop (#49722)
- **代码层面**：`scheduler_pp_mixin.py::event_loop_pp_disagg_prefill` 在 `process_prefill_chunk` 前加一行 `self.process_pending_chunked_abort()`。测试新增 `TestPPPrefillLoopAppliesChunkedAbort`。
- **新手概念课堂**：chunked prefill 把一个长 prompt 分块处理。如果在块之间收到 abort 请求，但没及时处理，下一块还会继续算——浪费算力，还可能让已 abort 的请求占着 slot。这行代码就是"每轮先清掉待处理的 abort"。
- **对你有什么影响**：PP + PD 分离 + chunked prefill 场景下，取消请求能及时生效，不再浪费 GPU。

### 14. [Fix] multimodal: keep placeholder token ids at multimodal positions of input_ids (#48385)
- **代码层面**：`mm_utils.py` 提取 `_mm_placeholder_token_id` 辅助函数；新增 `_fill_mm_placeholder_token_ids` 和 `_sorted_pad_values_tensor`；在 embedding 散射流程里用 mask 把占位 token id 填回 `input_ids` 的正确位置。
- **新手概念课堂**：多模态输入里，图片位置在 `input_ids` 里是占位符（如 `<image>`）。如果后续处理把这些位置覆盖成别的 id，模型就"看不见"图片了。这就像剧本里写着"[此处插入图片]"，结果被改成了"[此处插入文字]"，演员当然演错。
- **对你有什么影响**：图片/视频/音频输入的正确性提升，尤其是多模态 + padding 组合的场景。

### 15. [Doc] cookbook: Drop the PP chunk sizing flags from the Kimi-K3.1-Pro prefill cells (#49716)
- **代码层面**：`kimi-k3.1-pro.jsx` 里删掉两行 `--pp-min-chunked-prefill-size 4096` 和 `--pp-chunk-overhead-tokens 2000`。
- **新手概念课堂**：文档里的启动命令是给用户复制粘贴的。删掉这两个 flag 说明它们已经不再需要（可能默认值已优化，或已废弃）。
- **对你有什么影响**：照抄 cookbook 启动 Kimi-K3.1-Pro 时，命令更简洁，不会因为过时 flag 报错。

### 16. [Perf] kimi-k3.1: Per-M decode GEMM tactics and side-stream attention GEMMs (#49478)
- **代码层面**：`kimi_k3.py` 新增 `_K3_TGV_M_RANGES`、`_K3_GB300_TGV_M_RANGES`、`_K3_GB300_BFA_SIDE_TGV_M_RANGES` 三张表，按 `(n, k)` 映射到 `(m_lo, m_hi, tactic)`；新增 `_k3_side_stream_max_tokens()`；在 `_merge_bfa_weights` 里为 GB300 + cutedsl 后端构造 `TGVMRanges`。
- **新手概念课堂**：GEMM（矩阵乘）是 Transformer 的主要计算。不同的 M（batch 行数）适合不同的 kernel tactic（切块策略）。"Per-M tactics"就是"按 batch 大小选最优切法"，像裁缝按衣服尺码选不同的裁剪方案。
- **对你有什么影响**：Kimi-K3.1 在 GB300 上的 decode 吞吐提升，尤其是小 batch 场景。

### 17. [Feat] kv-events: Report caller multimodal routing labels as block spans (#45737)
- **代码层面**：`kv_events.py::BlockStored` 新增 `mm_routing_spans` 字段（dynamo 格式专用）；`encoder/receiver.py` 在 `_finish_assemble` 里调用 `attach_mm_routing_labels`；`dspark_verify_window.py` 修复 NPU kernel 未屏蔽被拒绝 draft token 的问题。
- **新手概念课堂**：KV 事件是"缓存块变动"的通知。多模态路由标签告诉路由器"这块缓存对应哪张图/哪段视频"，方便做亲和性调度。就像快递包裹上贴"内含易碎品"标签，分拣时特殊处理。
- **对你有什么影响**：用 KV 事件做多模态感知路由的用户，能拿到更细粒度的路由信息。

### 18. [Fix] kernels: Synchronize AOT MoE prefix before expert lookup (#49656)
- **代码层面**：`moe_align_kernel.cu` 里把"先写 `scan_buf` 再拷到 `prefix`"改成"直接写 `prefix`"，去掉中间 `__syncthreads()`；二分查找的 `left/right/mid` 从 `int` 改为 `uint32_t`，最后 `static_cast<int32_t>(left) - 2`。
- **新手概念课堂**：MoE 里要把 token 按专家分组。`prefix` 是"每个专家之前有多少 token"的前缀和。原来分两步写（先 `scan_buf` 再 `prefix`），中间需要同步；直接写 `prefix` 省掉一次同步，更快。`uint32_t` 避免负数 mid 的 UB。
- **对你有什么影响**：MoE 模型的 AOT kernel 略快，且更稳健（无符号二分避免溢出）。

### 19. [Feat] multimodal: Decode HEIF images and read relative BMP, TIFF and HEIF paths (#49673)
- **代码层面**：`media_io.py` 顶部 `try: from pillow_heif import register_heif_opener; register_heif_opener() except ImportError: pass`；`image_extension_names` 新增 `.bmp`、`.tif`、`.tiff`、`.heic`、`.heif`。测试扩展到这几种扩展名。
- **新手概念课堂**：HEIF 是苹果设备常用的图片格式，Pillow 默认不支持，需要 `pillow_heif` 插件注册 opener。就像你的播放器默认不认 `.mkv`，装个解码器就能放了。
- **对你有什么影响**：可以直接传 iPhone 拍的 HEIC 图片或 BMP/TIFF 给多模态模型，不用手动转格式。

### 20. [Fix] [AMD] disaggregation: Pair MoRI DSA state across PP prefill stages (#48382)
- **代码层面**：`mori/conn.py::_send_swa_dsa_state` 里，当 `state_type == DSA`、`pp_size > 1`、无 layer ids、`n_src != n_dst`、且非 hybrid MLA 时，走 `get_mla_kv_ptrs_with_pp` 按 PP stage 切片配对；否则走原来的 `build_transfer_entry_pairs`。测试新增 GLM-5.2 在 PP4 下的配对用例。
- **新手概念课堂**：PP 把模型切成多段，每段的 DSA state 数量不同（比如 79 层切 19/19/20/21）。如果按位置一一配对，就会错位——第 3 段的 state 对到第 2 段的槽上。这就像把 4 摞书按"第 1 本对第 1 本"配对，但每摞厚度不同，必须按"每摞的起始位置"配对。
- **对你有什么影响**：AMD + MoRI + PP + DSA 模型的用户，PD 分离传输不再错位，推理结果正确。

---

## 🔥 重要更新

- **#49756 拒绝 `--scheduler-recv-interval > 1` 配合 PP**：把原本只在特定 invariant-check 下才报的错，提前到参数校验阶段。生产环境配 PP 的人不会再踩"运行时才挂"的坑。
- **#49414 把 `top_logprobs` 限制在 `[0, vocab_size]`**：之前传超大值会让 `topk` 直接杀死调度器，现在返回 400。这是一个真实的线上稳定性修复。
- **#48385 修复多模态占位 token 被覆盖**：影响所有图片/视频/音频输入的正确性，属于"静默错误"类 bug，修复价值高。
- **#41495 NPU 权重缓存支持**：让昇腾平台也能用零拷贝权重共享，是 NPU 生态走向生产可用的重要一步。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)