# 🚀 vLLM & SGLang 每日更新 — 2026-10-22

> 自动生成于 2026-10-22 06:32 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：Rust 前端**
  - 一句话解释：vLLM 正在把 HTTP 服务层从 Python 逐步搬到 Rust，以获得更高的并发吞吐和更低的延迟。
  - 为什么社区在关注：今天有 3 个 commit（#62872、#62874、#62875）都在改 Rust 前端，说明 Rust 化已经不是实验，而是持续演进的主线。这反映了推理框架"Python 做模型、Rust 做服务"的分层趋势。
  - 对新手意味着什么：如果你会 Rust，又懂一点 LLM 服务，现在是切入 vLLM 生态的好时机；即使不写 Rust，理解"服务层与引擎层解耦"对排查线上问题也很有帮助。

- **关键词：LoRA 正确性**
  - 一句话解释：LoRA 是在冻结的主模型旁边挂一小块可训练参数，用来低成本定制模型。
  - 为什么社区在关注：今天有 4 个 commit（#61279、#48421、#61284、#60306）围绕 LoRA 打补丁——支持新模型、修复 CPU 算子、修复多流死锁、修复调度计数。说明 LoRA 在生产中用得越来越多，边角场景的 bug 开始集中暴露。
  - 对新手意味着什么：LoRA 是性价比最高的微调方式，搞懂它的推理侧实现（尤其是 batch 里混多个 adapter）是很有价值的技能。

- **关键词：多模态占位符（placeholder）**
  - 一句话解释：多模态模型在文本 prompt 里用特殊字符串标记"这里要插入一张图/一段音频"，推理时再替换成真正的特征。
  - 为什么社区在关注：#61196 修复了 Gemma 4 的占位符字符串不匹配导致图片替换失败的问题。这类 bug 很隐蔽——模型能跑，但图片根本没被"看到"。说明多模态支持正在快速铺开，细节坑很多。
  - 对新手意味着什么：做多模态应用时，"模型输出正常但答非所问"往往就是占位符或预处理对齐问题，学会看 processor 的输出是关键。

- **关键词：MoE（混合专家）**
  - 一句话解释：MoE 模型每层有很多"专家"子网络，每个 token 只激活其中几个，用更少算力换更大参数量。
  - 为什么社区在关注：#61749（自定义算子要求精确 stride）、#61786（EPLB 用 uint8 视图搬运 1 字节张量）、#61284（共享专家与 LoRA 的流冲突）三个 commit 都在 MoE 上。说明 MoE 已是主流架构，性能和正确性优化非常活跃。
  - 对新手意味着什么：MoE 的"专家并行/负载均衡/量化"是当前最热的研究和工程方向之一。

- **关键词：结构化输出（structured output）**
  - 一句话解释：强制模型输出符合 JSON Schema 或正则的格式，方便程序解析。
  - 为什么社区在关注：#62872 把"语法编译失败"从 500 服务器错误改成 400 客户端错误。说明结构化输出已进入生产可用阶段，错误语义的精细化是成熟标志。
  - 对新手意味着什么：做 Agent/工具调用时，结构化输出是必备能力；理解"约束编译"这一层能帮你快速定位"为什么我的 JSON 老是生成失败"。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：LoRA 的 token 级路由（-1 表示"这个 token 不用 LoRA"）**
  - **是什么**：在一个 batch 里，不同请求可能用不同 LoRA adapter，甚至有的请求完全不用 LoRA。vLLM 用 `token_lora_indices` 数组标记每个 token 该用哪个 adapter，`-1` 表示"基础模型，不加 LoRA"。
  - **为什么重要**：如果算子没处理 `-1`，就会用 Python 的负索引取到"最后一个 adapter"，导致基础模型请求被悄悄套上了别人的 LoRA——输出错乱但不会报错，极难排查。这正是 #48421 修的问题。
  - **动手试试**：
    ```python
    import torch

    # 模拟：2 个 token，token 0 用 LoRA 0，token 1 不用 LoRA
    lora_indices = torch.tensor([0, -1], dtype=torch.long)
    lora_weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 个 adapter

    # 错误写法：负索引会取到 adapter 1
    wrong = lora_weights[lora_indices]
    print("错误写法取到:", wrong)

    # 正确写法：先做 mask
    no_lora_mask = (lora_indices == -1).unsqueeze(-1)
    safe_indices = lora_indices.clamp(min=0)
    correct = lora_weights[safe_indices].masked_fill(no_lora_mask, 0)
    print("正确写法取到:", correct)
    ```
  - **预期结果**：你会看到错误写法第二行是 `[3.0, 4.0]`（adapter 1 的权重），正确写法第二行是 `[0.0, 0.0]`。理解这个差异，你就懂了 #48421 在修什么。

- **知识点 2：CUDA 多流（stream）与 NCCL 集合通信的死锁风险**
  - **是什么**：GPU 上可以同时跑多条"流"（stream），像多条并行流水线。MoE 的共享专家常被放到辅助流上与主计算重叠。但如果两个流都在同一个 NCCL 通信器上发起 all-gather/all-reduce，在 CUDA Graph 捕获后，重放顺序可能在不同 rank 间不一致，导致互相等待 → 死锁。
  - **为什么重要**：#61284 就是因为"全分片 LoRA + 共享专家辅助流"触发了这个死锁（issue #60845）。这是分布式推理里最典型的"看起来能跑、压力一大就挂"的问题。
  - **动手试试**（在单机多卡环境，或阅读代码理解即可）：
    ```python
    import torch

    # 观察 CUDA Graph 捕获时流的依赖关系
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        a = torch.randn(1000, 1000, device="cuda")
        b = a @ a

    with torch.cuda.stream(s2):
        c = torch.randn(1000, 1000, device="cuda")
        d = c @ c

    # 关键：两个流之间如果没有显式 event 同步，
    # 顺序就是"未定义"的 —— 在多卡集合通信里这就是隐患
    torch.cuda.synchronize()
    print("两流并行完成，但顺序未保证")
    ```
  - **预期结果**：单卡上只是并行完成；但请记住——如果 `b` 和 `d` 里各有一个 `dist.all_reduce`，而不同 rank 的流调度顺序不同，就会死锁。这就是为什么 vLLM 选择"直接禁用重叠"这种保守但正确的方案。

- **知识点 3：1 字节 dtype 与 NCCL 的兼容性（uint8 视图技巧）**
  - **是什么**：FP8（E8M0 scale）、FP4（e2m1fn_x2）这些新量化格式每个元素只占 1 字节甚至半字节。但 NCCL 的集合通信并不认识这些 dtype，只认识标准类型。
  - **为什么重要**：MoE 的 EPLB（专家负载均衡）需要在 GPU 间搬运专家权重。如果直接传 FP4 张量会报错或走低效路径。#61786 的做法是：把 1 字节 dtype 用 `.view(torch.uint8)` 变成字节视图——底层内存完全一样，只是告诉 NCCL"这是一堆字节"。
  - **动手试试**：
    ```python
    import torch

    # 构造一个 FP8 张量（如果硬件/版本支持）
    try:
        fp8 = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
        byte_view = fp8.view(torch.uint8)
        print("FP8 shape:", fp8.shape, "字节视图 shape:", byte_view.shape)
        print("共享存储:", fp8.data_ptr() == byte_view.data_ptr())

        # 修改字节视图 = 修改原张量
        byte_view.fill_(0x7F)
        print("原 FP8 张量也被改了:", fp8[0, :3])
    except Exception as e:
        print("当前环境不支持 FP8，跳过:", e)
    ```
  - **预期结果**：`data_ptr()` 相同说明是同一块内存的两种"视角"。这是零拷贝的类型转换技巧，在量化和分布式场景中非常常用。

- **知识点 4：前缀缓存（prefix caching）与多模态的"就绪检查"**
  - **是什么**：前缀缓存让相同 prompt 前缀的请求复用已算好的 KV cache。但多模态请求的"前缀"里包含图片 embedding，这些 embedding 可能是从远端编码器异步加载的——必须等它加载完才能复用。
  - **为什么重要**：#62834 修复了"前缀重放时没检查编码器就绪"的问题。如果不等，调度器会以为缓存可用，实际 embedding 还没到，导致结果错误或崩溃。这是"异步 + 缓存"组合的经典陷阱。
  - **动手试试**（用伪代码理解逻辑）：
    ```python
    # 模拟调度器的就绪检查
    class FakeFeature:
        def __init__(self, offset, length, identifier):
            self.offset, self.length, self.identifier = offset, length, identifier

    def ensure_cache_available(features, num_computed_tokens, loaded):
        """只有当图片位置已被计算，或 embedding 已加载，才算就绪"""
        return all(
            f.offset + f.length <= num_computed_tokens or f.identifier in loaded
            for f in features
        )

    feats = [FakeFeature(offset=383, length=184, identifier="img_1")]
    print("未加载时:", ensure_cache_available(feats, 0, set()))   # False
    print("已加载后:", ensure_cache_available(feats, 0, {"img_1"}))  # True
    print("已计算过:", ensure_cache_available(feats, 600, set()))  # True
    ```
  - **预期结果**：三个输出依次是 `False`、`True`、`True`。这就是 #62834 加到调度器里的核心判断逻辑。

- **知识点 5：结构化输出的"校验错误" vs "服务器错误"**
  - **是什么**：HTTP 400 表示"你的请求有问题"（客户端错误），500 表示"服务器内部炸了"。用户给了个无法编译的 JSON Schema，应该返回 400 而不是 500。
  - **为什么重要**：#62872 给 Rust 前端加了 `VALIDATION_ERROR` 错误码，把结构化输出编译失败映射为 400。这直接影响客户端重试策略——400 不该重试，500 该重试。错误的错误码会让客户端做无意义的疯狂重试。
  - **动手试试**：
    ```python
    # 用 Python 模拟错误分类逻辑
    class RequestError:
        VALIDATION_ERROR = "validation_error"
        def __init__(self, code, message, retryable):
            self.code, self.message, self.retryable = code, message, retryable

    def engine_error(err):
        if err is None:
            return 500, "unknown"
        if err.code == RequestError.VALIDATION_ERROR:
            return 400, err.message
        if err.retryable:
            return 503, err.message
        return 500, err.message

    print(engine_error(RequestError("validation_error", "bad grammar", False)))
    print(engine_error(RequestError("engine_paused", "paused", True)))
    print(engine_error(RequestError("internal", "boom", False)))
    ```
  - **预期结果**：依次输出 `(400, 'bad grammar')`、`(503, 'paused')`、`(500, 'boom')`。这就是 #62872 在 Rust `engine_error` 函数里实现的分类逻辑。

---

## 🔥 重要更新

1. **#61284 修复全分片 LoRA + 共享专家的 NCCL 死锁**：这是 issue #60845 的正式修复，直接决定 MoE + LoRA 组合在生产环境能否稳定运行。方案是保守地禁用辅助流重叠，牺牲一点性能换正确性。

2. **#61772 让 `/v1/embeddings` 支持 late chunking**：token embedding 模型现在可以通过标准 OpenAI 兼容接口做"固定长度分块 + 延迟池化"，这是 RAG 场景的重要能力补全，对做检索增强生成的开发者影响很大。

3. **#48421 修复 CPU LoRA 算子的 `-1` 索引 bug**：混合 batch 中"基础模型请求被套上别人的 LoRA"，属于静默错误——不报错但结果全错。所有跑 CPU LoRA 推理的用户都受影响。

4. **#62872 结构化输出编译失败返回 400**：错误语义精细化，影响所有用 JSON Schema 约束输出做 Agent 的应用的重试逻辑和可观测性。

---

## 📋 逐条解读

### 1. [Bugfix][Pooling] Keep the JinaVL score token when truncating score prompts (#62783)
- **代码层面**：改的是 `vllm/entrypoints/pooling/scoring/io_processor.py`。JinaVL 打分模型会在 prompt 末尾追加一个特殊"打分 token"，这个 token 的向量被用来算相似度。截断逻辑原来可能把这个 token 也截掉，现在通过引入 `reserve_tokens`（预留 token 数）确保截断时给它留位置。测试文件里新增了参数化测试验证截断后最后一个 token 仍是 100（打分 token）。
- **新手概念课堂**：想象你在写一封信，最后要盖一个"邮戳"才能寄出。截断就像"信太长要剪掉一部分"——你绝不能把邮戳剪掉，否则信就作废了。打分 token 就是那个邮戳。
- **对你有什么影响**：如果你用 JinaVL 做 rerank/打分，之前设置 `truncate_prompt_tokens` 时可能得到错误分数，现在修复了。

### 2. [Bugfix][Frontend] Cut stop strings and stop tokens on derender (#62719)
- **代码层面**：改的是 `docs/serving/online_serving/derenderer.md` 和 `tests/entrypoints/scale_out/derender/test_derender.py`。derender 是"把引擎原始 token 还原成文本"的组件。原来它只是简单 detokenize，现在会像引擎的 detokenizer 一样处理 stop token 和 stop string：遇到 stop 就把后面的内容切掉，并正确处理 `stop_reason`、`include_stop_str_in_output`、`min_tokens` 等参数。
- **新手概念课堂**：引擎生成时可能"多生成了一个 stop token"，就像录音机按停止键时又录了 0.1 秒杂音。derender 现在负责把这段杂音剪掉，让输出和原生 `vllm serve` 完全一致。
- **对你有什么影响**：使用分离式部署（generate server + derender server）时，输出文本不再多出奇怪的尾巴，流式和非流式行为更一致。

### 3. [Bugfix][MoE] Require exact strides for moe_forward custom ops (#61749)
- **代码层面**：改的是 MoE 自定义算子 `moe_forward` / `moe_forward_shared` 的输入校验。原来 Inductor（torch.compile 的后端）可能把"带 padding 的 routed 输入"的一个"跨步视图"（strided alias）传给共享专家，而共享专家的 NVFP4 量化 kernel 假设行是紧凑排列的——结果算错。现在算子要求精确 stride，不满足就报错或重新打包。
- **新手概念课堂**：想象一排书架，每个格子放一本书。padding 就像在书之间塞了空气。"跨步视图"是"只读每第 2 个格子"的读法。量化 kernel 假设"书是紧挨着的"，你给它一个跳跃的读法，它就会把空气也当成书来量化。修复就是：要么书真的紧挨着，要么先整理好再给。
- **对你有什么影响**：使用 torch.compile + NVFP4 MoE 时，共享专家的计算结果更可靠，不再出现"编译后精度下降"的诡异问题。

### 4. [Doc][Recipe] Use renamed Mamba prefix flag in Nemotron-3.5-Super-VL FP8 (#62451)
- **代码层面**：只改了一个 YAML 配方文件 `recipes/models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.yaml`，把参数 `--enable-mamba-fine-grained-prefix-cache` 换成新名字 `--enable-mamba-shared-prefix-checkpoint`。
- **新手概念课堂**：CLI 参数改名就像小区门牌号重编——老地址还能用一阵，但新地址才是正式的。配方文件是"一键复现某个模型最佳配置"的说明书，必须跟上参数改名。
- **对你有什么影响**：直接复制这个配方启动 Nemotron 模型的人不会因为参数失效而启动失败。

### 5. [Bugfix][Frontend] Render GPT-OSS chats with Harmony in /tokenize (#62786)
- **代码层面**：改的是 `vllm/entrypoints/serve/tokenize/protocol.py`，新增 `to_chat_completion_request()` 方法，把 `/tokenize` 请求的字段转换成 `ChatCompletionRequest`。原因是 GPT-OSS 模型用 Harmony 格式渲染 chat，`/tokenize` 之前没走这条路径，导致 tokenize 结果和 `/v1/chat/completions/render` 不一致。
- **新手概念课堂**：GPT-OSS 有自己的"对话格式方言"（Harmony），就像不同手机充电口。`/tokenize` 原来用了通用充电口，充不上电；现在会识别模型、切换对口。
- **对你有什么影响**：用 `/tokenize` 预估 token 数或做 prompt 调试时，结果和实际 chat 请求一致了。

### 6. [Core][Rust Frontend] Report structured-output compile failures as validation errors (#62872)
- **代码层面**：改的是 Rust 侧 `engine-core-client/src/protocol/output.rs`（新增 `VALIDATION_ERROR` 常量和 `is_validation_error()`）、`server/src/error.rs`（`engine_error` 函数新增 400 分支）、`server/src/grpc/convert.rs`（gRPC 映射为 `InvalidArgument`）。结构化输出（JSON Schema/正则）编译失败现在返回 HTTP 400 而非 500。
- **新手概念课堂**：去医院看病，400 是"你挂号信息填错了，去改"，500 是"医院系统崩了"。原来把"你 Schema 写错了"报成"医院崩了"，患者（客户端）就会一直重试。修复后语义正确。
- **对你有什么影响**：客户端 SDK 能正确区分"我的请求有问题"和"服务端有问题"，避免无意义重试；错误日志也更清晰。

### 7. [Bugfix][Model] Fix Gemma 4 image/audio placeholder strings (#61196)
- **代码层面**：改的是 `vllm/model_executor/models/diffusion_gemma.py` 和 `gemma4_mm.py` 里的 `get_placeholder_str` 返回值，以及对应测试。原来返回的占位符字符串和 processor 实际期望的不一致，导致 `--chat-template-content-format string` 模式下图片/音频占位符没被替换成特征。
- **新手概念课堂**：占位符就像快递单上的"此处贴面单"。如果模板印的是"贴面单"，系统却去找"贴快递单"，就永远贴不上。模型能跑，但图片被忽略，输出胡言乱语。
- **对你有什么影响**：Gemma 4 多模态在 string 内容格式下能正确"看到"图片和音频了。

### 8. Fix torchcodec lazy-import startup crash when native lib raises OSError (#61090)
- **代码层面**：改的是 `vllm/multimodal/media/audio.py`，把 `except (ImportError, RuntimeError)` 扩展为 `except (ImportError, RuntimeError, OSError)`。torchcodec 的原生库加载失败时抛的是 `OSError`，原来没捕获，导致 vLLM 启动直接崩溃。
- **新手概念课堂**：你装了个"音频解码插件"，但它的底层 DLL 缺依赖。插件自己报的是"我加载失败"（OSError），而 vLLM 只准备了"你没装"（ImportError）和"ffmpeg 缺失"（RuntimeError）两种应对方案，第三种情况直接懵了。
- **对你有什么影响**：即使 torchcodec 装坏了，vLLM 也能正常启动（音频功能降级），而不是整个服务起不来。

### 9. [Model][LoRA] Enable LoRA support for BERT cross-encoders (#61279)
- **代码层面**：改的是 `vllm/model_executor/models/bert.py`，让 `BertForSequenceClassification` 继承 `SupportsLoRA`，添加 `packed_modules_mapping`，并把 `nn.Linear` 分类头换成 `ReplicatedLinear`（支持 LoRA 注入和 TP 复制）。
- **新手概念课堂**：BERT cross-encoder 是"把问题和文档一起读，直接打分"的模型（适合 rerank）。LoRA 是"在原有能力上加一个小补丁学新技能"。原来这个模型"不接收补丁"，现在可以了。
- **对你有什么影响**：可以用 LoRA 微调 BERT reranker（如 ms-marco-MiniLM），低成本适配你的垂直领域排序任务。

### 10. [Rust Frontend] Match DeepSeek DSML markers by token identity (#62875)
- **代码层面**：改的是 Rust 侧 `parser/benches/deepseek_v32.rs` 和 DeepSeek V3.2 工具解析器。DeepSeek 的 `｜DSML｜` 是特殊 token，解析器现在通过 tokenizer 的 token id 来识别标记，而不是靠字符串匹配。benchmark 也改用带 token 归属的 chunk 喂入。
- **新手概念课堂**：字符串匹配像"用眼睛找'苹果'两个字"，token 匹配像"用字典查'苹果'这个词的编号"。前者遇到特殊 token、编码差异、分块边界就会漏；后者精确。
- **对你有什么影响**：DeepSeek V3.2 的工具调用解析在流式输出、特殊 token 场景下更准确，不会把工具调用误判成普通文本。

### 11. [Rust Frontend] Pass attributed deltas and the tokenizer to tool parsers (#62874)
- **代码层面**：改的是 Rust 侧 `chat/src/output/default/mod.rs`、`parser/tool/mod.rs` 等。工具解析器的工厂函数签名从 `create(name, tools)` 变成 `create(name, tools, tokenizer)`，并且流式 delta 现在带 token 归属信息（attributed deltas）。
- **新手概念课堂**：解析器原来拿到的是"一串字符"，不知道每个字符来自哪个 token；现在拿到的是"每个 token + 它的文本"，就像从"听写录音"升级成"带字幕的录音"。
- **对你有什么影响**：这是 #62875 的基础设施铺垫，让所有工具解析器都能做 token 级精确匹配，长期看会提升所有模型的工具调用解析准确率。

### 12. [Bugfix][EPD] Check encoder readiness for prefix replay (#62834)
- **代码层面**：改的是 `vllm/v1/core/sched/scheduler.py`。前缀重放（prefix replay）路径原来没检查多模态 embedding 是否已从编码器加载完成，现在加上了 `ec_connector.ensure_cache_available` 检查，未就绪就推迟请求。
- **新手概念课堂**：EPD = Encoder-Prefill-Decode 分离架构。图片先由编码器算成 embedding，再送进主模型。前缀缓存复用时，如果 embedding 还在路上，就像"菜谱说食材已备好，其实菜还在快递路上"——开火就出错。
- **对你有什么影响**：使用 EPD 分离部署 + 前缀缓存 + 多模态时，不再出现"缓存命中但结果错乱"的问题。

### 13. [Feature][Pooling] Serve late chunking through /v1/embeddings (#61772)
- **代码层面**：改的是文档和 `/v1/embeddings` 的 pooling 路径。现在 `token_embed` 任务的模型（如 ColBERT 类）可以通过标准 `/v1/embeddings` 接口返回 token 向量或固定长度 chunk 向量，支持 late chunking（先编码整篇文档，再按 chunk 池化）。
- **新手概念课堂**：普通 embedding 是"把整篇文章压成一个向量"；late chunking 是"先把每个词编码好，再按段落把词向量平均成段落向量"。好处是段落向量保留了上下文信息，比"先切段再编码"更准。
- **对你有什么影响**：做 RAG 时可以用 OpenAI 兼容接口直接拿到高质量 chunk 向量，不用自己写后处理。

### 14. [Core][Sleep Mode] Use torch.accelerator in CuMemAllocator free/sleep (#62701)
- **代码层面**：改的是 `vllm/device_allocator/cumem.py`，把 `torch.cuda.synchronize` / `torch.cuda.empty_cache` 换成 `torch.accelerator.synchronize` / `torch.accelerator.empty_cache`。这是为了支持非 CUDA 设备（如 ROCm、XPU）。
- **新手概念课堂**：`torch.cuda.xxx` 是"只认 NVIDIA 的遥控器"，`torch.accelerator.xxx` 是"万能遥控器"，插上 AMD 卡也能用。
- **对你有什么影响**：在 AMD GPU 上使用 sleep mode（显存休眠/唤醒）时不再报错。

### 15. [Bugfix][EPLB] Move 1-byte expert tensors as uint8 views (#61786)
- **代码层面**：改的是 `vllm/model_executor/models/interfaces.py` 的 `MixtureOfExperts.set_eplb_state`。注册专家权重时，如果元素大小是 1 字节（FP8、FP4x2），就用 `.view(torch.uint8)` 转成字节视图再交给 EPLB，因为 NCCL 不支持这些 dtype。
- **新手概念课堂**：见上面"知识点 3"。核心是"同一块内存，换个标签给 NCCL 看"。
- **对你有什么影响**：使用 FP8/FP4 量化的 MoE 模型 + 专家并行负载均衡时，权重搬运不再报错或走低效路径。

### 16. [Doc] DeepSeek-V4.1-Flash: fix TP4 c32 max-num-seqs in the B200 AgentX guide (#62971)
- **代码层面**：只改了一个 YAML 配方文件里的表格，把 TP4、c32 场景下的 `max-num-seqs` 从 `64` 改为 `128`。
- **新手概念课堂**：配方里的性能表格是"实测调优结果"。写错了会让人按错误配置跑，得到次优性能。这是纯文档修正。
- **对你有什么影响**：按官方配方部署 DeepSeek-V4.1-Flash 时，TP4 场景能获得作者实测的最佳并发配置。

### 17. [Bugfix][LoRA] Skip no-LoRA tokens (index -1) in CPU torch bgmv ops (#48421)
- **代码层面**：改的是 `vllm/lora/ops/torch_ops/lora_ops.py` 里的 `bgmv_expand`、`bgmv_shrink`、`bgmv_expand_slice`。用 `(lora_indices_tensor == -1)` 做 mask，对不用 LoRA 的 token 跳过计算（或保持原值），而不是用负索引取到最后一个 adapter。
- **新手概念课堂**：见上面"知识点 1"。这是"负索引陷阱"的经典案例。
- **对你有什么影响**：在 CPU 上跑混合 batch（部分请求用 LoRA、部分不用）时，基础模型请求的输出不再被污染。sgmv 因为委托给 bgmv，也一并修复。

### 18. [Bugfix][LoRA] Disable shared-expert aux stream with fully-sharded LoRA (#61284)
- **代码层面**：改的是 `vllm/model_executor/layers/fused_moe/runner/shared_experts.py`。在 `__init__` 里读取 `lora_config.fully_sharded_loras`，在 `_disable_shared_experts_overlap` 里新增分支：如果启用全分片 LoRA，直接返回 `True`（禁用辅助流重叠）。
- **新手概念课堂**：见上面"知识点 2"。两个流在同一个 NCCL 通信器上各发集合通信，CUDA Graph 重放时不同 rank 的顺序可能不一致 → 死锁。修复方式是"别并行，老老实实串行"。
- **对你有什么影响**：MoE 模型 + 全分片 LoRA 的组合不再偶发死锁，代价是共享专家不再与主计算重叠（性能略降，但正确）。

### 19. [Bugfix][Core] Count unscheduled running requests toward max_loras (#60306)
- **代码层面**：改的是 `vllm/v1/core/sched/scheduler.py`。`max_loras` 的计数原来只算"本步被调度的请求"，漏掉了"running 状态但本步被跳过"的请求（例如 PP 等待输出、encoder 预算不足）。现在这些请求也计入，避免超限。
- **新手概念课堂**：`max_loras` 是"同时最多加载几个 adapter"的硬限制，类似餐厅"最多同时开 3 个灶"。原来只数"正在炒菜的灶"，漏数了"菜已下锅、等出餐的灶"，结果开了第 4 个灶——超载。
- **对你有什么影响**：Pipeline Parallel 或带多模态 encoder 预算限制的场景下，LoRA 并发不会再悄悄超过 `max_loras`，避免显存暴涨或 adapter 换入换出抖动。

### 20. [ROCm][Perf][MiniMax-M3] MonoKernel: serve EAGLE3 verify steps (#62994)
- **代码层面**：改的是 AMD 专用的 MiniMax-M3 测试和 decode 路径。让 MonoKernel（把多个小 kernel 融合成一个大 kernel 的技术）支持 EAGLE3 投机解码的 verify 步骤——即一次验证 1 个主 token + 3 个草稿 token，最多 4 个请求（16 行）融合。
- **新手概念课堂**：EAGLE3 投机解码是"先让一个小模型猜 3 个词，再让大模型一次性验证"。verify 步骤的 batch 形状很特殊（每请求 4 行）。MonoKernel 是"把零散小任务打包成一个大批次"的优化。让两者兼容 = 在 AMD 卡上也能享受投机解码的加速。
- **对你有什么影响**：在 AMD GPU 上跑 MiniMax-M3 + EAGLE3 投机解码时，decode 阶段性能提升，且测试覆盖了 1-4 个请求的 verify 场景，稳定性更有保障。

## sglang
## 💡 今日关键词

- **关键词：PD 分离（Prefill/Decode Disaggregation）调度精细化**
  - 一句话通俗解释：把"读题"（prefill）和"答题"（decode）拆到不同机器上干，现在还能控制它们交替的节奏。
  - 为什么社区在关注：今天有 4 个 commit（#44001、#45641、#45108、#45492）都在打磨 PD 分离的调度、KV 传输和资源释放，说明大模型推理已经从"能跑"进入"抠延迟、抠吞吐"的工程深水区。
  - 对新手意味着什么：分布式推理调度是当下最缺人的方向之一。学会看 scheduler 代码、理解 prefill/decode 交替，比只会调 API 值钱得多。

- **关键词：量化（INT8 / FP8 / MXFP4）成为默认选项**
  - 一句话通俗解释：把模型权重和计算从 16 位压到 8 位甚至 4 位，省显存、提速度，代价是精度。
  - 为什么社区在关注：#45706 移除 `kitchen_int8` 别名、#44179 调整 convrot_int8 的 GEMM 策略、#45629 的 NVFP4 部署，说明量化不再是"实验特性"，而是生产标配，且工具链在收敛。
  - 对新手意味着什么：搞懂量化格式（W8A8、MXFP8、NVFP4）和它们的数值误差，是部署岗的硬技能。

- **关键词：AMD / ROCm 与异构硬件适配**
  - 一句话通俗解释：除了 NVIDIA，AMD 的 MI355X、NPU 等硬件也在被认真支持。
  - 为什么社区在关注：#45689、#45693 是 AMD cookbook 更新，#42976 是 NPU 内存池修复，说明 SGLang 在主动做多硬件后端，避免被单一厂商锁死。
  - 对新手意味着什么：如果你手上只有 AMD 卡或国产 NPU，现在正是参与生态建设、积累差异化经验的窗口期。

- **关键词：Diffusion（扩散模型）推理服务化**
  - 一句话通俗解释：文生图/文生视频模型也开始像 LLM 一样被"服务化"部署，有量化、有调度、有 ComfyUI 集成。
  - 为什么社区在关注：#45530、#44841、#45705、#45267 全是 diffusion 相关，MiniMax-H3、FastH3、Cosmos 等模型在快速接入。
  - 对新手意味着什么：多模态生成是下一个爆发点，懂 diffusion 推理管线的人会越来越吃香。

- **关键词：性能微优化（kernel 级 / 内存级）**
  - 一句话通俗解释：把 logits 直接写进缓冲区、去掉一次 host 同步、避免读陈旧元数据——都是"省一次拷贝、少一次等待"。
  - 为什么社区在关注：#45631、#45492、#45699 都是这种"看不见但很值钱"的优化，说明框架已进入性能打磨期。
  - 对新手意味着什么：学会用 profiler 找热点、理解 GPU 同步开销，是从"会用"到"会调"的分水岭。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：PD 分离中的 prefill/decode 交替节奏控制**
  - **是什么**：在 PD 分离架构里，prefill 节点算完一段 KV 后，decode 节点要跑若干轮解码，再回来接下一批 prefill。`--prefill-decode-interval` 就是控制"跑几轮 decode 再插一次 prefill"的旋钮。
  - **为什么重要**：如果 decode 轮数太少，prefill 频繁打断，decode 的 token 间延迟（ITL）抖动大；轮数太多，新请求的 TTFT 变长。不懂这个参数，线上会出现"P99 延迟忽高忽低"却找不到原因。
  - **动手试试**：
    ```bash
    # 启动一个带 PD interval 的服务（单机模拟，参数按你的模型调整）
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-7B-Instruct \
      --prefill-decode-interval 8 \
      --prefill-decode-interval-min-bs 4 \
      --port 30000
    ```
    然后用两个终端并发压测：一个发长 prompt（触发 prefill），一个持续发短请求（触发 decode），观察日志里 prefill 批次出现的频率。
  - **预期结果**：你会看到当 running batch 小于 `min-bs`（4）时，prefill 每轮都被允许进来（快速填满 batch）；batch 变大后，prefill 变成每 8 轮 decode 才插一次。这就是 #44001 新增的"门控"行为。

- **知识点 2：量化后端与 GEMM 分块策略**
  - **是什么**：INT8 量化后，矩阵乘法（GEMM）怎么切块、用哪种 CUTLASS 配置，会显著影响速度。`SGLANG_KITCHEN_INT8_MAX_ROWS` 控制一次 GEMM 最多处理多少行。
  - **为什么重要**：默认值从 8192 改成 0（不分块）后，MiniMax-H3 在 4090/5090/H200 上快了 1.3–3.9%。这说明"看起来更聪明的分块"不一定更快——额外调用和输出拷贝有成本。不懂这个，你会盲目调参越调越慢。
  - **动手试试**：
    ```bash
    # 对比两种设置下的 diffusion 推理耗时
    SGLANG_KITCHEN_INT8_MAX_ROWS=8192 sglang generate \
      --model-path <your-minimax-h3-path> --quantization convrot_int8 \
      --prompt "a cat" --num-inference-steps 8 2>&1 | tail -5

    SGLANG_KITCHEN_INT8_MAX_ROWS=0 sglang generate \
      --model-path <your-minimax-h3-path> --quantization convrot_int8 \
      --prompt "a cat" --num-inference-steps 8 2>&1 | tail -5
    ```
  - **预期结果**：第二次（不分块）通常更快。你会直观理解"优化不是越多越好，要看硬件和形状"。

- **知识点 3：避免不必要的 host 同步（GPU 异步编程）**
  - **是什么**：GPU 是异步执行的，CPU 想让 GPU 干活后马上读结果，就得 `synchronize()` 等它。这个等待会阻塞调度线程。#45492 用 CUDA event 把"等索引拷贝完成"从调度线程挪到了传输 worker 线程。
  - **为什么重要**：调度线程一旦被同步卡住，整个 batch 的调度都停了。这是分布式推理里最隐蔽的性能杀手之一。
  - **动手试试**：
    ```python
    import torch, time
    x = torch.randn(4096, 4096, device="cuda")

    # 反例：每次循环都同步
    t0 = time.time()
    for _ in range(1000):
        y = x @ x
        torch.cuda.synchronize()
    print("sync every iter:", time.time() - t0)

    # 正例：批量提交后再同步
    t0 = time.time()
    for _ in range(1000):
        y = x @ x
    torch.cuda.synchronize()
    print("sync once:", time.time() - t0)
    ```
  - **预期结果**：第二种明显更快。你会明白为什么"少一次同步"能带来可观的端到端收益。

- **知识点 4：KV Cache 容量与 context length 的关系**
  - **是什么**：#42976 新增了一条警告：如果 KV pool 能装的 token 数少于 `--context-length`，那么一个满长度请求根本放不下。
  - **为什么重要**：新手常犯的坑——设了 `--context-length 32768` 却因为 `--mem-fraction-static` 太低，实际池子只有 20000 token，长请求被拒或截断，还以为是模型问题。
  - **动手试试**：
    ```bash
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-7B-Instruct \
      --context-length 32768 \
      --mem-fraction-static 0.5 \
      --port 30000 2>&1 | grep -i "KV pool"
    ```
  - **预期结果**：如果池子不够，你会看到类似 `KV pool holds N tokens, fewer than --context-length 32768` 的警告。调高 `--mem-fraction-static` 或降低 `--context-length` 后警告消失。

- **知识点 5：HiCache 存储后端的热插拔管理**
  - **是什么**：#44726 给 rust-server 加了 `/hicache/storage-backend` 的查询、挂载、卸载、清空接口，还能通过 gRPC 调用。
  - **为什么重要**：以前 HiCache 的 L3 存储后端只能在启动时配置，现在可以运行时切换——比如高峰期挂 SSD、低峰期卸载释放资源。这是"运维友好"的重要一步。
  - **动手试试**：
    ```bash
    # 查询当前 HiCache 存储后端状态
    curl http://localhost:30000/hicache/storage-backend

    # 运行时挂载一个存储后端
    curl -X POST http://localhost:30000/hicache/storage-backend \
      -H "Content-Type: application/json" \
      -d '{"hicache_storage_backend":"file","hicache_write_policy":"write_through"}'

    # 清空缓存条目
    curl -X POST http://localhost:30000/hicache/storage-backend/clear
    ```
  - **预期结果**：第一次返回当前配置快照（可能是 null），挂载后返回 `{"success": true, ...}`，再查询能看到新后端。你会理解"配置即 API"的现代服务设计。

---

## 🔥 重要更新

- **#44001 scheduler: Gate --prefill-decode-interval on running batch size**：新增 `--prefill-decode-interval-min-bs`，让 PD 交替节奏能根据当前 batch 大小自适应。这是 PD 分离调度从"固定节奏"走向"动态节奏"的关键一步，直接影响线上 TTFT 与 ITL 的权衡。
- **#45492 pd: Send prefill KV chunks without a host sync on the scheduler thread**：把 KV 索引的 host 同步从调度线程挪到传输 worker，用 CUDA event 做异步协调。这是典型的"消除调度线程阻塞"优化，对高并发 PD 部署的吞吐提升明显。
- **#45631 logits: Write lm_head logits straight into the fp32 logits buffer**：lm_head 的输出直接写进预分配的 fp32 缓冲区，省掉一次中间 tensor 分配和拷贝。每个 decode step 都受益，是"小改动大收益"的典范。
- **#45706 diffusion: remove the deprecated kitchen_int8 alias**：正式移除 `kitchen_int8` 别名，统一为 `convrot_int8`。如果你还在用旧名字，升级后会报错——这是一个需要立刻检查的破坏性变更。

---

## 📋 逐条解读

### 1. [Feat] scheduler: Gate --prefill-decode-interval on running batch size (#44001)
- **代码层面**：在 `schedule.py` 新增参数 `prefill_decode_interval_min_bs`（默认 0），在 `scheduler.py` 的 `init_chunked_prefill` 里读取它；`validation_hook.py` 加了非负校验；文档同步更新。
- **新手概念课堂**：想象餐厅厨房（prefill）和传菜员（decode）。如果店里只有 2 桌客人，传菜员跑 8 趟再回厨房拿菜太浪费——应该让厨房随时补菜先把桌子填满。`min-bs` 就是"客人少于 N 桌时，厨房不用等"。
- **对你有什么影响**：低负载时新请求的 TTFT 更短（batch 快速填满），高负载时 decode 的 ITL 更稳。默认 0 表示行为不变，需要显式开启。

### 2. [Feat] systemone: serve Kev pointer head decision models (#45413)
- **代码层面**：新增 cookbook 文档 `docs/cookbook/autoregressive/Kev/kev.mdx`，描述如何用 SGLang 部署 Kev 系列决策模型（0.8B–27B），走 `/v1/systemone` 接口，禁用 radix cache、chunked prefill 和 CUDA graph。
- **新手概念课堂**：普通 LLM 是"生成文字"，Kev 这种决策模型是"直接给答案的概率"——把 LM head 换成 pointer head，比较选项 token 的隐状态，输出每个选项的 logit。像考试的选择题答题卡，而不是写作文。
- **对你有什么影响**：如果你做客服分流、工单分类这类任务，现在可以用 SGLang 直接部署决策模型，不用再套一层 LLM 生成再解析。

### 3. [Doc] cookbook: Refresh SemiAnalysis AgentX launches with five open InferenceX PRs (#45749)
- **代码层面**：更新 `docs/src/snippets/agentx/Qwen/qwen3.5.jsx` 等生成文件，同步 SemiAnalysis InferenceX 的最新 PR 状态和 launch 命令，移除 SA 专属配置。
- **新手概念课堂**：这是"自动生成的文档"——上游 benchmark 仓库的提交被脚本同步到 SGLang 文档里，保证 cookbook 里的命令和真实测试一致。类似"CI 自动更新 README"。
- **对你有什么影响**：cookbook 里的部署命令更可信、更新鲜。普通用户感知不强，但照着抄命令的人会更少踩坑。

### 4. [Refactor] disaggregation: Share the flag all-reduce and host-destination predicate in the decode transfer queue (#45641)
- **代码层面**：在 `decode.py` 里把重复出现的 `self.enable_host_receive and decode_req.host_staged` 抽成 `_is_host_destination()` 方法；把 `_all_reduce_release_flags` 替换为共享的 `all_reduce_flags()` 工具函数。
- **新手概念课堂**：这是"消除重复代码"的重构。就像家里好几个房间都有同一个开关逻辑，干脆做成一个统一开关面板。`all_reduce` 是分布式里的"大家投票取最大值"，确保所有 rank 对"是否保留资源"达成一致。
- **对你有什么影响**：无行为变化，但代码更易维护。对贡献者来说，以后改这块逻辑只需改一处。

### 5. [Feat] diffusion: support FastH3 Omni (Ref2VA 8-step PDD) (#45530)
- **代码层面**：在 `minimax_h3.py` pipeline config 里新增 VSA 相关配置（tile size、ref keep rate），支持加载 `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` 模型，文档补充了 2×H200 的延迟数据。
- **新手概念课堂**：PDD（Parallel Decoding Distillation）是"蒸馏"的一种——把 32 步的扩散过程压缩成 8 步，像把 32 页的说明书浓缩成 8 页精华版，速度快 4 倍但质量略降。
- **对你有什么影响**：做视频生成时可以用更少步数、更快出片。文档给了实测延迟（480p 图生视频约 27 秒），方便你评估是否满足业务需求。

### 6. [Refactor] diffusion: remove the deprecated kitchen_int8 alias (#45706)
- **代码层面**：删除 `method_names.py` 里的 `canonical_quantization_method` 映射，`kitchen_int8` 不再被接受，文档里所有 `kitchen_int8` 改成 `convrot_int8`。
- **新手概念课堂**：这是"清理历史包袱"。就像手机充电口从 Micro-USB 统一到 Type-C，旧接口虽然还能用但迟早要淘汰。别名机制是为了平滑迁移，现在过渡期结束了。
- **对你有什么影响**：**破坏性变更**——如果你的启动脚本里还写着 `--quantization kitchen_int8`，升级后会报错，改成 `convrot_int8` 即可。

### 7. [Fix] sgl-router: Do not abort forwarded requests on reqwest's default 30 s TCP_USER_TIMEOUT (#45696)
- **代码层面**：在 `proxy/mod.rs` 的 `build_client` 里，Linux/Android/Fuchsia 平台显式调用 `.tcp_user_timeout(None)` 关闭 reqwest 默认的 30 秒 TCP 超时；新增一个 40 秒的回归测试。
- **新手概念课堂**：`TCP_USER_TIMEOUT` 是"对方多久没确认收到数据就放弃"。reqwest 默认 30 秒，但推理引擎可能因为忙于处理长 prompt 而 30 秒没读 socket，导致请求被误杀。就像快递员等 30 秒没人签收就把包裹退回——其实人家只是在上厕所。
- **对你有什么影响**：高并发长 prompt 场景下，转发请求不再被莫名中断。这是一个"平时无感、出问题时救命"的修复。

### 8. [Doc] [AMD] cookbook: Add Muse Spark Open MI355X AgentX recipe (#45689)
- **代码层面**：在 `MuseSparkOpen.mdx` 新增 MI355X 8 卡 TP8·EP8 部署命令，包含 DFlash V2 投机解码、HiCache 配置和环境变量。
- **新手概念课堂**：这是"照着抄就能跑"的部署配方。TP8·EP8 指 8 卡张量并行 + 8 卡专家并行；DFlash 是投机解码（用小模型猜、大模型验），HiCache 是分层 KV 缓存。
- **对你有什么影响**：有 AMD MI355X 的团队可以直接复用这套命令，省去调参时间。

### 9. [Feat] rust-server: Add HiCache storage management endpoints (#44726)
- **代码层面**：新增 `proto/sglang/api/v1/hicache.proto`，定义 Get/Attach/Detach/Clear 四个 RPC；`service.proto` 注册；Python 侧 `api_types.py` 加对应类型。
- **新手概念课堂**：HiCache 是"KV 缓存的分层存储"——L1 在 GPU、L2 在 CPU、L3 在 SSD/远端。以前 L3 只能启动时配，现在可以运行时挂载/卸载，像给电脑热插拔移动硬盘。
- **对你有什么影响**：运维可以在不重启服务的情况下切换存储后端，比如高峰期挂上大容量 SSD，低峰期卸载释放资源。

### 10. [Fix] dsv4: Never read attention metadata for idle SWA stores (#45699)
- **代码层面**：在 `deepseek_v4_backend.py` 和 `deepseek_v4_backend_hip_radix.py` 的 `get_swa_out_cache_loc` 里，把 idle 分支提前到读取 `self.forward_metadata` 之前，避免读到陈旧或未设置的元数据。
- **新手概念课堂**：idle 是"这个 rank 这轮没活干但要陪跑"的状态。以前它会去读元数据，但元数据可能是上一轮的残留——就像空转的工人去拿上一批的图纸，可能拿错。现在直接走 dummy 路径。
- **对你有什么影响**：修复了 DeepSeek V4 在 SWA（滑动窗口注意力）场景下 idle rank 可能写错位置的 bug，提升稳定性。

### 11. [Feat] diffusion: Add worker-side ComfyUI compatibility patches (#44841)
- **代码层面**：在 `ComfyUI_SGLDiffusion/core/generator.py` 里新增 `worker_patches` 参数，通过 `scheduler_profiles` 上下文管理器在 worker 侧应用兼容性补丁；`options.py` 增加校验。
- **新手概念课堂**：ComfyUI 是流行的可视化工作流工具，但它的某些行为（比如调度器配置）和 SGLang 默认不一致。`worker_patches` 就是"翻译层"，让两边能对上话。
- **对你有什么影响**：用 ComfyUI + SGLang 做 diffusion 工作流的用户，兼容性问题更少，不用手动改配置。

### 12. [Doc] [AMD] dsv41: Update amd cookbook 1021 (#45693)
- **代码层面**：更新 DeepSeek-V4_1 的 AMD cookbook，刷新 ROCm 镜像 tag 到 `20261021`，调整 TP2/TP4 并发建议，移除 `AITER_BF16_FP8_MOE_BOUND` 和 `SGLANG_HIP_DSV41_INDEX_K_PAGE8` 等过时环境变量。
- **新手概念课堂**：这是"配方更新"——随着 ROCm 镜像和 kernel 优化演进，之前的最佳实践可能过时。比如 monokernel 后端不再兼容某个索引布局，就得去掉那个变量。
- **对你有什么影响**：照旧文档跑 AMD 部署的人可能会遇到兼容性问题，更新后能跑通。

### 13. [Perf] logits: Write lm_head logits straight into the fp32 logits buffer (#45631)
- **代码层面**：`logits_processor.py` 的 `_compute_lm_head` 新增 `out` 参数，`torch.mm` 直接写入预分配的 `next_token_logits_buffer`；`_copy_logits_to_buffer` 相应调整避免重复拷贝。
- **新手概念课堂**：以前是"算出来放临时篮子，再倒进大桶"，现在是"直接往大桶里倒"。省一次分配和一次拷贝，每个 decode step 都受益。
- **对你有什么影响**：decode 吞吐有小幅提升，尤其是大词表模型（如 128K vocab）效果更明显。

### 14. [Perf] diffusion: convrot_int8 comfy_kitchen issues one GEMM per layer by default (#44179)
- **代码层面**：`convrot_int8_comfy_kitchen.py` 里 `_MAX_ROWS_PER_CALL` 默认值从 8192 改为 0（不分块），文档同步更新说明。
- **新手概念课堂**：之前认为"大 GEMM 分块能避免 Stream-K 调度开销"，但实测发现分块带来的额外调用和输出拷贝成本更高。这是一个"反直觉优化"的案例——理论最优不等于实测最优。
- **对你有什么影响**：MiniMax-H3 等 diffusion 模型在 4090/5090/H200 上默认快 1.3–3.9%，无需改任何配置。

### 15. [Doc] cookbook: Add Muse Spark Open GB300 Dynamo + SGLang deployments measured with AA AgentPerf (#45629)
- **代码层面**：在 `MuseSparkOpen.mdx` 新增 5 套 GB300 部署配方（从 concurrency 1 到 160），包含 Dynamo frontend、etcd、SGLang worker 的完整命令。
- **新手概念课堂**：Dynamo 是 NVIDIA 的推理编排层，负责路由和 KV 感知调度；SGLang 是执行引擎。两者组合就像"调度员 + 厨师"，调度员决定谁去哪口锅。
- **对你有什么影响**：有 GB300 的团队可以直接抄这套配方，覆盖从低延迟到高吞吐的完整区间。

### 16. [Fix] hisparse: Keep PD destination host pages until the decode KV release (#45108)
- **代码层面**：`decode.py` 的 `_release_request` 里新增 `hisparse_coordinator.request_finished()` 调用，把原来在 `pop_transferred` 里提前释放的逻辑挪到真正释放 KV 时。
- **新手概念课堂**：HiSparse 模式下，KV 传输的目标是 coordinator 管理的 host 页面。如果提前释放，传输可能写到已回收的内存——就像快递还没到，你先把收货地址拆了。
- **对你有什么影响**：修复了 HiSparse + PD 分离场景下可能的 use-after-free 或数据损坏，提升稳定性。

### 17. [Perf] pd: Send prefill KV chunks without a host sync on the scheduler thread (#45492)
- **代码层面**：`TransferKVChunk` 新增 `indices_event` 字段；`mooncake/conn.py` 的 `transfer_worker` 里等待 `indices_event.synchronize()` 并把 tensor 转 numpy；`add_transfer_request` 和 `send` 透传该参数。
- **新手概念课堂**：调度线程是"大脑"，不该停下来等 GPU。现在用 CUDA event 把"等索引拷贝完成"这件事交给传输 worker 线程去等，大脑继续调度下一批。
- **对你有什么影响**：高并发 PD 部署下，调度吞吐提升，prefill 节点不再被 KV 传输的同步拖慢。

### 18. [Doc] diffusion: document prompt enhancer tasks and link Cosmos recipes (#45705)
- **代码层面**：`models_with_pe.mdx` 新增 `tasks`、`protocol`、`protocol_source` 字段说明，给出按任务（image/image_edit/video）定制 system prompt 和采样参数的示例；`model-catalog.jsx` 链接 Cosmos 配方。
- **新手概念课堂**：Prompt enhancer 是"把用户的一句话扩写成详细提示词"的中间模型。现在可以按任务类型给不同的扩写指令——视频扩写要保留运动和时间，图片扩写要保留主体。
- **对你有什么影响**：做 diffusion 应用时，可以用同一个 enhancer 服务处理多种任务，配置更灵活。

### 19. [Fix] [NPU] mla-pool: DCP allocator page and index-K bound, short-pool warning (#42976)
- **代码层面**：`memory_pool_npu.py` 的 `set_index_k_buffer` 加 OOB 检测；`tp_worker.py` 在 KV 池小于 context length 时打警告；`allocation.py` 把 NPU 加入 DCP 分支；`kv_cache_configurator.py` 适配 NPU allocator。
- **新手概念课堂**：DCP（Decode Context Parallel）下，allocator 的 page size 和物理 page size 可能不同。NPU 之前漏了这个分支，导致索引越界。新增的警告则是"体检报告"——池子装不下满长度请求时提前告诉你。
- **对你有什么影响**：NPU 用户部署 MLA 模型（如 DeepSeek）时更稳定，且能提前发现显存配置问题。

### 20. [Doc] diffusion/minimax-h3: Refresh VDN-H3 B200 benchmarks, fix 8-step bench presets and popular-models snippet (#45267)
- **代码层面**：更新 `MiniMax-H3.mdx` 的 VDN-H3 B200 基准表（SGLang 8 卡从 0.88 降到 0.85 s/NFE），修正 benchmark 脚本里 `num_inference_steps` 从 9 改为 8。
- **新手概念课堂**：`num_inference_steps` 多设 1 步会多跑一次完整 forward，基准数据就偏了。这是"benchmark 也要 code review"的典型案例。
- **对你有什么影响**：文档里的性能数据更准确，照着复现时不会因为步数不对而困惑。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)