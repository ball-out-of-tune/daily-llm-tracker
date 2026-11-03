# 🚀 vLLM & SGLang 每日更新 — 2026-11-03

> 自动生成于 2026-11-03 08:31 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：KV Offload（KV 缓存卸载）**
  - 一句话通俗解释：把 GPU 显存里放不下的 KV 缓存（注意力计算的历史记录）搬到 CPU 内存或磁盘上，需要时再拉回来，从而用更少显存跑更长的上下文。
  - 为什么社区在关注：今天有 2 个 commit（#64993、#64558）都在修 KV Offload 的调度和计数逻辑，说明这个功能已经从"能用"进入"抠细节正确性"的阶段，是长上下文推理的关键基础设施。
  - 对新手意味着什么：如果你在学推理优化，KV 缓存管理是绕不开的核心话题，理解"存哪里、什么时候存、怎么计数"是入门长上下文服务的敲门砖。

- **关键词：新硬件适配（SM110 / XPU / ROCm）**
  - 一句话通俗解释：让同一套代码在 NVIDIA 新架构（如 SM110）、Intel GPU（XPU）、AMD GPU（ROCm）上都能正确跑起来。
  - 为什么社区在关注：今天有 4 个 commit 涉及 SM110 内核修复、Intel CI 调整、ROCm AITER MoE 重构，说明 vLLM 正在全力摆脱"只支持 NVIDIA 主流卡"的标签，异构计算成为刚需。
  - 对新手意味着什么：只会写 CUDA 已经不够了，理解 Triton、跨平台抽象层（platform 接口）会让你在求职市场上更有竞争力。

- **关键词：MoE 内核优化（TRTLLM / AITER）**
  - 一句话通俗解释：混合专家模型（MoE）里"路由到不同专家"的计算非常复杂，专用内核能让它跑得更快。
  - 为什么社区在关注：今天有 2 个 commit（#65152、#63204）分别给 TRTLLM 内核加新激活函数、重构 AMD AITER 专家类，说明 MoE 已成为大模型主流架构，内核优化是性能主战场。
  - 对新手意味着什么：MoE 是当前最热门的模型架构之一，学会看懂 MoE 的 prepare/apply/finalize 三段式结构，就能读懂大部分推理框架的核心代码。

- **关键词：量化格式演进（NVFP4 + CSF 压缩）**
  - 一句话通俗解释：把模型权重从 16 位浮点压到 4 位浮点（NVFP4），甚至把量化用的缩放系数再压缩一遍（CSF），省显存又省带宽。
  - 为什么社区在关注：commit #64312 支持 CSF 压缩的 NVFP4 块缩放，说明 4 位量化正在从"能用"走向"极致压缩"，这是消费级显卡跑大模型的希望所在。
  - 对新手意味着什么：量化是性价比最高的推理加速手段，理解 FP8/FP4/块缩放的概念，能让你快速上手部署大模型。

- **关键词：投机解码 + KV 连接器联动**
  - 一句话通俗解释：投机解码（用一个小模型先猜几个 token）和 KV 连接器（跨实例复用缓存）结合时，需要提前计算"未来"的块哈希。
  - 为什么社区在关注：commit #64096 支持 lookahead block hashes，说明高级特性之间开始互相"打通"，这是框架成熟度的标志。
  - 对新手意味着什么：单点优化已经做得差不多了，下一个机会在于"多个优化技术如何协同"，这是进阶方向。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：KV 缓存卸载的"阈值计数"逻辑（来自 Commit 2）

- **是什么**：KV Offload 管理器有一个 `store_threshold` 参数，意思是"同一个 key 被至少 N 个不同请求看到过，才值得存到 CPU"。今天修的是：同一个请求重试时不能重复计数。
- **为什么重要**：如果计数错了，会导致该存的没存（浪费复用机会）或不该存的存了（浪费 CPU 内存）。这是典型的"状态机 bug"，不懂就会在长上下文场景下看到莫名其妙的缓存命中率下降。
- **动手试试**：
```python
# 简化版逻辑演示：为什么"每个请求只计一次"很重要
from collections import defaultdict

class SimpleStoreManager:
    def __init__(self, threshold):
        self.threshold = threshold
        self.counts = defaultdict(int)
        self.seen_by_request = defaultdict(set)  # key -> set of request ids

    def offer(self, keys, request_id):
        to_store = []
        for k in keys:
            if request_id in self.seen_by_request[k]:
                continue  # 关键：同一请求重试不重复计数
            self.seen_by_request[k].add(request_id)
            self.counts[k] += 1
            if self.counts[k] >= self.threshold:
                to_store.append(k)
        return to_store

m = SimpleStoreManager(threshold=2)
print(m.offer(["a", "b"], "req1"))  # []
print(m.offer(["a", "b"], "req1"))  # [] 重试，仍不计数
print(m.offer(["a", "b"], "req2"))  # ['a', 'b'] 第二个请求触发
```
- **预期结果**：你会看到重试的请求不会让计数虚增，只有真正不同的请求才会推动 key 达到阈值。

### 知识点 2：SSE 流式响应的解析陷阱（来自 Commit 18）

- **是什么**：SSE（Server-Sent Events）是流式输出的标准协议，格式是 `data: {...}\n\n`。但一个 TCP chunk 可能只包含半个事件，也可能包含 SSE 注释行（以 `:` 开头）或其他字段（如 `event: message`）。
- **为什么重要**：解析错了会导致 benchmark 客户端丢 token、卡死或统计错误。这是所有流式 API 客户端都会踩的坑。
- **动手试试**：
```python
# 演示：为什么不能"按 chunk 直接 json.loads"
chunks = [
    b'data: {"tok',
    b'en": "hello"}\n\n',
    b': this is a comment\n\n',
    b'event: message\ndata: {"token": "world"}\n\n',
]

buffer = b""
for c in chunks:
    buffer += c
    while b"\n\n" in buffer:
        event, buffer = buffer.split(b"\n\n", 1)
        lines = event.decode().splitlines()
        # 关键：只取 data: 行，跳过注释和其他字段
        data = next((l[6:] for l in lines if l.startswith("data: ")), None)
        if data:
            print("解析到:", data)
```
- **预期结果**：你会看到跨 chunk 的 JSON 被正确拼接，注释行被跳过，`event:` 字段被忽略，只输出两个 token。

### 知识点 3：种子（seed）的取值范围为什么是 [0, 2^32-1]（来自 Commit 13）

- **是什么**：Python 的 `random` 和 NumPy 的 `np.random.seed` 只接受 0 到 2^32-1 的整数。vLLM 用 Pydantic 的 `Field(ge=0, le=2**32-1)` 在配置层就拦截非法值。
- **为什么重要**：如果传了 `-1` 或 `2**32`，会在初始化后期某个深层调用里才报错，报错信息还看不懂。提前校验能让用户立刻知道问题在哪。
- **动手试试**：
```python
from pydantic import BaseModel, Field, ValidationError

class ModelConfig(BaseModel):
    seed: int = Field(default=0, ge=0, le=2**32 - 1)

for s in [0, 42, 2**32 - 1, -1, 2**32]:
    try:
        ModelConfig(seed=s)
        print(f"seed={s}: OK")
    except ValidationError as e:
        print(f"seed={s}: 被拦截 -> {e.errors()[0]['msg']}")
```
- **预期结果**：前三个通过，后两个被拦截并给出清晰的错误信息，而不是运行时崩溃。

### 知识点 4：Chat 模板不能"套娃"（来自 Commit 19）

- **是什么**：`openai-chat` 这类后端，服务端会自己把 messages 套上 chat template。如果客户端 benchmark 工具也套一遍，模型收到的是"双重模板"的 prompt，输出质量会崩。
- **为什么重要**：这是 benchmark 结果不可信的隐形杀手——你测出来的吞吐/质量，跟真实用户看到的完全不一样。
- **动手试试**：
```python
# 用 transformers 的 tokenizer 演示"双重模板"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [{"role": "user", "content": "你好"}]

once = tok.apply_chat_template(messages, tokenize=False)
twice = tok.apply_chat_template(
    [{"role": "user", "content": once}], tokenize=False
)
print("=== 套一次 ===")
print(once)
print("=== 套两次（错误示范）===")
print(twice)
```
- **预期结果**：你会看到第二次套模板后，特殊 token（如 `<|im_start|>`）出现了两遍，模型会困惑。

### 知识点 5：为什么"每个 config 独立"比"改环境变量"更安全（来自 Commit 17）

- **是什么**：以前 `VLLM_USE_BREAKABLE_CUDAGRAPH` 是进程级环境变量，一个模型设置了会影响同进程里所有模型。现在改成挂在 `compilation_config` 上，每个 config 独立。
- **为什么重要**：在测试套件或一个进程跑多个模型的场景下，环境变量"泄漏"会导致难以复现的诡异 bug——A 测试的配置影响了 B 测试的结果。
- **动手试试**：
```python
# 演示环境变量泄漏问题
import os
from types import SimpleNamespace

class OldStyle:
    @staticmethod
    def enable():
        os.environ["MY_FLAG"] = "1"
    @staticmethod
    def is_enabled():
        return os.environ.get("MY_FLAG") == "1"

class NewStyle:
    def __init__(self):
        self.flag = False
    def enable(self):
        self.flag = True

# 旧方式：A 开启了，B 无辜受牵连
a, b = OldStyle(), OldStyle()
a.enable()
print("旧方式 B 是否被影响:", b.is_enabled())  # True，泄漏！

# 新方式：各管各的
a, b = NewStyle(), NewStyle()
a.enable()
print("新方式 B 是否被影响:", b.flag)  # False，干净
```
- **预期结果**：你会直观看到环境变量的"污染"效应，以及 per-config 状态的隔离优势。

---

## 🔥 重要更新

- **#64558 KV Offload 计数 bug 修复**：如果同一个请求重试 `prepare_store`，会重复计入 `store_threshold`，导致本不该存储的 key 被提前存入或跳过统计。这是影响缓存命中率的正确性 bug，长上下文用户值得关注。
- **#64096 投机解码 + KV 连接器联动**：支持 lookahead block hashes，让投机解码场景下 KV 连接器能提前知道"未来会用到哪些块"，从而更好地预取。这是两个高级特性首次深度打通。
- **#64312 NVFP4 + CSF 压缩**：ModelOpt 量化新增 CSF 压缩的块缩放支持，进一步降低 4 位量化模型的显存占用，对消费级显卡部署大模型是好消息。
- **#65127 seed 范围校验**：之前传非法 seed 会在深层报错，现在初始化阶段就明确拦截。虽然小，但能省下大量排查时间。

---

## 📋 逐条解读

### 1. [Test][KV Offload] Update stale partial-tail scheduler expectations (#64993)

- **代码层面**：只改测试文件 `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py`。把过时的断言更新为符合新行为的期望值：`recurrent_event.block_size` 从 0 改成 4，`block_hashes` 从长度 1 改成具体等于某个 hash，`parent_block_hash` 从 None 改成上一个块的 hash。还引入了 `maybe_convert_block_hash` 辅助函数。
- **新手概念课堂**：这就像考试答案更新了——代码行为早就改了，但测试还在用旧答案，导致 CI 红灯。维护者需要把"标准答案"改成"新标准答案"。`block_size` 从 0 到 4 意味着"尾部不完整的块"现在按实际大小 4 来处理，而不是当作空块。
- **对你有什么影响**：普通用户无感知，但如果你在跑 vLLM 的测试套件，这条修复能让你不再看到无关的红灯。

### 2. [Bugfix][KV Offload] Count each key once per request toward store_threshold (#64558)

- **代码层面**：修改 CPU offload 管理器的计数逻辑。核心是给每个 key 记录"哪些请求已经 offer 过它"，同一请求重试时不再增加计数。测试里新增了 `test_store_threshold_does_not_count_retries_of_a_failed_offer`，验证 `prepare_store` 返回 None 后的重试不会虚增计数。
- **新手概念课堂**：想象一个"投票"系统，规则是"至少 2 个不同的人投票才采纳"。如果同一个人投了两次票都被算数，那规则就失效了。这里修复的就是"同一个人重复投票"的问题。
- **对你有什么影响**：使用 KV Offload 跑长上下文时，缓存命中率会更稳定，不会因为调度器重试而出现意外的"该存没存"或"不该存却存了"。

### 3. [Bugfix][Kernel] Fix cooperative top-k dispatch on SM110 (#56102)

- **代码层面**：在 `vllm/model_executor/layers/indexer_topk.py` 和 `qsa_indexer.py` 中，给"协作式 top-k 内核"的启用条件加上 `not is_device_capability_family(110)`。因为 SM110 架构不支持这个协作式内核，之前误判为支持会导致崩溃或错误结果。
- **新手概念课堂**：SM110 是 NVIDIA 新一代 GPU 架构（如 Blackwell 系列）。"协作式内核"指多个线程块协同完成一个 top-k 计算，需要特定硬件支持。这就像某些游戏只支持特定显卡型号，强行在旧卡上跑会花屏。
- **对你有什么影响**：如果你在 SM110 显卡上跑 DeepSeek-V4 等使用 top-k 索引器的模型，之前可能崩溃，现在会自动回退到兼容实现。

### 4. [Kernel] Enable Step5 SwiGLU in TRTLLM MoE (#65152)

- **代码层面**：在 `trtllm_bf16_moe.py` 中，给 `TrtLlmBf16Experts` 增加对 `MoEActivation.SWIGLUSTEP` 的支持。新增 `gemm1_activation_params` 字典，把 `gemm1_clamp_limit` 作为张量传给 FlashInfer 内核。同时更新 `moe_kernel_features.md` 文档，标注 TRTLLM 内核支持 `swiglustep` 激活。
- **新手概念课堂**：SwiGLU 是 MoE 里常用的门控激活函数，`SWIGLUSTEP` 是它的"阶跃变体"（clamp 限幅）。就像做菜时的"加盐"和"加盐+限盐"——后者控制咸度上限。TRTLLM 是 NVIDIA TensorRT-LLM 的缩写，是高性能推理内核库。
- **对你有什么影响**：使用 Step5 模型 + TRTLLM MoE 内核的用户，现在可以用上模型原生的激活函数，输出质量更接近原始模型。

### 5. [Refactor][DSv4.1] Remove duplicated code (#64974)

- **代码层面**：清理 DeepSeek-V4.1 模型代码里的重复逻辑。删除了 `_hf_config_int` 和 `_scheduler_config_int` 两个重复的辅助函数，直接访问 `hf_config.sliding_window` 和 `hf_config.index_topk`。还把 `_fill_short_context_topk_indices` 改名为 `fill_short_context_topk_indices`（去掉私有前缀，因为测试要用）。
- **新手概念课堂**：这是典型的"代码卫生"——同一段逻辑写了两遍，改一处忘另一处就会出 bug。重构就是"合并同类项"，让代码更易维护。
- **对你有什么影响**：普通用户无感知，但如果你在二次开发 DeepSeek-V4.1，代码会更清晰。

### 6. [EPD] Support DeepSeek-V4.1 image metadata-only inputs (#62802)

- **代码层面**：新增 `test_v41_image_metadata_only_roundtrip` 测试，验证在 EPD（Encoder-Prefill-Decode 分离）架构下，图片只需传元数据（`llm_grid` 网格信息）而不需要真正预处理图片。涉及 `PlaceholderMetadataResolver`、`collect_ec_item_metadata` 等组件。
- **新手概念课堂**：EPD 分离架构把"图像编码"和"文本生成"拆到不同实例。元数据传递就像快递单号——不需要真的把包裹搬来搬去，只要知道"包裹在哪、多大"就能调度。
- **对你有什么影响**：部署 DeepSeek-V4.1 多模态模型时，EPD 架构下可以省去重复的图像预处理，降低延迟。

### 7. [MM] Support mm_device_do_normalize for gemma3 (#64487)

- **代码层面**：在 `mm_processing.md` 文档里加入 gemma-3 支持；在 `test_mm_input_norm.py` 里新增 NCHW 布局的参考实现（`_reference_input_norm_nchw`）和 `InputLayout` 类型校验。核心是让 Gemma3 的图像归一化可以在 GPU 上融合执行。
- **新手概念课堂**：图像归一化通常是"减均值、除标准差"，在 CPU 上做很慢。`mm_device_do_normalize` 让这一步在 GPU 上做，就像"切菜在厨房切"而不是"切好再端到客厅"。
- **对你有什么影响**：跑 Gemma3 多模态模型时，图像预处理更快，整体延迟降低。

### 8. [Perf][Model Loader] Fault in mmap'd weight pages before host-to-device copies on integrated GPUs (#58868)

- **代码层面**：在 `vllm/model_executor/utils.py` 的 `copy_weight_` 函数中，对集成 GPU（如 Intel 核显、Apple Silicon）增加"预触碰"（prefault）逻辑：在 H2D 拷贝前先遍历 mmap 的权重页，触发缺页中断，把页加载到内存。
- **新手概念课堂**：集成 GPU 和 CPU 共享内存，H2D 拷贝其实是内存拷贝。如果权重文件是 mmap 的，第一次访问会触发"缺页中断"（从磁盘读页），这个开销如果落在拷贝过程中会很慢。提前"摸一遍"页，就像搬家前先把箱子都拆开检查一遍，而不是搬到一半才发现有东西要拆。
- **对你有什么影响**：在集成 GPU 上加载大模型会更快，尤其是首次加载。

### 9. [XPU][CI]Increase memory for Quantization test and deselect timeout case in v1/shutdown (#65097)

- **代码层面**：Buildkite CI 配置调整：Intel XPU 的量化测试内存从 16GB 加到 32GB；`v1/shutdown` 测试里 deselect 一个会超时的用例。
- **新手概念课堂**：这是"基础设施维护"——测试跑挂了不一定是代码 bug，可能只是机器内存不够或某个用例本身超时。就像实验室的电脑该升级了。
- **对你有什么影响**：无直接影响，但说明 Intel XPU 的 CI 在持续稳定化。

### 10. [XPU][Test] Use current platform device in test_compile_ranges (#63641)

- **代码层面**：`tests/compile/test_compile_ranges.py` 里把硬编码的 `torch.set_default_device("cuda")` 改成 `torch.set_default_device(current_platform.device_type)`。同时在 Intel CI 配置里加入这个测试。
- **新手概念课堂**：硬编码 "cuda" 就像"只认一种插座"，换成 Intel GPU 就插不上了。用 `current_platform.device_type` 就是"万能插座"，自动适配当前硬件。
- **对你有什么影响**：在非 NVIDIA 硬件上跑编译相关测试不再直接失败。

### 11. [Quantization] Support CSF-compressed NVFP4 block scales in ModelOpt (#64312)

- **代码层面**：在 `modelopt.py` 里新增 `weight_scale_encoding` 参数，支持 `CSF_SCALE_ENCODINGS` 中定义的编码格式。新增 `csf_weights_transform` 工具函数，在加载权重时把压缩的 scale 流解压。还加了 `get_checkpoint_weight_mapper` 来配对 CSF scale 流。
- **新手概念课堂**：NVFP4 是 4 位浮点量化，每个小块（16 个元素）共享一个 scale（缩放系数）。CSF 是"压缩 scale 格式"——把 scale 本身也压缩，进一步省显存。就像"压缩包里的压缩包"。
- **对你有什么影响**：使用 ModelOpt 量化的 NVFP4 模型，显存占用更低，能跑更大的模型或更长上下文。

### 12. [Bugfix][Step5] Size the KV block to the CSA tail ring (#65153)

- **代码层面**：新增 `test_csa_tail_ring_block_size.py` 测试，验证 Step5 模型的 CSA（Compressed Sparse Attention）尾部环形缓冲区的 block size 计算。核心是让 KV block size 匹配"尾部环形缓冲区"的大小，避免越界或浪费。
- **新手概念课堂**：CSA 的"尾部环形缓冲区"像录音机的循环录音带——只保留最近 N 秒。KV block 大小必须和这个环的大小对齐，否则要么装不下、要么浪费。就像买鞋要合脚。
- **对你有什么影响**：跑 Step5 模型时，KV 缓存分配更精确，不会因为 block size 不对齐而浪费显存或崩溃。

### 13. [Bugfix] Validate global model seed range (#65127)

- **代码层面**：`vllm/config/model.py` 里 `seed` 字段从 `int = 0` 改成 `Field(default=0, ge=0, le=2**32 - 1)`。新增测试验证 `seed=-1` 和 `seed=2**32` 会在初始化时抛出 ValueError。
- **新手概念课堂**：Pydantic 的 `Field(ge=..., le=...)` 是"配置层的守门员"，在数据进入系统前就检查合法性。就像机场安检——不合格的行李在门口就被拦下，而不是登机后才发现。
- **对你有什么影响**：传非法 seed 会立刻得到清晰报错，而不是在深层调用里看到莫名其妙的 NumPy 错误。

### 14. [Refactor][ROCm][MoE] Let each AiterExperts subclass own its AITER call args (#63204)

- **代码层面**：重构 AMD ROCm 的 AITER MoE 专家类。以前 `AiterFp8ChannelExperts.apply` 里把 topk=1 的 dispatch 逻辑藏在通用 AITER 专家里，现在让每个子类自己决定调用 `asm_moe_tkw1` 还是 `fused_moe`。新增测试 `test_aiter_fp8_channel_experts_owns_tkw1_dispatch` 验证 dispatch 正确。
- **新手概念课堂**：这是"职责分离"原则——每个类只管自己的事。以前是"大杂烩"函数，现在拆成"专人专岗"。`tkw1` 是 top-k=1 的专用快速路径，`fused_moe` 是通用路径。
- **对你有什么影响**：AMD GPU 上跑 MoE 模型时，topk=1 场景会走专用快速路径，性能更好；代码也更易维护。

### 15. [Bugfix][Multimodal] Pad MiniMax-M3 encoder graph cu_seqlens with zero-length sequences (#65226)

- **代码层面**：`encoder_cudagraph.py` 里，`pad_cu_seqlens` 函数从"用总 token 数填充尾部"改成"用最后一个真实 offset 填充尾部"（零长度序列）。删除了 `_encoder_cg_pad_totals` 字典和 `max_batch_size + 1` 的额外槽位。
- **新手概念课堂**：`cu_seqlens` 是变长注意力里的"累积序列长度"数组，告诉内核每个序列从哪开始到哪结束。如果尾部填充用"总长度"，内核会认为有一个超长序列；用"最后一个 offset"则等于零长度序列，内核直接跳过。就像排队时末尾放几个"空位"而不是"一个巨人"。
- **对你有什么影响**：MiniMax-M3 多模态模型在 CUDA Graph 编码器场景下不再返回 NaN，输出正常。

### 16. [KV Connector][Spec Decode] Support lookahead block hashes in KV connectors (#64096)

- **代码层面**：给 `VllmConfig` 增加 `num_hash_lookahead_tokens` 属性，让 KV 连接器在投机解码场景下能提前计算"未来会用到"的块哈希。修改了 Kimi-K3 KDA metadata builder 和多个 KV 连接器，测试里用 `num_hash_lookahead_tokens` 参数替代 monkeypatch。
- **新手概念课堂**：投机解码会一次猜多个 token，KV 连接器需要提前知道"这些猜的 token 会命中哪些缓存块"。lookahead hash 就是"预判未来"——像下棋时提前想好几步。
- **对你有什么影响**：同时使用投机解码和 KV 连接器（如 LMCache）时，缓存命中率更高，延迟更低。

### 17. [Bugfix][CI/Build] Use compilation config to control breakable cuda graph (#65091)

- **代码层面**：把 `VLLM_USE_BREAKABLE_CUDAGRAPH` 环境变量的控制逻辑迁移到 `CompilationConfig.use_breakable_cudagraph` 字段。删除了 `tests/conftest.py` 里为防环境变量泄漏而设的 monkeypatch。新增测试 `test_breakable_cudagraph_resolution_is_per_config` 验证每个 config 独立。
- **新手概念课堂**：环境变量是"全局开关"，一开全开。per-config 字段是"每个房间独立的开关"。前者会导致测试互相污染，后者干净隔离。就像办公室空调——中央空调一开全楼冷，独立空调各调各的。
- **对你有什么影响**：在一个进程里跑多个模型配置时，不会因为某个模型开了 breakable cudagraph 而影响其他模型。

### 18. [Bugfix][Benchmark] Preserve Rust SSE payloads across chunks and comments (#60177)

- **代码层面**：Rust benchmark 的 `openai_chat.rs` 里，删除了 `trim_bytes` 和"按 chunk 直接解析"的逻辑，改为把原始 chunk 交给 `StreamedResponseHandler` 缓冲，按 `\n\n` 分割事件后再提取 `data: ` 行。新增 `spawn_sse_server_with_body` 测试辅助函数。
- **新手概念课堂**：TCP 是"字节流"，不保证 chunk 边界和 SSE 事件边界对齐。一个事件可能被拆成两个 chunk，也可能一个 chunk 里有多个事件。正确做法是"先缓冲，再按分隔符切"。就像收快递——不能每来一个箱子就拆，要等所有箱子到齐再按单号分拣。
- **对你有什么影响**：用 Rust benchmark 工具测流式 API 时，不会再丢 token 或解析失败，统计数据更准确。

### 19. [Bugfix][Benchmark] Skip the client-side chat template on chat backends (#65245)

- **代码层面**：`datasets.py` 里新增 `CHAT_TEMPLATE_BACKENDS = ("openai-chat", "openai-responses", "vllm-chat")`，`get_samples` 里如果 backend 在这个列表里，自动跳过客户端 chat template。新增测试验证 chat 后端跳过、completion 后端保留。
- **新手概念课堂**：chat 后端（如 OpenAI Chat API）的服务端会自己套模板，客户端再套就是"双重模板"。就像你去餐厅点菜，服务员已经帮你翻译成厨房用语了，你再翻译一遍厨师就懵了。
- **对你有什么影响**：用 benchmark 工具测 chat 接口时，测出来的结果和真实用户看到的一致，不再因为双重模板而失真。

### 20. [Perf][DSpark][DSv4.1] Fuse the Markov sampling loop on NVIDIA by inverse CDF (#64442)

- **代码层面**：新增 `tests/models/deepseek_v41/nvidia/test_fused_markov.py`，验证 `FusedMarkovSampler` 和 `DSparkSpeculator` 的循环采样结果一致（greedy token 相同、采样 token 符合 softmax 分布）。核心是把 Markov 链的逐步采样循环融合成一个内核，用逆 CDF（inverse CDF）方法一次性采样。
- **新手概念课堂**：Markov 采样是"根据当前 token 预测下一个 token"的循环，每步都要算 softmax + 采样。融合内核把多步循环压成一次 GPU 调用，减少 kernel launch 开销。逆 CDF 采样是"按累积概率分布反查"——像抽奖时按概率区间找奖品。
- **对你有什么影响**：DeepSeek-V4.1 的 DSpark 投机解码在 NVIDIA GPU 上采样更快，端到端延迟降低。

## sglang
## 💡 今日关键词

- **关键词：异构计算（AMD / NVIDIA 双线适配）**
  - 一句话通俗解释：同一份代码要能在 NVIDIA、AMD 等不同厂商的 GPU 上都跑得快、跑得对。
  - 为什么社区在关注：今天有 3 个 commit 专门针对 AMD（gfx115x 权重填充、gfx950 确定性推理允许 AITER 解码、AITER 解码注意力），说明 SGLang 正在把 AMD 当作一等公民对待，而不是"能跑就行"。
  - 对新手意味着什么：如果你只会 CUDA，建议补一下 HIP / ROCm 的基本概念；AMD 生态的算子优化岗位和开源贡献机会正在变多，门槛比 CUDA 低。

- **关键词：确定性推理（Deterministic Inference）**
  - 一句话通俗解释：同样的输入、同样的批大小，每次跑出来的结果逐比特一致，方便复现和调试。
  - 为什么社区在关注：大模型推理的"不可复现"是 RL 训练、评测、bug 定位的大痛点。今天有 commit 放宽了确定性模式下可用的注意力后端（AITER 解码），说明社区在"确定性"和"性能"之间找平衡点。
  - 对新手意味着什么：这是理解 GPU 算子"为什么结果会飘"的绝佳切入点——归约顺序、批大小、split-K 都会影响浮点累加顺序。

- **关键词：导入图瘦身（Import Graph / 启动性能）**
  - 一句话通俗解释：Python 模块 import 时会执行代码，import 链越深启动越慢、内存越大；今天有多个 commit 在拆解 import 依赖。
  - 为什么社区在关注：SGLang 越来越大，"import 一个调度器却把整个 tokenizer 拉进来"这种问题会拖慢冷启动、增加多进程内存。今天有 3 个 commit（#48087、#47990、#43907）都在做这件事。
  - 对新手意味着什么：学会用 `python -X importtime` 和 `importlib` 分析依赖，是 Python 工程能力里很实用但常被忽视的技能。

- **关键词：融合算子（Kernel Fusion）**
  - 一句话通俗解释：把好几个小算子（卷积、激活、GLU）合成一个大 kernel，减少显存读写和启动开销。
  - 为什么社区在关注：今天有 SANA-Video 的 depthwise conv + GLU 融合、Qwen3 的 QK-norm/RoPE 表查融合、CuTe DSL NVFP4 MoE 支持 SiTU 激活。融合是推理性能优化最直接的杠杆。
  - 对新手意味着什么：Triton 是入门融合算子的最佳语言——比 CUDA 简单，又能直接写 GPU kernel，今天有好几个 commit 就是纯 Triton。

- **关键词：MoE 专家并行与权重布局（EPLB / Locality）**
  - 一句话通俗解释：MoE 模型有很多"专家"，要把它们分布到多卡上，还要在动态负载均衡时搬来搬去。
  - 为什么社区在关注：DeepGEMM Mega MoE 权重按 locality domain 本地化、EPLB 更新时处理非连续权重，说明 MoE 部署已经从"能跑"进入"精细调优"阶段。
  - 对新手意味着什么：MoE 是当前大模型的主流架构（DeepSeek、Kimi、Qwen-MoE），理解专家并行是进入推理框架核心的必经之路。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：Triton 融合算子——把"卷积 + GLU"写成一个 kernel**
  - **是什么**：GLU（Gated Linear Unit）是 `a * sigmoid(b)` 这种门控结构；depthwise 卷积是每个通道独立做 3×3 卷积。传统做法是"卷积 kernel → 写回显存 → 读出来做 GLU → 再写回"，融合后一次读写完成。
  - **为什么重要**：视频扩散模型（SANA-Video）里这类算子每帧都要跑，显存带宽是瓶颈。不融合会多两次全量显存往返，实测能差 20-30% 延迟。
  - **动手试试**：用一个最小 Triton kernel 体验"融合"的写法（需要 NVIDIA GPU + `pip install triton torch`）：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def fused_glu_kernel(X, Out, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(X + N + offs, mask=mask, other=0.0).to(tl.float32)
        # GLU: a * sigmoid(b)
        out = a * (1.0 / (1.0 + tl.exp(-b)))
        tl.store(Out + offs, out, mask=mask)

    N = 1024 * 1024
    x = torch.randn(2 * N, device="cuda", dtype=torch.float32)
    out = torch.empty(N, device="cuda", dtype=torch.float32)
    BLOCK = 1024
    fused_glu_kernel[(triton.cdiv(N, BLOCK),)](x, out, N=N, BLOCK=BLOCK)
    ref = x[:N] * torch.sigmoid(x[N:])
    print("max diff:", (out - ref).abs().max().item())
    ```
  - **预期结果**：输出 `max diff` 在 1e-6 量级。然后你可以用 `torch.profiler` 对比"分开写"和"融合写"的 kernel 数量和显存读写量，体会融合的价值。

- **知识点 2：权重行填充（Weight Row Padding）对抗缓存别名**
  - **是什么**：GPU 的 L2 缓存按"通道/bank"划分，如果矩阵每一行的字节数是 2KB 的整数倍，多行会落到同一个通道上，造成"缓存别名"（cache aliasing），带宽利用率骤降。填充 128 字节让行错开。
  - **为什么重要**：AMD gfx115x 上这个现象特别明显（commit #46327）。这是一个"不改数学、只改内存布局"的纯性能优化，输出完全不变——非常典型的 GPU 调优手法。
  - **动手试试**：在本地 CPU 上模拟"步长影响访问模式"的概念（不需要 GPU）：
    ```python
    import numpy as np, time

    N = 4096
    # 不填充：行字节数 = 4096 * 2 = 8192 = 4 * 2048，容易别名
    a = np.random.randn(N, N).astype(np.float16)
    # 填充 64 个元素（128 字节）后切片回来，逻辑形状不变但 stride 变了
    padded = np.zeros((N, N + 64), dtype=np.float16)
    padded[:, :N] = a
    b = padded[:, :N]

    print("a strides:", a.strides, "b strides:", b.strides)
    t0 = time.perf_counter(); _ = (a.T @ a).sum(); t1 = time.perf_counter()
    t2 = time.perf_counter(); _ = (b.T @ b).sum(); t3 = time.perf_counter()
    print(f"contiguous {t1-t0:.4f}s, padded-view {t3-t2:.4f}s")
    ```
  - **预期结果**：你会看到 `b.strides` 的第二维不是 `N*2` 而是 `(N+64)*2`，而计算结果的数值完全一致。在 GPU 上，这种 stride 差异会直接体现在带宽和延迟上——这就是 commit #46327 的全部精髓。

- **知识点 3：Python import 图与启动性能分析**
  - **是什么**：Python 的 `import` 是一条链，A import B，B import C…… 任何一环被拉进来都会执行模块级代码。`schedule_batch.py` 里放一个 `FINISH_ABORT` 常量，就会让所有 import 它的人把整个调度器拉进来。
  - **为什么重要**：commit #48087 把常量拆到 `req_types.py`，commit #47990 移除 re-export 垫片。这能显著降低冷启动时间和多进程内存（每个 worker 进程都要 import）。
  - **动手试试**：用标准库工具分析任意模块的 import 时间：
    ```bash
    # 看每个模块的 import 耗时（单位微秒）
    python -X importtime -c "import json" 2>&1 | sort -t'|' -k2 -n | tail -20
    ```
    再写一个小脚本，检测"循环 import"和"重导出"：
    ```python
    # check_imports.py
    import ast, sys, pathlib

    def find_reexports(path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    # 形如 from .x import Y 且 Y 在本文件未被使用，可能是垫片
                    if alias.asname is None:
                        yield node.module, alias.name

    for p in pathlib.Path(".").rglob("*.py"):
        for mod, name in find_reexports(p):
            if mod and "req_types" in mod:
                print(p, "->", mod, name)
    ```
  - **预期结果**：`importtime` 输出按耗时排序的模块列表，你能一眼看到哪些模块"拖后腿"；第二个脚本能帮你找出代码库里"从 A 导入但从 B 定义"的垫片，正是 #47990 在清理的东西。

- **知识点 4：确定性推理与"批不变性"（Batch Invariance）**
  - **是什么**：同一个请求，不管它和多少其他请求一起跑，输出都应逐比特相同。难点在于 GPU 的归约（求和）顺序会随批大小变化，浮点加法不满足结合律，结果就会飘。
  - **为什么重要**：commit #44489 讨论 AITER 解码注意力"按固定 256 token 分区归约"所以是批不变的，但 MLA 和投机解码路径不是。这就是在推理框架里做"确定性"审计的典型思路。
  - **动手试试**：用 PyTorch 演示"归约顺序影响结果"：
    ```python
    import torch
    torch.manual_seed(0)
    x = torch.randn(100000, device="cuda", dtype=torch.float32)

    # 不同归约顺序
    s1 = x.sum()
    s2 = x.view(100, 1000).sum(dim=0).sum()
    s3 = x.view(1000, 100).sum(dim=1).sum()
    print(f"{s1.item():.10f}\n{s2.item():.10f}\n{s3.item():.10f}")
    print("all equal?", s1 == s2 == s3)
    ```
  - **预期结果**：三个和通常不完全相等（差在 1e-3 量级）。这就是为什么"批大小改变 → 归约树改变 → 结果改变"。确定性推理要做的，就是让归约树与批大小无关。

- **知识点 5：MoE 权重布局与 P2P 传输的"稠密性"要求**
  - **是什么**：MoE 的专家权重在 EPLB（专家负载均衡）重排时要在 GPU 间 P2P 传输。如果权重的内存布局是"带洞的"（非连续 stride），就不能一次性整块发送，必须先拷到连续缓冲区。
  - **为什么重要**：commit #47976 给 DeepGEMM Mega MoE 权重做了"跨 locality domain 本地化"，导致布局非连续；commit #4709a3a2 的 `_is_dense` / `_int64_view` 就是为这种布局做适配。不懂这个，EPLB 更新会静默变慢或报错。
  - **动手试试**：用 PyTorch 检查张量是否"稠密"，并观察 view 的 stride：
    ```python
    import torch

    def is_dense(t):
        expected = 1
        for stride, size in sorted(zip(t.stride(), t.shape)):
            if size == 1: continue
            if stride != expected: return False
            expected *= size
        return True

    a = torch.randn(4, 8)
    b = torch.randn(4, 16)[:, :8]   # 切片，stride 非连续
    c = torch.zeros(4, 12)[:, :8]   # 带填充
    for name, t in [("a", a), ("b", b), ("c", c)]:
        print(name, "stride:", t.stride(), "is_dense:", is_dense(t))
    ```
  - **预期结果**：`a` 是稠密（stride `(8,1)`），`b`/`c` 不是（stride 是 `(16,1)` 或 `(12,1)`）。你就理解了为什么 P2P 发送前需要 `torch.empty_like(..., memory_format=torch.contiguous_format)` 再 `copy_`。

---

## 🔥 重要更新

- **#48087 / #47990 / #43907（导入图与调度器重构）**：三个 PR 联手拆解 SGLang 的 import 依赖和调度器接口，直接影响冷启动时间、多进程内存占用和代码可维护性，是"看不见但人人受益"的基础设施级改动。
- **#46327 / #44489（AMD 性能与确定性）**：AMD 平台同时拿到"权重行填充"和"确定性模式下允许 AITER 解码"，说明 AMD 的推理性能正在被认真对待，不再是二等公民。
- **#4709a3a2（MoE 权重本地化 + EPLB 适配）**：DeepGEMM Mega MoE 支持跨 locality domain 本地化权重，并配套处理了 P2P 传输的非连续布局，是 MoE 大规模部署的关键一步。
- **#47261（DCP 下禁用融合 RoPE KV 写入）**：一个典型的"性能优化在特定并行模式下不正确"的修复，提醒大家融合算子要考虑所有并行维度（这里是 DCP = Decode Context Parallel）。

---

## 📋 逐条解读

### 1. [Fix] Anthropic: accept thinking.display="updates"
- **代码层面**：改 `anthropic/protocol.py` 和 `anthropic/serving.py`。把 `AnthropicThinkingParam.display` 的 `Literal` 从 `["summarized", "omitted"]` 扩到 `["summarized", "omitted", "updates"]`，并简化了 serving 层的注释和日志逻辑（`updates` 和 `omitted` 一样，只打 warning 不真正抑制推理文本）。
- **新手概念课堂**：Anthropic 的 API 里 `thinking` 参数控制"扩展思考"（extended thinking）。`display` 决定思考过程对客户端可见多少——`summarized` 给摘要，`omitted` 不给，`updates` 是新增的"逐步更新"模式。SGLang 作为兼容层，需要"接受但可能不实现"某些字段，这叫"协议兼容性垫片"。
- **对你有什么影响**：如果你用 Anthropic SDK 调用 SGLang 的兼容端点，现在传 `display="updates"` 不会再被 400 拒绝，但思考文本仍会照常输出（日志里有 warning）。

### 2. [Perf] imports: Keep the detokenizer off TokenizerManager and the tokenizer off schedule_batch
- **代码层面**：把 `FINISH_ABORT`、`FINISH_LENGTH` 等常量从 `schedule_batch.py` 挪到新的 `req_types.py`，然后 beam_search、disaggregation 等模块改为从 `req_types` 导入常量、从 `schedule_batch` 导入 `Req`/`ScheduleBatch` 类。核心是把"常量定义"和"重型类定义"的 import 图切开。
- **新手概念课堂**：想象一个图书馆，`schedule_batch.py` 是"总馆"，里面有借书卡格式（常量）和整栋楼（类）。以前借一张卡要进总馆，现在卡单独放在"卡片亭"（`req_types.py`），不用进总馆。这就是 import 图瘦身。
- **对你有什么影响**：冷启动更快、每个 worker 进程内存更省。用户感知不明显，但大规模部署时启动时间能省几秒到几十秒。

### 3. [Perf] [AMD] unquant: Pad gfx115x linear and lm_head weight rows to avoid cache aliasing
- **代码层面**：在 `unquant.py` 新增 `maybe_pad_weight_for_gfx115()`，当权重是 2D、连续、bf16/fp16、且行字节数是 2048 的整数倍时，用 `F.pad` 在行尾加 128 字节再切片回来，得到一个"逻辑形状不变、物理 stride 改变"的视图。加环境变量 `SGLANG_DISABLE_GFX115_WEIGHT_PADDING` 做开关。
- **新手概念课堂**：GPU 的 L2 缓存像一排停车位，如果每辆车都停在同一个位置（行步长是 2KB 整数倍），就会互相挤占。加 128 字节填充相当于"错开一个车位"，大家都能停。
- **对你有什么影响**：在 AMD gfx1151 等显卡上跑非量化模型，线性层和 lm_head 会更快，输出完全不变。NVIDIA 用户无影响。

### 4. [Fix] test: Import ResolvedView from model_override_base in test_model_overrides
- **代码层面**：测试文件里 `ResolvedView` 的 import 来源从 `arg_groups.overrides` 改为 `arg_groups.model_override_base`，`_deterministic_attention_backend` 仍从 `overrides` 导入。配合 #47990 的"从定义处导入"重构。
- **新手概念课堂**：这是"re-export 垫片"的清理。以前 `overrides.py` 里 `from ... import ResolvedView` 只是为了方便别人导入，现在要求大家直接从定义处导入，避免 import 链变长。
- **对你有什么影响**：只影响开发者。如果你写测试或插件，应该从 `model_override_base` 导入 `ResolvedView`。

### 5. [Doc] diffusion: Set measured cookbook defaults for MiniMax-H3, FastH3 Omni, MOSS-Image and Qwen-Image-2.1
- **代码层面**：纯文档改动，更新 cookbook 里的启动命令、性能表格和配置片段。比如 FastH3 Omni 去掉了默认的 CPU offload（在 141GB/180GB 卡上常驻更快），Qwen-Image-2.1 的 B200 行改用 cuDNN SDPA 并更新延迟数据。
- **新手概念课堂**：cookbook 是"菜谱"，告诉你某个模型在某张卡上怎么启动、大概多快。CPU offload 是把部分权重放到内存、用时再搬回显存，省显存但慢。
- **对你有什么影响**：如果你照 cookbook 部署这些扩散模型，能直接拿到实测最优配置，少走弯路。

### 6. [Refactor] scheduler: Share one interface across request lists and abort through each list
- **代码层面**：新增 `_RequestList` 和 `ListKind`，把 `grammar_queue` 等请求列表统一成一个接口（支持 `join`、`matches` 等），abort 逻辑改为通过每个列表自己的 `_abort` 方法分发，而不是各写各的循环。
- **新手概念课堂**：以前调度器里有好几个"请求队列"（grammar 队列、waiting 队列、running 队列），每个队列的 abort 代码都要手写一遍。现在抽象出一个"队列接口"，像 USB 接口一样，插什么设备都能用。
- **对你有什么影响**：内部重构，用户无感知。但代码更不容易出 bug（比如某个队列漏了 abort 处理）。

### 7. [Perf] [NVIDIA] qwen3: Reuse the table QK-norm/RoPE kernel for MRoPE decode
- **代码层面**：扩展 `table_qk_norm_rope.py` 的 Triton kernel，增加 `axis_map_ptr`、`pos_stride`、`MROPE` 等参数，让原本用于普通 RoPE 的"查表"kernel 也能处理 MRoPE（多模态 RoPE，不同轴用不同位置）。同时把 `HEADS_PER_PROGRAM` 变成可配置，一次处理多个 head。
- **新手概念课堂**：RoPE（旋转位置编码）用 cos/sin 表旋转 Q/K。普通文本只有一个位置轴；多模态模型有"时间、行、列"三个轴（MRoPE），每个轴查不同的表。这个 commit 让同一个 kernel 支持两种模式，避免维护两份代码。
- **对你有什么影响**：Qwen3 系列多模态模型在 decode 阶段更快，尤其是 MRoPE 场景。

### 8. [Perf] [AMD] deterministic: Allow AITER decode attention on gfx950
- **代码层面**：`overrides.py` 里 `_deterministic_attention_backend` 现在允许 `decode_attention_backend="aiter"`，但仅限 gfx950、非 MLA 模型、无投机解码。`aiter_backend.py` 里增加检查：确定性模式下只允许 `paged_attention_ragged` 路径，禁用 unified-attention 和 5D-pool kernel。
- **新手概念课堂**：确定性推理要求"批不变"——AITER 的 paged decode 按固定 256 token 分区归约，所以批大小变了结果也不变；但 MLA 和投机解码路径按批大小分 KV，就不满足。这是"审计每个 kernel 的归约模式"的典型案例。
- **对你有什么影响**：AMD gfx950 用户在确定性模式下可以用 AITER 解码，性能更好；其他情况仍会报错提示。

### 9. [Perf] diffusion: Fuse SANA-Video depthwise convolution and GLU on SM120
- **代码层面**：新增 `sana_depthwise_glu_triton.py`，一个 Triton kernel 同时做 channels-last bf16 depthwise 3×3 卷积、bias 舍入和 GLU 门控。用 `round_bf16_to_fp32` 保证数值和参考实现一致。
- **新手概念课堂**：融合 kernel = 把"读输入→卷积→写中间→读中间→GLU→写输出"压缩成"读输入→卷积+GLU→写输出"，省两次全量显存往返。视频扩散每帧都要跑，省下来很可观。
- **对你有什么影响**：SANA-Video 在 SM120（Blackwell）上生成更快，输出数值不变。

### 10. [Doc] cookbook: Add MiniMax-M3.1-Flash GB300 MonoKernel and DEP16 AgentX points
- **代码层面**：纯文档/JSX 配置更新，给 MiniMax-M3.1-Flash 增加 GB300 MonoKernel 和 DEP16 AgentX 的实测点，调整 `item` 编号和 PR 引用。
- **新手概念课堂**：AgentX 是 SemiAnalysis 的推理基准，cookbook 里的 JSX 是"数据驱动文档"——配置和数据分离，渲染时生成表格。
- **对你有什么影响**：查 MiniMax-M3.1-Flash 部署数据时有更多实测参考。

### 11. [Perf] [NVIDIA] moe: Localize DeepGEMM Mega MoE weights across locality domains
- **代码层面**：新增环境变量 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE`；`expert_location_updater.py` 增加 `_is_dense` 和 `_int64_view`，在 EPLB 更新时把非连续权重先拷到连续缓冲区（能用 int64 view 就零拷贝），再 P2P 发送。
- **新手概念课堂**：MoE 的专家权重像图书馆的书，EPLB 是"重新分配书架"。如果书是"散页"（非连续布局），不能整摞搬，要先装订成册（拷贝到连续 buffer）。`_int64_view` 是"用更大的箱子装"来满足 8 字节对齐。
- **对你有什么影响**：DeepGEMM Mega MoE 在 EPLB 重排时更快、更不容易出错；不开这个开关则行为不变。

### 12. Revert "[Perf] kimi-k3: Seed a validated FlashInfer prefill MoE tactic"
- **代码层面**：删除 `kimi_k3_sm103_fi072_prefill.json` 和 `flashinfer_kimi_k3_tactics.py`，回滚 `flashinfer_autotune.py` 里加载预置 tactic 的逻辑。
- **新手概念课堂**：FlashInfer 的 AutoTuner 会为每种 shape 选最优 kernel 配置（tactic）。"预置 tactic"是把实测好的配置写死，跳过 autotune。回滚说明这个预置在真实场景下有问题（可能是版本不匹配或性能回退）。
- **对你有什么影响**：Kimi-K3 在 B30Z 上 prefill 会回到 autotune 流程，首次启动稍慢但更稳。

### 13. Revert "[Feat] diffusion/model: add the BRIA Fibo 2.0 transformer and text conditioning"
- **代码层面**：删除 `bria_fibo2.py` 等新增文件，回滚整个 BRIA Fibo 2.0 模型支持。
- **新手概念课堂**：模型适配 PR 被回滚通常是因为数值不一致、显存超限或依赖问题。这是开源协作的正常节奏——先合入再回滚，比一直挂着强。
- **对你有什么影响**：暂时不能用 SGLang 跑 BRIA Fibo 2.0，等修复后重新合入。

### 14. [Doc] cookbook: Refresh SemiAnalysis AgentX launches to InferenceX main c5e8b9b3 and revert #47560
- **代码层面**：纯文档，把 AgentX 数据同步到 InferenceX 仓库的最新 main，回滚之前的 B300 配置，更新 PR 引用和生成日期。
- **新手概念课堂**：cookbook 的数据是"生成"的，来自上游基准仓库。同步脚本把上游数据转成 JSX，避免手抄出错。
- **对你有什么影响**：AgentX 性能数据更新到最新，参考价值更高。

### 15. [Doc] qwen: Use TRT-LLM MHA for the B300 Qwen3-VL-4B recipe
- **代码层面**：文档和 JSX 配置里，B300 上 Qwen3-VL-4B-Instruct（BF16、TP1、无投机解码）的 `--attention-backend` 从 `flashinfer` 改为 `trtllm_mha`，因为这是 SGLang 在 B300 上的默认选择。
- **新手概念课堂**：注意力后端是"用哪个库算 attention"——FlashInfer、TRT-LLM MHA、FlashAttention 各有擅长。B300（Blackwell）上 TRT-LLM MHA 通常更快。
- **对你有什么影响**：照 cookbook 部署 Qwen3-VL-4B 能拿到更好性能；如果你手动写了 `--attention-backend flashinfer`，建议改掉。

### 16. [Fix] attention: Keep fused RoPE KV writes off DCP layouts
- **代码层面**：`enable_fused_set_kv_buffer` 增加检查：如果 `dcp_enabled`（Decode Context Parallel 开启），直接返回 `False`，禁用融合 RoPE KV 写入。测试新增 `TestDcpFusedStoreGate` 覆盖 dcp_size 1/2/4。
- **新手概念课堂**：融合 RoPE KV 写入 = 在旋转 Q/K 的同时把 K 写进 KV cache，省一次读写。但 DCP 模式下每个 rank 只负责部分 KV slot，融合 kernel 不知道"哪些 slot 归我"，会写错位置。
- **对你有什么影响**：开启 DCP 时不再用融合写入，正确性优先；不开 DCP 则性能不变。

### 17. [Refactor] imports: Import names from where they are defined; remove re-export shims
- **代码层面**：大规模 import 路径调整——比如 `sglang.bench_serving` 改为 `sglang.benchmark.serving`，`nsa/nsa_indexer.py` 改为 `dsa/dsa_indexer.py`，删除各处 re-export 垫片，文档和模板同步更新。
- **新手概念课堂**：re-export 垫片 = "A 文件里写 `from B import X`，只为让别人从 A 导入 X"。短期方便，长期让 import 图变乱、循环依赖难查。这个 PR 要求"从定义处导入"。
- **对你有什么影响**：如果你写了脚本 import `sglang.bench_serving`，需要改成 `sglang.benchmark.serving`；旧路径可能还能用但已弃用。

### 18. [Fix] scheduler: Tell a step's middle chunk from the batch's chunked request, not a per-request counter
- **代码层面**：删除 `Req.inflight_middle_chunks` 计数器，改用 `req is batch.chunked_req` 和 `req.inflight_steps` 判断"是否还在分块"。`prefill.py`、`dllm/mixin/scheduler.py`、`schedule_batch.py` 同步修改。
- **新手概念课堂**：chunked prefill 把长 prompt 切成多块逐步处理。以前用"每请求一个计数器"记录"还有几块没跑完"，但计数器在 overlap 调度、retract 等场景下容易算错。改用"它是不是当前 chunked_req"和"还有几步在飞"更可靠。
- **对你有什么影响**：长 prompt 分块预填在复杂调度场景下更不容易出现"请求卡住"或"提前释放"的 bug。

### 19. [Feat] moe: Allow SiTU on the CuTe DSL NVFP4 MoE (Kimi-K3 W4A16)
- **代码层面**：`flashinfer_cutedsl.py` 里激活类型映射增加 `"situ"`（映射到 `Swiglu` + `situ_beta`/`situ_linear_beta` 参数），断言和错误信息同步扩展；`modelopt_quant.py` 允许 `_is_cutedsl_v2_standard` 路径使用 SiTU。
- **新手概念课堂**：SiTU 是 SwiGLU 的变体，多两个 beta 参数控制门控形状。Kimi-K3 用它。CuTe DSL 是 NVIDIA 的 kernel 编写 DSL，FlashInfer 用它实现 NVFP4 MoE。
- **对你有什么影响**：Kimi-K3（W4A16 NVFP4）在 CuTe DSL 后端上能用原生 SiTU 激活，不用回退到慢路径。

### 20. [Refactor] rust-server: Make the advertised listener host an overridable classmethod
- **代码层面**：`rust_server/server.py` 新增 `_advertised_host()` 类方法，把"绑定地址是 0.0.0.0 时用本机 IP 代替"的逻辑抽出来，`gather_server_endpoints` 改为调用它。子类可覆盖以适配自定义服务发现。
- **新手概念课堂**：多机部署时，每个 rank 要告诉别人"我在哪"。如果绑定的是 `0.0.0.0`（所有网卡），别人不知道具体 IP，所以要换成实际网卡地址。抽成类方法后，特殊部署（比如 K8s Service）可以覆盖。
- **对你有什么影响**：普通用户无感知；自定义部署（如 K8s、服务网格）可以通过继承覆盖 `_advertised_host` 来发布正确的服务地址。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)