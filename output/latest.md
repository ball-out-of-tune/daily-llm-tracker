# 🚀 vLLM & SGLang 每日更新 — 2026-11-12

> 自动生成于 2026-11-12 08:48 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：XPU / 异构后端适配**
  - **一句话解释**：Intel XPU（GPU）平台上的测试和注意力后端选择逻辑被修复和重新启用，让 vLLM 能在 Intel GPU 上更稳定地跑。
  - **为什么社区在关注**：AMD、Intel、NVIDIA 三家 GPU 都在被 vLLM 主动适配，说明推理框架正在从"NVIDIA 专属"走向"多硬件后端"，异构计算不再是口号。
  - **对新手意味着什么**：如果你只会写 CUDA，现在补一手 ROCm / SYCL / XPU 的编程模型，会在未来几年非常吃香；哪怕只是理解"同一份 Python 代码如何在不同后端间切换"，也是加分项。

- **关键词：KV Cache 卸载与分层（Tiering / Offload）**
  - **一句话解释**：把 KV cache 从 GPU 显存"溢出"到 CPU 内存甚至更远的存储，并用共享内存区域在多个 worker 之间共享，从而支持更长上下文、更多并发。
  - **为什么社区在关注**：大模型上下文越来越长（百万 token），显存根本放不下，KV offload 是当前最热的系统级优化方向之一。
  - **对新手意味着什么**：这是"系统 + 推理"交叉的黄金地带。懂一点操作系统内存映射（mmap）、共享内存、NUMA 的人，在推理框架团队里非常抢手。

- **关键词：结构化输出 / Grammar 约束解码**
  - **一句话解释**：强制模型输出符合 JSON Schema、正则、EBNF 或 Lark 语法，保证下游程序能直接解析。
  - **为什么社区在关注**：Agent、函数调用、工具调用场景爆发，结构化输出从"锦上添花"变成"刚需"。今天有两个 commit 专门升级 xgrammar 并修复 Lark 语法处理。
  - **对新手意味着什么**：想进 Agent 方向，一定要理解 constrained decoding（约束解码）的原理：它不是后处理，而是在采样时就把非法 token 的概率置零。

- **关键词：编译与内存管理细节（torch.compile / CUDA allocator）**
  - **一句话解释**：PyTorch 的编译缓存和 CUDA 内存分配器的参数，会直接影响显存占用和对象生命周期。
  - **为什么社区在关注**：今天有 commit 修复"编译后的模型因字节码钩子未释放导致显存泄漏"，还有 commit 把 `max_split_size_mb` 从 20 调到 21 来复用碎片——这些都是生产环境真实踩过的坑。
  - **对新手意味着什么**：性能优化不只是写快 kernel，理解 allocator 行为、对象生命周期、缓存失效，才是资深工程师的分水岭。

- **关键词：Rust 前端 / 流式解析**
  - **一句话解释**：vLLM 正在把 tokenizer、工具调用解析器等前端逻辑用 Rust 重写，以获得更高吞吐和更低延迟。
  - **为什么社区在关注**：Python GIL 在流式场景下是瓶颈，Rust 重写前端是明显的性能趋势（今天有 3 个 Rust 相关 commit）。
  - **对新手意味着什么**：会 Rust 的 AI 工程师目前是稀缺资源；哪怕不写 Rust，理解"为什么前端要换语言"也能让你在架构讨论中有话语权。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：KV Cache 分层卸载与共享内存区域（mmap + populate_shard）

- **是什么**：当 GPU 显存不够放 KV cache 时，可以把一部分 KV 放到 CPU 内存里，甚至让多个 worker 进程共享同一块 mmap 内存区域（shared region），每个 worker 只"预触碰"（pre-fault）自己负责的那一段。
- **为什么重要**：不理解 mmap 的按需分页（demand paging）机制，你可能会遇到"进程启动时卡住几十秒"或"内存被重复占用"的问题。Commit 2 就是在优化：调度器不再对整个区域做 pre-fault，而是按 shard 只触碰自己那份。
- **动手试试**：
```python
# 演示：共享 mmap 区域 + 只触碰自己分片
import mmap, os, tempfile

path = tempfile.mktemp()
with open(path, "wb") as f:
    f.write(b"\x00" * (8 * 1024 * 1024))  # 8 MiB

with open(path, "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    rank, world_size = 1, 4
    shard_size = len(mm) // world_size
    start = rank * shard_size
    # 只触碰自己的分片
    for off in range(start, start + shard_size, 4096):
        _ = mm[off]
    print(f"rank {rank} touched bytes [{start}, {start+shard_size})")
    mm.close()
os.unlink(path)
```
- **预期结果**：你会看到每个 rank 只触碰自己的区间，进程 RSS 只增长约 2 MiB（而不是 8 MiB）。这说明"按 shard 预触碰"避免了所有 worker 都去 fault 整个区域。

---

### 知识点 2：结构化输出中的 Grammar 语法区分（EBNF vs Lark）

- **是什么**：约束解码支持多种语法描述格式。今天两个 commit 让 vLLM 能正确区分"这是 XGrammar EBNF 语法"还是"这是 Lark 语法"，并把语法类型一起传给后端，而不是简单粗暴地猜测。
- **为什么重要**：如果你把一个 Lark 语法误当成 EBNF 传给 xgrammar，会得到解析错误或错误约束，导致模型输出不符合预期。Commit 9/10 就是在修这个"语法类型丢失"的 bug。
- **动手试试**：
```python
# 用 xgrammar 做最小约束解码演示（需 pip install xgrammar）
import xgrammar as xgr
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
info = xgr.TokenizerInfo.from_huggingface(tokenizer)

# 一个简单的 JSON Schema 约束
schema = '{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}'
compiler = xgr.GrammarCompiler(info)
compiled = compiler.compile_json_schema(schema)

matcher = xgr.GrammarMatcher(compiled)
# 模拟采样：每一步只允许合法 token
print("grammar compiled OK, matcher created")
```
- **预期结果**：能成功编译 schema 并创建 matcher。若把 Lark 语法错误地传给 `compile_grammar`（EBNF 接口），会抛异常——这就是今天 commit 要修的"类型混淆"问题。

---

### 知识点 3：CUDA 内存分配器的 max_split_size_mb 参数

- **是什么**：PyTorch 的 caching allocator 会把大块显存切成小块复用。`max_split_size_mb` 控制"多大的块才允许被切分"。Commit 6 发现设为 20 时，1–10 MiB 的 tensor 释放后所在的 20 MiB 段"过大"而无法复用，最终撑爆显存；改成 21 后就能复用。
- **为什么重要**：这类 bug 极其隐蔽——训练/推理能跑，但显存缓慢增长直到 OOM。理解 allocator 的 split 策略，能帮你诊断"看起来没泄漏但显存一直涨"的问题。
- **动手试试**：
```python
import torch

# 查看当前 allocator 设置
print("backend:", torch.cuda.memory.get_allocator_backend())
print("settings:", torch._C._accelerator_getAllocatorSettings())

# 模拟 commit 6 的场景：反复分配/释放 1-10 MiB tensor
torch.cuda.empty_cache()
before = torch.cuda.memory_reserved()
for i in range(100):
    t = torch.empty((1 + i % 9) * 1024 * 1024 + 4096, dtype=torch.uint8, device="cuda")
    del t
after = torch.cuda.memory_reserved()
print(f"reserved grew by {(after - before) / 1024 / 1024:.1f} MiB")
```
- **预期结果**：在默认设置下增长可能较大；如果你手动设置 `max_split_size_mb:21`（通过 `torch._C._accelerator_setAllocatorSettings`），增长会被限制在 20 MiB 左右。这就是 commit 6 的核心。

---

### 知识点 4：torch.compile 的字节码钩子与对象生命周期

- **是什么**：`torch._dynamo.convert_frame.register_bytecode_hook` 会注册全局钩子，钩子闭包持有对编译模块的引用。如果清理时不 `handle.remove()`，模型和权重永远不会被 GC。Commit 14 修复了"子类也带 `@support_torch_compile` 导致注册两次钩子"的泄漏。
- **为什么重要**：在多模型加载/卸载、in-process 反复创建 LLM 的场景（如评测框架、多租户服务）中，这种泄漏会直接导致 OOM。
- **动手试试**：
```python
import torch
from torch._dynamo.convert_frame import _bytecode_hooks

print("initial hooks:", len(_bytecode_hooks))

# 注册一个钩子
handle = torch._dynamo.convert_frame.register_bytecode_hook(lambda *a, **k: None)
print("after register:", len(_bytecode_hooks))

handle.remove()
print("after remove:", len(_bytecode_hooks))
```
- **预期结果**：能看到钩子数量 0 → 1 → 0。如果你在 vLLM 中加载 Qwen3（继承 Qwen2，二者都 `@support_torch_compile`），修复前会看到 2 个钩子且清理后仍残留 1 个——这正是 commit 14 修的 bug。

---

### 知识点 5：Rust 流式解析器中的"标记匹配"

- **是什么**：MiniMax M3 的工具调用用特殊 token 作为分隔标记（如 `<]minimax[>[<`）。Commit 20 让解析器按"token 身份"而非"字符串拼写"匹配标记——只有当专用 token 真的产生了这个标记时才算数，普通 token 拼出同样字符串则视为普通文本。
- **为什么重要**：纯字符串匹配会被"模型恰好生成了相同字符"欺骗，导致误解析工具调用。按 token id 匹配是更鲁棒的做法。
- **动手试试**：
```python
# 演示：token 身份 vs 字符串拼写的区别
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

special = "<|im_start|>"
ids = tok.encode(special, add_special_tokens=False)
print("token ids for special string:", ids)
print("decoded back:", tok.decode(ids))

# 找 token id 是否存在
tid = tok.convert_tokens_to_ids(special)
print("convert_tokens_to_ids:", tid, "unk?", tid == tok.unk_token_id)
```
- **预期结果**：能看到特殊 token 有独立 id，而普通文本拼出的相同字符串会被切成多个 token。这正是 Rust 解析器要区分的情况。

---

## 🔥 重要更新

- **Commit 6（max_split_size_mb: 20 → 21）**：一个数字的改动，修复了生产环境中"释放的小 tensor 无法复用、显存持续增长"的隐蔽 bug，所有用 PyTorch native allocator 的 CUDA/ROCm 用户都会受益。
- **Commit 14（清理编译字节码钩子）**：修复 `torch.compile` 全局钩子导致的模型/权重泄漏，对频繁加载卸载模型的场景（评测、多租户）影响巨大。
- **Commit 11（tokenizer 池化）**：HF fast tokenizer 不是线程安全的，并发截断会串请求；这个 commit 用池化保证每请求隔离，是线上服务正确性修复。
- **Commit 9 + 10（xgrammar 0.2.9 + Lark 语法组合）**：结构化输出是 Agent 的地基，这两个 commit 让 Lark 语法也能走快速路径，直接影响工具调用可靠性。

---

## 📋 逐条解读

### 1. [XPU][CI] Re-enable `test_reset_prefix_cache_e2e` and `test_scheduler_e2e` on XPU (#62656)

- **代码层面**：改了 `.buildkite/intel_jobs/test-intel.yaml`（去掉 `--ignore` 两个测试文件）和 `vllm/platforms/xpu.py`。在 XPU 平台新增 `_get_flash_attn_path` 方法，统一处理"选 Flash Attention 后端"的逻辑：先检查 block_size 是否被 FlashAttentionBackend 支持，不支持就抛 `ValueError`，支持但不是 64 的倍数就打性能警告。原来 4 处重复的 `return AttentionBackendEnum.FLASH_ATTN.get_path()` 都改成调用这个新方法。
- **新手概念课堂**：把"选择注意力后端"想象成"点外卖时选配送方式"。以前代码里到处写"就用闪送"，现在统一到一个"调度台"，先检查你的地址（block_size）闪送能不能送，不能就报错，能但偏远就提醒你慢一点。
- **对你有什么影响**：Intel GPU 用户现在能跑到之前被禁用的两个 e2e 测试，说明 XPU 支持更完整了；如果你传了 Flash Attention 不支持的 block_size，会在启动时立刻报清晰错误，而不是等到 KV cache 初始化时才崩。

### 2. [Perf][KV Offload] Split the shared Tiering region and skip the scheduler pre-fault (#66391)

- **代码层面**：主要改 `vllm/v1/kv_offload/cpu/shared_offload_region.py` 和 `tiering/spec.py`。给共享 mmap 区域加了 `populate_shard=(rank, world_size)` 参数：每个 worker 只预触碰自己那 1/world_size 的区间，而不是所有 worker 都 fault 整个区域。调度器侧不再做 pre-fault，也不再 pin 页面。测试里新增断言 `region_calls[0]["populate_shard"] == (1, 4)`。
- **新手概念课堂**：想象 4 个人合租一个 8 层仓库（共享内存），以前每个人搬进来都要把 8 层都走一遍（pre-fault），浪费 4 倍时间。现在每人只走自己负责的 2 层。调度器（管理员）则完全不走了。
- **对你有什么影响**：启用 tiering KV offload 时，进程启动更快、内存占用更低；多 worker 场景下不再有"N 个进程同时 fault 同一块内存"的浪费。

### 3. [Model] Remove TeleChat2 and TeleChat3 (#67035)

- **代码层面**：从 `docs/models/supported_models.md` 删除两行模型条目；从 `tests/models/registry.py` 删除 `TeleChat2ForCausalLM` / `TeleChat3ForCausalLM` 的测试配置；删除 `tests/transformers_utils/test_config.py` 中专门处理 TeleChat3 的 `test_patch_legacy_rope_type_normalizes_telechat3_yarn` 测试；清理对应的 RoPE 兼容代码。
- **新手概念课堂**：这是"模型退役"。就像 App 下架旧版本——代码库里维护一个模型要持续跟进上游权重格式、RoPE 配置、tokenizer 兼容，成本很高。TeleChat 系列的维护者不再更新，vLLM 就把它移出官方支持列表。
- **对你有什么影响**：如果你还在用 TeleChat2/3，需要自己维护 fork 或改用其他推理框架；普通用户无感。

### 4. [Recipe] MiniMax-M3.1-Flash: sync the GB300 Dynamo AgentX presets to InferenceX #4998 (#67184)

- **代码层面**：改 `recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml`。把 GB300 AgentX 预设从"TP4 聚合 + DEP2 prefill"改成"DEP4 prefill"；更新镜像 tag 到固定 sha256；新增 `kv_offload: "off"`、`--stream-interval 20`、`--prefill-schedule-interval 24`、`VLLM_RS_TOKEN_IDS_AS_BYTES: "1"` 等参数；把原来的 `gpu_kv` / `dram_kv` 两个 point 简化成 `latency` 一个。
- **新手概念课堂**：这是"调参配方（recipe）"更新。就像游戏里的"推荐配置档"——官方根据最新 benchmark 结果，告诉你 MiniMax-M3.1-Flash 在 GB300 上怎么配参数最香。`stream-interval` 控制流式输出每多少 token 刷一次，Agent 场景需要 20 而不是默认的 1。
- **对你有什么影响**：如果你照着这个 recipe 部署 MiniMax-M3.1-Flash，能拿到更接近官方 benchmark 的性能；镜像用 sha256 固定，避免"latest 变了结果就变"。

### 5. [Misc] Use the b12x package shipped with flashinfer-python (#64765)

- **代码层面**：`setup.py` 里 `"b12x": ["b12x==1.5.1"]` 改成 `"b12x": []`（保留 key 向后兼容）；文档改成"b12x 随 flashinfer-python>=0.7.2 一起安装，无需单独装"；CI 去掉 `uv pip install b12x==1.5.1`；错误信息从 "Install with pip install vllm[b12x]" 改成 "requires flashinfer-python>=0.7.2"。
- **新手概念课堂**：这是"依赖合并"。以前 b12x 是独立小包，现在被吸收进 flashinfer 大包。就像你以前要单独装"字体包"，现在操作系统自带。好处是版本一致、少一个依赖；坏处是 flashinfer 版本变成硬约束。
- **对你有什么影响**：SM120/SM121（RTX 50 系 / Blackwell 消费卡）用户不用再手动装 b12x；但如果你锁定了旧版 flashinfer，b12x 后端会不可用。

### 6. [Bugfix] Use max_split_size_mb:21 so freed 1-10 MiB tensors are reused (#66561)

- **代码层面**：`vllm/v1/worker/gpu_worker.py` 新增常量 `_MAX_SPLIT_SIZE_MB = 21`，`load_model` 里从 `max_split_size_mb=20` 改成用这个常量。新增测试 `test_scoped_max_split_reuses_freed_small_tensor_segments`：循环分配/释放 100 个 1–10 MiB tensor，断言 reserved 增长 ≤ 20 MiB。测试文件 `test_k3_gfx942_int4_requant.py` 也改用这个常量。
- **新手概念课堂**：PyTorch allocator 把显存切成"段"（segment）。一个 1 MiB 的请求会被放进 20 MiB 的段里。当你释放它时，这个段因为"大于 max_split_size_mb=20"被标记为"不可再切分"，于是既不能复用也不能合并，就浪费了。改成 21 后，20 MiB 的段小于阈值，可以继续被切分复用。
- **对你有什么影响**：模型加载阶段（大量小 tensor 分配）显存占用明显下降，OOM 概率降低。这是那种"改一个数字救活整个线上服务"的经典修复。

### 7. [ROCm][Disagg][CI] Set token budgets for the MiniMax-M3 wide-EP cells (#66646)

- **代码层面**：只改 `.buildkite/amd-disagg/models.yaml`。给 MiniMax-M3 的 prefill EP 加 `--max-num-batched-tokens 2048`，给 decode EP 加 `--max-num-batched-tokens 64`。
- **新手概念课堂**：`max-num-batched-tokens` 是"一次前向最多处理多少 token"。prefill 阶段输入长、计算密集，给大预算（2048）能提高吞吐；decode 阶段每步只生成 1 token/请求，给小预算（64）能降低延迟、减少显存峰值。这是"分阶段调参"的典型。
- **对你有什么影响**：只是 CI 配置，普通用户无感；但如果你在 AMD 上跑 PD 分离部署，这个配置思路值得抄。

### 8. [Doc][Tests] Stop setting `gpu_memory_utilization` in most extensible KV cache cases (#67071)

- **代码层面**：大规模删除 CI/benchmark/测试脚本里的 `--gpu-memory-utilization 0.9`（涉及 genai-perf-tests.json、nightly-tests.json、多个 shell 脚本、auto_tune README 等），共 3485 行 diff 大部分是删除。
- **新手概念课堂**：`gpu_memory_utilization` 是"允许 vLLM 用多少比例的 GPU 显存"。以前测试里到处硬编码 0.9，导致"显存不够就调参"的坏习惯。新方案是让 vLLM 自动探测（默认 0.9 但会根据实际可用显存调整），测试不再手动指定，避免"测试通过但用户环境失败"。
- **对你有什么影响**：如果你抄了旧文档里的 `--gpu-memory-utilization 0.9`，可能在新版本上不是最优；建议让 vLLM 自己决定，只在明确需要时手动设。

### 9. [Rust Frontend] Bump xgrammar-structural-tag to 0.5 and compose Lark answers (#66692)

- **代码层面**：`rust/Cargo.toml` 和 `Cargo.lock` 把 `xgrammar-structural-tag` 从 0.4.1 升到 0.5.0。`rust/src/chat/src/output/structural_tag.rs` 里 `answer_format` 函数：以前遇到 Lark 语法直接 `return None`（不处理），现在用 `GrammarSyntax::Lark` 标记并交给 `Format::grammar(grammar, syntax)` 统一处理。测试也相应更新。
- **新手概念课堂**：这就像"翻译器升级"。以前遇到"方言"（Lark）就拒收，现在能识别"这是方言，用方言翻译器处理"。`GrammarSyntax` 枚举就是"语言标签"。
- **对你有什么影响**：用 Rust 前端 + Lark 语法做结构化输出的用户，现在能享受快速路径，而不是回退到慢的 Python 路径。

### 10. [Frontend] Bump xgrammar to 0.2.9 and compose Lark answers as Lark (#66691)

- **代码层面**：`requirements/common.txt` 里 `xgrammar == 0.2.8` 改成 `0.2.9`；`requirements/test/cpu.txt` 的 hash 列表整体替换；Python 侧 `structured_outputs` 相关代码同步支持 Lark 语法类型传递。
- **新手概念课堂**：这是 Commit 9 的 Python 侧孪生兄弟。Rust 和 Python 两套前端都要支持 Lark，版本必须对齐（0.2.9 和 0.5.0 是配套版本）。
- **对你有什么影响**：Python 用户也能用 Lark 语法做约束解码了；升级 xgrammar 可能带来一些行为变化，注意看 release notes。

### 11. [Bugfix][Frontend] Pool HF fast tokenizers in BaseRenderer so concurrent truncation stays per request (#65932)

- **代码层面**：新增 `tests/renderers/test_tokenizer_pool.py`（206 行）；`vllm/renderers/base.py` 引入 tokenizer 池化机制；`vllm/tokenizers/hf.py` 的 `ThreadSafeHFTokenizerMixin` 和 `get_cached_tokenizer` 配合池化。核心是：HF fast tokenizer 的 `truncation` 状态是全局可变的，并发调用会互相干扰，所以每个请求要从池里借一个独立实例。
- **新手概念课堂**：HF fast tokenizer 像"共享单车"——如果两个人同时骑一辆车改座椅高度（truncation 参数），就会打架。池化就是"每人发一辆车"，用完还回去。`ThreadSafeHFTokenizerMixin` 是"加锁版共享单车"，但锁会降低吞吐，池化更好。
- **对你有什么影响**：高并发场景下，不同请求的截断长度不再串扰——这是一个非常隐蔽但严重的正确性 bug，修复后线上输出更可靠。

### 12. [ROCm][Attention] Add an mxfp4_mla KV cache for rope-free sparse MLA (#60321)

- **代码层面**：新增 `tests/v1/attention/mxfp4_mla/test_mxfp4_mla_plumbing.py`；`vllm/v1/attention/ops/mxfp4_mla.py` 新增 `row_bytes` 等函数；`vllm/config/cache.py` 支持 `cache_dtype="mxfp4_mla"`；`ROCMAiterMLASparseBackend` 的 `supported_kv_cache_dtypes` 加入 `mxfp4_mla`；`MLAAttention.get_kv_cache_spec` 能返回 uint8 打包的 page。核心是把 MLA 的 KV cache 用 MXFP4 量化存储，每行 272 字节（512 宽 latent）。
- **新手概念课堂**：MLA（Multi-head Latent Attention）是 DeepSeek 系模型的注意力结构，KV cache 是"潜向量"而不是完整 K/V。MXFP4 是 4-bit 浮点量化。把两者结合，就是"把潜向量压到 4 bit 存"，显存直接减半以上。`row_bytes(512) = 512/2 + 512/32 = 272`（4 bit 数据 + 缩放因子）。
- **对你有什么影响**：AMD ROCm 用户在跑 GLM-5.3 等 MLA 模型时，KV cache 显存占用大幅下降，能支持更长上下文或更大 batch。

### 13. [Perf][Frontend] Trim per-chunk and per-request work in the completions stream (#66325)

- **代码层面**：`vllm/entrypoints/openai/completion/serving.py` 把 `json.dumps(delta_text, ensure_ascii=False)` 换成预创建的 `_encode_delta_text = json.JSONEncoder(ensure_ascii=False).encode`（避免每次创建 encoder）；把 `prompt_text` 的提取从"每个 chunk 都做"改成"只在 echo 时做"；`_DEFAULT_SAMPLING_PARAMS` 标记为 `ClassVar`（避免 pydantic 把它当实例字段）。
- **新手概念课堂**：这是"热路径微优化"。流式输出每个 token 都要走一遍这个函数，`json.dumps` 每次创建 encoder 对象是浪费。就像餐厅每道菜都重新洗一次锅，改成"一口锅反复用"。`ClassVar` 是告诉 pydantic"这是类常量，别给我复制到每个实例"。
- **对你有什么影响**：流式 completions 的 CPU 开销降低，高 QPS 下吞吐提升；延迟敏感场景能感知到。

### 14. [Bugfix] Remove compiled models' bytecode hooks at in-process shutdown (#67081)

- **代码层面**：`vllm/compilation/wrapper.py` 的 `__init__` 加 `if getattr(self, "_bytecode_hook_handle", None) is None` 防重复注册；`cleanup()` 里 `handle.remove()` 后把 `_bytecode_hook_handle = None`；新增 `remove_bytecode_hooks(model)` 函数遍历所有 `TorchCompileWithNoGuardsWrapper` 并 cleanup；`vllm/v1/engine/llm_engine.py` 的 `_cleanup_instance_caches` 改用新函数。新增测试 `test_llm_delete_inprocess_removes_bytecode_hooks`。
- **新手概念课堂**：`torch._dynamo` 的字节码钩子是"全局监听器"，闭包持有模型引用。Qwen3 继承 Qwen2，两者都 `@support_torch_compile`，所以 `__init__` 跑两次、注册两个钩子；`cleanup()` 只移除一个，另一个永远残留 → 模型和权重无法 GC。修复是"注册前检查是否已有"+"清理后置 None"。
- **对你有什么影响**：in-process 反复创建/销毁 LLM 的场景（评测、多租户、notebook）不再泄漏显存；这是那种"跑一天就 OOM"的 bug 的根治。

### 15. [ROCm][Bugfix] Route biased softmax/sigmoid top-k to AITER topk_gating (#67107)

- **代码层面**：`vllm/_aiter_ops.py` 的 `_rocm_aiter_topk_gating_impl` 新增 `correction_bias` 和 `routed_scaling_factor` 参数，透传给 `aiter.ops.topk.topk_gating`；`fused_topk_bias_router.py` 的 `fused_topk_bias` 新增分支：当 `_aiter_topk_gating_supported` 为真时走 AITER 融合 kernel，否则走原路径。
- **新手概念课堂**：MoE 路由要选 top-k 专家，还要加 bias（负载均衡）、乘 scaling factor。以前这些步骤是分开的 kernel，现在 AITER 的 `topk_gating` 一个 kernel 全干完。就像"洗切炒"三个厨师 vs 一个全能厨师。
- **对你有什么影响**：AMD GPU 上跑带 bias 的 MoE 模型（如 DeepSeek 系）时，路由 kernel 更快，且修复了"biased 路径没走融合 kernel"的性能 bug。

### 16. [Bugfix][Multimodal] Cache multi-page Unlimited-OCR-2 requests in the processor cache (#66638)

- **代码层面**：`tests/models/multimodal/processing/test_unlimited_ocr.py` 新增两个测试：`test_v2_multi_page_request_uses_processor_cache`（两页图片走缓存，第二次用 `[None, None]` 占位仍能得到相同 token_ids）和 `test_v2_partial_cache_hit_matches_uncached_request`（部分命中缓存的结果与完全不走缓存一致）；`vllm/model_executor/models/unlimited_ocr.py` 修复 processor cache key 的计算，让多页请求的每一页能独立命中缓存。
- **新手概念课堂**：多模态 processor cache 是"图片处理结果缓存"。以前多页 OCR 请求的 cache key 把整组图片当整体，一页变了全部失效。修复后每页独立 key，部分命中也能复用。就像"拼图缓存"改成"每块拼图单独缓存"。
- **对你有什么影响**：Unlimited-OCR-2 多页文档处理时，重复页面的预处理时间大幅减少；部分重复的请求也能享受缓存。

### 17. [Perf][mHC] use FlashInfer mega mHC for MHCFusedPostPreOp (GLM-5.3-Fl… (#65521)

- **代码层面**：新增 `benchmarks/kernels/benchmark_mhc_flashinfer_mega.py`（147 行）对比 TileLang 版和 FlashInfer mega 版的 mHC 融合算子；`vllm/model_executor/kernels/mhc/flashinfer_mega.py` 新增 `split_mhc_fn` 等；`MHCFusedPostPreOp` 在满足条件时路由到 FlashInfer mega kernel。
- **新手概念课堂**：mHC 是 GLM-5.3-Flash 里的"多 head 压缩"结构，Post-Pre 是"后处理 + 预处理"融合。FlashInfer 的 "mega" kernel 把更多步骤塞进一个 kernel，减少 launch 开销和 HBM 往返。benchmark 用 CUDA graph replay 100 对 [前驱 → mHC] 来测真实 serving 场景。
- **对你有什么影响**：GLM-5.3-Flash 在 NVIDIA GPU 上推理更快；如果你在做 kernel 优化，这个 benchmark 脚本是很好的"如何正确 benchmark 融合算子"的范本。

### 18. [KVConnector] Allocate the extensible KV cache exportable for EXPORTS_FD (LMCache MP) (#66674)

- **代码层面**：`examples/disaggregated/lmcache/README.md` 新增"MP mode with the extensible KV cache"章节；`tests/utils_/test_extensible_tensor.py` 新增 `test_only_posix_fd_importers_get_a_posix_fd`；`vllm/utils/vmm_driver.py` 新增 `MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`；`ExtensibleTensor` 支持 `export_handle_types` 参数；LMCache MP connector 声明 `"lmcache.mp.use_vmm_api": true` 时，vLLM 用 POSIX fd 导出 KV cache。
- **新手概念课堂**：CUDA VMM（虚拟内存管理）分配的显存不能用传统 CUDA IPC 共享。LMCache 要跨进程共享 KV cache，就得用 POSIX 文件描述符（fd）导出。就像"新式门锁（VMM）不能用旧钥匙（IPC），得配新钥匙（fd）"。
- **对你有什么影响**：用 LMCache MP + 可扩展 KV cache 做 PD 分离部署的用户，现在能正常工作了（之前会拒绝启动）；需要 LMCache 特定版本和同主机 `/dev/shm` 共享。

### 19. [Rust Frontend] Stream MiniMax M3 tool-call arguments (#66531)

- **代码层面**：`rust/src/parser/src/tool/minimax_m3.rs` 大改：`MinimaxM3Mode` 从 `Text/ToolBlock{scan}/Done` 扩展成 `Text/ToolBlock/Invoke{call, body}/Done`；新增 `InvokeBody` 枚举（`Parameters/Parameter/BufferedParameter`）；新增 `element_end_start` marker；支持参数文本"边到边流式输出"，而需要 schema 转换的参数则缓冲到 element 结束再处理。
- **新手概念课堂**：工具调用参数流式输出就像"直播 vs 录播"。简单参数（字符串）可以边生成边推给用户（直播）；需要按 JSON schema 转换的参数得等完整再转（录播）。这个 commit 让 MiniMax M3 的工具调用参数能流式吐出来，而不是等整个 invoke 结束。
- **对你有什么影响**：用 MiniMax M3 做 Agent 时，工具调用参数的 UI 能实时显示，首 token 延迟感知大幅降低。

### 20. [Rust Frontend] Match MiniMax M3 markers by token identity (#66767)

- **代码层面**：`rust/src/parser/src/tool/minimax_m3.rs` 引入 `MinimaxM3Markers` 结构体，`new(tokenizer)` 时用 `tokenizer.token_to_id(NAMESPACE)` 解析 namespace token；`Marker` 类型带 `Option<SpecialToken>`，匹配时要求"专用 token 产生了这个标记"；`Attributed<'i, MinimaxM3Markers>` 流类型携带 markers；`close_tag(name)` 动态生成结束标记。
- **新手概念课堂**：这是"防伪标记"。模型可能恰好生成 `<]minimax[>[<` 这些字符，但如果不是通过专用 token 产生的，就不该被当成工具调用分隔符。按 token id 匹配就像"验钞机"——不只看图案（字符串），还看防伪线（token id）。
- **对你有什么影响**：MiniMax M3 工具调用解析更鲁棒，不会被模型"恰好

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：DSA / 稀疏注意力后端矩阵化**
  - 一句话解释：DeepSeek 稀疏注意力（DSA）的 prefill/decode 后端现在有 `flashmla_sparse`、`trtllm`、`prims_ts` 等多个可选实现，可以按硬件和 KV 精度挑最合适的。
  - 为什么社区在关注：同一个模型能在不同 GPU（Hopper/Blackwell/AMD/NPU）上跑出最优性能，说明推理引擎正在从"一个后端打天下"走向"后端插件化"。
  - 对新手意味着什么：学推理优化不再只是调参，而是理解"算子-硬件-精度"的三角匹配，这是当前最抢手的工程能力。

- **关键词：AMD/ROCm 全面适配**
  - 一句话解释：今天有 4 个 commit 是 AMD 相关（Muse Spark 优化、流调度、MoE 权重 shuffle 标记、HIP 编译修复）。
  - 为什么社区在关注：MI300/MI355 正在成为 NVIDIA 的真实替代品，框架必须把 ROCm 当一等公民对待。
  - 对新手意味着什么：如果你只会 CUDA，补一点 HIP/ROCm 知识会让你在异构部署岗位上有明显优势。

- **关键词：Bit-exact 融合（逐位精确的算子融合）**
  - 一句话解释：把多个小算子合成一个大 kernel，但保证输出和原来逐个算的结果**每一位都相同**，不是"差不多"。
  - 为什么社区在关注：扩散模型/视频生成对数值极其敏感，一点点误差会累积成画面崩坏，所以"无损加速"成为刚需。
  - 对新手意味着什么：这是性能优化的高级形态，学会"先验证等价、再谈加速"的工程方法论比背 kernel 写法更重要。

- **关键词：CUDA Graph / 流调度 / PDL**
  - 一句话解释：把一串 GPU 操作录制成"图"一次性回放，减少 CPU 启动开销；PDL 是让相邻 kernel 提前启动的重叠技术。
  - 为什么社区在关注：LLM 推理的瓶颈越来越偏向 launch 开销和 stream 竞争，调度细节直接决定吞吐。
  - 对新手意味着什么：理解 stream、graph、同步点是排查"为什么 GPU 利用率上不去"的基本功。

- **关键词：投机解码（Speculative Decoding）兼容性治理**
  - 一句话解释：用小模型"猜"多个 token 再让大模型验证，加速生成；但不同算法与稀疏 KV 的组合不一定安全。
  - 为什么社区在关注：HiSparse 明确拒绝未验证的投机算法，说明社区开始用"白名单 + 显式报错"防止静默数据损坏。
  - 对新手意味着什么：学会"失败要响亮"（fail loud）的设计哲学，比写出能跑的代码更值钱。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：算子融合的"逐位精确"验证（Bit-exact fusion）**
  - **是什么**：把 `LayerNorm(x) * (1+scale) + shift` 这种多步计算合并成一个 Triton/CUDA kernel，但要求结果与 PyTorch 原生链**逐 bit 相同**。
  - **为什么重要**：视频/图像生成模型对数值误差极敏感，50 步去噪会把 1e-6 的偏差放大成肉眼可见的画面差异。所以加速前必须先证明"数学等价"，否则用户会投诉"画质变差了"。
  - **动手试试**：
    ```python
    import torch

    def eager_ln_mod(norm, x, scale, shift):
        return norm(x) * (1 + scale) + shift

    # 模拟一个"融合"版本：先算 mean/var 再一次性组合
    def fused_ln_mod(x, scale, shift, eps=1e-6):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        xhat = (x - mean) / torch.sqrt(var + eps)
        return xhat * (1 + scale) + shift

    torch.manual_seed(0)
    x = torch.randn(4, 128, 256, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    norm = torch.nn.LayerNorm(256).cuda().to(torch.bfloat16)

    a = eager_ln_mod(norm, x, scale, shift)
    b = fused_ln_mod(x, scale, shift)
    print("bit-exact?", torch.equal(a, b))
    print("max abs diff:", (a.float() - b.float()).abs().max().item())
    ```
  - **预期结果**：你会看到 `bit-exact? False`，`max abs diff` 是 1e-2 量级——因为 bf16 下 `var` 的算法和 LayerNorm 内部实现不同。这正好说明：**"看起来一样"和"逐位一样"是两回事**，真实融合 kernel 必须复刻 aten 的求和顺序（这也是 commit 6/17/19 里那些 `_ordered_*` 函数存在的原因）。

- **知识点 2：CUDA Stream 与"死锁"风险**
  - **是什么**：GPU 上的 stream 是任务队列；不同 stream 上的 kernel 可以并发，但如果两个 kernel 互相等待对方释放资源（比如占满所有 SM 的 monokernel），就会死锁。
  - **为什么重要**：Commit 10 就是修这个——Muse Spark 的 monokernel 需要"所有 workgroup 同时驻留"，如果 scheduler 在另一个 stream 上抢了一个 CU，就会永久卡住。这类 bug 表现为"服务偶发 hang 住"，极难排查。
  - **动手试试**：
    ```python
    import torch

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")

    # 在 s1 上排队
    with torch.cuda.stream(s1):
        c = a @ b
    # 在 s2 上排队，但依赖 c
    s2.wait_stream(s1)
    with torch.cuda.stream(s2):
        d = c @ b

    torch.cuda.synchronize()
    print("done, d mean =", d.mean().item())
    torch.cuda.synchronize()
    # 查看当前 stream 与默认 stream 是否相同
    print("current == default:", torch.cuda.current_stream() == torch.cuda.default_stream())
    ```
  - **预期结果**：能正常跑完。把 `s2.wait_stream(s1)` 删掉再跑，你会遇到结果错误或随机崩溃——这就是"缺少同步"的典型症状。理解这一点，你就明白为什么 commit 10 要**故意**让 scheduler 和 forward 共用同一个 stream。

- **知识点 3：MoE 权重的"shuffle 标记"与 kernel 选择**
  - **是什么**：AMD 的 AITER 库要求 MoE 权重按特定 16×16 分块重排（shuffle）后才能用最快的 kernel。代码里用 `weight.is_shuffled = True` 这个属性告诉下游"我已经排好了"。
  - **为什么重要**：Commit 11/12 修的是——`copy_or_rebind_param` 把数据拷进旧 Parameter 时把标记弄丢了，结果 kernel 以为权重没排过，走了慢路径或算错。这是"元数据与数据不同步"的经典坑。
  - **动手试试**：
    ```python
    import torch

    w = torch.nn.Parameter(torch.randn(64, 64))
    w.is_shuffled = True

    # 模拟 copy_or_rebind_param：拷贝数据到已有 Parameter
    new_data = torch.randn(64, 64)
    w.data.copy_(new_data)
    print("after copy_, is_shuffled =", getattr(w, "is_shuffled", False))

    # 正确做法：拷贝后重新打标记
    w.is_shuffled = True
    print("after re-mark, is_shuffled =", w.is_shuffled)
    ```
  - **预期结果**：第一次打印 `True`（因为属性挂在同一个对象上）——但如果实现是 `w = torch.nn.Parameter(new_data)` 重新绑定，属性就丢了。这解释了为什么 commit 11 的修复必须在 `copy_or_rebind_param` 之后**手动**补一句 `layer.w13_weight.is_shuffled = True`。

- **知识点 4：hipBLAS/ROCm 上的类型重载歧义**
  - **是什么**：C++ 里 `__ffsll` 有两个重载（`unsigned long long` 和 `long long`），传入 `uint64_t` 时编译器无法决定用哪个，报"ambiguous overload"。
  - **为什么重要**：Commit 14 就是修这个——HIP 平台的编译错误，一个 `static_cast<unsigned long long>` 解决。跨平台代码里这种"类型别名在不同平台宽度不同"的坑非常常见（`long` 在 Linux 是 64 位，Windows 是 32 位）。
  - **动手试试**：
    ```cpp
    // test_ffs.cpp
    #include <cstdint>
    #include <cstdio>

    // 模拟 HIP 的双重载
    inline int my_ffs(unsigned long long x) { return __builtin_ffsll((long long)x); }
    inline int my_ffs(long long x) { return __builtin_ffsll(x); }

    int main() {
        uint64_t mask = 0x10;
        // int r = my_ffs(mask);          // 编译错误：ambiguous
        int r = my_ffs(static_cast<unsigned long long>(mask));  // OK
        printf("%d\n", r);
    }
    ```
    ```bash
    g++ -std=c++17 test_ffs.cpp -o test_ffs && ./test_ffs
    ```
  - **预期结果**：注释掉的那行会报 `call of overloaded 'my_ffs(uint64_t&)' is ambiguous`，加上 `static_cast` 后输出 `5`（第 5 位是 1）。你就掌握了"用显式类型转换消除重载歧义"这个跨平台调试技巧。

- **知识点 5：投机解码与稀疏 KV 的"安全边界"**
  - **是什么**：投机解码让 draft 模型一次猜 k 个 token，target 模型批量验证。但如果 target 的 KV cache 是稀疏/分页的，某些算法（比如非线性的 tree attention）可能写到未映射的显存地址。
  - **为什么重要**：Commit 13 用白名单 `("EAGLE", "EAGLE3", "UNO")` 拒绝其他算法，并明确报错——因为**静默的显存越界写入 = 输出乱码但不崩溃**，比崩溃更难查。
  - **动手试试**：
    ```python
    VALIDATED = {"EAGLE", "EAGLE3", "UNO"}

    def check_spec(algo, topk, adaptive):
        if algo and algo.upper() != "NONE":
            if algo.upper() not in VALIDATED:
                raise ValueError(f"unvalidated algorithm {algo!r}, allowed: {VALIDATED}")
            if topk != 1:
                raise ValueError("only linear speculation (topk=1) supported")
            if adaptive:
                raise ValueError("adaptive speculation not supported with sparse KV")

    for args in [("EAGLE", 1, False), ("MEDUSA", 1, False), ("EAGLE", 4, False)]:
        try:
            check_spec(*args)
            print(args, "-> OK")
        except ValueError as e:
            print(args, "-> REJECTED:", e)
    ```
  - **预期结果**：`EAGLE` 通过，`MEDUSA` 被拒（不在白名单），`EAGLE topk=4` 被拒（非线性投机不支持）。这就是"fail loud"设计——宁可启动失败，也不要跑出错误结果。

---

## 🔥 重要更新

- **Commit 1（FlashInfer Prims-TS 稀疏 MLA 后端）**：为 DSA 增加了 SM100/SM103/SM107 上的新 decode/prefill 后端，配合 FP8 KV cache 使用，是 Blackwell 上稀疏注意力的重要性能补充。
- **Commit 5（KV cache 分片下的 gated allgather）**：用 PyTorch symmetric memory + CUDA multicast 把 all-gather 放到 copy engine 上跑（零 SM 占用），并支持前台传输暂停打断，对长上下文分布式推理吞吐有实质影响。
- **Commit 16（FlashInfer 0.7.4 升级）**：顺带修了 UNO 投机解码在 torch.compile 下无法 trace FlashInfer 流查询的问题——说明依赖升级会引发编译兼容性问题，值得警惕。
- **Commit 18（DSA native-head FlashMLA + DCP decode 预计算调度）**：把 decode 关键路径上的 `<<<1,32>>>` 调度 kernel 提前算好，并支持 native head 数，是 decode 延迟优化的典型手法。

---

## 📋 逐条解读

### 1. [Perf] dsa: Add an opt-in FlashInfer Prims-TS sparse MLA backend (#48602)
- **代码层面**：在 `exec_.py` 的 `dsa_prefill_backend` / `dsa_decode_backend` 选项列表里加了 `"prims_ts"`，并在 `overrides.py` 里加了对应的后端解析逻辑；文档同步更新。核心是让用户可以通过 `--dsa-decode-backend prims_ts` 显式选用 FlashInfer 的新稀疏 MLA kernel。
- **新手概念课堂**：就像手机相机有"自动模式"和"专业模式"——以前 DSA 只给你自动选后端，现在多了一个手动档 `prims_ts`，专门为 Blackwell + FP8 KV 调优。它不是默认，得你自己开。
- **对你有什么影响**：如果你在 B200/B300 上跑 DeepSeek 类模型且用 FP8 KV cache，可以试试这个后端看吞吐是否提升；普通用户无感知。

### 2. [Doc] diffusion: Fix feature guides that disagree with the code (#49675)
- **代码层面**：修了一堆文档与代码不一致的地方：`wan_repack.py` 的命令行参数从 `--input-path` 改成 `--model-type/--original-model-path/--quant-path`；`--kv-gather-degree` 的默认行为描述补上了"heads 能整除时用 Ulysses，否则用 K/V gather"；`--enable-breakable-cuda-graph` 的 quality 层级说明修正。
- **新手概念课堂**：文档就像菜谱，代码是实际做菜的过程。这次是"菜谱写错了步骤"，照着做会失败——比如按旧参数跑 `wan_repack.py` 会直接报参数错误。
- **对你有什么影响**：如果你之前照着旧文档转换 Wan2.2 量化权重失败了，现在文档修好了，重跑即可。

### 3. [Doc] README: update the readme logo (#49983)
- **代码层面**：把 README 里的 adoption 图片从外链 GitHub raw URL 换成仓库内的 `assets/adoption.png`。
- **新手概念课堂**：外链图片依赖第三方仓库可用性，一旦对方删库或改路径，README 就"裂图"了。放本地是"自给自足"。
- **对你有什么影响**：README 加载更稳定，国内网络访问 GitHub raw 慢的用户体验会改善。

### 4. [Perf] kimi-k3: Build the fused routing metadata in a separate 1024-thread PDL kernel (#49923)
- **代码层面**：`kimi_k3_fused_routing.cuh` 里把 routing metadata 的构建从原来复用 LargeRouter 的 block size 改成独立的 1024 线程 kernel；去掉了跨 launch 共享的 `counter`（原来最后一个 CTA 要重置它，导致 launch 不能重叠）；把 `permuted_idx_to_token_idx` 的 -1 填充改成所有线程并行做。
- **新手概念课堂**：原来像"一个厨师又炒菜又摆盘"，现在拆成"专门一个 1024 人的摆盘团队"，还能和下一道菜（PDL 提前启动）重叠。去掉共享 counter 就像去掉了"必须等上一位用完才能进"的门锁。
- **对你有什么影响**：Kimi-K3 模型在 MoE 路由阶段的启动延迟降低，decode 更快；普通用户感知为"响应稍微快一点"。

### 5. [Feat] Comm: Add gated allgather in kv cache sharding (#45947)
- **代码层面**：新增 `multicast_all_gather.py`，用 PyTorch symmetric memory + CUDA multicast 实现 all-gather，跑在 copy engine 上（不占 SM），并按 chunk 分片以便被前台 NVLink 传输暂停打断；环境变量 `SGLANG_ENABLE_KV_SHARD_ZERO_SM_GATHER` 的语义从"NCCL symmetric window"改为"PyTorch symmetric memory + multicast"。
- **新手概念课堂**：原来 all-gather 要占用计算核心（SM）来搬运数据，就像让厨师亲自去仓库搬菜；现在用 DMA（copy engine）搬，厨师继续炒菜。分 chunk 是为了"有人要用走廊时能立刻让路"。
- **对你有什么影响**：开启 `--enable-kv-cache-sharding` 的多卡部署，长上下文 prefill 时计算和通信重叠更好，吞吐提升；不支持 multicast 的 GPU 自动回退到 NCCL，无副作用。

### 6. [Perf] diffusion: Enable verified eager LN fusion for SANA-Video (#47253)
- **代码层面**：`sana_ln_modulate` 新增 `allow_eager` 参数；`_use_sana_bcg_fast_path` 统一判断是否走融合路径；SANA-Video 的两处调用传 `allow_eager=True`。原先默认流（非 CUDA graph）下强制走 eager，现在 SANA-Video 可以 eager 也用融合 kernel。
- **新手概念课堂**：融合 kernel 有"启动成本"（Python 调用开销）。小张量时融合反而更慢（+14% 墙钟时间），所以之前只在 CUDA graph 里用。SANA-Video 的激活值大，融合收益盖过启动成本，所以开个后门允许 eager 用。
- **对你有什么影响**：SANA-Video 在非 CUDA graph 模式下推理更快，且仍保持逐位精确（有 per-signature 验证 + 回退）。

### 7. [Chore] codeowners: paged-experts (#49867)
- **代码层面**：`.github/CODEOWNERS` 里把 `paged_experts.*` 的通配从精确文件名改成 `paged_experts*`，并新增测试目录、测试文件、文档的 owner 条目。
- **新手概念课堂**：CODEOWNERS 是"这块地归谁管"——PR 改到这些文件会自动请求对应 reviewer。通配符放宽是为了覆盖新加的 `.cuh`、`.py` 等变体文件。
- **对你有什么影响**：普通用户无感；贡献 paged-experts 相关代码的开发者会被自动 @ 到正确的 reviewer。

### 8. [Perf] [AMD] muse-spark: Prefill and decode optimization (#49704)
- **代码层面**：删除了 `3rdparty/amd/muse_spark/` 下的 README 和 patch 脚本（说明相关改动已合入主路径），并大幅修改了 ROCm 下 Muse Spark 的 prefill/decode 实现（diff 被截断，涉及 8000+ 行）。
- **新手概念课堂**：这就像"临时补丁"毕业了——原来要在容器里手动打 patch 才能跑，现在代码进了主仓库，镜像构建即用。
- **对你有什么影响**：在 MI355X 上部署 Muse Spark 更简单，不用再手动 apply patch；性能有 prefill/decode 优化。

### 9. [Perf] diffusion/h3: Calibrate quality=high with a per-step cache schedule (#49762)
- **代码层面**：MiniMax-H3 的 `quality="high"` 从"固定 Cache-DiT 配置"改成"按步数调度"：前 15 步全跑，16-25 步隔一步跑，之后每 8 步跑一次。文档里的基准表从"vs exact"改成"vs lossless"，因为 exact 和 lossless 只差舍入但 50 步后会变成不同样本，不能当参考。
- **新手概念课堂**：就像视频渲染的"关键帧 + 插值"——前段动作多必须精细算，后段变化小可以复用。之前拿 exact 当基准是错的，因为 exact 本身就和 lossless 不是同一个样本（蝴蝶效应）。
- **对你有什么影响**：`quality="high"` 从 1.40× 提速到 1.99×，SSIM 0.911 仍可接受；但如果你之前对比 exact 觉得"画质变差"，现在知道那是基准选错了。

### 10. [Fix] [AMD] scheduler: Keep Muse Spark's scheduler GPU work on the forward stream under monokernel (#47491)
- **代码层面**：新增 `_gpu_work_on_forward_stream()` 判断（HIP + `enable_amd_monokernel` + Muse Spark 架构）；满足时 `copy_stream` 直接复用 `forward_stream`，`schedule_stream` 也设为 `forward_stream`，避免另开 stream 抢占 CU。
- **新手概念课堂**：Muse Spark 的 monokernel 需要"所有 workgroup 同时在场"才能跑（它们互相等待）。如果 scheduler 在另一个 stream 上启动 kernel 抢了一个计算单元，monokernel 就永远等不齐人——死锁。解法是"大家都排同一条队"。
- **对你有什么影响**：修复 Muse Spark 在 monokernel 模式下的偶发 hang；代价是 scheduler 和 forward 不再并行，但 host 侧仍可重叠。

### 11. [Fix] [AMD] moe: Keep is_shuffled on AITER-shuffled unquantized MoE weights (#48219)
- **代码层面**：`unquant.py` 里 `copy_or_rebind_param` 调用 `shuffle_weight` 之后，手动补 `layer.w13_weight.is_shuffled = True` 和 `layer.w2_weight.is_shuffled = True`。
- **新手概念课堂**：`shuffle_weight` 把权重按 16×16 重排并打上"已重排"标签，但 `copy_or_rebind_param` 是"把新数据拷进旧盒子"，标签丢了。下游 `aiter.fused_moe` 看标签决定用哪个 kernel，标签丢了就走错路径。
- **对你有什么影响**：AMD 上非量化 MoE 模型（如某些 BF16 MoE）之前可能走了慢路径或算错，现在修复。

### 12. [Fix] [AMD] ct-fp8-moe: Mark the pre-shuffled CT FP8 MoE weights is_shuffled (#47870)
- **代码层面**：`compressed_tensors_w8a8_fp8_moe.py` 里两处 `torch.nn.Parameter(shuffle_weight(...))` 之后补 `is_shuffled = True`。和 commit 11 是同一类 bug，只是这里是"重新绑定 Parameter"而非"拷贝进旧 Parameter"。
- **新手概念课堂**：新建 Parameter 对象时属性不会自动继承，必须手动打标。就像新员工入职要重新发工牌，不能指望旧工牌自动转移。
- **对你有什么影响**：AMD 上 CT FP8 量化的 MoE 模型（如 DeepSeek 系列 FP8 权重）能正确走 preshuffled kernel，性能和正确性都修复。

### 13. [Feat] hisparse: reject unvalidated speculative algorithms by name (#48846)
- **代码层面**：`hisparse_hook.py` 新增 `HISPARSE_VALIDATED_SPEC_ALGORITHMS = ("EAGLE", "EAGLE3", "UNO")`；`validate_hisparse` 先检查算法是否在白名单，不在就报错并说明"会写到未映射显存导致静默损坏"；然后再检查 topk==1 和 adaptive 不支持。
- **新手概念课堂**：HiSparse 的 verify slot 是动态映射的，某些投机算法（如 MEDUSA 的 tree attention）不知道这个映射规则，会往"空地"写 KV——不崩溃但数据错了。白名单是"只允许走过验证的路"。
- **对你有什么影响**：开 `--enable-hisparse` 时如果配了不支持的投机算法，启动就会明确报错，而不是跑出乱码让你 debug 三天。

### 14. [Fix] hisparse: disambiguate __ffsll for HIP's dual overloads (#48840)
- **代码层面**：`hisparse_spec.cuh` 的 `first_set_lane` 里，ROCm 分支把 `__ffsll(mask)` 改成 `__ffsll(static_cast<unsigned long long>(mask))`。
- **新手概念课堂**：`uint64_t` 在 Linux 上是 `unsigned long`，能隐式转成 `unsigned long long` 也能转成 `long long`，编译器不知道选哪个就报 ambiguous。显式 cast 是"我指定用这个"。
- **对你有什么影响**：修复 HIP 平台编译错误，AMD 用户能正常编译 hisparse 相关 kernel。

### 15. [Fix] anthropic: Accept thinking enabled without budget_tokens (#47093)
- **代码层面**：Python 的 `AnthropicThinkingParam._validate_thinking_shape` 和 Rust router 的 `thinking_enabled` 都改成：`type="enabled"` 时 `budget_tokens` 可以缺省（视为开启 thinking），但如果给了且 <1024 仍报错。测试同步更新。
- **新手概念课堂**：Anthropic 官方 SDK 要求 `enabled` 必须带 `budget_tokens`，但一些兼容客户端省略了它。之前 SGLang 严格拒绝，导致这些客户端用不了。现在放宽为"可选，给了才校验"。
- **对你有什么影响**：用某些第三方 Anthropic 兼容客户端（省略 budget_tokens）现在能正常调用 SGLang 的 Anthropic 接口。

### 16. [Chore] deps: Bump FlashInfer to 0.7.4 (#49771)
- **代码层面**：Dockerfile、`pyproject.toml`、`engine.py` 的版本断言都从 0.7.3 升到 0.7.4；`uno_utils.py` 去掉 `_build_sparse_target_support_tensors` 上的 `@torch.compile(dynamic=True)`，因为 FlashInfer 0.7.4 的 scratch buffer 查询调用 `torch._C._cuda_getCurrentRawStream`，Dynamo 无法 trace。
- **新手概念课堂**：依赖升级不是免费的——新版本内部实现变了，可能和 `torch.compile` 不兼容。这里的解法是"这个函数别编译了，手动拆成两步"。
- **对你有什么影响**：升级 FlashInfer 后 UNO 投机解码仍能正常工作；如果你自己写代码调 FlashInfer + torch.compile，注意这个 stream 查询的坑。

### 17. [Perf] [NVIDIA] diffusion/joy: Reuse FP32 residual gate for large image updates on SM103 (#49391)
- **代码层面**：`joy_image.py` 的 `fused_add_gate` 新增 `_JOY_RESIDUAL_GATE` 验证门；大张量（≥16M 元素）、连续 BF16、SM103 时走新的 Triton kernel `residual_gate_fp32`（乘积保持 FP32 到最终 cast），小张量/其他布局走原生 `addcmul`。
- **新手概念课堂**：`addcmul` 在 BF16 下乘积会先降到 BF16 再相加，损失精度。新 kernel 把乘积留在 FP32，最后才 cast——更准，但只在"大张量 + SM103"这个组合下划算（小张量启动开销盖过收益）。
- **对你有什么影响**：Joy Image Edit 在 B300（SM103）上大图编辑更快且数值更接近原生 `addcmul`；其他硬件/小图无变化。

### 18. [Perf] dsa: Native-head FlashMLA and a precomputed schedule for DCP decode (#49980)
- **代码层面**：`_fast_flashmla_sched_shape` 扩展支持 native head 数（1≤heads<128 且 SM100）；`_maybe_precompute_flashmla_sched_meta` 新增 `schedule` 参数（`serial`/`parallel`/`windowed`），native head 时用 `sparse_decode_num_sm_parts` 算分区数；`dsa_backend.py` 相应调整。
- **新手概念课堂**：decode 时每个请求要算"怎么把 KV 分给各 SM"的调度表，原来每次 decode 都跑一个 `<<<1,32>>>` 小 kernel 现算，现在提前算好存起来。`windowed` 调度能在更短时间内得到"同样均衡"的表。
- **对你有什么影响**：DSA 模型 decode 延迟降低（省掉关键路径上的小 kernel），且支持更多 head 配置。

### 19. [Perf] [NVIDIA] diffusion/nomas: Fuse exact head-128 mean-square reduction (#49308)
- **代码层面**：`rmsnorm_cast_before_weight_bitexact.py` 把 `_ordered_rstd` 拆出 `_ordered_mean_square`，新增独立的 `_mean_square_kernel` 和 `rmsnorm_mean_square_bitexact` custom op，匹配 `x.float().square().mean(-1, keepdim=True)` 的逐位结果；有 `can_use_rmsnorm_mean_square` 前置条件检查。
- **新手概念课堂**：RMSNorm 需要先算 `mean(x²)`。原来这一步和后续的 cast/weight 乘混在一起，现在单独抽出来做成一个可复用的精确 kernel——就像把"切菜"从"炒菜"里拆出来，两边都能单独优化。
- **对你有什么影响**：Nomas 类扩散模型在 head_dim=128 的 BF16 输入上 RMSNorm 更快，且保持逐位精确。

### 20. [Test] mem_cache: Set sparse_prefill_small_heads on the page-sharded DSA test backend (#49987)
- **代码层面**：`test_page_sharded_dsa_attention.py` 的 `_attention_backend` 辅助函数里，手动设置 `backend.sparse_prefill_small_heads = _small_heads_prefill_enabled(...)`，模拟真实 `DeepseekSparseAttnBackend.__init__` 的行为。
- **新手概念课堂**：测试用 `__new__` 绕过 `__init__` 构造对象（为了只测 forward），但漏设了 `__init__` 里算好的一个字段，导致测试路径和真实路径不一致。修复是"手动补上真实 init 会做的事"。
- **对你有什么影响**：普通用户无感；这是测试正确性修复，避免测试通过但线上行为不同。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)