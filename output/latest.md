# 🚀 vLLM & SGLang 每日更新 — 2026-10-18

> 自动生成于 2026-10-18 06:33 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：ROCm / AMD GPU 适配**
  - **一句话通俗解释**：ROCm 是 AMD 版的 CUDA，让 AMD 显卡也能跑大模型推理。今天有多个 commit 在调 ROCm 的超时、镜像、同步逻辑。
  - **为什么社区在关注**：ROCm 10.0 成为默认镜像、AITER JIT 编译超时放宽到 30 分钟、NIXL 传输前加设备同步——这说明 vLLM 正在把 AMD 当作一等公民对待，而不只是"能跑就行"。异构计算（NVIDIA + AMD + Intel）已是常态。
  - **对新手意味着什么**：如果你只会 CUDA，现在是补 ROCm/HIP 的好时机。AMD 的软件栈人才缺口比 CUDA 大得多，且很多坑还没被踩平，正是能做出贡献的窗口期。

- **关键词：投机解码（Speculative Decoding）**
  - **一句话通俗解释**：让一个小模型先"猜"几个 token，大模型再一次性验证，猜对了就白赚速度。
  - **为什么社区在关注**：今天有 commit 修"动态投机解码 + 固定 K 提案器"的崩溃问题，说明这个特性已经从实验走向生产，开始处理边界组合。
  - **对新手意味着什么**：投机解码是当前推理加速最热的方向之一（EAGLE、MTP、Medusa 都是这个家族），理解它的调度逻辑是进入高性能推理团队的敲门砖。

- **关键词：KV Cache 传输 / NIXL / 分布式连接器**
  - **一句话通俗解释**：把已经算好的 KV 缓存从一个 GPU/节点搬到另一个，避免重复计算（PD 分离、前缀缓存共享都靠它）。
  - **为什么社区在关注**：今天有 3 个 commit 涉及 NIXL（EPLB 缓冲区同步、push 模式清理、MRV2 结果收集顺序），说明 PD 分离架构正在被大量生产环境使用，稳定性问题集中暴露。
  - **对新手意味着什么**：分布式推理的"数据搬运"是性能瓶颈所在，懂 RDMA/NIXL/CUDA Stream 同步的人非常稀缺。

- **关键词：量化（MXFP4 / FP8）+ LoRA**
  - **一句话通俗解释**：把权重从 16 位压到 4 位/8 位，省显存；LoRA 是给冻结的大模型挂小补丁做微调。
  - **为什么社区在关注**：今天新增了"FlashInfer TRTLLM MXFP4 专家层支持 LoRA"——把两个复杂特性叠加，说明量化 + 微调的组合是实际业务刚需。
  - **对新手意味着什么**：量化 kernel 是 GPU 编程的高价值领域，而 LoRA 是应用层最常用的定制手段，两者交叉处人才极少。

- **关键词：确定性 / 批次不变性（Batch Invariance）**
  - **一句话通俗解释**：同一个 prompt，不管跟谁拼成一个 batch，输出必须逐位相同。
  - **为什么社区在关注**：今天有 commit 打开 inductor 的 `batch_invariant` 开关，还有"训练-推理一致性"技能文档的增删反复。这是 RL 训练（on-policy）的硬需求。
  - **对新手意味着什么**：如果你对 RLHF/RLVR 感兴趣，这是绕不开的底层知识——浮点归约顺序、kernel 分块策略都会影响结果。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：设备流同步与"数据被覆盖"的竞态（来自 Commit 3）**
  - **是什么**：GPU 上的操作是异步的。你发起一个拷贝后立刻发起下一个操作，如果没同步，第二个操作可能读到还没写完的数据。
  - **为什么重要**：NIXL 的 RDMA READ 不经过 CUDA stream，所以"stream 上排好队"并不能保证数据就绪。这类 bug 表现为"偶发数据错乱"，极难复现，线上排查成本极高。
  - **动手试试**：
    ```python
    import torch

    # 模拟：stream A 写 buffer，另一个通道（这里用 CPU 侧读取）在没同步时读
    buf = torch.zeros(1024, device="cuda")
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        buf.fill_(1.0)          # 异步写

    # 错误做法：不同步直接读
    print("未同步读到:", buf.sum().item())   # 可能仍为 0

    # 正确做法
    s.synchronize()
    torch.cuda.current_stream().synchronize()
    print("同步后读到:", buf.sum().item())   # 一定是 1024
    ```
  - **预期结果**：第一次可能打印 0（说明竞态存在），同步后稳定打印 1024。你会理解为什么 vLLM 要在 `set_transfer_context` 里插 `synchronize()`。

- **知识点 2：投机解码的"固定 K vs 动态 K"（来自 Commit 10）**
  - **是什么**：固定 K = 每次固定猜 N 个 token；动态 K = 根据 batch size 自动调整猜几个。后者需要提案器（proposer）支持"按请求返回不同长度"。
  - **为什么重要**：不是所有提案器都能动态调整。强行开动态模式会导致形状不匹配崩溃，vLLM 的做法是"检测到不支持就降级并 warning"，而不是报错退出——这是生产级软件的容错设计。
  - **动手试试**：
    ```python
    # 观察 vLLM 的降级逻辑（不需要 GPU，纯读逻辑）
    from vllm.config import VllmConfig
    # 伪代码演示判断条件
    method = "custom_proposer"   # 不在白名单里
    whitelist = ("ngram", "eagle", "eagle3", "mtp", "draft_model", "dflash")
    use_v2 = False
    dynamic = True

    if dynamic and not use_v2 and method not in whitelist:
        print(f"警告：{method} 在 V1 上不支持动态投机解码，回退到静态 K")
    ```
  - **预期结果**：打印降级警告。理解"白名单 + 优雅降级"这个模式，它在 vLLM 里到处都是。

- **知识点 3：MXFP4 量化格式与反量化（来自 Commit 17）**
  - **是什么**：MXFP4 用 4 位存一个数（1 位符号 + 3 位尾数，取值只有 0/0.5/1/1.5/2/3/4/6），每 32 个数共享一个 8 位指数缩放因子。
  - **为什么重要**：4 位量化能把显存占用降到 FP16 的 1/4，但精度损失需要数学上理解清楚，否则会写出错误的参考实现，测试就失去意义。
  - **动手试试**：
    ```python
    import torch

    E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    def dequant_mxfp4(weight, scale):
        # weight: (..., K/2) uint8，每个字节装两个 4 位值
        nibbles = torch.stack([weight & 0xF, weight >> 4], dim=-1).flatten(-2).long()
        mag = torch.tensor(E2M1, dtype=torch.float64)
        vals = mag[nibbles & 7] * torch.where(nibbles >> 3 == 1, -1.0, 1.0)
        # scale: e8m0，实际指数 = scale - 127
        return vals * torch.pow(2.0, scale.double() - 127.0).repeat_interleave(32, dim=-1)

    w = torch.tensor([[0x21]], dtype=torch.uint8)   # 低4位=1, 高4位=2
    s = torch.tensor([[127]], dtype=torch.uint8)     # 2^0 = 1
    print(dequant_mxfp4(w, s))   # 期望 [0.5, 1.0]
    ```
  - **预期结果**：输出 `tensor([[0.5000, 1.0000]], dtype=torch.float64)`。你就掌握了 4 位量化的位运算本质。

- **知识点 4：CUDA Graph 捕获与显存预算（来自 Commit 19）**
  - **是什么**：CUDA Graph 把一串 kernel 录下来一次性重放，省掉 CPU 调度开销。但录制时要预留固定大小的显存缓冲区。
  - **为什么重要**：Gemma 4 的 bug 是"按最大帧数 × 每帧 token 数"算缓冲区，结果 4096 token 的预算算出 6.69 GiB，直接 OOM。正确做法是"行数 = token 预算 / 每行 token 数"。
  - **动手试试**：
    ```python
    def rows_for_budget(token_budget, row_tokens):
        return max(1, token_budget // row_tokens)

    # 错误：固定按最大帧数
    bad = 58 * 32
    # 正确：按预算切分
    good_280 = rows_for_budget(4096, 280)   # 14
    good_70  = rows_for_budget(4096, 70)    # 58

    print(f"错误行数={bad}, 正确(280tok)={good_280}, 正确(70tok)={good_70}")
    print(f"显存比: {bad*280 / (good_280*280):.1f}x")
    ```
  - **预期结果**：看到错误方案比正确方案多用约 132 倍行数。理解"按预算反推行数"这个思路，是所有显存池化设计的核心。

- **知识点 5：Rust 跨平台 socket 家族探测（来自 Commit 5）**
  - **是什么**：`SO_DOMAIN` 这个 socket 选项只在 Linux/Android/FreeBSD/Fuchsia 上有，macOS 没有。跨平台代码要从 `local_addr()` 反推地址族。
  - **为什么重要**：vLLM 的 Rust frontend 要支持 macOS 开发机。这类"某个 API 在某平台不存在"的问题，是系统编程的日常。
  - **动手试试**：
    ```rust
    // Cargo.toml: socket2 = "0.5"
    use std::os::fd::AsRawFd;

    fn main() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let sock: socket2::Socket = unsafe { socket2::Socket::from_raw_fd(l.as_raw_fd()) };
        // 跨平台写法
        println!("family = {:?}", sock.local_addr().unwrap().domain());
        std::mem::forget(l); // 避免 double close
    }
    ```
  - **预期结果**：在 macOS 和 Linux 上都能打印 `IPv4`。你会明白为什么 commit 用 `local_addr()?.domain()` 替换 `socket.domain()?`。

---

## 🔥 重要更新

1. **ROCm 10.0 成为默认发布镜像（Commit 2）**：整个 release pipeline 重构为"ROCm 10.0 默认 + ROCm 7.2 过渡保留"，并新增了 CUDA/ROCm 双通道的 perf-eval 触发。这是 AMD 支持从"实验性"转为"正式"的标志性事件。

2. **Qwen3 工具调用解析性能提升 190 倍（Commit 7）**：12 KB 的 write_file 调用从 2.1 秒降到 11 毫秒。原因是把"每次增量都重新跑完整解析器"改成"只扩展未闭合参数的值"。对使用 Qwen3 function calling 的线上服务是巨大收益。

3. **专家 LoRA 支持 FlashInfer TRTLLM MXFP4（Commit 17）**：MoE 架构 + 4 位量化 + LoRA 三者叠加，这是目前最前沿的组合，意味着可以在消费级显存上对 MoE 模型做定制微调。

4. **KV 传输结果收集提前到 drafter 之前（Commit 9）**：修复了"投机解码 + KV 连接器"组合下的顺序问题，新增 `defer_finalize` / `finalize_step` 两阶段 API。这是 PD 分离 + 投机解码同时使用的必要修复。

---

## 📋 逐条解读

### 1. [ROCm][CI] Give the MoE parallel-launch watchdog room for AITER JIT builds
- **代码层面**：改 `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`，把 `LAUNCH_TIMEOUT_S` 从固定 900 秒改成"ROCm 上 1800 秒，其他 900 秒"。判断依据是 `current_platform.is_rocm()`。
- **新手概念课堂**：JIT（即时编译）就像"点菜时才现做"——AITER 是 AMD 的算子库，有些 MoE kernel 没预先编译好，第一次调用要现场编译，一次要 10 分钟。测试的超时 watchdog 如果比编译时间短，就会误杀。
- **对你有什么影响**：如果你在 AMD 卡上跑 MoE 测试，之前会莫名超时失败，现在不会了。普通用户无感知。

### 2. [ROCm][CI] Make ROCm 10.0 the default image; keep ROCm 7.2 as -rocm72
- **代码层面**：大改 `.buildkite/release-pipeline.yaml`（4500+ 行 diff）。把原来单一的 "Build ROCm Wheel / Image" 组拆成 `build-rocm100`（默认，产出 `:nightly`/`:latest`）和 `-rocm72` 变体；新增 CUDA 和 ROCm 两条 perf-eval 触发流水线。
- **新手概念课堂**：发布流水线就像"工厂装配线"。以前只有一条 AMD 装配线，现在有两条并行（新版本 + 旧版本过渡），还加了一条"性能体检"支线，每晚自动跑基准测试。
- **对你有什么影响**：拉 `vllm/vllm-openai-rocm:latest` 现在拿到的是 ROCm 10.0 版本。如果你的驱动还是 ROCm 7.x，要显式拉 `-rocm72` 标签。

### 3. [Bugfix][EPLB] Wait for pending expert buffer copies before NIXL READs
- **代码层面**：改 `vllm/distributed/eplb/eplb_communicator.py`。`set_stream` 从空实现改为真正保存 stream；`set_transfer_context` 里在 ROCm 上加了 `current_stream().synchronize()` 和 `self._stream.synchronize()`，包在 `gpu_sync_allowed()` 上下文里。
- **新手概念课堂**：EPLB（专家并行负载均衡）会动态把"专家"从一张卡搬到另一张卡。搬运分两步：先拷到本地缓冲区，再用 NIXL 发 RDMA READ。NIXL 不走 CUDA stream，所以必须手动"等一等"，否则读到旧数据。
- **对你有什么影响**：修复了 AMD 上 MoE 负载均衡时偶发的输出错乱。使用 EPLB 的用户稳定性提升。

### 4. [NIXL] Drop unreachable producer-side notif counting in push mode
- **代码层面**：改 `push_worker.py` 的 `_get_new_notifs()`，删掉 push 模式下永远走不到的生产者侧计数分支；测试里相应删掉 `transfer_topo = MagicMock()` 和 `notified == set()` 断言。
- **新手概念课堂**：这是"死代码清理"。push 模式下通知只从消费者侧来，生产者侧那段代码逻辑上不可能执行，留着只会误导维护者。
- **对你有什么影响**：无功能变化，纯代码质量提升。但说明 NIXL 连接器正在被认真梳理，稳定性在变好。

### 5. [Bugfix][Rust Frontend] Fix macOS build: read inherited listener family from its address
- **代码层面**：改 `rust/src/engine-core-client/src/transport.rs` 的 `adopt_listener()`。把 `socket.domain()?`（依赖 `SO_DOMAIN`，macOS 没有）换成 `socket.local_addr()?.domain()`。新增一个 Unix socket 继承测试。
- **新手概念课堂**：`SO_DOMAIN` 是"问 socket 你是什么地址族"，但 macOS 不认这个问题。`local_addr()` 是"看你绑在哪"，所有平台都支持，从地址反推族更通用。
- **对你有什么影响**：Rust frontend 现在能在 macOS 上编译通过，Mac 开发者可以本地调试了。

### 6. [Bugfix] Enable inductor batch-invariant mode under VLLM_BATCH_INVARIANT
- **代码层面**：改 `vllm/model_executor/determinism/batch_invariant.py` 的 `override_envs_for_invariance()`，新增两行：导入 `torch._inductor.config` 并设置 `inductor_config.batch_invariant = True`。
- **新手概念课堂**：PyTorch 2.x 的 inductor 编译器有个"批次不变"开关，打开后它不会做"根据 batch size 选不同归约策略"的优化。之前 vLLM 只设了环境变量，忘了通知 inductor 本身。
- **对你有什么影响**：开启 `VLLM_BATCH_INVARIANT=1` 后，输出真正逐位可复现了。做 RL 训练对齐的用户受益最大。

### 7. [Perf][Parser] Stream Qwen3 tool argument values incrementally
- **代码层面**：改 Qwen3/MiMo 的流式工具调用解析器。新增 `TestArgValueExtender` 测试类。核心是把"每次收到新 chunk 就重新解析整个参数值"改成"只把新增字符追加到已打开的参数值上"。
- **新手概念课堂**：想象你在抄写一段长文本，旧做法是每来一个字就把整段重抄一遍（O(n²)），新做法是接着往下写（O(n)）。12 KB 从 2.1 秒到 11 毫秒就是这个差别。
- **对你有什么影响**：用 Qwen3 做 agent、生成大文件（代码、报告）时，流式输出的首字延迟和总耗时大幅降低。

### 8. [Bugfix][Benchmarks] Name the real cause when the /tokenize alignment probe fails
- **代码层面**：改 `vllm/benchmarks/serve.py`。把笼统的 `except Exception` 拆成 `ClientConnectionError`（服务器连不上）、`ClientResponseError`（404 时提示"可能没这个路由，也可能模型名不对"）、其他异常三类，分别给出具体提示。
- **新手概念课堂**：错误信息是给"凌晨三点被叫醒的工程师"看的。"404"可能是路由不存在，也可能是你写错了模型名——把这两种可能都说出来，能省半小时排查。
- **对你有什么影响**：跑 benchmark 时如果对齐探测失败，warning 会告诉你真正原因，而不是一句"skipping alignment"。

### 9. [MRV2] Collect KV transfer results before the drafter runs
- **代码层面**：改 `vllm/v1/worker/gpu/kv_connector.py`。`post_forward` 新增 `defer_finalize` 参数，新增 `finalize_step()` 方法。投机解码时，主模型 forward 后先收集 KV 传输结果，等 drafter 写完 KV 再执行 save 和清理。
- **新手概念课堂**：投机解码分两阶段（主模型验证 + drafter 生成下一轮草稿）。KV 传输的"收结果"必须早，"存结果"必须晚——早了会读到半成品，晚了会阻塞 drafter。
- **对你有什么影响**：同时使用 PD 分离 + 投机解码时不再有 KV 数据竞争，稳定性提升。

### 10. [Bugfix][Spec Decode] Disable dynamic speculative decoding for fixed-K proposers instead of crashing
- **代码层面**：改 `vllm/config/vllm.py`，新增 `_maybe_disable_dynamic_sd_for_v1()`。如果开启了动态投机解码、用的是 V1 model runner、且提案器不在白名单（ngram/eagle/eagle3/mtp/draft_model/dflash）里，就 warning 并回退到静态 K。
- **新手概念课堂**：动态 K 需要提案器能"按请求返回不同数量的草稿"。自定义提案器通常只支持固定 K。vLLM 选择"降级 + 警告"而不是"崩溃"，这是生产软件的成熟做法。
- **对你有什么影响**：用自定义提案器时不会再启动即崩，而是自动降级并告诉你原因。

### 11. [Rust Frontend] Validate bad_words limits during request lowering
- **代码层面**：改 `rust/src/text/src/lower/token_ids.rs`。新增两个错误类型 `TooManyBadWords`（默认上限 128）和 `TooManyBadWordTokens`（默认上限 1024），可通过 `VLLM_MAX_NUM_BAD_WORDS` / `VLLM_MAX_BAD_WORDS_TOTAL_TOKENS` 环境变量配置。
- **新手概念课堂**：`bad_words` 是"禁止生成的词表"。如果不限长度，恶意用户传一个百万词的列表就能打爆内存。这是"输入校验"的经典案例。
- **对你有什么影响**：Rust frontend 现在会明确拒绝超限请求并给出清晰错误信息，而不是默默吃掉内存。

### 12. [Benchmark] Add per-case kernel profiles to vllm bench forward
- **代码层面**：给 `vllm bench forward` 加 `--profile-dir DIR` 参数。每个 case 额外重放 3 次并用 `torch.profiler` 记录，输出 `kernels.json`（按 GPU 时间排序的 kernel 列表）和 `trace.json.gz`（Perfetto 可打开）。新增 `python -m vllm.benchmarks.forward.profile` 工具做单文件汇总或双文件对比。
- **新手概念课堂**：这是"给推理做体检"。以前只知道"这个 batch 花了 5 毫秒"，现在知道"其中 2 毫秒在 attention kernel、1.5 毫秒在 allreduce"。优化从"猜"变成"看"。
- **对你有什么影响**：做性能优化时能精确定位热点 kernel，A/B 对比有数据支撑。

### 13. [Feature] Skip MLA weight absorption when weights are pre-processed
- **代码层面**：改 `vllm/models/kimi_k3/nvidia/mla.py` 的 `process_weights_after_loading`。如果 `is_weights_pre_processed()` 为真且 `W_UV` 已存在，就跳过 `split_kv_b_proj` 和权重重排。
- **新手概念课堂**：MLA（多头潜在注意力）有个"权重吸收"步骤，把 `kv_b_proj` 拆成 `W_UK`/`W_UV` 并转置。如果权重缓存守护进程已经做过这一步，再做一次会就地修改共享的 CUDA 内存，导致缓存损坏。
- **对你有什么影响**：使用权重缓存（weight cache daemon）加载 Kimi K3 时不再有隐性内存破坏，加载更快。

### 14 & 15. Revert + Re-add "[Docs] Add train-inference consistency agent skill"
- **代码层面**：Commit 15 新增了一整套 `.agents/skills/train-inference-consistency/` 文档（SKILL.md + 十几个 guides），Commit 14 又把它整个删掉。这是"先加后撤"的成对操作。
- **新手概念课堂**：这是 AI agent 的"技能包"——给编码助手看的操作手册，教它如何帮用户做"训练和推理逐位对齐"。撤回可能是因为内容还没打磨好或需要重新组织。
- **对你有什么影响**：如果你在用 vLLM 的 AI 辅助开发，这个技能暂时不可用。但 `docs/features/batch_invariance.md` 仍然在，可以自己读。

### 16. [Doc] Sync DeepSeek-V4.1-Flash recipe with recipes#1136, #1137
- **代码层面**：改 `recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`。`min_vllm_version` 从 0.30.0 提到 0.31.0，默认模型换成 `DeepSeek-V4.1-Flash-1017`（10 月 17 日权重刷新），MI355X 的 pip 安装被禁用（因为 `--monokernel` 还没进 0.31.0 正式版）。
- **新手概念课堂**：recipe 是"开箱即用的部署配方"，包含模型 ID、镜像、环境变量、各硬件配置。版本号写错会导致用户拉到不兼容的 wheel。
- **对你有什么影响**：按 recipe 部署 DeepSeek-V4.1-Flash 会拿到更新的权重和正确的版本约束；MI355X 用户被引导到 nightly 镜像。

### 17. [Feature][LoRA][MoE] Expert LoRA for FlashInfer TRTLLM MXFP4 experts
- **代码层面**：新增 `TrtLlmMxfp4LoRAExperts` 类（在 `trtllm_lora_moe.py`）和 218 行的测试文件。测试用 float64 参考实现验证 4 位量化的反量化和 LoRA 增量计算。
- **新手概念课堂**：MoE 模型有几十上百个"专家"，LoRA 给每个专家的权重加一个小增量。MXFP4 是 4 位量化。三者叠加意味着：可以在 4 位量化的 MoE 模型上做 LoRA 微调，显存需求极低。
- **对你有什么影响**：如果你在 SM100（Blackwell）上微调 MoE 模型，现在可以用 4 位量化 + LoRA，显存占用大幅下降。

### 18. [XPU] Fix Event init failure w/ external
- **代码层面**：改 `vllm/v1/worker/xpu_model_runner.py` 的 `_xpu_event` 包装函数，除了已有的 `blocking` 参数，再吞掉 `external` 参数（`torch.xpu.Event` 不支持）。
- **新手概念课堂**：Intel XPU 的 PyTorch 后端 API 是 CUDA 的"不完全模仿"。CUDA 有的参数它不一定有。vLLM 用一个 wrapper 函数把不支持的参数悄悄丢掉，让上层代码不用关心后端差异。
- **对你有什么影响**：Intel GPU 用户不再因为 `external=True` 参数而初始化失败。

### 19. [Bugfix] Fix Gemma 4 encoder CUDA graph capture OOM at startup
- **代码层面**：改 `vllm/model_executor/models/gemma4_mm.py`。CUDA graph 捕获的行数从"最大帧数"改成 `max(1, token_budget // row_tokens)`；新增按"填充后宽度"打包混合尺寸图片的逻辑。测试验证 4096 token 预算下 280-token 行只需 14 行而非 1856 行。
- **新手概念课堂**：CUDA graph 录制要预分配固定大小的输入缓冲区。以前按"最坏情况"（每张图都最大）分配，导致 6.69 GiB 的 buffer 直接 OOM。正确做法是"按 token 预算反推行数"。
- **对你有什么影响**：Gemma 4 多模态模型启动不再 OOM，可以正常服务了。

### 20. [Perf][Bugfix] Prefer MNNVL for single-node FlashInfer allreduce again
- **代码层面**：改 `vllm/distributed/device_communicators/flashinfer_all_reduce.py` 的 `_resolve_fi_ar_backend()`。单节点场景从 `backend = "trtllm"` 改回 `backend = "mnnvl"`，并允许回退到 TRTLLM。测试期望值相应更新。
- **新手概念课堂**：单机多卡 allreduce 有两条路：TRTLLM 的 kernel 和 MNNVL（基于 NVSwitch 组播）。实测 MNNVL 在 H200/B200 的 TP4/TP8 上快 3-5%，但需要 NVSwitch 组播支持。所以策略是"优先 MNNVL，不支持就回退"。
- **对你有什么影响**：单机多卡推理吞吐提升 3-5%。这是"回退一次之前的错误决策"——之前为了稳妥改成了 TRTLLM，现在有数据支撑改回来。

## sglang
## 💡 今日关键词

- **Goodput（有效吞吐）**：只统计"按时完成、满足延迟目标"的请求数，而不是所有完成的请求数。
  - **为什么社区在关注**：LLM 服务从"能跑"进入"能卖"阶段，SLA/SLO 成了核心指标。bench_serving 新增 goodput 说明大家开始用业务视角而不是纯工程视角评估性能。
  - **对新手意味着什么**：性能优化不再是"QPS 越高越好"，你需要学会用 TTFT/TPOT/E2EL 这类延迟指标定义服务质量，这是进入推理工程岗的敲门砖。

- **Prefill/Decode 分离与图捕获（CUDA Graph / Breakable Graph）**：把推理拆成预填充和解码两个阶段，并用 CUDA Graph 把内核调用序列"录下来"重放以降低启动开销。
  - **为什么社区在关注**：DeepSeek-V4.1 相关 commit 密集出现，说明大模型架构（SWA、稀疏注意力）越来越复杂，图捕获需要"可断开"才能容纳动态逻辑。
  - **对新手意味着什么**：CUDA Graph 是推理框架性能的"必修课"，理解 capture/replay、静态 buffer、动态 shape 处理是进阶关键。

- **AMD ROCm 适配**：SGLang 在 AMD MI355X 上的 cookbook 更新到 ROCm 10 镜像，并加入 HIP 专用环境变量。
  - **为什么社区在关注**：NVIDIA 之外的第二算力平台正在被认真对待，"异构计算"从口号变成日常。
  - **对新手意味着什么**：如果你只会 CUDA，学一点 HIP/ROCm 会让你在求职时多一个差异化优势；很多 HIP API 与 CUDA 一一对应，迁移成本不高。

- **自定义算子直调（Custom Op Fast Path）**：在纯 eager 模式下绕过 PyTorch dispatcher 直接调用算子实现，省掉几微秒的分发开销。
  - **为什么社区在关注**：推理框架进入"微秒级抠性能"阶段，dispatcher 开销在 decode 阶段占比可观。
  - **对新手意味着什么**：理解 `torch.library`、dispatcher、dispatch mode、autograd 这些 PyTorch 底层机制，是从"调包侠"到"框架开发者"的分水岭。

- **多模态 tokenized 输入**：gRPC 的 GenerateRequest 新增 image/video/audio 字段，支持"已 tokenize + 多模态数据"一起送入。
  - **为什么社区在关注**：多模态推理正在从 demo 走向生产，需要与现有 token 化管线解耦。
  - **对新手意味着什么**：多模态 + 推理服务的交叉领域人才稀缺，值得投入。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：Goodput 与 SLO 达成率（性能评测方向）**
  - **是什么**：Goodput = 满足所有延迟目标（如 TTFT<500ms、TPOT<50ms、E2EL<20s）的成功请求数 / 秒。SLO attainment = 达标请求数 / 总发送请求数。
  - **为什么重要**：普通 throughput 会把"超时返回"的请求也算进去，看起来很美但用户已经走了。只看 goodput 才能找到"部署真正能承受的最大请求速率"。不懂这个，你会把系统压到过载还以为在优化。
  - **动手试试**：
    ```bash
    # 启动一个本地 sglang 服务后
    python3 -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port 30000 \
      --num-prompts 200 --request-rate 16 \
      --goodput ttft:500 tpot:50 e2el:20000
    ```
    再扫一遍 `--request-rate`：`4, 8, 16, 32, 64`，观察 `request_goodput` 和 `slo_attainment`。
  - **预期结果**：低速率时 goodput 随速率线性上升；超过某个点后 goodput 开始下降而 throughput 仍上升——那个拐点就是你的部署容量。

- **知识点 2：CUDA Graph 与 Breakable Graph（性能优化方向）**
  - **是什么**：CUDA Graph 把一串 GPU 内核调用录制成图，之后一次 replay 就能执行，省掉每次 launch 的 CPU 开销。Breakable Graph 允许图中间"断开"执行动态逻辑，再继续。
  - **为什么重要**：decode 阶段每步只有几微秒计算，kernel launch 开销可能占一半。DeepSeek-V4.1 的 SWA bounded replay 需要动态分支，普通 graph 装不下，所以引入 breakable 机制。
  - **动手试试**：
    ```python
    import torch
    # 感受 graph 的收益：对比 eager vs graph replay 的 launch 开销
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")

    # eager
    import time
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10000):
        c = a + b
    torch.cuda.synchronize(); print("eager:", time.perf_counter() - t0)

    # graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10000):
            c = a + b
    torch.cuda.synchronize(); t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize(); print("graph:", time.perf_counter() - t0)
    ```
  - **预期结果**：graph 版本明显更快（尤其在 kernel 很小、数量很多时）。注意 graph 要求固定 shape 和固定地址，这就是为什么框架里要预分配 buffer。

- **知识点 3：PyTorch Dispatcher 与自定义算子直调（框架底层方向）**
  - **是什么**：PyTorch 的 dispatcher 是一个多路分发层，根据 tensor 的 device、dtype、dispatch mode 等把调用路由到具体实现。每次调用都有几微秒开销。Commit 15 在"纯 eager + 无 grad + 纯 CUDA tensor"时直接调 `op_func` 绕过它。
  - **为什么重要**：decode 阶段小算子密集，dispatcher 开销累积可观。但绕过有风险：profiler 会记录不到、autograd 会失效、tensor subclass 会被误判——所以代码里做了严格的条件判断。
  - **动手试试**：
    ```python
    import torch
    from torch.library import custom_op

    @custom_op("myops::add_one", mutates_args=())
    def add_one(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    @add_one.register_fake
    def _(x): return torch.empty_like(x)

    x = torch.randn(1024, device="cuda")
    # 观察 dispatcher 是否被调用
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(100):
            add_one(x)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    ```
  - **预期结果**：profiler 里能看到 `myops::add_one` 这个 op 被记录。如果你手动绕过（直接调 `x + 1`），profiler 就看不到这个逻辑 op 了——这就是 commit 里要保留 profiler 路径的原因。

- **知识点 4：多模态 gRPC 请求结构（多模态服务方向）**
  - **是什么**：proto 里 `GenerateRequest` 新增 `image_data`、`video_data`、`audio_data` 三个 repeated string 字段，可以是 URL 也可以是 base64 data URI。Rust 侧把它们塞进 JSON dict 传给后端。
  - **为什么重要**：tokenized 输入（input_ids 已算好）+ 多模态数据同时传，意味着前端可以自己做 tokenize 和媒体预处理，后端只负责推理，职责解耦。
  - **动手试试**：
    ```python
    # 伪代码：构造一个带图片的 gRPC 请求
    from sglang.runtime.v1 import sglang_pb2
    req = sglang_pb2.GenerateRequest(
        input_ids=[2, 10, 20],
        image_data=["https://example.com/cat.png"],
        sampling_params=sglang_pb2.SamplingParams(temperature=0.5, max_new_tokens=64),
    )
    print(req)
    ```
    实际跑需要起 gRPC server，但看 proto 结构本身就能理解设计。
  - **预期结果**：打印出的 message 里能看到 image_data 字段。理解"URL 和 base64 都支持"意味着后端需要统一的媒体加载器。

- **知识点 5：Triton 内核启动 hook 与快速重放（内核/编译器方向）**
  - **是什么**：Triton 每次 launch 内核都会构造 metadata 并检查 hook（用于 profiler 等）。Commit 17 在"没有 hook 时"直接调 `compiled.run(...)` 并传 `None` 给 hook 参数，省掉 metadata 构造。
  - **为什么重要**：diffusion 模型每步要 launch 成百上千个小 Triton 内核，每次省几微秒累积起来很可观。但必须检测 hook 是否真的为空，否则 profiler 会失效。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl
    from triton import knobs

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    x = torch.randn(1024, device="cuda"); y = torch.randn(1024, device="cuda")
    out = torch.empty_like(x)
    # 第一次启动会编译
    add_kernel[(4,)](x, y, out, 1024, BLOCK=256)
    torch.cuda.synchronize()
    print("hooks:", knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    ```
  - **预期结果**：正常情况下 hook 为 None 或空 HookChain，说明可以走 fast path。如果你用 proton 之类的 profiler 挂上 hook，就会看到非空，此时必须退回标准路径。

---

## 🔥 重要更新

1. **#43967 bench_serving 新增 goodput 与 SLO 达成率**：性能评测从"吞吐量"升级到"业务有效吞吐"，这是把 benchmark 变成容量规划工具的关键一步，直接影响你怎么给线上服务定容量。

2. **#44948 / #44930 DeepSeek-V4.1 的 DP 行数交换与 breakable prefill graph**：两个 commit 配合，让 SWA bounded replay 能跑进可断开的 prefill CUDA Graph，同时修了跨 rank 行数不同步的 bug——这是大模型复杂注意力 + 图捕获协同的典型难题。

3. **#42833 gRPC 支持 tokenized 多模态生成**：proto 增加 image/video/audio 字段，多模态推理服务正式支持"前端 tokenize + 媒体分离传输"的架构，对生产部署影响大。

4. **#44158 自定义算子在纯 eager 下直调**：通过严格条件判断绕过 dispatcher，是推理框架"微秒级抠性能"的代表作，也是理解 PyTorch 分发机制的绝佳教材。

---

## 📋 逐条解读

### 1. [Feat] bench: report goodput and SLO attainment in bench_serving (#43967)
- **代码层面**：新增 `python/sglang/benchmark/goodput.py`，定义 `GoodputMetrics` 结构体和 `calculate_goodput()` 函数；`bench_serving` 新增 `--goodput ttft:500 tpot:50 e2el:20000` 参数；文档补充了 Goodput 章节和 JSONL 输出字段。单 token 响应的 TPOT 定义为 0。
- **新手概念课堂**：想象你去餐厅吃饭，`throughput` 是"今天厨房出了多少道菜"，`goodput` 是"有多少道菜在客人还愿意吃的时候端上来了"。菜出得再多，客人走了就是白做。SLO 就是你给客人承诺的"30 分钟内上菜"。
- **对你有什么影响**：以后跑 benchmark 会多出 `request_goodput` 和 `slo_attainment` 两列。做容量规划时，扫一遍 request-rate 找 goodput 峰值，比看 throughput 拐点更靠谱。

### 2. [Doc] sgl-router: Rewrite README and add architecture guide (#43592)
- **代码层面**：新增 `experimental/sgl-router/ARCHITECTURE.md`，236 行，讲清 `--chat-routing reorg` 的部署架构、请求生命周期、bucket 选择、负载感知路由、P/D 分离转发、admission control、失败重试。附 mermaid 流程图。
- **新手概念课堂**：Router 就像餐厅的领位员。Bucket 是"包间/大厅"这类分区，engine 是"桌子"。领位员先看你几个人（长度限制），再看哪个区有空桌（admission），然后在两个候选桌里选人少的那张（load-aware）。全满就告诉你"没位置"（503）。
- **对你有什么影响**：如果你要部署多模型/多 bucket 的推理集群，这份文档是必读的；里面还解释了 P/D 分离时 prefill 把 KV 直接传给 decode 的流程。

### 3. [Doc] [AMD] dsv41: Update amd cookbook 1016 (#44903)
- **代码层面**：DeepSeek-V4.1 的 AMD cookbook 镜像从 `v0.5.21-rocm720` 升到 `v0.5.22-rocm10`；新增两个 HIP 环境变量 `SGLANG_HIP_FP4_INDEXER_ROWGROUP=1` 和 `SGLANG_HIP_DSV41_INDEX_K_PAGE8=1`；并发 ≤16 时改用 `--monokernel-backend amd --disable-shared-experts-fusion`，32 时换回 `--enforce-shared-experts-fusion`。
- **新手概念课堂**：这就像同一款游戏在 PlayStation 和 Xbox 上要开不同的画质选项。ROCm 是 AMD 的 CUDA 对应物，`rocm10` 是版本号。环境变量是给底层 kernel 的"暗号"，开启特定优化。
- **对你有什么影响**：在 MI355X 上部署 DeepSeek-V4.1 的用户，照抄新命令能拿到更好的低延迟表现；注意并发阈值切换 flag 这个细节，别一股脑用一套参数。

### 4. [Fix] sgl-router: Fill the media fields when converting TextGenerate to Generate (#44951)
- **代码层面**：`experimental/sgl-router/src/proxy/grpc.rs` 的 `text_to_generate()` 里，补上 `image_data: Vec::new()`、`video_data: Vec::new()`、`audio_data: Vec::new()` 三个字段。因为 proto 新增了这三个字段，Rust 结构体初始化必须全部给出。
- **新手概念课堂**：这就像你给朋友寄包裹，表格新增了"易碎品""液体""电池"三栏，即使你没寄这些东西，也得每栏都写"无"，否则表格不完整寄不出去。Rust 的结构体初始化是严格的，少一个字段就编译不过。
- **对你有什么影响**：纯文本请求经过 router 时不会因为缺字段而 panic 或编译失败。属于配套 commit 10 的修复。

### 5. [Fix] deepseek-v4.1: Exchange late-layer DP row counts before the forward (#44948)
- **代码层面**：在 `base_attn_backend.py` 新增 `late_layer_num_tokens()` 钩子（默认返回 None）；`deepseek_v4_backend.py` 实现它，根据 extend 模式和 dspark 计算"晚层实际要跑的行数"；HIP radix 后端也实现；`forward_batch_info.py` 在 forward 前跨 DP rank 交换这个行数。
- **新手概念课堂**：DP（数据并行）下每个 GPU 处理不同请求，但某些层需要所有 GPU 一起做集合通信。如果各 GPU 的行数不一致，通信就会错位。这就像合唱团，每个人唱的拍子必须对齐，所以开唱前先喊一声"我这段有 8 拍"。
- **对你有什么影响**：修复了 DeepSeek-V4.1 在 DP + SWA bounded replay 场景下可能出现的通信错位/结果错误，属于正确性修复。

### 6. Revert "[Doc] cookbook: Add Muse Spark Open SWA ratio calculator" (#44947)
- **代码层面**：删除了 MuseSparkOpen.mdx 里的 SWA ratio 计算器章节、对应的 JSX 组件和测试文件 `muse-spark-swa-ratio.test.mjs`。回滚了一个之前加入的文档功能。
- **新手概念课堂**：Revert 就是"撤销"。可能计算器的公式有误，或者产品决策变化。SWA（滑动窗口注意力）的 KV 池和全注意力 KV 池需要按比例分配显存，这个计算器本来是帮用户估 `--swa-full-tokens-ratio` 的。
- **对你有什么影响**：如果你之前看过这个计算器，现在文档里没有了。需要自己按 `min(1, 2048/L)` 估算，或者等官方修正后重新上线。

### 7. [Perf] diffusion: classify residual_gate_add operands once per call (#44478)
- **代码层面**：`residual_gate_add_jit.py` 把"判断走哪个 kernel"的逻辑从 custom op 内部提取到 `_route()`，在调用前算一次 `mode` 传给 op；新增 `_TRANSPOSED = -1` 常量；op 签名从 `(residual, update, gate)` 变为 `(residual, update, gate, mode)`；`_fake_impl` 同步更新。还加了 `residual.get_device() == torch.cuda.current_device()` 的快路径。
- **新手概念课堂**：原来每次调用都要"看一遍三个张量的形状/步长/dtype"再决定走哪条路，现在提前判断好，把"结论"（mode）传进去。就像快递分拣，原来每件包裹到站才看地址，现在提前贴好区域标签。
- **对你有什么影响**：diffusion 模型里 residual + gate 这种逐元素操作调用极频繁，省掉重复判断能降低 CPU 侧开销，尤其在 eager 模式下。

### 8. [Perf] deepseek-v4.1: Run encoder SWA bounded replay in the breakable prefill graph (#44930)
- **代码层面**：`deepseek_v4_hook.py` 放宽了 prefill CUDA graph 的限制，允许 `Backend.BREAKABLE`（但 ROCm 上仍禁用）；`deepseek_v4_backend.py` 在 `refresh_for_breakable_cuda_graph_replay_` 里增加 `request_window_layout` 的拷贝；`init_forward_metadata_prefill` 里 graph 模式下 `num_groups` 改为按 request slot 数计算；新增 `_activate_request_window()`。
- **新手概念课堂**：CUDA Graph 要求所有张量地址固定，但 SWA 的"窗口历史"是动态的。Breakable Graph 允许在图中途"暂停"，执行一段动态逻辑再继续。这个 commit 就是让 encoder 的 SWA replay 能塞进这个"可暂停的图"里。
- **对你有什么影响**：DeepSeek-V4.1 在 prefill 阶段能更充分地利用 CUDA Graph，降低首 token 延迟。但 ROCm 上仍走老路（因为 HIP 的 replay 机制不同）。

### 9. [Perf] DFLASH: build the draft num_token_non_padded scalar with torch.full (#44119)
- **代码层面**：`dflash_worker_v2.py` 里把 `torch.tensor(bs * block_size, dtype=..., device=...)` 改成 `torch.full((), bs * block_size, dtype=..., device=...)`。一行改动。
- **新手概念课堂**：`torch.tensor(5)` 会先创建一个 Python 列表/标量再转换，`torch.full((), 5)` 直接分配一个 0 维张量。后者更快、更省内存，是"标量张量"的标准写法。
- **对你有什么影响**：微优化，单次省不了多少，但 speculative decoding 每步都调，累积起来有意义。也是学习 PyTorch 惯用法的好例子。

### 10. [Feat] runtime: Support tokenized multimodal generation (#42833)
- **代码层面**：`sglang.proto` 的 `GenerateRequest` 新增 `repeated string image_data = 19`、`video_data = 20`、`audio_data = 21`；Rust 生成的 `sglang.runtime.v1.rs` 同步更新；`request_utils.rs` 的 `build_generate_dict()` 把非空的媒体字段塞进 JSON dict；新增单测验证 tokenized 多模态请求保留生成参数。
- **新手概念课堂**：以前 gRPC 只能传 input_ids（纯文本 token），现在可以同时传"图片 URL"或"base64 图片数据"。这样前端可以自己做 tokenize 和媒体预处理，后端只负责推理，职责清晰。
- **对你有什么影响**：如果你在构建多模态推理服务，现在可以用 gRPC 高效传输"token + 图片"组合，不必走 HTTP JSON 的大 payload。

### 11. [Fix] dLLM: Size decode graphs to the default max running requests (#40941)
- **代码层面**：`base_cuda_graph_runner.py` 的 `get_batch_sizes_to_capture()` 里，如果 dLLM 配置需要独立 context encoding，则 `num_max_requests = min(num_max_requests, dllm_config.max_running_requests)`；`decode_cuda_graph_runner.py` 删除了原来在 `__init__` 里裁剪 capture_bs 的逻辑（挪到更早的地方）。
- **新手概念课堂**：dLLM（diffusion LLM）的 decode graph 批大小原来可能按 `req_to_token_pool.size` 来，但 dLLM 实际最大并发受 `max_running_requests` 限制。抓太大的 graph 浪费显存，抓太小又不够用。这个 commit 让 graph 尺寸对齐真实上限。
- **对你有什么影响**：dLLM 用户显存占用更合理，不会因为抓了过大的 graph 而 OOM。

### 12. [Feat] diffusion/model: support Cosmos Dreams and Cosmos Multiview (#39475)
- **代码层面**：新增对 `nvidia/Cosmos3-Nano-Sim-Bimanual`、`Cosmos3-Nano-Sim-Transfer`、`Cosmos3-Nano-Transfer-Auto`、`Cosmos3-Nano-Sim-Auto` 等 checkpoint 的支持；文档补充了输入格式（action rows、multiview object、lidar）、并行策略限制（TP 支持、SP/CFG 部分不支持）、`--component-residency` 用法；model-catalog.jsx 新增条目。diff 近 2.5 万行。
- **新手概念课堂**：Cosmos 是 NVIDIA 的世界模型/视频生成模型。Bimanual 是"双臂机器人"（输入动作，输出视频），Multiview 是"多摄像头视角"，Transfer 是"控制视频迁移"。这些 checkpoint 各有专属的 conditioning 格式，不能混用。
- **对你有什么影响**：做机器人仿真/视频生成的用户可以直接用 SGLang 部署这些模型。注意并行策略有限制（比如 Bimanual 不支持 SP），别乱开 flag。

### 13. [Doc] cookbook: Add DeepSeek V4.1 Flash 1017 variant (#44937)
- **代码层面**：`deepseek-v4_1.jsx` 配置从 `matchDims` 单维度匹配改为 `variants / quantizations / strategies / nodesOptions` 四维度；新增 `flash-1017` variant 和 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 模型名；所有 match 条件补全四个维度。
- **新手概念课堂**：文档的部署配置矩阵原来只按"策略"分（低延迟/高吞吐），现在多了"模型版本"和"量化方式"两个维度。就像买车从"只看颜色"变成"看颜色+配置+动力"。
- **对你有什么影响**：文档页面会出现 variant 下拉框，选 `Flash-1017` 能看到对应的部署命令。用户更容易找到匹配自己模型的配置。

### 14. [Doc] popular-models: Replace Kimi-K3 with Muse Spark Open (#44693)
- **代码层面**：`popular-models.jsx` 里删除 Kimi-K3 条目，新增 Muse Spark Open（Meta）作为 featured model，带 badge "New"、tags、hero 文案。注意 diff 末尾有个多余的 `},` 导致语法可能有问题（Kimi-K3 删除后残留的闭合括号）。
- **新手概念课堂**：这是文档首页的"热门模型"卡片列表。换掉 Kimi-K3 说明社区关注点转移，或者 Kimi-K3 已经过了"新品"期。
- **对你有什么影响**：首页会展示 Muse Spark Open 的入口。如果你对 Meta 的 SWA + 全注意力混合架构感兴趣，可以从这里进 cookbook。

### 15. [Perf] custom_op: call register_custom_op ops directly in plain eager execution (#44158)
- **代码层面**：`custom_op.py` 新增 `_call_directly_in_eager()`，把 `torch.compiler.is_compiling`、`_len_torch_dispatch_stack`、`_is_torch_function_mode_enabled`、`_profiler_enabled`、`is_grad_enabled` 这些检查函数在模块加载时绑定一次；`CustomOpWrapper.real_impl` 里如果 op 是"本地注册"的，就包一层 fast path。条件：非编译、无 dispatch mode、无 function mode、无 profiler、无 grad、所有 tensor 都是纯 `torch.Tensor`/`Parameter` 且在 CUDA 上。
- **新手概念课堂**：PyTorch 的 dispatcher 像一个"总机"，每次调用都要查表转接。这个 commit 说：如果你确定是"简单场景"（不编译、不求导、不 profile），那就别走总机，直接拨分机号。但一旦有任何"特殊需求"（比如 profiler 要记录），立刻退回总机。
- **对你有什么影响**：eager 模式下自定义算子调用更快，但如果你在用 profiler 调试，行为不变（仍走 dispatcher）。这是一个"安全的微优化"范例。

### 16. [Feat] diffusion: selectable sequence-parallel strategies for LingBot causal attention (#34385)
- **代码层面**：`lingbot_world.py` 新增 `_LINGBOT_SP_STRATEGIES = ("ulysses", "kvgather_replicated", "kvgather_sharded", "ring")`；`lingbot_sp_strategy()` 从环境变量 `SGLANG_LINGBOT_SP_STRATEGY` 读取（默认 ulysses）；新增 `_lingbot_merge_lse()` 做 online-softmax 合并；新增 `full_head_attn` 用于 kv-gather/ring 策略（这些策略保留所有 head，切分 query）。
- **新手概念课堂**：序列并行（SP）是把长序列切到多张卡上算。Ulysses 按 head 切（每个 rank 拿一部分 head，历史不跨 rank）；kv-gather 和 ring 按 token 切（每个 rank 拿一段 token，需要跨 rank 聚合 KV）。不同策略的通信模式和显存开销不同。
- **对你有什么影响**：LingBot 用户可以通过环境变量切换 SP 策略，根据序列长度和卡数选最优。默认 ulysses 最省通信。

### 17. [Perf] diffusion: Replay Triton launches without the runner's per-launch hook metadata (#44471)
- **代码层面**：`fast_launch.py` 的 `_Plan.replay()` 里，如果 `_launch_hooked()` 返回 False（没有 profiler hook），就直接调 `compiled.run(*grid, stream, function, metadata, None, None, None, *values)`，跳过 Triton runner 构造 hook metadata 的开销；`_plan()` 里保存 `(compiled, grid, run, function, metadata, constants, slots)` 而不是只保存 launcher。
- **新手概念课堂**：Triton 每次 launch 内核都要准备一份"元数据"给 hook 用（即使没人监听）。这个 commit 检测到"没人监听"时，直接传 `None` 给 hook 参数，省掉构造开销。就像开会时如果没人做会议记录，就不用准备记录本。
- **对你有什么影响**：diffusion 模型每步 launch 大量小内核，这个优化能明显降低 CPU 侧 launch 开销。但如果你挂了 profiler，会自动退回标准路径，不会丢数据。

### 18. [Perf] diffusion: Cache the fused linear-GELU verdict per linear (#44643)
- **代码层面**：`fused_linear_gelu_site.py` 新增 `_VERDICTS: weakref.WeakKeyDictionary`，把"这个 linear 能不能用 fused GELU"的判断缓存起来；拆出 `_input_independent_ok()` 只算输入无关的部分；`can_use_linear_gelu()` 先查缓存；`mount_fused_linear_gelu()` 和 `unmount_fused_linear_gelu()` 时清空缓存。
- **新手概念课堂**：判断"这个线性层能不能融合 GELU"要读架构、dtype、weight 等，每次 forward 都判断很浪费。缓存起来，mount/unmount 时清空（因为重新 mount 后 linear 可能变了）。用 `WeakKeyDictionary` 是为了 linear 被回收时缓存自动失效，不泄漏内存。
- **对你有什么影响**：diffusion 模型 forward 时每个 linear 少做一次判断，累积起来有收益。WeakKeyDictionary 是 Python 内存管理的实用技巧，值得学。

### 19. [Perf] distributed: cache the eager launch of custom all-reduce v2 per size (#44642)
- **代码层面**：`custom_all_reduce_v2.py` 新增 `_eager_plans: dict` 和 `_eager_plans_config`，按 `(nbytes, dtype)` 缓存 `(module, algo_name, use_multicast)`；`should_custom_ar()` 和 `custom_all_reduce()` 在非 graph 模式下走 `_eager_plan()`；config 变化时重建缓存；`_KERNEL_API_LOG_LEVEL` 非零时退回原路径（保留日志）。
- **新手概念课堂**：all-reduce 每次调用都要根据数据大小选算法（pull/push、multicast 与否）。这个选择逻辑不便宜，但同一大小的 all-reduce 会反复出现（比如固定 batch 的 decode）。缓存起来，第二次直接查表。
- **对你有什么影响**：eager 模式下的张量并行推理会更快，尤其 decode 阶段每层都做 all-reduce。graph 模式不受影响（另有路径）。

### 20. [Refactor] TCPCG: Consolidate attention eager regions and isolate Inkling policy (4/9) (#42288)
- **代码层面**：新增 `python/sglang/srt/layers/attention/graph_utils.py`（240 行），提供 `attention_input_scope()` 上下文管理器（临时切分 q/k/v 等张量、替换 forward_batch 的 out_cache_loc/positions/_attn_output，退出时恢复）和 `allocate_attention_outputs()`；`_zero_padded_tokens()` 处理 padding。这是 9 个 commit 系列的第 4 个，重构 attention 的 eager 区域。
- **新手概念课堂**：CUDA Graph 里跑 attention 时，有时需要"临时跳出图"执行一段 eager 逻辑（比如动态 shape 处理）。这个文件提供了一套"进入/退出 eager 区域"的脚手架：进入时把 batch 状态切到子集，退出时恢复原状，保证异常安全。
- **对你有什么影响**：这是内部重构，普通用户感知不到。但对想理解"graph 和 eager 如何混用"的开发者，`attention_input_scope` 是个很好的学习样本——看它如何在 `try/finally` 里保存和恢复状态。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)