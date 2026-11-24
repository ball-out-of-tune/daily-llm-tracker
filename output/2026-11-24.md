# 🚀 vLLM & SGLang 每日更新 — 2026-11-24

> 自动生成于 2026-11-24 11:34 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：ROCm / AMD 异构计算**
  - **一句话通俗解释**：ROCm 是 AMD 对标 CUDA 的 GPU 计算平台，vLLM 在持续把原本只在 NVIDIA 上跑得好的特性搬到 AMD 显卡上。
  - **为什么社区在关注**：今天有 3 个 commit（UMBP standalone、Zen CPU CI、ShardQ 修复）都和 AMD 平台相关，说明 vLLM 正在认真做"多硬件后端"而不是"NVIDIA 专属"。异构计算（NVIDIA + AMD + Intel + CPU）已是 vLLM 的核心战略。
  - **对新手意味着什么**：如果你只有 AMD 显卡（比如 RX 7900 / MI300），现在很多新特性都能跑了；如果你学 CUDA，也值得了解 HIP/ROCm 的对应概念，这是简历上的加分项。

- **关键词：KV Cache 卸载 / 前缀缓存（Prefix Caching）**
  - **一句话通俗解释**：把已经算过的注意力 KV 缓存存到 CPU 内存或远端，下次遇到相同前缀直接复用，省掉重复计算。
  - **为什么社区在关注**：今天 UMBP（Commit 1）和 KV Offload P2P（Commit 18）两个大 commit 都在优化这块，说明"长上下文 + 高并发"场景下 KV 缓存是最大的内存瓶颈，也是性能提升的主战场。
  - **对新手意味着什么**：理解 KV cache 是理解现代 LLM 推理系统的钥匙。学好了能看懂 vLLM 最核心的 PagedAttention 设计。

- **关键词：MoE 内核后端（FlashInfer / CuTe-DSL）**
  - **一句话通俗解释**：MoE（混合专家）模型里每个 token 只走部分专家，需要专门的高性能内核来调度，今天新增了一个 Hopper 上的 FlashInfer CuTe-DSL 后端。
  - **为什么社区在关注**：DeepSeek、Qwen3-MoE 等模型让 MoE 成为主流架构，各家内核库（FlashInfer、Triton、Cutlass、Marlin）都在抢这个赛道。
  - **对新手意味着什么**：如果你对 GPU 内核优化感兴趣，MoE 是最热门的方向之一，且门槛相对可控（很多是 Python + Triton）。

- **关键词：RL 训练 / 权重热更新（Weight Transfer）**
  - **一句话通俗解释**：强化学习训练时，训练进程要把新权重"热"传给推理引擎，不能停机重启。
  - **为什么社区在关注**：今天有 3 个 commit（Connex 后端、ShardedRDT 死锁修复、权重校验）都在做这块，说明 vLLM 正在成为 RLHF/RLVR 训练框架（如 verl、NeMo-RL）的默认 rollout 引擎。
  - **对新手意味着什么**：RL + 推理是当前最火的交叉方向，懂"训练-推理权重同步"的人才缺口很大。

- **关键词：Rust 前端 / vllm-rs**
  - **一句话通俗解释**：vLLM 正在用 Rust 重写 API 服务器前端，追求更低延迟、更强健壮性。
  - **为什么社区在关注**：今天有 3 个 Rust 相关 commit（glm5 渲染、gRPC 大小限制、RLIMIT_NOFILE），说明 Rust 前端已进入"打磨细节"阶段，不再是实验品。
  - **对新手意味着什么**：如果你会 Rust，现在正是参与 vLLM 前端的好时机，竞争小、影响大。

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 卸载与滑动窗口重建（Sliding Window Attention）**
  - **是什么**：有些模型（如 DeepSeek-V4）用滑动窗口注意力，只关注最近 N 个 token。为了省内存，可以把压缩后的 KV 存起来，需要时再"重放"最近窗口重建状态。
  - **为什么重要**：不搞懂这个，你会困惑为什么"命中缓存后结果略有偏差"——因为重建的窗口是近似的。真实场景中这决定了你能开多长的上下文、能扛多少并发。
  - **动手试试**：
    ```bash
    # 克隆 vllm 并查看 UMBP 文档
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    cat docs/features/umbp_store_connector_usage.md | head -60
    # 跑相关单测（需要 GPU）
    pytest tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py -k dsv4_swa_replay -v
    ```
  - **预期结果**：看到 `dsv4_swa_replay_enabled` 测试通过，理解"只有连接器要求时才开启重放"的逻辑。

- **知识点 2：ThreadPoolExecutor 死锁陷阱**
  - **是什么**：`with ThreadPoolExecutor(...)` 退出时会 `shutdown(wait=True)`，等待所有任务完成。如果某个任务因为异常永远阻塞，主线程就卡死了。
  - **为什么重要**：这是 Python 并发的经典坑，真实生产环境（尤其是分布式 RPC）里一旦踩上就是整机 hang 死，排查极难。Commit 9 就是修这个。
  - **动手试试**：
    ```python
    from concurrent.futures import ThreadPoolExecutor
    import threading, time

    release = threading.Event()

    def blocked():
        release.wait()  # 永远等下去

    # 危险写法：with 会卡死
    def bad():
        with ThreadPoolExecutor(max_workers=1) as exe:
            exe.submit(blocked)
            raise RuntimeError("gather failed")

    # 正确写法：手动 shutdown(wait=False)
    def good():
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            exe.submit(blocked)
            raise RuntimeError("gather failed")
        finally:
            exe.shutdown(wait=False, cancel_futures=True)
            release.set()

    good()
    print("good() 正常返回，没死锁")
    ```
  - **预期结果**：`bad()` 会卡住（可以用 Ctrl+C 中断），`good()` 立刻返回。你就理解了 Commit 9 的修复思路。

- **知识点 3：MoE 后端选择与内核对比**
  - **是什么**：MoE 层可以把"路由 + 专家计算"交给不同内核实现（Triton / Cutlass / FlashInfer / Marlin），每个内核适用的量化格式和 GPU 架构不同。
  - **为什么重要**：选错后端可能慢 2-3 倍，或者直接跑不起来。真实部署时要根据 GPU 型号（H100/A100/B200）和量化方式（BF16/FP8/NVFP4）挑后端。
  - **动手试试**：
    ```bash
    # 看 MoE 内核特性对比表
    sed -n '90,110p' docs/design/moe_kernel_features.md
    # 用不同后端跑 Qwen3-30B-A3B（需要 H100）
    vllm serve Qwen/Qwen3-30B-A3B --moe-backend=flashinfer_cutedsl
    # 对比 triton 后端
    vllm serve Qwen/Qwen3-30B-A3B --moe-backend=triton
    ```
  - **预期结果**：FlashInfer CuTe-DSL 在 H100 上吞吐更高（尤其 batch 大时），而 Triton 兼容性更好。你会看到 `--moe-backend` 参数如何影响性能。

- **知识点 4：RLIMIT_NOFILE 与高并发文件句柄**
  - **是什么**：Linux 每个进程能打开的文件/套接字数量有软限制（默认常是 1024），高并发服务会撞上 `EMFILE`（Too many open files）。
  - **为什么重要**：一个 LLM 服务要处理成百上千并发连接，每个连接一个 fd，1024 根本不够。Commit 6 在 Rust 前端启动时自动把软限制提到 65535。
  - **动手试试**：
    ```bash
    # 查看当前限制
    ulimit -n
    # Python 里查看和提升
    python3 -c "
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print('soft:', soft, 'hard:', hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    print('raised to:', resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    "
    ```
  - **预期结果**：看到 soft 从 1024 变成 65535（若 hard 允许）。你就懂了 Commit 6 里 `rlimit::increase_nofile_limit` 在干什么。

- **知识点 5：gRPC 消息大小限制**
  - **是什么**：gRPC 默认对单条消息大小有限制（通常 4MB），传大图、长 prompt 时会失败，需要显式调大 `max_decoding_message_size`。
  - **为什么重要**：多模态模型（图文混合）请求体可能几十 MB，不调这个限制会报 `ResourceExhausted`。Commit 11 把 HTTP 的 body 限制统一应用到 gRPC。
  - **动手试试**：
    ```python
    # 伪代码：gRPC 服务端设置消息上限
    from grpc import aio
    server = aio.server()
    # 实际 tonic 里是 .max_decoding_message_size(limit)
    limit = 32 * 1024 * 1024  # 32 MiB
    print(f"设置 gRPC 消息上限为 {limit} 字节")
    # 用 vllm-rs 启动时可通过环境变量控制
    # VLLM_HTTP_MAX_REQUEST_BODY_SIZE=67108864 vllm-rs serve ...
    ```
  - **预期结果**：理解 HTTP body 限制和 gRPC message 限制是"同一个旋钮"，改环境变量两者同时生效。

## 🔥 重要更新

- **Commit 1（UMBP standalone for DeepSeek-V4 on ROCm）**：AMD 用户终于能在 DeepSeek-V4 上用 UMBP 独立模式，配合 `rebuild_sliding_window` 大幅减少缓存对象数量，长上下文场景内存压力显著下降。这是 ROCm 生态的重要补强。
- **Commit 20（FlashInfer CuTe-DSL SM90 MoE 后端）**：H100 上多了一个高性能无量化 MoE 后端，支持多种激活函数（silu/gelu_tanh/swigluoai/situ/relu2_no_mul），对 Qwen3-MoE 等模型是直接的吞吐提升。
- **Commit 17（Connex 权重传输后端）**：新增基于 RDMA 的弹性权重传输后端，支持引擎动态加入/离开/故障，是 RL rollout 场景的重要基础设施，直接对标 NCCL 但更灵活。
- **Commit 9（ShardedRDT 死锁修复）**：修了一个会导致整个权重传输卡死的并发 bug，所有用 ShardedRDT 做 RL 权重的用户都该升级。

## 📋 逐条解读

### 1. [ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4-Pro (#63002)
- **代码层面**：改的是 `docs/features/umbp_store_connector_usage.md`（新增 `rebuild_sliding_window` 配置项说明）和 `tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py`。核心是把属性 `_has_kv_transfer` 重命名为 `_has_foreign_kv_writer`（语义更准确：有"外部写入者"而非"传输"），并新增 `dsv4_swa_replay_enabled()` 的判断逻辑：只有当连接器是 UMBP 且显式要求 `rebuild_sliding_window=True` 时才开启 SWA 重放。
- **新手概念课堂**：想象你在做笔记，滑动窗口就像"只记最近 100 句话"。UMBP 连接器把旧笔记压缩存起来，命中缓存后需要"重播最近 100 句"来恢复上下文——这就像凭记忆重写最近的笔记，可能略有出入（近似），但省了大量存储空间。
- **对你有什么影响**：AMD 显卡用户跑 DeepSeek-V4 时内存占用更低、能开更长上下文；但要注意重建的窗口是近似的，对精度极敏感的任务要谨慎开启。

### 2. [ZenCPU][CI] Push the zen CPU image so the zen5 tests can reuse it (#68136)
- **代码层面**：改 `.buildkite/hardware_tests/cpu.yaml`（加 `setup.py` 到依赖触发列表）和 `.buildkite/image_build/image_build_zen_cpu.sh`（新增 push 到 ECR 的步骤）。以前 zen 镜像只在 CI 里构建不推送，zen5 测试机每次都要重新构建，现在直接拉取。
- **新手概念课堂**：就像你做菜每次都要从头切菜，现在改成"提前备好料放冰箱（镜像仓库），下次直接拿"。CI 镜像复用是加速测试的标准手段。
- **对你有什么影响**：普通用户无感，但贡献者提交 PR 后 CI 会更快。

### 3. [Bugfix] Fix InstantTensor loading for HY4 Flash Engram weights (#68659)
- **代码层面**：改 `vllm/model_executor/model_loader/weight_utils.py` 里的 `_ENGRAM_WEIGHT_RE` 正则，新增对 HY4 Flash 模型 `engram(_layer)?\.multi_head_embedding\.embedding\.weight(_scale_inv)?` 的匹配。同时用注释标注了三种模型（DeepSeek-V4.1 / HY4 Flash / Qwen4 Exp）各自的命名。
- **新手概念课堂**：正则就像"找符合特定模式的字符串的筛子"。模型权重文件里有一堆名字，loader 要识别哪些是"特殊 embedding 权重"（需要特殊加载方式），正则就是那张筛网。
- **对你有什么影响**：用 HY4 Flash 模型时权重加载不再报错/漏加载。

### 4. [Docs] Update SPEED-Bench preparation script documentation (#65382)
- **代码层面**：改 `docs/benchmarking/cli.md` 和 `vllm/benchmarks/datasets/datasets.py`。SPEED-Bench 的准备脚本从 NVIDIA-NeMo/Skills 仓库迁移到 NVIDIA-NeMo/Gym 仓库，命令加了 `--format raw --config all` 参数；`DOWNLOAD_SCRIPT_URL` 常量重命名为 `DOWNLOAD_COMMAND`（因为现在是一整条命令不只是 URL）。
- **新手概念课堂**：就像食谱更新了——食材供应商换了地址（仓库迁移），还要求"按 raw 格式、全量下载"。跟着文档走就不会下错。
- **对你有什么影响**：跑 SPEED-Bench 基准测试的人要更新命令，否则会 404。

### 5. [Bugfix][Rust Frontend] glm5_next: render developer messages like the HF renderer (#68922)
- **代码层面**：改 `rust/src/chat/src/renderer/glm5_next/encoding.rs` 的 `messages()` 函数。GLM5 的 chat template 没有 `developer` role，所以要把开头的 system/developer 消息合并成一个 system 消息，后面的 developer 消息原地转成 system。新增 `append_content()` 函数处理文本拼接（用 `\n\n` 分隔）和多模态 part 合并。
- **新手概念课堂**：就像翻译——GLM5 的模板"词汇表"里没有"开发者"这个词，翻译器得把"开发者说的话"并入"系统提示"里，且要保持原文顺序和媒体位置。
- **对你有什么影响**：用 Rust 前端跑 GLM5 时，带 developer 消息的请求渲染结果与 HuggingFace 原生渲染器一致，不再出现格式错乱。

### 6. [Bugfix][Frontend] Raise the RLIMIT_NOFILE soft limit at vllm-rs startup (#68223)
- **代码层面**：改 `rust/src/cmd/src/main.rs`，新增 `raise_nofile_limit()` 函数，用 `rlimit` crate 把 `RLIMIT_NOFILE` 软限制提到 65535（不超过硬限制），启动时调用。`Cargo.toml` 加 `rlimit` 依赖。
- **新手概念课堂**：见上面知识点 4。就像给餐厅多配 65535 个"餐具位"，而不是默认的 1024 个，客人再多也不怕没位置。
- **对你有什么影响**：高并发部署 vllm-rs 不再因文件句柄耗尽而报 `EMFILE`。

### 7. [Governance] Add hickeyma as committer for API server (#69405)
- **代码层面**：改 `.github/CODEOWNERS`（把 `@hickeyma` 加进 scale_out、renderers 相关路径的 owner 列表）和 `docs/governance/committers.md`（新增 committer 条目）。
- **新手概念课堂**：CODEOWNERS 是 GitHub 的"责任人名单"，PR 改到这些路径会自动请求对应 owner review。就像公司里指定某部门文件的审批人。
- **对你有什么影响**：无功能影响，但提 PR 到这些路径会多一个 reviewer。

### 8. [XPU] Implement stateless_init_device_torch_dist_pg for XPU (#57196)
- **代码层面**：改 `vllm/platforms/xpu.py`，新增 `stateless_init_device_torch_dist_pg()` 类方法，用 `ProcessGroupXCCL`（Intel GPU 的集合通信后端）创建无状态进程组，供权重传输等场景用。
- **新手概念课堂**：进程组就像"开会的小组"，XCCL 是 Intel GPU 版的 NCCL（NVIDIA 的）。"无状态"意味着不用全局初始化，随时创建随时用。
- **对你有什么影响**：Intel GPU 用户现在能做分布式权重传输了。

### 9. [Bugfix][Weight Transfer] Avoid ThreadPoolExecutor(wait=True) deadlock (#69589)
- **代码层面**：改 `vllm/distributed/weight_transfer/sharded_rdt_trainer.py`，把 `with ThreadPoolExecutor(...) as exe:` 改成手动创建 + `try/finally`，异常时用 `shutdown(wait=False, cancel_futures=True)` 而非默认的 `wait=True`。测试文件新增"阻塞客户端"模拟真实 worker 卡住，验证错误能立即传播。
- **新手概念课堂**：见知识点 2。就像消防演习时发现有人被锁在房间里，不能等他自己出来（wait=True），要直接破门（wait=False）。
- **对你有什么影响**：RL 权重传输失败时不再整机卡死，错误能及时上报。

### 10. [Bugfix][DSA] Stage ShardQ indexer indices in fresh host memory each step (#66052)
- **代码层面**：改 `vllm/v1/attention/backends/mla/indexer.py`，把复用的 `CpuGpuBuffer` 换成每步新建的 `ShardQSplit` 张量。原因是 CUDA 异步拷贝是排队执行的，如果复用同一块 host 内存，下一步的写入会覆盖上一步还没上传完的数据。
- **新手概念课堂**：就像用快递箱——如果只有一个箱子，你往里塞新东西时上一个包裹可能还没寄出，就混了。每步用新箱子（fresh host memory）就不会串。
- **对你有什么影响**：DeepSeek V3.2 的 ShardQ 索引器在高并发下不再出现索引错乱。

### 11. [Bugfix][Rust Frontend] Apply the configurable request size limit to gRPC messages (#68923)
- **代码层面**：改 `rust/src/server/src/lib.rs`，把三个 gRPC service 的 `max_decoding_message_size` 从写死的 `DEFAULT_REQUEST_BODY_LIMIT_BYTES` 改成读 `request_body_limit_bytes` 配置（`None` 时用 `usize::MAX`）。CLI 帮助文本和文档同步更新说明"HTTP body 和 gRPC message 共用同一限制"。
- **新手概念课堂**：见知识点 5。以前 HTTP 有个"门"限制大小，gRPC 有另一个固定的"小门"，现在统一成一个可调的门。
- **对你有什么影响**：通过 `VLLM_HTTP_MAX_REQUEST_BODY_SIZE` 能同时控制 HTTP 和 gRPC 的请求大小，多模态大请求不再被 gRPC 层拒。

### 12. [Frontend][ToolParser] Stream Kimi K3 tool arguments incrementally (#67021)
- **代码层面**：改 `tests/tool_use/test_kimi_k3_tool_parser.py` 和 Kimi K3 tool parser 实现。以前工具调用参数要等完整才一次性输出，现在按 chunk 增量输出（测试从"1 个 delta"变成"3 个 delta"，拼接后 JSON 相同）。
- **新手概念课堂**：就像打字 vs 发语音——以前是憋完整段话再发（用户干等），现在是边说边发（用户实时看到），体验更好。
- **对你有什么影响**：用 Kimi K3 做 function calling 时，工具参数会流式返回，前端可以更早开始渲染。

### 13. [Perf] Enable native small-head FlashMLA kernels for HYV4/V4-Flash on SM90 (#68321)
- **代码层面**：改 `.buildkite/test_areas/attention.yaml`（加测试依赖）和 `tests/v1/attention/test_sparse_mla_backends.py`（新增 `with_sinks` 参数化测试），核心是新增 `vllm/models/hy_v4/nvidia/flashmla_sparse.py` 里的 `HYV4FlashMLASparseImpl`，支持 attention sink 和"小 head 数"的原生内核。
- **新手概念课堂**：MLA（Multi-head Latent Attention）是 DeepSeek 提出的省显存注意力。以前小 head 数（如 8 个）会走 fallback 慢路径，现在 H100 上有原生内核直接支持，还带 sink（注意力汇点，稳定长上下文）。
- **对你有什么影响**：HYV4/V4-Flash 模型在 H100 上推理更快，尤其小 head 配置。

### 14. [XPU][Qwen3.5] Support sequence parallelism with pipeline parallelism (#69314)
- **代码层面**：改 `vllm/models/qwen3_5/xpu/model.py`。去掉 `use_attn_reduce_scatter_for_moe` 里 `pipeline_parallel_size == 1` 的限制；`forward()` 里处理 residual 的序列并行切分；PP 非最后一层时把 hidden_states 和 residual 拼接后 all_gather 成完整 token（因为 PP 传输要求全量 token）。
- **新手概念课堂**：序列并行（SP）是把一条长序列切给多张卡算，流水并行（PP）是把不同层放不同卡。两者结合就像"流水线工人 + 每人处理半条产品"，需要小心地在工位间传递完整产品。
- **对你有什么影响**：Intel GPU 上 Qwen3.5 可以同时开 SP + PP，大模型部署更灵活。

### 15. [XPU][CI] Skip cpu_test cases in Intel V1 kv_connector unit tests (#69524)
- **代码层面**：改 `.buildkite/scripts/hardware_ci/run-intel-ci-test.sh`，给 pytest 加 `-m 'not cpu_test'`，跳过那些假设在 CPU 平台跑的用例。
- **新手概念课堂**：pytest 的 marker 就像给测试贴标签，"cpu_test"标签的用例在 GPU 机器上跑没意义，直接跳过省时间。
- **对你有什么影响**：无功能影响，Intel CI 更快更稳。

### 16. [Frontend][RL] Verify weights against a baseline in finish_weight_update (#59371)
- **代码层面**：改 `docs/training/weight_updates/weight_checker.md`（更新使用流程）和 `rust/src/server/src/routes/tests.rs`（新增测试）。核心是 `finish_weight_update` 接口现在可以接收 `baseline` 参数，在完成权重更新时自动比对 checksum，不匹配则抛 `RuntimeError`。
- **新手概念课堂**：就像快递签收时开箱验货——以前是先签收再单独验，现在签收时直接验，有问题当场拒收。
- **对你有什么影响**：RL 训练中权重传输出错能被及时发现，避免用错权重继续训练。

### 17. [RL] Add Connex weight transfer backend (#63805)
- **代码层面**：新增 `docs/training/weight_updates/engines/connex.md` 文档和对应后端实现。Connex 用单边 RDMA 读 + 树形扇出（已拉取的引擎可以服务后来的引擎），支持弹性成员变更和故障切换。
- **新手概念课堂**：见关键词"RL 训练 / 权重热更新"。就像"传话游戏"升级版——不是一个人挨个传，而是谁先听到谁就传给下一个人（树形），有人掉线自动换人（容错）。
- **对你有什么影响**：大规模 RL rollout 时权重分发更可靠、更弹性，引擎挂了不影响整体更新。

### 18. [Bugfix][KV Offload][P2P] Eliminate no-op store jobs and inflated write metrics (#69132)
- **代码层面**：改 `tests/v1/kv_offload/tiering/p2p/test_manager.py` 和 `tests/v1/kv_offload/tiering/test_backpressure.py`。核心是让主 tier 的 store 只在"生产者"场景才级联到 P2P tier，消费者/本地场景不产生空转的 P2P store job，同时修正被虚高的写指标。
- **新手概念课堂**：就像仓库入库——只有真正要出货的东西才登记到"待发区"，本地自用的东西不用走发货流程。以前所有东西都登记，导致"待发区"爆满且发货统计虚高。
- **对你有什么影响**：KV offload 的 P2P 场景下不再产生无用任务，监控指标更准确。

### 19. [Doc] Fix docs build RecursionError from lazily imported torch._inductor (#69580)
- **代码层面**：改 `docs/mkdocs/gen_files/generate_argparse.py`。旧逻辑用 `traceback` 找缺失模块的路径然后 auto_mock，遇到 `torch._inductor` 这种懒加载模块会无限递归。新逻辑直接用 `e.name` 拿到缺失模块名，塞进 `sys.modules` 一个 mock 就完事。
- **新手概念课堂**：就像找丢失的钥匙——以前是"根据脚印找房间再找钥匙"（可能绕圈），现在是"钥匙上写着名字，直接按名字找"。
- **对你有什么影响**：贡献文档时 `mkdocs build` 不再因 RecursionError 失败。

### 20. [Kernel][MoE] Add FlashInfer CuTe-DSL unquantized MoE backend for Hopper (SM90) (#66749)
- **代码层面**：新增 `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_sm90_moe.py`（`FlashInferCuteDSLSm90Experts`），更新 `docs/design/moe_kernel_features.md` 表格，新增 H100 CI 测试和 GSM8K 评测配置。支持多种激活函数（silu/gelu_tanh/swigluoai/situ/relu2_no_mul），无量化（BF16）。
- **新手概念课堂**：CuTe-DSL 是 NVIDIA 的"用 Python 写 GPU 内核"的新工具（类似 Triton 但更贴近硬件）。这个后端就像给 H100 配了一把"专用的 MoE 菜刀"，切 BF16 专家计算更快。
- **对你有什么影响**：H100 上跑 Qwen3-30B-A3B 等 MoE 模型时可以用 `--moe-backend=flashinfer_cutedsl` 获得更好吞吐，且兼容多种激活函数。

## sglang
## 💡 今日关键词

- **异构计算适配（NPU / AMD / NVIDIA 多平台）**
  - 一句话通俗解释：同一套推理框架要能在华为昇腾（NPU）、AMD GPU（gfx1250/gfx950）、NVIDIA Hopper/Blackwell 上都能跑得好。
  - 为什么社区在关注：今天 20 个 commit 里有 7 个直接和 NPU/AMD 相关（#51787、#49566、#49347、#48154、#52418 等），说明 SGLang 正从"NVIDIA 优先"走向"多后端一等公民"。
  - 对新手意味着什么：如果你只会写 CUDA，会越来越吃亏；懂 Triton + 多后端算子适配的人非常稀缺，是切入 AI Infra 的好方向。

- **确定性推理（Deterministic / Batch-Invariant）**
  - 一句话通俗解释：同样的问题，不管跟多少其他请求一起跑，输出必须一模一样。
  - 为什么社区在关注：#44194 修复了"batch 大小会改变输出"的 bug——这是做评测复现、RL 训练、合规审计的前提。
  - 对新手意味着什么：理解"浮点运算顺序会影响结果"是进阶必修课；做实验时能解释"为什么我复现不出论文数字"。

- **CUDA Graph / Breakable Graph 与 PP、CP 的结合**
  - 一句话通俗解释：把一段 GPU 计算"录下来"反复回放，省掉启动开销；现在还能和流水线并行（PP）、上下文并行（CP）组合。
  - 为什么社区在关注：#51721 让 zigzag 预填充 CP 的图回放支持 PP 阶段，是长上下文推理降延迟的关键。
  - 对新手意味着什么：CUDA Graph 是生产级推理框架的标配技能，值得专门花时间学。

- **KV Cache 与分层/分池管理（HiCache、Linker、Draft KV）**
  - 一句话通俗解释：把模型的"记忆"（KV cache）分层放在 GPU 显存、CPU 内存、远端存储里，按需搬运。
  - 为什么社区在关注：#49466、#49465、#51787 都在优化 cache 的加载顺序、掩码写入、越界写入，说明大规模服务里 cache 管理是性能与正确性的主战场。
  - 对新手意味着什么：KV cache 是 LLM 推理的核心数据结构，吃透它等于吃透半个推理引擎。

- **投机解码（Speculative Decoding）生态扩张**
  - 一句话通俗解释：用小模型"猜"几个 token，大模型一次性验证，加速生成。
  - 为什么社区在关注：#52754 把 ERNIE 5.5 Flash 的 draft 从 ECHO 换成 DSpark，说明投机解码方案正在快速迭代、多方案并存。
  - 对新手意味着什么：投机解码是当前推理加速性价比最高的手段之一，入门门槛不高，适合做项目练手。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：Triton 内核的 grid 维度限制（分布式 / 性能优化方向）

- **是什么**：CUDA 的 grid 在 y、z 方向有上限（通常 65535），如果把 token 数放在 `grid.y` 上，长序列 prefill 会直接崩。
- **为什么重要**：Commit #51702 就是把二维 grid `(k_blocks, m_blocks)` 压成一维 `(k_blocks * m_blocks,)`，在 kernel 里用取模/整除还原。不懂这个，你写的 Triton kernel 在小 batch 测试全过，一上长文本就报 `invalid configuration argument`。
- **动手试试**：
```python
import torch, triton, triton.language as tl

@triton.jit
def bad_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(1)  # 故意用 y 维
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs), mask=offs < n)

@triton.jit
def good_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs), mask=offs < n)

n = 100_000_000  # 1 亿个元素
x = torch.zeros(n, device="cuda")
BLOCK = 1024
# bad_kernel[(1, triton.cdiv(n, BLOCK))](x, n, BLOCK)   # 会报错
good_kernel[(triton.cdiv(n, BLOCK),)](x, n, BLOCK)      # 正常
print("ok")
```
- **预期结果**：注释掉的那行会抛 `RuntimeError: invalid configuration argument`；一维 grid 版本顺利跑完。你就理解了为什么 #51702 要"flatten the grid"。

---

### 知识点 2：Sampling 参数缓存与"写时失效"（后端工程方向）

- **是什么**：Commit #52570 给采样参数池加了一层缓存——如果这一批请求的 slot 和上次完全一样、中间没人写过，就直接复用上次 gather 出来的张量。
- **为什么重要**：decode 阶段每步都要 gather 采样参数（temperature/top_p/top_k），这是个 CPU→GPU 的小同步操作，在关键路径上。缓存能省掉重复的 `index_select` 和 host 端 numpy 索引。但代价是：**调用方绝对不能原地修改返回的张量**，否则会污染缓存。
- **动手试试**：用一个极简 Python 模型模拟"缓存 + 失效"逻辑：
```python
class Pool:
    def __init__(self):
        self.table = {}
        self._cache = None
    def write(self, slot, val):
        self.table[slot] = val
        self._cache = None          # 写时失效
    def gather(self, slots):
        key = tuple(slots)
        if self._cache and self._cache[0] == key:
            print("cache hit")
            return self._cache[1]
        print("cache miss")
        out = [self.table.get(s, 0) for s in slots]
        self._cache = (key, out)
        return out

p = Pool()
p.write(1, 0.7); p.write(4, 1)
print(p.gather([1, 4]))   # miss
print(p.gather([1, 4]))   # hit
p.write(1, 0.9)
print(p.gather([1, 4]))   # miss（因为写过）
```
- **预期结果**：`miss → hit → miss`。你会明白"缓存键 = slot 序列，写操作使缓存失效"这个通用模式，以及为什么文档里强调"Callers must not write them in place"。

---

### 知识点 3：批不变性（Batch Invariance）与浮点归约顺序（数值 / 确定性方向）

- **是什么**：Commit #44194 发现：`BT = min(chunk_size, next_pow2(T))`，当 T 是"整个 batch 的总 token 数"时，一条短序列的输出会因为它和长序列拼在一起而改变——因为 tile 大小变了，归约顺序变了，浮点结果就不 bitwise 相等了。
- **为什么重要**：这是"确定性推理"的核心坑。只要归约顺序依赖 batch 组成，你的评测结果就不可复现，RL 训练里 on-policy 数据也会漂移。
- **动手试试**：用 PyTorch 感受浮点归约顺序的影响：
```python
import torch
torch.manual_seed(0)
x = torch.randn(100000, device="cuda", dtype=torch.float32)
# 不同归约粒度
a = x.sum()
b = x.view(100, 1000).sum(dim=1).sum()
c = x.view(1000, 100).sum(dim=1).sum()
print(a.item(), b.item(), c.item())
print("a==b:", torch.equal(a, b), "a==c:", torch.equal(a, c))
```
- **预期结果**：三个值非常接近但通常不 bitwise 相等（`False`）。这就直观说明了：**改变分块方式 = 改变结果**。所以 #44194 在 batch-invariant 模式下把 `BT` 固定成 `chunk_size`，让每条序列的输出与 batchmates 无关。

---

### 知识点 4：RoPE + KV Cache 融合内核与"tiled"分块策略（性能优化方向）

- **是什么**：Commit #49566 在 gfx1250 上，当 T ≥ 256 时，把原来"每个 token 一个 program"的融合 RoPE+KV 写入内核，换成"每 16 个 token × 8 个 q head 一个 program"的分块版本。
- **为什么重要**：这是典型的 kernel 优化思路——**小 T 时并行度优先（一 token 一 program），大 T 时数据复用/访存效率优先（分块）**。同时它用环境变量 `SGLANG_FUSED_ROPE_KV_TILED` 做开关，并严格检查 dtype、stride、layout 等前置条件，否则回退到原内核——这是生产级 kernel 的正确写法。
- **动手试试**：用 Triton 写一个最小对比，感受"每元素一 program" vs "每 BLOCK 一 program"的差别：
```python
import torch, triton, triton.language as tl, time

@triton.jit
def per_elem(x_ptr, y_ptr, n):
    i = tl.program_id(0)
    if i < n: y_ptr[i] = x_ptr[i] * 2.0

@triton.jit
def blocked(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)

n = 1 << 20
x = torch.randn(n, device="cuda"); y = torch.empty_like(x)

def bench(fn):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(200): fn()
    torch.cuda.synchronize(); return (time.time()-t)/200*1e6

print("per_elem us:", bench(lambda: per_elem[(n,)](x, y, n)))
print("blocked  us:", bench(lambda: blocked[(triton.cdiv(n,1024),)](x, y, n, 1024)))
```
- **预期结果**：`blocked` 通常快 2–5 倍。你会直观理解为什么大 T 要走 tiled 路径，以及为什么 kernel 优化都要做"分块 + 掩码 + 回退条件"。

---

### 知识点 5：PP 流水线并行中的"设备张量不能直接 pickle"（分布式方向）

- **是什么**：Commit #52196 修复了 PP 场景下，输出消息里的 logprob 行还在 GPU 上就被 pickle 发送的问题；同时兼容"第一段收到的是 list 而不是 tensor"的情况。
- **为什么重要**：跨进程/跨节点通信时，**必须先把 GPU 张量拷回 host（.tolist() / .cpu()）再序列化**。否则要么报错，要么触发隐式同步，要么在多卡场景下出现难以复现的 hang。这是分布式推理里最常见的坑之一。
- **动手试试**：用 multiprocessing 复现"直接传 CUDA tensor"的问题（在单机多卡或单卡上都能观察）：
```python
import torch, multiprocessing as mp

def worker(q):
    t = q.get()
    print("received:", type(t), t.device)

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t = torch.randn(4, device="cuda")
    # q.put(t)                       # 很多场景会报错或极慢
    q.put(t.tolist())                # 正确做法：先回 host
    p = ctx.Process(target=worker, args=(q,)); p.start(); p.join()
```
- **预期结果**：直接 `put(cuda_tensor)` 会抛序列化错误或卡住；`tolist()` 版本正常。你就理解了 #52196 为什么要在发送前统一做 `pp_output_rows_to_host` / `pp_output_rows_to_lists`。

---

## 🔥 重要更新

- **#51787（NPU GLM-5.3-Flash KPool 越界写）**：删掉了一个错误的 `index_page_size` 参数，属于**内存越界写**级别的严重 bug，会导致结果错乱甚至崩溃，NPU 用户务必升级。
- **#44194（确定性推理 GDN 输出 tile 固定）**：修复了"batch 组成改变输出"的问题，是做可复现评测、RL 训练的基础设施性修复，影响面大。
- **#51721（zigzag prefill CP 图回放支持 PP）**：解除 `pp_size == 1` 限制，长上下文 + 流水线并行 + CUDA Graph 三者可以组合，是长文本推理延迟优化的重要一步。
- **#52196（PP 输出 logprob 先回 host 再 pickle）**：修复 PP 场景下的通信正确性问题，并为后续 AsyncPP 系列（1/9）铺路。

---

## 📋 逐条解读

### 1. [Fix] [NPU] mem_cache: Fix out-of-bounds KPool index writes in GLM-5.3-Flash (#51787)
- **代码层面**：`kv_cache_configurator.py` 的 `_build_ascend_mla_kv_pool` 里删掉了一行 `index_page_size=get_schedule().page_size`。这个参数传进去会让 KPool 的索引按错误的 page size 计算，导致写越界。
- **新手概念课堂**：KV cache 就像一个巨大的"储物柜"，每个柜子有编号。page size 是"一个柜子放几个 token 的 KV"。如果构建时告诉它错误的柜子大小，算出来的编号就会指到柜子外面去——这就是越界写，轻则数据错乱，重则进程崩溃。
- **对你有什么影响**：用昇腾 NPU 跑 GLM-5.3-Flash 的用户会从"偶发输出乱码/崩溃"变成稳定。其他硬件不受影响。

### 2. [Perf] [AMD] rotary_embedding: tile fused RoPE and KV write for large T on gfx1250 (#49566)
- **代码层面**：`fused_qk_rope_reshape_and_cache.py` 新增 `_use_tiled()` 判定函数和 tiled 版 Triton 内核；当 T ≥ 256、dtype 为 fp16/bf16/fp32、cos_sin 为 fp32、layout 为 flash + neox 等条件满足时启用。新增环境变量 `SGLANG_FUSED_ROPE_KV_TILED`（默认 true）。
- **新手概念课堂**：RoPE 是给 Q/K 向量"旋转"以编码位置信息；KV cache 写入是把它存起来。原来两个操作各干各的，融合内核一次干完。而"tiled"就是像切蛋糕一样，一次处理 16 个 token，而不是一个 token 一个线程——大块处理访存更连续。
- **对你有什么影响**：AMD gfx1250 用户处理长 prompt（≥256 token）时 prefill 更快；不想用可设 `SGLANG_FUSED_ROPE_KV_TILED=false` 回退。

### 3. [Fix] deterministic: Keep the GDN output tile fixed for varlen prefill (#44194)
- **代码层面**：`chunk_o.py` 的 `chunk_fwd_o` 中，当 `cu_seqlens is not None` 且开启了 batch-invariant 模式时，强制 `BT = chunk_size`，不再随总 T 缩小。新增测试验证"短序列单独跑 vs 和别的序列拼一起跑"输出 bitwise 相等。
- **新手概念课堂**：GDN（Gated Delta Net）是一种线性注意力。varlen 打包时 T 是整批总长度。原来 tile 大小随 T 变小，导致同一条序列在不同 batch 里被切得不一样，浮点归约顺序不同 → 结果不同。"批不变性"就是要求结果与 batchmates 无关。
- **对你有什么影响**：开启 batch-invariant 模式后，评测结果可复现；代价是短序列可能用了比需要更大的 tile，略慢一点点。

### 4. [Perf] sampling: Reuse the sampler params of an unchanged batch (#52570)
- **代码层面**：`sampler_params.py` 的 `SamplerParamsPool` 新增 `_last_gather` 缓存；`write()` 时清空缓存；`gather()` 时若 slot 序列（`tobytes()` 作为 key）与上次相同则直接返回。另把 `top_ks.masked_fill_` 改成非原地版本以免污染缓存。
- **新手概念课堂**：decode 每步都要把每个请求的采样参数（温度、top_p、top_k）从一张大表里"抓"出来。如果这一批请求和上一步完全一样（很常见，decode 时请求集合基本不变），就没必要重复抓——直接复用上次结果。
- **对你有什么影响**：高并发 decode 场景下，每步省掉一次 `index_select` 和 host 端索引，吞吐小幅提升。普通用户感知不到，但大规模服务累积起来可观。

### 5. [Fix] qwen2.5-vl: use per-frame cu_seqlens for video full attention (#47918)
- **代码层面**：`qwen2_5_vl.py` 中 `cu_seqlens` 从"每个视频一个 segment"改成"每个时间帧一个 segment"（`repeat_interleave(h*w, t)`），CUDA graph 的 `full_layout` 也同步改成逐帧累加。
- **新手概念课堂**：视频是一帧帧的，每帧内部 token 做全注意力，帧与帧之间不应该互相"看"。原来的 cu_seqlens 把整个视频当一个段，等于让所有帧互相注意，和 HuggingFace 的实现不一致。
- **对你有什么影响**：Qwen2.5-VL 处理视频时结果更正确，和 HF 对齐；CUDA graph 的 key 也修正了，不会再命中错误的捕获图。

### 6. [Perf] linker: Load DSpark draft KV with the last transfer layers (#49466)
- **代码层面**：`linker_pool_assembler.py` 中 `_sidecar_draft_entries` 新增 `num_layers` 参数，把 draft 层的映射从 `{i: i}` 改成 `{offset + i: i}`，让 draft KV 挂在目标模型的**尾部**传输层上；`registry.py` 给 draft pool 注册独立的 layer transfer counter。
- **新手概念课堂**：HiCache 分层加载时，各层 KV 是分批从远端搬进来的。投机解码的 draft 模型只在主模型算完之后才用，所以它的 KV 应该"晚点加载"——挂在最后几层后面，避免占用前面层的传输带宽。
- **对你有什么影响**：开启 HiCache + DSpark 投机解码时，首 token 延迟可能降低，因为 draft KV 不再抢前面的传输通道。

### 7. [Perf] [NVIDIA] diffusion/kernels: Look up the rounded SiLU in its table on Hopper too (#52491)
- **代码层面**：`activation.py` 的 `_silu_rounding_table_supported` 从 `props.major == 10` 改成 `props.major in (9, 10)`，即 Hopper（sm_90）也启用 128 KiB 共享内存的 SiLU 查表。
- **新手概念课堂**：SiLU 激活函数里有 `exp` 和除法，是 ALU 密集操作。但如果输入是 fp16/bf16，输出只有有限种取值，可以预计算一张 64K 项的查找表放进共享内存，直接查表代替计算。
- **对你有什么影响**：H100/H200 用户跑 diffusion 模型时激活层更快。测试里的 skip 提示也同步更新了。

### 8. [Doc] cookbook: Refresh SemiAnalysis AgentX launches to InferenceX main 1f8cf90a (#52550)
- **代码层面**：`_agentx.jsx` 及多个模型配置 jsx 更新：新增 `legacyLinkerConfig` 兼容旧参数拼写；Mooncake store 支持 `placement: "infra"`（独立节点，不跑 worker，只贡献内存）；tab id 加入 `placement` 避免冲突。
- **新手概念课堂**：Cookbook 是"照着抄就能跑"的部署配方。这次更新让旧版本提交的命令仍能正确渲染，同时支持了"Mooncake 存储跑在独立节点上"这种新部署形态。
- **对你有什么影响**：查 cookbook 时看到的启动命令更准确；用 Mooncake 做 KV 存储分层的用户多了一种部署选项。

### 9. [Refactor] [NPU] laser_attn: use the public sgl_kernel_npu operator entry (#52418)
- **代码层面**：`laser_attn.py` 从 `import attentions` + `torch.ops.attentions.la` 改成 `from sgl_kernel_npu.operators.attention import la`；新增 `_laser_attention_op` 包装，非 NPU 平台给出清晰报错。
- **新手概念课堂**：以前是通过"偷偷 import 一个包，让 torch 注册算子"的方式用 kernel，很脆弱。现在改成显式 import 公开 API，依赖关系清晰，报错信息也友好。
- **对你有什么影响**：NPU 用户装 `sgl-kernel-npu` 后 Laser Attention 更可靠；非 NPU 平台误用时能看到明确提示而不是莫名其妙的 AttributeError。

### 10. [Perf] linker: Fill the restorable-prefix mask with one vectorized store (#49465)
- **代码层面**：`unified_cache_linker.py` 的 `_sync_restorable_prefix` 把"for 循环逐个 `mask[pages] = 1`"改成"先收集 valid 列表，再 `mask[torch.tensor(valid)] = 1`"一次写入。
- **新手概念课堂**：GPU 上每个"标量索引赋值"都是一次独立的 kernel launch 或至少一次昂贵的 host-device 交互，微秒级。循环 100 次就是几百微秒，而这段代码在调度器关键路径上。
- **对你有什么影响**：HiCache 远端前缀匹配时调度延迟降低。新增测试也覆盖了"候选页越界应被丢弃"的边界情况。

### 11. [Doc] cookbook: Document FP32 state for Qwen3.5-122B FP8 on two H200s (#52609)
- **代码层面**：`Qwen3.5.mdx` 新增 5.4 节，给出双 H200 TP2 的完整启动命令，关键是 `--mamba-ssm-dtype float32`，并给出数据：FP32 状态 GSM8K 222/256 vs BF16 205/256，吞吐 2655 vs 2786 tokens/s（降 4.7%）。
- **新手概念课堂**：Mamba/线性注意力的"循环状态"如果用 BF16 存，长序列会累积误差。FP32 更准但更占内存、更慢。这是一个典型的"精度 vs 性能"权衡，而且作者给了实测数字。
- **对你有什么影响**：部署 Qwen3.5-122B 时可以直接抄这个配方；也学到了"遇到长文本质量下降，先试试把 SSM 状态升到 fp32"。

### 12. [Fix] pp: Copy the output message's device logprob rows to the host before pickling (AsyncPP 1/9) (#52196)
- **代码层面**：`batch_result_processor.py` 的 `_rows_to_lists` 兼容 list 输入；`move_logprobs_to_cpu` 对每个元素判断是否 tensor 再 `.tolist()`。`scheduler_pp_mixin.py` 在发送前调用 `pp_output_rows_to_lists` / `pp_output_rows_to_host`，并为异步拷贝单独记录 event 保证顺序。
- **新手概念课堂**：PP 是流水线并行，stage 之间要传中间结果。pickle 不能直接序列化 CUDA tensor，必须先拷回 host。异步拷贝还要用 event 保证"拷完了再发"。
- **对你有什么影响**：开启 PP + logprobs 输出时不再报错或卡死。这是 AsyncPP 系列 9 个 PR 的第 1 个，后续还会有更多异步优化。

### 13. [Perf] [NVIDIA] fused_moe: Add H20-3e Triton 3.8.0 config for E=257,N=256, tuned with real routing (#50928)
- **代码层面**：新增一个 JSON 配置文件 `E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json`，为不同 batch size（1/2/4/8/...）指定 BLOCK_SIZE_M/N/K、num_warps、num_stages。
- **新手概念课堂**：MoE 的 fused kernel 性能极度依赖 tile 大小，而最优 tile 随 batch size 变化。SGLang 用"按 (专家数, N, 设备, dtype, block_shape) 查表"的方式自动选配置。这个 PR 就是用真实路由分布（而不是均匀分布）调出来的 H20-3e 专用表。
- **对你有什么影响**：用 H20-3e 跑 DeepSeek 类 MoE（E=257 是 DeepSeek-V3 的专家数）时，MoE 层更快。

### 14. [Fix] humming: Fix MoE finalize grid overflow (#51702)
- **代码层面**：`moe_fused_mul_sum.py` 把 grid 从二维 `(k_blocks, m_blocks)` 改成一维 `(k_blocks * m_blocks,)`，kernel 内用 `pid % num_k_blocks` 和 `pid // num_k_blocks` 还原。
- **新手概念课堂**：CUDA grid 的 y 维上限是 65535。长 prefill 时 token 数可能上百万，`m_blocks` 直接爆掉。压成一维就绕过了这个限制。
- **对你有什么影响**：超长 prefill（比如 100 万 token）不再因 grid 溢出崩溃。这是"小 batch 测不出、长文本必炸"的经典 bug。

### 15. [Doc] cookbook: Use one hardware id, rtxpro6000, for the RTX PRO 6000 (#51951)
- **代码层面**：把散落的 `rtx6000`、`rtx6000pro` 统一成 `rtxpro6000`，涉及模板文件、多个模型 config jsx、dockerImages 映射等。
- **新手概念课堂**：Cookbook 的硬件 id 是配置数据的 key。同一个卡有多个 id 会导致"选了硬件但配方没匹配上"的诡异问题。统一命名是配置管理的卫生习惯。
- **对你有什么影响**：在 cookbook 里选 RTX PRO 6000 时，所有模型的配方都能正确显示。

### 16. [Doc] cookbook: Point the ERNIE 5.5 Flash recipes at the Preview ckpt (#52754)
- **代码层面**：`ERNIE-5.5-Flash.mdx` 和 `ernie-5.5-flash.jsx` 更新：模型路径改为 `baidu/ERNIE-5.5-Flash-Preview{,-FP8,-BF16}`；低延迟策略从 ECHO 改成 checkpoint 内置的 DSpark draft；更新了各硬件的验证状态表。
- **新手概念课堂**：DSpark 和 ECHO 是两种投机解码方案。DSpark 的 draft 直接打包在 checkpoint 里，不需要额外加载 draft 模型，部署更简单。
- **对你有什么影响**：照 cookbook 部署 ERNIE 5.5 Flash 时用的是最新 Preview 权重和更简单的投机解码配置。

### 17. [Perf] [AMD] mla: Read the fused_qkv_a slices in place in the gfx1250 q_a / kv_a RMSNorm, one launch (#49347)
- **代码层面**：`triton_qk_rmsnorm.py` 把原来的单次 `_rmsnorm_kernel` 改成 `_qk_rmsnorm_kernel`，一次 launch 同时处理 q 和 k；支持 q/k 是 fused_qkv_a 输出的列切片（各自有自己的 row stride），不再需要先拷成连续张量。
- **新手概念课堂**：MLA 里 q_a 和 kv_a 是从一个大张量 `fused_qkv_a` 里切出来的两列。原来要先 `.contiguous()` 拷贝一份才能做 RMSNorm，现在直接按 stride 读，省一次拷贝 + 一次 kernel launch。
- **对你有什么影响**：AMD gfx1250 上 MLA 模型的 prefill 更快，显存峰值略降。

### 18. [Perf] [AMD] mla: Use the gfx950 aiter RoPE, RMSNorm and MLA KV-row paths on gfx1250 (#48154)
- **代码层面**：`layernorm.py` 去掉 gfx1250 的特殊分支，统一用 `aiter.rmsnorm2d_fwd`；`factory.py` 把 `_aiter_rope_unsupported_arch` 重命名为 `_keeps_sglang_rope_modules`（gfx1250 保留 sglang 的 rope 模块因为 MLA 需要 `cos_sin_cache`）；`rope_variant.py` 里 DeepseekScalingRotaryEmbedding 直接调 aiter rope kernel。
- **新手概念课堂**：之前 gfx1250 因为 aiter 的 JIT 编译问题被迫走 Triton 回退路径。现在 aiter 修好了，gfx1250 可以直接复用 gfx950 的高性能 kernel，但 RoPE 的模块层要保留 sglang 自己的（因为接口不同）。
- **对你有什么影响**：AMD gfx1250 上 MLA 模型（DeepSeek 系列）的 RMSNorm、RoPE、KV 写入全部走 aiter 高性能路径，整体更快。

### 19. [Fix] benchmark: Use current SGLang and vLLM fused MoE APIs (#52111)
- **代码层面**：`benchmark_vllm_vs_sglang_fused_moe_triton.py` 从 vLLM 旧的 `fused_moe` 改成 `fused_experts` + `fp8_w8a8_moe_quant_config` + `fused_topk`；SGLang 侧改用 `MoeRunnerConfig` + `select_experts`。README 改为 `python -m benchmark.kernels...` 方式运行。
- **新手概念课堂**：vLLM 和 SGLang 的 MoE API 都在演进，benchmark 脚本如果还用旧 API 会 import 失败。改成 `-m` 模块方式运行是为了让相对 import（`.common_utils`）能正确解析。
- **对你有什么影响**：想对比 vLLM vs SGLang 的 MoE kernel 性能时，脚本能跑起来了；也提醒你写 benchmark 要跟着上游 API 走。

### 20. [Feat] cp: Replay zigzag prefill CP breakable graph on PP stages (#51721)
- **代码层面**：`zigzag_bcg.py` 删掉 `supports()` 里 `pp_size == 1` 的限制；文档更新为"PP 下预填充图回放需显式 `--cuda-graph-backend-prefill breakable`"；新增测试验证"后面的 PP stage 在 CP-local 行上回放"。
- **新手概念课堂**：CP（上下文并行）把一条长序列切给多张卡算；zigzag 是一种切法（每张卡拿头尾两段）。Breakable CUDA Graph 是"可以中途打断的图回放"。PP 是流水线。三者组合时，每个 PP stage 看到的 token 数不同，图的捕获桶要按 CP-local 的行数来选。
- **对你有什么影响**：长上下文 + 流水线并行的部署可以开启预填充 CUDA Graph，首 token 延迟进一步降低。注意仍需 `trtllm_mha` 后端且 CP size == TP size。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)