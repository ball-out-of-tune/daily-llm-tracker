# 🚀 vLLM & SGLang 每日更新 — 2026-10-19

> 自动生成于 2026-10-19 06:45 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词**：**AMD/ROCm 生态适配**
  - 一句话通俗解释：越来越多 vLLM 新功能和 bugfix 专门针对 AMD GPU（ROCm 平台）做适配，比如 MoRIIO KV 传输、AITER FP8 量化融合。
  - 为什么社区在关注：这说明 vLLM 已不满足于"只跑得好 NVIDIA"，AMD GPU 正在成为推理部署的一等公民。异构计算不再是口号，而是实实在在的工程投入。
  - 对新手意味着什么：如果你只会 CUDA，学一点 ROCm/HIP 会让你在简历上多一个"跨平台推理"标签；AMD 相关的 issue 和 PR 相对较少人做，是刷贡献的好方向。

- **关键词**：**量化（Quantization）持续下沉到 kernel 层**
  - 一句话通俗解释：INT8/FP8/W4A16 等量化不再只是"加载时转换权重"，而是深入到 CPU AMX、GEMM 调度、MoE 路由等 kernel 内部做精细优化。
  - 为什么社区在关注：量化是降低推理成本最直接的手段，社区在拼命榨干每一滴性能——从"能不能量化"进入"量化后怎么跑得最快"的阶段。
  - 对新手意味着什么：懂量化原理 + 会写/调 kernel 的人非常吃香。可以从读 `csrc/cpu/sgl-kernels/` 这类代码入手。

- **关键词**：**调度器与请求生命周期管理**
  - 一句话通俗解释：多个 commit 在修调度器（scheduler）的边界问题：暂停时拒绝新请求、编码器卡住时不要饿死后面的请求、异步 KV 加载的优先级。
  - 为什么社区在关注：大模型服务是长跑，调度器的每一个 corner case 都可能导致线上雪崩。这类"看不见的 bug"才是生产环境的真痛点。
  - 对新手意味着什么：想理解 vLLM 内核，调度器（`vllm/v1/core/sched/scheduler.py`）是必读模块，比模型代码更能锻炼系统思维。

- **关键词**：**多模态与模型适配加速**
  - 一句话通俗解释：一天之内新增 Gemma 4、ColUltraVBERT、DeepSeek V4.1 图像对齐等多个模型/多模态适配。
  - 为什么社区在关注：新模型层出不穷，vLLM 的"模型适配速度"本身就是竞争力。多模态（图文、检索）是当前最热的应用方向。
  - 对新手意味着什么：想快速贡献代码，写一个模型的适配 + 测试是最友好的入门 PR，模板清晰、review 快。

- **关键词**：**跨平台可移植性（XPU / CPU / 非 Linux）**
  - 一句话通俗解释：Intel XPU 支持 benchmark 图捕获、CPU 修复 3D 输入、非 Linux 平台用 `sys.platform` 守卫 mmap 调用。
  - 为什么社区在关注：vLLM 想覆盖所有硬件后端，"可移植性"从口号变成一行行 `if platform` 判断。
  - 对新手意味着什么：理解"平台抽象层"（`vllm/platforms`）是读懂 vLLM 多后端设计的关键。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：多维张量的"展平-计算-还原"模式（CPU INT8 线性层修复）**
  - **是什么**：很多底层 GEMM kernel 只接受 2D 矩阵 `[M, K]`，但模型里传进来的激活可能是 3D `[B, T, K]`。做法是先把前面维度"压平"成 2D 算完，再 reshape 回去。
  - **为什么重要**：不懂这个模式，你写自定义 kernel 时遇到 3D 输入就会崩（正如 Whisper encoder 踩的坑）。这是所有"batch × seq × dim"张量处理的通用套路。
  - **动手试试**：
    ```python
    import torch
    x = torch.randn(2, 3, 128)          # [B=2, T=3, K=128]
    w = torch.randn(64, 128)            # [N=64, K=128]
    # 错误做法：直接 mm 会因维度不匹配报错
    # 正确做法：展平 -> 计算 -> 还原
    x2d = x.reshape(-1, x.shape[-1])    # [6, 128]
    out2d = x2d @ w.T                   # [6, 64]
    out = out2d.reshape(x.shape[:-1] + (out2d.size(-1),))  # [2, 3, 64]
    print(out.shape)                    # torch.Size([2, 3, 64])
    ```
  - **预期结果**：输出 `torch.Size([2, 3, 64])`，且与逐 batch 计算的结果完全一致。你就掌握了 commit 4 的核心修复思路。

- **知识点 2：FP8 的指数偏置（bias）与反量化缩放**
  - **是什么**：FP8（E4M3/E5M2）用很少的位存浮点数，指数位比 FP32 少很多，所以"真实值 = 存储位 × 2^(某个偏移)"。kernel 里必须正确补偿这个偏移，否则数值会差 2^120 这种天文数字。
  - **为什么重要**：commit 9 就是修这个"underflow/overflow"——原来乘了 2^120 再算，容易溢出；改成在反量化时乘 2^-8，数值更稳。不懂偏置，你调 FP8 模型会得到 NaN 或全零。
  - **动手试试**：
    ```python
    import torch
    # 模拟 FP8 E4M3：值域约 ±448，最小正规数 2^-6
    x = torch.tensor([0.5, 1.0, 2.0, 448.0, 0.001])
    x_fp8 = x.to(torch.float8_e4m3fn)
    print("FP8 存储:", x_fp8)
    print("反量化回 FP32:", x_fp8.to(torch.float32))
    # 观察 0.001 是否被截断为 0（underflow）
    ```
  - **预期结果**：你会看到 0.001 变成 0（下溢），而 448 保持不变。这就解释了为什么 kernel 需要 `dequant_scale` 来"救回"小数值。

- **知识点 3：调度器中的"延迟队列"与公平性（避免请求饿死）**
  - **是什么**：调度器遍历等待队列时，如果遇到一个"暂时不能跑"的请求（比如等编码器），不能直接 `break` 结束整个调度，而应该把它挪到延迟队列，继续看后面的请求。
  - **为什么重要**：commit 15 的 bug 就是 `break` 导致第一个卡住的请求饿死了后面所有请求。这是经典的"队头阻塞（head-of-line blocking）"问题。
  - **动手试试**：
    ```python
    # 模拟有 bug 的调度
    def bad_schedule(queue):
        scheduled = []
        for req in queue:
            if req["blocked"]:
                break          # BUG: 直接结束
            scheduled.append(req["id"])
        return scheduled

    def good_schedule(queue):
        scheduled, deferred = [], []
        for req in queue:
            if req["blocked"]:
                deferred.append(req)   # 挪到延迟队列
                continue
            scheduled.append(req["id"])
        return scheduled, deferred

    q = [{"id": "A", "blocked": True}, {"id": "B", "blocked": False}]
    print("bad:", bad_schedule(q))     # []
    print("good:", good_schedule(q))   # (['B'], [{'id': 'A', ...}])
    ```
  - **预期结果**：`bad` 返回空列表（B 被饿死），`good` 正确调度了 B。你就理解了 commit 15 的修复价值。

- **知识点 4：CUDA Graph 与"何时能用图捕获"**
  - **是什么**：CUDA Graph 把一串 GPU kernel 调用录制成一张"图"，之后重放比逐个 launch 快很多。但不是所有模型/配置都能用——比如某些稀疏 MLA 在特定并行度下会 hang。
  - **为什么重要**：commit 14 给 DeepSeek-V3.2/GLM-5.2 开了 CUDA Graph 白名单，直接提升解码吞吐。理解"什么条件下能开图"是性能调优的基本功。
  - **动手试试**：
    ```python
    import torch
    # 简单演示 CUDA Graph（需要 GPU）
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        w = torch.randn(1024, 1024, device="cuda")
        # 预热（图捕获前必须 warmup）
        for _ in range(3):
            y = x @ w
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = x @ w
        g.replay()   # 重放，比普通 launch 快
        print("graph replay ok:", y.shape)
    else:
        print("需要 CUDA GPU")
    ```
  - **预期结果**：打印 `graph replay ok: torch.Size([1024, 1024])`。你会体会到"录制一次、重放多次"的思路。

- **知识点 5：跨平台代码守卫（`sys.platform`）**
  - **是什么**：`mmap` 的 `MADV_HUGEPAGE`、`fadvise` 等是 Linux 专有 API，在 macOS/Windows 上直接调用会报错。用 `if sys.platform != "linux": return` 提前退出。
  - **为什么重要**：commit 10 就是修这个。写跨平台库时，一个没守卫的系统调用就能让整个包在 Mac 上 import 失败。
  - **动手试试**：
    ```python
    import sys, mmap
    def try_hugepage(nbytes):
        if sys.platform != "linux":
            print(f"跳过：当前平台 {sys.platform} 不支持")
            return None
        m = mmap.mmap(-1, nbytes)
        m.madvise(mmap.MADV_HUGEPAGE)
        return m
    print(try_hugepage(1 << 20))
    ```
  - **预期结果**：在 Linux 上返回一个 mmap 对象；在 Mac 上打印"跳过"。你就理解了 commit 10 的防御性编程。

---

## 🔥 重要更新

1. **Commit 2：暂停生成时拒绝新请求（#51488）**——RL 训练场景的关键语义修复。之前暂停时新请求会"穿越边界"进入队列，导致权重更新期间混入旧权重推理。现在 `abort`/`wait` 模式下返回 HTTP 503，客户端重试即可，边界清晰。

2. **Commit 14：DSv3.2/GLM-5.2/5.3 稀疏 MLA 支持 decode-only CUDA Graph（#62377）**——直接解锁这两个热门模型的解码吞吐提升，之前只能 `cudagraph_mode=NONE` 硬扛。

3. **Commit 19：文本 LoRA 正确绑定到多模态模型的 LM 部分（#62224）**——之前给多模态模型加载纯文本 LoRA 会"静默失效"，现在能正确映射到 `language_model` 前缀，是 LoRA 用户的重大修复。

4. **Commit 18：Gemma 4 上线 `/v1/systemone` 和 `/v1/decisions`（#62303）**——新增结构化决策端点支持 Gemma 4 系列，紧跟 Google 新模型发布节奏。

---

## 📋 逐条解读

### 1. [Bugfix][Frontend] Honor --enable-log-outputs on /v1/completions (#62205)
- **代码层面**：修改 `tests/entrypoints/openai/completion/test_completion_error.py` 和对应的 serving 代码。让 `OpenAIServingCompletion` 接收并传递 `request_logger` 和 `enable_log_outputs` 参数，使 `/v1/completions` 端点也能像 chat 端点一样记录输出日志。
- **新手概念课堂**：想象你开了一家餐厅，`--enable-log-outputs` 是"记录每桌点了什么菜"的开关。之前这个开关只对"包间"（chat）生效，"大厅"（completions）不记录。这个 PR 就是让大厅也记账。
- **对你有什么影响**：如果你用 `/v1/completions` 做日志审计或调试，现在打开 `--enable-log-outputs` 就能看到完整输出记录，不用再抓包。

### 2. [Core] Reject new requests while generation is paused (#51488)
- **代码层面**：涉及 Rust 客户端 `pause_scheduler` → `pause_generation` 重命名、`EngineCoreOutput` 新增 `error` 字段、调度器在暂停态拒绝新请求并返回可重试错误、文档更新暂停语义表。核心是把"暂停"从"冻结队列"升级为"生成边界"。
- **新手概念课堂**：就像银行柜台"暂停营业"牌子。以前暂停只是"不叫号"，新客户还能进来排队；现在挂上牌子后，新客户直接被礼貌地请回（503），等恢复营业再来。
- **对你有什么影响**：做 RL 训练或权重热更新时，暂停期间客户端会收到 503，需要实现重试逻辑。好处是再也不会出现"用旧权重算了一半的请求混进新权重批次"。

### 3. [Bugfix][ROCm][MoRIIO] Support heterogeneous TP in WRITE mode (#61726)
- **代码层面**：修改 MoRIIO KV connector，支持 prefill 和 decode 实例使用不同的 TP 大小（只要互为倍数）。MLA latent 复制、KV head 切分、Mamba 状态切片都做了适配。新增 4 卡 MI355 测试。
- **新手概念课堂**：P/D 分离（Prefill/Decode 分离）就像"洗菜"和"炒菜"由不同厨师负责。以前两个厨师必须用同样多的灶台（TP 相同），现在洗菜用 2 个灶、炒菜用 1 个灶也行，只要能把菜传过去。
- **对你有什么影响**：部署 P/D 分离时，prefill 和 decode 可以按各自负载独立选择 TP，资源利用率更高。注意 xGMI 后端下"decode TP 更大"暂不支持。

### 4. [CPU][Bugfix] Flatten >2-D activations in AMX int8 W8A8 linear path (#61965)
- **代码层面**：`vllm/model_executor/kernels/linear/scaled_mm/cpu.py` 的 `_apply_weights_sgl` 中，先 `reshape(-1, last_dim)` 展平再调 kernel，算完再 reshape 回原形状。新增 3D 输入测试。
- **新手概念课堂**：见知识点 1。底层 GEMM 只会做"二维表格乘法"，三维数据要压扁成表格算完再折回去。
- **对你有什么影响**：Whisper 等 encoder 模型在 CPU 上跑 INT8 量化不再崩溃，输出形状正确。

### 5. [Benchmark][XPU] Support XPU in forward bench graph capture/profiling (#62375)
- **代码层面**：`vllm/benchmarks/forward/capture.py` 和 `profile.py` 中，把 CUDA 专有的 `torch.cuda.Event(external=True)`、`cuda.profiler.start/stop`、NVTX range 用 `current_platform.is_xpu()`/`is_cuda_alike()` 守卫，XPU 走简化路径。
- **新手概念课堂**：就像给不同品牌的相机配不同的快门线。CUDA 相机有"外部触发"接口，XPU 相机没有，就改用普通触发。功能一样，接线不同。
- **对你有什么影响**：在 Intel XPU 上跑 forward benchmark 不再 hang，可以正常做性能剖析。

### 6. [Bugfix][Frontend] Align DeepSeek V4.1 images with rendered content (#60792)
- **代码层面**：修改 `vllm/renderers/deepseek_v4.py` 和 tokenizer 相关代码，确保图像占位符与渲染后的实际内容对齐。新增同步/异步渲染的媒体测试。
- **新手概念课堂**：渲染模板就像"填空表格"，图像占位符是"此处贴照片"的方框。如果方框和实际贴的照片错位，模型就会"看图说话说错地方"。
- **对你有什么影响**：DeepSeek V4.1 多模态输入时，图像和文本对齐更准确，减少"图文不符"的幻觉。

### 7. [Fast Start] support ep (#55490)
- **代码层面**：权重缓存（weight_cache）支持专家并行（EP）。新增 2 卡 L4 分布式测试，`test_weight_cache.py` 增加 `DSV2_EP_CASE`，CI 把 distributed 测试拆到独立 job。
- **新手概念课堂**：权重缓存就像"提前把菜切好放冰箱"，下次开火直接用。EP（专家并行）是把 MoE 的不同专家分到不同 GPU。这个 PR 让"提前切好的菜"也能跨 GPU 共享。
- **对你有什么影响**：用 DeepSeek-V2-Lite 等 MoE 模型 + EP 部署时，冷启动更快（第二次启动直接读缓存）。

### 8. [Frontend] Raise VLLMValidationError for an unknown Harmony channel (#58385)
- **代码层面**：`vllm/entrypoints/openai/responses/harmony.py` 中，把 `raise ValueError` 改成 `raise VLLMValidationError`，并新增参数化测试。
- **新手概念课堂**：Harmony 是 OpenAI 的消息格式，"channel"是消息的"频道"（如 `final`、`analysis`）。遇到不认识的频道，以前抛"通用错误"，现在抛"请求校验错误"，客户端能区分是"你发错了"还是"服务器坏了"。
- **对你有什么影响**：用 Responses API 时，如果模型输出非法 channel，你会收到明确的 4xx 错误而非 5xx，便于排查。

### 9. [CPU] Fix underflow in FP8 AMX attention (#61329)
- **代码层面**：`csrc/cpu/cpu_attn_amx.hpp` 中，把原来"乘 2^120 再算"的偏置补偿改为"反量化时乘 2^-8/2^-16"，避免中间值溢出。`cpu_attn_fp8.hpp` 重写 `float_to_fp8_scalar` 为模板函数，用位运算正确处理 subnormal。
- **新手概念课堂**：见知识点 2。原来先把小数字放大 2^120 倍再算，容易溢出；现在改成"算的时候用正常尺度，最后再缩小"，数值更稳。
- **对你有什么影响**：CPU 上跑 FP8 KV cache 的 attention 不再出现 NaN/Inf，长上下文更稳定。

### 10. [BugFix][MyPy] Guard Linux-only mmap/fadvise calls with sys.platform (#61836)
- **代码层面**：`weight_utils.py` 的 `drop_checkpoint_cache` 和 `engram.py` 的 `_allocate_huge_page_storage` 开头加 `if sys.platform != "linux": return`。
- **新手概念课堂**：见知识点 5。Linux 独有的"大页内存"API 在 Mac/Windows 上不存在，提前返回避免崩溃。
- **对你有什么影响**：在非 Linux 开发机上 import vLLM 不再因这些函数报错。

### 11. [CPU] Optimize routed INT8 W8A8 MoE GEMM dispatch (#60740)
- **代码层面**：`csrc/cpu/sgl-kernels/gemm.h` 为 `int8_t` 特化 `can_use_brgemm`，降低 MoE 路由场景的阈值（`M > 3` 就用 BRGEMM）。`moe_int8.cpp` 重构 `tinygemm_kernel_vnni`，用 `convert_from_float_ext` 替代手写指令。
- **新手概念课堂**：BRGEMM 是"批量小矩阵乘法"的加速指令。MoE 每个专家分到的 token 很少（M 很小），以前阈值太高用不上加速；现在降低门槛，小批量也能加速。
- **对你有什么影响**：CPU 上跑 INT8 MoE 模型（如 Qwen3-MoE）吞吐提升，尤其是专家负载不均时。

### 12. [XPU] [Feature] Adding triton/sycl GDN toggle (#56835)
- **代码层面**：`arg_utils.py` 的 `--gdn-prefill-backend` 新增 `"sycl"` 选项。`qwen_gdn_linear_attn.py` 新增 `xpu_gdn_kernel_available()` 检测和 `_resolve_gdn_prefill_backend` 逻辑，SYCL 是融合整层算子（含 conv1d 和 decode）。OLMo 模型显式拒绝 sycl。
- **新手概念课堂**：GDN（Gated Delta Net）是线性注意力的一种。prefill 后端就像"算前置部分用哪个计算器"，现在 Intel GPU 上可以用 SYCL 专用计算器，比通用 Triton 更快。
- **对你有什么影响**：Intel XPU 上跑 Qwen GDN 系列模型，prefill 阶段可用 SYCL 加速，吞吐更高。

### 13. [ROCm][Compile] Fuse AITER static FP8 quant into MLA attention output (#62371)
- **代码层面**：`mla_attn_quant_fusion.py` 编译 pass 支持把 AITER 的 `rocm_aiter_per_tensor_quant` 融合进 MLA attention 输出。新增 ROCm AITER 专用测试。
- **新手概念课堂**：融合（fusion）就像"炒菜时盐和酱油一起倒"，而不是分两次。MLA 输出后本来要单独做 FP8 量化，现在直接在 attention kernel 里顺手做了，省一次显存往返。
- **对你有什么影响**：AMD GPU 上跑 DeepSeek 等 MLA 模型 + AITER + FP8，端到端延迟降低。

### 14. [PCP] Allow decode-only full CUDA graphs for DSv3.2/GLM-5.2/5.3 sparse MLA with DCP=1 (#62377)
- **代码层面**：`pcp_manager.py` 的 `validate_config` 中，对 `model_type in ("deepseek_v32", "glm_moe_dsa")` 且 DCP=1 时放行 CUDA Graph，其他稀疏 MLA 仍报错。
- **新手概念课堂**：见知识点 4。CUDA Graph 是"录播"，但有些模型录播会卡死。这个 PR 给两个特定模型开了"录播白名单"。
- **对你有什么影响**：部署 DeepSeek-V3.2 或 GLM-5.2/5.3 时，decode 阶段可开 CUDA Graph，吞吐显著提升。

### 15. [Bugfix][Core] Requeue encoder-stalled waiting requests (#59030)
- **代码层面**：`scheduler.py` 中，等待队列遇到"编码器输入为 0 token"的请求时，从 `break` 改为走 `skip_request` 的延迟队列（`deferred_waiting` 或 `kv_holding_waiting`）。新增 3 个回归测试。
- **新手概念课堂**：见知识点 3。队头一个人卡住，不能让他堵死整条队，要把他请到"等候区"让后面的人先办。
- **对你有什么影响**：多模态模型（如 LLaVA）服务时，一个等图像的请求不会拖慢后面纯文本请求，吞吐和公平性都改善。

### 16. [Bugfix][DSv4.1] Keep the Engram all-to-all buffers in one allocator (#62269)
- **代码层面**：`engram.py` 中，`exchanged_rows` 改为 `torch.empty_like(self.packed_rows)`，确保发送和接收 buffer 来自同一个 allocator（cumem pool vs caching allocator 混用会导致 NCCL hang）。
- **新手概念课堂**：NCCL 通信像"两个人传纸条"，如果纸条来自不同"纸张供应商"（allocator），传递时会卡住。统一供应商就好。
- **对你有什么影响**：DeepSeek V4.1 的 Engram 模块在 DP all-to-all 时不再 hang。

### 17. [Quantization][Autoround][ARK] Add ARK W4A16(moe) Support (#50396)
- **代码层面**：重构 `inc_wna16_scheme.py`，把 `_resolve_awq_moe`/`_resolve_gptq_moe` 函数替换为 `INCWNA16MoEScheme` 类，引入 ARK 后端检测（`_ARK_STATE`）。测试改为 monkeypatch 类而非函数。
- **新手概念课堂**：W4A16 是"权重 4 位、激活 16 位"量化，省显存但保持精度。ARK 是 Intel 的量化后端。这个 PR 让 AutoRound 量化的 MoE 模型能在 Intel 平台上用 ARK 加速。
- **对你有什么影响**：Intel CPU/XPU 上部署 AutoRound 量化的 MoE 模型（如 Qwen3-30B-A3B）有了新后端选择。

### 18. [Frontend] Serve Gemma 4 on /v1/systemone and /v1/decisions (#62303)
- **代码层面**：`structured_decisions` 策略选择中新增 `Gemma4ForCausalLM`、`Gemma4ForConditionalGeneration`、`Gemma4UnifiedForConditionalGeneration` 映射到 `NextTokenStrategy`。文档表格更新，新增 tokenizer 标签对齐测试。
- **新手概念课堂**：`/v1/decisions` 是"结构化决策"端点，让模型输出分类/路由结果。Gemma 4 现在支持这个端点，且不同尺寸（E2B vs 12B）的回复起始 token 不同，测试确保标签对齐。
- **对你有什么影响**：可以用 Gemma 4 做分类/路由任务，通过标准 `/v1/decisions` 端点调用。

### 19. [Bugfix][LoRA] Apply text-only LoRAs to the LM of multimodal wrappers (#62224)
- **代码层面**：`model_manager.py` 中新增 `_get_lora_layer_weights` 逻辑，通过 `mm_mapping` 把纯文本 LoRA 的模块名（如 `model.layers.0...`）映射到多模态 wrapper 的 `language_model.` 前缀。打包 MoE 时用 `_pop_lora_weights` 统一清理。
- **新手概念课堂**：多模态模型 = "眼睛 + 大脑"。你训练的纯文本 LoRA 只针对"大脑"，但多模态模型的"大脑"名字前面多了 `language_model.` 前缀，导致匹配不上、静默失效。这个 PR 加了"别名映射"。
- **对你有什么影响**：给 LLaVA/Qwen-VL 等多模态模型加载文本 LoRA 终于生效了，不再白加载。

### 20. [Model] Add ColUltraVBERT late-interaction retrieval model (#62237)
- **代码层面**：新增 `vllm/model_executor/models/colultravbert.py`，实现 UltraBERT 文本编码器 + SigLIP 视觉编码器的 late-interaction 检索模型，输出 per-token 向量供 MaxSim 打分。新增测试和文档。
- **新手概念课堂**：Late-interaction（如 ColBERT）是"每个 token 都留一个向量"，检索时做细粒度匹配，比单向量 embedding 更准但更占空间。ColUltraVBERT 是图文版。
- **对你有什么影响**：可以用 vLLM 部署图文检索服务，通过 `pooling_task="token_embed"` 拿到 per-token 向量做 MaxSim 打分。

## sglang
## 💡 今日关键词

- **关键词：DeepSeek-V4.1 的"分块 CUDA Graph"与 PD 分离**
  - 一句话解释：把大模型推理拆成 prefill（预填充）和 decode（解码）两个角色，分别跑在不同 GPU 上；同时把模型尾部若干层"录制"成 CUDA Graph 复用，减少启动开销。
  - 为什么社区在关注：DeepSeek 系列是开源推理的标杆，其 V4.1 的 PD 分离、MegaAttention 默认化、KV 布局握手校验，说明"生产级 PD 部署"正在从实验走向默认。
  - 对新手意味着什么：如果你学推理系统，PD 分离 + CUDA Graph 是必学组合，是当下大厂推理框架的核心竞争力。

- **关键词：异构硬件适配（AMD / Intel XPU / CPU）**
  - 一句话解释：同一份 SGLang 代码要能在 NVIDIA、AMD ROCm、Intel XPU、纯 CPU 上跑起来。
  - 为什么社区在关注：今天有 AMD MI325X/350X 的 ROCm 镜像说明、XPU 双向图像注意力修复、XPU 默认开启 fused QK-norm+RoPE、CPU 支持 LTX/MOVA/Muse 模型、CPU cgroup 内存感知等——异构已是主战场。
  - 对新手意味着什么：只会写 CUDA kernel 不够了，理解 ROCm/XPU 的差异、能读懂 platform hook，是差异化竞争力。

- **关键词：LoRA 多适配器热切换**
  - 一句话解释：一个服务同时挂多个 LoRA（低秩微调）适配器，请求时按需切换，不用重启。
  - 为什么社区在关注：diffusion 的 LoRA 切换 bug 修复（commit 8）说明多租户/多风格场景是真实需求，且切换时的"残留合并"是隐蔽坑。
  - 对新手意味着什么：学 LoRA 不只会 merge，还要懂 unmerge、缓存、disable 标志位的状态机。

- **关键词：量化与 dtype 精度（bf16 vs fp32 gate、FP8 KV）**
  - 一句话解释：用更低精度（bf16/fp8）存权重和 KV 缓存来省显存、提速度，但某些地方（如 MoE 路由 gate）必须保留 fp32 才稳定。
  - 为什么社区在关注：commit 9 把 bailing MoE gate 从 bf16 改回跟随 config（默认 fp32），因为 bf16 会扰动路由导致输出方差；commit 10 讨论 FP8 KV 布局。
  - 对新手意味着什么：量化不是"越低越好"，要理解哪些算子对精度敏感——这是调优的核心直觉。

- **关键词：CUDA Graph 的共享内存与 PDL（Programmatic Dependent Launch）**
  - 一句话解释：CUDA Graph 把一串 kernel 启动"录"下来一次性回放；PDL 让后一个 kernel 在前一个还没完全结束时就开始预热。
  - 为什么社区在关注：commit 11 修复 FlashMLA scheduler 动态共享内存超限，commit 5 给 GEMV kernel 加 PDL 开关——都是性能细节。
  - 对新手意味着什么：想写高性能 kernel，共享内存预算和 kernel 间依赖是绕不开的硬知识。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph 与"分桶录制"（row bucket graph）**
  - **是什么**：CUDA Graph 把一段固定的 GPU 操作序列录制下来，之后用不同输入"回放"，省去每次重新启动 kernel 的 CPU 开销。SGLang 新增 `row_bucket_graph_runner.py`，按"行数桶"（如 128 的倍数）各录一张图，运行时选最接近的桶。
  - **为什么重要**：大模型 decode 每步只处理很少 token，CPU 启动开销占比极高。不懂 CUDA Graph，你会以为 GPU 利用率低是算力问题，其实是"启动太慢"。分桶是为了处理变长 batch——图是静态的，输入行数却是动态的。
  - **动手试试**：
    ```python
    import torch
    # 1) 普通执行
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    def run(): return a @ b
    # 2) 录制
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        c = a @ b
    # 3) 回放
    g.replay()
    torch.cuda.synchronize()
    print("graph ok, out shape:", c.shape)
    ```
  - **预期结果**：看到 `graph ok`，说明你成功录制并回放了一张图。再试着改变 `a` 的 shape（比如 512 行）——回放会失败或结果错乱，这就是为什么需要"分桶"。

- **知识点 2：PD 分离与 KV 布局握手**
  - **是什么**：Prefill 服务器算完 prompt 的 KV 缓存后，通过网络传给 Decode 服务器继续生成。两边必须约定 KV 缓存的内存布局（多少字节/token、怎么排），否则传过去是乱码。
  - **为什么重要**：commit 10 给 PD 握手加了 `kv_layout` 字段，布局不一致时**首个请求就报 `KV layout mismatch`**。这是典型的"分布式系统里静默错误比崩溃更可怕"——以前只比较 dtype，布局不同也能通过检查，然后返回垃圾。
  - **动手试试**：不用真跑 PD，先理解握手协议：
    ```python
    # 模拟握手校验
    prefill_info = {"kv_cache_dtype_str": "fp8_e4m3", "kv_layout": "v41"}  # 528B/token
    decode_info  = {"kv_cache_dtype_str": "fp8_e4m3", "kv_layout": "v4"}   # 584B/token
    def check(pre, dec):
        if pre["kv_cache_dtype_str"] != dec["kv_cache_dtype_str"]:
            raise ValueError("dtype mismatch")
        if pre.get("kv_layout") != dec.get("kv_layout"):
            raise ValueError("KV layout mismatch")
    check(prefill_info, decode_info)  # 应抛错
    ```
  - **预期结果**：看到 `ValueError: KV layout mismatch`。这说明"光看 dtype 不够"，任何跨进程传输都要校验完整的内存契约。

- **知识点 3：MoE 路由的精度敏感性（bf16 vs fp32 gate）**
  - **是什么**：MoE（混合专家）模型里，一个"门控"（gate）网络决定每个 token 送给哪几个专家。commit 9 之前把 gate 跑在 bf16 提速 17%，但发现路由扰动导致输出方差，于是改回跟随 config 的 fp32。
  - **为什么重要**：路由是离散决策（top-k），微小的数值差异可能让 token 去不同的专家，输出就变了。这是"量化会改结果"的典型案例，不是 bug 而是精度权衡。
  - **动手试试**：
    ```python
    import torch
    torch.manual_seed(0)
    logits = torch.randn(1, 8)  # 8 个专家的打分
    topk = 2
    # fp32 路由
    idx32 = torch.topk(logits, topk).indices
    # 模拟 bf16 量化后的路由
    logits_bf16 = logits.to(torch.bfloat16).to(torch.float32)
    idx_bf16 = torch.topk(logits_bf16, topk).indices
    print("fp32:", idx32.tolist(), "bf16:", idx_bf16.tolist())
    print("same?", torch.equal(idx32, idx_bf16))
    ```
  - **预期结果**：多跑几个 seed，你会偶尔看到 `same? False`——这就是"bf16 扰动路由"的最小复现。真实模型里这种差异会被放大成输出方差。

- **知识点 4：共享内存预算与动态共享内存（FlashMLA scheduler 修复）**
  - **是什么**：GPU 每个 block 的共享内存（shared memory）有上限（常 48KB 静态 + 动态部分）。commit 11 里 scheduler kernel 的静态+动态共享内存加起来可能超默认限制，于是显式调用 `set_max_dynamic_smem_per_device` 申请 48KB 动态额度。
  - **为什么重要**：写 kernel 时"我的 shared 数组不大"是错觉——编译器还会分配静态部分。不懂这个，你会遇到莫名的 `too much shared data` 启动失败。
  - **动手试试**：
    ```python
    import torch
    props = torch.cuda.get_device_properties(0)
    print("shared mem per block:", props.shared_memory_per_block)
    print("shared mem per SM:", props.shared_memory_per_multiprocessor)
    ```
  - **预期结果**：看到类似 48KB / 100KB+ 的数字。这就是你的"预算表"，写 kernel 前先算清楚静态+动态总和。

- **知识点 5：CPU 上的 autocast 陷阱（temporary_module_fp32_dtype）**
  - **是什么**：commit 6 发现 `torch.autocast("cpu", dtype=torch.float32)` 在 CPU 上会**静默禁用** autocast（因为 CPU 支持列表只有 bf16/fp16），于是改用 `temporary_module_fp32_dtype` 手动把模块临时转 fp32。
  - **为什么重要**：这是"看似生效其实没生效"的经典坑。CPU 推理越来越常见，autocast 语义在 CPU/GPU 上不一致，不懂就会得到错误精度。
  - **动手试试**：
    ```python
    import torch, warnings
    warnings.simplefilter("always")
    with torch.autocast("cpu", dtype=torch.float32):
        print("autocast enabled?", torch.is_autocast_enabled("cpu"))
    ```
  - **预期结果**：会看到 warning "target dtype is not supported. Disabling autocast." 且 `enabled? False`。这就是 commit 6 注释里引用的 PyTorch 源码行为。

---

## 🔥 重要更新

1. **Commit 10 — DeepSeek-V4.1 PD 默认 MegaAttention + KV 布局进握手**：PD 部署从"能用"走向"默认安全"，布局不一致会在首个请求就报错，而不是静默返回垃圾。生产部署必读。
2. **Commit 3 — DeepSeek-V4.1 decoder 尾部层走 CUDA Graph**：新增 `row_bucket_graph_runner.py`，eager prefill 步骤的尾部层也能用图回放，是推理性能的系统级优化。
3. **Commit 9 — bailing MoE gate 精度回退**：bf16 gate 提速 17% 但因输出方差被撤回，改为跟随 config（默认 fp32）。提醒所有人：量化的"提速"可能以结果不稳定为代价。
4. **Commit 8 — Diffusion LoRA 切换残留合并修复**：切到新 adapter 时，旧 adapter 覆盖的层若未被新 adapter 覆盖，必须 unmerge + clear，否则权重被污染。多 LoRA 服务的关键 bug。

---

## 📋 逐条解读

### 1. [Doc] cookbook: GB300 full-set accuracy and tool follow-up output for Ling-3.1-flash
- **代码层面**：只改文档 `docs/cookbook/.../Ling-3.1-flash.mdx`。把"低延迟"和"高吞吐"两套配方分开写清楚：低延迟是 TP-only（EP=1）+ NEXTN 投机解码，高吞吐是 TP8+EP8 专家并行；新增 AMD MI325X/MI350X 的 ROCm 镜像说明（需自己 build 一个换分支的镜像）；更新性能数字（H200 TP8 269.2 tok/s vs TP8+EP8 239.5 tok/s）。
- **新手概念课堂**：TP（张量并行）是把一个大矩阵切给多张卡算；EP（专家并行）是把 MoE 的不同专家分给不同卡。类比：TP 像把一个三明治横着切给多人同时吃，EP 像把自助餐的不同菜分到不同桌上。
- **对你有什么影响**：如果你要在 H200/GB300/AMD 上部署 Ling-3.1-flash，照文档选配方即可；AMD 用户注意镜像可能缺 ROCm gated-MLA 修复，需按文档 build 薄镜像。

### 2. [Fix] [XPU] gemma3: Enable bidirectional image attention
- **代码层面**：改 `python/sglang/srt/layers/attention/xpu_backend.py`。新增 `_bidirectional_image_attn_args`，在 Gemma-3 多模态 prefill 时，把图像 token 的 attention 从因果（只看前面）改成双向（图像内部互相看），文本 token 仍保持因果。还处理了滑动窗口与双向 mask 不能共存的情况，以及有 prefix 缓存时的特殊路径。
- **新手概念课堂**：因果 attention 像"只能看前面的人"，双向 attention 像"整个图像里的人互相都能看见"。图像理解需要双向，文本生成需要因果——同一个模型里两种模式并存。
- **对你有什么影响**：在 Intel XPU 上用 Gemma-3 做图文理解，结果会更准确；普通 NVIDIA 用户不受影响。

### 3. [Perf] deepseek-v4.1: Run decoder replay tail layers from CUDA graphs on eager prefill steps
- **代码层面**：新增 `python/sglang/srt/model_executor/runner/row_bucket_graph_runner.py`（220 行），按行数桶录制 layer range 的 breakable CUDA Graph；新增环境变量 `SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS`（默认 2048）；在 `deepseek_v4_backend.py` 里让低比例压缩在 tail graph 内走 eager break。
- **新手概念课堂**：breakable CUDA Graph 像"录好的剧本，但中间留了几个可以临时改台词的位置"——大部分 kernel 回放，少数需要读实时状态的算子走 eager。
- **对你有什么影响**：DeepSeek-V4.1 在 prefill 阶段也能享受 CUDA Graph 的低启动开销，吞吐提升；普通用户无感知，但延迟更低。

### 4. [Fix] docker: Install numactl in runtime images
- **代码层面**：在 `docker/Dockerfile` 和 `Dockerfile.cu134` 的 runtime apt 安装列表里加 `numactl`。
- **新手概念课堂**：NUMA 是"多 CPU 插槽的内存分区"，跨插槽访问内存更慢。`numactl` 是绑定进程到特定 NUMA 节点的工具；`SGLANG_NUMA_BIND_V2` 需要它。
- **对你有什么影响**：用官方 runtime 镜像跑 NUMA 绑定时不再报"找不到 numactl"。

### 5. [Feat] nemotron: Port Nemotron 3.5 Super serving changes
- **代码层面**：改多个 JIT kernel（`hopper_bf16_gemv.cuh`、`sm120_fp8_gemv.cuh` 等），给 GEMV kernel 加 `kUsePDL` 模板参数和 `PDLTriggerSecondary/PDLWaitPrimary`；放宽 `kUnroll==1` 时的 K 对齐要求；还有大量 Nemotron 模型侧改动（截断显示）。
- **新手概念课堂**：PDL（Programmatic Dependent Launch）让后一个 kernel 在前一个还没完全结束时就开始"预热"，减少 kernel 间气泡。类比：接力赛里下一棒提前起跑。
- **对你有什么影响**：Nemotron 3.5 Super 在 Hopper/Blackwell 上推理更快；其他模型不受影响。

### 6. [Feat] [CPU] model: Support LTX and MOVA models on CPU
- **代码层面**：改 `ltx_2_vocoder.py` 和 `mova.py`，把 CPU 上的 `torch.autocast("cpu", dtype=torch.float32)`（会静默失效）替换成 `temporary_module_fp32_dtype`；`lora/pipeline.py` 用 `current_platform.synchronize()/empty_cache()` 替代直接调 device module。
- **新手概念课堂**：autocast 是"自动混合精度"——大部分算子用低精度，少数用高精度。但 CPU 上 fp32 autocast 不被支持，会静默关闭，导致你以为在跑 fp32 其实跑的是原 dtype。
- **对你有什么影响**：在纯 CPU 上跑 LTX/MOVA 视频生成模型，精度行为正确了。

### 7. [Feat] [CPU] model: Support Muse models on CPU
- **代码层面**：改 `muse_glimmer.py` 配置，新增 `head_dim` 和 `padded_projector_hidden_size` 字段；`update_config.py` 在 CPU TP 下对 `projector_hidden_size` 做 `round_up(projector_hidden_size, tp_size * 32)` 填充；`muse_glimmer.py` 模型代码适配填充后的维度。
- **新手概念课堂**：CPU 上做 TP 时，某些维度必须能被 TP size 整除，否则切不匀。填充（padding）就是"补几个空位让它能整除"，像排队时补几个人凑成整排。
- **对你有什么影响**：Muse 多模态模型能在 CPU 上做张量并行推理。

### 8. [Fix] diffusion: Unmerge LoRA layers the next adapter does not cover
- **代码层面**：`linear.py` 抽出 `clear_lora_weights()`；`unmerge_lora_weights` 去掉 `if self.disable_lora: return` 的早退；`pipeline.py` 在 `_apply_lora_to_layers` 里，若某层没有任何新 LoRA 且 `clear_existing`，先 `unmerge` 再 `clear` 再 `disable_lora=True`。新增单测验证。
- **新手概念课堂**：LoRA merge 是把低秩增量加到基座权重上；unmerge 是减回去。如果切换 adapter 时不 unmerge 旧的就直接 clear，基座权重里会残留旧增量——像换滤镜前没摘旧滤镜。
- **对你有什么影响**：多 LoRA 热切换服务不再出现"旧风格泄漏"。

### 9. [Fix] bailing: MoE gate dtype follows config.router_dtype
- **代码层面**：`bailing_moe_v3.py` 把硬编码的 `self.router_dtype = torch.bfloat16` 改成读 `config.router_dtype`，None 或 "fp32" 用 fp32，否则 bf16。注释说明 bf16 会扰动路由导致输出方差。
- **新手概念课堂**：MoE gate 是"分诊台"，决定 token 去哪个专家。分诊台的打分精度不够，token 可能被分错科室，结果就变了。
- **对你有什么影响**：bailing/ling-v3 系列模型输出更稳定、可复现；牺牲了一点 decode 速度换正确性。

### 10. [Perf] deepseek-v4.1: Default PD to MegaAttention, with the KV layout in the PD handshake
- **代码层面**：`deepseek_v4_hook.py` 去掉 `disaggregation_mode != "null"` 的排除，PD 也默认 `flashmla_mega`；`KVArgs` 和 `PrefillServerInfo` 新增 `kv_layout` 字段；握手时校验布局，不一致报 `KV layout mismatch`；trtllm backend 的 PD 断言文案更新（仍不允许）。
- **新手概念课堂**：PD 分离像"前端接待"（prefill 算 prompt）和"后端专家"（decode 逐字生成）分工。KV 缓存是接待记录，必须用同一种格式写，否则后端读不懂。
- **对你有什么影响**：DeepSeek-V4.1 的 PD 部署默认更快（MegaAttention），且布局错误会立刻报错而非静默出错。

### 11. [Fix] attention: Opt FlashMLA scheduler into dynamic shared memory
- **代码层面**：`flashmla_sched_meta.cuh` 加 `runtime::set_max_dynamic_smem_per_device<...>(device, 48*1024)`；新增测试 `test_schedule_at_shared_memory_boundary`，在 48KB 边界附近参数化验证。
- **新手概念课堂**：GPU block 的共享内存 = 静态（编译期固定）+ 动态（启动时申请）。默认动态额度有限，接近上限会启动失败。显式 opt-in 是"向驱动申请更大额度"。
- **对你有什么影响**：长上下文 FlashMLA 调度不再因共享内存超限崩溃。

### 12. [Perf] [AMD] deepseek-v4: Skip top-k refinement for proven positive-zero ties
- **代码层面**：`topk_v2.cuh` 把 `_pad` 改成 `tie_nonzero` 标志；在 zero bin 里检测是否有非零 tie value，若全是 0 则跳过昂贵的 refine_ties 路径；`topk_impl.cuh` 用 union 复用 `warp_sum` 和 `tie_nonzero` 省空间。
- **新手概念课堂**：top-k 在阈值处可能有很多"并列"（tie），需要二次精排。但如果并列值全是 0，且已知不会影响结果，就可以跳过精排——像考试并列第 10 名都是 0 分，不用再比小分。
- **对你有什么影响**：AMD 上 DeepSeek-V4 的 MoE top-k 更快。

### 13. [Perf] deepseek-v4.1: Stage late-layer DP row counts through pinned memory
- **代码层面**：`decoder_swa_bounded_replay.py` 里 `enter_late_layer_dp` 把 `gpu.new_tensor(counts)` 换成 `async_h2d(counts, ...)`，用 pinned memory 做异步 host-to-device 拷贝。
- **新手概念课堂**：pinned memory（页锁定内存）允许 GPU 直接 DMA 读写，不用先拷到临时缓冲区，异步传输可与计算重叠。类比：VIP 通道 vs 普通安检。
- **对你有什么影响**：DeepSeek-V4.1 尾部层 DP 行数同步更快，减少 CPU-GPU 同步等待。

### 14. [Fix] [CPU] memory: Honor cgroup memory.max when sizing CPU memory
- **代码层面**：主要是 CI 配置 `.github/workflows/pr-test-xeon.yml`，把 SPR 机器从"每台一个全机容器"改成"每台两个 socket 绑定容器"，用 `--memory` 限制容器内存；因为 SGLang 现在读 cgroup `memory.max` 来算 KV 池大小（PR #38260 + empty-NUMA-node 修复）。
- **新手概念课堂**：cgroup 是 Linux 的容器资源限制机制。以前 SGLang 看的是宿主机总内存，容器限了也不认；现在认 cgroup，两个容器并行不会互相 OOM。
- **对你有什么影响**：CPU 推理部署在容器里更安全，不会因为"以为有整机内存"而 OOM。

### 15. [Fix] Z-Image: Synchronize device after warmup request
- **代码层面**：`gpu_worker.py` 的 `_finalize_output_batch` 里，warmup 请求结束后（非 CPU 平台）调用 `torch.get_device_module().synchronize()`。
- **新手概念课堂**：warmup 请求不保存输出，所以没有下游操作等它的 GPU 工作完成；不同步的话，下一个请求的第一个阻塞操作会"替它付账"，表现为首请求延迟尖峰。
- **对你有什么影响**：Z-Image 服务冷启动后的第一个真实请求延迟更稳定。

### 16. [Doc] cookbook: Refresh SemiAnalysis AgentX launches for Qwen3.8-Flash-Next H200
- **代码层面**：只改 `docs/src/snippets/agentx/Qwen/qwen3.8-flash-next.jsx`。更新镜像到 `v0.5.22-cu130@sha256:...`，配置从 TP4·EP4 改为 TP2·EP2（2 GPU），环境变量顺序调整，来源 commit 更新。
- **新手概念课堂**：cookbook 是"经过验证的配方"，AgentX 是 SemiAnalysis 的推理基准。TP2 比 TP4 省一半卡还保持性能，说明优化到位。
- **对你有什么影响**：想复现 Qwen3.8-Flash-Next 的 AgentX 成绩，照新配方用 2 张 H200 即可。

### 17. [Perf] [XPU] server_args: Enable fused QK-norm + RoPE by default
- **代码层面**：`exec_.py` 把 `enable_fused_qk_norm_rope` 从 `bool=False` 改成 `Optional[bool]=None` + `BooleanOptionalAction`；`platform_hook.py` 在 XPU 上若为 None 则 `declare_resolution(...=True)`；文档同步更新。
- **新手概念课堂**：QK-norm 是给 query/key 做归一化，RoPE 是旋转位置编码。两者可以融合成一个 kernel，省一次内存往返。XPU 上融合是纯赚，所以默认开。
- **对你有什么影响**：Intel XPU 用户默认获得 fused kernel 加速；其他平台行为不变，可用 `--enable-fused-qk-norm-rope` 强制开。

### 18. [Fix] diffusion: size the in-context slot from both input grids for wan-animate-2
- **代码层面**：`wan_animate_2_block.py` 的 `_InContextLayout` 把 `max_tokens_per_frame` 从只看生成网格改成 `max(生成网格, 参考视频网格)`；新增单测覆盖"两个网格 16 对齐后差一行一列"的场景。
- **新手概念课堂**：参考图和驱动视频各自按自己的宽高比缩放到像素预算，16 对齐后网格可能不同（一个 80×44，一个 79×45）。in-context slot 必须按较大的那个开，否则参考 K/V 拷贝会溢出。
- **对你有什么影响**：wan-animate-2 在特定分辨率组合下不再崩溃或出花屏。

### 19. [Feat] dsv4.1: Return prompt logprobs under encoder SWA bounded replay
- **代码层面**：删掉 `schedule_batch.py` 和 `tokenizer_manager.py` 里"encoder SWA replay 不支持 cached prompt logprobs"的报错；`tp_worker.py` 把 tail metadata 的 `extend_seq_lens_cpu` 传给 `drop_folded_rows`；`encoder_swa_replay.py` 的 `drop_folded_rows` 新增 `tail_lens` 参数。
- **新手概念课堂**：SWA（滑动窗口注意力）bounded replay 会把长 prompt 折叠成更少的行来省显存；折叠后行号变了，logprob 要映射回原始行。以前不支持，现在支持了。
- **对你有什么影响**：DeepSeek-V4.1 在 SWA replay 模式下也能返回 prompt logprobs（用于 RL、打分、调试）。

### 20. [Feat] rust/grpc: Add ListModels, Detokenize, and Abort to api.v1
- **代码层面**：改 `proto/sglang/api/v1/generate.proto`、`sampling.proto`、`service.proto`。新增 `ListModels`/`Detokenize`/`Abort` RPC；`GenerateRequest` 新增 `lora_path`（支持标量或列表）、`routing_key`、`max_thinking_tokens`、`kv_hints`（版本化 KV 提示信封）；`SamplingParams` 新增 `choice`（字面量候选，服务端转义成正则）。
- **新手概念课堂**：gRPC 是高性能 RPC 框架，proto 是接口定义。`oneof` 表示"多选一"，`OptionalStringOrList` 表示"标量或列表"——让 Python 的动态类型在强类型 proto 里表达出来。
- **对你有什么影响**：用 Rust/gRPC 客户端（如 Dynamo）的团队现在能直接调 ListModels/Detokenize/Abort，不用绕 HTTP；LoRA 和 KV hints 也支持了。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)