# 🚀 vLLM & SGLang 每日更新 — 2026-08-07

> 自动生成于 2026-08-07 14:58 UTC | AI 解读: ❌ 仅原始数据

## vllm — 最近 24h commits

- **[[Bugfix] Fix get_open_port() livelock on DP-reserved ports and cover get_open_ports_list (#50965)](https://github.com/vllm-project/vllm/commit/448344c0e29383adfe606a5c7ede72dd74705321)** — aoshen02
  `448344c0` 2026-08-07T14:39:33Z
- **[[Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier in wait_for_layer_load (#48534)](https://github.com/vllm-project/vllm/commit/47228db84ce59321e6e464f20fabe0ef86e6aef5)** — limeward
  `47228db8` 2026-08-07T14:23:33Z
- **[[PD][NixlPush][Bugfix] Fix prefix caching (#48758)](https://github.com/vllm-project/vllm/commit/d4ecb75ba2f53a1e445cf5ac277ea8a5e78d516b)** — Nicolò Lucchesi
  `d4ecb75b` 2026-08-07T14:21:17Z
- **[[Model] Enable Qwen3.8 for AMD Rocm (#50068)](https://github.com/vllm-project/vllm/commit/f2bfad9167a29e963f29f9ea79f2811513566ea6)** — haic0
  `f2bfad91` 2026-08-07T13:19:07Z
- **[[Bugfix] Skip fetching revision for model when model and weights_model are different (#51260)](https://github.com/vllm-project/vllm/commit/a231c5ceac87451b6dcf5ccdf0eef7a3634bc5d4)** — music-dino
  `a231c5ce` 2026-08-07T12:55:50Z
- **[Fix ROCm architecture import on non-ROCm platforms (#51357)](https://github.com/vllm-project/vllm/commit/d5aae2b4641c5091e604e235617e51e60a564710)** — Xiaochang Wu
  `d5aae2b4` 2026-08-07T12:32:56Z
- **[feat: extended EPLB support for Mistral Large 3 and additional MoE backends (#48355)](https://github.com/vllm-project/vllm/commit/ae934ba8a5577c580c33e3489290ff7d8bf1f83e)** — Julien Debache
  `ae934ba8` 2026-08-07T12:32:52Z
- **[[XPU] quick fix online quantization UT break (#51365)](https://github.com/vllm-project/vllm/commit/8d9b52f7c2514490bdadfd5eb0c931e58625df2e)** — Yan Ma
  `8d9b52f7` 2026-08-07T10:25:31Z
- **[[Misc] Add and enable Triton kernel unit tests on XPU (#45694)](https://github.com/vllm-project/vllm/commit/6b5bec7bedffa949fbd393fce720faf35831d746)** — pmanczak
  `6b5bec7b` 2026-08-07T09:53:03Z
- **[[PD][PushConnector] Record last activity of remotes to allow clean up of stale ones (#50234)](https://github.com/vllm-project/vllm/commit/5ec47f3e48e7f6da9b6caa1c804b3887f832a788)** — Nicolò Lucchesi
  `5ec47f3e` 2026-08-07T09:43:29Z
- **[[Bugfix][Platform] Stop re-initializing NVML on every device-capability check (fixes #50381) (#50393)](https://github.com/vllm-project/vllm/commit/4f76c8ad9d8ec06e91ac9c84895e07a1913d7726)** — Sebastian Woo
  `4f76c8ad` 2026-08-07T08:30:00Z
- **[[Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16 (#50833)](https://github.com/vllm-project/vllm/commit/b8db7f4abd2c864d5a7045b6d36fa36c2c7bb1e1)** — Hank_
  `b8db7f4a` 2026-08-07T08:21:13Z
- **[[Refactor] Remove kernel dead code (#51051)](https://github.com/vllm-project/vllm/commit/c84789c40b506a40d6a1ec15a704d53397c564a6)** — Wentao Ye
  `c84789c4` 2026-08-07T08:20:19Z
- **[Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel  (#47972)](https://github.com/vllm-project/vllm/commit/da788334bc0683cc44a58b4624e3f5a3c09a09e0)** — jimmy-adams
  `da788334` 2026-08-07T08:19:31Z
- **[[ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes (#48847)](https://github.com/vllm-project/vllm/commit/0de0362ea1c69b93f9ed36126a1b5c94f0ce2f22)** — stefankoncarevic
  `0de0362e` 2026-08-07T08:19:26Z
- **[[Feat][Core] Add disk offloading support to SimpleCPUOffloadConnector (#49644)](https://github.com/vllm-project/vllm/commit/58fcaa0baaa32ba0c34e1119f6ce4554ef8a6256)** — Guanyi Chen
  `58fcaa0b` 2026-08-07T07:37:47Z
- **[[rl] Stateful Trainer Send: NCCL + Sparse NCCL [3/N] (#50902)](https://github.com/vllm-project/vllm/commit/21ea5b4fa1062a379dd7e6795497ad6becd5a856)** — Aaron Hao
  `21ea5b4f` 2026-08-07T07:35:59Z
- **[[ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path (#51253)](https://github.com/vllm-project/vllm/commit/43d691ec6b1d26d3ef3d8725a7c7e4d8556eb984)** — kliuae
  `43d691ec` 2026-08-07T06:11:18Z
- **[[CI][XPU] Work around intermittent segfault in Intel XPU CI with VLLM_DISABLE_COMPILE_CACHE=1 (#51337)](https://github.com/vllm-project/vllm/commit/b706fd1628b06c216a945176a9fedfa808324803)** — Chaojun Zhang
  `b706fd16` 2026-08-07T05:18:25Z
- **[[Bugfix] Fix Mamba all-mode CPU offload boundary alignment (#51100)](https://github.com/vllm-project/vllm/commit/c810e5ee9976ad86b81d1277b53e76d0ee639414)** — Qianxu Wang
  `c810e5ee` 2026-08-07T04:37:48Z
- **[[Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance (#51222)](https://github.com/vllm-project/vllm/commit/dd856e48bbf969e3f0e561e8c76f6e92c76e0795)** — Tianyu Guo
  `dd856e48` 2026-08-07T03:40:47Z
- **[[Feat] Support thinking_token_budget in Model Runner V2 (#46727)](https://github.com/vllm-project/vllm/commit/72c0d6765793e4c7242c3586274af3e1a8aca170)** — Chauncey
  `72c0d676` 2026-08-07T03:36:18Z
- **[[CI] Re-enable FI autotune in GSM8K config for Qwen3.5-35B-A3B (#51293)](https://github.com/vllm-project/vllm/commit/5ac2684976ee22c04fe0d2f968c6cf6096b383f2)** — Artem Perevedentsev
  `5ac26849` 2026-08-07T02:42:34Z
- **[fix pre-commit broken (#51341)](https://github.com/vllm-project/vllm/commit/0406ba22c431e5fd2000165b594323f5afa312a2)** — Kunshang Ji
  `0406ba22` 2026-08-07T01:17:39Z
- **[[Kernel] Support Nvfp4 Cutedsl Moe Swiglu-oai and Relu2(non-gated) Activation (#47106)](https://github.com/vllm-project/vllm/commit/e08111211bc94f5e7f5ebd0f071ef5300f6f5564)** — Bi Tiekai
  `e0811121` 2026-08-07T00:42:45Z
- **[[V1] Copy NaN-in-logits counts to host asynchronously (#51304)](https://github.com/vllm-project/vllm/commit/b1e12d142d8c9533f857f8da13d8fc368e95a8cd)** — Nick Hill
  `b1e12d14` 2026-08-07T00:05:03Z
- **[[CI] Exclude KV-connector subtree from broad source dependencies (#51046)](https://github.com/vllm-project/vllm/commit/d35eb6c44071ea806018841c490f0d2f3219c485)** — Nicolò Lucchesi
  `d35eb6c4` 2026-08-06T23:51:10Z
- **[[Quantization] Share online weight scales across TP (#49764)](https://github.com/vllm-project/vllm/commit/8170c23c4fa36ffdc5890e5df46b4825fd9d0745)** — Matej Sirovatka
  `8170c23c` 2026-08-06T23:34:33Z
- **[[Model Runner V2] Fix -1 placeholder draft token ids in rejection sam… (#50939)](https://github.com/vllm-project/vllm/commit/27930df9c2bd14047be35ff2a986ca72fc65631a)** — Giancarlo Delfin
  `27930df9` 2026-08-06T23:32:00Z
- **[docs(governance): refresh committers list, add TSC note, update project leads (#51300)](https://github.com/vllm-project/vllm/commit/9bca7d840d7fa4677e58e7d163ddd191cccc40b7)** — Simon Mo
  `9bca7d84` 2026-08-06T23:28:17Z
- **[[ModelRunner v2] Enable decoder token-wise pooling (#50931)](https://github.com/vllm-project/vllm/commit/946452961265514622b95ea170839b214f39c2a0)** — Taneem Ibrahim
  `94645296` 2026-08-06T22:21:52Z
- **[fix: resolve silent request skipping in PRIORITY scheduling (#49206)](https://github.com/vllm-project/vllm/commit/4d341ca829d7fbad351b3a5a17d1405e63dc5bf2)** — Tejas
  `4d341ca8` 2026-08-06T22:21:21Z
- **[[Misc] Upgrade fastsafetensors version, fix metadata is null (#50827)](https://github.com/vllm-project/vllm/commit/a07086e4032e66aacae60ac2fc01e738096e9569)** — rongfu.leng
  `a07086e4` 2026-08-06T21:48:56Z
- **[[ModelRunner V2] Minor indexing optimizations (#51210)](https://github.com/vllm-project/vllm/commit/c5d470ac4ccd17ac9663db0d1c0e2060e5ae15ad)** — Nick Hill
  `c5d470ac` 2026-08-06T21:44:13Z
- **[[Bugfix] Fix packed KV block zeroing stride (#50276)](https://github.com/vllm-project/vllm/commit/d6af803f434397222674e8ea5cc6f25b3a208e62)** — wangxian001
  `d6af803f` 2026-08-06T20:49:06Z
- **[[Mypy Fix] Mypy fix for "vllm/model_executor/models/[aA][bB]" (#48977)](https://github.com/vllm-project/vllm/commit/adc3e03517d2e7333a3bb2083bb4d394a2986876)** — Wentao Ye
  `adc3e035` 2026-08-06T19:30:15Z
- **[[ROCm][MLA] Use asm decode for non-divisor small head counts (#50578)](https://github.com/vllm-project/vllm/commit/d8eabdbfbe93ecc8a8d5cb8a55c5067a443a8796)** — vanshbhatia-amd
  `d8eabdbf` 2026-08-06T18:56:53Z
- **[[Frontend] Watch frontend processes during engine startup (#43417)](https://github.com/vllm-project/vllm/commit/7b9f2dad8920f115c1caea36e096e43c04c3da68)** — Bugen Zhao
  `7b9f2dad` 2026-08-06T18:17:24Z
- **[[ROCm][CI] Update AITER AR+RMS e2e fusion counts for final-norm coverage (#51273)](https://github.com/vllm-project/vllm/commit/4f851bef6c4ef3a691aa19f798b220819d19dde4)** — Divakar Verma
  `4f851bef` 2026-08-06T18:05:16Z
- **[[Bugfix] Keep mamba align prefill chunks block-aligned past last_cache_position (#51113)](https://github.com/vllm-project/vllm/commit/c56f169d9ae46ca420617e2cf5f0c9135da0f651)** — Yifan Qiao
  `c56f169d` 2026-08-06T17:21:00Z
- **[[Weight processing] Copy over `new_data` attributes in `replace_parameter` (#49601)](https://github.com/vllm-project/vllm/commit/81be2e09aebfd1c45b3ed9f73d2850da8a72984c)** — fxmarty-amd
  `81be2e09` 2026-08-06T17:03:37Z
- **[[Attention][MLA] Per-request scheduling for MLA chunked context (#50613)](https://github.com/vllm-project/vllm/commit/b38e111d3e4806a553ec2798e2b075da7a8b03d3)** — Matthew Bonanni
  `b38e111d` 2026-08-06T16:45:44Z
- **[Fully generalise input embedding handling in Transformers modelling backend (#51247)](https://github.com/vllm-project/vllm/commit/e7b8d5946095a594af2cf7ca3c314b9806cb7c32)** — Harry Mellor
  `e7b8d594` 2026-08-06T16:21:24Z
- **[[CI] Run basic fullgraph correctness on one GPU (#51271)](https://github.com/vllm-project/vllm/commit/566c80edf9e770524b1506a9d681922d4601c70c)** — Michael Goin
  `566c80ed` 2026-08-06T16:09:30Z
- **[attn_res kernel latency improvements (#50185)](https://github.com/vllm-project/vllm/commit/7b4ed49628abd7860a435d6798feef76a944cb02)** — gnovack
  `7b4ed496` 2026-08-06T15:53:20Z
- **[Update vllm to point to flash-attention commit that builds FA3 with torch stable API. (Retry) (#49599)](https://github.com/vllm-project/vllm/commit/41e7746b82b43dd3454cd842d1bcfc30665eddb2)** — Chris Leonard
  `41e7746b` 2026-08-06T15:37:46Z
- **[[VocabParallelEmbedding] fix extra_repr fields concat (#51224)](https://github.com/vllm-project/vllm/commit/62a86318de3655f970baf7c2ff89c81a72c1a1b3)** — Ning Xie
  `62a86318` 2026-08-06T12:22:18Z
- **[Remove the XPU branch of topk_softplus_sqrt (#51242)](https://github.com/vllm-project/vllm/commit/1e05b21d61e6126e4811313f39c961bf8b314470)** — Liangqiusong
  `1e05b21d` 2026-08-06T12:18:45Z
- **[[Bugfix][KV Offload] Clean up resources after initialization failure (#51227)](https://github.com/vllm-project/vllm/commit/46e6a83ce12b5968d956279a1bd4611de16d69eb)** — AlexHuang
  `46e6a83c` 2026-08-06T11:42:13Z
- **[[Bugfix][Model] Add missing fused_qkv_a_proj to Kimi-Linear packed_modules_mapping (#51249)](https://github.com/vllm-project/vllm/commit/5fba75aefeedcf5b6cc27abf9bc145b6a49873a7)** — yjz
  `5fba75ae` 2026-08-06T11:39:17Z

## sglang — 最近 24h commits

- **[[diffusion] Enable breakable CUDA graph for LTX-2 (H200 two-stage e2e 10.75 s -> 6.90 s, 1.56x) (#33885)](https://github.com/sgl-project/sglang/commit/d4be483efb2674385d39774232c11e4135217ea6)** — Xiaoyu Zhang
  `d4be483e` 2026-08-07T14:29:14Z
- **[[diffusion] feat: make scheduler rpc deadlines explicit (#33965)](https://github.com/sgl-project/sglang/commit/bc148dfdc8478bcc438e5200eb5c12ad05d0c9bb)** — Yifei Suo
  `bc148dfd` 2026-08-07T13:33:23Z
- **[Fix vae fast path test after the gate refactor (#33983)](https://github.com/sgl-project/sglang/commit/bc8c0370415547726d07081b05cfaabf9c65e306)** — Ke Bao
  `bc8c0370` 2026-08-07T12:45:19Z
- **[Fix Nemotron W4A16 NVFP4 MoE backend (#33543)](https://github.com/sgl-project/sglang/commit/4020bc95a7b5b88b8de5f354f4850a9b1f881298)** — danielafrimi
  `4020bc95` 2026-08-07T12:00:28Z
- **[[diffusion] UX: speed up tp and fsdp checkpoint loading (#33960)](https://github.com/sgl-project/sglang/commit/5ca734fc3d0e314fb0b0993f9f7bbcd5a991bdb5)** — Mick
  `5ca734fc` 2026-08-07T11:23:30Z
- **[[diffusion] fix: bind each rank to accelerator before distributed init (#33054)](https://github.com/sgl-project/sglang/commit/1034977318ea4f113142ba57f6fe7ac379e61630)** — Dayananda V
  `10349773` 2026-08-07T11:01:58Z
- **[[diffusion] fix: enable bcg with tp (#33421)](https://github.com/sgl-project/sglang/commit/acb64db9e27e8c6969fb1c69748e514d1375c68c)** — Yihao Wang
  `acb64db9` 2026-08-07T11:01:05Z
- **[Fix IndexError in Triton backend with pipeline parallelism (#30340)](https://github.com/sgl-project/sglang/commit/fe52b49827e6560692d0c6b7dff2a719f5731fde)** — Dayananda V
  `fe52b498` 2026-08-07T10:58:04Z
- **[[diffusion] feat: gate /health and /health_generate on warmup completion and add liveness endpoint (#33787)](https://github.com/sgl-project/sglang/commit/7af3d000f294f230e3b277ceeb022aaf6e16147f)** — Lennox Fu
  `7af3d000` 2026-08-07T09:59:23Z
- **[[diffusion] model: support lingbot-video moe 30b t2v (#32341)](https://github.com/sgl-project/sglang/commit/a42683eb629b0aed12e34bd4f2d5a59c61098dc6)** — Pan Li
  `a42683eb` 2026-08-07T09:57:01Z
- **[[diffusion] feat: make ring admission a backend capability (#33928)](https://github.com/sgl-project/sglang/commit/13938fed3f06a8df06f23961bc338b911066dd61)** — Mick
  `13938fed` 2026-08-07T09:55:44Z
- **[[diffusion] perf: build qwen's masked varlen metadata host-side (#33954)](https://github.com/sgl-project/sglang/commit/28b43bf693a8d93d9241ef0a71a11eb245f6f3de)** — Mick
  `28b43bf6` 2026-08-07T09:55:15Z
- **[Move SWA chunk-cap hatch tests into the registered suite (#33975)](https://github.com/sgl-project/sglang/commit/0756a1d2b070bb98f6c788c4c4c7346c68da164a)** — Ke Bao
  `0756a1d2` 2026-08-07T09:49:07Z
- **[[NPU] [DOC] Upgrade recommendeded sglang version on Ascend NPU (#33976)](https://github.com/sgl-project/sglang/commit/470807ef746977b2e6a9170ffe72b3fd551e771f)** — amote-i
  `470807ef` 2026-08-07T09:45:00Z
- **[[diffusion] Z-Image bit-exact fused qk-norm (H200 Turbo 1024px e2e -6.4%) (#33886)](https://github.com/sgl-project/sglang/commit/572434e2f6a855725fc4f56977f36f2e222ac939)** — Xiaoyu Zhang
  `572434e2` 2026-08-07T09:03:28Z
- **[Fix prefill CP graph overflow with larger bucket search (#33906)](https://github.com/sgl-project/sglang/commit/5e60363960db96d56b519617d7a75be871f15a0e)** — Baizhou Zhang
  `5e603639` 2026-08-07T08:33:14Z
- **[[CI] Share VLM engines and prune launch matrices on the per-commit H100/H200 suites (#33944)](https://github.com/sgl-project/sglang/commit/7395ee833e61c62b8928bd2a6cab4b825313e2f1)** — Liangsheng Yin
  `7395ee83` 2026-08-07T08:22:22Z
- **[feat: Add flashinfer mHC fusion for DSV4 (#33616)](https://github.com/sgl-project/sglang/commit/3ed2a0adf3d87b0f527c173a500dfb40d64b572f)** — Trevor Morris
  `3ed2a0ad` 2026-08-07T08:01:33Z
- **[[diffusion] fix: scope the masked-path replicated guard to sp runs (#33953)](https://github.com/sgl-project/sglang/commit/85d611a055a1503fe65fcf6dfccc2a3836e1aff4)** — Mick
  `85d611a0` 2026-08-07T07:45:50Z
- **[Fix fractional simulated acceptance in DSpark (#33463)](https://github.com/sgl-project/sglang/commit/5e58af150339ca2f570cdbed523724f19eceafb2)** — weireweire
  `5e58af15` 2026-08-07T07:45:07Z
- **[[diffusion] CI: exercise the default sp selection in CI (#33931)](https://github.com/sgl-project/sglang/commit/9aadacfc5325d8f1103c8e16bd34046b47393bf2)** — Mick
  `9aadacfc` 2026-08-07T06:20:19Z
- **[[diffusion] fix: minimax-h3 text encoder device mismatch under --text-encoder-cpu-offload (#33864)](https://github.com/sgl-project/sglang/commit/a79340dedd4780a7678c1bce64351f717994a56d)** — triple-mu
  `a79340de` 2026-08-07T06:04:20Z
- **[[AMD][DI][CI] 8/N Add GLM-5.2 MXFP4 1P1D DI/CI recipes (base + MTP + DP8/EP8) (#32120)](https://github.com/sgl-project/sglang/commit/fc9479243ef8920f8269fd97803e47e64318c94f)** — Zhaoyi Li
  `fc947924` 2026-08-07T05:56:40Z
- **[[srt] Batch scheduler cache frees (#33475)](https://github.com/sgl-project/sglang/commit/4d4f8023c405cade959afb5073e23505b2643a2a)** — Leon Gao
  `4d4f8023` 2026-08-07T04:49:59Z
- **[[diffusion] refactor: gate fast vae paths by quality (#33849)](https://github.com/sgl-project/sglang/commit/c2657cc4bfda2c656abe6320a7149331544d0139)** — Mick
  `c2657cc4` 2026-08-07T04:39:55Z
- **[[diffusion] fix: fix 4/8-step distilled minimax-h3 turbo lora merge (#33875)](https://github.com/sgl-project/sglang/commit/914644e81c9b6fc31d60a1ab5327a4884514c074)** — WenhaoZhang
  `914644e8` 2026-08-07T04:38:37Z
- **[[diffusion] chore: route zimage and hunyuanvideo attention through USPAttention (#33923)](https://github.com/sgl-project/sglang/commit/6dc77e490dd9428d1ec9bbf4ff45bc1abf060ac1)** — Mick
  `6dc77e49` 2026-08-07T04:36:23Z
- **[[diffusion] chore: derive h3 attention admission from backend capabilities (#33707)](https://github.com/sgl-project/sglang/commit/698f019a5a4ebc8762339559fcc00191e796fc1c)** — Mick
  `698f019a` 2026-08-07T04:35:15Z
- **[[misc] Remove break-graph debug log; reclaim pid-less /dev/shm leaks in CI (#33929)](https://github.com/sgl-project/sglang/commit/afa79330b8d2f63934f721857f4462a1f4cc9a8a)** — Liangsheng Yin
  `afa79330` 2026-08-07T04:18:01Z
- **[Clean GLM-5.2 NVFP4 cookbook (#33935)](https://github.com/sgl-project/sglang/commit/0c3a76fa0a5bfab410b645f4143e7e8e3cc25c77)** — Brayden Zhong
  `0c3a76fa` 2026-08-07T03:51:34Z
- **[fix(qwen2_5vl): replace in-place += with out-of-place + on expand view in decode path (#22634)](https://github.com/sgl-project/sglang/commit/453ea21dd32eb9cfe572b3fce8856169955a871b)** — Liu Zhenlong
  `453ea21d` 2026-08-07T03:51:09Z
- **[[Distributed] Propagate semantic group names to PyTorch process groups (#32900)](https://github.com/sgl-project/sglang/commit/f9e6888b5a3abd5917cb92bd7cc76ce3ee9e1d42)** — CangYue
  `f9e6888b` 2026-08-07T03:38:12Z
- **[[Diffusion]Skipping tensor copying for non-BCG GLM-Image workflows (#33688)](https://github.com/sgl-project/sglang/commit/c54dc4582fddc11151567809c53e8cb421c83b7d)** — Elizaveta Martirosian
  `c54dc458` 2026-08-07T03:19:58Z
- **[[CI] Refresh the CPU HF cache base only on main-ref runs (#33904)](https://github.com/sgl-project/sglang/commit/163b739b34023c30231062a46caecd62fc4aff83)** — Liangsheng Yin
  `163b739b` 2026-08-07T02:37:59Z
- **[fix: preserve priority for batched embedding requests (#32977)](https://github.com/sgl-project/sglang/commit/fe6a05a8e81f29f63e0bb8a721bdac5ac5762b1e)** — Nikhil Kulkarni
  `fe6a05a8` 2026-08-07T02:26:18Z
- **[fix(gdn): skip the -1 padding sentinel in the chunked extend kernel (#33810)](https://github.com/sgl-project/sglang/commit/db8f3cdd11eba31258157ce30912a30200361cae)** — Yuwei An
  `db8f3cdd` 2026-08-07T01:48:47Z
- **[[diffusion] feat: support K/V-gather style sequence parallel (CP-like) attention (#32667)](https://github.com/sgl-project/sglang/commit/1e08b865f9aec934fa172ef3b8f721ee0e7700b3)** — Mick
  `1e08b865` 2026-08-07T01:39:28Z
- **[[diffusion] CI: fix output-rank test fixture (#33878)](https://github.com/sgl-project/sglang/commit/9ee658d4f645f83852cc457059142058cedbac87)** — Mick
  `9ee658d4` 2026-08-07T01:33:50Z
- **[[Fix] Reformat /vertex_generate successful predictions (#33446)](https://github.com/sgl-project/sglang/commit/2c3ecf32f1a8be038712a1d663ced6eae722a1c1)** — Dustin Luong
  `2c3ecf32` 2026-08-07T00:00:48Z
- **[Fix paged SWA retraction resume accounting (#33794)](https://github.com/sgl-project/sglang/commit/af7c62e3378ee143f53e8536c21679d8aac337e0)** — Hao Zhang
  `af7c62e3` 2026-08-06T23:32:54Z
- **[Fix sgl-deep-ep builder dependencies (#33866)](https://github.com/sgl-project/sglang/commit/e0af47b03edf20f0be25faf2dfecf77675a50255)** — Baizhou Zhang
  `e0af47b0` 2026-08-06T23:30:02Z
- **[Fix inference mode mismatch in FlashInfer warmup (#33788)](https://github.com/sgl-project/sglang/commit/bae29f716acd3c46f1620131b397b011cfa72c08)** — Po-Han Huang (NVIDIA)
  `bae29f71` 2026-08-06T23:11:41Z
- **[[AMD] Enable gfx1250 sgl-kernel builds (#32466)](https://github.com/sgl-project/sglang/commit/b38caebf09e847611f663be237f6028aab1bd8de)** — Oguz Ulgen
  `b38caebf` 2026-08-06T22:47:17Z
- **[[AMD] perf: compact Triton extend-attention for ragged prefill (AMD/HIP-only) (#29677)](https://github.com/sgl-project/sglang/commit/18e6c61c21ad39725522c008190d2b540dd6228d)** — valechen
  `18e6c61c` 2026-08-06T21:46:10Z
- **[Fix Mistral-Large-3 EAGLE draft skipping DeepseekV2Model.__init__ (#33785)](https://github.com/sgl-project/sglang/commit/dd7e4c91e2e17e96c8d564c1ae321ccf05ea2287)** — Brayden Zhong
  `dd7e4c91` 2026-08-06T20:59:28Z
- **[[Kimi-K3] Allow DSPARK verify on cutedsl_mla (fold_sq) (#33650)](https://github.com/sgl-project/sglang/commit/971932d66117af03f5a4833d5fdf1ee42fba2c79)** — Yuhao Yang
  `971932d6` 2026-08-06T20:54:25Z
- **[fix(PP): size the mamba pool per pipeline stage, not per whole model (#33666)](https://github.com/sgl-project/sglang/commit/2fc557254b3aaf539e80266e52a6d1e1f8da9980)** — YAMY
  `2fc55725` 2026-08-06T20:10:43Z
- **[[Deps] Upgrade CUDA PyTorch stack to 2.13 (#28836)](https://github.com/sgl-project/sglang/commit/434e646282e5c7fcaeb5a2df38bc34dc704a0e58)** — Mohammad Miadh Angkad
  `434e6462` 2026-08-06T19:08:44Z
- **[[ModelOpt FP4] Support online MoE weight quantization (#33115)](https://github.com/sgl-project/sglang/commit/4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63)** — Ziang Li
  `4ad990ba` 2026-08-06T18:01:55Z
- **[[Disagg][StagingBuffer][2/2] Support radix cache (#30545)](https://github.com/sgl-project/sglang/commit/05c7ebf64c1a42590328435e6f7352cfd1bb45a8)** — YAMY
  `05c7ebf6` 2026-08-06T15:59:35Z

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)