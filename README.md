# 🚀 vLLM & SGLang 每日更新追踪

每天自动拉取 [vLLM](https://github.com/vllm-project/vllm) 和 [SGLang](https://github.com/sgl-project/sglang) 的最新 commits，并用 AI 翻译成**新手友好的中文解读**。

## 效果预览

每天生成一份报告 → `output/2026-08-08.md`：

```markdown
# 🚀 vLLM & SGLang 每日更新 — 2026-08-08

## vllm
### 🔥 重要更新
- **修复了 HiCache ratio=1.6 时缓存命中率下降的问题**

### 📋 逐条解读
### 1. Fix L2 eviction policy under low ratio
- **做了什么**：修复了快速缓存层空间不足时数据被错误淘汰的 bug
- **涉及概念**：KV Cache 就像厨师的备菜台——台面越大，做菜越快
- **影响**：使用低 hicache ratio 的用户不会再遇到性能骤降
```

## 快速开始

### 1. Fork 或创建仓库

```bash
gh repo create daily-llm-tracker --public --push --source .
```

### 2. 启用 AI 翻译（可选但推荐）

1. 去 [Anthropic Console](https://console.anthropic.com/) 获取 API Key
2. 在仓库 Settings → Secrets and variables → Actions → New repository secret
   - Name: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-...`

> 💰 费用极低：每天约 $0.01-0.03（使用 Claude Haiku），一个月不到 $1

> ⚠️ 不设置 API Key 也能用，但只会保存原始英文 commits，没有中文解读

### 3. 手动测试

在 Actions 页面 → "Daily Commit Fetch & Translate" → Run workflow

## 工作原理

```
定时触发 (每天 9:43 北京时间)
    ↓
GitHub API 拉取 commits
    ↓
过滤 (跳过 CI/typo/version bump)
    ↓
Anthropic API 翻译成中文  ← 需要 API Key
    ↓
写入 output/ 目录
    ↓
自动 commit & push 回仓库
```

## 本地运行

```bash
pip install -r requirements.txt  # 实际上不需要额外依赖，纯标准库
python fetch_and_translate.py
```
