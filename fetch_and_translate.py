"""
每日拉取 vLLM / SGLang 最新 commits，并用 AI 做新手向中文解读。
- 有 DEEPSEEK_API_KEY → AI 翻译（DeepSeek）
- 没有 → 只存原始 commits
"""
import os, sys, json, re
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError

REPOS = [
    ("vllm-project/vllm", "vllm"),
    ("sgl-project/sglang", "sglang"),
]
OUTPUT_DIR = "output"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 邮件配置（可选，不设置就不发邮件）
EMAIL_ENABLED = all(os.environ.get(k) for k in ["SMTP_SERVER", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"])

# ── 1. 拉取 commits ───────────────────────────────────────────
def fetch_commits(owner_repo: str, since_iso: str) -> list[dict]:
    """从 GitHub API 拉取 24h 内的 commits"""
    url = f"https://api.github.com/repos/{owner_repo}/commits?since={since_iso}&per_page=50"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    # 如果有 GITHUB_TOKEN 就用，提高 rate limit
    if token := os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except HTTPError as e:
        print(f"  ⚠ HTTP {e.code} fetching {owner_repo}")
        return []

    commits = []
    for c in data:
        msg = c.get("commit", {}).get("message", "")
        # 只取第一行作为标题
        title = msg.split("\n")[0].strip()
        commits.append({
            "sha": c.get("sha", "")[:8],
            "title": title,
            "full_message": msg,
            "author": c.get("commit", {}).get("author", {}).get("name", "unknown"),
            "date": c.get("commit", {}).get("author", {}).get("date", ""),
            "url": c.get("html_url", ""),
        })
    return commits


def filter_commits(commits: list[dict]) -> list[dict]:
    """过滤掉无意义的 commit（CI、typofix、version bump 等）"""
    skip_patterns = [
        r"(?i)^(ci|chore)\(?\w*\)?\s*:",
        r"(?i)^bump\s+version",
        r"(?i)^\d+\.\d+\.\d+$",
        r"(?i)^release",
        r"(?i)^update\s+(CHANGELOG|version|readme)",
        r"(?i)^nit\b",
        r"(?i)^fix\s+typo",
        r"(?i)^typo",
        r"(?i)^minor",
        r"(?i)^trivial",
        r"(?i)^format",
        r"(?i)^lint",
        r"(?i)^style\s*:",
        r"(?i)^docs\s*:\s*(fix|update)\s+typo",
    ]
    filtered = []
    for c in commits:
        title = c["title"]
        if not any(re.match(p, title) for p in skip_patterns):
            filtered.append(c)
    return filtered


# ── 2. AI 翻译 ─────────────────────────────────────────────────
def build_prompt(repo_name: str, commits: list[dict]) -> str:
    """构建给 AI 的 prompt"""
    lines = [f"以下是 {repo_name} 仓库最近 24 小时内的 {len(commits)} 个 commits：\n"]
    for i, c in enumerate(commits, 1):
        lines.append(f"{i}. [{c['sha']}] {c['title']}")
        lines.append(f"   作者: {c['author']}")
        # 包含完整 message（截断过长的）
        full = c["full_message"][:500]
        lines.append(f"   详情: {full}")
        lines.append("")
    lines.append("""请用中文对以上每个 commit 做新手向解读。格式要求：

## 🔥 重要更新（如果有的话）
（如果某个 commit 对普通用户影响很大，在这里强调）

## 📋 逐条解读

### 1. [commit 标题]
- **做了什么**：用一两句话通俗解释
- **涉及概念**：用简单比喻解释涉及的技术概念
- **影响**：对普通用户意味着什么

### 2. ...

## 💡 本周关键词
列出 2-3 个本期出现的技术关键词，每个一句话解释。

用 markdown 格式输出，不要输出其他内容。""")
    return "\n".join(lines)


def ai_translate(prompt: str) -> str:
    """调用 DeepSeek API 做翻译（兼容 OpenAI 格式）"""
    import urllib.request

    body = json.dumps({
        "model": "deepseek-chat",
        "max_tokens": 2048,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


# ── 3. 原始模式（无 AI） ───────────────────────────────────────
def raw_report(repo_name: str, commits: list[dict]) -> str:
    """无 AI 时的原始报告"""
    lines = [f"## {repo_name} — 最近 24h commits\n"]
    if not commits:
        lines.append("_无重要更新_\n")
    for c in commits:
        lines.append(f"- **[{c['title']}]({c['url']})** — {c['author']}")
        lines.append(f"  `{c['sha']}` {c['date']}")
    lines.append("")
    return "\n".join(lines)


# ── 3.5 邮件发送 ───────────────────────────────────────────────
def send_email(subject: str, body: str):
    """通过 SMTP 发送邮件报告"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_server = os.environ["SMTP_SERVER"]
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASS"]
    email_to = os.environ["EMAIL_TO"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = email_to

    # 纯文本版本
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 简易 HTML 版本（把 markdown 的 ## 转成 <h2>，链接转成 <a>）
    html_body = body
    html_body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html_body)
    html_body = re.sub(r"\n- ", r"\n<li>", html_body)
    html_body = re.sub(r"\n\n", r"<br><br>", html_body)
    html_body = f"<html><body><pre style='white-space:pre-wrap;font-family:system-ui,sans-serif'>{html_body}</pre></body></html>"
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, email_to, msg.as_string())
    print(f"📧 Email sent to {email_to}")


# ── 4. 主流程 ──────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=24)).isoformat()
    today_str = now.strftime("%Y-%m-%d")  # UTC date

    all_commits = {}
    for repo, name in REPOS:
        print(f"📡 Fetching {repo} ...")
        raw = fetch_commits(repo, since)
        filtered = filter_commits(raw)
        print(f"   {len(filtered)} significant commits (filtered out {len(raw) - len(filtered)})")
        all_commits[name] = filtered

    has_ai = bool(DEEPSEEK_KEY)
    print(f"\n🤖 AI translation: {'ENABLED' if has_ai else 'DISABLED (set DEEPSEEK_API_KEY to enable)'}")

    # 生成报告
    parts = [
        f"# 🚀 vLLM & SGLang 每日更新 — {today_str}\n",
        f"> 自动生成于 {now.strftime('%Y-%m-%d %H:%M UTC')} | AI 解读: {'✅' if has_ai else '❌ 仅原始数据'}\n",
    ]

    for repo, name in REPOS:
        commits = all_commits[name]
        if not commits:
            parts.append(f"## {name}\n_最近 24h 无重要更新_\n")
        elif has_ai:
            print(f"🧠 AI translating {name} ({len(commits)} commits) ...")
            prompt = build_prompt(name, commits)
            try:
                translation = ai_translate(prompt)
                parts.append(f"## {name}\n{translation}\n")
            except Exception as e:
                print(f"   ⚠ AI failed: {e}, falling back to raw")
                parts.append(raw_report(name, commits))
        else:
            parts.append(raw_report(name, commits))

    parts.append("---\n> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)")

    content = "\n".join(parts)

    # 写入文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily_path = os.path.join(OUTPUT_DIR, f"{today_str}.md")
    latest_path = os.path.join(OUTPUT_DIR, "latest.md")

    with open(daily_path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✅ Written to {daily_path} and {latest_path}")
    print(f"   Total size: {len(content)} chars")

    # 发送邮件（如果配置了 SMTP）
    if EMAIL_ENABLED:
        print(f"📧 Sending email ...")
        try:
            send_email(
                f"🚀 vLLM & SGLang 每日更新 — {today_str}",
                content,
            )
        except Exception as e:
            print(f"   ⚠ Email failed: {e}")


if __name__ == "__main__":
    main()
