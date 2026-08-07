"""
每日拉取 vLLM / SGLang 最新 commits + 代码 diff，用 DeepSeek AI 做新手向中文解读。
- 有 DEEPSEEK_API_KEY → AI 翻译 + 代码解读
- 没有 → 只存原始 commits
"""
import os, sys, json, re, html as html_mod
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError

REPOS = [
    ("vllm-project/vllm", "vllm"),
    ("sgl-project/sglang", "sglang"),
]
OUTPUT_DIR = "output"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
EMAIL_ENABLED = all(os.environ.get(k) for k in ["SMTP_SERVER", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"])

# 每个 commit 最多拉取的 diff 行数
MAX_DIFF_LINES = 80


# ── 1. GitHub API 请求 ──────────────────────────────────────────
def github_api(path: str, accept: str = "application/vnd.github+json") -> bytes:
    """封装 GitHub API 请求"""
    url = f"https://api.github.com/{path}"
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    if token := os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=60) as resp:
        return resp.read()


def fetch_commits(owner_repo: str, since_iso: str) -> list[dict]:
    """拉取 24h 内的 commits"""
    try:
        data = json.loads(github_api(
            f"repos/{owner_repo}/commits?since={since_iso}&per_page=50"
        ))
    except HTTPError as e:
        print(f"  ⚠ HTTP {e.code} fetching {owner_repo}")
        return []

    commits = []
    for c in data:
        msg = c.get("commit", {}).get("message", "")
        title = msg.split("\n")[0].strip()
        commits.append({
            "sha": c.get("sha", ""),
            "sha_short": c.get("sha", "")[:8],
            "title": title,
            "full_message": msg,
            "author": c.get("commit", {}).get("author", {}).get("name", "unknown"),
            "date": c.get("commit", {}).get("author", {}).get("date", ""),
            "url": c.get("html_url", ""),
        })
    return commits


def fetch_commit_diff(owner_repo: str, sha: str) -> str:
    """拉取单个 commit 的 diff（原始 unified diff 格式）"""
    try:
        diff_raw = github_api(
            f"repos/{owner_repo}/commits/{sha}",
            accept="application/vnd.github.v3.diff",
        ).decode("utf-8", errors="replace")
        lines = diff_raw.split("\n")
        if len(lines) > MAX_DIFF_LINES:
            diff_raw = "\n".join(lines[:MAX_DIFF_LINES]) + f"\n... (truncated, {len(lines) - MAX_DIFF_LINES} more lines)"
        return diff_raw
    except Exception as e:
        return f"(无法获取 diff: {e})"


def filter_commits(commits: list[dict]) -> list[dict]:
    """过滤无意义 commit，只保留改动实际代码的"""
    skip = [
        r"(?i)^(ci|chore)\(?\w*\)?\s*:",
        r"(?i)^bump\s+version",
        r"(?i)^\d+\.\d+\.\d+$",
        r"(?i)^release",
        r"(?i)^update\s+(CHANGELOG|version|readme)",
        r"(?i)^nit\b",
        r"(?i)^fix\s+typo",
        r"(?i)^typo",
        r"(?i)^minor\s*:",
        r"(?i)^trivial",
        r"(?i)^format\s*:",
        r"(?i)^lint",
        r"(?i)^style\s*:",
        r"(?i)^docs\s*:\s*(fix|update)\s+typo",
        r"(?i)^docs\(governance\)",
        r"(?i)^mypy\s+fix",
        r"(?i)^\[CI\]",
    ]
    out = []
    for c in commits:
        if not any(re.match(p, c["title"]) for p in skip):
            out.append(c)
    # 最多 20 个，避免 prompt 太长
    return out[:20]


# ── 2. AI 翻译 ─────────────────────────────────────────────────
def build_prompt(repo_name: str, commits: list[dict]) -> str:
    """构建含代码 diff 的 prompt"""
    parts = [f"你是资深程序员兼科普作者。以下是 {repo_name} 仓库最近 24 小时的 {len(commits)} 个重要 commits 及其代码变更。\n"]

    for i, c in enumerate(commits, 1):
        parts.append(f"---\n### Commit {i}: {c['title']}\n")
        parts.append(f"- 作者: {c['author']} | SHA: `{c['sha_short']}`\n")
        if c["full_message"]:
            msg = c["full_message"][:300]
            parts.append(f"- 提交说明: {msg}\n")
        if c.get("diff"):
            parts.append(f"\n**代码变更 (diff):**\n```diff\n{c['diff']}\n```\n")

    parts.append("""---
请从新手角度对以上每个 commit 做详细解读。**对于每个 commit，必须包含以下四个部分**：

## 🔥 重要更新
（筛选 2-4 个最有影响的，一两句话说明为什么重要）

## 📋 逐条解读

### 1. [commit 标题]
- **代码层面**：指出改的是哪个文件/函数，用通俗语言描述代码变更（比如"把 if 条件从 `x > 0` 改成 `x >= 0`，修复了边界情况"）
- **新手概念课堂**：用生活比喻解释涉及的技术概念（比如"端口就像公寓的门牌号，两个程序不能共用一个"）
- **对你有什么影响**：普通用户能感知的变化

（每个 commit 都按以上格式）

## 💡 今日关键词
列出 2-3 个出现的技术关键词，每个用一句话通俗解释。

用 markdown 格式输出，代码块用 ``` 包裹，不要输出其他闲聊内容。""")
    return "\n".join(parts)


def ai_translate(prompt: str) -> str:
    body = json.dumps({
        "model": "deepseek-chat",
        "max_tokens": 8192,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"},
    )
    with urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


# ── 3. 原始报告 ─────────────────────────────────────────────────
def raw_report(repo_name: str, commits: list[dict]) -> str:
    lines = [f"## {repo_name} — 最近 24h commits\n"]
    if not commits:
        lines.append("_无重要更新_\n")
    for c in commits:
        lines.append(f"- **[{c['title']}]({c['url']})** — {c['author']}")
        lines.append(f"  `{c['sha_short']}` {c['date']}")
    lines.append("")
    return "\n".join(lines)


# ── 4. 邮件 HTML 渲染 ──────────────────────────────────────────
def md_to_email_html(text: str) -> str:
    """把 markdown 转成 Gmail 友好的 HTML（不用 pre 包裹）"""
    lines = text.split("\n")
    out = []
    in_code = False
    in_list = False
    in_para = False

    def inline(s: str) -> str:
        """处理行内的 **bold** `code` [link](url)"""
        s = html_mod.escape(s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" style="color:#1a73e8">\1</a>', s)
        s = re.sub(r"`([^`]+)`", r'<code style="background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:0.9em">\1</code>', s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        return s

    def close_para():
        nonlocal in_para
        if in_para:
            out.append("</p>")
            in_para = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in lines:
        # 代码块
        if line.startswith("```"):
            close_para(); close_list()
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                lang = line[3:].strip() or "text"
                out.append(f'<pre style="background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:6px;overflow-x:auto;font-size:13px;line-height:1.5"><code>')
                in_code = True
            continue

        if in_code:
            out.append(html_mod.escape(line) if line else "&nbsp;")
            continue

        # 空行
        if not line.strip():
            close_para(); close_list()
            out.append("<br>")
            continue

        # 标题
        if line.startswith("### "):
            close_para(); close_list()
            out.append(f'<h3 style="color:#333;margin:16px 0 8px">{inline(line[4:])}</h3>')
        elif line.startswith("## "):
            close_para(); close_list()
            out.append(f'<h2 style="color:#1a1a1a;margin:20px 0 10px;border-bottom:2px solid #1a73e8;padding-bottom:6px">{inline(line[3:])}</h2>')
        elif line.startswith("# "):
            close_para(); close_list()
            out.append(f'<h1 style="color:#000;margin:24px 0 12px;font-size:22px">{inline(line[2:])}</h1>')
        elif line.startswith("> "):
            close_para(); close_list()
            out.append(f'<blockquote style="border-left:3px solid #1a73e8;padding:4px 12px;margin:8px 0;color:#555;background:#f8f9fa">{inline(line[2:])}</blockquote>')
        # 分隔线
        elif line.strip() == "---":
            close_para(); close_list()
            out.append('<hr style="border:none;border-top:1px solid #ddd;margin:16px 0">')
        # 列表项
        elif re.match(r"^- ", line):
            close_para()
            if not in_list:
                out.append('<ul style="padding-left:20px;margin:4px 0">')
                in_list = True
            out.append(f'<li style="margin:2px 0;line-height:1.6">{inline(line[2:])}</li>')
        # 普通段落
        else:
            close_list()
            if not in_para:
                out.append('<p style="margin:4px 0;line-height:1.7">')
                in_para = True
            else:
                out.append("<br>")
            out.append(inline(line))

    close_para(); close_list()

    body = "\n".join(out)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:720px;margin:0 auto;padding:20px;color:#222;background:#fff">
{body}
</body></html>"""


def send_email(subject: str, markdown_body: str):
    """发送 HTML 邮件"""
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

    # 纯文本
    msg.attach(MIMEText(markdown_body, "plain", "utf-8"))
    # HTML 渲染版
    msg.attach(MIMEText(md_to_email_html(markdown_body), "html", "utf-8"))

    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, email_to, msg.as_string())
    print(f"📧 Email sent to {email_to}")


# ── 5. 主流程 ──────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=24)).isoformat()
    today_str = now.strftime("%Y-%m-%d")

    all_commits = {}
    for repo, name in REPOS:
        print(f"📡 Fetching {repo} ...")
        raw = fetch_commits(repo, since)
        filtered = filter_commits(raw)
        print(f"   {len(filtered)} significant commits (filtered out {len(raw) - len(filtered)})")

        # 拉取每个 commit 的 diff
        if DEEPSEEK_KEY and filtered:
            print(f"   📥 Fetching diffs ...")
            for i, c in enumerate(filtered):
                owner, repo_name = repo.split("/")
                c["diff"] = fetch_commit_diff(repo, c["sha"])
                if i % 5 == 0:
                    print(f"      {i+1}/{len(filtered)} ...")

        all_commits[name] = filtered

    has_ai = bool(DEEPSEEK_KEY)
    print(f"\n🤖 AI translation: {'ENABLED' if has_ai else 'DISABLED'}")

    parts = [
        f"# 🚀 vLLM & SGLang 每日更新 — {today_str}\n",
        f"> 自动生成于 {now.strftime('%Y-%m-%d %H:%M UTC')} | AI 解读: {'✅ 含代码解读' if has_ai else '❌ 仅原始数据'}\n",
    ]

    for repo, name in REPOS:
        commits = all_commits[name]
        if not commits:
            parts.append(f"## {name}\n_最近 24h 无重要更新_\n")
        elif has_ai:
            print(f"🧠 AI translating {name} ({len(commits)} commits with diffs) ...")
            prompt = build_prompt(name, commits)
            try:
                translation = ai_translate(prompt)
                parts.append(f"## {name}\n{translation}\n")
            except Exception as e:
                print(f"   ⚠ AI failed: {e}, falling back to raw")
                # 即使 AI 失败，raw report 也带上 diff
                raw = raw_report(name, commits)
                for c in commits:
                    if c.get("diff"):
                        raw += f"\n<details><summary>📄 diff: {c['title'][:60]}</summary>\n\n```diff\n{c['diff']}\n```\n</details>\n"
                parts.append(raw)
        else:
            parts.append(raw_report(name, commits))

    parts.append("---\n> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)")

    content = "\n".join(parts)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily_path = os.path.join(OUTPUT_DIR, f"{today_str}.md")
    latest_path = os.path.join(OUTPUT_DIR, "latest.md")

    with open(daily_path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✅ Written to {daily_path} and {latest_path}")
    print(f"   Total size: {len(content)} chars")

    if EMAIL_ENABLED:
        print(f"📧 Sending email ...")
        try:
            send_email(f"🚀 vLLM & SGLang 每日更新 — {today_str}", content)
        except Exception as e:
            print(f"   ⚠ Email failed: {e}")


if __name__ == "__main__":
    main()
