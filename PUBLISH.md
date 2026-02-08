# Publish codex-mem to GitHub

## Prerequisites

- A GitHub repo: `<YOUR_ORG_OR_USER>/codex-mem`
- GitHub CLI authenticated (`gh auth login`) or browser-authenticated git credentials

## 1) Create repo (choose one)

### Option A: Web UI
Create `https://github.com/new` with name `codex-mem` under your org/user account.

### Option B: GitHub CLI

```bash
gh repo create codex-mem --public --source=. --remote=origin --push=false
```

## 2) Push current local branch

```bash
cd /ABS/PATH/codex-mem
git remote set-url origin https://github.com/<YOUR_ORG_OR_USER>/codex-mem.git
# recommended: use PAT once then store by credential helper
git push -u origin codex/init
```

## 3) If push auth fails, re-auth via GitHub CLI

```bash
gh auth login
git push -u origin codex/init
```

## 4) Open PR (optional)

```bash
# if main exists and you want PR from codex/init
# https://github.com/<YOUR_ORG_OR_USER>/codex-mem/compare/main...codex/init
```

## 5) Distribution channels beyond GitHub and X

- Product Hunt (launch post + demo GIF)
- Hacker News (`Show HN`)
- Reddit communities:
  - `r/LocalLLaMA`
  - `r/ChatGPTCoding`
  - `r/programming` (follow each subreddit rules)
- Dev.to article with install and smoke-test snippet
- Medium/Substack technical write-up
- Discord/Slack communities (AI engineering + agent tooling)
- Awesome lists PRs:
  - Awesome MCP / AI coding tool lists
  - agent memory tooling lists

## 6) Codex integration packaging checklist

- Keep MCP command deterministic and copy-pasteable
- Keep skill files in `Skills/codex-mem/`
- Keep one-command validation (`python3 Scripts/codex_mem_smoketest.py --root .`)
- Include screenshots/GIF of local web viewer in README
- Tag releases with clear changelog (`v0.x.y`)
