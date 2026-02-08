# Publish codex-mem to GitHub

## Prerequisites

- A GitHub repo: `StartripAI/codex-mem`
- A PAT with repo write permission

## 1) Create repo (choose one)

### Option A: Web UI
Create `https://github.com/new` with name `codex-mem` under org/user `StartripAI`.

### Option B: API (with PAT)

```bash
export GITHUB_TOKEN='<YOUR_PAT>'
curl -sS -X POST https://api.github.com/user/repos \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -d '{"name":"codex-mem","private":false}'
```

## 2) Push current local branch

```bash
cd /Users/alfred/projects/codex-mem
git remote set-url origin https://github.com/StartripAI/codex-mem.git
# recommended: use PAT once then store by credential helper
git push -u origin codex/init
```

## 3) If HTTPS prompts fail, use PAT in URL once

```bash
cd /Users/alfred/projects/codex-mem
git push -u https://<GITHUB_USERNAME>:<YOUR_PAT>@github.com/StartripAI/codex-mem.git codex/init
```

## 4) Open PR (optional)

```bash
# if main exists and you want PR from codex/init
# https://github.com/StartripAI/codex-mem/compare/main...codex/init
```
