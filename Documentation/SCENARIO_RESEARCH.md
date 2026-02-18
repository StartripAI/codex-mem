# Scenario Research (Why These Use Cases)

This note captures why `README` and the landing page prioritize specific workflows.

## Sources Reviewed

Primary docs / repos:
- OpenHands docs: [Your first OpenHands project](https://docs.all-hands.dev/modules/usage/your-first-project)
- Cline repository README: [cline/cline](https://github.com/cline/cline)
- Aider docs: [Usage](https://aider.chat/docs/usage.html)
- Aider docs: [Repository map](https://aider.chat/docs/repomap.html)
- Cursor docs: [Large codebases](https://cursor.com/docs/guides/advanced/large-codebases)
- Mem0 docs: [MCP docs index](https://docs.mem0.ai/mcp-docs/overview)

Social signal checks:
- r/cursor: [Debug Mode](https://www.reddit.com/r/cursor/comments/1pjb7s2/debug_mode/)
- r/cursor: [Cursor is spiralling into dead-ends?](https://www.reddit.com/r/cursor/comments/1is2zf7)

## High-Frequency Workflow Patterns

1. New-repo onboarding and architecture understanding
2. Issue-based bug triage and root-cause analysis
3. Debug loops from failing terminal/test output
4. Feature implementation with minimal-risk patching
5. Large-codebase refactoring with dependency awareness
6. PR review and regression risk scanning
7. Incident timeline reconstruction and forensics
8. Team handoff and next-command continuity

## How This Maps to Lakeside-mem

- `run-target` / `run-target-auto` provide a consistent execution entrypoint.
- 6-layer runtime keeps plan + retrieval + evidence + delivery explicit.
- `forced_next_input` preserves workflow continuity after every run.
- Hybrid retrieval helps keep token usage low in long debugging and implementation loops.
