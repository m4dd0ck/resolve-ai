# resolve-ai: rules for agents

Read `README.md` for how it works, `RUNBOOK.md` for procedures, `TASK.md` for what is open.
Global rules in `~/.claude/CLAUDE.md` still apply.

- Public repo: no personal data, account figures, third-party names or the owner's email in any file; the git gate blocks them.
- Real-time commits only; never backdate, never set `GIT_AUTHOR_DATE`.
- Releases follow hq CONVENTIONS (`~/developer/hq/CONVENTIONS.md`); the RUNBOOK names the steps.
- Docs follow the four-file skeleton: a fact lives in one file, and the commands table is generated.
- Traps: there is no LLM stage and never was, do not reintroduce the claim; the pipeline tests embed with the real model, so they need the model cached (first run needs network); never commit a `.db` or FAISS index file.
