# resolve-ai: rules for agents

Instructions for coding agents working in this repo; a human reader wants `README.md`. Read
`README.md` for how it works, `RUNBOOK.md` for procedures, `TASK.md` for what is open. The
owner's global agent rules apply on top of these.

- Public repo: no personal data, account figures, third-party names or the owner's email in any file; a pre-commit scan blocks them.
- Real-time commits only; never backdate, never set `GIT_AUTHOR_DATE`.
- Release steps: `RUNBOOK.md` §Release.
- README, RUNBOOK, CLAUDE and TASK each hold one kind of fact; the README commands table is
  generated, never hand-edited.
- Traps: there is no LLM stage and never was, do not reintroduce the claim; the pipeline tests embed with the real model, so they need the model cached (first run needs network); never commit a `.db` or FAISS index file.
