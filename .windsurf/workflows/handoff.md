---
description: Handoff protocol between AI sessions
auto_execution_mode: 3
---

# Session Handoff Protocol

Use this workflow when ending a work session to ensure smooth handoff to the next AI.

## Why This Matters

- Future AI sessions have no memory of previous work
- Clear handoffs prevent duplicate work and confusion
- Status documentation is the project's memory

## Handoff Checklist

### 1. Update Status Document

Update `/home/kali/code/docs/STATUS.md` with:

```markdown
# GENESIS Project Status

## Last Updated
[Date and time]

## Current Phase
[Phase A/B/C/D] - [Description]

## Completed This Session
- [What was done]
- [What was done]

## In Progress
- 🟡 [Task] - [Current state]

## Blocked
- 🔴 [Task] - [Why blocked] - [What's needed]

## Next Steps
1. [Immediate next task]
2. [Following task]
3. [Following task]

## Notes for Next Session
- [Important context]
- [Gotchas discovered]
- [Decisions made]
```

### 2. Commit Work

Ensure all changes are saved:
- Code files
- Test files
- Documentation updates

### 3. Run Tests

// turbo
```bash
pytest tests/ -v --tb=short
```

Document any failures in STATUS.md.

### 4. Summary Message

End the session with a clear summary:

```
## Session Summary

**Completed:**
- [List of completed items]

**In Progress:**
- [List of in-progress items]

**Blocked:**
- [List of blocked items with reasons]

**Next Steps:**
1. [What the next session should do first]
2. [What comes after]

**Important Notes:**
- [Any critical context for next session]
```

## Status Labels

Use these consistently:
- 🔴 `BLOCKED` — Cannot proceed
- 🟡 `IN_PROGRESS` — Being worked on
- 🟢 `COMPLETE` — Done and tested
- ⚪ `PENDING` — Not started
- 🔵 `REVIEW` — Needs human review

## What NOT to Do

- Don't leave work in broken state
- Don't skip the status update
- Don't assume next session remembers anything
- Don't leave uncommitted changes