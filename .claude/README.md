# Assistant context — restoring the train of thought

Everything an AI assistant needs to pick this cell up mid-stream is versioned here.
Without it a fresh session re-derives (and re-breaks) things this project already
settled on the real robot.

## What lives where

| Path | What it is |
|------|-----------|
| [`../CLAUDE.md`](../CLAUDE.md) | Project instructions: the editing loop, the working agreement, roadmap/status. Read first. |
| [`../AGENTS.md`](../AGENTS.md) | Tool-agnostic entry point (Codex, Cursor, a fresh Claude) — same agreement, no Claude-specific assumptions. |
| [`../docs/agent-debug-map.md`](../docs/agent-debug-map.md) | Fast orientation before opening the long handoff docs. |
| `memory/` | 44 accumulated notes — what was measured, what was tried and **refused**, which fixes are load-bearing. |
| `memory/MEMORY.md` | The index. One line per note; it is what gets loaded into context each session. |

`memory/` is the part that used to be at risk: Claude Code keeps it **outside** the
checkout, under `~/.claude/projects/<mangled-path>/memory/`, on a single machine and
in no backup. It is mirrored into the repo so it travels with the code.

## Restoring on a new machine

1. **Clone and put the memory back where Claude Code looks for it:**

   ```
   py -3.10 tools/sync_claude_memory.py --to-live
   ```

   The live path is derived from the repo's absolute path (every non-alphanumeric
   character becomes a dash). If your install differs, set `CLAUDE_MEMORY_DIR`.

2. **Reinstall the Superpowers skills plugin** (third-party, deliberately not
   vendored here — it is ~6.4 MB of someone else's code with its own release cycle):

   | | |
   |---|---|
   | Marketplace | `anthropics/claude-plugins-official` (GitHub) |
   | Plugin | `superpowers@claude-plugins-official` |
   | Version in use | **6.3.0** (`44c9b2d6e889982ac18c27d05a19fefe335194e1`) |

   Install with `/plugin marketplace add anthropics/claude-plugins-official` then
   `/plugin install superpowers`. Pin 6.3.0 if a newer release changes behaviour.

## Keeping the mirror honest

The repo copy is a snapshot, so it rots unless refreshed. Before committing a
session's work:

```
py -3.10 tools/sync_claude_memory.py           # live -> repo
py -3.10 tools/sync_claude_memory.py --check   # exit 1 if the two have drifted
```

Nothing is deleted without `--prune`, so a stale mirror cannot destroy live notes.
