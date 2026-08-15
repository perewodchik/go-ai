# Claude Code Sessions & Config Export (go-ai)

Exported on: **2026-08-15**  
Project: `/Users/perewodchik/Documents/projects/go-ai`

---

## 📂 What Was Exported

This archive contains everything Claude Code needs to preserve and resume your conversation sessions across account switches or machines:

```
.claude_export/
├── manifest.json              # Full index & metadata of all exported sessions
├── sessions/                  # 9 conversation JSONL transcripts (~26 MB total)
│   ├── ad7f5aa9-8a15-40c5-a794-8e334151344e.jsonl (Model self-play mode)
│   ├── dee3f4e5-5ee6-4f31-bf15-d8c4405468c3.jsonl (Training settings & modal slider)
│   ├── f01925de-6972-48bf-9035-63676be24cd9.jsonl (Record games in review section)
│   ├── 4ab0a959-ffb0-44cf-b48f-82ea583cdedd.jsonl (Refactor game history folders)
│   ├── 2664aa42-ecd2-4455-b50d-141aaa271196.jsonl (Play page mouse input bug)
│   ├── 9f1087b7-caf4-407c-84d9-5a7d5990db05.jsonl (2-eye group stone placement rule)
│   ├── 84803ec8-9602-4721-9cd8-41aec7b94db7.jsonl (Territory refactor in review games)
│   ├── 99deee3b-a017-4e77-9ec8-49f59004b38d.jsonl (Go learning feature analysis)
│   └── 60aa12c9-53af-40af-9323-5b555ba75dcc.jsonl (Initial project study & CLAUDE.md)
├── tasks/                     # Task subagent metadata
├── session-env/               # Session environment states
├── memory/                    # Project memory cache
├── project_config/            # Project-level configs
│   ├── settings.local.json    # Local permissions & launch preferences
│   ├── launch.json            # VSCode / Claude debugger configurations
│   ├── CLAUDE.md              # Project instructions & conventions
│   └── claude_json_project_entry.json # Permissions & trust flags from ~/.claude.json
├── restore_sessions.sh        # One-click restore script
└── README.md                  # This guide
```

---

## 🔄 How to Switch Accounts & Continue Your Sessions

### Method 1: On the Same Machine (Instant Resume)
In Claude Code, session transcripts are stored under `~/.claude/projects/-Users-perewodchik-Documents-projects-go-ai/`. Claude Code links sessions to the **project directory path**, not to your account ID.

1. **Log out of your current account:**
   ```bash
   claude logout
   ```
2. **Log into your new account:**
   ```bash
   claude login
   ```
3. **Resume your session:**
   - To continue your latest active session:
     ```bash
     claude --continue
     # or: claude -c
     ```
   - To interactively choose which session to resume from a list:
     ```bash
     claude --resume
     ```
   - To resume a specific session by ID (see session table below):
     ```bash
     claude --resume <session-id>
     ```

---

### Method 2: On a New Machine or Clean Profile
If you switch accounts on a different laptop or clean profile:

1. Clone or copy your project repository:
   ```bash
   cd /path/to/go-ai
   ```
2. Run the restore script:
   ```bash
   ./.claude_export/restore_sessions.sh
   ```
3. Start Claude Code with resume:
   ```bash
   claude --resume
   ```

---

## 📋 Session Index & Resume Commands

| Session ID | Size | Messages | Topics / Initial Prompt | Resume Command |
| :--- | :--- | :--- | :--- | :--- |
| `ad7f5aa9-8a15-40c5-a794-8e334151344e` | 1.91 MB | 122 | Model vs Model self-play mode in play section | `claude --resume ad7f5aa9-8a15-40c5-a794-8e334151344e` |
| `dee3f4e5-5ee6-4f31-bf15-d8c4405468c3` | 2.56 MB | 229 | Training settings modal slider overflow fix | `claude --resume dee3f4e5-5ee6-4f31-bf15-d8c4405468c3` |
| `f01925de-6972-48bf-9035-63676be24cd9` | 0.89 MB | 75 | Record user games to appear in review section | `claude --resume f01925de-6972-48bf-9035-63676be24cd9` |
| `4ab0a959-ffb0-44cf-b48f-82ea583cdedd` | 1.00 MB | 81 | Refactor game history folder structure | `claude --resume 4ab0a959-ffb0-44cf-b48f-82ea583cdedd` |
| `2664aa42-ecd2-4455-b50d-141aaa271196` | 3.52 MB | 312 | Play page mouse hover & stone placement fix | `claude --resume 2664aa42-ecd2-4455-b50d-141aaa271196` |
| `9f1087b7-caf4-407c-84d9-5a7d5990db05` | 2.23 MB | 210 | Training setting: disable placing stones in 2 eyes | `claude --resume 9f1087b7-caf4-407c-84d9-5a7d5990db05` |
| `84803ec8-9602-4721-9cd8-41aec7b94db7` | 1.52 MB | 59 | Refactor territory view in review games | `claude --resume 84803ec8-9602-4721-9cd8-41aec7b94db7` |
| `99deee3b-a017-4e77-9ec8-49f59004b38d` | 1.58 MB | 179 | Feature analysis for Go learning project | `claude --resume 99deee3b-a017-4e77-9ec8-49f59004b38d` |
| `60aa12c9-53af-40af-9323-5b555ba75dcc` | 9.80 MB | 551 | Initial project analysis & CLAUDE.md setup | `claude --resume 60aa12c9-53af-40af-9323-5b555ba75dcc` |
