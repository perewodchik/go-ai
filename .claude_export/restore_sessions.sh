#!/usr/bin/env bash
set -e

# ==============================================================================
# Claude Code Session Restoration Script for go-ai
# ==============================================================================

PROJECT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PROJECT_SLUG=$(echo "$PROJECT_DIR" | tr '/' '-')
CLAUDE_HOME="${HOME}/.claude"
EXPORT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================================="
echo " Restoring Claude Code sessions for go-ai"
echo "=========================================================="
echo " Project Directory: $PROJECT_DIR"
echo " Project Slug:      $PROJECT_SLUG"
echo " Target Claude Dir: $CLAUDE_HOME"
echo "=========================================================="

# Create target directories
TARGET_PROJECT_DIR="$CLAUDE_HOME/projects/$PROJECT_SLUG"
mkdir -p "$TARGET_PROJECT_DIR"
mkdir -p "$CLAUDE_HOME/tasks"
mkdir -p "$CLAUDE_HOME/session-env"

# 1. Restore Session Transcripts
if [ -d "$EXPORT_DIR/sessions" ]; then
    echo "[+] Restoring session transcripts to $TARGET_PROJECT_DIR..."
    cp -v "$EXPORT_DIR/sessions/"*.jsonl "$TARGET_PROJECT_DIR/" 2>/dev/null || true
fi

# 2. Restore Memory
if [ -d "$EXPORT_DIR/memory" ] && [ "$(ls -A "$EXPORT_DIR/memory" 2>/dev/null)" ]; then
    echo "[+] Restoring memory files..."
    mkdir -p "$TARGET_PROJECT_DIR/memory"
    cp -rv "$EXPORT_DIR/memory/"* "$TARGET_PROJECT_DIR/memory/" 2>/dev/null || true
fi

# 3. Restore Tasks
if [ -d "$EXPORT_DIR/tasks" ] && [ "$(ls -A "$EXPORT_DIR/tasks" 2>/dev/null)" ]; then
    echo "[+] Restoring subagent tasks to $CLAUDE_HOME/tasks/..."
    cp -rv "$EXPORT_DIR/tasks/"* "$CLAUDE_HOME/tasks/" 2>/dev/null || true
fi

# 4. Restore Session-env
if [ -d "$EXPORT_DIR/session-env" ] && [ "$(ls -A "$EXPORT_DIR/session-env" 2>/dev/null)" ]; then
    echo "[+] Restoring session environments to $CLAUDE_HOME/session-env/..."
    cp -rv "$EXPORT_DIR/session-env/"* "$CLAUDE_HOME/session-env/" 2>/dev/null || true
fi

# 5. Restore Project Config (.claude directory)
if [ -d "$EXPORT_DIR/project_config" ]; then
    echo "[+] Ensuring project .claude configs exist in $PROJECT_DIR/.claude..."
    mkdir -p "$PROJECT_DIR/.claude"
    if [ -f "$EXPORT_DIR/project_config/settings.local.json" ] && [ ! -f "$PROJECT_DIR/.claude/settings.local.json" ]; then
        cp "$EXPORT_DIR/project_config/settings.local.json" "$PROJECT_DIR/.claude/"
    fi
    if [ -f "$EXPORT_DIR/project_config/launch.json" ] && [ ! -f "$PROJECT_DIR/.claude/launch.json" ]; then
        cp "$EXPORT_DIR/project_config/launch.json" "$PROJECT_DIR/.claude/"
    fi
    if [ -f "$EXPORT_DIR/project_config/CLAUDE.md" ] && [ ! -f "$PROJECT_DIR/CLAUDE.md" ]; then
        cp "$EXPORT_DIR/project_config/CLAUDE.md" "$PROJECT_DIR/"
    fi
fi

echo ""
echo "=========================================================="
echo " Restoration complete!"
echo "=========================================================="
echo ""
echo "To resume your most recent session, run:"
echo "  cd \"$PROJECT_DIR\""
echo "  claude --continue"
echo ""
echo "Or to pick a session interactively:"
echo "  claude --resume"
echo ""
echo "Or resume a specific session by ID:"
cat "$EXPORT_DIR/manifest.json" | grep -E 'sessionId|resumeCommand' | sed 's/[",]//g'
echo "=========================================================="
