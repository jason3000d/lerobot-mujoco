#!/usr/bin/env bash
set -euo pipefail

# ---- Setup for lerobot-mujoco sim ----

# 1. System GL/GLFW libs for the MuJoCo passive viewer (not pip-installable).
if command -v apt-get >/dev/null 2>&1; then
    echo ">> Installing system GL dependencies (sudo required)..."
    sudo apt-get install -y libglfw3 libgl1 libosmesa6 libglew-dev patchelf
elif command -v brew >/dev/null 2>&1; then
    echo ">> Installing GLFW via Homebrew..."
    brew install glfw
else
    echo ">> WARNING: no apt-get or brew found; install GLFW manually." >&2
fi

# 2. Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    # Make uv usable in THIS shell right now (installer puts it in ~/.local/bin).
    export PATH="$HOME/.local/bin:$PATH"
    source "$HOME/.local/bin/env" 2>/dev/null || true

    # Persist it to the right shell profile so future terminals find it too.
    LINE='export PATH="$HOME/.local/bin:$PATH"'
    case "${SHELL:-}" in
        */zsh)  PROFILE="$HOME/.zshrc" ;;
        */bash) PROFILE="$HOME/.bash_profile" ;;
        *)      PROFILE="$HOME/.profile" ;;
    esac
    touch "$PROFILE"
    if ! grep -Fqs "$LINE" "$PROFILE"; then
        echo "$LINE" >> "$PROFILE"
        echo ">> Added uv to PATH in $PROFILE"
    fi
fi

# Verify uv is actually reachable before continuing.
if ! command -v uv >/dev/null 2>&1; then
    echo ">> ERROR: uv still not on PATH. Restart your terminal and re-run." >&2
    exit 1
fi
echo ">> uv found: $(uv --version)"

# 3. Create env + install everything from pyproject.toml, locked.
echo ">> Syncing environment from pyproject.toml..."
uv sync

echo ""
echo ">> Done."
if [[ "$(uname)" == "Darwin" ]]; then
    echo ">> Run with:  uv run mjpython robo-code/sim.py"
else
    echo ">> Run with:  uv run python robo-code/sim.py"
fi