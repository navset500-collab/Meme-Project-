#!/bin/zsh
PROJECT_DIR="${0:A:h}"
PYTHON_BIN="$HOME/webcam-face-control/.venv/bin/python"
RUNNING_PID=$(/usr/bin/pgrep -f '[f]ace-meme-studio/app.py' | /usr/bin/head -n 1)

if [[ -n "$RUNNING_PID" ]]; then
    /bin/kill -INT "$RUNNING_PID"
    exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    /usr/bin/osascript -e 'display alert "Face Meme Studio could not find its Python environment." message "Open the project README for setup help."'
    exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/app.py"