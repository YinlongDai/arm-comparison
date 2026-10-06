#!/bin/zsh
set -e
ROOT="${0:A:h}"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="$ROOT/.cache/uv"
    export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/python"
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi
exec .venv/bin/python viewer.py --open-browser "$@"
