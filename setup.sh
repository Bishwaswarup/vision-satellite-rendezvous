#!/usr/bin/env bash
#
# setup.sh — create and populate the project virtual environment.
#
# Run this ONCE, from the repository root, in a terminal on your own machine:
#
#     bash setup.sh
#
# It creates .venv/ inside this repository (which is where PyCharm, VS Code and
# most tooling look), installs the pinned requirements into it, and verifies the
# install by importing every third-party package the project needs.
#
# Options:
#     bash setup.sh --gpu       also install the optional vispy GPU renderer
#     bash setup.sh --dev       also install the optional test-reference deps
#     bash setup.sh --recreate  delete an existing .venv and start clean
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"
WITH_GPU=0
WITH_DEV=0
RECREATE=0

for arg in "$@"; do
  case "$arg" in
    --gpu)      WITH_GPU=1 ;;
    --dev)      WITH_DEV=1 ;;
    --recreate) RECREATE=1 ;;
    -h|--help)  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg  (try --help)"; exit 2 ;;
  esac
done

# ── Pick an interpreter ───────────────────────────────────────────────────────
# The project needs Python 3.10 or newer (it uses match-free but modern typing
# and numpy >= 1.26).
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    major="${ver%%.*}"; minor="${ver##*.}"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then PY="$cand"; break; fi
  fi
done

if [ -z "$PY" ]; then
  echo "ERROR: no Python 3.10+ interpreter found on PATH."
  echo "       Install one from python.org or via Homebrew:  brew install python@3.12"
  exit 1
fi

echo "interpreter : $PY  ($("$PY" --version 2>&1))"
echo "target venv : $VENV"

# ── Create ────────────────────────────────────────────────────────────────────
if [ -d "$VENV" ] && [ "$RECREATE" -eq 1 ]; then
  echo "removing existing venv ..."
  rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
  echo "creating venv ..."
  "$PY" -m venv "$VENV"
else
  echo "venv already exists — reusing it (pass --recreate to start clean)"
fi

VPY="$VENV/bin/python"
[ -x "$VPY" ] || VPY="$VENV/Scripts/python.exe"     # Git Bash on Windows
if [ ! -x "$VPY" ]; then
  echo "ERROR: could not find the venv interpreter under $VENV"
  exit 1
fi

# ── Install ───────────────────────────────────────────────────────────────────
echo
echo "upgrading pip ..."
"$VPY" -m pip install --upgrade pip --quiet

echo "installing requirements ..."
"$VPY" -m pip install -r requirements.txt --quiet

if [ "$WITH_GPU" -eq 1 ]; then
  echo "installing optional GPU renderer (vispy) ..."
  "$VPY" -m pip install "vispy>=0.14" "PyOpenGL>=3.1" --quiet || {
    echo "  WARNING: vispy install failed. It is optional — the 24 GPU tests"
    echo "           will simply skip. Everything else works without it."
  }
fi

if [ "$WITH_DEV" -eq 1 ]; then
  echo "installing optional test references (opencv) ..."
  "$VPY" -m pip install opencv-python-headless --quiet || {
    echo "  WARNING: opencv install failed. It is optional — the tests that"
    echo "           cross-check against it will skip."
  }
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo
echo "verifying ..."
"$VPY" - <<'PYEOF'
import importlib, sys

required = ['numpy', 'scipy', 'matplotlib', 'pytest']
optional = ['vispy', 'OpenGL', 'cv2']

bad = []
for name in required:
    try:
        m = importlib.import_module(name)
        print(f'  [ok ]  {name:<12s} {getattr(m, "__version__", "?")}')
    except Exception as exc:
        print(f'  [FAIL] {name:<12s} {exc}')
        bad.append(name)

for name in optional:
    try:
        m = importlib.import_module(name)
        print(f'  [opt ] {name:<12s} {getattr(m, "__version__", "present")}')
    except Exception:
        print(f'  [--  ] {name:<12s} not installed (optional)')

if bad:
    print('\nMISSING REQUIRED PACKAGES: ' + ', '.join(bad))
    sys.exit(1)
PYEOF

# ── Next steps ────────────────────────────────────────────────────────────────
cat <<EOF

------------------------------------------------------------------------------
Environment ready.

Activate it in a terminal:

    source .venv/bin/activate
    python main.py check

Point PyCharm at it:

    Settings / Preferences  ->  Project: vision-satellite-rendezvous
      ->  Python Interpreter  ->  gear icon  ->  Add Local Interpreter...
      ->  "Existing"  ->  Interpreter:

          $VPY

    (PyCharm normally auto-detects .venv in the project root. If it does not
     appear, use "Existing" and paste the path above.)

Note: there is an unrelated empty virtualenv one level up, at
      ../.venv  (i.e. VISION-BASED/.venv).
      It contains only pip and setuptools and nothing uses it. Safe to delete.
------------------------------------------------------------------------------
EOF
