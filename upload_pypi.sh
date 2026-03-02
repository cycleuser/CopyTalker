#!/usr/bin/env bash
#
# upload_pypi.sh - Build and upload CopyTalker to PyPI
#
# Usage:
#   ./upload_pypi.sh          # Upload to PyPI (production)
#   ./upload_pypi.sh test     # Upload to TestPyPI first
#   ./upload_pypi.sh build    # Build only, no upload
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Step 1: Check prerequisites ──────────────────────────────────────────────
info "Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || error "python3 not found"
python3 -m pip --version >/dev/null 2>&1 || error "pip not found"

# Install build tools if missing
for pkg in build twine; do
    if ! python3 -m $pkg --help >/dev/null 2>&1; then
        info "Installing $pkg..."
        python3 -m pip install --upgrade $pkg
    fi
done

# ── Step 2: Read version from __init__.py (single source of truth) ────────────
VERSION=$(python3 -c "
import re
with open('src/copytalker/__init__.py') as f:
    m = re.search(r'__version__\s*=\s*\"(.+?)\"', f.read())
    print(m.group(1) if m else 'UNKNOWN')
")

if [ "$VERSION" = "UNKNOWN" ]; then
    error "Could not read version from src/copytalker/__init__.py"
fi

info "Building version: $VERSION"

# ── Step 3: Clean previous builds ────────────────────────────────────────────
info "Cleaning previous builds..."
rm -rf build/ dist/ src/*.egg-info src/copytalker/*.egg-info *.egg-info

# ── Step 4: Build ────────────────────────────────────────────────────────────
info "Building sdist and wheel..."
python3 -m build

# Verify artifacts
if [ ! -d dist ] || [ -z "$(ls dist/)" ]; then
    error "Build failed - no artifacts in dist/"
fi

info "Build artifacts:"
ls -lh dist/

# ── Step 5: Check package ────────────────────────────────────────────────────
info "Running twine check..."
python3 -m twine check dist/*

# ── Step 6: Upload ───────────────────────────────────────────────────────────
MODE="${1:-}"

if [ "$MODE" = "build" ]; then
    info "Build-only mode. Skipping upload."
    info "Artifacts are in dist/"
    exit 0
fi

if [ "$MODE" = "test" ]; then
    info "Uploading to TestPyPI..."
    warn "Make sure you have a TestPyPI account: https://test.pypi.org/account/register/"
    warn "Configure token: python3 -m twine upload --repository testpypi dist/*"
    echo ""
    python3 -m twine upload --repository testpypi dist/*
    echo ""
    info "Uploaded to TestPyPI!"
    info "Install with: pip install -i https://test.pypi.org/simple/ copytalker==$VERSION"
    echo ""
    read -p "Upload to production PyPI as well? [y/N] " confirm
    if [[ "$confirm" != [yY] ]]; then
        info "Skipped production upload."
        exit 0
    fi
fi

info "Uploading to PyPI..."
warn "Make sure you have a PyPI account: https://pypi.org/account/register/"
warn "Use API token: https://pypi.org/manage/account/token/"
echo ""
python3 -m twine upload dist/*

echo ""
info "Successfully uploaded copytalker $VERSION to PyPI!"
info "Install with: pip install copytalker==$VERSION"
