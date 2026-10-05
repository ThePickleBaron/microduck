#!/usr/bin/env bash
# Re-vendor pollen-robotics/microduck_rl at a given commit or branch.
#   bash scripts/update_vendor.sh develop
set -euo pipefail
cd "$(dirname "$0")/.."
REF="${1:-develop}"
TMP="$(mktemp -d)"
git clone --quiet https://github.com/pollen-robotics/microduck_rl "$TMP/src"
git -C "$TMP/src" checkout --quiet "$REF"
SHA="$(git -C "$TMP/src" rev-parse HEAD)"
rm -rf third_party/microduck_rl
mkdir -p third_party
cp -r "$TMP/src" third_party/microduck_rl
rm -rf third_party/microduck_rl/.git "$TMP"
uv run --no-sync python scripts/sync_vendor_constraints.py
uv lock
echo "Vendored microduck_rl at $SHA. Update the commit in VENDORED.md, then: uv sync --extra dev && uv run pytest -q"
