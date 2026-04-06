#!/usr/bin/env bash
set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-$(pwd)}"

if [ -z "$PING_URL" ]; then
  echo "Usage: ./scripts/validate-submission.sh <ping_url> [repo_dir]" >&2
  exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"

echo "[1/4] Checking OpenEnv config"
python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('openenv.yaml').read_text(encoding='utf-8'))
assert cfg['name'] == 'customer-support-env'
assert cfg['version'] == '1.0.0'
assert cfg['entry_point'] == 'env:SupportEnv'
print('openenv.yaml OK')
PY

echo "[2/4] Running baseline"
python baseline.py --task hard >/dev/null

echo "[3/4] Running inference"
python inference.py >/dev/null

echo "[4/4] Pinging deployment"
code=$(curl -s -o /dev/null -w '%{http_code}' "$PING_URL")
if [ "$code" != "200" ]; then
  echo "Ping failed with status $code" >&2
  exit 1
fi

echo "Validation passed"
