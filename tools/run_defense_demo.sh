#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR=""
NUM_QUERIES=8
EVAL_MAX_QUERIES=60
SEED=42
NO_BUILD=0

usage() {
  cat <<'EOF'
Usage: tools/run_defense_demo.sh [options]

Options:
  --out-dir <path>          Output directory (default: models/defense_demo/<timestamp>)
  --num-queries <n>         Number of replayed online queries (default: 8)
  --eval-max-queries <n>    Max queries for reproducible offline eval (default: 60)
  --seed <n>                Random seed (default: 42)
  --no-build                Skip image rebuild, only ensure services are up
  -h, --help                Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --num-queries)
      NUM_QUERIES="${2:-}"
      shift 2
      ;;
    --eval-max-queries)
      EVAL_MAX_QUERIES="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --no-build)
      NO_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="models/defense_demo/$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_DIR"

if [[ "$NO_BUILD" -eq 1 ]]; then
  echo "[1/5] Starting services (no build)..."
  docker compose up -d
else
  echo "[1/5] Building and starting services..."
  docker compose up -d --build
fi

echo "Reloading app container to ensure latest code is active..."
docker compose restart app >/dev/null

echo "[2/5] Waiting for app readiness..."
for i in $(seq 1 90); do
  if docker compose exec -T app python - <<'PY' >/dev/null 2>&1
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:5000/model_info", timeout=5)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    echo "App is ready."
    break
  fi
  if [[ "$i" -eq 90 ]]; then
    echo "App did not become ready in time." >&2
    exit 1
  fi
  sleep 2
done

echo "[3/5] Replaying deterministic feedback session..."
REPLAY_JSON="$(docker compose exec -T app python tools/replay_feedback_simulation.py \
  --base-url http://127.0.0.1:5000 \
  --queries-file data/sample_queries.json \
  --num-queries "$NUM_QUERIES")"
printf "%s\n" "$REPLAY_JSON" > "$OUT_DIR/replay_session.json"

SESSION_ID="$(printf "%s" "$REPLAY_JSON" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["session_id"])')"
echo "Replay session id: $SESSION_ID"

echo "[4/5] Exporting online metrics and dashboard snapshot..."
docker compose exec -T app python - "$OUT_DIR" "$SESSION_ID" <<'PY'
import json
import os
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

out_dir = sys.argv[1]
session_id = sys.argv[2]
base = "http://127.0.0.1:5000"
os.makedirs(out_dir, exist_ok=True)

params = urlencode({"session_id": session_id})
with urlopen(base + "/export_data?" + params, timeout=20) as r:
    export_data = json.loads(r.read().decode("utf-8"))
with open(os.path.join(out_dir, "export_data.json"), "w", encoding="utf-8") as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)

with urlopen(base + "/research_dashboard", timeout=20) as r:
    dashboard_html = r.read().decode("utf-8")
with open(os.path.join(out_dir, "research_dashboard.html"), "w", encoding="utf-8") as f:
    f.write(dashboard_html)
PY

echo "[5/5] Running reproducible offline comparison..."
docker compose exec -T app python tools/run_repro_eval.py \
  --output-dir "$OUT_DIR/repro_eval" \
  --seed "$SEED" \
  --max-queries "$EVAL_MAX_QUERIES" \
  --ltr-source train \
  --lgbm-jobs 1 \
  --deterministic \
  > "$OUT_DIR/repro_eval.log" 2>&1

cat > "$OUT_DIR/README.md" <<EOF
# Defense Demo Artifacts

- replay_session.json: replay summary for simulated user session
- export_data.json: online feedback metrics snapshot (filtered to the replay session)
- research_dashboard.html: dashboard HTML snapshot
- repro_eval/: offline reproducible evaluation artifacts
  - comparison_results.json
  - comparison_table.md
  - split_indices.json
  - run_manifest.json
  - comparison_plot.png

Run command:
\`\`\`bash
tools/run_defense_demo.sh --out-dir "$OUT_DIR" --num-queries "$NUM_QUERIES" --eval-max-queries "$EVAL_MAX_QUERIES" --seed "$SEED"
\`\`\`
EOF

echo "Demo run completed."
echo "Artifacts saved to: $OUT_DIR"
echo "Key files:"
echo "- $OUT_DIR/export_data.json"
echo "- $OUT_DIR/repro_eval/comparison_table.md"
echo "- $OUT_DIR/repro_eval/run_manifest.json"
