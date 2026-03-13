#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR=""
SEED=42
NUM_BASELINE=25
NUM_SAMPLE=20
MAX_CLICKS_PER_QUERY=2
MODE_CYCLE="baseline,ltr,cross_encoder,hybrid"

usage() {
  cat <<'EOF'
Usage: tools/run_feedback_oracle_pipeline.sh [options]

Options:
  --out-dir <path>            Output directory (default: models/feedback_oracle/<timestamp>)
  --seed <n>                  Random seed (default: 42)
  --num-baseline <n>          Replay count for baseline_queries.json (default: 25)
  --num-sample <n>            Replay count for sample_queries.json (default: 20)
  --max-clicks <n>            Max clicks per query in replay (default: 2)
  --mode-cycle <csv>          Ranking mode cycle for replay
  -h, --help                  Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --num-baseline)
      NUM_BASELINE="${2:-}"
      shift 2
      ;;
    --num-sample)
      NUM_SAMPLE="${2:-}"
      shift 2
      ;;
    --max-clicks)
      MAX_CLICKS_PER_QUERY="${2:-}"
      shift 2
      ;;
    --mode-cycle)
      MODE_CYCLE="${2:-}"
      shift 2
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
  OUT_DIR="models/feedback_oracle/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"

echo "[1/8] Ensuring services are running..."
docker compose up -d >/dev/null
docker compose restart app >/dev/null

echo "[2/8] Waiting for app readiness..."
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

run_one_replay() {
  local tag="$1"
  local query_file="$2"
  local num_queries="$3"
  local out_prefix="$OUT_DIR/$tag"

  docker compose exec -T app python tools/replay_feedback_simulation.py \
    --base-url http://127.0.0.1:5000 \
    --queries-file "$query_file" \
    --num-queries "$num_queries" \
    --click-policy all \
    --max-clicks-per-query "$MAX_CLICKS_PER_QUERY" \
    --mode-cycle "$MODE_CYCLE" \
    --trace-path "$out_prefix.trace.json" \
    > "$out_prefix.summary.json"

  python3 - "$out_prefix.summary.json" "$out_prefix.session_id.txt" <<'PY'
import json, sys
summary_path, out_path = sys.argv[1], sys.argv[2]
with open(summary_path, "r", encoding="utf-8") as f:
    x = json.load(f)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(x["session_id"])
print(x["session_id"])
PY

  local session_id
  session_id="$(cat "$out_prefix.session_id.txt")"

  docker compose exec -T app python - "$out_prefix.export.json" "$session_id" <<'PY'
import json
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

out_path = sys.argv[1]
session_id = sys.argv[2]
base = "http://127.0.0.1:5000"
params = urlencode({"session_id": session_id})
with urlopen(base + "/export_data?" + params, timeout=30) as r:
    payload = json.loads(r.read().decode("utf-8"))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(len(payload.get("raw_events", [])))
PY
}

echo "[3/8] Replay #1 (baseline_queries)..."
run_one_replay "replay1" "data/baseline_queries.json" "$NUM_BASELINE"

echo "[4/8] Replay #2 (sample_queries)..."
run_one_replay "replay2" "data/sample_queries.json" "$NUM_SAMPLE"

echo "[5/8] Combining exports and traces..."
python3 - "$OUT_DIR" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

out_dir = sys.argv[1]
exp_paths = [
    os.path.join(out_dir, "replay1.export.json"),
    os.path.join(out_dir, "replay2.export.json"),
]
trace_paths = [
    os.path.join(out_dir, "replay1.trace.json"),
    os.path.join(out_dir, "replay2.trace.json"),
]

raw_events = []
for p in exp_paths:
    with open(p, "r", encoding="utf-8") as f:
        x = json.load(f)
    raw_events.extend(x.get("raw_events", []))

combined_export = {
    "export_timestamp": datetime.now(timezone.utc).isoformat(),
    "source_exports": exp_paths,
    "raw_events": raw_events,
}
with open(os.path.join(out_dir, "combined.export.json"), "w", encoding="utf-8") as f:
    json.dump(combined_export, f, indent=2, ensure_ascii=False)

search_traces = []
events = []
for p in trace_paths:
    with open(p, "r", encoding="utf-8") as f:
        x = json.load(f)
    search_traces.extend(x.get("search_traces", []))
    events.extend(x.get("events", []))

combined_trace = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source_traces": trace_paths,
    "search_traces": search_traces,
    "events": events,
}
with open(os.path.join(out_dir, "combined.trace.json"), "w", encoding="utf-8") as f:
    json.dump(combined_trace, f, indent=2, ensure_ascii=False)

print(len(raw_events), len(search_traces))
PY

FEEDBACK_DATA="$OUT_DIR/ltr_feedback_data.json"
echo "[6/8] Building feedback dataset..."
docker compose exec -T app python tools/build_feedback_ltr_data.py \
  --export-json "$OUT_DIR/combined.export.json" \
  --trace-json "$OUT_DIR/combined.trace.json" \
  --output "$FEEDBACK_DATA" \
  --use-confirmed-only \
  --candidate-source search \
  --fallback-mode baseline \
  --base-url http://127.0.0.1:5000 \
  > "$OUT_DIR/build_feedback.log" 2>&1

echo "[7/8] Running oracle (pretrained)..."
docker compose exec -T app python tools/run_oracle_experiment.py \
  --data "$FEEDBACK_DATA" \
  --output-dir "$OUT_DIR/oracle_pretrained" \
  --seed "$SEED" \
  --sample random \
  --ltr-model-path models/ltr_model.pkl \
  --ltr-source pretrained \
  --min-relevant-docs 1 \
  > "$OUT_DIR/oracle_pretrained.log" 2>&1

echo "[8/8] Running oracle (pretrained + no-prefix filter)..."
docker compose exec -T app python tools/run_oracle_experiment.py \
  --data "$FEEDBACK_DATA" \
  --output-dir "$OUT_DIR/oracle_pretrained_noprefix" \
  --seed "$SEED" \
  --sample random \
  --ltr-model-path models/ltr_model.pkl \
  --ltr-source pretrained \
  --min-relevant-docs 1 \
  --drop-prefix-queries \
  --min-query-terms 3 \
  > "$OUT_DIR/oracle_pretrained_noprefix.log" 2>&1

cat > "$OUT_DIR/README.md" <<EOF
# Feedback Oracle Pipeline Artifacts

- replay1.summary.json / replay2.summary.json: replay run summaries
- replay1.export.json / replay2.export.json: per-session export_data snapshots
- replay1.trace.json / replay2.trace.json: detailed replay traces
- combined.export.json / combined.trace.json: merged events + traces
- ltr_feedback_data.json: feedback-derived LTR dataset
- ltr_feedback_data.json.report.json: dataset build report
- oracle_pretrained/: oracle results on combined feedback data
- oracle_pretrained_noprefix/: oracle results with prefix-like queries filtered out

Command:
\`\`\`bash
tools/run_feedback_oracle_pipeline.sh --out-dir "$OUT_DIR" --seed "$SEED" --num-baseline "$NUM_BASELINE" --num-sample "$NUM_SAMPLE" --max-clicks "$MAX_CLICKS_PER_QUERY"
\`\`\`
EOF

echo "Pipeline completed."
echo "Artifacts: $OUT_DIR"
echo "Key files:"
echo "- $OUT_DIR/ltr_feedback_data.json.report.json"
echo "- $OUT_DIR/oracle_pretrained/oracle_results.json"
echo "- $OUT_DIR/oracle_pretrained_noprefix/oracle_results.json"
