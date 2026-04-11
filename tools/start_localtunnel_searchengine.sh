#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose up -d app
lt --port 5000 --local-host 127.0.0.1 --print-requests
