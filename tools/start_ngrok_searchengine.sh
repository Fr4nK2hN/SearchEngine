#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXED_URL="https://unadulating-otelia-dichotomously.ngrok-free.dev"

cd "$ROOT_DIR"

docker compose up -d app
ngrok http 5000 --url "$FIXED_URL"
