#!/usr/bin/env bash
set -euo pipefail

runtime_dir="${HOME}/BigDB_Proj/reports/service"
mkdir -p "$runtime_dir"
printf 'STARTING\n' > "$runtime_dir/current-url.txt"

/usr/local/bin/cloudflared tunnel --config /dev/null --protocol http2 \
  --url http://127.0.0.1:8086 --no-autoupdate 2>&1 |
while IFS= read -r line; do
  printf '%s\n' "$line"
  if [[ "$line" =~ https://[a-z0-9-]+\.trycloudflare\.com ]]; then
    printf '%s\n' "${BASH_REMATCH[0]}" > "$runtime_dir/current-url.txt"
  fi
done
