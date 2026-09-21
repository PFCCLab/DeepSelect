#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${PY_BIN:-/root/paddlejob/share-storage/gpfs/system-public/heqianyue/p2p_env/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
if [[ " $* " == *" --list "* ]]; then
  exec "$PY_BIN" "$HERE/run_suite.py" "$@"
fi
"$PY_BIN" "$HERE/run_suite.py" "$@"
tier="standard"
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--tier" ]]; then j=$((i+1)); tier="${!j}"; fi
done
case "$tier" in
  smoke) args=(--small-runs 20 --large-runs 5 --fresh-processes 2 --stream-rounds 3) ;;
  standard) args=(--small-runs 200 --large-runs 20 --fresh-processes 3 --stream-rounds 10) ;;
  stress) args=(--small-runs 1000 --large-runs 100 --fresh-processes 5 --stream-rounds 25) ;;
esac
exec "$PY_BIN" "$HERE/determinism.py" "${args[@]}"
