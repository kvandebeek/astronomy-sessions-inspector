#!/bin/sh
# Serves the review UI over fits_light_data.json. If that JSON hasn't been
# generated yet, runs run_light_stats.sh first (with defaults) so there is
# always fresh data to read.
#
# Pass --rescan to force a re-scan via run_light_stats.sh before serving.
# Any other args are passed through to fits_review_server.py, so
# run_review_server.sh -p 9000 still works as before.
set -e
cd "$(dirname "$0")"

PY=python3
[ -x ../.venv/bin/python ] && PY=../.venv/bin/python

JSON="$(pwd)/fits_light_data.json"

if [ "$1" = "--rescan" ]; then
    shift
    ./run_light_stats.sh --rescan
elif [ ! -f "$JSON" ]; then
    ./run_light_stats.sh
fi

exec "$PY" fits_review_server.py --json "$JSON" --root . "$@"
