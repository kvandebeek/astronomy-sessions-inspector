#!/bin/sh
# Serves the review UI over fits_light_data.json. If that JSON hasn't been
# generated yet, runs run_light_stats.sh first (with defaults) so there is
# always fresh data to read.
#
# Pass --rescan (anywhere among the arguments) to scan again via run_light_stats.sh
# before serving - only new or changed frames are measured - or --full-rescan to
# re-measure everything. Any other args are passed through to fits_review_server.py,
# so run_review_server.sh -p 9000 --open still works.
set -e
cd "$(dirname "$0")"

PY=python3
[ -x ../.venv/bin/python ] && PY=../.venv/bin/python

JSON="$(pwd)/fits_light_data.json"

# Pull the scan flags out of the argument list, keeping the rest in order.
RESCAN=""
for arg do
    shift
    case "$arg" in
        --rescan|--full-rescan) RESCAN="$arg" ;;
        *) set -- "$@" "$arg" ;;
    esac
done

if [ -n "$RESCAN" ]; then
    ./run_light_stats.sh "$RESCAN"
elif [ ! -f "$JSON" ]; then
    ./run_light_stats.sh
fi

exec "$PY" fits_review_server.py --json "$JSON" --root . "$@"
