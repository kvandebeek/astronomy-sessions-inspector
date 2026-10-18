#!/bin/sh
# Scans this telescope's folder (the one this script sits in), calibrating against
# the master bias/darks in darks/ (masterBias*.xisf + masterDark*.xisf together)
# and the flats at <session>/<any folder>/<filter>/FLAT/ (organize_flats.py puts
# them there), then writes fits_light_data.json for run_review_server.sh to read.
#
# Any extra args (e.g. --rescan, --solve, --darks DIR) are passed through and
# override the defaults below (a later --darks wins over the default one).
set -e
cd "$(dirname "$0")"

PY=python3
[ -x ../.venv/bin/python ] && PY=../.venv/bin/python

JSON="fits_light_data.json"
REPORT="fits_light_report.html"
DARKS="darks"

exec "$PY" fits_light_stats.py . \
    --json-cache "$JSON" \
    --output "$REPORT" \
    --darks "$DARKS" \
    "$@"
