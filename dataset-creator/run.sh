#!/usr/bin/env bash
# Launch dataset-creator. Port defaults to 8010 so it can run alongside the
# schedext annotator on 8000 -- both read and write the same annotation store.
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python dataset-creator/server.py "${1:-8010}"
