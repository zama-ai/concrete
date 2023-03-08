#!/bin/bash

# Run benchmarks while logging the intermediate results
# Publish findings in the progress tracker

set -e

# shellcheck disable=SC1091
if [ -f .env ]
then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
fi

DEV_VENV_PATH="/home/dev_user/dev_venv"

# shellcheck disable=SC1090,SC1091
if ! source "${DEV_VENV_PATH}/bin/activate"; then
    python3 -m venv "${DEV_VENV_PATH}"
    # shellcheck disable=SC1090,SC1091
    source "${DEV_VENV_PATH}/bin/activate"
fi

cd /src/ && make setup_env

mkdir -p /tmp/keycache
mkdir -p logs

initial_concrete_log=logs/$(date -u --iso-8601=seconds).concrete.log
make -s benchmark 2>&1 | tee -a "$initial_concrete_log"

final_concrete_log=logs/$(date -u --iso-8601=seconds).concrete.log
cat -s "$initial_concrete_log" | sed '1d; $d' > "$final_concrete_log"

# sed above removes the first and the last lines of the log
# which are empty to provide a nice console output
# but empty lines are useless for logs so we get rid of them

rm "$initial_concrete_log"
cp "$final_concrete_log" logs/latest.concrete.log

curl \
     -H 'Authorization: Bearer '"$CONCRETE_PROGRESS_TRACKER_TOKEN"'' \
     -H 'Content-Type: application/json' \
     -d @progress.json \
     -X POST "$CONCRETE_PROGRESS_TRACKER_URL"/measurement
