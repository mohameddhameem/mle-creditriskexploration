#!/bin/bash
# start_airflow.sh - Optimized sequential startup script for low-memory environments

set -e

echo "=== [1/3] Starting Airflow Webserver in background ==="
airflow webserver --port 8080 &
WEBSERVER_PID=$!

echo "=== [2/3] Waiting for Webserver to initialize and listen on port 8080 ==="
# Wait up to 60 seconds or until port 8080 is listening
for i in {1..60}; do
  if cat /proc/net/tcp | grep -i "00000000:1F90" > /dev/null || cat /proc/net/tcp6 | grep -i "00000000:1F90" > /dev/null; then
    echo "Webserver is listening on port 8080!"
    break
  fi
  if ! kill -0 $WEBSERVER_PID 2>/dev/null; then
    echo "ERROR: Webserver process died during startup."
    exit 1
  fi
  sleep 2
done

echo "=== [3/3] Starting Airflow Scheduler in foreground ==="
# exec replaces the shell process, preserving signal handling for container termination
exec airflow scheduler
