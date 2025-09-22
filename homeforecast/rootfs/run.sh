#!/usr/bin/env bashio
set -e

CONFIG_PATH=/data/options.json

bashio::log.info "Starting HomeForecast addon..."

# Start nginx in background
nginx

# Start the Python application
cd /app
exec python3 -u main.py