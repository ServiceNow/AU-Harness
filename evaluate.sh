#!/bin/bash

# Parse command line arguments
DEBUG=false

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

if [ "$DEBUG" = true ]; then
  echo "Starting in debug mode - waiting for debugger to attach on port 5678..."
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client evaluate.py "$@"
else
  python evaluate.py "$@"
fi