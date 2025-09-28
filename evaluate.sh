#!/bin/bash

# Parse command line arguments
DEBUG=false
ARGS=()

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      ARGS+=("$1")
      # Unknown option
      shift
      ;;
  esac
done

# Record start time
start=$(date +%s.%N)

if [ "$DEBUG" = true ]; then
  echo "Starting in debug mode - waiting for debugger to attach on port 5678..."
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client evaluate.py "$@"
else
  python evaluate.py "${ARGS[@]}"
fi

# Record end time
end=$(date +%s.%N)

# Compute elapsed time and round to 2 decimals
elapsed=$(echo "$end - $start" | bc)
printf "Elapsed time: %.2f seconds\n" "$elapsed"
