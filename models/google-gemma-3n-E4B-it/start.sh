#!/bin/bash
set -e

# Default values
PORT=8000
MODEL_PATH="google/gemma-3n-E4B-it"
HF_TOKEN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --hf-token=*)
      HF_TOKEN="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: ./start.sh --port=<port> --model-path=<model_path>"
      exit 1
      ;;
  esac
done

echo "Using MODEL_PATH: $MODEL_PATH"
echo "Using HF_TOKEN: $HF_TOKEN"
echo "Using PORT: $PORT"

export HF_TOKEN=$HF_TOKEN
export MODEL_PATH=$MODEL_PATH

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port $PORT
