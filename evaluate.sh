#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Shell wrapper for audiobench/evaluate.py
# -----------------------------------------------------------------------------
# This script provides a convenient way to invoke the AudioBench evaluator while
# documenting the two most common deployment scenarios for the model inference
# backend (vLLM):
#   1. "LOCAL SERVER"  – You start a vLLM/OpenAI-compatible HTTP server on the
#      same host (e.g. `python -m vllm.entrypoints.openai.api_server ...`).
#   2. "REMOTE ENDPOINT" – You point AudioBench to any reachable HTTP endpoint
#      (SageMaker, TGI, HF Inference Endpoints, an internal gateway, etc.).
#
# The evaluator itself only knows about a single required CLI flag, `--cfg`,
# which is the path to a YAML/JSON *benchmark configuration* file.  All other
# behaviour (model selection, dataset, metric, batch size, auth tokens…) is
# driven by **that config file**, *not* by additional CLI flags to evaluate.py.
#
# This wrapper therefore accepts the config path as its sole positional
# argument and forwards it verbatim to evaluate.py.  Any further args are
# forwarded unchanged to keep future-proof flexibility (e.g. should new flags
# be added upstream).
# -----------------------------------------------------------------------------
# USAGE
#   ./evaluate.sh path/to/benchmark.yaml [additional-evaluate.py-flags]
#
# EXAMPLES
# --------
# 1) vLLM SERVER ON LOCALHOST (no auth token)
#    Assume you launched vLLM like so
#        python -m vllm.entrypoints.openai.api_server \
#                --model meta-llama/Llama-3-8B-Instruct \
#                --host 0.0.0.0 --port 8000
#
#    Then a minimal benchmark YAML could look like:
#        ---
#        dataset: "audiobench_wavcaps_test"
#        metric: "word_error_rate"
#        batch_size: 8               # OPTIONAL (default=4 in evaluate.py)
#        models:
#          - name: "vllm"
#            info:
#              inference_type: "vllm"   # Required so Model class picks vLLM handler
#              url: "http://localhost:8000"  # Where your server listens
#              auth_token: ""               # OPTIONAL for local, omit or leave blank
#
#    Run the evaluation with:
#        ./evaluate.sh configs/vllm_local.yml
#
# 2) REMOTE HTTP ENDPOINT WITH AUTH TOKEN
#    e.g. An OpenAI-compatible gateway provided by your infra team.
#
#        ---
#        dataset: "audiobench_wavcaps_test"
#        metric: "word_error_rate"
#        models:
#          - name: "vllm_remote"
#            info:
#              inference_type: "vllm"           # Still vLLM/OpenAI-style REST
#              url: "https://my-gateway.acme.com/v1"
#              auth_token: "Bearer $MY_API_KEY" # OPTIONAL if gateway needs it
#        # batch_size not set ⇒ defaults to 4
#
#    Invoke with:
#        export MY_API_KEY="sk-…"
#        ./evaluate.sh configs/vllm_remote.yml
#
# REQUIRED VS OPTIONAL FIELDS (in YAML)
# -------------------------------------
# dataset           REQUIRED – must exist in audiobench_datasets.json
# metric            REQUIRED – must be recognised by Metric(name)
# models / model    At least one model definition is REQUIRED.
#   name            REQUIRED – corresponds to sub-module in audiobench.models
#   info            REQUIRED – free-form; keys depend on the model backend.
#     inference_type    OPTIONAL but strongly recommended ("vllm", "tgi", …)
#     url               OPTIONAL if model runs purely local; REQUIRED for HTTP.
#     auth_token        OPTIONAL – only needed when the endpoint enforces auth.
# batch_size        OPTIONAL – defaults to 4 if omitted.
# -----------------------------------------------------------------------------
# NOTE: evaluate.py lives in the package, so we call it with `python -m` to keep
# imports stable regardless of the current working directory.
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CONFIG_YAML_OR_JSON> [extra evaluate.py args]" >&2
  exit 1
fi

CONFIG_PATH="$1"
shift  # Remove first positional so "$@" contains any extra flags

NUM_SAMPLES_ARG=""
if [[ ! -z "${NUM_SAMPLES:-}" ]]; then
  NUM_SAMPLES_ARG="--num-samples ${NUM_SAMPLES}"
fi

python evaluate.py --cfg "${CONFIG_PATH}" $NUM_SAMPLES_ARG "$@"