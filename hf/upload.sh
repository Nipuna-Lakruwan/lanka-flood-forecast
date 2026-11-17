#!/usr/bin/env bash
# Publish the models to https://huggingface.co/nipuna02/lanka-flood-forecast (run after `hf auth login`).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=nipuna02/lanka-flood-forecast
uvx --from huggingface_hub hf repo create "$REPO" --repo-type model --exist-ok
uvx --from huggingface_hub hf upload "$REPO" hf/README.md README.md
uvx --from huggingface_hub hf upload "$REPO" models . --include "*.txt" --include "meta.json"
uvx --from huggingface_hub hf upload "$REPO" results/metrics.json results/metrics.json
echo "https://huggingface.co/$REPO"
