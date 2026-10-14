#!/usr/bin/env bash
# One-time setup: create/update the Conda environment and pull the TinyLlama model.
#   bash setup.sh
#   conda activate a10-summarizer && uvicorn app.main:app --port 8000
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not installed. Install Miniconda or Miniforge first: https://conda-forge.org/download/" >&2
  exit 1
fi

if conda env list | grep -qE '^a10-summarizer\s'; then
  echo "Updating Conda environment 'a10-summarizer'..."
  conda env update -n a10-summarizer -f environment.yml --prune
else
  echo "Creating Conda environment 'a10-summarizer'..."
  conda env create -f environment.yml
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download and run this script again." >&2
  exit 1
fi

if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  echo "Starting Ollama in the background..."
  (ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 20); do curl -sf http://localhost:11434/api/version >/dev/null && break; sleep 1; done
fi

echo "Pulling the TinyLlama model..."
ollama pull tinyllama

echo
echo "Setup complete. Start the app with:"
echo "  conda activate a10-summarizer"
echo "  uvicorn app.main:app --port 8000"
