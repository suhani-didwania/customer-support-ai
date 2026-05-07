#!/usr/bin/env bash
#
# One-shot project setup: venv, dependencies, .env, seed database, ingest PDFs.
# Idempotent - safe to re-run on an existing checkout.
#
# Usage:
#     ./setup.sh

set -euo pipefail

echo
echo "Customer Support AI - setup"
echo "---------------------------"
echo

# --- 1. Virtual environment ---
if [ ! -d "venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] Virtual environment already exists, reusing"
fi
# shellcheck source=/dev/null
source venv/bin/activate

# --- 2. Dependencies ---
echo "[2/5] Installing dependencies (this can take a few minutes)..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt > /dev/null
echo "      done"

# --- 3. .env ---
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo
    echo "[3/5] Created .env from template."
    echo "      Edit .env and add your OPENAI_API_KEY before continuing."
    echo
    read -rp "      Press Enter once you've added your API key (or Ctrl-C to abort)..."
else
    echo "[3/5] .env already present, skipping"
fi

# --- 4. Database ---
echo "[4/5] Seeding merchant database..."
python scripts/init_database.py

# --- 5. Vector index ---
echo "[5/5] Indexing policy PDFs into the vector store..."
if compgen -G "data/uploaded_pdfs/*.pdf" > /dev/null; then
    python scripts/ingest_documents.py
else
    echo
    echo "      No PDFs found in data/uploaded_pdfs/."
    echo "      Drop your policy PDFs in that folder and re-run:"
    echo "          python scripts/ingest_documents.py"
fi

echo
echo "Setup complete."
echo
echo "Launch the chat UI:"
echo "    streamlit run ui/app.py"
echo
echo "Or use the terminal client:"
echo "    python scripts/cli.py"
echo
echo "Or smoke-test the MCP server:"
echo "    python scripts/mcp_client_demo.py"
echo