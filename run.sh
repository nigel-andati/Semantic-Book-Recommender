#!/bin/bash
# Semantic Book Recommender - One-command run script (macOS/Linux)

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Edit .env and set USE_OFFLINE_EMBEDDINGS=true for no-API mode."
    echo ""
fi

echo "Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "Starting Book Recommender..."
python main.py
