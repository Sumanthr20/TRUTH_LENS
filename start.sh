#!/bin/bash
# TruthLens — Quick Start Script


echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║    TruthLens — AI Fake News Detection System     ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -r backend/requirements.txt -q

echo ""
echo "🤖 Training ML model..."
cd backend && python3 train_model.py
cd ..

echo ""
echo "🚀 Starting Flask server..."
echo "   Backend:  http://localhost:5000"
echo "   Frontend: open frontend/index.html in your browser"
echo ""
echo "   Or serve frontend via: python3 -m http.server 8080 (in frontend/)"
echo ""
python3 backend/app.py
