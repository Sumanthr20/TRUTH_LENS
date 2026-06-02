@echo off
echo.
echo ============================================================
echo    TruthLens - AI Fake News Detection System
echo    Sumanth
echo ============================================================
echo.

echo [1/3] Installing dependencies...
pip install -r backend\requirements.txt -q

echo.
echo [2/3] Training ML model...
cd backend
python train_model.py
cd ..

echo.
echo [3/3] Starting Flask server...
echo    Backend  : http://localhost:5000
echo    Frontend : Open frontend\index.html in your browser
echo.
python backend\app.py
pause
