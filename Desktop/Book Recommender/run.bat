@echo off
REM Semantic Book Recommender - One-command run script (Windows)
REM Ensures venv is activated and app runs with reproducible settings

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

if not exist ".env" (
    echo Creating .env from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Edit .env and set USE_OFFLINE_EMBEDDINGS=true for no-API mode.
    echo.
)

echo Installing dependencies...
pip install -r requirements.txt -q

echo.
echo Starting Book Recommender...
python main.py
