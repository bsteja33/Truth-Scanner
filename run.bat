@echo off
echo Starting Fake News Detection System...

echo Checking if model exists...
if not exist "models\ensemble_pipeline.pkl" (
    echo Model not found. Training model... (this may take a while)
    python src\train.py
)

echo Starting FastAPI backend...
cd api
start /b python -m uvicorn api:app --host 127.0.0.1 --port 8000
cd ..

timeout /t 3 >nul

echo Starting Streamlit frontend...
cd frontend
start /b python -m streamlit run app.py
cd ..

echo =============================================
echo Access the backend API at: http://127.0.0.1:8000
echo Access the interactive API docs at: http://127.0.0.1:8000/docs
echo Access the frontend at: http://localhost:8501
echo =============================================

echo Press any key to exit and stop servers...
pause >nul
exit
