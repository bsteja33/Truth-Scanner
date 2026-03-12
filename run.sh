#!/bin/bash
echo "Starting Fake News Detection System..."

# Train model if it doesn't exist
if [ ! -f "models/ensemble_pipeline.pkl" ]; then
    echo "Model not found. Training model... (this may take a while)"
    python src/train.py
fi

# Run API in the background
echo "Starting FastAPI backend..."
cd api
python -m uvicorn api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
cd ..

# Wait a second for API to start
sleep 2

# Run Streamlit frontend
echo "Starting Streamlit frontend..."
cd frontend
python -m streamlit run app.py &
FRONTEND_PID=$!
cd ..

echo "============================================="
echo "Access the backend API at: http://127.0.0.1:8000"
echo "Access the interactive API docs at: http://127.0.0.1:8000/docs"
echo "Access the frontend at: http://127.0.0.1:8501"
echo "============================================="

# Wait for background processes to exit
wait $API_PID
wait $FRONTEND_PID
