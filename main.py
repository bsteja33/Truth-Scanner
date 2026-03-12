import subprocess
import os
import time
import sys

def main():
    print("Starting Fake News Detection System...")

    # Check if model exists
    if not os.path.exists("models/ensemble_pipeline.pkl"):
        print("Model not found. Training model... (this may take a while)")
        subprocess.run(["python", "src/train.py"])

    # Start FastAPI
    print("Starting FastAPI backend...")
    api_process = subprocess.Popen("python -m uvicorn api:app --host 127.0.0.1 --port 8000", cwd="api", shell=True)
    
    time.sleep(3)

    # Start Streamlit
    print("Starting Streamlit frontend...")
    frontend_process = subprocess.Popen("python -m streamlit run app.py", cwd="frontend", shell=True)

    print("\n=============================================")
    print("Access the backend API at: http://127.0.0.1:8000")
    print("Access the interactive API docs at: http://127.0.0.1:8000/docs")
    print("Access the frontend at: http://localhost:8501")
    print("=============================================\n")

    try:
        api_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Shutting down processes...")
        api_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()
