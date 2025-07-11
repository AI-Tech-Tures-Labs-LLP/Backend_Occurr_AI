import os
import subprocess
import sys
import venv

VENV_DIR = "venv"
IS_WINDOWS = os.name == "nt"
PYTHON_BIN = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin", "python")
UVICORN_BIN = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin", "uvicorn")

APP_IMPORT_PATH = "app.main:app"  # Adjust this if your FastAPI app is elsewhere

def create_venv():
    if not os.path.exists(VENV_DIR):
        print("📦 Creating virtual environment...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print("✅ Virtual environment already exists.")

def install_requirements():
    print("📄 Installing dependencies...")
    subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "-r", "requirements.txt"])

def run_server():
    print("🚀 Starting FastAPI server with uvicorn...")
    subprocess.check_call([UVICORN_BIN, APP_IMPORT_PATH, "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    create_venv()
    install_requirements()
    run_server()
