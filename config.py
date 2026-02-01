import os
from pathlib import Path

def load_env_file(env_file='.env'):
    """Load environment variables from .env file"""
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"Warning: {env_file} not found. Using system environment variables.")
        return
    
    with open(env_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Load environment variables
load_env_file()

# Configuration from environment variables
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://postgres:dbda123@localhost:35432/postgres')

# Google OAuth credentials
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')

# Email settings
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

# Ollama settings
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')