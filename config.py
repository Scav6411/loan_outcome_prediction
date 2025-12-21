import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    MODEL_PATH = os.getenv("MODEL_PATH", "weights/best_weights.json")

settings = Settings()

def validate_settings():
    missing = []
    for key, value in settings.__dict__.items():
        if value is None:
            missing.append(key)

    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}"
        )
