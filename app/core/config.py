import os
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENV  = os.path.join(_ROOT, ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "AI Data Analysis Agent"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:latest"
    LLM_ENABLED: bool = True      # False bo'lsa agent faqat qoidaga asoslangan rejalashtiruvchini ishlatadi

    STORAGE_DIR:  str = "./storage"
    UPLOAD_DIR:   str = "./storage/uploads"
    CHARTS_DIR:   str = "./storage/charts"
    REPORTS_DIR:  str = "./storage/reports"
    MODELS_DIR:   str = "./storage/models"

    MAX_UPLOAD_MB: int = 50
    CORS_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    SECRET_KEY: str = "change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    @property
    def DB_PATH(self) -> str:
        return os.path.join(self.STORAGE_DIR, "db.sqlite")


settings = Settings()
