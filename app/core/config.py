from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env", env_prefix="LUM_")

    app_title: str = "Land Monitor API"
    app_version: str = "0.1.0"

    onnx_path:Path = Path("models/model.onnx")
    stats_path:Path = Path("models/eurosat_stats.json")
    class_names_path:Path = Path("models/class_names.json")

    max_upload_size:int = 5 * 1024 * 1024  # 10 MB

@lru_cache
def get_settings() -> Settings:
    return Settings()