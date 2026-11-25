
from __future__  import annotations
from functools import lru_cache

from pydantic_settings import BaseSettings , SettingsConfigDict

class Settings(BaseSettings):
    """Every knob is an env var with the WATHQA_ prefix"""
    model_config = SettingsConfigDict(
            env_prefix="WATHIQA_",env_file=".env", extra="ignore"
    )

    llm_provider : str = "groq"
    model : str = "openai/gpt-oss-120b" 

    auto_approve_limit : float = 2000.0
    confidence_threshold: float = .7
    max_investigation_steps : int = 3
    vat_rate :float = .14

@lru_cache
def get_settings() -> Settings:
    return Settings()