"""Application settings (env-driven)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "Nyaya"
    app_version: str = "1.0.0"

    # Security
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 10080  # 7 days

    # Database
    database_url: str = "sqlite+aiosqlite:///./nyaya.db"

    # LLM provider: auto | groq | huggingface | ollama | demo
    llm_provider: str = "auto"
    llm_model: str = ""
    llm_base_url: str = ""
    groq_api_key: str = ""
    hf_token: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout: float = 120.0

    # Embeddings: auto | huggingface | ollama | hash
    embeddings_provider: str = "auto"
    embeddings_model: str = "BAAI/bge-m3"

    # Vector store: auto | chroma | numpy
    vector_backend: str = "auto"
    chroma_dir: str = "./chroma_db"
    vector_dir: str = "./vector_store"

    # Knowledge base
    data_dir: str = "./app/data/legal"

    # Uploads
    uploads_dir: str = "./uploads"
    max_upload_mb: int = 15

    # CORS
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Seed admin
    admin_email: str = "admin@nyaya.demo"
    admin_password: str = "NyayaAdmin@2026"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
