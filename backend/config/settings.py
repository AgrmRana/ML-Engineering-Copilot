from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Database
    database_url: str
    test_database_url: str = ""
    
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-large"
    
    # Application
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    # Vector Database
    chroma_persist_dir: str = "./data/chroma"
    
    # Document Processing
    max_file_size_mb: int = 100
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval
    top_k_retrieval: int = 5
    min_confidence_score: float = 0.7
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
