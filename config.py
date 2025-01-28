from pydantic_settings import BaseSettings
import logging
from pathlib import Path

class Settings(BaseSettings):
    llm_model: str = "llama3.2:latest"
    temp_folder: str = "./_temp"
    chroma_path: str = "./chroma"
    logs_path: str = "./logs"
    collection_name: str = "LocalRAG"
    text_embedding_model: str = "nomic-embed-text"
    log_level: str = "INFO"
    chunk_size: int = 2048
    chunk_overlap: int = 16
    max_context_docs: int = 3
    archive_chroma_path: str = "./chroma_archive"
    archive_collection_name: str = "LocalRAG_Archive"
    
    @property
    def log_level_value(self) -> int:
        return getattr(logging, self.log_level.upper(), logging.INFO)
    
    def validate_paths(self):
        Path(self.temp_folder).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self.archive_chroma_path).mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.validate_paths()