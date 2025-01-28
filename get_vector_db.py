from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import settings
from logger_config import setup_logger
import os

logger = setup_logger('vector_db')
os.makedirs(settings.logs_path, exist_ok=True)

_vector_store = None

def get_vector_db():
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing vector database")
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        _vector_store = Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_path,
            embedding_function=OllamaEmbeddings(model=settings.text_embedding_model)
        )
    return _vector_store

def has_documents():
    try:
        count = len(get_vector_db().get().get('ids', []))
        logger.debug(f"Vector store document count: {count}")
        return count > 0
    except Exception as e:
        logger.error(f"Document check failed: {str(e)}")
        return False