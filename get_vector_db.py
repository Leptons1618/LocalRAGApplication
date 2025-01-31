from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import settings
from logger_config import setup_logger
import os

logger = setup_logger('vector_db')
os.makedirs(settings.logs_path, exist_ok=True)

_vector_store = None
_archive_store = None

def get_vector_db():
    global _vector_store
    if (_vector_store is None):
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

def archive_current_documents():
    """Archive current documents before clearing"""
    try:
        if not has_documents():
            return True
            
        logger.info("Archiving current documents")
        current_db = get_vector_db()
        
        global _archive_store
        if (_archive_store is None):
            _archive_store = Chroma(
                collection_name=settings.archive_collection_name,
                persist_directory=settings.archive_chroma_path,
                embedding_function=OllamaEmbeddings(model=settings.text_embedding_model)
            )
        
        # Transfer documents to archive
        current_data = current_db.get()
        if current_data.get('documents'):
            _archive_store.add_texts(
                texts=current_data['documents'],
                metadatas=current_data.get('metadatas', []),
                ids=current_data.get('ids', [])
            )
            logger.info(f"Archived {len(current_data['documents'])} documents")
            
        # Clear current DB
        current_db._collection.delete(current_data.get('ids', []))
        logger.info("Cleared current documents")
        return True
        
    except Exception as e:
        logger.error(f"Archive failed: {str(e)}")
        return False

def clear_documents():
    """Clear all documents from the current vector store"""
    try:
        if not has_documents():
            return True
            
        logger.info("Clearing current documents")
        current_db = get_vector_db()
        current_data = current_db.get()
        
        # Clear current DB
        if current_data.get('ids'):
            current_db._collection.delete(current_data.get('ids', []))
            logger.info("Cleared current documents")
        return True
        
    except Exception as e:
        logger.error(f"Clear documents failed: {str(e)}")
        return False

def get_relevant_sources(query: str, k: int = 4) -> list:
    """Get relevant source documents for a query"""
    try:
        if not has_documents():
            return []
            
        # Get documents with scores
        docs_and_scores = get_vector_db().similarity_search_with_score(query, k=k)
        
        # Filter out empty sources and include scores
        sources = []
        for doc, score in docs_and_scores:
            if doc.page_content.strip():
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)  # Convert to float for JSON serialization
                })
        
        return sources
    except Exception as e:
        logger.error(f"Error getting sources: {str(e)}")
        return []