import os
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_vector_db import get_vector_db
from logger_config import setup_logger
from config import settings

logger = setup_logger('embed')
os.makedirs(settings.logs_path, exist_ok=True)

def embed(file):
    if not file or not file.name.lower().endswith(('.pdf', '.txt')):
        logger.warning(f"Invalid file type: {file.name if file else 'No file'}")
        return False

    try:
        logger.info(f"Processing file: {file.name}")
        timestamp = datetime.now().timestamp()
        filename = f"{timestamp}_{secure_filename(file.name)}"
        file_path = os.path.join(settings.temp_folder, filename)
        
        # Write file contents manually instead of using save()
        with open(file_path, 'wb') as f:
            f.write(file.read())

        loader = PyPDFLoader(file_path) if filename.endswith('.pdf') else TextLoader(file_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = splitter.split_documents(loader.load())
        logger.debug(f"Split into {len(chunks)} chunks")

        if chunks:
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            get_vector_db().add_documents(chunks)
            return True

    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return False
    
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")

    return False