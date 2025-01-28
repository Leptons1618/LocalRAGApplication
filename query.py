from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from get_vector_db import get_vector_db, has_documents
from logger_config import setup_logger
from config import settings
import os

logger = setup_logger('query')
os.makedirs(settings.logs_path, exist_ok=True)

PROMPT_TEMPLATE = """You're a helpful AI assistant. Use this context to answer:
{context}

Question: {question}

Provide a concise, factual answer in Markdown. If unsure, say so."""

class QueryHandler:
    def __init__(self):
        logger.info(f"Initializing QueryHandler with model: {settings.llm_model}")
        self.llm = ChatOllama(model=settings.llm_model)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def get_welcome_message(self):
        return """**Welcome to RAG Chat!** 🌟\n\n- Upload documents to chat with them\n- Switch models in the sidebar\n- Clear history anytime"""

    def stream_query(self, query: str):
        try:
            logger.debug(f"Processing query: {query}")
            chain = self._build_chain()
            
            for chunk in chain.stream(query):
                yield chunk.content
                logger.debug("Received response chunk")
                
            logger.info("Query processed successfully")
            
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    def _build_chain(self):
        if not has_documents():
            logger.warning("No documents loaded - using base LLM")
            return self.llm
        
        logger.debug("Building RAG chain with documents")
        retriever = get_vector_db().as_retriever(
            search_kwargs={"k": settings.max_context_docs}
        )
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

def get_query_handler():
    return QueryHandler()