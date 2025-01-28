from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from get_vector_db import get_vector_db, has_documents, get_relevant_sources
from logger_config import setup_logger
from config import settings
import os
from datetime import datetime
import re

logger = setup_logger('query')
os.makedirs(settings.logs_path, exist_ok=True)

RAG_PROMPT = """You're a helpful AI assistant. Use this context to answer:
{context}

Question: {question}

Provide a concise, factual answer in Markdown. If unsure, say so."""

RELEVANCE_PROMPT = """Given this context and question, respond with 'relevant' or 'not relevant':
Context: {context}
Question: {question}
Response:"""

class QueryHandler:
    def __init__(self):
        logger.info(f"Initializing QueryHandler with model: {settings.llm_model}")
        self.llm = ChatOllama(model=settings.llm_model)
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        self.relevance_prompt = ChatPromptTemplate.from_template(RELEVANCE_PROMPT)
        self.last_sources = []

    def get_welcome_message(self):
        return """**Welcome to AXbot!** 🌟\n\n- Upload documents to chat with them\n- Switch models in the sidebar\n- Clear history anytime"""

    def stream_query(self, query: str, force_direct=False):
        try:
            if not has_documents() or force_direct:
                logger.info(f"Using direct chat mode - Reason: {'No documents' if not has_documents() else 'Forced direct'}")
                self.last_sources = []
                for chunk in self._direct_chat(query):
                    yield chunk
                return

            logger.debug("Checking query relevance to documents")
            if not self._is_query_relevant(query):
                logger.info("Query deemed not relevant to documents")
                self.last_sources = []
                yield "Your question doesn't seem related to the loaded documents. Would you like me to answer using general knowledge? (Yes/No)"
                return

            logger.info("Using RAG mode for relevant query")
            self.last_sources = get_relevant_sources(query)
            for chunk in self._rag_chat(query):
                yield chunk

        except Exception as e:
            self.last_sources = []
            error_msg = f"Query failed: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    def get_last_sources(self):
        """Get sources used in last query"""
        return self.last_sources if hasattr(self, 'last_sources') else []

    def _direct_chat(self, query: str):
        logger.debug("Processing direct chat query")
        for chunk in self.llm.stream(query):
            yield chunk.content

    def _rag_chat(self, query: str):
        logger.debug("Processing RAG query")
        try:
            # Get more documents for better context
            self.last_sources = get_relevant_sources(query, k=settings.max_context_docs)
            
            if not self.last_sources:
                logger.info("No relevant sources found for query")
                for chunk in self._direct_chat(query):
                    yield chunk
                return
            
            # Format context from relevant sources
            context = "\n\n".join([doc['content'] for doc in self.last_sources])
            
            # Use the chain with context
            chain = self.rag_prompt | self.llm
            for chunk in chain.stream({"context": context, "question": query}):
                yield chunk.content
                
        except Exception as e:
            logger.error(f"RAG chat error: {str(e)}")
            yield f"Error processing query: {str(e)}"

    def _is_query_relevant(self, query: str) -> bool:
        try:
            retriever = get_vector_db().as_retriever(
                search_kwargs={"k": 2}
            )
            docs = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])
            
            chain = self.relevance_prompt | self.llm
            response = chain.invoke({
                "context": context,
                "question": query
            })
            
            # Handle AIMessage object properly
            result = response.content if hasattr(response, 'content') else str(response)
            result = result.lower().strip()
            
            logger.debug(f"Relevance check result: {result}")
            return 'relevant' in result
            
        except Exception as e:
            logger.error(f"Relevance check failed: {str(e)}")
            return True  # Default to RAG mode on error

            yield f"Error processing query: {str(e)}"
def get_query_handler():
    return QueryHandler()

    def _is_query_relevant(self, query: str) -> bool:
        # Skip relevance check for basic conversational patterns
        basic_patterns = [
            r'\b(thanks|thank you|okay|ok|bye|goodbye)\b',
            r'\b(can you|could you|would you|will you)\b',
            r'\b(what|who|where|when|why|how)\b'
        ]
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in basic_patterns):
            logger.info("Detected conversational pattern, using direct chat")
            return False

        # Continue with document relevance check
        try:
            retriever = get_vector_db().as_retriever(
                search_kwargs={"k": 2}
            )
            docs = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])
            
            chain = self.relevance_prompt | self.llm
            response = chain.invoke({
                "context": context,
                "question": query
            })
            
            # Handle AIMessage object properly
            result = response.content if hasattr(response, 'content') else str(response)
            result = result.lower().strip()
            
            logger.debug(f"Relevance check result: {result}")
            return 'relevant' in result
            
        except Exception as e:
            logger.error(f"Relevance check failed: {str(e)}")
            return True  # Default to RAG mode on error

def get_query_handler():
    return QueryHandler()