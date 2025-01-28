import streamlit as st
from query import get_query_handler
from embed import embed
from get_vector_db import get_vector_db, archive_current_documents  # Add import
import os, time
import requests
from config import settings
from logger_config import setup_logger

logger = setup_logger('app')
os.makedirs(settings.logs_path, exist_ok=True)

# Constants
OLLAMA_API_TIMEOUT = 5  # seconds

def init_session():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": get_query_handler().get_welcome_message()
        }]
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False

def get_available_models():
    try:
        response = requests.get(
            'http://localhost:11434/api/tags',
            timeout=OLLAMA_API_TIMEOUT
        )
        if response.ok:
            models = [m['name'] for m in response.json()['models']]
            return sorted(models, key=lambda x: x != settings.llm_model)
    except requests.exceptions.RequestException:
        st.sidebar.error("⚠️ Could not connect to Ollama API")
    return [settings.llm_model]

def handle_file_upload(uploaded_files):
    if not uploaded_files:
        return

    with st.spinner(f"Processing {len(uploaded_files)} files..."):
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                if embed(file):
                    st.session_state.processed_files.add(file.name)
                else:
                    st.error(f"Failed to process {file.name}")

def render_chat():
    st.markdown("""
    <style>
    .upload-btn {
        position: fixed;
        left: 10px;
        bottom: 15px;
        z-index: 101;
    }
    .chat-input-container {
        position: relative;
        padding-left: 50px;
    }
    .title-section {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .stSelectbox {
        min-width: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Split header into columns for title and model selector
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.title("🤖 AXBot", help="Your AI Assistant")
    with col3:
        selected_model = st.selectbox(
            "Select Model",  # Added non-empty label
            get_available_models(),
            index=0,
            key="model_selector",
            label_visibility="hidden"  # Hide the label but keep it for accessibility
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Type your message...", key="main_chat_input"):
        logger.info(f"User query: {prompt}")
        
        # Check if this is a yes/no response to direct chat prompt
        last_message = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
        is_direct_prompt_response = "Would you like me to answer using general knowledge?" in last_message
        
        # Show user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Handle assistant response
        with st.chat_message("assistant"):
            try:
                full_response = []
                response_container = st.empty()
                
                # Handle Yes/No response to direct chat prompt
                if is_direct_prompt_response and prompt.lower().strip() in ['yes', 'y']:
                    logger.info("User opted for direct chat")
                    response_iterator = st.session_state.query_handler.stream_query(
                        st.session_state.messages[-2]["content"],  # Get original question
                        force_direct=True
                    )
                elif is_direct_prompt_response:
                    logger.info("User declined direct chat")
                    response_iterator = iter(["Okay, I'll only answer based on the documents I know about."])
                else:
                    response_iterator = st.session_state.query_handler.stream_query(prompt)

                for chunk in response_iterator:
                    full_response.append(chunk)
                    response_container.markdown(
                        f'{"".join(full_response)}<span class="stream-cursor">▋</span>', 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.02)
                
                final_response = "".join(full_response)
                response_container.markdown(final_response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_response
                })
                logger.info("Response generated successfully")
                
                # Show sources if available
                sources = st.session_state.query_handler.get_last_sources()
                if sources:
                    st.markdown("---\n**Reference Documents:**")
                    for idx, source in enumerate(sources, 1):
                        with st.expander(f"Source {idx}"):
                            st.markdown(source['content'])
                            if source.get('metadata'):
                                st.caption(f"Metadata: {source['metadata']}")
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)

def main():
    st.set_page_config(
        page_title="AXBot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if 'query_handler' not in st.session_state:
        st.session_state.query_handler = get_query_handler()
    
    init_session()

    # Simplified Sidebar
    with st.sidebar:
        st.title("📚 Documents")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": get_query_handler().get_welcome_message()
                }]
                st.rerun()
                
        with col2:
            if st.button("🗑️ New Session", use_container_width=True):
                if archive_current_documents():
                    st.session_state.clear()
                    st.rerun()
                else:
                    st.error("Failed to archive documents")
        
        # Document Upload Section
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        handle_file_upload(uploaded_files)

        # Document Status and List
        if st.session_state.processed_files:
            st.success(f"Loaded documents: {len(st.session_state.processed_files)}")
            st.write("Active Documents:")
            for file in sorted(st.session_state.processed_files):
                st.write(f"📄 {file}")
        else:
            st.info("No documents loaded yet")

        # Clear Documents Button
        if st.session_state.processed_files and st.button("🗑️ Clear Documents", use_container_width=True):
            st.session_state.processed_files.clear()
            st.rerun()

    # Main Chat Interface
    st.caption("Powered by Ollama LLMs")
    render_chat()

if __name__ == "__main__":
    main()