import streamlit as st
from query import get_query_handler
from embed import embed
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
            "",  # Empty label to remove header
            get_available_models(),
            index=0,
            key="model_selector"
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Type your message...", key="main_chat_input"):
        logger.info(f"User query: {prompt}")
        # Show user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show assistant response
        with st.chat_message("assistant"):
            try:
                full_response = []
                response_container = st.empty()
                
                for chunk in query_handler.stream_query(prompt):
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
    
    init_session()
    global query_handler
    query_handler = get_query_handler()

    # Simplified Sidebar
    with st.sidebar:
        st.title("📚 Documents")
        
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