# Local RAG Application - AXBot

A local Retrieval-Augmented Generation (RAG) chatbot powered by Ollama LLMs. Chat with your documents or have general conversations with AI assistance.

## Features

- 📚 Document-based conversations using RAG
- 💬 Smart conversation mode switching (RAG/Direct Chat)
- 🤖 Multiple LLM model support via Ollama
- 📑 PDF and TXT file support
- 💾 Automatic document archiving
- 🔍 Source reference display
- 📊 Detailed logging system

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Required LLM models pulled in Ollama

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LocalRAGApplication.git
cd LocalRAGApplication
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Ollama models:
```bash
ollama pull mistral
# Or any other supported model
```

## Project Workflow

1. **Document Processing**
   - Place documents in the `docs` folder
   - System automatically processes and indexes new documents
   - Embeddings are stored in the vector database

2. **Chat Interaction**
   - Start conversation with the bot
   - System automatically detects if query needs RAG
   - Retrieves relevant document segments when needed
   - Provides responses with source references

3. **Logging and Maintenance**
   - All interactions are logged
   - Document index is updated automatically
   - System maintains conversation history

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Choose interaction mode:
   - Document chat (RAG mode)
   - General conversation (Direct chat)

3. Enter your queries and interact naturally

## Configuration

Edit `config.yaml` to customize:
- LLM model selection
- Document processing parameters
- Vector store settings
- Logging preferences

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
