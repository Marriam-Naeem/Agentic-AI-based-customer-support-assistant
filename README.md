# AI Customer Support Assistant

Customer support system that handles refunds, technical support, and general questions using SmolAgents multi-agent framework.

## Features

- **Multi-Agent System**: Manager agent routes requests to specialized refund and support agents
- **Refund Processing**: Verifies orders and processes refunds automatically
- **Document Search**: Finds answers in company documents using vector search
- **Email Formatting**: Converts responses to professional customer service emails
- **Web Interface**: Simple chat interface for testing and interaction
- **Redis Caching**: Efficient response caching for improved performance

## Project Structure

```
customer-support-assistant/
├── data/
│   ├── chroma_db/              # ChromaDB vector database
│   ├── semantic_vector_store/  # Semantic vector storage
│   ├── orders_database.json    # Customer orders database
│   ├── processed_chunks.txt    # Document chunks for processing
│   └── semantic_chunks.txt     # Semantic document chunks
├── documents/                   # Company documents and policies
│   ├── Account Management and Security Policy.pdf
│   ├── Return and Refund Policy.pdf
│   ├── Shipping and Delivery Policy.pdf
│   ├── Technical Support And TroubleShooting.pdf
│   ├── company_config_json.json
│   └── support_tickets.csv
├── frontend.py                  # Web interface using Gradio
├── graph.py                     # Workflow graph definition
├── nodes.py                     # SmolAgents integration nodes
├── llm_setup.py                # Language model configuration
├── rag_tools.py                # RAG (Retrieval-Augmented Generation) tools
├── refund_tools.py             # Refund processing tools
├── redis_cache_manager.py      # Redis caching implementation
├── states.py                    # Application state management
├── settings.py                  # Configuration settings
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── uv.lock                     # UV dependency lock file
```

## Setup

1. **Install dependencies**
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```

2. **Environment variables**
   Create `.env` file:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   HUGGINGFACE_TOKEN=your_token  # Optional for embeddings
   REDIS_URL=redis://localhost:6379  # Optional for caching
   ```

3. **Add company documents**
   Place your company documents (PDF, CSV, JSON) in the `documents/` folder. The system will automatically process and chunk them for search.

4. **Run the application**
   ```bash
   python frontend.py
   ```
   Opens at `http://localhost:7860`

## How It Works

### Agents

1. **Manager Agent**: Routes requests to appropriate specialized agents
2. **Refund Agent**: Handles refund verification and processing
3. **Support Agent**: Searches documents for technical solutions  
4. **Formatter Agent**: Formats responses as professional emails

### Workflow

1. Manager agent analyzes incoming request
2. Routes to refund agent (for orders/billing) or support agent (for technical issues)
3. Specialized agent executes appropriate tools
4. Formatter agent creates professional email response

### Tools

- `refund_verification_tool`: Checks order eligibility
- `refund_processing_tool`: Processes approved refunds
- `document_search_tool`: Searches company documents using semantic similarity
- `pdf_processing_tool`: Converts and chunks PDF documents
- `cache_manager`: Manages Redis-based response caching

## Example Usage

### Refund Requests
```
"I want a refund for order #12345, my email is john@example.com"
```

### Technical Support
```
"How do I fix TechOffice Suite installation error 1603?"
"What's your return policy?"
"I can't login to my account"
```

### General Questions
```
"What are your shipping policies?"
"How do I contact customer service?"
```

## Configuration

- `settings.py`: API keys, model settings, agent prompts, and system configuration
- `documents/`: Add company documents here for automatic processing and search
- `company_config_json.json`: Company policies, rules, and configuration
- `llm_setup.py`: Language model configuration and setup

Built with SmolAgents, LangGraph, and Groq AI.
