# AI Customer Support Assistant

Customer support system that handles refunds, technical support, and general questions using SmolAgents multi-agent framework.

## Features

- **Multi-Agent System**: Manager agent routes requests to specialized refund and support agents
- **Refund Processing**: Verifies orders and processes refunds automatically
- **Document Search**: Finds answers in company documents using vector search
- **Email Formatting**: Converts responses to professional customer service emails
- **Web Interface**: Simple chat interface for testing

## Project Structure

```
customer-support-assistant/
├── data/
│   ├── semantic_vector_store/  # Vector database
│   ├── orders_database.json    # Customer orders
│   └── processed_chunks.txt    # Document chunks
├── documents/                   # Company documents
├── frontend.py                  # Web interface
├── graph.py                     # Workflow graph
├── nodes.py                     # SmolAgents integration
├── settings.py                  # Configuration
├── rag_tools.py                # Document search tools
└── refund_tools.py             # Refund tools
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
   GROQ_API_KEY=your_api_key
   HUGGINGFACE_TOKEN=your_token 
   GEMINI_API_KEY=your key
   ```

3. **Run the application**
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
- `document_search_tool`: Searches company documents

## Example Usage

- Refunds: "Refund order #12345, email: john@example.com"
- Technical: "How do I fix installation error 1603?"
- General: "What's your return policy?"

## Configuration

- `settings.py`: API keys, model settings, agent prompts
- `documents/`: Add company documents here for search
- `company_config_json.json`: Company policies and rules

Built with SmolAgents, LangGraph, and Groq AI.
