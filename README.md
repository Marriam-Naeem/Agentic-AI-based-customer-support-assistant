# AI Customer Support Assistant

A smart customer support system that automatically handles refund requests, technical problems, and general questions using AI agents and document search.

## 🚀 Features

- **Smart Routing**: Automatically figures out what type of help you need
- **Refund Processing**: Handles refund requests with order verification
- **Technical Support**: Searches through documents to solve technical problems
- **Document Search**: Finds answers in company policies and support guides
- **Error Handling**: Gracefully handles problems and escalates when needed
- **Clean Code**: Well-organized and easy to maintain

## 📁 Project Structure

```
customer-support-assistant/
├── data/                        # Database and search data
│   ├── chroma_db/              # Document search database
│   ├── orders_database.json    # Customer orders and data
│   └── processed_chunks.txt    # Document sections for search
├── documents/                   # Company documents and policies
│   ├── Account Management and Security Policy.pdf
│   ├── company_config_json.json
│   ├── Return and Refund Policy.pdf
│   ├── Shipping and Delivery Policy.pdf
│   ├── support_tickets.csv
│   └── Technical Support And TroubleShooting.pdf
├── frontend.py                  # Web interface (85 lines)
├── graph.py                     # Main workflow (35 lines)
├── llm_setup.py                 # AI model setup (58 lines)
├── nodes.py                     # Core logic (248 lines)
├── pyproject.toml              # Project settings
├── requirements.txt            # Python packages needed
├── settings.py                 # App settings and prompts
├── states.py                   # Data management
├── rag_tools.py                # Document search tools
├── refund_tools.py             # Refund handling tools
├── uv.lock                     # Package versions
└── updated_workflow.png        # Workflow diagram
```

## 🏗️ How It Works

### The Three Helpers

1. **Router**: Looks at your question and decides who can help best
2. **Refund Helper**: Handles refund requests and checks orders
3. **Support Helper**: Finds answers in documents and solves technical problems

### Keeping Track

The system remembers:
- What you asked and your session info
- What type of help you need
- Results from checking orders or searching documents
- Any errors or if you need human help
- The final answer for you

## 🛠️ Getting Started

### What You Need

- Python 3.8 or newer
- UV package manager (recommended) or pip

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer-support-assistant
   ```

2. **Install packages**
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up your settings**
   - Copy `settings.py` and add your API keys
   - Make sure all documents are in the `documents/` folder

## 🚀 Using the System

### Start the Web Interface

```bash
# Start the web app
python frontend.py
```

This opens a web page at `http://localhost:7860` where you can chat with the assistant.

This runs some test questions and shows how the system works.

### What the Web Interface Does

The web page gives you:
- **Chat Box**: Type your questions and get answers
- **Quick Responses**: Gets back to you right away
- **Clear Answers**: Shows you what type of help you're getting
- **Example Questions**: Built-in examples to try out
- **Error Messages**: Tells you if something goes wrong

### Example Questions

You can ask things like:

- **Refunds**: "I want a refund for order #12345, my email is john@example.com"
- **Technical Problems**: "How do I fix TechOffice Suite installation error 1603?"
- **General Questions**: "What's your return policy?" or "How do I track my order?"
- **Multiple Issues**: "My software won't install and I also need to know how to change my email address"

## 🔄 How It Processes Your Question

1. **Figure Out What You Need**: Looks at your question to see if it's about refunds, technical problems, or general questions
2. **Send to the Right Helper**: Routes your question to the best helper for the job
3. **Get the Answer**: 
   - **Refund Helper**: Checks your order and processes refunds
   - **Support Helper**: Searches documents to find answers
4. **Give You the Answer**: Sends back the right response or tells you if you need human help

## 🛠️ Making Changes

### Adding New Types of Questions

1. Update the routing logic in `graph.py`
2. Add new helper logic in `nodes.py`
3. Update the workflow to handle the new type

### Adding New Tools

1. Create new tools in `rag_tools.py` or `refund_tools.py`
2. Update the main logic in `nodes.py`
3. Change the workflow as needed

## 🔧 Configuration

Important files:

- `settings.py`: App settings, API keys, and how the AI should respond
- `company_config_json.json`: Company policies and rules
- `documents/`: Company documents and support guides
- `data/`: Search database and document sections

**Built with LangGraph, LangChain, Groq AI, and modern AI tools**