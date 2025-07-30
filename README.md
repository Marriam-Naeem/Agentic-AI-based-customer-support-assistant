# Customer Assistant - AI-Powered Support System

An intelligent customer support system built with LangGraph that handles refund requests and customer inquiries using AI agents and specialized tools.

## 🚀 Features

- **Smart Query Routing**: Automatically identifies and routes refund requests to specialized processing
- **Multi-Step Refund Processing**: Handles order verification, customer authentication, and refund processing
- **Tool Integration**: Executes real-world actions like checking order status and processing refunds
- **Error Handling**: Robust error management and escalation capabilities
- **Visual Workflow**: Built-in workflow visualization for debugging and understanding

## 📁 Project Structure

```
customerAssistant/
├── documents/                    # Support documents and data
│   ├── Account Management and Security Policy.pdf
│   ├── company_config_json.json
│   ├── Return and Refund Policy.pdf
│   ├── Shipping and Delivery Policy.pdf
│   ├── support_tickets.csv
│   └── Technical Support And TroubleShooting.pdf
├── frontend.py                  # Gradio web interface
├── graph.py                     # Main workflow definition
├── llm_setup.py                 # LLM model configuration
├── nodes.py                     # Workflow node implementations
├── pyproject.toml              # Project configuration
├── requirements.txt            # Python dependencies
├── settings.py                 # Application settings
├── states.py                   # State management
├── tools.py                    # Tool implementations
├── uv.lock                     # Dependency lock file
└── workflow.png               # Generated workflow visualization
```

## 🏗️ Architecture

### Workflow Components

1. **Router Node**: Analyzes customer queries and routes refund requests
2. **Refund Node**: Handles refund-specific logic and customer verification
3. **Tools Node**: Executes external tools for order checking and refund processing

### State Management

The system uses a `SupportState` object to track:
- Customer query and session information
- Verification results
- Processing results
- Error states
- Final responses

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customerAssistant
   ```

2. **Install dependencies**
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `settings.py` and configure your API keys
   - Ensure all required documents are in the `documents/` folder

## 🚀 Usage

### Running the Application

#### Option 1: Web Interface (Recommended)
```bash
# Run the Gradio frontend
python frontend.py
```

This will launch a web interface at `http://localhost:7860` where you can interact with the chatbot.



### Testing the Workflow

```bash
python graph.py
```

This will run the test workflow with sample customer queries and generate a workflow visualization.

### Web Interface Features

The Gradio frontend provides:
- **Chat Interface**: Natural conversation with the AI assistant
- **Real-time Processing**: Instant responses to customer queries
- **Visual Feedback**: Clear indication of query types and processing status
- **Example Queries**: Built-in examples to test different scenarios
- **Error Handling**: Graceful error messages and recovery

### Example Queries

The system can handle various customer requests:

- **Refund Requests**: "I want a refund for order #12345, my email is john@example.com"
- **General Support**: "How do I track my order?"
- **Invalid Orders**: "Refund for order 99999, email nobody@example.com"

## 🔄 Workflow Flow

1. **Query Analysis**: Customer message is analyzed to determine query type
2. **Routing**: Refund queries are routed to specialized processing
3. **Verification**: Order and customer details are verified
4. **Processing**: Refund is processed if verification passes
5. **Response**: Customer receives appropriate response or error message

## 🛠️ Customization

### Adding New Query Types

1. Update the `router_condition` function in `graph.py`
2. Add new nodes in `nodes.py`
3. Update the workflow graph

### Adding New Tools

1. Implement new tools in `tools.py`
2. Update the `NodeFunctions` class in `nodes.py`
3. Modify workflow conditions as needed

## 📊 Testing

The project includes comprehensive testing with multiple scenarios:

- Valid refund requests
- Invalid order numbers
- General support queries
- Error handling cases

Run tests with:
```bash
python graph.py
```

## 🔧 Configuration

Key configuration files:

- `settings.py`: Application settings and API configurations
- `company_config_json.json`: Company-specific policies and rules
- `documents/`: Support documents and reference materials

**Built with LangGraph, LangChain, and modern AI technologies**