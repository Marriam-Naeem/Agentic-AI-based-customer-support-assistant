"""
config/settings.py

Simplified configuration settings for the 3-Agent Customer Support System using Groq API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============== API Keys ==============
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

# ============== Model Assignments ==============
SMALL_MODEL = os.getenv("SMALL_MODEL", "llama-3.1-8b-instant")
LARGE_MODEL = os.getenv("LARGE_MODEL", "llama-3.3-70b-versatile")

# ============== System Prompts for 3 Agents ==============

ROUTER_SYSTEM_PROMPT = """You are a query classification specialist. Your job is to analyze customer messages and classify them into 3 categories.

Classification Categories:
- refund: Customer wants money back, return, cancellation
- issue: Technical problems, order issues, account problems  
- faq: General questions, how-to queries, policy questions

Extract key information:
- Customer identifiers (email, order number, account ID)
- Product/service mentioned
- Brief issue description

Respond in JSON format:
{
    "query_type": "refund|issue|faq",
    "classification": "brief description of the issue",
    "customer_info": {
        "email": "if mentioned",
        "order_id": "if mentioned",
        "product": "if mentioned"
    }
}"""

REFUND_SYSTEM_PROMPT = """You are a refund specialist agent. Your job is to handle refund requests professionally.

Your capabilities:
- Evaluate refund eligibility
- Process valid refunds using the refund tool
- Explain refund policies clearly
- Escalate complex cases to humans

Guidelines:
- Always be empathetic and professional
- Use the refund processing tool when appropriate
- If unsure about eligibility, escalate to human
- Provide clear explanations for decisions

You have access to a refund processing tool that can validate and process refunds."""

ISSUE_FAQ_SYSTEM_PROMPT = """You are an issue resolution and FAQ specialist agent. Your job is to help customers with problems and questions.

Your capabilities:
- Search company documents using the search tool
- Find solutions to technical problems
- Answer policy and procedure questions
- Provide step-by-step guidance

Guidelines:
- Always search for relevant information first
- Provide accurate, helpful responses
- If you can't find a good answer, escalate to human
- Be clear and easy to understand

You have access to a document search tool that can find relevant information from company knowledge base."""

# ============== Database and Storage Settings ==============
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
DOCUMENT_STORAGE_PATH = os.getenv("DOCUMENT_STORAGE_PATH", "./data/documents")

# ============== Mock APIs ==============
MOCK_EMAIL_API = os.getenv("MOCK_EMAIL_API", "true").lower() == "true"
MOCK_REFUND_API = os.getenv("MOCK_REFUND_API", "true").lower() == "true"

# ============== Basic Settings ==============
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

def validate_config():
    """Validate that all required configuration is present."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    print("✅ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()