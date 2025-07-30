"""
tools/rag_tool.py

Complete RAG document search tool with proposition-based chunking.
Supports PDF, CSV, JSON files with intelligent semantic chunking.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.tools import tool
from langchain.docstore.document import Document
import pandas as pd
import PyPDF2

# Fix deprecated imports
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

# Import settings with fallbacks
try:
    from settings import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, CHROMA_PERSIST_DIR
except ImportError:
    RAG_CHUNK_SIZE = 1000
    RAG_CHUNK_OVERLAP = 200
    CHROMA_PERSIST_DIR = "./data/chroma_db"

# Import embedding model from llm_setup
try:
    from llm_setup import embedding_model
except ImportError:
    embedding_model = None

class PropositionBasedChunker:
    """Advanced chunking system that extracts and groups atomic propositions."""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.max_propositions_per_chunk = 5
        self.min_chunk_size = 100
        self.max_chunk_size = 2000
    
    def extract_propositions(self, text: str) -> List[str]:
        """Extract atomic propositions from text using LLM."""
        if not self.llm:
            return self._fallback_proposition_extraction(text)
        
        try:
            prompt = f"""
            Extract atomic propositions from this text. Each proposition should be:
            1. A single, complete fact or statement
            2. Self-contained and independently meaningful
            3. Free of redundant information
            4. Clear and unambiguous
            
            Rules:
            - Break compound sentences into separate propositions
            - Remove filler words and unnecessary details
            - Keep only factual, actionable, or policy-related information
            - Each proposition should be 10-50 words
            
            Text to analyze:
            {text}
            
            Return ONLY a JSON array of propositions, no other text.
            Example: ["Proposition 1", "Proposition 2", "Proposition 3"]
            """
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean up the response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            propositions = json.loads(content)
            
            # Filter and clean propositions
            cleaned_propositions = []
            for prop in propositions:
                if isinstance(prop, str) and len(prop.strip()) > 10:
                    cleaned_prop = prop.strip()
                    if not cleaned_prop.startswith(('Example:', 'Rules:', 'Text:')):
                        cleaned_propositions.append(cleaned_prop)
            
            return cleaned_propositions[:50]  # Limit to prevent overwhelming
            
        except Exception as e:
            print(f"   ⚠️  LLM proposition extraction failed: {e}")
            return self._fallback_proposition_extraction(text)
    
    def _fallback_proposition_extraction(self, text: str) -> List[str]:
        """Fallback proposition extraction using simple sentence splitting."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        propositions = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                # Basic filtering for meaningful sentences
                if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'should', 'must']):
                    propositions.append(sentence)
        
        return propositions[:30]  # Limit for fallback
    
    def group_related_propositions(self, propositions: List[str]) -> List[List[str]]:
        """Group related propositions into semantic chunks."""
        if not propositions:
            return []
        
        if len(propositions) <= self.max_propositions_per_chunk:
            return [propositions]
        
        # Simple semantic grouping based on keyword overlap
        chunks = []
        current_chunk = []
        
        for prop in propositions:
            if len(current_chunk) >= self.max_propositions_per_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            
            current_chunk.append(prop)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def create_semantic_chunks(self, text: str, source: str, file_type: str) -> List[Dict[str, Any]]:
        """Create semantic chunks from text using proposition-based approach."""
        print(f"   🧠 Extracting propositions...")
        propositions = self.extract_propositions(text)
        print(f"   ✅ Extracted {len(propositions)} propositions")
        
        print(f"   🔗 Grouping related propositions...")
        proposition_groups = self.group_related_propositions(propositions)
        print(f"   ✅ Created {len(proposition_groups)} semantic chunks")
        
        chunks = []
        for i, group in enumerate(proposition_groups):
            chunk_content = ' '.join(group)
            
            # Skip chunks that are too small or too large
            if len(chunk_content) < self.min_chunk_size or len(chunk_content) > self.max_chunk_size:
                continue
            
            chunk_info = {
                'content': chunk_content,
                'propositions': group,
                'proposition_count': len(group),
                'chunk_id': i,
                'source': source,
                'file_type': file_type,
                'semantic_type': self._classify_chunk(group),
                'metadata': {
                    'source': source,
                    'file_type': file_type,
                    'chunk_id': i,
                    'file_name': os.path.basename(source),
                    'proposition_count': len(group),
                    'semantic_type': self._classify_chunk(group)
                }
            }
            chunks.append(chunk_info)
        
        return chunks
    
    def _classify_chunk(self, propositions: List[str]) -> str:
        """Classify the semantic type of a chunk based on its propositions."""
        text = ' '.join(propositions).lower()
        
        if any(word in text for word in ['refund', 'return', 'money back', 'cancel']):
            return 'refund_policy'
        elif any(word in text for word in ['shipping', 'delivery', 'tracking', 'package']):
            return 'shipping_policy'
        elif any(word in text for word in ['password', 'login', 'account', 'security']):
            return 'account_security'
        elif any(word in text for word in ['error', 'problem', 'issue', 'troubleshoot']):
            return 'technical_support'
        elif any(word in text for word in ['policy', 'terms', 'condition', 'rule']):
            return 'policy'
        else:
            return 'general_info'

class RAGSystem:
    """Complete RAG system with document processing and vector search."""
    
    def __init__(self, llm=None):
        self.persist_directory = CHROMA_PERSIST_DIR
        self.chunk_size = RAG_CHUNK_SIZE
        self.chunk_overlap = RAG_CHUNK_OVERLAP
        
        # Initialize proposition-based chunker with LLM
        self.text_splitter = PropositionBasedChunker(llm=llm)
        
        # Use embedding model from llm_setup
        if embedding_model is not None:
            self.embeddings = embedding_model
        else:
            # Fallback if embedding model not available
            try:
                import torch
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={
                        'device': 'cpu',
                        'torch_dtype': torch.float32,
                        'low_cpu_mem_usage': True
                    },
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'}
                )
        self.vectorstore = None
        self.processed_chunks = []
        self.is_initialized = False
        
        self._initialize_vectorstore()
        self._auto_initialize()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Check if vector store exists and has data
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                try:
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    # Check if it actually has documents
                    try:
                        test_search = self.vectorstore.similarity_search("test", k=1)
                        if len(test_search) > 0:
                            self.is_initialized = True
                            return
                    except:
                        pass
                except Exception as e:
                    pass
            
            # Create new vector store
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                # Try with minimal configuration
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            
        except Exception as e:
            pass
    
    def _auto_initialize(self):
        """Auto-initialize with available documents."""
        if self.is_initialized:
            return
        
        # Look for documents in documents directory
        documents_dir = os.path.join(os.getcwd(), "documents")
        
        if not os.path.exists(documents_dir):
            return
        
        # Check for default documents
        default_docs = [
            "./documents/company_config_json.json",
            "./documents/support_tickets.csv",
            "./documents/Account Management and Security Policy.pdf",
            "./documents/Return and Refund Policy.pdf",
            "./documents/Shipping and Delivery Policy.pdf",
            "./documents/Technical Support And TroubleShooting.pdf"
        ]
        
        existing_docs = []
        for doc_path in default_docs:
            if os.path.exists(doc_path):
                existing_docs.append(doc_path)
        
        if existing_docs:
            self.index_documents(existing_docs)
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            text = ""
            pages_with_content = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                    pages_with_content += 1
            
            doc.close()
            
            if not text.strip():
                return self._process_pdf_fallback(file_path)
            
            return text
            
        except ImportError:
            return self._process_pdf_fallback(file_path)
        except Exception as e:
            return self._process_pdf_fallback(file_path)
    
    def _process_pdf_fallback(self, file_path: str) -> str:
        """Fallback PDF text extraction using PyPDF2."""
        try:
            print(f"   📄 Trying PyPDF2 fallback extraction...")
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                pages_with_content = 0
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                            pages_with_content += 1
                        else:
                            print(f"      ⚠️  Page {page_num + 1} has no extractable text (PyPDF2)")
                    except Exception as page_error:
                        print(f"      ⚠️  Error extracting from page {page_num + 1}: {page_error}")
                
                print(f"   ✅ PyPDF2 extracted text from {pages_with_content}/{len(pdf_reader.pages)} pages")
                return text
                
        except Exception as e:
            print(f"   ❌ PyPDF2 extraction failed: {e}")
            return ""
    

    
    def _process_csv(self, file_path: str) -> str:
        """Convert CSV to readable text."""
        try:
            print(f"   📊 Processing CSV file...")
            df = pd.read_csv(file_path)
            
            text = f"CSV File: {os.path.basename(file_path)}\n"
            text += f"Total rows: {len(df)}\nColumns: {', '.join(df.columns.tolist())}\n\n"
            
            # Add all rows
            for idx, row in df.iterrows():
                row_parts = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        clean_value = str(value).replace('\n', ' ').replace('\r', ' ').strip()
                        row_parts.append(f"{col}: {clean_value}")
                text += f"Row {idx + 1}: {' | '.join(row_parts)}\n\n"
            
            # Add column summary
            text += "\n--- Column Information ---\n"
            for col in df.columns:
                sample_values = df[col].dropna().head(3).tolist()
                text += f"{col}: Examples - {', '.join(map(str, sample_values))}\n"
            
            print(f"   ✅ Processed {len(df)} rows with {len(df.columns)} columns")
            return text
        except Exception as e:
            print(f"   ❌ Error processing CSV {file_path}: {e}")
            return ""
    
    def _process_json(self, file_path: str) -> str:
        """Convert JSON to readable text."""
        try:
            print(f"   🗂️  Processing JSON file...")
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            text = f"JSON File: {os.path.basename(file_path)}\n\n"
            text += self._json_to_text(data)
            
            print(f"   ✅ Processed JSON structure")
            return text
        except Exception as e:
            print(f"   ❌ Error processing JSON {file_path}: {e}")
            return ""
    
    def _json_to_text(self, data: Any, prefix: str = "", max_depth: int = 10) -> str:
        """Convert JSON data to readable text format."""
        if max_depth <= 0:
            return f"{prefix}: [Max depth reached]\n"
        
        text = ""
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_prefix = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    text += f"\n--- {current_prefix} ---\n"
                    text += self._json_to_text(value, current_prefix, max_depth - 1)
                else:
                    clean_value = str(value).replace('\n', ' ').replace('\r', ' ').strip()
                    text += f"{current_prefix}: {clean_value}\n"
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_prefix = f"{prefix}[{i}]" if prefix else f"Item_{i}"
                
                if isinstance(item, (dict, list)):
                    text += f"\n--- {current_prefix} ---\n"
                    text += self._json_to_text(item, current_prefix, max_depth - 1)
                else:
                    clean_value = str(item).replace('\n', ' ').replace('\r', ' ').strip()
                    text += f"{current_prefix}: {clean_value}\n"
        
        return text
    
    def _save_chunks_to_file(self):
        """Save all processed chunks to text file."""
        try:
            os.makedirs("./data", exist_ok=True)
            output_path = "./data/processed_chunks.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Processed Document Chunks\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total chunks: {len(self.processed_chunks)}\n")
                f.write(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, chunk_info in enumerate(self.processed_chunks, 1):
                    f.write(f"CHUNK {i:04d}\n")
                    f.write(f"Source: {chunk_info['source']}\n")
                    f.write(f"File Type: {chunk_info['file_type']}\n")
                    f.write(f"Semantic Type: {chunk_info.get('semantic_type', 'unknown')}\n")
                    f.write(f"Proposition Count: {chunk_info.get('proposition_count', 0)}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"Content: {chunk_info['content']}\n")
                    if 'propositions' in chunk_info:
                        f.write(f"\nPropositions:\n")
                        for j, prop in enumerate(chunk_info['propositions'], 1):
                            f.write(f"  {j}. {prop}\n")
                    f.write("=" * 80 + "\n\n")
            
            
        except Exception as e:
            pass
    
    def index_documents(self, document_paths: List[str]):
        """Index all documents into single vector store."""
        all_documents = []
        self.processed_chunks = []
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                continue
            
            file_ext = os.path.splitext(doc_path)[1].lower()
            
            # Extract text based on file type
            if file_ext == '.pdf':
                full_text = self._process_pdf(doc_path)
                file_type = "pdf"
            elif file_ext == '.csv':
                full_text = self._process_csv(doc_path)
                file_type = "csv"
            elif file_ext == '.json':
                full_text = self._process_json(doc_path)
                file_type = "json"
            else:
                continue
            
            if not full_text.strip():
                continue
            
            # Split into chunks
            chunks = self.text_splitter.create_semantic_chunks(full_text, doc_path, file_type)
            
            for i, chunk_info in enumerate(chunks):
                if chunk_info['content'].strip():
                    # Create document
                    doc = Document(
                        page_content=chunk_info['content'],
                        metadata=chunk_info['metadata']
                    )
                    all_documents.append(doc)
                    
                    # Store for text file
                    self.processed_chunks.append(chunk_info)
        
        if all_documents:
            self._save_chunks_to_file()
            
            try:
                # Add to vector store in batches
                batch_size = 50  # Smaller batches for better stability
                
                for i in range(0, len(all_documents), batch_size):
                    batch = all_documents[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                
                # Persist the vector store (Chroma auto-persists in newer versions)
                self.is_initialized = True
                
            except Exception as e:
                self.is_initialized = False
        else:
            pass
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using vector similarity."""
        try:
            if not self.vectorstore or not self.is_initialized:
                return []
            
            # Simple search with basic error handling
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "source": doc.metadata.get("source", "unknown"),
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "file_type": doc.metadata.get("file_type", "unknown")
                })
            
            return formatted_results
        except Exception as e:
            return []


# Global RAG system instance - only create once
_rag_system = None

def get_rag_system(llm=None):
    """Get or create the global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem(llm=llm)
    return _rag_system

def reset_rag_system():
    """Reset the global RAG system instance to force reinitialization."""
    global _rag_system
    _rag_system = None


@tool
def document_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search through company documents using vector similarity.
    
    Args:
        query: The search query or question
        max_results: Maximum number of results to return
        
    Returns:
        JSON string with search results
    """
    try:
        if not query.strip():
            return json.dumps({
                "success": False,
                "error": "Empty search query provided"
            })
        
        rag_system = get_rag_system()
        results = rag_system.search_documents(query, k=max_results)
        
        if not results:
            return json.dumps({
                "success": False,
                "message": "No relevant documents found",
                "query": query,
                "results": []
            })
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result["content"],
                "source": result["source"],
                "file_name": result["file_name"],
                "file_type": result["file_type"],
                "similarity_score": result["similarity_score"]
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }, indent=2)
    
    except Exception as e:
        error_msg = str(e)
        if "meta tensor" in error_msg.lower():
            # Try to reset and reinitialize the RAG system
            try:
                reset_rag_system()
                rag_system = get_rag_system()
                results = rag_system.search_documents(query, k=max_results)
                
                if results:
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            "content": result["content"],
                            "source": result["source"],
                            "file_name": result["file_name"],
                            "file_type": result["file_type"],
                            "similarity_score": result["similarity_score"]
                        })
                    
                    return json.dumps({
                        "success": True,
                        "query": query,
                        "results_count": len(formatted_results),
                        "results": formatted_results
                    }, indent=2)
            except Exception as retry_error:
                error_msg = f"Search failed after retry: {str(retry_error)}"
        
        return json.dumps({
            "success": False,
            "error": error_msg,
            "query": query
        })


def get_document_search_tools():
    """Return list of document search tools for agent binding."""
    return [document_search_tool]


def add_documents_to_index(document_paths: List[str], llm=None):
    """Add documents to the RAG index."""
    rag_system = get_rag_system(llm=llm)
    rag_system.index_documents(document_paths)


# Test function
def test_rag_system(llm=None):
    """Test the RAG system."""
    rag_system = get_rag_system(llm=llm)
    
    if not rag_system.is_initialized:
        return
    
    test_queries = [
        "installation error",
        "password reset", 
        "refund policy"
    ]
    
    for query in test_queries:
        result_str = document_search_tool.invoke({"query": query, "max_results": 2})
        result = json.loads(result_str)


if __name__ == "__main__":
    test_rag_system()