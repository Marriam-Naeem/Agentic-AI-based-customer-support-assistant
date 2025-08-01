import json
import os
from typing import List, Dict, Any
from datetime import datetime
from langchain.tools import tool
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import PyPDF2

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    SemanticChunker = None

from settings import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, CHROMA_PERSIST_DIR

try:
    from llm_setup import embedding_model
except ImportError:
    embedding_model = None


class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len, separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_chunks(self, text: str, source: str, file_type: str) -> List[Dict[str, Any]]:
        text_chunks = self.text_splitter.split_text(text)
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {
                        "source": source, "file_name": os.path.basename(source),
                        "file_type": file_type, "chunk_index": i,
                        "chunk_type": "text", "processed_at": datetime.now().isoformat()
                    }
                })
        return chunks


class SemanticTextSplitter:
    def __init__(self, embeddings_model):
        if SemanticChunker is None:
            raise ImportError("SemanticChunker not available. Please install langchain-experimental.")
        self.semantic_splitter = SemanticChunker(
            embeddings_model, 
            breakpoint_threshold_type="gradient"
        )
    
    def create_chunks(self, text: str, source: str, file_type: str) -> List[Dict[str, Any]]:
        try:
            text_chunks = self.semantic_splitter.split_text(text)
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,  # Return as-is without strip()
                        "metadata": {
                            "source": source, "file_name": os.path.basename(source),
                            "file_type": file_type, "chunk_index": i,
                            "chunk_type": "semantic", "processed_at": datetime.now().isoformat()
                        }
                    })
            return chunks
        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            return []


class RAGSystem:
    def __init__(self):
        self.persist_directory = CHROMA_PERSIST_DIR
        self.semantic_persist_directory = os.path.join(os.path.dirname(CHROMA_PERSIST_DIR), "semantic_vector_store")
        self.text_splitter = SimpleTextSplitter(RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
        self.embeddings = self._get_embeddings()
        self.vectorstore = None
        self.semantic_vectorstore = None
        self.processed_chunks = []
        self.semantic_chunks = []
        self.is_initialized = False
        self.semantic_initialized = False
        self._initialize_vectorstore()
        self._initialize_semantic_vectorstore()
        self._auto_initialize()
    
    def _get_embeddings(self):
        if embedding_model is not None:
            return embedding_model
        try:
            import torch
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu', 'torch_dtype': torch.float32, 'low_cpu_mem_usage': True},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def _initialize_vectorstore(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                try:
                    self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
                    test_search = self.vectorstore.similarity_search("test", k=1)
                    if test_search:
                        self.is_initialized = True
                        return
                except Exception:
                    pass
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
    
    def _initialize_semantic_vectorstore(self):
        try:
            os.makedirs(self.semantic_persist_directory, exist_ok=True)
            if os.path.exists(self.semantic_persist_directory) and os.listdir(self.semantic_persist_directory):
                try:
                    self.semantic_vectorstore = Chroma(persist_directory=self.semantic_persist_directory, embedding_function=self.embeddings)
                    test_search = self.semantic_vectorstore.similarity_search("test", k=1)
                    if test_search:
                        self.semantic_initialized = True
                        return
                except Exception:
                    pass
            self.semantic_vectorstore = Chroma(persist_directory=self.semantic_persist_directory, embedding_function=self.embeddings)
        except Exception as e:
            print(f"Failed to initialize semantic vector store: {e}")
    
    def _auto_initialize(self):
        if self.is_initialized:
            return
        try:
            if self.vectorstore:
                test_search = self.vectorstore.similarity_search("test", k=1)
                if test_search:
                    self.is_initialized = True
                    return
        except Exception:
            pass
        
        documents_dir = "./documents"
        if os.path.exists(documents_dir):
            all_docs = [os.path.join(documents_dir, f) for f in os.listdir(documents_dir) 
                       if os.path.isfile(os.path.join(documents_dir, f)) and 
                       os.path.splitext(f)[1].lower() in ['.pdf', '.csv', '.json']]
            if all_docs:
                self.index_documents(all_docs)
    
    def _process_pdf(self, file_path: str) -> str:
        try:
            import fitz
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
            doc.close()
            return text if text.strip() else ""
        except ImportError:
            pass
        except Exception:
            pass
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception:
                        continue
                return text
        except Exception:
            return ""
    
    def _process_csv(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            text = f"CSV File: {os.path.basename(file_path)}\nTotal rows: {len(df)}\nColumns: {', '.join(df.columns.tolist())}\n\n"
            for idx, row in df.iterrows():
                row_parts = [f"{col}: {str(value).replace('\n', ' ').replace('\r', ' ').strip()}" 
                           for col in df.columns for value in [row[col]] 
                           if pd.notna(value) and str(value).strip()]
                text += f"Row {idx + 1}: {' | '.join(row_parts)}\n\n"
            text += "\n--- Column Information ---\n"
            for col in df.columns:
                sample_values = df[col].dropna().head(3).tolist()
                text += f"{col}: Examples - {', '.join(map(str, sample_values))}\n"
            return text
        except Exception:
            return ""
    
    def _process_json(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            text = f"JSON File: {os.path.basename(file_path)}\n\n"
            text += self._json_to_text(data)
            return text
        except Exception:
            return ""
    
    def _json_to_text(self, data: Any, prefix: str = "", max_depth: int = 10) -> str:
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
        try:
            os.makedirs("./data", exist_ok=True)
            with open("./data/processed_chunks.txt", 'w', encoding='utf-8') as f:
                f.write(f"Processed Document Chunks\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total chunks: {len(self.processed_chunks)}\nChunk size: {RAG_CHUNK_SIZE}, Overlap: {RAG_CHUNK_OVERLAP}\n")
                f.write("=" * 80 + "\n\n")
                for i, chunk_info in enumerate(self.processed_chunks, 1):
                    f.write(f"CHUNK {i:04d}\nSource: {chunk_info['metadata']['source']}\n")
                    f.write(f"File Type: {chunk_info['metadata']['file_type']}\nChunk Index: {chunk_info['metadata']['chunk_index']}\n")
                    f.write("-" * 60 + "\nContent: {}\n".format(chunk_info['content']) + "=" * 80 + "\n\n")
        except Exception:
            pass
    
    def _save_semantic_chunks_to_file(self):
        try:
            os.makedirs("./data", exist_ok=True)
            with open("./data/semantic_chunks.txt", 'w', encoding='utf-8') as f:
                f.write(f"Semantic Document Chunks\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total semantic chunks: {len(self.semantic_chunks)}\n")
                f.write("=" * 80 + "\n\n")
                for i, chunk_info in enumerate(self.semantic_chunks, 1):
                    f.write(f"SEMANTIC CHUNK {i:04d}\nSource: {chunk_info['metadata']['source']}\n")
                    f.write(f"File Type: {chunk_info['metadata']['file_type']}\nChunk Index: {chunk_info['metadata']['chunk_index']}\n")
                    f.write("-" * 60 + "\nContent: {}\n".format(chunk_info['content']) + "=" * 80 + "\n\n")
        except Exception:
            pass
    
    def index_documents(self, document_paths: List[str]):
        if self.is_initialized:
            return
        all_documents = []
        self.processed_chunks = []
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                continue
            file_ext = os.path.splitext(doc_path)[1].lower()
            
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
            
            chunks = self.text_splitter.create_chunks(full_text, doc_path, file_type)
            for chunk_info in chunks:
                if chunk_info['content'].strip():
                    doc = Document(page_content=chunk_info['content'], metadata=chunk_info['metadata'])
                    all_documents.append(doc)
                    self.processed_chunks.append(chunk_info)
        
        if all_documents:
            self._save_chunks_to_file()
            try:
                batch_size = 50
                for i in range(0, len(all_documents), batch_size):
                    batch = all_documents[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                self.is_initialized = True
            except Exception:
                self.is_initialized = False
    
    def index_documents_semantic(self, document_paths: List[str]):
        if self.semantic_initialized:
            return
        
        try:
            semantic_splitter = SemanticTextSplitter(self.embeddings)
        except ImportError:
            print("SemanticChunker not available. Please install langchain-experimental.")
            return
        
        all_semantic_documents = []
        self.semantic_chunks = []
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                continue
            file_ext = os.path.splitext(doc_path)[1].lower()
            
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
            
            semantic_chunks = semantic_splitter.create_chunks(full_text, doc_path, file_type)
            for chunk_info in semantic_chunks:
                if chunk_info['content'].strip():
                    doc = Document(page_content=chunk_info['content'], metadata=chunk_info['metadata'])
                    all_semantic_documents.append(doc)
                    self.semantic_chunks.append(chunk_info)
        
        if all_semantic_documents:
            self._save_semantic_chunks_to_file()
            try:
                batch_size = 50
                for i in range(0, len(all_semantic_documents), batch_size):
                    batch = all_semantic_documents[i:i + batch_size]
                    self.semantic_vectorstore.add_documents(batch)
                self.semantic_initialized = True
            except Exception as e:
                print(f"Error indexing semantic documents: {e}")
                self.semantic_initialized = False
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.vectorstore or not self.is_initialized:
                return []
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content, "metadata": doc.metadata,
                    "similarity_score": float(score), "source": doc.metadata.get("source", "unknown"),
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "file_type": doc.metadata.get("file_type", "unknown")
                })
            return formatted_results
        except Exception:
            return []
    
    def search_documents_semantic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.semantic_vectorstore or not self.semantic_initialized:
                return []
            results = self.semantic_vectorstore.similarity_search_with_score(query, k=k)
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content, "metadata": doc.metadata,
                    "similarity_score": float(score), "source": doc.metadata.get("source", "unknown"),
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "chunk_type": "semantic"
                })
            return formatted_results
        except Exception:
            return []


_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


@tool
def document_search_tool(query: str, max_results: int = 5) -> str:
    """Search through company documents using vector similarity."""
    try:
        if not query.strip():
            return json.dumps({"success": False, "error": "Empty search query provided"})
        
        rag_system = get_rag_system()
        results = rag_system.search_documents(query, k=max_results)
        
        if not results:
            return json.dumps({
                "success": False, "message": "No relevant documents found",
                "query": query, "results": []
            })
        
        formatted_results = [{
            "content": result["content"], "source": result["source"],
            "file_name": result["file_name"], "file_type": result["file_type"],
            "similarity_score": result["similarity_score"]
        } for result in results]
        
        # Pretty print retrieved context to console
        print("\n" + "="*80)
        print("RETRIEVED CONTEXT FOR EXAMINATION")
        print("="*80)
        print(f"Query: {query}")
        print(f"Results found: {len(formatted_results)}")
        print("="*80 + "\n")
        
        for i, result in enumerate(formatted_results, 1):
            print(f"Result {i}")
            print(f"Source: {result.get('source', 'Unknown')}")
            print(f"File: {result.get('file_name', 'Unknown')}")
            print(f"Type: {result.get('file_type', 'Unknown')}")
            print(f"Relevance Score: {result.get('similarity_score', 0):.4f}")
            print("-" * 60)
            print(f"Content:")
            print(result.get('content', 'No content available'))
            print("="*80 + "\n")
        
        return json.dumps({
            "success": True, "query": query,
            "results_count": len(formatted_results), "results": formatted_results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "query": query})


@tool
def semantic_document_search_tool(query: str, max_results: int = 5) -> str:
    """Search through company documents using semantic chunking and vector similarity."""
    try:
        if not query.strip():
            return json.dumps({"success": False, "error": "Empty search query provided"})
        
        rag_system = get_rag_system()
        results = rag_system.search_documents_semantic(query, k=max_results)
        
        if not results:
            return json.dumps({
                "success": False, "message": "No relevant semantic documents found",
                "query": query, "results": []
            })
        
        formatted_results = [{
            "content": result["content"], "source": result["source"],
            "file_name": result["file_name"], "file_type": result["file_type"],
            "similarity_score": result["similarity_score"], "chunk_type": "semantic"
        } for result in results]
        
        # Pretty print retrieved context to console
        print("\n" + "="*80)
        print("SEMANTIC RETRIEVED CONTEXT FOR EXAMINATION")
        print("="*80)
        print(f"Query: {query}")
        print(f"Results found: {len(formatted_results)}")
        print("="*80 + "\n")
        
        for i, result in enumerate(formatted_results, 1):
            print(f"Semantic Result {i}")
            print(f"Source: {result.get('source', 'Unknown')}")
            print(f"File: {result.get('file_name', 'Unknown')}")
            print(f"Type: {result.get('file_type', 'Unknown')}")
            print(f"Relevance Score: {result.get('similarity_score', 0):.4f}")
            print(f"Chunk Type: {result.get('chunk_type', 'semantic')}")
            print("-" * 60)
            print(f"Content:")
            print(result.get('content', 'No content available'))
            print("="*80 + "\n")
        
        return json.dumps({
            "success": True, "query": query,
            "results_count": len(formatted_results), "results": formatted_results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "query": query})


def get_document_search_tools():
    return [document_search_tool, semantic_document_search_tool]