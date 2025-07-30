#!/usr/bin/env python3
"""
test_proposition_chunking.py

Test script for the proposition-based chunking system.
"""

from rag_tools import PropositionBasedChunker

def test_chunking():
    """Test the proposition-based chunking system."""
    # Sample text for testing
    sample_text = """
    TechOffice Suite Installation Guide
    
    TechOffice Suite is a comprehensive office productivity software that includes word processing, 
    spreadsheet, and presentation tools. To install TechOffice Suite, follow these steps:
    
    1. Download the installer from our official website
    2. Run the installer as administrator
    3. Accept the license agreement
    4. Choose installation directory
    5. Wait for installation to complete
    
    If you encounter error 1603 during installation, try these troubleshooting steps:
    - Close all running applications
    - Disable antivirus temporarily
    - Run installer in compatibility mode
    - Check system requirements
    
    System Requirements:
    - Windows 10 or later
    - 4GB RAM minimum
    - 2GB free disk space
    - Internet connection for activation
    """
    
    # Initialize chunker
    chunker = PropositionBasedChunker()
    
    # Extract propositions
    propositions = chunker.extract_propositions(sample_text)
    
    # Group related propositions
    groups = chunker.group_related_propositions(propositions)
    
    # Create semantic chunks
    chunks = chunker.create_semantic_chunks(sample_text, "test_doc.txt", "txt")
    
    # Test with LLM if available
    try:
        from llm_setup import llm_models
        if llm_models:
            llm = llm_models.get("issue_faq_llm")
            if llm:
                chunker_with_llm = PropositionBasedChunker(llm=llm)
                llm_propositions = chunker_with_llm.extract_propositions(sample_text)
                llm_groups = chunker_with_llm.group_related_propositions(llm_propositions)
                llm_chunks = chunker_with_llm.create_semantic_chunks(sample_text, "test_doc.txt", "txt")
    except ImportError:
        pass

if __name__ == "__main__":
    test_chunking() 