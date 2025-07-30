#!/usr/bin/env python3
"""
Test script for proposition-based chunking functionality.
"""

import json
from rag_tools import PropositionBasedChunker, test_rag_system

def test_proposition_extraction():
    """Test proposition extraction with sample text."""
    print("🧪 Testing Proposition Extraction")
    print("=" * 40)
    
    # Sample text from a policy document
    sample_text = """
    Our refund policy allows customers to return items within 30 days of purchase. 
    The item must be in original condition with all packaging intact. 
    Refunds will be processed within 5-7 business days after we receive the returned item. 
    Shipping costs are non-refundable unless the item was defective or we made an error. 
    Customers can initiate a return by logging into their account and selecting the return option, 
    or by contacting our customer service team directly.
    """
    
    # Create chunker without LLM (will use fallback)
    chunker = PropositionBasedChunker()
    
    print("📝 Sample text:")
    print(sample_text.strip())
    print()
    
    # Extract propositions
    print("🔍 Extracting propositions...")
    propositions = chunker.extract_propositions(sample_text)
    
    print(f"✅ Extracted {len(propositions)} propositions:")
    for i, prop in enumerate(propositions, 1):
        print(f"  {i}. {prop}")
    
    print()
    
    # Group propositions
    print("🔗 Grouping related propositions...")
    groups = chunker.group_related_propositions(propositions)
    
    print(f"✅ Created {len(groups)} semantic groups:")
    for i, group in enumerate(groups, 1):
        print(f"  Group {i}:")
        for j, prop in enumerate(group, 1):
            print(f"    {j}. {prop}")
        print()
    
    # Create semantic chunks
    print("📦 Creating semantic chunks...")
    chunks = chunker.create_semantic_chunks(sample_text, "test_policy.txt", "txt")
    
    print(f"✅ Created {len(chunks)} semantic chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}:")
        print(f"    Type: {chunk['semantic_type']}")
        print(f"    Propositions: {chunk['proposition_count']}")
        print(f"    Content: {chunk['content'][:100]}...")
        print()

def test_with_llm():
    """Test with LLM if available."""
    try:
        from llm_setup import get_models
        print("🤖 Testing with LLM...")
        
        models = get_models()
        llm = models.get("issue_faq_llm")
        
        if llm:
            test_rag_system(llm=llm)
        else:
            print("⚠️  No LLM available for testing")
            
    except ImportError:
        print("⚠️  llm_setup not available, skipping LLM test")

if __name__ == "__main__":
    test_proposition_extraction()
    test_with_llm() 