#!/usr/bin/env python3
"""
Test script for IT Support Chatbot.

Tests the basic functionality without running the full API server.
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


def test_utilities():
    """Test utility functions."""
    print("\n" + "=" * 80)
    print("Testing Utility Functions")
    print("=" * 80)
    
    # Test LLM
    print("\n1. Testing LLM...")
    from utils.call_llm_groq import call_llm
    response = call_llm("Say 'Hello, World!' in exactly 2 words.")
    print(f"   Response: {response}")
    
    # Test embedding
    print("\n2. Testing Embedding...")
    from utils.embedding_local import get_embedding
    emb = get_embedding("test text")
    print(f"   Embedding dim: {len(emb)}")
    
    # Test intent classifier
    print("\n3. Testing Intent Classifier...")
    from utils.intent_classifier import classify_intent
    intent = classify_intent("My VPN is not working")
    print(f"   Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
    
    # Test chunker
    print("\n4. Testing Chunker...")
    from utils.chunker import chunk_text
    chunks = chunk_text("This is a test. " * 50, chunk_size=100, chunk_overlap=20)
    print(f"   Created {len(chunks)} chunks")
    
    # Test redactor
    print("\n5. Testing Redactor...")
    from utils.redactor import redact_text
    text = "My email is john@company.com and password is secret123"
    redacted = redact_text(text)
    print(f"   Original: {text}")
    print(f"   Redacted: {redacted}")
    
    print("\n✓ All utility tests passed")


def test_indexing():
    """Test document indexing flow."""
    print("\n" + "=" * 80)
    print("Testing Document Indexing Flow")
    print("=" * 80)
    
    from flows import get_flow
    from utils.chromadb_client import delete_collection
    
    # Switch to local mode for testing if server not available
    original_mode = os.getenv("CHROMADB_MODE")
    os.environ["CHROMADB_MODE"] = "local"
    
    try:
        # Delete old collection to recreate with cosine similarity
        print("\nDeleting old collection (if exists)...")
        try:
            delete_collection()
        except Exception as e:
            print(f"   No existing collection to delete: {e}")
        
        # Prepare shared store
        shared = {
            "source_dir": "./data/docs"
        }
        
        # Run indexing flow
        print("\nRunning indexing flow (local mode)...")
        flow = get_flow("indexing")
        flow.run(shared)
        
        print(f"\n✓ Indexing completed:")
        print(f"   Documents loaded: {len(shared.get('documents', []))}")
        print(f"   Chunks created: {len(shared.get('all_chunks', []))}")
        print(f"   Chunks indexed: {shared.get('indexed_count', 0)}")
    finally:
        # Restore original mode
        if original_mode:
            os.environ["CHROMADB_MODE"] = original_mode
        else:
            os.environ.pop("CHROMADB_MODE", None)


def test_query_flow():
    """Test query answering flow."""
    print("\n" + "=" * 80)
    print("Testing Query Flow")
    print("=" * 80)
    
    from flows import get_flow
    import uuid
    
    # Switch to local mode for testing if server not available
    original_mode = os.getenv("CHROMADB_MODE")
    os.environ["CHROMADB_MODE"] = "local"
    
    try:
        # Test queries
        test_queries = [
            "How do I set up VPN?",
            "My printer is not printing, what should I do?",
            "How do I reset my password?",
            "What is the capital of France?",  # Should not find relevant docs
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n{'-' * 80}")
            print(f"Query {i+1}: {query}")
            print(f"{'-' * 80}")
            
            # Prepare shared store
            shared = {
                "session_id": str(uuid.uuid4()),
                "user_id": "test_user",
                "user_query": query,
                "turn_count": 1,
                "response": {}
            }
            
            # Run query flow
            flow = get_flow("simple")  # Use simple flow for testing
            flow.run(shared)
            
            # Print results
            response = shared.get("response", {})
            print(f"\nIntent: {shared.get('intent', {}).get('intent', 'unknown')}")
            print(f"Retrieved docs: {len(shared.get('retrieved_docs', []))}")
            print(f"\nResponse:\n{response.get('text', 'No response')}")
    finally:
        # Restore original mode
        if original_mode:
            os.environ["CHROMADB_MODE"] = original_mode
        else:
            os.environ.pop("CHROMADB_MODE", None)


def main():
    """Run all tests."""
    try:
        # Test utilities first
        test_utilities()
        
        # Test indexing (only if docs exist)
        if os.path.exists("./data/docs"):
            test_indexing()
        else:
            print("\n⚠ Skipping indexing test: ./data/docs not found")
        
        # Test query flow
        test_query_flow()
        
        print("\n" + "=" * 80)
        print("✓ All tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
