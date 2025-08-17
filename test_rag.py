#!/usr/bin/env python3
"""
Test script for the RAG (Retrieval Augmented Generation) system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.main import RAGSystem
    print("✅ RAG system imported successfully")
    
    # Test RAG system initialization
    rag = RAGSystem()
    print("✅ RAG system initialized successfully")
    
    # Test document info (should be None initially)
    doc_info = rag.get_document_info()
    if doc_info is None:
        print("✅ Initial document state is correct (None)")
    else:
        print(f"❌ Unexpected initial document state: {doc_info}")
    
    # Test chat history (should be empty initially)
    chat_history = rag.get_chat_history()
    if len(chat_history) == 0:
        print("✅ Initial chat history is empty")
    else:
        print(f"❌ Unexpected initial chat history: {chat_history}")
    
    # Test starting new chat
    result = rag.start_new_chat()
    if result['success']:
        print("✅ New chat started successfully")
    else:
        print(f"❌ Failed to start new chat: {result['message']}")
    
    # Test asking question without document
    result = rag.ask_question("What is this document about?")
    if not result['success'] and "No document loaded" in result['message']:
        print("✅ Correctly prevented asking questions without document")
    else:
        print(f"❌ Unexpected behavior when asking question without document: {result}")
    
    # Test keyword extraction
    test_text = "This is a test document about artificial intelligence and machine learning."
    keywords = rag._extract_keywords(test_text)
    expected_keywords = ['test', 'document', 'about', 'artificial', 'intelligence', 'machine', 'learning']
    
    if set(keywords) == set(expected_keywords):
        print("✅ Keyword extraction working correctly")
    else:
        print(f"❌ Keyword extraction failed. Expected: {expected_keywords}, Got: {keywords}")
    
    # Test relevance calculation
    question_keywords = ['artificial', 'intelligence']
    document_keywords = ['artificial', 'intelligence', 'machine', 'learning', 'document']
    relevance = rag._calculate_relevance(question_keywords, document_keywords)
    
    if relevance == 1.0:  # Both keywords should match
        print("✅ Relevance calculation working correctly")
    else:
        print(f"❌ Relevance calculation failed. Expected: 1.0, Got: {relevance}")
    
    # Test relevance calculation with no matches
    question_keywords = ['python', 'programming']
    relevance = rag._calculate_relevance(question_keywords, document_keywords)
    
    if relevance == 0.0:  # No keywords should match
        print("✅ Relevance calculation with no matches working correctly")
    else:
        print(f"❌ Relevance calculation with no matches failed. Expected: 0.0, Got: {relevance}")
    
    print("\n🎉 RAG system test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
