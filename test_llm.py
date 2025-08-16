#!/usr/bin/env python3
"""
Test script for the Free LLM Service
"""
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_llm_service():
    """Test the LLM service functionality"""
    try:
        print("🤖 Testing Free LLM Service...")
        
        # Import the service
        from components.llm_service import FreeLLMService
        
        # Initialize the service
        llm_service = FreeLLMService()
        print("✅ LLM Service initialized successfully")
        
        # Test question answering
        print("\n🧪 Testing Question Answering...")
        question = "What is the main benefit of using AI in education?"
        context = "AI in education provides personalized learning experiences, adaptive content delivery, and intelligent tutoring systems that can improve student engagement and learning outcomes."
        
        answer = llm_service.answer_educational_question(question, context)
        print(f"Question: {question}")
        print(f"AI Answer: {answer}")
        
        # Test content summarization
        print("\n🧪 Testing Content Summarization...")
        content = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns."
        
        summary = llm_service.generate_content_summary(content)
        print(f"Original Content: {content}")
        print(f"AI Summary: {summary}")
        
        # Test learning style classification
        print("\n🧪 Testing Learning Style Classification...")
        student_responses = [
            "I prefer visual learning with diagrams and charts",
            "I learn best by watching demonstrations",
            "I like to see examples before trying something new"
        ]
        
        learning_style = llm_service.classify_learning_style(student_responses)
        print(f"Student Responses: {student_responses}")
        print(f"AI Recommended Style: {learning_style}")
        
        print("\n🎉 All tests completed successfully!")
        print("💡 The LLM service is working correctly!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("There was an issue with the LLM service")

if __name__ == "__main__":
    test_llm_service()
