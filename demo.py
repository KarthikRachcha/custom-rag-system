#!/usr/bin/env python3
"""
Demo script for the Custom RAG System
Shows how to use the system programmatically
"""

import sys
import os
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import RAGPipeline

def demo():
    """Demonstrate the RAG system capabilities"""
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 Custom RAG System Demo")
    print("=" * 50)
    
    # Initialize RAG pipeline
    print("\n📚 Initializing RAG system...")
    rag = RAGPipeline("data")
    
    try:
        # Initialize the system
        rag.initialize()
        
        # Demo questions
        demo_questions = [
            "What is LangGraph?",
            "What are the key concepts?",
            "What can LangGraph be used for?",
            "How does state management work?",
        ]
        
        print("\n🎯 Running demo questions...")
        print("-" * 30)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n❓ Question {i}: {question}")
            print("-" * 40)
            
            try:
                result = rag.query(question)
                
                if result and result['sources']:
                    print(f"🤖 Answer: {result['response']}")
                    print(f"📚 Sources: {len(result['sources'])} chunks found")
                    
                    # Show top source
                    if result['sources']:
                        top_source = result['sources'][0]
                        print(f"📖 Top source: {top_source['source']}")
                        if 'similarity' in top_source:
                            print(f"🎯 Similarity: {top_source['similarity']:.3f}")
                else:
                    print("❌ No relevant information found")
                    
            except Exception as e:
                print(f"❌ Error processing question: {e}")
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Try running 'python run.py' for interactive mode")
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("💡 Make sure you have:")
        print("   - OpenAI API key in .env file")
        print("   - Documents in the data/ folder")
        print("   - All dependencies installed")

if __name__ == "__main__":
    demo()
