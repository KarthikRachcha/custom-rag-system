import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize RAG pipeline
    rag = RAGPipeline("data")
    
    # Initialize the system
    rag.initialize()
    
    # Interactive query loop
    print("\nRAG System Ready! Type 'quit' to exit.")
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() == 'quit':
            break
        
        if question:
            result = rag.query(question)
            
            if result and result['sources']:
                print(f"\n{'='*60}")
                print(f"QUESTION: {result['question']}")
                print(f"{'='*60}")
                
                print(f"\n🤖 GENERATED RESPONSE:")
                print(f"{'─'*40}")
                print(result['response'])
                print(f"{'─'*40}")
                
                print(f"\n📚 SOURCES ({len(result['sources'])} chunks found):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n--- Source {i} ---")
                    print(f"File: {source['source']}")
                    if 'similarity' in source:
                        print(f"Similarity: {source['similarity']:.3f}")
                    print(f"Content: {source['content'][:150]}...")
            else:
                print("No relevant information found.")

if __name__ == "__main__":
    main()