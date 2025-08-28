import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from document_loader import DocumentLoader
from text_chunker import TextChunker
from embedding_system import EmbeddingSystem
from vector_store import SimpleVectorStore

# Load environment variables from .env file
load_dotenv()

class RAGPipeline:
    def __init__(self, data_dir: str, api_key: str = None):
        self.document_loader = DocumentLoader(data_dir)
        self.text_chunker = TextChunker()
        self.embedding_system = EmbeddingSystem(api_key)
        self.vector_store = SimpleVectorStore()
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the RAG system by loading and processing documents"""
        print("Loading documents...")
        documents = self.document_loader.load_documents()
        
        print("Splitting documents into chunks...")
        chunks = self.text_chunker.split_documents(documents)
        
        print("Generating embeddings...")
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_system.generate_embeddings(chunk_texts)
        
        # Filter out chunks that failed to get embeddings
        valid_chunks = []
        valid_embeddings = []
        
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)
        
        if not valid_chunks:
            print("Error: No valid embeddings generated. Please check your API key.")
            print("System will continue with basic text search functionality.")
            # Store ALL chunks without embeddings for basic text search
            self.vector_store.add_chunks(chunks, [])
            self.is_initialized = True
            print(f"RAG system initialized with {len(chunks)} chunks (basic mode)")
            return
        
        print("Storing in vector database...")
        self.vector_store.add_chunks(valid_chunks, valid_embeddings)
        
        self.is_initialized = True
        print(f"RAG system initialized with {len(valid_chunks)} chunks")
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system and return generated response"""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        # Try to generate embedding for the question
        question_embeddings = self.embedding_system.generate_embeddings([question])
        
        if not question_embeddings or len(question_embeddings) == 0:
            print("Warning: Could not generate embedding. Using basic text search.")
            retrieved_chunks = self._basic_text_search(question, top_k)
        else:
            question_embedding = question_embeddings[0]
            
            if not question_embedding:
                print("Warning: Invalid embedding generated. Using basic text search.")
                retrieved_chunks = self._basic_text_search(question, top_k)
            else:
                # Search for relevant chunks using embeddings
                search_results = self.vector_store.search(question_embedding, top_k)
                retrieved_chunks = [
                    {
                        'chunk': chunk,
                        'similarity': score,
                        'content': chunk['content'],
                        'source': chunk['source']
                    }
                    for chunk, score in search_results
                ]
        
        # Generate response using retrieved chunks
        generated_response = self.generate_response(question, retrieved_chunks)
        
        # Return both the response and the source chunks
        return {
            'response': generated_response,
            'sources': retrieved_chunks,
            'question': question
        }
    
    def _basic_text_search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Basic text search when embeddings are not available"""
        question_lower = question.lower()
        results = []
        
        print(f"Debug: Searching through {len(self.vector_store.chunks)} chunks")
        print(f"Debug: Question: '{question}' (lowercase: '{question_lower}')")
        
        for i, chunk in enumerate(self.vector_store.chunks):
            content_lower = chunk['content'].lower()
            print(f"Debug: Chunk {i}: '{chunk['content'][:100]}...'")
            
            # Simple keyword matching
            if question_lower in content_lower:
                print(f"Debug: Found exact match in chunk {i}")
                results.append({
                    'chunk': chunk,
                    'similarity': 1.0,  # High similarity for exact matches
                    'content': chunk['content'],
                    'source': chunk['source']
                })
            elif any(word in content_lower for word in question_lower.split()):
                print(f"Debug: Found partial match in chunk {i}")
                results.append({
                    'chunk': chunk,
                    'similarity': 0.5,  # Medium similarity for partial matches
                    'content': chunk['content'],
                    'source': chunk['source']
                })
        
        print(f"Debug: Found {len(results)} results")
        # Sort by similarity and return top_k results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def generate_response(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate a coherent answer using retrieved context and LLM"""
        if not retrieved_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Format the context from retrieved chunks
        context_parts = []
        for i, result in enumerate(retrieved_chunks, 1):
            chunk = result['chunk']
            context_parts.append(f"Source {i} ({chunk['source']}):\n{chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt for the LLM
        prompt = f"""Based on the following information, provide a comprehensive and accurate answer to the question.

Context Information:
{context}

Question: {question}

Instructions:
- Use only the information provided in the context
- If the context doesn't contain enough information to fully answer the question, say so
- Provide a clear, well-structured answer
- Cite the sources when possible

Answer:"""
        
        # Try to generate response using OpenAI
        try:
            response = self._generate_with_openai(prompt)
            return response
        except Exception as e:
            print(f"Error generating response with OpenAI: {e}")
            # Fallback to basic response formatting
            return self._format_basic_response(question, retrieved_chunks)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def _format_basic_response(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format a basic response when LLM is not available"""
        if not retrieved_chunks:
            return "No relevant information found."
        
        response = f"Based on the available information, here's what I found about '{question}':\n\n"
        
        for i, result in enumerate(retrieved_chunks, 1):
            chunk = result['chunk']
            response += f"{i}. From {chunk['source']}:\n{chunk['content']}\n\n"
        
        response += "Note: This is a basic summary. For a more coherent answer, please ensure your OpenAI API key is configured."
        return response