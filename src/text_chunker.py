from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks"""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._split_single_document(doc)
            chunks.extend(doc_chunks)
        
        return chunks
    
    def _split_single_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a single document into chunks"""
        content = document['content']
        chunks = []
        
        start = 0
        while start < len(content):
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(content):
                # Find the last sentence boundary within the chunk
                last_period = content.rfind('.', start, end)
                last_exclamation = content.rfind('!', start, end)
                last_question = content.rfind('?', start, end)
                
                # Find the latest sentence boundary
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk = {
                    'content': chunk_content,
                    'source': document['source'],
                    'chunk_id': len(chunks),
                    'start_char': start,
                    'end_char': end,
                    'metadata': document.get('metadata', {})
                }
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks