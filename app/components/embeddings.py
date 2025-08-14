"""
Embedding generation and management for educational content
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import pickle
import json
from dataclasses import asdict
import hashlib

from .data_processor import ContentChunk

class EmbeddingGenerator:
    """Generate and manage embeddings for educational content"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            if not text or not text.strip():
                return np.zeros(self.model.get_sentence_embedding_dimension())
            
            embedding = self.model.encode(text.strip(), convert_to_numpy=True)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding for text: {e}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batches"""
        try:
            if not texts:
                return []
            
            # Filter out empty texts
            valid_texts = [text if text and text.strip() else "" for text in texts]
            
            embeddings = self.model.encode(
                valid_texts, 
                batch_size=batch_size, 
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            return [emb for emb in embeddings]
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]
    
    def create_enhanced_text_for_embedding(self, chunk: ContentChunk) -> str:
        """
        Create enhanced text representation for better embeddings
        Combines content with metadata for context-aware embeddings
        """
        enhanced_parts = []
        
        # Add title and subject context
        if chunk.title:
            enhanced_parts.append(f"Title: {chunk.title}")
        
        if chunk.subject and chunk.topic:
            enhanced_parts.append(f"Subject: {chunk.subject}, Topic: {chunk.topic}")
        
        # Add learning objectives
        if chunk.learning_objectives:
            objectives_text = ", ".join(chunk.learning_objectives)
            enhanced_parts.append(f"Learning objectives: {objectives_text}")
        
        # Add difficulty level
        enhanced_parts.append(f"Difficulty: {chunk.difficulty_level}")
        
        # Add main content
        enhanced_parts.append(chunk.chunk_text)
        
        # Add important tags
        if chunk.tags:
            tags_text = ", ".join(chunk.tags[:5])  # Limit to top 5 tags
            enhanced_parts.append(f"Key concepts: {tags_text}")
        
        return " | ".join(enhanced_parts)
    
    def generate_chunk_embeddings(self, chunks: List[ContentChunk]) -> Dict[str, np.ndarray]:
        """Generate embeddings for content chunks"""
        if not chunks:
            return {}
        
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Create enhanced texts for embedding
        enhanced_texts = [self.create_enhanced_text_for_embedding(chunk) for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_batch_embeddings(enhanced_texts)
        
        # Create mapping from chunk_id to embedding
        chunk_embeddings = {}
        for chunk, embedding in zip(chunks, embeddings):
            chunk_embeddings[chunk.chunk_id] = embedding
        
        self.logger.info(f"Generated embeddings for {len(chunk_embeddings)} chunks")
        return chunk_embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_similar_chunks(self, query_embedding: np.ndarray, 
                           chunk_embeddings: Dict[str, np.ndarray],
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar chunks to a query embedding"""
        similarities = []
        
        for chunk_id, chunk_embedding in chunk_embeddings.items():
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], file_path: Path):
        """Save embeddings to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            embeddings_serializable = {
                chunk_id: embedding.tolist() 
                for chunk_id, embedding in embeddings.items()
            }
            
            with open(file_path, 'w') as f:
                json.dump(embeddings_serializable, f, indent=2)
            
            self.logger.info(f"Saved embeddings for {len(embeddings)} chunks to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load embeddings from file"""
        try:
            with open(file_path, 'r') as f:
                embeddings_data = json.load(f)
            
            # Convert lists back to numpy arrays
            embeddings = {
                chunk_id: np.array(embedding) 
                for chunk_id, embedding in embeddings_data.items()
            }
            
            self.logger.info(f"Loaded embeddings for {len(embeddings)} chunks from {file_path}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            return {}

class QueryProcessor:
    """Process user queries for semantic search"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess user query for better search"""
        # Basic cleaning
        query = query.strip()
        
        # Add context hints for educational content
        educational_keywords = [
            "learn", "understand", "explain", "teach", "how", "what", "why",
            "concept", "example", "practice", "exercise"
        ]
        
        query_lower = query.lower()
        has_educational_context = any(keyword in query_lower for keyword in educational_keywords)
        
        if not has_educational_context and len(query.split()) < 3:
            # Add learning context for short queries
            query = f"I want to learn about {query}"
        
        return query
    
    def create_contextual_query(self, query: str, student_profile: Optional[Dict] = None) -> str:
        """Create contextual query based on student profile"""
        enhanced_query = self.preprocess_query(query)
        
        if student_profile:
            # Add difficulty level context
            difficulty = student_profile.get('current_level', 'intermediate')
            enhanced_query += f" at {difficulty} level"
            
            # Add learning style context
            learning_style = student_profile.get('learning_style')
            if learning_style == 'visual':
                enhanced_query += " with visual examples and diagrams"
            elif learning_style == 'practical':
                enhanced_query += " with hands-on examples and exercises"
            elif learning_style == 'reading':
                enhanced_query += " with detailed explanations and text"
        
        return enhanced_query
    
    def generate_query_embedding(self, query: str, student_profile: Optional[Dict] = None) -> np.ndarray:
        """Generate embedding for user query with student context"""
        contextual_query = self.create_contextual_query(query, student_profile)
        return self.embedding_generator.generate_text_embedding(contextual_query)
    
    def expand_query_with_synonyms(self, query: str) -> List[str]:
        """Expand query with educational synonyms and related terms"""
        # Simple synonym expansion - in production, use a more sophisticated approach
        synonym_map = {
            "learn": ["study", "understand", "grasp", "master"],
            "explain": ["describe", "clarify", "illustrate", "demonstrate"],
            "solve": ["resolve", "work out", "figure out", "calculate"],
            "understand": ["comprehend", "grasp", "learn", "know"],
            "example": ["instance", "case", "illustration", "sample"],
            "practice": ["exercise", "drill", "rehearse", "train"]
        }
        
        expanded_queries = [query]
        words = query.lower().split()
        
        for word in words:
            if word in synonym_map:
                for synonym in synonym_map[word][:2]:  # Limit to 2 synonyms
                    expanded_query = query.lower().replace(word, synonym)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries[:3]  # Limit to 3 variations

# Utility functions for embedding analysis
def analyze_embedding_quality(embeddings: Dict[str, np.ndarray], chunks: List[ContentChunk]) -> Dict[str, Any]:
    """Analyze the quality and distribution of embeddings"""
    if not embeddings or not chunks:
        return {}
    
    embedding_values = list(embeddings.values())
    dimensions = len(embedding_values[0]) if embedding_values else 0
    
    # Calculate statistics
    all_values = np.concatenate(embedding_values)
    
    analysis = {
        'total_embeddings': len(embeddings),
        'embedding_dimensions': dimensions,
        'mean_value': float(np.mean(all_values)),
        'std_value': float(np.std(all_values)),
        'min_value': float(np.min(all_values)),
        'max_value': float(np.max(all_values)),
    }
    
    # Analyze by subject
    subject_embeddings = {}
    chunk_dict = {chunk.chunk_id: chunk for chunk in chunks}
    
    for chunk_id, embedding in embeddings.items():
        if chunk_id in chunk_dict:
            subject = chunk_dict[chunk_id].subject
            if subject not in subject_embeddings:
                subject_embeddings[subject] = []
            subject_embeddings[subject].append(embedding)
    
    # Calculate average similarity within subjects
    subject_coherence = {}
    for subject, embs in subject_embeddings.items():
        if len(embs) > 1:
            similarities = []
            for i in range(len(embs)):
                for j in range(i+1, len(embs)):
                    sim = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]))
                    similarities.append(sim)
            subject_coherence[subject] = float(np.mean(similarities)) if similarities else 0.0
    
    analysis['subject_coherence'] = subject_coherence
    
    return analysis

# Example usage
if __name__ == "__main__":
    from app.config import config
    from .data_processor import ContentProcessor
    import json
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load processed chunks
    chunks_file = config.PROCESSED_DATA_DIR / "processed_chunks.json"
    if chunks_file.exists():
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        
        # Convert to ContentChunk objects
        chunks = []
        for chunk_data in chunks_data:
            chunk = ContentChunk(**chunk_data)
            chunks.append(chunk)
        
        # Generate embeddings
        embedding_gen = EmbeddingGenerator()
        embeddings = embedding_gen.generate_chunk_embeddings(chunks)
        
        # Save embeddings
        embeddings_file = config.PROCESSED_DATA_DIR / "chunk_embeddings.json"
        embedding_gen.save_embeddings(embeddings, embeddings_file)
        
        # Analyze embedding quality
        analysis = analyze_embedding_quality(embeddings, chunks)
        print("\nEmbedding Analysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        # Test query processing
        query_processor = QueryProcessor(embedding_gen)
        test_query = "explain algebra basics"
        query_embedding = query_processor.generate_query_embedding(test_query)
        
        # Find similar chunks
        similar_chunks = embedding_gen.find_similar_chunks(query_embedding, embeddings, top_k=3)
        print(f"\nTop 3 similar chunks for query '{test_query}':")
        for chunk_id, similarity in similar_chunks:
            print(f"  {chunk_id}: {similarity:.3f}")
    
    else:
        print("Please run data_processor.py first to generate processed chunks.")