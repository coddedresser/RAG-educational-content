"""
Vector-based retrieval system using FAISS for educational content
"""
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import pickle
from dataclasses import asdict

from .data_processor import ContentChunk
from .embeddings import EmbeddingGenerator, QueryProcessor

class EducationalRetriever:
    """Advanced retrieval system for educational content using FAISS"""
    
    def __init__(self, persist_directory: Path, collection_name: str = "educational_content"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.index = None
        self.chunks = []
        self.embeddings = []
        self.embedding_generator = EmbeddingGenerator()
        self.query_processor = QueryProcessor(self.embedding_generator)
        self.logger = logging.getLogger(__name__)
        # Add collection attribute for compatibility
        self.collection = self
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize FAISS index"""
        try:
            # Try to load existing index
            index_path = self.persist_directory / f"{self.collection_name}_index.faiss"
            chunks_path = self.persist_directory / f"{self.collection_name}_chunks.pkl"
            
            if index_path.exists() and chunks_path.exists():
                self.index = faiss.read_index(str(index_path))
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                self.embeddings = self.index.reconstruct_n(0, self.index.ntotal)
                self.logger.info(f"Loaded existing FAISS index with {len(self.chunks)} chunks")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(384)  # 384 for all-MiniLM-L6-v2
                self.logger.info(f"Created new FAISS index: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error initializing FAISS database: {e}")
            # Create basic index as fallback
            self.index = faiss.IndexFlatIP(384)
            self.logger.info("Created fallback FAISS index")
    
    # Add count method for compatibility
    def count(self):
        """Return number of chunks in the collection"""
        return len(self.chunks)
    
    # Add get method for compatibility
    def get(self, where=None, include=None):
        """Get chunks with optional filtering (simplified for FAISS)"""
        if where and 'content_id' in where:
            content_id = where['content_id']
            chunk = next((c for c in self.chunks if c.chunk_id == content_id), None)
            if chunk:
                return {
                    'documents': [chunk.chunk_text],
                    'metadatas': [chunk.__dict__],
                    'embeddings': [self.embeddings[self.chunks.index(chunk)]]
                }
        return {'documents': [], 'metadatas': [], 'embeddings': []}

    def add_chunks_to_database(self, chunks: List[ContentChunk], batch_size: int = 100):
        """Add content chunks to ChromaDB"""
        if not chunks:
            return
        
        self.logger.info(f"Adding {len(chunks)} chunks to database...")
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in batch:
                # Create enhanced text for embedding
                enhanced_text = self.embedding_generator.create_enhanced_text_for_embedding(chunk)
                
                # Generate embedding
                embedding = self.embedding_generator.generate_text_embedding(enhanced_text)
                
                ids.append(chunk.chunk_id)
                documents.append(chunk.chunk_text)
                embeddings.append(embedding.tolist())
                
                # Prepare metadata (ChromaDB requires simple types)
                metadata = {
                    'content_id': chunk.content_id,
                    'title': chunk.title,
                    'subject': chunk.subject,
                    'topic': chunk.topic,
                    'difficulty_level': chunk.difficulty_level,
                    'content_type': chunk.content_type,
                    'estimated_time': chunk.estimated_time,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'learning_objectives': json.dumps(chunk.learning_objectives),
                    'prerequisites': json.dumps(chunk.prerequisites),
                    'tags': json.dumps(chunk.tags),
                    'word_count': len(chunk.chunk_text.split())
                }
                metadatas.append(metadata)
            
            # Add to collection
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                self.logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            except Exception as e:
                self.logger.error(f"Error adding batch to database: {e}")
        
        self.logger.info(f"Successfully added {len(chunks)} chunks to database")
    
    def create_metadata_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create ChromaDB metadata filters"""
        chroma_filters = {}
        
        if 'subject' in filters and filters['subject']:
            chroma_filters['subject'] = {'$eq': filters['subject']}
        
        if 'difficulty_level' in filters and filters['difficulty_level']:
            chroma_filters['difficulty_level'] = {'$eq': filters['difficulty_level']}
        
        if 'content_type' in filters and filters['content_type']:
            chroma_filters['content_type'] = {'$eq': filters['content_type']}
        
        if 'min_time' in filters or 'max_time' in filters:
            time_filter = {}
            if 'min_time' in filters:
                time_filter['$gte'] = filters['min_time']
            if 'max_time' in filters:
                time_filter['$lte'] = filters['max_time']
            chroma_filters['estimated_time'] = time_filter
        
        return chroma_filters
    
    def semantic_search(self, 
                       query: str, 
                       top_k: int = 5,
                       student_profile: Optional[Dict] = None,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search on educational content
        """
        try:
            # Generate query embedding with student context
            query_embedding = self.query_processor.generate_query_embedding(
                query, student_profile
            )
            
            # Create metadata filters
            where_filters = {}
            if filters:
                where_filters = self.create_metadata_filters(filters)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, 50),  # Limit to prevent memory issues
                where=where_filters if where_filters else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            processed_results = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to similarity score (1 - distance for cosine distance)
                    similarity_score = 1 - distance
                    
                    # Parse JSON fields
                    try:
                        learning_objectives = json.loads(metadata.get('learning_objectives', '[]'))
                    except:
                        learning_objectives = []
                    
                    try:
                        prerequisites = json.loads(metadata.get('prerequisites', '[]'))
                    except:
                        prerequisites = []
                    
                    try:
                        tags = json.loads(metadata.get('tags', '[]'))
                    except:
                        tags = []
                    
                    result = {
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'learning_objectives': learning_objectives,
                        'prerequisites': prerequisites,
                        'tags': tags
                    }
                    processed_results.append(result)
            
            # Apply student-specific re-ranking
            if student_profile:
                processed_results = self._rerank_by_student_profile(processed_results, student_profile)
            
            return processed_results[:top_k]
        
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _rerank_by_student_profile(self, results: List[Dict], student_profile: Dict) -> List[Dict]:
        """Re-rank results based on student profile preferences"""
        student_level = student_profile.get('current_level', 'intermediate')
        learning_style = student_profile.get('learning_style', 'reading')
        subjects_of_interest = student_profile.get('subjects_of_interest', [])
        
        # Define level preferences (prefer current level, then adjacent levels)
        level_scores = {
            'beginner': {'beginner': 1.0, 'intermediate': 0.7, 'advanced': 0.3},
            'intermediate': {'beginner': 0.6, 'intermediate': 1.0, 'advanced': 0.8},
            'advanced': {'beginner': 0.4, 'intermediate': 0.8, 'advanced': 1.0}
        }
        
        # Content type preferences by learning style
        style_preferences = {
            'visual': {'video': 1.2, 'reading': 0.9, 'exercise': 1.0, 'quiz': 0.8},
            'auditory': {'video': 1.3, 'reading': 1.0, 'exercise': 0.9, 'quiz': 0.7},
            'kinesthetic': {'exercise': 1.3, 'project': 1.2, 'quiz': 1.0, 'reading': 0.8},
            'reading': {'reading': 1.2, 'lesson': 1.1, 'exercise': 0.9, 'video': 0.7}
        }
        
        # Apply re-ranking
        for result in results:
            metadata = result['metadata']
            base_score = result['similarity_score']
            
            # Level preference multiplier
            content_level = metadata.get('difficulty_level', 'intermediate')
            level_multiplier = level_scores.get(student_level, {}).get(content_level, 0.5)
            
            # Learning style multiplier
            content_type = metadata.get('content_type', 'lesson')
            style_multiplier = style_preferences.get(learning_style, {}).get(content_type, 1.0)
            
            # Subject interest multiplier
            content_subject = metadata.get('subject', '')
            subject_multiplier = 1.1 if content_subject in subjects_of_interest else 1.0
            
            # Calculate final score
            final_score = base_score * level_multiplier * style_multiplier * subject_multiplier
            result['final_score'] = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return results
    
    def hybrid_search(self, 
                     query: str,
                     keywords: List[str] = None,
                     top_k: int = 5,
                     student_profile: Optional[Dict] = None,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search
        """
        # Get semantic search results
        semantic_results = self.semantic_search(
            query, top_k * 2, student_profile, filters
        )
        
        # If keywords provided, perform keyword filtering
        if keywords:
            keyword_filtered = []
            for result in semantic_results:
                content_text = result['content'].lower()
                metadata_text = f"{result['metadata'].get('title', '')} {result['metadata'].get('subject', '')} {result['metadata'].get('topic', '')}".lower()
                full_text = f"{content_text} {metadata_text}"
                
                # Check if any keyword is present
                if any(keyword.lower() in full_text for keyword in keywords):
                    result['keyword_match'] = True
                    # Boost score for keyword matches
                    result['final_score'] = result.get('final_score', result['similarity_score']) * 1.2
                else:
                    result['keyword_match'] = False
                
                keyword_filtered.append(result)
            
            # Sort by final score again
            keyword_filtered.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            return keyword_filtered[:top_k]
        
        return semantic_results[:top_k]
    
    def get_related_content(self, content_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find content related to a specific content item"""
        try:
            # Get the original content
            results = self.collection.get(
                where={'content_id': content_id},
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if not results['documents']:
                return []
            
            # Use the first result as the base
            base_embedding = results['embeddings'][0]
            base_metadata = results['metadatas'][0]
            
            # Find similar content
            similar_results = self.collection.query(
                query_embeddings=[base_embedding],
                n_results=top_k + 5,  # Get extra to filter out the original
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process and filter results
            related_content = []
            if similar_results['documents'] and similar_results['documents'][0]:
                for doc, metadata, distance in zip(
                    similar_results['documents'][0],
                    similar_results['metadatas'][0],
                    similar_results['distances'][0]
                ):
                    # Skip the original content
                    if metadata.get('content_id') == content_id:
                        continue
                    
                    similarity_score = 1 - distance
                    related_content.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score
                    })
            
            return related_content[:top_k]
        
        except Exception as e:
            self.logger.error(f"Error finding related content: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get all metadata to analyze distribution
            all_results = self.collection.get(include=['metadatas'])
            metadatas = all_results['metadatas']
            
            # Analyze distribution
            subjects = {}
            difficulty_levels = {}
            content_types = {}
            
            for metadata in metadatas:
                # Subject distribution
                subject = metadata.get('subject', 'Unknown')
                subjects[subject] = subjects.get(subject, 0) + 1
                
                # Difficulty distribution
                difficulty = metadata.get('difficulty_level', 'Unknown')
                difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
                
                # Content type distribution
                content_type = metadata.get('content_type', 'Unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                'total_chunks': count,
                'subject_distribution': subjects,
                'difficulty_distribution': difficulty_levels,
                'content_type_distribution': content_types
            }
        
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the collection (use with caution!)"""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate it
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Educational content for RAG system"}
            )
            
            self.logger.info("Collection cleared successfully")
        
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")

# Example usage and testing
if __name__ == "__main__":
    from app.config import config
    from .data_processor import ContentProcessor
    import json
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create directories
    config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize retriever
    retriever = EducationalRetriever(config.VECTOR_DB_DIR)
    
    # Load processed chunks if available
    chunks_file = config.PROCESSED_DATA_DIR / "processed_chunks.json"
    
    if chunks_file.exists():
        print("Loading processed chunks...")
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        
        # Convert to ContentChunk objects
        chunks = []
        for chunk_data in chunks_data:
            chunk = ContentChunk(**chunk_data)
            chunks.append(chunk)
        
        # Add chunks to database if collection is empty
        if retriever.collection.count() == 0:
            print("Adding chunks to vector database...")
            retriever.add_chunks_to_database(chunks)
        
        # Test semantic search
        test_queries = [
            "explain basic algebra concepts",
            "python programming variables",
            "beginner mathematics",
            "programming fundamentals"
        ]
        
        print("\n" + "="*50)
        print("Testing Semantic Search")
        print("="*50)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.semantic_search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['metadata']['subject']}] {result['metadata']['title']}")
                print(f"     Similarity: {result['similarity_score']:.3f}")
                print(f"     Content: {result['content'][:100]}...")
        
        # Test with student profile
        sample_student_profile = {
            'current_level': 'beginner',
            'learning_style': 'visual',
            'subjects_of_interest': ['Mathematics', 'Programming']
        }
        
        print("\n" + "="*50)
        print("Testing Personalized Search")
        print("="*50)
        
        query = "learn programming basics"
        results = retriever.semantic_search(
            query, 
            top_k=3, 
            student_profile=sample_student_profile
        )
        
        print(f"\nPersonalized results for query: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['metadata']['subject']}] {result['metadata']['title']}")
            print(f"     Final Score: {result.get('final_score', result['similarity_score']):.3f}")
            print(f"     Difficulty: {result['metadata']['difficulty_level']}")
        
        # Show collection stats
        print("\n" + "="*50)
        print("Collection Statistics")
        print("="*50)
        
        stats = retriever.get_collection_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    else:
        print("No processed chunks found. Please run data_processor.py first.")