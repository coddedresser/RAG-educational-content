"""
Educational content processing and chunking module
"""
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import hashlib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class ContentChunk:
    """Represents a chunk of educational content"""
    chunk_id: str
    content_id: str
    title: str
    subject: str
    topic: str
    difficulty_level: str
    learning_objectives: List[str]
    prerequisites: List[str]
    content_type: str
    estimated_time: int
    tags: List[str]
    chunk_text: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]

class ContentProcessor:
    """Process and chunk educational content for RAG system"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 102):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
    def load_content_from_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load educational content from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            self.logger.info(f"Loaded {len(content)} content items from {file_path}")
            return content
        except Exception as e:
            self.logger.error(f"Error loading content from {file_path}: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\"\'`]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def extract_key_concepts(self, text: str, max_concepts: int = 10) -> List[str]:
        """Extract key concepts from text using TF-IDF"""
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(
                max_features=max_concepts,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = tfidf_matrix.mean(axis=0).A1
            concept_scores = list(zip(feature_names, mean_scores))
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [concept[0] for concept in concept_scores[:max_concepts]]
        
        except Exception as e:
            self.logger.warning(f"Error extracting concepts: {e}")
            return []
    
    def semantic_chunking(self, text: str, content_id: str) -> List[str]:
        """
        Create semantic chunks based on sentence boundaries and content structure
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, start new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk if needed
                overlap_text = ""
                if self.chunk_overlap > 0:
                    overlap_words = current_chunk.split()[-self.chunk_overlap:]
                    overlap_text = " ".join(overlap_words) + " "
                
                current_chunk = overlap_text + sentence + " "
                current_length = len(current_chunk.split())
            else:
                current_chunk += sentence + " "
                current_length += sentence_length
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_chunk_id(self, content_id: str, chunk_index: int, chunk_text: str) -> str:
        """Generate unique chunk ID"""
        text_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        return f"{content_id}_chunk_{chunk_index}_{text_hash}"
    
    def chunk_content(self, content_item: Dict[str, Any]) -> List[ContentChunk]:
        """
        Chunk a single content item into smaller pieces
        """
        content_text = content_item.get('content', '')
        if not content_text:
            return []
        
        # Preprocess the content
        processed_text = self.preprocess_text(content_text)
        
        # Create semantic chunks
        text_chunks = self.semantic_chunking(processed_text, content_item['content_id'])
        
        if not text_chunks:
            return []
        
        # Create ContentChunk objects
        content_chunks = []
        total_chunks = len(text_chunks)
        
        for i, chunk_text in enumerate(text_chunks):
            # Extract key concepts from this chunk
            chunk_concepts = self.extract_key_concepts(chunk_text, max_concepts=5)
            
            # Enhance tags with extracted concepts
            enhanced_tags = list(content_item.get('tags', []))
            enhanced_tags.extend(chunk_concepts)
            enhanced_tags = list(set(enhanced_tags))  # Remove duplicates
            
            chunk = ContentChunk(
                chunk_id=self.generate_chunk_id(content_item['content_id'], i, chunk_text),
                content_id=content_item['content_id'],
                title=content_item.get('title', ''),
                subject=content_item.get('subject', ''),
                topic=content_item.get('topic', ''),
                difficulty_level=content_item.get('difficulty_level', 'intermediate'),
                learning_objectives=content_item.get('learning_objectives', []),
                prerequisites=content_item.get('prerequisites', []),
                content_type=content_item.get('content_type', 'lesson'),
                estimated_time=content_item.get('estimated_time', 15),
                tags=enhanced_tags,
                chunk_text=chunk_text,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata={
                    **content_item.get('metadata', {}),
                    'chunk_word_count': len(chunk_text.split()),
                    'chunk_concepts': chunk_concepts
                }
            )
            content_chunks.append(chunk)
        
        return content_chunks
    
    def process_content_batch(self, content_items: List[Dict[str, Any]]) -> List[ContentChunk]:
        """
        Process multiple content items and return all chunks
        """
        all_chunks = []
        
        for content_item in content_items:
            try:
                chunks = self.chunk_content(content_item)
                all_chunks.extend(chunks)
                self.logger.info(f"Processed {content_item['content_id']}: {len(chunks)} chunks")
            except Exception as e:
                self.logger.error(f"Error processing {content_item.get('content_id', 'unknown')}: {e}")
        
        self.logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
    
    def save_chunks_to_json(self, chunks: List[ContentChunk], output_path: Path):
        """Save processed chunks to JSON file"""
        try:
            chunks_data = []
            for chunk in chunks:
                chunk_dict = {
                    'chunk_id': chunk.chunk_id,
                    'content_id': chunk.content_id,
                    'title': chunk.title,
                    'subject': chunk.subject,
                    'topic': chunk.topic,
                    'difficulty_level': chunk.difficulty_level,
                    'learning_objectives': chunk.learning_objectives,
                    'prerequisites': chunk.prerequisites,
                    'content_type': chunk.content_type,
                    'estimated_time': chunk.estimated_time,
                    'tags': chunk.tags,
                    'chunk_text': chunk.chunk_text,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'metadata': chunk.metadata
                }
                chunks_data.append(chunk_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving chunks to {output_path}: {e}")
    
    def create_content_summary(self, chunks: List[ContentChunk]) -> Dict[str, Any]:
        """Create summary statistics of processed content"""
        if not chunks:
            return {}
        
        summary = {
            'total_chunks': len(chunks),
            'total_content_items': len(set(chunk.content_id for chunk in chunks)),
            'subjects': list(set(chunk.subject for chunk in chunks)),
            'difficulty_levels': list(set(chunk.difficulty_level for chunk in chunks)),
            'content_types': list(set(chunk.content_type for chunk in chunks)),
            'avg_chunk_length': sum(len(chunk.chunk_text.split()) for chunk in chunks) / len(chunks),
            'total_estimated_time': sum(chunk.estimated_time for chunk in chunks),
            'topics_count': len(set(chunk.topic for chunk in chunks)),
        }
        
        # Subject distribution
        subject_counts = {}
        for chunk in chunks:
            subject_counts[chunk.subject] = subject_counts.get(chunk.subject, 0) + 1
        summary['subject_distribution'] = subject_counts
        
        # Difficulty distribution
        difficulty_counts = {}
        for chunk in chunks:
            difficulty_counts[chunk.difficulty_level] = difficulty_counts.get(chunk.difficulty_level, 0) + 1
        summary['difficulty_distribution'] = difficulty_counts
        
        return summary

def create_sample_content_file(output_path: Path, sample_content: List[Dict[str, Any]]):
    """Create sample content file for testing"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_content, f, indent=2, ensure_ascii=False)
        print(f"Sample content saved to {output_path}")
    except Exception as e:
        print(f"Error creating sample content file: {e}")

# Example usage
if __name__ == "__main__":
    from app.config import config, SAMPLE_CONTENT
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create directories
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create sample content file
    sample_file = config.RAW_DATA_DIR / "sample_content.json"
    create_sample_content_file(sample_file, SAMPLE_CONTENT)
    
    # Initialize processor
    processor = ContentProcessor()
    
    # Load and process content
    content_items = processor.load_content_from_json(sample_file)
    chunks = processor.process_content_batch(content_items)
    
    # Save processed chunks
    output_file = config.PROCESSED_DATA_DIR / "processed_chunks.json"
    processor.save_chunks_to_json(chunks, output_file)
    
    # Create summary
    summary = processor.create_content_summary(chunks)
    print("\nContent Processing Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")