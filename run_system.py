#!/usr/bin/env python3
"""
Educational RAG System - Complete Setup and Demo Script
This script initializes the entire system and demonstrates all components
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import config, create_directories, SAMPLE_CONTENT
from app.components.data_processor import ContentProcessor, ContentChunk
from app.components.embeddings import EmbeddingGenerator
from app.components.retriever import EducationalRetriever
from app.components.student_profile import StudentProfileManager, StudentProfile
from app.components.learning_path import LearningPathGenerator, save_learning_path
from app.components.progress_tracker import ProgressTracker

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('educational_rag.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def initialize_system():
    """Initialize all system components"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Initializing Educational RAG System...")
    
    # Create necessary directories
    create_directories()
    logger.info("✅ Created directory structure")
    
    # Initialize components
    components = {}
    
    try:
        # Data Processor
        logger.info("📚 Initializing Content Processor...")
        components['processor'] = ContentProcessor()
        
        # Embedding Generator
        logger.info("🔢 Initializing Embedding Generator...")
        components['embedding_generator'] = EmbeddingGenerator()
        
        # Vector Retriever
        logger.info("🔍 Initializing Vector Retriever...")
        components['retriever'] = EducationalRetriever(
            config.VECTOR_DB_DIR,
            config.CHROMA_COLLECTION_NAME
        )
        
        # Student Profile Manager
        logger.info("👤 Initializing Student Profile Manager...")
        components['profile_manager'] = StudentProfileManager(
            config.MODELS_DIR / "student_profiles.db"
        )
        
        # Learning Path Generator
        logger.info("🛤️ Initializing Learning Path Generator...")
        components['path_generator'] = LearningPathGenerator()
        
        # Progress Tracker
        logger.info("📊 Initializing Progress Tracker...")
        components['progress_tracker'] = ProgressTracker(
            config.MODELS_DIR / "progress_tracking.db"
        )
        
        logger.info("✅ All components initialized successfully!")
        return components
    
    except Exception as e:
        logger.error(f"❌ Error initializing system: {e}")
        raise

def process_sample_content(components):
    """Process and load sample educational content"""
    logger = logging.getLogger(__name__)
    logger.info("📖 Processing sample educational content...")
    
    processor = components['processor']
    retriever = components['retriever']
    path_generator = components['path_generator']
    
    try:
        # Check if content already exists
        if retriever.collection.count() > 0:
            logger.info("📚 Content already exists in vector database")
            
            # Load existing content for path generation
            chunks = []
            for content_item in SAMPLE_CONTENT:
                item_chunks = processor.chunk_content(content_item)
                chunks.extend(item_chunks)
            
            path_generator.build_prerequisite_graph(chunks)
            return chunks
        
        # Process sample content
        chunks = []
        for i, content_item in enumerate(SAMPLE_CONTENT, 1):
            logger.info(f"Processing content item {i}/{len(SAMPLE_CONTENT)}: {content_item['title']}")
            item_chunks = processor.chunk_content(content_item)
            chunks.extend(item_chunks)
        
        logger.info(f"✅ Created {len(chunks)} content chunks from {len(SAMPLE_CONTENT)} items")
        
        # Add to vector database
        logger.info("💾 Adding content to vector database...")
        retriever.add_chunks_to_database(chunks, batch_size=10)
        
        # Build prerequisite graph
        logger.info("🔗 Building prerequisite graph...")
        path_generator.build_prerequisite_graph(chunks)
        
        logger.info("✅ Sample content processed and loaded successfully!")
        return chunks
    
    except Exception as e:
        logger.error(f"❌ Error processing content: {e}")
        raise

def create_demo_student(components):
    """Create a demo student profile"""
    logger = logging.getLogger(__name__)
    logger.info("👤 Creating demo student profile...")
    
    profile_manager = components['profile_manager']
    
    try:
        # Create student with learning style assessment
        assessment_responses = ["A", "C", "A", "D", "B"]  # Mixed learning style
        
        student_profile = profile_manager.create_student_profile(
            student_id="demo_student_001",
            name="Alex Demo Student",
            email="alex.demo@educational-rag.com",
            assessment_responses=assessment_responses
        )
        
        # Update profile with preferences
        student_profile.current_level = "intermediate"
        student_profile.subjects_of_interest = ["Mathematics", "Programming"]
        student_profile.learning_pace = "medium"
        
        profile_manager._save_profile_to_db(student_profile)
        
        logger.info(f"✅ Created demo student: {student_profile.name}")
        logger.info(f"   Learning Style: {student_profile.learning_style}")
        logger.info(f"   Level: {student_profile.current_level}")
        logger.info(f"   Interests: {', '.join(student_profile.subjects_of_interest)}")
        
        return student_profile
    
    except Exception as e:
        logger.error(f"❌ Error creating demo student: {e}")
        raise

def demonstrate_search(components, student_profile):
    """Demonstrate content search functionality"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Demonstrating content search...")
    
    retriever = components['retriever']
    
    test_queries = [
        "explain basic algebra concepts",
        "python programming variables",
        "introduction to mathematics for beginners",
        "programming fundamentals with examples"
    ]
    
    try:
        for query in test_queries:
            logger.info(f"🔎 Searching: '{query}'")
            
            # Perform semantic search
            results = retriever.semantic_search(
                query=query,
                top_k=3,
                student_profile=student_profile.__dict__,
                filters={"difficulty_level": student_profile.current_level}
            )
            
            logger.info(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                score = result['similarity_score']
                logger.info(f"   {i}. [{metadata['subject']}] {metadata['title']} (Score: {score:.3f})")
            
            time.sleep(1)  # Brief pause between queries
        
        logger.info("✅ Search demonstration completed!")
    
    except Exception as e:
        logger.error(f"❌ Error demonstrating search: {e}")
        raise

def demonstrate_learning_path(components, student_profile, content_chunks):
    """Demonstrate learning path generation"""
    logger = logging.getLogger(__name__)
    logger.info("🛤️ Demonstrating learning path generation...")
    
    path_generator = components['path_generator']
    
    try:
        # Generate learning path for Mathematics
        logger.info("📊 Generating Mathematics learning path...")
        math_path = path_generator.generate_learning_path(
            student_profile=student_profile,
            target_subject="Mathematics",
            target_objectives=["Understanding variables", "Basic algebraic expressions", "Simple equations"],
            available_content=content_chunks,
            max_path_length=10
        )
        
        logger.info(f"✅ Generated Mathematics path with {len(math_path.path_items)} items:")
        logger.info(f"   Total estimated time: {math_path.estimated_total_time} minutes")
        logger.info(f"   Difficulty progression: {math_path.difficulty_progression}")
        
        for i, item in enumerate(math_path.path_items[:5], 1):  # Show first 5 items
            checkpoint = "🎯" if item.is_checkpoint else ""
            logger.info(f"   {i}. {checkpoint}[{item.difficulty_level}] {item.title} ({item.estimated_time}min)")
        
        if len(math_path.path_items) > 5:
            logger.info(f"   ... and {len(math_path.path_items) - 5} more items")
        
        # Save the path
        path_file = config.MODELS_DIR / f"demo_path_{math_path.path_id}.json"
        save_learning_path(math_path, str(path_file))
        logger.info(f"💾 Saved learning path to: {path_file}")
        
        # Generate Programming path
        logger.info("💻 Generating Programming learning path...")
        prog_path = path_generator.generate_learning_path(
            student_profile=student_profile,
            target_subject="Programming",
            target_objectives=["Understanding variables", "Data types", "Basic programming concepts"],
            available_content=content_chunks,
            max_path_length=8
        )
        
        logger.info(f"✅ Generated Programming path with {len(prog_path.path_items)} items:")
        for i, item in enumerate(prog_path.path_items[:3], 1):  # Show first 3 items
            checkpoint = "🎯" if item.is_checkpoint else ""
            logger.info(f"   {i}. {checkpoint}[{item.difficulty_level}] {item.title} ({item.estimated_time}min)")
        
        return math_path, prog_path
    
    except Exception as e:
        logger.error(f"❌ Error demonstrating learning path: {e}")
        raise

def demonstrate_progress_tracking(components, student_profile, learning_path):
    """Demonstrate progress tracking"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Demonstrating progress tracking...")
    
    progress_tracker = components['progress_tracker']
    profile_manager = components['profile_manager']
    
    try:
        # Simulate learning sessions
        for i, path_item in enumerate(learning_path.path_items[:3]):  # Simulate first 3 items
            logger.info(f"🎓 Simulating learning session for: {path_item.title}")
            
            # Start session
            session_id = progress_tracker.start_learning_session(
                student_id=student_profile.student_id,
                content_id=path_item.content_id,
                subject=path_item.subject,
                difficulty_level=path_item.difficulty_level,
                content_type=path_item.content_type
            )
            
            # Simulate some study time
            time.sleep(0.5)
            
            # End session with performance score
            performance_score = 0.75 + (i * 0.1)  # Improving performance
            success = progress_tracker.end_learning_session(
                session_id=session_id,
                completion_status="completed",
                performance_score=performance_score,
                interactions={
                    "questions_answered": 5 + i,
                    "hints_used": 3 - i,
                    "time_spent_seconds": (15 + i * 5) * 60
                }
            )
            
            if success:
                logger.info(f"   ✅ Completed with score: {performance_score:.2f}")
                
                # Update student profile
                profile_manager.update_student_progress(
                    student_id=student_profile.student_id,
                    completed_content_id=path_item.content_id,
                    performance_score=performance_score,
                    session_duration=path_item.estimated_time
                )
        
        # Get progress analytics
        logger.info("📈 Generating progress analytics...")
        analytics = progress_tracker.get_learning_analytics(
            student_id=student_profile.student_id,
            time_range_days=7
        )
        
        logger.info("✅ Progress Analytics Summary:")
        logger.info(f"   📚 Total Sessions: {analytics.get('total_sessions', 0)}")
        logger.info(f"   ⏱️ Total Study Time: {analytics.get('total_study_time', 0)} minutes")
        logger.info(f"   📊 Average Performance: {analytics.get('average_performance', 0):.2f}")
        logger.info(f"   ✅ Completion Rate: {analytics.get('completion_rate', 0):.1f}%")
        
        # Show recommendations
        recommendations = analytics.get('recommendations', [])
        if recommendations:
            logger.info("💡 Personalized Recommendations:")
            for rec in recommendations[:3]:
                logger.info(f"   • {rec}")
        
        logger.info("✅ Progress tracking demonstration completed!")
    
    except Exception as e:
        logger.error(f"❌ Error demonstrating progress tracking: {e}")
        raise

def show_system_statistics(components):
    """Display system statistics"""
    logger = logging.getLogger(__name__)
    logger.info("📈 System Statistics Summary:")
    
    retriever = components['retriever']
    
    try:
        # Vector database stats
        stats = retriever.get_collection_stats()
        logger.info(f"   📚 Total Content Chunks: {stats.get('total_chunks', 0)}")
        
        # Subject distribution
        subject_dist = stats.get('subject_distribution', {})
        logger.info("   📊 Content by Subject:")
        for subject, count in subject_dist.items():
            logger.info(f"      • {subject}: {count} chunks")
        
        # Difficulty distribution
        difficulty_dist = stats.get('difficulty_distribution', {})
        logger.info("   📈 Content by Difficulty:")
        for difficulty, count in difficulty_dist.items():
            logger.info(f"      • {difficulty}: {count} chunks")
        
        # Content type distribution
        type_dist = stats.get('content_type_distribution', {})
        logger.info("   📋 Content by Type:")
        for content_type, count in type_dist.items():
            logger.info(f"      • {content_type}: {count} chunks")
        
        logger.info("✅ System statistics displayed!")
    
    except Exception as e:
        logger.error(f"❌ Error showing system statistics: {e}")

def main():
    """Main demonstration function"""
    print("=" * 70)
    print("🎓 EDUCATIONAL RAG SYSTEM - COMPLETE DEMO")
    print("=" * 70)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize system
        components = initialize_system()
        
        # Process content
        content_chunks = process_sample_content(components)
        
        # Create demo student
        student_profile = create_demo_student(components)
        
        # Demonstrate search
        demonstrate_search(components, student_profile)
        
        # Demonstrate learning path generation
        math_path, prog_path = demonstrate_learning_path(
            components, student_profile, content_chunks
        )
        
        # Demonstrate progress tracking
        demonstrate_progress_tracking(components, student_profile, math_path)
        
        # Show system statistics
        show_system_statistics(components)
        
        print("\n" + "=" * 70)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n🚀 To start the web interface, run:")
        print("   streamlit run app/main.py")
        print("\n📊 Check the logs in: educational_rag.log")
        print("💾 Data stored in: vector_db/ and models/")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()