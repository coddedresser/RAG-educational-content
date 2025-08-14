# 🎓 Educational Content RAG with Learning Path Generation

A comprehensive Retrieval-Augmented Generation (RAG) system that creates personalized learning paths by retrieving and organizing educational content based on student level, learning style, and progress tracking.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [API Reference](#api-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Deployment](#deployment)
- [Contributing](#contributing)

## ✨ Features

### Core Functionality
- **🔍 Semantic Content Search**: Advanced RAG-based content retrieval using ChromaDB and sentence transformers
- **👤 Student Profiling**: Comprehensive student profiles with learning style assessment
- **🛤️ Personalized Learning Paths**: AI-generated learning sequences optimized for individual students
- **📊 Progress Tracking**: Real-time progress monitoring and analytics
- **🎯 Adaptive Learning**: Dynamic content difficulty adjustment based on performance
- **📈 Learning Analytics**: Detailed insights into learning patterns and performance

### Advanced Features
- **🧠 Learning Style Assessment**: Scientific questionnaire-based learning style identification
- **📐 Prerequisite Mapping**: Intelligent content sequencing based on dependencies
- **⚡ Adaptive Recommendations**: Real-time content suggestions based on performance
- **🏆 Gamification Elements**: Streaks, milestones, and achievement tracking
- **📱 Responsive UI**: Modern Streamlit interface with interactive components

## 🏗️ System Architecture

```
educational-rag-system/
├── app/
│   ├── main.py                 # Streamlit application
│   ├── config.py              # Configuration settings
│   └── components/
│       ├── data_processor.py   # Content processing & chunking
│       ├── embeddings.py       # Embedding generation
│       ├── retriever.py        # Vector search & retrieval
│       ├── learning_path.py    # Learning path generation
│       ├── student_profile.py  # Student profiling system
│       └── progress_tracker.py # Progress tracking & analytics
├── data/
│   ├── raw/                   # Raw educational content
│   ├── processed/             # Processed & chunked content
│   └── sample/                # Sample data for demo
├── models/
│   └── student_profiles/      # Stored student profiles
├── vector_db/                 # ChromaDB vector database
├── requirements.txt
└── README.md
```

### Technology Stack

- **Framework**: Streamlit for web interface
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Database**: SQLite for structured data
- **ML Libraries**: scikit-learn, numpy, pandas
- **Visualization**: Plotly, matplotlib
- **NLP**: NLTK, spacy for text processing

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- 2GB+ disk space for models and data

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/your-username/educational-rag-system.git
cd educational-rag-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional)
```bash
cp .env.example .env
# Edit .env with your API keys if needed
```

4. **Initialize the system**
```bash
python -m app.components.data_processor
python -m app.components.embeddings
```

5. **Run the application**
```bash
streamlit run app/main.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage

### Getting Started

1. **Create a Student Profile**
   - Navigate to the "Student Profile" tab
   - Fill in basic information (name, email)
   - Complete the learning style assessment
   - Select subjects of interest

2. **Search for Content**
   - Use the "Content Search" tab
   - Enter natural language queries like "explain basic algebra"
   - Apply filters for subject, difficulty, and content type
   - Browse personalized results

3. **Generate Learning Path**
   - Go to the "Learning Path" tab
   - Select target subject and difficulty progression
   - Define learning objectives
   - Review and start the generated path

4. **Track Progress**
   - Monitor learning activity in "Progress Dashboard"
   - View detailed analytics and performance metrics
   - Get personalized recommendations
   - Track learning streaks and achievements

### Sample Queries

The system supports natural language queries such as:
- "Explain basic algebra concepts for beginners"
- "Python programming tutorials for intermediate level"
- "Interactive math exercises with visual examples"
- "Advanced calculus problems with step-by-step solutions"

## 🧩 Components

### Data Processor (`data_processor.py`)
- **Content Chunking**: Semantic chunking with configurable size and overlap
- **Metadata Extraction**: Automatic extraction of learning objectives and concepts
- **Content Validation**: Ensures content quality and completeness
- **Format Support**: JSON, text, and structured educational content

### Embedding Generator (`embeddings.py`)
- **Model**: sentence-transformers/all-MiniLM-L6-v2 for semantic embeddings
- **Enhanced Context**: Combines content with metadata for better representations
- **Batch Processing**: Efficient processing of large content collections
- **Similarity Computing**: Cosine similarity for content matching

### Retriever (`retriever.py`)
- **Vector Search**: ChromaDB-powered semantic search
- **Hybrid Search**: Combines semantic and keyword-based retrieval
- **Personalization**: Student profile-aware result ranking
- **Filtering**: Advanced metadata filtering capabilities

### Student Profiler (`student_profile.py`)
- **Learning Style Assessment**: 5-question scientific assessment
- **Profile Management**: Comprehensive student data management
- **Progress Integration**: Seamless integration with learning analytics
- **Adaptive Profiling**: Dynamic profile updates based on behavior

### Learning Path Generator (`learning_path.py`)
- **Prerequisite Mapping**: Intelligent content dependency resolution
- **Personalized Sequencing**: Student-specific content ordering
- **Difficulty Progression**: Configurable learning curve optimization
- **Milestone Creation**: Automatic checkpoint and milestone generation

### Progress Tracker (`progress_tracker.py`)
- **Session Management**: Detailed learning session tracking
- **Performance Analytics**: Comprehensive progress metrics
- **Pattern Recognition**: Learning behavior analysis
- **Goal Setting**: Learning objective management and tracking

## 📊 Evaluation Metrics

### System Performance
- **Retrieval Accuracy**: Precision@K, Recall@K, NDCG@K
- **Response Relevance**: RAGAS evaluation metrics
- **System Latency**: Query response time < 2 seconds
- **Scalability**: Supports 1000+ content items, 100+ concurrent users

### Educational Effectiveness
- **Learning Outcome Achievement**: Objective completion rates
- **Time-to-Competency**: Efficiency of learning paths
- **Student Engagement**: Session duration and completion rates
- **Knowledge Retention**: Long-term performance tracking

### Sample Metrics
```python
# Retrieval Performance
precision_at_5 = 0.85
recall_at_5 = 0.78
avg_response_time = 1.2  # seconds

# Educational Effectiveness
avg_completion_rate = 0.82
learning_efficiency = 0.75  # objectives per hour
student_satisfaction = 4.3  # out of 5.0
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app/main.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure environment variables
4. Deploy with one click

### HuggingFace Spaces
1. Create new Space on [HuggingFace](https://huggingface.co/spaces)
2. Upload code and requirements.txt
3. Configure as Streamlit application
4. Deploy automatically

## 🔧 Configuration

### Environment Variables
```bash
# Optional: For enhanced features
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_hf_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///educational_rag.db

# System Settings
DEBUG=False
LOG_LEVEL=INFO
```

### System Configuration (`config.py`)
```python
# Vector Database Settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 102
SIMILARITY_THRESHOLD = 0.7

# Learning Path Settings
MAX_PATH_LENGTH = 20
DIFFICULTY_PROGRESSION = 0.1

# UI Settings
PAGE_TITLE = "🎓 Educational Learning Assistant"
```

## 📈 Performance Optimization

### Vector Database Optimization
- Index optimization for faster retrieval
- Batch processing for large datasets
- Memory-efficient embedding storage
- Query result caching

### Learning Path Optimization
- Prerequisite graph caching
- Incremental path updates
- Parallel path generation
- Smart content pre-loading

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/test_components.py -v
```

### Integration Tests
```bash
python -m pytest tests/test_integration.py -v
```

### Performance Tests
```bash
python scripts/performance_benchmark.py
```

## 📚 API Reference

### Core Classes

#### `EducationalRetriever`
```python
retriever = EducationalRetriever(persist_directory, collection_name)

# Semantic search
results = retriever.semantic_search(
    query="algebra basics",
    top_k=5,
    student_profile=profile_dict,
    filters={"subject": "Mathematics"}
)

# Hybrid search
results = retriever.hybrid_search(
    query="python programming",
    keywords=["variables", "functions"],
    top_k=10
)
```

#### `LearningPathGenerator`
```python
path_generator = LearningPathGenerator()
path_generator.build_prerequisite_graph(content_chunks)

# Generate personalized path
learning_path = path_generator.generate_learning_path(
    student_profile=student,
    target_subject="Mathematics",
    target_objectives=["algebra", "equations"],
    available_content=chunks
)
```

#### `StudentProfileManager`
```python
profile_manager = StudentProfileManager(db_path)

# Create profile
profile = profile_manager.create_student_profile(
    student_id="student_001",
    name="John Doe",
    email="john@example.com",
    assessment_responses=["A", "B", "C", "A", "D"]
)

# Update progress
profile_manager.update_student_progress(
    student_id="student_001",
    completed_content_id="math_001",
    performance_score=0.85,
    session_duration=25
)
```

## 🔍 Troubleshooting

### Common Issues

**Issue**: ChromaDB initialization fails
```bash
Solution: Ensure sufficient disk space and permissions
rm -rf vector_db/
python -m app.components.retriever
```

**Issue**: Embedding model download fails
```bash
Solution: Check internet connection and HuggingFace access
pip install --upgrade sentence-transformers
```

**Issue**: Streamlit app crashes on startup
```bash
Solution: Check Python version and dependencies
python --version  # Should be 3.8+
pip install --upgrade streamlit
```

### Performance Issues

**Slow search queries**:
- Reduce `top_k` parameter
- Enable result caching
- Optimize chunk size

**Memory usage**:
- Use smaller embedding models
- Implement batch processing
- Clear unused session state

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run tests and linting
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all functions
- Maintain test coverage > 80%

### Contribution Areas
- 🐛 Bug fixes and improvements
- ✨ New features and components
- 📚 Documentation and examples
- 🧪 Test coverage expansion
- 🚀 Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ChromaDB** for the excellent vector database
- **Sentence Transformers** for semantic embeddings
- **Streamlit** for the amazing web framework
- **HuggingFace** for model hosting and tools
- **NetworkX** for graph processing capabilities

## 📞 Support

- 📧 Email: support@educational-rag.com
- 💬 Discord: [Educational RAG Community](https://discord.gg/educational-rag)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/educational-rag-system/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/educational-rag-system/wiki)

## 🗺️ Roadmap

### Version 2.0 (Upcoming)
- [ ] Multi-modal content support (images, videos, audio)
- [ ] Advanced NLP with transformer models
- [ ] Real-time collaborative learning
- [ ] Mobile app development
- [ ] Advanced analytics dashboard

### Version 2.1
- [ ] Multi-language support
- [ ] Integration with LMS platforms
- [ ] Advanced assessment tools
- [ ] Peer learning features
- [ ] Content creation tools

### Long-term Goals
- [ ] AI-powered content generation
- [ ] Virtual tutoring assistant
- [ ] Blockchain-based credentialing
- [ ] VR/AR learning experiences
- [ ] Global content marketplace

---

Made with ❤️ by the Educational RAG Team

**⭐ Star this repository if you find it useful!**