"""
Configuration settings for the Educational RAG System
"""
import os
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings
class Config(BaseSettings):
    """Application configuration"""
    
    # Application Settings
    APP_NAME: str = "Educational RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Keys (from environment variables)
    OPENAI_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Vector Database Settings
    CHROMA_COLLECTION_NAME: str = "educational_content"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_PERSIST: bool = True
    
    # Chunking Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 102  # 20% overlap
    MIN_CHUNK_SIZE: int = 100
    MAX_CHUNK_SIZE: int = 1000
    
    # Retrieval Settings
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Learning Path Settings
    MAX_PATH_LENGTH: int = 50
    MIN_PATH_LENGTH: int = 3
    DEFAULT_DIFFICULTY_PROGRESSION: float = 0.1  # 10% increase per step
    
    # Student Profile Settings
    LEARNING_STYLES: List[str] = ["visual", "auditory", "kinesthetic", "reading"]
    DIFFICULTY_LEVELS: List[str] = ["beginner", "intermediate", "advanced"]
    LEARNING_PACES: List[str] = ["slow", "medium", "fast"]
    
    # Content Categories
    SUBJECTS: List[str] = [
        "Mathematics", "Science", "History", "Literature", "Programming",
        "Physics", "Chemistry", "Biology", "Geography", "Art"
    ]
    
    CONTENT_TYPES: List[str] = [
        "lesson", "exercise", "quiz", "video", "reading", "project"
    ]
    
    # Progress Tracking
    PROGRESS_DB_NAME: str = "student_progress.db"
    MASTERY_THRESHOLD: float = 0.8  # 80% to consider mastered
    
    # UI Settings
    PAGE_TITLE: str = "🎓 Educational Learning Assistant"
    SIDEBAR_TITLE: str = "Navigation"
    
    # Assessment Settings
    ASSESSMENT_QUESTIONS_PER_TOPIC: int = 5
    PASSING_SCORE: float = 0.7
    
    # Time Estimates (in minutes)
    CONTENT_TIME_ESTIMATES: Dict[str, int] = {
        "lesson": 15,
        "exercise": 10,
        "quiz": 5,
        "video": 20,
        "reading": 12,
        "project": 60
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global configuration instance
config = Config()

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.VECTOR_DB_DIR,
        config.MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Learning objectives mapping for different subjects
LEARNING_OBJECTIVES_MAP = {
    "Mathematics": {
        "beginner": [
            "Basic arithmetic operations",
            "Number patterns and sequences",
            "Simple geometry shapes",
            "Basic fractions and decimals"
        ],
        "intermediate": [
            "Algebraic expressions and equations",
            "Coordinate geometry",
            "Probability and statistics basics",
            "Advanced fractions and percentages"
        ],
        "advanced": [
            "Calculus fundamentals",
            "Advanced statistics",
            "Complex number systems",
            "Advanced geometry and trigonometry"
        ]
    },
    "Science": {
        "beginner": [
            "Scientific method basics",
            "States of matter",
            "Simple machines",
            "Weather and climate"
        ],
        "intermediate": [
            "Chemical reactions",
            "Energy and motion",
            "Cell biology",
            "Earth's systems"
        ],
        "advanced": [
            "Molecular biology",
            "Thermodynamics",
            "Quantum physics basics",
            "Environmental science"
        ]
    },
    "Programming": {
        "beginner": [
            "Programming fundamentals",
            "Variables and data types",
            "Control structures",
            "Basic input/output"
        ],
        "intermediate": [
            "Object-oriented programming",
            "Data structures",
            "Algorithm design",
            "Database basics"
        ],
        "advanced": [
            "Advanced algorithms",
            "System design",
            "Machine learning basics",
            "Software architecture"
        ]
    }
}

# Prerequisites mapping
PREREQUISITES_MAP = {
    "Advanced algebra": ["Basic algebra", "Arithmetic operations"],
    "Calculus": ["Advanced algebra", "Functions and graphs"],
    "Object-oriented programming": ["Programming fundamentals", "Control structures"],
    "Machine learning": ["Programming fundamentals", "Statistics", "Linear algebra"],
    "Chemical reactions": ["Atomic structure", "Elements and compounds"],
    "Cell biology": ["Basic biology", "Scientific method"],
}

# Sample educational content for demonstration
SAMPLE_CONTENT = [
    {
        "content_id": "math_001",
        "title": "Introduction to Algebra",
        "subject": "Mathematics",
        "topic": "Basic Algebra",
        "difficulty_level": "beginner",
        "learning_objectives": ["Understanding variables", "Basic algebraic expressions"],
        "prerequisites": ["Basic arithmetic"],
        "content_type": "lesson",
        "estimated_time": 15,
        "tags": ["algebra", "variables", "expressions"],
        "content": """
        Algebra is a branch of mathematics that uses letters and symbols to represent numbers and quantities in formulas and equations. 
        
        A variable is a letter or symbol that represents an unknown value. For example, in the equation x + 3 = 7, 'x' is a variable.
        
        An algebraic expression is a combination of variables, numbers, and operations. Examples include:
        - 2x + 5 (linear expression)
        - x² + 3x - 2 (quadratic expression)
        
        Key concepts:
        1. Variables can represent any number
        2. We can perform operations on variables just like numbers
        3. The goal is often to find the value of the variable
        """,
        "metadata": {"author": "Dr. Smith", "last_updated": "2024-01-15"}
    },
    {
        "content_id": "prog_001",
        "title": "Variables in Python",
        "subject": "Programming",
        "topic": "Python Basics",
        "difficulty_level": "beginner",
        "learning_objectives": ["Understanding variables", "Variable assignment", "Data types"],
        "prerequisites": ["Basic computer literacy"],
        "content_type": "lesson",
        "estimated_time": 20,
        "tags": ["python", "variables", "programming"],
        "content": """
        Variables in Python are containers that store data values. Unlike other programming languages, Python has no command for declaring a variable.
        
        Creating Variables:
        ```python
        x = 5
        y = "Hello"
        z = 3.14
        ```
        
        Variable Names:
        - Must start with a letter or underscore
        - Can contain letters, numbers, and underscores
        - Case-sensitive (age and Age are different)
        
        Data Types:
        - int: Integer numbers (5, -3, 100)
        - float: Decimal numbers (3.14, -2.5)
        - str: Text strings ("Hello", 'World')
        - bool: True/False values
        
        You can check the type of a variable using type():
        ```python
        print(type(x))  # <class 'int'>
        ```
        """,
        "metadata": {"author": "Prof. Johnson", "last_updated": "2024-01-20"}
    }
]