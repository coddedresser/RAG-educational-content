"""
Configuration settings for the Educational RAG System
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    """Configuration class for the Educational RAG System"""
    
    # Base directory configuration
    BASE_DIR: Path = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "processed_data"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # File paths
    LOG_FILE: Path = LOGS_DIR / "educational_rag.log"
    CONFIG_FILE: Path = BASE_DIR / "config.json"
    
    # Database configuration
    DATABASE_URL: str = "sqlite:///educational_rag.db"
    
    # Model configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Search configuration
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 50
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Learning path configuration
    MAX_PATH_LENGTH: int = 10
    MIN_PATH_LENGTH: int = 3
    DEFAULT_ESTIMATED_TIME: int = 30
    
    # Student profile configuration
    DEFAULT_LEARNING_STYLE: str = "visual"
    DEFAULT_LEVEL: str = "beginner"
    SUPPORTED_SUBJECTS: List[str] = [
        "Mathematics", "Programming", "Science", "Literature", 
        "History", "Geography", "Art", "Music", "Physical Education"
    ]
    
    # Content types
    CONTENT_TYPES: List[str] = [
        "lesson", "exercise", "quiz", "video", "reading", 
        "project", "assessment", "tutorial"
    ]
    
    # Difficulty levels
    DIFFICULTY_LEVELS: List[str] = ["beginner", "intermediate", "advanced"]
    
    # API configuration
    OPENAI_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance configuration
    BATCH_SIZE: int = 100
    MAX_WORKERS: int = 4
    CACHE_TTL: int = 3600  # 1 hour
    
    # UI configuration
    PAGE_TITLE: str = "Educational RAG System"
    PAGE_ICON: str = "��"
    LAYOUT: str = "wide"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global config instance
config = Config()

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        config.DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.VECTOR_DB_DIR,
        config.MODELS_DIR,
        config.LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

# Sample educational content for demonstration
SAMPLE_CONTENT = [
    {
        "title": "Introduction to Algebra",
        "subject": "Mathematics",
        "topic": "Algebra Basics",
        "difficulty_level": "beginner",
        "content_type": "lesson",
        "estimated_time": 15,
        "learning_objectives": [
            "Understanding variables",
            "Basic algebraic expressions",
            "Simple equations"
        ],
        "prerequisites": [],
        "tags": ["algebra", "variables", "equations", "beginner"],
        "content": """
        Algebra is a branch of mathematics that deals with symbols and the rules for manipulating these symbols.
        In algebra, we use letters (variables) to represent unknown values and solve equations.
        
        Key Concepts:
        1. Variables: Letters that represent unknown values (e.g., x, y, z)
        2. Constants: Fixed numbers (e.g., 5, -3, 0.5)
        3. Expressions: Combinations of variables and constants (e.g., 2x + 3)
        4. Equations: Mathematical statements with equals sign (e.g., 2x + 3 = 7)
        
        Example:
        Solve for x: 2x + 3 = 7
        
        Solution:
        Step 1: Subtract 3 from both sides
        2x + 3 - 3 = 7 - 3
        2x = 4
        
        Step 2: Divide both sides by 2
        2x ÷ 2 = 4 ÷ 2
        x = 2
        
        Therefore, x = 2
        """
    },
    {
        "title": "Variables in Python",
        "subject": "Programming",
        "topic": "Python Basics",
        "difficulty_level": "beginner",
        "content_type": "lesson",
        "estimated_time": 20,
        "learning_objectives": [
            "Understanding variables",
            "Data types",
            "Basic programming concepts"
        ],
        "prerequisites": [],
        "tags": ["python", "variables", "programming", "beginner"],
        "content": """
        Variables are fundamental building blocks in programming. They allow us to store and manipulate data.
        
        What is a Variable?
        A variable is a named container that stores a value. Think of it as a labeled box where you can put different types of data.
        
        Creating Variables in Python:
        In Python, you create a variable by assigning a value to a name using the equals sign (=).
        
        Examples:
        
        1. String Variables:
        name = "Alice"
        message = "Hello, World!"
        
        2. Numeric Variables:
        age = 25
        height = 5.8
        temperature = -5
        
        3. Boolean Variables:
        is_student = True
        is_working = False
        
        4. List Variables:
        numbers = [1, 2, 3, 4, 5]
        colors = ["red", "green", "blue"]
        
        Variable Naming Rules:
        - Use descriptive names (e.g., 'user_age' instead of 'a')
        - Start with a letter or underscore
        - Can contain letters, numbers, and underscores
        - Case sensitive (age ≠ Age)
        - Avoid Python keywords (if, for, while, etc.)
        
        Good Examples:
        - user_name
        - total_score
        - is_active
        - first_name
        
        Bad Examples:
        - 1name (starts with number)
        - user-name (contains hyphen)
        - if (Python keyword)
        
        Using Variables:
        Once you've created a variable, you can use it in expressions and operations.
        
        Example:
        x = 5
        y = 3
        sum = x + y
        print(sum)  # Output: 8
        
        Reassigning Variables:
        Variables can be changed by assigning new values to them.
        
        Example:
        age = 25
        print(age)  # Output: 25
        
        age = 26
        print(age)  # Output: 26
        
        Practice Exercise:
        Create variables for:
        1. Your name
        2. Your age
        3. Your favorite number
        4. Whether you're a student (True/False)
        
        Then print all the information in a sentence.
        """
    }
]

# Additional configuration functions
def get_content_by_subject(subject: str) -> List[Dict[str, Any]]:
    """Get content filtered by subject"""
    return [content for content in SAMPLE_CONTENT if content["subject"] == subject]

def get_content_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """Get content filtered by difficulty level"""
    return [content for content in SAMPLE_CONTENT if content["difficulty_level"] == difficulty]

def get_content_by_type(content_type: str) -> List[Dict[str, Any]]:
    """Get content filtered by content type"""
    return [content for content in SAMPLE_CONTENT if content["content_type"] == content_type]

def get_available_subjects() -> List[str]:
    """Get list of available subjects"""
    return list(set(content["subject"] for content in SAMPLE_CONTENT))

def get_available_topics(subject: str = None) -> List[str]:
    """Get list of available topics, optionally filtered by subject"""
    if subject:
        return list(set(content["topic"] for content in SAMPLE_CONTENT if content["subject"] == subject))
    return list(set(content["topic"] for content in SAMPLE_CONTENT))

# Configuration validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required directories
    required_dirs = [
        config.DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.VECTOR_DB_DIR,
        config.MODELS_DIR,
        config.LOGS_DIR
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"Directory does not exist: {directory}")
    
    # Check content validity
    for i, content in enumerate(SAMPLE_CONTENT):
        required_fields = ["title", "subject", "topic", "difficulty_level", "content_type"]
        for field in required_fields:
            if field not in content:
                errors.append(f"Content {i}: Missing required field '{field}'")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Initialize configuration
if __name__ == "__main__":
    try:
        create_directories()
        validate_config()
        print("✅ Configuration loaded successfully!")
        print(f"📁 Base directory: {config.BASE_DIR}")
        print(f"📚 Sample content: {len(SAMPLE_CONTENT)} items")
        print(f"�� Available subjects: {', '.join(get_available_subjects())}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")