"""
Student profiling and learning style assessment system
"""
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum
import numpy as np

class LearningStyle(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING = "reading"

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class LearningPace(Enum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"

@dataclass
class StudentProfile:
    """Student profile data structure"""
    student_id: str
    name: str
    email: str
    current_level: str
    learning_style: str
    subjects_of_interest: List[str]
    learning_pace: str
    completed_content: List[str]
    knowledge_gaps: List[str]
    performance_history: Dict[str, Any]
    preferences: Dict[str, Any]
    created_at: str
    updated_at: str
    total_study_time: int = 0  # in minutes
    streak_days: int = 0
    last_activity: Optional[str] = None

class LearningStyleAssessment:
    """Learning style assessment questionnaire and analysis"""
    
    def __init__(self):
        self.questions = [
            {
                "id": 1,
                "question": "When learning something new, I prefer to:",
                "options": {
                    "A": "See diagrams, charts, or visual representations",
                    "B": "Listen to explanations or discussions",
                    "C": "Try it out hands-on or through practice",
                    "D": "Read detailed written instructions"
                }
            },
            {
                "id": 2,
                "question": "I remember information best when:",
                "options": {
                    "A": "I can visualize it or see it in my mind",
                    "B": "I hear it explained or discuss it with others",
                    "C": "I practice or apply it immediately",
                    "D": "I write it down or read about it"
                }
            },
            {
                "id": 3,
                "question": "When solving problems, I prefer to:",
                "options": {
                    "A": "Draw diagrams or create visual maps",
                    "B": "Talk through the problem out loud",
                    "C": "Work through examples step by step",
                    "D": "Research and read about similar problems"
                }
            },
            {
                "id": 4,
                "question": "In a learning environment, I work best when:",
                "options": {
                    "A": "There are visual aids and colorful materials",
                    "B": "I can ask questions and participate in discussions",
                    "C": "I can move around and interact with materials",
                    "D": "I have quiet time to read and reflect"
                }
            },
            {
                "id": 5,
                "question": "When following directions, I prefer:",
                "options": {
                    "A": "Maps, diagrams, or visual guides",
                    "B": "Verbal instructions or audio guidance",
                    "C": "To figure it out by trial and error",
                    "D": "Written step-by-step instructions"
                }
            }
        ]
        
        self.scoring_map = {
            "A": LearningStyle.VISUAL,
            "B": LearningStyle.AUDITORY,
            "C": LearningStyle.KINESTHETIC,
            "D": LearningStyle.READING
        }
    
    def calculate_learning_style(self, responses: List[str]) -> Tuple[LearningStyle, Dict[str, int]]:
        """Calculate learning style based on assessment responses"""
        scores = {style.value: 0 for style in LearningStyle}
        
        for response in responses:
            if response in self.scoring_map:
                style = self.scoring_map[response]
                scores[style.value] += 1
        
        # Find dominant learning style
        dominant_style = max(scores.items(), key=lambda x: x[1])
        return LearningStyle(dominant_style[0]), scores
    
    def get_learning_style_description(self, style: LearningStyle) -> Dict[str, Any]:
        """Get detailed description of learning style"""
        descriptions = {
            LearningStyle.VISUAL: {
                "name": "Visual Learner",
                "description": "You learn best through visual aids, diagrams, charts, and images.",
                "strengths": [
                    "Remembers faces and places well",
                    "Enjoys visual arts and creativity",
                    "Can visualize information easily",
                    "Good at reading maps and charts"
                ],
                "learning_tips": [
                    "Use mind maps and diagrams",
                    "Highlight important information in different colors",
                    "Watch educational videos",
                    "Create visual summaries of content"
                ],
                "preferred_content": ["video", "infographic", "diagram", "chart"]
            },
            LearningStyle.AUDITORY: {
                "name": "Auditory Learner",
                "description": "You learn best through listening, speaking, and verbal instruction.",
                "strengths": [
                    "Good at remembering spoken information",
                    "Enjoys discussions and debates",
                    "Can follow verbal directions well",
                    "Often thinks out loud"
                ],
                "learning_tips": [
                    "Listen to podcasts and audio lectures",
                    "Discuss topics with others",
                    "Read content aloud",
                    "Use music or rhymes to remember information"
                ],
                "preferred_content": ["audio", "lecture", "discussion", "podcast"]
            },
            LearningStyle.KINESTHETIC: {
                "name": "Kinesthetic Learner",
                "description": "You learn best through hands-on experience and physical activity.",
                "strengths": [
                    "Good at learning through practice",
                    "Remembers what they do",
                    "Enjoys building and creating",
                    "Good hand-eye coordination"
                ],
                "learning_tips": [
                    "Use hands-on activities and experiments",
                    "Take breaks for physical movement",
                    "Practice skills immediately",
                    "Use manipulatives and real objects"
                ],
                "preferred_content": ["exercise", "simulation", "project", "lab"]
            },
            LearningStyle.READING: {
                "name": "Reading/Writing Learner",
                "description": "You learn best through reading and writing activities.",
                "strengths": [
                    "Enjoys reading and writing",
                    "Good at organizing information",
                    "Remembers written information well",
                    "Prefers written instructions"
                ],
                "learning_tips": [
                    "Take detailed notes while learning",
                    "Rewrite information in your own words",
                    "Create written summaries",
                    "Use lists and written plans"
                ],
                "preferred_content": ["text", "article", "book", "written_exercise"]
            }
        }
        
        return descriptions.get(style, {})

class StudentProfileManager:
    """Manage student profiles and learning data"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.assessment = LearningStyleAssessment()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for student profiles"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create student profiles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS student_profiles (
                        student_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE,
                        current_level TEXT NOT NULL,
                        learning_style TEXT NOT NULL,
                        subjects_of_interest TEXT,
                        learning_pace TEXT NOT NULL,
                        completed_content TEXT,
                        knowledge_gaps TEXT,
                        performance_history TEXT,
                        preferences TEXT,
                        total_study_time INTEGER DEFAULT 0,
                        streak_days INTEGER DEFAULT 0,
                        last_activity TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Create learning sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        session_id TEXT PRIMARY KEY,
                        student_id TEXT NOT NULL,
                        content_id TEXT NOT NULL,
                        session_start TEXT NOT NULL,
                        session_end TEXT,
                        duration_minutes INTEGER,
                        completion_status TEXT,
                        performance_score REAL,
                        difficulty_level TEXT,
                        subject TEXT,
                        FOREIGN KEY (student_id) REFERENCES student_profiles (student_id)
                    )
                """)
                
                # Create assessments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS assessments (
                        assessment_id TEXT PRIMARY KEY,
                        student_id TEXT NOT NULL,
                        assessment_type TEXT NOT NULL,
                        questions_data TEXT,
                        responses TEXT,
                        results TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (student_id) REFERENCES student_profiles (student_id)
                    )
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def create_student_profile(self, 
                             student_id: str,
                             name: str,
                             email: str,
                             assessment_responses: List[str] = None) -> StudentProfile:
        """Create a new student profile"""
        
        # Determine learning style from assessment
        if assessment_responses:
            learning_style, style_scores = self.assessment.calculate_learning_style(assessment_responses)
            learning_style_value = learning_style.value
            
            # Store assessment results
            self._save_assessment(student_id, "learning_style", assessment_responses, style_scores)
        else:
            learning_style_value = "reading"  # Default
        
        # Create profile
        now = datetime.now().isoformat()
        profile = StudentProfile(
            student_id=student_id,
            name=name,
            email=email,
            current_level=DifficultyLevel.BEGINNER.value,
            learning_style=learning_style_value,
            subjects_of_interest=[],
            learning_pace=LearningPace.MEDIUM.value,
            completed_content=[],
            knowledge_gaps=[],
            performance_history={},
            preferences={},
            created_at=now,
            updated_at=now,
            total_study_time=0,
            streak_days=0,
            last_activity=None
        )
        
        # Save to database
        self._save_profile_to_db(profile)
        self.logger.info(f"Created profile for student: {name} ({student_id})")
        
        return profile
    
    def _save_profile_to_db(self, profile: StudentProfile):
        """Save student profile to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO student_profiles 
                    (student_id, name, email, current_level, learning_style, 
                     subjects_of_interest, learning_pace, completed_content, 
                     knowledge_gaps, performance_history, preferences,
                     total_study_time, streak_days, last_activity, 
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.student_id,
                    profile.name,
                    profile.email,
                    profile.current_level,
                    profile.learning_style,
                    json.dumps(profile.subjects_of_interest),
                    profile.learning_pace,
                    json.dumps(profile.completed_content),
                    json.dumps(profile.knowledge_gaps),
                    json.dumps(profile.performance_history),
                    json.dumps(profile.preferences),
                    profile.total_study_time,
                    profile.streak_days,
                    profile.last_activity,
                    profile.created_at,
                    profile.updated_at
                ))
                
                conn.commit()
        
        except Exception as e:
            self.logger.error(f"Error saving profile to database: {e}")
            raise
    
    def load_student_profile(self, student_id: str) -> Optional[StudentProfile]:
        """Load student profile from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM student_profiles WHERE student_id = ?
                """, (student_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Convert row to profile
                profile = StudentProfile(
                    student_id=row[0],
                    name=row[1],
                    email=row[2],
                    current_level=row[3],
                    learning_style=row[4],
                    subjects_of_interest=json.loads(row[5]) if row[5] else [],
                    learning_pace=row[6],
                    completed_content=json.loads(row[7]) if row[7] else [],
                    knowledge_gaps=json.loads(row[8]) if row[8] else [],
                    performance_history=json.loads(row[9]) if row[9] else {},
                    preferences=json.loads(row[10]) if row[10] else {},
                    total_study_time=row[11],
                    streak_days=row[12],
                    last_activity=row[13],
                    created_at=row[14],
                    updated_at=row[15]
                )
                
                return profile
        
        except Exception as e:
            self.logger.error(f"Error loading profile: {e}")
            return None
    
    def update_student_progress(self, 
                               student_id: str,
                               completed_content_id: str,
                               performance_score: float,
                               session_duration: int):
        """Update student progress after completing content"""
        profile = self.load_student_profile(student_id)
        if not profile:
            return
        
        # Update completed content
        if completed_content_id not in profile.completed_content:
            profile.completed_content.append(completed_content_id)
        
        # Update performance history
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in profile.performance_history:
            profile.performance_history[today] = []
        
        profile.performance_history[today].append({
            'content_id': completed_content_id,
            'score': performance_score,
            'duration': session_duration
        })
        
        # Update study time and streak
        profile.total_study_time += session_duration
        profile.last_activity = datetime.now().isoformat()
        
        # Update streak
        profile.streak_days = self._calculate_streak(profile.performance_history)
        
        # Check for level progression
        self._check_level_progression(profile)
        
        profile.updated_at = datetime.now().isoformat()
        self._save_profile_to_db(profile)
    
    def _calculate_streak(self, performance_history: Dict) -> int:
        """Calculate current learning streak in days"""
        if not performance_history:
            return 0
        
        # Sort dates
        dates = sorted(performance_history.keys(), reverse=True)
        today = datetime.now().date()
        
        streak = 0
        for date_str in dates:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            days_diff = (today - date).days
            
            if days_diff == streak:
                streak += 1
            else:
                break
        
        return streak
    
    def _check_level_progression(self, profile: StudentProfile):
        """Check if student should progress to next level"""
        if not profile.performance_history:
            return
        
        # Get recent performance (last 10 sessions)
        recent_scores = []
        for sessions in list(profile.performance_history.values())[-5:]:  # Last 5 days
            for session in sessions:
                recent_scores.append(session['score'])
        
        if len(recent_scores) < 5:  # Need minimum sessions
            return
        
        avg_score = np.mean(recent_scores[-10:])  # Last 10 sessions
        
        # Progression thresholds
        if profile.current_level == 'beginner' and avg_score >= 0.8:
            profile.current_level = 'intermediate'
            self.logger.info(f"Student {profile.student_id} progressed to intermediate level")
        
        elif profile.current_level == 'intermediate' and avg_score >= 0.85:
            profile.current_level = 'advanced'
            self.logger.info(f"Student {profile.student_id} progressed to advanced level")
    
    def identify_knowledge_gaps(self, student_id: str, subject: str) -> List[str]:
        """Identify knowledge gaps based on performance"""
        profile = self.load_student_profile(student_id)
        if not profile:
            return []
        
        # Analyze performance history to find weak areas
        weak_topics = []
        topic_scores = {}
        
        # This would be enhanced with actual content topic mapping
        # For now, return placeholder gaps
        if subject == "Mathematics":
            potential_gaps = ["algebra", "geometry", "fractions", "equations"]
        elif subject == "Programming":
            potential_gaps = ["variables", "functions", "loops", "data_structures"]
        else:
            potential_gaps = ["fundamentals", "intermediate_concepts", "advanced_topics"]
        
        # Simple logic: if average score in recent sessions is low, identify gaps
        recent_scores = []
        for sessions in list(profile.performance_history.values())[-3:]:
            for session in sessions:
                recent_scores.append(session['score'])
        
        if recent_scores and np.mean(recent_scores) < 0.7:
            weak_topics = potential_gaps[:2]  # Return first 2 as gaps
        
        return weak_topics
    
    def _save_assessment(self, student_id: str, assessment_type: str, 
                        responses: List[str], results: Dict):
        """Save assessment results to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                assessment_id = f"{student_id}_{assessment_type}_{int(datetime.now().timestamp())}"
                
                cursor.execute("""
                    INSERT INTO assessments 
                    (assessment_id, student_id, assessment_type, 
                     questions_data, responses, results, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    assessment_id,
                    student_id,
                    assessment_type,
                    json.dumps(self.assessment.questions),
                    json.dumps(responses),
                    json.dumps(results),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
        
        except Exception as e:
            self.logger.error(f"Error saving assessment: {e}")
    
    def get_student_analytics(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a student"""
        profile = self.load_student_profile(student_id)
        if not profile:
            return {}
        
        # Calculate various metrics
        total_content = len(profile.completed_content)
        recent_performance = []
        
        for sessions in list(profile.performance_history.values())[-7:]:
            for session in sessions:
                recent_performance.append(session['score'])
        
        analytics = {
            'profile_summary': {
                'name': profile.name,
                'level': profile.current_level,
                'learning_style': profile.learning_style,
                'total_study_time': profile.total_study_time,
                'streak_days': profile.streak_days
            },
            'progress_metrics': {
                'completed_content_count': total_content,
                'average_recent_score': np.mean(recent_performance) if recent_performance else 0,
                'knowledge_gaps_count': len(profile.knowledge_gaps),
                'subjects_count': len(profile.subjects_of_interest)
            },
            'recommendations': self._generate_recommendations(profile)
        }
        
        return analytics
    
    def _generate_recommendations(self, profile: StudentProfile) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Based on learning style
        style_desc = self.assessment.get_learning_style_description(
            LearningStyle(profile.learning_style)
        )
        recommendations.extend(style_desc.get('learning_tips', [])[:2])
        
        # Based on performance
        if profile.streak_days == 0:
            recommendations.append("Try to study consistently every day to build a learning streak")
        elif profile.streak_days < 7:
            recommendations.append(f"Great! You're on a {profile.streak_days}-day streak. Keep it up!")
        
        # Based on knowledge gaps
        if profile.knowledge_gaps:
            recommendations.append(f"Focus on improving: {', '.join(profile.knowledge_gaps[:2])}")
        
        return recommendations[:5]  # Limit to 5 recommendations

# Example usage
if __name__ == "__main__":
    from app.config import config
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create database directory
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize profile manager
    db_path = config.MODELS_DIR / "student_profiles.db"
    profile_manager = StudentProfileManager(db_path)
    
    # Sample assessment responses (A=visual, B=auditory, C=kinesthetic, D=reading)
    sample_responses = ["A", "B", "C", "A", "D"]
    
    # Create sample student profile
    profile = profile_manager.create_student_profile(
        student_id="student_001",
        name="John Doe",
        email="john.doe@example.com",
        assessment_responses=sample_responses
    )
    
    print(f"Created profile for: {profile.name}")
    print(f"Learning Style: {profile.learning_style}")
    print(f"Current Level: {profile.current_level}")
    
    # Simulate some learning progress
    profile_manager.update_student_progress(
        student_id="student_001",
        completed_content_id="math_001",
        performance_score=0.85,
        session_duration=25
    )
    
    # Get analytics
    analytics = profile_manager.get_student_analytics("student_001")
    print("\nStudent Analytics:")
    for key, value in analytics.items():
        print(f"{key}: {value}")