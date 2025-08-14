"""
Progress tracking and analytics system for educational content
"""
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

@dataclass
class LearningSession:
    """Individual learning session data"""
    session_id: str
    student_id: str
    content_id: str
    session_start: str
    session_end: Optional[str]
    duration_minutes: int
    completion_status: str  # started, completed, abandoned
    performance_score: float
    difficulty_level: str
    subject: str
    content_type: str
    interactions: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ProgressMetrics:
    """Student progress metrics"""
    student_id: str
    total_study_time: int
    sessions_completed: int
    average_score: float
    streak_days: int
    last_activity: str
    subjects_studied: List[str]
    difficulty_progression: Dict[str, int]
    learning_velocity: float  # content items per hour
    mastery_levels: Dict[str, float]  # subject -> mastery level

class ProgressTracker:
    """Track and analyze student learning progress"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize progress tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Learning sessions table
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
                        content_type TEXT,
                        interactions TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Progress snapshots table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS progress_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        student_id TEXT NOT NULL,
                        snapshot_date TEXT NOT NULL,
                        metrics_data TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Knowledge assessments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_assessments (
                        assessment_id TEXT PRIMARY KEY,
                        student_id TEXT NOT NULL,
                        subject TEXT,
                        topic TEXT,
                        assessment_type TEXT,
                        questions_data TEXT,
                        responses TEXT,
                        score REAL,
                        mastery_level REAL,
                        assessment_date TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Learning goals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_goals (
                        goal_id TEXT PRIMARY KEY,
                        student_id TEXT NOT NULL,
                        goal_type TEXT,
                        target_subject TEXT,
                        target_objectives TEXT,
                        target_date TEXT,
                        current_progress REAL DEFAULT 0.0,
                        status TEXT DEFAULT 'active',
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
                
                conn.commit()
                self.logger.info("Progress tracking database initialized")
        
        except Exception as e:
            self.logger.error(f"Error initializing progress database: {e}")
            raise
    
    def start_learning_session(self, 
                             student_id: str,
                             content_id: str,
                             subject: str,
                             difficulty_level: str,
                             content_type: str) -> str:
        """Start a new learning session"""
        session_id = f"session_{student_id}_{int(datetime.now().timestamp())}"
        
        session = LearningSession(
            session_id=session_id,
            student_id=student_id,
            content_id=content_id,
            session_start=datetime.now().isoformat(),
            session_end=None,
            duration_minutes=0,
            completion_status="started",
            performance_score=0.0,
            difficulty_level=difficulty_level,
            subject=subject,
            content_type=content_type,
            interactions={},
            metadata={}
        )
        
        self._save_session(session)
        self.logger.info(f"Started learning session: {session_id}")
        return session_id
    
    def end_learning_session(self, 
                           session_id: str,
                           completion_status: str,
                           performance_score: float,
                           interactions: Dict[str, Any] = None) -> bool:
        """End a learning session and record results"""
        try:
            # Load existing session
            session = self._load_session(session_id)
            if not session:
                self.logger.error(f"Session not found: {session_id}")
                return False
            
            # Update session
            session.session_end = datetime.now().isoformat()
            session.completion_status = completion_status
            session.performance_score = performance_score
            
            if interactions:
                session.interactions.update(interactions)
            
            # Calculate duration
            start_time = datetime.fromisoformat(session.session_start)
            end_time = datetime.fromisoformat(session.session_end)
            session.duration_minutes = int((end_time - start_time).total_seconds() / 60)
            
            # Save updated session
            self._save_session(session)
            
            # Update progress metrics
            self._update_progress_metrics(session)
            
            self.logger.info(f"Ended learning session: {session_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error ending session {session_id}: {e}")
            return False
    
    def _save_session(self, session: LearningSession):
        """Save learning session to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO learning_sessions 
                    (session_id, student_id, content_id, session_start, session_end,
                     duration_minutes, completion_status, performance_score,
                     difficulty_level, subject, content_type, interactions, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.student_id,
                    session.content_id,
                    session.session_start,
                    session.session_end,
                    session.duration_minutes,
                    session.completion_status,
                    session.performance_score,
                    session.difficulty_level,
                    session.subject,
                    session.content_type,
                    json.dumps(session.interactions),
                    json.dumps(session.metadata)
                ))
                
                conn.commit()
        
        except Exception as e:
            self.logger.error(f"Error saving session: {e}")
            raise
    
    def _load_session(self, session_id: str) -> Optional[LearningSession]:
        """Load learning session from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM learning_sessions WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return LearningSession(
                    session_id=row[0],
                    student_id=row[1],
                    content_id=row[2],
                    session_start=row[3],
                    session_end=row[4],
                    duration_minutes=row[5] or 0,
                    completion_status=row[6] or "started",
                    performance_score=row[7] or 0.0,
                    difficulty_level=row[8],
                    subject=row[9],
                    content_type=row[10],
                    interactions=json.loads(row[11]) if row[11] else {},
                    metadata=json.loads(row[12]) if row[12] else {}
                )
        
        except Exception as e:
            self.logger.error(f"Error loading session: {e}")
            return None
    
    def _update_progress_metrics(self, session: LearningSession):
        """Update student progress metrics after session"""
        # This would trigger comprehensive progress calculation
        # For now, we'll create a snapshot
        metrics = self.calculate_progress_metrics(session.student_id)
        self._save_progress_snapshot(session.student_id, metrics)
    
    def calculate_progress_metrics(self, student_id: str) -> ProgressMetrics:
        """Calculate comprehensive progress metrics for a student"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all completed sessions
                cursor.execute("""
                    SELECT * FROM learning_sessions 
                    WHERE student_id = ? AND completion_status = 'completed'
                    ORDER BY session_start
                """, (student_id,))
                
                sessions = cursor.fetchall()
                
                if not sessions:
                    return self._empty_metrics(student_id)
                
                # Calculate metrics
                total_study_time = sum(session[5] for session in sessions if session[5])
                sessions_completed = len(sessions)
                
                scores = [session[7] for session in sessions if session[7] is not None]
                average_score = np.mean(scores) if scores else 0.0
                
                # Calculate streak
                streak_days = self._calculate_learning_streak(sessions)
                
                # Get last activity
                last_session = sessions[-1] if sessions else None
                last_activity = last_session[3] if last_session else ""
                
                # Subject analysis
                subjects_studied = list(set(session[9] for session in sessions if session[9]))
                
                # Difficulty progression
                difficulty_counts = defaultdict(int)
                for session in sessions:
                    if session[8]:  # difficulty_level
                        difficulty_counts[session[8]] += 1
                
                # Learning velocity (items per hour)
                if total_study_time > 0:
                    learning_velocity = (sessions_completed / total_study_time) * 60
                else:
                    learning_velocity = 0.0
                
                # Mastery levels by subject
                mastery_levels = self._calculate_mastery_levels(sessions)
                
                return ProgressMetrics(
                    student_id=student_id,
                    total_study_time=total_study_time,
                    sessions_completed=sessions_completed,
                    average_score=average_score,
                    streak_days=streak_days,
                    last_activity=last_activity,
                    subjects_studied=subjects_studied,
                    difficulty_progression=dict(difficulty_counts),
                    learning_velocity=learning_velocity,
                    mastery_levels=mastery_levels
                )
        
        except Exception as e:
            self.logger.error(f"Error calculating progress metrics: {e}")
            return self._empty_metrics(student_id)
    
    def _empty_metrics(self, student_id: str) -> ProgressMetrics:
        """Return empty progress metrics"""
        return ProgressMetrics(
            student_id=student_id,
            total_study_time=0,
            sessions_completed=0,
            average_score=0.0,
            streak_days=0,
            last_activity="",
            subjects_studied=[],
            difficulty_progression={},
            learning_velocity=0.0,
            mastery_levels={}
        )
    
    def _calculate_learning_streak(self, sessions: List[Tuple]) -> int:
        """Calculate current learning streak in days"""
        if not sessions:
            return 0
        
        # Group sessions by date
        session_dates = set()
        for session in sessions:
            session_date = datetime.fromisoformat(session[3]).date()
            session_dates.add(session_date)
        
        # Sort dates
        sorted_dates = sorted(session_dates, reverse=True)
        
        # Calculate streak from most recent date
        today = datetime.now().date()
        streak = 0
        
        for i, date in enumerate(sorted_dates):
            days_ago = (today - date).days
            
            if days_ago == i:  # Consecutive days
                streak += 1
            elif days_ago == i + 1 and i == 0:  # Allow 1 day gap at start
                streak += 1
            else:
                break
        
        return streak
    
    def _calculate_mastery_levels(self, sessions: List[Tuple]) -> Dict[str, float]:
        """Calculate mastery levels for each subject"""
        subject_sessions = defaultdict(list)
        
        for session in sessions:
            subject = session[9]  # subject column
            score = session[7]   # performance_score column
            
            if subject and score is not None:
                subject_sessions[subject].append(score)
        
        mastery_levels = {}
        for subject, scores in subject_sessions.items():
            # Use weighted average with more weight on recent sessions
            if scores:
                weights = np.array([1.0 + 0.1 * i for i in range(len(scores))])
                weighted_avg = np.average(scores, weights=weights)
                mastery_levels[subject] = min(weighted_avg, 1.0)
        
        return mastery_levels
    
    def _save_progress_snapshot(self, student_id: str, metrics: ProgressMetrics):
        """Save progress snapshot to database"""
        try:
            snapshot_id = f"snapshot_{student_id}_{int(datetime.now().timestamp())}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO progress_snapshots 
                    (snapshot_id, student_id, snapshot_date, metrics_data)
                    VALUES (?, ?, ?, ?)
                """, (
                    snapshot_id,
                    student_id,
                    datetime.now().isoformat(),
                    json.dumps(asdict(metrics))
                ))
                
                conn.commit()
        
        except Exception as e:
            self.logger.error(f"Error saving progress snapshot: {e}")
    
    def get_learning_analytics(self, 
                             student_id: str,
                             time_range_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive learning analytics for a student"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=time_range_days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get sessions in time range
                cursor.execute("""
                    SELECT * FROM learning_sessions 
                    WHERE student_id = ? AND session_start >= ?
                    ORDER BY session_start
                """, (student_id, cutoff_date))
                
                sessions = cursor.fetchall()
                
                if not sessions:
                    return {"message": "No learning activity in the specified time range"}
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame(sessions, columns=[
                    'session_id', 'student_id', 'content_id', 'session_start',
                    'session_end', 'duration_minutes', 'completion_status',
                    'performance_score', 'difficulty_level', 'subject',
                    'content_type', 'interactions', 'metadata', 'created_at'
                ])
                
                # Analyze patterns
                analytics = {
                    'time_range_days': time_range_days,
                    'total_sessions': len(df),
                    'completed_sessions': len(df[df['completion_status'] == 'completed']),
                    'total_study_time': df['duration_minutes'].sum(),
                    'average_session_duration': df['duration_minutes'].mean(),
                    'completion_rate': len(df[df['completion_status'] == 'completed']) / len(df) * 100,
                    'average_performance': df['performance_score'].mean(),
                    'subject_breakdown': df['subject'].value_counts().to_dict(),
                    'difficulty_progression': df['difficulty_level'].value_counts().to_dict(),
                    'content_type_preferences': df['content_type'].value_counts().to_dict(),
                    'learning_patterns': self._analyze_learning_patterns(df),
                    'recommendations': self._generate_learning_recommendations(df)
                }
                
                return analytics
        
        except Exception as e:
            self.logger.error(f"Error generating learning analytics: {e}")
            return {"error": str(e)}
    
    def _analyze_learning_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze learning patterns from session data"""
        patterns = {}
        
        # Time patterns
        df['hour'] = pd.to_datetime(df['session_start']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['session_start']).dt.day_name()
        
        patterns['preferred_study_hours'] = df['hour'].value_counts().head(3).to_dict()
        patterns['preferred_study_days'] = df['day_of_week'].value_counts().head(3).to_dict()
        
        # Performance patterns
        avg_score_by_hour = df.groupby('hour')['performance_score'].mean()
        patterns['best_performance_hours'] = avg_score_by_hour.nlargest(3).to_dict()
        
        avg_score_by_difficulty = df.groupby('difficulty_level')['performance_score'].mean()
        patterns['performance_by_difficulty'] = avg_score_by_difficulty.to_dict()
        
        # Session length patterns
        avg_duration_by_type = df.groupby('content_type')['duration_minutes'].mean()
        patterns['duration_by_content_type'] = avg_duration_by_type.to_dict()
        
        return patterns
    
    def _generate_learning_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate personalized learning recommendations"""
        recommendations = []
        
        # Completion rate recommendations
        completion_rate = len(df[df['completion_status'] == 'completed']) / len(df) * 100
        if completion_rate < 70:
            recommendations.append("Try shorter learning sessions to improve completion rate")
        
        # Performance recommendations
        avg_performance = df['performance_score'].mean()
        if avg_performance < 0.7:
            recommendations.append("Consider reviewing prerequisite topics before tackling new content")
        elif avg_performance > 0.9:
            recommendations.append("You're excelling! Try more challenging content to accelerate learning")
        
        # Time-based recommendations
        study_hours = df['hour'].value_counts()
        if len(study_hours) > 0:
            best_hour = study_hours.index[0]
            recommendations.append(f"Your most active learning time is {best_hour}:00. Consider scheduling important topics then")
        
        # Subject balance recommendations
        subjects = df['subject'].value_counts()
        if len(subjects) > 1:
            most_studied = subjects.index[0]
            least_studied = subjects.index[-1]
            if subjects[most_studied] > subjects[least_studied] * 3:
                recommendations.append(f"Consider spending more time on {least_studied} to maintain balanced learning")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def record_assessment(self, 
                        student_id: str,
                        subject: str,
                        topic: str,
                        assessment_type: str,
                        questions_data: List[Dict],
                        responses: List[Any],
                        score: float) -> str:
        """Record assessment results and calculate mastery level"""
        assessment_id = f"assessment_{student_id}_{int(datetime.now().timestamp())}"
        
        # Calculate mastery level based on score and assessment difficulty
        mastery_level = min(score, 1.0)  # Simple mapping for now
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO knowledge_assessments 
                    (assessment_id, student_id, subject, topic, assessment_type,
                     questions_data, responses, score, mastery_level, assessment_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    assessment_id,
                    student_id,
                    subject,
                    topic,
                    assessment_type,
                    json.dumps(questions_data),
                    json.dumps(responses),
                    score,
                    mastery_level,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
                self.logger.info(f"Recorded assessment: {assessment_id}")
                return assessment_id
        
        except Exception as e:
            self.logger.error(f"Error recording assessment: {e}")
            return ""
    
    def get_student_sessions(self, 
                           student_id: str,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent learning sessions for a student"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM learning_sessions 
                    WHERE student_id = ?
                    ORDER BY session_start DESC
                    LIMIT ?
                """, (student_id, limit))
                
                rows = cursor.fetchall()
                
                sessions = []
                for row in rows:
                    session_data = {
                        'session_id': row[0],
                        'student_id': row[1],
                        'content_id': row[2],
                        'session_start': row[3],
                        'session_end': row[4],
                        'duration_minutes': row[5],
                        'completion_status': row[6],
                        'performance_score': row[7],
                        'difficulty_level': row[8],
                        'subject': row[9],
                        'content_type': row[10],
                        'interactions': json.loads(row[11]) if row[11] else {},
                        'metadata': json.loads(row[12]) if row[12] else {}
                    }
                    sessions.append(session_data)
                
                return sessions
        
        except Exception as e:
            self.logger.error(f"Error getting student sessions: {e}")
            return []
    
    def create_learning_goal(self, 
                           student_id: str,
                           goal_type: str,
                           target_subject: str,
                           target_objectives: List[str],
                           target_date: str) -> str:
        """Create a learning goal for a student"""
        goal_id = f"goal_{student_id}_{int(datetime.now().timestamp())}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO learning_goals 
                    (goal_id, student_id, goal_type, target_subject, 
                     target_objectives, target_date, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal_id,
                    student_id,
                    goal_type,
                    target_subject,
                    json.dumps(target_objectives),
                    target_date,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                self.logger.info(f"Created learning goal: {goal_id}")
                return goal_id
        
        except Exception as e:
            self.logger.error(f"Error creating learning goal: {e}")
            return ""
    
    def update_goal_progress(self, goal_id: str, progress: float):
        """Update progress on a learning goal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE learning_goals 
                    SET current_progress = ?, updated_at = ?
                    WHERE goal_id = ?
                """, (progress, datetime.now().isoformat(), goal_id))
                
                conn.commit()
        
        except Exception as e:
            self.logger.error(f"Error updating goal progress: {e}")

# Example usage
if __name__ == "__main__":
    from app.config import config
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create database directory
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress tracker
    db_path = config.MODELS_DIR / "progress_tracking.db"
    tracker = ProgressTracker(db_path)
    
    # Simulate learning session
    student_id = "student_001"
    session_id = tracker.start_learning_session(
        student_id=student_id,
        content_id="math_001",
        subject="Mathematics",
        difficulty_level="beginner",
        content_type="lesson"
    )
    
    print(f"Started session: {session_id}")
    
    # Simulate session completion
    success = tracker.end_learning_session(
        session_id=session_id,
        completion_status="completed",
        performance_score=0.85,
        interactions={"questions_answered": 5, "hints_used": 2}
    )
    
    print(f"Session ended: {success}")
    
    # Get progress metrics
    metrics = tracker.calculate_progress_metrics(student_id)
    print(f"Progress metrics: {asdict(metrics)}")
    
    # Get learning analytics
    analytics = tracker.get_learning_analytics(student_id)
    print(f"Learning analytics: {analytics}")