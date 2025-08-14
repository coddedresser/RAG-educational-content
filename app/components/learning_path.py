"""
Learning path generation and optimization system
"""
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import networkx as nx
import numpy as np
from collections import defaultdict, deque

from .student_profile import StudentProfile
from .data_processor import ContentChunk

@dataclass
class LearningPathItem:
    """Individual item in a learning path"""
    content_id: str
    title: str
    subject: str
    topic: str
    difficulty_level: str
    learning_objectives: List[str]
    prerequisites: List[str]
    content_type: str
    estimated_time: int
    order_index: int
    is_checkpoint: bool = False
    completion_status: str = "not_started"  # not_started, in_progress, completed
    mastery_score: float = 0.0

@dataclass
class LearningPath:
    """Complete learning path for a student"""
    path_id: str
    student_id: str
    target_subject: str
    target_objectives: List[str]
    difficulty_progression: str  # gradual, moderate, steep
    estimated_total_time: int
    path_items: List[LearningPathItem]
    milestones: List[Dict[str, Any]]
    created_at: str
    updated_at: str
    completion_percentage: float = 0.0
    current_item_index: int = 0

class PrerequisiteGraph:
    """Manages prerequisite relationships between learning content"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.topic_dependencies = {}
        self.logger = logging.getLogger(__name__)
    
    def add_content(self, content_chunk: ContentChunk):
        """Add content and its prerequisites to the graph"""
        content_id = content_chunk.content_id
        
        # Add node
        self.graph.add_node(content_id, 
                          title=content_chunk.title,
                          subject=content_chunk.subject,
                          topic=content_chunk.topic,
                          difficulty=content_chunk.difficulty_level,
                          objectives=content_chunk.learning_objectives,
                          estimated_time=content_chunk.estimated_time)
        
        # Add prerequisite edges
        for prereq in content_chunk.prerequisites:
            # Find content that covers this prerequisite
            prereq_content = self._find_content_by_topic(prereq, content_chunk.subject)
            if prereq_content:
                self.graph.add_edge(prereq_content, content_id)
    
    def _find_content_by_topic(self, topic: str, subject: str) -> Optional[str]:
        """Find content ID that covers a specific topic"""
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if (node_data.get('subject') == subject and 
                any(topic.lower() in obj.lower() for obj in node_data.get('objectives', []))):
                return node
        return None
    
    def get_prerequisites(self, content_id: str) -> List[str]:
        """Get all prerequisites for a content item"""
        if content_id not in self.graph:
            return []
        return list(self.graph.predecessors(content_id))
    
    def get_dependents(self, content_id: str) -> List[str]:
        """Get all content that depends on this item"""
        if content_id not in self.graph:
            return []
        return list(self.graph.successors(content_id))
    
    def topological_sort(self, content_ids: List[str]) -> List[str]:
        """Sort content items in prerequisite order"""
        subgraph = self.graph.subgraph(content_ids)
        try:
            return list(nx.topological_sort(subgraph))
        except nx.NetworkXError:
            # Handle cycles by using approximate topological sort
            self.logger.warning("Cycle detected in prerequisites, using approximate sort")
            return self._approximate_topological_sort(content_ids)
    
    def _approximate_topological_sort(self, content_ids: List[str]) -> List[str]:
        """Approximate topological sort when cycles exist"""
        # Simple approach: sort by difficulty level and in-degree
        content_with_metrics = []
        
        for content_id in content_ids:
            if content_id in self.graph:
                node_data = self.graph.nodes[content_id]
                in_degree = self.graph.in_degree(content_id)
                
                # Map difficulty to numeric value
                difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
                difficulty_score = difficulty_map.get(node_data.get('difficulty', 'intermediate'), 2)
                
                content_with_metrics.append((content_id, in_degree, difficulty_score))
            else:
                content_with_metrics.append((content_id, 0, 1))
        
        # Sort by in-degree (prerequisites first) then by difficulty
        content_with_metrics.sort(key=lambda x: (x[1], x[2]))
        return [item[0] for item in content_with_metrics]

class PathOptimizer:
    """Optimize learning paths based on student profile and learning science"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Learning science parameters
        self.difficulty_progression_rates = {
            'gradual': 0.05,    # 5% increase per step
            'moderate': 0.10,   # 10% increase per step
            'steep': 0.15       # 15% increase per step
        }
        
        self.learning_style_preferences = {
            'visual': {'video': 1.3, 'reading': 0.9, 'exercise': 1.0, 'quiz': 0.8},
            'auditory': {'video': 1.2, 'reading': 1.0, 'exercise': 0.9, 'quiz': 0.7},
            'kinesthetic': {'exercise': 1.3, 'project': 1.2, 'quiz': 1.0, 'reading': 0.8},
            'reading': {'reading': 1.2, 'lesson': 1.1, 'exercise': 0.9, 'video': 0.7}
        }
    
    def optimize_content_sequence(self, 
                                content_items: List[ContentChunk],
                                student_profile: StudentProfile,
                                target_objectives: List[str]) -> List[ContentChunk]:
        """Optimize the sequence of content based on student profile"""
        
        # Filter content relevant to target objectives
        relevant_content = self._filter_relevant_content(content_items, target_objectives)
        
        # Score and rank content based on student profile
        scored_content = []
        for content in relevant_content:
            score = self._calculate_content_score(content, student_profile)
            scored_content.append((content, score))
        
        # Sort by score (higher is better)
        scored_content.sort(key=lambda x: x[1], reverse=True)
        
        # Apply difficulty progression
        optimized_content = self._apply_difficulty_progression(
            [item[0] for item in scored_content], 
            student_profile
        )
        
        return optimized_content
    
    def _filter_relevant_content(self, 
                                content_items: List[ContentChunk], 
                                target_objectives: List[str]) -> List[ContentChunk]:
        """Filter content relevant to learning objectives"""
        relevant_content = []
        
        for content in content_items:
            # Check if content objectives overlap with target objectives
            content_objectives = [obj.lower() for obj in content.learning_objectives]
            target_objectives_lower = [obj.lower() for obj in target_objectives]
            
            # Simple keyword matching (can be enhanced with semantic similarity)
            relevance_score = 0
            for target_obj in target_objectives_lower:
                for content_obj in content_objectives:
                    if any(word in content_obj for word in target_obj.split()):
                        relevance_score += 1
            
            if relevance_score > 0:
                relevant_content.append(content)
        
        return relevant_content
    
    def _calculate_content_score(self, 
                               content: ContentChunk, 
                               student_profile: StudentProfile) -> float:
        """Calculate relevance score for content based on student profile"""
        score = 1.0
        
        # Learning style preference
        learning_style = student_profile.learning_style
        content_type = content.content_type
        
        style_multiplier = self.learning_style_preferences.get(
            learning_style, {}
        ).get(content_type, 1.0)
        score *= style_multiplier
        
        # Difficulty appropriateness
        student_level = student_profile.current_level
        content_level = content.difficulty_level
        
        level_compatibility = {
            ('beginner', 'beginner'): 1.0,
            ('beginner', 'intermediate'): 0.6,
            ('beginner', 'advanced'): 0.2,
            ('intermediate', 'beginner'): 0.7,
            ('intermediate', 'intermediate'): 1.0,
            ('intermediate', 'advanced'): 0.8,
            ('advanced', 'beginner'): 0.5,
            ('advanced', 'intermediate'): 0.9,
            ('advanced', 'advanced'): 1.0
        }
        
        level_multiplier = level_compatibility.get((student_level, content_level), 0.5)
        score *= level_multiplier
        
        # Subject interest
        if content.subject in student_profile.subjects_of_interest:
            score *= 1.2
        
        # Time preference (favor shorter content for slow learners)
        if student_profile.learning_pace == 'slow' and content.estimated_time > 30:
            score *= 0.8
        elif student_profile.learning_pace == 'fast' and content.estimated_time < 10:
            score *= 0.9
        
        return score
    
    def _apply_difficulty_progression(self, 
                                    content_items: List[ContentChunk],
                                    student_profile: StudentProfile) -> List[ContentChunk]:
        """Apply optimal difficulty progression"""
        if not content_items:
            return []
        
        # Group by difficulty
        difficulty_groups = {
            'beginner': [],
            'intermediate': [],
            'advanced': []
        }
        
        for content in content_items:
            difficulty_groups[content.difficulty_level].append(content)
        
        # Create progression based on student level and pace
        progression = []
        student_level = student_profile.current_level
        
        if student_level == 'beginner':
            progression.extend(difficulty_groups['beginner'][:3])  # Start with 3 beginner items
            progression.extend(difficulty_groups['intermediate'][:2])  # Add 2 intermediate
            progression.extend(difficulty_groups['beginner'][3:5])  # More beginner for reinforcement
            progression.extend(difficulty_groups['intermediate'][2:])  # Rest of intermediate
            progression.extend(difficulty_groups['advanced'][:2])  # Touch on advanced
            
        elif student_level == 'intermediate':
            progression.extend(difficulty_groups['beginner'][:1])  # Quick review
            progression.extend(difficulty_groups['intermediate'][:4])  # Focus on intermediate
            progression.extend(difficulty_groups['advanced'][:2])  # Introduce advanced
            progression.extend(difficulty_groups['intermediate'][4:])  # More intermediate
            progression.extend(difficulty_groups['advanced'][2:])  # More advanced
            
        else:  # advanced
            progression.extend(difficulty_groups['intermediate'][:2])  # Quick review
            progression.extend(difficulty_groups['advanced'])  # Focus on advanced
            progression.extend(difficulty_groups['intermediate'][2:])  # Fill with intermediate
        
        return progression[:20]  # Limit path length

class LearningPathGenerator:
    """Main class for generating personalized learning paths"""
    
    def __init__(self):
        self.prerequisite_graph = PrerequisiteGraph()
        self.optimizer = PathOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def build_prerequisite_graph(self, content_chunks: List[ContentChunk]):
        """Build the prerequisite graph from content chunks"""
        self.logger.info(f"Building prerequisite graph from {len(content_chunks)} content items")
        
        for chunk in content_chunks:
            self.prerequisite_graph.add_content(chunk)
        
        self.logger.info(f"Graph built with {self.prerequisite_graph.graph.number_of_nodes()} nodes and {self.prerequisite_graph.graph.number_of_edges()} edges")
    
    def generate_learning_path(self,
                             student_profile: StudentProfile,
                             target_subject: str,
                             target_objectives: List[str],
                             available_content: List[ContentChunk],
                             max_path_length: int = 20) -> LearningPath:
        """Generate a personalized learning path"""
        
        self.logger.info(f"Generating learning path for {student_profile.name}")
        self.logger.info(f"Target: {target_subject}, Objectives: {target_objectives}")
        
        # Filter content by subject
        subject_content = [
            chunk for chunk in available_content 
            if chunk.subject.lower() == target_subject.lower()
        ]
        
        if not subject_content:
            self.logger.warning(f"No content found for subject: {target_subject}")
            return self._create_empty_path(student_profile, target_subject, target_objectives)
        
        # Optimize content sequence
        optimized_content = self.optimizer.optimize_content_sequence(
            subject_content, student_profile, target_objectives
        )
        
        # Apply prerequisite ordering
        content_ids = [chunk.content_id for chunk in optimized_content]
        ordered_ids = self.prerequisite_graph.topological_sort(content_ids)
        
        # Reorder content based on prerequisites
        ordered_content = []
        content_dict = {chunk.content_id: chunk for chunk in optimized_content}
        
        for content_id in ordered_ids:
            if content_id in content_dict:
                ordered_content.append(content_dict[content_id])
        
        # Add any remaining content not in the graph
        for chunk in optimized_content:
            if chunk not in ordered_content:
                ordered_content.append(chunk)
        
        # Limit path length
        final_content = ordered_content[:max_path_length]
        
        # Create learning path items
        path_items = []
        total_time = 0
        
        for i, content in enumerate(final_content):
            path_item = LearningPathItem(
                content_id=content.content_id,
                title=content.title,
                subject=content.subject,
                topic=content.topic,
                difficulty_level=content.difficulty_level,
                learning_objectives=content.learning_objectives,
                prerequisites=content.prerequisites,
                content_type=content.content_type,
                estimated_time=content.estimated_time,
                order_index=i,
                is_checkpoint=self._is_checkpoint(i, len(final_content))
            )
            path_items.append(path_item)
            total_time += content.estimated_time
        
        # Create milestones
        milestones = self._create_milestones(path_items, target_objectives)
        
        # Determine difficulty progression
        difficulty_progression = self._analyze_difficulty_progression(path_items)
        
        # Create learning path
        path_id = f"path_{student_profile.student_id}_{int(datetime.now().timestamp())}"
        
        learning_path = LearningPath(
            path_id=path_id,
            student_id=student_profile.student_id,
            target_subject=target_subject,
            target_objectives=target_objectives,
            difficulty_progression=difficulty_progression,
            estimated_total_time=total_time,
            path_items=path_items,
            milestones=milestones,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.logger.info(f"Generated learning path with {len(path_items)} items, estimated time: {total_time} minutes")
        
        return learning_path
    
    def _create_empty_path(self, student_profile: StudentProfile, 
                          target_subject: str, target_objectives: List[str]) -> LearningPath:
        """Create an empty learning path when no content is available"""
        path_id = f"empty_path_{student_profile.student_id}_{int(datetime.now().timestamp())}"
        
        return LearningPath(
            path_id=path_id,
            student_id=student_profile.student_id,
            target_subject=target_subject,
            target_objectives=target_objectives,
            difficulty_progression="gradual",
            estimated_total_time=0,
            path_items=[],
            milestones=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def _is_checkpoint(self, index: int, total_items: int) -> bool:
        """Determine if an item should be a checkpoint"""
        # Create checkpoints at 25%, 50%, 75% and end
        checkpoints = [int(total_items * p) for p in [0.25, 0.5, 0.75, 1.0]]
        return index + 1 in checkpoints
    
    def _create_milestones(self, path_items: List[LearningPathItem], 
                          target_objectives: List[str]) -> List[Dict[str, Any]]:
        """Create learning milestones"""
        milestones = []
        
        # Group objectives by checkpoint
        checkpoint_indices = [i for i, item in enumerate(path_items) if item.is_checkpoint]
        
        if not checkpoint_indices:
            return milestones
        
        objectives_per_milestone = len(target_objectives) // len(checkpoint_indices)
        
        for i, checkpoint_idx in enumerate(checkpoint_indices):
            start_obj_idx = i * objectives_per_milestone
            end_obj_idx = min((i + 1) * objectives_per_milestone, len(target_objectives))
            
            milestone_objectives = target_objectives[start_obj_idx:end_obj_idx]
            
            milestone = {
                'milestone_id': f"milestone_{i + 1}",
                'title': f"Milestone {i + 1}",
                'description': f"Complete {checkpoint_idx + 1} learning items",
                'objectives': milestone_objectives,
                'target_item_index': checkpoint_idx,
                'completion_criteria': {
                    'min_items_completed': checkpoint_idx + 1,
                    'min_average_score': 0.7
                }
            }
            milestones.append(milestone)
        
        return milestones
    
    def _analyze_difficulty_progression(self, path_items: List[LearningPathItem]) -> str:
        """Analyze the difficulty progression of the path"""
        if not path_items:
            return "gradual"
        
        difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        difficulties = [difficulty_map.get(item.difficulty_level, 2) for item in path_items]
        
        # Calculate progression rate
        if len(difficulties) < 2:
            return "gradual"
        
        # Calculate average increase per step
        increases = [difficulties[i] - difficulties[i-1] for i in range(1, len(difficulties))]
        avg_increase = np.mean(increases) if increases else 0
        
        if avg_increase <= 0.1:
            return "gradual"
        elif avg_increase <= 0.2:
            return "moderate"
        else:
            return "steep"
    
    def update_path_progress(self, learning_path: LearningPath, 
                           completed_item_index: int, mastery_score: float) -> LearningPath:
        """Update learning path progress"""
        if completed_item_index < len(learning_path.path_items):
            item = learning_path.path_items[completed_item_index]
            item.completion_status = "completed"
            item.mastery_score = mastery_score
            
            # Update current item index
            learning_path.current_item_index = min(
                completed_item_index + 1, 
                len(learning_path.path_items) - 1
            )
            
            # Calculate completion percentage
            completed_items = sum(
                1 for item in learning_path.path_items 
                if item.completion_status == "completed"
            )
            learning_path.completion_percentage = (completed_items / len(learning_path.path_items)) * 100
            
            learning_path.updated_at = datetime.now().isoformat()
        
        return learning_path
    
    def adaptive_path_adjustment(self, learning_path: LearningPath, 
                               student_performance: Dict[str, float],
                               available_content: List[ContentChunk]) -> LearningPath:
        """Adaptively adjust learning path based on student performance"""
        
        # Analyze performance trends
        recent_scores = list(student_performance.values())[-5:]  # Last 5 scores
        avg_recent_score = np.mean(recent_scores) if recent_scores else 0.7
        
        # If performance is low, add remedial content
        if avg_recent_score < 0.6:
            self._add_remedial_content(learning_path, available_content)
        
        # If performance is high, potentially skip or accelerate
        elif avg_recent_score > 0.9:
            self._optimize_for_acceleration(learning_path)
        
        learning_path.updated_at = datetime.now().isoformat()
        return learning_path
    
    def _add_remedial_content(self, learning_path: LearningPath, 
                            available_content: List[ContentChunk]):
        """Add remedial content for struggling students"""
        current_index = learning_path.current_item_index
        
        if current_index < len(learning_path.path_items):
            current_item = learning_path.path_items[current_index]
            
            # Find easier content on the same topic
            remedial_content = [
                chunk for chunk in available_content
                if (chunk.subject == current_item.subject and
                    chunk.topic == current_item.topic and
                    chunk.difficulty_level == 'beginner' and
                    chunk.content_id not in [item.content_id for item in learning_path.path_items])
            ]
            
            # Insert remedial content
            for i, content in enumerate(remedial_content[:2]):  # Add max 2 remedial items
                remedial_item = LearningPathItem(
                    content_id=content.content_id,
                    title=f"Review: {content.title}",
                    subject=content.subject,
                    topic=content.topic,
                    difficulty_level=content.difficulty_level,
                    learning_objectives=content.learning_objectives,
                    prerequisites=content.prerequisites,
                    content_type=content.content_type,
                    estimated_time=content.estimated_time,
                    order_index=current_index + i + 0.5,  # Insert between items
                    is_checkpoint=False
                )
                learning_path.path_items.insert(current_index + i + 1, remedial_item)
            
            # Reorder indices
            for i, item in enumerate(learning_path.path_items):
                item.order_index = i
    
    def _optimize_for_acceleration(self, learning_path: LearningPath):
        """Optimize path for high-performing students"""
        # Skip some beginner content or mark as optional
        for item in learning_path.path_items[learning_path.current_item_index:]:
            if (item.difficulty_level == 'beginner' and 
                item.completion_status == 'not_started'):
                # Mark as optional rather than removing
                if 'optional' not in item.title.lower():
                    item.title = f"Optional: {item.title}"

# Utility functions
def save_learning_path(learning_path: LearningPath, file_path: str):
    """Save learning path to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(learning_path), f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving learning path: {e}")

def load_learning_path(file_path: str) -> Optional[LearningPath]:
    """Load learning path from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert path_items back to LearningPathItem objects
        path_items = [LearningPathItem(**item) for item in data['path_items']]
        data['path_items'] = path_items
        
        return LearningPath(**data)
    except Exception as e:
        logging.error(f"Error loading learning path: {e}")
        return None

# Example usage
if __name__ == "__main__":
    from app.config import config, SAMPLE_CONTENT
    from .student_profile import StudentProfile, StudentProfileManager
    from .data_processor import ContentProcessor, ContentChunk
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample content chunks
    processor = ContentProcessor()
    sample_chunks = []
    
    for content_item in SAMPLE_CONTENT:
        chunks = processor.chunk_content(content_item)
        sample_chunks.extend(chunks)
    
    # Create sample student profile
    sample_profile = StudentProfile(
        student_id="student_001",
        name="John Doe",
        email="john@example.com",
        current_level="beginner",
        learning_style="visual",
        subjects_of_interest=["Mathematics", "Programming"],
        learning_pace="medium",
        completed_content=[],
        knowledge_gaps=[],
        performance_history={},
        preferences={},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Generate learning path
    path_generator = LearningPathGenerator()
    path_generator.build_prerequisite_graph(sample_chunks)
    
    learning_path = path_generator.generate_learning_path(
        student_profile=sample_profile,
        target_subject="Mathematics",
        target_objectives=["Understanding variables", "Basic algebraic expressions"],
        available_content=sample_chunks
    )
    
    # Display results
    print(f"\nGenerated Learning Path for {sample_profile.name}")
    print(f"Subject: {learning_path.target_subject}")
    print(f"Total Items: {len(learning_path.path_items)}")
    print(f"Estimated Time: {learning_path.estimated_total_time} minutes")
    print(f"Difficulty Progression: {learning_path.difficulty_progression}")
    
    print("\nPath Items:")
    for item in learning_path.path_items:
        checkpoint_indicator = " 🎯" if item.is_checkpoint else ""
        print(f"  {item.order_index + 1}. [{item.difficulty_level}] {item.title}{checkpoint_indicator}")
        print(f"     Time: {item.estimated_time}min, Type: {item.content_type}")
    
    print(f"\nMilestones: {len(learning_path.milestones)}")
    for milestone in learning_path.milestones:
        print(f"  - {milestone['title']}: {milestone['description']}")