"""
Main Streamlit application for Educational RAG System
"""
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import config, create_directories, SAMPLE_CONTENT
from app.components.data_processor import ContentProcessor, ContentChunk
from app.components.embeddings import EmbeddingGenerator
from app.components.retriever import EducationalRetriever
from app.components.student_profile import (
    StudentProfileManager, LearningStyleAssessment, StudentProfile
)
from app.components.learning_path import LearningPathGenerator, save_learning_path, load_learning_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .learning-path-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .checkpoint-item {
        border-left: 4px solid #28a745;
        background-color: #f8fff9;
    }
    .completed-item {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
    }
    .current-item {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class EducationalRAGApp:
    """Main application class"""
    
    def __init__(self):
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            # Create necessary directories
            create_directories()
            
            # Initialize session state
            if 'system_initialized' not in st.session_state:
                with st.spinner('Initializing Educational RAG System...'):
                    self._init_session_state()
                    self._load_or_create_sample_data()
                st.session_state.system_initialized = True
                st.success("System initialized successfully!")
        
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            logger.error(f"System initialization error: {e}")
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'current_student' not in st.session_state:
            st.session_state.current_student = None
        
        if 'current_path' not in st.session_state:
            st.session_state.current_path = None
        
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        
        # Initialize components
        st.session_state.profile_manager = StudentProfileManager(
            config.MODELS_DIR / "student_profiles.db"
        )
        
        st.session_state.retriever = EducationalRetriever(
            config.VECTOR_DB_DIR,
            config.CHROMA_COLLECTION_NAME
        )
        
        st.session_state.path_generator = LearningPathGenerator()
        
        st.session_state.assessment = LearningStyleAssessment()
    
    def _load_or_create_sample_data(self):
        """Load or create sample educational content"""
        # Check if we have content in the vector database
        if st.session_state.retriever.collection.count() == 0:
            st.info("Loading sample educational content...")
            
            # Process sample content
            processor = ContentProcessor()
            chunks = []
            
            for content_item in SAMPLE_CONTENT:
                item_chunks = processor.chunk_content(content_item)
                chunks.extend(item_chunks)
            
            # Add to vector database
            st.session_state.retriever.add_chunks_to_database(chunks)
            
            # Build prerequisite graph
            st.session_state.path_generator.build_prerequisite_graph(chunks)
            
            # Store chunks in session state for path generation
            st.session_state.content_chunks = chunks
            
            st.success(f"Loaded {len(chunks)} content chunks from {len(SAMPLE_CONTENT)} source documents")
        else:
            # Load existing content for path generation
            if 'content_chunks' not in st.session_state:
                # This is a simplified version - in production, you'd load from database
                processor = ContentProcessor()
                chunks = []
                for content_item in SAMPLE_CONTENT:
                    item_chunks = processor.chunk_content(content_item)
                    chunks.extend(item_chunks)
                st.session_state.content_chunks = chunks
                st.session_state.path_generator.build_prerequisite_graph(chunks)
    
    def run(self):
        """Run the main application"""
        # Main header
        st.markdown('<div class="main-header">🎓 Educational Learning Assistant</div>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown(f'<div class="section-header">{config.SIDEBAR_TITLE}</div>', 
                       unsafe_allow_html=True)
            
            page = st.radio("Choose a page:", [
                "🏠 Home",
                "👤 Student Profile", 
                "🔍 Content Search",
                "🛤️ Learning Path",
                "📊 Progress Dashboard",
                "⚙️ System Admin"
            ])
        
        # Route to appropriate page
        if page == "🏠 Home":
            self.show_home_page()
        elif page == "👤 Student Profile":
            self.show_student_profile_page()
        elif page == "🔍 Content Search":
            self.show_content_search_page()
        elif page == "🛤️ Learning Path":
            self.show_learning_path_page()
        elif page == "📊 Progress Dashboard":
            self.show_progress_dashboard()
        elif page == "⚙️ System Admin":
            self.show_admin_page()
    
    def show_home_page(self):
        """Display the home page"""
        st.markdown('<div class="section-header">Welcome to Your Personal Learning Assistant</div>', 
                   unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_content = st.session_state.retriever.collection.count()
            st.metric("Total Content Items", total_content)
        
        with col2:
            current_student = st.session_state.current_student
            student_level = current_student.current_level if current_student else "Not Set"
            st.metric("Current Level", student_level)
        
        with col3:
            study_time = current_student.total_study_time if current_student else 0
            st.metric("Total Study Time", f"{study_time} min")
        
        with col4:
            streak = current_student.streak_days if current_student else 0
            st.metric("Learning Streak", f"{streak} days")
        
        # Quick actions
        st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Create Learning Path", use_container_width=True):
                st.switch_page("Learning Path")
        
        with col2:
            if st.button("🔍 Search Content", use_container_width=True):
                st.switch_page("Content Search")
        
        with col3:
            if st.button("📊 View Progress", use_container_width=True):
                st.switch_page("Progress Dashboard")
        
        # System overview
        if current_student:
            st.markdown('<div class="section-header">Your Learning Overview</div>', 
                       unsafe_allow_html=True)
            
            # Recent activity
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Learning Style Profile")
                style_desc = st.session_state.assessment.get_learning_style_description(
                    st.session_state.assessment.scoring_map.get('A')  # Default to visual
                )
                if current_student.learning_style:
                    for style in st.session_state.assessment.scoring_map.values():
                        if style.value == current_student.learning_style:
                            style_desc = st.session_state.assessment.get_learning_style_description(style)
                            break
                
                st.write(f"**{style_desc.get('name', 'Unknown')}**")
                st.write(style_desc.get('description', ''))
                
                if style_desc.get('learning_tips'):
                    st.write("**Tips for you:**")
                    for tip in style_desc['learning_tips'][:3]:
                        st.write(f"• {tip}")
            
            with col2:
                st.subheader("Subjects of Interest")
                if current_student.subjects_of_interest:
                    for subject in current_student.subjects_of_interest:
                        st.write(f"• {subject}")
                else:
                    st.write("No subjects selected yet")
        else:
            st.info("👋 Welcome! Please create or select a student profile to get started.")
    
    def show_student_profile_page(self):
        """Display student profile management page"""
        st.markdown('<div class="section-header">Student Profile Management</div>', 
                   unsafe_allow_html=True)
        
        # Profile selection/creation tabs
        tab1, tab2, tab3 = st.tabs(["Current Profile", "Create New Profile", "Learning Style Assessment"])
        
        with tab1:
            if st.session_state.current_student:
                self._display_current_profile()
            else:
                st.info("No student profile selected. Create a new profile or load an existing one.")
        
        with tab2:
            self._show_profile_creation_form()
        
        with tab3:
            self._show_learning_style_assessment()
    
    def _display_current_profile(self):
        """Display current student profile information"""
        student = st.session_state.current_student
        
        st.subheader(f"Profile: {student.name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Email:** {student.email}")
            st.write(f"**Current Level:** {student.current_level}")
            st.write(f"**Learning Style:** {student.learning_style}")
            st.write(f"**Learning Pace:** {student.learning_pace}")
        
        with col2:
            st.write(f"**Total Study Time:** {student.total_study_time} minutes")
            st.write(f"**Streak Days:** {student.streak_days}")
            st.write(f"**Completed Content:** {len(student.completed_content)} items")
            st.write(f"**Knowledge Gaps:** {len(student.knowledge_gaps)} identified")
        
        # Subjects of interest
        st.subheader("Subjects of Interest")
        if student.subjects_of_interest:
            cols = st.columns(len(student.subjects_of_interest))
            for i, subject in enumerate(student.subjects_of_interest):
                cols[i].write(f"🎯 {subject}")
        else:
            st.write("No subjects selected")
        
        # Edit profile button
        if st.button("Edit Profile"):
            st.session_state.editing_profile = True
            st.rerun()
    
    def _show_profile_creation_form(self):
        """Show form for creating new student profile"""
        st.subheader("Create New Student Profile")
        
        with st.form("create_profile_form"):
            name = st.text_input("Full Name*", placeholder="Enter your full name")
            email = st.text_input("Email*", placeholder="Enter your email address")
            
            col1, col2 = st.columns(2)
            
            with col1:
                current_level = st.selectbox("Current Level", config.DIFFICULTY_LEVELS)
                learning_pace = st.selectbox("Learning Pace", config.LEARNING_PACES)
            
            with col2:
                learning_style = st.selectbox("Learning Style", config.LEARNING_STYLES)
                subjects = st.multiselect("Subjects of Interest", config.SUBJECTS)
            
            submitted = st.form_submit_button("Create Profile")
            
            if submitted:
                if name and email:
                    # Create student profile
                    student_id = f"student_{int(datetime.now().timestamp())}"
                    
                    try:
                        profile = st.session_state.profile_manager.create_student_profile(
                            student_id=student_id,
                            name=name,
                            email=email
                        )
                        
                        # Update profile with additional info
                        profile.current_level = current_level
                        profile.learning_style = learning_style
                        profile.learning_pace = learning_pace
                        profile.subjects_of_interest = subjects
                        
                        # Save updated profile
                        st.session_state.profile_manager._save_profile_to_db(profile)
                        
                        # Set as current student
                        st.session_state.current_student = profile
                        
                        st.success(f"Profile created successfully for {name}!")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error creating profile: {e}")
                else:
                    st.error("Please fill in all required fields (marked with *)")
    
    def _show_learning_style_assessment(self):
        """Show learning style assessment questionnaire"""
        st.subheader("Learning Style Assessment")
        st.write("Answer these questions to determine your optimal learning style:")
        
        with st.form("learning_style_assessment"):
            responses = []
            
            for question in st.session_state.assessment.questions:
                st.write(f"**{question['question']}**")
                
                response = st.radio(
                    f"Question {question['id']}",
                    list(question['options'].keys()),
                    format_func=lambda x: f"{x}. {question['options'][x]}",
                    key=f"q_{question['id']}",
                    label_visibility="collapsed"
                )
                responses.append(response)
            
            submitted = st.form_submit_button("Get My Learning Style")
            
            if submitted:
                # Calculate learning style
                learning_style, scores = st.session_state.assessment.calculate_learning_style(responses)
                
                st.success("Assessment Complete!")
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Your Learning Style")
                    style_desc = st.session_state.assessment.get_learning_style_description(learning_style)
                    st.write(f"**{style_desc['name']}**")
                    st.write(style_desc['description'])
                
                with col2:
                    st.subheader("Score Breakdown")
                    score_df = pd.DataFrame(list(scores.items()), columns=['Style', 'Score'])
                    fig = px.bar(score_df, x='Style', y='Score', title='Learning Style Scores')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Learning tips
                st.subheader("Personalized Learning Tips")
                for tip in style_desc.get('learning_tips', []):
                    st.write(f"• {tip}")
                
                # Update current student profile if exists
                if st.session_state.current_student:
                    if st.button("Update My Profile with These Results"):
                        st.session_state.current_student.learning_style = learning_style.value
                        st.session_state.profile_manager._save_profile_to_db(st.session_state.current_student)
                        st.success("Profile updated with new learning style!")
                        st.rerun()
    
    def show_content_search_page(self):
        """Display content search and discovery page"""
        st.markdown('<div class="section-header">Content Search & Discovery</div>', 
                   unsafe_allow_html=True)
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search for educational content:",
                placeholder="e.g., 'explain basic algebra', 'python variables', 'introduction to calculus'"
            )
        
        with col2:
            search_button = st.button("🔍 Search", use_container_width=True)
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                subject_filter = st.selectbox("Subject", ["All"] + config.SUBJECTS)
                difficulty_filter = st.selectbox("Difficulty", ["All"] + config.DIFFICULTY_LEVELS)
            
            with col2:
                content_type_filter = st.selectbox("Content Type", ["All"] + config.CONTENT_TYPES)
                max_time_filter = st.slider("Max Time (minutes)", 5, 120, 60)
            
            with col3:
                if st.session_state.current_student:
                    personalized = st.checkbox("Personalized Results", value=True)
                else:
                    personalized = False
                    st.info("Create a profile for personalized results")
        
        # Perform search
        if search_button and search_query:
            with st.spinner("Searching content..."):
                # Build filters
                filters = {}
                if subject_filter != "All":
                    filters['subject'] = subject_filter
                if difficulty_filter != "All":
                    filters['difficulty_level'] = difficulty_filter
                if content_type_filter != "All":
                    filters['content_type'] = content_type_filter
                filters['max_time'] = max_time_filter
                
                # Get student profile for personalization
                student_profile = st.session_state.current_student if personalized else None
                
                # Search
                results = st.session_state.retriever.semantic_search(
                    query=search_query,
                    top_k=10,
                    student_profile=student_profile.__dict__ if student_profile else None,
                    filters=filters
                )
                
                st.session_state.search_results = results
        
        # Display results
        if st.session_state.search_results:
            st.subheader(f"Search Results ({len(st.session_state.search_results)} found)")
            
            for i, result in enumerate(st.session_state.search_results):
                with st.container():
                    st.markdown(f"""
                    <div class="learning-path-item">
                        <h4>{result['metadata']['title']}</h4>
                        <p><strong>Subject:</strong> {result['metadata']['subject']} | 
                           <strong>Difficulty:</strong> {result['metadata']['difficulty_level']} | 
                           <strong>Type:</strong> {result['metadata']['content_type']} | 
                           <strong>Time:</strong> {result['metadata']['estimated_time']} min</p>
                        <p><strong>Relevance Score:</strong> {result['similarity_score']:.3f}</p>
                        <p>{result['content'][:300]}{'...' if len(result['content']) > 300 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button(f"Study Now", key=f"study_{i}"):
                            st.info("Study mode would open here (not implemented in demo)")
                    
                    with col2:
                        if st.button(f"Add to Path", key=f"add_path_{i}"):
                            st.info("Would add to learning path (not implemented in demo)")
        
        # Quick search suggestions
        if not st.session_state.search_results:
            st.subheader("Popular Search Topics")
            col1, col2, col3 = st.columns(3)
            
            suggestions = [
                "Basic algebra concepts", "Python programming", "Introduction to calculus",
                "Scientific method", "World history", "Grammar fundamentals",
                "Statistics basics", "Chemistry elements", "Art techniques"
            ]
            
            for i, suggestion in enumerate(suggestions):
                col = [col1, col2, col3][i % 3]
                if col.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.search_query = suggestion
                    st.rerun()
    
    def show_learning_path_page(self):
        """Display learning path generation and management page"""
        st.markdown('<div class="section-header">Learning Path Management</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.current_student:
            st.warning("Please create a student profile first to generate learning paths.")
            return
        
        # Tabs for different path operations
        tab1, tab2, tab3 = st.tabs(["Generate New Path", "Current Path", "Path History"])
        
        with tab1:
            self._show_path_generation_form()
        
        with tab2:
            self._show_current_learning_path()
        
        with tab3:
            self._show_path_history()
    
    def _show_path_generation_form(self):
        """Show form for generating new learning paths"""
        st.subheader("Generate Personalized Learning Path")
        
        with st.form("generate_path_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_subject = st.selectbox("Target Subject", config.SUBJECTS)
                path_length = st.slider("Maximum Path Length", 5, 30, 15)
            
            with col2:
                difficulty_progression = st.selectbox(
                    "Difficulty Progression", 
                    ["gradual", "moderate", "steep"],
                    help="How quickly should the difficulty increase?"
                )
            
            # Learning objectives
            st.subheader("Learning Objectives")
            objectives_text = st.text_area(
                "Enter your learning objectives (one per line):",
                placeholder="Understanding variables\nBasic algebraic expressions\nSolving simple equations",
                help="What do you want to learn or master?"
            )
            
            submitted = st.form_submit_button("Generate Learning Path")
            
            if submitted:
                if objectives_text.strip():
                    objectives = [obj.strip() for obj in objectives_text.split('\n') if obj.strip()]
                    
                    with st.spinner("Generating personalized learning path..."):
                        try:
                            # Generate learning path
                            learning_path = st.session_state.path_generator.generate_learning_path(
                                student_profile=st.session_state.current_student,
                                target_subject=target_subject,
                                target_objectives=objectives,
                                available_content=st.session_state.content_chunks,
                                max_path_length=path_length
                            )
                            
                            # Save path
                            st.session_state.current_path = learning_path
                            
                            # Save to file
                            path_file = config.MODELS_DIR / f"path_{learning_path.path_id}.json"
                            save_learning_path(learning_path, str(path_file))
                            
                            st.success(f"Learning path generated with {len(learning_path.path_items)} items!")
                            st.info(f"Estimated total time: {learning_path.estimated_total_time} minutes")
                            
                            # Show path preview
                            self._display_path_preview(learning_path)
                        
                        except Exception as e:
                            st.error(f"Error generating learning path: {e}")
                            logger.error(f"Path generation error: {e}")
                else:
                    st.error("Please enter at least one learning objective.")
    
    def _show_current_learning_path(self):
        """Display current learning path with progress tracking"""
        if not st.session_state.current_path:
            st.info("No active learning path. Generate a new path to get started.")
            return
        
        path = st.session_state.current_path
        
        # Path overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", len(path.path_items))
        
        with col2:
            st.metric("Completed", 
                     sum(1 for item in path.path_items if item.completion_status == "completed"))
        
        with col3:
            st.metric("Progress", f"{path.completion_percentage:.1f}%")
        
        with col4:
            st.metric("Est. Time", f"{path.estimated_total_time} min")
        
        # Progress bar
        progress_bar = st.progress(path.completion_percentage / 100)
        
        # Learning path items
        st.subheader("Learning Path Items")
        
        for i, item in enumerate(path.path_items):
            # Determine item style
            item_class = "learning-path-item"
            if item.completion_status == "completed":
                item_class += " completed-item"
            elif i == path.current_item_index:
                item_class += " current-item"
            elif item.is_checkpoint:
                item_class += " checkpoint-item"
            
            # Display item
            st.markdown(f"""
            <div class="{item_class}">
                <h4>{item.order_index + 1}. {item.title} {"🎯" if item.is_checkpoint else ""}</h4>
                <p><strong>Subject:</strong> {item.subject} | 
                   <strong>Difficulty:</strong> {item.difficulty_level} | 
                   <strong>Type:</strong> {item.content_type} | 
                   <strong>Time:</strong> {item.estimated_time} min</p>
                <p><strong>Objectives:</strong> {', '.join(item.learning_objectives[:3])}</p>
                <p><strong>Status:</strong> {item.completion_status.replace('_', ' ').title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons for current/next items
            if item.completion_status != "completed":
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button(f"Start Learning", key=f"start_{i}"):
                        st.info("Learning interface would open here")
                
                with col2:
                    if st.button(f"Mark Complete", key=f"complete_{i}"):
                        # Simulate completion
                        item.completion_status = "completed"
                        item.mastery_score = 0.8  # Simulated score
                        
                        # Update path progress
                        updated_path = st.session_state.path_generator.update_path_progress(
                            path, i, 0.8
                        )
                        st.session_state.current_path = updated_path
                        
                        # Update student progress
                        st.session_state.profile_manager.update_student_progress(
                            st.session_state.current_student.student_id,
                            item.content_id,
                            0.8,
                            item.estimated_time
                        )
                        
                        st.success("Item marked as completed!")
                        st.rerun()
        
        # Milestones
        if path.milestones:
            st.subheader("Learning Milestones")
            for milestone in path.milestones:
                completed_items = sum(1 for item in path.path_items[:milestone['target_item_index'] + 1] 
                                   if item.completion_status == "completed")
                target_items = milestone['target_item_index'] + 1
                milestone_progress = (completed_items / target_items) * 100
                
                st.write(f"**{milestone['title']}**")
                st.write(f"Progress: {milestone_progress:.1f}% ({completed_items}/{target_items} items)")
                st.progress(milestone_progress / 100)
                st.write(f"Objectives: {', '.join(milestone['objectives'])}")
                st.write("---")
    
    def _display_path_preview(self, learning_path):
        """Display a preview of the generated learning path"""
        st.subheader("Learning Path Preview")
        
        # Path summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Subject:** {learning_path.target_subject}")
            st.write(f"**Total Items:** {len(learning_path.path_items)}")
        
        with col2:
            st.write(f"**Estimated Time:** {learning_path.estimated_total_time} minutes")
            st.write(f"**Difficulty Progression:** {learning_path.difficulty_progression}")
        
        with col3:
            st.write(f"**Milestones:** {len(learning_path.milestones)}")
            st.write(f"**Target Objectives:** {len(learning_path.target_objectives)}")
        
        # First few items preview
        st.write("**First 5 Items:**")
        for i, item in enumerate(learning_path.path_items[:5]):
            checkpoint_indicator = "🎯 " if item.is_checkpoint else ""
            st.write(f"{i+1}. {checkpoint_indicator}[{item.difficulty_level}] {item.title} ({item.estimated_time}min)")
    
    def _show_path_history(self):
        """Show learning path history"""
        st.subheader("Learning Path History")
        st.info("Path history feature would be implemented here, showing previously generated and completed paths.")
    
    def show_progress_dashboard(self):
        """Display student progress dashboard"""
        st.markdown('<div class="section-header">Progress Dashboard</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.current_student:
            st.warning("Please create a student profile to view progress.")
            return
        
        student = st.session_state.current_student
        
        # Overall progress metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Study Time", f"{student.total_study_time} min")
        
        with col2:
            st.metric("Learning Streak", f"{student.streak_days} days")
        
        with col3:
            st.metric("Content Completed", len(student.completed_content))
        
        with col4:
            avg_score = 0.8  # Placeholder
            st.metric("Average Score", f"{avg_score:.1%}")
        
        # Charts and visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Study time over time (mock data)
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            study_times = np.random.poisson(25, 30)  # Mock study times
            
            df = pd.DataFrame({'Date': dates, 'Study Time': study_times})
            fig = px.line(df, x='Date', y='Study Time', title='Daily Study Time')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Subject progress (mock data)
            subjects = ['Mathematics', 'Programming', 'Science']
            progress = [75, 60, 45]  # Mock progress percentages
            
            fig = px.bar(x=subjects, y=progress, title='Subject Progress')
            fig.update_layout(yaxis_title='Progress (%)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed progress
        st.subheader("Detailed Progress")
        
        if student.performance_history:
            # Convert performance history to dataframe for display
            performance_data = []
            for date, sessions in student.performance_history.items():
                for session in sessions:
                    performance_data.append({
                        'Date': date,
                        'Content ID': session['content_id'],
                        'Score': session['score'],
                        'Duration (min)': session['duration']
                    })
            
            if performance_data:
                df = pd.DataFrame(performance_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No learning activity recorded yet. Complete some content to see your progress!")
        
        # Recommendations
        st.subheader("Recommendations")
        analytics = st.session_state.profile_manager.get_student_analytics(student.student_id)
        recommendations = analytics.get('recommendations', [])
        
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.write("Complete some learning activities to get personalized recommendations!")
    
    def show_admin_page(self):
        """Display system administration page"""
        st.markdown('<div class="section-header">System Administration</div>', 
                   unsafe_allow_html=True)
        
        # System statistics
        st.subheader("System Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_content = st.session_state.retriever.collection.count()
            st.metric("Total Content Items", total_content)
        
        with col2:
            # This would count actual students in production
            st.metric("Total Students", 1 if st.session_state.current_student else 0)
        
        with col3:
            # This would count actual paths in production
            st.metric("Learning Paths Created", 1 if st.session_state.current_path else 0)
        
        # System actions
        st.subheader("System Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Vector Database", type="secondary"):
                if st.confirm("Are you sure? This will delete all content from the vector database."):
                    st.session_state.retriever.clear_collection()
                    st.success("Vector database cleared!")
        
        with col2:
            if st.button("Reload Sample Content"):
                st.session_state.retriever.clear_collection()
                self._load_or_create_sample_data()
                st.success("Sample content reloaded!")
        
        # Collection statistics
        st.subheader("Content Statistics")
        stats = st.session_state.retriever.get_collection_stats()
        
        if stats:
            for key, value in stats.items():
                if isinstance(value, dict):
                    st.write(f"**{key.replace('_', ' ').title()}:**")
                    for sub_key, sub_value in value.items():
                        st.write(f"  • {sub_key}: {sub_value}")
                else:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Database info
        st.subheader("Database Information")
        st.write(f"**Vector Database Path:** {config.VECTOR_DB_DIR}")
        st.write(f"**Student Profiles Database:** {config.MODELS_DIR / 'student_profiles.db'}")
        st.write(f"**Configuration:** {config.APP_NAME} v{config.APP_VERSION}")

# Main application entry point
def main():
    """Main application entry point"""
    try:
        app = EducationalRAGApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")
        
        # Show error details in development
        if config.DEBUG:
            st.exception(e)

if __name__ == "__main__":
    main()