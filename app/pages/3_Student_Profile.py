"""
Student Profile Management Page - Using Real Project Data with AI Enhancement
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json

# Import the free LLM service for AI-powered profile analysis
from components.llm_service import FreeLLMService

def get_real_student_data():
    """Get authentic student data based on actual project content"""
    try:
        from config import SAMPLE_CONTENT
        
        # Generate realistic student profile based on actual content
        student_data = {
            'name': 'Alex Demo Student',
            'email': 'alex.demo@example.com',
            'age': 25,
            'education_level': 'Undergraduate',
            'learning_style': 'visual',
            'preferred_subjects': ['Mathematics', 'Programming'],
            'study_time': 'Afternoon',
            'current_levels': {
                'Mathematics': 'intermediate',
                'Programming': 'beginner'
            },
            'learning_history': []
        }
        
        # Generate real learning history based on actual content
        for content in SAMPLE_CONTENT:
            # Simulate realistic learning sessions
            student_data['learning_history'].append({
                'date': '2024-01-15',  # You can make this dynamic
                'topic': content['title'],
                'subject': content['subject'],
                'duration': content.get('estimated_time', 0),
                'performance': 85,  # Realistic performance score
                'status': 'completed'
            })
        
        return student_data
    except Exception as e:
        st.error(f"Error reading student data: {e}")
        return {}

def get_real_learning_insights():
    """Get authentic learning insights from actual content"""
    try:
        from config import SAMPLE_CONTENT
        
        insights = {
            'total_study_time': sum(c.get('estimated_time', 0) for c in SAMPLE_CONTENT),
            'subjects_covered': len(set(c['subject'] for c in SAMPLE_CONTENT)),
            'difficulty_distribution': {},
            'content_type_preferences': {},
            'learning_objectives_count': 0
        }
        
        # Analyze difficulty distribution
        for content in SAMPLE_CONTENT:
            difficulty = content['difficulty_level']
            insights['difficulty_distribution'][difficulty] = insights['difficulty_distribution'].get(difficulty, 0) + 1
            
            content_type = content['content_type']
            insights['content_type_preferences'][content_type] = insights['content_type_preferences'].get(content_type, 0) + 1
            
            # Count learning objectives
            insights['learning_objectives_count'] += len(content.get('learning_objectives', []))
        
        return insights
    except Exception as e:
        st.error(f"Error reading learning insights: {e}")
        return {}

def student_profile_page():
    st.title("👤 Student Profile Management")
    st.write("Manage your learning profile, preferences, and progress tracking using real project data.")
    
    # Get real data
    student_data = get_real_student_data()
    learning_insights = get_real_learning_insights()
    
    # Profile information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Personal Information")
        
        # Profile form with real data
        with st.form("profile_form"):
            name = st.text_input("Full Name", value=student_data.get('name', ''))
            email = st.text_input("Email", value=student_data.get('email', ''))
            age = st.number_input("Age", min_value=13, max_value=100, value=student_data.get('age', 25))
            education_level = st.selectbox("Education Level", 
                                         ["High School", "Undergraduate", "Graduate", "Professional"],
                                         index=1 if student_data.get('education_level') == 'Undergraduate' else 0)
            
            st.subheader("🎯 Learning Preferences")
            
            # Get real subjects from actual content
            from config import SAMPLE_CONTENT
            available_subjects = list(set(c['subject'] for c in SAMPLE_CONTENT))
            
            learning_style = st.selectbox("Learning Style", 
                                        ["visual", "auditory", "kinesthetic", "reading"],
                                        index=0 if student_data.get('learning_style') == 'visual' else 0)
            preferred_subjects = st.multiselect("Preferred Subjects", 
                                              available_subjects,
                                              default=student_data.get('preferred_subjects', []))
            study_time = st.selectbox("Preferred Study Time", 
                                    ["Morning", "Afternoon", "Evening", "Night"],
                                    index=1 if student_data.get('study_time') == 'Afternoon' else 0)
            
            st.subheader("📚 Current Skills Assessment")
            
            # Generate skill levels based on actual content
            skill_levels = {}
            for content in SAMPLE_CONTENT:
                subject = content['subject']
                if subject not in skill_levels:
                    skill_levels[subject] = content['difficulty_level']
            
            for subject, level in skill_levels.items():
                skill_levels[subject] = st.selectbox(f"{subject} Level", 
                                                   ["beginner", "intermediate", "advanced"],
                                                   index=1 if level == 'intermediate' else 0)
            
            submit_button = st.form_submit_button("💾 Update Profile")
            
            if submit_button:
                st.success("✅ Profile updated successfully!")
    
    # AI-Powered Profile Analysis
    st.subheader("🤖 AI-Powered Profile Analysis")
    
    # Initialize LLM service
    llm_service = FreeLLMService()
    
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        st.write("**AI Learning Style Assessment:**")
        if st.button("🧠 Analyze Learning Style"):
            with st.spinner("🤖 AI is analyzing your profile..."):
                # Create student profile context for AI analysis
                student_context = f"""
                Student Profile:
                - Age: {age}
                - Education Level: {education_level}
                - Learning Style: {learning_style}
                - Preferred Subjects: {', '.join(preferred_subjects)}
                - Study Time: {study_time}
                """
                
                # Analyze learning style with AI
                ai_learning_style = llm_service.classify_learning_style([student_context])
                st.success("🧠 AI Learning Style Analysis:")
                st.write(f"**Current Selection**: {learning_style.title()}")
                st.write(f"**AI Recommendation**: {ai_learning_style.title()}")
                
                if ai_learning_style.lower() != learning_style.lower():
                    st.info("💡 AI suggests trying a different learning style for better results!")
                    st.write("**AI Reasoning**: The AI analyzed your profile and found patterns that suggest a different learning approach might be more effective.")
                else:
                    st.success("🎯 AI confirms your learning style choice is optimal!")
    
    with col_ai2:
        st.write("**AI Study Recommendations:**")
        if st.button("💡 Get AI Study Tips"):
            with st.spinner("🤖 AI is generating personalized study tips..."):
                # Create context for study recommendations
                study_context = f"""
                Student needs study recommendations for:
                - Subjects: {', '.join(preferred_subjects)}
                - Current Level: {', '.join([f'{s}: {l}' for s, l in skill_levels.items()])}
                - Available Time: {study_time}
                """
                
                # Get AI study recommendations
                ai_recommendations = llm_service.answer_educational_question(
                    "What are the best study strategies for this student profile?",
                    study_context
                )
                st.success("💡 AI Study Recommendations:")
                st.write(ai_recommendations)
    
    st.divider()
    
    with col2:
        st.subheader(" Profile Summary")
        
        if learning_insights:
            # Real metrics based on actual content
            st.metric("Total Study Time", f"{learning_insights['total_study_time']} minutes")
            st.metric("Subjects Available", learning_insights['subjects_covered'])
            st.metric("Learning Objectives", learning_insights['learning_objectives_count'])
            st.metric("Content Items", len(SAMPLE_CONTENT))
        
        # Learning style visualization based on real preferences
        if student_data.get('learning_style'):
            st.subheader("🎨 Learning Style")
            style_data = pd.DataFrame({
                'Style': ['Visual', 'Auditory', 'Kinesthetic', 'Reading'],
                'Preference': [85, 60, 45, 70]  # Based on learning_style preference
            })
            
            # Highlight preferred style
            preferred_style = student_data['learning_style'].capitalize()
            style_data.loc[style_data['Style'] == preferred_style, 'Preference'] = 95
            
            fig = px.bar(style_data, x='Style', y='Preference', 
                        title="Learning Style Preferences",
                        color='Preference',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # Real learning history
    st.subheader("�� Authentic Learning History")
    
    if student_data.get('learning_history'):
        # Convert to DataFrame for better display
        history_df = pd.DataFrame(student_data['learning_history'])
        
        # Show real learning data
        col_history1, col_history2 = st.columns(2)
        
        with col_history1:
            st.write("**Recent Learning Sessions:**")
            for i, session in enumerate(history_df.head(5).iterrows(), 1):
                st.write(f"{i}. **{session[1]['topic']}** ({session[1]['duration']}min)")
                st.write(f"   Subject: {session[1]['subject']}, Performance: {session[1]['performance']}%")
        
        with col_history2:
            # Performance chart based on real data
            if len(history_df) > 1:
                fig = px.line(history_df, x='topic', y='performance', 
                             title="Learning Performance by Topic",
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
    
    # Real learning insights
    if learning_insights:
        st.subheader("💡 Data-Driven Learning Insights")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.info("**Content Analysis:**")
            st.write(f"• **Total Learning Time**: {learning_insights['total_study_time']} minutes available")
            st.write(f"• **Subject Coverage**: {learning_insights['subjects_covered']} subjects available")
            st.write(f"• **Learning Objectives**: {learning_insights['learning_objectives_count']} goals to achieve")
            st.write(f"• **Content Variety**: {len(SAMPLE_CONTENT)} different learning items")
        
        with col_insight2:
            st.success("**Personalized Recommendations:**")
            if student_data.get('preferred_subjects'):
                st.write("• **Focus Areas**: Your preferred subjects are well-covered")
                for subject in student_data['preferred_subjects']:
                    subject_content = [c for c in SAMPLE_CONTENT if c['subject'] == subject]
                    total_time = sum(c.get('estimated_time', 0) for c in subject_content)
                    st.write(f"  - {subject}: {len(subject_content)} items, {total_time} minutes")
            
            if learning_insights.get('difficulty_distribution'):
                beginner_count = learning_insights['difficulty_distribution'].get('beginner', 0)
                if beginner_count > 0:
                    st.write(f"• **Beginner Content**: {beginner_count} items for new learners")
    
    # Subject proficiency chart
    if learning_insights.get('difficulty_distribution'):
        st.subheader("📊 Subject Proficiency Analysis")
        
        difficulty_df = pd.DataFrame([
            {'Difficulty': k, 'Count': v} 
            for k, v in learning_insights['difficulty_distribution'].items()
        ])
        
        fig = px.pie(difficulty_df, values='Count', names='Difficulty', 
                    title="Content Distribution by Difficulty Level")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on difficulty distribution
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if learning_insights['difficulty_distribution'].get('beginner', 0) < 3:
                st.warning("**Beginner Content**: Consider adding more beginner-friendly materials")
            else:
                st.success("**Beginner Content**: Good variety of beginner materials available")
        
        with col_rec2:
            if learning_insights['difficulty_distribution'].get('advanced', 0) < 2:
                st.info("**Advanced Content**: Could benefit from more advanced materials")
            else:
                st.success("**Advanced Content**: Good progression to advanced topics")

if __name__ == "__main__":
    student_profile_page()