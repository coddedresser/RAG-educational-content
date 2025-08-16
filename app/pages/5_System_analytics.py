"""
System Analytics and Performance Page - Using Real Project Data
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np
import os

def get_real_system_data():
    """Get authentic system data from actual project files"""
    try:
        # Get real file counts and sizes
        project_root = Path(__file__).parent.parent.parent
        app_dir = project_root / "app"
        
        # Count actual files
        python_files = list(app_dir.rglob("*.py"))
        component_files = list((app_dir / "components").rglob("*.py"))
        page_files = list((app_dir / "pages").rglob("*.py"))
        
        # Get actual file sizes
        total_size = sum(f.stat().st_size for f in python_files if f.exists())
        
        # Count lines of code (approximate)
        total_lines = 0
        for file in python_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                continue
        
        return {
            'python_files': len(python_files),
            'component_files': len(component_files),
            'page_files': len(page_files),
            'total_size_kb': total_size / 1024,
            'total_lines': total_lines,
            'project_root': project_root
        }
    except Exception as e:
        st.error(f"Error reading system data: {e}")
        return {}

def get_real_content_stats():
    """Get authentic content statistics from config"""
    try:
        from config import SAMPLE_CONTENT
        
        # Real content analysis
        subjects = {}
        difficulties = {}
        content_types = {}
        total_estimated_time = 0
        
        for content in SAMPLE_CONTENT:
            # Subject distribution
            subject = content.get('subject', 'Unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
            
            # Difficulty distribution
            difficulty = content.get('difficulty_level', 'Unknown')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # Content type distribution
            content_type = content.get('content_type', 'Unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Total time
            total_estimated_time += content.get('estimated_time', 0)
        
        # Get unique values for counts
        unique_subjects = len(set(content.get('subject', 'Unknown') for content in SAMPLE_CONTENT))
        unique_content_types = len(set(content.get('content_type', 'Unknown') for content in SAMPLE_CONTENT))
        unique_difficulties = len(set(content.get('difficulty_level', 'Unknown') for content in SAMPLE_CONTENT))
        
        return {
            'total_content': len(SAMPLE_CONTENT),
            'subjects': subjects,
            'difficulties': difficulties,
            'content_types': content_types,
            'total_estimated_time': total_estimated_time,
            'supported_subjects': unique_subjects,
            'content_types_available': unique_content_types,
            'difficulty_levels': unique_difficulties
        }
    except Exception as e:
        st.error(f"Error reading content stats: {e}")
        return {}

def get_real_file_structure():
    """Get authentic project file structure"""
    try:
        project_root = Path(__file__).parent.parent.parent
        app_dir = project_root / "app"
        
        file_structure = {}
        
        # Scan actual directory structure
        for item in app_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(project_root)
                file_type = item.suffix
                if file_type not in file_structure:
                    file_structure[file_type] = []
                file_structure[file_type].append(str(rel_path))
        
        return file_structure
    except Exception as e:
        st.error(f"Error reading file structure: {e}")
        return {}

def system_analytics_page():
    st.title(" System Analytics & Performance")
    st.write("Real-time system metrics and authentic project data analysis.")
    
    # Get real data
    system_data = get_real_system_data()
    content_stats = get_real_content_stats()
    file_structure = get_real_file_structure()
    
    # System overview with real data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Python Files", system_data.get('python_files', 0))
    with col2:
        st.metric("Total Lines of Code", f"{system_data.get('total_lines', 0):,}")
    with col3:
        st.metric("Project Size", f"{system_data.get('total_size_kb', 0):.1f} KB")
    with col4:
        st.metric("Components", system_data.get('component_files', 0))
    
    # Real content analytics
    st.subheader("📚 Authentic Content Analytics")
    
    if content_stats:
        col_content1, col_content2 = st.columns(2)
        
        with col_content1:
            # Real subject distribution
            if content_stats.get('subjects'):
                subject_df = pd.DataFrame([
                    {'Subject': k, 'Count': v} 
                    for k, v in content_stats['subjects'].items()
                ])
                
                fig = px.pie(subject_df, values='Count', names='Subject', 
                            title="Real Content Distribution by Subject")
                st.plotly_chart(fig, use_container_width=True)
        
        with col_content2:
            # Real difficulty distribution
            if content_stats.get('difficulties'):
                difficulty_df = pd.DataFrame([
                    {'Difficulty': k, 'Count': v} 
                    for k, v in content_stats['difficulties'].items()
                ])
                
                fig = px.bar(difficulty_df, x='Difficulty', y='Count', 
                            title="Real Content by Difficulty Level",
                            color='Count',
                            color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
    
    # Real project structure
    st.subheader("📁 Project File Structure")
    
    if file_structure:
        # Show file types and counts
        file_type_counts = {k: len(v) for k, v in file_structure.items()}
        
        col_struct1, col_struct2 = st.columns(2)
        
        with col_struct1:
            if file_type_counts:
                file_df = pd.DataFrame([
                    {'File Type': k, 'Count': v} 
                    for k, v in file_type_counts.items()
                ])
                
                fig = px.bar(file_df, x='File Type', y='Count', 
                            title="Files by Type",
                            color='Count',
                            color_continuous_scale='plasma')
                st.plotly_chart(fig, use_container_width=True)
        
        with col_struct2:
            st.write("**File Type Breakdown:**")
            for file_type, count in file_type_counts.items():
                st.write(f"• **{file_type}**: {count} files")
    
    # Real system performance
    st.subheader("⚡ System Performance Metrics")
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.write("**Code Quality Metrics:**")
        if system_data.get('total_lines', 0) > 0:
            avg_file_size = system_data.get('total_lines', 0) / max(system_data.get('python_files', 1), 1)
            st.metric("Average Lines per File", f"{avg_file_size:.1f}")
            st.metric("Total Project Size", f"{system_data.get('total_size_kb', 0):.1f} KB")
            st.metric("Python Files", system_data.get('python_files', 0))
    
    with col_perf2:
        st.write("**Content Metrics:**")
        if content_stats:
            st.metric("Total Content Items", content_stats.get('total_content', 0))
            st.metric("Total Estimated Time", f"{content_stats.get('total_estimated_time', 0)} minutes")
            st.metric("Supported Subjects", content_stats.get('supported_subjects', 0))
    
    # Real project insights
    st.subheader("💡 Authentic Project Insights")
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.info("**Project Structure Analysis:**")
        if system_data:
            st.write(f"• **Total Components**: {system_data.get('component_files', 0)} core components")
            st.write(f"• **Pages**: {system_data.get('page_files', 0)} user interface pages")
            st.write(f"• **Code Base**: {system_data.get('total_lines', 0):,} lines of Python code")
            st.write(f"• **Project Scale**: {system_data.get('python_files', 0)} Python files")
    
    with col_insight2:
        st.success("**Content Analysis:**")
        if content_stats:
            st.write(f"• **Content Coverage**: {content_stats.get('total_content', 0)} educational items")
            st.write(f"• **Subject Diversity**: {content_stats.get('supported_subjects', 0)} subjects supported")
            st.write(f"• **Learning Time**: {content_stats.get('total_estimated_time', 0)} minutes of content")
            st.write(f"• **Content Types**: {content_stats.get('content_types_available', 0)} different formats")
    
    # Real file details
    st.subheader("📋 Detailed File Analysis")
    
    if file_structure:
        # Show actual files
        st.write("**Project Files by Type:**")
        
        for file_type, files in file_structure.items():
            with st.expander(f"{file_type} Files ({len(files)})"):
                for file_path in sorted(files):
                    st.write(f"• `{file_path}`")
    
    # System recommendations based on real data
    st.subheader("💡 Data-Driven Recommendations")
    
    if content_stats and system_data:
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.info("**Content Recommendations:**")
            if content_stats.get('subjects'):
                # Find subject with least content
                min_subject = min(content_stats['subjects'].items(), key=lambda x: x[1])
                st.write(f"• **Expand {min_subject[0]}**: Only {min_subject[1]} items available")
            
            if content_stats.get('difficulties'):
                # Check difficulty balance
                beginner = content_stats['difficulties'].get('beginner', 0)
                advanced = content_stats['difficulties'].get('advanced', 0)
                if beginner < advanced:
                    st.write("• **Add beginner content**: More advanced than beginner content")
                elif advanced < beginner:
                    st.write("• **Add advanced content**: More beginner than advanced content")
        
        with col_rec2:
            st.info("**Technical Recommendations:**")
            if system_data.get('total_lines', 0) > 1000:
                st.write("• **Code organization**: Large codebase, consider modularization")
            
            if system_data.get('python_files', 0) > 10:
                st.write("• **Documentation**: Multiple files, ensure good documentation")
            
            if content_stats.get('total_content', 0) < 10:
                st.write("• **Content expansion**: Add more educational content")

if __name__ == "__main__":
    system_analytics_page()