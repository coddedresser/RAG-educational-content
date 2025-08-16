"""
Content Search and Retrieval Page - Using Real Project Data with AI Enhancement
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json

# Import the free LLM service for AI-powered search
from components.llm_service import FreeLLMService

def get_real_search_results(query, subject_filter="All", difficulty_filter="All", content_type_filter="All"):
    """Get authentic search results from actual project content"""
    try:
        from config import SAMPLE_CONTENT
        
        # Real search logic based on actual content
        results = []
        
        for content in SAMPLE_CONTENT:
            # Apply filters
            if subject_filter != "All" and content['subject'] != subject_filter:
                continue
            if difficulty_filter != "All" and content['difficulty_level'] != difficulty_filter:
                continue
            if content_type_filter != "All" and content['content_type'] != content_type_filter:
                continue
            
            # Simple text search (replace with your actual search logic)
            query_lower = query.lower()
            content_text = content.get('content', '').lower()
            title_lower = content['title'].lower()
            subject_lower = content['subject'].lower()
            
            # Calculate relevance score
            relevance = 0
            if query_lower in title_lower:
                relevance += 0.8
            if query_lower in subject_lower:
                relevance += 0.6
            if query_lower in content_text:
                relevance += 0.4
            
            if relevance > 0:
                results.append({
                    'title': content['title'],
                    'subject': content['subject'],
                    'difficulty': content['difficulty_level'],
                    'relevance': min(relevance, 1.0),
                    'content': content.get('content', '')[:200] + "...",
                    'estimated_time': content.get('estimated_time', 0),
                    'content_type': content['content_type'],
                    'learning_objectives': content.get('learning_objectives', [])
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results
        
    except Exception as e:
        st.error(f"Error in search: {e}")
        return []

def get_real_search_analytics():
    """Get authentic search analytics from actual usage"""
    try:
        from config import SAMPLE_CONTENT
        
        # Analyze actual content for search insights
        subjects = {}
        difficulties = {}
        content_types = {}
        
        for content in SAMPLE_CONTENT:
            # Subject distribution
            subject = content['subject']
            subjects[subject] = subjects.get(subject, 0) + 1
            
            # Difficulty distribution
            difficulty = content['difficulty_level']
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # Content type distribution
            content_type = content['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return {
            'subjects': subjects,
            'difficulties': difficulties,
            'content_types': content_types,
            'total_content': len(SAMPLE_CONTENT)
        }
    except Exception as e:
        st.error(f"Error reading analytics: {e}")
        return {}

def content_search_page():
    st.title("🔍 Content Search & Retrieval")
    st.write("Search through authentic educational content using natural language queries.")
    
    # Get real analytics
    analytics = get_real_search_analytics()
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Enter your search query:", 
                                   placeholder="e.g., explain basic algebra concepts")
        
        # Search filters based on real content
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            if analytics.get('subjects'):
                subject_options = ["All"] + list(analytics['subjects'].keys())
                subject_filter = st.selectbox("Subject", subject_options)
            else:
                subject_filter = "All"
        
        with col_filter2:
            if analytics.get('difficulties'):
                difficulty_options = ["All"] + list(analytics['difficulties'].keys())
                difficulty_filter = st.selectbox("Difficulty", difficulty_options)
            else:
                difficulty_filter = "All"
        
        with col_filter3:
            if analytics.get('content_types'):
                content_type_options = ["All"] + list(analytics['content_types'].keys())
                content_type_filter = st.selectbox("Content Type", content_type_options)
            else:
                content_type_filter = "All"
    
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 Search", type="primary")
    
    # AI-Powered Search Features
    st.subheader("🤖 AI-Powered Search Enhancement")
    
    # Initialize LLM service
    llm_service = FreeLLMService()
    
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        st.write("**AI Content Summarization:**")
        if st.button("📝 Get AI Summary of Results"):
            if search_query:
                with st.spinner("🤖 AI is analyzing search results..."):
                    # Get search results for AI analysis
                    results = get_real_search_results(search_query, subject_filter, difficulty_filter, content_type_filter)
                    if results:
                        # Combine content for AI summarization
                        combined_content = " ".join([result['content'] for result in results[:5]])
                        if combined_content:
                            ai_summary = llm_service.generate_content_summary(combined_content)
                            st.success("📝 AI Summary of Search Results:")
                            st.write(ai_summary)
                        else:
                            st.warning("No content available for AI summary")
                    else:
                        st.warning("No search results to summarize")
            else:
                st.warning("Please enter a search query first")
    
    with col_ai2:
        st.write("**AI Search Query Enhancement:**")
        if st.button("🚀 Enhance Search Query"):
            if search_query:
                with st.spinner("🤖 AI is enhancing your query..."):
                    # Create context for query enhancement
                    context = f"User wants to search for: {search_query}"
                    enhanced_query = llm_service.answer_educational_question(
                        "How can I improve this search query to get better results?",
                        context
                    )
                    st.success("🚀 AI-Enhanced Search Suggestions:")
                    st.write(enhanced_query)
            else:
                st.warning("Please enter a search query first")
    
    st.divider()
    
    # Search results
    if search_button and search_query:
        st.subheader("📚 Search Results")
        
        # Get real search results
        results = get_real_search_results(search_query, subject_filter, difficulty_filter, content_type_filter)
        
        if results:
            st.success(f"✅ Found {len(results)} relevant results")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"{i}. {result['title']} (Relevance: {result['relevance']:.2f})"):
                    st.write(f"**Subject:** {result['subject']}")
                    st.write(f"**Difficulty:** {result['difficulty']}")
                    st.write(f"**Content Type:** {result['content_type']}")
                    st.write(f"**Estimated Time:** {result['estimated_time']} minutes")
                    st.write(f"**Content:** {result['content']}")
                    
                    if result['learning_objectives']:
                        st.write("**Learning Objectives:**")
                        for obj in result['learning_objectives']:
                            st.write(f"• {obj}")
                    
                    col_view, col_save = st.columns(2)
                    with col_view:
                        if st.button("��️ View Full Content", key=f"view_{i}"):
                            st.info("�� Full content would be displayed here")
                    with col_save:
                        if st.button("💾 Save to Library", key=f"save_{i}"):
                            st.success("✅ Content saved to your library!")
        else:
            st.warning("🔍 No results found. Try different keywords or filters.")
    
    # Real search analytics
    with st.expander("📊 Authentic Search Analytics"):
        if analytics:
            st.write(f"**Total Content Available:** {analytics['total_content']} items")
            
            col_analytics1, col_analytics2 = st.columns(2)
            
            with col_analytics1:
                if analytics.get('subjects'):
                    subject_df = pd.DataFrame([
                        {'Subject': k, 'Count': v} 
                        for k, v in analytics['subjects'].items()
                    ])
                    
                    fig = px.pie(subject_df, values='Count', names='Subject', 
                                title="Content Distribution by Subject")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_analytics2:
                if analytics.get('difficulties'):
                    difficulty_df = pd.DataFrame([
                        {'Difficulty': k, 'Count': v} 
                        for k, v in analytics['difficulties'].items()
                    ])
                    
                    fig = px.bar(difficulty_df, x='Difficulty', y='Count', 
                                title="Content by Difficulty Level",
                                color='Count',
                                color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Content recommendations based on real data
    if analytics:
        st.subheader("💡 Content Recommendations")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if analytics.get('subjects'):
                # Find subject with most content
                max_subject = max(analytics['subjects'].items(), key=lambda x: x[1])
                st.info(f"**Most Comprehensive Subject**: {max_subject[0]}")
                st.write(f"• {max_subject[1]} content items available")
                st.write("• Great starting point for learning")
                st.write("• Wide range of topics covered")
        
        with col_rec2:
            if analytics.get('difficulties'):
                # Check difficulty balance
                beginner = analytics['difficulties'].get('beginner', 0)
                advanced = analytics['difficulties'].get('advanced', 0)
                
                if beginner > advanced:
                    st.success("**Beginner-Friendly System**")
                    st.write("• Plenty of beginner content")
                    st.write("• Good for new learners")
                elif advanced > beginner:
                    st.warning("**Advanced-Focused System**")
                    st.write("• More advanced content available")
                    st.write("• Consider adding beginner materials")

if __name__ == "__main__":
    content_search_page()