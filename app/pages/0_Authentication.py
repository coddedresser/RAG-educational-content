"""
Authentication Page for Educational RAG System
First page users see - handles login and registration
"""
import streamlit as st
import sys
from pathlib import Path

# Fix import paths
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
app_dir = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

try:
    from components.auth import AuthUI
except ImportError:
    from app.components.auth import AuthUI

def main():
    """Main authentication page"""
    
    # Page configuration
    st.set_page_config(
        page_title="Authentication - Educational RAG System",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide the default Streamlit menu and footer
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        </style>
        """, unsafe_allow_html=True)
    
    # Main authentication interface
    auth_ui = AuthUI()
    auth_ui.show_auth_page()

if __name__ == "__main__":
    main()
