"""
Authentication Component for Educational RAG System
Handles user registration, login, session management, and security
"""
import streamlit as st
import hashlib
import secrets
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class UserAuth:
    """User authentication and management system"""
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.sessions_file = Path("data/sessions.json")
        self.users_file.parent.mkdir(exist_ok=True)
        self._load_users()
        self._load_sessions()
    
    def _load_users(self):
        """Load existing users from file"""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                self.users = {}
                self._save_users()
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            self.users = {}
    
    def _save_users(self):
        """Save users to file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _load_sessions(self):
        """Load active sessions from file"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    self.sessions = json.load(f)
            else:
                self.sessions = {}
                self._save_sessions()
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to file"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return salt, hashed.hex()
    
    def _verify_password(self, password: str, salt: str, hashed: str) -> bool:
        """Verify password against stored hash"""
        try:
            _, new_hash = self._hash_password(password, salt)
            return new_hash == hashed
        except Exception:
            return False
    
    def register_user(self, username: str, email: str, password: str, full_name: str = "") -> Dict[str, Any]:
        """Register a new user"""
        # Validation
        if not username or not email or not password:
            return {"success": False, "message": "All fields are required"}
        
        if len(password) < 8:
            return {"success": False, "message": "Password must be at least 8 characters long"}
        
        if username in self.users:
            return {"success": False, "message": "Username already exists"}
        
        if any(user['email'] == email for user in self.users.values()):
            return {"success": False, "message": "Email already registered"}
        
        # Create user
        salt, hashed_password = self._hash_password(password)
        user_id = secrets.token_hex(8)
        
        user_data = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "full_name": full_name,
            "password_hash": hashed_password,
            "password_salt": salt,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "profile_completed": False,
            "learning_style": None,
            "preferences": {},
            "progress": {}
        }
        
        self.users[username] = user_data
        self._save_users()
        
        logger.info(f"New user registered: {username}")
        return {"success": True, "message": "Registration successful!", "user_id": user_id}
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate and login user"""
        if username not in self.users:
            return {"success": False, "message": "Invalid username or password"}
        
        user = self.users[username]
        
        if not self._verify_password(password, user['password_salt'], user['password_hash']):
            return {"success": False, "message": "Invalid username or password"}
        
        # Create session
        session_token = secrets.token_hex(32)
        session_data = {
            "user_id": user['user_id'],
            "username": username,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        self.sessions[session_token] = session_data
        self._save_sessions()
        
        # Update last login
        user['last_login'] = datetime.now().isoformat()
        self._save_users()
        
        logger.info(f"User logged in: {username}")
        return {
            "success": True, 
            "message": "Login successful!", 
            "session_token": session_token,
            "user_data": user
        }
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user and remove session"""
        if session_token in self.sessions:
            del self.sessions[session_token]
            self._save_sessions()
            logger.info(f"User logged out: {self.sessions.get(session_token, {}).get('username', 'Unknown')}")
            return True
        return False
    
    def get_user_from_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get user data from session token"""
        if session_token not in self.sessions:
            return None
        
        session = self.sessions[session_token]
        
        # Check if session expired
        expires_at = datetime.fromisoformat(session['expires_at'])
        if datetime.now() > expires_at:
            del self.sessions[session_token]
            self._save_sessions()
            return None
        
        username = session['username']
        if username in self.users:
            return self.users[username]
        
        return None
    
    def update_user_profile(self, username: str, profile_data: Dict[str, Any]) -> bool:
        """Update user profile information"""
        if username not in self.users:
            return False
        
        # Update allowed fields
        allowed_fields = ['full_name', 'learning_style', 'preferences', 'profile_completed']
        for field in allowed_fields:
            if field in profile_data:
                self.users[username][field] = profile_data[field]
        
        self._save_users()
        return True
    
    def change_password(self, username: str, current_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password"""
        if username not in self.users:
            return {"success": False, "message": "User not found"}
        
        user = self.users[username]
        
        if not self._verify_password(current_password, user['password_salt'], user['password_hash']):
            return {"success": False, "message": "Current password is incorrect"}
        
        if len(new_password) < 8:
            return {"success": False, "message": "New password must be at least 8 characters long"}
        
        # Update password
        salt, hashed_password = self._hash_password(new_password)
        user['password_hash'] = hashed_password
        user['password_salt'] = salt
        
        self._save_users()
        logger.info(f"Password changed for user: {username}")
        return {"success": True, "message": "Password changed successfully!"}
    
    def delete_user(self, username: str, password: str) -> Dict[str, Any]:
        """Delete user account"""
        if username not in self.users:
            return {"success": False, "message": "User not found"}
        
        user = self.users[username]
        
        if not self._verify_password(password, user['password_salt'], user['password_hash']):
            return {"success": False, "message": "Password is incorrect"}
        
        # Remove user sessions
        user_id = user['user_id']
        sessions_to_remove = [token for token, session in self.sessions.items() 
                            if session['user_id'] == user_id]
        
        for token in sessions_to_remove:
            del self.sessions[token]
        
        # Remove user
        del self.users[username]
        
        self._save_users()
        self._save_sessions()
        
        logger.info(f"User account deleted: {username}")
        return {"success": True, "message": "Account deleted successfully!"}
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_tokens = []
        
        for token, session in self.sessions.items():
            expires_at = datetime.fromisoformat(session['expires_at'])
            if current_time > expires_at:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.sessions[token]
        
        if expired_tokens:
            self._save_sessions()
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")

class AuthUI:
    """User interface for authentication"""
    
    def __init__(self):
        self.auth = UserAuth()
    
    def show_auth_page(self):
        """Display authentication page with login/register options"""
        st.title("🔐 User Authentication")
        st.markdown("Sign in to access your personalized learning experience or create a new account.")
        
        # Clean up expired sessions
        self.auth.cleanup_expired_sessions()
        
        # Check if user is already logged in
        if 'session_token' in st.session_state:
            user = self.auth.get_user_from_session(st.session_state.session_token)
            if user:
                self._show_user_dashboard(user)
                return
        
        # Authentication tabs
        tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])
        
        with tab1:
            self._show_login_form()
        
        with tab2:
            self._show_register_form()
    
    def _show_login_form(self):
        """Display login form"""
        st.subheader("Welcome Back!")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            with col2:
                if st.form_submit_button("Forgot Password?", use_container_width=True):
                    st.info("Password reset functionality coming soon!")
            
            if submit:
                if username and password:
                    result = self.auth.login_user(username, password)
                    if result['success']:
                        st.session_state.session_token = result['session_token']
                        st.session_state.user_data = result['user_data']
                        st.success(result['message'])
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please fill in all fields")
    
    def _show_register_form(self):
        """Display registration form"""
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            username = st.text_input("Username", placeholder="Choose a unique username")
            email = st.text_input("Email", placeholder="Enter your email address")
            password = st.text_input("Password", type="password", placeholder="Create a password (min 8 chars)")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            # Password strength indicator
            if password:
                strength = self._check_password_strength(password)
                st.progress(strength['score'] / 4)
                st.write(f"Password strength: {strength['label']} ({strength['score']}/4)")
            
            submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            
            if submit:
                if not all([full_name, username, email, password, confirm_password]):
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    result = self.auth.register_user(username, email, password, full_name)
                    if result['success']:
                        st.success(result['message'])
                        st.info("You can now sign in with your new account!")
                    else:
                        st.error(result['message'])
    
    def _check_password_strength(self, password: str) -> Dict[str, Any]:
        """Check password strength"""
        score = 0
        feedback = []
        
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("At least 8 characters")
        
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Include uppercase letters")
        
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Include lowercase letters")
        
        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Include numbers")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Include special characters")
        
        # Cap score at 4
        score = min(score, 4)
        
        labels = ["Very Weak", "Weak", "Fair", "Good", "Strong"]
        return {
            "score": score,
            "label": labels[score],
            "feedback": feedback
        }
    
    def _show_user_dashboard(self, user: Dict[str, Any]):
        """Display user dashboard after login"""
        st.success(f"Welcome back, {user.get('full_name', user['username'])}! 🎉")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("👤 Your Profile")
            st.write(f"**Username:** {user['username']}")
            st.write(f"**Email:** {user['email']}")
            st.write(f"**Member since:** {user['created_at'][:10]}")
            
            if user.get('profile_completed'):
                st.success("✅ Profile completed")
            else:
                st.warning("⚠️ Complete your profile for personalized learning")
        
        with col2:
            st.subheader("📊 Quick Stats")
            if user.get('progress'):
                st.metric("Learning Sessions", len(user['progress']))
            else:
                st.metric("Learning Sessions", 0)
            
            if user.get('learning_style'):
                st.metric("Learning Style", user['learning_style'])
            else:
                st.metric("Learning Style", "Not set")
        
        with col3:
            st.subheader("⚙️ Account")
            if st.button("🔒 Change Password"):
                st.session_state.show_change_password = True
            
            if st.button("🗑️ Delete Account"):
                st.session_state.show_delete_account = True
            
            if st.button("🚪 Sign Out"):
                self._logout_user()
        
        # Change password form
        if st.session_state.get('show_change_password'):
            self._show_change_password_form(user)
        
        # Delete account form
        if st.session_state.get('show_delete_account'):
            self._show_delete_account_form(user)
        
        st.divider()
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📚 Start Learning", type="primary", use_container_width=True):
                st.info("Navigate to Learning Path or Content Search to begin!")
        
        with col2:
            if st.button("👤 Complete Profile", use_container_width=True):
                st.info("Go to Student Profile page to complete your learning profile!")
        
        with col3:
            if st.button("📊 View Progress", use_container_width=True):
                st.info("Check your Progress Tracking page for detailed analytics!")
    
    def _show_change_password_form(self, user: Dict[str, Any]):
        """Display password change form"""
        st.subheader("🔒 Change Password")
        
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit = st.form_submit_button("Change Password", type="primary")
            with col2:
                cancel = st.form_submit_button("Cancel")
            
            if submit:
                if not all([current_password, new_password, confirm_password]):
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 8:
                    st.error("New password must be at least 8 characters long")
                else:
                    result = self.auth.change_password(user['username'], current_password, new_password)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.show_change_password = False
                        st.rerun()
                    else:
                        st.error(result['message'])
            
            if cancel:
                st.session_state.show_change_password = False
                st.rerun()
    
    def _show_delete_account_form(self, user: Dict[str, Any]):
        """Display account deletion form"""
        st.subheader("🗑️ Delete Account")
        st.warning("⚠️ This action cannot be undone. All your data will be permanently deleted.")
        
        with st.form("delete_account_form"):
            password = st.text_input("Enter your password to confirm", type="password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit = st.form_submit_button("Delete Account", type="secondary")
            with col2:
                cancel = st.form_submit_button("Cancel")
            
            if submit:
                if password:
                    result = self.auth.delete_user(user['username'], password)
                    if result['success']:
                        st.success(result['message'])
                        self._logout_user()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter your password to confirm")
            
            if cancel:
                st.session_state.show_delete_account = False
                st.rerun()
    
    def _logout_user(self):
        """Logout current user"""
        if 'session_token' in st.session_state:
            self.auth.logout_user(st.session_state.session_token)
            del st.session_state.session_token
            del st.session_state.user_data
            st.success("Successfully signed out!")
            st.rerun()
    
    def require_auth(self):
        """Decorator to require authentication for pages"""
        if 'session_token' not in st.session_state:
            st.error("🔐 Please sign in to access this page")
            st.stop()
        
        user = self.auth.get_user_from_session(st.session_state.session_token)
        if not user:
            del st.session_state.session_token
            st.error("🔐 Session expired. Please sign in again")
            st.stop()
        
        return user
