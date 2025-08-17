"""
Test script for the authentication system
Run this to verify the auth components work correctly
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.components.auth import UserAuth
    print("✅ Authentication component imported successfully")
    
    # Test user creation
    auth = UserAuth()
    print("✅ UserAuth initialized successfully")
    
    # Test user registration
    result = auth.register_user(
        username="testuser",
        email="test@example.com",
        password="TestPassword123!",
        full_name="Test User"
    )
    
    if result['success']:
        print("✅ User registration test passed")
    else:
        print(f"❌ User registration failed: {result['message']}")
    
    # Test user login
    login_result = auth.login_user("testuser", "TestPassword123!")
    
    if login_result['success']:
        print("✅ User login test passed")
        session_token = login_result['session_token']
        
        # Test session validation
        user = auth.get_user_from_session(session_token)
        if user:
            print("✅ Session validation test passed")
        else:
            print("❌ Session validation failed")
        
        # Test logout
        if auth.logout_user(session_token):
            print("✅ User logout test passed")
        else:
            print("❌ User logout failed")
            
    else:
        print(f"❌ User login failed: {login_result['message']}")
    
    print("\n🎉 Authentication system test completed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
