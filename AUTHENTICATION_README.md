# 🔐 Authentication System for Educational RAG System

## Overview
This document explains how to use the new authentication system that has been added to your Educational RAG System.

## Features Added

### 1. User Registration & Login
- **Sign Up**: Create new accounts with username, email, and password
- **Sign In**: Authenticate existing users
- **Password Security**: Secure password hashing with PBKDF2 and salt
- **Session Management**: 24-hour session tokens for secure access

### 2. User Management
- **Profile Management**: Update personal information and preferences
- **Password Changes**: Secure password update functionality
- **Account Deletion**: Remove user accounts with password confirmation
- **Learning Progress**: Track individual user learning progress

### 3. Security Features
- **Password Strength**: Real-time password strength validation
- **Session Expiration**: Automatic session cleanup and expiration
- **Secure Storage**: User data stored in encrypted JSON files
- **Input Validation**: Comprehensive form validation and sanitization

## How to Use

### First Time Setup
1. **Navigate to Authentication Page**: The system now starts with an authentication page
2. **Create Account**: Click "Sign Up" tab and fill in your details
3. **Complete Profile**: After registration, complete your learning profile
4. **Start Learning**: Access all system features with your personalized account

### Daily Usage
1. **Sign In**: Use your username and password to access the system
2. **Access Features**: All pages now require authentication
3. **Manage Account**: Use sidebar options to manage your account
4. **Sign Out**: Use the sign out button when finished

### Account Management
- **Change Password**: Access via account section in sidebar
- **Update Profile**: Modify your learning preferences and information
- **Delete Account**: Remove your account if needed (irreversible)

## File Structure

```
app/
├── components/
│   └── auth.py              # Authentication logic and UI
├── pages/
│   ├── 0_Authentication.py  # Main authentication page
│   ├── 1_Learning_Path.py   # Now requires authentication
│   ├── 2_Content_search.py  # Now requires authentication
│   ├── 3_Student_Profile.py # Now requires authentication
│   ├── 4_Progress_Tracking.py # Now requires authentication
│   └── 5_System_analytics.py # Now requires authentication
├── main.py                   # Updated with authentication checks
└── config.py                 # Updated with data directory setup

data/                         # User data storage (auto-created)
├── users.json               # User accounts and profiles
└── sessions.json            # Active user sessions
```

## Security Implementation

### Password Security
- **Hashing**: PBKDF2 with SHA-256 (100,000 iterations)
- **Salt**: Unique 16-byte salt per user
- **Storage**: Only hashed passwords stored, never plain text

### Session Security
- **Tokens**: 32-byte random session tokens
- **Expiration**: 24-hour automatic expiration
- **Cleanup**: Automatic cleanup of expired sessions

### Data Protection
- **Local Storage**: User data stored locally in JSON files
- **Validation**: Comprehensive input validation and sanitization
- **Access Control**: All pages require valid authentication

## User Experience

### Authentication Flow
1. **Landing Page**: Users see authentication page first
2. **Registration**: Simple sign-up form with validation
3. **Login**: Quick access for returning users
4. **Dashboard**: Personalized welcome and quick actions
5. **Navigation**: Seamless access to all features

### Personalization
- **Welcome Messages**: Personalized greetings throughout the system
- **User Context**: All features now show user-specific information
- **Progress Tracking**: Individual learning progress per user
- **Preferences**: User-specific learning preferences and settings

## Technical Details

### Dependencies
- **Streamlit**: UI framework for authentication forms
- **Hashlib**: Python standard library for password hashing
- **Secrets**: Cryptographically secure random generation
- **JSON**: Data storage and serialization
- **Pathlib**: File system operations

### Data Storage
- **Users**: Stored in `data/users.json`
- **Sessions**: Stored in `data/sessions.json`
- **Auto-creation**: Data directory created automatically
- **Backup**: Consider backing up user data regularly

### Error Handling
- **Validation Errors**: Clear error messages for form issues
- **Session Errors**: Automatic redirect to authentication
- **System Errors**: Graceful fallbacks and user notifications

## Migration from Previous Version

### For Existing Users
- **No Data Loss**: All existing functionality preserved
- **New Features**: Authentication adds security and personalization
- **Same Interface**: Familiar UI with enhanced security

### For Developers
- **Code Updates**: All pages now include authentication checks
- **New Components**: `auth.py` component for authentication logic
- **Session Management**: Streamlit session state integration

## Best Practices

### For Users
- **Strong Passwords**: Use unique, strong passwords
- **Regular Updates**: Keep your profile information current
- **Secure Access**: Don't share your login credentials
- **Sign Out**: Always sign out when using shared devices

### For Administrators
- **Data Backup**: Regular backup of user data files
- **Security Monitoring**: Monitor for unusual access patterns
- **Updates**: Keep the system updated with latest security patches

## Troubleshooting

### Common Issues
1. **Session Expired**: Simply sign in again
2. **Password Issues**: Use password change functionality
3. **Account Locked**: Contact system administrator
4. **Data Loss**: Check data directory and backup files

### Support
- **Documentation**: Refer to this README for guidance
- **Error Messages**: System provides clear error information
- **Logs**: Check system logs for detailed error information

## Future Enhancements

### Planned Features
- **Password Reset**: Email-based password recovery
- **Two-Factor Authentication**: Enhanced security options
- **Social Login**: Integration with external authentication providers
- **Role-Based Access**: Different permission levels for users
- **Audit Logging**: Detailed access and activity logging

### Integration Opportunities
- **Database Backend**: Migration to SQL database
- **Cloud Storage**: User data synchronization across devices
- **API Access**: RESTful API for external integrations
- **Mobile App**: Native mobile application support

---

**Note**: This authentication system provides a solid foundation for user management while maintaining the existing functionality of your Educational RAG System. All features are now secure and personalized for individual users.
