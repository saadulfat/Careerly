from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import User, InterviewSession
from database import db
import re
from werkzeug.security import check_password_hash

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username format"""
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if len(password) > 128:
        return False, "Password is too long"
    return True, "Password is valid"

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if request.method == 'GET':
        # If user is already logged in, redirect to home
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        return render_template('signup.html')
    
    try:
        data = request.get_json()
        
        # Extract and validate required fields
        required_fields = ['first_name', 'last_name', 'email', 'username', 'password']
        for field in required_fields:
            if not data.get(field) or not data[field].strip():
                return jsonify({
                    'success': False,
                    'message': f'{field.replace("_", " ").title()} is required'
                }), 400
        
        first_name = data['first_name'].strip()
        last_name = data['last_name'].strip()
        email = data['email'].strip().lower()
        username = data['username'].strip().lower()
        password = data['password']
        
        # Optional fields
        profession = data.get('profession', '').strip() if data.get('profession') else None
        experience_level = data.get('experience_level', 'Beginner')
        phone = data.get('phone', '').strip() if data.get('phone') else None
        
        # Validate email format
        if not validate_email(email):
            return jsonify({
                'success': False,
                'message': 'Please enter a valid email address'
            }), 400
        
        # Validate username format
        if not validate_username(username):
            return jsonify({
                'success': False,
                'message': 'Username must be 3-20 characters long and contain only letters, numbers, and underscores'
            }), 400
        
        # Validate password
        password_valid, password_message = validate_password(password)
        if not password_valid:
            return jsonify({
                'success': False,
                'message': password_message
            }), 400
        
        # Check if email already exists
        if User.email_exists(email):
            return jsonify({
                'success': False,
                'message': 'An account with this email already exists'
            }), 400
        
        # Check if username already exists
        if User.username_exists(username):
            return jsonify({
                'success': False,
                'message': 'This username is already taken'
            }), 400
        
        # Validate experience level
        valid_levels = ['Beginner', 'Intermediate', 'Advanced']
        if experience_level not in valid_levels:
            experience_level = 'Beginner'
        
        # Create new user
        user = User.create_user(
            email=email,
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            profession=profession,
            experience_level=experience_level,
            phone=phone
        )
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully! Please log in to continue.',
            'user_id': user.public_id
        }), 201
        
    except Exception as e:
        print(f"Signup error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while creating your account. Please try again.'
        }), 500

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'GET':
        # If user is already logged in, redirect to home
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        return render_template('login.html')
    
    try:
        data = request.get_json()
        
        # Extract login credentials
        login_input = data.get('login', '').strip()
        password = data.get('password', '')
        remember = data.get('remember', False)
        
        if not login_input or not password:
            return jsonify({
                'success': False,
                'message': 'Please enter both email/username and password'
            }), 400
        
        # Authenticate user
        user = User.authenticate(login_input, password)
        
        if not user:
            return jsonify({
                'success': False,
                'message': 'Invalid email/username or password'
            }), 401
        
        # Check if user account is active
        if not user.is_active:
            return jsonify({
                'success': False,
                'message': 'Your account has been deactivated. Please contact support.'
            }), 401
        
        # Log in the user
        login_user(user, remember=remember)
        
        # Update last login timestamp
        user.update_last_login()
        
        # Get redirect URL from session or default to home
        next_page = session.pop('next_page', None)
        redirect_url = next_page if next_page else url_for('home')
        
        return jsonify({
            'success': True,
            'message': 'Login successful!',
            'redirect_url': redirect_url,
            'user': {
                'id': user.public_id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name
            }
        }), 200
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while logging in. Please try again.'
        }), 500

@auth_bp.route('/logout', methods=['POST', 'GET'])
@login_required
def logout():
    """User logout"""
    try:
        logout_user()
        
        # Clear session data related to interviews
        interview_keys = ['questions', 'answers', 'role', 'interview_type', 'level', 
                         'interview_mode', 'current_question', 'csv_session_id']
        for key in interview_keys:
            session.pop(key, None)
        
        if request.method == 'POST':
            return jsonify({
                'success': True,
                'message': 'Logged out successfully',
                'redirect_url': url_for('auth.login')
            }), 200
        else:
            flash('You have been logged out successfully', 'success')
            return redirect(url_for('auth.login'))
            
    except Exception as e:
        print(f"Logout error: {str(e)}")
        if request.method == 'POST':
            return jsonify({
                'success': False,
                'message': 'An error occurred while logging out'
            }), 500
        else:
            flash('An error occurred while logging out', 'error')
            return redirect(url_for('auth.login'))

@auth_bp.route('/profile', methods=['GET'])
@login_required
def profile():
    """Get user profile"""
    try:
        return jsonify({
            'success': True,
            'user': current_user.to_dict()
        }), 200
    except Exception as e:
        print(f"Profile error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error retrieving profile'
        }), 500

@auth_bp.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        data = request.get_json()
        
        # Extract allowed fields
        allowed_fields = ['first_name', 'last_name', 'phone', 'profession', 'experience_level']
        update_data = {}
        
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
        
        # Validate experience level if provided
        if 'experience_level' in update_data:
            valid_levels = ['Beginner', 'Intermediate', 'Advanced']
            if update_data['experience_level'] not in valid_levels:
                return jsonify({
                    'success': False,
                    'message': 'Invalid experience level'
                }), 400
        
        # Update profile
        current_user.update_profile(**update_data)
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': current_user.to_dict()
        }), 200
        
    except Exception as e:
        print(f"Profile update error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error updating profile'
        }), 500

@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    try:
        data = request.get_json()
        
        old_password = data.get('old_password', '')
        new_password = data.get('new_password', '')
        confirm_password = data.get('confirm_password', '')
        
        if not old_password or not new_password or not confirm_password:
            return jsonify({
                'success': False,
                'message': 'All password fields are required'
            }), 400
        
        if new_password != confirm_password:
            return jsonify({
                'success': False,
                'message': 'New passwords do not match'
            }), 400
        
        # Validate new password
        password_valid, password_message = validate_password(new_password)
        if not password_valid:
            return jsonify({
                'success': False,
                'message': password_message
            }), 400
        
        # Change password
        if current_user.change_password(old_password, new_password):
            return jsonify({
                'success': True,
                'message': 'Password changed successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Current password is incorrect'
            }), 400
            
    except Exception as e:
        print(f"Change password error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error changing password'
        }), 500

@auth_bp.route('/sessions', methods=['GET'])
@login_required
def user_sessions():
    """Get user's interview sessions"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        sessions_pagination = InterviewSession.get_user_sessions(
            current_user.id, page=page, per_page=per_page
        )
        
        sessions = [session.to_dict() for session in sessions_pagination.items]
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'pagination': {
                'page': sessions_pagination.page,
                'per_page': sessions_pagination.per_page,
                'total': sessions_pagination.total,
                'pages': sessions_pagination.pages,
                'has_next': sessions_pagination.has_next,
                'has_prev': sessions_pagination.has_prev
            }
        }), 200
        
    except Exception as e:
        print(f"Sessions error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error retrieving sessions'
        }), 500

@auth_bp.route('/check-email', methods=['POST'])
def check_email():
    """Check if email is available"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({
                'success': False,
                'message': 'Email is required'
            }), 400
        
        if not validate_email(email):
            return jsonify({
                'success': False,
                'message': 'Invalid email format'
            }), 400
        
        exists = User.email_exists(email)
        
        return jsonify({
            'success': True,
            'available': not exists,
            'message': 'Email is available' if not exists else 'Email is already taken'
        }), 200
        
    except Exception as e:
        print(f"Check email error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error checking email availability'
        }), 500

@auth_bp.route('/check-username', methods=['POST'])
def check_username():
    """Check if username is available"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip().lower()
        
        if not username:
            return jsonify({
                'success': False,
                'message': 'Username is required'
            }), 400
        
        if not validate_username(username):
            return jsonify({
                'success': False,
                'message': 'Invalid username format'
            }), 400
        
        exists = User.username_exists(username)
        
        return jsonify({
            'success': True,
            'available': not exists,
            'message': 'Username is available' if not exists else 'Username is already taken'
        }), 200
        
    except Exception as e:
        print(f"Check username error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error checking username availability'
        }), 500