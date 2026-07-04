from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import relationship
from database import db
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_login import UserMixin
import uuid

class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    profession = db.Column(db.String(100), nullable=True)
    experience_level = db.Column(db.String(20), default='Beginner')
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    interview_sessions = db.relationship('InterviewSession', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, email, username, password, first_name, last_name, **kwargs):
        self.email = email.lower().strip()
        self.username = username.lower().strip()
        self.first_name = first_name.strip()
        self.last_name = last_name.strip()
        self.set_password(password)
        
        # Optional fields
        self.phone = kwargs.get('phone', '').strip() if kwargs.get('phone') else None
        self.profession = kwargs.get('profession', '').strip() if kwargs.get('profession') else None
        self.experience_level = kwargs.get('experience_level', 'Beginner')

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)

    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()

    @property
    def full_name(self):
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"

    def to_dict(self):
        """Convert user to dictionary for JSON responses"""
        return {
            'id': self.public_id,
            'email': self.email,
            'username': self.username,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name,
            'phone': self.phone,
            'profession': self.profession,
            'experience_level': self.experience_level,
            'is_active': self.is_active,
            'email_verified': self.email_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

    def __repr__(self):
        return f'<User {self.username}>'

    # Class methods for database queries
    @classmethod
    def create_user(cls, email, username, password, first_name, last_name, **kwargs):
        """Create a new user"""
        try:
            user = cls(
                email=email,
                username=username,
                password=password,
                first_name=first_name,
                last_name=last_name,
                **kwargs
            )
            db.session.add(user)
            db.session.commit()
            return user
        except Exception as e:
            db.session.rollback()
            raise e

    @classmethod
    def get_by_email(cls, email):
        """Get user by email"""
        return cls.query.filter_by(email=email.lower().strip()).first()

    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        return cls.query.filter_by(username=username.lower().strip()).first()

    @classmethod
    def get_by_id(cls, user_id):
        """Get user by ID"""
        return cls.query.get(user_id)

    @classmethod
    def get_by_public_id(cls, public_id):
        """Get user by public ID"""
        return cls.query.filter_by(public_id=public_id).first()

    @classmethod
    def email_exists(cls, email):
        """Check if email already exists"""
        return cls.query.filter_by(email=email.lower().strip()).first() is not None

    @classmethod
    def username_exists(cls, username):
        """Check if username already exists"""
        return cls.query.filter_by(username=username.lower().strip()).first() is not None

    @classmethod
    def authenticate(cls, login, password):
        """Authenticate user with email/username and password"""
        # Try to find user by email first, then by username
        user = cls.get_by_email(login) or cls.get_by_username(login)
        
        if user and user.check_password(password) and user.is_active:
            user.update_last_login()
            return user
        return None

    @classmethod
    def get_all_users(cls, page=1, per_page=20):
        """Get paginated list of users"""
        return cls.query.order_by(cls.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

    def update_profile(self, **kwargs):
        """Update user profile"""
        try:
            allowed_fields = ['first_name', 'last_name', 'phone', 'profession', 'experience_level']
            
            for field in allowed_fields:
                if field in kwargs:
                    value = kwargs[field].strip() if kwargs[field] else None
                    setattr(self, field, value)
            
            self.updated_at = datetime.utcnow()
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise e

    def change_password(self, old_password, new_password):
        """Change user password"""
        if not self.check_password(old_password):
            return False
        
        try:
            self.set_password(new_password)
            self.updated_at = datetime.utcnow()
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise e

    def deactivate(self):
        """Deactivate user account"""
        try:
            self.is_active = False
            self.updated_at = datetime.utcnow()
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise e


class InterviewSession(db.Model):
    """Model to store interview session data"""
    
    __tablename__ = 'interview_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    interview_type = db.Column(db.String(50), nullable=False)
    level = db.Column(db.String(20), nullable=False)
    interview_mode = db.Column(db.String(20), default='text')
    
    # Session data
    questions = db.Column(db.JSON, nullable=True)
    answers = db.Column(db.JSON, nullable=True)
    total_questions = db.Column(db.Integer, default=0)
    completed_questions = db.Column(db.Integer, default=0)
    
    # Scoring
    average_scores = db.Column(db.JSON, nullable=True)
    overall_score = db.Column(db.Float, default=0.0)
    
    # Session status
    status = db.Column(db.String(20), default='active')  # active, completed, abandoned
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    duration_minutes = db.Column(db.Integer, default=0)
    
    def __init__(self, user_id, role, interview_type, level, interview_mode='text'):
        self.user_id = user_id
        self.role = role
        self.interview_type = interview_type
        self.level = level
        self.interview_mode = interview_mode

    @classmethod
    def create_session(cls, user_id, role, interview_type, level, interview_mode='text'):
        """Create a new interview session"""
        try:
            session = cls(
                user_id=user_id,
                role=role,
                interview_type=interview_type,
                level=level,
                interview_mode=interview_mode
            )
            db.session.add(session)
            db.session.commit()
            return session
        except Exception as e:
            db.session.rollback()
            raise e

    @classmethod
    def get_by_session_id(cls, session_id):
        """Get session by session ID"""
        return cls.query.filter_by(session_id=session_id).first()

    @classmethod
    def get_user_sessions(cls, user_id, page=1, per_page=10):
        """Get user's interview sessions"""
        return cls.query.filter_by(user_id=user_id).order_by(
            cls.started_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)

    def update_questions(self, questions):
        """Update session questions"""
        try:
            self.questions = questions
            self.total_questions = len(questions)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def add_answer(self, answer_data):
        """Add answer to session"""
        try:
            if not self.answers:
                self.answers = []
            
            self.answers.append(answer_data)
            self.completed_questions = len(self.answers)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def complete_session(self, average_scores, overall_score):
        """Mark session as completed"""
        try:
            self.status = 'completed'
            self.completed_at = datetime.utcnow()
            self.average_scores = average_scores
            self.overall_score = overall_score
            
            # Calculate duration
            if self.started_at and self.completed_at:
                duration = self.completed_at - self.started_at
                self.duration_minutes = int(duration.total_seconds() / 60)
            
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def to_dict(self):
        """Convert session to dictionary"""
        return {
            'session_id': self.session_id,
            'role': self.role,
            'interview_type': self.interview_type,
            'level': self.level,
            'interview_mode': self.interview_mode,
            'total_questions': self.total_questions,
            'completed_questions': self.completed_questions,
            'average_scores': self.average_scores,
            'overall_score': self.overall_score,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_minutes': self.duration_minutes
        }