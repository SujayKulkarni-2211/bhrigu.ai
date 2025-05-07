# pro/app/models/user.py
from datetime import datetime
import uuid
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    # Profile information
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    organization = db.Column(db.String(100), nullable=True)
    
    # Account status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    
    # Pro subscription status
    is_pro = db.Column(db.Boolean, default=False)
    pro_since = db.Column(db.DateTime, nullable=True)
    pro_expires = db.Column(db.DateTime, nullable=True)
    
    # Additional purchased features
    has_k_fold = db.Column(db.Boolean, default=False)
    has_api_hosting = db.Column(db.Boolean, default=False)
    
    # Email notification preferences
    notify_on_completion = db.Column(db.Boolean, default=True)
    notify_on_error = db.Column(db.Boolean, default=True)
    
    # Payment information
    stripe_customer_id = db.Column(db.String(100), nullable=True)
    stripe_subscription_id = db.Column(db.String(100), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    experiments = db.relationship('Experiment', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def __init__(self, email, username, password, **kwargs):
        """Initialize a user with email, username and password"""
        self.email = email
        self.username = username
        self.set_password(password)
        
        # Set any additional properties passed as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def set_password(self, password):
        """Set the password hash from a plain text password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if the provided password matches the hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        """Get the user's full name, or username if names not set"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        else:
            return self.username
    
    def has_active_pro_subscription(self):
        """Check if user has an active Pro subscription"""
        if not self.is_pro or not self.pro_expires:
            return False
        
        return self.pro_expires > datetime.utcnow()
    
    def can_use_quantum_models(self):
        """Check if user can use quantum models (Pro only)"""
        return self.has_active_pro_subscription()
    
    def can_create_new_project(self):
        """Check if user can create a new project"""
        if self.has_active_pro_subscription():
            return True
        
        from flask import current_app
        # Count user's active projects
        from app.models.experiment import Experiment
        
        active_project_count = Experiment.query.filter_by(
            user_id=self.id, 
            is_deleted=False
        ).count()
        
        return active_project_count < current_app.config['FREE_MAX_PROJECTS']
    
    def has_active_experiment(self):
        """Check if user has any active experiment running"""
        from app.models.experiment import Experiment
        
        running_statuses = ['training', 'tuning']
        return Experiment.query.filter(
            Experiment.user_id == self.id,
            Experiment.status.in_(running_statuses),
            Experiment.is_deleted == False
        ).first() is not None
    
    def days_until_subscription_renewal(self):
        """Get number of days until subscription renewal"""
        if not self.has_active_pro_subscription():
            return 0
            
        delta = self.pro_expires - datetime.utcnow()
        return max(0, delta.days)
    
    def __repr__(self):
        """String representation of the user"""
        return f'<User {self.username}>'