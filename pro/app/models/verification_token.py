# pro/app/models/verification_token.py
from datetime import datetime
import uuid
from app import db

class VerificationToken(db.Model):
    """Model for storing email verification tokens"""
    __tablename__ = 'verification_tokens'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    token = db.Column(db.String(255), nullable=False, unique=True)
    token_type = db.Column(db.String(20), nullable=False)  # 'email_verification', 'password_reset'
    is_used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    def __repr__(self):
        return f'<VerificationToken {self.id}>'