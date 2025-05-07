# pro/app/models/file.py
from datetime import datetime
import uuid
from app import db

class File(db.Model):
    """Model for storing file metadata"""
    __tablename__ = 'files'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    experiment_id = db.Column(db.String(36), db.ForeignKey('experiments.id'), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
    file_type = db.Column(db.String(50), nullable=False)  # MIME type
    file_extension = db.Column(db.String(10), nullable=False)
    file_category = db.Column(db.String(20), nullable=False)  # 'dataset', 'profile_image', 'report', etc.
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_deleted = db.Column(db.Boolean, default=False)
    deleted_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<File {self.id}>'
