# app/models/experiment.py
from datetime import datetime
import uuid
from app import db

class GuestExperiment(db.Model):
    """Model for tracking guest user experiments"""
    __tablename__ = 'guest_experiments'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(100), nullable=False)
    dataset_filename = db.Column(db.String(255), nullable=True)
    dataset_original_name = db.Column(db.String(255), nullable=True)
    problem_type = db.Column(db.String(20), nullable=True)  # 'classification' or 'regression'
    data_type = db.Column(db.String(20), nullable=True)     # 'tabular' or 'image'
    
    # Track status
    status = db.Column(db.String(20), default='created')  # created, data_uploaded, analyzed, model_selected, training, completed, failed
    current_step = db.Column(db.String(100), nullable=True)
    
    # Dataset info
    target_column = db.Column(db.String(100), nullable=True)
    feature_columns = db.Column(db.Text, nullable=True)  # JSON string of column names
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    dataset_info = db.relationship('DatasetInfo', backref='experiment', uselist=False, cascade='all, delete-orphan')
    model_runs = db.relationship('ModelRun', backref='experiment', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<GuestExperiment {self.id}>'

class DatasetInfo(db.Model):
    """Model for storing dataset analysis information"""
    __tablename__ = 'dataset_info'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = db.Column(db.String(36), db.ForeignKey('guest_experiments.id'), nullable=False)
    
    # Dataset statistics
    num_rows = db.Column(db.Integer, nullable=True)
    num_columns = db.Column(db.Integer, nullable=True)
    column_names = db.Column(db.Text, nullable=True)  # JSON string of column names
    column_types = db.Column(db.Text, nullable=True)  # JSON string of column types
    missing_values = db.Column(db.Text, nullable=True)  # JSON string of missing value counts
    
    # Numerical columns stats
    numeric_columns = db.Column(db.Text, nullable=True)  # JSON string of numeric column names
    categorical_columns = db.Column(db.Text, nullable=True)  # JSON string of categorical column names
    statistics = db.Column(db.Text, nullable=True)  # JSON string of descriptive statistics
    
    # Visualization data
    correlation_matrix = db.Column(db.Text, nullable=True)  # Base64 encoded image
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<DatasetInfo {self.id}>'

class ModelRun(db.Model):
    """Model for tracking individual ML model runs"""
    __tablename__ = 'model_runs'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = db.Column(db.String(36), db.ForeignKey('guest_experiments.id'), nullable=False)
    
    # Model information
    algorithm = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)  # 'classical', 'neural', 'quantum'
    hyperparameters = db.Column(db.Text, nullable=True)  # JSON string of hyperparameters
    
    # Status and outputs
    status = db.Column(db.String(20), default='created')  # created, training, completed, failed, timed_out
    logs = db.Column(db.Text, nullable=True)  # Training logs
    tuning_logs = db.Column(db.Text, nullable=True)  # Hyperparameter tuning logs
    model_path = db.Column(db.String(255), nullable=True)  # Path to saved model file
    metrics = db.Column(db.Text, nullable=True)  # JSON string of performance metrics
    report_path = db.Column(db.String(255), nullable=True)  # Path to generated report
    hyperparameters_tuning = db.Column(db.Boolean, default=False)  # Whether hyperparameter tuning was performed
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    tuning_metrics = db.Column(db.Text, nullable=True)  # JSON string of tuning metrics
    tuning_results = db.Column(db.Text, nullable=True)  # JSON string of tuning results
    tuning_report_path = db.Column(db.String(255), nullable=True)  # Path to tuning report
    comprehensive_report_path = db.Column(db.String(255), nullable=True)  # Path to comprehensive report
    feature_importances = db.Column(db.Text, nullable=True)  # JSON string of feature importances
    confusion_matrix = db.Column(db.Text, nullable=True)  # JSON string of confusion matrix
    classification_report = db.Column(db.Text, nullable=True)  # Text of classification report
    
    def __repr__(self):
        return f'<ModelRun {self.id} - {self.algorithm}>'