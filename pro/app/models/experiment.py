# pro/app/models/experiment.py
from datetime import datetime
import uuid
from app import db

class Experiment(db.Model):
    """Model for tracking user experiments"""
    __tablename__ = 'experiments'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    # Basic experiment info
    
    name = db.Column(db.String(100), nullable=True)
    # Add this line with the other fields in the Experiment class
    description = db.Column(db.Text, nullable=True)  # Project description
    dataset_filename = db.Column(db.String(255), nullable=True)
    dataset_original_name = db.Column(db.String(255), nullable=True)
    file_path = db.Column(db.String(255), nullable=True) # Path to the dataset file
    problem_type = db.Column(db.String(20), nullable=True)  # 'classification' or 'regression'
    data_type = db.Column(db.String(20), nullable=True)     # 'tabular' or 'image'
    
    status = db.Column(db.String(20), default='created')  # created, data_uploaded, analyzed, model_selected, training, completed, failed
    current_step = db.Column(db.String(100), nullable=True)
    # Add this with the other fields in the Experiment class
     # Path to the dataset file
    
    # Dataset info
    target_column = db.Column(db.String(100), nullable=True)
    feature_columns = db.Column(db.Text, nullable=True)  # JSON string of column names
    
    # Additional options
    use_k_fold = db.Column(db.Boolean, default=False)
    k_fold_value = db.Column(db.Integer, default=5)
    deploy_as_api = db.Column(db.Boolean, default=False)
    
    # Deletion flag (for soft deletes)
    is_deleted = db.Column(db.Boolean, default=False)
    deleted_at = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # API info (if deployed)
    api_endpoint = db.Column(db.String(255), nullable=True)
    api_key = db.Column(db.String(100), nullable=True)
    
    # Job control
    job_id = db.Column(db.String(100), nullable=True)  # ID for background job tracking
    
    # Relationships
    dataset_info = db.relationship('ProDatasetInfo', backref='experiment', uselist=False, cascade='all, delete-orphan')
    model_runs = db.relationship('ProModelRun', backref='experiment', lazy=True, cascade='all, delete-orphan')
    
    def soft_delete(self):
        """Mark experiment as deleted without removing from database"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        db.session.commit()
    
    def __repr__(self):
        return f'<Experiment {self.id}>'


class ProDatasetInfo(db.Model):
    """Model for storing dataset analysis information (Pro version)"""
    __tablename__ = 'pro_dataset_info'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = db.Column(db.String(36), db.ForeignKey('experiments.id'), nullable=False)
    
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
        return f'<ProDatasetInfo {self.id}>'


class ProModelRun(db.Model):
    """Model for tracking individual ML model runs (Pro version)"""
    __tablename__ = 'pro_model_runs'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = db.Column(db.String(36), db.ForeignKey('experiments.id'), nullable=False)
    
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
    # Add these lines in the ProModelRun class definition
    api_endpoint = db.Column(db.String(255), nullable=True)
    api_key = db.Column(db.String(100), nullable=True)
    # Additional fields for advanced reporting
    tuning_metrics = db.Column(db.Text, nullable=True)  # JSON string of tuning metrics
    tuning_results = db.Column(db.Text, nullable=True)  # JSON string of tuning results
    tuning_report_path = db.Column(db.String(255), nullable=True)  # Path to tuning report
    comprehensive_report_path = db.Column(db.String(255), nullable=True)  # Path to comprehensive report
    feature_importances = db.Column(db.Text, nullable=True)  # JSON string of feature importances
    confusion_matrix = db.Column(db.Text, nullable=True)  # JSON string of confusion matrix
    classification_report = db.Column(db.Text, nullable=True)  # Text of classification report
    
    def __repr__(self):
        return f'<ProModelRun {self.id} - {self.algorithm}>'