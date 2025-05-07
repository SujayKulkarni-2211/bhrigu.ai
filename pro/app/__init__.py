# pro/app/__init__.py
import os
import json
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_mail import Mail

from datetime import datetime
import numpy as np

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
mail = Mail()

login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Custom JSON encoder for handling numpy arrays and other complex types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def create_app(config_name='default'):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Load configuration
    from app.config import config
    app.config.from_object(config[config_name])
    
    # Set custom JSON encoder
    app.json_encoder = CustomJSONEncoder
    
    # Add these lines for SQLite threading support
    from sqlalchemy.pool import StaticPool
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'connect_args': {'check_same_thread': False},
        'poolclass': StaticPool
    }
    
    # Create necessary folders if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'profile_images'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'reports'), exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
    
    # Fix for absolute path
    app.config['UPLOAD_FOLDER'] = os.path.abspath(app.config['UPLOAD_FOLDER'])
    app.config['MODEL_FOLDER'] = os.path.abspath(app.config['MODEL_FOLDER'])
    app.config['REPORT_FOLDER'] = os.path.abspath(app.config['REPORT_FOLDER'])
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    mail.init_app(app)
    
    # Set up user loader for Flask-Login
    from app.models.user import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(user_id)
    
    # Register blueprints
    with app.app_context():
        # Import blueprints here to avoid circular imports
        from app.routes.auth import auth_bp
        from app.routes.dashboard import dashboard_bp
        from app.routes.payment import payment_bp
        from app.routes.project import project_bp
        
        app.register_blueprint(auth_bp)  # Root routes are in auth_bp
        app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
        app.register_blueprint(payment_bp, url_prefix='/payment')
        app.register_blueprint(project_bp, url_prefix='/project')
        
        # Create database tables if they don't exist
        db.create_all()
    
    from sqlalchemy.pool import StaticPool
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'connect_args': {'check_same_thread': False},
        'poolclass': StaticPool
    }
    
    # Custom template filters for safe JSON handling
    @app.template_filter('tojson_safe')
    def tojson_safe_filter(obj):
        """Filter for safely converting objects to JSON in templates"""
        try:
            return json.dumps(obj, cls=CustomJSONEncoder)
        except:
            return '"{}"'
    
    @app.template_filter('parse_json')
    def parse_json_filter(json_str, default=None):
        """Filter for safely parsing JSON strings in templates"""
        if not json_str:
            return default
        try:
            return json.loads(json_str)
        except:
            return default
    
    # Error handlers with improved error messages
    @app.errorhandler(404)
    def not_found_error(error):
        if request.is_xhr or request.path.startswith('/api/'):
            return jsonify(error="Not found"), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        import traceback
        error_traceback = traceback.format_exc()
        app.logger.error(f"500 error: {str(error)}\n{error_traceback}")
        
        if request.is_xhr or request.path.startswith('/api/'):
            return jsonify(error="Internal server error"), 500
            
        return render_template(
            'errors/500.html', 
            error=str(error),
            traceback=error_traceback if app.debug else None
        ), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        db.session.rollback()
        import traceback
        error_traceback = traceback.format_exc()
        app.logger.error(f"Unhandled exception: {str(error)}\n{error_traceback}")
        
        if request.is_xhr or request.path.startswith('/api/'):
            return jsonify(error=str(error)), 500
            
        return render_template(
            'errors/error.html',
            error_message=str(error),
            traceback=error_traceback if app.debug else None
        ), 500

    # Template utility functions
    @app.context_processor
    def inject_now():
        return {'current_year': datetime.utcnow().year}

    @app.context_processor
    def utility_processor():
        def getStatusColor(status):
            status_colors = {
                'created': 'secondary',
                'data_uploaded': 'info',
                'analyzed': 'primary',
                'model_selected': 'primary',
                'training': 'warning',
                'tuning': 'warning',
                'completed': 'success',
                'failed': 'danger',
                'timed_out': 'danger'
            }
            return status_colors.get(status, 'secondary')
        
        # Safe JSON parsing for template
        def safe_json(json_str, default=None):
            if not json_str:
                return default
            try:
                return json.loads(json_str)
            except:
                return default
        
        # Fix file paths
        def resolve_file_path(file_path):
            """Resolve file path if it doesn't exist"""
            if os.path.exists(file_path):
                return file_path
                
            # Try common patterns
            base_name = os.path.basename(file_path)
            
            # Check in upload folder/datasets
            alt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', base_name)
            if os.path.exists(alt_path):
                return alt_path
                
            # Check in upload folder
            alt_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name)
            if os.path.exists(alt_path):
                return alt_path
                
            return file_path
    
        return {
            'getStatusColor': getStatusColor,
            'safe_json': safe_json,
            'resolve_file_path': resolve_file_path,
            'json': json  # Make json module available in templates
        }
    
    # Fix for request handling
    from flask import request, g
    
    @app.before_request
    def before_request():
        """Set up data for each request"""
        g.json_safe = CustomJSONEncoder().encode
        
        # Add AJAX detection
        request.is_xhr = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    return app