"""
Bhrigu.ai - Flask application initialization
"""

import os
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
load_dotenv()
# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app(config_name='development'):
    """Create and configure the Flask application"""
    # Create Flask app
    app = Flask(__name__, instance_relative_config=True)
    
    # Load default configuration from config.py
    from app.config import config
    app.config.from_object(config[config_name])
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Ensure upload, models, and reports directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
    
    # Initialize extensions
    db.init_app(app)
    
    # Register blueprints
    from app.routes.main import main_bp
    app.register_blueprint(main_bp)
    
    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errors/500.html'), 500
    
    # Create database tables if they don't exist
    with app.app_context():
        from app.models import experiment
        db.create_all()
    
    # Add Jinja template filter for JSON parsing
    @app.template_filter('from_json')
    def from_json(value):
        import json
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return []
    
    return app