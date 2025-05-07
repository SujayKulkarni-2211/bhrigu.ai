# pro/app/config.py
import os
from datetime import timedelta

class Config:
    """Base configuration."""
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-bhrigu-pro-change-in-production')
    
    # SQLite database settings
    # Change this line in config.py (around line 9):
    # In config.py, update the SQLite connection string with check_same_thread=False parameter
    SQLALCHEMY_DATABASE_URI = os.environ.get('PRO_DATABASE_URL', 'sqlite:///bhrigu_pro.db?check_same_thread=False')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {'pool_size': 10, 'max_overflow': 20,'pool_recycle': 1800, 'pool_pre_ping': True}
    # Upload settings
    
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
    ALLOWED_DATASET_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    ALLOWED_REPORT_EXTENSIONS = {'pdf', 'docx', 'pptx'}
    ALLOWED_MODEL_EXTENSIONS = {'h5', 'pkl', 'joblib'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    GUEST_SERVICE_URL = os.environ.get('GUEST_SERVICE_URL') or 'http://localhost:5000'
    
    # Model settings
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    REPORT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    
    # User limitations
    FREE_MAX_PROJECTS = 7    # Maximum projects for free users
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    # Stripe settings
    STRIPE_PUBLIC_KEY = os.environ.get('STRIPE_PUBLIC_KEY', 'pk_test_your_test_key')
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_your_test_key')
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', 'sk_test_your_test_key')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', 'sk_test_your_test_key')
    
    # Email settings for notifications
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', 'no-reply@bhrigu.ai')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', 'password')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'no-reply@bhrigu.ai')
    SECURITY_PASSWORD_SALT = os.environ.get('SECURITY_PASSWORD_SALT', 'saltyben')
    
    # Pricing configuration
    PRICING = {
        'pro_subscription': {
            'name': 'Pro Subscription',
            'price': 2499,
            'period': '6 months',
            'description': '6-month subscription to Pro features',
            'features': [
                'Access to all ML models including neural networks',
                'Access to quantum machine learning models',
                'No limit on number of projects',
                'Run experiments without time limit',
                'Advanced hyperparameter tuning',
                'Unlimited experiments',
                'Custom model development',
                'Priority support'
            ],
            'stripe_price_id': 'price_pro_subscription'
        },
        'k_fold_validation': {
            'name': 'K-Fold Cross Validation',
            'price': 99,
            'description': 'One-time payment per project',
            'stripe_price_id': 'price_k_fold'
        },
        'api_hosting': {
            'name': 'ML Model API Hosting',
            'price': 199,
            'description': 'Deploy your model as a REST API',
            'stripe_price_id': 'price_api_hosting'
        }
    }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    # In production, use a proper secret key
    SECRET_KEY = os.environ.get('SECRET_KEY')
    # In production, use PostgreSQL or other production DB
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    # Set proper Stripe keys in production
    STRIPE_PUBLIC_KEY = os.environ.get('STRIPE_PUBLIC_KEY')
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')

# Dictionary to map environment names to config objects
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}