"""
Bhrigu.ai - Automated Machine Learning Platform
Entry point for the Flask application
"""
import os

from app import create_app
env = os.environ.get('FLASK_ENV', 'production')
app = create_app(env)

if __name__ == '__main__':
    app.run(debug=True, port=5000)