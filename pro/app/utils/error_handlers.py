import os
import json
import traceback
from flask import render_template, jsonify, current_app, request

def register_error_handlers(app):
    """Register error handlers for the Flask application"""
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors"""
        if request.is_xhr or request.path.startswith('/api/'):
            return jsonify(error="Not found"), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        """Handle 500 errors"""
        error_traceback = traceback.format_exc()
        
        # Log the error
        app.logger.error(f"500 error: {str(e)}\n{error_traceback}")
        
        if request.is_xhr or request.path.startswith('/api/'):
            return jsonify(error="Internal server error"), 500
            
        return render_template(
            'errors/500.html', 
            error=str(e),
            traceback=error_traceback if app.debug else None
        ), 500
    
    @app.errorhandler(403)
    def forbidden(e):
        """Handle 403 errors"""
        if request.is_xhr or request.path.startswith('/api/'):
            return jsonify(error="Forbidden"), 403
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Handle unhandled exceptions"""
        error_traceback = traceback.format_exc()
        
        # Log the error
        app.logger.error(f"Unhandled exception: {str(e)}\n{error_traceback}")
        
        if request.is_xhr or request.path.startswith('/api/'):
            return jsonify(error=str(e)), 500
            
        return render_template(
            'errors/error.html',
            error_message=str(e),
            traceback=error_traceback if app.debug else None
        ), 500
