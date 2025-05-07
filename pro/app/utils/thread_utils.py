# pro/app/utils/thread_utils.py
import os
import json
import traceback
from threading import Thread
from datetime import datetime
import numpy as np

def create_app_context_thread(app, target_func, *args, **kwargs):
    """
    Create a thread with Flask application context and db session management
    
    Args:
        app: Flask application object
        target_func: Function to run in thread
        *args, **kwargs: Arguments to pass to target_func
        
    Returns:
        Thread object that's ready to start
    """
    def wrapper(*args, **kwargs):
        with app.app_context():
            # Import db inside app context to avoid circular imports
            from app import db
            
            # Create a new session for this thread
            session = db.create_scoped_session()
            
            try:
                # Add session to kwargs
                kwargs['session'] = session
                result = target_func(*args, **kwargs)
                return result
            except Exception as e:
                # Log the error and traceback
                error_message = f"{str(e)}\n{traceback.format_exc()}"
                print(f"Thread error: {error_message}")
                
                # Try to call error callback if provided
                if 'error_callback' in kwargs:
                    try:
                        error_callback = kwargs.pop('error_callback')
                        error_callback(str(e))
                    except Exception as callback_error:
                        print(f"Error in error callback: {str(callback_error)}")
                
                # Rollback session if there was an error
                try:
                    session.rollback()
                except:
                    pass
            finally:
                # Always close the session
                try:
                    session.close()
                except:
                    pass
                print("Thread completed, session closed")
    
    thread = Thread(target=wrapper, args=args, kwargs=kwargs)
    thread.daemon = True  # Make thread daemon so it exits when main program exits
    return thread

def serialize_for_json(obj):
    """
    Convert objects to JSON-serializable format
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: serialize_for_json(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    return obj

def resolve_file_path(app, file_path):
    """
    Attempt to resolve the correct file path if the original path is not found
    
    Args:
        app: Flask application object
        file_path: Original file path to resolve
        
    Returns:
        Resolved file path or original path if can't resolve
    """
    if os.path.exists(file_path):
        return file_path
    
    # Try to fix file path using common patterns
    base_name = os.path.basename(file_path)
    
    # Check in UPLOAD_FOLDER/datasets
    alt_path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', base_name)
    if os.path.exists(alt_path1):
        return alt_path1
    
    # Check in UPLOAD_FOLDER directly
    alt_path2 = os.path.join(app.config['UPLOAD_FOLDER'], base_name)
    if os.path.exists(alt_path2):
        return alt_path2
    
    # Check absolute path based on just the filename
    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        if base_name in files:
            return os.path.join(root, base_name)
    
    # If we can't resolve, return original
    return file_path