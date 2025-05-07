# pro/app/utils/file_handling.py
import os
import uuid
import imghdr
import requests
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename
from flask import current_app
from app import db
from app.models.file import File
from datetime import datetime

def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, category, user_id=None, experiment_id=None):
    """
    Save an uploaded file and record it in the database
    
    Args:
        file: The uploaded file object
        category: File category (dataset, profile_image, report, etc.)
        user_id: User ID
        experiment_id: Experiment ID (optional)
        
    Returns:
        Tuple of (file_path, file_id, original_filename) or (None, None, None)
    """
    if not file or not file.filename:
        return None, None, None
    
    # Determine allowed extensions based on category
    if category == 'dataset':
        allowed_extensions = current_app.config['ALLOWED_DATASET_EXTENSIONS']
    elif category == 'profile_image':
        allowed_extensions = current_app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif category == 'report':
        allowed_extensions = current_app.config['ALLOWED_REPORT_EXTENSIONS']
    else:
        allowed_extensions = current_app.config['ALLOWED_EXTENSIONS']
    
    if not allowed_file(file.filename, allowed_extensions):
        return None, None, None
    
    # Create a secure filename
    original_filename = secure_filename(file.filename)
    file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    
    # Create a unique filename
    unique_id = str(uuid.uuid4())
    stored_filename = f"{unique_id}_{original_filename}"
    
    # Determine the upload folder based on category
    if category == 'dataset':
        upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'datasets')
    elif category == 'profile_image':
        upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'profile_images')
    elif category == 'report':
        upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'reports')
    else:
        upload_folder = current_app.config['UPLOAD_FOLDER']
    
    # Create upload folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)
    
    # In file_handling.py
    file_path = os.path.abspath(os.path.join(upload_folder, stored_filename))
    file.save(file_path)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Determine MIME type
    if category == 'profile_image':
        file_type = f"image/{imghdr.what(file_path) or 'jpeg'}"
    elif category == 'dataset':
        if file_extension == 'csv':
            file_type = 'text/csv'
        elif file_extension in ['xls', 'xlsx']:
            file_type = 'application/vnd.ms-excel'
        else:
            file_type = 'application/octet-stream'
    elif category == 'report':
        if file_extension == 'pdf':
            file_type = 'application/pdf'
        else:
            file_type = 'application/octet-stream'
    else:
        file_type = 'application/octet-stream'
    
    # Record file in database if user_id is provided
    if user_id:
        file_record = File(
            user_id=user_id,
            experiment_id=experiment_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            file_extension=file_extension,
            file_category=category
        )
        
        db.session.add(file_record)
        db.session.commit()
        
        return file_path, file_record.id, original_filename
    
    return file_path, unique_id, original_filename

def save_profile_image(file, user_id):
    """
    Save a profile image and optimize it
    
    Args:
        file: The uploaded image file
        user_id: User ID
        
    Returns:
        Path to the saved image or None
    """
    if not file or not file.filename:
        return None
    
    # Check if file is an allowed image
    allowed_extensions = current_app.config['ALLOWED_IMAGE_EXTENSIONS']
    if not allowed_file(file.filename, allowed_extensions):
        return None
    
    # Save the original file
    file_path, file_id, _ = save_uploaded_file(file, 'profile_image', user_id)
    
    if not file_path:
        return None
    
    try:
        # Open and optimize the image
        img = Image.open(file_path)
        
        # Resize if needed
        max_size = current_app.config.get('PROFILE_IMAGE_MAX_SIZE', 500)
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))
        
        # Convert to RGB if mode is not compatible
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save optimized image
        img.save(file_path, format='JPEG', quality=85, optimize=True)
        
        return file_path
    except Exception as e:
        current_app.logger.error(f"Error optimizing profile image: {str(e)}")
        return file_path  # Return original path if optimization fails

def save_profile_image_from_url(image_url, user_id):
    """
    Download and save a profile image from URL
    
    Args:
        image_url: URL of the image
        user_id: User ID
        
    Returns:
        Path to the saved image or None
    """
    try:
        # Download image
        response = requests.get(image_url, timeout=5)
        if response.status_code != 200:
            return None
        
        # Create a file-like object from the response content
        file_content = BytesIO(response.content)
        
        # Validate that it's an image
        try:
            img = Image.open(file_content)
            img_format = img.format.lower()
            if img_format not in ['jpeg', 'jpg', 'png', 'gif']:
                return None
        except:
            return None
        
        # Reset file pointer
        file_content.seek(0)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.{img_format}"
        
        # Determine upload folder
        upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'profile_images')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(upload_folder, filename)
        
        # Resize if needed
        max_size = current_app.config.get('PROFILE_IMAGE_MAX_SIZE', 500)
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))
        
        # Convert to RGB if mode is not compatible
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save optimized image
        img.save(file_path, format='JPEG', quality=85, optimize=True)
        
        # Record file in database
        file_size = os.path.getsize(file_path)
        file_type = f"image/jpeg"
        
        file_record = File(
            user_id=user_id,
            experiment_id=None,
            original_filename=filename,
            stored_filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            file_extension='jpg',
            file_category='profile_image'
        )
        
        db.session.add(file_record)
        db.session.commit()
        
        return file_path
    except Exception as e:
        current_app.logger.error(f"Error saving profile image from URL: {str(e)}")
        return None

def delete_file(file_id, user_id=None):
    """
    Delete a file (soft delete in database, keep on disk)
    
    Args:
        file_id: File ID
        user_id: User ID (for permission check)
        
    Returns:
        True if successfully deleted, False otherwise
    """
    file_record = File.query.get(file_id)
    
    if not file_record:
        return False
    
    # Check if user has permission to delete the file
    if user_id and file_record.user_id != user_id:
        return False
    
    # Mark as deleted in database
    file_record.is_deleted = True
    file_record.deleted_at = datetime.utcnow()
    db.session.commit()
    
    return True

def permanently_delete_file(file_id, user_id=None):
    """
    Permanently delete a file (from disk and database)
    
    Args:
        file_id: File ID
        user_id: User ID (for permission check)
        
    Returns:
        True if successfully deleted, False otherwise
    """
    file_record = File.query.get(file_id)
    
    if not file_record:
        return False
    
    # Check if user has permission to delete the file
    if user_id and file_record.user_id != user_id:
        return False
    
    # Delete file from disk
    if os.path.exists(file_record.file_path):
        try:
            os.remove(file_record.file_path)
        except:
            return False
    
    # Delete from database
    db.session.delete(file_record)
    db.session.commit()
    
    return True