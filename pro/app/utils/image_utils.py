# pro/app/utils/image_utils.py
import os
import uuid
import random
from PIL import Image, ImageDraw, ImageFont
from flask import current_app

def generate_initial_avatar(initials, user_id=None, size=500):
    """
    Generate an avatar image with user initials on a colored background
    
    Args:
        initials: User initials (1-2 characters)
        user_id: User ID for filename
        size: Image size in pixels (square)
        
    Returns:
        Path to the generated image
    """
    # Limit initials to 2 characters
    initials = initials[:2].upper()
    
    # Generate a unique filename
    unique_id = user_id or str(uuid.uuid4())
    filename = f"avatar_{unique_id}.jpg"
    
    # Determine upload folder
    upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'profile_images')
    os.makedirs(upload_folder, exist_ok=True)
    
    # Generate a random background color (ensure it's not too light)
    background_colors = [
        (66, 133, 244),   # Google Blue
        (219, 68, 55),    # Google Red
        (244, 180, 0),    # Google Yellow
        (15, 157, 88),    # Google Green
        (98, 71, 170),    # Purple
        (0, 131, 143),    # Teal
        (16, 120, 121),   # Dark Teal
        (198, 40, 40),    # Dark Red
        (46, 125, 50),    # Dark Green
        (124, 77, 255),   # Bright Purple
        (21, 101, 192),   # Dark Blue
        (230, 74, 25),    # Orange
        (0, 96, 100),     # Deep Teal
        (136, 14, 79),    # Pink
        (74, 20, 140)     # Indigo
    ]
    
    # Use hash of initials to consistently get same color for same user
    color_index = hash(initials) % len(background_colors)
    background_color = background_colors[color_index]
    
    # Calculate text color (white or black based on background brightness)
    brightness = (0.299 * background_color[0] + 0.587 * background_color[1] + 0.114 * background_color[2]) / 255
    text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
    
    # Create a new image with the background color
    img = Image.new('RGB', (size, size), background_color)
    draw = ImageDraw.Draw(img)
    
    # Load a font
    try:
        # Try to load a built-in font
        font_size = int(size * 0.4)
        font_path = os.path.join(current_app.root_path, 'static', 'fonts', 'Roboto-Bold.ttf')
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Fall back to default font
            font = ImageFont.load_default()
    except Exception as e:
        current_app.logger.error(f"Error loading font: {str(e)}")
        font = ImageFont.load_default()
    
    # Get text size
    text_width, text_height = draw.textsize(initials, font=font) if hasattr(draw, 'textsize') else (size//2, size//2)
    
    # Calculate text position (centered)
    text_position = ((size - text_width) // 2, (size - text_height) // 2)
    
    # Draw text
    draw.text(text_position, initials, font=font, fill=text_color)
    
    # Save the image
    file_path = os.path.join(upload_folder, filename)
    img.save(file_path, format='JPEG', quality=90, optimize=True)
    
    return file_path

def create_thumbnail(source_path, width=200, height=200):
    """
    Create a thumbnail from an image
    
    Args:
        source_path: Path to the source image
        width: Thumbnail width
        height: Thumbnail height
        
    Returns:
        Path to the generated thumbnail
    """
    if not os.path.exists(source_path):
        return None
    
    try:
        # Open the source image
        img = Image.open(source_path)
        
        # Generate a unique filename
        unique_id = str(uuid.uuid4())
        thumbnail_filename = f"thumb_{unique_id}.jpg"
        
        # Determine thumbnail folder (same directory as source)
        source_dir = os.path.dirname(source_path)
        thumbnail_path = os.path.join(source_dir, thumbnail_filename)
        
        # Create thumbnail
        img.thumbnail((width, height), Image.LANCZOS)
        
        # Save thumbnail
        img.save(thumbnail_path, format='JPEG', quality=85, optimize=True)
        
        return thumbnail_path
    except Exception as e:
        current_app.logger.error(f"Error creating thumbnail: {str(e)}")
        return None