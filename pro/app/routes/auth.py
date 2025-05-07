# pro/app/routes/auth.py
import os
import json
import uuid
import random
import string
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.urls import url_parse
from app import db, login_manager
from app.models.user import User
from app.models.otp import OTP
from app.models.verification_token import VerificationToken
from app.utils.email import send_password_reset_email, send_verification_email, send_otp_email
from functools import wraps
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from flask_oauthlib.client import OAuth
import pyotp

# Initialize the Blueprint
auth_bp = Blueprint('auth', __name__)

# Initialize OAuth
oauth = OAuth()

# Configure Google OAuth
google = oauth.remote_app(
    'google',
    consumer_key=os.environ.get('GOOGLE_CLIENT_ID'),
    consumer_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    request_token_params={
        'scope': 'email profile'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return User.query.get(user_id)

# Index route serves as the login page
@auth_bp.route('/', methods=['GET'])
def index():
    """Main landing page / login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    return render_template('auth/login.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    # Check if there's a redirect parameter from Guest service
    next_page = request.args.get('next')
    
    # Handle the login form submission
    if request.method == 'POST':
        # Existing login logic
        email = request.form.get('email')
        password = request.form.get('password')
        remember_me = 'remember_me' in request.form
        
        user = User.query.filter_by(email=email).first()
        
        if user is None or not user.check_password(password):
            flash('Invalid email or password', 'danger')
            return render_template('auth/login.html', title='Sign In')
        
        if not user.is_verified:
            session['unverified_email'] = email
            flash('Please verify your email first', 'warning')
            return redirect(url_for('auth.resend_verification'))
        
        login_user(user, remember=remember_me)
        
        if next_page and url_parse(next_page).netloc == '':
            return redirect(next_page)
        return redirect(url_for('dashboard.index'))
    
    # If this is a GET request, just render the login template
    # Pass the next parameter to the template if it exists
    return render_template('auth/login.html', title='Sign In', next=next_page)

# Add a new route to handle redirection back to Guest service if needed
@auth_bp.route('/return-to-guest')
def return_to_guest():
    guest_url = os.environ.get('GUEST_SERVICE_URL', 'http://localhost:5000')
    return redirect(f"{guest_url}/")

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('auth/register.html')
        
        # Check if email or username already exists
        email_exists = User.query.filter_by(email=email.lower()).first()
        if email_exists:
            flash('Email already registered', 'danger')
            return render_template('auth/register.html')
        
        username_exists = User.query.filter_by(username=username).first()
        if username_exists:
            flash('Username already taken', 'danger')
            return render_template('auth/register.html')
        
        # Create a new user
        new_user = User(
            email=email.lower(),
            username=username,
            password=password,
            is_verified=False  # User needs to verify email
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        # Generate verification token
        token = generate_verification_token(new_user.email)
        verification_url = url_for('auth.verify_email', token=token, _external=True)
        
        # Send verification email
        send_verification_email(new_user.email, verification_url)
        
        flash('Registration successful! Please check your email to verify your account.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html')

@auth_bp.route('/complete-profile', methods=['GET', 'POST'])
@login_required
def complete_profile():
    """Complete user profile after OAuth registration"""
    # If profile is already complete, redirect to dashboard
    if current_user.first_name and current_user.last_name:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        organization = request.form.get('organization', '')
        
        # Update user profile
        current_user.first_name = first_name
        current_user.last_name = last_name
        current_user.organization = organization
        
        # Handle profile image upload
        if 'profile_image' in request.files and request.files['profile_image'].filename:
            from app.utils.file_handling import save_profile_image
            profile_image_path = save_profile_image(
                request.files['profile_image'],
                current_user.id
            )
            current_user.profile_image = profile_image_path
        else:
            # Generate profile image with user's initials
            from app.utils.image_utils import generate_initial_avatar
            initials = first_name[0].upper() + (last_name[0].upper() if last_name else '')
            avatar_path = generate_initial_avatar(initials, current_user.id)
            current_user.profile_image = avatar_path
        
        db.session.commit()
        flash('Profile completed successfully!', 'success')
        return redirect(url_for('dashboard.index'))
    
    return render_template('auth/complete_profile.html')

@auth_bp.route('/verify-email/<token>')
def verify_email(token):
    """Verify user email with token"""
    try:
        email = verify_token(token)
        user = User.query.filter_by(email=email).first()
        
        if not user:
            flash('Invalid verification link', 'danger')
            return redirect(url_for('auth.login'))
        
        if user.is_verified:
            flash('Your account is already verified. Please log in.', 'info')
            return redirect(url_for('auth.login'))
        
        # Verify the user
        user.is_verified = True
        db.session.commit()
        
        flash('Your account has been verified successfully! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    except SignatureExpired:
        flash('The verification link has expired. Please request a new one.', 'danger')
        return redirect(url_for('auth.resend_verification'))
    
    except BadSignature:
        flash('Invalid verification link', 'danger')
        return redirect(url_for('auth.login'))

@auth_bp.route('/resend-verification', methods=['GET', 'POST'])
def resend_verification():
    """Resend verification email"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email.lower()).first()
        
        if not user:
            flash('Email not found', 'danger')
            return render_template('auth/resend_verification.html')
        
        if user.is_verified:
            flash('Your account is already verified. Please log in.', 'info')
            return redirect(url_for('auth.login'))
        
        # Generate verification token
        token = generate_verification_token(user.email)
        verification_url = url_for('auth.verify_email', token=token, _external=True)
        
        # Send verification email
        send_verification_email(user.email, verification_url)
        
        flash('Verification email has been resent. Please check your inbox.', 'success')
        return redirect(url_for('auth.login'))
    
    email = request.args.get('email', '')
    return render_template('auth/resend_verification.html', email=email)

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password requests"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email.lower()).first()
        
        if not user:
            # For security reasons, we'll still show a success message
            flash('If the email is registered, a password reset link has been sent.', 'info')
            return redirect(url_for('auth.login'))
        
        # Generate password reset token
        token = generate_verification_token(user.email)
        reset_url = url_for('auth.reset_password', token=token, _external=True)
        
        # Send password reset email
        send_password_reset_email(user.email, reset_url)
        
        flash('If the email is registered, a password reset link has been sent.', 'info')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/forgot_password.html')

@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Reset password with token"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    try:
        email = verify_token(token)
        user = User.query.filter_by(email=email).first()
        
        if not user:
            flash('Invalid reset link', 'danger')
            return redirect(url_for('auth.login'))
        
        if request.method == 'POST':
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('auth/reset_password.html', token=token)
            
            user.set_password(password)
            db.session.commit()
            
            flash('Your password has been reset successfully! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        
        return render_template('auth/reset_password.html', token=token)
    
    except SignatureExpired:
        flash('The reset link has expired. Please request a new one.', 'danger')
        return redirect(url_for('auth.forgot_password'))
    
    except BadSignature:
        flash('Invalid reset link', 'danger')
        return redirect(url_for('auth.login'))

@auth_bp.route('/login-with-otp', methods=['GET', 'POST'])
def login_with_otp():
    """Login with one-time password via email"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email.lower()).first()
        
        if not user:
            flash('Email not found', 'danger')
            return render_template('auth/login_with_otp.html')
        
        if not user.is_verified:
            flash('Please verify your email before logging in.', 'warning')
            return redirect(url_for('auth.resend_verification', email=email))
        
        # Generate OTP
        otp_code = ''.join(random.choices(string.digits, k=6))
        otp_expires = datetime.utcnow() + timedelta(minutes=10)
        
        # Save OTP to database
        otp = OTP(
            user_id=user.id,
            code=otp_code,
            expires_at=otp_expires
        )
        db.session.add(otp)
        db.session.commit()
        
        # Send OTP to user's email
        send_otp_email(user.email, otp_code)
        
        # Store email in session for verification
        session['otp_email'] = user.email
        
        flash('An OTP has been sent to your email. It will expire in 10 minutes.', 'info')
        return redirect(url_for('auth.verify_otp'))
    
    return render_template('auth/login_with_otp.html')

@auth_bp.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    """Verify OTP for login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    # Check if OTP email is in session
    if 'otp_email' not in session:
        flash('Please request an OTP first', 'warning')
        return redirect(url_for('auth.login_with_otp'))
    
    if request.method == 'POST':
        otp_code = request.form.get('otp')
        email = session['otp_email']
        
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Invalid session', 'danger')
            return redirect(url_for('auth.login'))
        
        # Find active OTP for user
        otp = OTP.query.filter_by(
            user_id=user.id,
            code=otp_code,
            is_used=False
        ).first()
        
        if not otp or otp.expires_at < datetime.utcnow():
            flash('Invalid or expired OTP', 'danger')
            return render_template('auth/verify_otp.html')
        
        # Mark OTP as used
        otp.is_used = True
        db.session.commit()
        
        # Clear session
        session.pop('otp_email', None)
        
        # Login user
        login_user(user)
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard.index'))
    
    return render_template('auth/verify_otp.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))

# Google OAuth routes
@auth_bp.route('/login/google')
def login_google():
    """Initiate Google OAuth login"""
    return google.authorize(callback=url_for('auth.google_authorized', _external=True))

@auth_bp.route('/login/google/callback')
def google_authorized():
    """Handle Google OAuth callback"""
    resp = google.authorized_response()
    if resp is None or resp.get('access_token') is None:
        flash('Access denied: reason={} error={}'.format(
            request.args['error_reason'],
            request.args['error_description']
        ), 'danger')
        return redirect(url_for('auth.login'))
    
    session['google_token'] = (resp['access_token'], '')
    user_info = google.get('userinfo')
    
    # Extract user information
    google_id = user_info.data['id']
    email = user_info.data['email']
    username = email.split('@')[0]  # Use first part of email as username
    first_name = user_info.data.get('given_name', '')
    last_name = user_info.data.get('family_name', '')
    profile_image = user_info.data.get('picture', '')
    
    # Check if user exists
    user = User.query.filter_by(email=email).first()
    
    if user:
        # Update OAuth information
        user.google_id = google_id
        
        # Update profile info if not set
        if not user.first_name and first_name:
            user.first_name = first_name
        if not user.last_name and last_name:
            user.last_name = last_name
        if not user.profile_image and profile_image:
            # Save profile image from URL
            from app.utils.file_handling import save_profile_image_from_url
            profile_image_path = save_profile_image_from_url(
                profile_image,
                user.id
            )
            user.profile_image = profile_image_path
        
        db.session.commit()
    else:
        # Create a new user
        # Generate a secure random password
        random_password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        
        user = User(
            email=email,
            username=generate_unique_username(username),
            password=random_password,
            google_id=google_id,
            first_name=first_name,
            last_name=last_name,
            is_verified=True  # OAuth users are auto-verified
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Save profile image if available
        if profile_image:
            from app.utils.file_handling import save_profile_image_from_url
            profile_image_path = save_profile_image_from_url(
                profile_image,
                user.id
            )
            user.profile_image = profile_image_path
            db.session.commit()
    
    # Login user
    login_user(user)
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Redirect to dashboard or profile completion
    if not user.first_name or not user.last_name:
        return redirect(url_for('auth.complete_profile'))
    
    flash('Login successful!', 'success')
    return redirect(url_for('dashboard.index'))

@google.tokengetter
def get_google_oauth_token():
    """Retrieve OAuth token from session"""
    return session.get('google_token')

# Helper functions
def generate_verification_token(email):
    """Generate a secure token for email verification"""
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.dumps(email, salt=current_app.config['SECURITY_PASSWORD_SALT'])

def verify_token(token, expiration=3600):
    """Verify token and return email"""
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.loads(
        token,
        salt=current_app.config['SECURITY_PASSWORD_SALT'],
        max_age=expiration
    )

def generate_unique_username(base_username):
    """Generate a unique username by appending numbers if needed"""
    username = base_username
    counter = 1
    
    while User.query.filter_by(username=username).first():
        username = f"{base_username}{counter}"
        counter += 1
    
    return username