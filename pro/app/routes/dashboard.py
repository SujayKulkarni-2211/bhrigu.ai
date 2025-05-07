# pro/app/routes/dashboard.py
import os
import json
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.user import User
from app.models.experiment import Experiment
from app.utils.file_handling import save_profile_image
from app.utils.file_handling import save_uploaded_file, allowed_file

# Initialize the Blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def index():
    """Dashboard home page displaying all user projects"""
    # Get user's active projects
    projects = Experiment.query.filter_by(
        user_id=current_user.id,
        is_deleted=False
    ).order_by(Experiment.updated_at.desc()).all()
    
    # Count projects by status
    status_counts = {
        'total': len(projects),
        'in_progress': sum(1 for p in projects if p.status in ['data_uploaded', 'analyzed', 'model_selected', 'training']),
        'completed': sum(1 for p in projects if p.status == 'completed'),
        'failed': sum(1 for p in projects if p.status == 'failed')
    }
    
    # Check if user can create new projects (based on subscription)
    can_create_new = current_user.can_create_new_project()
    
    # Get user's subscription status
    is_pro = current_user.is_pro and current_user.has_active_pro_subscription()
    pro_days_remaining = current_user.days_until_subscription_renewal() if is_pro else 0
    
    return render_template(
        'dashboard/index.html',
        user=current_user,
        projects=projects,
        status_counts=status_counts,
        can_create_new=can_create_new,
        is_pro=is_pro,
        pro_days_remaining=pro_days_remaining
    )

@dashboard_bp.route('/dashboard/new-project', methods=['GET', 'POST'])
@login_required
def new_project():
    """Create a new project"""
    # Check if user can create new projects
    if not current_user.can_create_new_project():
        flash('You have reached the maximum number of projects. Please upgrade to Pro to create more projects.', 'warning')
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        # Get form data
        project_name = request.form.get('project_name', '').strip()
        project_description = request.form.get('project_description', '').strip()
        
        if not project_name:
            flash('Project name is required', 'danger')
            return redirect(url_for('dashboard.new_project'))
        
        # Create a new experiment
        experiment = Experiment(
            user_id=current_user.id,
            name=project_name,
            description=project_description,
            status='created'
        )
        
        db.session.add(experiment)
        db.session.commit()
        
        flash('Project created successfully!', 'success')
        return redirect(url_for('dashboard.index'))
    
    return render_template('dashboard/new_project.html')

@dashboard_bp.route('/dashboard/project/<experiment_id>')
@login_required
def view_project(experiment_id):
    """View project details"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Redirect to the project.view route in the project blueprint
    return redirect(url_for('project.view', experiment_id=experiment.id))

@dashboard_bp.route('/dashboard/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page"""
    if request.method == 'POST':
        # Get form data
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        organization = request.form.get('organization', '')
        
        # Update user profile
        current_user.first_name = first_name
        current_user.last_name = last_name
        current_user.organization = organization
        
        # Handle profile image upload
        if 'profile_image' in request.files and request.files['profile_image'].filename:
            profile_image_path = save_profile_image(
                request.files['profile_image'],
                current_user.id
            )
            current_user.profile_image = profile_image_path
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('dashboard.profile'))
    
    # Get user's subscription info
    is_pro = current_user.is_pro and current_user.has_active_pro_subscription()
    pro_days_remaining = current_user.days_until_subscription_renewal() if is_pro else 0
    
    # Get user's active experiments count
    active_experiments = Experiment.query.filter_by(
        user_id=current_user.id,
        is_deleted=False
    ).count()
    
    # Get feature access
    can_use_quantum = current_user.can_use_quantum_models()
    has_k_fold = current_user.has_k_fold
    has_api_hosting = current_user.has_api_hosting
    
    return render_template(
        'dashboard/profile.html',
        user=current_user,
        is_pro=is_pro,
        pro_days_remaining=pro_days_remaining,
        active_experiments=active_experiments,
        can_use_quantum=can_use_quantum,
        has_k_fold=has_k_fold,
        has_api_hosting=has_api_hosting
    )

@dashboard_bp.route('/dashboard/request-password-change', methods=['POST'])
@login_required
def request_password_change():
    """Request a password change"""
    from app.utils.email import send_password_reset_email
    from app.routes.auth import generate_verification_token
    
    # Generate password reset token
    token = generate_verification_token(current_user.email)
    reset_url = url_for('auth.reset_password', token=token, _external=True)
    
    # Send password reset email
    send_password_reset_email(current_user.email, reset_url)
    
    flash('A password reset link has been sent to your email.', 'info')
    return redirect(url_for('dashboard.profile'))

@dashboard_bp.route('/dashboard/delete-project/<experiment_id>', methods=['POST'])
@login_required
def delete_project(experiment_id):
    """Soft delete a project"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Soft delete the experiment
    experiment.soft_delete()
    flash('Project has been deleted.', 'success')
    return redirect(url_for('dashboard.index'))