# pro/app/utils/email.py
import os
from flask import current_app, render_template
from flask_mail import Message
from app import mail
from threading import Thread

def send_async_email(app, msg):
    """Send email asynchronously"""
    with app.app_context():
        mail.send(msg)

def send_email(subject, recipients, html_body, text_body=None, sender=None, attachments=None):
    """Send an email with the application context"""
    if sender is None:
        sender = current_app.config['MAIL_DEFAULT_SENDER']
    
    msg = Message(subject, recipients=recipients, html=html_body, body=text_body, sender=sender)
    
    # Add attachments if provided
    if attachments:
        for attachment in attachments:
            if isinstance(attachment, tuple) and len(attachment) == 3:
                filename, content_type, data = attachment
                msg.attach(filename=filename, content_type=content_type, data=data)
    
    # Send email asynchronously
    Thread(
        target=send_async_email,
        args=(current_app._get_current_object(), msg)
    ).start()

def send_verification_email(email, verification_url):
    """Send account verification email"""
    subject = "Verify Your Bhrigu AI Account"
    recipients = [email]
    
    # Render email templates
    html_body = render_template(
        'email/verify_email.html',
        verification_url=verification_url
    )
    text_body = render_template(
        'email/verify_email.txt',
        verification_url=verification_url
    )
    
    send_email(subject, recipients, html_body, text_body)

def send_password_reset_email(email, reset_url):
    """Send password reset email"""
    subject = "Reset Your Bhrigu AI Password"
    recipients = [email]
    
    # Render email templates
    html_body = render_template(
        'email/reset_password.html',
        reset_url=reset_url
    )
    text_body = render_template(
        'email/reset_password.txt',
        reset_url=reset_url
    )
    
    send_email(subject, recipients, html_body, text_body)

def send_otp_email(email, otp):
    """Send one-time password email"""
    subject = "Your Bhrigu AI Login Code"
    recipients = [email]
    
    # Render email templates
    html_body = render_template(
        'email/otp_email.html',
        otp=otp
    )
    text_body = render_template(
        'email/otp_email.txt',
        otp=otp
    )
    
    send_email(subject, recipients, html_body, text_body)

def send_welcome_email(email, user_name):
    """Send welcome email after registration"""
    subject = "Welcome to Bhrigu AI"
    recipients = [email]
    
    # Render email templates
    html_body = render_template(
        'email/welcome.html',
        user_name=user_name
    )
    text_body = render_template(
        'email/welcome.txt',
        user_name=user_name
    )
    
    send_email(subject, recipients, html_body, text_body)

def send_subscription_confirmation(email, user_name, plan, expires_at):
    """Send confirmation email for subscription purchase"""
    subject = f"Your Bhrigu AI {plan} Subscription"
    recipients = [email]
    
    # Render email templates
    html_body = render_template(
        'email/subscription_confirmation.html',
        user_name=user_name,
        plan=plan,
        expires_at=expires_at
    )
    text_body = render_template(
        'email/subscription_confirmation.txt',
        user_name=user_name,
        plan=plan,
        expires_at=expires_at
    )
    
    send_email(subject, recipients, html_body, text_body)

def send_model_completion_notification(email, user_name, experiment_name, model_type, metrics):
    """Send notification when model training is complete"""
    from datetime import datetime
    from flask import url_for
    
    subject = f"Your {model_type} Model is Ready"
    recipients = [email]
    
    # Create timestamp for email
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct model URL (if you have the experiment and model IDs)
    model_url = ""  # If you don't have IDs, leave empty or use a dashboard URL
    
    # Render email templates
    html_body = render_template(
        'email/model_complete.html',
        user_name=user_name,
        model_name=f"{experiment_name} ({model_type})",  # Combine these for a proper model name
        experiment_name=experiment_name,
        model_type=model_type,
        model_url=model_url,
        metrics=metrics,
        timestamp=timestamp
    )
    text_body = render_template(
        'email/model_complete.txt',
        user_name=user_name,
        model_name=f"{experiment_name} ({model_type})",
        experiment_name=experiment_name, 
        model_type=model_type,
        model_url=model_url,
        metrics=metrics,
        timestamp=timestamp
    )
    
    send_email(subject, recipients, html_body, text_body)
    
def send_error_notification(email, user_name, experiment_name, model_type, error_message):
    """Send notification when model training fails"""
    from datetime import datetime
    
    # Add current timestamp
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    subject = f"Error in Your {model_type} Model Training"
    recipients = [email]
    
    # Render email templates
    html_body = render_template(
        'email/model_error.html',
        user_name=user_name,
        experiment_name=experiment_name,
        model_type=model_type,
        error_message=error_message,
        timestamp=current_time
    )
    text_body = render_template(
        'email/model_error.txt',
        user_name=user_name,
        experiment_name=experiment_name,
        model_type=model_type,
        error_message=error_message,
        timestamp=current_time
    )
    
    send_email(subject, recipients, html_body, text_body)