# pro/app/routes/payment.py
import os
import json
import uuid
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_required, current_user
from app import db
from app.models.user import User
from app.models.subscription import Subscription, Transaction, ApiUsage
from app.models.experiment import Experiment
from app.utils.email import send_subscription_confirmation

# Initialize the Blueprint
payment_bp = Blueprint('payment', __name__)

@payment_bp.route('/payment/upgrade-to-pro')
@login_required
def upgrade_to_pro():
    """Pro subscription upgrade page"""
    # Check if user already has Pro
    if current_user.is_pro and current_user.has_active_pro_subscription():
        flash('You already have an active Pro subscription', 'info')
        return redirect(url_for('dashboard.index'))
    
    # Get pricing from config
    pricing = current_app.config.get('PRICING', {})
    pro_pricing = pricing.get('pro_subscription', {})
    
    # Get monthly and yearly prices
    monthly_price = 249
    yearly_price = 2499
    monthly_features = pro_pricing.get('features', [])
    
    return render_template(
        'payment/upgrade_to_pro.html',
        monthly_price=monthly_price,
        yearly_price=yearly_price,
        features=monthly_features
    )

@payment_bp.route('/payment/process-pro-payment', methods=['POST'])
@login_required
def process_pro_payment():
    """Process Pro subscription payment"""
    # Check if user already has Pro
    if current_user.is_pro and current_user.has_active_pro_subscription():
        flash('You already have an active Pro subscription', 'info')
        return redirect(url_for('dashboard.index'))
    
    # Get subscription type
    subscription_type = request.form.get('subscription_type', 'yearly')
    
    # Determine price and duration
    if subscription_type == 'monthly':
        price = 249
        duration_days = 30
    else:  # yearly
        price = 2499
        duration_days = 365
    
    # Simulate payment processing
    # In a real app, you would integrate with a payment gateway like Razorpay
    try:
        # Create a recurring subscription for Pro
        subscription = Subscription(
            user_id=current_user.id,
            plan_type='pro',
            subscription_id=f"pro_{uuid.uuid4()}",
            payment_id=f"payment_{uuid.uuid4()}",
            amount=price,
            currency='INR',
            status='active',
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=duration_days),
            is_recurring=True,
            billing_cycle=subscription_type,
            auto_renew=True
        )
        
        db.session.add(subscription)
        
        # Create transaction record
        transaction = Transaction(
            user_id=current_user.id,
            subscription_id=subscription.id,
            transaction_id=f"txn_{uuid.uuid4()}",
            payment_method='credit_card',  # This would come from payment gateway
            amount=price,
            currency='INR',
            status='completed',
            description=f"Pro {subscription_type.capitalize()} Subscription",
            transaction_type='subscription'
        )
        
        db.session.add(transaction)
        
        # Update user
        current_user.is_pro = True
        current_user.pro_since = datetime.utcnow()
        current_user.pro_expires = datetime.utcnow() + timedelta(days=duration_days)
        current_user.stripe_subscription_id = subscription.subscription_id
        
        db.session.commit()
        
        # Send subscription confirmation email
        send_subscription_confirmation(
            current_user.email, 
            current_user.get_full_name(),
            f"Pro {subscription_type.capitalize()}",
            subscription.end_date
        )
        
        flash(f'Payment successful! Your Pro subscription is now active.', 'success')
        return redirect(url_for('dashboard.index'))
    
    except Exception as e:
        flash(f'Payment failed: {str(e)}', 'danger')
        return redirect(url_for('payment.upgrade_to_pro'))

@payment_bp.route('/payment/cancel-subscription', methods=['POST'])
@login_required
def cancel_subscription():
    """Cancel Pro subscription"""
    # Check if user has an active subscription
    if not current_user.is_pro or not current_user.has_active_pro_subscription():
        flash('You do not have an active Pro subscription to cancel', 'warning')
        return redirect(url_for('dashboard.profile'))
    
    try:
        # Find active subscription
        subscription = Subscription.query.filter_by(
            user_id=current_user.id,
            plan_type='pro',
            status='active'
        ).first()
        
        if not subscription:
            flash('No active subscription found', 'warning')
            return redirect(url_for('dashboard.profile'))
        
        # Update subscription record
        subscription.status = 'cancelled'
        subscription.cancelled_at = datetime.utcnow()
        subscription.auto_renew = False
        
        # In a real implementation, this would also call the payment gateway's API
        # to cancel the subscription on their end
        
        # Keep user access until the subscription period ends
        # (pro_expires date remains unchanged)
        
        db.session.commit()
        
        flash('Your Pro subscription has been cancelled. You will have access until the end of your billing period.', 'info')
        return redirect(url_for('dashboard.profile'))
    
    except Exception as e:
        flash(f'Error cancelling subscription: {str(e)}', 'danger')
        return redirect(url_for('dashboard.profile'))

@payment_bp.route('/payment/k-fold/<experiment_id>')
@login_required
def pay_for_k_fold(experiment_id):
    """Payment page for K-fold cross-validation"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user already has k-fold for this experiment
    existing_subscription = Subscription.query.filter_by(
        user_id=current_user.id,
        plan_type='k_fold',
        experiment_id=experiment.id,
        status='active'
    ).first()
    
    if existing_subscription or current_user.has_k_fold:
        flash('You already have K-fold cross-validation enabled for this experiment', 'info')
        return redirect(url_for('project.k_fold_evaluation', experiment_id=experiment.id))
    
    # Get pricing from config
    pricing = current_app.config.get('PRICING', {})
    k_fold_pricing = pricing.get('k_fold_validation', {})
    price = k_fold_pricing.get('price', 99)
    
    return render_template(
        'payment/k_fold_payment.html',
        experiment=experiment,
        price=price
    )

@payment_bp.route('/payment/process-k-fold-payment/<experiment_id>', methods=['POST'])
@login_required
def process_k_fold_payment(experiment_id):
    """Process payment for K-fold cross-validation"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user already has k-fold subscription
    if current_user.has_k_fold:
        flash('You already have K-fold cross-validation enabled', 'info')
        return redirect(url_for('project.k_fold_evaluation', experiment_id=experiment.id))
    
    # Get pricing from config
    pricing = current_app.config.get('PRICING', {})
    k_fold_pricing = pricing.get('k_fold_validation', {})
    price = k_fold_pricing.get('price', 99)
    
    # Simulate payment processing
    try:
        # Create a one-time subscription for k-fold
        subscription = Subscription(
            user_id=current_user.id,
            plan_type='k_fold',
            subscription_id=f"k_fold_{uuid.uuid4()}",
            payment_id=f"payment_{uuid.uuid4()}",
            amount=price,
            currency='INR',
            status='active',
            start_date=datetime.utcnow(),
            experiment_id=experiment.id,
            is_one_time=True
        )
        
        db.session.add(subscription)
        
        # Create transaction record
        transaction = Transaction(
            user_id=current_user.id,
            subscription_id=subscription.id,
            transaction_id=f"txn_{uuid.uuid4()}",
            payment_method='credit_card',  # This would come from payment gateway
            amount=price,
            currency='INR',
            status='completed',
            description=f"K-Fold Cross-Validation for {experiment.name}",
            transaction_type='payment'
        )
        
        db.session.add(transaction)
        
        # Update user
        current_user.has_k_fold = True
        
        db.session.commit()
        
        # Send subscription confirmation email
        send_subscription_confirmation(
            current_user.email,
            current_user.get_full_name(),
            'K-Fold Cross-Validation',
            None  # No expiration for one-time purchase
        )
        
        flash('Payment successful! K-fold cross-validation is now enabled for this project.', 'success')
        return redirect(url_for('project.k_fold_evaluation', experiment_id=experiment.id))
    
    except Exception as e:
        flash(f'Payment failed: {str(e)}', 'danger')
        return redirect(url_for('payment.pay_for_k_fold', experiment_id=experiment.id))

@payment_bp.route('/payment/api-hosting/<experiment_id>/<model_run_id>')
@login_required
def pay_for_api_hosting(experiment_id, model_run_id):
    """Payment page for API hosting"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = experiment.model_runs.filter_by(id=model_run_id).first_or_404()
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user already has API hosting
    if current_user.has_api_hosting:
        flash('You already have API hosting enabled', 'info')
        return redirect(url_for('project.deploy_model_api', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Get pricing from config
    pricing = current_app.config.get('PRICING', {})
    api_pricing = pricing.get('api_hosting', {})
    price = api_pricing.get('price', 199)
    
    return render_template(
        'payment/api_hosting_payment.html',
        experiment=experiment,
        model_run=model_run,
        price=price
    )

@payment_bp.route('/payment/process-api-hosting-payment/<experiment_id>/<model_run_id>', methods=['POST'])
@login_required
def process_api_hosting_payment(experiment_id, model_run_id):
    """Process payment for API hosting"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = experiment.model_runs.filter_by(id=model_run_id).first_or_404()
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user already has API hosting
    if current_user.has_api_hosting:
        flash('You already have API hosting enabled', 'info')
        return redirect(url_for('project.deploy_model_api', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Get pricing from config
    pricing = current_app.config.get('PRICING', {})
    api_pricing = pricing.get('api_hosting', {})
    price = api_pricing.get('price', 199)
    
    # Simulate payment processing
    try:
        # Create a subscription for API hosting (valid for 30 days)
        subscription = Subscription(
            user_id=current_user.id,
            plan_type='api_hosting',
            subscription_id=f"api_{uuid.uuid4()}",
            payment_id=f"payment_{uuid.uuid4()}",
            amount=price,
            currency='INR',
            status='active',
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=30),  # 30 days validity
            experiment_id=experiment.id,
            is_one_time=True
        )
        
        db.session.add(subscription)
        
        # Create transaction record
        transaction = Transaction(
            user_id=current_user.id,
            subscription_id=subscription.id,
            transaction_id=f"txn_{uuid.uuid4()}",
            payment_method='credit_card',  # This would come from payment gateway
            amount=price,
            currency='INR',
            status='completed',
            description=f"API Hosting for {experiment.name}",
            transaction_type='payment'
        )
        
        db.session.add(transaction)
        
        # Create API usage tracking
        api_usage = ApiUsage(
            user_id=current_user.id,
            experiment_id=experiment.id,
            request_count=0,
            total_tokens=0,
            successful_requests=0,
            failed_requests=0,
            monthly_limit=10000,  # Default limit
            usage_reset_date=datetime.utcnow() + timedelta(days=30)  # 30 days from now
        )
        
        db.session.add(api_usage)
        
        # Update user
        current_user.has_api_hosting = True
        
        db.session.commit()
        
        # Send subscription confirmation email
        send_subscription_confirmation(
            current_user.email,
            current_user.get_full_name(),
            'API Hosting',
            subscription.end_date
        )
        
        flash('Payment successful! API hosting is now enabled for your account.', 'success')
        return redirect(url_for('project.deploy_model_api', experiment_id=experiment.id, model_run_id=model_run.id))
    
    except Exception as e:
        flash(f'Payment failed: {str(e)}', 'danger')
        return redirect(url_for('payment.pay_for_api_hosting', experiment_id=experiment.id, model_run_id=model_run.id))

@payment_bp.route('/payment/transaction-history')
@login_required
def transaction_history():
    """View transaction history"""
    # Get all user transactions
    transactions = Transaction.query.filter_by(
        user_id=current_user.id
    ).order_by(Transaction.created_at.desc()).all()
    
    # Get active subscriptions
    active_subscriptions = Subscription.query.filter_by(
        user_id=current_user.id,
        status='active'
    ).all()
    
    return render_template(
        'payment/transaction_history.html',
        transactions=transactions,
        active_subscriptions=active_subscriptions
    )

@payment_bp.route('/payment/receipt/<transaction_id>')
@login_required
def view_receipt(transaction_id):
    """View receipt for a transaction"""
    # Get transaction
    transaction = Transaction.query.filter_by(
        id=transaction_id, 
        user_id=current_user.id
    ).first_or_404()
    
    # Get associated subscription
    subscription = None
    if transaction.subscription_id:
        subscription = Subscription.query.get(transaction.subscription_id)
    
    return render_template(
        'payment/receipt.html',
        transaction=transaction,
        subscription=subscription
    )

@payment_bp.route('/payment/request-refund/<transaction_id>', methods=['POST'])
@login_required
def request_refund(transaction_id):
    """Request a refund for a transaction"""
    # Get transaction
    transaction = Transaction.query.filter_by(
        id=transaction_id, 
        user_id=current_user.id
    ).first_or_404()
    
    # Check if transaction is eligible for refund
    # For simplicity, only allow refunds for transactions less than 24 hours old
    if (datetime.utcnow() - transaction.created_at).total_seconds() > 86400:
        flash('This transaction is not eligible for refund', 'warning')
        return redirect(url_for('payment.transaction_history'))
    
    # Get refund reason
    reason = request.form.get('reason', 'Customer requested refund')
    
    # Simulate refund processing
    try:
        # Update transaction status
        transaction.status = 'refunded'
        transaction.refund_id = f"refund_{uuid.uuid4()}"
        transaction.refund_reason = reason
        transaction.refunded_amount = transaction.amount
        
        # If associated with a subscription, cancel it
        if transaction.subscription_id:
            subscription = Subscription.query.get(transaction.subscription_id)
            if subscription:
                subscription.status = 'cancelled'
                subscription.cancelled_at = datetime.utcnow()
                
                # Revoke benefits based on subscription type
                if subscription.plan_type == 'pro':
                    current_user.is_pro = False
                    current_user.pro_expires = datetime.utcnow()
                elif subscription.plan_type == 'k_fold':
                    current_user.has_k_fold = False
                elif subscription.plan_type == 'api_hosting':
                    current_user.has_api_hosting = False
        
        db.session.commit()
        
        flash('Refund request has been processed successfully', 'success')
        return redirect(url_for('payment.transaction_history'))
    
    except Exception as e:
        flash(f'Error processing refund: {str(e)}', 'danger')
        return redirect(url_for('payment.transaction_history'))

@payment_bp.route('/api/payment/check-subscription-status')
@login_required
def check_subscription_status():
    """API endpoint to check subscription status"""
    # Prepare response data
    response = {
        'is_pro': current_user.is_pro and current_user.has_active_pro_subscription(),
        'has_k_fold': current_user.has_k_fold,
        'has_api_hosting': current_user.has_api_hosting,
    }
    
    # Add Pro expiration date if applicable
    if current_user.pro_expires and current_user.is_pro:
        days_remaining = current_user.days_until_subscription_renewal()
        response['pro_days_remaining'] = days_remaining
        response['pro_expires'] = current_user.pro_expires.isoformat()
    
    return jsonify(response)

@payment_bp.route('/payment/renew-subscription/<subscription_id>', methods=['POST'])
@login_required
def renew_subscription(subscription_id):
    """Manually renew a subscription"""
    # Get subscription
    subscription = Subscription.query.filter_by(
        id=subscription_id, 
        user_id=current_user.id
    ).first_or_404()
    
    # Check if subscription is active but expiring soon
    if subscription.status != 'active':
        flash('Only active subscriptions can be renewed', 'warning')
        return redirect(url_for('payment.transaction_history'))
    
    try:
        # Calculate new end date based on billing cycle
        if subscription.billing_cycle == 'monthly':
            new_end_date = subscription.end_date + timedelta(days=30)
            price = 249
        else:  # yearly
            new_end_date = subscription.end_date + timedelta(days=365)
            price = 2499
        
        # Update subscription
        subscription.end_date = new_end_date
        
        # Create transaction record
        transaction = Transaction(
            user_id=current_user.id,
            subscription_id=subscription.id,
            transaction_id=f"txn_{uuid.uuid4()}",
            payment_method='credit_card',  # This would come from payment gateway
            amount=price,
            currency='INR',
            status='completed',
            description=f"Renewal: {subscription.plan_type.capitalize()} Subscription",
            transaction_type='subscription'
        )
        
        db.session.add(transaction)
        
        # Update user
        if subscription.plan_type == 'pro':
            current_user.pro_expires = new_end_date
        
        db.session.commit()
        
        # Send subscription confirmation email
        send_subscription_confirmation(
            current_user.email,
            current_user.get_full_name(),
            f"{subscription.plan_type.capitalize()} Subscription",
            subscription.end_date
        )
        
        flash('Your subscription has been renewed successfully', 'success')
        return redirect(url_for('payment.transaction_history'))
    
    except Exception as e:
        flash(f'Error renewing subscription: {str(e)}', 'danger')
        return redirect(url_for('payment.transaction_history'))