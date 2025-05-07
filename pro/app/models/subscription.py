# pro/app/models/subscription.py
from datetime import datetime
import uuid
from app import db

class Subscription(db.Model):
    """Model for tracking user subscriptions"""
    __tablename__ = 'subscriptions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    plan_type = db.Column(db.String(20), nullable=False)  # 'pro', 'k_fold', 'api_hosting'
    subscription_id = db.Column(db.String(100), nullable=True)  # Payment provider subscription ID
    payment_id = db.Column(db.String(100), nullable=True)  # Payment provider payment ID
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='INR')
    status = db.Column(db.String(20), nullable=False)  # 'active', 'cancelled', 'failed', 'pending'
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=True)  # Null for ongoing subscriptions
    
    # For one-time purchases (k_fold, api_hosting)
    experiment_id = db.Column(db.String(36), db.ForeignKey('experiments.id'), nullable=True)
    is_one_time = db.Column(db.Boolean, default=False)
    
    # For recurring subscriptions (pro)
    is_recurring = db.Column(db.Boolean, default=False)
    billing_cycle = db.Column(db.String(20), nullable=True)  # 'monthly', 'yearly'
    auto_renew = db.Column(db.Boolean, default=True)
    
    # Receipt and invoice info
    receipt_url = db.Column(db.String(512), nullable=True)
    invoice_id = db.Column(db.String(100), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    cancelled_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<Subscription {self.id}>'
    
    def is_active(self):
        """Check if subscription is active"""
        return (
            self.status == 'active' and 
            (self.end_date is None or self.end_date > datetime.utcnow())
        )
    
    def days_remaining(self):
        """Get number of days until subscription expires"""
        if not self.end_date or not self.is_active():
            return 0
            
        delta = self.end_date - datetime.utcnow()
        return max(0, delta.days)

class Transaction(db.Model):
    """Model for tracking payment transactions"""
    __tablename__ = 'transactions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    subscription_id = db.Column(db.String(36), db.ForeignKey('subscriptions.id'), nullable=True)
    
    transaction_id = db.Column(db.String(100), nullable=False)  # Payment provider transaction ID
    payment_method = db.Column(db.String(50), nullable=False)  # 'credit_card', 'upi', etc.
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='INR')
    status = db.Column(db.String(20), nullable=False)  # 'completed', 'pending', 'failed', 'refunded'
    
    # Transaction details
    description = db.Column(db.String(255), nullable=True)
    transaction_type = db.Column(db.String(20), nullable=False)  # 'payment', 'refund', 'subscription'
    
    # Payment method details (stored securely)
    payment_method_details = db.Column(db.Text, nullable=True)  # JSON string with redacted payment details
    
    # Refund information
    refund_id = db.Column(db.String(100), nullable=True)
    refund_reason = db.Column(db.String(255), nullable=True)
    refunded_amount = db.Column(db.Float, nullable=True)
    
    # Error information
    error_code = db.Column(db.String(100), nullable=True)
    error_message = db.Column(db.String(255), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Transaction {self.id}>'

class ApiUsage(db.Model):
    """Model for tracking API usage for hosted models"""
    __tablename__ = 'api_usage'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    experiment_id = db.Column(db.String(36), db.ForeignKey('experiments.id'), nullable=False)
    
    # Usage statistics
    request_count = db.Column(db.Integer, default=0)
    total_tokens = db.Column(db.Integer, default=0)
    successful_requests = db.Column(db.Integer, default=0)
    failed_requests = db.Column(db.Integer, default=0)
    
    # Limits and quotas
    monthly_limit = db.Column(db.Integer, nullable=True)
    usage_reset_date = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_request_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<ApiUsage {self.id}>'
    
    def increment_usage(self, tokens=0, success=True):
        """Increment API usage counters"""
        self.request_count += 1
        self.total_tokens += tokens
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.last_request_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def is_within_limits(self):
        """Check if usage is within limits"""
        if not self.monthly_limit:
            return True
        
        # Reset usage if past reset date
        if self.usage_reset_date and self.usage_reset_date < datetime.utcnow():
            self.request_count = 0
            self.total_tokens = 0
            self.successful_requests = 0
            self.failed_requests = 0
            
            # Set next reset date to first day of next month
            now = datetime.utcnow()
            if now.month == 12:
                self.usage_reset_date = datetime(now.year + 1, 1, 1)
            else:
                self.usage_reset_date = datetime(now.year, now.month + 1, 1)
        
        return self.request_count < self.monthly_limit