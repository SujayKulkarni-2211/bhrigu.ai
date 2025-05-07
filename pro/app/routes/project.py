# pro/app/routes/project.py
import os
import json
import uuid
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.user import User
from app.models.experiment import Experiment, ProDatasetInfo, ProModelRun
from app.models.subscription import Subscription, ApiUsage
from app.utils.data_processing import analyze_dataset_stats
from app.utils.file_handling import save_uploaded_file, allowed_file
from app.utils.ml_utils import (
    get_model_recommendations, get_gemini_recommendations, train_model_async,
    tune_hyperparameters_async, compare_models, get_available_algorithms,
    deploy_model_as_api, train_with_kfold, get_default_hyperparameters
)
from app.models.user import User

from app.utils.ml_report_utils import (
    generate_model_report, generate_tuning_report, generate_comparison_report,
    generate_comprehensive_report
)
from app.utils.email import (
    send_model_completion_notification, send_error_notification,
    send_subscription_confirmation
)
from threading import Thread

# Initialize the Blueprint
project_bp = Blueprint('project', __name__)

@project_bp.route('/project/<experiment_id>')
@login_required
def view(experiment_id):
    """Project dashboard page"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # If no dataset uploaded yet, redirect to upload page
    if experiment.status == 'created':
        return redirect(url_for('project.upload_dataset', experiment_id=experiment.id))
    
    # Get experiment data
    dataset_info = experiment.dataset_info
    model_runs = experiment.model_runs
    
    # Check if user is pro
    is_pro = current_user.is_pro and current_user.has_active_pro_subscription()
    
    # Count experiments and trials
    num_experiments = Experiment.query.filter_by(
        user_id=current_user.id,
        is_deleted=False
    ).count()
    
    num_trials = ProModelRun.query.join(Experiment).filter(
        Experiment.user_id == current_user.id,
        Experiment.is_deleted == False
    ).count()
    
    # Get available algorithms based on user's subscription
    available_algorithms = {}
    if experiment.problem_type:
        available_algorithms = get_available_algorithms(
            problem_type=experiment.problem_type,
            model_type='all',  # We'll filter in the template
            is_pro=is_pro
        )
    
    feature_columns_list = json.loads(experiment.feature_columns) if experiment.feature_columns else []
    return render_template(
        'projecttemp/view.html',
        experiment=experiment,
        dataset_info=dataset_info,
        model_runs=model_runs,
        is_pro=is_pro,
        has_k_fold=current_user.has_k_fold,
        has_api_hosting=current_user.has_api_hosting,
        num_experiments=num_experiments,
        num_trials=num_trials,
        available_algorithms=available_algorithms,json=json,
        feature_columns_list=feature_columns_list
    )

@project_bp.route('/project/<experiment_id>/upload', methods=['GET', 'POST'])
@login_required
def upload_dataset(experiment_id):
    """Upload dataset for a project"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'dataset' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['dataset']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        # Save the uploaded file
        file_path, filename, original_filename = save_uploaded_file(
            file=file,
            user_id=current_user.id,
            experiment_id=experiment.id,
            category='dataset'
        )
        
        if not file_path:
            flash('Invalid file type. Please upload a CSV or Excel file.', 'danger')
            return redirect(request.url)
        
        # Update experiment with dataset info
        experiment.dataset_filename = filename
        experiment.dataset_original_name = original_filename
        experiment.file_path = file_path
        experiment.status = 'data_uploaded'
        print(f"File saved at: {file_path}")
        
        db.session.commit()
        
        flash('Dataset uploaded successfully!', 'success')
        return redirect(url_for('project.analyze_dataset', experiment_id=experiment.id))
    
    return render_template('projecttemp/upload_dataset.html',json=json, experiment=experiment)

@project_bp.route('/project/<experiment_id>/analyze', methods=['GET', 'POST'])
@login_required
def analyze_dataset(experiment_id):
    """Analyze dataset for a project"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if dataset exists
    if not experiment.dataset_filename:
        flash('Please upload a dataset first', 'warning')
        return redirect(url_for('project.upload_dataset', experiment_id=experiment.id))
    
    file_path = experiment.file_path
    
    if request.method == 'POST':
        # Get form data
        problem_type = request.form.get('problem_type')
        data_type = request.form.get('data_type', 'tabular')  # Default to tabular
        target_column = request.form.get('target_column')
        feature_columns = request.form.getlist('feature_columns')
        
        if not problem_type or not target_column or not feature_columns:
            flash('Please select problem type, target column, and at least one feature column', 'danger')
            return redirect(request.url)
        
        # Update experiment
        experiment.problem_type = problem_type
        experiment.data_type = data_type
        experiment.target_column = target_column
        experiment.feature_columns = json.dumps(feature_columns)
        experiment.status = 'analyzed'
        
        db.session.commit()
        
        flash('Dataset analysis completed!', 'success')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    # Analyze dataset if not already done
    if not experiment.dataset_info:
        try:
            data_info = analyze_dataset_stats(file_path)
            
            # Create dataset info record
            dataset_info = ProDatasetInfo(
                experiment_id=experiment.id,
                num_rows=data_info['num_rows'],
                num_columns=data_info['num_cols'],
                column_names=json.dumps(data_info['columns']),
                column_types=json.dumps(data_info['column_types']),
                missing_values=json.dumps(data_info['missing_values']),
                numeric_columns=json.dumps(data_info['numeric_columns']),
                categorical_columns=json.dumps(data_info['categorical_columns']),
                statistics=json.dumps(data_info['statistics']),
                correlation_matrix=data_info['correlation_image']
            )
            
            db.session.add(dataset_info)
            db.session.commit()
        except Exception as e:
            flash(f'Error analyzing dataset: {str(e)}', 'danger')
            return redirect(url_for('project.upload_dataset', experiment_id=experiment.id))
    else:
        # Get data info from database
        dataset_info = experiment.dataset_info
        data_info = {
            'num_rows': dataset_info.num_rows,
            'num_cols': dataset_info.num_columns,
            'columns': json.loads(dataset_info.column_names) if dataset_info.column_names else [],
            'column_types': json.loads(dataset_info.column_types) if dataset_info.column_types else {},
            'missing_values': json.loads(dataset_info.missing_values) if dataset_info.missing_values else {},
            'numeric_columns': json.loads(dataset_info.numeric_columns) if dataset_info.numeric_columns else [],
            'categorical_columns': json.loads(dataset_info.categorical_columns) if dataset_info.categorical_columns else [],
            'statistics': json.loads(dataset_info.statistics) if dataset_info.statistics else {},
            'correlation_image': dataset_info.correlation_matrix,
            'categorical_values': {}
        }
        for column in data_info['categorical_columns']:
            data_info['categorical_values'][column] = {}
    
    return render_template(
        'projecttemp/analyze.html',
        experiment=experiment,
        data_info=data_info
    )




@project_bp.route('/project/<experiment_id>/model-recommendations')
@login_required
def model_recommendations(experiment_id):
    """Get model recommendations page"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if dataset is analyzed
    if not experiment.problem_type or not experiment.target_column or not experiment.feature_columns:
        flash('Please analyze your dataset first', 'warning')
        return redirect(url_for('project.analyze_dataset', experiment_id=experiment.id))
    
    # Get dataset info
    dataset_info = experiment.dataset_info
    data_info = {
        'num_rows': dataset_info.num_rows,
        'num_cols': dataset_info.num_columns,
        'numeric_columns': json.loads(dataset_info.numeric_columns) if dataset_info.numeric_columns else [],
        'categorical_columns': json.loads(dataset_info.categorical_columns) if dataset_info.categorical_columns else [],
        'statistics': json.loads(dataset_info.statistics) if dataset_info.statistics else {},
        'columns': json.loads(dataset_info.column_names) if dataset_info.column_names else [],
        'column_types': json.loads(dataset_info.column_types) if dataset_info.column_types else {}
    }
    
    # Check if user has k-fold subscription
    has_k_fold = current_user.has_k_fold
    
    return render_template(
        'projecttemp/model_recommendations.html',
        experiment=experiment,
        data_info=data_info,json=json,
        has_k_fold=has_k_fold
    )

@project_bp.route('/project/<experiment_id>/gemini-recommendations')
@login_required
def gemini_recommendations(experiment_id):
    """Get model recommendations using Gemini"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get dataset info
    dataset_info = experiment.dataset_info
    data_info = {
        'num_rows': dataset_info.num_rows,
        'num_cols': dataset_info.num_columns,
        'numeric_columns': json.loads(dataset_info.numeric_columns) if dataset_info.numeric_columns else [],
        'categorical_columns': json.loads(dataset_info.categorical_columns) if dataset_info.categorical_columns else [],
        'statistics': json.loads(dataset_info.statistics) if dataset_info.statistics else {},
        'columns': json.loads(dataset_info.column_names) if dataset_info.column_names else [],
        'column_types': json.loads(dataset_info.column_types) if dataset_info.column_types else {}
        
    }
    
    # Get model recommendations
    recommendations = get_gemini_recommendations(experiment.problem_type, data_info)
    
    # Get available models based on user's subscription
    is_pro = current_user.is_pro and current_user.has_active_pro_subscription()
    available_models = get_available_algorithms(
        problem_type=experiment.problem_type,
        is_pro=is_pro
    )
    available_models_data = []
    for algorithm in available_models:
        # Determine model type
        if algorithm.startswith('Q'):
            model_type = 'quantum'
        elif algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU']:
            model_type = 'neural'
        else:
            model_type = 'classical'
            
        model_data = {
            'algorithm': algorithm,
            'model_type': model_type,
            'description': f"A {model_type} model for {experiment.problem_type}",
            'suitable_for': experiment.problem_type
        }
        available_models_data.append(model_data)
    
    return render_template(
        'projecttemp/gemini_recommendations.html',
        experiment=experiment,
        recommendations=recommendations,
        available_models=available_models_data,json=json,
        data_info=data_info
    )

@project_bp.route('/project/<experiment_id>/k-fold-evaluation', methods=['GET', 'POST'])
@login_required
def k_fold_evaluation(experiment_id):
    """K-Fold Cross-Validation Evaluation"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user has k-fold subscription
    if not current_user.has_k_fold:
        # Show payment page
        return render_template(
            'projecttemp/k_fold_payment.html',
            experiment=experiment,json=json,
            price=99  # Price in INR
        )
    
    if request.method == 'POST':
        # Get K-fold value
        k_fold_value = int(request.form.get('k_fold_value', 5))
        
        # Update experiment
        experiment.use_k_fold = True
        experiment.k_fold_value = k_fold_value
        db.session.commit()
        
        # Start K-fold evaluation
        return redirect(url_for('project.run_k_fold', experiment_id=experiment.id))
    
    return render_template(
        'projecttemp/k_fold_setup.html',json=json,
        experiment=experiment
    )

@project_bp.route('/project/<experiment_id>/pay-for-k-fold', methods=['POST'])
@login_required
def pay_for_k_fold(experiment_id):
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
    
    # Simulate payment processing
    # In a real app, you would integrate with a payment gateway like Razorpay
    try:
        # Create a one-time subscription for k-fold
        subscription = Subscription(
            user_id=current_user.id,
            plan_type='k_fold',
            subscription_id=f"k_fold_{uuid.uuid4()}",
            payment_id=f"payment_{uuid.uuid4()}",
            amount=99,
            currency='INR',
            status='active',
            start_date=datetime.utcnow(),
            experiment_id=experiment.id,
            is_one_time=True
        )
        
        db.session.add(subscription)
        
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
        return redirect(url_for('project.k_fold_evaluation', experiment_id=experiment.id))

@project_bp.route('/project/<experiment_id>/run-k-fold')
@login_required
def run_k_fold(experiment_id):
    """Run K-fold cross-validation"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user has k-fold subscription
    if not current_user.has_k_fold:
        flash('K-fold cross-validation is not enabled for your account', 'warning')
        return redirect(url_for('project.k_fold_evaluation', experiment_id=experiment.id))
    
    # Get available algorithms based on user's subscription
    is_pro = current_user.is_pro and current_user.has_active_pro_subscription()
    available_algorithms = get_available_algorithms(
        problem_type=experiment.problem_type,
        is_pro=is_pro
    )
    
    # Create a model run for each algorithm to track k-fold results
    k_fold_runs = []
    for algorithm in available_algorithms:
        model_type = 'quantum' if algorithm.startswith('Q') else 'neural' if algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU'] else 'classical'
        
        # Skip quantum models for non-pro users
        if model_type == 'quantum' and not is_pro:
            continue
        
        # Create model run
        model_run = ProModelRun(
            experiment_id=experiment.id,
            algorithm=algorithm,
            model_type=model_type,
            status='created',
            hyperparameters_tuning=False
        )
        
        db.session.add(model_run)
        k_fold_runs.append(model_run)
    
    db.session.commit()
    
    # Start k-fold evaluations in background for each algorithm
    # This is a placeholder - you would implement these as background tasks
    for model_run in k_fold_runs:
        # Update status
        model_run.status = 'training'
        model_run.started_at = datetime.utcnow()
        model_run.logs = f"Starting {model_run.algorithm} with {experiment.k_fold_value}-fold cross-validation...\n"
        db.session.commit()
        
        # Run k-fold evaluation in background
        run_k_fold_async(
            model_run.id,
            model_run.algorithm,
            experiment.problem_type,
            experiment.file_path,
            experiment.target_column,
            json.loads(experiment.feature_columns),
            experiment.k_fold_value,
            current_app.config['MODEL_FOLDER'],
            # Pro users get more processing time
            current_app.config['PRO_MAX_TRAINING_TIME'] if is_pro else current_app.config['FREE_MAX_TRAINING_TIME'],
            notify_completion=current_user.notify_on_completion,
            notify_error=current_user.notify_on_error
        )
    
    return render_template(
        'projecttemp/k_fold_running.html',json=json,
        experiment=experiment,
        k_fold_value=experiment.k_fold_value,
        k_fold_runs=k_fold_runs
    )

def run_k_fold_async(model_run_id, algorithm, problem_type, file_path, target_column, 
                    feature_columns, k_fold_value, model_folder, time_limit,
                    notify_completion=True, notify_error=True):
    """Start k-fold cross-validation in a background thread"""
    def task():
        try:
            # Get model run
            model_run = ProModelRun.query.get(model_run_id)
            experiment = Experiment.query.get(model_run.experiment_id)
            user = User.query.get(experiment.user_id)
            
            # Run k-fold cross-validation
            results = train_with_kfold(
                algorithm=algorithm,
                problem_type=problem_type,
                file_path=file_path,
                target_column=target_column,
                feature_columns=feature_columns,
                n_splits=k_fold_value,
                model_folder=model_folder,
                time_limit=time_limit
            )
            
            # Update model run with results
            model_run.status = 'completed'
            model_run.metrics = json.dumps(results.get('metrics', {}))
            model_run.feature_importances = json.dumps(results.get('feature_importances', {}))
            model_run.model_path = results.get('model_path')
            model_run.completed_at = datetime.utcnow()
            
            # Add fold-specific metrics
            model_run.tuning_metrics = json.dumps({
                'fold_metrics': results.get('fold_metrics', []),
                'best_fold': results.get('best_fold', 0),
                'best_fold_metrics': results.get('best_fold_metrics', {})
            })
            
            db.session.commit()
            
            # Send completion notification
            if notify_completion and user.notify_on_completion:
                send_model_completion_notification(
                    user.email,
                    user.get_full_name(),
                    experiment.name,
                    model_run.algorithm,
                    results.get('metrics', {})
                )
                
        except Exception as e:
            # Log error
            error_message = str(e)
            
            # Update model run with error
            model_run.status = 'failed'
            model_run.logs += f"Error: {error_message}\n"
            model_run.completed_at = datetime.utcnow()
            db.session.commit()
            
            # Send error notification
            if notify_error and user.notify_on_error:
                send_error_notification(
                    user.email,
                    user.get_full_name(),
                    experiment.name,
                    model_run.algorithm,
                    error_message
                )
    
    # Start background thread
    Thread(target=task).start()

@project_bp.route('/project/<experiment_id>/k-fold-results')
@login_required
def k_fold_results(experiment_id):
    """Show K-fold cross-validation results"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get k-fold model runs
    k_fold_runs = ProModelRun.query.filter_by(
        experiment_id=experiment.id
    ).order_by(ProModelRun.created_at.desc()).all()
    
    # Prepare data for visualization
    algorithms = []
    scores = []
    
    for run in k_fold_runs:
        if run.status == 'completed' and run.metrics:
            metrics = json.loads(run.metrics)
            # Get primary metric (accuracy for classification, r2 for regression)
            primary_metric = metrics.get('accuracy' if experiment.problem_type == 'classification' else 'r2', 0)
            
            algorithms.append(run.algorithm)
            scores.append(primary_metric)
    
    return render_template(
        'projecttemp/k_fold_results.html',
        experiment=experiment,
        k_fold_runs=k_fold_runs,
        algorithms=json.dumps(algorithms),json=json,
        scores=json.dumps(scores)
    )

@project_bp.route('/project/<experiment_id>/select-model/<algorithm>')
@login_required
def select_model(experiment_id, algorithm):
    """Select a model for a new trial"""
    print(f"select_model route called with experiment_id={experiment_id}, algorithm={algorithm}")
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if algorithm is valid for user's subscription
    is_pro = current_user.is_pro and current_user.has_active_pro_subscription()
    
    # Determine model type
    model_type = 'quantum' if algorithm.startswith('Q') else 'neural' if algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU'] else 'classical'
    
    # If quantum algorithm but not pro, redirect to pro upgrade
    if model_type == 'quantum' and not is_pro:
        flash('Quantum algorithms are only available with Pro subscription', 'warning')
        return redirect(url_for('project.upgrade_to_pro', experiment_id=experiment.id))
    
    # If advanced neural network but not pro, limit options
    if model_type == 'neural' and not is_pro and algorithm not in ['MLP', 'FNN']:
        flash('Advanced neural networks are only available with Pro subscription', 'warning')
        return redirect(url_for('project.upgrade_to_pro', experiment_id=experiment.id))
    
    # Create a new model run (trial)
    model_run = ProModelRun(
        experiment_id=experiment.id,
        algorithm=algorithm,
        model_type=model_type,
        status='created'
    )
    
    db.session.add(model_run)
    db.session.commit()
    
    # Redirect to hyperparameter choice
    return redirect(url_for('project.hyperparameter_choice', 
                           experiment_id=experiment.id, 
                           model_run_id=model_run.id))

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/hyperparameter-choice', methods=['GET'])
@login_required
def hyperparameter_choice(experiment_id, model_run_id):
    """Choose between automatic tuning and manual configuration"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    dataset_info = experiment.dataset_info
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    return render_template(
        'projecttemp/hyperparameter_choice.html',
        experiment=experiment,json=json,dataset_info=dataset_info,
        model_run=model_run
    )

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/configure', methods=['GET', 'POST'])
@login_required
def configure_hyperparameters(experiment_id, model_run_id):
    """Configure hyperparameters for a trial"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    dataset_info = experiment.dataset_info
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    if request.method == 'POST':
        # Get hyperparameters from form
        hyperparameters = {}
        
        # Process form data based on algorithm type
        if model_run.algorithm in ['RandomForest', 'RandomForestRegressor']:
            hyperparameters['n_estimators'] = int(request.form.get('n_estimators', 100))
            hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') and request.form.get('max_depth') != 'None' else None
            hyperparameters['min_samples_split'] = int(request.form.get('min_samples_split', 2))
            hyperparameters['min_samples_leaf'] = int(request.form.get('min_samples_leaf', 1))
            if experiment.problem_type == 'classification':
                hyperparameters['criterion'] = request.form.get('criterion', 'gini')
            else:
                hyperparameters['criterion'] = request.form.get('criterion', 'squared_error')
        
        elif model_run.algorithm == 'LogisticRegression':
            hyperparameters['C'] = float(request.form.get('C', 1.0))
            hyperparameters['max_iter'] = int(request.form.get('max_iter', 1000))
            hyperparameters['solver'] = request.form.get('solver', 'liblinear')
            hyperparameters['penalty'] = request.form.get('penalty', 'l2')
        
        elif model_run.algorithm == 'LinearRegression':
            hyperparameters['fit_intercept'] = request.form.get('fit_intercept', 'True') == 'True'
            hyperparameters['normalize'] = request.form.get('normalize', 'False') == 'True'
        
        elif model_run.algorithm in ['SVC', 'SVR']:
            hyperparameters['C'] = float(request.form.get('C', 1.0))
            hyperparameters['kernel'] = request.form.get('kernel', 'rbf')
            hyperparameters['gamma'] = request.form.get('gamma', 'scale')
            
            if model_run.algorithm == 'SVC':
                hyperparameters['probability'] = True  # Always enable probability for better reporting
        
        elif model_run.algorithm in ['DecisionTree', 'DecisionTreeRegressor']:
            hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') and request.form.get('max_depth') != 'None' else None
            hyperparameters['min_samples_split'] = int(request.form.get('min_samples_split', 2))
            hyperparameters['min_samples_leaf'] = int(request.form.get('min_samples_leaf', 1))
            
            if experiment.problem_type == 'classification':
                hyperparameters['criterion'] = request.form.get('criterion', 'gini')
            else:
                hyperparameters['criterion'] = request.form.get('criterion', 'squared_error')
        
        elif model_run.algorithm in ['KNN', 'KNeighborsRegressor']:
            hyperparameters['n_neighbors'] = int(request.form.get('n_neighbors', 5))
            hyperparameters['weights'] = request.form.get('weights', 'uniform')
            hyperparameters['algorithm'] = request.form.get('algorithm', 'auto')
            hyperparameters['p'] = int(request.form.get('p', 2))
        
        elif model_run.algorithm == 'GradientBoosting' or model_run.algorithm == 'GradientBoostingRegressor':
            hyperparameters['n_estimators'] = int(request.form.get('n_estimators', 100))
            hyperparameters['learning_rate'] = float(request.form.get('learning_rate', 0.1))
            hyperparameters['max_depth'] = int(request.form.get('max_depth', 3))
            hyperparameters['subsample'] = float(request.form.get('subsample', 1.0))
        
        elif model_run.algorithm in ['MLP', 'FNN']:
            hyperparameters['layers'] = [int(x) for x in request.form.getlist('layers') if x and x.isdigit()]
            hyperparameters['activation'] = request.form.get('activation', 'relu')
            hyperparameters['dropout'] = float(request.form.get('dropout', 0.2))
            hyperparameters['learning_rate'] = float(request.form.get('learning_rate', 0.001))
            hyperparameters['epochs'] = int(request.form.get('epochs', 100))
            hyperparameters['batch_size'] = int(request.form.get('batch_size', 32))
            hyperparameters['early_stopping'] = request.form.get('early_stopping', 'True') == 'True'
            hyperparameters['patience'] = int(request.form.get('patience', 10))
        
        elif model_run.algorithm in ['RNN', 'LSTM', 'GRU']:
            hyperparameters['layers'] = [int(x) for x in request.form.getlist('layers') if x and x.isdigit()]
            hyperparameters['dropout'] = float(request.form.get('dropout', 0.2))
            hyperparameters['learning_rate'] = float(request.form.get('learning_rate', 0.001))
            hyperparameters['epochs'] = int(request.form.get('epochs', 100))
            hyperparameters['batch_size'] = int(request.form.get('batch_size', 32))
            hyperparameters['early_stopping'] = request.form.get('early_stopping', 'True') == 'True'
            hyperparameters['patience'] = int(request.form.get('patience', 10))
        
        # Quantum algorithms
        elif model_run.algorithm in ['QSVM', 'QKE']:
            hyperparameters['qubits'] = int(request.form.get('qubits', 4))
            hyperparameters['feature_map'] = request.form.get('feature_map', 'ZZFeatureMap')
        
        elif model_run.algorithm in ['QNN', 'QRegression']:
            hyperparameters['qubits'] = int(request.form.get('qubits', 4))
            hyperparameters['layers'] = int(request.form.get('layers', 2))
            hyperparameters['ansatz'] = request.form.get('ansatz', 'StronglyEntanglingLayers')
            hyperparameters['learning_rate'] = float(request.form.get('learning_rate', 0.01))
            hyperparameters['epochs'] = int(request.form.get('epochs', 50))
        
        # Update model run with hyperparameters
        model_run.hyperparameters = json.dumps(hyperparameters)
        db.session.commit()
        
        # Redirect to training page
        return redirect(url_for('project.train_model', 
                               experiment_id=experiment.id, 
                               model_run_id=model_run.id, dataset_info=dataset_info))
    
    # Get default hyperparameters for algorithm
    default_params = get_default_hyperparameters(
        algorithm=model_run.algorithm,
        problem_type=experiment.problem_type
    )
    
    # Override with existing hyperparameters if available
    if model_run.hyperparameters:
        try:
            existing_params = json.loads(model_run.hyperparameters)
            default_params.update(existing_params)
        except:
            pass
    
    return render_template(
        'projecttemp/configure_hyperparameters.html',
        experiment=experiment,
        model_run=model_run,json=json,
        default_params=default_params,
        dataset_info=dataset_info
    )

# Fixed tune_hyperparameters function for project.py

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/tune')
@login_required
def tune_hyperparameters(experiment_id, model_run_id):
    """Run hyperparameter tuning for a trial"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    dataset_info = experiment.dataset_info
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    # Update model run status
    model_run.status = 'tuning'
    model_run.hyperparameters_tuning = True
    model_run.started_at = datetime.utcnow()
    model_run.logs = "Starting hyperparameter optimization...\n"
    db.session.commit()
    
    # Get base hyperparameters
    base_params = get_default_hyperparameters(
        algorithm=model_run.algorithm,
        problem_type=experiment.problem_type
    )
    
    # Capture the app context
    app = current_app._get_current_object()
    
    # Define callback function
    def tuning_complete_callback(success, results=None, error=None):
        """Callback for when tuning completes"""
        with app.app_context():
            # Create a new session for this callback
            session = db.create_scoped_session()
            try:
                # Update model run with results
                model_run = session.query(ProModelRun).get(model_run_id)
                if not model_run:
                    return
                
                if success and results:
                    # Update model with best parameters
                    model_run.hyperparameters = json.dumps(results.get('best_params', {}))
                    
                    # Clean tuning results for JSON serialization
                    clean_tuning_results = []
                    for result in results.get('tuning_results', []):
                        clean_result = {}
                        for k, v in result.items():
                            if isinstance(v, (int, float, str, bool)) or v is None:
                                clean_result[k] = v
                            elif hasattr(v, 'tolist'):  # numpy array
                                clean_result[k] = v.tolist()
                            else:
                                clean_result[k] = str(v)
                        clean_tuning_results.append(clean_result)
                    
                    model_run.tuning_results = json.dumps(clean_tuning_results)
                    
                    # Clean metrics for JSON serialization
                    clean_metrics = {}
                    for k, v in results.get('metrics', {}).items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            clean_metrics[k] = v
                        elif hasattr(v, 'tolist'):  # numpy array
                            clean_metrics[k] = v.tolist()
                        else:
                            clean_metrics[k] = str(v)
                    
                    model_run.tuning_metrics = json.dumps(clean_metrics)
                    model_run.status = 'created'  # Reset to created so user can train with these params
                    model_run.logs += "Hyperparameter tuning completed successfully.\n"
                    model_run.completed_at = datetime.utcnow()
                    
                    # Generate tuning report if in Pro version
                    if 'best_params' in results and results.get('best_params'):
                        try:
                            tuning_report_path = generate_tuning_report(
                                model_run.id,
                                model_run.algorithm,
                                experiment.problem_type,
                                clean_tuning_results,
                                results.get('best_params', {}),
                                clean_metrics,
                                app.config['REPORT_FOLDER'],
                                experiment_name=experiment.name
                            )
                            model_run.tuning_report_path = tuning_report_path
                        except Exception as e:
                            model_run.logs += f"Error generating tuning report: {str(e)}\n"
                else:
                    # Update with error
                    model_run.status = 'failed'
                    model_run.logs += f"Hyperparameter tuning failed: {error or 'Unknown error'}\n"
                    model_run.completed_at = datetime.utcnow()
                
                session.commit()
                
                # Send notifications
                experiment = session.query(Experiment).get(model_run.experiment_id)
                user = session.query(User).get(experiment.user_id)
                
                if user:
                    if success and user.notify_on_completion:
                        send_model_completion_notification(
                            user.email,
                            user.get_full_name(),
                            experiment.name,
                            f"{model_run.algorithm} Hyperparameter Tuning",
                            clean_metrics if success else {}
                        )
                    elif not success and user.notify_on_error:
                        send_error_notification(
                            user.email,
                            user.get_full_name(),
                            experiment.name,
                            f"{model_run.algorithm} Hyperparameter Tuning",
                            error or "Unknown error occurred during tuning"
                        )
            except Exception as e:
                print(f"Error in tuning_complete_callback: {str(e)}")
                session.rollback()
            finally:
                session.close()
    
    # Define the tuning task function
    def tuning_task():
        try:
            # Use the captured app context
            with app.app_context():
                # Create a new session for this task
                session = db.create_scoped_session()
                try:
                    # Get fresh data from database
                    fresh_model_run = session.query(ProModelRun).get(model_run_id)
                    fresh_experiment = session.query(Experiment).get(fresh_model_run.experiment_id)
                    
                    # Get the file path
                    file_path = fresh_experiment.file_path
                    if not os.path.exists(file_path):
                        # Try to resolve the file path
                        base_name = os.path.basename(file_path)
                        alternate_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', base_name)
                        if os.path.exists(alternate_path):
                            file_path = alternate_path
                        else:
                            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
                    
                    # Get feature columns
                    feature_columns = []
                    if fresh_experiment.feature_columns:
                        try:
                            feature_columns = json.loads(fresh_experiment.feature_columns)
                        except json.JSONDecodeError:
                            raise ValueError("Invalid feature columns data")
                    
                    # Run tuning
                    from app.utils.ml_utils import tune_hyperparameters_pipeline
                    results = tune_hyperparameters_pipeline(
                        model_run_id=fresh_model_run.id,
                        algorithm=fresh_model_run.algorithm,
                        problem_type=fresh_experiment.problem_type,
                        file_path=file_path,
                        target_column=fresh_experiment.target_column,
                        feature_columns=feature_columns,
                        base_hyperparameters=base_params,
                        model_folder=app.config['MODEL_FOLDER'],
                        time_limit=app.config.get('PRO_MAX_TRAINING_TIME', 3600)
                    )
                    
                    # Call callback with results
                    tuning_complete_callback(True, results)
                except Exception as e:
                    import traceback
                    print(f"Error in tuning_task: {str(e)}\n{traceback.format_exc()}")
                    # Call callback with error
                    tuning_complete_callback(False, None, str(e))
                finally:
                    session.close()
        except Exception as e:
            import traceback
            print(f"Error in tuning_task outer block: {str(e)}\n{traceback.format_exc()}")
            try:
                tuning_complete_callback(False, None, str(e))
            except Exception as callback_error:
                print(f"Error in error callback: {str(callback_error)}")
    
    # Start tuning in background
    from threading import Thread
    tuning_thread = Thread(target=tuning_task)
    tuning_thread.daemon = True
    tuning_thread.start()
    
    return render_template(
        'projecttemp/tuning_status.html',
        experiment=experiment,
        model_run=model_run,
        json=json,
        dataset_info=dataset_info,
    )

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/tuning-status')
@login_required
def tuning_status(experiment_id, model_run_id):
    """Get hyperparameter tuning status (AJAX endpoint)"""
    # Get model run
    model_run = ProModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current user
    experiment = Experiment.query.get(model_run.experiment_id)
    if experiment.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Get tuning metrics if completed
    tuning_metrics = json.loads(model_run.tuning_metrics) if model_run.tuning_metrics else None
    best_params = json.loads(model_run.hyperparameters) if model_run.hyperparameters else None
    
    # Return status info
    return jsonify({
        'status': model_run.status,
        'logs': model_run.logs,
        'tuning_metrics': tuning_metrics,
        'best_params': best_params
    })

# Fixed train_model function for project.py

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/train')
@login_required
def train_model(experiment_id, model_run_id):
    """Train a model for a trial"""
    try:
        # Get experiment and model run
        experiment = Experiment.query.get_or_404(experiment_id)
        model_run = ProModelRun.query.get_or_404(model_run_id)
        
        # Check if this experiment belongs to the current user
        if experiment.user_id != current_user.id:
            flash('Access denied', 'danger')
            return redirect(url_for('dashboard.index'))
        
        # Check if model run belongs to this experiment
        if model_run.experiment_id != experiment.id:
            flash('Invalid model for this experiment', 'danger')
            return redirect(url_for('project.view', experiment_id=experiment.id))
        
        # Get file path and hyperparameters
        file_path = experiment.file_path
        hyperparameters = {}
        
        if model_run.hyperparameters:
            try:
                hyperparameters = json.loads(model_run.hyperparameters)
            except json.JSONDecodeError:
                flash('Error loading hyperparameters, using defaults instead', 'warning')
        
        feature_columns = []
        if experiment.feature_columns:
            try:
                feature_columns = json.loads(experiment.feature_columns)
            except json.JSONDecodeError:
                flash('Error loading feature columns', 'warning')
        
        # Get dataset_info for the template
        dataset_info = experiment.dataset_info
        
        # Update model status
        model_run.status = 'training'
        model_run.started_at = datetime.utcnow()
        model_run.logs = "Starting model training...\n"
        db.session.commit()
        
        # Determine if user is pro
        is_pro = current_user.is_pro and current_user.has_active_pro_subscription() if hasattr(current_user, 'is_pro') else False
        
        # Capture the app context
        app = current_app._get_current_object()
        
        # Define the training task function with properly captured file_path
        def training_task(captured_file_path, captured_feature_columns, captured_hyperparameters):
            try:
                # Use the captured app context
                with app.app_context():
                    # Create a new session for this task
                    session = db.create_scoped_session()
                    try:
                        # Get fresh data from database
                        fresh_model_run = session.query(ProModelRun).get(model_run_id)
                        fresh_experiment = session.query(Experiment).get(fresh_model_run.experiment_id)
                        user = session.query(User).get(fresh_experiment.user_id)
                        
                        # Verify path exists or find an alternative
                        local_file_path = captured_file_path
                        if not os.path.exists(local_file_path):
                            # Try to resolve the file path
                            base_name = os.path.basename(local_file_path)
                            alternate_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', base_name)
                            if os.path.exists(alternate_path):
                                local_file_path = alternate_path
                                fresh_model_run.logs += f"Fixed file path to {local_file_path}\n"
                                session.commit()
                            else:
                                raise FileNotFoundError(f"Dataset file not found at: {local_file_path}")
                        
                        # Start the training
                        from app.utils.ml_utils import MLModel
                        
                        # Determine model type
                        model_type = 'quantum' if fresh_model_run.algorithm.startswith('Q') else \
                                    'neural' if fresh_model_run.algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU'] else \
                                    'classical'
                                    
                        # Check if pro user has access to this model type
                        if model_type == 'quantum' and not is_pro:
                            raise PermissionError("Quantum models are only available with Pro subscription")
                        
                        # Initialize model
                        model = MLModel(
                            model_type=model_type,
                            algorithm=fresh_model_run.algorithm,
                            problem_type=fresh_experiment.problem_type,
                            params=captured_hyperparameters,
                            is_pro=is_pro,
                            experiment_id=fresh_experiment.id,
                            model_run_id=fresh_model_run.id
                        )
                        
                        # Preprocess data
                        fresh_model_run.logs += "Preprocessing dataset...\n"
                        session.commit()
                        
                        from app.utils.data_processing import preprocess_data
                        processed_data = preprocess_data(
                            local_file_path, 
                            fresh_experiment.target_column, 
                            captured_feature_columns
                        )
                        
                        X = processed_data['X']
                        y = processed_data['y']
                        
                        # Train model
                        fresh_model_run.logs += "Data preprocessed. Starting model training...\n"
                        session.commit()
                        
                        # Set start time for time limit checking
                        start_time = datetime.utcnow()
                        
                        # Train the model
                        model.train(X, y, test_size=0.2, random_state=42)
                        
                        # Check if time limit exceeded
                        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
                        time_limit = app.config.get('PRO_MAX_TRAINING_TIME', 3600) if is_pro else app.config.get('FREE_MAX_TRAINING_TIME', 1800)
                        
                        if elapsed_time > time_limit:
                            fresh_model_run.status = 'timed_out'
                            fresh_model_run.logs += "Training timed out. Maximum allowed time exceeded.\n"
                            session.commit()
                            
                            if user and hasattr(user, 'notify_on_error') and user.notify_on_error:
                                from app.utils.email import send_error_notification
                                send_error_notification(
                                    user.email,
                                    user.get_full_name(),
                                    fresh_experiment.name,
                                    fresh_model_run.algorithm,
                                    "Training timed out - maximum allowed time exceeded"
                                )
                            return
                        
                        # Helper function to convert numpy values to Python types
                        def convert_to_serializable(obj):
                            import numpy as np
                            if isinstance(obj, np.integer):
                                return int(obj)
                            elif isinstance(obj, np.floating):
                                return float(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, np.bool_):
                                return bool(obj)
                            return obj
                        
                        # Save model
                        fresh_model_run.logs += "Training completed. Saving model and generating report...\n"
                        session.commit()
                        
                        # Ensure directory exists
                        model_folder = app.config['MODEL_FOLDER']
                        os.makedirs(model_folder, exist_ok=True)
                        model_path = os.path.join(model_folder, f"{fresh_model_run.id}_model.pkl")
                        model.save(model_path)
                        
                        # Generate report
                        report_path = model.generate_report(model_folder, fresh_experiment.name)
                        
                        # Extract feature importances if available
                        feature_importances_dict = None
                        if model.feature_importances is not None:
                            # Match with feature names
                            try:
                                importances = model.feature_importances
                                if hasattr(importances, 'tolist'):
                                    importances = importances.tolist()
                                    
                                if len(captured_feature_columns) == len(importances):
                                    feature_importances_dict = {feature: convert_to_serializable(value) 
                                                              for feature, value in zip(captured_feature_columns, importances)}
                                else:
                                    # If length mismatch, use generic names
                                    feature_importances_dict = {f"feature_{i}": convert_to_serializable(value) 
                                                              for i, value in enumerate(importances)}
                            except Exception as fi_error:
                                fresh_model_run.logs += f"Warning: Error processing feature importances: {str(fi_error)}\n"
                        
                        # Convert metrics to serializable format
                        serializable_metrics = {}
                        for k, v in model.metrics.items():
                            serializable_metrics[k] = convert_to_serializable(v)
                        
                        # Save confusion matrix if available
                        confusion_matrix_json = None
                        if hasattr(model, 'confusion_matrix_data') and model.confusion_matrix_data is not None:
                            try:
                                cm_data = model.confusion_matrix_data
                                if hasattr(cm_data, 'tolist'):
                                    cm_data = cm_data.tolist()
                                confusion_matrix_json = json.dumps(cm_data)
                            except Exception as cm_error:
                                fresh_model_run.logs += f"Warning: Error processing confusion matrix: {str(cm_error)}\n"
                        
                        # Update model run with results
                        fresh_model_run.status = 'completed'
                        fresh_model_run.metrics = json.dumps(serializable_metrics)
                        fresh_model_run.feature_importances = json.dumps(feature_importances_dict) if feature_importances_dict else None
                        fresh_model_run.model_path = model_path
                        fresh_model_run.report_path = report_path
                        fresh_model_run.completed_at = datetime.utcnow()
                        
                        # Save confusion matrix and classification report if available
                        if confusion_matrix_json:
                            fresh_model_run.confusion_matrix = confusion_matrix_json
                        
                        if hasattr(model, 'classification_report_data') and model.classification_report_data is not None:
                            fresh_model_run.classification_report = model.classification_report_data
                        
                        fresh_model_run.logs += "Model training completed successfully!\n"
                        session.commit()
                        
                        # Send completion notification
                        if user and hasattr(user, 'notify_on_completion') and user.notify_on_completion:
                            try:
                                from app.utils.email import send_model_completion_notification
                                send_model_completion_notification(
                                    user.email,
                                    user.get_full_name(),
                                    fresh_experiment.name,
                                    fresh_model_run.algorithm,
                                    serializable_metrics
                                )
                            except Exception as email_error:
                                print(f"Error sending completion notification: {str(email_error)}")
                    
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        print(f"Training error: {str(e)}\n{error_details}")
                        
                        # Get the models in case they weren't fetched yet
                        user = None
                        if 'fresh_model_run' not in locals():
                            fresh_model_run = session.query(ProModelRun).get(model_run_id)
                        
                        if 'fresh_experiment' not in locals():
                            fresh_experiment = session.query(Experiment).get(fresh_model_run.experiment_id)
                            user = session.query(User).get(fresh_experiment.user_id)
                        
                        # Update with error
                        fresh_model_run.status = 'failed'
                        fresh_model_run.logs += f"Model training failed: {str(e)}\n"
                        fresh_model_run.completed_at = datetime.utcnow()
                        session.commit()
                        
                        # Send error notification
                        if user and hasattr(user, 'notify_on_error') and user.notify_on_error:
                            try:
                                from app.utils.email import send_error_notification
                                send_error_notification(
                                    user.email,
                                    user.get_full_name(),
                                    fresh_experiment.name,
                                    fresh_model_run.algorithm,
                                    str(e)
                                )
                            except Exception as email_error:
                                print(f"Error sending error notification: {str(email_error)}")
                    
                    finally:
                        # Close the session
                        session.close()
                        print("Training thread complete, session closed")
            
            except Exception as outer_e:
                print(f"Outer training error: {str(outer_e)}")
                import traceback
                traceback.print_exc()
        
        # Create a dictionary with all the model_run attributes used in the template
        model_run_dict = {
            'id': model_run.id,
            'algorithm': model_run.algorithm,
            'model_type': model_run.model_type,
            'status': model_run.status,
            'logs': model_run.logs
        }
        
        # Start training in background thread with captured variables
        from threading import Thread
        training_thread = Thread(
            target=training_task, 
            args=(file_path, feature_columns, hyperparameters)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Pass the dictionary instead of the SQLAlchemy object
        return render_template(
            'projecttemp/train_model.html',
            experiment=experiment,
            model_run=model_run,
            json=json,
            dataset_info=dataset_info
        )
        
    except Exception as e:
        # Log any errors
        print(f"Error in train_model route: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Show error template with detailed message
        return render_template(
            'errors/error.html',
            error_message=f"An error occurred: {str(e)}",
            traceback=traceback.format_exc() if current_app.debug else None
        )
 
@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/training-status')
@login_required
def training_status(experiment_id, model_run_id):
    """Get model training status (AJAX endpoint)"""
    try:
        # Create a new session for this request to avoid threading issues
        session = db.create_scoped_session()
        
        # Get model run
        model_run = session.query(ProModelRun).get(model_run_id)
        if not model_run:
            return jsonify({'error': 'Model run not found'}), 404
        
        # Check if this experiment belongs to the current user
        experiment = session.query(Experiment).get(model_run.experiment_id)
        if not experiment or experiment.user_id != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        
        # Get metrics if completed
        metrics = None
        if model_run.metrics:
            try:
                metrics = json.loads(model_run.metrics)
            except json.JSONDecodeError:
                metrics = {'error': 'Error parsing metrics JSON'}
        
        # Return status info
        return jsonify({
            'status': model_run.status,
            'logs': model_run.logs,
            'metrics': metrics
        })
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in training_status: {str(e)}\n{error_traceback}")
        
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    finally:
        session.close()
        
@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/results')
@login_required
def model_results(experiment_id, model_run_id):
    """Display model results page for a trial"""
    try:
        # Get experiment and model run
        experiment = Experiment.query.get_or_404(experiment_id)
        model_run = ProModelRun.query.get_or_404(model_run_id)
        
        # Check if this experiment belongs to the current user
        if experiment.user_id != current_user.id:
            flash('Access denied', 'danger')
            return redirect(url_for('dashboard.index'))
        
        # Check if model run belongs to this experiment
        if model_run.experiment_id != experiment.id:
            flash('Invalid model for this experiment', 'danger')
            return redirect(url_for('project.view', experiment_id=experiment.id))
        
        # Check if model is completed
        if model_run.status != 'completed':
            flash('Model training is not yet complete', 'warning')
            return redirect(url_for('project.train_model', experiment_id=experiment.id, model_run_id=model_run.id))
        
        # Get metrics
        metrics = {}
        if model_run.metrics:
            try:
                metrics = json.loads(model_run.metrics)
            except json.JSONDecodeError:
                flash('Error parsing metrics data', 'warning')
        
        # Get feature importances if available
        feature_importances = {}
        if model_run.feature_importances:
            try:
                feature_importances = json.loads(model_run.feature_importances)
            except json.JSONDecodeError:
                flash('Error parsing feature importance data', 'warning')
        
        # Get confusion matrix if available (for classification)
        confusion_matrix = None
        if model_run.confusion_matrix:
            try:
                confusion_matrix = json.loads(model_run.confusion_matrix)
            except json.JSONDecodeError:
                flash('Error parsing confusion matrix data', 'warning')
        
        # Process classification report - THIS IS THE NEW CODE
        classification_report_data = None
        try:
            if model_run.classification_report:
                # Check if it's valid JSON first
                try:
                    classification_report_data = json.loads(model_run.classification_report)
                except json.JSONDecodeError:
                    # If not JSON, it's probably a string representation of the report
                    # Store it as a string to display in a <pre> tag
                    classification_report_data = model_run.classification_report
        except Exception as e:
            print(f"Error processing classification report: {str(e)}")
            # Just leave it as None
        
        # Check if user has API hosting subscription
        has_api_hosting = current_user.has_api_hosting if hasattr(current_user, 'has_api_hosting') else False
        
        # Check if this model is already deployed as API
        is_deployed = False
        if hasattr(model_run, 'api_endpoint') and hasattr(model_run, 'api_key'):
            is_deployed = model_run.api_endpoint is not None and model_run.api_key is not None
        
        dataset_info = experiment.dataset_info
        
        # Add default value for accuracy if it doesn't exist in metrics
        if experiment.problem_type == 'classification' and 'accuracy' not in metrics:
            metrics['accuracy'] = 0.0
        
        return render_template(
            'projecttemp/model_results.html',
            experiment=experiment,
            model_run=model_run,
            metrics=metrics,
            feature_importances=feature_importances,
            confusion_matrix=confusion_matrix,
            classification_report_data=classification_report_data,  # Pass the processed data
            has_api_hosting=has_api_hosting,
            json=json,
            is_deployed=is_deployed, 
            dataset_info=dataset_info
        )
    except Exception as e:
        # Log any errors
        print(f"Error in model_results route: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Show error template with detailed message
        return render_template(
            'errors/error.html',
            error_message=f"An error occurred: {str(e)}",
            traceback=traceback.format_exc() if current_app.debug else None
        )
        
@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/deploy-api', methods=['GET', 'POST'])
@login_required
def deploy_model_api(experiment_id, model_run_id):
    """Deploy model as API"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    # Check if model is completed
    if model_run.status != 'completed':
        flash('Model training is not yet complete', 'warning')
        return redirect(url_for('project.train_model', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Check if user has API hosting subscription
    if not current_user.has_api_hosting:
        # Show payment page
        return render_template(
            'projecttemp/api_hosting_payment.html',
            experiment=experiment,
            model_run=model_run,json=json,
            price=199  # Price in INR
        )
    
    if request.method == 'POST':
        # Deploy model as API
        # Generate API key
        api_key = str(uuid.uuid4())
        
        # Deploy model
        api_info = deploy_model_as_api(
            model_path=model_run.model_path,
            metadata_path=os.path.join(os.path.dirname(model_run.model_path), f"{os.path.basename(model_run.model_path)}_metadata.json"),
            api_key=api_key,
            user_id=current_user.id,
            experiment_id=experiment.id
        )
        
        if 'error' in api_info:
            flash(f'Error deploying model: {api_info["error"]}', 'danger')
            return redirect(url_for('project.model_results', experiment_id=experiment.id, model_run_id=model_run.id))
        
        # Update model run with API info
        model_run.api_endpoint = api_info['api_endpoint']
        model_run.api_key = api_info['api_key']
        db.session.commit()
        
        # Create API usage record
        api_usage = ApiUsage(
            user_id=current_user.id,
            experiment_id=experiment.id,
            request_count=0,
            total_tokens=0,
            successful_requests=0,
            failed_requests=0,
            monthly_limit=10000,  # Default limit
            usage_reset_date=(datetime.utcnow() + timedelta(days=30))  # 30 days from now
        )
        
        db.session.add(api_usage)
        db.session.commit()
        
        flash('Model deployed as API successfully!', 'success')
        return redirect(url_for('project.api_details', experiment_id=experiment.id, model_run_id=model_run.id))
    
    return render_template(
        'projecttemp/deploy_api.html',
        experiment=experiment,json=json,
        model_run=model_run
    )

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/pay-for-api-hosting', methods=['POST'])
@login_required
def pay_for_api_hosting(experiment_id, model_run_id):
    """Process payment for API hosting"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    # Check if user already has API hosting
    if current_user.has_api_hosting:
        flash('You already have API hosting enabled', 'info')
        return redirect(url_for('project.deploy_model_api', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Simulate payment processing
    # In a real app, you would integrate with a payment gateway like Razorpay
    try:
        # Create a one-time subscription for API hosting
        subscription = Subscription(
            user_id=current_user.id,
            plan_type='api_hosting',
            subscription_id=f"api_{uuid.uuid4()}",
            payment_id=f"payment_{uuid.uuid4()}",
            amount=199,
            currency='INR',
            status='active',
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=30),  # 30 days validity
            experiment_id=experiment.id,
            is_one_time=True
        )
        
        db.session.add(subscription)
        
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
        return redirect(url_for('project.deploy_model_api', experiment_id=experiment.id, model_run_id=model_run.id))

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/api-details')
@login_required
def api_details(experiment_id, model_run_id):
    """Display API details page"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    # Check if model has API endpoint
    if not model_run.api_endpoint or not model_run.api_key:
        flash('Model is not deployed as API yet', 'warning')
        return redirect(url_for('project.deploy_model_api', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Get API usage stats
    api_usage = ApiUsage.query.filter_by(
        user_id=current_user.id,
        experiment_id=experiment.id
    ).first()
    
    # Generate sample code snippets
    python_snippet = f"""
import requests
import json

# API Endpoint
url = "{model_run.api_endpoint}"

# Your API Key
headers = {{
    "Content-Type": "application/json",
    "Authorization": "Bearer {model_run.api_key}"
}}

# Sample data for prediction
data = {{
    "features": [
        # Your feature values here
    ]
}}

# Make prediction request
response = requests.post(url, headers=headers, data=json.dumps(data))

# Parse response
if response.status_code == 200:
    prediction = response.json()
    print(f"Prediction: {{prediction}}")
else:
    print(f"Error: {{response.status_code}} - {{response.text}}")
"""
    
    javascript_snippet = f"""
// API Endpoint
const url = "{model_run.api_endpoint}";

// Your API Key
const headers = {{
    "Content-Type": "application/json",
    "Authorization": "Bearer {model_run.api_key}"
}};

// Sample data for prediction
const data = {{
    "features": [
        // Your feature values here
    ]
}};

// Make prediction request
fetch(url, {{
    method: 'POST',
    headers: headers,
    body: JSON.stringify(data)
}})
.then(response => response.json())
.then(prediction => {{
    console.log("Prediction:", prediction);
}})
.catch(error => {{
    console.error("Error:", error);
}});
"""
    
    curl_snippet = f"""
curl -X POST \\
  {model_run.api_endpoint} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {model_run.api_key}" \\
  -d '{{
    "features": [
        // Your feature values here
    ]
}}'
"""
    
    return render_template(
        'projecttemp/api_details.html',
        experiment=experiment,
        model_run=model_run,
        api_usage=api_usage,
        python_snippet=python_snippet,
        javascript_snippet=javascript_snippet,json=json,
        curl_snippet=curl_snippet
    )

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/download-model')
@login_required
def download_model(experiment_id, model_run_id):
    """Download trained model"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    # Check if model is completed and has a path
    if model_run.status != 'completed' or not model_run.model_path:
        flash('Model is not ready for download', 'warning')
        return redirect(url_for('project.model_results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Check if file exists
    if not os.path.exists(model_run.model_path):
        flash('Model file not found', 'danger')
        return redirect(url_for('project.model_results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Return file for download
    return send_file(
        model_run.model_path,
        as_attachment=True,
        download_name=f"{experiment.name}_{model_run.algorithm}_model.pkl",
        mimetype='application/octet-stream'
    )

@project_bp.route('/project/<experiment_id>/trial/<model_run_id>/download-report')
@login_required
def download_model_report(experiment_id, model_run_id):
    """Download model report"""
    # Get experiment and model run
    experiment = Experiment.query.get_or_404(experiment_id)
    model_run = ProModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if model run belongs to this experiment
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    # Check if model is completed
    if model_run.status != 'completed':
        flash('Model training is not complete', 'warning')
        return redirect(url_for('project.model_results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Check if report exists or generate it
    if not model_run.report_path or not os.path.exists(model_run.report_path):
        try:
            # Get metrics
            metrics = json.loads(model_run.metrics) if model_run.metrics else {}
            
            # Get feature importances and other data if available
            feature_importances = json.loads(model_run.feature_importances) if model_run.feature_importances else None
            confusion_matrix = json.loads(model_run.confusion_matrix) if model_run.confusion_matrix else None
            classification_report_data = model_run.classification_report
            
            # For quantum models, include circuit visualization
            quantum_circuit = None
            if model_run.model_type == 'quantum' and model_run.quantum_circuit:
                quantum_circuit = json.loads(model_run.quantum_circuit)
            
            # Generate report
            report_path = generate_model_report(
                model_run.id,
                model_run.algorithm,
                experiment.problem_type,
                metrics,
                feature_importances,
                confusion_matrix,
                classification_report_data,
                current_app.config['REPORT_FOLDER'],
                tuning_report_path=model_run.tuning_report_path,
                experiment_name=experiment.name,
                quantum_circuit=quantum_circuit
            )
            
            # Update model run with report path
            model_run.report_path = report_path
            db.session.commit()
            
        except Exception as e:
            flash(f'Error generating model report: {str(e)}', 'danger')
            return redirect(url_for('project.model_results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Return file for download
    return send_file(
        model_run.report_path,
        as_attachment=True,
        download_name=f"{experiment.name}_{model_run.algorithm}_report.pdf",
        mimetype='application/pdf'
    )

@project_bp.route('/project/<experiment_id>/trials-summary')
@login_required
def trials_summary(experiment_id):
    """Display summary of all trials for an experiment"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get all model runs for this experiment
    model_runs = ProModelRun.query.filter_by(
        experiment_id=experiment.id
    ).order_by(ProModelRun.created_at.desc()).all()
    
    # Prepare data for comparison visualization
    comparison_data = []
    
    # Group by algorithm and model type for clearer comparison
    algorithms = []
    scores = []
    
    for model_run in model_runs:
        if model_run.status == 'completed' and model_run.metrics:
            metrics = json.loads(model_run.metrics)
            
            # Create a summary dict
            summary = {
                'id': model_run.id,
                'algorithm': model_run.algorithm,
                'model_type': model_run.model_type,
                'created_at': model_run.created_at,
                'completed_at': model_run.completed_at,
                'metrics': metrics
            }
            
            comparison_data.append(summary)
            
            # For visualization
            algorithms.append(model_run.algorithm)
            
            # Get primary metric (accuracy for classification, r2 for regression)
            if experiment.problem_type == 'classification':
                score = metrics.get('accuracy', metrics.get('f1', 0))
            else:
                score = metrics.get('r2', metrics.get('explained_variance', 0))
            
            scores.append(score)
    
    return render_template(
        'projecttemp/trials_summary.html',
        experiment=experiment,
        model_runs=model_runs,
        comparison_data=comparison_data,
        algorithms=json.dumps(algorithms),json=json,
        scores=json.dumps(scores)
    )

@project_bp.route('/project/<experiment_id>/download-comparison-report')
@login_required
def download_comparison_report(experiment_id):
    """Download comparison report for all trials"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get all completed model runs for this experiment
    model_runs = ProModelRun.query.filter_by(
        experiment_id=experiment.id,
        status='completed'
    ).all()
    
    if not model_runs:
        flash('No completed models found for comparison', 'warning')
        return redirect(url_for('project.trials_summary', experiment_id=experiment.id))
    
    # Prepare data for comparison report
    models_data = []
    for model_run in model_runs:
        if model_run.metrics:
            model_data = {
                'model_run_id': model_run.id,
                'algorithm': model_run.algorithm,
                'problem_type': experiment.problem_type,
                'metrics': json.loads(model_run.metrics),
                'feature_importances': json.loads(model_run.feature_importances) if model_run.feature_importances else None
            }
            models_data.append(model_data)
    
    if not models_data:
        flash('No valid model data found for comparison', 'warning')
        return redirect(url_for('project.trials_summary', experiment_id=experiment.id))
    
    # Generate comparison report
    try:
        report_path = generate_comparison_report(
            models_data=models_data,
            output_folder=current_app.config['REPORT_FOLDER'],
            experiment_name=experiment.name
        )
        
        # Save report path to experiment
        experiment.comparison_report_path = report_path
        db.session.commit()
        
        # Return file for download
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"{experiment.name}_comparison_report.pdf",
            mimetype='application/pdf'
        )
    
    except Exception as e:
        flash(f'Error generating comparison report: {str(e)}', 'danger')
        return redirect(url_for('project.trials_summary', experiment_id=experiment.id))

@project_bp.route('/project/<experiment_id>/experiments-summary')
@login_required
def experiments_summary(experiment_id):
    """Display summary of all experiments for a project"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get all experiments for this user
    experiments = Experiment.query.filter_by(
        user_id=current_user.id,
        is_deleted=False
    ).order_by(Experiment.created_at.desc()).all()
    
    # Get best model for each experiment
    experiment_summary = []
    for exp in experiments:
        # Get best model run for this experiment
        best_model_run = ProModelRun.query.filter_by(
            experiment_id=exp.id,
            status='completed'
        ).order_by(ProModelRun.created_at.desc()).first()
        
        if best_model_run and best_model_run.metrics:
            metrics = json.loads(best_model_run.metrics)
            
            summary = {
                'id': exp.id,
                'name': exp.name,
                'problem_type': exp.problem_type,
                'created_at': exp.created_at,
                'best_model': {
                    'id': best_model_run.id,
                    'algorithm': best_model_run.algorithm,
                    'model_type': best_model_run.model_type,
                    'metrics': metrics
                }
            }
            
            experiment_summary.append(summary)
    
    return render_template(
        'projecttemp/experiments_summary.html',
        experiment=experiment,
        experiments=experiments,json=json,
        experiment_summary=experiment_summary
    )

@project_bp.route('/project/<experiment_id>/download-project-report')
@login_required
def download_project_report(experiment_id):
    """Download comprehensive project report"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Generate comprehensive project report
    try:
        # Get all completed model runs
        model_runs = ProModelRun.query.join(Experiment).filter(
            Experiment.user_id == current_user.id,
            Experiment.is_deleted == False,
            ProModelRun.status == 'completed'
        ).all()
        
        # Prepare data for report
        models_data = []
        for model_run in model_runs:
            if model_run.metrics:
                exp = Experiment.query.get(model_run.experiment_id)
                
                model_data = {
                    'model_run_id': model_run.id,
                    'algorithm': model_run.algorithm,
                    'problem_type': exp.problem_type,
                    'experiment_name': exp.name,
                    'metrics': json.loads(model_run.metrics),
                    'feature_importances': json.loads(model_run.feature_importances) if model_run.feature_importances else None
                }
                models_data.append(model_data)
        
        if not models_data:
            flash('No completed models found for project report', 'warning')
            return redirect(url_for('project.experiments_summary', experiment_id=experiment.id))
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(
            models_data=models_data,
            output_folder=current_app.config['REPORT_FOLDER'],
            experiment_name=f"Project Report - {current_user.username}"
        )
        
        # Return file for download
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"project_report.pdf",
            mimetype='application/pdf'
        )
    
    except Exception as e:
        flash(f'Error generating project report: {str(e)}', 'danger')
        return redirect(url_for('project.experiments_summary', experiment_id=experiment.id))

@project_bp.route('/project/<experiment_id>/upgrade-to-pro')
@login_required
def upgrade_to_pro(experiment_id):
    """Upgrade to Pro subscription page"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user already has Pro
    if current_user.is_pro and current_user.has_active_pro_subscription():
        flash('You already have a Pro subscription', 'info')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    return render_template(
        'projecttemp/upgrade_to_pro.html',
        experiment=experiment,json=json,
        monthly_price=249,  # Monthly price in INR
        yearly_price=2499   # Yearly price in INR
    )

@project_bp.route('/project/<experiment_id>/pay-for-pro', methods=['POST'])
@login_required
def pay_for_pro(experiment_id):
    """Process payment for Pro subscription"""
    # Get experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current user
    if experiment.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Check if user already has Pro
    if current_user.is_pro and current_user.has_active_pro_subscription():
        flash('You already have a Pro subscription', 'info')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
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
        
        # Update user
        current_user.is_pro = True
        current_user.pro_since = datetime.utcnow()
        current_user.pro_expires = datetime.utcnow() + timedelta(days=duration_days)
        
        db.session.commit()
        
        # Send subscription confirmation email
        send_subscription_confirmation(
            current_user.email, 
            current_user.get_full_name(),
            f"Pro {subscription_type.capitalize()}",
            subscription.end_date
        )
        
        flash(f'Payment successful! Your Pro subscription is now active.', 'success')
        return redirect(url_for('project.view', experiment_id=experiment.id))
    
    except Exception as e:
        flash(f'Payment failed: {str(e)}', 'danger')
        return redirect(url_for('project.upgrade_to_pro', experiment_id=experiment.id))

    
def create_app_context_thread(target_func, *args, **kwargs):
    """Create a thread with Flask application context and db session management"""
    # Get the actual application object (not proxy)
    app = current_app._get_current_object()
    is_pro = current_user.is_pro if hasattr(current_user, 'is_pro') else False
    def wrapper(*args, **kwargs):
        try:
            with app.app_context():
                # Create a new session for this thread
                session = db.create_scoped_session()
                # Run the function with the new session
                try:
                    result = target_func(session=session, *args, **kwargs)
                    return result
                except Exception as e:
                    # Rollback the session if there's an error
                    session.rollback()
                    raise
                finally:
                    # Close the session when done
                    session.close()
        except Exception as e:
            print(f"Thread error: {str(e)}")
            # Handle error inside app context
            with app.app_context():
                if 'error_callback' in kwargs:
                    error_callback = kwargs.pop('error_callback')
                    error_callback(str(e))
    
    thread = Thread(target=wrapper, args=args, kwargs=kwargs)
    return thread