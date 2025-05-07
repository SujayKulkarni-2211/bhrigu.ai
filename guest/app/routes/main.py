# app/routes/main.py
import os
import json
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session, jsonify, send_file
from werkzeug.utils import secure_filename
from app import db
from app.models.experiment import GuestExperiment, DatasetInfo, ModelRun
from app.utils.data_processing import save_uploaded_file, analyze_dataset
from app.utils.ml_utils import get_model_recommendations, train_model_async, CLASSIFICATION_MODELS, REGRESSION_MODELS, get_gemini_recommendations
from datetime import datetime
import time
from datetime import datetime
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

main_bp = Blueprint('main', __name__)
# Ensure guest has a session ID for tracking experiments
@main_bp.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


PRO_SERVICE_URL = os.environ.get('PRO_SERVICE_URL', 'http://bhrigu-pro:5001')
@main_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html',PRO_SERVICE_URL=PRO_SERVICE_URL)

@main_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload dataset page"""
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
            file, 
            current_app.config['UPLOAD_FOLDER'],
            current_app.config['ALLOWED_EXTENSIONS']
        )
        
        if not file_path:
            flash('Invalid file type. Please upload a CSV or Excel file.', 'danger')
            return redirect(request.url)
        
        # Create a new guest experiment
        experiment = GuestExperiment(
            session_id=session['session_id'],
            dataset_filename=filename,
            dataset_original_name=original_filename,
            status='data_uploaded'
        )
        
        db.session.add(experiment)
        db.session.commit()
        
        flash('Dataset uploaded successfully!', 'success')
        return redirect(url_for('main.analyze', experiment_id=experiment.id))
    
    return render_template('upload.html')

@main_bp.route('/analyze/<experiment_id>', methods=['GET', 'POST'])
def analyze(experiment_id):
    """Analyze dataset page"""
    # Get experiment
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if dataset exists
    if not experiment.dataset_filename:
        flash('Please upload a dataset first', 'warning')
        return redirect(url_for('main.upload'))
    
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], experiment.dataset_filename)
    
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
        return redirect(url_for('main.select_model', experiment_id=experiment.id))
    
    # Analyze dataset if not already done
    if not experiment.dataset_info:
        try:
            data_info = analyze_dataset(file_path)
            
            # Create dataset info record
            dataset_info = DatasetInfo(
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
            return redirect(url_for('main.upload'))
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
            'correlation_image': dataset_info.correlation_matrix
        }
    
    return render_template(
        'analyze.html',
        experiment=experiment,
        data_info=data_info
    )

@main_bp.route('/select-model/<experiment_id>', methods=['GET', 'POST'])
def select_model(experiment_id):
    """Select model page"""
    # Get experiment
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if dataset is analyzed
    if not experiment.problem_type or not experiment.target_column or not experiment.feature_columns:
        flash('Please analyze your dataset first', 'warning')
        return redirect(url_for('main.analyze', experiment_id=experiment.id))
    
    # Get dataset info
    dataset_info = experiment.dataset_info
    data_info = {
        'num_rows': dataset_info.num_rows,
        'num_cols': dataset_info.num_columns,
        'numeric_columns': json.loads(dataset_info.numeric_columns) if dataset_info.numeric_columns else [],
        'categorical_columns': json.loads(dataset_info.categorical_columns) if dataset_info.categorical_columns else []
    }
    
    if request.method == 'POST':
        # Get selected algorithm
        algorithm = request.form.get('algorithm')
        
        if not algorithm:
            flash('Please select an algorithm', 'danger')
            return redirect(request.url)
        
        # Create model run
        model_run = ModelRun(
            experiment_id=experiment.id,
            algorithm=algorithm,
            model_type='classical',  # Only classical models for guest users
            status='created'
        )
        
        db.session.add(model_run)
        db.session.commit()
        
        # Update experiment status
        experiment.status = 'model_selected'
        db.session.commit()
        
        flash('Model selected! Configure hyperparameters to start training.', 'success')
        return redirect(url_for('main.hyperparameter_choice', experiment_id=experiment.id, algorithm=algorithm))
    
    # Get model recommendations
    recommendations = get_gemini_recommendations(experiment.problem_type, data_info)
    
    # Get available models based on problem type
    if experiment.problem_type == 'classification':
        available_models = CLASSIFICATION_MODELS
    else:  # regression
        available_models = REGRESSION_MODELS
    
    return render_template(
        'select_model.html',
        experiment=experiment,
        recommendations=recommendations,
        available_models=available_models
    )

@main_bp.route('/configure-model/<experiment_id>/<model_run_id>', methods=['GET', 'POST'])
def configure_model(experiment_id, model_run_id):
    """Configure model hyperparameters page"""
    # Get experiment and model run
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('main.select_model', experiment_id=experiment.id))
    
    if request.method == 'POST':
        # Get hyperparameters based on algorithm
        hyperparameters = {}
        find_best = 'find_best_hyperparameters' in request.form
        if experiment.problem_type == 'classification':
            if model_run.algorithm == 'RandomForest':
                hyperparameters['n_estimators'] = int(request.form.get('n_estimators', 100))
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
                hyperparameters['criterion'] = request.form.get('criterion', 'gini')
            elif model_run.algorithm == 'LogisticRegression':
                hyperparameters['C'] = float(request.form.get('C', 1.0))
                hyperparameters['max_iter'] = int(request.form.get('max_iter', 1000))
            elif model_run.algorithm == 'SVC':
                hyperparameters['C'] = float(request.form.get('C', 1.0))
                hyperparameters['kernel'] = request.form.get('kernel', 'rbf')
                hyperparameters['gamma'] = request.form.get('gamma', 'scale')
            elif model_run.algorithm == 'DecisionTree':
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
                hyperparameters['criterion'] = request.form.get('criterion', 'gini')
            elif model_run.algorithm == 'KNN':
                hyperparameters['n_neighbors'] = int(request.form.get('n_neighbors', 5))
                hyperparameters['weights'] = request.form.get('weights', 'uniform')
        else:  # regression
            if model_run.algorithm == 'RandomForestRegressor':
                hyperparameters['n_estimators'] = int(request.form.get('n_estimators', 100))
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
            elif model_run.algorithm == 'LinearRegression':
                pass  # No hyperparameters for Linear Regression
            elif model_run.algorithm == 'SVR':
                hyperparameters['C'] = float(request.form.get('C', 1.0))
                hyperparameters['kernel'] = request.form.get('kernel', 'rbf')
                hyperparameters['gamma'] = request.form.get('gamma', 'scale')
            elif model_run.algorithm == 'DecisionTreeRegressor':
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
        
        # Update model run with hyperparameters
        model_run.hyperparameters = json.dumps(hyperparameters)
        if find_best:
            model_run.hyperparameters_tuning = True
            
        db.session.commit()
        
        if find_best:
            flash('Hyperparameter optimization started! This may take a few minutes.', 'info')
            return redirect(url_for('main.tune_hyperparameters', experiment_id=experiment.id, model_run_id=model_run.id))
        else:
            flash('Hyperparameters configured! You can now start training.', 'success')
            # If you've created a new route for training with parameters
            return redirect(url_for('main.train_model', experiment_id=experiment.id, model_run_id=model_run.id))
        
    
    # Get default hyperparameters based on algorithm
    if experiment.problem_type == 'classification':
        model_info = CLASSIFICATION_MODELS.get(model_run.algorithm, {})
    else:  # regression
        model_info = REGRESSION_MODELS.get(model_run.algorithm, {})
    
    default_params = model_info.get('params', {})
    
    return render_template(
        'train_model.html',
        experiment=experiment,
        model_run=model_run,
        default_params=default_params,
        is_configuring=True
    )



@main_bp.route('/model-status/<model_run_id>')
def model_status(model_run_id):
    """Get model training status (AJAX endpoint)"""
    # Get model run
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    experiment = GuestExperiment.query.get(model_run.experiment_id)
    if experiment.session_id != session['session_id']:
        return jsonify({'error': 'Access denied'}), 403
    
    # Get metrics if completed
    metrics = json.loads(model_run.metrics) if model_run.metrics else None
    
    # Return status info
    return jsonify({
        'status': model_run.status,
        'logs': model_run.logs,
        'metrics': metrics
    })

@main_bp.route('/results/<experiment_id>/<model_run_id>')
def results(experiment_id, model_run_id):
    """Display model results page"""
    # Get experiment and model run
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('main.select_model', experiment_id=experiment.id))
    
    # Check if model is completed
    if model_run.status != 'completed':
        flash('Model training is not yet complete', 'warning')
        return redirect(url_for('main.train_model', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Get metrics
    metrics = json.loads(model_run.metrics) if model_run.metrics else {}
    
    return render_template(
        'results.html',
        experiment=experiment,
        model_run=model_run,
        metrics=metrics
    )

@main_bp.route('/download-model/<model_run_id>')
def download_model(model_run_id):
    """Download trained model"""
    # Get model run
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    experiment = GuestExperiment.query.get(model_run.experiment_id)
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model is completed and has a path
    if model_run.status != 'completed' or not model_run.model_path:
        flash('Model is not ready for download', 'warning')
        return redirect(url_for('main.results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Get file path
    file_path = model_run.model_path
    
    # Check if file exists
    if not os.path.exists(file_path):
        flash('Model file not found', 'danger')
        return redirect(url_for('main.results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Return file for download
    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"{experiment.dataset_original_name}_{model_run.algorithm}_model.pkl",
        mimetype='application/octet-stream'
    )

# Add this new route to app/routes/main.py
@main_bp.route('/tune-hyperparameters/<experiment_id>/<model_run_id>')
def tune_hyperparameters(experiment_id, model_run_id):
    """Run hyperparameter optimization"""
    # Get experiment and model run
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('main.select_model', experiment_id=experiment.id))
    
    # Get file path and base hyperparameters
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], experiment.dataset_filename)
    base_hyperparameters = json.loads(model_run.hyperparameters) if model_run.hyperparameters else {}
    feature_columns = json.loads(experiment.feature_columns) if experiment.feature_columns else []
    
    # Update model status
    model_run.status = 'tuning'
    model_run.started_at = datetime.utcnow()
    model_run.logs = "Starting hyperparameter optimization...\n"
    db.session.commit()
    
    # Start hyperparameter tuning in background
    from app.utils.ml_utils import tune_hyperparameters_async
    tune_hyperparameters_async(
        model_run.id,
        model_run.algorithm,
        experiment.problem_type,
        file_path,
        experiment.target_column,
        feature_columns,
        base_hyperparameters,
        current_app.config['MODEL_FOLDER'],
        current_app.config['GUEST_MAX_TRAINING_TIME']
    )
    
    return render_template(
        'tuning_status.html',
        experiment=experiment,
        model_run=model_run,
        is_configuring=False,
        is_tuning=True
    )
# Add these imports to app/routes/main.py
from app.utils.ml_report_utils import (generate_tuning_report, generate_model_report,
                                     generate_comprehensive_report, extract_feature_importances)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Add these new routes to app/routes/main.py after the existing select_model route

@main_bp.route('/hyperparameter-choice/<experiment_id>', methods=['GET'])
def hyperparameter_choice(experiment_id):
    """Page to choose between automatic tuning and manual configuration"""
    # Get experiment
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if dataset is analyzed
    if not experiment.problem_type or not experiment.target_column or not experiment.feature_columns:
        flash('Please analyze your dataset first', 'warning')
        return redirect(url_for('main.analyze', experiment_id=experiment.id))
    
    # Create a model run (which will be updated based on the choice)
    model_run = ModelRun(
        experiment_id=experiment.id,
        algorithm=request.args.get('algorithm', 'RandomForest' if experiment.problem_type == 'classification' else 'RandomForestRegressor'),
        model_type='classical',
        status='created'
    )
    
    db.session.add(model_run)
    db.session.commit()
    
    return render_template(
        'hyperparameter_choice.html',
        experiment=experiment,
        model_id=model_run.id,
        model_algorithm=model_run.algorithm
    )

@main_bp.route('/start-hyperparameter-tuning/<experiment_id>/<model_id>')
def start_hyperparameter_tuning(experiment_id, model_id):
    """Start hyperparameter tuning process"""
    # Get experiment and model run
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    model_run = ModelRun.query.get_or_404(model_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('main.hyperparameter_choice', experiment_id=experiment.id))
    
    # Get file path
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], experiment.dataset_filename)
    feature_columns = json.loads(experiment.feature_columns) if experiment.feature_columns else []
    
    # Update model status
    model_run.status = 'tuning'
    model_run.started_at = datetime.utcnow()
    model_run.logs = "Starting hyperparameter optimization...\n"
    model_run.hyperparameters_tuning = True
    db.session.commit()
    
    # Start hyperparameter tuning in background
    from app.utils.ml_utils import tune_hyperparameters_async
    tune_hyperparameters_async(
        model_run.id,
        model_run.algorithm,
        experiment.problem_type,
        file_path,
        experiment.target_column,
        feature_columns,
        {},  # Empty dict for base hyperparameters - we'll try a wide range
        current_app.config['MODEL_FOLDER'],
        current_app.config['GUEST_MAX_TRAINING_TIME']
    )
    
    return render_template(
        'tuning_status.html',
        experiment=experiment,
        model_run=model_run
    )

@main_bp.route('/tuning-status/<model_run_id>')
def tuning_status(model_run_id):
    """Get hyperparameter tuning status (AJAX endpoint)"""
    # Get model run
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    experiment = GuestExperiment.query.get(model_run.experiment_id)
    if experiment.session_id != session['session_id']:
        return jsonify({'error': 'Access denied'}), 403
    
    # Get tuning metrics if completed
    tuning_metrics = json.loads(model_run.tuning_metrics) if model_run.tuning_metrics else None
    
    # Return status info
    return jsonify({
        'status': model_run.status,
        'logs': model_run.logs,
        'tuning_metrics': tuning_metrics
    })

@main_bp.route('/configure-hyperparameters/<experiment_id>/<model_id>', methods=['GET', 'POST'])
@main_bp.route('/configure-hyperparameters/<experiment_id>/<model_id>/<mode>', methods=['GET', 'POST'])
def configure_hyperparameters(experiment_id, model_id, mode=None):
    """Configure hyperparameters page (after tuning or manual)"""
    # Get experiment and model run
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    model_run = ModelRun.query.get_or_404(model_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('main.hyperparameter_choice', experiment_id=experiment.id))
    
    # Check if mode is valid
    if mode and mode not in ['tuning', 'manual']:
        flash('Invalid configuration mode', 'danger')
        return redirect(url_for('main.hyperparameter_choice', experiment_id=experiment.id))
    
    # Set tuning completed flag
    tuning_completed = model_run.hyperparameters_tuning and model_run.status == 'created'
    
    # Get tuning metrics if available
    tuning_metrics = json.loads(model_run.tuning_metrics) if hasattr(model_run, 'tuning_metrics') and model_run.tuning_metrics else None
    
    if request.method == 'POST':
        # Get hyperparameters based on algorithm
        hyperparameters = {}
        
        if experiment.problem_type == 'classification':
            if model_run.algorithm == 'RandomForest':
                hyperparameters['n_estimators'] = int(request.form.get('n_estimators', 100))
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
                hyperparameters['criterion'] = request.form.get('criterion', 'gini')
                hyperparameters['min_samples_split'] = int(request.form.get('min_samples_split', 2))
            elif model_run.algorithm == 'LogisticRegression':
                hyperparameters['C'] = float(request.form.get('C', 1.0))
                hyperparameters['max_iter'] = int(request.form.get('max_iter', 1000))
                hyperparameters['solver'] = request.form.get('solver', 'liblinear')
            elif model_run.algorithm == 'SVC':
                hyperparameters['C'] = float(request.form.get('C', 1.0))
                hyperparameters['kernel'] = request.form.get('kernel', 'rbf')
                hyperparameters['gamma'] = request.form.get('gamma', 'scale')
            elif model_run.algorithm == 'DecisionTree':
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
                hyperparameters['criterion'] = request.form.get('criterion', 'gini')
                hyperparameters['min_samples_split'] = int(request.form.get('min_samples_split', 2))
            elif model_run.algorithm == 'KNN':
                hyperparameters['n_neighbors'] = int(request.form.get('n_neighbors', 5))
                hyperparameters['weights'] = request.form.get('weights', 'uniform')
                hyperparameters['p'] = int(request.form.get('p', 2))
        else:  # regression
            if model_run.algorithm == 'RandomForestRegressor':
                hyperparameters['n_estimators'] = int(request.form.get('n_estimators', 100))
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
                hyperparameters['min_samples_split'] = int(request.form.get('min_samples_split', 2))
            elif model_run.algorithm == 'LinearRegression':
                hyperparameters['fit_intercept'] = request.form.get('fit_intercept', 'True') == 'True'
            elif model_run.algorithm == 'SVR':
                hyperparameters['C'] = float(request.form.get('C', 1.0))
                hyperparameters['kernel'] = request.form.get('kernel', 'rbf')
                hyperparameters['gamma'] = request.form.get('gamma', 'scale')
            elif model_run.algorithm == 'DecisionTreeRegressor':
                hyperparameters['max_depth'] = int(request.form.get('max_depth', 10)) if request.form.get('max_depth') else None
                hyperparameters['min_samples_split'] = int(request.form.get('min_samples_split', 2))
        
        # Update model run with hyperparameters
        model_run.hyperparameters = json.dumps(hyperparameters)
        db.session.commit()
        
        if 'train_full_model' in request.form:
            # Redirect to train model page
            return redirect(url_for('main.train_model', experiment_id=experiment.id, model_run_id=model_run.id))
        else:
            flash('Hyperparameters saved! You can now start training.', 'success')
    
    # Get existing hyperparameters
    hyperparams = json.loads(model_run.hyperparameters) if model_run.hyperparameters else {}
    
    # Get default hyperparameters based on algorithm if no existing hyperparameters
    if not hyperparams:
        if experiment.problem_type == 'classification':
            model_info = CLASSIFICATION_MODELS.get(model_run.algorithm, {})
        else:  # regression
            model_info = REGRESSION_MODELS.get(model_run.algorithm, {})
        
        hyperparams = model_info.get('params', {})
    
    return render_template(
        'configure_hyperparameters.html',
        experiment=experiment,
        model_run=model_run,
        hyperparams=hyperparams,
        tuning_completed=tuning_completed,
        tuning_metrics=tuning_metrics
    )

@main_bp.route('/download-tuning-report/<model_run_id>')
def download_tuning_report(model_run_id):
    """Download tuning report PDF"""
    # Get model run
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    experiment = GuestExperiment.query.get(model_run.experiment_id)
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model has tuning report
    if not model_run.tuning_report_path or not os.path.exists(model_run.tuning_report_path):
        # Generate the report if it doesn't exist
        if model_run.hyperparameters_tuning and model_run.tuning_results:
            try:
                # Prepare data for report
                tuning_results = json.loads(model_run.tuning_results)
                best_params = json.loads(model_run.hyperparameters)
                metrics = json.loads(model_run.tuning_metrics) if model_run.tuning_metrics else {}
                
                # Generate report
                report_path = generate_tuning_report(
                    model_run.id,
                    model_run.algorithm,
                    experiment.problem_type,
                    tuning_results,
                    best_params,
                    metrics,
                    current_app.config['REPORT_FOLDER'],
                    experiment_name=experiment.dataset_original_name
                )
                
                # Update model run with report path
                model_run.tuning_report_path = report_path
                db.session.commit()
                
            except Exception as e:
                flash(f'Error generating tuning report: {str(e)}', 'danger')
                return redirect(url_for('main.configure_hyperparameters', experiment_id=experiment.id, model_id=model_run.id))
        else:
            flash('No tuning report available for this model', 'warning')
            return redirect(url_for('main.configure_hyperparameters', experiment_id=experiment.id, model_id=model_run.id))
    
    # Return file for download
    return send_file(
        model_run.tuning_report_path,
        as_attachment=True,
        download_name=f"{experiment.dataset_original_name}_{model_run.algorithm}_tuning_report.pdf",
        mimetype='application/pdf'
    )

@main_bp.route('/download-classification-report/<model_run_id>')
def download_classification_report(model_run_id):
    """Download classification report PDF"""
    # Get model run
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    experiment = GuestExperiment.query.get(model_run.experiment_id)
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model is completed
    if model_run.status != 'completed':
        flash('Model training is not complete', 'warning')
        return redirect(url_for('main.results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Check if report exists
    if not model_run.report_path or not os.path.exists(model_run.report_path):
        # Generate report if it doesn't exist
        try:
            # Get metrics
            metrics = json.loads(model_run.metrics) if model_run.metrics else {}
            
            # Generate report
            report_path = generate_model_report(
                model_run.id,
                model_run.algorithm,
                experiment.problem_type,
                metrics,
                None,  # We don't have feature importances here
                None,  # We don't have confusion matrix here
                None,  # We don't have classification report text here
                current_app.config['REPORT_FOLDER'],
                tuning_report_path=model_run.tuning_report_path if hasattr(model_run, 'tuning_report_path') else None,
                experiment_name=experiment.dataset_original_name
            )
            
            # Update model run with report path
            model_run.report_path = report_path
            db.session.commit()
            
        except Exception as e:
            flash(f'Error generating model report: {str(e)}', 'danger')
            return redirect(url_for('main.results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Return file for download
    return send_file(
        model_run.report_path,
        as_attachment=True,
        download_name=f"{experiment.dataset_original_name}_{model_run.algorithm}_model_report.pdf",
        mimetype='application/pdf'
    )

@main_bp.route('/download-comprehensive-report/<model_run_id>')
def download_comprehensive_report(model_run_id):
    """Download comprehensive report PDF (model + tuning)"""
    # Get model run
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    experiment = GuestExperiment.query.get(model_run.experiment_id)
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model is completed
    if model_run.status != 'completed':
        flash('Model training is not complete', 'warning')
        return redirect(url_for('main.results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Generate comprehensive report
    try:
        # Get metrics
        metrics = json.loads(model_run.metrics) if model_run.metrics else {}
        
        # Get tuning results if available
        tuning_results = json.loads(model_run.tuning_results) if hasattr(model_run, 'tuning_results') and model_run.tuning_results else None
        best_params = json.loads(model_run.hyperparameters) if model_run.hyperparameters else {}
        
        # Get feature importances, confusion matrix, and classification report if available
        feature_importances = json.loads(model_run.feature_importances) if hasattr(model_run, 'feature_importances') and model_run.feature_importances else None
        confusion_matrix_data = json.loads(model_run.confusion_matrix) if hasattr(model_run, 'confusion_matrix') and model_run.confusion_matrix else None
        classification_report_data = model_run.classification_report if hasattr(model_run, 'classification_report') and model_run.classification_report else None
        
        # Generate report
        report_path = generate_comprehensive_report(
            model_run.id,
            model_run.algorithm,
            experiment.problem_type,
            metrics,
            feature_importances,
            confusion_matrix_data,
            classification_report_data,
            tuning_results,
            best_params,
            current_app.config['REPORT_FOLDER'],
            experiment_name=experiment.dataset_original_name
        )
        
        # Update model run with comprehensive report path
        model_run.comprehensive_report_path = report_path
        db.session.commit()
        
    except Exception as e:
        flash(f'Error generating comprehensive report: {str(e)}', 'danger')
        return redirect(url_for('main.results', experiment_id=experiment.id, model_run_id=model_run.id))
    
    # Return file for download
    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"{experiment.dataset_original_name}_{model_run.algorithm}_comprehensive_report.pdf",
        mimetype='application/pdf'
    )


@main_bp.route('/train-model/<experiment_id>/<model_run_id>')
def train_model(experiment_id, model_run_id):
    """Train a machine learning model with time limit and generate reports"""
    # Get experiment and model run
    experiment = GuestExperiment.query.get_or_404(experiment_id)
    model_run = ModelRun.query.get_or_404(model_run_id)
    
    # Check if this experiment belongs to the current session
    if experiment.session_id != session['session_id']:
        flash('Access denied', 'danger')
        return redirect(url_for('main.upload'))
    
    # Check if model run belongs to this experiment
    if model_run.experiment_id != experiment.id:
        flash('Invalid model for this experiment', 'danger')
        return redirect(url_for('main.select_model', experiment_id=experiment.id))
    
    # Get file path and hyperparameters
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], experiment.dataset_filename)
    hyperparameters = json.loads(model_run.hyperparameters) if model_run.hyperparameters else {}
    feature_columns = json.loads(experiment.feature_columns) if experiment.feature_columns else []
    algorithm = model_run.algorithm
    problem_type = experiment.problem_type
    model_folder = current_app.config['MODEL_FOLDER']
    time_limit = current_app.config['GUEST_MAX_TRAINING_TIME']
    target_column = experiment.target_column
    
    # Start model training in background
    from app.utils.ml_utils import train_full_model_async
    train_full_model_async(
        model_run.id,
        algorithm,
        problem_type,
        file_path,
        target_column,
        feature_columns,
        hyperparameters,
        model_folder,
        time_limit
    )
    
    # Update model status
    model_run.status = 'training'
    model_run.started_at = datetime.utcnow()
    db.session.commit()
    
    return render_template(
        'train_model.html',
        experiment=experiment,
        model_run=model_run,
        is_configuring=False,
        is_tuning=False
    )
            
@main_bp.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'ok'})