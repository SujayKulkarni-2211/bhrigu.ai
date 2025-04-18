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
main_bp = Blueprint('main', __name__)
# Ensure guest has a session ID for tracking experiments
@main_bp.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@main_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')

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
        return redirect(url_for('main.configure_model', experiment_id=experiment.id, model_run_id=model_run.id))
    
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

@main_bp.route('/train-model/<experiment_id>/<model_run_id>')
def train_model(experiment_id, model_run_id):
    """Train model page"""
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
    
    # Check if model is already training or completed
    if model_run.status in ['training', 'completed', 'failed', 'timed_out']:
        return render_template(
            'train_model.html',
            experiment=experiment,
            model_run=model_run,
            is_configuring=False
        )
    
    # Get file path and hyperparameters
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], experiment.dataset_filename)
    hyperparameters = json.loads(model_run.hyperparameters) if model_run.hyperparameters else {}
    feature_columns = json.loads(experiment.feature_columns) if experiment.feature_columns else []
    
    # Start model training in background
    train_model_async(
        model_run.id,
        model_run.algorithm,
        experiment.problem_type,
        file_path,
        experiment.target_column,
        feature_columns,
        hyperparameters,
        current_app.config['MODEL_FOLDER'],
        current_app.config['GUEST_MAX_TRAINING_TIME']
    )
    
    # Update experiment status
    experiment.status = 'training'
    experiment.started_at = db.func.now()
    db.session.commit()
    
    flash('Model training started! This may take a few minutes.', 'info')
    return render_template(
        'train_model.html',
        experiment=experiment,
        model_run=model_run,
        is_configuring=False
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
        'train_model.html',
        experiment=experiment,
        model_run=model_run,
        is_configuring=False,
        is_tuning=True
    )

@main_bp.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'ok'})