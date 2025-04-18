# app/utils/ml_utils.py
import os
import json
import pickle
import time
import threading
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from google import genai

# Dictionary of available models
CLASSIFICATION_MODELS = {
    'RandomForest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'criterion': 'gini'
        }
    },
    'LogisticRegression': {
        'class': LogisticRegression,
        'params': {
            'max_iter': 1000,
            'C': 1.0
        }
    },
    'SVC': {
        'class': SVC,
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        }
    },
    'DecisionTree': {
        'class': DecisionTreeClassifier,
        'params': {
            'max_depth': 10,
            'criterion': 'gini'
        }
    },
    'KNN': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'weights': 'uniform'
        }
    },
        'MLP': {
            'class': None,  # Placeholder
            'params': {
                'hidden_layer_sizes': (100, 100),
                'activation': 'relu',
                'solver': 'adam',
                'learning_rate': 'adaptive'
            },
            'disabled': True,
            'disabled_reason': 'Available for logged-in users'
        },
        'CNN': {
            'class': None,  # Placeholder
            'params': {
                'layers': 3,
                'filters': [32, 64, 128],
                'kernel_size': 3,
                'pool_size': 2
            },
            'disabled': True,
            'disabled_reason': 'Available for logged-in users'
        },
        # Quantum models (disabled for guest users)
        'QSVM': {
            'class': None,  # Placeholder
            'params': {
                'feature_map': 'ZZFeatureMap',
                'shots': 1024
            },
            'disabled': True,
            'disabled_reason': 'Available for Pro users'
        },
        'QNN': {
            'class': None,  # Placeholder
            'params': {
                'qubits': 4,
                'layers': 2
            },
            'disabled': True,
            'disabled_reason': 'Available for Pro users'
        }
}

REGRESSION_MODELS = {
    'RandomForestRegressor': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 10
        }
    },
    'LinearRegression': {
        'class': LinearRegression,
        'params': {}
    },
    'SVR': {
        'class': SVR,
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        }
    },
    'DecisionTreeRegressor': {
        'class': DecisionTreeRegressor,
        'params': {
            'max_depth': 10
        }
    }
}

def get_model_recommendations(problem_type, data_info):
    """Get basic model recommendations based on dataset characteristics"""
    recommendations = []
    
    # Get dataset characteristics
    num_samples = data_info['num_rows']
    num_features = len(data_info['numeric_columns']) + len(data_info['categorical_columns'])
    has_categorical = len(data_info['categorical_columns']) > 0
    
    if problem_type == 'classification':
        # Default recommendations
        recommendations.append({
            'algorithm': 'RandomForest',
            'model_type': 'classical',
            'confidence': 0.85,
            'reason': 'Random Forest typically performs well on a variety of datasets and is less prone to overfitting.'
        })
        
        # Check for linear relationships (oversimplified heuristic)
        if num_features < 10 and not has_categorical:
            recommendations.append({
                'algorithm': 'LogisticRegression',
                'model_type': 'classical',
                'confidence': 0.75,
                'reason': 'Logistic Regression works well for smaller datasets with fewer features.'
            })
        
        # Add SVC for smaller datasets
        if num_samples < 10000:
            recommendations.append({
                'algorithm': 'SVC',
                'model_type': 'classical',
                'confidence': 0.70,
                'reason': 'Support Vector Machines can handle complex decision boundaries.'
            })
        
        # Add KNN for smaller datasets
        if num_samples < 5000:
            recommendations.append({
                'algorithm': 'KNN',
                'model_type': 'classical',
                'confidence': 0.65,
                'reason': 'K-Nearest Neighbors works well on smaller datasets with meaningful distance metrics.'
            })
            
        # Add Decision Tree
        recommendations.append({
            'algorithm': 'DecisionTree',
            'model_type': 'classical',
            'confidence': 0.60,
            'reason': 'Decision Trees provide interpretable results and handle mixed data types well.'
        })
    
    elif problem_type == 'regression':
        # Default recommendations
        recommendations.append({
            'algorithm': 'RandomForestRegressor',
            'model_type': 'classical',
            'confidence': 0.85,
            'reason': 'Random Forest Regressor typically performs well on a variety of datasets and is less prone to overfitting.'
        })
        
        # Check for linear relationships (oversimplified heuristic)
        if num_features < 10 and not has_categorical:
            recommendations.append({
                'algorithm': 'LinearRegression',
                'model_type': 'classical',
                'confidence': 0.75,
                'reason': 'Linear Regression works well for smaller datasets with linear relationships.'
            })
        
        # Add SVR for smaller datasets
        if num_samples < 10000:
            recommendations.append({
                'algorithm': 'SVR',
                'model_type': 'classical',
                'confidence': 0.70,
                'reason': 'Support Vector Regression can handle complex non-linear relationships.'
            })
            
        # Add Decision Tree Regressor
        recommendations.append({
            'algorithm': 'DecisionTreeRegressor',
            'model_type': 'classical',
            'confidence': 0.60,
            'reason': 'Decision Trees provide interpretable results and handle mixed data types well.'
        })
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    return recommendations

def train_model_async(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, hyperparameters, model_folder, time_limit=3600):
    """Start model training in a background thread with a time limit"""
    thread = threading.Thread(
        target=train_model,
        args=(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, hyperparameters, model_folder, time_limit)
    )
    thread.daemon = True
    thread.start()
    
    return thread

def train_model(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, hyperparameters, model_folder, time_limit=3600):
    """Train a machine learning model with time limit"""
    from app import create_app, db
    from app.models.experiment import ModelRun
    
    # Create Flask app context
    app = create_app()
    
    with app.app_context():
        # Start timer
        start_time = time.time()
        
        # Get model run from database
        model_run = ModelRun.query.get(model_run_id)
        
        if not model_run:
            print(f"Model run {model_run_id} not found")
            return
        
        # Update status
        model_run.status = 'training'
        model_run.started_at = datetime.utcnow()
        model_run.logs = "Starting model training...\n"
        db.session.commit()
        
        try:
            # Log preprocessing
            update_model_logs(model_run_id, "Preprocessing dataset...")
            
            # Preprocess data
            from app.utils.data_processing import preprocess_data
            processed_data = preprocess_data(file_path, target_column, feature_columns)
            
            # Split data
            update_model_logs(model_run_id, "Splitting data into training and test sets...")
            X = processed_data['X']
            y = processed_data['y']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Set model class and parameters based on problem type
            if problem_type == 'classification':
                model_info = CLASSIFICATION_MODELS.get(algorithm)
            else:  # regression
                model_info = REGRESSION_MODELS.get(algorithm)
            
            if not model_info:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Merge default parameters with provided hyperparameters
            model_params = model_info['params'].copy()
            if hyperparameters:
                model_params.update(hyperparameters)
            
            # Initialize model
            update_model_logs(model_run_id, f"Initializing {algorithm} model...")
            model = model_info['class'](**model_params)
            
            # Train model with time limit check
            update_model_logs(model_run_id, f"Training {algorithm} model...")
            train_start_time = time.time()
            
            model.fit(X_train, y_train)
            
            # Check if we've exceeded the time limit
            current_time = time.time()
            if current_time - start_time > time_limit:
                update_model_logs(model_run_id, "Training timed out. Maximum allowed time exceeded.")
                model_run.status = 'timed_out'
                db.session.commit()
                return
            
            # Evaluate model
            update_model_logs(model_run_id, "Evaluating model performance...")
            y_pred = model.predict(X_test)
            
            if problem_type == 'classification':
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                }
            else:  # regression
                metrics = {
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'r2': float(r2_score(y_test, y_pred))
                }
            
            # Save model
            update_model_logs(model_run_id, "Saving model...")
            model_filename = f"{model_run_id}.pkl"
            model_path = os.path.join(model_folder, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'encoders': processed_data['encoders'],
                    'scaler': processed_data['scaler'],
                    'metrics': metrics
                }, f)
            
            # Update model run
            update_model_logs(model_run_id, "Model training completed successfully!")
            model_run.status = 'completed'
            model_run.completed_at = datetime.utcnow()
            model_run.metrics = json.dumps(metrics)
            model_run.model_path = model_path
            
            # Generate report path (to be implemented in another utility)
            report_filename = f"{model_run_id}_report.pdf"
            report_path = os.path.join(model_folder, report_filename)
            model_run.report_path = report_path
            
            db.session.commit()
            
        except Exception as e:
            # Log error
            error_msg = f"Error during model training: {str(e)}"
            update_model_logs(model_run_id, error_msg)
            
            # Update model run
            model_run.status = 'failed'
            db.session.commit()

def update_model_logs(model_run_id, log_message):
    """Update the logs for a model run"""
    from app import db
    from app.models.experiment import ModelRun
    
    model_run = ModelRun.query.get(model_run_id)
    if model_run:
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        if model_run.logs:
            model_run.logs += f"\n[{timestamp}] {log_message}"
        else:
            model_run.logs = f"[{timestamp}] {log_message}"
        
        db.session.commit()
        print(f"Model {model_run_id}: {log_message}")
# Add this to app/utils/ml_utils.py

def get_gemini_recommendations(problem_type, data_info):
    """Get model recommendations using Gemini API"""
    try:
        # Try to import the Gemini API client
        from google import genai
        import os
        import re
        import json
        
        # Get API key from environment variable for security
        api_key = os.environ.get('GEMINI_API_KEY')
        
        # If no API key, fall back to mock recommendations
        if not api_key:
            # Log that we're using mock data due to missing API key
            print("Warning: GEMINI_API_KEY not found in environment variables. Using mock recommendations.")
            return get_mock_recommendations(problem_type, data_info)
        
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Prepare the prompt with dataset information
        prompt = f"""
        See you are sitting in an automl platform and you are trying to build a model for a dataset.Since you are being used as an api and I am an app using you in my services you are strictly asked to follow the format.Because if you become conversational and don't give the response in the way we want then the app can collapse. Remember you are not conversing with anyone but just serving the requests sent.
        I have a dataset with the following characteristics:
        - Problem type: {problem_type}
        - Number of rows: {data_info['num_rows']}
        - Number of columns: {data_info['num_cols']}
        - Numeric features: {len(data_info['numeric_columns'])}
        - Categorical features: {len(data_info['categorical_columns'])}
        
        Please recommend the best machine learning algorithms for this dataset. For each algorithm, provide:
        1. Algorithm name (one of: RandomForest, LogisticRegression, SVC, DecisionTree, KNN, XGBoost, LinearRegression, SVR, MLP, CNN, QSVM, QNN)
        2. Model type (one of: classical, neural, quantum)
        3. Confidence score (between 0 and 1)
        4. Reason for recommendation
        5. Whether it should be disabled for guest users (neural and quantum models should be disabled)
        
        Format the response as a JSON array of objects with keys: algorithm, model_type, confidence, reason, disabled, disabled_reason
        
        """
        
        # Generate response from Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Extract the raw text
        raw_text = response.text
        
        # Extract JSON content - handling potential code block wrapping
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, raw_text)
        
        if json_match:
            # Extract JSON content from code block
            json_str = json_match.group(1)
        else:
            # Try to use the raw text directly
            json_str = raw_text
        
        # Parse the JSON response
        recommendations = json.loads(json_str)
        
        # Validate and clean the response
        validated_recommendations = []
        for rec in recommendations:
            # Ensure all required fields are present
            if all(key in rec for key in ['algorithm', 'model_type', 'confidence', 'reason']):
                # Add disabled flag for neural and quantum models if not already present
                if rec['model_type'] == 'neural' and not rec.get('disabled'):
                    rec['disabled'] = True
                    rec['disabled_reason'] = 'Available for logged-in users'
                elif rec['model_type'] == 'quantum' and not rec.get('disabled'):
                    rec['disabled'] = True
                    rec['disabled_reason'] = 'Available for Pro users'
                
                validated_recommendations.append(rec)
        
        # Sort by confidence
        validated_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return validated_recommendations
    
    except Exception as e:
        # Log the error and response for debugging
        print(f"Error using Gemini API: {str(e)}. Falling back to mock recommendations.")
        
        # Fall back to mock recommendations if Gemini API fails
        return get_mock_recommendations(problem_type, data_info)

def get_mock_recommendations(problem_type, data_info):
    """Get model recommendations using Gemini API (simulated for now)"""
    # In a real implementation, this would call the Gemini API
    # For now, we'll return enhanced mock recommendations
    
    recommendations = []
    
    if problem_type == 'classification':
        # More sophisticated recommendations with better explanations
        recommendations = [
            {
                'algorithm': 'RandomForest',
                'model_type': 'classical',
                'confidence': 0.92,
                'reason': 'Random Forest is ideal for your dataset with mixed feature types and moderate dimensionality. It handles non-linear relationships well and provides good performance with default settings.'
            },
            {
                'algorithm': 'XGBoost',
                'model_type': 'classical',
                'confidence': 0.89,
                'reason': 'Gradient boosting techniques like XGBoost often achieve state-of-the-art results on tabular data. It would work well with your numerical features.'
            },
            {
                'algorithm': 'LogisticRegression',
                'model_type': 'classical',
                'confidence': 0.75,
                'reason': 'For interpretability and baseline performance, Logistic Regression provides a solid foundation. Best if your data has mostly linear relationships.'
            },
            {
                'algorithm': 'MLP',
                'model_type': 'neural',
                'confidence': 0.85,
                'reason': 'A Neural Network could capture complex patterns in your data. This model requires more computational resources but often produces excellent results.',
                'disabled': True,
                'disabled_reason': 'Available for logged-in users'
            },
            {
                'algorithm': 'CNN',
                'model_type': 'neural',
                'confidence': 0.70,
                'reason': 'Convolutional Neural Networks excel at spatial data. If your features have spatial relationships, this could be effective.',
                'disabled': True,
                'disabled_reason': 'Available for logged-in users'
            },
            {
                'algorithm': 'QSVM',
                'model_type': 'quantum',
                'confidence': 0.82,
                'reason': 'Quantum Support Vector Machine could provide improved classification boundaries using quantum computing principles.',
                'disabled': True,
                'disabled_reason': 'Available for Pro users'
            }
        ]
    else:  # regression
        recommendations = [
            {
                'algorithm': 'RandomForestRegressor',
                'model_type': 'classical',
                'confidence': 0.90,
                'reason': 'Random Forest Regressor handles non-linear relationships and interactions between features effectively, making it ideal for your dataset.'
            },
            {
                'algorithm': 'XGBoostRegressor',
                'model_type': 'classical',
                'confidence': 0.88,
                'reason': 'XGBoost often provides excellent regression performance through its gradient boosting approach. Particularly good for your numerical features.'
            },
            {
                'algorithm': 'LinearRegression',
                'model_type': 'classical',
                'confidence': 0.70,
                'reason': 'For a simple baseline and maximum interpretability, Linear Regression provides a good starting point. Best for linear relationships.'
            },
            {
                'algorithm': 'MLPRegressor',
                'model_type': 'neural',
                'confidence': 0.83,
                'reason': 'A Neural Network Regressor could model complex non-linear patterns in your data. Good for datasets with intricate feature interactions.',
                'disabled': True,
                'disabled_reason': 'Available for logged-in users'
            },
            {
                'algorithm': 'QRegression',
                'model_type': 'quantum',
                'confidence': 0.78,
                'reason': 'Quantum Regression leverages quantum computing principles for potentially improved predictions on complex relationships.',
                'disabled': True,
                'disabled_reason': 'Available for Pro users'
            }
        ]
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    return recommendations

def tune_hyperparameters_async(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, base_hyperparameters, model_folder, time_limit=3600):
    """Start hyperparameter tuning in a background thread with a time limit"""
    thread = threading.Thread(
        target=tune_hyperparameters,
        args=(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, base_hyperparameters, model_folder, time_limit)
    )
    thread.daemon = True
    thread.start()
    
    return thread

def tune_hyperparameters(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, base_hyperparameters, model_folder, time_limit=3600):
    """Tune hyperparameters for a machine learning model with time limit"""
    from app import create_app, db
    from app.models.experiment import ModelRun
    from sklearn.model_selection import GridSearchCV
    
    # Create Flask app context
    app = create_app('development')
    
    with app.app_context():
        # Start timer
        start_time = time.time()
        
        # Get model run from database
        model_run = ModelRun.query.get(model_run_id)
        
        if not model_run:
            print(f"Model run {model_run_id} not found")
            return
        
        # Update status
        model_run.status = 'tuning'
        model_run.logs = "Starting hyperparameter optimization...\n"
        db.session.commit()
        
        try:
            # Preprocess data
            update_model_logs(model_run_id, "Preprocessing dataset for hyperparameter tuning...")
            from app.utils.data_processing import preprocess_data
            processed_data = preprocess_data(file_path, target_column, feature_columns)
            
            # Split data
            update_model_logs(model_run_id, "Splitting data into training and validation sets...")
            X = processed_data['X']
            y = processed_data['y']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Set model class and parameter grid based on algorithm and problem type
            if problem_type == 'classification':
                model_info = CLASSIFICATION_MODELS.get(algorithm)
                if algorithm == 'RandomForest':
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'criterion': ['gini', 'entropy']
                    }
                elif algorithm == 'XGBoost':
                        param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'max_depth': [3, 6, 9],
                        'min_child_weight': [1, 3, 5],
                        'subsample': [0.7, 0.8, 0.9],
                        'colsample_bytree': [0.7, 0.8, 0.9]
                        }
                elif algorithm == 'LogisticRegression':
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'solver': ['liblinear', 'lbfgs'],
                        'max_iter': [1000]
                    }
                elif algorithm == 'SVC':
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                elif algorithm == 'DecisionTree':
                    param_grid = {
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'criterion': ['gini', 'entropy']
                    }
                elif algorithm == 'KNN':
                    param_grid = {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2]  # Manhattan or Euclidean
                    }
                else:
                    param_grid = {}  # Default empty grid
            else:  # regression
                model_info = REGRESSION_MODELS.get(algorithm)
                if algorithm == 'RandomForestRegressor':
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                elif algorithm == 'LinearRegression':
                    param_grid = {}  # Linear Regression has minimal hyperparameters
                elif algorithm == 'SVR':
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                elif algorithm == 'DecisionTreeRegressor':
                    param_grid = {
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                
                elif algorithm == 'XGBoostRegressor':
                        param_grid = {
                                'n_estimators': [50, 100, 200],
                                'learning_rate': [0.01, 0.1, 0.3],
                                'max_depth': [3, 6, 9],
                                'min_child_weight': [1, 3, 5],
                                'subsample': [0.7, 0.8, 0.9],
                                'colsample_bytree': [0.7, 0.8, 0.9]
                            }
                else:
                    param_grid = {}  # Default empty grid
            
            if not model_info:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Merge default parameters with provided hyperparameters
            model_params = model_info['params'].copy()
            if base_hyperparameters:
                model_params.update(base_hyperparameters)
            
            # Initialize model
            update_model_logs(model_run_id, f"Initializing {algorithm} for hyperparameter tuning...")
            model_class = model_info['class']
            if not model_class:
                raise ValueError(f"Model class not available for {algorithm} in guest mode")
                
            model = model_class(**model_params)
            
            # Check if we have a parameter grid
            if not param_grid:
                update_model_logs(model_run_id, f"No hyperparameters to tune for {algorithm}. Using default values.")
                best_params = model_params
            else:
                # Set up grid search
                update_model_logs(model_run_id, f"Starting grid search with {len(param_grid)} parameters...")
                if problem_type == 'classification':
                    scoring = 'f1_weighted'
                else:
                    scoring = 'neg_mean_squared_error'
                
                grid_search = GridSearchCV(
                    model, 
                    param_grid, 
                    cv=3,  # Use 3-fold CV to save time
                    scoring=scoring,
                    n_jobs=-1  # Use all cores
                )
                
                # Run grid search with time limit check
                update_model_logs(model_run_id, "Testing different hyperparameter combinations...")
                
                # Log parameter sets being tested
                grid_points = 1
                for param, values in param_grid.items():
                    grid_points *= len(values)
                    update_model_logs(model_run_id, f"Testing {param}: {values}")
                
                update_model_logs(model_run_id, f"Total parameter combinations: {grid_points}")
                
                try:
                    grid_search.fit(X_train, y_train)
                    
                    # Check if we've exceeded the time limit
                    current_time = time.time()
                    if current_time - start_time > time_limit:
                        update_model_logs(model_run_id, "Hyperparameter tuning timed out. Using best parameters found so far.")
                        model_run.status = 'timed_out'
                        best_params = grid_search.best_params_
                    else:
                        # Get best parameters
                        best_params = grid_search.best_params_
                        update_model_logs(model_run_id, f"Best parameters found: {best_params}")
                        
                        # Report best score
                        if problem_type == 'classification':
                            update_model_logs(model_run_id, f"Best F1 score: {grid_search.best_score_:.4f}")
                        else:
                            update_model_logs(model_run_id, f"Best neg MSE: {grid_search.best_score_:.4f}")
                except Exception as e:
                    update_model_logs(model_run_id, f"Error during grid search: {str(e)}")
                    update_model_logs(model_run_id, "Falling back to default parameters")
                    best_params = model_params
            
            # Update model run with best hyperparameters
            model_run.hyperparameters = json.dumps(best_params)
            model_run.status = 'created'  # Reset to created so user can start training
            update_model_logs(model_run_id, "Hyperparameter optimization completed. Ready for training!")
            db.session.commit()
            
        except Exception as e:
            # Log error
            error_msg = f"Error during hyperparameter tuning: {str(e)}"
            update_model_logs(model_run_id, error_msg)
            
            # Update model run
            model_run.status = 'failed'
            db.session.commit()