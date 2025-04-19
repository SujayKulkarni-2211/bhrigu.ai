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
        1. Algorithm name (one of: RandomForest, LogisticRegression, SVC, DecisionTree, KNN, LinearRegression, SVR, MLP, CNN, QSVM, QNN)
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

# Add these imports to app/utils/ml_utils.py
from app.utils.ml_report_utils import generate_tuning_report, extract_feature_importances
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Update the tune_hyperparameters function in app/utils/ml_utils.py
def tune_hyperparameters(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, base_hyperparameters, model_folder, time_limit=3600):
    """Tune hyperparameters for a machine learning model with time limit and save detailed results"""
    from app import create_app, db
    from app.models.experiment import ModelRun
    
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
            
            # Define parameter grids based on algorithm and problem type
            param_grid, param_dist = get_parameter_space(algorithm, problem_type)
            
            if not param_grid and not param_dist:
                update_model_logs(model_run_id, f"No tunable parameters for {algorithm}. Using default values.")
                best_params = base_hyperparameters if base_hyperparameters else {}
                tuning_results = []
                score = 0
            else:
                # Set model class based on algorithm and problem type
                if problem_type == 'classification':
                    model_info = CLASSIFICATION_MODELS.get(algorithm)
                else:  # regression
                    model_info = REGRESSION_MODELS.get(algorithm)
                
                if not model_info:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                # Create base model with default or provided params
                model_params = model_info['params'].copy()
                if base_hyperparameters:
                    model_params.update(base_hyperparameters)
                
                model_class = model_info['class']
                
                # Set scoring metric
                if problem_type == 'classification':
                    scoring = 'f1_weighted'
                else:
                    scoring = 'neg_mean_squared_error'
                
                update_model_logs(model_run_id, f"Starting hyperparameter search for {algorithm}...")
                
                # First do a broad randomized search
                update_model_logs(model_run_id, "Phase 1: Broad randomized search...")
                random_search = RandomizedSearchCV(
                    model_class(**model_params),
                    param_distributions=param_dist,
                    n_iter=20,  # Try 20 random combinations
                    cv=3,
                    scoring=scoring,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Run randomized search with time monitoring
                search_start_time = time.time()
                random_search.fit(X_train, y_train)
                
                # Check for timeout
                current_time = time.time()
                if current_time - start_time > time_limit * 0.7:  # Use 70% of time limit
                    update_model_logs(model_run_id, "Time limit approaching. Using best parameters from randomized search.")
                    best_params = random_search.best_params_
                    cv_results = random_search.cv_results_
                    
                    # Format results for storage
                    tuning_results = []
                    for i in range(len(cv_results['params'])):
                        result = cv_results['params'][i].copy()
                        result['score'] = cv_results['mean_test_score'][i]
                        tuning_results.append(result)
                else:
                    # Now do a focused grid search around the best parameters
                    update_model_logs(model_run_id, "Phase 2: Focused grid search around best parameters...")
                    
                    # Create a focused parameter grid based on best randomized search results
                    focused_param_grid = create_focused_param_grid(random_search.best_params_, algorithm)
                    
                    grid_search = GridSearchCV(
                        model_class(**{k: v for k, v in model_params.items() if k not in focused_param_grid}),
                        param_grid=focused_param_grid,
                        cv=5,  # More folds for more accurate results
                        scoring=scoring,
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_params = grid_search.best_params_
                    
                    # Combine results from both searches
                    random_results = []
                    for i in range(len(random_search.cv_results_['params'])):
                        result = random_search.cv_results_['params'][i].copy()
                        result['score'] = random_search.cv_results_['mean_test_score'][i]
                        result['search_phase'] = 'random'
                        random_results.append(result)
                    
                    grid_results = []
                    for i in range(len(grid_search.cv_results_['params'])):
                        result = grid_search.cv_results_['params'][i].copy()
                        result['score'] = grid_search.cv_results_['mean_test_score'][i]
                        result['search_phase'] = 'grid'
                        grid_results.append(result)
                    
                    tuning_results = random_results + grid_results
                
                # Log results
                update_model_logs(model_run_id, f"Best parameters found: {best_params}")
                
                # Train a final model with the best parameters and evaluate on validation set
                update_model_logs(model_run_id, "Training model with best parameters for evaluation...")
                best_model = model_class(**{**model_params, **best_params})
                best_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = best_model.predict(X_val)
                
                # Calculate metrics
                if problem_type == 'classification':
                    metrics = {
                        'accuracy': float(accuracy_score(y_val, y_pred)),
                        'precision': float(precision_score(y_val, y_pred, average='weighted', zero_division=0)),
                        'recall': float(recall_score(y_val, y_pred, average='weighted', zero_division=0)),
                        'f1': float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
                    }
                else:  # regression
                    metrics = {
                        'mse': float(mean_squared_error(y_val, y_pred)),
                        'mae': float(mean_absolute_error(y_val, y_pred)),
                        'r2': float(r2_score(y_val, y_pred))
                    }
                
                update_model_logs(model_run_id, f"Validation metrics with best parameters: {metrics}")
                
                # Store metrics
                model_run.tuning_metrics = json.dumps(metrics)
                
                # Generate tuning report
                update_model_logs(model_run_id, "Generating tuning report...")
                
                # Create reports directory if it doesn't exist
                report_folder = os.path.join(os.path.dirname(model_folder), 'reports')
                os.makedirs(report_folder, exist_ok=True)
                
                # Get experiment name
                from app.models.experiment import GuestExperiment
                experiment = GuestExperiment.query.get(model_run.experiment_id)
                experiment_name = experiment.dataset_original_name if experiment else None
                
                # Generate report
                report_path = generate_tuning_report(
                    model_run_id,
                    algorithm,
                    problem_type,
                    tuning_results,
                    best_params,
                    metrics,
                    report_folder,
                    experiment_name=experiment_name
                )
                
                # Store report path
                model_run.tuning_report_path = report_path
            
            # Update model run with best hyperparameters and tuning results
            model_run.hyperparameters = json.dumps(best_params)
            model_run.tuning_results = json.dumps(tuning_results)
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

def get_parameter_space(algorithm, problem_type):
    """Define parameter spaces for hyperparameter tuning"""
    param_grid = {}
    param_dist = {}
    
    if algorithm == 'RandomForest' or algorithm == 'RandomForestRegressor':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy'] if algorithm == 'RandomForest' else ['squared_error', 'absolute_error']
        }
        
        param_dist = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
        if algorithm == 'RandomForest':
            param_dist['criterion'] = ['gini', 'entropy']
        else:
            param_dist['criterion'] = ['squared_error', 'absolute_error', 'poisson']
            
    elif algorithm == 'LogisticRegression':
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'max_iter': [1000]
        }
        
        param_dist = {
            'C': uniform(0.01, 100),
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'max_iter': [1000, 2000, 5000]
        }
        
    elif algorithm == 'LinearRegression':
        param_grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
        
        param_dist = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
        
    elif algorithm == 'SVC' or algorithm == 'SVR':
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        
        param_dist = {
            'C': uniform(0.01, 50),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': randint(2, 5)  # For poly kernel
        }
        
    elif algorithm == 'DecisionTree' or algorithm == 'DecisionTreeRegressor':
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        if algorithm == 'DecisionTree':
            param_grid['criterion'] = ['gini', 'entropy']
            param_dist = {
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'criterion': ['gini', 'entropy']
            }
        else:
            param_grid['criterion'] = ['squared_error', 'absolute_error', 'friedman_mse']
            param_dist = {
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'criterion': ['squared_error', 'absolute_error', 'friedman_mse']
            }
            
    elif algorithm == 'KNN':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        
        param_dist = {
            'n_neighbors': randint(3, 20),
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    
    return param_grid, param_dist

def create_focused_param_grid(best_params, algorithm):
    """Create a focused parameter grid around the best parameters from randomized search"""
    focused_grid = {}
    
    # Depending on the algorithm, create appropriate focused ranges
    if algorithm == 'RandomForest' or algorithm == 'RandomForestRegressor':
        if 'n_estimators' in best_params:
            n_est = best_params['n_estimators']
            focused_grid['n_estimators'] = [max(10, n_est - 50), n_est, min(500, n_est + 50)]
        
        if 'max_depth' in best_params and best_params['max_depth'] is not None:
            max_d = best_params['max_depth']
            focused_grid['max_depth'] = [max(3, max_d - 5), max_d, min(100, max_d + 5)]
        
        if 'min_samples_split' in best_params:
            min_split = best_params['min_samples_split']
            focused_grid['min_samples_split'] = [max(2, min_split - 2), min_split, min_split + 2]
            
    elif algorithm == 'LogisticRegression':
        if 'C' in best_params:
            c_val = best_params['C']
            focused_grid['C'] = [c_val / 3, c_val, c_val * 3]
    elif algorithm == 'SVC' or algorithm == 'SVR':
        if 'C' in best_params:
            c_val = best_params['C']
            focused_grid['C'] = [c_val / 3, c_val, c_val * 3]
        
        if 'gamma' in best_params and best_params['gamma'] not in ['scale', 'auto']:
            g_val = best_params['gamma']
            focused_grid['gamma'] = [g_val / 3, g_val, g_val * 3]
            
    elif algorithm == 'DecisionTree' or algorithm == 'DecisionTreeRegressor':
        if 'max_depth' in best_params and best_params['max_depth'] is not None:
            max_d = best_params['max_depth']
            focused_grid['max_depth'] = [max(3, max_d - 3), max_d, min(50, max_d + 3)]
            
        if 'min_samples_split' in best_params:
            min_split = best_params['min_samples_split']
            focused_grid['min_samples_split'] = [max(2, min_split - 1), min_split, min_split + 1]
            
    elif algorithm == 'KNN':
        if 'n_neighbors' in best_params:
            n = best_params['n_neighbors']
            focused_grid['n_neighbors'] = [max(1, n - 2), n, n + 2]
    
    # For any parameter with categorical values, just use the best value found
    categorical_params = ['criterion', 'kernel', 'weights', 'solver', 'algorithm']
    for param in categorical_params:
        if param in best_params:
            focused_grid[param] = [best_params[param]]
    
    return focused_grid

# Update the tune_hyperparameters_async function
def tune_hyperparameters_async(model_run_id, algorithm, problem_type, file_path, target_column, 
                              feature_columns, base_hyperparameters, model_folder, time_limit=3600):
    """Start hyperparameter tuning in a background thread with a time limit"""
    thread = threading.Thread(
        target=tune_hyperparameters,
        args=(model_run_id, algorithm, problem_type, file_path, target_column, 
              feature_columns, base_hyperparameters, model_folder, time_limit)
    )
    thread.daemon = True
    thread.start()
    
    return thread

def train_full_model_async(model_run_id, algorithm, problem_type, file_path, target_column, 
                          feature_columns, hyperparameters, model_folder, time_limit=3600):
    """Start full model training in a background thread with a time limit"""
    thread = threading.Thread(
        target=_train_model_impl,  # Using a separate implementation function
        args=(model_run_id, algorithm, problem_type, file_path, target_column, 
              feature_columns, hyperparameters, model_folder, time_limit)
    )
    thread.daemon = True
    thread.start()
    
    return thread

def _train_model_impl(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, hyperparameters, model_folder, time_limit=3600):
    """Implementation of model training with time limit and report generation"""
    from app import create_app, db
    from app.models.experiment import ModelRun
    from app.utils.ml_report_utils import generate_model_report, generate_comprehensive_report
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
                
                # Generate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                model_run.confusion_matrix = json.dumps(cm.tolist())
                
                # Generate classification report
                cr = classification_report(y_test, y_pred)
                model_run.classification_report = cr
            else:  # regression
                metrics = {
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'r2': float(r2_score(y_test, y_pred))
                }
            
            # Extract feature importances if available
            feature_importances = extract_feature_importances(model, feature_columns)
            if feature_importances:
                model_run.feature_importances = json.dumps(feature_importances)
            
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
            
            # Generate report
            update_model_logs(model_run_id, "Generating model report...")
            report_folder = os.path.join(os.path.dirname(model_folder), 'reports')
            os.makedirs(report_folder, exist_ok=True)
            
            # Get experiment for experiment name
            from app.models.experiment import GuestExperiment
            experiment = GuestExperiment.query.get(model_run.experiment_id)
            experiment_name = experiment.dataset_original_name if experiment else None
            
            # Get tuning results if this model had hyperparameter tuning
            tuning_results = json.loads(model_run.tuning_results) if hasattr(model_run, 'tuning_results') and model_run.tuning_results else None
            
            if tuning_results:
                # Generate comprehensive report including tuning info
                report_path = generate_comprehensive_report(
                    model_run_id,
                    algorithm,
                    problem_type,
                    metrics,
                    feature_importances,
                    cm.tolist() if problem_type == 'classification' else None,
                    cr if problem_type == 'classification' else None,
                    tuning_results,
                    model_params,
                    report_folder,
                    experiment_name=experiment_name
                )
                model_run.comprehensive_report_path = report_path
            else:
                # Generate standard model report
                report_path = generate_model_report(
                    model_run_id,
                    algorithm,
                    problem_type,
                    metrics,
                    feature_importances,
                    cm.tolist() if problem_type == 'classification' else None,
                    cr if problem_type == 'classification' else None,
                    report_folder,
                    experiment_name=experiment_name
                )
            
            # Update model run
            update_model_logs(model_run_id, "Model training completed successfully!")
            model_run.status = 'completed'
            model_run.completed_at = datetime.utcnow()
            model_run.metrics = json.dumps(metrics)
            model_run.model_path = model_path
            model_run.report_path = report_path
            
            db.session.commit()
            
        except Exception as e:
            # Log error
            error_msg = f"Error during model training: {str(e)}"
            update_model_logs(model_run_id, error_msg)
            
            # Update model run
            model_run.status = 'failed'
            db.session.commit()
