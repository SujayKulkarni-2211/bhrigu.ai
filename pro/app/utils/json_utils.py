# pro/app/utils/json_utils.py
import json
import numpy as np
from datetime import datetime, date
from decimal import Decimal

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types like numpy arrays, dates, etc."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            # Try to convert objects to dictionaries
            try:
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            except:
                return str(obj)
        return super().default(obj)

def safe_dumps(obj, **kwargs):
    """
    Convert object to JSON string safely
    
    Args:
        obj: Object to convert to JSON
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string representation of object
    """
    return json.dumps(obj, cls=JSONEncoder, **kwargs)

def safe_loads(json_string, default=None):
    """
    Parse JSON string safely
    
    Args:
        json_string: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    if not json_string:
        return default
        
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing JSON: {str(e)}")
        return default

def serialize_model_metrics(metrics):
    """
    Safely serialize machine learning model metrics
    
    Args:
        metrics: Dictionary of model metrics
        
    Returns:
        JSON string of serialized metrics
    """
    if not metrics:
        return json.dumps({})
        
    # Clean metrics dictionary by removing non-serializable values
    clean_metrics = {}
    for key, value in metrics.items():
        # Skip functions, modules, or complex objects
        if callable(value) or str(type(value)).startswith("<class 'module'") or key.startswith('_'):
            continue
            
        # Convert numpy types and other special types
        if isinstance(value, np.integer):
            clean_metrics[key] = int(value)
        elif isinstance(value, np.floating):
            clean_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            clean_metrics[key] = value.tolist()
        elif isinstance(value, (datetime, date)):
            clean_metrics[key] = value.isoformat()
        elif isinstance(value, Decimal):
            clean_metrics[key] = float(value)
        else:
            # Try to keep the value as is, the JSONEncoder will handle it
            clean_metrics[key] = value
            
    return safe_dumps(clean_metrics)

def parse_model_run_data(model_run):
    """
    Parse model run data safely for templates
    
    Args:
        model_run: ProModelRun object
        
    Returns:
        Dictionary with safely parsed model run data
    """
    model_data = {
        'id': model_run.id,
        'algorithm': model_run.algorithm,
        'model_type': model_run.model_type,
        'status': model_run.status,
        'logs': model_run.logs,
        'created_at': model_run.created_at,
        'started_at': model_run.started_at,
        'completed_at': model_run.completed_at,
        'model_path': model_run.model_path,
        'report_path': model_run.report_path,
        'hyperparameters_tuning': model_run.hyperparameters_tuning
    }
    
    # Parse hyperparameters
    model_data['hyperparameters'] = safe_loads(model_run.hyperparameters, {})
    
    # Parse metrics
    model_data['metrics'] = safe_loads(model_run.metrics, {})
    
    # Parse feature importances
    model_data['feature_importances'] = safe_loads(model_run.feature_importances, {})
    
    # Parse confusion matrix
    model_data['confusion_matrix'] = safe_loads(model_run.confusion_matrix, None)
    
    # Include classification report as-is
    model_data['classification_report'] = model_run.classification_report
    
    # Parse tuning metrics
    model_data['tuning_metrics'] = safe_loads(model_run.tuning_metrics, {})
    
    # Parse tuning results
    model_data['tuning_results'] = safe_loads(model_run.tuning_results, [])
    
    return model_data