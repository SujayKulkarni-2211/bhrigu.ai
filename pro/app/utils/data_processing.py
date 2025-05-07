# app/utils/data_processing.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import uuid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from flask import current_app
from app.models.experiment import Experiment
from app.utils.file_handling import save_uploaded_file
from flask import flash, redirect, url_for
from app import db

def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Updated functions for data_processing.py

def read_dataset(file_path):
    """Read dataset from CSV or Excel"""
    import os
    import pandas as pd
    from flask import current_app
    
    print(f"Reading file from: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    # Try to resolve file path if it doesn't exist
    if not os.path.exists(file_path):
        base_name = os.path.basename(file_path)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        # Try dataset directory
        alt_path = os.path.join(upload_folder, 'datasets', base_name)
        if os.path.exists(alt_path):
            file_path = alt_path
            print(f"Resolved file path to: {file_path}")
        elif os.path.exists(os.path.join(upload_folder, base_name)):
            # Try upload folder root
            file_path = os.path.join(upload_folder, base_name)
            print(f"Resolved file path to: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find dataset file: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        # Try with different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully read CSV file with {encoding} encoding")
                return df
            except Exception as e:
                print(f"Error reading CSV with {encoding} encoding: {str(e)}")
                continue
                
        # If all encodings fail, try without specifying encoding
        return pd.read_csv(file_path)
    
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def analyze_dataset_stats(file_path):
    """Analyze dataset and return summary statistics"""
    from flask import current_app
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64
    
    print(f"Attempting to analyze dataset from: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    # Read dataset
    df = read_dataset(file_path)
    
    # Basic info
    num_rows, num_cols = df.shape
    
    # Column types
    column_types = {col: str(df[col].dtype) for col in df.columns}
    
    # Missing values
    missing_values = df.isnull().sum().to_dict()
    
    # Descriptive statistics for numerical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Safe conversion for numeric statistics
    def safe_numeric_stats(df, columns):
        if not columns:
            return {}
        
        stats = {}
        try:
            # Convert to regular Python types for JSON serialization
            desc_df = df[columns].describe()
            
            for col in desc_df.columns:
                col_stats = {}
                for stat, value in desc_df[col].items():
                    # Convert numpy types to native Python types
                    if isinstance(value, (np.integer, np.floating)):
                        col_stats[stat] = float(value)
                    else:
                        col_stats[stat] = value
                stats[col] = col_stats
            
            return stats
        except Exception as e:
            print(f"Error calculating numeric statistics: {str(e)}")
            return {}
    
    statistics = safe_numeric_stats(df, numeric_columns)
    
    # Value counts for categorical columns (top 5 values)
    categorical_values = {}
    for col in categorical_columns:
        try:
            # Convert to regular Python types for JSON serialization
            values = df[col].value_counts().head(5).to_dict()
            categorical_values[col] = {str(k): int(v) for k, v in values.items()}
        except Exception as e:
            print(f"Error calculating categorical values for {col}: {str(e)}")
            categorical_values[col] = {"Error": "Could not analyze values"}
    
    # Generate correlation matrix
    correlation_image = None
    if len(numeric_columns) > 1:
        try:
            plt.figure(figsize=(10, 8))
            correlation = df[numeric_columns].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            
            # Save to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            correlation_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close()
        except Exception as e:
            print(f"Error generating correlation matrix: {str(e)}")
    
    # Sample data (first 5 rows)
    sample_data = []
    try:
        # Convert to regular Python types for JSON serialization
        for i, row in df.head(5).iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, (np.integer, np.floating)):
                    val = float(val) if isinstance(val, np.floating) else int(val)
                elif isinstance(val, np.bool_):
                    val = bool(val)
                elif pd.isna(val):
                    val = None
                else:
                    val = str(val)
                row_dict[col] = val
            sample_data.append(row_dict)
    except Exception as e:
        print(f"Error converting sample data: {str(e)}")
    
    # Return dataset info
    dataset_info = {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'columns': df.columns.tolist(),
        'column_types': column_types,
        'missing_values': missing_values,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'statistics': statistics,
        'categorical_values': categorical_values,
        'correlation_image': correlation_image,
        'sample_data': sample_data
    }
    
    return dataset_info

# Fix for the preprocess_data function's scaling code to avoid the pandas warning

def preprocess_data(file_path, target_column, feature_columns=None):
    """Preprocess data for model training"""
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from flask import current_app
    
    # Make sure file exists
    if not os.path.exists(file_path):
        # Try to resolve the file path
        base_name = os.path.basename(file_path)
        
        # Try upload folder/datasets path
        alt_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'datasets', base_name)
        if os.path.exists(alt_path):
            file_path = alt_path
            print(f"Resolved file path to: {file_path}")
        else:
            # Try upload folder directly
            alt_path = os.path.join(current_app.config['UPLOAD_FOLDER'], base_name)
            if os.path.exists(alt_path):
                file_path = alt_path
                print(f"Resolved file path to: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Read dataset
    df = read_dataset(file_path)
    
    # Convert object columns to string to handle potential mixed types
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    
    # Select features and target
    if feature_columns:
        # Make sure all feature columns exist in the dataframe
        valid_features = [col for col in feature_columns if col in df.columns]
        if len(valid_features) != len(feature_columns):
            missing = set(feature_columns) - set(valid_features)
            print(f"Warning: Some feature columns were not found in the dataset: {missing}")
        
        if not valid_features:
            raise ValueError("No valid feature columns found in the dataset")
            
        X = df[valid_features].copy()  # Use .copy() to avoid SettingWithCopyWarning
    else:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        X = df.drop(columns=[target_column]).copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
        
    y = df[target_column].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Handle missing values
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    
    # Impute missing values
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        try:
            X[col] = encoder.fit_transform(X[col])
            encoders[col] = encoder
        except Exception as e:
            print(f"Error encoding column {col}: {str(e)}")
            # Use a simpler approach for problematic columns
            X[col] = pd.factorize(X[col])[0]
    
    # Scale numerical features - fixing the SettingWithCopyWarning
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        # Fix by operating on the entire dataframe at once or properly using loc
        X_numeric = X[numerical_cols].values
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        
        # Update the scaled values in the dataframe
        for i, col in enumerate(numerical_cols):
            X.loc[:, col] = X_numeric_scaled[:, i]
    
    processed_data = {
        'X': X,
        'y': y,
        'original_X': df.drop(columns=[target_column]),
        'original_y': df[target_column],
        'encoders': encoders,
        'scaler': scaler
    }
    
    return processed_data

