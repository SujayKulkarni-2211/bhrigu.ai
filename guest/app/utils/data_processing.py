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

def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder, allowed_extensions):
    """Save an uploaded file with a unique name"""
    if file and allowed_file(file.filename, allowed_extensions):
        # Create a secure filename
        original_filename = secure_filename(file.filename)
        # Create a unique filename
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        # Save the file
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        return file_path, unique_filename, original_filename
    
    return None, None, None

def read_dataset(file_path):
    """Read dataset from CSV or Excel"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return df

def analyze_dataset(file_path):
    """Analyze dataset and return summary statistics"""
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
    
    statistics = {}
    if numeric_columns:
        statistics = df[numeric_columns].describe().to_dict()
    
    # Value counts for categorical columns (top 5 values)
    categorical_values = {}
    for col in categorical_columns:
        categorical_values[col] = df[col].value_counts().head(5).to_dict()
    
    # Generate correlation matrix
    correlation_image = None
    if len(numeric_columns) > 1:
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
    
    # Sample data (first 5 rows)
    sample_data = df.head(5).to_dict(orient='records')
    
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

def preprocess_data(file_path, target_column, feature_columns=None):
    """Preprocess data for model training"""
    df = read_dataset(file_path)
    
    # Select features and target
    if feature_columns:
        X = df[feature_columns]
    else:
        X = df.drop(columns=[target_column])
    
    y = df[target_column]
    
    # Handle missing values
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    
    # Impute missing values
    if numerical_cols.size > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    if categorical_cols.size > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder
    
    # Scale numerical features
    scaler = StandardScaler()
    for col in numerical_cols:
        X[col] = scaler.fit_transform(X[[col]])
    
    processed_data = {
        'X': X,
        'y': y,
        'original_X': df.drop(columns=[target_column]),
        'original_y': df[target_column],
        'encoders': encoders,
        'scaler': scaler
    }
    
    return processed_data

