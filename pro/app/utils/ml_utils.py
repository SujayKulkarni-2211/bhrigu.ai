# pro/app/utils/ml_utils.py
import os
import json
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from threading import Thread

# Import machine learning algorithms
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# Try to import neural network libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("Warning: TensorFlow not installed. Neural network models will not be available.")

# Try to import quantum computing libraries
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: PennyLane not installed. Quantum models will not be available.")

# Import ML reporting utilities
from app.utils.ml_report_utils import (
    generate_model_report, generate_tuning_report, create_model_card,
    export_model_predictions, visualize_training_history
)


def debug_data_types(y_test, y_pred):
    """Debug function to print type information"""
    print(f"y_test shape: {y_test.shape}, type: {type(y_test)}, dtype: {y_test.dtype if hasattr(y_test, 'dtype') else 'N/A'}")
    print(f"y_pred shape: {y_pred.shape}, type: {type(y_pred)}, dtype: {y_pred.dtype if hasattr(y_pred, 'dtype') else 'N/A'}")
    
    if hasattr(y_pred, 'ndim') and y_pred.ndim == 2:
        print(f"y_pred is 2D with shape {y_pred.shape}")
        print(f"First few values: {y_pred[:5]}")
        
class MLModel:
    """Base class for machine learning models"""
    
    def __init__(self, model_type='classical', algorithm=None, problem_type=None, 
                params=None, is_pro=False, experiment_id=None, model_run_id=None):
        """
        Initialize a machine learning model
        
        Parameters
        ----------
        model_type : str
            Type of model ('classical', 'neural', 'quantum')
        algorithm : str
            Name of the algorithm
        problem_type : str
            Type of problem ('classification', 'regression')
        params : dict
            Hyperparameters for the model
        is_pro : bool
            Whether user has pro subscription
        experiment_id : str
            ID of the experiment
        model_run_id : str
            ID of the model run
        """
        self.model_type = model_type
        self.algorithm = algorithm
        self.problem_type = problem_type
        self.params = params or {}
        self.is_pro = is_pro
        self.experiment_id = experiment_id
        self.model_run_id = model_run_id
        
        self.model = None
        self.feature_importances = None
        self.metrics = {}
        self.history = None
        self.confusion_matrix_data = None
        self.classification_report_data = None
        self.scaler = None
        self.label_encoder = None
        self.best_params = None
        self.tuning_results = None
        self.quantum_circuit = None
        
        # Initialize model based on algorithm
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on the algorithm"""
        # Classical ML models
        if self.model_type == 'classical':
            if self.problem_type == 'classification':
                self._initialize_classification_model()
            elif self.problem_type == 'regression':
                self._initialize_regression_model()
            else:
                raise ValueError(f"Unsupported problem type: {self.problem_type}")
        
        # Neural network models
        elif self.model_type == 'neural':
            if not NEURAL_AVAILABLE:
                raise ImportError("TensorFlow is required for neural network models")
            
            self._initialize_neural_model()
        
        # Quantum models
        elif self.model_type == 'quantum':
            if not QUANTUM_AVAILABLE:
                raise ImportError("PennyLane is required for quantum models")
            
            if not self.is_pro:
                raise PermissionError("Quantum models are only available with Pro subscription")
            
            self._initialize_quantum_model()
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _initialize_classification_model(self):
        """Initialize a classification model"""
        if self.algorithm == 'LogisticRegression':
            self.model = LogisticRegression(**self.params)
        
        elif self.algorithm == 'RandomForest':
            self.model = RandomForestClassifier(**self.params)
        
        elif self.algorithm == 'GradientBoosting':
            self.model = GradientBoostingClassifier(**self.params)
        
        elif self.algorithm == 'SVM':
            self.model = SVC(**self.params, probability=True)
        
        elif self.algorithm == 'KNN':
            self.model = KNeighborsClassifier(**self.params)
        
        elif self.algorithm == 'DecisionTree':
            self.model = DecisionTreeClassifier(**self.params)
        
        elif self.algorithm == 'NaiveBayes':
            self.model = GaussianNB(**self.params)
        
        else:
            raise ValueError(f"Unsupported classification algorithm: {self.algorithm}")
    
    def _initialize_regression_model(self):
        """Initialize a regression model"""
        if self.algorithm == 'LinearRegression':
            self.model = LinearRegression(**self.params)
        
        elif self.algorithm == 'Ridge':
            self.model = Ridge(**self.params)
        
        elif self.algorithm == 'Lasso':
            self.model = Lasso(**self.params)
        
        elif self.algorithm == 'RandomForest':
            self.model = RandomForestRegressor(**self.params)
        
        elif self.algorithm == 'GradientBoosting':
            self.model = GradientBoostingRegressor(**self.params)
        
        elif self.algorithm == 'SVR':
            self.model = SVR(**self.params)
        
        elif self.algorithm == 'KNN':
            self.model = KNeighborsRegressor(**self.params)
        
        elif self.algorithm == 'DecisionTree':
            self.model = DecisionTreeRegressor(**self.params)
        
        else:
            raise ValueError(f"Unsupported regression algorithm: {self.algorithm}")
    
    def _initialize_neural_model(self):
        """Initialize a neural network model"""
        # Initialize as None; will be built during training with input shape
        self.model = None
        self.model_config = self.params.copy()
        
        # Default configs if not provided
        self.model_config.setdefault('layers', [64, 32])
        self.model_config.setdefault('activation', 'relu')
        self.model_config.setdefault('dropout', 0.2)
        self.model_config.setdefault('learning_rate', 0.001)
        self.model_config.setdefault('epochs', 100)
        self.model_config.setdefault('batch_size', 32)
        self.model_config.setdefault('early_stopping', True)
        self.model_config.setdefault('patience', 10)
    
    def _initialize_quantum_model(self):
        """Initialize a quantum machine learning model"""
        # Initialize as None; will be built during training with feature dimensions
        self.model = None
        self.quantum_config = self.params.copy()
        
        # Default configs if not provided
        self.quantum_config.setdefault('qubits', 4)
        self.quantum_config.setdefault('layers', 2)
        
        if self.algorithm in ['QSVM', 'QKE']:
            self.quantum_config.setdefault('feature_map', 'ZZFeatureMap')
        elif self.algorithm in ['QNN', 'QVQE', 'QRegression']:
            self.quantum_config.setdefault('ansatz', 'StronglyEntanglingLayers')
            self.quantum_config.setdefault('optimizer', 'Adam')
            self.quantum_config.setdefault('learning_rate', 0.01)
            self.quantum_config.setdefault('epochs', 50)
    
    # Replace the _build_neural_model method in the MLModel class in ml_utils.py

    def _build_neural_model(self, input_shape, num_classes=None):
        """
        Build a neural network model with specified input shape
        
        Parameters
        ----------
        input_shape : tuple
            Shape of the input data
        num_classes : int
            Number of classes for classification
        """
        import tensorflow as tf
        
        print(f"Building neural model: {self.algorithm}, Input shape: {input_shape}, Num classes: {num_classes}")
        
        if self.algorithm == 'MLP' or self.algorithm == 'FNN':
            # Create a simple feedforward neural network
            model = tf.keras.Sequential()
            
            # Input layer
            model.add(tf.keras.layers.Dense(self.model_config['layers'][0], activation=self.model_config['activation'], 
                        input_shape=input_shape))
            model.add(tf.keras.layers.Dropout(self.model_config['dropout']))
            
            # Hidden layers
            for units in self.model_config['layers'][1:]:
                model.add(tf.keras.layers.Dense(units, activation=self.model_config['activation']))
                model.add(tf.keras.layers.Dropout(self.model_config['dropout']))
            
            # Output layer
            if self.problem_type == 'classification':
                if num_classes is None:
                    num_classes = 2  # Default to binary classification
                    
                if num_classes == 2:
                    # Binary classification
                    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
                else:
                    # Multi-class classification
                    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
                    loss = 'categorical_crossentropy'
                    metrics = ['accuracy']
                
                print(f"Created classification model with {num_classes} classes")
            else:  # regression
                model.add(tf.keras.layers.Dense(1, activation='linear'))
                loss = 'mean_squared_error'
                metrics = ['mean_squared_error', 'mean_absolute_error']
                
                print("Created regression model")
            
            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate']),
                loss=loss,
                metrics=metrics
            )
            
            # Save the model
            self.model = model
            
            # Print model summary for debugging
            print(f"Model summary:")
            model.summary()
            
        # (Rest of the method for other model types can remain the same)
    
    def _build_quantum_model(self, n_features):
        """
        Build a quantum machine learning model
        
        Parameters
        ----------
        n_features : int
            Number of features
        """
        n_qubits = self.quantum_config.get('qubits', 4)
        n_qubits = min(n_qubits, n_features)  # Can't use more qubits than features
        
        if self.algorithm == 'QSVM':
            # Use PennyLane's QSVM implementation with kernel
            dev = qml.device("default.qubit", wires=n_qubits)
            
            # Define the quantum kernel function
            @qml.qnode(dev)
            def kernel(x1, x2):
                # Encode the data
                for i in range(n_qubits):
                    qml.RX(x1[i], wires=i)
                
                # Apply feature map (e.g., ZZFeatureMap)
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(x1[i] * x1[j], wires=j)
                        qml.CNOT(wires=[i, j])
                
                # Apply inverse feature map on x2
                for i in range(n_qubits-1, -1, -1):
                    for j in range(n_qubits-1, i, -1):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(-x2[i] * x2[j], wires=j)
                        qml.CNOT(wires=[i, j])
                
                for i in range(n_qubits):
                    qml.RX(-x2[i], wires=i)
                
                # Return expectation value
                return qml.expval(qml.PauliZ(0))
            
            # Wrapper for scikit-learn compatibility
            def quantum_kernel(X1, X2):
                K = np.zeros((len(X1), len(X2)))
                for i, x1 in enumerate(X1):
                    for j, x2 in enumerate(X2):
                        K[i, j] = kernel(x1[:n_qubits], x2[:n_qubits])
                return K
            
            # Create a custom kernel SVM
            from sklearn.svm import SVC
            self.model = SVC(kernel=quantum_kernel)
            
            # Store quantum circuit config for visualization
            self.quantum_circuit = {
                'qubits': n_qubits,
                'feature_map': self.quantum_config.get('feature_map', 'ZZFeatureMap'),
                'circuit_type': 'feature_map'
            }
        
        elif self.algorithm == 'QKE':
            # Quantum Kernel Estimator
            dev = qml.device("default.qubit", wires=n_qubits)
            
            # Define the quantum kernel function
            @qml.qnode(dev)
            def kernel(x1, x2):
                # Encode the data
                for i in range(n_qubits):
                    qml.RX(x1[i], wires=i)
                
                # Apply feature map
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(x1[i] * x1[j], wires=j)
                        qml.CNOT(wires=[i, j])
                
                # Apply inverse feature map on x2
                for i in range(n_qubits-1, -1, -1):
                    for j in range(n_qubits-1, i, -1):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(-x2[i] * x2[j], wires=j)
                        qml.CNOT(wires=[i, j])
                
                for i in range(n_qubits):
                    qml.RX(-x2[i], wires=i)
                
                # Return expectation value
                return qml.expval(qml.PauliZ(0))
            
            # Wrapper for scikit-learn compatibility
            def quantum_kernel(X1, X2):
                K = np.zeros((len(X1), len(X2)))
                for i, x1 in enumerate(X1):
                    for j, x2 in enumerate(X2):
                        K[i, j] = kernel(x1[:n_qubits], x2[:n_qubits])
                return K
            
            # Create a custom kernel estimator based on problem type
            if self.problem_type == 'classification':
                from sklearn.svm import SVC
                self.model = SVC(kernel=quantum_kernel)
            else:  # regression
                from sklearn.svm import SVR
                self.model = SVR(kernel=quantum_kernel)
            
            # Store quantum circuit config for visualization
            self.quantum_circuit = {
                'qubits': n_qubits,
                'feature_map': self.quantum_config.get('feature_map', 'ZZFeatureMap'),
                'circuit_type': 'feature_map'
            }
        
        elif self.algorithm in ['QNN', 'QRegression']:
            # Quantum Neural Network
            n_layers = self.quantum_config.get('layers', 2)
            
            dev = qml.device("default.qubit", wires=n_qubits)
            
            # Define the QNN circuit
            @qml.qnode(dev)
            def qnode(inputs, weights):
                # Encode inputs
                for i in range(n_qubits):
                    qml.RX(inputs[i], wires=i)
                
                # Parameterized quantum circuit
                for layer in range(n_layers):
                    for qubit in range(n_qubits):
                        qml.Rot(*weights[layer, qubit], wires=qubit)
                    
                    # Entanglement
                    for qubit in range(n_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])
                
                # Return expectation value
                return qml.expval(qml.PauliZ(0))
            
            # Initialize weights
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            
            # Create Quantum model with classical optimizer
            from pennylane.optimize import AdamOptimizer
            
            class QuantumModel:
                def __init__(self, qnode, weight_shapes, learning_rate=0.01):
                    self.qnode = qnode
                    self.weight_shapes = weight_shapes
                    self.weights = np.random.uniform(0, 2*np.pi, size=qml.templates.StronglyEntanglingLayers.shape(n_layers, n_qubits))
                    self.optimizer = AdamOptimizer(learning_rate)
                
                def fit(self, X, y, epochs=50):
                    X_resized = self._resize_features(X)
                    for epoch in range(epochs):
                        # Update weights
                        self.weights = self.optimizer.step(
                            lambda weights: self._loss(weights, X_resized, y),
                            self.weights
                        )
                    return self
                
                def predict(self, X):
                    X_resized = self._resize_features(X)
                    predictions = np.array([self._predict_single(x) for x in X_resized])
                    return predictions
                
                def _predict_single(self, x):
                    # Scale prediction between 0 and 1 for classification
                    return (self.qnode(x, self.weights) + 1) / 2
                
                def _loss(self, weights, X, y):
                    predictions = np.array([self.qnode(x, weights) for x in X])
                    # Scale predictions between 0 and 1
                    predictions = (predictions + 1) / 2
                    # MSE loss
                    loss = np.mean((predictions - y) ** 2)
                    return loss
                
                def _resize_features(self, X):
                    # Resize features to match number of qubits
                    if X.shape[1] > n_qubits:
                        return X[:, :n_qubits]
                    elif X.shape[1] < n_qubits:
                        padded = np.zeros((X.shape[0], n_qubits))
                        padded[:, :X.shape[1]] = X
                        return padded
                    return X
            
            # Create the model
            self.model = QuantumModel(
                qnode, 
                weight_shapes,
                learning_rate=self.quantum_config.get('learning_rate', 0.01)
            )
            
            # Store quantum circuit config for visualization
            self.quantum_circuit = {
                'qubits': n_qubits,
                'layers': n_layers,
                'ansatz': self.quantum_config.get('ansatz', 'StronglyEntanglingLayers'),
                'circuit_type': 'variational'
            }
        
        else:
            raise ValueError(f"Unsupported quantum algorithm: {self.algorithm}")
    
    def preprocess_data(self, X, y=None):
        """
        Preprocess the data (scaling, encoding, etc.) within the MLModel class
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        
        Returns
        -------
        dict
            Dictionary containing the preprocessed data
        """
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        result = {}
        
        # Split data if both X and y are provided
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            result['X_train'] = X_train
            result['X_test'] = X_test
            result['y_train'] = y_train
            result['y_test'] = y_test
        else:
            result['X'] = X
            if y is not None:
                result['y'] = y
        
        # Scale numerical features
        if 'X_train' in result:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            result['X_train'] = self.scaler.fit_transform(result['X_train'])
            result['X_test'] = self.scaler.transform(result['X_test'])
        elif 'X' in result:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            result['X'] = self.scaler.fit_transform(result['X'])
        
        # Encode target for classification (if needed)
        if self.problem_type == 'classification' and y is not None:
            from sklearn.preprocessing import LabelEncoder
            if 'y_train' in result:
                unique_values = np.unique(y_train)
                if len(unique_values) > 2 or not isinstance(y_train[0], (int, bool, np.integer)):
                    self.label_encoder = LabelEncoder()
                    result['y_train'] = self.label_encoder.fit_transform(y_train)
                    result['y_test'] = self.label_encoder.transform(y_test)
                    result['n_classes'] = len(self.label_encoder.classes_)
                else:
                    result['n_classes'] = 2
            else:
                unique_values = np.unique(y)
                if len(unique_values) > 2 or not isinstance(y[0], (int, bool, np.integer)):
                    self.label_encoder = LabelEncoder()
                    result['y'] = self.label_encoder.fit_transform(y)
                    result['n_classes'] = len(self.label_encoder.classes_)
                else:
                    result['n_classes'] = 2
        
        return result
    
    def train(self, X, y, validation_data=None, test_size=0.2, random_state=42):
        """Train the model"""
        # Preprocess the data
        data = self.preprocess_data(X, y)
        
        # Manually split the data if not already split in preprocess_data
        if 'X_train' not in data:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=test_size, random_state=random_state
            )
            data['X_train'] = X_train
            data['X_test'] = X_test
            data['y_train'] = y_train 
            data['y_test'] = y_test
        
        # Use validation data if provided, otherwise use test data
        if validation_data is not None:
            X_val, y_val = validation_data
            if self.scaler is not None:
                X_val = self.scaler.transform(X_val)
            
            if self.label_encoder is not None:
                y_val = self.label_encoder.transform(y_val)
        else:
            X_val, y_val = data.get('X_test'), data.get('y_test')
        
        # Train different model types
        if self.model_type == 'classical':
            self._train_classical(data)
        
        elif self.model_type == 'neural':
            self._train_neural(data, X_val, y_val)
        
        elif self.model_type == 'quantum':
            self._train_quantum(data)
        
        # Skip metrics calculation for now - we'll handle it separately
        # self._calculate_metrics(data)
        
        # For classification models, manually calculate metrics
        if self.problem_type == 'classification':
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
            
            X_test, y_test = data['X_test'], data['y_test']
            
            # For Keras/TensorFlow model
            if hasattr(self.model, 'predict') and hasattr(self.model, 'output_shape'):
                # Get raw predictions
                raw_predictions = self.model.predict(X_test)
                
                # Convert continuous predictions to binary
                if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
                    # Binary classification
                    binary_predictions = (raw_predictions > 0.5).astype(int).flatten()
                elif raw_predictions.ndim == 2 and raw_predictions.shape[1] > 1:
                    # Multi-class
                    binary_predictions = np.argmax(raw_predictions, axis=1)
                else:
                    binary_predictions = (raw_predictions > 0.5).astype(int)
                
                # Make sure y_test is properly shaped
                if y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.flatten()
                
                # Calculate metrics
                try:
                    self.metrics['accuracy'] = float(accuracy_score(y_test, binary_predictions))
                    self.metrics['precision'] = float(precision_score(y_test, binary_predictions, average='weighted', zero_division=0))
                    self.metrics['recall'] = float(recall_score(y_test, binary_predictions, average='weighted', zero_division=0))
                    self.metrics['f1'] = float(f1_score(y_test, binary_predictions, average='weighted', zero_division=0))
                    
                    # Confusion matrix and report
                    self.confusion_matrix_data = confusion_matrix(y_test, binary_predictions)
                    self.classification_report_data = classification_report(y_test, binary_predictions)
                except Exception as e:
                    print(f"Error calculating metrics manually: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.metrics['error'] = str(e)
            else:
                # For classical models, use the original method
                try:
                    self._calculate_metrics(data)
                except Exception as e:
                    print(f"Error in original _calculate_metrics: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.metrics['error'] = str(e)
        else:
            # For regression models, use the original method
            try:
                self._calculate_metrics(data)
            except Exception as e:
                print(f"Error in original _calculate_metrics for regression: {str(e)}")
                self.metrics['error'] = str(e)
        
        # Store problem type in metrics
        self.metrics['problem_type'] = self.problem_type
        
        return self
    
    def _train_classical(self, data):
        """
        Train a classical ML model
        
        Parameters
        ----------
        data : dict
            Dictionary containing the preprocessed data
        """
        X_train, y_train = data['X_train'], data['y_train']
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get feature importances if available
        self._extract_feature_importances()
    
    # Replace the _train_neural method in the MLModel class in ml_utils.py

    # Fix for the _train_neural method in the MLModel class in ml_utils.py

    def _train_neural(self, data, X_val, y_val):
        """
        Train a neural network model
        
        Parameters
        ----------
        data : dict
            Dictionary containing the preprocessed data
        X_val : array-like
            Validation feature matrix
        y_val : array-like
            Validation target vector
        """
        import tensorflow as tf
        import numpy as np
        
        X_train, y_train = data['X_train'], data['y_train']
        
        # Build the neural network model if not built yet
        if self.model is None:
            # Determine input shape
            if self.algorithm in ['RNN', 'LSTM', 'GRU', 'CNN']:
                # Reshape for recurrent/convolutional networks if 2D data
                if len(X_train.shape) == 2:
                    # Add time dimension (assume each sample is a sequence)
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                
                input_shape = X_train.shape[1:]
            else:
                input_shape = (X_train.shape[1],)
            
            # Get number of classes for classification
            if self.problem_type == 'classification':
                num_classes = data.get('n_classes')
                print(f"Number of classes detected: {num_classes}")
            else:
                num_classes = None
            
            # Build the model
            self._build_neural_model(input_shape, num_classes)
        
        # Prepare callbacks
        callbacks = []
        if self.model_config.get('early_stopping', True):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.model_config.get('patience', 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # For classification, properly prepare target data
        if self.problem_type == 'classification':
            n_classes = data.get('n_classes')
            
            # Get the output shape to determine format needed
            output_shape = self.model.output_shape
            output_units = output_shape[-1]
            
            print(f"Model output shape: {output_shape}, Units: {output_units}")
            print(f"Target shape: {np.array(y_train).shape}")
            
            # For binary classification with a single output unit
            if output_units == 1:
                # Need to reshape to match output
                y_train_prepared = np.array(y_train).reshape(-1, 1)
                y_val_prepared = np.array(y_val).reshape(-1, 1)
                
            # For multi-class classification with multiple output units
            elif output_units > 1:
                # One-hot encode the targets
                from tensorflow.keras.utils import to_categorical
                y_train_prepared = to_categorical(y_train, n_classes)
                y_val_prepared = to_categorical(y_val, n_classes)
                print(f"Converted targets to one-hot encoding. Shape: {y_train_prepared.shape}")
            else:
                # Fallback - use as is
                y_train_prepared = y_train
                y_val_prepared = y_val
        else:
            # For regression, reshape if needed
            if len(np.array(y_train).shape) == 1:
                y_train_prepared = np.array(y_train).reshape(-1, 1)
                y_val_prepared = np.array(y_val).reshape(-1, 1)
            else:
                y_train_prepared = y_train
                y_val_prepared = y_val
        
        # Train the model
        try:
            print(f"Training model with prepared data: X_train shape: {X_train.shape}, y_train shape: {np.array(y_train_prepared).shape}")
            self.history = self.model.fit(
                X_train, y_train_prepared,
                epochs=self.model_config.get('epochs', 100),
                batch_size=self.model_config.get('batch_size', 32),
                validation_data=(X_val, y_val_prepared),
                callbacks=callbacks,
                verbose=1  # Changed to 1 to see progress
            ).history
            
            # Store history in metrics for reporting
            self.metrics['history'] = self.history
            
            # Store model config in metrics for reporting
            self.metrics['model_config'] = self.model_config
            
        except Exception as e:
            # Log the error with complete information
            error_message = str(e)
            print(f"Training error: {error_message}")
            print(f"X_train shape: {X_train.shape}, y_train shape: {np.array(y_train_prepared).shape}")
            print(f"Model output shape: {self.model.output_shape}")
            
            # Re-raise the exception to be handled by caller
            raise
    
    def _extract_feature_importances(self):
        """Extract feature importances if available"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            if len(self.model.coef_.shape) == 1:
                self.feature_importances = np.abs(self.model.coef_)
            else:
                self.feature_importances = np.mean(np.abs(self.model.coef_), axis=0)
                
    def _calculate_metrics(self, data):
        """
        Calculate performance metrics
        
        Parameters
        ----------
        data : dict
            Dictionary containing the preprocessed data
        """
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, 
            confusion_matrix, classification_report, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        X_test, y_test = data['X_test'], data['y_test']
        
        try:
            print(f"Problem type: {self.problem_type}")
            print(f"Algorithm: {self.algorithm}")
            print(f"Model type: {self.model_type}")
            
            # Make predictions
            if self.problem_type == 'classification':
                # Get raw predictions
                raw_predictions = self.model.predict(X_test)
                print(f"Raw predictions shape: {raw_predictions.shape}")
                
                # Convert to binary predictions
                if hasattr(self.model, 'output_shape'):  # For Keras/TF models
                    print("Handling Keras/TF model")
                    # For binary classification with sigmoid
                    if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
                        print("Binary classification with sigmoid")
                        y_prob = raw_predictions.flatten()
                        y_pred = (y_prob > 0.5).astype(int)
                    # For multi-class with softmax
                    elif raw_predictions.ndim == 2 and raw_predictions.shape[1] > 1:
                        print("Multi-class classification with softmax")
                        y_prob = raw_predictions
                        y_pred = np.argmax(y_prob, axis=1)
                    else:
                        print(f"Other prediction shape: {raw_predictions.shape}")
                        y_prob = raw_predictions
                        y_pred = (y_prob > 0.5).astype(int)
                else:  # For scikit-learn models
                    print("Handling scikit-learn model")
                    y_pred = raw_predictions
                    
                    # Try to get probabilities if available
                    if hasattr(self.model, 'predict_proba'):
                        try:
                            y_prob = self.model.predict_proba(X_test)
                            if y_prob.shape[1] == 2:  # Binary classification
                                y_prob = y_prob[:, 1]
                        except:
                            y_prob = None
                    else:
                        y_prob = None
                
                # Ensure y_test is 1D if needed
                print(f"y_test shape before: {y_test.shape if hasattr(y_test, 'shape') else 'no shape'}")
                if hasattr(y_test, 'ndim') and y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.flatten()
                print(f"y_test shape after: {y_test.shape if hasattr(y_test, 'shape') else 'no shape'}")
                
                print(f"y_pred shape: {y_pred.shape}")
                print(f"y_test sample: {y_test[:5]}")
                print(f"y_pred sample: {y_pred[:5]}")
                
                # Calculate classification metrics
                self.metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
                
                # For binary classification
                if len(np.unique(y_test)) <= 2:
                    self.metrics['precision'] = float(precision_score(y_test, y_pred, average='binary', zero_division=0))
                    self.metrics['recall'] = float(recall_score(y_test, y_pred, average='binary', zero_division=0))
                    self.metrics['f1'] = float(f1_score(y_test, y_pred, average='binary', zero_division=0))
                    
                    if y_prob is not None:
                        self.metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
                else:
                    # For multiclass classification
                    self.metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                    self.metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                    self.metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                
                # Confusion matrix
                self.confusion_matrix_data = confusion_matrix(y_test, y_pred)
                
                # Classification report
                self.classification_report_data = classification_report(y_test, y_pred)
                
            else:  # regression
                # Make predictions
                y_pred = self.model.predict(X_test)
                
                # Ensure y_test and y_pred have the same shape
                if hasattr(y_pred, 'shape') and hasattr(y_test, 'shape') and y_pred.shape != y_test.shape:
                    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                    if y_test.ndim == 2 and y_test.shape[1] == 1:
                        y_test = y_test.flatten()
                
                # Calculate regression metrics
                self.metrics['mse'] = float(mean_squared_error(y_test, y_pred))
                self.metrics['rmse'] = float(np.sqrt(self.metrics['mse']))
                self.metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
                self.metrics['r2'] = float(r2_score(y_test, y_pred))
                
                # Add more regression metrics if needed
                self.metrics['explained_variance'] = float(np.var(y_pred) / np.var(y_test))
                
                # Calculate residuals for analysis
                residuals = y_test - y_pred
                self.metrics['mean_residual'] = float(np.mean(residuals))
                self.metrics['std_residual'] = float(np.std(residuals))
            
            # Store problem type in metrics
            self.metrics['problem_type'] = self.problem_type
            
        except Exception as e:
            import traceback
            print(f"Error calculating metrics: {str(e)}")
            print(traceback.format_exc())
            self.metrics['error'] = str(e)
            self.metrics['problem_type'] = self.problem_type
    
    def tune_hyperparameters(self, X, y, param_grid, cv=5, n_iter=10, method='random', 
                           scoring=None, n_jobs=-1):
        """
        Tune hyperparameters using grid or randomized search
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        param_grid : dict
            Grid of hyperparameters to search
        cv : int
            Number of cross-validation folds
        n_iter : int
            Number of iterations for randomized search
        method : str
            Search method ('grid' or 'random')
        scoring : str
            Scoring metric for evaluation
        n_jobs : int
            Number of parallel jobs
        
        Returns
        -------
        self
            Model with tuned hyperparameters
        """
        # Only for classical models for now
        if self.model_type != 'classical':
            raise ValueError("Hyperparameter tuning is only supported for classical models")
        
        # Preprocess data
        data = self.preprocess_data(X, y)
        X_processed = data.get('X', data.get('X_train'))
        y_processed = data.get('y', data.get('y_train'))
        
        # Set default scoring based on problem type
        if scoring is None:
            if self.problem_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
        
        # Initialize the search
        if method == 'grid':
            search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=True
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                self.model, 
                param_grid, 
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unsupported search method: {method}")
        
        # Run the search
        search.fit(X_processed, y_processed)
        
        # Update model with best estimator
        self.model = search.best_estimator_
        
        # Store tuning results
        self.best_params = search.best_params_
        self.tuning_results = []
        
        for params, mean_score, std_score in zip(
            search.cv_results_['params'],
            search.cv_results_['mean_test_score'],
            search.cv_results_['std_test_score']
        ):
            self.tuning_results.append({
                **params,
                'mean_test_score': mean_score,
                'std_test_score': std_score
            })
        
        # Extract feature importances if available
        self._extract_feature_importances()
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        
        Returns
        -------
        array-like
            Predictions
        """
        # Scale input data if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type == 'neural' and self.algorithm in ['RNN', 'LSTM', 'GRU', 'CNN']:
            # Reshape for recurrent/convolutional networks if needed
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Decode predictions for classification if label encoder exists
        if self.problem_type == 'classification' and self.label_encoder is not None:
            # If binary classification with single output
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                if isinstance(predictions[0], (np.floating, float)):
                    predictions = (predictions > 0.5).astype(int)
                    
                predictions = self.label_encoder.inverse_transform(predictions.ravel())
            else:
                # For multiclass with probabilities
                class_indices = np.argmax(predictions, axis=1)
                predictions = self.label_encoder.inverse_transform(class_indices)
        
        return predictions
    
    def save(self, filepath, save_format=None):
        """
        Save the model to disk
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        save_format : str
            Format to save the model ('pickle', 'joblib', or 'tensorflow')
        
        Returns
        -------
        str
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Special handling for neural network models
        if self.model_type == 'neural':
            # Change file extension for TensorFlow models
            keras_path = os.path.splitext(filepath)[0] + '.keras'
            
            # Save the model
            self.model.save(keras_path)
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'algorithm': self.algorithm,
                'problem_type': self.problem_type,
                'params': self.params,
                'metrics': self.metrics,
                'best_params': self.best_params,
                'feature_importances': self.feature_importances.tolist() if self.feature_importances is not None else None,
                'confusion_matrix_data': self.confusion_matrix_data.tolist() if self.confusion_matrix_data is not None else None,
                'classification_report_data': self.classification_report_data,
                'timestamp': datetime.now().isoformat(),
                'actual_model_path': keras_path  # Save the actual path used
            }
            
            # Save scaler and encoders separately
            if self.scaler is not None:
                scaler_path = os.path.join(os.path.dirname(filepath), f"{os.path.basename(filepath)}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                metadata['scaler_path'] = scaler_path
            
            if self.label_encoder is not None:
                encoder_path = os.path.join(os.path.dirname(filepath), f"{os.path.basename(filepath)}_encoder.pkl")
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                metadata['encoder_path'] = encoder_path
            
            # Save metadata
            metadata_path = os.path.join(os.path.dirname(filepath), f"{os.path.basename(filepath)}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return keras_path
        
        # For non-neural models, use original method
        else:
            # Determine save format for traditional models
            if save_format is None:
                save_format = 'pickle'
            
            # Save the model
            if save_format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(self.model, f)
            elif save_format == 'joblib':
                joblib.dump(self.model, filepath)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'algorithm': self.algorithm,
                'problem_type': self.problem_type,
                'params': self.params,
                'metrics': self.metrics,
                'best_params': self.best_params,
                'feature_importances': self.feature_importances.tolist() if self.feature_importances is not None else None,
                'confusion_matrix_data': self.confusion_matrix_data.tolist() if self.confusion_matrix_data is not None else None,
                'classification_report_data': self.classification_report_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save scaler and encoders separately
            if self.scaler is not None:
                scaler_path = os.path.join(os.path.dirname(filepath), f"{os.path.basename(filepath)}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                metadata['scaler_path'] = scaler_path
            
            if self.label_encoder is not None:
                encoder_path = os.path.join(os.path.dirname(filepath), f"{os.path.basename(filepath)}_encoder.pkl")
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                metadata['encoder_path'] = encoder_path
            
            # Save metadata
            metadata_path = os.path.join(os.path.dirname(filepath), f"{os.path.basename(filepath)}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return filepath
    @classmethod
    def load(cls, filepath, metadata_path=None):
        """
        Load a model from disk
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        metadata_path : str
            Path to the metadata file (if None, inferred from filepath)
        
        Returns
        -------
        MLModel
            Loaded model
        """
        # Infer metadata path if not provided
        if metadata_path is None:
            metadata_path = os.path.join(os.path.dirname(filepath), f"{os.path.basename(filepath)}_metadata.json")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Initialize model
        model = cls(
            model_type=metadata['model_type'],
            algorithm=metadata['algorithm'],
            problem_type=metadata['problem_type'],
            params=metadata['params']
        )
        
        # Load the model
        if metadata['model_type'] == 'neural':
            if not NEURAL_AVAILABLE:
                raise ImportError("TensorFlow is required to load neural network models")
            
            # Load TensorFlow model
            model.model = tf.keras.models.load_model(filepath)
        elif os.path.exists(filepath):
            # Try pickle first
            try:
                with open(filepath, 'rb') as f:
                    model.model = pickle.load(f)
            except:
                # Try joblib if pickle fails
                model.model = joblib.load(filepath)
        
        # Load scaler if available
        if 'scaler_path' in metadata and os.path.exists(metadata['scaler_path']):
            with open(metadata['scaler_path'], 'rb') as f:
                model.scaler = pickle.load(f)
        
        # Load label encoder if available
        if 'encoder_path' in metadata and os.path.exists(metadata['encoder_path']):
            with open(metadata['encoder_path'], 'rb') as f:
                model.label_encoder = pickle.load(f)
        
        # Set other attributes
        model.metrics = metadata['metrics']
        model.best_params = metadata['best_params']
        model.feature_importances = np.array(metadata['feature_importances']) if metadata['feature_importances'] else None
        model.confusion_matrix_data = np.array(metadata['confusion_matrix_data']) if metadata['confusion_matrix_data'] else None
        model.classification_report_data = metadata['classification_report_data']
        
        return model
    
    def generate_report(self, output_folder, experiment_name=None):
        """
        Generate a comprehensive report for the model
        
        Parameters
        ----------
        output_folder : str
            Folder to save the report
        experiment_name : str
            Name of the experiment
        
        Returns
        -------
        str
            Path to the generated report
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Determine report type based on model type
        if self.model_type == 'quantum' and QUANTUM_AVAILABLE and self.quantum_circuit:
            # Generate quantum report
            from app.utils.ml_report_utils import generate_quantum_report
            
            return generate_quantum_report(
                model_run_id=self.model_run_id or 'model',
                algorithm=self.algorithm,
                quantum_circuit=self.quantum_circuit,
                metrics=self.metrics,
                feature_importances=self.feature_importances,
                output_folder=output_folder,
                experiment_name=experiment_name,
                is_pro=self.is_pro
            )
        else:
            # Generate standard model report
            feature_importances_dict = None
            if self.feature_importances is not None:
                # Create feature names if not available
                feature_names = [f"feature_{i}" for i in range(len(self.feature_importances))]
                feature_importances_dict = dict(zip(feature_names, self.feature_importances))
            
            return generate_model_report(
                model_run_id=self.model_run_id or 'model',
                algorithm=self.algorithm,
                problem_type=self.problem_type,
                metrics=self.metrics,
                feature_importances=feature_importances_dict,
                confusion_matrix_data=self.confusion_matrix_data,
                classification_report_data=self.classification_report_data,
                output_folder=output_folder,
                tuning_report_path=self._generate_tuning_report(output_folder, experiment_name) if self.tuning_results else None,
                experiment_name=experiment_name,
                quantum_circuit=getattr(self, 'quantum_circuit', None)
            )
    
    def _generate_tuning_report(self, output_folder, experiment_name=None):
        """
        Generate a hyperparameter tuning report
        
        Parameters
        ----------
        output_folder : str
            Folder to save the report
        experiment_name : str
            Name of the experiment
        
        Returns
        -------
        str
            Path to the generated report
        """
        if not self.tuning_results or not self.best_params:
            return None
        
        return generate_tuning_report(
            model_run_id=self.model_run_id or 'model',
            algorithm=self.algorithm,
            problem_type=self.problem_type,
            tuning_results=self.tuning_results,
            best_params=self.best_params,
            metrics=self.metrics,
            output_folder=output_folder,
            experiment_name=experiment_name
        )
    
    def export_predictions(self, X, y_true, output_path=None, feature_names=None):
        """
        Export predictions to a CSV file
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y_true : array-like
            True target values
        output_path : str
            Path to save the CSV file
        feature_names : list
            Names of the features
        
        Returns
        -------
        str or pandas.DataFrame
            Path to the CSV file or DataFrame with predictions
        """
        # Make predictions
        y_pred = self.predict(X)
        
        return export_model_predictions(
            y_true=y_true,
            y_pred=y_pred,
            feature_names=feature_names,
            X=X if feature_names else None,
            output_path=output_path
        )
    
    def _calculate_metrics(self, data):
        """
        Calculate performance metrics
        
        Parameters
        ----------
        data : dict
            Dictionary containing the preprocessed data
        """
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, 
            confusion_matrix, classification_report, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        X_test, y_test = data['X_test'], data['y_test']
        
        # Make predictions
        if self.problem_type == 'classification':
            # Get raw predictions
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)
                # For binary classification, take the probability of the positive class
                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                elif y_prob.shape[1] == 1:
                    y_prob = y_prob.ravel()
            elif hasattr(self.model, 'predict'):
                raw_predictions = self.model.predict(X_test)
                
                # Handle TensorFlow/Keras models which output continuous values
                if hasattr(self.model, 'output_shape') and isinstance(raw_predictions, np.ndarray):
                    # If we have a binary classification with sigmoid activation
                    if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
                        y_prob = raw_predictions.ravel()
                    # If we have multi-class with softmax
                    elif raw_predictions.ndim == 2 and raw_predictions.shape[1] > 1:
                        y_prob = raw_predictions
                    else:
                        y_prob = None
                else:
                    y_prob = None
            else:
                y_prob = None
            
            # Convert continuous predictions to class labels
            if hasattr(self.model, 'predict'):
                if hasattr(self.model, 'output_shape') and hasattr(self.model, 'layers'):
                    # This is a Keras model - need to threshold the outputs
                    raw_preds = self.model.predict(X_test)
                    if raw_preds.ndim == 2 and raw_preds.shape[1] == 1:
                        # Binary classification with sigmoid
                        y_pred = (raw_preds > 0.5).astype(int).ravel()
                    elif raw_preds.ndim == 2 and raw_preds.shape[1] > 1:
                        # Multi-class with softmax
                        y_pred = np.argmax(raw_preds, axis=1)
                    else:
                        y_pred = raw_preds
                else:
                    # Standard sklearn model
                    y_pred = self.model.predict(X_test)
            else:
                # Fallback if no predict method
                y_pred = (y_prob > 0.5).astype(int) if y_prob is not None else None
            
            # Ensure y_test and y_pred have the same shape
            if y_pred is not None and y_pred.shape != y_test.shape:
                if y_pred.ndim == 1 and y_test.ndim == 2 and y_test.shape[1] == 1:
                    # y_test is column vector, y_pred is 1D
                    y_test = y_test.ravel()
                elif y_pred.ndim == 2 and y_pred.shape[1] == 1 and y_test.ndim == 1:
                    # y_pred is column vector, y_test is 1D
                    y_pred = y_pred.ravel()
            
            # Calculate classification metrics
            try:
                self.metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
                
                # For binary classification
                if len(np.unique(y_test)) == 2:
                    self.metrics['precision'] = float(precision_score(y_test, y_pred, average='binary', zero_division=0))
                    self.metrics['recall'] = float(recall_score(y_test, y_pred, average='binary', zero_division=0))
                    self.metrics['f1'] = float(f1_score(y_test, y_pred, average='binary', zero_division=0))
                    
                    if y_prob is not None:
                        self.metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
                else:
                    # For multiclass classification
                    self.metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                    self.metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                    self.metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    
                    if y_prob is not None and y_prob.ndim == 2:
                        try:
                            self.metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
                        except:
                            pass  # Skip if not applicable
                
                # Confusion matrix
                self.confusion_matrix_data = confusion_matrix(y_test, y_pred)
                
                # Classification report
                self.classification_report_data = classification_report(y_test, y_pred)
            except Exception as e:
                print(f"Error calculating classification metrics: {str(e)}")
                import traceback
                traceback.print_exc()
                self.metrics['error'] = str(e)
        
        else:  # regression
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Ensure y_test and y_pred have the same shape
            if y_pred.shape != y_test.shape:
                if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                    y_pred = y_pred.ravel()
                if y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.ravel()
            
            try:
                # Calculate regression metrics
                self.metrics['mse'] = float(mean_squared_error(y_test, y_pred))
                self.metrics['rmse'] = float(np.sqrt(self.metrics['mse']))
                self.metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
                self.metrics['r2'] = float(r2_score(y_test, y_pred))
                
                # Add more regression metrics if needed
                self.metrics['explained_variance'] = float(np.var(y_pred) / np.var(y_test))
                
                # Calculate residuals for analysis
                residuals = y_test - y_pred
                self.metrics['mean_residual'] = float(np.mean(residuals))
                self.metrics['std_residual'] = float(np.std(residuals))
            except Exception as e:
                print(f"Error calculating regression metrics: {str(e)}")
                import traceback
                traceback.print_exc()
                self.metrics['error'] = str(e)
        
        # Store problem type in metrics
        self.metrics['problem_type'] = self.problem_type

def get_available_algorithms(problem_type, model_type='classical', is_pro=False):
    """
    Get available algorithms for a given problem type and model type
    
    Parameters
    ----------
    problem_type : str
        Type of problem ('classification' or 'regression')
    model_type : str
        Type of model ('classical', 'neural', 'quantum')
    is_pro : bool
        Whether user has pro subscription
    
    Returns
    -------
    list
        List of available algorithms
    """
    if problem_type == 'classification':
        if model_type == 'classical':
            return [
                'LogisticRegression',
                'RandomForest',
                'GradientBoosting',
                'SVM',
                'KNN',
                'DecisionTree',
                'NaiveBayes'
            ]
        elif model_type == 'neural':
            if not NEURAL_AVAILABLE:
                return []
            
            return [
                'MLP',
                'CNN',
                'RNN',
                'LSTM',
                'GRU'
            ]
        elif model_type == 'quantum':
            if not QUANTUM_AVAILABLE or not is_pro:
                return []
            
            return [
                'QSVM',
                'QKE',
                'QNN'
            ]
    
    elif problem_type == 'regression':
        if model_type == 'classical':
            return [
                'LinearRegression',
                'Ridge',
                'Lasso',
                'RandomForest',
                'GradientBoosting',
                'SVR',
                'KNN',
                'DecisionTree'
            ]
        elif model_type == 'neural':
            if not NEURAL_AVAILABLE:
                return []
            
            return [
                'MLP',
                'CNN',
                'RNN',
                'LSTM',
                'GRU'
            ]
        elif model_type == 'quantum':
            if not QUANTUM_AVAILABLE or not is_pro:
                return []
            
            return [
                'QKE',
                'QRegression'
            ]
    
    return []

def get_default_hyperparameters(algorithm, problem_type):
    """
    Get default hyperparameters for a given algorithm
    
    Parameters
    ----------
    algorithm : str
        Name of the algorithm
    problem_type : str
        Type of problem ('classification' or 'regression')
    
    Returns
    -------
    dict
        Default hyperparameters
    """
    # Classification algorithms
    if problem_type == 'classification':
        if algorithm == 'LogisticRegression':
            return {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000
            }
        elif algorithm == 'RandomForest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        elif algorithm == 'GradientBoosting':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
        elif algorithm == 'SVM':
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
        elif algorithm == 'KNN':
            return {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            }
        elif algorithm == 'DecisionTree':
            return {
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        elif algorithm == 'NaiveBayes':
            return {}
        elif algorithm == 'MLP':
            return {
                'layers': [64, 32],
                'activation': 'relu',
                'dropout': 0.2,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping': True,
                'patience': 10
            }
        elif algorithm == 'CNN':
            return {
                'layers': [64, 32],
                'activation': 'relu',
                'dropout': 0.2,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping': True,
                'patience': 10
            }
        elif algorithm in ['RNN', 'LSTM', 'GRU']:
            return {
                'layers': [64, 32],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping': True,
                'patience': 10
            }
        elif algorithm in ['QSVM', 'QKE']:
            return {
                'qubits': 4,
                'feature_map': 'ZZFeatureMap'
            }
        elif algorithm == 'QNN':
            return {
                'qubits': 4,
                'layers': 2,
                'ansatz': 'StronglyEntanglingLayers',
                'learning_rate': 0.01,
                'epochs': 50
            }
    
    # Regression algorithms
    elif problem_type == 'regression':
        if algorithm == 'LinearRegression':
            return {}
        elif algorithm == 'Ridge':
            return {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            }
        elif algorithm == 'Lasso':
            return {
                'alpha': 1.0,
                'random_state': 42
            }
        elif algorithm == 'RandomForest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        elif algorithm == 'GradientBoosting':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
        elif algorithm == 'SVR':
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale'
            }
        elif algorithm == 'KNN':
            return {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            }
        elif algorithm == 'DecisionTree':
            return {
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        elif algorithm == 'MLP':
            return {
                'layers': [64, 32],
                'activation': 'relu',
                'dropout': 0.2,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping': True,
                'patience': 10
            }
        elif algorithm in ['RNN', 'LSTM', 'GRU']:
            return {
                'layers': [64, 32],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping': True,
                'patience': 10
            }
        elif algorithm == 'QKE':
            return {
                'qubits': 4,
                'feature_map': 'ZZFeatureMap'
            }
        elif algorithm == 'QRegression':
            return {
                'qubits': 4,
                'layers': 2,
                'ansatz': 'StronglyEntanglingLayers',
                'learning_rate': 0.01,
                'epochs': 50
            }
    
    # Default empty dict if algorithm not found
    return {}

def get_hyperparameter_grid(algorithm, problem_type):
    """
    Get hyperparameter grid for tuning a given algorithm
    
    Parameters
    ----------
    algorithm : str
        Name of the algorithm
    problem_type : str
        Type of problem ('classification' or 'regression')
    
    Returns
    -------
    dict
        Hyperparameter grid
    """
    # Classification algorithms
    if problem_type == 'classification':
        if algorithm == 'LogisticRegression':
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
            }
        elif algorithm == 'RandomForest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif algorithm == 'GradientBoosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        elif algorithm == 'SVM':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 1.0]
            }
        elif algorithm == 'KNN':
            return {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        elif algorithm == 'DecisionTree':
            return {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        elif algorithm == 'NaiveBayes':
            return {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
    
    # Regression algorithms
    elif problem_type == 'regression':
        if algorithm == 'LinearRegression':
            return {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        elif algorithm == 'Ridge':
            return {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        elif algorithm == 'Lasso':
            return {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'selection': ['cyclic', 'random']
            }
        elif algorithm == 'RandomForest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif algorithm == 'GradientBoosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        elif algorithm == 'SVR':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 1.0],
                'epsilon': [0.1, 0.2, 0.5]
            }
        elif algorithm == 'KNN':
            return {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        elif algorithm == 'DecisionTree':
            return {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
            }
    
    # Neural networks (simplified grid since most params are adjusted during build)
    elif algorithm in ['MLP', 'FNN', 'RNN', 'LSTM', 'GRU', 'CNN']:
        return {
            'learning_rate': [0.01, 0.001, 0.0001],
            'dropout': [0.1, 0.2, 0.3],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 200]
        }
    
    # Quantum algorithms (simplified grid)
    elif algorithm in ['QSVM', 'QKE', 'QNN', 'QRegression']:
        return {
            'qubits': [2, 4, 6],
            'layers': [1, 2, 3]
        }
    
    # Default empty dict if algorithm not found
    return {}

def train_test_evaluate(X, y, algorithm, problem_type, model_type='classical', params=None, 
                       test_size=0.2, random_state=42, is_pro=False, experiment_id=None, 
                       model_run_id=None, output_folder=None, use_k_fold=False, k_fold_value=5):
    """
    Train, test, and evaluate a machine learning model
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    algorithm : str
        Name of the algorithm
    problem_type : str
        Type of problem ('classification' or 'regression')
    model_type : str
        Type of model ('classical', 'neural', 'quantum')
    params : dict
        Hyperparameters for the model
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    is_pro : bool
        Whether user has pro subscription
    experiment_id : str
        ID of the experiment
    model_run_id : str
        ID of the model run
    output_folder : str
        Folder to save model outputs
    use_k_fold : bool
        Whether to use k-fold cross-validation
    k_fold_value : int
        Number of folds for cross-validation
    
    Returns
    -------
    dict
        Results of the model training and evaluation
    """
    # Initialize results
    results = {
        'algorithm': algorithm,
        'problem_type': problem_type,
        'model_type': model_type,
        'params': params,
        'metrics': {},
        'feature_importances': None,
        'model_path': None,
        'report_path': None
    }
    
    try:
        # Create model
        model = MLModel(
            model_type=model_type,
            algorithm=algorithm,
            problem_type=problem_type,
            params=params,
            is_pro=is_pro,
            experiment_id=experiment_id,
            model_run_id=model_run_id
        )
        
        # Train model (with or without k-fold)
        if use_k_fold:
            results.update(train_with_kfold(
                model, X, y, k_fold_value, random_state, output_folder
            ))
        else:
            # Train and evaluate on a single train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train the model
            model.train(X_train, y_train, validation_data=(X_test, y_test))
            
            # Get metrics
            results['metrics'] = model.metrics
            
            # Get feature importances
            if model.feature_importances is not None:
                # Create feature names if not available
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances))]
                results['feature_importances'] = dict(zip(feature_names, model.feature_importances))
            
            # Save model if output_folder provided
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                
                # Save model
                model_path = os.path.join(output_folder, f"{algorithm}_{model_run_id}.pkl")
                model.save(model_path)
                results['model_path'] = model_path
                
                # Generate report
                report_path = model.generate_report(output_folder, experiment_id)
                results['report_path'] = report_path
                
                # Save predictions
                predictions_path = os.path.join(output_folder, f"{algorithm}_{model_run_id}_predictions.csv")
                model.export_predictions(X_test, y_test, predictions_path)
                results['predictions_path'] = predictions_path
        
        return results
    
    except Exception as e:
        # Log error
        import traceback
        error_details = traceback.format_exc()
        
        results['error'] = str(e)
        results['error_details'] = error_details
        
        return results

def train_with_kfold(model, X, y, n_splits=5, random_state=42, output_folder=None):
    """
    Train a model using k-fold cross-validation
    
    Parameters
    ----------
    model : MLModel
        Model to train
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    n_splits : int
        Number of folds
    random_state : int
        Random seed for reproducibility
    output_folder : str
        Folder to save model outputs
    
    Returns
    -------
    dict
        Results of the k-fold training
    """
    results = {
        'metrics': {},
        'fold_metrics': [],
        'best_fold': 0,
        'best_fold_metrics': {},
        'feature_importances': None
    }
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Prepare scaled X if needed
    X_scaled = None
    if hasattr(model, 'scaler') and model.scaler is not None:
        X_scaled = model.scaler.transform(X)
    else:
        # Create a new scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.scaler = scaler
    
    # Prepare encoded y if needed
    y_encoded = y
    if model.problem_type == 'classification':
        if len(np.unique(y)) > 2 or not isinstance(y[0], (int, bool, np.integer)):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            model.label_encoder = label_encoder
    
    # Performance metrics for each fold
    fold_metrics = []
    best_model = None
    best_score = -float('inf')
    best_fold = 0
    
    # For each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        # Get train/test data for this fold
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # Train model on this fold
        model.train(X_train, y_train, validation_data=(X_test, y_test))
        
        # Store fold metrics
        fold_result = {
            'fold': fold + 1,
            'metrics': model.metrics.copy()
        }
        fold_metrics.append(fold_result)
        
        # Track best model (based on primary metric)
        current_score = get_primary_score(model.metrics, model.problem_type)
        if current_score > best_score:
            best_score = current_score
            best_fold = fold + 1
            best_model = model
            # Save feature importances if available
            if model.feature_importances is not None:
                # Create feature names if not available
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances))]
                results['feature_importances'] = dict(zip(feature_names, model.feature_importances))
    
    # Compute average metrics
    avg_metrics = {}
    for metric in fold_metrics[0]['metrics']:
        if isinstance(fold_metrics[0]['metrics'][metric], (int, float)) and not isinstance(fold_metrics[0]['metrics'][metric], bool):
            avg_metrics[metric] = np.mean([fold['metrics'][metric] for fold in fold_metrics])
    
    # Store results
    results['metrics'] = avg_metrics
    results['fold_metrics'] = fold_metrics
    results['best_fold'] = best_fold
    results['best_fold_metrics'] = fold_metrics[best_fold - 1]['metrics']
    
    # Save best model if output_folder provided
    if output_folder and best_model:
        os.makedirs(output_folder, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_folder, f"{best_model.algorithm}_kfold_best.pkl")
        best_model.save(model_path)
        results['model_path'] = model_path
        
        # Generate report
        report_path = best_model.generate_report(output_folder, f"{best_model.experiment_id}_kfold")
        results['report_path'] = report_path
    
    return results

def get_primary_score(metrics, problem_type):
    """
    Get the primary score from metrics based on problem type
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    problem_type : str
        Type of problem ('classification' or 'regression')
    
    Returns
    -------
    float
        Primary score
    """
    if problem_type == 'classification':
        # For classification, prioritize metrics in this order
        for metric in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
            if metric in metrics:
                return metrics[metric]
    else:  # regression
        # For regression, prioritize metrics in this order (note: higher is better)
        if 'r2' in metrics:
            return metrics['r2']
        elif 'mse' in metrics:
            return -metrics['mse']  # Negative because lower MSE is better
        elif 'rmse' in metrics:
            return -metrics['rmse']  # Negative because lower RMSE is better
        elif 'mae' in metrics:
            return -metrics['mae']  # Negative because lower MAE is better
    
    # If no recognized metric found, return 0
    return 0

def compare_models(models_results, output_folder=None, experiment_name=None):
    """
    Compare multiple models and generate a comparison report
    
    Parameters
    ----------
    models_results : list of dict
        List of model results
    output_folder : str
        Folder to save the comparison report
    experiment_name : str
        Name of the experiment
    
    Returns
    -------
    dict
        Comparison results
    """
    from app.utils.ml_report_utils import generate_comparison_report
    
    # Filter out models with errors
    valid_models = [result for result in models_results if 'error' not in result]
    
    if not valid_models:
        return {'error': 'No valid models to compare'}
    
    # Prepare data for comparison report
    models_data = []
    for result in valid_models:
        model_data = {
            'model_run_id': result.get('model_run_id', 'unknown'),
            'algorithm': result['algorithm'],
            'problem_type': result['problem_type'],
            'metrics': result['metrics'],
            'feature_importances': result.get('feature_importances')
        }
        models_data.append(model_data)
    
    # Generate comparison report if output_folder provided
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
        report_path = generate_comparison_report(
            models_data=models_data,
            output_folder=output_folder,
            experiment_name=experiment_name
        )
        
        return {
            'models': models_data,
            'report_path': report_path,
            'best_model': find_best_model(models_data)
        }
    
    return {
        'models': models_data,
        'best_model': find_best_model(models_data)
    }

def find_best_model(models_data):
    """
    Find the best model from a list of models
    
    Parameters
    ----------
    models_data : list of dict
        List of model data
    
    Returns
    -------
    dict
        Best model data
    """
    if not models_data:
        return None
    
    problem_type = models_data[0]['problem_type']
    best_model = None
    best_score = -float('inf')
    
    for model in models_data:
        score = get_primary_score(model['metrics'], problem_type)
        if score > best_score:
            best_score = score
            best_model = model
    
    return best_model

def tune_hyperparameters_async(model_run_id, algorithm, problem_type, file_path, target_column, 
                              feature_columns, base_hyperparameters, model_folder, time_limit=3600,
                              model_type=None, notify_completion=True, notify_error=True, is_pro=False):
    """Start hyperparameter tuning in a background thread with a time limit"""
    # Import at top level so they're available in the thread
    from flask import current_app
    import os
    import json
    import traceback
    from datetime import datetime
    import numpy as np
    from threading import Thread
    
    # Get app context object before starting thread
    app = current_app._get_current_object()
    
    def task():
        from app import db
        from app.models.experiment import Experiment, ProModelRun
        from app.models.user import User
        from app.utils.email import send_model_completion_notification, send_error_notification
        from app.utils.data_processing import preprocess_data
        
        # Use the captured app object for context
        with app.app_context():
            # Create a new session for this thread
            session = db.create_scoped_session()
            
            try:
                # Get model run and related info
                model_run = session.query(ProModelRun).get(model_run_id)
                if not model_run:
                    print(f"Error: Model run {model_run_id} not found in thread")
                    return
                
                experiment = session.query(Experiment).get(model_run.experiment_id)
                user = session.query(User).get(experiment.user_id)
                
                # Update status
                model_run.status = 'tuning'
                model_run.logs = "Starting hyperparameter tuning...\n"  # Reset logs to avoid duplication
                model_run.hyperparameters_tuning = True
                model_run.started_at = datetime.utcnow()
                session.commit()
                
                # Check if file exists
                if not os.path.exists(file_path):
                    # Try to fix file path
                    base_name = os.path.basename(file_path)
                    alternate_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', base_name)
                    if os.path.exists(alternate_path):
                        file_path = alternate_path
                        model_run.logs += f"Fixed file path to {file_path}\n"
                        session.commit()
                    else:
                        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
                
                # Determine model type if not provided
                actual_model_type = model_type
                if actual_model_type is None:
                    actual_model_type = 'quantum' if algorithm.startswith('Q') else \
                               'neural' if algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU'] else \
                               'classical'
                
                # Check if pro user has access to this model type
                if actual_model_type == 'quantum' and not is_pro:
                    raise PermissionError("Quantum models are only available with Pro subscription")
                
                # Process the dataset
                model_run.logs += "Preprocessing dataset...\n"
                session.commit()
                
                processed_data = preprocess_data(file_path, target_column, feature_columns)
                X = processed_data['X']
                y = processed_data['y']
                
                model_run.logs += f"Preprocessed data shapes: X: {X.shape}, y: {len(y)}\n"
                session.commit()
                
                # Get parameter grid based on algorithm
                from app.utils.ml_utils import get_hyperparameter_grid
                param_grid = get_hyperparameter_grid(algorithm, problem_type)
                
                model_run.logs += "Starting hyperparameter search...\n"
                session.commit()
                
                # Use tune_hyperparameters_pipeline for the actual tuning
                from app.utils.ml_utils import tune_hyperparameters_pipeline
                results = tune_hyperparameters_pipeline(
                    X=X,
                    y=y,
                    algorithm=algorithm,
                    problem_type=problem_type,
                    model_type=actual_model_type,
                    param_grid=param_grid,
                    cv=5,
                    n_iter=10,
                    method='random',
                    scoring=None,  # Auto-select based on problem type
                    n_jobs=-1,
                    random_state=42,
                    is_pro=is_pro,
                    experiment_id=experiment.id,
                    model_run_id=model_run_id,
                    output_folder=model_folder
                )
                
                # Handle successful tuning
                if 'error' not in results:
                    # Helper function to convert numpy values to Python types
                    def convert_to_serializable(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        return obj
                    
                    # Update model run with results
                    model_run.status = 'completed'
                    model_run.hyperparameters = json.dumps(results.get('best_params', {}))
                    
                    # Process tuning results
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
                    
                    # Process metrics
                    clean_metrics = {}
                    for k, v in results.get('metrics', {}).items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            clean_metrics[k] = v
                        elif hasattr(v, 'tolist'):  # numpy array
                            clean_metrics[k] = v.tolist()
                        else:
                            clean_metrics[k] = str(v)
                    
                    model_run.tuning_metrics = json.dumps(clean_metrics)
                    model_run.model_path = results.get('model_path')
                    model_run.tuning_report_path = results.get('tuning_report_path')
                    model_run.report_path = results.get('report_path')
                    model_run.completed_at = datetime.utcnow()
                    model_run.logs += "Hyperparameter tuning completed successfully!\n"
                    session.commit()
                    
                    # Send completion notification
                    if notify_completion and hasattr(user, 'notify_on_completion') and user.notify_on_completion:
                        try:
                            send_model_completion_notification(
                                user.email,
                                user.get_full_name(),
                                experiment.name,
                                f"{algorithm} Hyperparameter Tuning",
                                clean_metrics
                            )
                            model_run.logs += "Sent completion notification email.\n"
                            session.commit()
                        except Exception as email_error:
                            model_run.logs += f"Error sending completion notification: {str(email_error)}\n"
                            session.commit()
                            print(f"Error sending completion notification: {str(email_error)}")
                else:
                    # Handle error
                    error_message = results.get('error', 'Unknown error')
                    error_details = results.get('error_details', '')
                    
                    model_run.status = 'failed'
                    model_run.logs += f"Hyperparameter tuning failed: {error_message}\n"
                    if error_details:
                        model_run.logs += f"Error details: {error_details}\n"
                    model_run.completed_at = datetime.utcnow()
                    session.commit()
                    
                    # Send error notification
                    if notify_error and hasattr(user, 'notify_on_error') and user.notify_on_error:
                        try:
                            send_error_notification(
                                user.email,
                                user.get_full_name(),
                                experiment.name,
                                f"{algorithm} Hyperparameter Tuning",
                                error_message
                            )
                            model_run.logs += "Sent error notification email.\n"
                            session.commit()
                        except Exception as email_error:
                            model_run.logs += f"Error sending error notification: {str(email_error)}\n"
                            session.commit()
                            print(f"Error sending error notification: {str(email_error)}")
                
            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Tuning error: {str(e)}\n{error_details}")
                
                # Update with error
                if 'model_run' in locals():
                    model_run.status = 'failed'
                    model_run.logs += f"Hyperparameter tuning failed: {str(e)}\n"
                    model_run.logs += f"Error details: {error_details}\n"
                    model_run.completed_at = datetime.utcnow()
                    session.commit()
                
                # Send error notification
                if notify_error and 'user' in locals() and 'experiment' in locals():
                    try:
                        send_error_notification(
                            user.email,
                            user.get_full_name(),
                            experiment.name,
                            f"{algorithm} Hyperparameter Tuning",
                            str(e)
                        )
                        if 'model_run' in locals():
                            model_run.logs += "Sent error notification email.\n"
                            session.commit()
                    except Exception as email_error:
                        if 'model_run' in locals():
                            model_run.logs += f"Error sending error notification: {str(email_error)}\n"
                            session.commit()
                        print(f"Error sending error notification: {str(email_error)}")
            
            finally:
                # Close the session
                session.close()
                print("Tuning thread complete, session closed")
    
    # Start background thread
    thread = Thread(target=task)
    thread.daemon = True
    thread.start()
    
    # Return a placeholder dictionary that will be updated by the thread
    return {
        'status': 'started',
        'algorithm': algorithm,
        'problem_type': problem_type
    }

def tune_hyperparameters_pipeline(X, y, algorithm, problem_type, model_type='classical', 
                                param_grid=None, cv=5, n_iter=10, method='random', 
                                scoring=None, n_jobs=-1, random_state=42, is_pro=False,
                                experiment_id=None, model_run_id=None, output_folder=None):
    """
    Complete pipeline for hyperparameter tuning
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    algorithm : str
        Name of the algorithm
    problem_type : str
        Type of problem ('classification' or 'regression')
    model_type : str
        Type of model ('classical', 'neural', 'quantum')
    param_grid : dict
        Grid of hyperparameters to search
    cv : int
        Number of cross-validation folds
    n_iter : int
        Number of iterations for randomized search
    method : str
        Search method ('grid' or 'random')
    scoring : str
        Scoring metric for evaluation
    n_jobs : int
        Number of parallel jobs
    random_state : int
        Random seed for reproducibility
    is_pro : bool
        Whether user has pro subscription
    experiment_id : str
        ID of the experiment
    model_run_id : str
        ID of the model run
    output_folder : str
        Folder to save outputs
    
    Returns
    -------
    dict
        Results of the hyperparameter tuning
    """
    # Initialize results
    results = {
        'algorithm': algorithm,
        'problem_type': problem_type,
        'model_type': model_type,
        'best_params': None,
        'best_score': None,
        'tuning_results': None,
        'model_path': None,
        'tuning_report_path': None,
        'metrics': {}
    }
    
    try:
        # Get default params if not provided
        base_params = get_default_hyperparameters(algorithm, problem_type)
        
        # Create model
        model = MLModel(
            model_type=model_type,
            algorithm=algorithm,
            problem_type=problem_type,
            params=base_params,
            is_pro=is_pro,
            experiment_id=experiment_id,
            model_run_id=model_run_id
        )
        
        # Use default param grid if not provided
        if param_grid is None:
            param_grid = get_hyperparameter_grid(algorithm, problem_type)
        
        # Tune hyperparameters
        model.tune_hyperparameters(
            X=X,
            y=y,
            param_grid=param_grid,
            cv=cv,
            n_iter=n_iter,
            method=method,
            scoring=scoring,
            n_jobs=n_jobs
        )
        
        # Train model with best params
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        model.train(X_train, y_train, validation_data=(X_test, y_test))
        
        # Get results
        results['best_params'] = model.best_params
        results['tuning_results'] = model.tuning_results
        results['metrics'] = model.metrics
        
        # Get feature importances
        if model.feature_importances is not None:
            # Create feature names if not available
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances))]
            results['feature_importances'] = dict(zip(feature_names, model.feature_importances))
        
        # Save model and reports if output_folder provided
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            
            # Save model
            model_path = os.path.join(output_folder, f"{algorithm}_tuned_{model_run_id}.pkl")
            model.save(model_path)
            results['model_path'] = model_path
            
            # Generate tuning report
            tuning_report_path = model._generate_tuning_report(output_folder, experiment_id)
            results['tuning_report_path'] = tuning_report_path
            
            # Generate model report
            report_path = model.generate_report(output_folder, experiment_id)
            results['report_path'] = report_path
        
        return results
    
    except Exception as e:
        # Log error
        import traceback
        error_details = traceback.format_exc()
        
        results['error'] = str(e)
        results['error_details'] = error_details
        
        return results

def create_quantum_models():
    """
    Additional quantum models that can be integrated
    This is a placeholder function to document potential quantum algorithms
    
    Returns
    -------
    dict
        Dictionary of quantum model implementations
    """
    if not QUANTUM_AVAILABLE:
        return {}
    
    quantum_models = {
        'QKNN': {
            'description': 'Quantum K-Nearest Neighbors algorithm',
            'implementation': 'Uses quantum feature maps and distance calculations'
        },
        'QRF': {
            'description': 'Quantum Random Forest',
            'implementation': 'Ensemble of quantum decision trees with quantum feature encoding'
        },
        'QDT': {
            'description': 'Quantum Decision Tree',
            'implementation': 'Decision tree with quantum feature encoding and quantum splitting criteria'
        },
        'QSVM': {
            'description': 'Quantum Support Vector Machine',
            'implementation': 'SVM with quantum kernel methods'
        },
        'VQC': {
            'description': 'Variational Quantum Classifier',
            'implementation': 'Parameterized quantum circuit for classification'
        },
        'QRNN': {
            'description': 'Quantum Recurrent Neural Network',
            'implementation': 'RNN with quantum cells for sequence processing'
        }
    }
    
    return quantum_models

def get_quantum_circuit_visualization(circuit_type, n_qubits=4, layers=2, data_dimension=None):
    """
    Generate a visualization of a quantum circuit
    
    Parameters
    ----------
    circuit_type : str
        Type of quantum circuit
    n_qubits : int
        Number of qubits
    layers : int
        Number of layers in the circuit
    data_dimension : int
        Dimension of the input data
    
    Returns
    -------
    str
        Base64 encoded image of the circuit or None if PennyLane not available
    """
    if not QUANTUM_AVAILABLE:
        return None
    
    try:
        # Create device
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define circuit based on type
        if circuit_type == 'feature_map':
            @qml.qnode(dev)
            def circuit(x):
                # Encode data
                for i in range(n_qubits):
                    qml.RX(x[i % len(x)], wires=i)
                
                # Apply ZZ feature map
                for l in range(layers):
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            qml.CNOT(wires=[i, j])
                            qml.RZ(x[i % len(x)] * x[j % len(x)], wires=j)
                            qml.CNOT(wires=[i, j])
                
                return qml.expval(qml.PauliZ(0))
            
            # Create sample data
            if data_dimension is None:
                data_dimension = n_qubits
            sample_data = np.random.rand(data_dimension) * 2 * np.pi
            
            # Draw circuit
            fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(sample_data)
            
        elif circuit_type == 'variational':
            @qml.qnode(dev)
            def circuit(x, weights):
                # Encode data
                for i in range(n_qubits):
                    qml.RX(x[i % len(x)], wires=i)
                
                # Apply variational circuit
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                
                return qml.expval(qml.PauliZ(0))
            
            # Create sample data and weights
            if data_dimension is None:
                data_dimension = n_qubits
            sample_data = np.random.rand(data_dimension) * 2 * np.pi
            sample_weights = np.random.rand(layers, n_qubits, 3) * 2 * np.pi
            
            # Draw circuit
            fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(sample_data, sample_weights)
        
        else:
            return None
        
        # Save figure to buffer
        from io import BytesIO
        import base64
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Encode image
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    except Exception as e:
        print(f"Error generating quantum circuit visualization: {str(e)}")
        return None

def deploy_model_as_api(model_path, metadata_path=None, api_key=None, user_id=None, experiment_id=None):
    """
    Prepare a model for API deployment
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
    metadata_path : str
        Path to the model metadata
    api_key : str
        API key for authentication
    user_id : str
        ID of the user
    experiment_id : str
        ID of the experiment
    
    Returns
    -------
    dict
        API deployment information
    """
    try:
        # Load model and metadata
        model = MLModel.load(model_path, metadata_path)
        
        # Generate API endpoint
        import uuid
        endpoint_id = str(uuid.uuid4())
        
        # Create API endpoint URL (placeholder for actual implementation)
        api_endpoint = f"/api/v1/predict/{endpoint_id}"
        
        # Generate API key if not provided
        if api_key is None:
            import secrets
            api_key = secrets.token_urlsafe(32)
        
        # Create deployment info
        deployment_info = {
            'api_endpoint': api_endpoint,
            'api_key': api_key,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'user_id': user_id,
            'experiment_id': experiment_id,
            'model_type': model.model_type,
            'algorithm': model.algorithm,
            'problem_type': model.problem_type,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        # In a real implementation, this would save the model to a serving environment
        # and set up the API endpoints
        
        return deployment_info
    
    except Exception as e:
        return {'error': str(e)}

def get_neural_network_visualization(model, output_path=None):
    """
    Generate a visualization of a neural network architecture
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        The neural network model
    output_path : str
        Path to save the visualization
    
    Returns
    -------
    str
        Base64 encoded image or path to saved image
    """
    if not NEURAL_AVAILABLE:
        return None
    
    try:
        # Plot model architecture
        from tensorflow.keras.utils import plot_model
        
        if output_path:
            # Save visualization to file
            plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True)
            return output_path
        else:
            # Return base64 encoded image
            import io
            import base64
            
            buf = io.BytesIO()
            plot_model(model, to_file=buf, format='png', show_shapes=True, show_layer_names=True)
            buf.seek(0)
            
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
    
    except Exception as e:
        print(f"Error generating neural network visualization: {str(e)}")
        return None

def get_gemini_recommendations(problem_type, data_info):
    """Get model recommendations using Gemini API for the Pro version"""
    try:
        print("In get recomendation.")

        # Try to import the Gemini API client
        from google import genai
        import os
        import re
        import json
        from flask import current_app
        
        # Get API key from environment variable or configuration
        gemini_api_key = os.environ.get('GEMINI_API_KEY') or current_app.config.get('GEMINI_API_KEY')
        print("ApI key: ", gemini_api_key)

        # If no API key, fall back to basic recommendations
        if not gemini_api_key:
            print("Warning: GEMINI_API_KEY not found. Using basic recommendations.")
            return get_model_recommendations(problem_type, data_info)
        
        # Initialize Gemini client
        client = genai.Client(api_key=gemini_api_key)
        print("Client initialized.")

        # Get available algorithms based on problem type
        available_algorithms = {}
        available_classical = get_available_algorithms(problem_type, model_type='classical', is_pro=True)
        available_neural = get_available_algorithms(problem_type, model_type='neural', is_pro=True)
        available_quantum = get_available_algorithms(problem_type, model_type='quantum', is_pro=True)
        print("Available algorithms: ", available_classical, available_neural, available_quantum)
        
        # Prepare the prompt with dataset information and available models
        prompt = f"""
        You're being used as an API in an AutoML platform. Please provide a complete analysis of the dataset provided. Return the response as a JSON object with the following structure:

        {{
          "models": [
            {{
              "algorithm": "Algorithm Name",
              "model_type": "classical/neural/quantum", 
              "confidence": 0.XX,
              "reason": "Reason for recommendation"
            }}
          ],
          "insights": "Text describing dataset insights",
          "preprocessing": "Text describing preprocessing recommendations",
          "expectations": "Text describing performance expectations",
          "feature_importances": [
            {{
              "name": "Feature name",
              "importance": XX
            }}
          ],
          "issues": [
            {{
              "title": "Issue title",
              "description": "Issue description",
              "severity": "High/Medium/Low"
            }}
          ]
        }}

        Dataset characteristics:
        - Problem type: {problem_type}
        - Number of rows: {data_info['num_rows']}
        - Number of columns: {data_info.get('num_cols')}
        - Numeric features: {len(data_info.get('numeric_columns', []))}
        - Categorical features: {len(data_info.get('categorical_columns', []))}

        Available models:
        - Classical: {available_classical}
        - Neural: {available_neural}
        - Quantum: {available_quantum}

        Include EXACTLY 6 model recommendations (3 classical, 2 neural, 1 quantum).
        Feature importance should include the top 5 features.
        Include at least 2 potential issues with their severity.
        """
        
        # Generate response from Gemini
        model = 'gemini-2.0-flash'
        response = client.models.generate_content(model=model, contents=prompt)
        print("Response received.")
        print("Response text: ", response.text)
        
        # Extract the JSON content
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, response.text)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.text
        
        # Debug
        print("Extracted JSON string:", json_str)
        
        # Clean the text to ensure it's valid JSON
        json_str = re.sub(r'[\n\r\t]', '', json_str)
        
        # Parse the JSON response
        try:
            recommendations = json.loads(json_str)
            print("Successfully parsed JSON")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Problematic JSON string: {json_str}")
            return get_model_recommendations(problem_type, data_info)
        
        # Process the models list
        formatted_recommendations = recommendations
        
        # Process models with the correct format for the template
        # The issue is likely here - format each model object correctly
        if 'models' in formatted_recommendations:
            for i, model in enumerate(formatted_recommendations['models']):
                # Convert confidence to percentage score
                if 'confidence' in model:
                    model['score'] = int(model['confidence'] * 100)
                
                # Add UI-specific fields
                if 'model_type' in model:
                    if model['model_type'] == 'classical':
                        model['color_class'] = 'primary'
                        model['icon'] = 'chart-line'
                    elif model['model_type'] == 'neural':
                        model['color_class'] = 'info'
                        model['icon'] = 'brain'
                    elif model['model_type'] == 'quantum':
                        model['color_class'] = 'warning'
                        model['icon'] = 'atom'
                    else:
                        model['color_class'] = 'primary'
                        model['icon'] = 'robot'
                        
                # Ensure name field is present
                if 'algorithm' in model and 'name' not in model:
                    model['name'] = model['algorithm']
                    
        # Set top algorithm if models exist
        if 'models' in formatted_recommendations and formatted_recommendations['models']:
            formatted_recommendations['top_algorithm'] = formatted_recommendations['models'][0]['algorithm']
        
        # Process issues to add severity_class
        if 'issues' in formatted_recommendations and formatted_recommendations['issues']:
            for issue in formatted_recommendations['issues']:
                if 'severity' in issue:
                    if issue['severity'] == 'High':
                        issue['severity_class'] = 'danger'
                    elif issue['severity'] == 'Medium':
                        issue['severity_class'] = 'warning'
                    else:
                        issue['severity_class'] = 'info'
                        
        # Convert feature importance to percentages
        if 'feature_importances' in formatted_recommendations and formatted_recommendations['feature_importances']:
            for feature in formatted_recommendations['feature_importances']:
                if 'importance' in feature:
                    feature['importance'] = int(feature['importance'] * 100)
                    
        return formatted_recommendations
    
    except Exception as e:
        # Log error and fall back to basic recommendations
        print(f"Error using Gemini API: {str(e)}. Falling back to basic recommendations.")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        return get_model_recommendations(problem_type, data_info)

# Fix for the ml_utils.py file - Remove duplicate train_model_async function and improve implementation
# Replace the duplicate train_model_async function in ml_utils.py with this improved version

def train_model_async(model_run_id, algorithm, problem_type, file_path, target_column, 
                     feature_columns, hyperparameters, model_folder, time_limit=3600,
                     model_type=None, notify_completion=True, notify_error=True, is_pro=False):
    """Start model training in a background thread with a time limit for Pro version"""
    from flask import current_app
    import os
    import json
    import traceback
    from datetime import datetime
    import numpy as np
    import threading
    
    # Get app context object before starting thread
    app = current_app._get_current_object()
    
    # Create a separate thread that doesn't use TensorFlow's monitoring features
    def task():
        # Set environment variables for TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
        os.environ['PYTHONUNBUFFERED'] = '1'      # Ensure output is not buffered
        
        # Use a separate thread that doesn't share state with Flask's reloader
        from app import db
        from app.models.experiment import Experiment, ProModelRun
        from app.models.user import User
        from app.utils.email import send_model_completion_notification, send_error_notification
        from app.utils.data_processing import preprocess_data
        from sklearn.model_selection import train_test_split
        
        with app.app_context():
            # Create a new session to avoid DB connection conflicts
            session = db.create_scoped_session()
            
            try:
                # Get model run and related info
                model_run = session.query(ProModelRun).get(model_run_id)
                if not model_run:
                    print(f"Error: Model run {model_run_id} not found in thread")
                    return
                
                experiment = session.query(Experiment).get(model_run.experiment_id)
                user = session.query(User).get(experiment.user_id)
                
                # Update status
                model_run.status = 'training'
                model_run.logs = "Starting model training...\n"
                session.commit()
                
                # Check if file exists
                if not os.path.exists(file_path):
                    # Try to fix file path
                    base_name = os.path.basename(file_path)
                    alternate_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', base_name)
                    if os.path.exists(alternate_path):
                        file_path = alternate_path
                        model_run.logs += f"Fixed file path to {file_path}\n"
                        session.commit()
                    else:
                        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
                
                # Determine model type if not provided
                actual_model_type = model_type
                if actual_model_type is None:
                    actual_model_type = 'quantum' if algorithm.startswith('Q') else \
                               'neural' if algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU'] else \
                               'classical'
                
                # Process the dataset
                model_run.logs += "Preprocessing dataset...\n"
                session.commit()
                
                processed_data = preprocess_data(file_path, target_column, feature_columns)
                X = processed_data['X']
                y = processed_data['y']
                
                # Train the model in a way that's less likely to trigger Flask reloads
                if actual_model_type == 'neural':
                    # For neural models, we'll use a simpler approach to avoid Flask reload issues
                    model_run.logs += "Preparing to train neural network...\n"
                    session.commit()
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                    
                    # Start the training as a standalone process
                    model_run.logs += "Starting neural network training...\n"
                    session.commit()
                    
                    # Define a special training function for neural networks
                    def train_neural_model():
                        from app.utils.ml_utils import MLModel
                        
                        try:
                            # Initialize the model
                            model = MLModel(
                                model_type=actual_model_type,
                                algorithm=algorithm,
                                problem_type=problem_type,
                                params=hyperparameters,
                                is_pro=is_pro,
                                experiment_id=experiment.id,
                                model_run_id=model_run_id
                            )
                            
                            # Create the data dictionary
                            data = {
                                'X_train': X_train,
                                'X_test': X_test,
                                'y_train': y_train,
                                'y_test': y_test,
                                'n_classes': processed_data.get('n_classes', 2)
                            }
                            
                            # Directly call the training method
                            model._train_neural(data, X_test, y_test)
                            
                            # Calculate metrics
                            model._calculate_metrics(data)
                            
                            # Save the model
                            os.makedirs(model_folder, exist_ok=True)
                            model_path = os.path.join(model_folder, f"{model_run_id}_model.pkl")
                            model.save(model_path)
                            
                            # Generate a report if possible
                            try:
                                report_path = model.generate_report(model_folder, experiment.name)
                            except Exception as report_error:
                                print(f"Error generating report: {str(report_error)}")
                                report_path = None
                            
                            # Convert metrics to serializable format
                            def convert_to_serializable(obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, np.bool_):
                                    return bool(obj)
                                return obj
                            
                            serializable_metrics = {}
                            for k, v in model.metrics.items():
                                serializable_metrics[k] = convert_to_serializable(v)
                            
                            # Process feature importances
                            feature_importances_dict = None
                            if model.feature_importances is not None:
                                if len(feature_columns) == len(model.feature_importances):
                                    feature_importances_dict = {
                                        feature: convert_to_serializable(value) 
                                        for feature, value in zip(feature_columns, model.feature_importances)
                                    }
                                else:
                                    feature_importances_dict = {
                                        f"feature_{i}": convert_to_serializable(value) 
                                        for i, value in enumerate(model.feature_importances)
                                    }
                            
                            # Process confusion matrix
                            confusion_matrix_json = None
                            if hasattr(model, 'confusion_matrix_data') and model.confusion_matrix_data is not None:
                                cm_data = model.confusion_matrix_data
                                if hasattr(cm_data, 'tolist'):
                                    cm_data = cm_data.tolist()
                                confusion_matrix_json = json.dumps(cm_data)
                            
                            # Create a results file
                            results_path = os.path.join(model_folder, f"{model_run_id}_results.json")
                            with open(results_path, 'w') as f:
                                json.dump({
                                    'status': 'completed',
                                    'metrics': serializable_metrics,
                                    'feature_importances': feature_importances_dict,
                                    'model_path': model_path,
                                    'report_path': report_path,
                                    'confusion_matrix': confusion_matrix_json,
                                    'classification_report': getattr(model, 'classification_report_data', None)
                                }, f)
                            
                            # Update database in a new session
                            with app.app_context():
                                new_session = db.create_scoped_session()
                                try:
                                    updated_run = new_session.query(ProModelRun).get(model_run_id)
                                    updated_run.status = 'completed'
                                    updated_run.metrics = json.dumps(serializable_metrics)
                                    updated_run.feature_importances = json.dumps(feature_importances_dict) if feature_importances_dict else None
                                    updated_run.model_path = model_path
                                    updated_run.report_path = report_path
                                    updated_run.completed_at = datetime.utcnow()
                                    
                                    if confusion_matrix_json:
                                        updated_run.confusion_matrix = confusion_matrix_json
                                    
                                    if hasattr(model, 'classification_report_data') and model.classification_report_data is not None:
                                        updated_run.classification_report = model.classification_report_data
                                    
                                    updated_run.logs += "Model training completed successfully!\n"
                                    new_session.commit()
                                    
                                    # Send completion notification
                                    if notify_completion:
                                        try:
                                            send_model_completion_notification(
                                                user.email,
                                                user.get_full_name(),
                                                experiment.name,
                                                algorithm,
                                                serializable_metrics
                                            )
                                        except Exception as email_error:
                                            print(f"Error sending completion notification: {str(email_error)}")
                                except Exception as db_error:
                                    print(f"Error updating database: {str(db_error)}")
                                finally:
                                    new_session.close()
                                    
                        except Exception as e:
                            print(f"Error in neural training: {str(e)}")
                            import traceback
                            error_details = traceback.format_exc()
                            
                            # Update database with error
                            with app.app_context():
                                error_session = db.create_scoped_session()
                                try:
                                    error_run = error_session.query(ProModelRun).get(model_run_id)
                                    error_run.status = 'failed'
                                    error_run.logs += f"Model training failed: {str(e)}\n"
                                    error_run.logs += f"Error details: {error_details}\n"
                                    error_run.completed_at = datetime.utcnow()
                                    error_session.commit()
                                    
                                    # Send error notification
                                    if notify_error:
                                        try:
                                            send_error_notification(
                                                user.email,
                                                user.get_full_name(),
                                                experiment.name,
                                                algorithm,
                                                str(e)
                                            )
                                        except Exception as email_error:
                                            print(f"Error sending error notification: {str(email_error)}")
                                except Exception as db_error:
                                    print(f"Error updating database with error: {str(db_error)}")
                                finally:
                                    error_session.close()
                    
                    # Start the neural training in a dedicated thread
                    neural_thread = threading.Thread(target=train_neural_model)
                    neural_thread.daemon = True
                    neural_thread.start()
                    
                    # Return from this function to let Flask continue
                    model_run.logs += "Neural network training process started. You will receive an email when it completes.\n"
                    session.commit()
                    
                else:
                    # For non-neural models, use the regular approach
                    from app.utils.ml_utils import MLModel
                    
                    # Initialize model
                    model = MLModel(
                        model_type=actual_model_type,
                        algorithm=algorithm,
                        problem_type=problem_type,
                        params=hyperparameters,
                        is_pro=is_pro,
                        experiment_id=experiment.id,
                        model_run_id=model_run_id
                    )
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                    
                    # Create the data dictionary
                    data = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test
                    }
                    
                    # Train classical model
                    model_run.logs += "Training model...\n"
                    session.commit()
                    
                    model._train_classical(data)
                    model._calculate_metrics(data)
                    
                    # Save model
                    model_run.logs += "Training completed. Saving model and generating report...\n"
                    session.commit()
                    
                    os.makedirs(model_folder, exist_ok=True)
                    model_path = os.path.join(model_folder, f"{model_run_id}_model.pkl")
                    model.save(model_path)
                    
                    # Generate report
                    report_path = model.generate_report(model_folder, experiment.name)
                    
                    # Helper function to convert numpy values to Python types
                    def convert_to_serializable(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        return obj
                    
                    # Process results
                    serializable_metrics = {}
                    for k, v in model.metrics.items():
                        serializable_metrics[k] = convert_to_serializable(v)
                    
                    feature_importances_dict = None
                    if model.feature_importances is not None:
                        if len(feature_columns) == len(model.feature_importances):
                            feature_importances_dict = {
                                feature: convert_to_serializable(value) 
                                for feature, value in zip(feature_columns, model.feature_importances)
                            }
                        else:
                            feature_importances_dict = {
                                f"feature_{i}": convert_to_serializable(value) 
                                for i, value in enumerate(model.feature_importances)
                            }
                    
                    # Save confusion matrix if available
                    confusion_matrix_json = None
                    if hasattr(model, 'confusion_matrix_data') and model.confusion_matrix_data is not None:
                        cm_data = model.confusion_matrix_data
                        if hasattr(cm_data, 'tolist'):
                            cm_data = cm_data.tolist()
                        confusion_matrix_json = json.dumps(cm_data)
                    
                    # Update model run with results
                    model_run.status = 'completed'
                    model_run.metrics = json.dumps(serializable_metrics)
                    model_run.feature_importances = json.dumps(feature_importances_dict) if feature_importances_dict else None
                    model_run.model_path = model_path
                    model_run.report_path = report_path
                    model_run.completed_at = datetime.utcnow()
                    
                    # Save confusion matrix and classification report if available
                    if confusion_matrix_json:
                        model_run.confusion_matrix = confusion_matrix_json
                    
                    if hasattr(model, 'classification_report_data') and model.classification_report_data is not None:
                        model_run.classification_report = model.classification_report_data
                    
                    model_run.logs += "Model training completed successfully!\n"
                    session.commit()
                    
                    # Send completion notification
                    if notify_completion and hasattr(user, 'notify_on_completion') and user.notify_on_completion:
                        try:
                            send_model_completion_notification(
                                user.email,
                                user.get_full_name(),
                                experiment.name,
                                algorithm,
                                serializable_metrics
                            )
                            model_run.logs += "Sent completion notification email.\n"
                            session.commit()
                        except Exception as email_error:
                            model_run.logs += f"Error sending completion notification: {str(email_error)}\n"
                            session.commit()
                            print(f"Error sending completion notification: {str(email_error)}")
                
            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Training error: {str(e)}\n{error_details}")
                
                # Update with error
                if 'model_run' in locals():
                    model_run.status = 'failed'
                    model_run.logs += f"Model training failed: {str(e)}\n"
                    model_run.logs += f"Error details: {error_details}\n"
                    model_run.completed_at = datetime.utcnow()
                    session.commit()
                
                # Send error notification
                if notify_error and 'user' in locals() and 'experiment' in locals():
                    try:
                        send_error_notification(
                            user.email,
                            user.get_full_name(),
                            experiment.name,
                            algorithm,
                            str(e)
                        )
                        if 'model_run' in locals():
                            model_run.logs += "Sent error notification email.\n"
                            session.commit()
                    except Exception as email_error:
                        if 'model_run' in locals():
                            model_run.logs += f"Error sending error notification: {str(email_error)}\n"
                            session.commit()
                        print(f"Error sending error notification: {str(email_error)}")
            
            finally:
                # Close the session
                session.close()
                print("Training thread complete, session closed")
    
    # Start background thread
    thread = threading.Thread(target=task)
    thread.daemon = True
    thread.start()
    
    # Return a placeholder dictionary that will be updated by the thread
    return {
        'status': 'started',
        'algorithm': algorithm,
        'problem_type': problem_type
    }


def _train_model_impl(model_run_id, algorithm, problem_type, file_path, target_column, feature_columns, hyperparameters, model_folder, time_limit=3600, model_type=None, session=None):
    """Implementation of model training"""
    from datetime import datetime
    import os
    import json
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import time
    
    start_time = time.time()
    results = {}
    
    try:
        # Update logs
        if session:
            from app.models.experiment import ProModelRun
            model_run = session.query(ProModelRun).get(model_run_id)
            model_run.logs += "Preprocessing dataset...\n"
            session.commit()
        
        # Preprocess data
        from app.utils.data_processing import preprocess_data
        processed_data = preprocess_data(file_path, target_column, feature_columns)
        
        # Split data
        X = processed_data['X']
        y = processed_data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Update logs
        if session:
            model_run.logs += "Splitting data into training and test sets...\n"
            session.commit()
        
        # Determine model class and parameters
        if problem_type == 'classification':
            if algorithm == 'RandomForest':
                from sklearn.ensemble import RandomForestClassifier
                model_class = RandomForestClassifier
            elif algorithm == 'LogisticRegression':
                from sklearn.linear_model import LogisticRegression
                model_class = LogisticRegression
            elif algorithm == 'SVC':
                from sklearn.svm import SVC
                model_class = SVC
            elif algorithm == 'DecisionTree':
                from sklearn.tree import DecisionTreeClassifier
                model_class = DecisionTreeClassifier
            elif algorithm == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                model_class = KNeighborsClassifier
            # Neural networks and quantum would go here
            else:
                raise ValueError(f"Unsupported classification algorithm: {algorithm}")
        else:  # regression
            if algorithm == 'RandomForestRegressor':
                from sklearn.ensemble import RandomForestRegressor
                model_class = RandomForestRegressor
            elif algorithm == 'LinearRegression':
                from sklearn.linear_model import LinearRegression
                model_class = LinearRegression
            elif algorithm == 'SVR':
                from sklearn.svm import SVR
                model_class = SVR
            elif algorithm == 'DecisionTreeRegressor':
                from sklearn.tree import DecisionTreeRegressor
                model_class = DecisionTreeRegressor
            # Neural networks and quantum would go here
            else:
                raise ValueError(f"Unsupported regression algorithm: {algorithm}")
        
        # Initialize model
        if session:
            model_run.logs += f"Initializing {algorithm} model...\n"
            session.commit()
        
        model = model_class(**hyperparameters)
        
        # Train model
        if session:
            model_run.logs += f"Training {algorithm} model...\n"
            session.commit()
        
        model.fit(X_train, y_train)
        
        # Check if time limit exceeded
        current_time = time.time()
        if current_time - start_time > time_limit:
            return {'error': 'Training timed out. Maximum allowed time exceeded.'}
        
        # Evaluate model
        if session:
            model_run.logs += "Evaluating model performance...\n"
            session.commit()
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if problem_type == 'classification':
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Generate classification report
            cr = classification_report(y_test, y_pred)
            results['classification_report'] = cr
        else:  # regression
            metrics = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
        
        # Extract feature importances if available
        feature_importances = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, feature in enumerate(feature_columns):
                feature_importances[feature] = float(importances[i]) if i < len(importances) else 0.0
        
        # Save model
        if session:
            model_run.logs += "Saving model...\n"
            session.commit()
        
        os.makedirs(model_folder, exist_ok=True)
        model_filename = f"{model_run_id}_model.pkl"
        model_path = os.path.join(model_folder, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'encoders': processed_data.get('encoders', {}),
                'scaler': processed_data.get('scaler', None),
                'metrics': metrics
            }, f)
        
        # Return results
        results['metrics'] = metrics
        results['model_path'] = model_path
        results['feature_importances'] = feature_importances
        
        return results
    
    except Exception as e:
        print(f"Error in _train_model_impl: {str(e)}")
        return {'error': str(e)}

def get_model_recommendations(problem_type, data_info):
    """Get basic model recommendations based on dataset characteristics"""
    recommendations = []
    
    # Get dataset characteristics
    num_samples = data_info.get('num_rows', 0)
    num_features = len(data_info.get('numeric_columns', [])) + len(data_info.get('categorical_columns', []))
    has_categorical = len(data_info.get('categorical_columns', [])) > 0
    
    if problem_type == 'classification':
        # Classical model recommendations
        recommendations.append({
            'algorithm': 'RandomForest',
            'model_type': 'classical',
            'confidence': 0.92,
            'reason': 'Random Forest typically performs well on a variety of datasets and is less prone to overfitting.'
        })
        
        recommendations.append({
            'algorithm': 'LogisticRegression',
            'model_type': 'classical',
            'confidence': 0.80,
            'reason': 'Logistic Regression provides good performance for linear relationships and high interpretability.'
        })
        
        # Add SVC for smaller datasets
        if num_samples < 10000:
            recommendations.append({
                'algorithm': 'SVC',
                'model_type': 'classical',
                'confidence': 0.78,
                'reason': 'Support Vector Classifier can handle complex decision boundaries and works well on smaller datasets.'
            })
        
        # Add KNN for smaller datasets
        if num_samples < 5000:
            recommendations.append({
                'algorithm': 'KNN',
                'model_type': 'classical',
                'confidence': 0.70,
                'reason': 'K-Nearest Neighbors works well for smaller datasets with meaningful distance metrics.'
            })
            
        # Add Decision Tree
        recommendations.append({
            'algorithm': 'DecisionTree',
            'model_type': 'classical',
            'confidence': 0.65,
            'reason': 'Decision Trees provide interpretable results and handle mixed data types well.'
        })
        
        # Neural network models (pro only)
        if num_features > 5 and num_samples > 1000:
            recommendations.append({
                'algorithm': 'MLP',
                'model_type': 'neural',
                'confidence': 0.85,
                'reason': 'Multi-Layer Perceptron can capture complex patterns and non-linear relationships in the data.'
            })
            
            if num_samples > 5000:
                recommendations.append({
                    'algorithm': 'CNN',
                    'model_type': 'neural',
                    'confidence': 0.75,
                    'reason': 'Convolutional Neural Networks can detect local patterns and work well with structured data.'
                })
                
                recommendations.append({
                    'algorithm': 'LSTM',
                    'model_type': 'neural',
                    'confidence': 0.72,
                    'reason': 'Long Short-Term Memory networks can capture sequential patterns in the data.'
                })
        
        # Quantum models (pro only)
        if num_features < 20 and num_samples < 5000:
            recommendations.append({
                'algorithm': 'QSVM',
                'model_type': 'quantum',
                'confidence': 0.78,
                'reason': 'Quantum Support Vector Machine leverages quantum computing for potentially improved classification boundaries.'
            })
            
            recommendations.append({
                'algorithm': 'QNN',
                'model_type': 'quantum',
                'confidence': 0.76,
                'reason': 'Quantum Neural Network combines quantum computing with neural network principles for complex pattern recognition.'
            })
    
    elif problem_type == 'regression':
        # Classical model recommendations
        recommendations.append({
            'algorithm': 'RandomForest',
            'model_type': 'classical',
            'confidence': 0.90,
            'reason': 'Random Forest Regressor handles non-linear relationships and interactions between features effectively.'
        })
        
        # Add Linear Regression for smaller feature sets
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
                'confidence': 0.78,
                'reason': 'Support Vector Regression can handle complex non-linear relationships.'
            })
            
        # Add Decision Tree Regressor
        recommendations.append({
            'algorithm': 'DecisionTree',
            'model_type': 'classical',
            'confidence': 0.70,
            'reason': 'Decision Trees provide interpretable results and handle mixed data types well.'
        })
        
        # Neural network models (pro only)
        if num_features > 5 and num_samples > 1000:
            recommendations.append({
                'algorithm': 'MLP',
                'model_type': 'neural',
                'confidence': 0.82,
                'reason': 'Multi-Layer Perceptron can model complex non-linear relationships for regression problems.'
            })
            
            if num_samples > 5000:
                recommendations.append({
                    'algorithm': 'RNN',
                    'model_type': 'neural',
                    'confidence': 0.75,
                    'reason': 'Recurrent Neural Networks are effective for time series or sequential regression problems.'
                })
                
                recommendations.append({
                    'algorithm': 'GRU',
                    'model_type': 'neural',
                    'confidence': 0.72,
                    'reason': 'Gated Recurrent Units can capture temporal dependencies efficiently.'
                })
        
        # Quantum models (pro only)
        if num_features < 20 and num_samples < 5000:
            recommendations.append({
                'algorithm': 'QKE',
                'model_type': 'quantum',
                'confidence': 0.75,
                'reason': 'Quantum Kernel Estimator can potentially improve regression accuracy on complex data.'
            })
            
            recommendations.append({
                'algorithm': 'QRegression',
                'model_type': 'quantum',
                'confidence': 0.72,
                'reason': 'Quantum Regression leverages quantum computing principles for potentially improved predictions.'
            })
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Mark models that require pro subscription
    for rec in recommendations:
        if rec['model_type'] == 'quantum':
            rec['pro_only'] = True
        elif rec['model_type'] == 'neural' and rec['algorithm'] not in ['MLP', 'FNN']:
            rec['pro_only'] = True
    
    return recommendations

# Add these functions to ml_utils.py to fix JSON serialization issues

def safe_json_serialize(obj):
    """
    Convert objects to JSON-serializable types
    
    Parameters
    ----------
    obj : any
        Object to convert
    
    Returns
    -------
    any
        JSON-serializable version of the object
    """
    import numpy as np
    from datetime import datetime
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        try:
            return {k: safe_json_serialize(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
        except:
            return str(obj)
    return obj

def serialize_metrics(metrics):
    """
    Safely serialize metrics for JSON storage
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics from model evaluation
    
    Returns
    -------
    dict
        Serializable metrics
    """
    if not metrics:
        return {}
        
    serializable_metrics = {}
    for key, value in metrics.items():
        # Skip non-serializable objects or functions
        if callable(value) or key.startswith('_'):
            continue
            
        # Convert to serializable format
        serializable_metrics[key] = safe_json_serialize(value)
    
    return serializable_metrics

def serialize_feature_importances(feature_importances, feature_names=None):
    """
    Safely serialize feature importances with feature names
    
    Parameters
    ----------
    feature_importances : array-like
        Feature importance values
    feature_names : list, optional
        Names of features
    
    Returns
    -------
    dict
        Feature importances with names as keys
    """
    if feature_importances is None:
        return None
        
    # Convert to list if it's a numpy array
    if hasattr(feature_importances, 'tolist'):
        importances = feature_importances.tolist()
    else:
        importances = list(feature_importances)
    
    # Match with feature names
    if feature_names and len(feature_names) == len(importances):
        # Create dictionary with feature names as keys
        result = {str(name): float(value) for name, value in zip(feature_names, importances)}
    else:
        # Use generic feature names
        result = {f"feature_{i}": float(value) for i, value in enumerate(importances)}
    
    return result

    