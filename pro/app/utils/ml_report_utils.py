# pro/app/utils/ml_report_utils.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import seaborn as sns
from datetime import datetime
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Try to import pennylane for quantum circuit visualization
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: PennyLane not installed. Quantum circuit visualization will not be available.")

def generate_model_report(model_run_id, algorithm, problem_type, metrics, feature_importances, 
                          confusion_matrix_data, classification_report_data, output_folder, 
                          tuning_report_path=None, experiment_name=None, quantum_circuit=None):
    """
    Generate a comprehensive PDF report for the trained model
    
    Parameters:
    -----------
    model_run_id : str
        ID of the model run
    algorithm : str
        Name of the algorithm
    problem_type : str
        'classification' or 'regression'
    metrics : dict
        Performance metrics for the model
    feature_importances : dict or None
        Feature importances if available (feature name -> importance score)
    confusion_matrix_data : numpy array or None
        Confusion matrix for classification models
    classification_report_data : dict or None
        Classification report for classification models
    output_folder : str
        Folder to save the report
    tuning_report_path : str, optional
        Path to the tuning report if exists
    experiment_name : str, optional
        Name of the experiment
    quantum_circuit : dict, optional
        Quantum circuit configuration for visualization
        
    Returns:
    --------
    str : Path to the generated report
    """
    # Create the output filename
    report_filename = f"{model_run_id}_model_report.pdf"
    report_path = os.path.join(output_folder, report_filename)
    
    # Create a PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create a custom style for code/parameters
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
    )
    
    # Content elements
    elements = []
    
    # Title
    title = f"Model Report: {algorithm}"
    elements.append(Paragraph(title, title_style))
    
    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if experiment_name:
        elements.append(Paragraph(f"Experiment: {experiment_name}", normal_style))
    elements.append(Paragraph(f"Problem Type: {problem_type.capitalize()}", normal_style))
    elements.append(Paragraph(f"Generated: {timestamp}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Model Performance Section
    elements.append(Paragraph("Model Performance", heading_style))
    
    # Create table data for metrics
    metrics_data = [["Metric", "Value"]]
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics_data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
    
    # Create table for metrics
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Feature Importances Section (if available)
    if feature_importances and len(feature_importances) > 0:
        elements.append(Paragraph("Feature Importances", heading_style))
        
        # Create plot for feature importances
        fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
        
        # Sort importances and limit to top 15 if there are many
        importance_df = pd.DataFrame({
            'feature': list(feature_importances.keys()),
            'importance': list(feature_importances.values())
        }).sort_values('importance', ascending=False)
        
        if len(importance_df) > 15:
            importance_df = importance_df.head(15)
            subtitle = "Top 15 features shown"
        else:
            subtitle = ""
        
        # Plot horizontal bar chart
        importance_df = importance_df.sort_values('importance')  # Sort ascending for bottom-to-top display
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')
        if subtitle:
            ax.text(0.5, 0.01, subtitle, transform=ax.transAxes, ha='center', fontsize=8)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        
        # Add plot to report
        img = Image(buf, width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.25*inch))
        
        plt.close()
    
    # Classification specific sections
    if problem_type == 'classification' and confusion_matrix_data is not None:
        # Confusion Matrix
        elements.append(Paragraph("Confusion Matrix", heading_style))
        
        # Plot confusion matrix
        fig, ax = plt.figure(figsize=(6, 5)), plt.gca()
        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        
        # Add plot to report
        img = Image(buf, width=5*inch, height=4.5*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.1*inch))
        
        plt.close()
        
        # Classification Report
        if classification_report_data:
            elements.append(Paragraph("Classification Report", heading_style))
            report_text = classification_report_data.replace(' ', '&nbsp;')
            report_text = report_text.replace('\n', '<br/>')
            p = Paragraph(f"<font face='Courier'>{report_text}</font>", normal_style)
            elements.append(p)
            elements.append(Spacer(1, 0.25*inch))
    
    # Quantum Circuit Visualization (if available)
    if quantum_circuit and QUANTUM_AVAILABLE:
        elements.append(Paragraph("Quantum Circuit", heading_style))
        
        try:
            # Extract circuit parameters
            n_qubits = quantum_circuit.get('qubits', 4)
            n_layers = quantum_circuit.get('layers', 2)
            
            # Create a simple circuit representation based on the algorithm
            if algorithm in ['QSVM', 'QuantumKernel']:
                elements.append(Paragraph("Quantum Feature Map Circuit", heading2_style))
                
                # Generate a sample feature map circuit
                feature_map_type = quantum_circuit.get('feature_map', 'ZZFeatureMap')
                elements.append(Paragraph(f"Feature Map Type: {feature_map_type}", normal_style))
                
                # Create visualization
                fig, ax = plt.figure(figsize=(7, n_qubits * 0.7)), plt.gca()
                
                # Create a simple circuit with the specified feature map
                dev = qml.device("default.qubit", wires=n_qubits)
                
                @qml.qnode(dev)
                def circuit(x):
                    # Basic encoding: Angle encoding for each feature to a qubit
                    for i in range(n_qubits):
                        qml.RX(x[i % len(x)], wires=i)
                    
                    # Add entanglement based on feature map type
                    if feature_map_type == 'ZZFeatureMap':
                        # Add ZZ entanglement
                        for i in range(n_qubits):
                            for j in range(i+1, n_qubits):
                                qml.CNOT(wires=[i, j])
                                qml.RZ(x[i % len(x)] * x[j % len(x)], wires=j)
                                qml.CNOT(wires=[i, j])
                    
                    return qml.expval(qml.PauliZ(0))
                
                # Create some sample data for visualization
                sample_data = np.random.rand(n_qubits) * 2 * np.pi
                circuit(sample_data)
                
                # Draw the circuit
                fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(sample_data)
                
                # Save figure to buffer
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                # Add plot to report
                img = Image(buf, width=6*inch, height=n_qubits * 0.7 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                
                plt.close()
            
            elif algorithm in ['QNN', 'QRegression', 'QRNN', 'QLSTM']:
                elements.append(Paragraph("Quantum Neural Network Circuit", heading2_style))
                
                # Create visualization
                fig, ax = plt.figure(figsize=(7, n_qubits * 0.7)), plt.gca()
                
                # Create a simple circuit with parametrized gates
                dev = qml.device("default.qubit", wires=n_qubits)
                
                @qml.qnode(dev)
                def circuit(inputs, weights):
                    # Encode inputs
                    for i in range(n_qubits):
                        qml.RX(inputs[i % len(inputs)], wires=i)
                    
                    # Parametrized quantum layers
                    for layer in range(n_layers):
                        for qubit in range(n_qubits):
                            qml.Rot(*weights[layer, qubit], wires=qubit)
                        
                        # Add entanglement
                        for qubit in range(n_qubits - 1):
                            qml.CNOT(wires=[qubit, qubit + 1])
                        qml.CNOT(wires=[n_qubits - 1, 0])  # Connect last to first
                    
                    return qml.expval(qml.PauliZ(0))
                
                # Create some sample data for visualization
                sample_inputs = np.random.rand(n_qubits) * 2 * np.pi
                sample_weights = np.random.rand(n_layers, n_qubits, 3) * 2 * np.pi
                
                # Draw the circuit
                circuit(sample_inputs, sample_weights)
                fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(sample_inputs, sample_weights)
                
                # Save figure to buffer
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                # Add plot to report
                img = Image(buf, width=6*inch, height=n_qubits * 0.7 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                
                plt.close()
            
            # Add quantum circuit parameters
            elements.append(Paragraph("Quantum Circuit Parameters", heading2_style))
            
            # Create table data for quantum parameters
            quantum_params_data = [["Parameter", "Value"]]
            for param, value in quantum_circuit.items():
                if param not in ['model_type', 'class']:
                    quantum_params_data.append([param, str(value)])
            
            # Create table for quantum parameters
            quantum_params_table = Table(quantum_params_data, colWidths=[2.5*inch, 3*inch])
            quantum_params_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(quantum_params_table)
            elements.append(Spacer(1, 0.25*inch))
            
        except Exception as e:
            elements.append(Paragraph(f"Error generating quantum circuit visualization: {str(e)}", normal_style))
    
    # Neural Network Architecture (if applicable)
    if algorithm in ['MLP', 'FNN', 'RNN', 'GRU', 'LSTM', 'GNN']:
        elements.append(Paragraph("Neural Network Architecture", heading_style))
        
        try:
            # Extract model architecture from the algorithm
            if 'model_config' in metrics:
                model_config = metrics['model_config']
                
                # Network topology visualization
                elements.append(Paragraph("Network Topology", heading2_style))
                
                # Visualize a simple network topology based on the architecture
                if algorithm in ['MLP', 'FNN']:
                    # Get layer sizes
                    layer_sizes = model_config.get('layers', [64, 32, 1])
                    
                    # Create a simple visualization
                    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
                    
                    # Draw network diagram
                    left = 0.1
                    right = 0.9
                    bottom = 0.1
                    top = 0.9
                    
                    # Compute positions
                    n_layers = len(layer_sizes)
                    v_spacing = (top - bottom) / max(layer_sizes)
                    h_spacing = (right - left) / (n_layers - 1)
                    
                    # Input layer
                    layer_top = top - (layer_sizes[0] - 1) * v_spacing / 2
                    for i in range(layer_sizes[0]):
                        pos = layer_top - i * v_spacing
                        circle = plt.Circle((left, pos), v_spacing/4, fill=True, color='skyblue')
                        ax.add_artist(circle)
                    
                    # Hidden layers and connections
                    for l in range(1, n_layers):
                        layer_top = top - (layer_sizes[l] - 1) * v_spacing / 2
                        for i in range(layer_sizes[l]):
                            pos = layer_top - i * v_spacing
                            circle = plt.Circle((left + l * h_spacing, pos), v_spacing/4, fill=True, color='skyblue')
                            ax.add_artist(circle)
                            
                            # Add some random connections for visualization
                            layer_top_prev = top - (layer_sizes[l-1] - 1) * v_spacing / 2
                            for j in range(layer_sizes[l-1]):
                                if np.random.rand() < 0.3:  # Add only some connections for clarity
                                    prev_pos = layer_top_prev - j * v_spacing
                                    ax.plot([left + (l-1) * h_spacing, left + l * h_spacing],
                                          [prev_pos, pos], 'gray', alpha=0.3)
                    
                    # Add layer labels
                    plt.text(left, top + v_spacing/4, f'Input Layer\n{layer_sizes[0]} neurons', 
                           horizontalalignment='center')
                    
                    for l in range(1, n_layers-1):
                        plt.text(left + l * h_spacing, top + v_spacing/4, 
                               f'Hidden Layer {l}\n{layer_sizes[l]} neurons', 
                               horizontalalignment='center')
                    
                    plt.text(left + (n_layers-1) * h_spacing, top + v_spacing/4, 
                           f'Output Layer\n{layer_sizes[-1]} neurons', 
                           horizontalalignment='center')
                    
                    # Set limits and remove axes
                    ax.set_xlim(left - v_spacing, right + v_spacing)
                    ax.set_ylim(bottom - v_spacing, top + v_spacing)
                    ax.axis('off')
                    plt.title(f'{algorithm} Architecture')
                    
                    # Save figure to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Add plot to report
                    img = Image(buf, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.25*inch))
                    
                    plt.close()
                
                elif algorithm in ['RNN', 'GRU', 'LSTM']:
                    # Recurrent network visualization
                    elements.append(Paragraph(f"{algorithm} Unfolded View", normal_style))
                    
                    # Create a simple RNN/LSTM/GRU illustration
                    fig, ax = plt.figure(figsize=(8, 4)), plt.gca()
                    
                    # Parameters
                    time_steps = 4
                    h_spacing = 0.2
                    v_spacing = 0.3
                    
                    # Draw unfolded RNN/LSTM/GRU
                    for t in range(time_steps):
                        # Cell
                        rect = plt.Rectangle((t * h_spacing, 0.5), 0.15, 0.25, 
                                           edgecolor='blue', facecolor='skyblue', alpha=0.8)
                        ax.add_patch(rect)
                        
                        # Input
                        arrow = plt.Arrow(t * h_spacing + 0.075, 0.3, 0, 0.2, width=0.05, color='black')
                        ax.add_patch(arrow)
                        
                        # Output
                        arrow = plt.Arrow(t * h_spacing + 0.075, 0.75, 0, 0.2, width=0.05, color='black')
                        ax.add_patch(arrow)
                        
                        # Recurrent connection (if not the last time step)
                        if t < time_steps - 1:
                            arrow = plt.Arrow(t * h_spacing + 0.15, 0.625, h_spacing - 0.15, 0, 
                                            width=0.05, color='red')
                            ax.add_patch(arrow)
                        
                        # Labels
                        plt.text(t * h_spacing + 0.075, 0.2, f'$x_{t}$', horizontalalignment='center')
                        plt.text(t * h_spacing + 0.075, 0.95, f'$h_{t}$', horizontalalignment='center')
                        plt.text(t * h_spacing + 0.075, 0.625, algorithm, horizontalalignment='center')
                    
                    # Set limits and remove axes
                    ax.set_xlim(-0.1, time_steps * h_spacing + 0.2)
                    ax.set_ylim(0, 1.2)
                    ax.axis('off')
                    plt.title(f'{algorithm} Unfolded Through Time')
                    
                    # Save figure to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Add plot to report
                    img = Image(buf, width=6*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.25*inch))
                    
                    plt.close()
                
                elif algorithm == 'GNN':
                    # Graph Neural Network visualization
                    elements.append(Paragraph("Graph Neural Network", normal_style))
                    
                    # Create a simple GNN illustration
                    fig, ax = plt.figure(figsize=(6, 6)), plt.gca()
                    
                    # Create a random graph for visualization
                    n_nodes = 10
                    pos = {i: (np.cos(2 * np.pi * i / n_nodes), np.sin(2 * np.pi * i / n_nodes)) 
                         for i in range(n_nodes)}
                    
                    # Draw nodes
                    for i, (x, y) in pos.items():
                        circle = plt.Circle((x, y), 0.1, fill=True, color='skyblue', edgecolor='blue')
                        ax.add_artist(circle)
                        plt.text(x, y, str(i), horizontalalignment='center', verticalalignment='center')
                    
                    # Draw edges (randomly)
                    for i in range(n_nodes):
                        for j in range(i+1, n_nodes):
                            if np.random.rand() < 0.3:
                                x1, y1 = pos[i]
                                x2, y2 = pos[j]
                                plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)
                    
                    # Set limits and remove axes
                    ax.set_xlim(-1.2, 1.2)
                    ax.set_ylim(-1.2, 1.2)
                    ax.axis('off')
                    plt.title('Graph Neural Network Structure')
                    
                    # Save figure to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Add plot to report
                    img = Image(buf, width=4*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.25*inch))
                    
                    plt.close()
                
                # Model Parameters Table
                elements.append(Paragraph("Model Parameters", heading2_style))
                
                # Create table data for neural network parameters
                nn_params_data = [["Parameter", "Value"]]
                for param, value in model_config.items():
                    nn_params_data.append([param, str(value)])
                
                # Create table for neural network parameters
                nn_params_table = Table(nn_params_data, colWidths=[2.5*inch, 3*inch])
                nn_params_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                elements.append(nn_params_table)
                elements.append(Spacer(1, 0.25*inch))
                
                # Training History (if available)
                if 'history' in metrics:
                    history = metrics['history']
                    elements.append(Paragraph("Training History", heading2_style))
                    
                    # Plot training history
                    fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
                    
                    # Plot loss
                    if 'loss' in history:
                        epochs = range(1, len(history['loss']) + 1)
                        plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
                        
                        # Plot validation loss if available
                        if 'val_loss' in history:
                            plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
                    
                    plt.title('Training and Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save figure to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Add plot to report
                    img = Image(buf, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                    
                    plt.close()
                    
                    # Plot accuracy if it's a classification problem
                    if problem_type == 'classification' and 'accuracy' in history:
                        fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
                        
                        epochs = range(1, len(history['accuracy']) + 1)
                        plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
                        
                        # Plot validation accuracy if available
                        if 'val_accuracy' in history:
                            plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
                        
                        plt.title('Training and Validation Accuracy')
                        plt.xlabel('Epochs')
                        plt.ylabel('Accuracy')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        # Save figure to buffer
                        buf = BytesIO()
                        plt.savefig(buf, format='png', dpi=300)
                        buf.seek(0)
                        
                        # Add plot to report
                        img = Image(buf, width=6*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 0.25*inch))
                        
                        plt.close()
            else:
                elements.append(Paragraph("Neural network architecture information not available.", normal_style))
        
        except Exception as e:
            elements.append(Paragraph(f"Error generating neural network architecture visualization: {str(e)}", normal_style))
    
    # If this model had hyperparameter tuning, mention the separate report
    if tuning_report_path:
        elements.append(Paragraph("Hyperparameter Tuning", heading_style))
        elements.append(Paragraph("This model was trained after hyperparameter tuning. A separate tuning report is available with detailed information about the tuning process and results.", normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path

def visualize_training_history(history, output_path=None, show_plot=False):
    """
    Visualize training history from a model's training process
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history, with keys like 'loss', 'accuracy', 'val_loss', etc.
    output_path : str, optional
        Path to save the visualization (if None, the plot is not saved)
    show_plot : bool, default=False
        Whether to display the plot (should be False for server environment)
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure object
    """
    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Plot training & validation loss
    if 'loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss')
        
        # Plot validation loss if available
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot training & validation metrics (accuracy or others)
    metric_found = False
    for metric in ['accuracy', 'f1', 'auc', 'precision', 'recall']:
        if metric in history:
            metric_found = True
            epochs = range(1, len(history[metric]) + 1)
            axes[1].plot(epochs, history[metric], 'b-', label=f'Training {metric.capitalize()}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                axes[1].plot(epochs, history[val_metric], 'r-', label=f'Validation {metric.capitalize()}')
            
            axes[1].set_title(f'Training and Validation {metric.capitalize()}')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel(metric.capitalize())
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.7)
            break  # Only plot the first metric found
    
    # If no metrics found, remove the second subplot
    if not metric_found:
        fig.delaxes(axes[1])
        plt.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    
    return fig

def create_model_card(model_info, output_path=None, format='markdown'):
    """
    Generate a model card for the trained model, documenting its uses, limitations, and ethical considerations
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing information about the model, including:
        - name: Name of the model
        - version: Version of the model
        - description: Short description of the model
        - algorithm: Algorithm used
        - problem_type: 'classification' or 'regression'
        - metrics: Performance metrics
        - limitations: List of limitations
        - ethical_considerations: List of ethical considerations
        - intended_uses: List of intended uses
        - creator: Name of the model creator
    output_path : str, optional
        Path to save the model card (if None, the model card is returned as a string)
    format : str, default='markdown'
        Format of the model card ('markdown' or 'html')
        
    Returns:
    --------
    str : Model card content or path to the saved model card
    """
    # Default values if not provided
    model_info.setdefault('name', 'Unnamed Model')
    model_info.setdefault('version', '1.0.0')
    model_info.setdefault('description', 'No description provided')
    model_info.setdefault('algorithm', 'Unknown')
    model_info.setdefault('problem_type', 'Unknown')
    model_info.setdefault('metrics', {})
    model_info.setdefault('limitations', [])
    model_info.setdefault('ethical_considerations', [])
    model_info.setdefault('intended_uses', [])
    model_info.setdefault('creator', 'Unknown')
    
    # Create model card content in Markdown format
    if format.lower() == 'markdown':
        content = f"""# Model Card: {model_info['name']}

## Model Details

- **Name:** {model_info['name']}
- **Version:** {model_info['version']}
- **Type:** {model_info['problem_type'].capitalize()}
- **Algorithm:** {model_info['algorithm']}
- **Created by:** {model_info['creator']}
- **Created on:** {datetime.now().strftime("%Y-%m-%d")}

## Model Description

{model_info['description']}

## Performance Metrics

"""
        # Add metrics
        for metric, value in model_info['metrics'].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                content += f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n"
        
        # Add intended uses
        content += "\n## Intended Uses\n\n"
        if model_info['intended_uses']:
            for use in model_info['intended_uses']:
                content += f"- {use}\n"
        else:
            content += "- Not specified\n"
        
        # Add limitations
        content += "\n## Limitations\n\n"
        if model_info['limitations']:
            for limitation in model_info['limitations']:
                content += f"- {limitation}\n"
        else:
            content += "- Not specified\n"
        
        # Add ethical considerations
        content += "\n## Ethical Considerations\n\n"
        if model_info['ethical_considerations']:
            for consideration in model_info['ethical_considerations']:
                content += f"- {consideration}\n"
        else:
            content += "- Not specified\n"
    
    elif format.lower() == 'html':
        # Create HTML content (simplified for illustration)
        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Card: {model_info['name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333366; }}
        .section {{ margin-bottom: 20px; }}
        .metrics {{ display: table; }}
        .metric {{ display: table-row; }}
        .metric-name {{ display: table-cell; font-weight: bold; padding-right: 10px; }}
        .metric-value {{ display: table-cell; }}
        ul {{ margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>Model Card: {model_info['name']}</h1>
    
    <div class="section">
        <h2>Model Details</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-name">Name:</div>
                <div class="metric-value">{model_info['name']}</div>
            </div>
            <div class="metric">
                <div class="metric-name">Version:</div>
                <div class="metric-value">{model_info['version']}</div>
            </div>
            <div class="metric">
                <div class="metric-name">Type:</div>
                <div class="metric-value">{model_info['problem_type'].capitalize()}</div>
            </div>
            <div class="metric">
                <div class="metric-name">Algorithm:</div>
                <div class="metric-value">{model_info['algorithm']}</div>
            </div>
            <div class="metric">
                <div class="metric-name">Created by:</div>
                <div class="metric-value">{model_info['creator']}</div>
            </div>
            <div class="metric">
                <div class="metric-name">Created on:</div>
                <div class="metric-value">{datetime.now().strftime("%Y-%m-%d")}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Model Description</h2>
        <p>{model_info['description']}</p>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics">
"""
        # Add metrics
        for metric, value in model_info['metrics'].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                content += f"""            <div class="metric">
                <div class="metric-name">{metric.replace('_', ' ').title()}:</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
"""
        
        content += """        </div>
    </div>
    
    <div class="section">
        <h2>Intended Uses</h2>
        <ul>
"""
        # Add intended uses
        if model_info['intended_uses']:
            for use in model_info['intended_uses']:
                content += f"            <li>{use}</li>\n"
        else:
            content += "            <li>Not specified</li>\n"
        
        content += """        </ul>
    </div>
    
    <div class="section">
        <h2>Limitations</h2>
        <ul>
"""
        # Add limitations
        if model_info['limitations']:
            for limitation in model_info['limitations']:
                content += f"            <li>{limitation}</li>\n"
        else:
            content += "            <li>Not specified</li>\n"
        
        content += """        </ul>
    </div>
    
    <div class="section">
        <h2>Ethical Considerations</h2>
        <ul>
"""
        # Add ethical considerations
        if model_info['ethical_considerations']:
            for consideration in model_info['ethical_considerations']:
                content += f"            <li>{consideration}</li>\n"
        else:
            content += "            <li>Not specified</li>\n"
        
        content += """        </ul>
    </div>
</body>
</html>
"""
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'markdown' or 'html'.")
    
    # Save the model card if output_path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path
    
    # Otherwise, return the content
    return content

def export_model_predictions(y_true, y_pred, feature_names=None, X=None, output_path=None, index=None):
    """
    Export model predictions to a CSV file with actual and predicted values
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    feature_names : list, optional
        Names of the features
    X : array-like, optional
        Feature matrix (needed if feature_names are provided)
    output_path : str, optional
        Path to save the CSV file (if None, a DataFrame is returned)
    index : array-like, optional
        Index for the DataFrame (e.g., IDs or timestamps)
        
    Returns:
    --------
    pandas.DataFrame or str : DataFrame with predictions or path to the saved CSV file
    """
    # Create a DataFrame with predictions
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred
    })
    
    # Set index if provided
    if index is not None:
        df.index = index
    
    # Add features if provided
    if feature_names is not None and X is not None:
        for i, name in enumerate(feature_names):
            df[f'feature_{name}'] = X[:, i]
    
    # Calculate error
    if np.issubdtype(df['actual'].dtype, np.number) and np.issubdtype(df['predicted'].dtype, np.number):
        df['error'] = df['predicted'] - df['actual']
        df['abs_error'] = np.abs(df['error'])
    
    # Save to CSV if output_path is provided
    if output_path:
        df.to_csv(output_path, index=True)
        return output_path
    
    # Otherwise, return the DataFrame
    return df

def generate_comparison_report(models_data, output_folder, experiment_name=None):
    """
    Generate a comparison report for multiple models
    
    Parameters:
    -----------
    models_data : list of dicts
        Each dict contains information about a model:
        - model_run_id: ID of the model run
        - algorithm: Name of the algorithm
        - metrics: Performance metrics
        - feature_importances: Feature importances (optional)
    output_folder : str
        Folder to save the report
    experiment_name : str, optional
        Name of the experiment
        
    Returns:
    --------
    str : Path to the generated report
    """
    # Ensure we have at least 2 models to compare
    if len(models_data) < 2:
        raise ValueError("At least 2 models are required for comparison")
    
    # Get problem type from the first model (assume all models are the same type)
    problem_type = models_data[0].get('problem_type', 'classification')
    
    # Create the output filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_filename = f"model_comparison_{timestamp}.pdf"
    report_path = os.path.join(output_folder, report_filename)
    
    # Create a PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Content elements
    elements = []
    
    # Title
    title = "Model Comparison Report"
    elements.append(Paragraph(title, title_style))
    
    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if experiment_name:
        elements.append(Paragraph(f"Experiment: {experiment_name}", normal_style))
    elements.append(Paragraph(f"Problem Type: {problem_type.capitalize()}", normal_style))
    elements.append(Paragraph(f"Generated: {timestamp}", normal_style))
    elements.append(Paragraph(f"Models Compared: {len(models_data)}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Performance Comparison Table
    elements.append(Paragraph("Performance Comparison", heading_style))
    
    # Get all unique metrics across all models
    all_metrics = set()
    for model_data in models_data:
        metrics = model_data.get('metrics', {})
        for metric in metrics:
            if isinstance(metrics[metric], (int, float)) and not isinstance(metrics[metric], bool):
                all_metrics.add(metric)
    
    # Create table data for metrics comparison
    metrics_data = [["Metric"] + [model_data.get('algorithm', 'Unknown') for model_data in models_data]]
    
    for metric in sorted(all_metrics):
        row = [metric.replace('_', ' ').title()]
        for model_data in models_data:
            metrics = model_data.get('metrics', {})
            value = metrics.get(metric, "N/A")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row.append(f"{value:.4f}")
            else:
                row.append("N/A")
        metrics_data.append(row)
    
    # Create table for metrics comparison
    col_width = 4.5*inch / len(models_data)
    metrics_table = Table(metrics_data, colWidths=[2*inch] + [col_width] * len(models_data))
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Bar chart comparing key metrics
    elements.append(Paragraph("Performance Visualization", heading_style))
    
    # Select top metrics to visualize (limit to 5)
    metrics_to_plot = []
    if problem_type == 'classification':
        priority_metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    else:  # regression
        priority_metrics = ['r2', 'mse', 'rmse', 'mae', 'mape']
    
    for metric in priority_metrics:
        if metric in all_metrics:
            metrics_to_plot.append(metric)
        
        if len(metrics_to_plot) >= 5:
            break
    
    if not metrics_to_plot and all_metrics:
        # If none of the priority metrics found, use the first 5 metrics
        metrics_to_plot = list(sorted(all_metrics))[:5]
    
    if metrics_to_plot:
        # Create bar chart for each metric
        for metric in metrics_to_plot:
            fig, ax = plt.figure(figsize=(8, 4)), plt.gca()
            
            algorithms = []
            values = []
            
            for model_data in models_data:
                algo = model_data.get('algorithm', 'Unknown')
                algorithms.append(algo)
                
                metrics = model_data.get('metrics', {})
                value = metrics.get(metric, 0)
                values.append(value if isinstance(value, (int, float)) else 0)
            
            # Plot bar chart
            bars = ax.bar(algorithms, values, width=0.6, alpha=0.7, color='skyblue')
            
            # Add value labels on top of bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)
            
            ax.set_xlabel('Algorithm')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save figure to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            
            # Add plot to report
            img = Image(buf, width=6*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
            
            plt.close()
    
    # Feature Importance Comparison (if available)
    models_with_importances = [model_data for model_data in models_data 
                              if model_data.get('feature_importances') and len(model_data['feature_importances']) > 0]
    
    if models_with_importances:
        elements.append(Paragraph("Feature Importance Comparison", heading_style))
        
        # Get all unique features across all models
        all_features = set()
        for model_data in models_with_importances:
            all_features.update(model_data['feature_importances'].keys())
        
        # Create table data for feature importance comparison
        importances_data = [["Feature"] + [model_data.get('algorithm', 'Unknown') for model_data in models_with_importances]]
        
        for feature in sorted(all_features):
            row = [feature]
            for model_data in models_with_importances:
                importances = model_data.get('feature_importances', {})
                value = importances.get(feature, "N/A")
                if isinstance(value, (int, float)):
                    row.append(f"{value:.4f}")
                else:
                    row.append("N/A")
            importances_data.append(row)
        
        # Create table for feature importance comparison
        col_width = 4.5*inch / len(models_with_importances)
        importances_table = Table(importances_data, colWidths=[2*inch] + [col_width] * len(models_with_importances))
        importances_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(importances_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Create visualization of top features across models
        elements.append(Paragraph("Top Features Across Models", heading2_style))
        
        # Combine importances from all models
        combined_importances = {}
        for feature in all_features:
            feature_values = []
            for model_data in models_with_importances:
                importances = model_data.get('feature_importances', {})
                value = importances.get(feature, 0)
                if isinstance(value, (int, float)):
                    feature_values.append(value)
            
            if feature_values:
                combined_importances[feature] = sum(feature_values) / len(feature_values)
        
        # Sort and limit to top 15
        top_features = sorted(combined_importances.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_features:
            # Create plot
            fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
            
            feature_names = [feature for feature, _ in top_features]
            importance_values = [value for _, value in top_features]
            
            # Reverse order for bottom-to-top display in horizontal bar chart
            feature_names.reverse()
            importance_values.reverse()
            
            # Plot horizontal bar chart
            ax.barh(feature_names, importance_values)
            ax.set_xlabel('Average Importance')
            ax.set_title('Top Features Across Models')
            plt.tight_layout()
            
            # Save figure to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            
            # Add plot to report
            img = Image(buf, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
            
            plt.close()
    
    # Recommendations Section
    elements.append(Paragraph("Recommendations", heading_style))
    
    # Find the best model based on key metrics
    best_model = None
    best_score = -float('inf')
    
    # Determine which metric to use for comparison
    if problem_type == 'classification':
        primary_metrics = ['accuracy', 'f1', 'auc']
    else:  # regression
        primary_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    # Find best model
    for model_data in models_data:
        metrics = model_data.get('metrics', {})
        for metric in primary_metrics:
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                # For error metrics (those with 'error' in the name), lower is better
                score = -metrics[metric] if 'error' in metric.lower() else metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model = model_data
                break  # Use first available metric in priority order
    
    # Add recommendations text
    if best_model:
        best_algo = best_model.get('algorithm', 'Unknown')
        elements.append(Paragraph(f"Based on the comparison, the {best_algo} model shows the best overall performance for this problem. This model is recommended for deployment.", normal_style))
    else:
        elements.append(Paragraph("No clear best model could be determined. Please review the metrics and choose a model based on your specific requirements.", normal_style))
    
    # Additional considerations
    elements.append(Paragraph("Additional Considerations", heading2_style))
    elements.append(Paragraph("When selecting a model, consider the following factors:", normal_style))
    
    considerations = [
        "Model performance on specific metrics relevant to your use case",
        "Computational efficiency and inference speed requirements",
        "Interpretability needs (some models offer better explainability)",
        "Robustness to outliers and edge cases",
        "Deployment constraints and integration requirements"
    ]
    
    for consideration in considerations:
        elements.append(Paragraph(f"• {consideration}", normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path

def generate_explainability_report(model_run_id, algorithm, feature_explanations, global_explanations, 
                                 output_folder, experiment_name=None):
    """
    Generate an explainability report for model interpretability
    
    Parameters:
    -----------
    model_run_id : str
        ID of the model run
    algorithm : str
        Name of the algorithm
    feature_explanations : dict
        Feature-level explanations (e.g., SHAP values, feature importances)
    global_explanations : dict
        Global model explanations
    output_folder : str
        Folder to save the report
    experiment_name : str, optional
        Name of the experiment
        
    Returns:
    --------
    str : Path to the generated report
    """
    # Create the output filename
    report_filename = f"{model_run_id}_explainability_report.pdf"
    report_path = os.path.join(output_folder, report_filename)
    
    # Create a PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Content elements
    elements = []
    
    # Title
    title = f"Model Explainability Report: {algorithm}"
    elements.append(Paragraph(title, title_style))
    
    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if experiment_name:
        elements.append(Paragraph(f"Experiment: {experiment_name}", normal_style))
    elements.append(Paragraph(f"Generated: {timestamp}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Introduction
    elements.append(Paragraph("Model Explainability Overview", heading_style))
    elements.append(Paragraph(
        f"This report provides insights into how the {algorithm} model makes predictions, "
        f"helping to understand its behavior and decision-making process. "
        f"Explainability is critical for building trust, identifying biases, and ensuring "
        f"model fairness and compliance with regulations.",
        normal_style
    ))
    elements.append(Spacer(1, 0.25*inch))
    
    # Feature Explanations
    if feature_explanations:
        elements.append(Paragraph("Feature-Level Explanations", heading_style))
        
        # SHAP Values Visualization (if available)
        if 'shap_values' in feature_explanations:
            elements.append(Paragraph("SHAP Values", heading2_style))
            elements.append(Paragraph(
                "SHAP (SHapley Additive exPlanations) values explain how much each feature contributes to the prediction "
                "for individual instances, accounting for feature interactions.",
                normal_style
            ))
            
            # Check if summary plot is provided
            if 'shap_summary_plot' in feature_explanations:
                # Add the encoded image
                try:
                    plot_data = feature_explanations['shap_summary_plot']
                    img_data = base64.b64decode(plot_data)
                    img = Image(BytesIO(img_data), width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                except Exception as e:
                    elements.append(Paragraph(f"Error displaying SHAP summary plot: {str(e)}", normal_style))
            
            # Check if waterfall plots are provided
            if 'shap_waterfall_plots' in feature_explanations:
                for i, plot_data in enumerate(feature_explanations['shap_waterfall_plots'][:3]):  # Limit to 3 examples
                    elements.append(Paragraph(f"Example {i+1} Explanation", heading2_style))
                    try:
                        img_data = base64.b64decode(plot_data)
                        img = Image(BytesIO(img_data), width=6*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 0.1*inch))
                    except Exception as e:
                        elements.append(Paragraph(f"Error displaying SHAP waterfall plot: {str(e)}", normal_style))
        
        # Partial Dependence Plots (if available)
        if 'partial_dependence' in feature_explanations:
            elements.append(Paragraph("Partial Dependence Plots", heading2_style))
            elements.append(Paragraph(
                "Partial dependence plots show how the model's prediction changes as a feature value changes, "
                "averaging out the effects of all other features.",
                normal_style
            ))
            
            for feature, plot_data in feature_explanations['partial_dependence'].items():
                try:
                    # Add the plot
                    elements.append(Paragraph(f"Feature: {feature}", normal_style))
                    img_data = base64.b64decode(plot_data)
                    img = Image(BytesIO(img_data), width=5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                except Exception as e:
                    elements.append(Paragraph(f"Error displaying partial dependence plot for {feature}: {str(e)}", normal_style))
        
        # Feature Importance
        if 'feature_importance' in feature_explanations:
            elements.append(Paragraph("Feature Importance", heading2_style))
            elements.append(Paragraph(
                "Feature importance indicates the relative importance of each feature to the model's predictions.",
                normal_style
            ))
            
            importances = feature_explanations['feature_importance']
            
            # Create plot
            fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
            
            # Sort importances and limit to top 15 if there are many
            importance_df = pd.DataFrame({
                'feature': list(importances.keys()),
                'importance': list(importances.values())
            }).sort_values('importance', ascending=False)
            
            if len(importance_df) > 15:
                importance_df = importance_df.head(15)
                subtitle = "Top 15 features shown"
            else:
                subtitle = ""
            
            # Plot horizontal bar chart
            importance_df = importance_df.sort_values('importance')  # Sort ascending for bottom-to-top display
            ax.barh(importance_df['feature'], importance_df['importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            if subtitle:
                ax.text(0.5, 0.01, subtitle, transform=ax.transAxes, ha='center', fontsize=8)
            plt.tight_layout()
            
            # Save figure to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            
            # Add plot to report
            img = Image(buf, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
            
            plt.close()
    
    # Global Explanations
    if global_explanations:
        elements.append(Paragraph("Global Model Explanations", heading_style))
        
        # Decision Tree Surrogate (if available)
        if 'surrogate_tree' in global_explanations:
            elements.append(Paragraph("Decision Tree Surrogate Model", heading2_style))
            elements.append(Paragraph(
                "A simplified decision tree that approximates the behavior of the complex model, "
                "providing an interpretable representation of its decision-making process.",
                normal_style
            ))
            
            try:
                # Add the encoded image
                plot_data = global_explanations['surrogate_tree']
                img_data = base64.b64decode(plot_data)
                img = Image(BytesIO(img_data), width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.25*inch))
            except Exception as e:
                elements.append(Paragraph(f"Error displaying surrogate tree plot: {str(e)}", normal_style))
        
        # Feature Interactions (if available)
        if 'feature_interactions' in global_explanations:
            elements.append(Paragraph("Feature Interactions", heading2_style))
            elements.append(Paragraph(
                "Feature interactions measure how features work together to influence the model's predictions.",
                normal_style
            ))
            
            try:
                # Add the encoded image
                plot_data = global_explanations['feature_interactions']
                img_data = base64.b64decode(plot_data)
                img = Image(BytesIO(img_data), width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.25*inch))
            except Exception as e:
                elements.append(Paragraph(f"Error displaying feature interactions plot: {str(e)}", normal_style))
    
    # Interpretation and Recommendations
    elements.append(Paragraph("Interpretation & Recommendations", heading_style))
    
    # Add default interpretation text
    elements.append(Paragraph(
        "Based on the explainability analysis, here are the key insights about the model's behavior:",
        normal_style
    ))
    
    # Add insights based on available explanations
    insights = []
    
    if feature_explanations and 'feature_importance' in feature_explanations:
        # Get top features
        importances = feature_explanations['feature_importance']
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_features:
            top_features_text = ", ".join([f"{feature}" for feature, _ in top_features])
            insights.append(f"The most influential features for this model are: {top_features_text}")
    
    if global_explanations and 'model_behavior' in global_explanations:
        insights.append(global_explanations['model_behavior'])
    
    if not insights:
        insights = [
            "Review feature importance to understand which variables drive predictions the most",
            "Look for unexpected patterns in how features influence predictions",
            "Consider whether the model's behavior aligns with domain knowledge"
        ]
    
    for insight in insights:
        elements.append(Paragraph(f"• {insight}", normal_style))
    
    elements.append(Spacer(1, 0.1*inch))
    
    # Recommendations
    elements.append(Paragraph("Recommendations for Model Use", heading2_style))
    
    recommendations = [
        "Monitor predictions for instances where important features have unusual values",
        "Consider collecting more data for areas where the model shows uncertainty",
        "When deploying the model, provide explanations alongside predictions for critical decisions",
        "Review edge cases where feature interactions might lead to unexpected predictions"
    ]
    
    for recommendation in recommendations:
        elements.append(Paragraph(f"• {recommendation}", normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path
                    

def generate_tuning_report(model_run_id, algorithm, problem_type, tuning_results, best_params, 
                          metrics, output_folder, experiment_name=None):
    """
    Generate a PDF report for hyperparameter tuning results
    
    Parameters:
    -----------
    model_run_id : str
        ID of the model run
    algorithm : str
        Name of the algorithm
    problem_type : str
        'classification' or 'regression'
    tuning_results : list of dicts
        Each dict contains hyperparameters and corresponding scores
    best_params : dict
        The best hyperparameters found
    metrics : dict
        Performance metrics for the best hyperparameters
    output_folder : str
        Folder to save the report
    experiment_name : str, optional
        Name of the experiment
        
    Returns:
    --------
    str : Path to the generated report
    """
    # Create the output filename
    report_filename = f"{model_run_id}_tuning_report.pdf"
    report_path = os.path.join(output_folder, report_filename)
    
    # Create a PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create a custom style for code/parameters
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
    )
    
    # Content elements
    elements = []
    
    # Title
    title = f"Hyperparameter Tuning Report: {algorithm}"
    elements.append(Paragraph(title, title_style))
    
    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if experiment_name:
        elements.append(Paragraph(f"Experiment: {experiment_name}", normal_style))
    elements.append(Paragraph(f"Problem Type: {problem_type.capitalize()}", normal_style))
    elements.append(Paragraph(f"Generated: {timestamp}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Best Parameters Section
    elements.append(Paragraph("Best Hyperparameters", heading_style))
    
    # Create table data for best parameters
    best_params_data = [["Parameter", "Value"]]
    for param, value in best_params.items():
        best_params_data.append([param, str(value)])
    
    # Create table for best parameters
    best_params_table = Table(best_params_data, colWidths=[2.5*inch, 3*inch])
    best_params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(best_params_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Best Model Performance Section
    elements.append(Paragraph("Best Model Performance", heading_style))
    
    # Create table data for metrics
    metrics_data = [["Metric", "Value"]]
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics_data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
    
    # Create table for metrics
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Tuning Results Section
    elements.append(Paragraph("Hyperparameter Tuning Results", heading_style))
    
    # Convert tuning_results to DataFrame for easier manipulation
    try:
        import pandas as pd
        df_results = pd.DataFrame(tuning_results)
        
        # Get the score column (assume it's 'score', 'mean_test_score', or similar)
        score_col = None
        for col in df_results.columns:
            if 'score' in col.lower():
                score_col = col
                break
        
        if score_col is not None:
            # Plot tuning results
            elements.append(Paragraph("Parameter Importance", heading2_style))
            
            # For each hyperparameter, create a plot
            param_cols = [col for col in df_results.columns if col != score_col and not col.startswith('param_')]
            
            for param in param_cols:
                try:
                    # Create the plot
                    fig, ax = plt.figure(figsize=(7, 4)), plt.gca()
                    
                    # Sort values
                    plot_data = df_results[[param, score_col]].sort_values(param)
                    
                    # Convert to numeric if possible
                    try:
                        plot_data[param] = pd.to_numeric(plot_data[param])
                    except:
                        pass  # Keep as is if not numeric
                    
                    # Plot the parameter vs score
                    plt.plot(plot_data[param], plot_data[score_col], 'o-')
                    plt.title(f'{param} vs. {score_col}')
                    plt.xlabel(param)
                    plt.ylabel(score_col)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # If the parameter has many unique values, rotate xticks
                    if len(plot_data[param].unique()) > 5:
                        plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    
                    # Save figure to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Add plot to report
                    img = Image(buf, width=6*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                    
                    plt.close()
                except Exception as e:
                    elements.append(Paragraph(f"Error plotting parameter {param}: {str(e)}", normal_style))
            
            # Plot distribution of scores
            fig, ax = plt.figure(figsize=(7, 4)), plt.gca()
            
            plt.hist(df_results[score_col], bins=20, alpha=0.7, color='skyblue')
            plt.axvline(df_results[score_col].max(), color='red', linestyle='--', 
                       label=f'Best Score: {df_results[score_col].max():.4f}')
            plt.title('Distribution of Scores')
            plt.xlabel(score_col)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save figure to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            
            # Add plot to report
            img = Image(buf, width=6*inch, height=3.5*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
            
            plt.close()
        
        # Detailed Results Table
        elements.append(Paragraph("Detailed Tuning Results", heading2_style))
        
        # Get top 10 results if there are many
        if len(df_results) > 10:
            top_results = df_results.nlargest(10, score_col)
            elements.append(Paragraph("Top 10 Parameter Combinations:", normal_style))
        else:
            top_results = df_results
            elements.append(Paragraph("All Parameter Combinations:", normal_style))
        
        # Create table data for detailed results
        detailed_data = [["Rank", "Score"] + param_cols]
        
        for i, (_, row) in enumerate(top_results.iterrows()):
            row_data = [str(i+1), f"{row[score_col]:.4f}"]
            for param in param_cols:
                row_data.append(str(row[param]))
            detailed_data.append(row_data)
        
        # Create table for detailed results
        col_widths = [0.5*inch, 1*inch] + [2*inch] * len(param_cols)
        detailed_table = Table(detailed_data, colWidths=col_widths)
        detailed_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(detailed_table)
        elements.append(Spacer(1, 0.25*inch))
    
    except Exception as e:
        elements.append(Paragraph(f"Error generating tuning results visualizations: {str(e)}", normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path

def generate_comprehensive_report(model_run_id, algorithm, problem_type, metrics, feature_importances, 
                                 confusion_matrix_data, classification_report_data, tuning_results, 
                                 best_params, output_folder, experiment_name=None, quantum_circuit=None):
    """
    Generate a comprehensive PDF report combining model results and tuning information
    
    Parameters:
    -----------
    model_run_id : str
        ID of the model run
    algorithm : str
        Name of the algorithm
    problem_type : str
        'classification' or 'regression'
    metrics : dict
        Performance metrics for the model
    feature_importances : dict or None
        Feature importances if available (feature name -> importance score)
    confusion_matrix_data : numpy array or None
        Confusion matrix for classification models
    classification_report_data : dict or None
        Classification report for classification models
    tuning_results : list of dicts
        Each dict contains hyperparameters and corresponding scores
    best_params : dict
        The best hyperparameters found
    output_folder : str
        Folder to save the report
    experiment_name : str, optional
        Name of the experiment
    quantum_circuit : dict, optional
        Quantum circuit configuration for visualization
        
    Returns:
    --------
    str : Path to the generated report
    """
    # Create the output filename
    report_filename = f"{model_run_id}_comprehensive_report.pdf"
    report_path = os.path.join(output_folder, report_filename)
    
    # Create a PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create a custom style for code/parameters
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
    )
    
    # Content elements
    elements = []
    
    # Title
    title = f"Comprehensive ML Model Report: {algorithm}"
    elements.append(Paragraph(title, title_style))
    
    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if experiment_name:
        elements.append(Paragraph(f"Experiment: {experiment_name}", normal_style))
    elements.append(Paragraph(f"Problem Type: {problem_type.capitalize()}", normal_style))
    elements.append(Paragraph(f"Generated: {timestamp}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Model Overview Section
    elements.append(Paragraph("Model Overview", heading_style))
    
    # Add algorithm description based on type
    model_type = 'Quantum' if algorithm.startswith('Q') else 'Neural Network' if algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU'] else 'Classical'
    
    algorithm_descriptions = {
        'RandomForest': 'An ensemble learning method that constructs multiple decision trees during training and outputs the class or mean prediction of the individual trees.',
        'LogisticRegression': 'A statistical model that uses a logistic function to model a binary dependent variable.',
        'SVC': 'A support vector machine for classification that finds the hyperplane that best separates classes in a high-dimensional space.',
        'SVR': 'A support vector machine for regression that finds the hyperplane that best fits the data in a high-dimensional space.',
        'DecisionTree': 'A flowchart-like tree structure where an internal node represents a feature, a branch represents a decision rule, and a leaf node represents the outcome.',
        'KNN': 'A non-parametric method that classifies cases based on their similarity to other cases.',
        'MLP': 'A class of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.',
        'CNN': 'A class of deep neural networks most commonly applied to analyzing visual imagery but also used for non-image data that has a grid-like topology.',
        'RNN': 'A class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.',
        'LSTM': 'A recurrent neural network architecture that is capable of learning long-term dependencies.',
        'GRU': 'A gating mechanism in recurrent neural networks similar to LSTM but with fewer parameters.',
        'QSVM': 'A quantum version of support vector machines that employs quantum computing to enhance classification performance.',
        'QNN': 'A quantum neural network that combines quantum computing principles with neural network architecture.',
        'QKE': 'A quantum kernel estimator that uses quantum computing to implement kernel methods for machine learning.',
        'QRegression': 'A quantum regression model that leverages quantum computing for potentially improved predictions.'
    }
    
    if algorithm in algorithm_descriptions:
        elements.append(Paragraph(f"Algorithm Type: {model_type}", normal_style))
        elements.append(Paragraph(f"Description: {algorithm_descriptions[algorithm]}", normal_style))
    else:
        elements.append(Paragraph(f"Algorithm Type: {model_type}", normal_style))
    
    elements.append(Spacer(1, 0.15*inch))
    
    # Performance Metrics Section
    elements.append(Paragraph("Model Performance", heading_style))
    
    # Create table data for metrics
    metrics_data = [["Metric", "Value"]]
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics_data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
    
    # Create table for metrics
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Classification specific sections
    if problem_type == 'classification' and confusion_matrix_data is not None:
        # Confusion Matrix
        elements.append(Paragraph("Confusion Matrix", heading_style))
        
        # Plot confusion matrix
        fig, ax = plt.figure(figsize=(6, 5)), plt.gca()
        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        
        # Add plot to report
        img = Image(buf, width=5*inch, height=4.5*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.1*inch))
        
        plt.close()
        
        # Classification Report
        if classification_report_data:
            elements.append(Paragraph("Classification Report", heading_style))
            report_text = classification_report_data.replace(' ', '&nbsp;')
            report_text = report_text.replace('\n', '<br/>')
            p = Paragraph(f"<font face='Courier'>{report_text}</font>", normal_style)
            elements.append(p)
            elements.append(Spacer(1, 0.25*inch))
    
    # Feature Importances Section (if available)
    if feature_importances and len(feature_importances) > 0:
        elements.append(Paragraph("Feature Importances", heading_style))
        
        # Create plot for feature importances
        fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
        
        # Sort importances and limit to top 15 if there are many
        importance_df = pd.DataFrame({
            'feature': list(feature_importances.keys()),
            'importance': list(feature_importances.values())
        }).sort_values('importance', ascending=False)
        
        if len(importance_df) > 15:
            importance_df = importance_df.head(15)
            subtitle = "Top 15 features shown"
        else:
            subtitle = ""
        
        # Plot horizontal bar chart
        importance_df = importance_df.sort_values('importance')  # Sort ascending for bottom-to-top display
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')
        if subtitle:
            ax.text(0.5, 0.01, subtitle, transform=ax.transAxes, ha='center', fontsize=8)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        
        # Add plot to report
        img = Image(buf, width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.25*inch))
        
        plt.close()
    
    # Hyperparameter Tuning Section
    if tuning_results and best_params:
        elements.append(Paragraph("Hyperparameter Tuning", heading_style))
        
        # Best Parameters Section
        elements.append(Paragraph("Best Hyperparameters", heading2_style))
        
        # Create table data for best parameters
        best_params_data = [["Parameter", "Value"]]
        for param, value in best_params.items():
            best_params_data.append([param, str(value)])
        
        # Create table for best parameters
        best_params_table = Table(best_params_data, colWidths=[2.5*inch, 3*inch])
        best_params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(best_params_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Create visualizations of tuning results
        if len(tuning_results) > 1:
            elements.append(Paragraph("Hyperparameter Tuning Results", heading2_style))
            
            # Convert tuning results to DataFrame
            results_df = pd.DataFrame(tuning_results)
            
            # For each parameter, plot its effect on the score
            for param in best_params.keys():
                if param in results_df.columns and len(results_df[param].unique()) > 1:
                    elements.append(Paragraph(f"Effect of {param}", heading2_style))
                    
                    # Create plot
                    fig, ax = plt.figure(figsize=(7, 4)), plt.gca()
                    
                    # Sort by parameter value if numeric
                    if pd.api.types.is_numeric_dtype(results_df[param]):
                        plot_df = results_df.sort_values(by=param)
                        ax.plot(plot_df[param], plot_df['score'], marker='o', linestyle='-')
                    else:
                        # For categorical parameters, use bar plot
                        plot_df = results_df.groupby(param)['score'].mean().reset_index()
                        ax.bar(plot_df[param].astype(str), plot_df['score'])
                    
                    ax.set_xlabel(param)
                    ax.set_ylabel('Score')
                    ax.set_title(f'Effect of {param} on Model Performance')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # Save figure to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Add plot to report
                    img = Image(buf, width=6*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                    
                    plt.close()
    
    # Quantum Circuit Visualization (if available)
    if quantum_circuit and 'feature_map' in quantum_circuit:
        elements.append(Paragraph("Quantum Circuit Visualization", heading_style))
        
        elements.append(Paragraph("Quantum Circuit Parameters", heading2_style))
        
        # Create table data for quantum parameters
        quantum_params_data = [["Parameter", "Value"]]
        for param, value in quantum_circuit.items():
            quantum_params_data.append([param, str(value)])
        
        # Create table for quantum parameters
        quantum_params_table = Table(quantum_params_data, colWidths=[2.5*inch, 3*inch])
        quantum_params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(quantum_params_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add a note about quantum circuit visualization
        elements.append(Paragraph("Note: Quantum circuit visualization requires PennyLane and is only available in the interactive version.", normal_style))
        elements.append(Spacer(1, 0.25*inch))
    
    # Neural Network Architecture Visualization (if applicable)
    if algorithm in ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU', 'GNN']:
        elements.append(Paragraph("Neural Network Architecture", heading_style))
        
        # Add model architecture details if available in metrics
        if 'model_config' in metrics:
            model_config = metrics['model_config']
            
            # Create table data for neural network parameters
            nn_params_data = [["Parameter", "Value"]]
            for param, value in model_config.items():
                nn_params_data.append([param, str(value)])
            
            # Create table for neural network parameters
            nn_params_table = Table(nn_params_data, colWidths=[2.5*inch, 3*inch])
            nn_params_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(nn_params_table)
            elements.append(Spacer(1, 0.25*inch))
            
            # Training History (if available)
            if 'history' in metrics:
                history = metrics['history']
                elements.append(Paragraph("Training History", heading2_style))
                
                # Plot training history
                fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
                
                # Plot loss
                if 'loss' in history:
                    epochs = range(1, len(history['loss']) + 1)
                    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
                    
                    # Plot validation loss if available
                    if 'val_loss' in history:
                        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
                
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Save figure to buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                
                # Add plot to report
                img = Image(buf, width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                
                plt.close()
                
                # Plot accuracy if it's a classification problem
                if problem_type == 'classification' and 'accuracy' in history:
                    fig, ax = plt.figure(figsize=(7, 5)), plt.gca()
                    
                    epochs = range(1, len(history['accuracy']) + 1)
                    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
                    
                    # Plot validation accuracy if available
                    if 'val_accuracy' in history:
                        plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
                    
                    plt.title('Training and Validation Accuracy')
                    plt.xlabel('Epochs')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save figure to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Add plot to report
                    img = Image(buf, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.25*inch))
                    
                    plt.close()
        else:
            elements.append(Paragraph("Neural network architecture information not available.", normal_style))
    
    # Recommendations Section
    elements.append(Paragraph("Recommendations and Best Practices", heading_style))
    
    # Add recommendations based on algorithm type
    if model_type == 'Classical':
        elements.append(Paragraph("• Consider feature engineering or selection to improve model performance.", normal_style))
        elements.append(Paragraph("• Explore ensemble methods by combining this model with others for better predictions.", normal_style))
        elements.append(Paragraph("• Regularly retrain the model as new data becomes available.", normal_style))
    elif model_type == 'Neural Network':
        elements.append(Paragraph("• Experiment with different network architectures or layer configurations.", normal_style))
        elements.append(Paragraph("• Consider using early stopping and learning rate scheduling for better convergence.", normal_style))
        elements.append(Paragraph("• Try data augmentation techniques to improve model generalization.", normal_style))
        elements.append(Paragraph("• Monitor for overfitting and apply regularization if needed.", normal_style))
    elif model_type == 'Quantum':
        elements.append(Paragraph("• Experiment with different feature encodings for the quantum circuit.", normal_style))
        elements.append(Paragraph("• Increase the number of qubits or circuit layers for more complex problems.", normal_style))
        elements.append(Paragraph("• Consider hybrid quantum-classical approaches for better performance.", normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path