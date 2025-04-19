# app/utils/ml_report_utils.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

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
    
    # Performance Metrics Section
    elements.append(Paragraph("Performance Metrics", heading_style))
    
    # Create table data for metrics
    metrics_data = [["Metric", "Value"]]
    for metric, value in metrics.items():
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
    
    # Create visualizations if we have tuning results
    if tuning_results and len(tuning_results) > 1:
        elements.append(Paragraph("Hyperparameter Tuning Results", heading_style))
        
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
        
        # Add table with all tuning results
        elements.append(Paragraph("All Hyperparameter Combinations Tested", heading2_style))
        
        # Create table data for all results
        # First limit to important columns
        important_cols = ['score'] + list(best_params.keys())
        filtered_cols = [col for col in important_cols if col in results_df.columns]
        results_df_filtered = results_df[filtered_cols].copy()
        
        # Sort by score (descending)
        results_df_filtered = results_df_filtered.sort_values(by='score', ascending=False)
        
        # Format scores to 4 decimal places
        results_df_filtered['score'] = results_df_filtered['score'].apply(lambda x: f"{x:.4f}")
        
        # Convert DataFrame to list of lists for table
        tuning_table_data = [filtered_cols]  # Header row
        for _, row in results_df_filtered.iterrows():
            tuning_table_data.append([str(row[col]) for col in filtered_cols])
        
        # Create table with all results (limit to top 20 rows if large)
        if len(tuning_table_data) > 21:  # 20 + header
            tuning_table_data = tuning_table_data[:21]
            elements.append(Paragraph("(Showing top 20 results)", normal_style))
        
        # Calculate column widths - distribute evenly
        col_width = 6.5 * inch / len(filtered_cols)
        col_widths = [col_width] * len(filtered_cols)
        
        tuning_table = Table(tuning_table_data, colWidths=col_widths)
        tuning_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        elements.append(tuning_table)
    
    # Build the PDF
    doc.build(elements)
    
    return report_path

def generate_model_report(model_run_id, algorithm, problem_type, metrics, feature_importances, 
                          confusion_matrix_data, classification_report_data, output_folder, 
                          tuning_report_path=None, experiment_name=None):
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
    
    # If this model had hyperparameter tuning, mention the separate report
    if tuning_report_path:
        elements.append(Paragraph("Hyperparameter Tuning", heading_style))
        elements.append(Paragraph("This model was trained after hyperparameter tuning. A separate tuning report is available with detailed information about the tuning process and results.", normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path

def generate_comprehensive_report(model_run_id, algorithm, problem_type, metrics, feature_importances, 
                                 confusion_matrix_data, classification_report_data, tuning_results, 
                                 best_params, output_folder, experiment_name=None):
    """
    Generate a comprehensive PDF report combining model results and tuning information
    
    Parameters are similar to the individual report functions
        
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
    
    # Model Performance Section
    elements.append(Paragraph("Model Performance", heading_style))
    
    # Create table data for metrics
    metrics_data = [["Metric", "Value"]]
    for metric, value in metrics.items():
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
    
    # Build the PDF
    doc.build(elements)
    
    return report_path

def extract_feature_importances(model, feature_names):
    """Extract feature importances from a model if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            return dict(zip(feature_names, importances))
    except:
        pass
    return None