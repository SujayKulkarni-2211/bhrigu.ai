# Bhrigu.ai - Automated Machine Learning Platform

Bhrigu.ai is a web-based AutoML platform that enables users to upload datasets, analyze them, and automatically select, train, and tune machine learning models without requiring programming knowledge.

## Features (Step 1: Non-Logged-In User Experience)

Current implementation includes:
- Modern, responsive UI with dark/light mode toggle
- Dataset upload and analysis
- Data preprocessing and visualization
- AI-assisted model selection with recommendations
- Enhanced hyperparameter tuning workflow:
  - Automatic hyperparameter optimization
  - Manual hyperparameter configuration
  - Detailed tuning reports with visualizations
- Real-time model training with progress logging
- Comprehensive model evaluation:
  - Performance metrics visualization
  - PDF report generation (tuning, model performance, comprehensive)
  - Feature importance analysis
  - Confusion matrix for classification models
- Trained model download (.pkl format)

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/bhrigu-ai.git
cd bhrigu-ai
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional)
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application
```bash
flask run
```

6. Open your browser and navigate to `http://127.0.0.1:5000`

## Machine Learning Workflow

Bhrigu.ai implements a step-by-step ML workflow for users:

1. **Data Upload**: Support for CSV and Excel files
2. **Data Analysis**: Automatic detection of data types, statistics, and visualizations
3. **Model Selection**: AI-powered recommendations based on dataset characteristics
4. **Hyperparameter Optimization**:
   - **Automatic Tuning**: Grid search and randomized search with cross-validation
   - **Manual Configuration**: Expert-level control over hyperparameters
5. **Model Training**: Complete training with selected hyperparameters
6. **Model Evaluation**: Comprehensive metrics and visualizations
7. **Report Generation**: Detailed PDF reports for each stage of the process
8. **Model Download**: Export trained models for use in other applications

## Project Structure

```
bhrigu-ai/
├── app/                      # Flask application
│   ├── __init__.py           # App initialization
│   ├── config.py             # Configuration settings
│   ├── models/               # Database models
│   ├── routes/               # Route definitions
│   ├── static/               # Static assets (CSS, JS, images)
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   ├── templates/            # HTML templates
│   └── utils/                # Utility functions
│       ├── data_processing.py  # Data preprocessing utilities
│       ├── ml_utils.py         # ML training and tuning utilities
│       └── ml_report_utils.py  # Report generation utilities
├── instance/                 # Application instance files
├── uploads/                  # User uploaded files
├── models/                   # Saved ML models
├── reports/                  # Generated PDF reports
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore file
├── requirements.txt          # Python dependencies
└── run.py                    # Application entry point
```

## Supported Algorithms

### Classification
- Random Forest
- Logistic Regression
- Support Vector Machine (SVC)
- Decision Tree
- K-Nearest Neighbors (KNN)

### Regression
- Random Forest Regressor
- Linear Regression
- Support Vector Regressor (SVR)
- Decision Tree Regressor

## Development Roadmap

- **Current Phase (Step 1.5)**: Enhanced ML workflow with hyperparameter tuning and report generation ✅
- **Step 2**: User login, dashboard, and experiment management
- **Step 3**: Multiple ML algorithms and expanded report generation
- **Step 4**: K-fold validation for model recommendations
- **Step 5**: Neural network models (e.g., MLP, CNN)
- **Step 6**: Image data support
- **Step 7**: Advanced regression problems
- **Step 8**: Quantum ML with PennyLane
- **Step 9**: API hosting and model deployment
- **Step 10**: Enhanced visualizations and interpretability
- **Step 11**: Pro mode features with advanced customization
- **Step 12**: Ultra mode with ensemble learning
- **Step 13**: Payment integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.