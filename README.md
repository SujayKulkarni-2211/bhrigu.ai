# Bhrigu.ai - Automated Machine Learning Platform

Bhrigu.ai is a web-based AutoML platform that enables users to upload datasets, analyze them, and automatically select, train, and tune machine learning models without requiring programming knowledge.

## Features (Step 1: Non-Logged-In User Experience)

Current implementation includes:
- Modern, responsive UI with dark/light mode toggle
- Dataset upload and analysis
- Data preprocessing visualization
- Model selection with AI-based recommendations
- Hyperparameter configuration
- Real-time model training with progress logging
- Performance metrics visualization
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
├── instance/                 # Application instance files
├── uploads/                  # User uploaded files
├── models/                   # Saved ML models
├── reports/                  # Generated reports
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore file
├── requirements.txt          # Python dependencies
└── run.py                    # Application entry point
```

## Development Roadmap

- **Current Phase (Step 1)**: Non-logged-in user features and core ML functionality
- **Step 2**: User login, dashboard, and experiment management
- **Step 3**: Multiple ML algorithms and report generation
- **Step 4**: K-fold validation for model recommendations
- **Step 5**: Neural network models
- **Step 6**: Image data support
- **Step 7**: Regression problems
- **Step 8**: Quantum ML with PennyLane
- **Step 9**: API hosting
- **Step 10**: Enhanced reporting
- **Step 11**: Pro mode features
- **Step 12**: Ultra mode
- **Step 13**: Payment integration
- **Steps 14-19**: DevOps pipeline and deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.