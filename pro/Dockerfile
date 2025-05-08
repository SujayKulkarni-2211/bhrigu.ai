FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and application code
COPY . .

# Install packages in specific order to avoid conflicts
RUN pip install --no-cache-dir setuptools==68.0.0 wheel==0.40.0 && \
    pip install --no-cache-dir flask==2.3.3 werkzeug==2.3.7 && \
    pip install --no-cache-dir jinja2==3.1.2 && \
    pip install --no-cache-dir sqlalchemy==2.0.16 && \
    pip install --no-cache-dir flask-sqlalchemy==3.1.1 && \
    pip install --no-cache-dir flask-login==0.6.2 && \
    pip install --no-cache-dir flask-wtf==1.1.1 && \
    pip install --no-cache-dir flask-mail==0.9.1 && \
    pip install --no-cache-dir Flask-Migrate && \
    pip install --no-cache-dir python-dotenv==1.0.0 && \
    pip install --no-cache-dir email-validator==2.0.0 && \
    pip install --no-cache-dir oauthlib==2.1.0 && \
    pip install --no-cache-dir Flask-OAuthlib==0.9.6 && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir pandas==2.0.3 && \
    pip install --no-cache-dir scikit-learn==1.2.2 && \
    pip install --no-cache-dir matplotlib==3.7.2 && \
    pip install --no-cache-dir seaborn==0.12.2 && \
    pip install --no-cache-dir joblib==1.3.1 && \
    pip install --no-cache-dir mysqlclient==2.1.1 && \
    pip install --no-cache-dir sqlite3-api==2.0.4 && \
    # Install other common packages that might be needed
    pip install --no-cache-dir requests==2.31.0 && \
    # Attempt to install the rest from requirements.txt but don't fail if some packages conflict
    pip install --no-cache-dir -r requirements.txt || echo "Some packages from requirements.txt were not installed due to conflicts"

# Create necessary directories if they don't exist
RUN mkdir -p uploads/datasets uploads/profile_images uploads/reports models reports

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV GUEST_SERVICE_URL=http://bhrigu-guest:5000

# Expose the port
EXPOSE 5001

# Run the application
CMD ["python", "run.py"]