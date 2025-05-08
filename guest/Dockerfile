FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including MySQL client
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install setuptools first to avoid build issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools==68.0.0 wheel==0.40.0

# Copy requirements first for better caching
COPY requirements.txt .

# Fix for version issues - remove problematic packages and version conflicts
RUN grep -v "sqlite3-api==0.1.0" requirements.txt | grep -v "haystack" | grep -v "oauthlib" | grep -v "Flask-OAuthlib" > requirements_fixed.txt && \
    echo "sqlite3-api==2.0.4" >> requirements_fixed.txt && \
    echo "oauthlib==2.1.0" >> requirements_fixed.txt && \
    echo "Flask-OAuthlib==0.9.6" >> requirements_fixed.txt

# Install dependencies from fixed requirements.txt
RUN pip install --no-cache-dir -r requirements_fixed.txt

# Copy application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p uploads models reports

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PRO_SERVICE_URL=http://bhrigu-pro:5001

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "run.py"]