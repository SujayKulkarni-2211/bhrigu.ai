FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

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