#!/bin/bash

# Exit on any error
set -e

echo "Starting Bhrigu.AI deployment process..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create necessary directories if they don't exist
echo "Creating necessary directories..."
mkdir -p guest/uploads guest/models guest/reports pro/uploads pro/models pro/reports

# Check for .env files and create them if they don't exist, copying from examples
if [ ! -f "./guest/.env" ] && [ -f "./guest/.env.example" ]; then
    echo "Creating guest/.env file from example..."
    cp ./guest/.env.example ./guest/.env
    echo "Please update guest/.env with your actual values before running the application"
fi

if [ ! -f "./pro/.env" ] && [ -f "./pro/.env.example" ]; then
    echo "Creating pro/.env file from example..."
    cp ./pro/.env.example ./pro/.env
    echo "Please update pro/.env with your actual values before running the application"
fi

# Check if .env files exist
if [ ! -f "./guest/.env" ] || [ ! -f "./pro/.env" ]; then
    echo "Error: .env files are missing. Please create .env files from the .env.example files."
    exit 1
fi

# Build and run the containers
echo "Building and starting the containers..."
docker-compose up -d --build

# Wait for containers to be ready
echo "Waiting for containers to be ready..."
sleep 10

echo "Deployment complete!"
echo "Guest service: http://localhost:5000"
echo "Pro service: http://localhost:5001"

# Print instructions
echo ""
echo "================================"
echo "INSTRUCTIONS"
echo "================================"
echo "1. Open your browser and go to http://localhost:5000 to access the Guest interface"
echo "2. Click on the Login button to be redirected to the Pro service"
echo "3. To stop the containers, run: docker-compose down"
echo "4. To view logs, run: docker-compose logs -f"
echo "================================"