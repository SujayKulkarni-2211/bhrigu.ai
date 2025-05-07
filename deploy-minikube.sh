#!/bin/bash

# Exit on any error
set -e

# Set variables
GUEST_IMAGE="bhrigu-guest:latest"
PRO_IMAGE="bhrigu-pro:latest"

echo "Starting Minikube deployment for Bhrigu.AI"

# Check if Minikube is running
if ! minikube status | grep -q "Running"; then
    echo "Starting Minikube..."
    minikube start
else
    echo "Minikube is already running."
fi

# Enable Ingress addon if not already enabled
if ! minikube addons list | grep -q "ingress: enabled"; then
    echo "Enabling Ingress addon..."
    minikube addons enable ingress
fi

# Point shell to minikube's docker-daemon
echo "Setting up Docker environment for Minikube..."
eval $(minikube docker-env)

# Build Docker images (using minikube's Docker)
echo "Building Guest Docker image..."
cd guest
docker build -t $GUEST_IMAGE .
cd ..

echo "Building Pro Docker image..."
cd pro
docker build -t $PRO_IMAGE .
cd ..

# Update deployment.yaml with local image names
sed -i.bak "s|image: .*bhrigu-guest:.*|image: $GUEST_IMAGE|g" k8s/deployment.yaml
sed -i.bak "s|image: .*bhrigu-pro:.*|image: $PRO_IMAGE|g" k8s/deployment.yaml

# Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f k8s/pvcs.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/bhrigu-guest
kubectl rollout status deployment/bhrigu-pro

# Set up host entries for local testing
MINIKUBE_IP=$(minikube ip)
echo "Minikube is running at $MINIKUBE_IP"
echo "Add the following entry to your /etc/hosts file to access the application:"
echo "$MINIKUBE_IP bhrigu.local"

# Run tests and generate report
echo "Running tests and generating report..."
cd guest
python generate_test_summary.py

# Display service URLs
echo "Deployment complete!"
echo "Guest service: http://bhrigu.local"
echo "Pro service: http://bhrigu.local/pro"
echo "To access services directly (bypassing Ingress):"
minikube service bhrigu-guest --url
minikube service bhrigu-pro --url

echo "Deployment completed successfully!"