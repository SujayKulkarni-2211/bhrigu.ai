#!/bin/bash
# Script to deploy Bhrigu.ai to Minikube

set -e

# Check if Minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "Error: Minikube is not installed. Please install Minikube first."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if Minikube is running
if ! minikube status | grep -q "Running"; then
    echo "Starting Minikube..."
    minikube start
fi

# Enable ingress addon if not already enabled
if ! minikube addons list | grep -q "ingress.*enabled"; then
    echo "Enabling Ingress addon..."
    minikube addons enable ingress
fi

# Set Docker environment to use Minikube's Docker daemon
eval $(minikube docker-env)

# Check for Docker Hub username or use local images
DOCKER_HUB_USERNAME=${DOCKER_HUB_USERNAME:-""}

if [ -z "$DOCKER_HUB_USERNAME" ]; then
    echo "No DOCKER_HUB_USERNAME environment variable found. Using local images."
    
    # Build Docker images locally
    echo "Building Docker images locally..."
    docker build -t bhrigu-guest:latest ./guest
    docker build -t bhrigu-pro:latest ./pro
    
    # Update deployment.yaml to use local images
    sed -i "s|\${DOCKER_HUB_USERNAME}/bhrigu-guest:latest|bhrigu-guest:latest|g" k8s/deployment.yaml
    sed -i "s|\${DOCKER_HUB_USERNAME}/bhrigu-pro:latest|bhrigu-pro:latest|g" k8s/deployment.yaml
else
    echo "Using Docker Hub images from $DOCKER_HUB_USERNAME"
    sed -i "s|\${DOCKER_HUB_USERNAME}|$DOCKER_HUB_USERNAME|g" k8s/deployment.yaml
fi

# Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f k8s/pvcs.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/bhrigu-guest
kubectl rollout status deployment/bhrigu-pro

# Get Minikube IP
MINIKUBE_IP=$(minikube ip)
echo ""
echo "===================================="
echo "Bhrigu.ai is now running on Minikube"
echo "===================================="
echo ""
echo "Add the following entry to your /etc/hosts file:"
echo "$MINIKUBE_IP bhrigu.local"
echo ""
echo "Access the application at:"
echo "- Guest interface: http://bhrigu.local"
echo "- Pro interface: http://bhrigu.local/pro"
echo ""
echo "Current Kubernetes resources:"
kubectl get pods
echo ""
kubectl get services
echo ""
kubectl get ingress