# This script deploys Bhrigu.ai to a Minikube cluster
set -e

# Check if Minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "Minikube is not installed. Please install Minikube first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Start Minikube if it's not running
if ! minikube status | grep -q "Running"; then
    echo "Starting Minikube..."
    minikube start --driver=docker --cpus=4 --memory=4g
fi

# Enable Ingress addon if not already enabled
if ! minikube addons list | grep -q "ingress.*enabled"; then
    echo "Enabling Ingress addon..."
    minikube addons enable ingress
fi

# Create a namespace for Bhrigu.ai if it doesn't exist
if ! kubectl get namespace bhrigu &> /dev/null; then
    echo "Creating 'bhrigu' namespace..."
    kubectl create namespace bhrigu
fi