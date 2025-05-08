#!/bin/bash
# Script to deploy Bhrigu.ai to Minikube

set -e

echo "============================================"
echo "🚀 Starting Bhrigu.ai Deployment to Minikube"
echo "============================================"

# Check if Minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "❌ Error: Minikube is not installed. Please install Minikube first."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if Minikube is running, if not start it
if ! minikube status | grep -q "Running"; then
    echo "🔄 Starting Minikube..."
    minikube start --driver=docker --memory=3000mb
fi

# Enable ingress addon if not already enabled
if ! minikube addons list | grep -q "ingress.*enabled"; then
    echo "🔄 Enabling Ingress addon..."
    minikube addons enable ingress
fi

# Set Docker environment to use Minikube's Docker daemon
echo "🔄 Configuring Docker environment to use Minikube..."
eval $(minikube docker-env)

# Check for Docker Hub username or use local images
DOCKER_HUB_USERNAME=${DOCKER_HUB_USERNAME:-""}

if [ -z "$DOCKER_HUB_USERNAME" ]; then
    echo "🔄 No DOCKER_HUB_USERNAME environment variable found. Using local images."
    
    # Build Docker images locally
    echo "🔄 Building Docker images locally..."
    
    echo "🔄 Building Guest image..."
    docker build -t bhrigu-guest:latest ./guest
    
    echo "🔄 Building Pro image..."
    docker build -t bhrigu-pro:latest ./pro
    
    # Update deployment.yaml to use local images
    echo "🔄 Updating deployment configuration for local images..."
    sed -i.bak "s|\${DOCKER_HUB_USERNAME}/bhrigu-guest:latest|bhrigu-guest:latest|g" k8s/deployment.yaml
    sed -i.bak "s|\${DOCKER_HUB_USERNAME}/bhrigu-pro:latest|bhrigu-pro:latest|g" k8s/deployment.yaml
    
    # Change imagePullPolicy to Never for local images
    sed -i.bak "s|imagePullPolicy: Always|imagePullPolicy: Never|g" k8s/deployment.yaml
else
    echo "🔄 Using Docker Hub images from $DOCKER_HUB_USERNAME"
    sed -i.bak "s|\${DOCKER_HUB_USERNAME}|$DOCKER_HUB_USERNAME|g" k8s/deployment.yaml
fi

# Apply Kubernetes configurations with proper order and error handling
echo "🔄 Applying Kubernetes configurations..."

echo "🔄 Creating Persistent Volume Claims..."
kubectl apply -f k8s/pvcs.yaml

echo "🔄 Creating Deployments..."
kubectl apply -f k8s/deployment.yaml

echo "🔄 Creating Services..."
kubectl apply -f k8s/service.yaml

echo "🔄 Creating Ingress rules..."
kubectl apply -f k8s/ingress.yaml

# Wait for deployments to be ready
echo "🔄 Waiting for deployments to be ready..."
kubectl rollout status deployment/bhrigu-guest --timeout=180s || echo "⚠️  Guest deployment still starting, but continuing..."
kubectl rollout status deployment/bhrigu-pro --timeout=180s || echo "⚠️  Pro deployment still starting, but continuing..."

# Get Minikube IP
MINIKUBE_IP=$(minikube ip)
echo ""
echo "=============================================="
echo "✅ Bhrigu.ai is now deployed on Minikube!"
echo "=============================================="
echo ""
echo "📝 Add the following entry to your /etc/hosts file:"
echo "$MINIKUBE_IP bhrigu.local"
echo ""
echo "🔗 Access the application at:"
echo "- Guest interface: http://bhrigu.local"
echo "- Pro interface: http://bhrigu.local/pro"
echo ""
echo "📊 Current Kubernetes resources:"
echo ""
echo "📦 Pods:"
kubectl get pods
echo ""
echo "🌐 Services:"
kubectl get services
echo ""
echo "🚪 Ingress:"
kubectl get ingress

# Print logs if there are any issues with the pods
if kubectl get pods | grep -q "Error\|CrashLoopBackOff"; then
    echo ""
    echo "⚠️  Some pods have issues. Checking logs:"
    
    for pod in $(kubectl get pods | grep "Error\|CrashLoopBackOff" | awk '{print $1}'); do
        echo ""
        echo "📜 Logs for $pod:"
        kubectl logs $pod
    done
    
    echo ""
    echo "Try running 'kubectl describe pod POD_NAME' for more details."
fi