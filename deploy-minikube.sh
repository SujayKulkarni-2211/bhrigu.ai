#!/bin/bash
# Script to deploy Bhrigu.ai to Minikube - Guaranteed to work version

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

# Stop any running Minikube to start fresh
echo "🔄 Stopping any running Minikube instance..."
minikube stop || echo "No Minikube running, will start fresh."

# Start Minikube with sufficient resources
echo "🔄 Starting Minikube with sufficient resources..."
minikube start --driver=docker --memory=3000mb --cpus=2

# Enable ingress addon
echo "🔄 Enabling Ingress addon..."
minikube addons enable ingress

# Set Docker environment to use Minikube's Docker daemon
echo "🔄 Configuring Docker environment to use Minikube..."
eval $(minikube docker-env)

# Fix sqlite3-api version in requirements.txt files if needed
echo "🔄 Checking and fixing requirements files if needed..."

if grep -q "sqlite3-api==0.1.0" ./guest/requirements.txt; then
    echo "🔧 Fixing sqlite3-api version in guest requirements.txt..."
    grep -v "sqlite3-api==0.1.0" ./guest/requirements.txt > ./guest/requirements_fixed.txt
    echo "sqlite3-api==2.0.4" >> ./guest/requirements_fixed.txt
    mv ./guest/requirements_fixed.txt ./guest/requirements.txt
fi

if grep -q "sqlite3-api==0.1.0" ./pro/requirements.txt; then
    echo "🔧 Fixing sqlite3-api version in pro requirements.txt..."
    grep -v "sqlite3-api==0.1.0" ./pro/requirements.txt > ./pro/requirements_fixed.txt
    echo "sqlite3-api==2.0.4" >> ./pro/requirements_fixed.txt
    mv ./pro/requirements_fixed.txt ./pro/requirements.txt
fi

# Build Docker images locally with improved caching
echo "🔄 Building Docker images locally..."

echo "🔄 Building Guest image..."
docker build -t bhrigu-guest:latest ./guest

echo "🔄 Building Pro image..."
docker build -t bhrigu-pro:latest ./pro

# Update deployment.yaml to use local images
echo "🔄 Updating deployment configuration for local images..."
cp k8s/deployment.yaml k8s/deployment.yaml.bak
sed -i "s|\${DOCKER_HUB_USERNAME}/bhrigu-guest:latest|bhrigu-guest:latest|g" k8s/deployment.yaml
sed -i "s|\${DOCKER_HUB_USERNAME}/bhrigu-pro:latest|bhrigu-pro:latest|g" k8s/deployment.yaml

# Change imagePullPolicy to Never for local images
sed -i "s|imagePullPolicy: Always|imagePullPolicy: Never|g" k8s/deployment.yaml

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

# Wait for deployments to be ready with timeout
echo "🔄 Waiting for deployments to be ready..."
kubectl rollout status deployment/bhrigu-guest --timeout=300s || echo "⚠️  Guest deployment still starting, but continuing..."
kubectl rollout status deployment/bhrigu-pro --timeout=300s || echo "⚠️  Pro deployment still starting, but continuing..."

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
echo ""
echo "🔍 For direct access without modifying hosts file, use these URLs:"
echo "Guest: $(minikube service bhrigu-guest --url)"
echo "Pro: $(minikube service bhrigu-pro --url)"

# Check if there are any pods in error state and display logs
if kubectl get pods | grep -q "Error\|CrashLoopBackOff"; then
    echo ""
    echo "⚠️  Some pods have issues. Checking logs:"
    
    for pod in $(kubectl get pods | grep "Error\|CrashLoopBackOff" | awk '{print $1}'); do
        echo ""
        echo "📜 Logs for pod $pod:"
        kubectl logs $pod
        echo "-----------------------------------"
    done
    
    echo ""
    echo "If you need to troubleshoot further, run: ./troubleshoot.sh"
fi