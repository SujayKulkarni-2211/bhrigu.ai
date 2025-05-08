#!/bin/bash
# Reset script for Bhrigu.ai Kubernetes deployment
# Use this script if you need to completely start over

set -e

echo "============================================"
echo "🔄 Bhrigu.ai Complete Reset"
echo "============================================"

echo "⚠️  WARNING: This will delete all Kubernetes resources and start fresh."
echo "Press Ctrl+C to cancel or Enter to continue..."
read

# Stop Minikube
echo "🔄 Stopping Minikube..."
minikube stop || echo "No Minikube running."

# Delete Minikube
echo "🔄 Deleting Minikube..."
minikube delete || echo "No Minikube to delete."

# Start fresh Minikube with sufficient resources
echo "🔄 Starting fresh Minikube with sufficient resources..."
minikube start --driver=docker --memory=3000mb --cpus=2

# Enable ingress addon
echo "🔄 Enabling Ingress addon..."
minikube addons enable ingress

# Set Docker environment to use Minikube's Docker daemon
echo "🔄 Configuring Docker environment to use Minikube..."
eval $(minikube docker-env)

# Build Docker images with no cache to ensure fresh start
echo "🔄 Building Docker images with no cache..."

echo "🔄 Building Guest image..."
docker build --no-cache -t bhrigu-guest:latest ./guest

echo "🔄 Building Pro image..."
docker build --no-cache -t bhrigu-pro:latest ./pro

# Update deployment.yaml to use local images
echo "🔄 Updating deployment configuration for local images..."
cp k8s/deployment.yaml k8s/deployment.yaml.bak
sed -i "s|\${DOCKER_HUB_USERNAME}/bhrigu-guest:latest|bhrigu-guest:latest|g" k8s/deployment.yaml
sed -i "s|\${DOCKER_HUB_USERNAME}/bhrigu-pro:latest|bhrigu-pro:latest|g" k8s/deployment.yaml

# Change imagePullPolicy to Never for local images
sed -i "s|imagePullPolicy: Always|imagePullPolicy: Never|g" k8s/deployment.yaml

# Apply Kubernetes configurations with proper order
echo "🔄 Applying Kubernetes configurations..."

echo "🔄 Creating Persistent Volume Claims..."
kubectl apply -f k8s/pvcs.yaml

echo "🔄 Creating Deployments..."
kubectl apply -f k8s/deployment.yaml

echo "🔄 Creating Services..."
kubectl apply -f k8s/service.yaml

echo "🔄 Creating Ingress rules..."
kubectl apply -f k8s/ingress.yaml

# Wait for deployments to be ready with longer timeout
echo "🔄 Waiting for deployments to be ready (this may take a few minutes)..."
kubectl rollout status deployment/bhrigu-guest --timeout=300s || echo "⚠️  Guest deployment still starting, but continuing..."
kubectl rollout status deployment/bhrigu-pro --timeout=300s || echo "⚠️  Pro deployment still starting, but continuing..."

# Get Minikube IP
MINIKUBE_IP=$(minikube ip)
echo ""
echo "=============================================="
echo "✅ Bhrigu.ai has been completely reset and redeployed!"
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