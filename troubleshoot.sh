#!/bin/bash
# Troubleshooting script for Bhrigu.ai Kubernetes deployment

set -e

echo "============================================"
echo "🔍 Bhrigu.ai Deployment Troubleshooter"
echo "============================================"

# Check Minikube status
echo "🔍 Checking Minikube status..."
if ! minikube status | grep -q "Running"; then
    echo "⚠️  Minikube is not running. Starting Minikube..."
    minikube start --driver=docker --memory=3000mb --cpus=2
else
    echo "✅ Minikube is running."
fi

# Check if ingress is enabled
echo "🔍 Checking if Ingress is enabled..."
if ! minikube addons list | grep -q "ingress.*enabled"; then
    echo "⚠️  Ingress addon is not enabled. Enabling it now..."
    minikube addons enable ingress
else
    echo "✅ Ingress addon is enabled."
fi

# Set Docker environment to use Minikube's Docker daemon
eval $(minikube docker-env)

# Check if Docker images exist
echo "🔍 Checking Docker images..."
if ! docker images | grep -q "bhrigu-guest"; then
    echo "⚠️  Guest Docker image not found. Rebuilding..."
    docker build -t bhrigu-guest:latest ./guest
else
    echo "✅ Guest Docker image exists."
fi

if ! docker images | grep -q "bhrigu-pro"; then
    echo "⚠️  Pro Docker image not found. Rebuilding..."
    docker build -t bhrigu-pro:latest ./pro
else
    echo "✅ Pro Docker image exists."
fi

# Check deployments
echo "🔍 Checking deployments..."
if ! kubectl get deployments | grep -q "bhrigu-guest"; then
    echo "⚠️  Bhrigu Guest deployment not found. Reapplying deployment..."
    kubectl apply -f k8s/deployment.yaml
else
    echo "✅ Bhrigu Guest deployment exists."
    
    # Check if pods are running
    if ! kubectl get pods -l app=bhrigu-guest | grep -q "Running"; then
        echo "⚠️  Bhrigu Guest pods are not running. Restarting deployment..."
        kubectl rollout restart deployment bhrigu-guest
    else
        echo "✅ Bhrigu Guest pods are running."
    fi
fi

if ! kubectl get deployments | grep -q "bhrigu-pro"; then
    echo "⚠️  Bhrigu Pro deployment not found. Reapplying deployment..."
    kubectl apply -f k8s/deployment.yaml
else
    echo "✅ Bhrigu Pro deployment exists."
    
    # Check if pods are running
    if ! kubectl get pods -l app=bhrigu-pro | grep -q "Running"; then
        echo "⚠️  Bhrigu Pro pods are not running. Restarting deployment..."
        kubectl rollout restart deployment bhrigu-pro
    else
        echo "✅ Bhrigu Pro pods are running."
    fi
fi

# Check services
echo "🔍 Checking services..."
if ! kubectl get services | grep -q "bhrigu-guest"; then
    echo "⚠️  Bhrigu Guest service not found. Reapplying service..."
    kubectl apply -f k8s/service.yaml
else
    echo "✅ Bhrigu Guest service exists."
fi

if ! kubectl get services | grep -q "bhrigu-pro"; then
    echo "⚠️  Bhrigu Pro service not found. Reapplying service..."
    kubectl apply -f k8s/service.yaml
else
    echo "✅ Bhrigu Pro service exists."
fi

# Check ingress
echo "🔍 Checking ingress..."
if ! kubectl get ingress | grep -q "bhrigu-ingress"; then
    echo "⚠️  Bhrigu ingress not found. Reapplying ingress..."
    kubectl apply -f k8s/ingress.yaml
else
    echo "✅ Bhrigu ingress exists."
fi

# Check PVCs
echo "🔍 Checking persistent volume claims..."
if ! kubectl get pvc | grep -q "bhrigu-guest-uploads-pvc"; then
    echo "⚠️  Bhrigu Guest PVCs not found. Reapplying PVCs..."
    kubectl apply -f k8s/pvcs.yaml
else
    echo "✅ Bhrigu Guest PVCs exist."
fi

if ! kubectl get pvc | grep -q "bhrigu-pro-uploads-pvc"; then
    echo "⚠️  Bhrigu Pro PVCs not found. Reapplying PVCs..."
    kubectl apply -f k8s/pvcs.yaml
else
    echo "✅ Bhrigu Pro PVCs exist."
fi

# Display pod logs if there are issues
echo "🔍 Checking for pod issues..."
if kubectl get pods | grep -q "Error\|CrashLoopBackOff"; then
    echo "⚠️  Found pods with issues. Checking logs..."
    
    # Get problematic pods
    PROBLEM_PODS=$(kubectl get pods | grep "Error\|CrashLoopBackOff" | awk '{print $1}')
    
    for pod in $PROBLEM_PODS; do
        echo ""
        echo "📜 Logs for pod $pod:"
        kubectl logs $pod
        echo "-----------------------------------"
    done
    
    echo "🔄 Attempting to restart problematic pods..."
    kubectl get deployments | grep "bhrigu" | awk '{print $1}' | xargs kubectl rollout restart deployment
else
    echo "✅ No pod issues found."
fi

# Check if we need to force recreate the deployments
echo "🔍 Checking if deployments need to be recreated..."
if kubectl get pods | grep -q "CrashLoopBackOff"; then
    echo "⚠️  Pods in CrashLoopBackOff state detected. Attempting force recreation..."
    
    # Delete and recreate deployments
    kubectl delete -f k8s/deployment.yaml
    sleep 5
    kubectl apply -f k8s/deployment.yaml
    
    echo "🔄 Waiting for new deployments to start..."
    sleep 10
    kubectl get pods
fi

# Get Minikube IP and access URLs
MINIKUBE_IP=$(minikube ip)
echo ""
echo "=============================================="
echo "✅ Troubleshooting complete!"
echo "=============================================="
echo ""
echo "📝 Make sure you have the following entry in your /etc/hosts file:"
echo "$MINIKUBE_IP bhrigu.local"
echo ""
echo "🔗 Access the application at:"
echo "- Guest interface: http://bhrigu.local"
echo "- Pro interface: http://bhrigu.local/pro"
echo ""
echo "📊 Current Kubernetes resources:"
kubectl get all
echo ""
echo "🔌 To access directly without hosts file:"
echo "Guest: $(minikube service bhrigu-guest --url)"
echo "Pro: $(minikube service bhrigu-pro --url)"
echo ""
echo "If problems persist, you can try a complete reset with:"
echo "./reset-deployment.sh"