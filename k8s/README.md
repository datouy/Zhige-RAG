# ChineseRAGKB Kubernetes Deployment Guide
# Version: 0.3.0

# Prerequisites
# - Kubernetes 1.24+
# - NVIDIA GPU Operator (for GPU support)
# - NGINX Ingress Controller
# - metrics-server
# - PostgreSQL 15+ (external or via operator)
# - Redis 7+ (optional)

# Quick Start
# 1. Create namespace
kubectl create namespace chineseragkb

# 2. Apply configurations (in order)
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# 3. Check deployment status
kubectl -n chineseragkb get pods
kubectl -n chineseragkb get services
kubectl -n chineseragkb get ingress

# 4. View logs
kubectl -n chineseragkb logs -l app=chineseragkb -f

# Scaling
# Manual scaling
kubectl -n chineseragkb scale deployment chineseragkb-api --replicas=3

# Enable HPA
kubectl -n chineseragkb autoscale deployment chineseragkb-api --min=2 --max=10 --cpu-percent=70

# Update Strategy
# Rolling update
kubectl -n chineseragkb set image deployment/chineseragkb-api api=chineseragkb/api:0.4.0

# Rollback
kubectl -n chineseragkb rollout undo deployment/chineseragkb-api

# Debugging
# Port forward for local testing
kubectl -n chineseragkb port-forward svc/chineseragkb-api 8000:80

# Check pod resources
kubectl -n chineseragkb top pods

# Describe deployment
kubectl -n chineseragkb describe deployment chineseragkb-api

# Troubleshooting
# 1. Check events
kubectl -n chineseragkb get events --sort-by='.lastTimestamp'

# 2. Check pod logs
kubectl -n chineseragkb logs <pod-name> --previous

# 3. Check ConfigMap
kubectl -n chineseragkb get configmap chineseragkb-config -o yaml

# 4. Check Secrets
kubectl -n chineseragkb get secret chineseragkb-secrets -o jsonpath='{.data}' | keys

# Clean up
kubectl delete -f ingress.yaml
kubectl delete -f hpa.yaml
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kubectl delete -f pvc.yaml
kubectl delete -f secret.yaml
kubectl delete -f configmap.yaml
kubectl delete namespace chineseragkb
