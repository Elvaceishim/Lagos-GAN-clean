# Deployment notes

- Local dev: docker-compose (uses .env).
- CI: runs pytest and builds image (.github/workflows/ci.yml).
- Production (k8s): create k8s secret from .env and apply k8s manifests.

Example k8s secret from .env:

```bash
kubectl create secret generic gan-api-key --from-literal=GAN_API_KEYS="$(grep '^GAN_API_KEYS=' .env | cut -d= -f2-)"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
