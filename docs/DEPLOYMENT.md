# Deployment notes

- Local dev: docker-compose (uses .env).
- CI: runs pytest and builds image (.github/workflows/ci.yml).
- Production (k8s): create k8s secret from .env and apply k8s manifests.


