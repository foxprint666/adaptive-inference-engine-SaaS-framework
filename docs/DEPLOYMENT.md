# Deployment

Local development starts with:

```bash
docker compose -f docker-compose.multitenant.yml up --build
```

Production deployment still needs persistent registry wiring, secrets,
TLS/mTLS, and Kubernetes manifests.
