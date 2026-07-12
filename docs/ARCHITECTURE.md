# Architecture

The current platform is a runnable SaaS foundation with three local services:

- `inference`: tenant-aware prediction endpoint with runtime selection.
- `admin-api`: control plane for tenant/model registration and retraining requests.
- `worker`: tenant-isolated telemetry summarization.

The next production steps are persistent registry storage, real retraining
orchestration, stronger auth, and dynamic Envoy cluster management.
