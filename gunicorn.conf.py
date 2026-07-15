"""
gunicorn.conf.py

Production Gunicorn configuration for CPU-bound multi-tenant inference server.
Sized dynamically based on physical cores to prevent thread context-switching.
"""

import os

bind = "0.0.0.0:8080"
worker_class = "uvicorn.workers.UvicornWorker"

# CPU-bound: workers = physical cores (usually half of logical threads)
# fallback to 2 if core detection fails or returns small number
cores = os.cpu_count() or 4
workers = max(2, cores // 2)

# Connection & resource tuning
worker_connections = 1000
timeout = 120
keepalive = 5
preload_app = True          # Fork-safe: loads models in parent, shares memory

# Memory safety recycles
max_requests = 1000
max_requests_jitter = 100
graceful_timeout = 30

# Logging
loglevel = "warning"
access_log = "-"
errorlog = "-"
