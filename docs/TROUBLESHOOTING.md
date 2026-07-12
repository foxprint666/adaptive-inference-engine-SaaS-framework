# Troubleshooting

- If `python` is not found on Windows, use a virtual environment or an absolute
  Python executable path.
- If API tests skip, install dependencies from `requirements.txt`.
- If inference cannot write telemetry, confirm Redis is running and `REDIS_URL`
  points to the correct host.
