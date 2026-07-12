# Multi-Tenancy

Tenant isolation currently uses:

- `X-Tenant-ID` request headers.
- In-memory tenant/model registry records.
- Redis hash-tag key prefixes such as `{client-a}:fraudnet-v1:telemetry_queue`.

The next step is enforcing tenant ownership from a persistent control-plane
database rather than process-local state.
