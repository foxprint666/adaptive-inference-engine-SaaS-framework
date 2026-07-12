import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI not installed")
@unittest.skipUnless(importlib.util.find_spec("jose"), "python-jose not installed")
class Phase3ControlPlaneTests(unittest.TestCase):
    def test_admin_api_registers_tenant_and_model(self):
        from fastapi.testclient import TestClient

        from admin_api.app import app

        client = TestClient(app)
        tenant_response = client.post(
            "/register-tenant",
            json={
                "tenant_id": "client-a",
                "tenant_name": "Client A",
                "contact_email": "ops@example.com",
                "tier": "standard",
            },
        )
        self.assertEqual(tenant_response.status_code, 201)

        token = client.post("/auth/token?tenant_id=client-a").json()["access_token"]
        model_response = client.post(
            "/models/register",
            headers={"X-Tenant-ID": "client-a", "Authorization": f"Bearer {token}"},
            json={
                "model_id": "fraudnet-v1",
                "model_version": "1.0.0",
                "storage_path": "/models/model_baseline_client-a.pt",
                "config_path": "inference/config_fraudnet.json",
                "schema_definition": {"amount": {"type": "float"}},
                "drift_thresholds": {"psi_threshold": 0.25, "auc_threshold": 0.72},
                "framework": "pytorch",
            },
        )
        self.assertEqual(model_response.status_code, 200)
        self.assertEqual(model_response.json()["tenant_id"], "client-a")


if __name__ == "__main__":
    unittest.main()
