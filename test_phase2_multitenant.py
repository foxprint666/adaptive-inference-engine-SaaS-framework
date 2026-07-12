import unittest


class Phase2MultiTenantTests(unittest.TestCase):
    def test_registry_isolates_models_by_tenant(self):
        from inference.tenant_model_registry import TenantModelRegistry

        registry = TenantModelRegistry()
        registry.register_model(
            tenant_id="client-a",
            model_id="fraudnet-v1",
            model_version="1.0.0",
            storage_path="/models/a.pt",
            config_path="inference/config_fraudnet.json",
            schema_definition={"amount": {"type": "float"}},
            drift_thresholds={"psi_threshold": 0.25},
            framework="pytorch",
        )
        registry.register_model(
            tenant_id="client-b",
            model_id="fraudnet-v1",
            model_version="1.0.0",
            storage_path="/models/b.pt",
            config_path="inference/config_fraudnet.json",
            schema_definition={"amount": {"type": "float"}},
            drift_thresholds={"psi_threshold": 0.25},
            framework="pytorch",
        )

        self.assertEqual(
            registry.get_latest_model("client-a", "fraudnet-v1").storage_path,
            "/models/a.pt",
        )
        self.assertEqual(
            registry.get_latest_model("client-b", "fraudnet-v1").storage_path,
            "/models/b.pt",
        )

    def test_redis_keys_use_tenant_hash_tags(self):
        from inference.tenant_redis_client import TenantRedisClient

        client = TenantRedisClient.__new__(TenantRedisClient)
        client.prefix = "{client-a}"
        self.assertEqual(
            client._format_key("fraudnet-v1:telemetry_queue"),
            "{client-a}:fraudnet-v1:telemetry_queue",
        )


if __name__ == "__main__":
    unittest.main()
