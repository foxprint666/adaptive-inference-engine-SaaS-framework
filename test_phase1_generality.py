import importlib.util
import unittest


class Phase1GeneralityTests(unittest.TestCase):
    def test_dynamic_schema_accepts_configured_features(self):
        from inference.model_runtime import ModelRuntime

        class DemoRuntime(ModelRuntime):
            def load(self, model_path: str) -> None:
                self.model = object()

            def predict(self, features):
                self.validate_features(features)
                return {"prediction": 0, "probability": 0.1}

        runtime = DemoRuntime("inference/config_fraudnet.json")
        self.assertTrue(
            runtime.validate_features(
                {
                    "amount": 1.0,
                    "distance": 2.0,
                    "velocity": 3.0,
                    "age": 4.0,
                    "risk_score": 0.5,
                }
            )
        )

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed")
    def test_churn_runtime_predicts_without_saved_model(self):
        from inference.churn_runtime import ChurnRuntime

        runtime = ChurnRuntime("inference/config_churn.json")
        runtime.load("missing-demo-model.pkl")
        result = runtime.predict(
            {
                "customer_age": 30,
                "tenure_months": 6,
                "monthly_spend": 49.0,
                "support_tickets": 2,
                "contract_type": 0,
            }
        )
        self.assertIn(result["prediction"], [0, 1])
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
