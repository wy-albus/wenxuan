import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


class MLPModelTests(unittest.TestCase):
    def test_network_has_required_hidden_layers_and_output_shape(self):
        from src.models.mlp_model import TabularMLP

        model = TabularMLP(4, [3, 5], [2, 3])
        output = model(torch.zeros(6, 4), torch.zeros(6, 2, dtype=torch.long))

        self.assertEqual(tuple(output.shape), (6,))
        linear_sizes = [(layer.in_features, layer.out_features) for layer in model.mlp if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([size[1] for size in linear_sizes], [256, 128, 64, 1])
        self.assertTrue(any(isinstance(layer, torch.nn.LayerNorm) for layer in model.mlp))

    def test_unknown_category_zero_has_a_valid_embedding(self):
        from src.models.mlp_model import TabularMLP

        model = TabularMLP(2, [4], [3])
        result = model(torch.ones(3, 2), torch.zeros(3, 1, dtype=torch.long))

        self.assertTrue(torch.isfinite(result).all())

    def test_weighted_mse_normalizes_by_weight_sum(self):
        from src.models.mlp_model import weighted_mse_loss

        prediction = torch.tensor([0.0, 2.0])
        target = torch.tensor([1.0, 0.0])
        weight = torch.tensor([1.0, 3.0])

        self.assertAlmostEqual(weighted_mse_loss(prediction, target, weight).item(), 3.25)

    def test_target_transform_and_inverse_are_non_negative(self):
        from src.models.mlp_model import inverse_prediction, transform_target

        target = np.array([-1.0, 0.0, 3.0, np.nan])
        transformed = transform_target(target)

        np.testing.assert_allclose(transformed, np.log1p([0.0, 0.0, 3.0, 0.0]))
        np.testing.assert_allclose(inverse_prediction(transformed), [0.0, 0.0, 3.0, 0.0])

    def test_model_bundle_reloads_with_identical_predictions(self):
        from src.models.mlp_model import TabularMLP, load_model_bundle, save_model_bundle

        torch.manual_seed(42)
        model = TabularMLP(3, [4], [2])
        numeric = torch.randn(5, 3)
        categorical = torch.tensor([[0], [1], [2], [3], [0]])
        model.eval()
        expected = model(numeric, categorical)
        metadata = {"numeric_features": ["a", "b", "c"], "horizon": "1m"}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_bundle(model, path, metadata)
            loaded, loaded_metadata = load_model_bundle(path, map_location="cpu")
            loaded.eval()

            torch.testing.assert_close(expected, loaded(numeric, categorical))
            self.assertEqual(loaded_metadata["horizon"], "1m")

    def test_model_bundle_sanitizes_torch_version_for_weights_only_load(self):
        from src.models.mlp_model import TabularMLP, load_model_bundle, save_model_bundle

        model = TabularMLP(1, [], [])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_bundle(model, path, {"torch_version": torch.__version__})
            _, metadata = load_model_bundle(path)

        self.assertIs(type(metadata["torch_version"]), str)


if __name__ == "__main__":
    unittest.main()
