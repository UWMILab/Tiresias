import unittest

import numpy as np

from ldp.policy import HORIZON
from rankers.dataset.build_candidate_metric_labels import metric_vectors


class MetricVectorsTest(unittest.TestCase):
    def test_rmse_uses_squared_spatial_norm_then_temporal_mean(self):
        horizon = HORIZON["navigation"]
        truth = np.zeros((horizon, 3))
        candidate = np.zeros((1, horizon, 3))
        candidate[0, 1, 0] = 1.
        candidate[0, 2, 0] = 3.
        values = metric_vectors("navigation", candidate, truth, threshold=1.)
        self.assertAlmostEqual(float(values["ATE"][0]), 15. / 4., places=6)
        self.assertAlmostEqual(float(values["RPE"][0]), np.sqrt(2. / 3.), places=6)

    def test_one_corresponding_step_has_zero_rpe(self):
        candidate = np.zeros((1, HORIZON["navigation"], 3))
        candidate[0, 0, :2] = (3., 4.)
        values = metric_vectors("navigation", candidate, np.zeros((1, 3)), 1.)
        self.assertEqual(float(values["ATE"][0]), 5.)
        self.assertEqual(float(values["RPE"][0]), 0.)

    def test_different_reference_length_remains_supported(self):
        horizon = HORIZON["navigation"]
        truth = np.zeros((3, 3))
        candidate = np.zeros((1, horizon, 3))
        values = metric_vectors("navigation", candidate, truth, threshold=1.)
        self.assertEqual(float(values["ATE"][0]), 0.)
        self.assertEqual(float(values["RPE"][0]), 0.)
        self.assertEqual(float(values["nDTW"][0]), 1.)


if __name__ == "__main__":
    unittest.main()
