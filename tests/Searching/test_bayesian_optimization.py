import unittest

import numpy as np
import torch

from panther.utils.SkAutoTuner.Searching.BayesianOptimization import (
    BayesianOptimization,
)


class TestBayesianOptimization(unittest.TestCase):
    def setUp(self):
        self.param_space = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": [0.01, 0.1, 1],
        }
        self.optimizer = BayesianOptimization(max_trials=10, random_trials=3, seed=42)
        self.optimizer.initialize(self.param_space)

    def test_initialization(self):
        self.assertEqual(self.optimizer.max_trials, 10)
        self.assertEqual(self.optimizer.random_trials, 3)
        self.assertEqual(self.optimizer.current_trial, 0)
        self.assertIsNotNone(self.optimizer.bounds)
        self.assertEqual(len(self.optimizer._param_mapping), len(self.param_space))

    def test_param_conversion(self):
        params = {"C": 1, "kernel": "rbf", "gamma": 0.1}
        point = self.optimizer._params_to_point(params)
        self.assertIsInstance(point, torch.Tensor)
        self.assertEqual(point.shape, (1, len(self.param_space)))

        # Test normalization: C=1 (index 1 of 3) -> 1 / (3-1) = 0.5
        # kernel='rbf' (index 1 of 2) -> 1 / (2-1) = 1.0
        # gamma=0.1 (index 1 of 3) -> 1 / (3-1) = 0.5
        expected_point = torch.tensor([[0.5, 1.0, 0.5]], dtype=torch.float64)
        self.assertTrue(torch.allclose(point, expected_point))

        converted_params = self.optimizer._point_to_params(point)
        self.assertEqual(params, converted_params)

        # Test edge case: single option parameter
        single_option_space = {"single": ["value"]}
        self.optimizer.initialize(single_option_space)
        params_single = {"single": "value"}
        point_single = self.optimizer._params_to_point(params_single)
        # For single option, normalized value should be 0.5
        expected_point_single = torch.tensor([[0.5]], dtype=torch.float64)
        self.assertTrue(torch.allclose(point_single, expected_point_single))
        converted_params_single = self.optimizer._point_to_params(point_single)
        self.assertEqual(params_single, converted_params_single)

        # Re-initialize with original param_space for other tests
        self.optimizer.initialize(self.param_space)

    def test_random_search(self):
        for i in range(self.optimizer.random_trials):
            params = self.optimizer.get_next_params()
            self.assertIsNotNone(params)
            self.assertIn("C", params)
            self.assertIn("kernel", params)
            self.assertIn("gamma", params)
            # Simulate evaluation
            self.optimizer.update(params, np.random.rand())
        self.assertEqual(self.optimizer.current_trial, self.optimizer.random_trials)
        self.assertIsNotNone(self.optimizer.train_x)
        self.assertIsNotNone(self.optimizer.train_y)
        self.assertEqual(len(self.optimizer.train_x), self.optimizer.random_trials)
        self.assertEqual(len(self.optimizer.train_y), self.optimizer.random_trials)

    def test_bayesian_optimization_search(self):
        # Run random trials first
        for _ in range(self.optimizer.random_trials):
            params = self.optimizer.get_next_params()
            self.optimizer.update(params, np.random.rand())

        # Now, BO should kick in
        self.assertIsNone(self.optimizer.model)  # Model not created until BO phase
        params_bo = self.optimizer.get_next_params()
        self.assertIsNotNone(params_bo)
        self.assertIsNotNone(self.optimizer.model)  # Model should be created now
        self.optimizer.update(params_bo, np.random.rand())
        self.assertEqual(self.optimizer.current_trial, self.optimizer.random_trials + 1)

    def test_update_and_best_params(self):
        params1 = {"C": 0.1, "kernel": "linear", "gamma": 0.01}
        score1 = 0.8
        self.optimizer.update(params1, score1)
        self.assertEqual(self.optimizer.best_value, score1)
        self.assertEqual(self.optimizer.best_params, params1)
        self.assertEqual(self.optimizer.get_best_score(), score1)
        self.assertEqual(self.optimizer.get_best_params(), params1)

        params2 = {"C": 1, "kernel": "rbf", "gamma": 0.1}
        score2 = 0.9  # Better score
        self.optimizer.update(params2, score2)
        self.assertEqual(self.optimizer.best_value, score2)
        self.assertEqual(self.optimizer.best_params, params2)

        params3 = {"C": 10, "kernel": "linear", "gamma": 1}
        score3 = 0.7  # Worse score
        self.optimizer.update(params3, score3)
        self.assertEqual(
            self.optimizer.best_value, score2
        )  # Best should still be score2
        self.assertEqual(self.optimizer.best_params, params2)

    def test_save_and_load_state(self):
        import os
        import tempfile

        # Run some trials
        for _ in range(5):
            params = self.optimizer.get_next_params()
            if params is None:
                break
            self.optimizer.update(params, np.random.rand())

        original_best_score = self.optimizer.get_best_score()
        original_best_params = self.optimizer.get_best_params()
        original_current_trial = self.optimizer.current_trial

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmpfile:
            filepath = tmpfile.name

        self.optimizer.save_state(filepath)

        new_optimizer = BayesianOptimization(max_trials=10, random_trials=3, seed=42)
        # Initialize with the same param_space is crucial before loading
        new_optimizer.initialize(self.param_space)
        new_optimizer.load_state(filepath)

        self.assertEqual(new_optimizer.current_trial, original_current_trial)
        self.assertEqual(new_optimizer.get_best_score(), original_best_score)
        self.assertEqual(new_optimizer.get_best_params(), original_best_params)
        self.assertTrue(torch.allclose(new_optimizer.train_x, self.optimizer.train_x))
        self.assertTrue(torch.allclose(new_optimizer.train_y, self.optimizer.train_y))
        self.assertEqual(new_optimizer.train_y_raw, self.optimizer.train_y_raw)

        # Test loading model state if it exists
        if self.optimizer.model is not None:
            self.assertIsNotNone(new_optimizer.model)
            # A more thorough check would involve comparing model parameters,
            # but for now, just checking existence is a good start.

        os.remove(filepath)

    def test_is_finished(self):
        self.assertFalse(self.optimizer.is_finished())
        for i in range(self.optimizer.max_trials):
            params = self.optimizer.get_next_params()
            if params is None:  # Should not happen if max_trials not reached
                self.fail(
                    f"get_next_params returned None at trial {i + 1} but max_trials is {self.optimizer.max_trials}"
                )
            self.optimizer.update(params, np.random.rand())

        self.assertTrue(self.optimizer.is_finished())
        self.assertIsNone(
            self.optimizer.get_next_params()
        )  # Should be None after finishing

    def test_reset(self):
        # Run some trials
        for _ in range(self.optimizer.random_trials + 2):  # Go past random trials
            params = self.optimizer.get_next_params()
            if params is None:
                break
            self.optimizer.update(params, np.random.rand())

        self.assertNotEqual(self.optimizer.current_trial, 0)
        self.assertIsNotNone(self.optimizer.best_value)
        self.assertIsNotNone(self.optimizer.model)  # Model should exist after BO phase

        self.optimizer.reset()
        # Re-initialize is needed after reset as per class design
        self.optimizer.initialize(self.param_space)

        self.assertEqual(self.optimizer.current_trial, 0)
        self.assertIsNone(self.optimizer.best_value)
        self.assertIsNone(self.optimizer.best_params)
        self.assertIsNone(self.optimizer.model)  # Model should be reset
        self.assertEqual(len(self.optimizer.train_x), 0)
        self.assertEqual(len(self.optimizer.train_y), 0)
        self.assertEqual(len(self.optimizer.train_y_raw), 0)

    def test_acquisition_functions(self):
        # Run random trials first to have some data
        for _ in range(self.optimizer.random_trials):
            params = self.optimizer.get_next_params()
            self.optimizer.update(params, np.random.rand())

        acquisition_types = ["ei", "ucb", "logei", "pi", "mes", "lcb", "pm", "psd"]
        for acq_type in acquisition_types:
            with self.subTest(acquisition_type=acq_type):
                optimizer = BayesianOptimization(
                    max_trials=5,
                    random_trials=1,  # Use fewer random trials for faster test
                    acquisition_type=acq_type,
                    seed=42,
                )
                optimizer.initialize(self.param_space)

                # Populate with some initial data similar to the main optimizer
                optimizer.train_x = self.optimizer.train_x.clone()
                optimizer.train_y = self.optimizer.train_y.clone()
                optimizer.train_y_raw = list(self.optimizer.train_y_raw)
                optimizer.current_trial = self.optimizer.current_trial
                optimizer.best_value = self.optimizer.best_value
                optimizer.best_params = self.optimizer.best_params

                # First BO step
                params = optimizer.get_next_params()
                self.assertIsNotNone(
                    params, f"get_next_params returned None for {acq_type}"
                )
                self.assertIsNotNone(
                    optimizer.model, f"Model not created for {acq_type}"
                )

                # Simulate evaluation
                try:
                    optimizer.update(params, np.random.rand())
                except Exception as e:
                    self.fail(f"Update failed for {acq_type} with params {params}: {e}")

                self.assertIn(acq_type, optimizer.acquisition_type)  # Check it was set


if __name__ == "__main__":
    unittest.main()
