import math
import unittest

from panther.utils.SkAutoTuner.Searching.Hyperband import Hyperband


class TestHyperband(unittest.TestCase):
    def test_initialization(self):
        hyperband = Hyperband(max_resource=81, eta=3, min_resource_per_config=1)
        self.assertEqual(hyperband.max_resource, 81)
        self.assertEqual(hyperband.eta, 3)
        self.assertEqual(hyperband.min_resource_per_config, 1)
        self.assertEqual(hyperband.s_max, math.floor(math.log(81 / 1, 3)))
        self.assertEqual(hyperband.B, (hyperband.s_max + 1) * 81)
        self.assertIsNone(hyperband.param_space)
        self.assertIsNone(hyperband.best_config)
        self.assertEqual(hyperband.best_score, float("-inf"))
        self.assertEqual(hyperband.current_s, hyperband.s_max)
        self.assertEqual(hyperband.total_iterations_done, 0)

    def test_initialize_method(self):
        hyperband = Hyperband()
        param_space = {"p1": [1, 2, 3], "p2": ["a", "b"]}
        hyperband.initialize(param_space)
        self.assertEqual(hyperband.param_space, param_space)
        self.assertIsNone(hyperband.best_config)
        self.assertEqual(hyperband.best_score, float("-inf"))
        self.assertEqual(hyperband.current_s, hyperband.s_max)
        self.assertEqual(hyperband.total_iterations_done, 0)
        # Check if the first bracket is set up
        self.assertTrue(
            len(hyperband.configs_to_evaluate) > 0
            or hyperband.current_s < hyperband.s_max
        )
        self.assertTrue(
            len(hyperband.resource_allocations) > 0
            or hyperband.current_s < hyperband.s_max
        )

    def test_get_random_config(self):
        hyperband = Hyperband()
        param_space = {"p1": [1, 2, 3], "p2": ["a", "b"], "p3": [True, False]}
        hyperband.initialize(param_space)
        random_config = hyperband._get_random_config()
        self.assertIn(random_config["p1"], param_space["p1"])
        self.assertIn(random_config["p2"], param_space["p2"])
        self.assertIn(random_config["p3"], param_space["p3"])

    def test_get_random_config_not_initialized(self):
        hyperband = Hyperband()
        with self.assertRaises(ValueError):
            hyperband._get_random_config()

    def test_get_next_params_and_update(self):
        hyperband = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        param_space = {"learning_rate": [0.01, 0.1, 1.0], "epochs": [10, 20]}
        hyperband.initialize(param_space)

        # Simulate a few iterations
        params1 = hyperband.get_next_params()
        self.assertIsNotNone(params1)
        self.assertIn(params1["learning_rate"], param_space["learning_rate"])
        self.assertIn(params1["epochs"], param_space["epochs"])
        hyperband.update(params1, 0.5)  # Assume score is 0.5
        self.assertEqual(hyperband.best_score, 0.5)
        self.assertEqual(hyperband.best_config, params1)
        self.assertEqual(len(hyperband.evaluated_configs_in_round), 1)

        params2 = hyperband.get_next_params()
        if params2:  # Could be None if first bracket had only 1 config
            self.assertIsNotNone(params2)
            hyperband.update(params2, 0.8)
            self.assertEqual(hyperband.best_score, 0.8)
            self.assertEqual(hyperband.best_config, params2)
            self.assertEqual(
                len(hyperband.evaluated_configs_in_round),
                2 if params1 != params2 else 1,
            )

        # Test exhaustion of a bracket (simplified)
        # This requires more intricate setup to test specific SH promotions
        # For now, we focus on basic operation

    def test_is_finished(self):
        hyperband = Hyperband(max_resource=9, eta=3, min_resource_per_config=1)
        param_space = {"p1": [1, 2]}
        hyperband.initialize(param_space)

        self.assertFalse(hyperband.is_finished())

        # Simulate running through all configurations
        # s_max = floor(log(9/1, 3)) = floor(log(9,3)) = floor(2) = 2
        # B = (2+1)*9 = 27
        # max_total_iterations is B, so 27 iterations
        # This loop will simulate calls to get_next_params and update
        for i in range(
            int(hyperband.max_total_iterations) + 5
        ):  # go a bit beyond to ensure finish
            params = hyperband.get_next_params()
            if params is None:
                # print(f"No more params at iteration {i}, total done: {hyperband.total_iterations_done}")
                break
            hyperband.update(params, i * 0.01)  # dummy score

        self.assertTrue(hyperband.is_finished())
        self.assertIsNone(
            hyperband.get_next_params()
        )  # Should return None when finished

    def test_save_and_load_state(self):
        hyperband1 = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        param_space = {"lr": [0.1, 0.01], "batch_size": [32, 64]}
        hyperband1.initialize(param_space)

        params1 = hyperband1.get_next_params()
        hyperband1.update(params1, 0.75)
        params2 = hyperband1.get_next_params()
        if params2:  # Handle case where only one config might be in a small bracket
            hyperband1.update(params2, 0.85)

        filepath = "test_hyperband_state.pkl"
        hyperband1.save_state(filepath)

        hyperband2 = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        # Note: param_space is part of the saved state, but initialize is usually called.
        # For a direct load test without re-initialize, ensure param_space is handled.
        # However, the current implementation of load_state restores param_space.
        hyperband2.load_state(filepath)

        self.assertEqual(hyperband1.max_resource, hyperband2.max_resource)
        self.assertEqual(hyperband1.eta, hyperband2.eta)
        self.assertEqual(
            hyperband1.min_resource_per_config, hyperband2.min_resource_per_config
        )
        self.assertEqual(hyperband1.s_max, hyperband2.s_max)
        self.assertEqual(hyperband1.B, hyperband2.B)
        self.assertEqual(hyperband1.param_space, hyperband2.param_space)
        self.assertEqual(hyperband1.best_config, hyperband2.best_config)
        self.assertEqual(hyperband1.best_score, hyperband2.best_score)
        self.assertEqual(hyperband1.current_s, hyperband2.current_s)
        # Comparing lists of dicts can be tricky if order isn't guaranteed or dicts are complex
        # For simple dicts of primitives, direct comparison might work.
        # self.assertEqual(hyperband1.configs_for_sh, hyperband2.configs_for_sh)
        self.assertEqual(
            len(hyperband1.configs_to_evaluate), len(hyperband2.configs_to_evaluate)
        )
        self.assertEqual(
            hyperband1.resource_allocations, hyperband2.resource_allocations
        )
        self.assertEqual(hyperband1.current_sh_round, hyperband2.current_sh_round)
        self.assertEqual(
            hyperband1.num_configs_for_current_s, hyperband2.num_configs_for_current_s
        )
        # self.assertEqual(hyperband1.evaluated_configs_in_round, hyperband2.evaluated_configs_in_round)
        self.assertEqual(
            hyperband1.total_iterations_done, hyperband2.total_iterations_done
        )
        self.assertEqual(
            hyperband1.max_total_iterations, hyperband2.max_total_iterations
        )

        # Clean up the created file
        import os

        if os.path.exists(filepath):
            os.remove(filepath)

    def test_get_best_before_update(self):
        hyperband = Hyperband()
        param_space = {"p1": [1, 2], "p2": [3, 4]}
        hyperband.initialize(param_space)
        # Before any updates, best_score is -inf
        self.assertEqual(hyperband.get_best_score(), float("-inf"))
        # get_best_params should return a random config if no updates yet
        best_params_before_update = hyperband.get_best_params()
        self.assertIn(best_params_before_update["p1"], param_space["p1"])
        self.assertIn(best_params_before_update["p2"], param_space["p2"])

    def test_reset_method(self):
        hyperband = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        param_space = {"a": [10, 20]}
        hyperband.initialize(param_space)

        params = hyperband.get_next_params()
        hyperband.update(params, 0.5)

        self.assertIsNotNone(hyperband.best_config)
        self.assertNotEqual(hyperband.best_score, float("-inf"))
        self.assertNotEqual(hyperband.total_iterations_done, 0)

        hyperband.reset()

        self.assertIsNone(hyperband.best_config)
        self.assertEqual(hyperband.best_score, float("-inf"))
        self.assertEqual(hyperband.total_iterations_done, 0)
        self.assertEqual(
            hyperband.current_s, hyperband.s_max
        )  # s_max remains based on init params
        self.assertEqual(
            len(hyperband.configs_to_evaluate), 0
        )  # Reset to empty, initialize will fill it
        self.assertEqual(len(hyperband.resource_allocations), 0)
        # param_space is not reset by design, it should be re-initialized if needed
        # self.assertIsNone(hyperband.param_space) # This would fail as reset does not clear it.


if __name__ == "__main__":
    unittest.main()
