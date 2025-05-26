import os
import unittest

from panther.utils.SkAutoTuner.Searching.RandomSearch import RandomSearch


class TestRandomSearch(unittest.TestCase):
    def setUp(self):
        self.param_space = {"param1": [1, 2], "param2": ["a", "b"]}
        self.max_trials = 5
        self.search = RandomSearch(max_trials=self.max_trials)
        self.state_filepath = "test_random_search_state.pkl"

    def tearDown(self):
        if os.path.exists(self.state_filepath):
            os.remove(self.state_filepath)

    def test_initialization(self):
        search_default = RandomSearch()
        self.assertEqual(search_default.max_trials, 20)
        self.assertEqual(self.search.max_trials, self.max_trials)
        self.assertEqual(self.search.current_trial, 0)
        self.assertEqual(self.search.param_combinations, [])
        self.assertEqual(self.search.history, [])
        self.assertEqual(self.search.best_score, -float("inf"))
        self.assertIsNone(self.search.best_params)

    def test_initialize(self):
        self.search.initialize(self.param_space)
        self.assertEqual(self.search.param_space, self.param_space)
        self.assertEqual(self.search.current_trial, 0)
        self.assertEqual(len(self.search.param_combinations), 4)  # 2*2 combinations
        self.assertEqual(self.search.history, [])
        self.assertEqual(self.search.best_score, -float("inf"))
        self.assertIsNone(self.search.best_params)

    def test_generate_combinations(self):
        self.search.initialize(self.param_space)
        expected_combinations = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]
        # Order doesn't matter for param_combinations as it's popped randomly
        self.assertEqual(
            len(self.search.param_combinations), len(expected_combinations)
        )
        for combo in self.search.param_combinations:
            self.assertIn(combo, expected_combinations)
        for expected_combo in expected_combinations:
            self.assertIn(expected_combo, self.search.param_combinations)

    def test_get_next_params_valid(self):
        self.search.initialize(self.param_space)
        params = self.search.get_next_params()
        self.assertIsNotNone(params)
        self.assertIn(params["param1"], self.param_space["param1"])
        self.assertIn(params["param2"], self.param_space["param2"])
        self.assertEqual(self.search.current_trial, 1)
        self.assertEqual(len(self.search.param_combinations), 3)

    def test_get_next_params_no_more_trials(self):
        self.search.initialize(self.param_space)
        self.search.current_trial = self.search.max_trials  # Manually set to max_trials
        params = self.search.get_next_params()
        self.assertIsNone(params)

    def test_get_next_params_no_more_combinations(self):
        small_param_space = {"param1": [1]}
        search_small = RandomSearch(max_trials=2)
        search_small.initialize(small_param_space)

        params1 = search_small.get_next_params()
        self.assertIsNotNone(params1)
        self.assertEqual(search_small.current_trial, 1)
        self.assertEqual(len(search_small.param_combinations), 0)

        params2 = search_small.get_next_params()  # No more combinations
        self.assertIsNone(params2)
        self.assertEqual(
            search_small.current_trial, 1
        )  # Trial doesn't increment if no params returned

    def test_update_history_and_best_score(self):
        self.search.initialize(self.param_space)
        params1 = {"param1": 1, "param2": "a"}
        score1 = 10.0
        self.search.update(params1, score1)

        self.assertEqual(len(self.search.history), 1)
        self.assertEqual(self.search.history[0]["params"], params1)
        self.assertEqual(self.search.history[0]["score"], score1)
        self.assertEqual(self.search.best_score, score1)
        self.assertEqual(self.search.best_params, params1)

        params2 = {"param1": 2, "param2": "b"}
        score2 = 20.0
        self.search.update(params2, score2)

        self.assertEqual(len(self.search.history), 2)
        self.assertEqual(self.search.history[1]["params"], params2)
        self.assertEqual(self.search.history[1]["score"], score2)
        self.assertEqual(self.search.best_score, score2)
        self.assertEqual(self.search.best_params, params2)

    def test_update_lower_score(self):
        self.search.initialize(self.param_space)
        params1 = {"param1": 1, "param2": "a"}
        score1 = 20.0
        self.search.update(params1, score1)

        params2 = {"param1": 2, "param2": "b"}
        score2 = 10.0  # Lower score
        self.search.update(params2, score2)

        self.assertEqual(len(self.search.history), 2)
        self.assertEqual(
            self.search.best_score, score1
        )  # Should remain the higher score
        self.assertEqual(
            self.search.best_params, params1
        )  # Should remain params of higher score

    def test_save_and_load_state(self):
        self.search.initialize(self.param_space)
        # Perform some operations
        params1 = self.search.get_next_params()
        self.search.update(params1, 10.0)
        params2 = self.search.get_next_params()
        self.search.update(params2, 5.0)

        # Save state
        self.search.save_state(self.state_filepath)
        self.assertTrue(os.path.exists(self.state_filepath))

        # Create a new instance and load state
        new_search = RandomSearch(max_trials=self.max_trials)
        new_search.load_state(self.state_filepath)

        self.assertEqual(new_search.param_space, self.search.param_space)
        self.assertEqual(new_search.max_trials, self.search.max_trials)
        self.assertEqual(new_search.current_trial, self.search.current_trial)
        self.assertEqual(new_search.param_combinations, self.search.param_combinations)
        self.assertEqual(new_search.history, self.search.history)
        self.assertEqual(new_search.best_score, self.search.best_score)
        self.assertEqual(new_search.best_params, self.search.best_params)

    def test_get_best_params_and_score(self):
        self.search.initialize(self.param_space)
        self.assertIsNone(self.search.get_best_params())
        self.assertEqual(self.search.get_best_score(), -float("inf"))

        params1 = {"param1": 1, "param2": "a"}
        score1 = 10.0
        self.search.update(params1, score1)
        self.assertEqual(self.search.get_best_params(), params1)
        self.assertEqual(self.search.get_best_score(), score1)

        params2 = {"param1": 2, "param2": "b"}
        score2 = 5.0  # Lower score
        self.search.update(params2, score2)
        self.assertEqual(
            self.search.get_best_params(), params1
        )  # Should still be params1
        self.assertEqual(self.search.get_best_score(), score1)  # Should still be score1

    def test_reset(self):
        self.search.initialize(self.param_space)
        self.search.get_next_params()
        self.search.update({"param1": 1, "param2": "a"}, 10.0)

        self.search.reset()

        self.assertEqual(self.search.current_trial, 0)
        self.assertEqual(len(self.search.param_combinations), 4)  # Regenerated
        self.assertEqual(self.search.history, [])
        self.assertEqual(self.search.best_score, -float("inf"))
        self.assertIsNone(self.search.best_params)
        self.assertEqual(
            self.search.param_space, self.param_space
        )  # Should be preserved
        self.assertEqual(self.search.max_trials, self.max_trials)  # Should be preserved

    def test_reset_no_initial_param_space(self):
        # Test reset when param_space was never initialized
        search_no_init = RandomSearch(max_trials=10)
        search_no_init.current_trial = 5
        search_no_init.history.append({"params": {"p": 1}, "score": 1})
        search_no_init.best_score = 1
        search_no_init.best_params = {"p": 1}

        search_no_init.reset()

        self.assertEqual(search_no_init.current_trial, 0)
        self.assertEqual(
            search_no_init.param_combinations, []
        )  # No param_space to generate from
        self.assertEqual(search_no_init.history, [])
        self.assertEqual(search_no_init.best_score, -float("inf"))
        self.assertIsNone(search_no_init.best_params)
        self.assertEqual(search_no_init.param_space, {})  # Should be empty

    # def test_is_finished(self):
    #     self.search.initialize(self.param_space)
    #     self.assertFalse(self.search.is_finished())

    #     for _ in range(self.max_trials):
    #         params = self.search.get_next_params()
    #         if params is None:  # Could happen if max_trials > num_combinations
    #             break
    #     self.assertTrue(self.search.is_finished())

    def test_is_finished_exact_trials(self):
        search_exact = RandomSearch(max_trials=2)
        param_space_exact = {"p1": [1, 2, 3]}  # 3 combinations
        search_exact.initialize(param_space_exact)

        self.assertFalse(search_exact.is_finished())
        search_exact.get_next_params()  # trial 1
        self.assertFalse(search_exact.is_finished())
        search_exact.get_next_params()  # trial 2
        self.assertTrue(search_exact.is_finished())

        # Try getting more params
        self.assertIsNone(search_exact.get_next_params())
        self.assertTrue(search_exact.is_finished())  # Still finished


if __name__ == "__main__":
    unittest.main()
