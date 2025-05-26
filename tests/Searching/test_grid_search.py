import os
import unittest

from panther.utils.SkAutoTuner.Searching.GridSearch import GridSearch


class TestGridSearch(unittest.TestCase):
    def setUp(self):
        self.grid_search = GridSearch()
        self.param_space = {"param1": [1, 2], "param2": ["a", "b"]}
        self.expected_combinations = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]
        self.test_filepath = "test_grid_search_state.pkl"

    def tearDown(self):
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)

    def test_initialization_default(self):
        self.assertEqual(self.grid_search.param_space, {})
        self.assertEqual(self.grid_search.current_idx, 0)
        self.assertEqual(self.grid_search.param_combinations, [])
        self.assertEqual(self.grid_search.history, [])
        self.assertEqual(self.grid_search.best_score, -float("inf"))
        self.assertIsNone(self.grid_search.best_params)

    def test_initialize(self):
        self.grid_search.initialize(self.param_space)
        self.assertEqual(self.grid_search.param_space, self.param_space)
        self.assertEqual(self.grid_search.current_idx, 0)
        self.assertEqual(len(self.grid_search.param_combinations), 4)
        # Order might vary depending on dict iteration, so check content
        for combo in self.expected_combinations:
            self.assertIn(combo, self.grid_search.param_combinations)
        self.assertEqual(self.grid_search.history, [])
        self.assertEqual(self.grid_search.best_score, -float("inf"))
        self.assertIsNone(self.grid_search.best_params)

    def test_generate_combinations(self):
        self.grid_search.initialize(
            self.param_space
        )  # initialize calls _generate_combinations
        self.assertEqual(len(self.grid_search.param_combinations), 4)
        for combo in self.expected_combinations:
            self.assertIn(combo, self.grid_search.param_combinations)

        empty_space = {}
        self.grid_search.initialize(empty_space)
        self.assertEqual(
            self.grid_search.param_combinations, [{}]
        )  # Product of empty gives one empty dict

        single_param_space = {"p1": [1, 2, 3]}
        self.grid_search.initialize(single_param_space)
        expected_single = [{"p1": 1}, {"p1": 2}, {"p1": 3}]
        self.assertEqual(len(self.grid_search.param_combinations), 3)
        for combo in expected_single:
            self.assertIn(combo, self.grid_search.param_combinations)

    def test_get_next_params(self):
        self.grid_search.initialize(self.param_space)

        # To ensure order for this test, we sort the generated combinations
        # based on a canonical representation (e.g., sorted items string)
        # This makes the test deterministic if internal dict ordering changes.
        self.grid_search.param_combinations.sort(key=lambda d: str(sorted(d.items())))

        # We also sort self.expected_combinations to match this deterministic order.
        sorted_expected_combinations = sorted(
            self.expected_combinations, key=lambda d: str(sorted(d.items()))
        )

        for i in range(len(sorted_expected_combinations)):
            params = self.grid_search.get_next_params()
            self.assertEqual(params, sorted_expected_combinations[i])
            self.assertEqual(self.grid_search.current_idx, i + 1)

        self.assertIsNone(self.grid_search.get_next_params())
        self.assertEqual(
            self.grid_search.current_idx, len(sorted_expected_combinations)
        )

    def test_update(self):
        self.grid_search.initialize(self.param_space)
        params1 = {"param1": 1, "param2": "a"}
        score1 = 10.0
        self.grid_search.update(params1, score1)

        self.assertEqual(len(self.grid_search.history), 1)
        self.assertEqual(self.grid_search.history[0]["params"], params1)
        self.assertEqual(self.grid_search.history[0]["score"], score1)
        self.assertEqual(self.grid_search.best_score, score1)
        self.assertEqual(self.grid_search.best_params, params1)

        params2 = {"param1": 1, "param2": "b"}
        score2 = 20.0
        self.grid_search.update(params2, score2)

        self.assertEqual(len(self.grid_search.history), 2)
        self.assertEqual(self.grid_search.history[1]["params"], params2)
        self.assertEqual(self.grid_search.history[1]["score"], score2)
        self.assertEqual(self.grid_search.best_score, score2)
        self.assertEqual(self.grid_search.best_params, params2)

        params3 = {"param1": 2, "param2": "a"}
        score3 = 5.0  # Lower score
        self.grid_search.update(params3, score3)

        self.assertEqual(len(self.grid_search.history), 3)
        self.assertEqual(self.grid_search.history[2]["params"], params3)
        self.assertEqual(self.grid_search.history[2]["score"], score3)
        self.assertEqual(
            self.grid_search.best_score, score2
        )  # Best score should not change
        self.assertEqual(
            self.grid_search.best_params, params2
        )  # Best params should not change

    def test_save_and_load_state(self):
        self.grid_search.initialize(self.param_space)
        params1 = self.grid_search.get_next_params()
        self.grid_search.update(params1, 10.0)

        self.grid_search.save_state(self.test_filepath)

        new_grid_search = GridSearch()
        new_grid_search.load_state(self.test_filepath)

        self.assertEqual(new_grid_search.param_space, self.grid_search.param_space)
        self.assertEqual(new_grid_search.current_idx, self.grid_search.current_idx)
        # Sort for comparison as order might not be preserved by pickle/dict
        self.assertEqual(
            sorted(
                new_grid_search.param_combinations, key=lambda d: str(sorted(d.items()))
            ),
            sorted(
                self.grid_search.param_combinations,
                key=lambda d: str(sorted(d.items())),
            ),
        )
        self.assertEqual(new_grid_search.history, self.grid_search.history)
        self.assertEqual(new_grid_search.best_score, self.grid_search.best_score)
        self.assertEqual(new_grid_search.best_params, self.grid_search.best_params)

    def test_get_best_params_and_score(self):
        self.assertEqual(self.grid_search.get_best_params(), None)
        self.assertEqual(self.grid_search.get_best_score(), -float("inf"))

        self.grid_search.initialize(self.param_space)
        params1 = {"param1": 1, "param2": "a"}
        score1 = 10.0
        self.grid_search.update(params1, score1)
        self.assertEqual(self.grid_search.get_best_params(), params1)
        self.assertEqual(self.grid_search.get_best_score(), score1)

        params2 = {"param1": 1, "param2": "b"}
        score2 = 5.0  # Lower score
        self.grid_search.update(params2, score2)
        self.assertEqual(
            self.grid_search.get_best_params(), params1
        )  # Should remain params1
        self.assertEqual(
            self.grid_search.get_best_score(), score1
        )  # Should remain score1

    def test_reset(self):
        self.grid_search.initialize(self.param_space)
        self.grid_search.get_next_params()  # current_idx = 1
        self.grid_search.update({"param1": 1, "param2": "a"}, 10)

        original_param_space = self.grid_search.param_space.copy()
        original_combinations_count = len(self.grid_search.param_combinations)

        self.grid_search.reset()

        self.assertEqual(self.grid_search.current_idx, 0)
        self.assertEqual(self.grid_search.history, [])
        self.assertEqual(self.grid_search.best_score, -float("inf"))
        self.assertIsNone(self.grid_search.best_params)
        self.assertEqual(
            self.grid_search.param_space, original_param_space
        )  # param_space should be preserved
        self.assertEqual(
            len(self.grid_search.param_combinations), original_combinations_count
        )  # combinations should be preserved

    def test_is_finished(self):
        self.grid_search.initialize(self.param_space)
        self.assertFalse(self.grid_search.is_finished())

        for _ in range(len(self.expected_combinations)):
            self.assertFalse(
                self.grid_search.is_finished()
            )  # Should not be finished until all are GOTTEN
            self.grid_search.get_next_params()

        self.assertTrue(
            self.grid_search.is_finished()
        )  # Now all params have been retrieved

        # Test with an empty param space
        empty_search = GridSearch()
        empty_search.initialize({})
        # It generates one empty dict combination [{}]
        self.assertFalse(empty_search.is_finished())
        empty_search.get_next_params()  # Gets the {}
        self.assertTrue(empty_search.is_finished())

    def test_save_load_preserves_combination_order_if_possible(self):
        self.grid_search.initialize(self.param_space)
        # Make param_combinations deterministic for this test
        self.grid_search.param_combinations.sort(
            key=lambda d: (d["param1"], d["param2"])
        )

        self.grid_search.get_next_params()
        self.grid_search.update({"param1": 1, "param2": "a"}, 5)
        self.grid_search.save_state(self.test_filepath)

        loaded_gs = GridSearch()
        loaded_gs.load_state(self.test_filepath)

        self.assertEqual(
            self.grid_search.param_combinations, loaded_gs.param_combinations
        )
        self.assertEqual(self.grid_search.current_idx, loaded_gs.current_idx)


if __name__ == "__main__":
    unittest.main()
