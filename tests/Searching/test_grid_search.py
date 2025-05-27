import random

import pytest

from panther.utils.SkAutoTuner.Searching.GridSearch import GridSearch


class TestGridSearch:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.grid_search = GridSearch()
        self.param_space = {"param1": [1, 2], "param2": ["a", "b"]}
        self.expected_combinations = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]
        self.test_filepath = tmp_path / "test_grid_search_state.pkl"

        self.original_random_state = random.getstate()
        yield
        random.setstate(self.original_random_state)

    def test_initialization_default(self):
        assert self.grid_search.param_space == {}
        assert self.grid_search.current_idx == 0
        assert self.grid_search.param_combinations == []
        assert self.grid_search.history == []
        assert self.grid_search.best_score == -float("inf")
        assert self.grid_search.best_params is None

    def test_initialize(self):
        self.grid_search.initialize(self.param_space)
        assert self.grid_search.param_space == self.param_space
        assert self.grid_search.current_idx == 0
        assert len(self.grid_search.param_combinations) == 4
        generated_combos_set = {
            frozenset(combo.items()) for combo in self.grid_search.param_combinations
        }
        expected_combos_set = {
            frozenset(combo.items()) for combo in self.expected_combinations
        }
        assert generated_combos_set == expected_combos_set
        assert self.grid_search.history == []
        assert self.grid_search.best_score == -float("inf")
        assert self.grid_search.best_params is None

    def test_generate_combinations(self):
        self.grid_search.initialize(self.param_space)
        assert len(self.grid_search.param_combinations) == 4
        generated_combos_set = {
            frozenset(combo.items()) for combo in self.grid_search.param_combinations
        }
        expected_combos_set = {
            frozenset(combo.items()) for combo in self.expected_combinations
        }
        assert generated_combos_set == expected_combos_set

        empty_space = {}
        self.grid_search.initialize(empty_space)
        assert self.grid_search.param_combinations == [{}]

        single_param_space = {"p1": [1, 2, 3]}
        self.grid_search.initialize(single_param_space)
        expected_single = [{"p1": 1}, {"p1": 2}, {"p1": 3}]
        generated_single_set = {
            frozenset(combo.items()) for combo in self.grid_search.param_combinations
        }
        expected_single_set = {frozenset(combo.items()) for combo in expected_single}
        assert len(self.grid_search.param_combinations) == 3
        assert generated_single_set == expected_single_set

    def test_get_next_params(self):
        self.grid_search.initialize(self.param_space)

        sorted_param_names = sorted(self.param_space.keys())
        all_fetched_params = []
        while True:
            params = self.grid_search.get_next_params()
            if params is None:
                break
            all_fetched_params.append(params)

        assert len(all_fetched_params) == len(self.expected_combinations)
        assert {frozenset(p.items()) for p in all_fetched_params} == {
            frozenset(p.items()) for p in self.expected_combinations
        }
        assert self.grid_search.current_idx == len(self.expected_combinations)
        assert self.grid_search.get_next_params() is None

    def test_update(self):
        self.grid_search.initialize(self.param_space)
        params1 = {"param1": 1, "param2": "a"}
        score1 = 10.0
        self.grid_search.update(params1, score1)

        assert len(self.grid_search.history) == 1
        assert self.grid_search.history[0]["params"] == params1
        assert self.grid_search.history[0]["score"] == score1
        assert self.grid_search.best_score == score1
        assert self.grid_search.best_params == params1

        params2 = {"param1": 1, "param2": "b"}
        score2 = 20.0
        self.grid_search.update(params2, score2)

        assert len(self.grid_search.history) == 2
        assert self.grid_search.history[1]["params"] == params2
        assert self.grid_search.history[1]["score"] == score2
        assert self.grid_search.best_score == score2
        assert self.grid_search.best_params == params2

        params3 = {"param1": 2, "param2": "a"}
        score3 = 5.0
        self.grid_search.update(params3, score3)

        assert len(self.grid_search.history) == 3
        assert self.grid_search.history[2]["params"] == params3
        assert self.grid_search.history[2]["score"] == score3
        assert self.grid_search.best_score == score2
        assert self.grid_search.best_params == params2

    def test_save_and_load_state(self):
        self.grid_search.initialize(self.param_space)
        params1 = self.grid_search.get_next_params()
        if params1:
            self.grid_search.update(params1, 10.0)

        self.grid_search.save_state(str(self.test_filepath))

        new_grid_search = GridSearch()
        new_grid_search.load_state(str(self.test_filepath))

        assert new_grid_search.param_space == self.grid_search.param_space
        assert new_grid_search.current_idx == self.grid_search.current_idx
        assert new_grid_search.param_combinations == self.grid_search.param_combinations
        assert new_grid_search.history == self.grid_search.history
        assert new_grid_search.best_score == self.grid_search.best_score
        assert new_grid_search.best_params == self.grid_search.best_params

    def test_get_best_params_and_score(self):
        assert self.grid_search.get_best_params() is None
        assert self.grid_search.get_best_score() == -float("inf")

        self.grid_search.initialize(self.param_space)
        params1_from_iter = self.grid_search.param_combinations[0]
        score1 = 10.0
        self.grid_search.update(params1_from_iter, score1)
        assert self.grid_search.get_best_params() == params1_from_iter
        assert self.grid_search.get_best_score() == score1

        if len(self.grid_search.param_combinations) > 1:
            params2_from_iter = self.grid_search.param_combinations[1]
            score2 = 5.0
            self.grid_search.update(params2_from_iter, score2)
            assert self.grid_search.get_best_params() == params1_from_iter
            assert self.grid_search.get_best_score() == score1

    def test_reset(self):
        self.grid_search.initialize(self.param_space)
        self.grid_search.get_next_params()
        if self.grid_search.param_combinations:
            param_to_update = self.grid_search.param_combinations[0]
            self.grid_search.update(param_to_update, 10)

        original_param_space = self.grid_search.param_space.copy()
        original_combinations_content_set = {
            frozenset(c.items()) for c in self.grid_search.param_combinations
        }

        self.grid_search.reset()

        assert self.grid_search.current_idx == 0
        assert self.grid_search.history == []
        assert self.grid_search.best_score == -float("inf")
        assert self.grid_search.best_params is None
        assert self.grid_search.param_space == original_param_space
        assert {
            frozenset(c.items()) for c in self.grid_search.param_combinations
        } == original_combinations_content_set

    def test_is_finished(self):
        self.grid_search.initialize(self.param_space)
        assert not self.grid_search.is_finished()

        for _ in range(len(self.expected_combinations)):
            assert not (self.grid_search.is_finished())
            self.grid_search.get_next_params()

        assert self.grid_search.is_finished()

        empty_search = GridSearch()
        empty_search.initialize({})
        assert not empty_search.is_finished()
        empty_search.get_next_params()
        assert empty_search.is_finished()

    def test_save_load_preserves_combination_order(self):
        self.grid_search.initialize(self.param_space)
        param_taken = self.grid_search.get_next_params()
        if param_taken:
            self.grid_search.update(param_taken, 5)
        self.grid_search.save_state(str(self.test_filepath))

        loaded_gs = GridSearch()
        loaded_gs.load_state(str(self.test_filepath))

        assert self.grid_search.param_combinations == loaded_gs.param_combinations
        assert self.grid_search.current_idx == loaded_gs.current_idx

    def test_initialize_empty_param_space_values(self):
        gs = GridSearch()
        gs.initialize({"param1": []})
        assert gs.param_space == {"param1": []}
        assert gs.param_combinations == []
