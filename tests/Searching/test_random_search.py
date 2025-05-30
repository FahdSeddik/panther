import random

import pytest

from panther.tuner.SkAutoTuner.Searching.RandomSearch import RandomSearch


class TestRandomSearch:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.param_space = {"param1": [1, 2], "param2": ["a", "b"]}
        self.max_trials = 5
        self.search = RandomSearch(max_trials=self.max_trials)
        self.state_filepath = tmp_path / "test_random_search_state.pkl"

        self.original_random_state = random.getstate()
        yield
        random.setstate(self.original_random_state)

    def test_initialization(self):
        search_default = RandomSearch()
        assert search_default.max_trials == 20
        assert self.search.max_trials == self.max_trials
        assert self.search.current_trial == 0
        assert self.search.param_combinations == []
        assert self.search.history == []
        assert self.search.best_score == -float("inf")
        assert self.search.best_params is None

    def test_initialize(self):
        self.search.initialize(self.param_space)
        assert self.search.param_space == self.param_space
        assert self.search.current_trial == 0
        assert len(self.search.param_combinations) == 4  # 2*2 combinations
        assert self.search.history == []
        assert self.search.best_score == -float("inf")
        assert self.search.best_params is None

    def test_generate_combinations(self):
        self.search.initialize(self.param_space)
        expected_combinations = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]
        assert len(self.search.param_combinations) == len(expected_combinations)
        generated_combos_set = {
            frozenset(combo.items()) for combo in self.search.param_combinations
        }
        expected_combos_set = {
            frozenset(combo.items()) for combo in expected_combinations
        }
        assert generated_combos_set == expected_combos_set

    def test_get_next_params_valid(self):
        self.search.initialize(self.param_space)
        params = self.search.get_next_params()
        assert params is not None
        assert params["param1"] in self.param_space["param1"]
        assert params["param2"] in self.param_space["param2"]
        assert self.search.current_trial == 1
        assert len(self.search.param_combinations) == 3

    def test_get_next_params_no_more_trials(self):
        self.search.initialize(self.param_space)
        self.search.current_trial = self.search.max_trials
        params = self.search.get_next_params()
        assert params is None

    def test_get_next_params_no_more_combinations(self):
        small_param_space = {"param1": [1]}
        search_small = RandomSearch(max_trials=2)
        search_small.initialize(small_param_space)

        params1 = search_small.get_next_params()
        assert params1 is not None
        assert search_small.current_trial == 1
        assert len(search_small.param_combinations) == 0

        params2 = search_small.get_next_params()
        assert params2 is None
        assert search_small.current_trial == 1

    def test_update_history_and_best_score(self):
        self.search.initialize(self.param_space)
        params1 = {"param1": 1, "param2": "a"}
        score1 = 10.0
        self.search.update(params1, score1)

        assert len(self.search.history) == 1
        assert self.search.history[0]["params"] == params1
        assert self.search.history[0]["score"] == score1
        assert self.search.best_score == score1
        assert self.search.best_params == params1

        params2 = {"param1": 2, "param2": "b"}
        score2 = 20.0
        self.search.update(params2, score2)

        assert len(self.search.history) == 2
        assert self.search.history[1]["params"] == params2
        assert self.search.history[1]["score"] == score2
        assert self.search.best_score == score2
        assert self.search.best_params == params2

    def test_update_lower_score(self):
        self.search.initialize(self.param_space)
        params1 = {"param1": 1, "param2": "a"}
        score1 = 20.0
        self.search.update(params1, score1)

        params2 = {"param1": 2, "param2": "b"}
        score2 = 10.0
        self.search.update(params2, score2)

        assert len(self.search.history) == 2
        assert self.search.best_score == score1
        assert self.search.best_params == params1

    def test_save_and_load_state(self):
        self.search.initialize(self.param_space)
        params1 = self.search.get_next_params()
        self.search.update(params1, 10.0)
        params2 = self.search.get_next_params()
        self.search.update(params2, 5.0)

        self.search.save_state(str(self.state_filepath))
        assert self.state_filepath.exists()

        new_search = RandomSearch(max_trials=self.max_trials)
        new_search.load_state(str(self.state_filepath))

        assert new_search.param_space == self.search.param_space
        assert new_search.max_trials == self.search.max_trials
        assert new_search.current_trial == self.search.current_trial
        assert new_search.param_combinations == self.search.param_combinations
        assert new_search.history == self.search.history
        assert new_search.best_score == self.search.best_score
        assert new_search.best_params == self.search.best_params

    def test_get_best_params_and_score(self):
        self.search.initialize(self.param_space)
        assert self.search.get_best_params() is None
        assert self.search.get_best_score() == -float("inf")

        params1 = {"param1": 1, "param2": "a"}
        score1 = 10.0
        self.search.update(params1, score1)
        assert self.search.get_best_params() == params1
        assert self.search.get_best_score() == score1

        params2 = {"param1": 2, "param2": "b"}
        score2 = 5.0
        self.search.update(params2, score2)
        assert self.search.get_best_params() == params1
        assert self.search.get_best_score() == score1

    def test_reset(self):
        self.search.initialize(self.param_space)
        self.search.get_next_params()
        self.search.update({"param1": 1, "param2": "a"}, 10.0)

        self.search.reset()

        assert self.search.current_trial == 0
        assert len(self.search.param_combinations) == 4
        assert self.search.history == []
        assert self.search.best_score == -float("inf")
        assert self.search.best_params is None
        assert self.search.param_space == self.param_space
        assert self.search.max_trials == self.max_trials

    def test_reset_no_initial_param_space(self):
        search_no_init = RandomSearch(max_trials=10)
        search_no_init.current_trial = 5
        search_no_init.history.append({"params": {"p": 1}, "score": 1})
        search_no_init.best_score = 1
        search_no_init.best_params = {"p": 1}

        search_no_init.reset()

        assert search_no_init.current_trial == 0
        assert search_no_init.param_combinations == []
        assert search_no_init.history == []
        assert search_no_init.best_score == -float("inf")
        assert search_no_init.best_params is None
        assert search_no_init.param_space == {}

    def test_is_finished_exact_trials(self):
        search_exact = RandomSearch(max_trials=2)
        param_space_exact = {"p1": [1, 2, 3]}
        search_exact.initialize(param_space_exact)

        assert not search_exact.is_finished()
        search_exact.get_next_params()
        assert not search_exact.is_finished()
        search_exact.get_next_params()
        assert search_exact.is_finished()

        assert search_exact.get_next_params() is None
        assert search_exact.is_finished()

    def test_initialize_with_empty_param_space(self):
        rs = RandomSearch()
        rs.initialize({})
        assert rs.param_space == {}
        assert rs.param_combinations == [{}]

    def test_initialize_with_empty_param_values(self):
        rs = RandomSearch()
        rs.initialize({"param1": []})
        assert rs.param_space == {"param1": []}
        assert rs.param_combinations == []
