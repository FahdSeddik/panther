import math
import random

import pytest

from panther.utils.SkAutoTuner.Searching.Hyperband import Hyperband


class TestHyperband:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.tmp_path = tmp_path
        self.original_random_state = random.getstate()
        yield
        random.setstate(self.original_random_state)

    def test_initialization(self):
        hyperband = Hyperband(max_resource=81, eta=3, min_resource_per_config=1)
        assert hyperband.max_resource == 81
        assert hyperband.eta == 3
        assert hyperband.min_resource_per_config == 1
        assert hyperband.s_max == math.floor(math.log(81 / 1, 3))
        assert hyperband.B == (hyperband.s_max + 1) * 81
        assert hyperband.param_space is None
        assert hyperband.best_config is None
        assert hyperband.best_score == float("-inf")
        assert hyperband.current_s == hyperband.s_max
        assert hyperband.total_iterations_done == 0

    def test_initialize_method(self):
        hyperband = Hyperband()
        param_space = {"p1": [1, 2, 3], "p2": ["a", "b"]}
        hyperband.initialize(param_space)
        assert hyperband.param_space == param_space
        assert hyperband.best_config is None
        assert hyperband.best_score == float("-inf")
        assert hyperband.current_s == hyperband.s_max
        assert hyperband.total_iterations_done == 0
        assert (
            len(hyperband.configs_to_evaluate) > 0
            or hyperband.current_s < hyperband.s_max
        )
        assert (
            len(hyperband.resource_allocations) > 0
            or hyperband.current_s < hyperband.s_max
        )

    def test_get_random_config(self):
        hyperband = Hyperband()
        param_space = {"p1": [1, 2, 3], "p2": ["a", "b"], "p3": [True, False]}
        hyperband.initialize(param_space)
        random_config = hyperband._get_random_config()
        assert random_config["p1"] in param_space["p1"]
        assert random_config["p2"] in param_space["p2"]
        assert random_config["p3"] in param_space["p3"]

    def test_get_random_config_not_initialized(self):
        hyperband = Hyperband()
        with pytest.raises(ValueError):
            hyperband._get_random_config()

    def test_get_next_params_and_update(self):
        hyperband = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        param_space = {"learning_rate": [0.01, 0.1, 1.0], "epochs": [10, 20]}
        hyperband.initialize(param_space)

        params1 = hyperband.get_next_params()
        assert params1 is not None
        assert params1["learning_rate"] in param_space["learning_rate"]
        assert params1["epochs"] in param_space["epochs"]
        hyperband.update(params1, 0.5)
        assert hyperband.best_score == 0.5
        assert hyperband.best_config == params1
        assert len(hyperband.evaluated_configs_in_round) == 1

        params2 = hyperband.get_next_params()
        if params2:
            assert params2 is not None
            hyperband.update(params2, 0.8)
            assert hyperband.best_score == 0.8
            assert hyperband.best_config == params2
            assert len(hyperband.evaluated_configs_in_round) == (
                2 if params1 != params2 else 1
            )

    def test_is_finished(self):
        hyperband = Hyperband(max_resource=9, eta=3, min_resource_per_config=1)
        param_space = {"p1": [1, 2]}
        hyperband.initialize(param_space)

        assert not hyperband.is_finished()

        for i in range(int(hyperband.max_total_iterations) + 5):
            params = hyperband.get_next_params()
            if params is None:
                break
            hyperband.update(params, i * 0.01)

        assert hyperband.is_finished()
        assert hyperband.get_next_params() is None

    def test_save_and_load_state(self):
        filepath = self.tmp_path / "test_hyperband_state.json"
        hyperband1 = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        param_space = {"lr": [0.1, 0.01], "batch_size": [32, 64]}
        hyperband1.initialize(param_space)

        params1 = hyperband1.get_next_params()
        hyperband1.update(params1, 0.75)
        params2 = hyperband1.get_next_params()
        if params2:
            hyperband1.update(params2, 0.85)

        hyperband1.save_state(str(filepath))

        hyperband2 = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        hyperband2.load_state(str(filepath))

        assert hyperband1.max_resource == hyperband2.max_resource
        assert hyperband1.eta == hyperband2.eta
        assert hyperband1.min_resource_per_config == hyperband2.min_resource_per_config
        assert hyperband1.s_max == hyperband2.s_max
        assert hyperband1.B == hyperband2.B
        assert hyperband1.param_space == hyperband2.param_space
        assert hyperband1.best_config == hyperband2.best_config
        assert hyperband1.best_score == hyperband2.best_score
        assert hyperband1.current_s == hyperband2.current_s

        assert len(hyperband1.configs_to_evaluate) == len(
            hyperband2.configs_to_evaluate
        )
        assert hyperband1.resource_allocations == hyperband2.resource_allocations
        assert hyperband1.current_sh_round == hyperband2.current_sh_round
        assert (
            hyperband1.num_configs_for_current_s == hyperband2.num_configs_for_current_s
        )
        assert hyperband1.total_iterations_done == hyperband2.total_iterations_done
        assert hyperband1.max_total_iterations == hyperband2.max_total_iterations

    def test_get_best_before_update(self):
        hyperband = Hyperband()
        param_space = {"p1": [1, 2], "p2": [3, 4]}
        hyperband.initialize(param_space)
        assert hyperband.get_best_score() == float("-inf")
        best_params_before_update = hyperband.get_best_params()
        assert best_params_before_update["p1"] in param_space["p1"]
        assert best_params_before_update["p2"] in param_space["p2"]

    def test_reset_method(self):
        hyperband = Hyperband(max_resource=27, eta=3, min_resource_per_config=1)
        param_space = {"a": [10, 20]}
        original_s_max = hyperband.s_max

        hyperband.initialize(param_space)
        params = hyperband.get_next_params()
        hyperband.update(params, 0.5)

        assert hyperband.best_config is not None
        assert hyperband.best_score != float("-inf")
        assert hyperband.total_iterations_done != 0

        preserved_param_space = hyperband.param_space

        hyperband.reset()

        assert hyperband.best_config is None
        assert hyperband.best_score == float("-inf")
        assert hyperband.total_iterations_done == 0
        assert hyperband.current_s == original_s_max
        assert len(hyperband.configs_to_evaluate) == 0
        assert len(hyperband.resource_allocations) == 0
        assert hyperband.param_space == preserved_param_space

    def test_initialize_empty_param_space(self):
        hyperband = Hyperband()
        with pytest.raises(ValueError):
            hyperband.initialize({})

    def test_initialize_empty_param_values(self):
        hyperband = Hyperband()
        with pytest.raises(IndexError, match="Cannot choose from an empty sequence"):
            hyperband.initialize({"p1": []})
