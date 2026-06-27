"""
Unit tests for panther.tuner.SkAutoTuner.Searching.GridSearch.
"""

from panther.tuner.SkAutoTuner.Configs.ParamSpec import Categorical, Float, Int
from panther.tuner.SkAutoTuner.Searching.GridSearch import GridSearch


def drain(search):
    params = []
    while not search.is_finished():
        next_params = search.get_next_params()
        if next_params is None:
            break
        params.append(next_params)
    return params


def test_param_specs_expand_to_grid():
    search = GridSearch(max_iterations=10)

    search.initialize(
        {
            "num_terms": Categorical([1, 2]),
            "low_rank": Int(8, 16, step=8),
            "dropout": Float(0.0, 0.1, step=0.1),
        }
    )

    assert drain(search) == [
        {"num_terms": 1, "low_rank": 8, "dropout": 0.0},
        {"num_terms": 1, "low_rank": 8, "dropout": 0.1},
        {"num_terms": 1, "low_rank": 16, "dropout": 0.0},
        {"num_terms": 1, "low_rank": 16, "dropout": 0.1},
        {"num_terms": 2, "low_rank": 8, "dropout": 0.0},
        {"num_terms": 2, "low_rank": 8, "dropout": 0.1},
        {"num_terms": 2, "low_rank": 16, "dropout": 0.0},
        {"num_terms": 2, "low_rank": 16, "dropout": 0.1},
    ]


def test_reinitialize_preserves_requested_budget():
    search = GridSearch(max_iterations=3)

    search.initialize({"x": [1, 2]})
    assert search.max_iterations == 2

    search.initialize({"x": [1, 2, 3, 4, 5]})

    assert search.max_iterations == 3
    assert drain(search) == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_save_load_preserves_iteration_budget(tmp_path):
    path = tmp_path / "grid.pkl"
    search = GridSearch(max_iterations=3)
    search.initialize({"x": [1, 2, 3, 4, 5]})
    search.get_next_params()
    search.save_state(str(path))

    loaded = GridSearch()
    loaded.load_state(str(path))

    assert loaded.requested_max_iterations == 3
    assert loaded.max_iterations == 3
    assert drain(loaded) == [{"x": 2}, {"x": 3}]


def test_is_finished_handles_overshot_iteration():
    search = GridSearch(max_iterations=2)
    search.initialize({"x": [1, 2]})
    search.current_idx = 3

    assert search.is_finished()
    assert search.get_next_params() is None
