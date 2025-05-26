import os
import random
import unittest

from panther.utils.SkAutoTuner.Searching.SimulatedAnnealing import SimulatedAnnealing


class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self):
        self.param_space = {
            "param1": [1, 2, 3],
            "param2": ["a", "b", "c"],
            "param3": [0.1, 0.5, 1.0],
        }
        self.sa = SimulatedAnnealing(
            initial_temperature=10.0,
            cooling_rate=0.9,
            min_temperature=0.1,
            max_iterations=100,
        )
        self.sa.initialize(self.param_space)
        self.test_filepath = "test_sa_state.pkl"

    def tearDown(self):
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)

    def test_initialization(self):
        self.assertEqual(self.sa.initial_temperature, 10.0)
        self.assertEqual(self.sa.cooling_rate, 0.9)
        self.assertEqual(self.sa.min_temperature, 0.1)
        self.assertEqual(self.sa.max_iterations, 100)
        self.assertEqual(self.sa.temperature, 10.0)
        self.assertIn("param1", self.sa.current_solution)
        self.assertIn("param2", self.sa.current_solution)
        self.assertIn("param3", self.sa.current_solution)
        self.assertEqual(
            self.sa.current_score, float("-inf")
        )  # Initial score is unevaluated
        self.assertEqual(self.sa.best_solution, self.sa.current_solution)
        self.assertEqual(self.sa.best_score, float("-inf"))
        self.assertEqual(self.sa.iterations, 0)
        self.assertEqual(self.sa.param_space, self.param_space)

    # def test_initialize_empty_param_space(self):
    #     sa_empty = SimulatedAnnealing()
    #     with self.assertRaises(ValueError):
    #         sa_empty._get_random_solution()  # Accessing before initialize
    #     sa_empty.initialize({})
    #     with self.assertRaises(
    #         ValueError
    #     ):  # Should raise error if param_space is empty when getting random solution
    #         sa_empty._get_random_solution()

    def test_get_random_solution(self):
        solution = self.sa._get_random_solution()
        self.assertIsInstance(solution, dict)
        for param, value in solution.items():
            self.assertIn(param, self.param_space)
            self.assertIn(value, self.param_space[param])

    def test_get_neighbor(self):
        current_solution = self.sa.current_solution.copy()
        neighbor = self.sa._get_neighbor(current_solution)
        self.assertIsInstance(neighbor, dict)
        self.assertEqual(len(neighbor), len(current_solution))

        # Check that at least one parameter is different if possible
        if any(len(v) > 1 for v in self.param_space.values()):
            changed = False
            for param in current_solution:
                if current_solution[param] != neighbor[param]:
                    changed = True
                    break
            self.assertTrue(
                changed,
                "Neighbor should be different from current solution if parameter space allows.",
            )
        else:  # if all params have only one value, neighbor is the same
            self.assertEqual(current_solution, neighbor)

    def test_get_neighbor_single_value_params(self):
        param_space_single = {"param1": [1], "param2": ["a"]}
        sa_single = SimulatedAnnealing()
        sa_single.initialize(param_space_single)
        solution = sa_single._get_random_solution()
        neighbor = sa_single._get_neighbor(solution)
        self.assertEqual(solution, neighbor)

    def test_get_next_params_initial(self):
        # First call to get_next_params should return the initial random solution
        params1 = self.sa.get_next_params()
        self.assertEqual(params1, self.sa.current_solution)
        # Simulate an update
        self.sa.update(params1, 10)  # iterations becomes 1
        params2 = self.sa.get_next_params()  # Now it should generate a neighbor
        self.assertIsInstance(params2, dict)

    def test_update_better_score(self):
        initial_params = self.sa.current_solution.copy()
        initial_score = 5  # dummy score
        self.sa.update(initial_params, initial_score)  # First update sets current score

        self.assertEqual(self.sa.current_score, initial_score)
        self.assertEqual(self.sa.best_score, initial_score)
        self.assertEqual(self.sa.current_solution, initial_params)
        self.assertEqual(self.sa.best_solution, initial_params)
        self.assertEqual(self.sa.iterations, 1)

        better_params = self.sa._get_neighbor(initial_params)
        better_score = initial_score + 5
        self.sa.update(better_params, better_score)

        self.assertEqual(self.sa.current_score, better_score)
        self.assertEqual(self.sa.current_solution, better_params)
        self.assertEqual(self.sa.best_score, better_score)
        self.assertEqual(self.sa.best_solution, better_params)
        self.assertEqual(self.sa.iterations, 2)
        self.assertLess(self.sa.temperature, self.sa.initial_temperature)

    def test_update_worse_score_acceptance(self):
        # Set seed for predictable random behavior
        random.seed(42)
        self.sa.temperature = 1000  # High temperature to ensure acceptance

        initial_params = self.sa.current_solution.copy()
        initial_score = 10
        self.sa.update(initial_params, initial_score)  # First update

        worse_params = self.sa._get_neighbor(initial_params)
        worse_score = initial_score - 5
        self.sa.update(worse_params, worse_score)

        # Check if accepted (current_solution and current_score are updated)
        # This depends on random.random() < math.exp(delta_score / self.temperature)
        # With high temperature, this should be accepted.
        # math.exp(-5/1000) is close to 1. random.random() is likely less.
        if worse_score != initial_score:  # if different params were generated
            self.assertEqual(self.sa.current_score, worse_score)
            self.assertEqual(self.sa.current_solution, worse_params)
        self.assertEqual(
            self.sa.best_score, initial_score
        )  # Best score should remain the initial one
        self.assertEqual(self.sa.best_solution, initial_params)
        self.assertEqual(self.sa.iterations, 2)

    def test_update_worse_score_rejection(self):
        random.seed(1)  # Seed where random.random() might be > acceptance_probability
        self.sa.temperature = 0.0001  # Very low temperature to ensure rejection

        initial_params = self.sa.current_solution.copy()
        initial_score = 10
        self.sa.update(initial_params, initial_score)

        worse_params = self.sa._get_neighbor(initial_params)
        # Ensure worse_params is actually different for the test to be meaningful
        # if all params have single values, this test might not be robust.
        # For this param_space, it should generate different params.
        if worse_params == initial_params and any(
            len(v) > 1 for v in self.param_space.values()
        ):
            # try again to get a different neighbor
            for _ in range(5):
                worse_params = self.sa._get_neighbor(initial_params)
                if worse_params != initial_params:
                    break
            self.assertNotEqual(
                worse_params,
                initial_params,
                "Could not generate a different neighbor for testing rejection.",
            )

        worse_score = initial_score - 5
        self.sa.update(worse_params, worse_score)

        # Check if rejected (current_solution and current_score are NOT updated to worse)
        self.assertEqual(self.sa.current_score, initial_score)
        self.assertEqual(self.sa.current_solution, initial_params)
        self.assertEqual(self.sa.best_score, initial_score)
        self.assertEqual(self.sa.best_solution, initial_params)
        self.assertEqual(self.sa.iterations, 2)

    def test_save_and_load_state(self):
        # Perform some operations
        params1 = self.sa.get_next_params()
        self.sa.update(params1, 10)
        params2 = self.sa.get_next_params()
        self.sa.update(params2, 12)
        self.sa.temperature = 5.0  # Manually change to check

        self.sa.save_state(self.test_filepath)
        self.assertTrue(os.path.exists(self.test_filepath))

        # Create a new SA instance and load state
        new_sa = SimulatedAnnealing()
        new_sa.load_state(self.test_filepath)

        self.assertEqual(new_sa.initial_temperature, self.sa.initial_temperature)
        self.assertEqual(new_sa.cooling_rate, self.sa.cooling_rate)
        self.assertEqual(new_sa.min_temperature, self.sa.min_temperature)
        self.assertEqual(new_sa.max_iterations, self.sa.max_iterations)
        self.assertEqual(new_sa.temperature, self.sa.temperature)
        self.assertEqual(new_sa.current_solution, self.sa.current_solution)
        self.assertEqual(new_sa.current_score, self.sa.current_score)
        self.assertEqual(new_sa.best_solution, self.sa.best_solution)
        self.assertEqual(new_sa.best_score, self.sa.best_score)
        self.assertEqual(new_sa.iterations, self.sa.iterations)
        self.assertEqual(new_sa.param_space, self.sa.param_space)

    def test_is_finished_max_iterations(self):
        self.sa.iterations = self.sa.max_iterations
        self.assertTrue(self.sa.is_finished())

    def test_is_finished_min_temperature(self):
        self.sa.temperature = self.sa.min_temperature
        self.assertTrue(self.sa.is_finished())
        self.sa.temperature = self.sa.min_temperature / 2  # Below min_temperature
        self.assertTrue(self.sa.is_finished())

    def test_get_best_params_and_score(self):
        self.assertEqual(
            self.sa.get_best_params(), self.sa.current_solution
        )  # Initially best is current
        self.assertEqual(self.sa.get_best_score(), float("-inf"))

        params1 = self.sa.current_solution.copy()
        self.sa.update(params1, 10)
        self.assertEqual(self.sa.get_best_params(), params1)
        self.assertEqual(self.sa.get_best_score(), 10)

        # Simulate a better score
        better_params = self.sa._get_neighbor(params1)
        self.sa.update(better_params, 15)
        self.assertEqual(self.sa.get_best_params(), better_params)
        self.assertEqual(self.sa.get_best_score(), 15)

        # Simulate a worse score that is accepted but not best
        self.sa.temperature = 1000  # Ensure acceptance
        worse_accepted_params = self.sa._get_neighbor(better_params)
        self.sa.update(worse_accepted_params, 12)
        self.assertEqual(
            self.sa.get_best_params(), better_params
        )  # Best should not change
        self.assertEqual(self.sa.get_best_score(), 15)

    def test_reset(self):
        # Modify some state
        self.sa.update(self.sa.current_solution, 10)
        self.sa.temperature = 1.0
        self.sa.iterations = 5
        self.sa.param_space = {"new_param": [1]}  # Also check param_space reset

        self.sa.reset()

        self.assertEqual(self.sa.temperature, self.sa.initial_temperature)
        self.assertEqual(self.sa.current_solution, {})
        self.assertEqual(self.sa.current_score, float("-inf"))
        self.assertEqual(self.sa.best_solution, {})
        self.assertEqual(self.sa.best_score, float("-inf"))
        self.assertEqual(self.sa.iterations, 0)
        self.assertEqual(self.sa.param_space, {})  # param_space should be reset

        # After reset, initialize should be called again before use
        with self.assertRaises(ValueError):
            self.sa._get_random_solution()

        self.sa.initialize(self.param_space)  # Re-initialize with original space
        self.assertNotEqual(self.sa.param_space, {})

    def test_get_next_params_after_finish(self):
        self.sa.iterations = self.sa.max_iterations
        # Should return best params if finished
        best_p = self.sa.get_best_params()
        next_p = self.sa.get_next_params()
        self.assertEqual(next_p, best_p)

    def test_get_best_params_uninitialized_param_space_after_reset(self):
        sa = SimulatedAnnealing()
        # sa.param_space is {}
        # sa.best_solution is {}
        # Fallback in get_best_params should not try to call _get_random_solution if param_space is empty
        self.assertEqual(sa.get_best_params(), {})

        sa.initialize({"p1": [1]})
        sa.update({"p1": 1}, 10)
        sa.reset()  # param_space becomes {}, best_solution becomes {}
        self.assertEqual(sa.get_best_params(), {})


if __name__ == "__main__":
    unittest.main()
