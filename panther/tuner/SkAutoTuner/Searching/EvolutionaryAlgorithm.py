import copy
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from .SearchAlgorithm import SearchAlgorithm


class EvolutionaryAlgorithm(SearchAlgorithm):
    """
    Implements an Evolutionary Algorithm (specifically a Genetic Algorithm) for hyperparameter tuning.

    This algorithm maintains a population of candidate solutions (sets of parameters)
    and iteratively applies genetic operators like selection, crossover, and mutation
    to evolve better solutions over generations.
    """

    def __init__(
        self,
        population_size: int,
        n_generations: int,
        mutation_rate: float,
        crossover_rate: float,
        tournament_size: int,
        elitism_count: int = 1,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the Evolutionary Algorithm.

        Args:
            population_size: The number of individuals (parameter sets) in each generation.
            n_generations: The number of generations to run the algorithm for.
            mutation_rate: The probability (0.0 to 1.0) of mutating a gene (parameter) in an individual.
            crossover_rate: The probability (0.0 to 1.0) that crossover will occur between two parents.
            tournament_size: The number of individuals to select for a tournament in tournament selection.
            elitism_count: The number of best individuals from the current generation to carry over to the next.
            random_seed: Optional seed for the random number generator for reproducibility.
        """
        if not isinstance(population_size, int) or population_size <= 0:
            raise ValueError("population_size must be a positive integer.")
        if not isinstance(n_generations, int) or n_generations <= 0:
            raise ValueError("n_generations must be a positive integer.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0.")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be between 0.0 and 1.0.")
        if not isinstance(tournament_size, int) or tournament_size <= 0:
            raise ValueError("tournament_size must be a positive integer.")
        if not isinstance(elitism_count, int) or elitism_count < 0:
            raise ValueError("elitism_count must be a non-negative integer.")
        if elitism_count >= population_size:
            raise ValueError("elitism_count must be less than population_size.")
        if tournament_size > population_size:
            # Warning, not strictly an error if population is small, but less effective.
            # For now, allow it, but good to note.
            pass

        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

        if random_seed is not None:
            random.seed(random_seed)

        self.param_space: Optional[Dict[str, List[Any]]] = None
        self.population: List[
            Dict[str, Any]
        ] = []  # List of individuals {'params': Dict, 'score': Optional[float]}
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = -float("inf")
        self.current_generation: int = 0
        self.evaluations_count: int = 0
        self._current_individual_index: int = (
            0  # Index for dispatching individuals from the current population
        )

    def initialize(self, param_space: Dict[str, List[Any]]):
        """
        Initialize the search algorithm with the parameter space and create the initial population.
        """
        if not param_space:
            raise ValueError("Parameter space cannot be empty.")
        for name, values in param_space.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"Parameter '{name}' must have a non-empty list of possible values."
                )

        self.param_space = param_space
        self.reset()

    def _create_individual(self) -> Dict[str, Any]:
        """Helper function to create a random individual based on param_space."""
        if self.param_space is None:
            raise RuntimeError("param_space is not initialized.")
        individual_params = {}
        for param_name, possible_values in self.param_space.items():
            individual_params[param_name] = random.choice(possible_values)
        return individual_params

    def reset(self):
        """Reset the search algorithm to its initial state (requires param_space to be set)."""
        if self.param_space is None:
            raise ValueError(
                "Parameter space not initialized. Call initialize() first."
            )

        self.population = []
        for _ in range(self.population_size):
            self.population.append({"params": self._create_individual(), "score": None})

        self.best_params = {}
        self.best_score = -float("inf")
        self.current_generation = 0
        self.evaluations_count = 0
        self._current_individual_index = 0

    def get_next_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the next set of parameters to try. Evolves to a new generation if the current one is complete.
        Returns None if the search is finished.
        """
        if self.is_finished():
            return None

        if self._current_individual_index >= self.population_size:
            # Current generation fully dispatched, time to evolve
            self._evolve_population()
            self.current_generation += 1
            self._current_individual_index = 0

            # Check if finished after evolution (due to reaching n_generations)
            if self.is_finished():
                return None

        if not self.population or self._current_individual_index >= len(
            self.population
        ):
            # Should not happen if logic is correct, means population is exhausted unexpectedly
            return None

        # Get next individual to evaluate
        individual_to_evaluate = self.population[self._current_individual_index]

        # Ensure it hasn't been scored from a previous partial generation (e.g. after load_state)
        # Or simply rely on the flow: only unevaluated individuals are processed by _evolve_population setting score to None

        self._current_individual_index += 1
        self.evaluations_count += 1

        # Return a deep copy to prevent external modification of internal state
        return copy.deepcopy(individual_to_evaluate["params"])

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.
        """
        if self.param_space is None:  # Should be initialized before any update
            return

        # Find the individual in the population that matches `params` and assign its score
        # This assumes `params` is one of the individuals dispatched from the current generation
        # and its score hasn't been set yet for this generation.
        found_individual = False
        for individual in self.population:
            # We need to compare params carefully. If params were deepcopied, this direct comparison should work.
            # This relies on the 'params' dict not having its 'score' field yet from THIS generation's eval.
            if individual["params"] == params and individual["score"] is None:
                individual["score"] = score
                found_individual = True  # noqa: F841
                break

        # If not found, it might be an issue, or it was already scored (e.g. if update is called twice for same params)
        # For simplicity, we assume valid usage where update is called once per dispatched param set.

        if score > self.best_score:
            self.best_score = score
            self.best_params = copy.deepcopy(params)

    def _all_individuals_scored(self) -> bool:
        """Checks if all individuals in the current population have been scored."""
        if not self.population:
            return (
                False  # Or True if population is empty, depending on desired behavior
            )
        return all(indiv.get("score") is not None for indiv in self.population)

    def _evolve_population(self):
        """Evolve the population to the next generation using selection, crossover, and mutation."""
        if not self._all_individuals_scored():
            # This state should ideally not be reached if get_next_params and update are used correctly.
            # It means we are trying to evolve before all scores are in.
            # For robustness, one might wait or raise an error. Here, we proceed, but scores might be missing.
            # print("Warning: Evolving population with potentially unscored individuals.") # For debugging
            pass  # Or handle more gracefully

        new_population: List[Dict[str, Any]] = []

        # 1. Elitism: Carry over the best individuals
        # Sort by score (descending, hence -indiv['score']). Handle None scores by treating them as worst.
        sorted_population = sorted(
            [indiv for indiv in self.population if indiv["score"] is not None],
            key=lambda x: x["score"],
            reverse=True,
        )

        for i in range(self.elitism_count):
            if i < len(sorted_population):
                elite_individual = sorted_population[i]
                new_population.append(
                    {"params": copy.deepcopy(elite_individual["params"]), "score": None}
                )

        # 2. Generate offspring using selection, crossover, and mutation
        while len(new_population) < self.population_size:
            parent1_struct = self._tournament_selection()
            parent2_struct = self._tournament_selection()

            # Ensure parents were actually selected (e.g. population not empty of scored individuals)
            if parent1_struct is None or parent2_struct is None:
                # Fallback: create random individuals if selection fails (e.g. no scored individuals)
                child1_params = self._create_individual()
                child2_params = self._create_individual()
            else:
                parent1_params = parent1_struct["params"]
                parent2_params = parent2_struct["params"]

                if random.random() < self.crossover_rate:
                    child1_params, child2_params = self._uniform_crossover(
                        parent1_params, parent2_params
                    )
                else:
                    child1_params = copy.deepcopy(parent1_params)
                    child2_params = copy.deepcopy(parent2_params)

            mutated_child1_params = self._mutate(child1_params)
            new_population.append({"params": mutated_child1_params, "score": None})

            if len(new_population) < self.population_size:
                mutated_child2_params = self._mutate(child2_params)
                new_population.append({"params": mutated_child2_params, "score": None})

        self.population = new_population[: self.population_size]  # Ensure correct size

    def _tournament_selection(self) -> Optional[Dict[str, Any]]:
        """Selects an individual using tournament selection."""
        # Consider only individuals that have a score
        scored_population = [
            indiv for indiv in self.population if indiv["score"] is not None
        ]
        if not scored_population:
            return None  # Cannot select if no individuals have scores

        if self.tournament_size > len(scored_population):
            # If tournament is larger than available scored population, compete among all available
            actual_tournament_size = len(scored_population)
        else:
            actual_tournament_size = self.tournament_size

        if actual_tournament_size == 0:  # Should be caught by not scored_population
            return None

        tournament_contenders = random.sample(scored_population, actual_tournament_size)

        best_in_tournament = sorted(
            tournament_contenders, key=lambda x: x["score"], reverse=True
        )[0]
        return (
            best_in_tournament  # Returns the full struct {'params': ..., 'score': ...}
        )

    def _uniform_crossover(
        self, parent1_params: Dict[str, Any], parent2_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Performs uniform crossover between two parents."""
        if self.param_space is None:
            raise RuntimeError("param_space is not initialized.")

        child1_params = {}
        child2_params = {}
        for param_name in self.param_space.keys():
            if random.random() < 0.5:
                child1_params[param_name] = parent1_params[param_name]
                child2_params[param_name] = parent2_params[param_name]
            else:
                child1_params[param_name] = parent2_params[param_name]
                child2_params[param_name] = parent1_params[param_name]
        return child1_params, child2_params

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutates an individual's parameters."""
        if self.param_space is None:
            raise RuntimeError("param_space is not initialized.")

        mutated_params = copy.deepcopy(params)
        for param_name in self.param_space.keys():
            if random.random() < self.mutation_rate:
                possible_values = self.param_space[param_name]
                if possible_values:  # Ensure there are values to choose from
                    # Simple mutation: pick any random value.
                    # Could be enhanced to pick a *different* value if desired.
                    mutated_params[param_name] = random.choice(possible_values)
        return mutated_params

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best set of parameters found so far."""
        return copy.deepcopy(self.best_params)

    def get_best_score(self) -> float:
        """Get the best score achieved so far."""
        return self.best_score

    def is_finished(self) -> bool:
        """Check if the search algorithm has finished (e.g., n_generations reached)."""
        return self.current_generation >= self.n_generations

    def save_state(self, filepath: str):
        """Save the current state of the search algorithm to a file."""
        state = {
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "tournament_size": self.tournament_size,
            "elitism_count": self.elitism_count,
            "param_space": self.param_space,
            "population": self.population,  # population includes params and scores
            "best_params": self.best_params,
            "best_score": self.best_score,
            "current_generation": self.current_generation,
            "evaluations_count": self.evaluations_count,
            "_current_individual_index": self._current_individual_index,
            "random_state": random.getstate(),  # Save Python's random generator state
        }
        try:
            with open(filepath, "w") as f:
                json.dump(state, f, indent=4)
        except (IOError, TypeError) as e:
            # Handle cases where state is not JSON serializable or file cannot be written
            # print(f"Error saving state: {e}") # Or raise a custom exception
            raise RuntimeError(f"Failed to save state to {filepath}: {e}")

    def load_state(self, filepath: str):
        """Load the state of the search algorithm from a file."""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            # print(f"Error loading state: {e}")
            raise RuntimeError(f"Failed to load state from {filepath}: {e}")

        self.population_size = state["population_size"]
        self.n_generations = state["n_generations"]
        self.mutation_rate = state["mutation_rate"]
        self.crossover_rate = state["crossover_rate"]
        self.tournament_size = state["tournament_size"]
        self.elitism_count = state["elitism_count"]
        self.param_space = state["param_space"]
        self.population = state["population"]
        self.best_params = state["best_params"]
        self.best_score = state["best_score"]
        self.current_generation = state["current_generation"]
        self.evaluations_count = state["evaluations_count"]
        self._current_individual_index = state["_current_individual_index"]

        if "random_state" in state:
            # Ensure the loaded state is converted to a tuple for random.setstate
            random_state_from_file = state["random_state"]
            if isinstance(random_state_from_file, list):
                random.setstate(tuple(random_state_from_file))
            else:
                # If it's already a tuple (e.g. if loaded from a non-JSON source in future)
                random.setstate(random_state_from_file)

        # After loading, the algorithm is ready to continue from where it left off.
        # Ensure that loaded population scores are correctly handled (e.g. None for unevaluated)


# Example Usage (Illustrative - not part of the class itself)
if __name__ == "__main__":
    param_space_example = {
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [32, 64, 128],
        "optimizer": ["adam", "sgd"],
    }

    ga = EvolutionaryAlgorithm(
        population_size=10,
        n_generations=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        tournament_size=3,
        elitism_count=1,
        random_seed=42,
    )

    ga.initialize(param_space_example)

    print(
        f"Running GA for {ga.n_generations} generations with population size {ga.population_size}"
    )

    for gen in range(
        ga.n_generations * ga.population_size + 5
    ):  # Try to get a few more to see None
        if ga.is_finished():
            print(
                f"Search finished at evaluation {ga.evaluations_count}, generation {ga.current_generation}."
            )
            break

        print(
            f"\n--- Generation: {ga.current_generation}, Evaluation: {ga.evaluations_count + 1} ---"
        )
        current_params = ga.get_next_params()

        if current_params is None:
            print(
                "Received None from get_next_params, likely search is finished or waiting for evolution."
            )
            if ga.is_finished():
                print("Confirmed finished.")
                break
            else:
                print(
                    "All params for current generation dispatched. Next call should evolve."
                )
                continue

        print(f"Trying params: {current_params}")

        # Simulate evaluation
        mock_score = sum(
            current_params.get(k, 0)
            if isinstance(current_params.get(k), (int, float))
            else random.random()
            for k in current_params
        )
        mock_score += random.uniform(-0.1, 0.1)  # Add some noise
        if current_params.get("optimizer") == "adam":
            mock_score += 0.5

        print(f"Achieved score: {mock_score}")
        ga.update(current_params, mock_score)
        print(
            f"Best score so far: {ga.get_best_score()}, Best params: {ga.get_best_params()}"
        )

        # Illustrate save/load (optional)
        if ga.evaluations_count == 15:
            print("\nSAVING STATE...")
            ga.save_state("ga_state.json")
            print("LOADING STATE...")
            # Create a new instance or re-initialize to demonstrate loading
            ga_loaded = EvolutionaryAlgorithm(
                1, 1, 0, 0, 1
            )  # Dummy params, will be overwritten
            ga_loaded.load_state("ga_state.json")
            ga = ga_loaded  # Continue with the loaded state
            print(
                f"Resumed. Current gen: {ga.current_generation}, evals: {ga.evaluations_count}"
            )
            print(f"Population size of loaded: {len(ga.population)}")

    print("\n--- Final Results ---")
    print(f"Best score: {ga.get_best_score()}")
    print(f"Best parameters: {ga.get_best_params()}")
    print(f"Total evaluations: {ga.evaluations_count}")
    print(f"Generations run: {ga.current_generation}")
