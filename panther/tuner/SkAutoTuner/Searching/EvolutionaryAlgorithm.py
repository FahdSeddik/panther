import pickle
from typing import Any, Dict, List

import torch

from .SearchAlgorithm import SearchAlgorithm


class EvolutionaryAlgorithm(SearchAlgorithm):
    """
    Abstract base class for evolutionary search algorithms.

    This class provides a framework for implementing evolutionary algorithms
    for hyperparameter tuning. It includes common operations such as
    initialization, selection, crossover, and mutation.
    """

    def __init__(
        self,
        population_size=5,
        n_generations=10,
        mutation_rate=0.5,
        crossover_rate=0.5,
        tournament_size=5,
        elitism_count=1,
        selection_type="roulette_wheel_selection",
        crossover_type="uniform_crossover",
    ):
        """
        Initializes the EvolutionaryAlgorithm.

        Args:
            population_size: The number of individuals in each generation.
            n_generations: The total number of generations to evolve.
            mutation_rate: The probability of mutating an individual.
            crossover_rate: The probability of performing crossover between two parents.
            tournament_size: The number of individuals participating in a tournament selection.
            elitism_count: The number of best individuals to carry over to the next generation.
            selection_type: The method used for selecting parents (e.g., "roulette_wheel_selection").
            crossover_type: The method used for crossover (e.g., "uniform_crossover").
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.__checks()
        self.reset()

    def __checks(self):
        """
        Validates the initialization parameters.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if self.population_size <= 0:
            raise ValueError("population_size must be greater than 0")
        if self.n_generations <= 0:
            raise ValueError("n_generations must be greater than 0")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("mutation_rate must be between 0 and 1")
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("crossover_rate must be between 0 and 1")
        if self.tournament_size <= 0:
            raise ValueError("tournament_size must be greater than 0")
        if self.elitism_count < 0:
            raise ValueError("elitism_count must be greater than or equal to 0")
        if self.selection_type not in [
            "roulette_wheel_selection",
            "tournament_selection",
            "random_selection",
        ]:
            raise ValueError(
                f"selection_type must be one of ['roulette_wheel_selection', 'tournament_selection', 'random_selection'], got {self.selection_type}"
            )
        if self.crossover_type not in [
            "uniform_crossover",
            "single_point_crossover",
            "two_point_crossover",
        ]:
            raise ValueError(
                f"crossover_type must be one of ['uniform_crossover', 'single_point_crossover', 'two_point_crossover'], got {self.crossover_type}"
            )

    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with the parameter space.
        This method sets up the initial population for the evolutionary process.

        Args:
            param_space: Dictionary of parameter names and their possible values.
        """
        self.reset()
        self.param_space = param_space
        self.indexed_param_space = self._generate_indexed_param_space()
        self.tournament_size = min(self.tournament_size, self.population_size)

        i = 0
        # Create the initial population by randomly selecting from the indexed parameter space
        while i < self.population_size:
            choice = torch.random.randint(
                low=0, high=len(self.indexed_param_space), size=1, dtype=torch.int32
            )[0]

            params = self.indexed_param_space[choice]
            self.indexed_param_space.pop(choice)
            self.population_to_eval[i] = params
            i = i + 1

    def _my_product(self, value_lists: List[List[Any]]) -> List[tuple]:
        """
        Computes the Cartesian product of a list of lists.
        Example: _my_product([[1, 2], ['a', 'b']]) -> [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

        Args:
            value_lists: A list of lists, where each inner list contains values for a parameter.

        Returns:
            A list of tuples, where each tuple is a unique combination of values.
        """
        if not value_lists:
            return [()]  # Product of no lists is a list with one empty tuple

        # Initialize with a list containing an empty tuple.
        # Each item from the subsequent lists will be appended to these tuples.
        product_tuples = [()]

        for current_list_values in value_lists:
            if not current_list_values:
                # If any input list is empty, the Cartesian product is empty.
                return []

            # Build the new product by combining existing tuples with items from the current list
            product_tuples = [
                existing_tuple + (item,)
                for existing_tuple in product_tuples
                for item in current_list_values
            ]

        return product_tuples

    def _generate_indexed_param_space(self):
        """
        Generates an indexed parameter space from the provided param_space.
        The indexed parameter space is a list of dictionaries, where each dictionary
        represents a unique combination of parameter values. This is used for
        initializing the population and for mutation if new individuals are needed.

        Raises:
            ValueError: If param_space is None.

        Returns:
            List[Dict[str, Any]]: The indexed parameter space.
        """
        if self.param_space is None:
            raise ValueError("param_space is None")

        param_names = list(self.param_space.keys())
        value_lists = list(self.param_space.values())

        values_combined = self._my_product(value_lists)
        indexed_param_space = [{} for _ in range(len(values_combined))]

        i = 0
        for value_combination in values_combined:
            param = {}
            j = 0
            for value in value_combination:
                param[param_names[j]] = value
                j = j + 1
            indexed_param_space[i] = param
            i = i + 1

        return indexed_param_space

    def get_next_params(self) -> Dict[str, Any]:
        """
        Get the next set of parameters to try.

        Returns:
            Dictionary of parameter names and values to try, or None if the search is finished.
        """
        if self.is_finished():
            return None

        if len(self.population_to_eval) > 0:
            # If there are individuals in the current population to evaluate
            param = self.population_to_eval[self.current_individual_index]
            self.population_to_eval.pop(self.current_individual_index)
            self.current_individual_index = self.current_individual_index + 1
            return param
        else:
            # If the current population has been evaluated, evolve to the next generation
            self._evolve()
            # Recursively call get_next_params to get an individual from the new population
            self.get_next_params()

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.
        This involves recording the score of the evaluated parameters and
        updating the best-known parameters if the current score is better.

        Args:
            params: Dictionary of parameter names and values that were tried.
            score: The evaluation score for the parameters.
        """
        params["score"] = score
        self.evaluated_population.append(params)

        if self.best_score < score:
            self.best_score = score
            self.best_params = params

    def _evolve(self):
        """
        Evolves the population to the next generation.
        This method implements the core evolutionary loop:
        1. Normalizes scores.
        2. Applies elitism.
        3. Performs selection, crossover, and mutation to generate the new population.
        """
        self.curr_generation = self.curr_generation + 1
        new_population = []

        # Normalize scores for selection (e.g., roulette wheel)
        scores = torch.tensor([indv["score"] for indv in self.evaluated_population])
        min_score = scores.min()
        max_score = scores.max()
        for indv in self.evaluated_population:
            # Add a small epsilon to avoid division by zero if all scores are the same
            indv["score"] = indv["score"] - min_score / ((max_score - min_score) + 1e-4)

        population_sorted_normalized = sorted(
            self.evaluated_population, key=lambda indv: -indv["score"]
        )

        # 1. Elitism: Carry over the best individuals to the next generation
        new_population[: self.elitism_count] = population_sorted_normalized[
            : self.elitism_count
        ]

        # 2. Genetic operations: Fill the rest of the new population
        while len(new_population) < self.population_size:
            parent1 = self._selection(population_sorted_normalized)
            parent2 = self._selection(population_sorted_normalized)

            child1 = None
            child2 = None

            # Crossover
            if (
                parent1 is not None
                and parent2 is not None
                and self.crossover_rate > torch.rand().item()
            ):
                child1, child2 = self._cross_over(parent1, parent2)
            else:
                # If crossover doesn't happen or parents are missing,
                # either keep parents or generate new random individuals
                if parent1 is not None:
                    child1 = parent1.copy()
                elif (
                    len(new_population) < self.population_size
                ):  # Ensure we don't exceed population size
                    choice = torch.random.randint(
                        low=0,
                        high=len(
                            self.indexed_param_space
                        ),  # Ensure index is within bounds
                        size=1,
                        dtype=torch.int32,
                    )[0]

                    params = self.indexed_param_space[choice]
                    self.indexed_param_space.pop(
                        choice
                    )  # Avoid re-selecting the same random individual
                    child1 = params

                if parent2 is not None:
                    child2 = parent2.copy()
                elif (
                    len(new_population) < self.population_size
                ):  # Ensure we don't exceed population size
                    choice = torch.random.randint(
                        low=0,
                        high=len(
                            self.indexed_param_space
                        ),  # Ensure index is within bounds
                        size=1,
                        dtype=torch.int32,
                    )[0]

                    params = self.indexed_param_space[choice]
                    self.indexed_param_space.pop(
                        choice
                    )  # Avoid re-selecting the same random individual
                    child2 = params

            # Mutation
            if child1 is not None:
                child1 = self._mutate(child1)

            if child2 is not None:
                child2 = self._mutate(child2)

            if child1 is not None:
                new_population.append(child1)

            if (
                child2 is not None and len(new_population) < self.population_size
            ):  # Ensure we don't exceed population size
                new_population.append(child2)

        self.population_to_eval = new_population
        self.evaluated_population = []
        self.current_individual_index = 0

    def _roulette_wheel_selection(self, population_sorted_normalized):
        """
        Performs roulette wheel selection.
        Individuals are selected based on their fitness (score), where higher
        fitness means a higher probability of being selected.

        Args:
            population_sorted_normalized: A list of individuals sorted by score (descending)
                                          and scores normalized.

        Returns:
            The selected individual.
        """
        scores = torch.tensor([indv["score"] for indv in population_sorted_normalized])
        probabilities = scores / scores.sum()
        selected_index = torch.multinomial(probabilities, 1).item()
        indv = population_sorted_normalized[selected_index]
        population_sorted_normalized.pop(
            selected_index
        )  # Remove selected individual to avoid re-selection in the same step
        return indv

    def _tournament_selection(self, population_sorted_normalized):
        """
        Performs tournament selection.
        A subset of individuals (tournament) is randomly selected from the population,
        and the fittest individual from this tournament is chosen as a parent.

        Args:
            population_sorted_normalized: A list of individuals sorted by score (descending)
                                          and scores normalized.

        Returns:
            The selected individual.
        """
        if not population_sorted_normalized:
            return None

        tournament_candidates_indices = torch.randperm(
            len(population_sorted_normalized)
        )[: self.tournament_size]

        tournament_candidates = [
            population_sorted_normalized[i] for i in tournament_candidates_indices
        ]

        # need to sort them again unfortunatelly
        tournament_candidates_sorted = sorted(
            tournament_candidates, key=lambda indv: -indv["score"]
        )

        # The population is already sorted by score, so the first one is the best
        winner = tournament_candidates_sorted[0]

        # Find and remove the winner from the original population list to avoid re-selection.
        # We remove the specific 'winner' object by identity to ensure the correct individual is removed,
        # which is more robust if multiple individuals have identical parameters.
        index_to_remove = -1
        for i, indv in enumerate(population_sorted_normalized):
            if indv is winner:  # Compare by object identity
                index_to_remove = i
                break

        if index_to_remove != -1:
            population_sorted_normalized.pop(index_to_remove)
        # else:
        # If winner is not found by identity, it indicates an issue.
        # The previous value-based comparison might have masked this if it found a
        # different object with the same parameters.
        # For robustness, one might add an assertion or logging here, e.g.:
        # print("Warning: Winner object not found for removal in tournament selection by identity.")
        # This situation would imply an unexpected state in population management.

        return winner

    def _random_selection(self, population_sorted_normalized):
        """
        Performs random selection.
        An individual is chosen randomly from the population.

        Args:
            population_sorted_normalized: A list of individuals (can be unsorted for this method).

        Returns:
            The selected individual.
        """
        if not population_sorted_normalized:
            return None
        choice = torch.random.randint(
            low=0,
            high=len(population_sorted_normalized),  # Ensure index is within bounds
            size=1,
            dtype=torch.int32,
        )[0]

        indv = population_sorted_normalized[choice]
        population_sorted_normalized.pop(choice)
        return indv

    def _selection(self, population_sorted_normalized):
        """
        Selects individuals from the population for reproduction.
        Currently supports roulette wheel selection.

        Args:
            population_sorted_normalized: A list of individuals sorted by score (descending)
                                          and scores normalized.

        Returns:
            The selected individual.

        Raises:
            NotImplementedError: If the specified selection_type is not implemented.
        """
        if self.selection_type == "roulette_wheel_selection":
            return self._roulette_wheel_selection(population_sorted_normalized)
        elif self.selection_type == "tournament_selection":
            return self._tournament_selection(population_sorted_normalized)
        elif self.selection_type == "random_selection":
            return self._random_selection(population_sorted_normalized)
        else:
            raise (
                NotImplementedError(
                    f"Selection type {self.selection_type} is not implemented."
                )
            )

    def _cross_over(self, parent1, parent2):
        """
        Performs crossover between two parents to create two children.
        The type of crossover is determined by `self.crossover_type`.

        Args:
            parent1: The first parent individual (dictionary of parameters).
            parent2: The second parent individual (dictionary of parameters).

        Returns:
            A tuple containing two children (child1, child2).
        """
        if self.crossover_type == "uniform_crossover":
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_type == "single_point_crossover":
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_type == "two_point_crossover":
            return self._two_point_crossover(parent1, parent2)
        else:
            raise NotImplementedError(
                f"Crossover type {self.crossover_type} is not implemented."
            )

    def _uniform_crossover(self, parent1, parent2):
        """
        Performs uniform crossover between two parents to create two children.
        For each parameter, the value is randomly chosen from one of the parents.

        Args:
            parent1: The first parent individual.
            parent2: The second parent individual.

        Returns:
            A tuple containing two children.
        """
        child1 = {}
        child2 = {}

        for param in parent1.keys():
            if param == "score":
                continue
            if torch.rand().item() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]

        return child1, child2

    def _single_point_crossover(self, parent1, parent2):
        """
        Performs single-point crossover between two parents.
        A single crossover point is selected, and parameters are swapped
        between parents after this point to create two children.

        Args:
            parent1: The first parent individual.
            parent2: The second parent individual.

        Returns:
            A tuple containing two children.
        """
        child1 = {}
        child2 = {}
        params_keys = [k for k in parent1.keys() if k != "score"]
        if not params_keys:  # No parameters to crossover
            return parent1.copy(), parent2.copy()

        crossover_point = torch.randint(0, len(params_keys) + 1, (1,)).item()

        for i, key in enumerate(params_keys):
            if i < crossover_point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        return child1, child2

    def _two_point_crossover(self, parent1, parent2):
        """
        Performs two-point crossover between two parents.
        Two crossover points are selected, and parameters between these
        points are swapped between parents.

        Args:
            parent1: The first parent individual.
            parent2: The second parent individual.

        Returns:
            A tuple containing two children.
        """
        child1 = {}
        child2 = {}
        params_keys = [k for k in parent1.keys() if k != "score"]
        if (
            not params_keys or len(params_keys) < 2
        ):  # Not enough parameters for two points
            return self._uniform_crossover(parent1, parent2)  # fallback to uniform

        point1 = torch.randint(0, len(params_keys), (1,)).item()
        point2 = torch.randint(0, len(params_keys), (1,)).item()

        start_point = min(point1, point2)
        end_point = max(point1, point2)

        for i, key in enumerate(params_keys):
            if start_point <= i < end_point:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
            else:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
        return child1, child2

    def _mutate(self, indv):
        """
        Performs mutation on an individual.
        For each parameter in the individual, there is a `mutation_rate` chance
        that its value will be changed to a randomly selected value from its
        possible range in the `param_space`.

        Args:
            indv: The individual (dictionary of parameters) to mutate.

        Returns:
            The mutated individual.
        """
        new_indv = indv.copy()

        for param in indv.keys():
            if param == "score":
                continue

            values = self.param_space[param]

            choice = torch.random.randint(
                low=0, high=len(values), size=1, dtype=torch.int32
            )[0]

            if torch.rand().item() < self.mutation_rate:
                new_indv[param] = values[choice]

        return new_indv

    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.

        Args:
            filepath: The path to the file where the state should be saved.
        """
        state = {
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "tournament_size": self.tournament_size,
            "elitism_count": self.elitism_count,
            "selection_type": self.selection_type,
            "param_space": self.param_space,
            "curr_generation": self.curr_generation,
            "indexed_param_space": self.indexed_param_space,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "population_to_eval": self.population_to_eval,
            "evaluated_population": self.evaluated_population,
            "current_individual_index": self.current_individual_index,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.

        Args:
            filepath: The path to the file from which the state should be loaded.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.population_size = state["population_size"]
        self.n_generations = state["n_generations"]
        self.mutation_rate = state["mutation_rate"]
        self.crossover_rate = state["crossover_rate"]
        self.tournament_size = state["tournament_size"]
        self.elitism_count = state["elitism_count"]
        self.selection_type = state["selection_type"]
        self.param_space = state["param_space"]
        self.curr_generation = state["curr_generation"]
        self.indexed_param_space = state["indexed_param_space"]
        self.best_params = state["best_params"]
        self.best_score = state["best_score"]
        self.population_to_eval = state["population_to_eval"]
        self.evaluated_population = state["evaluated_population"]
        self.current_individual_index = state["current_individual_index"]

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best set of parameters found so far.

        Returns:
            Dictionary of the best parameter names and values.
        """
        return self.best_params

    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.

        Returns:
            The best score.
        """
        return self.best_score

    def reset(self):
        """
        Reset the search algorithm to its initial state.
        """
        self.param_space = None
        self.curr_generation = 0
        self.indexed_param_space = []
        self.best_params = None
        self.best_score = -float("inf")
        self.population_to_eval = [
            {} for _ in range(self.population_size)
        ]  # individuals to be evaluated in the current generation
        self.evaluated_population = []  # individuals that have been evaluated in the current generation, with their scores
        self.current_individual_index = 0

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search (e.g., budget exhausted).

        Returns:
            True if the search is finished, False otherwise.
        """
        return self.curr_generation == self.n_generations
