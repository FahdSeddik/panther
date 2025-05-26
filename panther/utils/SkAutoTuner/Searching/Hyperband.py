import math
import pickle
import random
from typing import Any, Dict, List, Tuple

from .SearchAlgorithm import SearchAlgorithm


class Hyperband(SearchAlgorithm):
    def __init__(
        self, max_resource: int = 81, eta: int = 3, min_resource_per_config: int = 1
    ):
        self.max_resource = (
            max_resource  # R: Maximum resources allocated to a single configuration
        )
        self.eta = eta  # Defines the proportion of configurations discarded in each round of Successive Halving
        self.min_resource_per_config = (
            min_resource_per_config  # r: minimum resources per config
        )

        self.s_max = math.floor(
            math.log(self.max_resource / self.min_resource_per_config, self.eta)
        )
        self.B = (
            self.s_max + 1
        ) * self.max_resource  # Total budget for one Hyperband iteration

        self.param_space: Dict[str, List] = None
        self.best_config: Dict[str, Any] = None
        self.best_score: float = float("-inf")

        # State for current Hyperband iteration
        self.current_s: int = self.s_max
        self.configs_for_sh: List[
            Tuple[Dict[str, Any], float]
        ] = []  # (config, score) for current SH bracket
        self.configs_to_evaluate: List[Dict[str, Any]] = []
        self.resource_allocations: List[
            int
        ] = []  # Resources for each round in current SH
        self.current_sh_round: int = 0  # Index for self.resource_allocations
        self.num_configs_for_current_s: int = 0
        self.evaluated_configs_in_round: List[Tuple[Dict[str, Any], float]] = []
        self.total_iterations_done = 0  # For is_finished, if we want a hard stop
        self.max_total_iterations = self.B  # A loose upper bound, can be configured

    def initialize(self, param_space: Dict[str, List]):
        self.param_space = param_space
        self.best_config = None
        self.best_score = float("-inf")
        self.current_s = self.s_max
        self.total_iterations_done = 0
        self._setup_next_sh_bracket()

    def _get_random_config(self) -> Dict[str, Any]:
        if not self.param_space:
            raise ValueError("Parameter space not initialized.")
        return {
            param: random.choice(values) for param, values in self.param_space.items()
        }

    def _setup_next_sh_bracket(self):
        if self.current_s < 0:
            # All Hyperband iterations (brackets) are done
            self.configs_to_evaluate = []
            return

        n = math.ceil(
            self.B
            / self.max_resource
            / (self.current_s + 1)
            * (self.eta**self.current_s)
        )
        self.num_configs_for_current_s = int(n)
        r = (
            self.max_resource
            / (self.eta**self.current_s)
            * self.min_resource_per_config
        )  # Initial resource for this s
        if r < self.min_resource_per_config:
            r = self.min_resource_per_config

        self.configs_for_sh = []  # Reset for the new bracket
        self.configs_to_evaluate = [
            self._get_random_config() for _ in range(self.num_configs_for_current_s)
        ]

        self.resource_allocations = []
        self.current_sh_round = 0
        self.evaluated_configs_in_round = []

        for i in range(self.current_s + 1):
            n_i = math.floor(n / (self.eta**i))
            r_i = r * (self.eta**i)
            if r_i > self.max_resource:
                r_i = self.max_resource
            if n_i > 0:
                self.resource_allocations.append(int(r_i))
            else:
                break  # No more configs to run in this bracket

        if not self.resource_allocations:  # If r was already too high, or n too low
            self.current_s -= 1
            self._setup_next_sh_bracket()
            return

        # The first set of evaluations will use self.resource_allocations[0]
        # The parameter to be passed to the user is the actual config
        # The resource level is an internal concept for Hyperband's update logic
        # For now, we just need to queue up the initial configs for the first round of SH
        # get_next_params() will pop from self.configs_to_evaluate

    def get_next_params(self) -> Dict[str, Any]:
        if self.is_finished():
            return None

        if not self.configs_to_evaluate:
            # This implies a round of SH is finished or a bracket is finished.
            # We need to process results and set up the next round/bracket.
            self._process_sh_round_results()

            if not self.configs_to_evaluate and self.current_s < 0:
                # All brackets completed
                return None
            elif not self.configs_to_evaluate:
                # This could happen if _process_sh_round_results decided there's nothing more
                # or if a bracket resulted in no evaluatable configurations.
                # Try to set up next bracket or indicate finish.
                if self.current_s >= 0:
                    self._setup_next_sh_bracket()
                    if (
                        not self.configs_to_evaluate
                    ):  # Still no configs after trying next bracket
                        return None
                else:
                    return None  # Should be caught by is_finished

        if not self.configs_to_evaluate:
            # If after all processing, there are still no configs, then we are done.
            return None

        # Return the next config to evaluate. The resource level is implicitly
        # self.resource_allocations[self.current_sh_round]
        # The actual resource value is not passed back with the params, as the SearchAlgorithm
        # interface doesn't support it. The user of Hyperband (e.g., an AutoTuner)
        # would need to be aware of how to interpret this if resources are dynamic.
        # For now, we assume the score reflects performance for the implicitly managed resource.
        next_config = self.configs_to_evaluate.pop(0)
        return next_config

    def _process_sh_round_results(self):
        # This method is called when self.configs_to_evaluate is empty,
        # meaning all configs for the current SH round have been dispatched.
        # Now we need their results from self.evaluated_configs_in_round.

        if (
            not self.evaluated_configs_in_round
        ):  # No results came in, or called prematurely
            if not self.configs_for_sh and self.current_sh_round == 0:
                # If it's the first round of a new bracket and configs_for_sh is empty,
                # it implies the initial set from _setup_next_sh_bracket was empty or
                # get_next_params was called when it shouldn't have been.
                # This state should ideally be handled by ensuring _setup_next_sh_bracket
                # always provides configs if the bracket is valid.
                if self.current_s >= 0:
                    self.current_s -= 1  # Try to move to next bracket
                    self._setup_next_sh_bracket()
                return
            else:
                # Waiting for more results from the current round
                return

        # Sort the evaluated configs by score (descending)
        self.evaluated_configs_in_round.sort(key=lambda x: x[1], reverse=True)

        self.current_sh_round += 1

        if self.current_sh_round >= len(self.resource_allocations):
            # Finished all rounds for the current s (bracket)
            self.current_s -= 1
            self.evaluated_configs_in_round = []  # Clear for next bracket
            self._setup_next_sh_bracket()
        else:
            # Promote top configurations for the next round of SH
            num_to_keep = math.floor(len(self.evaluated_configs_in_round) / self.eta)
            if num_to_keep == 0 and len(self.evaluated_configs_in_round) > 0:
                num_to_keep = 1  # Always keep at least one if there were results

            self.configs_to_evaluate = [
                config
                for config, score in self.evaluated_configs_in_round[:num_to_keep]
            ]
            self.evaluated_configs_in_round = []  # Clear for the next set of evaluations

            if not self.configs_to_evaluate and self.current_s >= 0:
                # No configs promoted, means this bracket is effectively done.
                # Move to the next s.
                self.current_s -= 1
                self._setup_next_sh_bracket()

    def update(self, params: Dict[str, Any], score: float):
        self.total_iterations_done += 1
        if score > self.best_score:
            self.best_score = score
            self.best_config = params

        # Store the result of the evaluated config
        self.evaluated_configs_in_round.append((params, score))

        # Note: The actual processing of these results to select top k for the next round
        # happens in get_next_params() when it realizes it needs more configs to dispatch.
        # This is because update() is called for each config one by one.

    def save_state(self, filepath: str):
        state = {
            "max_resource": self.max_resource,
            "eta": self.eta,
            "min_resource_per_config": self.min_resource_per_config,
            "s_max": self.s_max,
            "B": self.B,
            "param_space": self.param_space,
            "best_config": self.best_config,
            "best_score": self.best_score,
            "current_s": self.current_s,
            "configs_for_sh": self.configs_for_sh,  # May not be perfectly serializable if it contains complex objects not intended for pickle
            "configs_to_evaluate": self.configs_to_evaluate,
            "resource_allocations": self.resource_allocations,
            "current_sh_round": self.current_sh_round,
            "num_configs_for_current_s": self.num_configs_for_current_s,
            "evaluated_configs_in_round": self.evaluated_configs_in_round,
            "total_iterations_done": self.total_iterations_done,
            "max_total_iterations": self.max_total_iterations,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.max_resource = state["max_resource"]
        self.eta = state["eta"]
        self.min_resource_per_config = state["min_resource_per_config"]
        self.s_max = state["s_max"]
        self.B = state["B"]
        self.param_space = state["param_space"]
        self.best_config = state["best_config"]
        self.best_score = state["best_score"]
        self.current_s = state["current_s"]
        self.configs_for_sh = state["configs_for_sh"]
        self.configs_to_evaluate = state["configs_to_evaluate"]
        self.resource_allocations = state["resource_allocations"]
        self.current_sh_round = state["current_sh_round"]
        self.num_configs_for_current_s = state["num_configs_for_current_s"]
        self.evaluated_configs_in_round = state["evaluated_configs_in_round"]
        self.total_iterations_done = state["total_iterations_done"]
        self.max_total_iterations = state["max_total_iterations"]

    def get_best_params(self) -> Dict[str, Any]:
        if self.best_config is None and self.param_space:
            return self._get_random_config()  # Fallback if no evaluations yet
        return self.best_config

    def get_best_score(self) -> float:
        return self.best_score

    def reset(self):
        # param_space is assumed to be set again via initialize
        # self.param_space = None
        self.best_config = None
        self.best_score = float("-inf")
        self.current_s = self.s_max
        self.configs_for_sh = []
        self.configs_to_evaluate = []
        self.resource_allocations = []
        self.current_sh_round = 0
        self.num_configs_for_current_s = 0
        self.evaluated_configs_in_round = []
        self.total_iterations_done = 0
        # Call initialize() externally after reset to set up the first bracket

    def is_finished(self) -> bool:
        # Hyperband naturally finishes when current_s < 0 (all brackets done)
        # Or, we can add a max total iterations if desired.
        # A simple check is if current_s < 0 and no more configs are queued.
        if self.current_s < 0 and not self.configs_to_evaluate:
            return True
        if self.total_iterations_done >= self.max_total_iterations:
            # print(f"Hyperband finished due to max_total_iterations: {self.total_iterations_done}")
            return True
        return False
