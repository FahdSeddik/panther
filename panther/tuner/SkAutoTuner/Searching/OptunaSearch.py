"""
Optuna-backed search algorithm for industry-standard HPO.

This integrates the Optuna framework with the SKAutoTuner's SearchAlgorithm interface,
providing state-of-the-art samplers (TPE, CMA-ES, etc.) and optional pruning support.
"""

import pickle
from typing import Any, Dict, List, Optional, Union

import optuna
from optuna.samplers import BaseSampler, TPESampler

from ..Configs.ParamSpec import Categorical, Float, Int, ParamSpec
from .SearchAlgorithm import SearchAlgorithm


class OptunaSearch(SearchAlgorithm):
    """
    Optuna-backed search algorithm implementing the SearchAlgorithm interface.

    This provides access to Optuna's powerful samplers (TPE by default) while
    maintaining compatibility with the existing SKAutoTuner workflow.

    Args:
        n_trials: Maximum number of trials to run (default: 100)
        sampler: Optuna sampler to use (default: TPESampler)
        direction: Optimization direction, "maximize" or "minimize" (default: "maximize")
        study_name: Optional name for the study (useful for persistence)
        storage: Optional Optuna storage URL (e.g., "sqlite:///study.db")
        load_if_exists: Whether to load existing study if storage is provided
        seed: Random seed for reproducibility

    Example:
        >>> search = OptunaSearch(n_trials=50, seed=42)
        >>> search.initialize({"num_terms": [1, 2, 3], "low_rank": Int(8, 128, step=8)})
        >>> while not search.is_finished():
        ...     params = search.get_next_params()
        ...     if params is None:
        ...         break
        ...     score = evaluate(params)
        ...     search.update(params, score)
    """

    def __init__(
        self,
        n_trials: int = 100,
        sampler: Optional[BaseSampler] = None,
        direction: str = "maximize",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False,
        seed: Optional[int] = None,
    ):
        self.n_trials = n_trials
        self.direction = direction
        self.study_name = study_name or "skautotuner_study"
        self.storage = storage
        self.load_if_exists = load_if_exists
        self.seed = seed

        # Create sampler with seed if provided
        self.sampler: BaseSampler
        if sampler is None:
            self.sampler = TPESampler(seed=seed)
        else:
            self.sampler = sampler

        # State
        self._study: Optional[optuna.Study] = None
        self._param_space: Dict[str, ParamSpec] = {}
        self._pending_trial: Optional[optuna.trial.Trial] = None
        self._trial_count: int = 0
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None

    def initialize(self, param_space: Dict[str, Union[List, ParamSpec]]):
        """
        Initialize the search algorithm with the parameter space.

        Args:
            param_space: Dictionary of parameter names to their specifications.
                         Supports both legacy list format and ParamSpec types.
        """
        # Normalize param space: convert lists to Categorical
        self._param_space = {}
        for name, spec in param_space.items():
            if isinstance(spec, list):
                self._param_space[name] = Categorical(spec)
            elif isinstance(spec, (Categorical, Int, Float)):
                self._param_space[name] = spec
            else:
                raise TypeError(
                    f"Parameter '{name}' has unsupported type {type(spec).__name__}. "
                    f"Expected list, Categorical, Int, or Float."
                )

        # Create or load study
        self._study = optuna.create_study(
            study_name=self.study_name,
            sampler=self.sampler,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
        )

        # Reset counters
        self._trial_count = 0
        self._pending_trial = None
        self._best_params = None
        self._best_score = None

    def _suggest_param(
        self, trial: optuna.trial.Trial, name: str, spec: ParamSpec
    ) -> Any:
        """Suggest a parameter value using Optuna's trial API."""
        if isinstance(spec, Categorical):
            return trial.suggest_categorical(name, spec.choices)
        elif isinstance(spec, Int):
            if spec.step == 1:
                return trial.suggest_int(name, spec.low, spec.high, log=spec.log)
            else:
                return trial.suggest_int(
                    name, spec.low, spec.high, step=spec.step, log=spec.log
                )
        elif isinstance(spec, Float):
            if spec.step is not None:
                return trial.suggest_float(
                    name, spec.low, spec.high, step=spec.step, log=spec.log
                )
            else:
                return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        else:
            raise TypeError(f"Unknown param spec type: {type(spec)}")

    def get_next_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the next set of parameters to try.

        Returns:
            Dictionary of parameter names and values to try, or None if finished.
        """
        if self._study is None:
            raise RuntimeError("OptunaSearch not initialized. Call initialize() first.")

        if self.is_finished():
            return None

        # Create a new trial using ask()
        self._pending_trial = self._study.ask()
        self._trial_count += 1

        # Sample parameters
        params = {}
        for name, spec in self._param_space.items():
            params[name] = self._suggest_param(self._pending_trial, name, spec)

        return params

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.

        Args:
            params: Dictionary of parameter names and values that were tried
            score: The evaluation score for the parameters
        """
        if self._study is None:
            raise RuntimeError("OptunaSearch not initialized. Call initialize() first.")

        if self._pending_trial is None:
            raise RuntimeError(
                "No pending trial to update. Call get_next_params() first."
            )

        # Complete the trial using tell()
        self._study.tell(self._pending_trial, score)
        self._pending_trial = None

        # Update best tracking
        if self._best_score is None or self._is_better(score, self._best_score):
            self._best_score = score
            self._best_params = params.copy()

    def _is_better(self, new_score: float, old_score: float) -> bool:
        """Check if new score is better than old score based on direction."""
        if self.direction == "maximize":
            return new_score > old_score
        else:
            return new_score < old_score

    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.

        Note: If using storage (SQLite, etc.), the study is automatically persisted.
        This method saves additional state for full restoration.
        """
        state = {
            "n_trials": self.n_trials,
            "direction": self.direction,
            "study_name": self.study_name,
            "storage": self.storage,
            "seed": self.seed,
            "param_space": self._param_space,
            "trial_count": self._trial_count,
            "best_params": self._best_params,
            "best_score": self._best_score,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.

        Note: If using storage, the study will be loaded from storage.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.n_trials = state["n_trials"]
        self.direction = state["direction"]
        self.study_name = state["study_name"]
        self.storage = state["storage"]
        self.seed = state["seed"]
        self._param_space = state["param_space"]
        self._trial_count = state["trial_count"]
        self._best_params = state["best_params"]
        self._best_score = state["best_score"]

        # Reload or create study
        if self.storage:
            self._study = optuna.load_study(
                study_name=self.study_name, storage=self.storage
            )
        else:
            # Without storage, study trials are lost; reinitialize
            self._study = optuna.create_study(
                study_name=self.study_name,
                sampler=TPESampler(seed=self.seed),
                direction=self.direction,
            )

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the best set of parameters found so far.

        Returns:
            Dictionary of the best parameter names and values, or None if no params yet.
        """
        if self._study is not None and len(self._study.trials) > 0:
            try:
                return self._study.best_params
            except ValueError:
                # No completed trials yet
                return self._best_params
        return self._best_params

    def get_best_score(self) -> Optional[float]:
        """
        Get the best score achieved so far.

        Returns:
            The best score, or None if no score yet.
        """
        if self._study is not None and len(self._study.trials) > 0:
            try:
                return self._study.best_value
            except ValueError:
                # No completed trials yet
                return self._best_score
        return self._best_score

    def reset(self):
        """
        Reset the search algorithm to its initial state.
        """
        if self._param_space:
            # Reinitialize with same param space
            param_space = self._param_space.copy()
            self.initialize(param_space)
        else:
            self._study = None
            self._pending_trial = None
            self._trial_count = 0
            self._best_params = None
            self._best_score = None

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search.

        Returns:
            True if the budget (n_trials) is exhausted, False otherwise.
        """
        return self._trial_count >= self.n_trials

    # --- Additional Optuna-specific methods ---

    @property
    def study(self) -> Optional[optuna.Study]:
        """Access the underlying Optuna study for advanced operations."""
        return self._study

    def get_trials_dataframe(self):
        """
        Get a pandas DataFrame of all trials.

        Useful for analysis and visualization.
        """
        if self._study is None:
            return None
        return self._study.trials_dataframe()

    def set_user_attr(self, key: str, value: Any):
        """Set a user attribute on the current pending trial."""
        if self._pending_trial is not None:
            self._pending_trial.set_user_attr(key, value)

    def report_intermediate(self, value: float, step: int):
        """
        Report an intermediate objective value for pruning.

        Args:
            value: Intermediate objective value
            step: Step of the trial (e.g., epoch number)
        """
        if self._pending_trial is not None:
            self._pending_trial.report(value, step)

    def should_prune(self) -> bool:
        """
        Check if the current trial should be pruned.

        Returns:
            True if the trial should be pruned, False otherwise.
        """
        if self._pending_trial is not None:
            return self._pending_trial.should_prune()
        return False
