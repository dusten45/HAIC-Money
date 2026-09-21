"""Package-time inference selection; rewritten only inside a submission ZIP."""

PLANNER_ENABLED = True
PLANNER_SETTINGS = {
    "horizon": 4,
    "population": 16,
    "iterations": 2,
    "candidate_batch_size": 8,
    "uncertainty_cost": 1.0,
}
STRICT_CHECKPOINT_LOADING = False
CONTROLLER_MODE = "auto"
