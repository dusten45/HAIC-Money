"""DrQ-v2 extensions kept separate from the native trainer."""

from .residual_options import (
    FeatureOptionReplay,
    OPTION_COUNT,
    OptionTransition,
    OptionTransitionAccumulator,
    ResidualOption,
    ResidualOptionQ,
    ResidualOptionState,
    apply_native_residual,
    double_q_targets,
    polyak_update_option_target,
    select_greedy_option,
    train_option_q_step,
)

__all__ = [
    "FeatureOptionReplay",
    "OPTION_COUNT",
    "OptionTransition",
    "OptionTransitionAccumulator",
    "ResidualOption",
    "ResidualOptionQ",
    "ResidualOptionState",
    "apply_native_residual",
    "double_q_targets",
    "polyak_update_option_target",
    "select_greedy_option",
    "train_option_q_step",
]
