# Preserve hazard memory and near clear-road turn support

Memory-corridor V1 fixes a concrete missing optimizer constraint and completes
15.20/19.72/18.78/21.14s without contacts. Arc-clear V2 fixes a different
clear-road tracking problem and completes15.44/19.08/16.52/20.92s, with one
track4 contact. These separate results do not constitute a combined result.

Embed exact frozen arc-clear V2 with its public class renamed; append only
memory-corridor V1's exact `_route` method. This preserves the independently
tested steering and remembered-constraint implementations, in one controller
with shared route and steering state. No selector or steering handoff is used.

Actual missed-circle route and clear-bend envelope regressions must both pass,
including one memory transport per action, finite invalid-input recovery and
reset. Freeze source/defaults before a fresh mandatory four-cell benchmark.
No holdout, adoption or tyre safety certificate is claimed by this experiment.
