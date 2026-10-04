"""Offline compute-only batch assembly from three immutable NumPy sources."""
import argparse
import ast
import hashlib
from pathlib import Path


PARENT_SHA = 'b70af66e0ba39ea975d53909ec6d7a19df2a73005a036465d71dd7ac8caeff8a'
MODEL_SHA = '8ae7239b81d9b41e649e2ef704c3a6bbf3a8b2a933ed499b7bf13f566cb82d53'
PLANNER_SHA = '263909e1163fdce76e6704246e7a612c7b940f40dab89a34338bd1c57bfbaeca'
MODEL_EXPORTS = ('BATCH_DT', 'BATCH_ARRAY_SHAPES', 'BATCH_SCALARS',
                 '_copy_batch', '_rotate_batch', 'predict_batch')

ADAPTER = '''class _BudgetedBatchControlPlanner(FixedBatchControlPlanner):
    """Retain the owner's original deadline through model and geometry work."""

    def __init__(self, batch_callback, geometry):
        super().__init__(batch_callback, geometry)
        owner = getattr(batch_callback, '__self__', None)
        if not isinstance(owner, _PredictiveV2Reference):
            raise ValueError('expected the owning camera agent batch callback')
        # Only this agent's explicit public method reads its private deadline.
        # Constructor checks must not raise a timeout outside inherited act's
        # planner error handling; the first model/geometry call checks it.
        self._check_budget = owner.check_planning_budget

    def _geometry(self, positions, angles, velocities):
        self._check_budget()
        result = super()._geometry(positions, angles, velocities)
        self._check_budget()
        return result

    def plan(self, supplied_states, previous_action):
        result = super().plan(supplied_states, previous_action)
        if result.get('feasible'):
            try:
                # Aggregation also consumes the original per-action budget.
                self._check_budget()
            except _PlanningBudgetExceeded as error:
                return {'feasible': False, 'action': None, 'reason': 'planning_error',
                        'error': type(error).__name__+': '+str(error)[:160],
                        'model_steps': self.model_steps, 'batch_calls': self.batch_calls,
                        'batched_model_rows': self.batched_model_rows}
        return result


class Agent(_PredictiveV2Reference):
    """Same camera policy and action history with batched model computation."""

    def check_planning_budget(self):
        if time.perf_counter() >= self._predictive_deadline:
            raise _PlanningBudgetExceeded('camera planning time budget reached')

    def _predictive_step(self, state, commands):
        self.check_planning_budget()
        # Count actual raw row predictions, not batch callback invocations.
        # Planner model_steps separately preserves scalar logical telemetry.
        self.predictive_model_calls += len(commands)
        result = predict_batch(state, commands)
        self.check_planning_budget()
        return result


# Inherited V2 act looks up this global when creating its planning object.
# Scalar predict_step remains unchanged for CameraObserver.advance.
FixedControlPlanner = _BudgetedBatchControlPlanner
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_nodes(text, names):
    tree = ast.parse(text)
    exports = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            assert name not in exports
            exports[name] = ast.get_source_segment(text, node)
    assert set(exports) == set(names)
    return '\n\n'.join(exports[name] for name in names)+'\n'


def build_text(root):
    base = Path(root)/'agents/apex_2026'
    research = base/'research/speed_20261005'
    parent = base/'fast_predictive_v2_agent.py'
    model = research/'predictive_batch_model.py'
    planner = research/'predictive_batch_control.py'
    assert digest(parent) == PARENT_SHA
    assert digest(model) == MODEL_SHA
    assert digest(planner) == PLANNER_SHA
    prefix = parent.read_text()
    assert prefix.endswith('\n')
    result = prefix+'\n\n_PredictiveV2Reference = Agent\n_ScalarPlannerReference = FixedControlPlanner\n\n'
    result += export_nodes(model.read_text(), MODEL_EXPORTS)+'\n'
    result += export_nodes(planner.read_text(), ('FixedBatchControlPlanner',))+'\n\n'+ADAPTER
    ast.parse(result)
    assert result.startswith(prefix)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    text = build_text(args.root)
    assert not args.output.exists(), 'use a new immutable output path'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
