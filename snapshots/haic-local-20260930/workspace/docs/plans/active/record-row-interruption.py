"""Coordinator record repair after the Docker parent process disappeared."""
from pathlib import Path
from haic_research.config import load_config
from haic_research.records import run_transaction
from haic_research.commands import _validated_plan, _replay, _event
from haic_research.state import WorkflowState

config=load_config(Path.cwd())
with run_transaction(config.run_root/'fast-completion-row-repair-20260929',config=config) as run:
    _validated_plan(config,run)
    history=_replay(run)
    assert history.execution_started and not history.execution_finished
    approval=next(a for a in history.approvals if a.stage=='execution')
    run.append(_event('EXECUTION_FAILED',WorkflowState.EVALUATE,approval_ref=approval.source_ref,
                     execution_status='failed',error='Docker Desktop backend terminated; execution parent disappeared at21/24 episodes. See engine_interruption.json.',
                     resource_usage={'returncode':None,'infrastructure_interrupted':True,'episode_files':21}))
print('Recorded infrastructure failure; original event history preserved.')
