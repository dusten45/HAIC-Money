"""One stateless pedal substitution before the frozen arrival-speed block.

The candidate packager inserts this call after the existing far tracker update.
No detector, projection, target, steering, or timer is replaced here.
"""

import math

import numpy as np


def apply_sprint72_relief(agent, action, near, far, impact_before):
    """Mutate only eligible pedals; the original downstream code records history."""
    info = agent.last
    speed = info['pixel_speed']
    centers = info['road_centers']
    parent = info['parent_action']
    original = action.copy()
    proposal = max(speed, 72.)
    # Reuse the acceleration layer's values and exact floating-point comparison.
    space_ok = (near is None and 42 in centers and 54 in centers
                and abs(centers[54] - 42.) < 3.
                and info['free_samples'] >= 4
                and abs(float(parent[0])) < .18
                and info['free_distance'] >= max(18., proposal * .4))
    clearing = ((impact_before > 0 or info['impact_proxy_trigger'])
                and not info['braking_proxy_veto'])
    eligible = (agent.steps > 10 and math.isfinite(speed) and speed >= 72.
                and info['target_speed'] == 60. and space_ok
                and far is None and agent.track is None and not clearing
                and agent.recovery_left == 0 and not info['geometry_repaired']
                and action[1] == action.dtype.type(0.)
                and action[2] == action.dtype.type(.15)
                and action[0] == parent[0])
    brake = float(np.clip(.02 * (speed - 72.), 0., .15)) if eligible else None
    if eligible:
        action[1], action[2] = 0., brake
    # These fields are observations only. Existing target/sprint flags keep their
    # original meanings; FarHazard and ContactContinuity write the issued brake.
    info.update(sprint72_eligible=bool(eligible),
                sprint72_applied=not np.array_equal(action, original),
                sprint72_space_ok=bool(space_ok),
                sprint72_pre_action=original.tolist(),
                sprint72_prearrival_action=action.tolist(),
                sprint72_brake_formula=brake,
                sprint72_impact_before=int(impact_before))
