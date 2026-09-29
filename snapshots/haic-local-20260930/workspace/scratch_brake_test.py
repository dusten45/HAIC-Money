import numpy as np

from training.evaluate_closed_loop import create_training_environment
from training.site_maps import SiteMapSpec

site_map = SiteMapSpec(
    map_id="official-track1-seed42",
    map_kind="official",
    track_id=1,
    seed=42,
    obstacle_mode="official",
    obstacles=(),
    max_steps=2000,
    frame_skip=4,
)
env = create_training_environment(track_id=None, seed=42, max_decisions=600, site_map=site_map)
observation, info = env.reset()

# warm up to speed by full gas for a while, then slam full brake and measure decel
speeds = []
for i in range(80):
    action = np.array([0.0, 0.12, 0.0], dtype=np.float32)
    transition = env.step_transition(action)
    speed = transition.labels.speed
    speeds.append(speed)

print("speed after warm-up gas:", speeds[-1])

brake_speeds = [speeds[-1]]
for i in range(20):
    action = np.array([0.0, 0.0, 0.28], dtype=np.float32)
    transition = env.step_transition(action)
    brake_speeds.append(transition.labels.speed)

print("speeds during full brake:", brake_speeds)
decision_seconds = 4/50.0
decels = [(brake_speeds[i+1]-brake_speeds[i])/decision_seconds for i in range(len(brake_speeds)-1)]
print("decelerations:", decels)
print("max decel magnitude:", min(decels))
