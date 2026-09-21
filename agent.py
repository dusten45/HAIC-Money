import numpy as np
import torch
from torch import nn

from action_smoothing import (
    append_action_control_plane,
    action_control_fingerprint,
    action_smoothing_fingerprint,
    build_action_smoother,
    normalize_action_smoothing,
    normalize_action_control,
)
from action_representation import (
    action_representation_fingerprint,
    map_policy_action,
    normalize_action_representation,
)

MODEL_FILENAME = "model.pt"

# Small single-observation CNN inference is faster and more predictable without
# the default large CPU thread pool.
torch.set_num_threads(1)


class NatureFeatures(nn.Module):
    def __init__(self, input_channels=4, action_dimensions=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

    def forward(self, observation):
        return self.linear(self.cnn(observation.float()))


class Baseline1Actor(nn.Module):
    """Deterministic actor equivalent to the Baseline 1 SB3 CnnPolicy."""

    def __init__(self, input_channels=4, action_dimensions=3):
        super().__init__()
        self.input_channels = int(input_channels)
        self.features_extractor = NatureFeatures(self.input_channels)
        self.policy_net = nn.Sequential()
        self.action_dimensions = int(action_dimensions)
        self.action_net = nn.Linear(512, self.action_dimensions)

    def forward(self, observation):
        features = self.features_extractor(observation)
        return self.action_net(self.policy_net(features))

    def predict_action(self, observation):
        mean = self.forward(observation)
        return torch.stack(
            (
                mean[..., 0].clamp(-1.0, 1.0),
                mean[..., 1].clamp(0.0, 1.0),
                mean[..., 2].clamp(0.0, 1.0),
            ),
            dim=-1,
        )


class Agent:
    def __init__(self):
        """Loads only the deterministic actor needed by the submission."""
        payload = torch.load(
            MODEL_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            action_smoothing = payload.get("action_smoothing")
            smoothing_fingerprint = payload.get("action_smoothing_fingerprint")
            action_control = payload.get("action_control")
            control_fingerprint = payload.get("action_control_fingerprint")
            input_channels = payload.get("input_channels", 4)
            action_representation = payload.get("action_representation")
            representation_fingerprint = payload.get("action_representation_fingerprint")
        else:
            state_dict = payload
            action_smoothing = None
            smoothing_fingerprint = None
            action_control = None
            control_fingerprint = None
            input_channels = 4
            action_representation = None
            representation_fingerprint = None
        action_smoothing = normalize_action_smoothing(action_smoothing)
        action_control = normalize_action_control(action_control)
        action_representation = normalize_action_representation(action_representation)
        if (
            smoothing_fingerprint is not None
            and smoothing_fingerprint != action_smoothing_fingerprint(action_smoothing)
        ):
            raise ValueError("model action smoothing fingerprint does not match config")
        if (
            control_fingerprint is not None
            and control_fingerprint != action_control_fingerprint(action_control)
        ):
            raise ValueError("model action control fingerprint does not match config")
        if int(input_channels) != action_control["input_channels"]:
            raise ValueError("model input channels do not match action control config")
        if (
            representation_fingerprint is not None
            and representation_fingerprint != action_representation_fingerprint(action_representation)
        ):
            raise ValueError("model action representation fingerprint does not match config")
        action_dimensions = (
            sum(action_representation["nvec"])
            if action_representation["method"] != "continuous_box"
            else 3
        )
        self.model = Baseline1Actor(
            input_channels=int(input_channels), action_dimensions=action_dimensions
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.action_smoothing = action_smoothing
        self.action_control = action_control
        self.action_representation = action_representation
        self.smoother = build_action_smoother(self.action_smoothing)
        self.reset(None)

    def reset(self, observation):
        self.smoother.reset(initial_action=self.action_smoothing["initial_action"])

    @torch.inference_mode()
    def act(self, observation) -> np.ndarray:
        controlled = append_action_control_plane(
            observation,
            self.action_control,
            self.smoother.last_action,
        )
        state = torch.as_tensor(controlled).unsqueeze(0)
        raw_action = self.model(state).squeeze(0).numpy()
        if self.action_representation["method"] != "continuous_box":
            steering_dimensions, longitudinal_dimensions = self.action_representation["nvec"]
            raw_action = map_policy_action(
                np.asarray(
                    (
                        np.argmax(raw_action[:steering_dimensions]),
                        np.argmax(raw_action[steering_dimensions:steering_dimensions + longitudinal_dimensions]),
                    ),
                    dtype=np.int64,
                ),
                self.action_representation,
            )
        else:
            raw_action = np.asarray([
                np.clip(raw_action[0], -1.0, 1.0),
                np.clip(raw_action[1], 0.0, 1.0),
                np.clip(raw_action[2], 0.0, 1.0),
            ], dtype=np.float32)
        action = self.smoother.smooth(raw_action)
        return np.asarray(action, dtype=np.float32)
