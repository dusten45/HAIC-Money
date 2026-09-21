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
DRQ_ACTOR_FORMAT = "haic-drq-v2-actor-v1"

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


class DrQFeatures(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Flatten(), nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim), nn.Tanh(),
        )

    def forward(self, observation):
        return self.linear(self.convolution(observation.float() - 0.5))


class DrQActor(nn.Module):
    """Inference-only architecture matching the native DrQ actor state keys."""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.encoder = DrQFeatures(feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, 3)

    def forward(self, observation):
        return torch.tanh(self.policy(self.trunk(self.encoder(observation))))


class Agent:
    def __init__(self, model_path=None):
        """Loads only the deterministic actor needed by the submission."""
        payload = torch.load(
            MODEL_FILENAME if model_path is None else model_path,
            map_location="cpu",
            weights_only=True,
        )
        self.format = payload.get("format") if isinstance(payload, dict) else None
        if self.format == DRQ_ACTOR_FORMAT:
            config = payload.get("config", {})
            observation_spec = payload.get("observation_spec", {})
            action_spec = payload.get("action_spec", {})
            expected_observation = {
                "shape": (4, 84, 84), "dtype": "float32", "channel_order": "CHW",
                "low": 0.0, "high": 1.0, "uint8_scale": 255,
                "control_plane_fingerprint": None,
            }
            expected_action = {
                "native_low": (-1.0, -1.0, -1.0), "native_high": (1.0, 1.0, 1.0),
                "official_low": (-1.0, 0.0, 0.0), "official_high": (1.0, 1.0, 1.0),
                "frame_skip": 4, "order": ("steer", "gas", "brake"),
                "method": "symmetric-native-to-haic-box",
            }
            for name, actual, expected in (
                ("observation", observation_spec, expected_observation),
                ("action", action_spec, expected_action),
            ):
                if not isinstance(actual, dict):
                    raise ValueError(f"invalid DrQ {name} spec")
                for key, value in expected.items():
                    recorded = actual.get(key)
                    if isinstance(value, tuple) and isinstance(recorded, (list, tuple)):
                        recorded = tuple(recorded)
                    if key not in actual or recorded != value:
                        raise ValueError(f"unsupported DrQ {name} spec: {key}")
            if (
                not isinstance(config, dict)
                or config.get("observation_shape") not in ((4, 84, 84), [4, 84, 84])
                or config.get("action_dim") != 3
                or any(type(config.get(key)) is not int or config[key] <= 0
                       for key in ("feature_dim", "hidden_dim"))
            ):
                raise ValueError("invalid DrQ actor architecture config")
            if (
                normalize_action_smoothing(payload.get("action_smoothing"))
                != normalize_action_smoothing()
                or normalize_action_control(payload.get("action_control"))
                != normalize_action_control()
                or normalize_action_representation(payload.get("action_representation"))
                != normalize_action_representation()
            ):
                raise ValueError("DrQ export must use the frozen unsmoothed action contract")
            state_dict = payload.get("state_dict")
            if not isinstance(state_dict, dict) or not state_dict or any(
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32 or not torch.isfinite(value).all()
                for value in state_dict.values()
            ):
                raise ValueError("invalid or non-finite DrQ actor state")
            for key, shape in (
                ("encoder.linear.1.weight", (config["feature_dim"], 512)),
                ("trunk.0.weight", (config["hidden_dim"], config["feature_dim"])),
                ("policy.weight", (3, config["hidden_dim"])),
            ):
                if key not in state_dict or tuple(state_dict[key].shape) != shape:
                    raise ValueError(f"DrQ actor config does not match state: {key}")
            with torch.random.fork_rng(devices=[]):
                self.model = DrQActor(config["feature_dim"], config["hidden_dim"])
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.export_metadata = {key: value for key, value in payload.items() if key != "state_dict"}
            self.reset(None)
            return
        if self.format is not None:
            raise ValueError(f"unsupported model format: {self.format}")
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
        if self.format == DRQ_ACTOR_FORMAT:
            # This actor is feed-forward; there is no state to carry across tracks.
            return
        self.smoother.reset(initial_action=self.action_smoothing["initial_action"])

    @torch.inference_mode()
    def act(self, observation) -> np.ndarray:
        if self.format == DRQ_ACTOR_FORMAT:
            observation = np.asarray(observation)
            if (
                observation.shape != (4, 84, 84) or observation.dtype != np.float32
                or not np.isfinite(observation).all()
                or np.any(observation < 0.0) or np.any(observation > 1.0)
            ):
                raise ValueError("DrQ observation must be float32 CHW (4, 84, 84) in [0, 1]")
            native = self.model(torch.as_tensor(np.ascontiguousarray(observation)).unsqueeze(0)).squeeze(0).numpy()
            action = np.clip(native, -1.0, 1.0).astype(np.float32)
            action[1:] = (action[1:] + 1.0) * 0.5
            if not np.isfinite(action).all():
                raise ValueError("DrQ actor produced a non-finite action")
            return action
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
