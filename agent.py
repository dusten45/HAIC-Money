import numpy as np
import torch
from torch import nn


MODEL_FILENAME = "model.pt"

# Small single-observation CNN inference is faster and more predictable without
# the default large CPU thread pool.
torch.set_num_threads(1)


class NatureFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
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

    def __init__(self):
        super().__init__()
        self.features_extractor = NatureFeatures()
        self.policy_net = nn.Sequential()
        self.action_net = nn.Linear(512, 3)

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
        self.model = Baseline1Actor()
        state_dict = torch.load(
            MODEL_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def reset(self, observation):
        pass

    @torch.inference_mode()
    def act(self, observation) -> np.ndarray:
        state = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
        action = self.model.predict_action(state).squeeze(0).numpy()
        return action.astype(np.float32, copy=False)
