import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from dreamer_v3 import DreamerV3Agent, DreamerV3Config, symexp, symlog


def diagnose_world_model(checkpoint_path: Path, output_path: Path | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint["config"]
    cfg = DreamerV3Config(**cfg_dict)
    cfg.device = device
    agent = DreamerV3Agent(cfg)
    agent.load_checkpoint(checkpoint_path)

    agent.encoder.eval()
    agent.rssm.eval()
    agent.decoder.eval()
    agent.reward_head.eval()
    agent.continue_head.eval()
    agent.actor.eval()
    agent.critic.eval()

    if agent.replay.size < agent.config.seq_len + 1:
        raise ValueError(f"replay has only {agent.replay.size} steps, need at least {agent.config.seq_len + 1}")

    # Sample batch of sequences
    batch = agent.replay.sample_sequence(batch_size=16, seq_len=agent.config.seq_len)
    obs = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    is_first = batch["is_first"].to(device)
    is_terminal = batch["is_terminal"].to(device)

    B, T = obs.shape[:2]

    with torch.no_grad():
        # 1. Observation encoding
        embeds = agent.encoder(obs.view(B * T, 4, 84, 84)).view(B, T, -1)

        # 2. RSSM Sequence Rollout
        h = torch.zeros(B, agent.rssm.hidden_dim, device=device)
        z = torch.zeros(B, agent.rssm.stoch_dim, device=device)
        prev_a = torch.zeros(B, 3, device=device)

        h_seq, z_seq = [], []
        pr_logits_seq, po_logits_seq = [], []

        for t in range(T):
            first_mask = is_first[:, t].unsqueeze(-1).float()
            h = h * (1.0 - first_mask)
            z = z * (1.0 - first_mask)
            prev_a = prev_a * (1.0 - first_mask)

            prior_h, prior_z, pr_logits, pr_probs = agent.rssm.step_prior(h, z, prev_a)
            post_h, post_z, po_logits, po_probs = agent.rssm.step_post(h, z, prev_a, embeds[:, t])

            h, z = post_h, post_z
            prev_a = actions[:, t]

            h_seq.append(h)
            z_seq.append(z)
            pr_logits_seq.append(pr_logits)
            po_logits_seq.append(po_logits)

        h_stack = torch.stack(h_seq, dim=1)
        z_stack = torch.stack(z_seq, dim=1)
        states = torch.cat([h_stack, z_stack], dim=-1)

        # 3. Observation reconstruction quality
        rec_obs = agent.decoder(states.view(B * T, -1)).view(B, T, 4, 84, 84)
        obs_mse = F.mse_loss(rec_obs, obs).item()

        # 4. Reward prediction quality
        pred_reward_sym = agent.reward_head(states).squeeze(-1)
        pred_reward = symexp(pred_reward_sym)
        target_reward_sym = symlog(rewards)
        reward_sym_mse = F.mse_loss(pred_reward_sym, target_reward_sym).item()
        reward_mae = F.l1_loss(pred_reward, rewards).item()

        # Correlation between predicted and true rewards
        pred_flat = pred_reward.view(-1).cpu().numpy()
        true_flat = rewards.view(-1).cpu().numpy()
        if np.std(pred_flat) > 1e-6 and np.std(true_flat) > 1e-6:
            reward_corr = float(np.corrcoef(pred_flat, true_flat)[0, 1])
        else:
            reward_corr = 0.0

        # 5. Continuation / Terminal prediction
        pred_cont_logits = agent.continue_head(states).squeeze(-1)
        pred_cont_prob = torch.sigmoid(pred_cont_logits)
        cont_targets = (1.0 - is_terminal.float())
        cont_bce = F.binary_cross_entropy_with_logits(pred_cont_logits, cont_targets).item()
        cont_acc = ((pred_cont_prob >= 0.5) == (cont_targets >= 0.5)).float().mean().item()

        # 6. KL divergence between prior and posterior
        pr_logits_stack = torch.stack(pr_logits_seq, dim=1)
        po_logits_stack = torch.stack(po_logits_seq, dim=1)
        _, pr_probs = agent.rssm.get_dist(pr_logits_stack)
        _, po_probs = agent.rssm.get_dist(po_logits_stack)
        eps = 1e-7
        po_p = po_probs.clamp(min=eps)
        pr_p = pr_probs.clamp(min=eps)
        kl = (po_p * (torch.log(po_p) - torch.log(pr_p))).sum(dim=-1).sum(dim=-1)
        mean_kl = kl.mean().item()

        # 7. Latent imagination stability
        flat_states = states.view(B * T, -1)
        sample_size = min(B * T, 64)
        perm = torch.randperm(B * T, device=device)[:sample_size]
        init_s = flat_states[perm]
        im_h, im_z = torch.split(init_s, [agent.rssm.hidden_dim, agent.rssm.stoch_dim], dim=-1)

        H = agent.config.imagination_horizon
        im_states = [torch.cat([im_h, im_z], dim=-1)]
        im_rewards = []
        im_values = []

        for step in range(H):
            curr_s = torch.cat([im_h, im_z], dim=-1)
            act, _ = agent.actor(curr_s, deterministic=True)
            im_h, im_z, _, _ = agent.rssm.step_prior(im_h, im_z, act)
            next_s = torch.cat([im_h, im_z], dim=-1)

            r = symexp(agent.reward_head(next_s).squeeze(-1))
            v = symexp(agent.critic(next_s).squeeze(-1))

            im_states.append(next_s)
            im_rewards.append(r)
            im_values.append(v)

        im_states = torch.stack(im_states)  # (H+1, S, state_dim)
        im_rewards = torch.stack(im_rewards)  # (H, S)
        im_values = torch.stack(im_values)  # (H, S)

        latent_finite = bool(torch.isfinite(im_states).all().item())
        values_finite = bool(torch.isfinite(im_values).all().item())
        rewards_finite = bool(torch.isfinite(im_rewards).all().item())

        report = {
            "checkpoint": str(checkpoint_path),
            "environment_steps": agent.environment_steps,
            "gradient_steps": agent.gradient_steps,
            "world_model": {
                "obs_reconstruction_mse": float(obs_mse),
                "reward_symlog_mse": float(reward_sym_mse),
                "reward_mae": float(reward_mae),
                "reward_correlation": float(reward_corr),
                "continue_bce": float(cont_bce),
                "continue_accuracy": float(cont_acc),
                "mean_kl_divergence": float(mean_kl),
            },
            "latent_imagination": {
                "horizon": H,
                "latent_states_finite": latent_finite,
                "values_finite": values_finite,
                "rewards_finite": rewards_finite,
                "imagined_reward_mean": float(im_rewards.mean().item()),
                "imagined_value_mean": float(im_values.mean().item()),
                "imagined_value_std": float(im_values.std().item()),
            },
            "diagnostics_passed": (
                obs_mse < 0.2
                and reward_sym_mse < 2.0
                and cont_acc > 0.8
                and math.isfinite(mean_kl)
                and latent_finite
                and values_finite
            ),
        }

    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose learned DreamerV3 World Model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--output", type=Path, help="Path to output JSON")
    args = parser.parse_args()

    result = diagnose_world_model(args.checkpoint, args.output)
    print(json.dumps(result, indent=2))
