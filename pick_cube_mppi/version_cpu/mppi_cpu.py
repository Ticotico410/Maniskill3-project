"""
CPU-based MPPI for PickCube-v1
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional
from mani_skill.envs.sapien_env import BaseEnv


class MPPIController:
    """
    MPPI controller repeatedly does `set_state → rollout → restore` on the same env.
    It uses the negative sum of dense rewards as cost (via env.compute_dense_reward).
    """

    def __init__(
            self,
            env: BaseEnv,
            horizon: int,
            num_samples: int,
            lambda_: float,
            noise_sigma: float,
            action_low: Optional[np.ndarray] = None,
            action_high: Optional[np.ndarray] = None,
    ):
        # store environment and algorithm hyper‑parameters
        self.env = env
        self.H = horizon
        self.N = num_samples
        self.lam = lambda_
        self.sigma = noise_sigma

        # action dimensions and bounds
        self.act_dim = env.action_space.shape[0]
        self.low = action_low if action_low is not None else env.action_space.low
        self.high = action_high if action_high is not None else env.action_space.high

        # nominal control sequence (H × act_dim)
        self.u = np.zeros((self.H, self.act_dim), dtype=np.float32)

        # Store costs
        self.last_costs: Optional[np.ndarray] = None

    def update_control(self) -> np.ndarray:
        """
        1) Sample N candidate sequences
        2) Roll out each
        3) Compute soft‑min weights
        4) Reconstruct nominal sequence and return its first action.
        """
        # Save current state
        cur_state = self.env.unwrapped.get_state()

        # Sample N candidate control sequences (N, H, act_dim)
        cand = self._sample_sequences()

        # Rollout and compute cost
        costs = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            costs[i] = self._sequence_cost(cur_state, cand[i])
        self.last_costs = costs  # Save cost

        # soft‑minimum: exponentiate negative costs
        shifted = costs - costs.min()
        w = np.exp(-self.lam * shifted)
        w /= w.sum() + 1e-12

        # debug prints: cost stats
        print(f"[MPPI] costs: min={costs.min():.4f}, mean={costs.mean():.4f}, max={costs.max():.4f}")
        # top-3 weights
        top3 = np.sort(w)[-3:][::-1]
        print(f"[MPPI] top-3 weights: {top3.tolist()}")

        # update nominal sequence as weighted average of candidates
        self.u = np.tensordot(w, cand, axes=(0, 0))

        # Return the first action
        action = self.u[0].astype(np.float32)
        print(f"[MPPI] selected action: {action}")

        return action

    def _sample_sequences(self) -> np.ndarray:
        """
        Add Gaussian noise to the nominal sequence and clip to action bounds.
        """
        noise = np.random.normal(0.0, self.sigma, (self.N, self.H, self.act_dim))
        seq = self.u[None, :, :] + noise
        return np.clip(seq, self.low, self.high).astype(np.float32)

    def _sequence_cost(self, init_state, action_seq) -> float:
        """
        Restore the env to init_state, roll out H steps applying action_seq,
        accumulate -dense_reward, plus a small control penalty.
        """
        # Rollout env
        env = self.env.unwrapped
        env.set_state(init_state)

        total_reward = 0.0
        obs = env.get_obs()  # initial observation
        for a in action_seq:
            obs, _, _, _, info = env.step(a)
            # ensure obs entries are torch tensors for compute_dense_reward
            obs_t = {k: torch.as_tensor(v) if not isinstance(v, dict) else v
                     for k, v in obs.items()}
            act_t = torch.as_tensor(a[None, :], dtype=torch.float32)
            r_t = env.compute_dense_reward(obs_t, act_t, info)
            total_reward += r_t.item()

        # cost = negative reward
        cost = -total_reward
        return cost
