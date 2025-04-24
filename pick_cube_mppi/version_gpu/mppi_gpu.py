"""
GPU‐parallel MPPI for PickCube-v1
"""

import sapien.physx as physx
try:
    physx.enable_gpu()
except RuntimeError:
    pass
physx.enable_gpu = lambda: None

import torch
import numpy as np
import gymnasium as gym
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
        device: str = "cuda",
    ):
        # store environment and algorithm hyper‑parameters
        self.env = env
        self.H = horizon
        self.N = num_samples
        self.lam = lambda_
        self.sigma = noise_sigma
        self.device = device

        # action dimensions and bounds
        self.act_dim = env.action_space.shape[0]
        self.low  = torch.tensor(env.action_space.low,  device=device)
        self.high = torch.tensor(env.action_space.high, device=device)

        # # nominal control sequence (H × act_dim)
        self.U = torch.zeros((self.H, self.act_dim), device=device)

        # parallel rollout env：num_envs = N
        self.rollout_env = gym.make(
            env.spec.id,
            num_envs=self.N,
            robot_uids="panda",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            render_mode=None,
        )
        self.rollout_env.reset(seed=0)

        # 上一次的 costs，用于监控
        self.last_costs = None

    def update_control(self) -> np.ndarray:
        """
        1) Sample N candidate sequences
        2) Roll out each
        3) Compute soft‑min weights
        4) Reconstruct nominal sequence and return its first action.
        """
        # 1) 备份 & 同步主环境状态
        x0 = self.env.unwrapped.get_state()
        for i in range(self.N):
            self.rollout_env.unwrapped.set_state(x0, env_idx=i)

        # 2) 在 GPU 上采样噪声并生成 N 条候选序列
        noise = torch.randn((self.N, self.H, self.act_dim), device=self.device) * self.sigma
        candidates = (self.U.unsqueeze(0) + noise).clamp(self.low, self.high)

        # 3) 并行 rollout：累积 dense‐reward 作为 cost
        # total_cost = torch.zeros(self.N, device=self.device)
        total_cost = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        obs_batch, _ = self.rollout_env.reset(seed=None)

        for t in range(self.H):
            actions_t = candidates[:, t, :]
            obs_batch, _, _, _, info = self.rollout_env.step(actions_t)

            # 将 obs_batch（含两层 dict）展平
            flat_obs: dict[str, torch.Tensor] = {}
            for top_k, top_v in obs_batch.items():
                if isinstance(top_v, dict):
                    for sub_k, sub_v in top_v.items():
                        flat_obs[sub_k] = torch.as_tensor(
                            sub_v, dtype=torch.float32, device=self.device
                        )
                else:
                    flat_obs[top_k] = torch.as_tensor(
                        top_v, dtype=torch.float32, device=self.device
                    )

            # 调用 dense reward
            dense_r = self.rollout_env.compute_dense_reward(
                flat_obs,                # 观测扁平 dict
                actions_t,               # 当前动作 batch (N, act_dim)
                info                     # info 列表
            )
            total_cost += -dense_r     # MPPI cost = –reward

        # 4) 计算软最小权重
        costs = total_cost
        beta = costs.min()
        w = torch.exp(-(costs - beta) / self.lam)
        w = w / (w.sum() + 1e-12)
        self.last_costs = costs.detach().cpu().numpy()

        # 5) 更新名义序列 U
        self.U = torch.einsum("n,n h d->h d", w, candidates).clamp(self.low, self.high)

        # 6) 返回首步动作
        action = self.U[0].detach().cpu().numpy().astype(np.float32)
        return action
