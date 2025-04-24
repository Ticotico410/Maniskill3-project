"""
mppi_maniskill.py —— Official MPPI × ManiSkill PickCube-v1 Integration
-------------------------------------------------------
* Each trajectory is rolled out in parallel environments (num_envs = K)
* running_cost = -dense_reward + regularization terms
* Noise covariance separates joint std and gripper std
"""

import numpy as np
import gymnasium as gym
import mani_skill.envs
import sapien.physx as physx
import torch
import matplotlib.pyplot as plt

from mppi import MPPIPlanner, MPPIConfig


# --------------------------------------------------
# Enable GPU PhysX once
# --------------------------------------------------
try:
    physx.enable_gpu()
except RuntimeError:
    pass
physx.enable_gpu = lambda: None  # Prevent repeated calls


# --------------------------------------------------
# MPPI wrapper for PickCube-v1 task
# --------------------------------------------------
class PickCubeMPPI:
    def __init__(
        self,
        horizon: int = 30,
        num_samples: int = 128,
        joint_std: float = 0.25,
        grip_std: float = 0.05,
        lambda_: float = 1.0,
        device: str = "cuda",
    ):
        # Set compute device and planning parameters
        self.device = torch.device(device)
        self.K = num_samples
        ACTION_DIM = 8
        self.nx = 8  # state dimension: 7 joints + 1 gripper position

        # ---------- Parallel simulation environment setup ----------
        self.roll_env = gym.make(
            "PickCube-v1",
            num_envs=self.K,
            robot_uids="panda",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            render_mode=None,
        )
        self._last_obs, _ = self.roll_env.reset(seed=0)

        # ---------- Noise covariance configuration ----------
        noise_diag = [joint_std ** 2] * 7 + [grip_std ** 2]
        noise_sigma = np.diag(noise_diag).tolist()

        # ---------- MPPI configuration ----------
        cfg = MPPIConfig(
            num_samples=num_samples,
            horizon=horizon,
            noise_sigma=noise_sigma,
            noise_mu=[0.0] * ACTION_DIM,
            u_min=[-1.0] * ACTION_DIM,
            u_max=[1.0] * ACTION_DIM,
            lambda_=lambda_,
            device=str(self.device),
            mppi_mode="halton-spline",
            sampling_method="random",
        )

        # Placeholder for last action batch
        self._last_action_batch = torch.zeros(self.K, ACTION_DIM, device=self.device)

        # ---------- Instantiate MPPI planner ----------
        self.planner = MPPIPlanner(
            cfg,
            nx=self.nx,
            dynamics=self._dynamics,
            running_cost=self._running_cost,
        )

        # Current state buffer for planner (qpos for 7 joints + gripper)
        self.current_qpos = torch.zeros((self.K, self.nx), device=self.device)

    # --------------------------------------------------
    # Dynamics callback: perform parallel step and update state
    # --------------------------------------------------
    def _dynamics(self, state, u, t=None):
        # Convert action batch to numpy for Gym
        act_np = u.cpu().numpy()
        self._last_action_batch = u
        self._last_obs, _, _, _, self._last_info = self.roll_env.step(act_np)

        # Extract qpos (first 8 dims) as next state
        qpos = torch.as_tensor(
            self._last_obs["agent"]["qpos"][:, :8],
            dtype=torch.float32,
            device=self.device,
        )
        self.current_qpos = qpos
        return qpos, u

    # --------------------------------------------------
    # Running cost: negative dense reward plus regularization
    # --------------------------------------------------
    def _running_cost(self, _state):
        # Compute dense reward from environment
        dense_r = self.roll_env.compute_dense_reward(
            self._last_obs, self._last_action_batch, self._last_info
        ).to(self.device)

        # Additional regularization: joint velocity and control effort
        qvel = torch.as_tensor(
            self._last_obs["agent"]["qvel"][:, :8], dtype=torch.float32, device=self.device
        )
        reg_vel = 0.01 * torch.norm(qvel, dim=1)
        reg_ctrl = 0.001 * torch.sum(self._last_action_batch ** 2, dim=1)

        # Total cost = -reward + velocity + effort
        cost = -dense_r + reg_vel + reg_ctrl
        return cost  # shape (K,)

    # --------------------------------------------------
    # Public interface: get next action and shift nominal plan
    # --------------------------------------------------
    @torch.no_grad()
    def command(self) -> torch.Tensor:
        """Return the next action for the current state."""
        return self.planner.command(self.current_qpos)

    def shift_nominal(self):
        """Shift the nominal control sequence after execution."""
        self.planner.U = torch.roll(self.planner.U, -1, dims=0)
        self.planner.U[-1].zero_()


# ------------------------------------------------------
# Minimal demo execution
# ------------------------------------------------------
if __name__ == "__main__":
    vis_env = gym.make(
        "PickCube-v1",
        num_envs=1,
        robot_uids="panda",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )
    vis_env.reset(seed=0)

    mppi = PickCubeMPPI(
        horizon=10,
        num_samples=10,
        joint_std=0.25,
        grip_std=0.05,
        lambda_=0.8,
        device="cuda",
    )

    costs = []
    for step in range(150):
        print(f"=== step {step} ===")
        action = mppi.command().cpu().numpy()
        _, _, term, trunc, _ = vis_env.step(action)
        vis_env.render()

        # Record minimum cost per iteration
        loss = float(mppi.planner.cost_total.min().cpu())
        print(f"[Loss] min cost = {loss:.4f}")
        costs.append(loss)

        if term.any() or trunc.any():
            print("Task completed.")
            break
        mppi.shift_nominal()

    vis_env.close()
    plt.plot(costs)
    plt.xlabel("MPPI iteration")
    plt.ylabel("Min cost")
    plt.title("PickCube-v1 MPPI Convergence")
    plt.show()
