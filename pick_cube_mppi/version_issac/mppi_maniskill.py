import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenObservationWrapper
import sapien.physx as physx
import torch
from typing import Any, Dict
from mppi import MPPIPlanner, MPPIConfig


# Enable GPU PhysX once
try:
    physx.enable_gpu()
except RuntimeError:
    pass
physx.enable_gpu = lambda: None  # Prevent subsequent calls


def flatten_obs(obs: Dict[str, Any], device):
    """
    Flatten a nested observation dictionary and convert values to torch.Tensor.

    Args:
        obs (Dict[str, Any]): Nested observation dictionary.
        device: Torch device to place tensors.

    Returns:
        Dict[str, torch.Tensor]: Flattened observation tensors.
    """
    # TODO: This function may change FlattenObservationWrapper, could try
    flat: Dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[subkey] = torch.as_tensor(subvalue, dtype=torch.float32, device=device)
        else:
            flat[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return flat


class PickCubeMPPI:
    """
    Wrapper that combines MPPIPlanner with a parallel ManiSkill PickCube-v1 environment.
    """
    # TODO: Tune the params
    def __init__(
        self,
        horizon: int = 25,
        num_samples: int = 64,
        noise_joint: float = 0.4,
        noise_grip: float = 0.15,
        lambda_: float = 0.3,
        device: str = "cuda",
    ):
        # Set device and number of samples
        self.device = torch.device(device)
        self.K = num_samples
        ACTION_DIM = 8

        # ---------- Create parallel simulation environment ----------
        self.roll_env = gym.make(
            "PickCube-v1",
            num_envs=self.K,
            robot_uids="panda",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            render_mode=None,
        )
        self.roll_env.reset(seed=0)

        # ---------- Build diagonal covariance matrix for action noise ----------
        noise_diag = [noise_joint**2]*7 + [noise_grip**2]
        noise_mat = np.diag(noise_diag).tolist()

        # ---------- Configuration of MPPI ----------
        cfg = MPPIConfig(
            num_samples=num_samples,
            horizon=horizon,
            noise_sigma=noise_mat,        # Covariance matrix (8×8)
            noise_mu=[0.0]*ACTION_DIM,    # Mean vector
            u_min=[-1.0]*ACTION_DIM,      # Lower action bounds
            u_max=[ 1.0]*ACTION_DIM,      # Upper action bounds
            lambda_=lambda_,              # Inverse temperature
            device=str(self.device),      # Compute device
            mppi_mode ="simple",          # MPPI mode
            sampling_method="random",     # Sampling mode
        )

        # State dimension placeholder
        nx = 1
        self._last_action_batch = torch.zeros(self.K, ACTION_DIM, device=self.device)
        self._last_obs = None
        self._last_info = None

        # Instantiate the MPPI planner
        self.planner = MPPIPlanner(
            cfg, nx,
            dynamics=self._dynamics,
            running_cost=self._running_cost,
        )
        # Dummy state for batch planning
        self.dummy_state = torch.zeros((self.K, nx), device=self.device)

    # ---------- MPPI callback for dynamics ---------- #
    def _dynamics(self, state, u, t=None):
        """
        Parallel simulation step for MPPI.

        Args:
            state: Placeholder state tensor.
            u: Action batch (K×8).

        Returns:
            (state, u): State and action returned unchanged.
        """
        self._last_action_batch = u
        self._last_obs, _, _, _, self._last_info = self.roll_env.step(u)
        return state, u

    # ---------- MPPI callback for running cost ---------- #
    def _running_cost(self, _state):
        """
        Compute cost from negative dense reward of PickCube environment.

        Args:
            _state: Placeholder state (unused).

        Returns:
            Tensor of costs (K,).
        """
        flat = flatten_obs(self._last_obs, self.device)
        dense_r = self.roll_env.compute_dense_reward(
            flat, self._last_action_batch, self._last_info
        )  # Torch tensor of shape (K,)
        return -dense_r  # MPPI uses cost = -reward

    # ---------- Public interface ---------- #
    @torch.no_grad()
    def command(self) -> torch.Tensor:
        """
        Compute the next action using MPPI planner.

        Returns:
            Torch tensor of shape (ACTION_DIM,).
        """
        return self.planner.command(self.dummy_state)

    def shift_nominal(self):
        """
        Shift the nominal control sequence by one step after execution.
        """
        self.planner.U = torch.roll(self.planner.U, -1, dims=0)
        self.planner.U[-1].zero_()


if __name__ == "__main__":
    # Create a single environment for visualization
    vis_env = gym.make(
        "PickCube-v1",
        num_envs=1,
        robot_uids="panda",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )
    vis_env.reset(seed=0)

    # Instantiate MPPI for demo
    mppi = PickCubeMPPI(
        horizon=25,
        num_samples=64,
        noise_joint=0.5,
        noise_grip=0.2,
        lambda_=0.4,
        device="cuda"
    )

    costs = []
    for step in range(100):
        print(f"=== step {step} ===")
        act = mppi.command().cpu().numpy()
        obs, r, term, trunc, _ = vis_env.step(act)
        vis_env.render()

        # Record and print loss
        loss = float(mppi.planner.cost_total.min().cpu())
        print(f"[Loss] min cost = {loss:.4f}")
        costs.append(loss)

        if term.any() or trunc.any():
            break

        mppi.shift_nominal()

    vis_env.close()
    plt.plot(costs)
    plt.xlabel("MPPI iteration")
    plt.ylabel("Min cost")
    plt.title("PickCube-v1 Loss")
    plt.show()
