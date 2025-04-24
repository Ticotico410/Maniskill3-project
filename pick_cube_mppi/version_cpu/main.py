import matplotlib.pyplot as plt
import gymnasium as gym

from src.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mppi_cpu import MPPIController


# Visualize the simulation：num_envs=1
env = gym.make(
    "PickCube-v1",
    num_envs=1,
    robot_uids="panda",
    obs_mode="state_dict",
    control_mode="pd_joint_delta_pos",
    render_mode="human",
)
obs, _ = env.reset(seed=0)

# Load and print goal threshold
goal_thresh = PICK_CUBE_CONFIGS["panda"]["goal_thresh"]
print(f"[Init] goal_thresh = {goal_thresh} m\n")

# Initialize MPPI controller
mppi = MPPIController(
    env=env,
    horizon=15,
    num_samples=5,
    lambda_=0.25,
    noise_sigma=0.30,
)
cost_history = []

# Main control loop
step = 0
MAX_STEPS = 50
done = False
while not done and step < MAX_STEPS:
    step += 1
    print(f"\n===== Step {step} =====")

    # MPPI motion plan
    action = mppi.update_control()

    # Execute action
    obs, reward, terminated, truncated, info = env.step(action)
    term_flag = bool(terminated[0].item())
    success_flag = bool(env.unwrapped.evaluate()["success"][0].item())
    print(f"[Env] reward={float(reward):.4f}, terminated={term_flag}, success={success_flag}\n")

    # Record and print MPPI min cost
    min_cost = float(mppi.last_costs.min())
    cost_history.append(min_cost)
    print(f"[Cost] min cost = {min_cost:.4f}")

    # Prepare for next iteration
    done = term_flag or success_flag
    env.render()

    # Rollout control sequence
    mppi.u[:-1] = mppi.u[1:]
    mppi.u[-1] = 0.0

env.close()

# Plot cost curve
plt.figure()
plt.plot(cost_history, marker='o')
plt.title("MPPI Minimum Cost per Iteration (CPU)")
plt.xlabel("Iteration")
plt.ylabel("Min Cost")
plt.grid(True)
plt.show()
