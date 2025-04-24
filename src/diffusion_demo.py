import numpy as np

# Simulation parameters
N = 5          # Number of candidate sequences
H = 5          # Horizon (steps per sequence)
d = 3          # Action dimension
sigma = 0.5    # Noise standard deviation

# 1) Initialize nominal sequence (all zeros)
u_nominal = np.zeros((H, d), dtype=np.float32)

# 2) Sample noise: shape (N, H, d)
noise = np.random.normal(0, sigma, size=(N, H, d)).astype(np.float32)

# 3) Generate candidate sequences by adding noise
candidates = u_nominal[None, :, :] + noise

# Display the results
print("Nominal sequence (u_nominal):")
print(u_nominal, "\n")

for i in range(N):
    print(f"Candidate sequence {i} (u_nominal + noise[{i}]):")
    print(candidates[i], "\n")
