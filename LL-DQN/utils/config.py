import torch

# Device configuration
DEVICE = torch.device("cpu")

# Environment
ENV_NAME = "LunarLander-v3"
SEED = 0

# Replay Buffer Parameters
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64

# Discounting and Update Parameters
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

# Optimizer / Training
LR = 5e-4