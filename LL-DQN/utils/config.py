# config.py
import torch

# Device configuration (GPU if available, else CPU)
DEVICE = torch.device("cpu")

# Environment
ENV_NAME = "LunarLander-v3"
SEED = 0

# Replay Buffer Parameters
BUFFER_SIZE = int(1e5)   # Maximum number of experiences the buffer can hold
BATCH_SIZE = 64          # Number of experiences sampled per training step

# Discounting and Update Parameters
GAMMA = 0.99             # Discount factor for future rewards
TAU = 1e-3               # Soft update factor for target network (polyak averaging)
UPDATE_EVERY = 4         # Number of steps between network updates

# Optimizer / Training
LR = 5e-4                # Learning rate for the Q-network optimizer
