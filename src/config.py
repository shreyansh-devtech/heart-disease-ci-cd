# This file stores all configurations in one place.

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data path
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")

# Model path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")

# Random state for reproducibility
RANDOM_STATE = 42

# Test split ratio
TEST_SIZE = 0.2
    