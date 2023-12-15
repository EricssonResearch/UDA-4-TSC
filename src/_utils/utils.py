import numpy as np
import random
from transformers import set_seed


"""
utils.py should contain only stuff that all stages in the pipeline depends on
"""


def set_random_seed(seed: int):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        set_seed(seed)
