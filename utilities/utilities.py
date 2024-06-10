import os
import json
import platform
import torch
from torch import Tensor

def get_device() -> torch.device:
    if platform.system().upper() == "DARWIN":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif platform.system().upper() == "WINDOWS":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")