import argparse
import torch

from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count


UNIQUE_ID = datetime.now().strftime("%Y%m%d_%H%M") # unique ID based on current date and time (YYYYMMDD_HHMM)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    
    return torch.device(device_type)

DEVICE = get_device()

CPU_COUNT: int = cpu_count()
NUM_WORKERS: int = min(8, CPU_COUNT // 2) if DEVICE.type != "cpu" else 0
PERSIST_WORK: bool = NUM_WORKERS > 0
PIN_MEM: bool = DEVICE.type == "cuda"
