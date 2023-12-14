import os
from pathlib import Path

from gondar.settings.auto_config import Gconfig

if Gconfig["SAVE_CHECKPOINT"] and Gconfig["ALLOW_PARENT"]:
    cache_dir = Path.cwd() / Path(Gconfig["CACHE_DIRECTORY"])

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=Gconfig["ALLOW_PARENT"])

__all__ = ["Gconfig"]
