import os

from gondar.settings.config import IdentityConfig, NetworkConfig

os.environ.update(**IdentityConfig)
os.environ.update(**NetworkConfig)
