import os

from dotenv import load_dotenv

from gondar.settings.config import IdentityConfig, NetworkConfig

os.environ.update(**IdentityConfig)
os.environ.update(**NetworkConfig)

load_dotenv(verbose=True)
