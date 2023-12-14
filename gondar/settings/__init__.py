from typing import Dict

from dotenv import dotenv_values

from gondar.settings import config

gconfig: Dict = {}
for klass in dir(config):
    if klass.endswith("Config"):
        k = getattr(config, klass)
        gconfig.update(k.to_dict())

gconfig.update(dotenv_values())
