from typing import Dict

from dotenv import dotenv_values

from gondar.settings import default_config

Gconfig: Dict = {}
for config_cls in dir(default_config):
    if config_cls.endswith("Config"):
        k = getattr(default_config, config_cls)
        Gconfig.update(k.to_dict())

Gconfig.update(dotenv_values())
