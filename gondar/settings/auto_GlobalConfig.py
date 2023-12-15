import functools
import inspect
from typing import Any, Dict

from dotenv import dotenv_values

from gondar.exception import EnviromentError
from gondar.settings import _defaultConfig


class DotDict(dict):
    def __init__(self):
        super().__init__()

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)  # Set proper exception, not KeyError


class GlobalConfig(object):
    """
    A safe global config which can only be modified by the restricted methods.
    """

    SAFE_WARN = "Safe config is not allow to modify."

    def __init__(self):
        self.__config: DotDict = DotDict()

    def restrict_to(self, func):
        def set_config(settings: Dict):
            self.__config.update(settings)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(set_config, *args, **kwargs)

        return wrapper

    def __getattr__(self, key):
        if hasattr(self.__config, key):
            return getattr(self.__config, key)
        else:
            raise EnviromentError(f"Found no {key} parameter within global config.")

    def __setattr__(self, __name: str, __value: Any) -> None:
        calling_frame = inspect.stack()[1]
        if calling_frame.function == "__init__":
            super().__setattr__(__name, __value)
        else:
            raise EnviromentError(self.SAFE_WARN)

    def __delattr__(self, __name: str) -> None:
        raise EnviromentError(self.SAFE_WARN)


Gconfig = GlobalConfig()


@Gconfig.restrict_to
def __init_Gconfig(set_config):
    configs: Dict = {}
    for dc in dir(_defaultConfig):
        if (not dc.startswith("__")) and (dc.endswith("Config")):
            c = getattr(_defaultConfig, dc)
            configs.update(c.to_dict())

    set_config(configs)


@Gconfig.restrict_to
def _update_Gconfig(set_config, new_config: Dict):
    set_config(new_config)


__init_Gconfig()
_update_Gconfig(dotenv_values())
