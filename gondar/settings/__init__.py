from dotenv import dotenv_values

from gondar.settings._DefaultConfig import GondarGlobalConfig

custom_envs = dotenv_values(encoding="utf-8")
Gconfig = GondarGlobalConfig(**custom_envs)


__all__ = ["Gconfig"]
