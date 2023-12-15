class __baseConfig(object):
    @classmethod
    def to_dict(cls):
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        }


class _CacheConfig(__baseConfig):
    """
    CacheConfig
    """

    SAVE_CHECKPOINT: bool = False

    ALLOW_PARENT: bool = True
    CACHE_DIRECTORY: str = ".local/"

    USE_SHELVE: bool = True
    SHELVE_NAME: str = "local.shelf"


class _IdentityConfig(__baseConfig):
    """
    Identity configuration.
    """

    EMAIL: str | None = None


class _NetworkConfig(__baseConfig):
    """
    Network configuration
    """

    HTTP_PROXY: str | None = None
    HTTPS_PROXY: str | None = None
    FTP_PROXY: str | None = None

    MAX_RETRY: int | None = 3  # times
    RETRY_GAP: int | None = 5  # sec

    TIMEOUT: int | None = 120  # sec


class _PerformanceConfig(__baseConfig):
    """
    Performance configuration
    """

    USE_MULTIPROCESSING: bool = False
    USE_MAX_PROCESSOR: bool = False
    USE_PROCESSOR: int = 1
