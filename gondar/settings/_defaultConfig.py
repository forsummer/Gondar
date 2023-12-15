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


class _SqlConfig(__baseConfig):
    """
    SQL config
    """


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
    USEABLE_PROCESSOR: int = 1

    USE_MULTITHREADING: bool = False
    USEABLE_THREADS: int = 1
