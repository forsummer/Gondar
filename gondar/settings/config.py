from gondar.utils.base import baseConfig


class IdentityConfig(baseConfig):
    """
    Identity configuration.
    """

    EMAIL: str | None = None


class NetworkConfig(baseConfig):
    """
    Network configuration
    """

    HTTP_PROXY: str | None = None
    HTTPS_PROXY: str | None = None
    FTP_PROXY: str | None = None

    MAX_RETRY: int | None = 3  # times
    RETRY_GAP: int | None = 5  # sec

    TIMEOUT: int | None = 120  # sec


class PerformanceConfig(baseConfig):
    """
    Performance configuration
    """
