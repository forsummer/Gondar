from enum import Enum


class IdentityConfig(Enum):
    """
    Identity configuration.
    """

    EMAIL: str | None = None


class NetworkConfig(Enum):
    """
    Network configuration
    """

    HTTP_PROXY: str | None = None
    HTTPS_PROXY: str | None = None
    FTP_PROXY: str | None = None
