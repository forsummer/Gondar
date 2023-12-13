from enum import Enum

class Identity(Enum):
    """
    Identity configuration.
    """
    EMAIL: str | None = None
    
class Network(Enum):
    """
    Network configuration
    """
    HTTP_PROXY: str | None = None
    HTTPS_PROXY: str | None = None
    FTP_PROXY: str | None = None
    