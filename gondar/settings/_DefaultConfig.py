from pydantic import Field, create_model

from gondar.utils import DEFAULT_FALSE, POS_INT, POS_NUM, STR, GondarPydanticModel


class _IdentityConfig(GondarPydanticModel):
    """
    Identity configuration.
    """

    # User info
    EMAIL: STR = Field(default=None, examples=["explorer@email.com"])


class _NetworkConfig(GondarPydanticModel):
    """
    Network configuration
    """

    # Waiting
    MAX_RETRY: POS_INT = Field(default=3, lt=10)  # times
    RETRY_GAP: POS_NUM = Field(default=5, lt=5 * 60)  # sec
    TIMEOUT: POS_NUM = Field(default=30, lt=60 * 60)  # sec

    # Network proxy
    HTTP_PROXY: STR = Field(default=None, examples=["http://127.0.0.1:7890"])
    HTTPS_PROXY: STR = Field(default=None, examples=["https://127.0.0.1:7891"])
    FTP_PROXY: STR = Field(default=None, examples=["ftp://127.0.0.1:7892"])


class _PerformanceConfig(GondarPydanticModel):
    """
    Performance configuration
    """

    # Multiprocessing
    USE_MULTIPROCESSING: DEFAULT_FALSE
    USE_MAX_PROCESSOR: DEFAULT_FALSE
    USEABLE_PROCESSOR: POS_INT = Field(default=1)

    # Multithreading
    USE_MULTITHREADING: DEFAULT_FALSE
    USEABLE_THREADS: POS_INT = Field(default=1)


GondarGlobalConfig = create_model(
    "GondarGlobalConfig",
    __base__=(_IdentityConfig, _NetworkConfig, _PerformanceConfig),
)
