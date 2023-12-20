import inspect
from typing import Any, Union

from pydantic import BaseModel, Field, create_model
from typing_extensions import Annotated

from gondar.exception import EnviromentError

(
    _POS_INT,
    _POS_NUM,
    _STRICT_STR,
    _DEFAULT_TRUE,
    _DEFAULT_FALSE,
) = (
    Annotated[int, Field(gt=0, kw_only=True)],
    Annotated[Union[int, float], Field(gt=0, kw_only=True)],
    Annotated[str, Field(strict=True, kw_only=True)],
    Annotated[bool, Field(default=True, kw_only=True)],
    Annotated[bool, Field(default=False, kw_only=True)],
)


class __GondarConfig(BaseModel):
    class Config:
        validate_assignment = True

    def __setattr__(self, __name: str, __value: Any) -> None:
        calling_frame = inspect.stack()[1]
        if calling_frame.function == "__init__":
            super().__setattr__(__name, __value)
        else:
            raise EnviromentError("Gondar config is not allow to modify.")


class _IdentityConfig(__GondarConfig):
    """
    Identity configuration.
    """

    # User info
    EMAIL: _STRICT_STR = Field(default=None, examples=["explorer@email.com"])


class _NetworkConfig(__GondarConfig):
    """
    Network configuration
    """

    # Waiting
    MAX_RETRY: _POS_INT = Field(default=3, lt=10)  # times
    RETRY_GAP: _POS_NUM = Field(default=5, lt=5 * 60)  # sec
    TIMEOUT: _POS_NUM = Field(default=30, lt=60 * 60)  # sec

    # Network proxy
    HTTP_PROXY: _STRICT_STR = Field(default=None, examples=["http://127.0.0.1:7890"])
    HTTPS_PROXY: _STRICT_STR = Field(default=None, examples=["https://127.0.0.1:7890"])
    FTP_PROXY: _STRICT_STR = Field(default=None, examples=["ftp://127.0.0.1:7890"])


class _PerformanceConfig(__GondarConfig):
    """
    Performance configuration
    """

    # Multiprocessing
    USE_MULTIPROCESSING: _DEFAULT_FALSE
    USE_MAX_PROCESSOR: _DEFAULT_FALSE
    USEABLE_PROCESSOR: _POS_INT = Field(default=1)

    # Multithreading
    USE_MULTITHREADING: _DEFAULT_FALSE
    USEABLE_THREADS: _POS_INT = Field(default=1)


GondarGlobalConfig = create_model(
    "Gconfig",
    __base__=(_IdentityConfig, _NetworkConfig, _PerformanceConfig),
)
