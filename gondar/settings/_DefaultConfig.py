from inspect import stack as InspectStack
from typing import Any

from pydantic import BaseModel, Field, create_model
from typing_extensions import Annotated

from gondar.exception import ConfigError
from gondar.utils import DEFAULT_FALSE, POS_INT, POS_NUM, STR


class GondarConfigModel(BaseModel):
    class Config:
        validate_assignment = True

    def __setattr__(self, __name: str, __value: Any) -> None:
        calling_frame = InspectStack()[1]
        if calling_frame.function == "__init__":
            super().__setattr__(__name, __value)
        else:
            raise ConfigError(f"Gondar safe config {__name} is not allow to be modify.")


class _IdentityConfig(GondarConfigModel):
    """
    Identity configuration.
    """

    # User info
    EMAIL: Annotated[STR, Field(default=None, examples=["explorer@email.com"])]


class _NetworkConfig(GondarConfigModel):
    """
    Network configuration
    """

    # Waiting
    MAX_RETRY: Annotated[POS_INT, Field(default=3, lt=10)]  # times
    RETRY_GAP: Annotated[POS_NUM, Field(default=5, lt=5 * 60)]  # sec
    TIMEOUT: Annotated[POS_NUM, Field(default=30, lt=60 * 60)]  # sec

    # Network proxy
    HTTP_PROXY: Annotated[STR, Field(default=None, examples=["http://127.0.0.1:7890"])]
    HTTPS_PROXY: Annotated[
        STR, Field(default=None, examples=["https://127.0.0.1:7891"])
    ]
    FTP_PROXY: Annotated[STR, Field(default=None, examples=["ftp://127.0.0.1:7892"])]


class _PerformanceConfig(GondarConfigModel):
    """
    Performance configuration
    """

    # Multiprocessing
    USE_MULTIPROCESSING: DEFAULT_FALSE
    USE_MAX_PROCESSOR: DEFAULT_FALSE
    USEABLE_PROCESSOR: Annotated[POS_INT, Field(default=1)]

    # Multithreading
    USE_MULTITHREADING: DEFAULT_FALSE
    USEABLE_THREADS: Annotated[POS_INT, Field(default=1)]


class _LLMConfig(GondarConfigModel):
    """
    LLM configuration
    """

    # OpenAI

    # AzureOpenAI
    AZURE_OPENAI_ENDPOINT: Annotated[STR, Field(default=None)]
    AZURE_OPENAI_KEY: Annotated[STR, Field(default=None)]
    AZURE_DEPLOYMENT: Annotated[STR, Field(default=None)]
    AZURE_API_VERSION: Annotated[STR, Field(default=None)]


GondarGlobalConfig = create_model(
    "GondarGlobalConfig",
    __base__=(
        _IdentityConfig,
        _NetworkConfig,
        _PerformanceConfig,
        _LLMConfig,
    ),
)
