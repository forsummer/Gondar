from abc import abstractmethod
from typing import Any, Dict, Literal

from pydantic import model_validator

from gondar.utils import POS_INT, STR, VALID, Messages
from gondar.utils.base import GondarModel
from gondar.utils.types import POS_FLOAT


class LLM(GondarModel):
    """LLM core.

    Wrapping the LLM API for easily invoking.
    """

    def __call__(self, messages: Messages) -> Any:
        return self.invoke(messages)

    @abstractmethod
    def invoke(self, messages: Messages) -> Any:
        ...


class AzureJSON(LLM):
    azure_openai_endpoint: STR
    azure_deployment: STR
    azure_openai_key: STR
    azure_api_version: STR = "2023-12-01-preview"

    model_version: Literal[
        "gpt-4-1106-preview", "gpt-35-turbo-1106"
    ] = "gpt-4-1106-preview"  # Since 12/07/2023
    response_format: Dict = {"type": "json_object"}
    temperature: POS_FLOAT | None = None
    max_tokens: POS_INT | None = None
    frequency_penalty: POS_FLOAT | None = None
    seed: POS_INT | None = None

    max_retries: POS_INT = 2  # times
    timeout: POS_INT = 120  # seconds

    @model_validator(mode="before")
    @classmethod
    def validate_import(cls, values: Dict) -> Dict:
        try:
            from openai import AzureOpenAI
        except ImportError as error:
            raise error("Failed to import AzureOpenAI from openai packages.")

        return values

    @model_validator(mode="after")
    def validate_client(self) -> VALID:
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            azure_deployment=self.azure_deployment,
            api_version=self.azure_api_version,
            api_key=self.azure_openai_key,
        )

    def invoke(self, messages: Messages) -> Dict:
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model_version,
            response_format=self.response_format,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            seed=self.seed,
        )
