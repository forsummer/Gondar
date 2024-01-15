import json
from typing import Any, Dict, Generator, List, Literal

import polars as pl
from pydantic import BaseModel, model_validator

from gondar import Gconfig
from gondar.utils.types import POS_FLOAT, POS_INT, STR, VALID


class GondarModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Message(GondarModel):
    role: str
    content: str


class AzureOpenAIWrapper(GondarModel):
    """
    Azure OpenAI API Document: \n
    https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode?tabs=python
    """

    client: Any = None

    azure_openai_endpoint: STR
    azure_deployment: STR
    azure_openai_key: STR
    azure_api_version: STR = "2023-12-01-preview"

    model: Literal[
        "gpt-4-1106-preview", "gpt-35-turbo-1106"
    ] = "gpt-4-1106-preview"  # Since 12/07/2023
    response_format: Dict = {"type": "json_object"}
    temperature: POS_FLOAT = 0.0

    max_retries: POS_INT = 2  # times
    timeout: POS_INT = 300  # seconds

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

    def invoke(self, messages: List[Message]) -> Dict:
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            response_format=self.response_format,
            temperature=self.temperature,
            seed=1001,
        )


class MessageTemplate(GondarModel):
    role: str
    template: str

    def fill(self, **kwargs):
        return Message(role=self.role, content=self.template.format(**kwargs))


class MessagesWrapper(GondarModel):
    template_store: List[MessageTemplate]

    def generate(
        self,
        motivation: str,
        tabular: pl.DataFrame,
    ) -> List[Message]:
        tabular = tabular.to_pandas().to_json()

        _messages: Generator[Message] = (
            template.fill(motivation=motivation, tabular=tabular)
            for template in self.template_store
        )

        return [mes for mes in _messages]


def df_to_json(df: pl.DataFrame):
    headers = df.columns

    rows = df.rows()
    rows = {f"row_{i}": list(rows[i]) for i in range(len(rows))}

    return json.dumps(
        {
            "headers": headers,
            "data": rows,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    df = pl.read_parquet("test_df.parquet")
    df_string = df_to_json(df)

    helpful_assistant: Dict[str, str] = {
        "role": "system",
        "template": """
        You are an intelligent research assistant.
        """,
    }

    text_extract: Dict[str, str] = {
        "role": "user",
        "template": """
        Motivation:
        {motivation}
        
        Tabular JSON:
        {tabular}
        """,
    }

    self_check: Dict[str, str] = {
        "role": "assistant",
        "template": """     
        Let me strictly check if the reference text contains the data required by the headers: {heads}.
        If not satisfied, record as 'No'; if satisfied, record as 'Yes'.
        
        I will finish all my tasks with all your requirement.
        Here is the prefectest list print as JSON:
        """,
    }

    templates: List[MessageTemplate] = [
        MessageTemplate(**helpful_assistant),
        MessageTemplate(**text_extract),
        MessageTemplate(**self_check),
    ]

    messagesWrapper = MessagesWrapper(template_store=templates)

    motivation = """
    Retrieve engineered strain of Chlamydomonas reinhardtii and the production of any types of terpenes.
    """

    llm = AzureOpenAIWrapper(
        azure_openai_endpoint=Gconfig.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Gconfig.AZURE_DEPLOYMENT,
        azure_openai_key=Gconfig.AZURE_OPENAI_KEY,
        model="gpt-4-1106-preview",
    )

    mes = messagesWrapper.generate(
        motivation=motivation,
        tabular=df_string,
    )

    res = llm.invoke(mes)
    print(res.usage)
    c = res.choices[0].message.content.strip()
    print(c)
    cd = json.loads(c)
    df = pl.DataFrame(data=cd["data"]).transpose()
    df = df.rename(dict(zip(df.columns, cd["headers"])))
    print(df)
