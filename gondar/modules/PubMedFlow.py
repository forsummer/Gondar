from abc import abstractmethod
from typing import Annotated, Any, Callable, Dict, Generator, Iterator, List, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from gondar import Gconfig
from gondar.utils.Message import Message
from gondar.utils.types import POS_FLOAT, POS_INT, STR, VALID


class GondarModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class AzureOpenAIWrapper(GondarModel):
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


class Message(GondarModel):
    role: str
    content: str


class MessageTemplate(GondarModel):
    role: str
    template: str

    def fill(self, **kwargs):
        return Message(role=self.role, content=self.template.format(**kwargs))


class MessagesWrapper(GondarModel):
    template_store: List[MessageTemplate]

    def generate(
        self, reference: str, heads: List[str], motivation: str
    ) -> List[Message]:
        heads = str(heads)

        _messages: Generator[Message] = (
            template.fill(reference=reference, heads=heads, motivation=motivation)
            for template in self.template_store
        )

        return [mes for mes in _messages]


if __name__ == "__main__":
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    import json

    import polars as pl

    from gondar.tools import EntrezAPIWrapper

    entrez = EntrezAPIWrapper(retmax=3)
    data = entrez.load("(Chlamydomonas reinhardtii) AND (Terpene)")

    # Each Requirement is fucking crucial.
    helpful_assistant: Dict[str, str] = {
        "role": "system",
        "template": """
        You are an intelligent research assistant.

        Your tasks:
        * Understand the user's motivation.
        * Select one of your abilities required for analyzing each type of header.
        * Retrieve relevant information based on the provided headers and corresponding descriptions.
        * Organize into a concise, tidy structured data table.
        
        Present the table as a JSON:
        {{ 
            headers: [header1, header2, ...],
            data: {{row1: [column1, column2, ...], row2: [column1, column2, ...], ...}}
        }}
        
        Requirements:
        * You pay thorough attention to the entire reference text.
        * Your output information should be sourced directly from the provided reference text, DO NOT make any modifications or alterations.
        * You drop the table if any header is being evaluate as 'unsatisfactory'.
        * You should explode the table to prevent too much information in the same row.
        * You must ensure consistent column count for each row.
        * You DO NOT output the description of headers.
        * DO NOT output '\\n'.
        """,
    }

    text_extract: Dict[str, str] = {
        "role": "user",
        "template": """
        Motivation:
        {motivation}
        
        Refrence Text:
        {reference}
        
        Headers:
        {heads}
        
        Take a deep breath, remember all your tasks, and all Requirements.
        Print JSON object:
        """,
    }

    templates: List[MessageTemplate] = [
        MessageTemplate(**helpful_assistant),
        MessageTemplate(**text_extract),
    ]
    messagesWrapper = MessagesWrapper(template_store=templates)

    heads = [
        "Engineered strains. Description: An exact Name or ID.",
        "Terpenes Types. Description: Types name or ID.",
        "Terpenes Production. Description: A production values.",
        "Key Protocol. Description: A brief about the key protocol.",
    ]

    motivation = """
    Retrieve engineered strain of Chlamydomonas reinhardtii and the production of any types of terpenes.
    """

    body = "\n\n".join(data[0]["body"])

    splitted = tokenizer.encode(body)

    def messages_iter():
        for i in range(0, len(splitted), 40_000):
            yield messagesWrapper.generate(
                reference=tokenizer.decode(splitted.ids[i : i + 40_000]),
                heads=heads,
                motivation=motivation,
            )

    llm = AzureOpenAIWrapper(
        azure_openai_endpoint=Gconfig.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Gconfig.AZURE_DEPLOYMENT,
        azure_openai_key=Gconfig.AZURE_OPENAI_KEY,
        model="gpt-4-1106-preview",
    )

    print(data[1]["article"])
    for body in data[1]["body"]:
        mes = messagesWrapper.generate(
            reference=body,
            heads=heads,
            motivation=motivation,
        )

        r = llm.invoke(mes)
        print(r.usage)
        c = r.choices[0].message.content.strip()
        cd = json.loads(c)
        print(c)
        if cd["data"] != {}:
            df = pl.DataFrame(data=cd["data"]).transpose()
            df = df.rename(dict(zip(df.columns, cd["headers"])))
            print(df)

            print("\n")

    # for mes in messages_iter():
    #     r = llm.invoke(mes)
    #     print(r.usage)
    #     c = r.choices[0].message.content.strip()
    #     cd = json.loads(c)
    #     print(c)
    #     if cd["data"] != {}:
    #         df = pl.DataFrame(data=cd["data"]).transpose()
    #         df = df.rename(dict(zip(df.columns, cd["headers"])))
    #         print(df)

    #         print("\n")
