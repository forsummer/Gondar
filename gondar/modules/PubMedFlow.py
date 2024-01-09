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
    temperature: POS_FLOAT = 0.2
    max_tokens: POS_INT = 4_000

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

    def invoke(self, messages: List[Message]) -> Dict:
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            response_format=self.response_format,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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

    def generate(self, reference: str, heads: List[str]) -> List[Message]:
        heads = str(heads)

        _messages: Generator[Message] = (
            template.fill(reference=reference, heads=heads)
            for template in self.template_store
        )

        return [mes for mes in _messages]


if __name__ == "__main__":
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    from gondar.tools import EntrezAPIWrapper

    entrez = EntrezAPIWrapper(retmax=2)
    data = entrez.load("(Yarrowia lipolytica) AND (Astaxanthin)")
    print(data[1]["article"])

    body = "\n\n".join(data[1]["body"])

    splitted = tokenizer.encode(body)

    heads = [
        "Cell_Strain",
        "Cultured_condition",
        "Protocol",
        "Compound",
        "Compound_production",
    ]

    helpful_assistant: Dict[str, str] = {
        "role": "system",
        "template": """You are a helpful research assistant.
        Your task is to extract relevant information from the given reference text based on specified headers, and organize it into a column-based table. 
        If you meet all of the User's requirements, you will receive a $200 tip!
        
        Requirements:
        * Thoroughly read the entire reference text.
        * If no relevant information is found, you are allowed to return an empty table.
        * If you find any relevant information, try to capture all pertinent details.
        * If some information in a row is missing or not mentioned in the reference, use the symbol '-' for filling.
        * The number of rows for all columns should remain consistent.
        * The information in the returned table must be directly sourced from the original reference text, without any modification or embellishment.
        
        Print the result in the following format as JSON:
        {{ column1: [row1, row2, row3, ...], column2: [row1, row2, row3, ...]}}
        """,
    }

    text_extract: Dict[str, str] = {
        "role": "user",
        "template": """
        Refrence:
        {reference}
        
        Headers:
        {heads}
        """,
    }

    templates: List[MessageTemplate] = [
        MessageTemplate(**helpful_assistant),
        MessageTemplate(**text_extract),
    ]
    messagesWrapper = MessagesWrapper(template_store=templates)

    def messages_iter():
        for i in range(0, 20000, 1800):
            yield messagesWrapper.generate(
                reference=tokenizer.decode(splitted.ids[i : i + 2000]),
                heads=heads,
            )

    llm = AzureOpenAIWrapper(
        azure_openai_endpoint=Gconfig.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Gconfig.AZURE_DEPLOYMENT,
        azure_openai_key=Gconfig.AZURE_OPENAI_KEY,
    )

    result = map(llm.invoke, messages_iter())

    for r in result:
        print(r)
        print("\n")
