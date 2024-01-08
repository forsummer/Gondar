from abc import abstractmethod
from typing import Annotated, Any, Callable, Dict, Generator, Iterator, List, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

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

    data = "\n\n".join(data[1]["tables"])

    splitted = tokenizer.encode(data)

    heads = [
        "Cell_Strain",
        "Cultured_condition",
        "Astaxanthin production",
    ]

    helpful_assistant: Dict[str, str] = {
        "role": "system",
        "template": """You are a helpful research assistant.
        Your task is to extract relevant information from the given reference text based on specified headers, and organize it into a column-based table. 
        
        NOTE!!!:
        1. Thoroughly read the entire reference text.
        2. If no relevant information is found, return an empty table.
        3. If there is relevant information, try to capture all pertinent details.
        4. The information in the returned table must be directly sourced from the original reference text, without any modification or embellishment.
        
        Print the result in the following format as JSON:
        {{ column1: [row1, row2, row3, ...], column2: [row1, row2, row3, ...]}}
        """,
    }

    text_extract: Dict[str, str] = {
        "role": "user",
        "template": "Refrence: {reference} \n\nHeaders: {heads}",
    }

    templates: List[MessageTemplate] = [
        MessageTemplate(**helpful_assistant),
        MessageTemplate(**text_extract),
    ]
    messagesWrapper = MessagesWrapper(template_store=templates)

    def messages_iter():
        for i in range(0, 20000, 4800):
            yield messagesWrapper.generate(
                reference=tokenizer.decode(splitted.ids[i : i + 5000]),
                heads=heads,
            )

    llm = AzureOpenAIWrapper(
        azure_openai_endpoint="https://protogabio-gpt.openai.azure.com",
        azure_deployment="protogabio-gpt4",
        azure_openai_key="671368cabc444adfbe2ff0c7873e98ac",
    )

    result = map(llm.invoke, messages_iter())

    for r in result:
        print(r)
        print("\n")
