"""
Document Structuring Workflow Based on PubMed Data

---------------------------------

The scripts should be included following models:\n

1. Tools: The Interface of External API or Function,
such as entrez (Network access interface for external resources)
and Aliyun function computation. \n

2. Document: The data model coming from Tools.

3. PromptTemplate: The string template that use for organize the prompt.

4. Messages: The data model sending to Agent (LLM).

5. Agent: LLM interface.

6. Parser: The data (processing) model that parsering the coming data from Agent (LLM).

7. Callback: Various plug-and-play callback models,
such as the callback model for recording Tokens Usage
or the callback model for interacting with the frontend.

8. Memory: A model for interfacing with databases,
which may include but is not limited to Memory, Redis, or databases like Postgre, Mongo, etc.
Used to provide the model with cached, persistent memory, or historical records.

---------------------------------

Combining the above models into a workflow (Flow) model.
This Flow model will directly provide interfaces for external access.

"""

import json
from typing import Any, Dict, Generator, Iterator, List, Literal

import polars as pl
from pydantic import BaseModel, model_validator

from gondar import Gconfig
from gondar.tools import EntrezAPIWrapper
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
    seed: POS_INT = 1001

    max_retries: POS_INT = 2  # times
    timeout: POS_INT = 90  # seconds

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
            seed=self.seed,
        )


class PromptTemplate(GondarModel):
    template_store: List[Message]

    def generate(self, **kwargs) -> List[Message]:
        _messages: Generator[Message] = (
            Message(role=mes.role, content=mes.content.format(**kwargs))
            for mes in self.template_store
        )

        return [mes for mes in _messages]


class DocumentBodyExtractPromptTemplate(PromptTemplate):
    system: Dict[str, str] = {
        "role": "system",
        "content": """
        Assistant is an excellent data crawler.
        Assistant scraping a data list to help the user complete the purpose by referencing the provided headers and examples from the document.

        Assistant output following data types:
        - Named Entity: A noun or term with not exceeding 5 words.
        - Value/Units: A number or a range of number, better with units. 
        - Brief: A concise description with not exceeding 31 words.

        Assistant's self-requirements:
        - Thoroughly paying attention to the entire document with no lazy.
        - Directly scraping data from the document with reasonable inference.

        Assistant will output the JSON step by step:
        1. Understand user's purpose.
        2. Output the name of each header in the "headers", excluding the data type.
        3. Output the data types corresponding to each header in the "data type".
        4. Carefully consider which arguments contain information that aligns with all headers.
        5. Record the indices of arguments that satisfy all headers in the 'Available argument'. Output an empty list if found no satisfied arguments.
        6. Output the data as an empty list if Available argument is empty list.
        if Available argument is not emtpy list, continue:
            1. Extract data entries that match the user's purpose, user-specified headers, and the data types corresponding to each header from the available arguments.
            2. Append the index of the referenced argument to the first column of each entry.
            3. Data entries should be similar to the entry examples provided by the user.
            4. Expand the data list to ensure only one object is described in an entry.
            5. Ensure consistent column count for each row of entry.

        JSON format:
        {{ 
            headers: [header1, header2, ...],
            data type: [type1, type2, ...]
            Available argument: [0,1, ...],
            data: {{entry1: [column1, column2, ...], entry2: [column1, column2, ...], ...}},
        }}
        """,
    }

    user: Dict[str, str] = {
        "role": "user",
        "content": """
        User's JSON:
        {{
            "Purpose": {purpose},
            "Headers": {headers},
            "Document": {document},
            "Entry examples": {examples},
        }}
        """,
    }

    assistant: Dict[str, str] = {
        "role": "assistant",
        "content": """
        Let's do it with great enthusiasm and vigor!

        First, I understand the user's purpose, the meaning of each header and the corresponding data type.
        Then, I retrieve arguments that have the aligned data for {headers}.
        Finally, I output data entries based on the arguments and entry examples.

        JSON object:
        """,
    }

    template_store: List[Message] = [
        Message(**system),
        Message(**user),
        Message(**assistant),
    ]


class TabularTrimmingPromptTemplate(PromptTemplate):
    system: Dict[str, str] = {
        "role": "system",
        "content": """
        Assistant's Identity:
        Assistant is a meticulous data analyst.

        Assistant's Task:
        Check if the user-specified strict header aligns with the user purpose and the reference data types. If not, delete the misalign and incomplete data entries.

        Assistant's self-requirements:
        - Assistant pay thorough attention to the entire Tabular JSON.

        Reference data types:
        - Named Entity: A noun or term with not exceeding 5 words.
        - Value/Units: An exact value with units. For examples: 1mg/L, 2%, 3-fold, 4μg/L·h-1, 5% increase. 
        - Brief: A concise description with not exceeding 31 words.

        Assistant present JSON object:
        {{
            headers: [header1, header2, ...],
            data type: [type1, type2, ...],
            delete: [0,4,17,...],
        }}

        Assistant will carefully analyze step by step:
        1. Understand user's purpose.
        2. Confirm the headers that the user wants to check rigorously. Record as "headers".
        3. Check if each header is to extract Named Entity, Value/Units, or a Brief. Record as "data type".
        4. Retrieve data entry with misaligned data types and names against the headers. Record as an integer within 'delete'.
        5. Retrieve incomplete data. Record as an integer within 'delete'.
        """,
    }

    user: Dict[str, str] = {
        "role": "user",
        "content": """
        User's JSON:
        {{
            "Purpose": {purpose},
            "Strict": {strict},
            "Tabular": {tabular},
        }}
        """,
    }

    assistant: Dict[str, str] = {
        "role": "assistant",
        "content": """
        Let's do it with great enthusiasm and vigor!

        First, I check the data types of each header.
        Then, I check data entry with misaligned data types and names against the headers.
        Next, I check incomplete data entry.
        Finally, I record index of misaligned entries and incomplete entries.

        JSON object:
        """,
    }

    template_store: List[Message] = [
        Message(**system),
        Message(**user),
        Message(**assistant),
    ]


def df_to_json(df: pl.DataFrame):
    headers = df.columns

    rows = df.rows()
    rows = {f"entry_{i}": list(rows[i]) for i in range(len(rows))}

    return json.dumps(
        {
            "headers": headers,
            "data": rows,
        },
        ensure_ascii=False,
    )


def wrap_batch(content: List[str], load: int = 4096) -> Iterator[List[str]]:
    content.reverse()
    batch = []

    while content:
        batch.append(content.pop())
        if sum(map(len, batch)) >= load:
            yield batch
            batch.clear()
        else:
            if not content:
                yield batch


if __name__ == "__main__":
    pl.Config.set_tbl_rows(50)

    custom_kw = "(Yarrowia lipolytica) AND (astaxanthin)"

    custom_headers: List[str] = [
        "Named Entity: Strain",
        "Named Entity: Compound Type",
        "Value/Units: Compound Production",
        "Named Entity: Pathway or Gene",
    ]

    custom_purpose = "Retrieve strain of Yarrowia lipolytica and the production of any types of Compound."

    custom_examples = {
        "data": {"entry1": ["Yarrowia lipolytica", "Astaxanthin", "3 mg/L", "EcAcrBp"]},
    }

    entrez = EntrezAPIWrapper(retmax=3)

    doc = entrez.load(custom_kw)

    llm = AzureOpenAIWrapper(
        azure_openai_endpoint=Gconfig.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Gconfig.AZURE_DEPLOYMENT,
        azure_openai_key=Gconfig.AZURE_OPENAI_KEY,
    )

    doc_extract = DocumentBodyExtractPromptTemplate()

    total_prompt = 0
    total_comp = 0

    report = []

    for i in range(len(doc)):
        dfs = []

        if doc[i]["body"] == []:
            continue

        print(doc[i]["article"])

        for batch in wrap_batch(doc[i]["body"]):
            formatted_batch = {f"argument {i}": batch[i] for i in range(len(batch))}

            print(formatted_batch)

            mes = doc_extract.generate(
                document=json.dumps(formatted_batch, ensure_ascii=False),
                headers=str(custom_headers),
                purpose=custom_purpose,
                examples=str(custom_examples),
            )

            try:
                res = llm.invoke(mes)

                print(res.usage)
                total_comp += res.usage.completion_tokens
                total_prompt += res.usage.prompt_tokens

                res_content = res.choices[0].message.content.strip()
                print(res_content, "\n")
                res_json = json.loads(res_content)

                if res_json["data"] != {}:
                    df = pl.DataFrame(
                        data=[v for k, v in res_json["data"].items()], orient="row"
                    )
                    df = df.rename(dict(zip(df.columns, ["ref"] + res_json["headers"])))
                    print(df)

                    print("\n")

                    dfs.append(df)

            except Exception as e:
                print(e)
                continue

        sum_df: pl.DataFrame = pl.concat(dfs)
        print(sum_df)

        json_df = df_to_json(sum_df)

        tabular_trim = TabularTrimmingPromptTemplate()

        mes = tabular_trim.generate(
            purpose=custom_purpose,
            tabular=json_df,
            headers=custom_headers,
            strict=str(["Strain", "Compound Type", "Compound Production"]),
        )

        try:
            res = llm.invoke(mes)

            print(res.usage)
            total_comp += res.usage.completion_tokens
            total_prompt += res.usage.prompt_tokens

            res_content = res.choices[0].message.content.strip()
            print(res_content)
            res_json = json.loads(res_content)

            if res_json["delete"] != {}:
                deleted_index = [int(i) for i in res_json.get("delete")]
                filter_df = sum_df.filter(
                    ~pl.arange(0, pl.count()).is_in(deleted_index)
                )
                filter_df = filter_df.with_columns(
                    [
                        pl.lit(v).alias(k)
                        for k, v in doc[i].items()
                        if k not in ["body", "tables"]
                    ]
                )

            print(filter_df)

            report.append(filter_df.unique(subset=filter_df.columns))

        except Exception as e:
            print(e)
            continue

        print(f"total prompt: {total_prompt}")
        print(f"total completions: {total_comp}")

        report_df: pl.DataFrame = pl.concat(report)
        report_df.write_csv("test_df.csv", separator=",")

    print(report_df)
