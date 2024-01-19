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
from typing import Any, Dict, Generator, List, Literal

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

    max_retries: POS_INT = 1  # times
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
        Assistant's Identity:
        Assistant is an excellent data crawler.

        Assistant's Task:
        Assistant scraping a data list to help the user complete the purpose by referencing the provided headers and examples from the document.

        Assistant's self-requirements:
        - Thoroughly paying attention to the entire document with no lazy.
        - Directly scraping data from the document with reasonable inference.

        JSON format of data list:
        {{ 
            headers: [header1, header2, ...],
            sufficient: [No/Yes, No/Yes, ...],
            specified: [No/Yes, No/Yes, ...],
            type matching: [No/Yes, No/Yes, ...],
            data: {{entry1: [column1, column2, ...], entry2: [column1, column2, ...], ...}},
        }}

        Assistant will carefully consider step by step before scraping:
        1. Understand user's purpose.
        2. Determine whether each header is to extract Named Entity, Values/Unit, or a Brief.
        3. Determine whether the Document contains sufficient and specified data. Record as 'sufficient' and 'specified'.
        4. Determine whether the data type reported in the Document match the header data type (Named Entity, Values/Unit, or Brief)? Record as 'type matching'.
        5. Check if the Document contains the data required by the headers, output empty data list if the 'sufficient', 'specified' or 'type matching' for any header is 'No'.
        6. Learn from the data entry examples provided by the user.
        7. Scrap data entry one by one from Document that satisfies user Purpose and all headers. Record as 'data'.
        8. Explode the data list to ensure only one object is described in an entry.
        9. Ensure consistent column count for each row of entry.

        Assistant output following types of data:
        - Named Entity: Must be a noun or term with not exceeding 7 words.
        - Values/Unit: Must be a value and unit
        - Brief: Must be a concise description with not exceeding 30 words.
        """,
    }

    user: Dict[str, str] = {
        "role": "user",
        "content": """
        Document:
        {document}

        Purpose:
        {purpose}
        
        Headers:
        {headers}
        
        Data entry examples:
        {examples}
        """,
    }

    assistant: Dict[str, str] = {
        "role": "assistant",
        "content": """
        Let's do it with great enthusiasm and vigor!
        
        First, I check the data types of each header.
        Then, I check whether the data in the document satisfies 'sufficient', 'specified', or 'type matching' for each header: {headers}.
        Finally, I refer to the data entry examples to print the data entries I have found.
    
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
        Assistant is a meticulous data analyst.
        
        Assistant will analyze step by step:
        1. User's purpose is the task scenario.
        2. Specify whether each header is to extract Named Entity, Values/Unit, or a Brief.
        3. Check data with mismatched data types and names against the headers.
        4. Check incomplete data.
        5. Record index of mismatching entries and incomplete entries as an integer within 'delete'.
        
        Assistant present JSON object:
        {{
            delete: [1,2,...],
        }}
        
        Requirements:
        - Assistant pay thorough attention to the entire Tabular JSON.
        """,
    }

    user: Dict[str, str] = {
        "role": "user",
        "content": """
        Purpose:
        {purpose}
        
        Tabular JSON:
        {tabular}
        """,
    }

    assistant: Dict[str, str] = {
        "role": "assistant",
        "content": """
        I will finish all my tasks with all your requirements.
        Here is the prefectest list print as JSON:
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


if __name__ == "__main__":
    pl.Config.set_tbl_rows(100)

    custom_kw = "(Yarrowia lipolytica) AND (astaxanthin)"

    custom_headers: List[str] = [
        "Named Entity: Strain of Yarrowia lipolytica",
        "Named Entity: Compound Types",
        "Values/Unit: Compound Production",
    ]
    custom_headers = str(custom_headers)

    custom_purpose = "Retrieve strain of Yarrowia lipolytica and the production of any types of Compound."

    custom_examples = """
    {{
        data: {{entry1: ["Yarrowia lipolytica", "Astaxanthin", "3 mg/L"], entry2: ["Y. li", "β-Carotene", "20 mg/"]}}
    }}
    """

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

        for body in doc[i]["body"]:
            mes = doc_extract.generate(
                document=body,
                headers=custom_headers,
                purpose=custom_purpose,
                examples=custom_examples,
            )

            try:
                res = llm.invoke(mes)

                print(res.usage)
                total_comp += res.usage.completion_tokens
                total_prompt += res.usage.prompt_tokens

                res_content = res.choices[0].message.content.strip()
                print(res_content)
                res_json = json.loads(res_content)

                if res_json["data"] != {}:
                    df = pl.DataFrame(data=res_json["data"]).transpose()
                    df = df.rename(dict(zip(df.columns, res_json["headers"])))
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
        )

        res = llm.invoke(mes)

        print(res.usage)
        total_comp += res.usage.completion_tokens
        total_prompt += res.usage.prompt_tokens

        res_content = res.choices[0].message.content.strip()
        print(res_content)
        res_json = json.loads(res_content)

        if res_json["delete"] != {}:
            deleted_index = [int(i) for i in res_json.get("delete")]
            filter_df = sum_df.filter(~pl.arange(0, pl.count()).is_in(deleted_index))

        print(filter_df)

        report.append(filter_df.unique(subset=filter_df.columns))

    report: pl.DataFrame = pl.concat(report)
    print(report)

    report.write_csv("test_df.csv", separator=",")

    print(f"total prompt: {total_prompt}")
    print(f"total completions: {total_comp}")
