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

    model: Literal["gpt-4-1106-preview", "gpt-35-turbo-1106"] = (
        "gpt-4-1106-preview"  # Since 12/07/2023
    )
    response_format: Dict = {"type": "json_object"}
    temperature: POS_FLOAT = 0.0
    seed: POS_INT = 1001

    max_tokens: POS_INT = 2000
    max_retries: POS_INT = 1  # times
    timeout: POS_INT = 60  # seconds

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
            max_tokens=self.max_tokens,
        )


class PromptTemplate(GondarModel):
    template_store: List[Message]

    def generate(self, **kwargs) -> List[Message]:
        _messages: Generator[Message] = (
            Message(role=mes.role, content=mes.content.format(**kwargs))
            for mes in self.template_store
        )

        return [mes for mes in _messages]


# TODO: The highlight prompt is not good enough. Considering about embedding with cosine similarity.
class DocumentBodyExtractPromptTemplate(PromptTemplate):
    Identity: Dict[str, str] = {
        "role": "system",
        "content": """
        Assistant is an smart data analyst.
        Assistant's task is create a data table to complete the researcher's purpose by referencing the headers and examples.

        Assistant will:
        - Thoroughly paying attention to content provided by researcher.
        - Directly extract information from researcher's document with reasonable inference.
        - Try to earn the researcher's tip.
        """,
    }

    researchers: Dict[str, str] = {
        "role": "user",
        "content": """
        researcher Purpose: {purpose}
        researcher Headers: {headers}
        researcher Document: {document}
        researcher Examples: {examples}

        If you do well, I will give you a $10 tip.
        """,
    }

    Rules: Dict[str, str] = {
        "role": "system",
        "content": """
        Legal data types:
        - Entity: A noun or term with not exceeding 5 words.
        - Number: Must include a number, can be also a range or a change of number, with units.
        - Brief: A concise description with not exceeding 31 words.

        Output the JSON after thinking step by step:
        Step1. Output the headers in "headers", excluding the data types.
        Step2. Output the data types corresponding to each header in "data types".
        Step3. Highlight the reference that can sufficiently fill all columns of data entry, output the index (Int) of those reference in "highlight".
        Step4. Output the data as an empty list if highlight is empty list.

        if highlight is not emtpy list, continue output the JSON after thinking step by step:
        Step1. Extract data entries summarized from the highlight.
        Step2. Ensure consistent column count of entries and headers.
        Step3. Focus on up to three of the highlights which is the most relating to this data entry, append the index at the last column.

        Hints:
        1. Fewer highlights in the references can lead to a more focused generation of satisfactory results.
        2. Highlight too much references is wasted.
        3. Referencing legal data types can generate data entry that is easier for researchers to understand.

        JSON format:
        {{
            headers: [header1, header2, ...],
            data types: [column1, column2, ...],
            highlight: [1,2,3,4,...],
            data: {{entry1: [column1, column2, ..., [1]], entry2: [column1, column2, ..., [2,3]], ...}},
        }}

        JSON object: {{
        """,
    }

    template_store: List[Message] = [
        Message(**Identity),
        Message(**researchers),
        Message(**Rules),
    ]


class TabularTrimmingPromptTemplate(PromptTemplate):
    Identity: Dict[str, str] = {
        "role": "system",
        "content": """Assistant is an smart data analyst.
        The assistant trims the data table by checking researcher-specified headers and removing low quality data entries.

        Assistant will:
        - Thoroughly paying attention to the entire researcher's JSON.
        - Will not be influenced by contents of researcher's JSON when making judgments.
        - Try to earn the researcher's tip.
        """,
    }

    researchers: Dict[str, str] = {
        "role": "user",
        "content": """researcher's JSON:
        {{
            "Purpose": {purpose},
            "data": {table},
        }}

        If you do well, I will give you a $10 tip.
        """,
    }

    Rules: Dict[str, str] = {
        "role": "system",
        "content": """Legal data types:
        - Entity: A noun or term with not exceeding 5 words.
        - Number: Must include a number with units.
        - Brief: A concise description with not exceeding 31 words.
        
        Strictly output the JSON step by step:
        1. Output the header of the researcher's data table in the "headers".
        2. Output the data types corresponding to the header in "data type".
        Continue:
            1. Find data entries with missing information.
            2. Find data entries with illegal data type.
            3. Find data entries that do not meet the researcher's purpose.
            4. Output the indices (Int) of the data entries found above in the "deleted entry".

        JSON format:
        {{
            headers: [header1, header2, ...],
            data type: [column1, column2, ...],
            deleted entry: [0,4,17,...],
        }}

        JSON object: {{
        """,
    }

    template_store: List[Message] = [
        Message(**Identity),
        Message(**researchers),
        Message(**Rules),
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


def wrap_batch(
    content: List[str], len_load: int = 100_000, num_load: int = 500
) -> Iterator[List[str]]:
    content.reverse()
    batch = []

    while content:
        batch.append(content.pop())
        if (len(batch) == num_load) or (sum(map(len, batch)) >= len_load):
            yield batch
            batch.clear()
        else:
            if not content:
                yield batch


if __name__ == "__main__":
    pl.Config.set_tbl_rows(100)

    # custom_ids_path = ".local/PMC.ids.txt"
    # with open(custom_ids_path, "r") as file:
    #     custom_ids = file.readlines()
    custom_query = "(Chlamydomonas reinhardtii) AND (Fatty Acid)"

    custom_headers: List[str] = [
        "Entity: Strain",
        "Number: Fatty Acid Types",
        "Entity: Fatty Acid Yield",
        "Entity: Related Key Methods",
        "Entity: Related Genes",
        "Entity: Related Pathways",
    ]
    strict_headers: List[str] = [
        "Strain",
        "Fatty Acid Types",
        "Fatty Acid Yield",
    ]

    custom_purpose = "Find Fatty acid production yield that generate by Chlamydomonas reinhardtii strain. As well as any related key methods, genes or pathway, if available."

    custom_examples = {
        "data": {
            "entry1": ["C. reinhardtii", "DHA", "3 mg/L", "control pH", "ACLp", "MVA"],
        },
    }

    entrez = EntrezAPIWrapper(retmax=10)

    doc = entrez.load(query=custom_query)

    llm = AzureOpenAIWrapper(
        azure_openai_endpoint=Gconfig.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Gconfig.AZURE_DEPLOYMENT,
        azure_openai_key=Gconfig.AZURE_OPENAI_KEY,
    )

    doc_extract = DocumentBodyExtractPromptTemplate()
    tabular_trim = TabularTrimmingPromptTemplate()

    total_prompt = 0
    total_comp = 0

    report = []

    for i in range(0, len(doc)):
        dfs = []

        if doc[i]["body"] == []:
            continue

        print(doc[i]["article"])

        for batch in wrap_batch(doc[i]["body"]):
            formatted_batch = {f"reference {i}": batch[i] for i in range(len(batch))}

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
                        data=list(res_json["data"].values()), orient="row"
                    )
                    df = df.rename(dict(zip(df.columns, res_json["headers"] + ["ref"])))
                    df = df.with_columns(
                        pl.Series(
                            "reference",
                            [
                                [batch[index] for index in indexes]
                                for indexes in df["ref"]
                            ],
                        )
                    ).drop("ref")
                    print(df)

                    print("\n")

                    dfs.append(df)

            except Exception as e:
                print(e)
                continue

        sum_df: pl.DataFrame = pl.concat(dfs)
        print(sum_df.with_row_count("id"))

        json_df = df_to_json(sum_df.select(strict_headers))
        print(json_df)

        try:
            mes = tabular_trim.generate(
                purpose=custom_purpose,
                table=json_df,
            )

            res = llm.invoke(mes)

            print(res.usage)
            total_comp += res.usage.completion_tokens
            total_prompt += res.usage.prompt_tokens

            res_content = res.choices[0].message.content.strip()
            print(res_content)
            res_json = json.loads(res_content)

            if delete_index := res_json.get("deleted entry", None):
                deleted_index = [int(i) for i in delete_index]
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

            print(filter_df.with_row_count("id"))

            report.append(filter_df.unique(subset=filter_df.columns))
            print(f"total prompt: {total_prompt}")
            print(f"total completions: {total_comp}")

            report_df: pl.DataFrame = pl.concat(report)
            report_df.write_csv("test_df.csv", separator=",")

        except Exception as e:
            print(e)
            continue

    print(report_df)
