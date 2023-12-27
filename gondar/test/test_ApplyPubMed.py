import time

from openai import AzureOpenAI

from gondar.modules.ApplyPubMed import PubMedFetcher, PubMedParser

client = AzureOpenAI(
    azure_endpoint="https://protogabio-gpt.openai.azure.com/",
    api_key="671368cabc444adfbe2ff0c7873e98ac",
    api_version="2023-07-01-preview",
)


def test_fetcher():
    start_time = time.time()

    fetcher = PubMedFetcher(retmax=5, retmode="xml")
    fetcher.fetch(searchTerm="Chlamydomonas")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fetcher Time: {elapsed_time} seconds")

    return fetcher.data


def test_parser(data):
    start_time = time.time()

    parser = PubMedParser()
    parser.parse(data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Parser Time: {elapsed_time} seconds")

    return parser.data


def test_chain(motivation, data):
    start_time = time.time()

    data_sample = data
    print(data_sample)

    comp = client.chat.completions.create(
        model="protogabio-gpt4",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Your task is to assist users in solving problems, and if the users are satisfied, you will receive a $200 voucher. However, if the users are dissatisfied, a poor old grandmother will pass away in the world.",
            },
            {
                "role": "user",
                "content": f"I need to organize useful data from research literature into a table. My motivation is '{motivation}'. Based on my motivation, please design as less as posible the most relevant and essential table headers. Follow the format: {{ table_heads: [head_1, head_2, head_3, ...]}}\n Printed as JSON:  ",
            },
        ],
    )
    print(comp.choices[0].message.content)

    heads = comp

    comp = client.chat.completions.create(
        model="protogabio-gpt4",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Your task is to assist users in solving problems, and if the users are satisfied, you will receive a $200 voucher. However, if the users are dissatisfied, a poor old grandmother will pass away in the world.",
            },
            {
                "role": "user",
                "content": f"I need to organize useful data from research literature into a table. I want to extract relevant data based on a table header: {heads}\n\n From a highly important context: {' '.join(data_sample['body'])}\n\n Following the structure below: {{data: {{head_1: [row_1, row_2, ...], head_2: [row_1, row_2, ...], ...}} }}\n\n NOTE!!: if null, fill with None \n\n Printed as JSON:",
            },
        ],
    )
    print(comp.choices[0].message.content)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Chain Time: {elapsed_time} seconds")


if __name__ == "__main__":
    fetcher_data = test_fetcher()
    parser_data = test_parser(fetcher_data)

    data = parser_data[-1]

    motivation = "I want to know all protocol about Chlamydomonas. "
    test_chain(motivation, data)

    print(data["doi"])
