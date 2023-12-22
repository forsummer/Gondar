import time

from gondar.modules.ApplyPubMed import PubMedFetcher, PubMedParser, PubMedPublisher


def test_fetcher():
    start_time = time.time()

    fetcher = PubMedFetcher(retmax=10, retmode="xml")
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


def test_publisher(data):
    start_time = time.time()

    publisher = PubMedPublisher()
    publisher.publish(data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Publisher Time: {elapsed_time} seconds")

    return publisher.data


if __name__ == "__main__":
    fetcher_data = test_fetcher()
    parser_data = test_parser(fetcher_data)
    publisher_data = test_publisher(parser_data)

    print(publisher_data)
