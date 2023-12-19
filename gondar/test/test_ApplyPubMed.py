from gondar.modules.ApplyPubMed import PubMedFetcher, PubMedParser


def test_fetcher():
    fetcher = PubMedFetcher(retmax=5, retmode="xml")
    fetcher.fetch(searchTerm="Chalamydomonas")

    return fetcher.data


def test_parser(data):
    parser = PubMedParser()

    parser.parse(data)

    return parser.data


if __name__ == "__main__":
    data = test_fetcher()

    data = test_parser(data)

    print(data)
