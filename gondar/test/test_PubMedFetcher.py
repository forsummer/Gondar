from gondar.modules.PubMedFetcher import PubMedFetcher

if __name__ == "__main__":
    fetcher = PubMedFetcher(retmax=5, retmode="xml")
    fetcher.fetch(searchTerm="Chalamydomonas")

    print(fetcher.data)
