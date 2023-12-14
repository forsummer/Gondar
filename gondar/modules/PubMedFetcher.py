from inspect import Parameter
from typing import Any, Dict, List

from Bio import Entrez
from bs4 import BeautifulSoup

from gondar.exception import EnviromentError
from gondar.settings import gconfig
from gondar.utils.base import baseFetcher, baseParser


class PubMedFetcher(baseFetcher):
    # Use "pmc" instead of "pubmed" for fetching free full text.
    _DB: str = "pmc"

    _EXTRACT_ID_TAG: str = "Id"

    # Ref to: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    _ESEARCH_OPTIONS: List[Parameter] = [
        Parameter("retstart", kind=Parameter.KEYWORD_ONLY, default=0),
        Parameter("retmax", kind=Parameter.KEYWORD_ONLY, default=20),
        Parameter("retmode", kind=Parameter.KEYWORD_ONLY, default="xml"),
        Parameter("rettype", kind=Parameter.KEYWORD_ONLY, default="uilist"),
        Parameter("sort", kind=Parameter.KEYWORD_ONLY, default="relevance"),
        Parameter("datetype", kind=Parameter.KEYWORD_ONLY, default="pdat"),
        Parameter("reldate", kind=Parameter.KEYWORD_ONLY, default=None),
        Parameter("mindate", kind=Parameter.KEYWORD_ONLY, default=None),
        Parameter("maxdate", kind=Parameter.KEYWORD_ONLY, default=None),
    ]

    _OPTIONS: List[Parameter | None] = _ESEARCH_OPTIONS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch(self, searchTerm: str):
        self._pre_fetch()

        self._fetch(searchTerm)

        self._post_fetch()

    def _pre_fetch(self):
        """
        Prepare Entrez settings.
        """

        Entrez.email: str | None = gconfig.get("EMAIL", None)
        if Entrez.email is None:
            raise EnviromentError("Found no user email in ENVIRON for Entrez API!")

        Entrez.max_tries: int | None = gconfig.get("MAX_RETRY")
        Entrez.sleep_between_tries: int | None = gconfig.get("RETRY_GAP")

    def _fetch(self, searchTerm: str):
        """
        Use Entrez.esearch to collect IDs of target article.
        Then use Entrez.efetch to get the full text of article (default as XML).
        """

        try:
            with Entrez.esearch(
                db=self._DB, term=searchTerm, **self._default_options
            ) as searchHandle:
                searchResults = BeautifulSoup(
                    searchHandle.read(), self._default_options["retmode"]
                )
                id_set = [
                    id_tag.text
                    for id_tag in searchResults.find_all(self._EXTRACT_ID_TAG)
                ]
        except Exception as e:
            raise e

        try:
            with Entrez.efetch(db=self._DB, id=id_set) as fetchHandle:
                self.data = BeautifulSoup(fetchHandle.read(), "xml")
        except Exception as e:
            raise e

    def _post_fetch(self):
        """
        Do nothing after fetching.
        """


class PubMedParser(baseParser):
    def __init__(self) -> None:
        super().__init__()

    def withParser():
        ...

    def parse(self):
        ...

    def _pre_parse(self):
        ...

    def _parse(self):
        ...

    def _post_parse(self):
        ...
