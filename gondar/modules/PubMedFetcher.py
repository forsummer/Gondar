import logging
import os
from inspect import Parameter
from typing import Any, Dict, List

from Bio import Entrez
from bs4 import BeautifulSoup

from gondar.exception import EnviromentError, NetIOError
from gondar.utils.baseFetcher import baseFetcher


class PubMedFetcher(baseFetcher):
    __DB: str = "pubmed"

    __EXTRACT_ID_TAG: str = "Id"

    __ESEARCH_OPTIONS: List[Parameter] = [
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

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self._default_options: Dict[str, Any] = {
            arg.name: arg.default for arg in self.__ESEARCH_OPTIONS
        }
        self.reset_default_options(**kwargs)

        self.data = None

    def reset_default_options(self, **kwargs) -> None:
        exist_params = set(self._default_options.keys()) & set(kwargs.keys())
        if exist_params is not None:
            self._default_options.update(
                {k: v for k, v in kwargs.items() if k in exist_params}
            )

    def fetch(self, searchTerm: str):
        self._pre_fetch()

        self._fetch(searchTerm)

        self._post_fetch()

    def _pre_fetch(self) -> None:
        """
        Prepare Entrez settings.
        """

        Entrez.email: str | None = os.environ.get("EMAIL", None)
        if Entrez.email is None:
            e = EnviromentError("Found no user email in ENVIRON for Entrez API!")
            logging.error(e)
            raise e

        Entrez.max_tries: int | None = os.environ.get("MAX_RETRY")
        Entrez.sleep_between_tries: int | None = os.environ.get("RETRY_GAP")

    def _fetch(self, searchTerm: str) -> None:
        """
        Use Entrez.esearch to collect IDs of target article.
        Then use Entrez.efetch to get the full text of article (default as XML).
        """

        try:
            with Entrez.esearch(
                db=self.__DB, term=searchTerm, **self._default_options
            ) as searchHandle:
                searchResults = BeautifulSoup(searchHandle.read(), self._MODE)
                id_set = [
                    id_tag.text
                    for id_tag in searchResults.find_all(self.__EXTRACT_ID_TAG)
                ]
        except NetIOError("Entrez search failed") as e:
            logging.error(e)
            raise e

        try:
            with Entrez.efetch(
                db=self.__DB, id=id_set, **self._default_options
            ) as fetchHandle:
                self.data = fetchHandle.read()
        except NetIOError("Entrez fetch failed") as e:
            logging.error(e)
            raise e

    def _post_fetch(self) -> None:
        """
        Do nothing after fetching.
        """
