import multiprocessing as mp
from inspect import Parameter
from typing import Iterator, List

from Bio import Entrez
from bs4 import BeautifulSoup, PageElement

from gondar.exception import EnviromentError
from gondar.settings import Gconfig
from gondar.utils.base import baseFetcher, baseParser


class PubMedFetcher(baseFetcher):
    """
    Fast bulk fetch interested full text from PMC with Entrez.

    NOTICE:
    Must be provide an email within .env:

    EMAIL=example@mail.com
    """

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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fetch(self, searchTerm: str) -> None:
        self._pre_fetch()

        self._fetch(searchTerm)

        self._post_fetch()

    def _pre_fetch(self) -> None:
        """
        Prepare Entrez settings.
        """

        Entrez.email: str | None = Gconfig.EMAIL
        if Entrez.email is None:
            raise EnviromentError("Found no user email in ENVIRON for Entrez API!")

        Entrez.max_tries: int | None = Gconfig.MAX_RETRY
        Entrez.sleep_between_tries: int | None = Gconfig.RETRY_GAP

    def _fetch(self, searchTerm: str) -> None:
        """
        Use Entrez.esearch to collect IDs of target article.
        Then use Entrez.efetch to get the full text of article (default as XML).
        """

        # Esearch for related pmc id
        try:
            with Entrez.esearch(
                db=self._DB, term=searchTerm, **self._default_options
            ) as handle:
                searchResults = BeautifulSoup(
                    handle.read(), self._default_options["retmode"]
                )

                id_set = [
                    id_tag.text
                    for id_tag in searchResults.find_all(self._EXTRACT_ID_TAG)
                ]

        except Exception as e:
            raise e

        # Efetch for full text
        try:
            with Entrez.efetch(db=self._DB, id=id_set) as handle:
                self.data = BeautifulSoup(
                    handle.read(), self._default_options["retmode"]
                )

        except Exception as e:
            raise e

    def _post_fetch(self) -> None:
        """
        Do nothing after fetching.
        """


class PubMedParser(baseParser):
    _OPTIONS: List[Parameter | None] = [
        Parameter("bsEncoding", kind=Parameter.KEYWORD_ONLY, default="xml"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = []

    def parse(self, data: BeautifulSoup) -> None:
        self._pre_parse()

        self._parse(data)

        self._post_parse()

    def _pre_parse(self) -> None:
        self.use_mp: bool = Gconfig.USE_MULTIPROCESSING
        self.processor: int = int(Gconfig.USEABLE_PROCESSOR)

        if self.use_mp and Gconfig.USE_MAX_PROCESSOR:
            self.processor: int = mp.cpu_count()

    def _parse(self, data: BeautifulSoup) -> None:
        articles: Iterator[PageElement] = (
            BeautifulSoup(str(a), self._default_options["bsEncoding"])
            for a in data.find_all("article")
        )  # This step is crucial. Due to PageElement is un-serializable.

        if self.use_mp:
            self._MpPipeline(articles)

        else:
            self._LoopPipeline(articles)

    def _post_parse(self) -> None:
        """
        Do nothing after parsing.
        """

    def _pipeline(self, article: PageElement) -> None:
        ...

    def _LoopPipeline(self, articles: Iterator) -> None:
        for a in articles:
            self._pipeline(a)

    def _MpPipeline(self, articles: Iterator) -> None:
        with mp.Pool(processes=self.processor) as pool:
            pool.map(self._pipeline, articles)
