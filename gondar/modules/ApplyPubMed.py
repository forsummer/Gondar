from inspect import Parameter
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Iterator, List, Literal

from Bio import Entrez
from bs4 import BeautifulSoup, PageElement, Tag

from gondar.exception import EnviromentError, ModuleError
from gondar.settings import Gconfig
from gondar.utils.base import baseFetcher, baseParser, basePublisher


class PubMedFetcher(baseFetcher):
    """
    Fast bulk fetch interested full text from PMC with Entrez.

    NOTICE:
    Must be provide an email within .env:

    EMAIL=example@mail.com
    """

    # Use "pmc" instead of "pubmed" for fetching free full text.
    _DB: str = "pmc"
    _BS_ENCODING: str = "xml"
    _EXTRACT_ID_TAG: str = "Id"

    # Ref to: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    _ESEARCH_OPTIONS: List[Parameter | None] = [
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
                searchResults = BeautifulSoup(handle.read(), self._BS_ENCODING)

                id_set: List[str] = [
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
        # TODO: May print some fetched example to let user confirm if everything OK.
        """


(
    _ARTICLE_TITLE_TAG,
    _JOURNAL_TITLE_TAG,
    _ARTICLE_ID_TAG,
    _PUBDATE_TAG,
    _SECTION_TAG,
    _TABLE_TAG,
    _TABLE_ROW_TAG,
    _TABLE_HEAD_TAG,
    _TABLE_DATA_TAG,
) = (
    "article-title",
    "journal-title",
    "article-id",
    "pub-date",
    "sec",
    "table",
    "tr",
    "th",
    "td",
)

(
    _BLANK_LINKER,
    _SPACE_LINKER,
    _SLASH_LINKER,
) = (
    "",
    " ",
    "/",
)

_filter_meta: Callable[[Tag | None, str], str] = (
    lambda content, linker: linker.join(content.stripped_strings)
    if content is not None
    else ""
)


def _get_Meta(article: BeautifulSoup) -> Dict[str, str]:
    return {
        "article": _filter_meta(
            article.find(_ARTICLE_TITLE_TAG),
            _BLANK_LINKER,
        ),
        "journal": _filter_meta(
            article.find(_JOURNAL_TITLE_TAG),
            _BLANK_LINKER,
        ),
        "pmcid": _filter_meta(
            article.find(_ARTICLE_ID_TAG, attrs={"pub-id-type": "pmc"}),
            _BLANK_LINKER,
        ),
        "doi": _filter_meta(
            article.find(_ARTICLE_ID_TAG, attrs={"pub-id-type": "doi"}),
            _BLANK_LINKER,
        ),
        "pubdate": _filter_meta(
            article.find(_PUBDATE_TAG, attrs={"pub-type": "epub"}),
            _SLASH_LINKER,
        ),
    }


def _get_Body(article: BeautifulSoup) -> Dict[str, List]:
    sections: Iterator[PageElement] = (
        section for section in article.find_all(_SECTION_TAG)
    )

    section_contents: Iterator[str] = (
        _SPACE_LINKER.join(section.stripped_strings) if section is not None else ""
        for section in sections
    )

    return {
        "body": list(section_contents),
    }


_filter_row: Callable[[Tag | None], List[str]] = lambda row: [
    _SPACE_LINKER.join(content.stripped_strings)
    for content in row.find_all([_TABLE_HEAD_TAG, _TABLE_DATA_TAG])
]


def _filter_table(rows: List[Tag]) -> Dict[str, List[str]]:
    """Make a col-based table is easier than row-base table."""
    table_dict = {}

    for (col_content,) in zip(_filter_row(row) for row in rows):
        table_dict[col_content[0]] = col_content[1:]

    return table_dict


def _get_Tabular(article: BeautifulSoup) -> Dict[str, List]:
    tables: Iterator[PageElement] = (table for table in article.find_all(_TABLE_TAG))

    tabulars: Iterator[Dict[str, List[str]]] = (
        _filter_table(table.find_all(_TABLE_ROW_TAG)) for table in tables
    )

    return {
        "tabular": list(tabulars),
    }


class PubMedParser(baseParser):
    _BS_ENCODING: str = "xml"
    _TAG_ARTICLE: str = "article"

    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data: List = []
        self._pipeline: Callable[[BeautifulSoup], Dict] = None

    def _pre_parse(self) -> None:
        self._max_processor: int = cpu_count()
        self.use_mp: bool = Gconfig.USE_MULTIPROCESSING
        self.processor: int = (
            min(Gconfig.USEABLE_PROCESSOR, self._max_processor)
            if not Gconfig.USE_MAX_PROCESSOR
            else self._max_processor
        )

        self._pipeline: Callable[[BeautifulSoup], Dict] = self.assmPipeline(
            _get_Meta,
            _get_Body,
            _get_Tabular,
        )

    def _parse(self, source: BeautifulSoup) -> None:
        articles: Iterator[BeautifulSoup] = (
            BeautifulSoup(str(article), self._BS_ENCODING)
            for article in source.find_all(self._TAG_ARTICLE)
        )  # Stringify and de-stringify the PageElement is crucial step.

        if self._pipeline is not None:
            if self.use_mp:
                self.data = self._MpPipeline(self.processor, articles)
            else:
                self.data = self._LoopPipeline(articles)
        else:
            raise ModuleError("Found no useable pipeline.")

    def _post_parse(self) -> None:
        """
        Do nothing after parsing.
        # TODO: May print some parsed example to let user confirm if everything OK.
        """


class PubMedPublisher(basePublisher):
    _VALID_PUBLISH_TYPES: List[str] = [
        "csv",  # Human-readable tabular, lower volume fast IO.
        "excel",  # Human-readable tabular, terrible choice.
        "json",  # Json~
        "feather",  # Col-based persistent storage, larger disk usage, faster IO
        "parquet",  # Col-based persistent storage, lower disk usage, lower IO
        "avro",  # Row-based persistent storage, higher efficience for dense writing of amounts of data
    ]

    _OPTIONS: List[Parameter | None] = [
        Parameter("publish_type", kind=Parameter.KEYWORD_ONLY, default="parquet")
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _pre_publish(self) -> None:
        if self._default_options["publish_type"] in self._VALID_PUBLISH_TYPES:
            self._publish_type: str = self._default_options["publish_type"]
        else:
            raise ModuleError(
                f"Invalid publish type of {self._default_options['publish_type']}"
            )

    def _publish(self, source: List) -> None:
        ...

    def _post_publish(self) -> None:
        ...

    @classmethod
    def _publish_dataframe(
        cls,
    ):
        ...
