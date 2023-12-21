from multiprocessing import cpu_count
from typing import Callable, Dict, Iterator, List

from Bio import Entrez
from bs4 import BeautifulSoup, PageElement, Tag
from pydantic import AfterValidator, Field
from typing_extensions import Annotated

from gondar.exception import ConfigError, ModuleError
from gondar.settings import Gconfig
from gondar.utils import (
    POS_INT,
    STR,
    VALID_CHOICES,
    BaseFetcher,
    BaseParser,
    BasePublisher,
    GondarPydanticModel,
)


class PubMedFetcher(BaseFetcher):
    """
    Fast bulk fetch interested full text from PMC with Entrez.

    NOTICE:\n
    Must be provide an email:\n
    EMAIL=example@mail.com
    """

    # Use "pmc" instead of "pubmed" for fetching free full text.
    DB: STR = "pmc"
    BS_ENCODING: STR = "xml"
    EXTRACT_ID_TAG: STR = "Id"

    class Options(GondarPydanticModel):
        # Ref to: https://www.ncbi.nlm.nih.gov/books/NBK25499/
        restart: Annotated[POS_INT, Field(default=0)]
        retmax: Annotated[POS_INT, Field(default=20)]
        retmode: Annotated[STR, Field(default="xml")]
        rettype: Annotated[STR, Field(default="uilist")]
        sort: Annotated[STR, Field(default="relevance")]
        datetype: Annotated[STR, Field(default="pdat")]
        reldate: STR
        mindate: STR
        maxdate: STR

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _pre_fetch(self) -> None:
        """
        Prepare Entrez settings.
        """

        Entrez.email: str | None = Gconfig.EMAIL
        if Entrez.email is None:
            raise ConfigError(
                "Found no user email in Global configuration for Entrez API!"
            )

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
                db=self.DB, term=searchTerm, **dict(self.OPT)
            ) as handle:
                searchResults = BeautifulSoup(handle.read(), self.BS_ENCODING)

                id_set: List[str] = [
                    id_tag.text
                    for id_tag in searchResults.find_all(self.EXTRACT_ID_TAG)
                ]
        except Exception as e:
            raise e

        # Efetch for full text
        try:
            with Entrez.efetch(db=self.DB, id=id_set) as handle:
                self.data = BeautifulSoup(handle.read(), self.OPT.retmode)
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


def _get_Tables(article: BeautifulSoup) -> Dict[str, List]:
    tables: Iterator[PageElement] = (table for table in article.find_all(_TABLE_TAG))

    f_tables: Iterator[Dict[str, List[str]]] = (
        _filter_table(table.find_all(_TABLE_ROW_TAG)) for table in tables
    )

    return {
        "tables": list(f_tables),
    }


class PubMedParser(BaseParser):
    BS_ENCODING: STR = "xml"
    TAG_ARTICLE: STR = "article"

    class Options(GondarPydanticModel):
        ...

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
            _get_Tables,
        )

    def _parse(self, source: BeautifulSoup) -> None:
        articles: Iterator[BeautifulSoup] = (
            BeautifulSoup(str(article), self.BS_ENCODING)
            for article in source.find_all(self.TAG_ARTICLE)
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


class PubMedPublisher(BasePublisher):
    class Options(GondarPydanticModel):
        PUBLISTH_TYPE: Annotated[
            STR,
            AfterValidator(
                VALID_CHOICES(["csv", "excel", "json", "feather", "parquet", "avro"])
            ),
            Field(default="csv"),
        ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _pre_publish(self) -> None:
        if self.OPTIONS["publish_type"] in self._VALID_PUBLISH_TYPES:
            self._publish_type: str = self.OPTIONS["publish_type"]
        else:
            raise ModuleError(f"Invalid publish type of {self.OPTIONS['publish_type']}")

    def _publish(self, source: List) -> None:
        ...

    def _post_publish(self) -> None:
        ...

    @classmethod
    def _publish_dataframe(
        cls,
    ):
        ...
