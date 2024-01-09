from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Iterator, List, Literal

from Bio import Entrez
from bs4 import BeautifulSoup
from openai import AzureOpenAI
from pydantic import AfterValidator, Field
from typing_extensions import Annotated

from gondar.exception import ConfigError, ModuleError
from gondar.settings import Gconfig
from gondar.utils import (
    POS_INT,
    STR,
    VALID_CHOICES,
    BaseChain,
    BaseFetcher,
    BaseParser,
    BasePublisher,
    GondarConfigModel,
)


def removeAllAttrs(soup: BeautifulSoup):
    """
    Recursively remove all attributes. \n
    This step is crucial for saving tokens usage (and your dollars).
    """
    if hasattr(soup, "attrs"):
        soup.attrs = {}
    if hasattr(soup, "contents"):
        for child in soup.contents:
            removeAllAttrs(child)


class PubMedFetcher(BaseFetcher):
    """
    Fast bulk fetch interested full text from PMC with Entrez.

    NOTICE:\n
    Must be provide an email:\n
    EMAIL=example@mail.com
    """

    DB: STR = "pmc"
    BS_ENCODING: STR = "xml"
    EXTRACT_ID_TAG: STR = "Id"

    class Options(GondarConfigModel):
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


_ARTICLE_TITLE_TAG: Literal["article-title"] = "article-title"
_JOURNAL_TITLE_TAG: Literal["journal-title"] = "journal-title"
_ARTICLE_ID_TAG: Literal["article-id"] = "article-id"
_PUBDATE_TAG: Literal["pub-date"] = "pub-date"
_SECTION_TAG: Literal["sec"] = "sec"
_TABLE_TAG: Literal["table-wrap"] = "table-wrap"

_BLANK_LINKER: Literal[""] = ""
_SPACE_LINKER: Literal[" "] = " "
_SLASH_LINKER: Literal["/"] = "/"


_filter_meta: Callable[[BeautifulSoup | None, str], str] = (
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
    sections: Iterator[BeautifulSoup] = (
        section for section in article.find_all(_SECTION_TAG)
    )

    section_contents: Iterator[str] = (
        _SPACE_LINKER.join(section.stripped_strings) if section is not None else ""
        for section in sections
    )

    return {
        "body": list(section_contents),
    }


def _get_Tables(article: BeautifulSoup) -> Dict[str, List]:
    tables: Iterator[BeautifulSoup] = (table for table in article.find_all(_TABLE_TAG))

    unwraped_tables: List[str] = []
    for t in tables:
        removeAllAttrs(t)
        unwraped_tables.append(str(t))

    return {
        "tables": unwraped_tables,
    }


class PubMedParser(BaseParser):
    BS_ENCODING: STR = "xml"
    TAG_ARTICLE: STR = "article"

    class Options(GondarConfigModel):
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


class PubMedChain(BaseChain):
    class Options(GondarConfigModel):
        llm: Annotated[
            STR,
            AfterValidator(
                VALID_CHOICES(
                    "azure",
                )
            ),
            Field(default="azure"),
        ]
        embedding: Annotated[
            STR,
            AfterValidator(
                VALID_CHOICES(
                    "azure",
                )
            ),
            Field(default=None),
        ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data: List = []

    def _pre_run(self):
        self.client = AzureOpenAI(
            azure_endpoint=...,
            azure_deployment=...,
            api_version=...,
            api_key=...,
            timeout=...,
            max_retries=...,
        )

    def _run(self, target: str, source: List[Dict]):
        self.table_heads = self._chain1(target)

        self.data = map(self._chain2, source)

    def _post_run(self):
        ...

    def _chain1(self, target: str):
        client = AzureOpenAI(
            azure_endpoint=...,
            azure_deployment=...,
            api_version=...,
            api_key=...,
            timeout=...,
            max_retries=...,
        )
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": ...,
                },
                {
                    "role": "user",
                    "content": "How do I output all files in a directory using Python?",
                },
            ],
        )

        return completion

    def _chain2(self, source: Dict):
        heads = self.table_heads
        completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": ...,
                },
                {
                    "role": "user",
                    "content": "How do I output all files in a directory using Python?",
                },
            ],
        )
