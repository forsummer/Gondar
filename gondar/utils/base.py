import multiprocessing as mp
from abc import abstractmethod
from inspect import Parameter
from typing import Any, Callable, Dict, Iterator, List


class baseModel(object):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        self._default_options: Dict[str, Any] = self.add_default_options()

        # Safely update outsource args
        self.reset_default_options(**kwargs)

        self.data: Any = None

    def add_default_options(cls):
        _options = cls._OPTIONS

        return (
            {arg.name: arg.default for arg in _options} if _options is not None else {}
        )

    def reset_default_options(self, **kwargs):
        updated_options = set(self._default_options.keys()) & set(kwargs.keys())
        if updated_options is not None:
            self._default_options.update(
                {k: v for k, v in kwargs.items() if k in updated_options}
            )


class baseFetcher(baseModel):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data = None

    def fetch(self, searchTerm: Any) -> None:
        self._pre_fetch()

        self._fetch(searchTerm)

        self._post_fetch()

    @abstractmethod
    def _pre_fetch(self):
        """
        The pre-processing before _fetch.
        Usually use for specify some config.
        """

    @abstractmethod
    def _fetch(self):
        """
        The actual fetch processing.
        """

    @abstractmethod
    def _post_fetch(self):
        """
        The post-processing after _fetch.
        """

    async def afetch(self):
        ...

    async def _afetch(self):
        ...


class baseParser(baseModel):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data = None
        self._pipeline: Callable[[Any], Dict] = None

    class assmPipeline:
        def __init__(self, *workers) -> None:
            self.regWorkers = workers

        def __call__(self, source: Any) -> Dict[str, Any]:
            output: Dict = {}

            for worker in self.regWorkers:
                output.update(worker(source))

            return output

    def parse(self, source: Any):
        self._pre_parse()

        self._parse(source)

        self._post_parse()

    @abstractmethod
    def _pre_parse(self):
        ...

    @abstractmethod
    def _parse(self):
        ...

    @abstractmethod
    def _post_parse(self):
        ...

    def _LoopPipeline(self, sourceIter: Iterator[Any]) -> List:
        return [self._pipeline(source) for source in sourceIter]

    def _MpPipeline(self, processes: int, sourceIter: Iterator[Any]) -> List:
        with mp.Pool(processes=processes) as pool:
            return pool.map(self._pipeline, sourceIter)


class basePublisher(baseModel):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    def publish(self):
        ...

    @abstractmethod
    def _pre_publish(self):
        ...

    @abstractmethod
    def _publish(self):
        ...

    @abstractmethod
    def _post_publish(self):
        ...

    async def apublish(self):
        ...

    async def _apublish(self):
        ...
