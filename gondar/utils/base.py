import multiprocessing as mp
from abc import abstractmethod
from inspect import stack as InspectStack
from typing import Any, Callable, Dict, Iterator, List

from pydantic import BaseModel

from gondar.exception import ConfigError


class GondarPydanticModel(BaseModel):
    class Config:
        validate_assignment = True

    def __setattr__(self, __name: str, __value: Any) -> None:
        calling_frame = InspectStack()[1]
        if calling_frame.function == "__init__":
            super().__setattr__(__name, __value)
        else:
            raise ConfigError("Gondar safe config is not allow to be modify.")


class BaseGondarModel(object):
    class Options(GondarPydanticModel):
        ...

    def __init__(self, **kwargs) -> None:
        self.OPT = self.__set_options(**kwargs)

        self.data: Any = None

    def __set_options(cls, **kwargs):
        return cls.Options(**kwargs)


class BaseFetcher(BaseGondarModel):
    class Options(GondarPydanticModel):
        ...

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
    def _fetch(self, searchTerm: Any):
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


class BaseParser(BaseGondarModel):
    class Options(GondarPydanticModel):
        ...

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
    def _parse(self, source: Any):
        ...

    @abstractmethod
    def _post_parse(self):
        ...

    def _LoopPipeline(self, sourceIter: Iterator[Any]) -> List:
        return [self._pipeline(source) for source in sourceIter]

    def _MpPipeline(self, processes: int, sourceIter: Iterator[Any]) -> List:
        with mp.Pool(processes=processes) as pool:
            return pool.map(self._pipeline, sourceIter)


class BasePublisher(BaseGondarModel):
    class Options(GondarPydanticModel):
        ...

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data = None

    def publish(self, source: Any):
        self._pre_publish()

        self._publish(source)

        self._post_publish()

    @abstractmethod
    def _pre_publish(self):
        ...

    @abstractmethod
    def _publish(self, source: Any):
        ...

    @abstractmethod
    def _post_publish(self):
        ...

    async def apublish(self):
        ...

    async def _apublish(self):
        ...
