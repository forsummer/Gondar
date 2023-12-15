from abc import abstractmethod
from inspect import Parameter
from typing import Any, Dict, List, Set


class baseModel(object):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        self._default_options: Dict[str, Any] = self.add_default_options()

        # Safely update outsource args
        self.reset_default_options(**kwargs)

        self.data = None

    def add_default_options(cls):
        _options: List[Parameter | None] = cls._OPTIONS

        return (
            {arg.name: arg.default for arg in _options} if _options is not None else {}
        )

    def reset_default_options(self, **kwargs):
        updated_options: Set[str] = set(self._default_options.keys()) & set(
            kwargs.keys()
        )
        if updated_options is not None:
            self._default_options.update(
                {k: v for k, v in kwargs.items() if k in updated_options}
            )


class baseFetcher(baseModel):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_data(self):
        return self.data

    @abstractmethod
    def fetch(self):
        """
        The full fetch progress.
        """

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

        self.workflow = None

    @abstractmethod
    def parse(self):
        ...

    @abstractmethod
    def _pre_parse(self):
        ...

    @abstractmethod
    def _parse(self):
        ...

    @abstractmethod
    def _post_parse(self):
        ...

    @abstractmethod
    def _pipeline(self):
        ...

    def _LoopPipeline(self):
        ...

    def _MpPipeline(self):
        ...


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
