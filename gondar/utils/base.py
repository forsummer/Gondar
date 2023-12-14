import functools
import shelve
from abc import abstractmethod
from inspect import Parameter
from typing import Any, Dict, List, Set, Type

from gondar.settings import Gconfig


class baseModel(object):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        self._default_options: Dict[str, Any] = self.add_default_options()

        # Safely update outsource args
        self.reset_default_options(**kwargs)

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

    def save_checkpoint(self, key, value, save_path: str | None = None):
        save_path: str = save_path or (
            Gconfig["CACHE_DIRECTORY"] + Gconfig["SHELVE_NAME"]
        )

        with shelve.open(save_path, "c") as shelf:
            shelf[key] = value


class baseFetcher(baseModel):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data = None

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


class baseParser(baseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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


class basePublisher(baseModel):
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


class basePipe(baseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
