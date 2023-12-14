import functools
from abc import abstractmethod
from inspect import Parameter
from typing import Any, Dict, List, Set, Type


class baseConfig(object):
    @classmethod
    def to_dict(cls):
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        }


class baseModel(object):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        self.data = None

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


class baseFetcher(baseModel):
    _OPTIONS: List[Parameter | None] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data = None

    def get_data(self):
        return self.data

    @property
    def dtype(self) -> Type:
        return type(self.data)

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

    @classmethod
    def withParser(cls):
        @functools.wraps()
        def wrapper():
            return

        return wrapper

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
