from abc import abstractmethod


class baseConfig(object):
    @classmethod
    def to_dict(cls):
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        }


class baseFetcher(object):
    def __init__(self) -> None:
        ...

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


class baseParser(object):
    ...
