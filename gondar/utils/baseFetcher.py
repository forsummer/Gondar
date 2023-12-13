from abc import abstractmethod


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
