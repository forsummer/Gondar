import datetime


class _DefaultErrorMessage(dict):
    """
    Default error messages.
    """

    error_message: dict[str, str] = {
        "BaseError": "Occur exception",
        "EnviromentError": "Enviroment exception",
    }

    def __getitem__(self, __key: str) -> str:
        return self.error_message[__key]


class BaseError(Exception):
    """
    As a base class for all exception types.
    """

    def __init__(self, msg: str = None) -> None:
        if msg == None:
            self.__msg = _DefaultErrorMessage[type(self)]
        else:
            self.__msg = msg

    def __str__(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return "[{}] [{}]: {}".format(current_time, type(self), self.__msg)

    @property
    def msg(self) -> str:
        return self.__msg


class EnviromentError(BaseError):
    """
    Enviroment exception.
    """

    def __init__(self, msg: str = None) -> None:
        super().__init__(msg)
