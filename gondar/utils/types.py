from typing import Any, Union

from pydantic import Field
from typing_extensions import Annotated

from gondar.exception import ModuleError

# String
STR = Annotated[str, Field(default=None, kw_only=True)]

# Number
POS_INT = Annotated[int, Field(gt=0, kw_only=True)]
POS_NUM = Annotated[Union[int, float], Field(gt=0, kw_only=True)]

# Bool
DEFAULT_TRUE = Annotated[bool, Field(default=True, kw_only=True)]
DEFAULT_FALSE = Annotated[bool, Field(default=False, kw_only=True)]


# Validate choice
class VALID_CHOICES:
    def __init__(self, *choices) -> None:
        self._choices = list(choices)

    def __call__(self, choice: Any) -> Any:
        if choice in self._choices:
            return choice
        else:
            raise ModuleError(f"{choice} is not in optional choices: {self._choices}")
