from typing import Union

from pydantic import Field
from typing_extensions import Annotated

# String
STR = Annotated[str, Field(kw_only=True)]

# Number
POS_INT = Annotated[int, Field(gt=0, kw_only=True)]
POS_NUM = Annotated[Union[int, float], Field(gt=0, kw_only=True)]

# Bool
DEFAULT_TRUE = Annotated[bool, Field(default=True, kw_only=True)]
DEFAULT_FALSE = Annotated[bool, Field(default=False, kw_only=True)]
