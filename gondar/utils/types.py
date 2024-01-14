from typing import Literal, Union

from pydantic import Field
from typing_extensions import Annotated

# String
STR = Annotated[str, Field(default=None, kw_only=True)]

# Number
POS_INT = Annotated[int, Field(gt=0, kw_only=True)]
POS_FLOAT = Annotated[float, Field(gt=0, kw_only=True)]
POS_NUM = Annotated[Union[int, float], Field(gt=0, kw_only=True)]

# Bool
DEFAULT_TRUE = Annotated[bool, Field(default=True, kw_only=True)]
DEFAULT_FALSE = Annotated[bool, Field(default=False, kw_only=True)]

# Validation
VALID: Literal["valid"] = "valid"
INVALID: Literal["invalid"] = "invalid"
