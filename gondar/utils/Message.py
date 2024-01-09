from typing import Annotated, Any, Dict, List, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from gondar.utils import STR


class Message(BaseModel):
    role: STR
    content: STR


class Responses(BaseModel):
    response: Dict
