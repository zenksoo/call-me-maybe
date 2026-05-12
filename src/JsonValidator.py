from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any


class TestCaseSchema(BaseModel):
    prompt: str = Field(min_length=1)


class FunctionDefSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]
