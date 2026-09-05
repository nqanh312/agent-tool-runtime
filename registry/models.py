"""Define the metadata required to register an agent tool."""

from typing import Callable
from pydantic import BaseModel


class ToolDefinition(BaseModel):
    """Describe a callable tool and its access requirements."""
    name: str
    description: str
    input_schema: dict
    required_permissions: list[str]
    handler: Callable
    model_visible: bool = True

    class Config:
        arbitrary_types_allowed = True
