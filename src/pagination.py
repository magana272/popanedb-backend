"""
Global pagination utilities
"""

from typing import TypeVar, Generic, List
from pydantic import BaseModel

from src.config import settings

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Standard pagination parameters"""
    limit: int = settings.DEFAULT_LIMIT
    offset: int = 0

    def __init__(self, limit: int = settings.DEFAULT_LIMIT, offset: int = 0):
        # Enforce max limit
        if limit > settings.MAX_LIMIT:
            limit = settings.MAX_LIMIT
        super().__init__(limit=limit, offset=offset)


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper"""
    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool

    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        limit: int,
        offset: int
    ) -> "PaginatedResponse[T]":
        return cls(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + len(items)) < total
        )
