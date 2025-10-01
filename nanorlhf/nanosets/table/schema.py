from dataclasses import dataclass
from typing import Tuple

from nanorlhf.nanosets.table.field import Field


@dataclass(frozen=True)
class Schema:
    """
    A Schema defines the structure (fields) of a Table or RecordBatch.
    This ensures that all RecordBatches and Tables using it have consistent column names, types, and nullability.

    Args:
        fields (tuple[Field, ...]): A tuple of Field objects describing each column.

    Example:
        >>> from nanorlhf.nanosets.dtype.dtype import INT32, DataType
        >>> schema = Schema((
        ...     Field("id", INT32, nullable=False),
        ...     Field("name", DataType("string")),
        ... ))
        >>> schema.names()
        ['id', 'name']
        >>> schema.index("name")
        1
    """

    fields: Tuple[Field, ...]

    def names(self):
        """Return a list of field names in order."""
        return [f.name for f in self.fields]

    def index(self, name: str) -> int:
        """Return the index of the field with the given name."""
        return self.names().index(name)
