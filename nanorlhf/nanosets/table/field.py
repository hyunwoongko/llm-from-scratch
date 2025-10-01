from dataclasses import dataclass

from nanorlhf.nanosets.dtype.dtype import DataType


@dataclass(frozen=True)
class Field:
    """
    A field represents a single column definition in a schema.
    This only describes metadata, not store actual data.

    Args:
        name (str): name of column
        dtype (DataType): data type of the column
        nullable (bool): whether the column can contain null values

    Examples:
        >>> from nanorlhf.nanosets.dtype.dtype import INT32
        >>> Field("age", INT32, False)
        Field(name='age', dtype=DataType(name='int32'), nullable=False)

    Notes:
        `frozen=True` makes the Field instances immutable,
        so once created, their attributes cannot be changed.
    """

    name: str
    dtype: DataType
    nullable: bool = True
