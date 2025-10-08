from abc import ABC, abstractmethod
from typing import Optional
from typing import TypeVar, Generic

from nanorlhf.nanosets.core.bitmap import Bitmap
from nanorlhf.nanosets.dtype.dtype import DataType


class Array(ABC):
    """
    Abstract base class for all array types.

    Args:
        dtype (DataType): data type of the array
        length (int): length of the array
        validity (Optional[Bitmap]): validity bitmap, if None all elements are valid
    """

    def __init__(self, dtype: DataType, length: int, validity: Optional[Bitmap] = None):
        self.dtype = dtype
        self.length = length
        self.validity = validity

    def __len__(self):
        """
        Get the length of the array.

        Returns:
            int: length of the array
        """
        return self.length

    def is_null(self, i: int) -> bool:
        """
        Check if the i-th element is null.

        Args:
            i (int): index of the element

        Returns:
            bool: True if the element is null, False otherwise
        """
        if self.validity is None:
            return False
        return not self.validity.is_valid(i)

    @abstractmethod
    def to_pylist(self) -> list:
        """
        Convert the array to a Python list, respecting null values.

        Returns:
            list: Python list representation of the array
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pylist(cls, data: list) -> "Array":
        """
        Create an Array from a Python list.

        Args:
            data (list): Python list to convert

        Returns:
            Array: Converted Array
        """
        raise NotImplementedError


# element type
E = TypeVar('E')

# array type
# - bound=Array: `A` must be a subclass of Array
# - covariant=True: `A` can be a more derived type than specified
A = TypeVar('A', bound=Array, covariant=True)


class ArrayBuilder(Generic[E, A], ABC):
    """
    Abstract base class for all array builder classes.
    When we build an arbitrary array, we usually don't know the length in advance.
    So we use a builder to incrementally build the array.
    """

    @abstractmethod
    def append(self, value: E) -> "ArrayBuilder[E, A]":
        """
        Append a single value to the builder.

        Args:
            value (E): value to append

        Returns:
            ArrayBuilder: self for method chaining
        """
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> A:
        """
        Finalize the builder and return the built array.

        Returns:
            A: built array
        """
        raise NotImplementedError
