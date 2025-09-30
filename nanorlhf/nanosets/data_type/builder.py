from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from nanorlhf.nanosets.data_type.array import Array

# element type
E = TypeVar('E')

# array type
# - bound=Array: `A` must be a subclass of Array
# - covariant=True: `A` can be a more derived type than specified
A = TypeVar('A', bound=Array, covariant=True)


class Builder(Generic[E, A], ABC):
    """
    Abstract base class for builder classes.
    When we build an arbitrary array, we usually don't know the length in advance.
    So we use a builder to incrementally build the array.
    """

    @abstractmethod
    def append(self, value: E) -> "Builder[E, A]":
        """
        Append a single value to the builder.

        Args:
            value (E): value to append

        Returns:
            Builder: self for method chaining
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
