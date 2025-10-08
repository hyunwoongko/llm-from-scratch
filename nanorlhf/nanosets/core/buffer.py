from dataclasses import dataclass


@dataclass
class Buffer:
    """
    A simple buffer class that wraps a `memoryview`, for zero-copy data handling.

    Discussion:
        Q. What is zero-copy?
            Typically, when passing or converting data to another variable,
            Python often copies the data to a new memory location.

            For example:
            >>> a = b"hello world"
            >>> b = a[:]  # This creates a copy of `a` in a new memory location.

            In this case, `b` contains exactly the same content as `a`,
            but different copies exist in memory, so a and b are in different locations.

            But `view` doesn't copy the data, it provides a view of the original data.
            So it shares the same memory location as the original data.

            >>> data = bytearray(b"hello world")
            >>> view = memoryview(data)  # No copy!

        Q. Why should we use `memoryview` instead of `bytes` or `bytearray`?
            `bytes` is immutable, so any modification creates a new copy.
            `bytearray` is mutable, but slicing it creates a new copy, so it cannot provide a true zero-copy view.

            In contrast, `memoryview` allows direct access and modification of the data without copying,
            making it ideal for efficient zero-copy data handling.

            So we can modify the original data through the view like this:
            >>> view[0] = ord('H')
            >>> print(data == bytearray(b"Hello world"))  # True

        Q. Why is this important for Arrow-like implementation?
            Libraries like Arrow aim to minimize unnecessary data copies for speed and memory efficiency.
            By using zero-copy techniques, they can handle large datasets more efficiently.
    """

    # `memoryview` is a built-in Python class that provides
    # a view of the memory of another binary object (like bytes, bytearray, etc.) without copying it.
    data: memoryview

    def __len__(self):
        """
        Get the length of the buffer

        Returns:
            int: length of the buffer
        """
        return len(self.data)

    @classmethod
    def from_bytes(cls, b: bytes) -> "Buffer":
        """
        Create a Buffer from bytes.

        Args:
            b (bytes): input bytes

        Returns:
            Buffer: a Buffer instance wrapping the bytes
        """
        return cls(memoryview(b))

    @classmethod
    def from_bytearray(cls, b: bytearray) -> "Buffer":
        """
        Create a Buffer from bytearray.

        Args:
            b (bytearray): input bytearray

        Returns:
            Buffer: a Buffer instance wrapping the bytearray
        """
        return cls(memoryview(b))
