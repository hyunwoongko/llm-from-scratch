import struct
from typing import List, Optional

from nanorlhf.nanosets.base.bitmap import Bitmap
from nanorlhf.nanosets.base.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import (
    DataType,
    PrimitiveType,
    FMT,
    BOOL,
    INT32,
    INT64,
    INT32_MIN,
    INT32_MAX,
    FLOAT32,
    FLOAT64,
)


def infer_primitive_dtype(vals: List[Optional[PrimitiveType]]) -> DataType:
    """
    Infer the most suitable primitive DataType (BOOL, INT64, FLOAT64)

    Args:
        vals (List[Optional[PrimitiveType]]): List of Python scalars (bool/int/float) or None

    Returns:
        DataType: Inferred DataType (BOOL, INT64, FLOAT64)

    Discussion:
        Q. What is the inference rule?
            - Any float present  -> FLOAT64
            - Else any int       -> INT64
            - Else any bool      -> BOOL
            - Else (all None)    -> ValueError
    """
    saw_float = False
    saw_int = False
    saw_bool = False

    for v in vals:
        if v is None:
            continue

        if isinstance(v, bool):
            saw_bool = True
            continue  # bool is a subclass of int; keep separate

        if isinstance(v, float):
            saw_float = True
        elif isinstance(v, int):
            saw_int = True
        else:
            raise TypeError(f"Unsupported value type: {type(v).__name__}")

    if saw_float:
        return FLOAT64
    if saw_int:
        return INT64
    if saw_bool:
        return BOOL

    raise ValueError("Cannot infer dtype from all-None input.")


class PrimitiveArray(Array):
    """
    PrimitiveArray is an Array that holds primitive fixed-size data types like int32, float64, etc.

    Args:
        dtype (DataType): data type of the array
        values (Buffer): buffer holding the actual data
        length (int): length of the array
        validity (Optional[Bitmap]): validity bitmap, if None all elements are valid

    Examples:
            >>> PrimitiveArray.from_pylist([1, None, 3]).to_pylist()
            [1, None, 3]
            >>> PrimitiveArray.from_pylist([1.0, 2, True]).to_pylist()
            [1.0, 2.0, 1.0]
            >>> PrimitiveArray.from_pylist([True, False, None]).to_pylist()
            [True, False, None]
            >>> PrimitiveArray.from_pylist([1, 2, 3], dtype=INT32).to_pylist()
            [1, 2, 3]
    """

    def __init__(self, dtype: DataType, values: Buffer, length: int, validity: Optional[Bitmap] = None):
        assert dtype in FMT, f"Unsupported data type: {dtype}"
        super().__init__(dtype, length, validity)
        self.values = values
        self.fmt, self.itemsize = FMT[dtype]

        # sanity check
        expected = length * self.itemsize
        if len(self.values.data) != expected:
            raise ValueError(
                f"values size mismatch: expected {expected} bytes, got {len(self.values.data)}"
            )

    def to_pylist(self) -> list:
        """
        Convert the PrimitiveArray to a Python list, respecting null values.

        Returns:
            list: Python list representation of the array
        """
        output = []
        _memoryview = self.values.data

        for i in range(self.length):
            offset = i * self.itemsize
            if self.is_null(i):
                output.append(None)
            else:
                value = struct.unpack_from(self.fmt, _memoryview, offset)[0]
                output.append(value)

        return output

    @classmethod
    def from_pylist(
        cls,
        data: List[Optional[PrimitiveType]],
        dtype: Optional[DataType] = None,
    ) -> "PrimitiveArray":
        """
        Build a PrimitiveArray from a Python list of scalars using PrimitiveArrayBuilder.

        Args:
            data: A list of Python scalars (bool/int/float) or None for nulls.
            dtype: Optional target primitive dtype (BOOL, INT32, INT64, FLOAT32, FLOAT64).
                If omitted, dtype is inferred by `infer_primitive_type(data)`.

        Returns:
            PrimitiveArray: Immutable primitive column with an optional validity bitmap
                (created only if at least one element is null).
        """

        # 1) Resolve target dtype (infer if needed)
        target = dtype if dtype is not None else infer_primitive_dtype(data)
        if target not in FMT:
            raise TypeError(f"Unsupported dtype for PrimitiveArray: {target}")

        # 2) Build using the existing PrimitiveArrayBuilder (values must be pre-validated here)
        b = PrimitiveArrayBuilder(target)

        if target is BOOL:
            for v in data:
                if v is None:
                    b.append(None)
                elif isinstance(v, bool):
                    b.append(v)
                else:
                    raise TypeError("BOOL dtype expects `bool` or `None`.")

        elif target in (INT32, INT64):
            for v in data:
                if v is None:
                    b.append(None)
                    continue
                if isinstance(v, bool):
                    b.append(int(v))
                elif isinstance(v, int):
                    if target is INT32 and not (INT32_MIN <= v <= INT32_MAX):
                        raise OverflowError(f"Value {v} out of int32 range")
                    b.append(v)
                elif isinstance(v, float):
                    raise TypeError(
                        "Float value provided for integer dtype. "
                        "Use FLOAT32/FLOAT64 or omit dtype for inference."
                    )
                else:
                    raise TypeError(
                        f"Integer dtype expects `int`/`bool` or `None`, got {type(v).__name__}"
                    )

        elif target in (FLOAT32, FLOAT64):
            for v in data:
                if v is None:
                    b.append(None)
                elif isinstance(v, (bool, int, float)):
                    b.append(float(v))
                else:
                    raise TypeError(
                        f"Float dtype expects float/int/bool or None, got {type(v).__name__}"
                    )

        else:
            raise TypeError(f"Unhandled dtype: {target}")

        return b.finish()


class PrimitiveArrayBuilder(ArrayBuilder[PrimitiveType, PrimitiveArray]):
    """
    Builder for constructing PrimitiveArray incrementally.

    Args:
        dtype (DataType): data type of the array to build
    """

    def __init__(self, dtype: DataType):
        assert dtype in FMT, f"Unsupported dtype: {dtype}"
        self.dtype = dtype
        self.fmt, self.itemsize = FMT[dtype]
        self.values: List[PrimitiveType] = []
        self.validity: List[int] = []

    def append(self, x: Optional[PrimitiveType]) -> "PrimitiveArrayBuilder":
        """
        Append a value to the builder. Use None for nulls.

        Args:
            x (Optional[Prim]): value to append, or None for null

        Returns:
            PrimitiveArrayBuilder: self for chaining
        """
        if x is None:
            self.validity.append(0)
            if self.dtype is BOOL:
                self.values.append(False)
            else:
                # placeholder for null
                self.values.append(0)
        else:
            self.validity.append(1)
            self.values.append(x)
        return self

    def finish(self) -> PrimitiveArray:
        """
        Finalize the builder and return the built PrimitiveArray.

        Returns:
            PrimitiveArray: built PrimitiveArray

        Discussion:
            Q. Why freeze the `bytearray` with `bytes(...)` first?
                The `bytearray` is mutable, meaning that other objects can modify its content even after
                the `StringArray` is created. Converting it to `bytes` creates an immutable copy of the data,
                ensuring the buffer will never change once finalized.

                This prevents aliasing issues — a situation where multiple variables reference the same
                memory block. If the data remained mutable, modifying one reference could unintentionally
                alter the data seen by others.

                For example:
                    >>> buf = bytearray(b"hello")
                    >>> view = memoryview(buf)
                    >>> buf[0] = ord('H')
                    >>> print(view.tobytes())  # b'Hello' — `view` changed too!

                Here, both `buf` and `view` share the same underlying memory.
                When `buf` is mutated, `view` reflects that change automatically.

                By freezing the data:
                    >>> buf = bytearray(b"hello")
                    >>> frozen = bytes(buf)
                    >>> buf[0] = ord('H')
                    >>> print(frozen)  # b'hello' — safely unchanged

                This guarantees immutability, ensuring that once a `StringArray` is finalized,
                internal data remains immutable and protected from accidental modification through shared references.

        """
        num_items = len(self.values)
        total_bytes = num_items * self.itemsize
        raw_buffer = bytearray(total_bytes)

        # Pack values into the bytearray
        offset = 0
        for value in self.values:
            struct.pack_into(self.fmt, raw_buffer, offset, value)
            offset += self.itemsize

        # Create the Buffer
        frozen_values = bytes(raw_buffer)
        values_buffer = Buffer.from_bytes(frozen_values)

        # Validity bitmap (if any nulls)
        validity = Bitmap.from_pylist(self.validity)

        # Create and return the PrimitiveArray
        return PrimitiveArray(self.dtype, values_buffer, num_items, validity)


