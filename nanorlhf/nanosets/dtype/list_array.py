import struct
from typing import Any, Iterable, Optional, TypeVar
from typing import List

from nanorlhf.nanosets.base.bitmap import Bitmap
from nanorlhf.nanosets.base.buffer import Buffer
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import LIST, PrimitiveType
from nanorlhf.nanosets.dtype.primitive_array import infer_primitive_dtype, PrimitiveArrayBuilder
from nanorlhf.nanosets.dtype.string_array import StringArrayBuilder

ChildE = TypeVar("ChildE")


def infer_child_builder(rows: List[Optional[Iterable[Any]]]) -> ArrayBuilder:
    """
    Infer a builder for elements of a ListArray column.

    The function inspects the provided list rows and returns an appropriate builder:
      - Primitive elements (bool/int/float): `PrimitiveArrayBuilder(infer_primitive_dtype(...))`
      - String elements (str): `StringArrayBuilder()`
      - Nested lists: `ListArrayBuilder(infer_child_builder(rewritten_inner_rows))`
      - Dict elements: `StructArrayBuilder(<inferred struct fields>)`

    Args:
      rows (List[Optional[Iterable[Any]]]): A list where each item is an iterable (list/tuple)
          of child elements, or None to represent a null list.

    Returns:
      `Builder`: A builder instance suitable for the list's element type.
    """
    # 1. Find a representative non-null element to choose a branch.
    sample: Any = None
    for r in rows:
        if r is None:
            continue
        for e in r:
            if e is not None:
                sample = e
                break
        if sample is not None:
            break

    if sample is None:
        raise ValueError("Cannot infer element type: all rows are None or empty.")

    # 2. Branch by the sample's type.
    # 2.1. Nested lists
    if isinstance(sample, (list, tuple)):
        # Build "inner rows":
        #     each child list becomes one inner row for the nested ListArrayBuilder.
        inner_rows: List[Optional[Iterable[Any]]] = []
        for r in rows:
            if r is None:
                continue  # null outer list contributes no child lists
            for sub in r:
                if sub is None:
                    inner_rows.append(None)  # null inner list
                elif isinstance(sub, (list, tuple)):
                    inner_rows.append(sub)  # an inner list (possibly empty)
                else:
                    raise TypeError(
                        f"Expected nested list elements, found {type(sub).__name__}"
                    )
        inner_child = infer_child_builder(inner_rows)
        return ListArrayBuilder(inner_child)

    # 2.2. Dict → struct
    if isinstance(sample, dict):
        dict_elems: List[Optional[dict]] = []
        for row in rows:
            if row is None:
                continue
            for elem in row:
                if elem is None:
                    dict_elems.append(None)
                elif isinstance(elem, dict):
                    dict_elems.append(elem)
                else:
                    raise TypeError(f"Mixed element types: expected dict, got {type(elem).__name__}")
        from nanorlhf.nanosets.dtype.struct_array import get_struct_array_builder_from_rows
        return get_struct_array_builder_from_rows(dict_elems)

    # 2.3. Strings
    if isinstance(sample, str):
        # Validate all non-null elements across rows are str
        for r in rows:
            if r is None:
                continue
            for e in r:
                if e is None:
                    continue
                if not isinstance(e, str):
                    raise TypeError(
                        f"Mixed element types: expected str, got {type(e).__name__}"
                    )
        return StringArrayBuilder()

    # 2.4. Primitives (bool/int/float)
    if isinstance(sample, (bool, int, float)):
        # Collect primitive values (and None) and ensure there are no foreign types.
        prims: List[Optional[PrimitiveType]] = []
        for r in rows:
            if r is None:
                continue
            for e in r:
                if e is None:
                    prims.append(None)
                    continue
                if isinstance(e, (bool, int, float)):
                    prims.append(e)
                else:
                    raise TypeError(
                        f"Mixed element types: expected primitive, got {type(e).__name__}"
                    )

        # Decide BOOL / INT64 / FLOAT64
        dt = infer_primitive_dtype(prims)
        return PrimitiveArrayBuilder(dt)

    # 2.5. Otherwise unsupported
    raise TypeError(f"Unsupported element type for list: {type(sample).__name__}")


class ListArray(Array):
    """
    ListArray is an Array that holds variable-length lists of elements.
    It's conceptually similar to StringArray, but instead of strings,
    it holds lists of arbitrary elements.

    Args:
        offsets (np.ndarray): start and end positions boundaries of each list
        values (Array): child array holding the actual list elements
        validity (Optional[Bitmap]): validity bitmap, if None all elements are valid

    Discussion:
        Q. How is this similar to StringArray?
            Both ListArray and StringArray use the same "offsets + values" pattern.

            - StringArray
                - offsets: int32[n+1]
                - values : uint8 buffer holding all UTF-8 bytes
                - element i = bytes[offsets[i] : offsets[i+1]]

            - ListArray
                - offsets: int32[n+1]
                - values : a *child Array* that concatenates all list items
                - element i = child[offsets[i] : offsets[i+1]]  (a sublist)

            Example (ListArray):
                Suppose `values = [10, 11, 12, 13, 14]` and `offsets = [0, 2, 5]`.

                Then:
                  row 0 -> values[0:2] -> [10, 11]
                  row 1 -> values[2:5] -> [12, 13, 14]

            The key difference is that StringArray's `values` is a primitive
            byte buffer (np.uint8) for text, whereas ListArray's `values` is an arbitrary Array.
            It can itself be a PrimitiveArray, StringArray, another ListArray, or a StructArray.
            This makes nested structures like List<List<int32>> or List<Struct{...}> natural.
    """

    def __init__(self, offsets: Buffer, values: Array, validity: Optional[Bitmap] = None):
        if (len(offsets.data) % 4) != 0:
            raise ValueError("offsets buffer length must be a multiple of 4 bytes (int32).")

        num_offsets = len(offsets.data) // 4

        if num_offsets < 1:
            raise ValueError("offsets must contain at least 1 int32 (length n+1).")

        num_lists = num_offsets - 1
        super().__init__(LIST, num_lists, validity)

        self.offsets = offsets
        self.values = values

    def offset_at(self, i: int) -> int:
        """
        Read int32 offset[i] (little-endian) without NumPy.

        Args:
            i (int): index of the offset to read

        Returns:
            int: the i-th offset value
        """
        return struct.unpack_from("<i", self.offsets.data, i * 4)[0]

    def to_pylist(self) -> List[Optional[List]]:
        """
        Convert the ListArray to a Python list of lists, respecting null values.

        Returns:
            List: Python list representation of the array

        Notes:
            This supposes 2D lists. For example, if the ListArray represents
            [1, 2, None, 3, 4, 5] with offsets [0, 2, 2, 5], the output will be:
            [
                [1, 2],
                None,
                [3, 4, 5]
            ]
        """
        output = []
        child_pylist = self.values.to_pylist()

        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                start = int(self.offset_at(i))
                end = int(self.offset_at(i + 1))
                output.append(child_pylist[start:end])
        return output

    @classmethod
    def from_pylist(cls, data: List[Optional[Iterable[Any]]]) -> "ListArray":
        """
        Build a ListArray from a Python list of iterables (or None), with type inference.

        This function infers the element type of the list column and constructs a
        `ListArrayBuilder` with an appropriate child builder:

        - Primitive elements (bool/int/float): `PrimitiveArrayBuilder(infer_primitive_dtype(...))`
        - String elements (str): `StringArrayBuilder()`
        - Nested lists: `ListArrayBuilder(<recursively inferred child builder>)`
        - Dict elements: `StructArrayBuilder(<inferred struct fields>)`

        Args:
            data (List[Optional[Iterable[Any]]]): A list where each item is an iterable (list/tuple)
                of child elements or None to represent a null list.
                Elements may be primitives, strings, or nested lists (arbitrary depth).

        Returns:
            ListArray: Immutable list column with an optional validity bitmap.

        Examples:
            >>> prims = ListArray.from_pylist([[1, 2], None, [3, 4, 5]])
            >>> nested = ListArray.from_pylist([[[1], [2]], None, [[3, 4], []]])
            >>> strings = ListArray.from_pylist([["foo", "bar"], None, ["baz"]])
        """
        child_builder = infer_child_builder(data)
        array_builder = ListArrayBuilder(child_builder)
        for row in data:
            array_builder.append(row)
            # row can be None (null list) or any iterable (including empty)
        return array_builder.finish()


class ListArrayBuilder(ArrayBuilder[Iterable[ChildE], ListArray]):
    """
    ListArrayBuilder incrementally builds a ListArray
    for arbitrary element types by composing a child builder.

    The child builder is responsible for appending elements of the list,
    while this ListArrayBuilder manages list row boundaries (offsets) and nulls (validity bitmap).

    Args:
        child_builder (Builder): builder for the child element type

    Discussion:
        Q. How is this similar to StringArrayBuilder?
            They share the same architectural pattern:
              1) Maintain cumulative offsets of length n+1 starting at 0.
              2) Track per-row validity and pack it into a 1-bit bitmap via _pack_bitmap.
              3) Freeze internal buffers on finish() to produce an immutable Array.

            In short:
                - StringArrayBuilder
                    offsets: int32[n+1] over UTF-8 byte length
                    values : uint8 flat byte buffer of all strings
                    finish : returns StringArray(offsets, values, validity)

                - ListArrayBuilder
                    offsets: int32[n+1] over element counts per list row
                    values : produced by value_builder.finish() — an arbitrary child Array
                    finish : returns ListArray(offsets, child, validity)
    """

    def __init__(self, child_builder: ArrayBuilder):
        self.child_builder = child_builder
        self.offsets = [0]
        self.validity = []

    def append(self, seq: Optional[Iterable[Any]]) -> "ListArrayBuilder":
        """
        Append a single list (or None) to the builder.

        Args:
            seq (Optional[Iterable[Any]]): list to append, or None for null

        Returns:
            ListArrayBuilder: self for method chaining
        """
        if seq is None:
            self.validity.append(0)
            self.offsets.append(self.offsets[-1])
            return self

        self.validity.append(1)
        count = 0
        for x in seq:
            self.child_builder.append(x)
            count += 1
        self.offsets.append(self.offsets[-1] + count)

    def finish(self) -> ListArray:
        """
        Finalize the builder and return the built ListArray.

        Returns:
            ListArray: built ListArray
        """
        num_offsets = len(self.offsets)
        raw_offsets = bytearray(num_offsets * 4)

        # Pack offsets into the bytearray
        byte_offset = 0
        for offset in self.offsets:
            struct.pack_into("<i", raw_offsets, byte_offset, offset)
            byte_offset += 4

        frozen_offsets = bytes(raw_offsets)
        offsets_buffer = Buffer.from_bytes(frozen_offsets)

        # Finalize child array
        child_array = self.child_builder.finish()

        # Validity bitmap (if any nulls)
        validity = Bitmap.from_pylist(self.validity)

        # Create and return ListArray
        return ListArray(offsets_buffer, child_array, validity)
