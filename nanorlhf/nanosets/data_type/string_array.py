import struct
from typing import Optional, List

from nanorlhf.nanosets.base.bitmap import Bitmap
from nanorlhf.nanosets.base.buffer import Buffer
from nanorlhf.nanosets.data_type.array import Array
from nanorlhf.nanosets.data_type.builder import Builder
from nanorlhf.nanosets.data_type.data_type import STRING


class StringArray(Array):
    """
    StringArray is an Array that holds variable-length UTF-8 strings.

    Args:
        offsets (np.ndarray): start and end positions boundaries of each string
        values (np.ndarray): concatenated string data as uint8 bytes
        validity (Optional[Bitmap]): validity bitmap, if None all elements are valid

    Discussion:
        Q. Why offsets + values?
            Strings are variable-length, which means each string can have a different number of bytes.
            Instead of storing each string separately (like Python lists do), Arrow stores all string bytes
            together in a single continuous memory block called `values`, and keeps another array called
            `offsets` that marks where each string starts and ends.

            ----------------------------------------------------------------
            Python list representation (pointer-based)
            ----------------------------------------------------------------
            >>> data = ["hello", "data", "science"]

            Internally, Python stores object pointers (PyObject*) to each string:

            | Index | Value    | Actual Storage Type         |
            |-------|----------|-----------------------------|
            | 0     | "hello"  | PyObject* (points to str)   |
            | 1     | "data"   | PyObject* (points to str)   |
            | 2     | "science"| PyObject* (points to str)   |

            Each string lives at a separate memory address, so the data is scattered across memory.
            As a result:
            - Cache locality is poor (data is not contiguous).
            - SIMD vectorization is impossible.
            - Each access requires Python object indirection.
            - Serialization requires walking through every object.

            ----------------------------------------------------------------
            Arrow-style representation (offsets + values)
            ----------------------------------------------------------------
            >>> offsets = [0, 5, 9, 16]
            >>> values  = b"hellodatascience"

            Here, `values` stores all bytes back-to-back, and `offsets`
            records where each string begins and ends:

            | String | Start | End | Slice (values[start:end]) | Bytes          | Decoded String |
            |--------|-------|-----|---------------------------|----------------|----------------|
            | 0      | 0     | 5   | values[0:5]               | b"hello"       | "hello"        |
            | 1      | 5     | 9   | values[5:9]               | b"data"        | "data"         |
            | 2      | 9     | 16  | values[9:16]              | b"science"     | "science"      |

            This design provides several key advantages:
            1. Compact storage: All string bytes are tightly packed without padding or wasted memory.
            2. Contiguous layout: Data resides in a single continuous memory region, improving CPU cache locality.
            3. Zero-copy slicing: Subarrays can be created by adjusting offsets, without copying bytes.
            4. Cross-language compatibility: Both `offsets` and `values` are primitive arrays (int32, uint8),
               allowing direct sharing between languages like C++, Python, Java, and Rust.

            In short, the (offsets + values) design lets Arrow-style arrays
            store variable-length strings efficiently while maintaining a compact,
            contiguous, and zero-copy memory layout.
    """

    def __init__(self, offsets: Buffer, values: Buffer, validity: Optional[Bitmap] = None):
        if (len(offsets.data) % 4) != 0:
            raise ValueError("offsets buffer length must be a multiple of 4 bytes (int32).")

        num_offsets = len(offsets.data) // 4

        if num_offsets < 1:
            raise ValueError("offsets must contain at least 1 int32 (length n+1).")

        # The `offsets` buffer defines string boundaries.
        # For n strings, we need n+1 offsets because the last value marks the end of the final string.
        #
        # For example, suppose we have three strings:
        # ["hi", "arrow", "world"]
        #
        # The start and end boundaries of each string are as follows:
        # | String  | Start | End  |
        # |---------|-------|------|
        # | "hi"    | 0     | 2    |
        # | "arrow" | 2     | 7    |
        # | "world" | 7     | 12   |
        #
        # Therefore, the offsets array becomes:
        # >>> offsets = [0, 2, 7, 12]
        #
        # The total number of strings is one less than the number of offsets:
        # >>> num_offsets - 1 = 3
        super().__init__(STRING, num_offsets - 1, validity)

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

    def to_pylist(self) -> List[Optional[str]]:
        """
        Convert the StringArray to a Python list, respecting null values.

        Returns:
            list: Python list representation of the array

        Discussion:
            Q. How does this function work internally?
                The StringArray stores all strings in a single contiguous `values` buffer (as uint8),
                and each string's start and end positions are defined by the `offsets` array.

                To reconstruct the original Python strings, we:
                1. Iterate through each index `i` in the array.
                2. Check if the element is null using `is_null(i)`.
                3. If valid, extract the corresponding byte slice:
                       start = self.offset_at(i)
                       end   = self.offset_at(i + 1)
                       byte_slice = memoryview(self.values.data[start:end])
                4. Convert that byte slice into a UTF-8 string:
                       byte_slice.decode("utf-8")
                5. Append the decoded string (or None) to the output list.

            Q. Example walkthrough
                Suppose we have the following data:

                    offsets = [0, 2, 7, 12]
                    values  = b"hiarrowworld"

                The data represents three strings: ["hi", "arrow", "world"]

                Here's how each element is reconstructed:

                    | Index | Start | End | Slice (bytes) | Decoded string |
                    |------:|-------|-----|----------------|----------------|
                    | 0     | 0     | 2   | b"hi"          | "hi"           |
                    | 1     | 2     | 7   | b"arrow"       | "arrow"        |
                    | 2     | 7     | 12  | b"world"       | "world"        |

                So, `to_pylist()` will return:
                    ["hi", "arrow", "world"]
        """

        output = []
        for i in range(self.length):
            if self.is_null(i):
                output.append(None)
            else:
                start = self.offset_at(i)
                end = self.offset_at(i + 1)
                byte_slice = bytes(self.values.data[start:end])
                output.append(byte_slice.decode("utf-8"))
        return output

    @classmethod
    def from_pylist(cls, data: List[Optional[str]]) -> "StringArray":
        """
        Create a StringArray from a Python list of strings (or None for nulls).

        Args:
            data (List[Optional[str]]): list of strings or None

        Returns:
            StringArray: constructed StringArray

        Discussion:
            Q. How does this function work internally?
                This method uses the StringBuilder to incrementally build the StringArray.
                It iterates through each element in the input Python list:
                - If the element is a valid string, it appends it to the builder.
                - If the element is None, it appends a null to the builder.

                After processing all elements, it calls `finish()` on the builder
                to produce an immutable StringArray.

            Q. Example usage
                >>> data = ["hello", None, "world"]
                >>> string_array = StringArray.from_pylist(data)
                >>> print(string_array.to_pylist())
                ["hello", None, "world"]
        """
        builder = StringBuilder()
        for s in data:
            builder.append(s)
        return builder.finish()


class StringBuilder(Builder[str, StringArray]):
    """
    A simple builder to incrementally construct a column of UTF-8 strings.

    Internally it builds three Arrow-style components:

      - values: a single byte buffer (`uint8`) that stores all string bytes back-to-back
      - offsets: an int32 array of length n+1 that stores string boundaries
      - validity: a bitmap where 1 means valid and 0 means null

    Discussion:
        Q. Why start offsets at 0 and keep length n+1?
            For n strings we need n+1 boundaries: the first start (0) and the final end.
            Example:
                strings = ["hi", "arrow", "world"]
                values  = b"hiarrowworld"
                offsets = [0, 2, 7, 12]   # length = 3 strings => 4 offsets

        Q. What about nulls?
            A null has no bytes, so its start and end offsets are equal.
            We record 0 in the validity list for that position and repeat the previous offset.
    """

    def __init__(self):
        # Offsets always start from 0.
        # For n logical strings, there will be n+1 offsets.
        self.offsets: List[int] = [0]

        # A growable byte buffer holding UTF-8-encoded data.
        self.values: bytearray = bytearray()

        # Validity flags, 1 = valid, 0 = null.
        self.validity: List[int] = []

    def append(self, s: Optional[str]) -> "StringBuilder":
        """
        Append a string or None to the builder.
        Returns self for method chaining.

        Args:
            s (Optional[str]): The string to append, or None for a null value.

        Returns:
            StringBuilder: self

        Discussion:
            Q. How does this work internally?
                1) If `s` is None:
                    - Append 0 to validity (null).
                    - Do NOT change the byte buffer.
                    - Repeat the last offset (same start and end).

                2) If `s` is a valid string:
                    - Encode to UTF-8 bytes and extend the byte buffer.
                    - Append 1 to validity (valid).
                    - Append the new cumulative byte length as the next offset.

            Q. Why cumulative length for offsets?
                offsets[i] is the start of string i, and offsets[i+1] is its end.
                Recording the running total length lets us reconstruct
                values[offsets[i]: offsets[i+1]] without extra bookkeeping.
        """
        if s is None:
            self.validity.append(0)
            # Repeat the last offset (no bytes added)
            self.offsets.append(self.offsets[-1])
            return self

        if not isinstance(s, str):
            raise TypeError(f"StringBuilder.append expects str or None, got {type(s).__name__}")

        b = s.encode("utf-8")
        self.values.extend(b)
        self.validity.append(1)
        self.offsets.append(len(self.values))
        return self

    def finish(self) -> StringArray:
        """
        Finalize the builder and return a `StringArray`.

        Returns:
            `StringArray`: A finalized, immutable `StringArray` built from appended strings.

        Discussion:
            Q. What does `finish()` do?
                1) Convert the offsets list to a contiguous byte buffer (int32 little-endian).
                2) Freeze the offsets_buffer and values_buffer into immutable `Buffer` objects.
                3) Create a validity bitmap if there are any nulls.
                4) Construct and return the `StringArray` with these components.
        """
        # Pack offsets into little-endian int32 buffer
        offsets_raw_buffer = bytearray(len(self.offsets) * 4)
        byte_offset = 0
        for offset in self.offsets:
            struct.pack_into("<i", offsets_raw_buffer, byte_offset, offset)
            byte_offset += 4

        # Freeze the buffers into immutable bytes
        frozen_offsets = bytes(offsets_raw_buffer)
        offsets_buffer = Buffer.from_bytes(frozen_offsets)

        frozen_values = bytes(self.values)
        values_buffer = Buffer.from_bytes(frozen_values)

        # Validity bitmap (if any nulls)
        validity = Bitmap.from_pylist(self.validity)

        # Create and return the StringArray
        return StringArray(offsets_buffer, values_buffer, validity)
