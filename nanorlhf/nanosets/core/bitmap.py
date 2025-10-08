from typing import Optional, List

from nanorlhf.nanosets.core.buffer import Buffer


class Bitmap:
    """
    Bitmap is a simple class that stores the validity (nullness) of elements.
    Each bit in the bitmap represents whether the corresponding element is valid or null.

    Args:
        size (int): The number of elements the bitmap can hold.
        buffer (Optional[Buffer]): Optional buffer to initialize the bitmap from.

    Notes:
        It stores 1 for valid values and 0 for nulls.
        e.g. [1, None, 1, None] -> 1010 -> 0b00001010 (in binary).

    Examples:
        >>> bm = Bitmap(10)
        >>> bm.set_valid(0, True)
        >>> print(bm.is_valid(0))  # True
        >>> bm.set_valid(3, False)
        >>> print(bm.is_valid(3))  # False

    Discussion:
        Q. Why do we need to store null (missing) information separately?
            In real-world datasets, some elements may be missing (e.g. None or NaN).
            If nulls are stored directly inside the data array, it mixes different Python object types,
            such as integers and NoneType, breaking the uniform memory layout.

            >>> data = [10, None, 30, 40]

            In Python, this list internally stores object pointers (PyObject*), not raw integers.
            Each element in the list is a pointer to a Python object allocated in a different memory location like:

            | Index | Value | Actual Storage Type            |
            |------:|-------|--------------------------------|
            | 0     | 10    | PyObject* (points to int)      |
            | 1     | None  | PyObject* (points to NoneType) |
            | 2     | 30    | PyObject* (points to int)      |
            | 3     | 40    | PyObject* (points to int)      |

            This scattered memory layout leads to several issues:
            - No SIMD vectorization: The CPU cannot process multiple elements at once.
            - Cache inefficiency: Values are distributed across non-contiguous memory.
            - High overhead: Each access goes through a Python object wrapper.

            By separating values and their validity information,
            the numeric data can be stored as a compact, contiguous memory block (e.g. int32, float64, etc.),
            while the validity is tracked by a lightweight bitmap that uses only 1 bit per element.

            >>> values  = [10, 0, 30, 40]   # int32 array
            >>> bitmap  = [1, 0, 1, 1]       # 1 means valid, 0 means null

            This design provides three major advantages:
            1. Speed: Contiguous memory enables SIMD and vectorized operations.
            2. Memory efficiency: Validity information requires only 1 bit per value.
            3. Interoperability: The structure is language-agnostic and can be shared
               across C, Python, Java, Rust, and others without serialization overhead.

            In short, separating null validity from data values allows Arrow-style arrays
            to achieve both high performance and flexibility.

        Q. What is SIMD vectorization, and why does it matter?
            SIMD stands for Single Instruction, Multiple Data.
            It allows the CPU to perform the same operation on multiple elements simultaneously.
            For example, instead of executing four separate additions:
            >>> # Scalar operations (no SIMD)
            >>> [1+1, 2+2, 3+3, 4+4]  # processed one by one

            With SIMD, the CPU can execute them all at once using vectorized instructions:
            >>> # Vectorized operations (with SIMD)
            >>> [1, 2, 3, 4] + [1, 2, 3, 4]  # computed together internally

            SIMD works only when data is stored contiguously in memory.
            If data is scattered (like Python objects), the CPU cannot load or process multiple values efficiently.
            This is why Arrow, NumPy, and Pandas all require contiguous memory layouts.
            They allow low-level vectorized operations that fully utilize CPU hardware capabilities.

        Q. What is cache inefficiency, and why does it matter?
            Modern CPUs are much faster than main memory, so they use a small, high-speed memory
            called the CPU cache to temporarily store recently accessed data.

            When data is stored contiguously (for example, in a NumPy or Arrow array),
            the CPU can load an entire sequence of elements into the cache in a single loading operation.
            This allows fast sequential processing because nearby elements are already cached.
            >>> # Contiguous memory (cache-friendly)
            >>> [10, 20, 30, 40]  # loaded into cache together

            However, when data is scattered in memory (like a list of PyObject* pointers),
            each element may reside in a completely different memory location.
            The CPU must repeatedly fetch data from main memory instead of reusing cached data,
            causing frequent "cache misses" that slow down processing dramatically.
            >>> # Scattered memory (cache-unfriendly)
            >>> [10, None, 30, 40]  # each object stored separately

            This is why Arrow stores all values in contiguous buffers
            ensuring that sequential reads fully benefit from CPU caching and memory prefetching.
    """

    def __init__(self, size: int, buffer: Optional[Buffer] = None):
        self.num_bits = int(size)
        # 1 byte is 8 bits, and `(size + 7) // 8` computes
        # the number of bytes needed to store `size` bits.
        num_bytes = (self.num_bits + 7) // 8

        if buffer is None:
            # allocate writable storage
            self.buffer = Buffer.from_bytearray(bytearray(num_bytes))
        else:
            if len(buffer.data) < num_bytes:
                raise ValueError(
                    f"Bitmap buffer too small: need {num_bytes} bytes, got {len(buffer.data)}"
                )
            self.buffer = buffer  # may be read-only if it wraps `bytes`

    def _check_bounds(self, i: int) -> None:
        """
        Check if the bit index is within bounds.

        Args:
            i (int): bit index to check
        """
        if not (0 <= i < self.num_bits):
            raise IndexError(f"bit index out of range: {i}")

    def _ensure_writable(self) -> None:
        """Ensure the bitmap buffer is writable."""
        if self.buffer.data.readonly:
            raise RuntimeError("Bitmap buffer is read-only; cannot mutate bits.")

    def set_valid(self, i: int, valid: bool = True):
        """
        Set the validity of the i-th element.

        Args:
            i (int): index of the element
            valid (bool): True for valid, False for invalid (null)

        Discussion:
            Q. How does this function work internally?
                Each bit in the bitmap represents the validity of one element.
                A bit value of 1 means valid, and 0 means null.
                The bitmap is stored as a NumPy array of bytes (uint8),
                so each byte holds the validity of 8 elements.

                >>> i = 10
                >>> byte, bit = divmod(i, 8)
                Here, `byte` is the index of the byte, and `bit` is the position inside that byte.

                For example:
                    i = 0  -> (byte=0, bit=0)
                    i = 7  -> (byte=0, bit=7)
                    i = 8  -> (byte=1, bit=0)
                    i = 9  -> (byte=1, bit=1)

            Q. How are bitwise operations used here?
                - To set a bit to 1 (mark valid):
                    self.buffer.data[byte] |= (1 << bit)
                    The OR operator (|) turns that bit to 1 while keeping other bits unchanged.

                - To clear a bit to 0 (mark null):
                    self.buffer.data &= ~(1 << bit) & 0xFF
                    The NOT (~) flips the mask bits (so only the target bit becomes 0),
                    and AND (&) keeps all other bits intact.

                - Why `& 0xFF` when clearing a bit?
                    In Python, integers can grow beyond 8 bits, so when we use the NOT operator (~),
                    it produces a negative number with an infinite series of leading 1s in binary.
                    This can lead to unintended consequences when performing bitwise operations on bytes.

                    By applying `& 0xFF`, we ensure that only the lowest 8 bits are kept,
                    effectively masking out any higher bits that could interfere with our byte-level operation.
                    This guarantees that the result remains within the valid range of a byte (0-255).

            Q. Example walkthrough
                Suppose we have byte = 00100100 (binary) and bit = 3.

                mask = (1 << 3)  # 00001000 (binary)

                - When setting 3rd bit valid (True):
                    00100100
                  | 00001000
                  = 00101100  (bit 3 set to 1)

                - When setting 3rd bit invalid (False):
                    00101100
                  & 11110111 (which is ~00001000)
                  = 00100100 (bit 3 cleared to 0)
                  & 11111111 (=0xFF, to keep it within a byte)
                  = 00100100
        """
        # sanity checks
        self._check_bounds(i)
        self._ensure_writable()

        # `divmod` returns (quotient=몫, remainder=나머지) as a tuple
        byte, bit = divmod(i, 8)
        b = self.buffer.data[byte]
        if valid:
            b |= (1 << bit)
        else:
            b &= ~(1 << bit) & 0xFF
        self.buffer.data[byte] = b

    def is_valid(self, i: int) -> bool:
        """
        Check if the i-th element is valid.

        Args:
            i (int): index of the element

        Returns:
            bool: True if valid, False if invalid (null)

        Notes:
            - The method runs in O(1) time, regardless of the bitmap size.
            - Using bitmaps avoids storing one boolean per element (which would use 1 byte each),
              saving up to 8× memory.

        Discussion:
            Q. How does this function check the bit?
                It locates which byte and bit represent the given element’s validity,
                then uses a bitwise AND (&) to test whether the target bit is set.

                >>> byte, bit = divmod(i, 8)
                >>> mask = (1 << bit)
                >>> (self.buffer.data[byte] & mask) != 0

                If the result is nonzero, the bit is 1 (valid).
                If it is zero, the bit is 0 (null).

            Q. Example walkthrough
                Suppose the byte is 00100100 (binary):

                - For bit=2:
                    mask = 00000100
                    00100100 & 00000100 = 00000100  -> nonzero -> True

                - For bit=1:
                    mask = 00000010
                    00100100 & 00000010 = 00000000  -> zero -> False
        """
        self._check_bounds(i)
        byte, bit = divmod(i, 8)
        return (self.buffer.data[byte] & (1 << bit)) != 0

    @classmethod
    def from_pylist(cls, bits: List[int]) -> Optional["Bitmap"]:
        """
        Build a bitmap from a 0/1 list.
        This method packs and pads the bits into bytes.

        Args:
            bits: 0/1 per element (1 = valid, 0 = null).

        Returns:
            Bitmap: New bitmap whose logical size is len(bits).

        Discussion:
            Q. What is packing and padding?
                - Packing:
                    Storing multiple boolean flags as individual bits inside bytes
                    (8 flags per byte) instead of using one byte per flag.

                - Padding:
                    When the number of flags N is not a multiple of 8, the final byte has
                    unused (extra) bit positions. Those high bits are explicitly set to 0 so
                    the bitmap occupies an integer number of bytes with a deterministic value.

                - Example:
                    bits = [1, 0, 1, 1, 1]  # N = 5
                    packed byte steps (little-endian):
                        initial state:  xxxxxxxx
                            → set i=0 → xxxxxxx1
                            → set i=1 → xxxxxx01
                            → set i=2 → xxxxx101
                            → set i=3 → xxxx1101
                            → set i=4 → xxx11101
                    padding bits (positions 5..7) = 0
                    stored byte = 0b00011101

            Q. Why packing matters?
                Storing one boolean per element would use 1 byte each,
                while a bitmap uses only 1 bit per element, saving up to 8× memory.

                For example, for 1 million elements:
                    - Using bytes: 1,000,000 bytes (about 1 MB)
                    - Using bitmap: 125,000 bytes (about 125 KB)

                This is especially important for large datasets,
                where memory savings can significantly improve performance
                by reducing cache misses and memory bandwidth usage.

            Q. What is little-endian bit order?
                In little-endian bit order, the least significant bit (bit 0)
                corresponds to the lowest index (index 0). This is the reverse
                of how humans usually read binary numbers.

                For example:
                    bits = [1, 0, 1, 0, 0, 0, 0, 0]
                    Packed byte = 0b00000101 = 5

                This means:
                    bit 0 (LSB) → index 0 (1)
                    bit 1 → index 1 (0)
                    bit 2 → index 2 (1)
                    bit 3 → index 3 (0)
                    ...

                Arrow and NumPy both use little-endian order for consistency across
                platforms. It allows fast unpacking and efficient bit masking (`1 << bit`),
                because the first element’s validity directly maps to the lowest bit in memory.

            Q. Why use little-endian instead of big-endian?
                Big-endian order (most significant bit first) is more intuitive for
                human reading, but little-endian is better optimized for modern CPUs.

                - In big-endian layout, bits are stored left-to-right, like `10000000`
                  meaning bit 0 is the leftmost (most significant bit).
                - In little-endian layout, bits are stored right-to-left, like `00000001`
                  meaning bit 0 is the rightmost (least significant bit).

                Modern CPUs (x86, AMD64, ARM, Apple M1, etc.) use little-endian internally
                because it matches how hardware performs arithmetic and memory access.

                Advantages of little-endian:
                    1. CPUs perform arithmetic starting from the least significant bit (rightmost bit).
                       For example, when adding 1 + 1:
                           0b00000001 (binary, not storing order)
                         + 0b00000001 (binary, not storing order)
                         = 0b00000010 (binary, not storing order)

                       The CPU begins by processing bit 0 first, then carries over to higher bits as needed.
                       Storing the least significant part first in memory matches this natural flow.
                       So the CPU can store result bits first as it computes, without waiting for the higher bits.
                       This makes arithmetic operations like addition and multiplication simpler and faster to implement.

                    2. Reading integers of different sizes becomes simpler.
                       For instance, if 4 bytes in memory are:
                           [0x78, 0x56, 0x34, 0x12]
                       then:
                           - as a 1-byte integer → 0 x 7 | 8 | x | x | x | x | x | x
                           - as a 2-byte integer → 0 x 5 | 6 | 7 | 8 | x | x | x | x
                           - as a 4-byte integer → 0 x 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
                                                       ↳ start reading from here
                       Notice how you can just "extend" the number by reading more bytes.
                       You don’t need to reorder or reinterpret them — this is not true for big-endian systems.

                       In big-endian systems, however, the most significant byte is stored first,
                       so the CPU must reverse the order (called byte-swapping) to interpret the value correctly.
                       Little-endian avoids this entirely because its memory layout naturally aligns
                       with how CPUs process numbers.

                    3. Memory compatibility and efficiency.
                       Almost all modern CPUs (x86, AMD64, ARM) are little-endian,
                       and C, NumPy, and Arrow all follow the same memory layout.
                       This means binary data can be shared directly between languages
                       (C, Python, Rust, Java, etc.) without copying or converting.

                In short:
                    Big-endian is more human-friendly (easier to read),
                    Little-endian is more machine-friendly (faster for CPUs).
                    Arrow adopts little-endian to maximize performance and ensure
                    cross-platform consistency with NumPy and native C memory layouts.
        """
        # if all valid, return None for no bitmap
        if 0 not in bits:
            return None

        bitmap = cls(size=len(bits))
        for i, v in enumerate(bits):
            if v == 1:
                bitmap.set_valid(i, True)
            elif v == 0:
                pass  # default is already 0
            else:
                raise ValueError(f"Invalid bit value at index {i}: {v} (must be 0 or 1)")
        return bitmap

    def to_pylist(self) -> List[int]:
        """
        Unpack bitmap to a list of 0/1

        Returns:
            List[int]: 0/1 per element (1 = valid, 0 = null)
        """
        return [1 if self.is_valid(i) else 0 for i in range(self.num_bits)]
