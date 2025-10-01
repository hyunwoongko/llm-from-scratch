"""
Notes:
    IPC Stands for 'Inter-Process Communication'.
"""

import json
import mmap
import struct
from typing import List

from nanorlhf.nanosets.base.bitmap import Bitmap
from nanorlhf.nanosets.base.buffer import Buffer
from nanorlhf.nanosets.dtype.dtype import DataType
from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArray
from nanorlhf.nanosets.dtype.string_array import StringArray
from nanorlhf.nanosets.dtype.list_array import ListArray
from nanorlhf.nanosets.dtype.struct_array import StructArray
from nanorlhf.nanosets.table.field import Field
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.schema import Schema
from nanorlhf.nanosets.table.table import Table

MAGIC = b"NANO0"


def write_table(fp, table: Table):
    """
    Serialize a `Table` into a compact binary IPC format

    Args:
        fp (file-like): writable binary file-like object
        table (Table): Table to serialize

    Notes:
        The file layout is as follows:
        - MAGIC (5 bytes)
        - Header Length (4 bytes)
        - Header (JSON)
        - Buffers (Blobs)

    Examples:
        >>> with open("table.nano", "wb") as f:
        ...    write_table(f, table)

    Discussion:
        Q. What is a magic string?
            A magic string (or "magic number") is a short fixed sequence of bytes
            written at the beginning of a file to uniquely identify its format.
            Many file formats use it (e.g., PNG = b"\\x89PNG", ZIP = b"PK\\x03\\x04").
            It allows quick detection of file type and prevents misinterpretation.
            We use `b"NANO0"` as our magic string, meaning "This is a Nanoset IPC file, version 0".

        Q. Why header in Json?
            The header is stored as a JSON object for human readability and cross-language interoperability,
            while the actual data (buffers) are written as raw binary blobs for performance and memory efficiency.
            The header is very small compared to the data, so the overhead of JSON is negligible.

        Q. What is blob?
            The term ‘blob’ refers to the raw binary data buffers (e.g. values, offsets, validity bitmaps)
            written sequentially after the header.
    """
    # Flat list of byte blobs to be written after the header
    blobs: List[bytes] = []

    def add_buf(b: Buffer) -> int:
        """Append a buffer's bytes and return its index for referencing in the header."""
        i = len(blobs)
        # Ensure a stable snapshot: convert to bytes (may copy if underlying is mutable)
        blobs.append(b.data)
        return i

    def dtype_meta(dt: DataType):
        """Encode DataType into a minimal JSON-friendly metadata object."""
        return {"kind": dt.name}

    def encode_array(arr):
        """
        Encode an array's metadata, collect its underlying buffers, and
        return a JSON-serializable dictionary. Supports:
        - PrimitiveArray
        - StringArray
        - ListArray (recursively encodes child)
        - StructArray (recursively encodes children)
        """
        meta = {"dtype": dtype_meta(arr.dtype), "length": arr.length}

        if arr.validity is not None:
            meta["validity"] = add_buf(arr.validity.buf)

        if isinstance(arr, PrimitiveArray):
            meta["kind"] = "primitive"
            # values buffer holds raw fixed-width items (little-endian pack)
            meta["values"] = add_buf(arr.values)

        elif isinstance(arr, StringArray):
            meta["kind"] = "string"
            # offsets:int32 (little-endian), values:uint8 (UTF-8 bytes)
            meta["offsets"] = add_buf(arr.offsets)
            meta["values"] = add_buf(arr.values)

        elif isinstance(arr, ListArray):
            meta["kind"] = "list"
            # offsets:int32 (little-endian) + child array (recursive)
            meta["offsets"] = add_buf(arr.offsets)
            meta["child"] = encode_array(arr.values)

        elif isinstance(arr, StructArray):
            meta["kind"] = "struct"
            # struct has no offsets; it is a fixed-shape record of children
            meta["names"] = arr.names
            meta["children"] = [encode_array(ch) for ch in arr.children]

        else:
            raise TypeError(f"unsupported array type: {type(arr).__name__}")

        return meta

    # Build the header (encode_array() calls above will populate `blobs`)
    header = {
        "schema": {
            "fields": [
                {"name": f.name, "dtype": dtype_meta(f.dtype), "nullable": f.nullable}
                for f in table.schema.fields
            ]
        },
        "batches": [
            {"length": b.length, "columns": [encode_array(arr) for arr in b.columns]}
            for b in table.batches
        ],
        "buffers": [],  # will be filled with offsets/lengths right after
    }

    # Compute offsets for the collected blobs, in the same concatenation order
    offset = 0
    for blob in blobs:
        header["buffers"].append({"offset": offset, "length": len(blob)})
        offset += len(blob)

    # Serialize header as JSON (keep defaults for readability)
    header_bytes = json.dumps(header).encode("utf-8")

    # Write the file: MAGIC, header length, header bytes, then all blobs
    fp.write(MAGIC)
    fp.write(struct.pack("<I", len(header_bytes)))  # '<I' means little-endian unsigned int
    fp.write(header_bytes)
    for blob in blobs:
        fp.write(blob)


def read_table(path: str) -> Table:
    """
    Memory-map a serialized `Table` from a file written using `write_table()`.

    Args:
        path (str): Path to the `.nano` file.

    Returns:
        Table: A fully reconstructed `Table` backed by memory-mapped buffers.

    Examples:
        >>> table = read_table("table.nano")

    Discussion:
        Q. What is `mmap`?
            Before understanding `mmap`, it helps to know how RAM is structured in an operating system.

            In modern OSes, RAM is conceptually divided into:
              - Kernel space: managed by the OS; used for disk caches, I/O buffers, and device control.
              - User space: used by individual programs; isolated from the kernel for safety.

            Normally, reading a file with standard I/O (e.g., `fp.read()`) follows this path:
                [Disk] → Copy → [Kernel space] → Copy → [User space]

            The first copy loads data from disk into the kernel space,
            and the second copy moves it into the user space.
            This double-copy increases CPU overhead and wastes memory bandwidth.

            `mmap` (memory mapping) removes that second copy.
            User space has its own memory area called the virtual address space (or virtual memory).
            When we use `mmap`, instead of copying data into user space,
            it maps the addresses of kernel space directly into the user virtual address space.

                [Disk] → Copy → [Kernel space] ↔ [User virtual address space]

            So the file data itself remains in the kernel space and user space just has pointers to it.

        Q. What is a page in memory?
            A page is the smallest fixed-size block of memory managed by the operating system’s
            virtual memory system. Most systems use 4 KB per page.

            When the OS copies data from disk into kernel space,
            it doesn't load individual bytes; it loads an entire page (4KB) at a time.
            These pages are stored in the kernel space in a structure called the page cache.

        Q. Is the entire file copied into the kernel page cache at once?
            No. When the user program first reads from a mapped region, a page fault occurs,
            prompting the OS to copy the **only needed page** from the disk into the kernel space.
            This is called **demand paging** or **lazy loading**.

            Before 1st access:
                [User virtual address space] → Page Fault → OS → [Disk] → Copy → [Kernel space]

            After 1st access:
                [User virtual address space] ↔ [Kernel space]

            This can be summarized in the following table:
                +----------------------------------------------------------------------------------------------------+
                |                    Memory Mapping States                     |                                     |
                +------------------+-------------------------------------------+-------------------------------------+
                | Stage            | User Space (Process Memory)               | Kernel Space (Page Cache)           |
                +------------------+-------------------------------------------+-------------------------------------+
                | `mmap()` called  | Space for virtual addresses reserved      | No file data loaded (still on disk) |
                | After 1st access | Address → Kernel page mapping established | File page loaded into page cache    |
                | Later accesses   | Reads from mapped addresses               | File page remains in page cache     |
                +------------------+-------------------------------------------+-------------------------------------+

        Q. How is `mmap` different from `memoryview`?
            Both `mmap` and `memoryview` provide zero-copy access, but at different levels:
            The key difference is where the zero-copy happens, OS-level vs Python-level.

            `mmap`: works at the OS level.
                Maps a page address directly into virtual memory.
                Avoids copying data between kernel and user space.

            `memoryview`: works at the Python level.
                Provides a zero-copy view into existing memory
                (e.g., bytes, NumPy arrays, or `mmap` objects)
                but doesn't handle disk I/O or paging.

            In short:
                - `mmap`: OS-level zero-copy (disk ↔ virtual memory)
                - `memoryview`: Python-level zero-copy (RAM ↔ Python object)
    """
    # Open the file and create a read-only memory map.
    # On some platforms (e.g., Windows), keeping the file handle alive is safer while the map is in use.
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        # Verify magic string
        if mm.read(len(MAGIC)) != MAGIC:
            raise ValueError("Invalid file format: missing magic string")

        # Read header length and header JSON
        (len_header,) = struct.unpack("<I", mm.read(4))
        header_string = mm.read(len_header).decode("utf-8")
        header = json.loads(header_string)

        # Calculate total length of concatenated data buffers
        total = sum(b["length"] for b in header["buffers"])

        # Create a zero-copy memoryview into the mmap for all buffer data
        data_start = mm.tell()
        base = memoryview(mm)[data_start : data_start + total]

        # Materialize Buffer views using offsets/lengths from the header
        buf_views: List[Buffer] = []
        for b in header["buffers"]:
            start = b["offset"]
            end = start + b["length"]
            buf_views.append(Buffer(base[start:end]))

        def meta_to_dtype(m):
            """Decode DataType metadata from header."""
            return DataType(m["kind"])

        def decode_array(m):
            """
            Reconstruct array objects from metadata and buffer indices.
            Supports:
              - primitive, string, list, struct
            """
            dt = meta_to_dtype(m["dtype"])
            validity = Bitmap(m["length"], buf_views[m["validity"]]) if "validity" in m else None

            kind = m["kind"]
            if kind == "primitive":
                # values: raw fixed-width items buffer
                values = buf_views[m["values"]]
                return PrimitiveArray(dt, values, m["length"], validity)

            if kind == "string":
                offsets = buf_views[m["offsets"]]
                values = buf_views[m["values"]]
                return StringArray(offsets, values, validity)

            if kind == "list":
                offsets = buf_views[m["offsets"]]
                child = decode_array(m["child"])
                return ListArray(offsets, child, validity)

            if kind == "struct":
                names = m["names"]
                children = [decode_array(cm) for cm in m["children"]]
                return StructArray(names, children, validity)

            raise TypeError(f"unsupported array kind: {kind!r}")

        # Reconstruct schema
        fields = tuple(
            Field(fld["name"], meta_to_dtype(fld["dtype"]), fld.get("nullable", True))
            for fld in header["schema"]["fields"]
        )
        schema = Schema(fields)

        # Reconstruct batches/columns
        batches = []
        for b in header["batches"]:
            cols = [decode_array(col_meta) for col_meta in b["columns"]]
            batches.append(RecordBatch(schema, cols))

        # Build final table
        return Table(batches)

    except Exception:
        # Cleanup on error
        mm.close()
        f.close()
        raise
