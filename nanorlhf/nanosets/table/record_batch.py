from typing import List, Union, Optional, Dict, Any

from nanorlhf.nanosets.data_type.array import Array
from nanorlhf.nanosets.data_type.struct_array import StructArray
from nanorlhf.nanosets.table.field import Field
from nanorlhf.nanosets.table.schema import Schema


class RecordBatch:
    """
    A RecordBatch represents a batch of rows sharing the same Schema.

    Args:
        schema (Schema): The schema describing column names and data types.
        columns (list[Array]): A list of Array objects holding column data.

    Examples:
        >>> from nanorlhf.nanosets.table.schema import Schema
        >>> from nanorlhf.nanosets.table.field import Field
        >>> from nanorlhf.nanosets.data_type.data_type import INT32, FLOAT32, STRING
        >>> from nanorlhf.nanosets.data_type.primitive_array import PrimitiveArray
        >>> from nanorlhf.nanosets.data_type.string_array import StringArray
        >>> schema = Schema(
        >>>     (
        ...              Field("x", INT32),
        ...              Field("y", FLOAT32),
        ...              Field("z", STRING)
        ...     )
        ... )
        >>> arr_x = PrimitiveArray.from_pylist([1, 2, 3], dtype=INT32)
        >>> arr_y = PrimitiveArray.from_pylist([1.0, 2.0, 3.0], dtype=FLOAT32)
        >>> arr_z = StringArray.from_pylist(["hi", "there", "!"])
        >>> batch = RecordBatch(schema, [arr_x, arr_y, arr_z])
        >>> batch.column("x").to_pylist()
        [1, 2, 3]
        >>> batch.column("z").to_pylist()
        ['hi', 'there', '!']

    Discussion:
        Q. What is the relationship between `RecordBatch` and `Table`?
            A `RecordBatch` is a *physical chunk* of columnar data,
            while a `Table` is a *logical abstraction* combines multiple `RecordBatches`
            into a single, unified dataset.

            Each `RecordBatch` stores its columns in contiguous memory
            (e.g. each column is a compact numpy or Arrow buffer).
            This ensures high cache locality and efficient vectorized access.

            A `Table`, on the other hand, does not store all of its data in one continuous memory region.
            Its columns are logically unified across multiple RecordBatch instances,
            but physically split into several contiguous segments, one per batch.

            Example:
                RecordBatch1:
                    x = [1, 2, 3]
                    y = ["a", "b", "c"]

                RecordBatch2:
                    x = [4, 5]
                    y = ["d", "e"]

                Table([RecordBatch1, RecordBatch2]):
                    | x | y |
                    |---|---|
                    | 1 | a |
                    | 2 | b |
                    | 3 | c |
                    | 4 | d |
                    | 5 | e |

            In short:
                - `RecordBatch`: one chunk of contiguous columnar data
                - `Table`: multiple RecordBatches logically concatenated

            This separation is essential for:
                1. Streaming processing: Data can be processed batch by batch
                   without loading the entire dataset into memory.
                2. Parallelism: Each `RecordBatch` can be processed independently,
                   enabling multicore or distributed execution.
                3. Cache efficiency: Small batches fit into CPU caches,
                   minimizing memory latency and improving performance.
                4. I/O efficiency: Datasets can be saved and loaded
                   in independent chunks (used by Arrow IPC, Parquet, etc.).

        Q. Why does `RecordBatch` store data in a columnar layout instead of row-major?
            1. Contiguous memory and cache efficiency
               Columns of the same data type are stored together in a single contiguous buffer.
               This drastically improves CPU cache locality when scanning or aggregating a single column.
               In contrast, row-oriented storage interleaves different data types (e.g., int, float, string),
               causing more cache misses and wasted memory reads.

            2. Vectorization and SIMD acceleration
               Since column data is contiguous and homogeneous (e.g., all `int32` or all `float64`),
               the CPU can apply SIMD instructions to process multiple values in parallel.
               For example, computing the sum of column `x` can be done by scanning only the `x` buffer
               without touching unrelated data from other columns.

            3. Efficient handling of nulls and variable-length data
               Variable-length data such as strings are stored using two buffers, `values(uint8)` and `offsets(int32)`
               avoiding thousands of scattered Python objects and improving cache and serialization efficiency.

            4. Trade-offs
               - Columnar format is not ideal for frequent row-level updates:
                 modifying one row requires updating multiple column buffers.
               - Reconstructing full rows (e.g., in `to_pylist()`) requires assembling values
                 from multiple column buffers, which adds some overhead.
               In short: columnar layout favors *read-heavy analytical workloads*,
               while row layout suits *write-heavy transactional workloads*.

               During deep learning model training, data is read far more frequently than it is written,
               making the columnar format well-suited for this workload.

            Example:
                Row-major layout:
                    [x1, y1, z1][x2, y2, z2][x3, y3, z3]
                    # mixed types interleaved → poor cache use

                Columnar layout:
                    x: [x1, x2, x3, ...]
                    y: [y1, y2, y3, ...]
                    z: [z1, z2, z3, ...]
                    # contiguous columns → better scanning & compression

                - Row layout: must read full (x, y, z) tuples even if only `x` is needed.
                - Columnar layout: can read just `x` buffer - minimal I/O, higher cache hit rate, SIMD-friendly.
    """

    def __init__(self, schema: Schema, columns: List[Array]):
        assert len(schema.fields) == len(columns), "Number of columns must match schema fields"
        lens = {len(c) for c in columns}
        assert len(lens) == 1, "All columns in a RecordBatch must have the same length."

        self.schema = schema
        self.columns = columns
        self.length = next(iter(lens))

    def column(self, i_or_name: Union[int, str]):
        """
        Retrieve a column by index or name.

        Args:
            i_or_name (int or str): Column index (int) or name (str)

        Returns:
            Array: The requested column array
        """
        if isinstance(i_or_name, int):
            return self.columns[i_or_name]
        elif isinstance(i_or_name, str):
            idx = self.schema.index(i_or_name)
            return self.columns[idx]
        else:
            raise TypeError("Argument must be an integer index or a string column name.")

    def to_pylist(self) -> List[Dict[str, Any]]:
        """Convert this RecordBatch into a list of Python row dictionaries.

        Returns:
            List[Dict[str, Any]]: One dict per row, mapping field name -> value.
                Child arrays may contain None (nulls), nested lists, or nested dicts,
                depending on the column type.

        Discussion:
            Implementation details:
              1) Each column array is converted once with `to_pylist()` to avoid
                 per-element array method calls inside the row loop.
              2) Rows are assembled by indexing into those per-column Python lists.
              3) If the schema has zero fields, this returns a list of empty dicts.

            Complexity:
              - Time:  O(num_columns * column_to_pylist + num_rows * num_columns)
              - Space: O(num_columns * num_rows) for the intermediate per-column lists.
        """
        rows: List[Dict[str, Any]] = []
        per_column_python_lists = [column.to_pylist() for column in self.columns]

        for row_index in range(self.length):
            row: Dict[str, Any] = {}
            for field, column_values in zip(self.schema.fields, per_column_python_lists):
                row[field.name] = column_values[row_index]
            rows.append(row)

        return rows

    @classmethod
    def from_pylist(cls, rows: List[Optional[Dict[str, Any]]], *, strict_keys: bool = False) -> "RecordBatch":
        """Build a RecordBatch from Python dict rows (or None) with schema inference.

        This constructs a StructArray from the input rows (handling nested lists/structs and nulls),
        derives a Schema (names, dtypes, and nullability) from the StructArray's children,
        and returns a RecordBatch sharing those child column arrays (zero-copy).

        Args:
            rows: A list where each item is either:
                - dict mapping field name -> value (value can be primitive, str, list, or dict), or
                - None to represent a null row.

        Returns:
            RecordBatch: A single-batch columnar container with inferred schema.

        Raises:
            TypeError/ValueError: If per-field values mix incompatible types or no fields can be inferred.

        Examples:
            >>> rows = [
            ...     {"id": 1, "name": "a", "tags": ["ml", "rl"]},
            ...     None,
            ...     {"id": 2, "name": "b"}  # missing 'tags' → treated as None
            ... ]
            >>> batch = RecordBatch.from_pylist(rows)
            >>> batch.schema.names()
            ['id', 'name', 'tags']
            >>> batch.column('name').to_pylist()
            ['a', None, 'b']
        """
        # 1) Column construction & type inference (supports nesting) via StructArray
        struct = StructArray.from_pylist(rows, strict_keys=strict_keys)

        # 2) Derive schema from children (name, dtype, nullable)
        fields = tuple(
            Field(
                name=name,
                dtype=child.dtype,
                nullable=(child.validity is not None),
            )
            for name, child in zip(struct.names, struct.children)
        )
        schema = Schema(fields)

        # 3) Return a RecordBatch that shares the child arrays (no extra copies)
        return cls(schema, struct.children)
