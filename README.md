# Charmander

**C**ross-platform **H**andling of **A**rray, **R**ecursive, **M**apping, **A**nd **N**ested **D**ata **E**xchange **R**untime

Convert between Polars schemas and PySpark schemas with ease.

Charmander provides simple, bidirectional conversion functions to transform schemas between Polars and PySpark, supporting all complex types including nested structures, arrays, and maps.

## Installation

```bash
pip install charmander
```

## Requirements

- Python >= 3.8
- polars >= 0.19.0
- pyspark >= 3.0.0

## Quick Start

### Converting Polars Schema to PySpark

Charmander supports **three Polars schema formats** - use whichever is most convenient:

```python
import polars as pl
from charmander import to_pyspark_schema

# Format 1: Dictionary
polars_schema_dict = {
    "name": pl.String,
    "age": pl.Int32,
    "score": pl.Float64,
    "tags": pl.List(pl.String),
}

# Format 2: pl.Schema object
polars_schema_schema = pl.Schema({
    "name": pl.String,
    "age": pl.Int32,
    "score": pl.Float64,
    "tags": pl.List(pl.String),
})

# Format 3: List of tuples
polars_schema_list = [
    ("name", pl.String),
    ("age", pl.Int32),
    ("score", pl.Float64),
    ("tags", pl.List(pl.String)),
]

# All three formats work identically!
pyspark_schema = to_pyspark_schema(polars_schema_dict)
# or: to_pyspark_schema(polars_schema_schema)
# or: to_pyspark_schema(polars_schema_list)

print(pyspark_schema)
# StructType([StructField('name', StringType(), True),
#             StructField('age', IntegerType(), True),
#             StructField('score', DoubleType(), True),
#             StructField('tags', ArrayType(StringType(), True), True)])
```

### Converting PySpark Schema to Polars

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from charmander import to_polars_schema

# Define a PySpark schema
pyspark_schema = StructType([
    StructField("name", StringType()),
    StructField("age", IntegerType()),
    StructField("score", DoubleType()),
    StructField("tags", ArrayType(StringType())),
])

# Convert to Polars schema
polars_schema = to_polars_schema(pyspark_schema)
print(polars_schema)
# Schema({'name': <class 'polars.datatypes.String'>, 'age': <class 'polars.datatypes.Int32'>, ...})

# Use directly with Polars DataFrame
df = pl.DataFrame({}, schema=polars_schema)
```

## Features

- **Bidirectional Conversion**: Convert schemas in both directions (Polars ↔ PySpark)
- **Multiple Schema Formats**: Supports `pl.Schema`, `dict[str, pl.DataType]`, and `Iterable[tuple[str, pl.DataType]]` formats
- **Native Polars Integration**: Returns `pl.Schema` objects from `to_polars_schema` for seamless DataFrame integration
- **Comprehensive Type Support**: Supports all primitive and complex types
- **Nested Structures**: Handles deeply nested structs, arrays, and maps
- **Type Safety**: Clear error messages for unsupported types
- **Simple API**: Functional, stateless functions - easy to use and understand

## Supported Types

### Primitive Types

| Polars | PySpark |
|--------|---------|
| `Int8` | `ByteType` |
| `Int16` | `ShortType` |
| `Int32` | `IntegerType` |
| `Int64` | `LongType` |
| `UInt8` | `ShortType` |
| `UInt16` | `IntegerType` |
| `UInt32` | `LongType` |
| `Float32` | `FloatType` |
| `Float64` | `DoubleType` |
| `Boolean` | `BooleanType` |
| `String` / `Utf8` | `StringType` |
| `Date` | `DateType` |
| `Datetime` | `TimestampType` |
| `Decimal` | `DecimalType` |
| `Binary` | `BinaryType` |
| `Null` | `NullType` |
| `Categorical` | `StringType` |
| `Enum` | `StringType` |
| `Int128` | `DecimalType` |

**PySpark Types:**

| PySpark | Polars |
|---------|--------|
| `ByteType` | `Int8` |
| `ShortType` | `Int32` |
| `IntegerType` | `Int32` |
| `LongType` | `Int64` |
| `FloatType` | `Float32` |
| `DoubleType` | `Float64` |
| `BooleanType` | `Boolean` |
| `StringType` | `String` |
| `VarcharType` | `String` |
| `CharType` | `String` |
| `DateType` | `Date` |
| `TimestampType` | `Datetime` |
| `TimestampNTZType` | `Datetime` |
| `DecimalType` | `Decimal` |
| `BinaryType` | `Binary` |
| `NullType` | `Null` |

### Complex Types

- **Arrays/Lists**: Fully supported with nested arrays
- **Structs**: Fully supported with nested structs
- **Maps**: PySpark `MapType` converts to Polars `Struct` (with `key` and `value` fields)

## Limitations

### Type Conversions with Information Loss

Some type conversions result in information loss or semantic changes:

- **UInt64 → LongType**: PySpark doesn't support unsigned 64-bit integers, so `UInt64` maps to signed `LongType`. Values greater than `2^63 - 1` may cause issues.

- **Duration → StringType**: Polars `Duration` types are converted to PySpark `StringType` as PySpark doesn't have a native duration type. The semantic meaning is lost.

- **Time → TimestampType**: Polars `Time` types are converted to PySpark `TimestampType`, which may not be the ideal representation.

- **Decimal precision/scale**: When converting Polars `Decimal` to PySpark `DecimalType`, default precision (10) and scale (0) are used. Precision and scale information is not preserved when converting from PySpark to Polars.

- **MapType → Struct**: PySpark `MapType` is converted to a Polars `Struct` with `key` and `value` fields. This changes the data structure from a map to a struct representation.

### Nullability

- **Polars → PySpark**: All fields are created with `nullable=True`, as Polars schemas don't explicitly track nullability at the schema definition level.

- **PySpark → Polars**: The `nullable` attribute from PySpark `StructField` is not preserved, as Polars schemas don't track nullability per field. All Polars fields can contain nulls by default.

### Input Validation

Charmander validates schemas before conversion:

- **Duplicate field names**: Raises `SchemaError` if duplicate field names are detected
- **Empty field names**: Raises `SchemaError` if any field name is an empty string
- **Invalid field types**: Raises `SchemaError` if field types are `None`
- **Invalid field name types**: Raises `SchemaError` if field names are not strings

### Datetime Timezone Handling

- Polars `Datetime` types can have timezone information (e.g., `pl.Datetime(time_unit="ms", time_zone="UTC")`)
- When converting to PySpark `TimestampType`, timezone information is not preserved
- `TimestampNTZType` (PySpark 3.4+) is converted to Polars `Datetime` without timezone information
- The timezone metadata is lost in conversion, but the timestamp value is preserved

## Advanced Examples

### Nested Structures

```python
import polars as pl
from charmander import to_pyspark_schema

# Define a nested Polars schema
polars_schema = {
    "user": pl.Struct([
        pl.Field("name", pl.String),
        pl.Field("address", pl.Struct([
            pl.Field("street", pl.String),
            pl.Field("city", pl.String),
            pl.Field("zip", pl.Int32),
        ])),
    ]),
}

pyspark_schema = to_pyspark_schema(polars_schema)
```

### Arrays with Nested Types

```python
import polars as pl
from charmander import to_pyspark_schema

# Nested arrays
polars_schema = {
    "matrix": pl.List(pl.List(pl.Float64)),
    "tags": pl.List(pl.String),
}

pyspark_schema = to_pyspark_schema(polars_schema)
```

### Round-Trip Conversion

```python
import polars as pl
from charmander import to_pyspark_schema, to_polars_schema

# Start with Polars schema (any format works)
original = {
    "name": pl.String,
    "age": pl.Int32,
    "scores": pl.List(pl.Float64),
}

# Convert to PySpark and back
pyspark = to_pyspark_schema(original)
converted_back = to_polars_schema(pyspark)  # Returns pl.Schema

# Verify types match (pl.Schema supports dict-like access)
assert converted_back["name"] == original["name"]
assert converted_back["age"] == original["age"]
assert isinstance(converted_back, pl.Schema)
```

## Error Handling

Charmander provides clear error messages through custom exceptions. All exceptions inherit from `ConversionError`, so you can catch all conversion errors at once or handle them individually:

```python
from charmander import ConversionError, UnsupportedTypeError, SchemaError

# Example 1: Handle specific error types
try:
    schema = to_pyspark_schema(invalid_schema)
except SchemaError as e:
    print(f"Invalid schema structure: {e}")
    # Handles: duplicate field names, empty field names, invalid field types, etc.
except UnsupportedTypeError as e:
    print(f"Unsupported type: {e}")
    # Handles: types that cannot be converted between Polars and PySpark
except ConversionError as e:
    print(f"General conversion error: {e}")
    # Catches all conversion-related errors (base class)

# Example 2: Catch all conversion errors
try:
    schema = to_pyspark_schema(invalid_schema)
except ConversionError as e:
    print(f"Conversion failed: {e}")
    # This will catch SchemaError, UnsupportedTypeError, and any future error types

# Example 3: Common error scenarios
try:
    # Invalid iterable format
    schema = to_pyspark_schema([("name", pl.String), "invalid"])
except SchemaError as e:
    print(f"Schema validation failed: {e}")
    # Output: "Invalid schema format: <class 'list'>. Expected iterable of (field_name, type) tuples. Item at index 1 is not a tuple: 'invalid'"

try:
    # Duplicate field names
    schema = to_pyspark_schema([("name", pl.String), ("name", pl.Int32)])
except SchemaError as e:
    print(f"Duplicate field: {e}")
    # Output: "Invalid schema format: <class 'list'>. Duplicate field name found: 'name'"

try:
    # Unsupported type
    schema = to_pyspark_schema({"field": some_unsupported_type})
except UnsupportedTypeError as e:
    print(f"Unsupported type: {e}")
    # Output includes list of supported types
```

## API Reference

### `to_pyspark_schema(polars_schema)`

Convert a Polars schema to a PySpark `StructType`.

**Parameters:**
- `polars_schema`: Polars schema in any supported format:
  - `pl.Schema` object
  - `dict[str, pl.DataType]`: Dictionary mapping field names to types
  - `Iterable[tuple[str, pl.DataType]]`: Iterable of (field_name, type) tuples (e.g., list or tuple of tuples)

**Returns:**
- `pyspark.sql.types.StructType`: PySpark schema

**Raises:**
- `SchemaError`: If the schema structure is invalid
- `UnsupportedTypeError`: If a type cannot be converted

**Example:**
```python
import polars as pl
from charmander import to_pyspark_schema

# All three formats work:
schema1 = {"name": pl.String, "age": pl.Int32}
schema2 = pl.Schema({"name": pl.String, "age": pl.Int32})
schema3 = [("name", pl.String), ("age", pl.Int32)]

pyspark_schema = to_pyspark_schema(schema1)  # or schema2, or schema3
```

### `to_polars_schema(pyspark_schema)`

Convert a PySpark `StructType` to a Polars schema.

**Parameters:**
- `pyspark_schema` (`pyspark.sql.types.StructType`): PySpark schema

**Returns:**
- `pl.Schema`: Polars Schema object mapping field names to Polars types

**Raises:**
- `SchemaError`: If the schema structure is invalid
- `UnsupportedTypeError`: If a type cannot be converted

**Example:**
```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from charmander import to_polars_schema

pyspark_schema = StructType([
    StructField("name", StringType()),
    StructField("age", IntegerType())
])

polars_schema = to_polars_schema(pyspark_schema)
# Returns pl.Schema object - use directly with Polars DataFrames
df = pl.DataFrame({}, schema=polars_schema)
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Project Structure

```
charmander/
├── charmander/
│   ├── __init__.py          # Public API
│   ├── converters.py         # Core conversion functions
│   ├── type_mappings.py      # Type mapping dictionaries
│   └── errors.py             # Custom exceptions
├── tests/
│   ├── test_converters.py    # Conversion tests
│   └── test_type_mappings.py # Type mapping tests
└── pyproject.toml            # Package configuration
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Inspiration

This project is inspired by [poldantic](https://github.com/eddiethedean/poldantic), which provides similar functionality for converting between Pydantic models and Polars schemas.

