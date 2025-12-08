# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- Initial release of charmander
- Support for converting Polars schemas to PySpark schemas (`to_pyspark_schema`)
- Support for converting PySpark schemas to Polars schemas (`to_polars_schema`)
- Support for three Polars schema input formats:
  - `pl.Schema` objects
  - `dict[str, pl.DataType]` dictionaries
  - `Iterable[tuple[str, pl.DataType]]` (list/tuple of tuples)
- Comprehensive type mappings for all primitive and complex types
- Support for nested structures (structs, arrays, maps)
- Custom exception classes: `ConversionError`, `UnsupportedTypeError`, `SchemaError`
- Input validation for duplicate field names, empty field names, and invalid types
- Full test coverage with 67 tests

### Features
- Bidirectional schema conversion between Polars and PySpark
- Handles all Polars primitive types (Int8-Int128, UInt8-UInt64, Float32/64, Boolean, String, Date, Datetime, Decimal, Binary, Null, Categorical, Enum)
- Handles all PySpark primitive types (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, BooleanType, StringType, DateType, TimestampType, DecimalType, BinaryType, NullType, VarcharType, CharType, TimestampNTZType)
- Support for complex types: Lists/Arrays, Structs, Maps (converted to Struct in Polars)
- Returns `pl.Schema` objects for direct use with Polars DataFrames

### Limitations
- Decimal precision/scale information is not preserved (defaults used)
- Datetime timezone information is not preserved
- MapType is converted to Struct in Polars (with 'key' and 'value' fields)
- Nullability information is not preserved (all fields nullable in PySpark, all fields can contain nulls in Polars)

