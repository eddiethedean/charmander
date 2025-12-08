"""Charmander: Convert between Polars schemas and PySpark schemas."""

__version__ = "0.2.0"

from charmander.converters import to_pyspark_schema, to_polars_schema
from charmander.errors import ConversionError, UnsupportedTypeError, SchemaError

__all__ = [
    "to_pyspark_schema",
    "to_polars_schema",
    "ConversionError",  # Base exception for all conversion errors
    "UnsupportedTypeError",  # Raised when a type cannot be converted
    "SchemaError",  # Raised when a schema structure is invalid
]
