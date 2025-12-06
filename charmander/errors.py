"""Custom exception classes for charmander."""


class ConversionError(Exception):
    """Base exception for conversion issues."""

    pass


class UnsupportedTypeError(ConversionError):
    """Raised when a type cannot be converted."""

    pass


class SchemaError(ConversionError):
    """Raised when a schema structure is invalid."""

    pass
