"""Core conversion functions between Polars and PySpark schemas."""

from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    import polars as pl
    from pyspark.sql.types import (
        StructType,
        StructField,
        ArrayType,
        MapType,
        DataType,
    )
else:
    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore[assignment]

    try:
        from pyspark.sql.types import (
            StructType,
            StructField,
            ArrayType,
            MapType,
            DataType,
        )
    except ImportError:
        StructType = None  # type: ignore[assignment]
        StructField = None  # type: ignore[assignment]
        ArrayType = None  # type: ignore[assignment]
        MapType = None  # type: ignore[assignment]
        DataType = None  # type: ignore[assignment]

from charmander.errors import SchemaError, UnsupportedTypeError
from charmander.type_mappings import get_pyspark_type, get_polars_type


def to_pyspark_schema(polars_schema: Union[Dict[str, Any], pl.Schema]) -> StructType:
    """
    Convert a Polars schema to a PySpark StructType.

    Args:
        polars_schema: Polars schema as a dictionary mapping field names to types,
                      or a polars.Schema object

    Returns:
        PySpark StructType representing the converted schema

    Raises:
        SchemaError: If the schema structure is invalid
        UnsupportedTypeError: If a type cannot be converted

    Example:
        >>> import polars as pl
        >>> schema = {"name": pl.String, "age": pl.Int32, "score": pl.Float64}
        >>> pyspark_schema = to_pyspark_schema(schema)
    """
    if pl is None:
        raise ImportError("polars is not installed")
    if StructType is None:
        raise ImportError("pyspark is not installed")

    # Convert polars.Schema to dict if needed
    if isinstance(polars_schema, pl.Schema):
        schema_dict = dict(polars_schema)
    elif isinstance(polars_schema, dict):
        schema_dict = polars_schema
    else:
        raise SchemaError(
            f"Invalid schema type: {type(polars_schema)}. Expected dict or pl.Schema"
        )

    fields = []
    for field_name, field_type in schema_dict.items():
        pyspark_field = _convert_polars_type_to_pyspark_field(field_name, field_type)
        fields.append(pyspark_field)

    return StructType(fields)


def _convert_polars_type_to_pyspark_field(
    field_name: str, polars_type: Any, nullable: bool = True
) -> StructField:
    """
    Convert a Polars type to a PySpark StructField.

    Args:
        field_name: Name of the field
        polars_type: Polars data type (class or instance)
        nullable: Whether the field is nullable

    Returns:
        PySpark StructField
    """
    # Handle Optional types
    if hasattr(polars_type, "__origin__") and hasattr(polars_type, "__args__"):
        # This is a typing.Optional or Union type
        if polars_type.__origin__ is Union:
            args = polars_type.__args__
            if len(args) == 2 and type(None) in args:
                # It's an Optional
                non_null_type = args[0] if args[1] is type(None) else args[1]
                return _convert_polars_type_to_pyspark_field(
                    field_name, non_null_type, nullable=True
                )

    # Handle List/Array types
    if pl is not None and isinstance(polars_type, pl.List):
        element_type = polars_type.inner
        element_field = _convert_polars_type_to_pyspark_field(
            "element", element_type, nullable=True
        )
        spark_array_type = ArrayType(element_field.dataType, containsNull=True)
        return StructField(field_name, spark_array_type, nullable=nullable)

    # Handle Struct types
    if pl is not None and isinstance(polars_type, pl.Struct):
        struct_fields = []
        for field in polars_type.fields:
            nested_field = _convert_polars_type_to_pyspark_field(
                field.name, field.dtype
            )
            struct_fields.append(nested_field)
        spark_struct_type = StructType(struct_fields)
        return StructField(field_name, spark_struct_type, nullable=nullable)

    # Handle Object types - convert to StringType
    if pl is not None and isinstance(polars_type, pl.Object):
        from pyspark.sql.types import StringType

        return StructField(field_name, StringType(), nullable=nullable)

    # Handle primitive types
    try:
        spark_type_class = get_pyspark_type(polars_type)
        # If it's a class, instantiate it
        if isinstance(spark_type_class, type):
            spark_type = spark_type_class()
        else:
            spark_type = spark_type_class
        return StructField(field_name, spark_type, nullable=nullable)
    except UnsupportedTypeError:
        raise UnsupportedTypeError(
            f"Cannot convert Polars type {type(polars_type)} to PySpark type for field '{field_name}'"
        )


def to_polars_schema(pyspark_schema: StructType) -> Dict[str, Any]:
    """
    Convert a PySpark StructType to a Polars schema dictionary.

    Args:
        pyspark_schema: PySpark StructType to convert

    Returns:
        Dictionary mapping field names to Polars types

    Raises:
        SchemaError: If the schema structure is invalid
        UnsupportedTypeError: If a type cannot be converted

    Example:
        >>> from pyspark.sql.types import StructType, StructField, StringType, IntegerType
        >>> schema = StructType([
        ...     StructField("name", StringType()),
        ...     StructField("age", IntegerType())
        ... ])
        >>> polars_schema = to_polars_schema(schema)
    """
    if pl is None:
        raise ImportError("polars is not installed")
    if StructType is None:
        raise ImportError("pyspark is not installed")

    if not isinstance(pyspark_schema, StructType):
        raise SchemaError(f"Expected StructType, got {type(pyspark_schema)}")

    schema_dict = {}
    for field in pyspark_schema.fields:
        polars_type = _convert_pyspark_field_to_polars_type(field)
        schema_dict[field.name] = polars_type

    return schema_dict


def _convert_pyspark_field_to_polars_type(field: StructField) -> Any:
    """
    Convert a PySpark StructField to a Polars type.

    Args:
        field: PySpark StructField to convert

    Returns:
        Polars data type

    Raises:
        UnsupportedTypeError: If the type cannot be converted
    """
    spark_type = field.dataType

    # Handle ArrayType
    if isinstance(spark_type, ArrayType):
        element_type = _convert_pyspark_type_to_polars_type(spark_type.elementType)
        return pl.List(element_type)

    # Handle MapType - convert to Struct in Polars
    if isinstance(spark_type, MapType):
        key_type = _convert_pyspark_type_to_polars_type(spark_type.keyType)
        value_type = _convert_pyspark_type_to_polars_type(spark_type.valueType)
        # Represent Map as Struct with 'key' and 'value' fields
        # This is a limitation: Polars doesn't have native Map type
        return pl.Struct(
            [
                pl.Field("key", key_type),
                pl.Field("value", value_type),
            ]
        )

    # Handle StructType
    if isinstance(spark_type, StructType):
        struct_fields = []
        for nested_field in spark_type.fields:
            nested_type = _convert_pyspark_field_to_polars_type(nested_field)
            struct_fields.append(pl.Field(nested_field.name, nested_type))
        return pl.Struct(struct_fields)

    # Handle primitive types
    try:
        polars_type_class = get_polars_type(spark_type)
        return polars_type_class
    except UnsupportedTypeError:
        raise UnsupportedTypeError(
            f"Cannot convert PySpark type {type(spark_type)} to Polars type for field '{field.name}'"
        )


def _convert_pyspark_type_to_polars_type(spark_type: DataType) -> Any:
    """
    Convert a PySpark DataType to a Polars type (helper for nested types).

    Args:
        spark_type: PySpark DataType to convert

    Returns:
        Polars data type

    Raises:
        UnsupportedTypeError: If the type cannot be converted
    """
    # Handle ArrayType
    if isinstance(spark_type, ArrayType):
        element_type = _convert_pyspark_type_to_polars_type(spark_type.elementType)
        return pl.List(element_type)

    # Handle MapType
    if isinstance(spark_type, MapType):
        key_type = _convert_pyspark_type_to_polars_type(spark_type.keyType)
        value_type = _convert_pyspark_type_to_polars_type(spark_type.valueType)
        return pl.Struct(
            [
                pl.Field("key", key_type),
                pl.Field("value", value_type),
            ]
        )

    # Handle StructType
    if isinstance(spark_type, StructType):
        struct_fields = []
        for nested_field in spark_type.fields:
            nested_type = _convert_pyspark_field_to_polars_type(nested_field)
            struct_fields.append(pl.Field(nested_field.name, nested_type))
        return pl.Struct(struct_fields)

    # Handle primitive types
    try:
        polars_type_class = get_polars_type(spark_type)
        return polars_type_class
    except UnsupportedTypeError:
        raise UnsupportedTypeError(
            f"Cannot convert PySpark type {type(spark_type)} to Polars type"
        )
