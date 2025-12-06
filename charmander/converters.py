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

    Note:
        All fields are created with nullable=True, as Polars schemas don't
        explicitly track nullability at the schema definition level.

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

    # Validate and convert fields
    field_names = []
    fields = []
    for field_name, field_type in schema_dict.items():
        # Validate field name
        if not isinstance(field_name, str):
            raise SchemaError(
                f"Field name must be a string, got {type(field_name)}: {field_name!r}"
            )
        if not field_name:
            raise SchemaError("Field names cannot be empty strings")
        if field_name in field_names:
            raise SchemaError(f"Duplicate field name found: {field_name!r}")

        # Validate field type
        if field_type is None:
            raise SchemaError(f"Field type cannot be None for field {field_name!r}")

        field_names.append(field_name)
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
            # DecimalType requires special handling - use default precision/scale
            # Note: Polars Decimal doesn't expose precision/scale in schema definition
            from pyspark.sql.types import DecimalType

            if spark_type_class is DecimalType:
                spark_type = DecimalType()  # Default precision=10, scale=0
            else:
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

    Note:
        The nullable attribute from PySpark StructField is not preserved,
        as Polars schemas don't track nullability at the schema definition level.
        All Polars fields can contain nulls by default.

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
    field_names = set()
    for field in pyspark_schema.fields:
        # Validate field name
        if not isinstance(field.name, str):
            raise SchemaError(
                f"Field name must be a string, got {type(field.name)}: {field.name!r}"
            )
        if not field.name:
            raise SchemaError("Field names cannot be empty strings")
        if field.name in field_names:
            raise SchemaError(f"Duplicate field name found: {field.name!r}")

        field_names.add(field.name)
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

    # Handle DecimalType - Polars Decimal doesn't preserve precision/scale
    from pyspark.sql.types import DecimalType

    if isinstance(spark_type, DecimalType):
        return pl.Decimal

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

    # Handle DecimalType - Polars Decimal doesn't preserve precision/scale
    from pyspark.sql.types import DecimalType

    if isinstance(spark_type, DecimalType):
        return pl.Decimal

    # Handle primitive types
    try:
        polars_type_class = get_polars_type(spark_type)
        return polars_type_class
    except UnsupportedTypeError:
        raise UnsupportedTypeError(
            f"Cannot convert PySpark type {type(spark_type)} to Polars type"
        )
