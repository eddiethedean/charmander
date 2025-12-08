"""Type mappings between Polars and PySpark schemas."""

from typing import TYPE_CHECKING, Dict, Type, Any

if TYPE_CHECKING:
    import polars as pl
    from pyspark.sql import types as spark_types
else:
    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore[assignment]

    try:
        from pyspark.sql import types as spark_types
    except ImportError:
        spark_types = None  # type: ignore[assignment]

from charmander.errors import UnsupportedTypeError


# Mapping from Polars data types to PySpark data types
POLARS_TO_PYSPARK: Dict[Type, Type] = {}

# Mapping from PySpark data types to Polars data types
PYSPARK_TO_POLARS: Dict[Type, Type] = {}


def _init_mappings():
    """Initialize the type mappings."""
    global POLARS_TO_PYSPARK, PYSPARK_TO_POLARS

    if pl is None or spark_types is None:
        return

    # Primitive numeric types
    POLARS_TO_PYSPARK[pl.Int8] = spark_types.ByteType
    POLARS_TO_PYSPARK[pl.Int16] = spark_types.ShortType
    POLARS_TO_PYSPARK[pl.Int32] = spark_types.IntegerType
    POLARS_TO_PYSPARK[pl.Int64] = spark_types.LongType
    POLARS_TO_PYSPARK[pl.UInt8] = spark_types.ShortType  # UInt8 fits in ShortType
    POLARS_TO_PYSPARK[pl.UInt16] = spark_types.IntegerType  # UInt16 fits in IntegerType
    POLARS_TO_PYSPARK[pl.UInt32] = spark_types.LongType  # UInt32 fits in LongType
    POLARS_TO_PYSPARK[pl.UInt64] = (
        spark_types.LongType
    )  # PySpark doesn't have UInt64, use LongType
    POLARS_TO_PYSPARK[pl.Int128] = (
        spark_types.DecimalType
    )  # PySpark doesn't support Int128, use DecimalType
    POLARS_TO_PYSPARK[pl.Float32] = spark_types.FloatType
    POLARS_TO_PYSPARK[pl.Float64] = spark_types.DoubleType

    # Boolean
    POLARS_TO_PYSPARK[pl.Boolean] = spark_types.BooleanType

    # String types
    POLARS_TO_PYSPARK[pl.String] = spark_types.StringType
    POLARS_TO_PYSPARK[pl.Utf8] = spark_types.StringType
    POLARS_TO_PYSPARK[pl.Categorical] = (
        spark_types.StringType
    )  # Categorical maps to StringType
    POLARS_TO_PYSPARK[pl.Enum] = spark_types.StringType  # Enum maps to StringType

    # Date and time types
    POLARS_TO_PYSPARK[pl.Date] = spark_types.DateType
    POLARS_TO_PYSPARK[pl.Datetime] = spark_types.TimestampType
    POLARS_TO_PYSPARK[pl.Time] = spark_types.TimestampType  # Time maps to TimestampType
    POLARS_TO_PYSPARK[pl.Duration] = spark_types.StringType  # Duration as string

    # Decimal and Binary types
    POLARS_TO_PYSPARK[pl.Decimal] = spark_types.DecimalType
    POLARS_TO_PYSPARK[pl.Binary] = spark_types.BinaryType

    # Null type
    POLARS_TO_PYSPARK[pl.Null] = spark_types.NullType

    # Complex types are handled specially in converters
    # List, Struct, Object, Map are converted recursively

    # Reverse mapping: PySpark to Polars
    PYSPARK_TO_POLARS[spark_types.ByteType] = pl.Int8
    PYSPARK_TO_POLARS[spark_types.ShortType] = (
        pl.Int32
    )  # Short maps to Int32 to be safe
    PYSPARK_TO_POLARS[spark_types.IntegerType] = pl.Int32
    PYSPARK_TO_POLARS[spark_types.LongType] = pl.Int64
    PYSPARK_TO_POLARS[spark_types.FloatType] = pl.Float32
    PYSPARK_TO_POLARS[spark_types.DoubleType] = pl.Float64
    PYSPARK_TO_POLARS[spark_types.BooleanType] = pl.Boolean
    PYSPARK_TO_POLARS[spark_types.StringType] = pl.String
    PYSPARK_TO_POLARS[spark_types.VarcharType] = pl.String  # Varchar maps to String
    PYSPARK_TO_POLARS[spark_types.CharType] = pl.String  # Char maps to String
    PYSPARK_TO_POLARS[spark_types.DateType] = pl.Date
    PYSPARK_TO_POLARS[spark_types.TimestampType] = pl.Datetime
    PYSPARK_TO_POLARS[spark_types.TimestampNTZType] = (
        pl.Datetime
    )  # Timestamp without timezone
    PYSPARK_TO_POLARS[spark_types.DecimalType] = pl.Decimal
    PYSPARK_TO_POLARS[spark_types.BinaryType] = pl.Binary
    PYSPARK_TO_POLARS[spark_types.NullType] = pl.Null

    # Complex types are handled specially in converters


# Initialize mappings on import
_init_mappings()


def get_pyspark_type(polars_type: Any) -> Type:  # type: ignore[no-any-unused]
    # polars_type can be a class (pl.String) or instance (pl.Decimal(...), pl.List(...))
    # so Any is used to accept both
    """
    Get the corresponding PySpark type for a Polars type.

    Args:
        polars_type: Polars data type instance or class

    Returns:
        PySpark type class

    Raises:
        UnsupportedTypeError: If the Polars type cannot be mapped
    """
    if spark_types is None:
        raise ImportError("pyspark is not installed")

    # Handle both class and instance
    polars_type_class = (
        type(polars_type) if not isinstance(polars_type, type) else polars_type
    )

    # Handle complex types
    if pl is not None:
        if isinstance(polars_type, pl.List):
            # List types are handled in converters recursively
            return spark_types.ArrayType
        elif isinstance(polars_type, pl.Struct):
            # Struct types are handled in converters recursively
            return spark_types.StructType
        elif isinstance(polars_type, pl.Object):
            # Object type maps to StringType as fallback
            return spark_types.StringType

    # Check direct mapping
    if polars_type_class in POLARS_TO_PYSPARK:
        return POLARS_TO_PYSPARK[polars_type_class]

    # Provide helpful error message with suggestions
    supported = [str(t.__name__) for t in POLARS_TO_PYSPARK.keys()]
    raise UnsupportedTypeError(
        f"Unsupported Polars type: {polars_type_class}. "
        f"Supported types: {', '.join(supported)}. "
        "If you're using a complex type (List, Struct, Map), ensure it's used correctly."
    )


def get_polars_type(pyspark_type: Any) -> Type:  # type: ignore[no-any-unused]
    # pyspark_type can be a class (StringType) or instance (ArrayType(...), StructType(...))
    # so Any is used to accept both
    """
    Get the corresponding Polars type for a PySpark type.

    Args:
        pyspark_type: PySpark data type instance or class

    Returns:
        Polars type class

    Raises:
        UnsupportedTypeError: If the PySpark type cannot be mapped
    """
    if pl is None:
        raise ImportError("polars is not installed")

    # Handle both class and instance
    pyspark_type_class = (
        type(pyspark_type) if not isinstance(pyspark_type, type) else pyspark_type
    )

    # Handle complex types
    if spark_types is not None:
        if isinstance(pyspark_type, spark_types.ArrayType):
            # Array types are handled in converters recursively
            return pl.List
        elif isinstance(pyspark_type, spark_types.StructType):
            # Struct types are handled in converters recursively
            return pl.Struct
        elif isinstance(pyspark_type, spark_types.MapType):
            # Map types are handled in converters recursively
            return pl.Struct  # Maps are represented as Struct in Polars

    # Check direct mapping
    if pyspark_type_class in PYSPARK_TO_POLARS:
        return PYSPARK_TO_POLARS[pyspark_type_class]

    # Provide helpful error message with suggestions
    supported = [str(t.__name__) for t in PYSPARK_TO_POLARS.keys()]
    raise UnsupportedTypeError(
        f"Unsupported PySpark type: {pyspark_type_class}. "
        f"Supported types: {', '.join(supported)}. "
        "If you're using a complex type (ArrayType, StructType, MapType), ensure it's used correctly."
    )
