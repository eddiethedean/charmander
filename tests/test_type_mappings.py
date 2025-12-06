"""Tests for type mappings."""

import pytest

try:
    import polars as pl
except ImportError:
    pytest.skip("polars not installed", allow_module_level=True)

try:
    from pyspark.sql import types as spark_types
except ImportError:
    pytest.skip("pyspark not installed", allow_module_level=True)

from charmander.type_mappings import get_pyspark_type, get_polars_type
from charmander.errors import UnsupportedTypeError


class TestPolarsToPySparkMappings:
    """Test mappings from Polars to PySpark types."""

    def test_primitive_int_types(self):
        """Test primitive integer type mappings."""
        assert get_pyspark_type(pl.Int8) == spark_types.ByteType
        assert get_pyspark_type(pl.Int16) == spark_types.ShortType
        assert get_pyspark_type(pl.Int32) == spark_types.IntegerType
        assert get_pyspark_type(pl.Int64) == spark_types.LongType

    def test_primitive_float_types(self):
        """Test primitive float type mappings."""
        assert get_pyspark_type(pl.Float32) == spark_types.FloatType
        assert get_pyspark_type(pl.Float64) == spark_types.DoubleType

    def test_boolean_type(self):
        """Test boolean type mapping."""
        assert get_pyspark_type(pl.Boolean) == spark_types.BooleanType

    def test_string_types(self):
        """Test string type mappings."""
        assert get_pyspark_type(pl.String) == spark_types.StringType
        assert get_pyspark_type(pl.Utf8) == spark_types.StringType

    def test_date_time_types(self):
        """Test date and time type mappings."""
        assert get_pyspark_type(pl.Date) == spark_types.DateType
        assert get_pyspark_type(pl.Datetime) == spark_types.TimestampType

    def test_unsupported_type(self):
        """Test that unsupported types raise UnsupportedTypeError."""
        with pytest.raises(UnsupportedTypeError):
            get_pyspark_type(int)  # Python int is not a Polars type

    def test_list_type_returns_array_type(self):
        """Test that List types return ArrayType class."""
        list_type = pl.List(pl.String)
        assert get_pyspark_type(list_type) == spark_types.ArrayType

    def test_struct_type_returns_struct_type(self):
        """Test that Struct types return StructType class."""
        struct_type = pl.Struct(
            [pl.Field("name", pl.String), pl.Field("age", pl.Int32)]
        )
        assert get_pyspark_type(struct_type) == spark_types.StructType


class TestPySparkToPolarsMappings:
    """Test mappings from PySpark to Polars types."""

    def test_primitive_int_types(self):
        """Test primitive integer type mappings."""
        assert get_polars_type(spark_types.ByteType) == pl.Int8
        assert get_polars_type(spark_types.IntegerType) == pl.Int32
        assert get_polars_type(spark_types.LongType) == pl.Int64

    def test_primitive_float_types(self):
        """Test primitive float type mappings."""
        assert get_polars_type(spark_types.FloatType) == pl.Float32
        assert get_polars_type(spark_types.DoubleType) == pl.Float64

    def test_boolean_type(self):
        """Test boolean type mapping."""
        assert get_polars_type(spark_types.BooleanType) == pl.Boolean

    def test_string_type(self):
        """Test string type mapping."""
        assert get_polars_type(spark_types.StringType) == pl.String

    def test_date_time_types(self):
        """Test date and time type mappings."""
        assert get_polars_type(spark_types.DateType) == pl.Date
        assert get_polars_type(spark_types.TimestampType) == pl.Datetime

    def test_unsupported_type(self):
        """Test that unsupported types raise UnsupportedTypeError."""
        with pytest.raises(UnsupportedTypeError):
            get_polars_type(int)  # Python int is not a PySpark type

    def test_array_type_returns_list(self):
        """Test that ArrayType returns List class."""
        array_type = spark_types.ArrayType(spark_types.StringType())
        assert get_polars_type(array_type) == pl.List

    def test_struct_type_returns_struct(self):
        """Test that StructType returns Struct class."""
        struct_type = spark_types.StructType(
            [
                spark_types.StructField("name", spark_types.StringType()),
                spark_types.StructField("age", spark_types.IntegerType()),
            ]
        )
        assert get_polars_type(struct_type) == pl.Struct
