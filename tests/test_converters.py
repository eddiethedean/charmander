"""Tests for schema converters."""

import pytest

try:
    import polars as pl
except ImportError:
    pytest.skip("polars not installed", allow_module_level=True)

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        FloatType,
        DoubleType,
        BooleanType,
        ArrayType,
        MapType,
        ByteType,
        LongType,
    )
except ImportError:
    pytest.skip("pyspark not installed", allow_module_level=True)

from charmander import to_pyspark_schema, to_polars_schema
from charmander.errors import SchemaError


class TestPolarsToPySparkConversion:
    """Test conversion from Polars schemas to PySpark schemas."""

    def test_simple_schema_conversion(self):
        """Test conversion of a simple schema with primitive types."""
        polars_schema = {
            "name": pl.String,
            "age": pl.Int32,
            "score": pl.Float64,
            "is_active": pl.Boolean,
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 4
        assert pyspark_schema.fields[0].name == "name"
        assert isinstance(pyspark_schema.fields[0].dataType, StringType)
        assert pyspark_schema.fields[1].name == "age"
        assert isinstance(pyspark_schema.fields[1].dataType, IntegerType)
        assert pyspark_schema.fields[2].name == "score"
        assert isinstance(pyspark_schema.fields[2].dataType, DoubleType)
        assert pyspark_schema.fields[3].name == "is_active"
        assert isinstance(pyspark_schema.fields[3].dataType, BooleanType)

    def test_polars_schema_object_conversion(self):
        """Test conversion using polars.Schema object."""
        polars_schema = pl.Schema(
            {
                "id": pl.Int64,
                "email": pl.String,
            }
        )
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 2

    def test_nested_struct_conversion(self):
        """Test conversion of nested struct types."""
        polars_schema = {
            "name": pl.String,
            "address": pl.Struct(
                [
                    pl.Field("street", pl.String),
                    pl.Field("city", pl.String),
                    pl.Field("zip", pl.Int32),
                ]
            ),
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        address_field = pyspark_schema.fields[1]
        assert address_field.name == "address"
        assert isinstance(address_field.dataType, StructType)
        assert len(address_field.dataType.fields) == 3

    def test_array_conversion(self):
        """Test conversion of array/list types."""
        polars_schema = {
            "tags": pl.List(pl.String),
            "scores": pl.List(pl.Float64),
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        tags_field = pyspark_schema.fields[0]
        assert tags_field.name == "tags"
        assert isinstance(tags_field.dataType, ArrayType)
        assert isinstance(tags_field.dataType.elementType, StringType)

        scores_field = pyspark_schema.fields[1]
        assert scores_field.name == "scores"
        assert isinstance(scores_field.dataType, ArrayType)
        assert isinstance(scores_field.dataType.elementType, DoubleType)

    def test_nested_array_conversion(self):
        """Test conversion of nested arrays."""
        polars_schema = {
            "matrix": pl.List(pl.List(pl.Float64)),
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        matrix_field = pyspark_schema.fields[0]
        assert isinstance(matrix_field.dataType, ArrayType)
        inner_array = matrix_field.dataType.elementType
        assert isinstance(inner_array, ArrayType)
        assert isinstance(inner_array.elementType, DoubleType)

    def test_complex_nested_structure(self):
        """Test conversion of complex nested structures."""
        polars_schema = {
            "user": pl.Struct(
                [
                    pl.Field("name", pl.String),
                    pl.Field(
                        "contact",
                        pl.Struct(
                            [
                                pl.Field("email", pl.String),
                                pl.Field("phones", pl.List(pl.String)),
                            ]
                        ),
                    ),
                ]
            ),
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        user_field = pyspark_schema.fields[0]
        assert isinstance(user_field.dataType, StructType)
        contact_field = user_field.dataType.fields[1]
        assert isinstance(contact_field.dataType, StructType)
        phones_field = contact_field.dataType.fields[1]
        assert isinstance(phones_field.dataType, ArrayType)

    def test_invalid_schema_type(self):
        """Test that invalid schema types raise SchemaError."""
        with pytest.raises(SchemaError):
            to_pyspark_schema("not a schema")

    def test_all_numeric_types(self):
        """Test all numeric type conversions."""
        polars_schema = {
            "int8": pl.Int8,
            "int16": pl.Int16,
            "int32": pl.Int32,
            "int64": pl.Int64,
            "uint8": pl.UInt8,
            "float32": pl.Float32,
            "float64": pl.Float64,
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 7


class TestPySparkToPolarsConversion:
    """Test conversion from PySpark schemas to Polars schemas."""

    def test_simple_schema_conversion(self):
        """Test conversion of a simple schema with primitive types."""
        pyspark_schema = StructType(
            [
                StructField("name", StringType()),
                StructField("age", IntegerType()),
                StructField("score", DoubleType()),
                StructField("is_active", BooleanType()),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, dict)
        assert "name" in polars_schema
        assert polars_schema["name"] == pl.String
        assert polars_schema["age"] == pl.Int32
        assert polars_schema["score"] == pl.Float64
        assert polars_schema["is_active"] == pl.Boolean

    def test_nested_struct_conversion(self):
        """Test conversion of nested struct types."""
        pyspark_schema = StructType(
            [
                StructField("name", StringType()),
                StructField(
                    "address",
                    StructType(
                        [
                            StructField("street", StringType()),
                            StructField("city", StringType()),
                            StructField("zip", IntegerType()),
                        ]
                    ),
                ),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, dict)
        assert isinstance(polars_schema["address"], pl.Struct)
        assert len(polars_schema["address"].fields) == 3

    def test_array_conversion(self):
        """Test conversion of array types."""
        pyspark_schema = StructType(
            [
                StructField("tags", ArrayType(StringType())),
                StructField("scores", ArrayType(DoubleType())),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, dict)
        assert isinstance(polars_schema["tags"], pl.List)
        assert polars_schema["tags"].inner == pl.String
        assert isinstance(polars_schema["scores"], pl.List)
        assert polars_schema["scores"].inner == pl.Float64

    def test_nested_array_conversion(self):
        """Test conversion of nested arrays."""
        pyspark_schema = StructType(
            [
                StructField("matrix", ArrayType(ArrayType(DoubleType()))),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, dict)
        assert isinstance(polars_schema["matrix"], pl.List)
        assert isinstance(polars_schema["matrix"].inner, pl.List)
        assert polars_schema["matrix"].inner.inner == pl.Float64

    def test_map_conversion(self):
        """Test conversion of map types."""
        pyspark_schema = StructType(
            [
                StructField("metadata", MapType(StringType(), StringType())),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, dict)
        # MapType is converted to Struct with 'key' and 'value' fields
        assert isinstance(polars_schema["metadata"], pl.Struct)
        assert len(polars_schema["metadata"].fields) == 2

    def test_complex_nested_structure(self):
        """Test conversion of complex nested structures."""
        pyspark_schema = StructType(
            [
                StructField(
                    "user",
                    StructType(
                        [
                            StructField("name", StringType()),
                            StructField(
                                "contact",
                                StructType(
                                    [
                                        StructField("email", StringType()),
                                        StructField("phones", ArrayType(StringType())),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, dict)
        assert isinstance(polars_schema["user"], pl.Struct)
        contact_field = next(
            f for f in polars_schema["user"].fields if f.name == "contact"
        )
        assert isinstance(contact_field.dtype, pl.Struct)
        phones_field = next(f for f in contact_field.dtype.fields if f.name == "phones")
        assert isinstance(phones_field.dtype, pl.List)

    def test_invalid_schema_type(self):
        """Test that invalid schema types raise SchemaError."""
        with pytest.raises(SchemaError):
            to_polars_schema("not a schema")

    def test_all_numeric_types(self):
        """Test all numeric type conversions."""
        pyspark_schema = StructType(
            [
                StructField("byte", ByteType()),
                StructField("int", IntegerType()),
                StructField("long", LongType()),
                StructField("float", FloatType()),
                StructField("double", DoubleType()),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, dict)
        assert polars_schema["byte"] == pl.Int8
        assert polars_schema["int"] == pl.Int32
        assert polars_schema["long"] == pl.Int64
        assert polars_schema["float"] == pl.Float32
        assert polars_schema["double"] == pl.Float64


class TestRoundTripConversion:
    """Test round-trip conversions (Polars -> PySpark -> Polars)."""

    def test_simple_schema_round_trip(self):
        """Test round-trip conversion of a simple schema."""
        original = {
            "name": pl.String,
            "age": pl.Int32,
            "score": pl.Float64,
        }
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)

        assert converted_back["name"] == original["name"]
        assert converted_back["age"] == original["age"]
        assert converted_back["score"] == original["score"]

    def test_nested_struct_round_trip(self):
        """Test round-trip conversion of nested struct."""
        original = {
            "user": pl.Struct(
                [
                    pl.Field("name", pl.String),
                    pl.Field("age", pl.Int32),
                ]
            ),
        }
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)

        assert isinstance(converted_back["user"], pl.Struct)
        assert len(converted_back["user"].fields) == 2

    def test_array_round_trip(self):
        """Test round-trip conversion of arrays."""
        original = {
            "tags": pl.List(pl.String),
            "numbers": pl.List(pl.Int64),
        }
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)

        assert isinstance(converted_back["tags"], pl.List)
        assert converted_back["tags"].inner == pl.String
        assert isinstance(converted_back["numbers"], pl.List)
        assert converted_back["numbers"].inner == pl.Int64
