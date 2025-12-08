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
        DecimalType,
        BinaryType,
        NullType,
        TimestampNTZType,
        VarcharType,
        CharType,
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

    def test_iterable_tuple_format_conversion(self):
        """Test conversion using list of tuples format."""
        polars_schema = [
            ("name", pl.String),
            ("age", pl.Int32),
            ("score", pl.Float64),
        ]
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 3
        assert pyspark_schema.fields[0].name == "name"
        assert isinstance(pyspark_schema.fields[0].dataType, StringType)
        assert pyspark_schema.fields[1].name == "age"
        assert isinstance(pyspark_schema.fields[1].dataType, IntegerType)
        assert pyspark_schema.fields[2].name == "score"
        assert isinstance(pyspark_schema.fields[2].dataType, DoubleType)

    def test_iterable_tuple_of_tuples_format(self):
        """Test conversion using tuple of tuples format."""
        polars_schema = (
            ("id", pl.Int64),
            ("email", pl.String),
        )
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 2
        assert pyspark_schema.fields[0].name == "id"
        assert isinstance(pyspark_schema.fields[0].dataType, LongType)
        assert pyspark_schema.fields[1].name == "email"
        assert isinstance(pyspark_schema.fields[1].dataType, StringType)

    def test_iterable_format_with_nested_structures(self):
        """Test iterable format with nested struct types."""
        polars_schema = [
            ("name", pl.String),
            (
                "address",
                pl.Struct(
                    [
                        pl.Field("street", pl.String),
                        pl.Field("city", pl.String),
                        pl.Field("zip", pl.Int32),
                    ]
                ),
            ),
        ]
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 2
        address_field = pyspark_schema.fields[1]
        assert address_field.name == "address"
        assert isinstance(address_field.dataType, StructType)
        assert len(address_field.dataType.fields) == 3

    def test_all_three_formats_produce_identical_results(self):
        """Test that all three schema formats produce identical PySpark schemas."""
        # Format 1: Dictionary
        schema_dict = {
            "name": pl.String,
            "age": pl.Int32,
            "score": pl.Float64,
        }
        pyspark_dict = to_pyspark_schema(schema_dict)

        # Format 2: pl.Schema object
        schema_schema = pl.Schema(schema_dict)
        pyspark_schema = to_pyspark_schema(schema_schema)

        # Format 3: List of tuples
        schema_list = [("name", pl.String), ("age", pl.Int32), ("score", pl.Float64)]
        pyspark_list = to_pyspark_schema(schema_list)

        # All should produce identical results
        assert (
            len(pyspark_dict.fields)
            == len(pyspark_schema.fields)
            == len(pyspark_list.fields)
        )
        for i in range(len(pyspark_dict.fields)):
            assert (
                pyspark_dict.fields[i].name
                == pyspark_schema.fields[i].name
                == pyspark_list.fields[i].name
            )
            assert (
                type(pyspark_dict.fields[i].dataType)
                is type(pyspark_schema.fields[i].dataType)
                is type(pyspark_list.fields[i].dataType)
            )

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

    def test_invalid_iterable_format(self):
        """Test that invalid iterable formats raise SchemaError."""
        # Not a tuple
        with pytest.raises(SchemaError):
            to_pyspark_schema([("name", pl.String), "invalid"])

        # Tuple with wrong length
        with pytest.raises(SchemaError):
            to_pyspark_schema([("name", pl.String, "extra")])

        # Empty list should work (empty schema)
        empty_schema = []
        pyspark_schema = to_pyspark_schema(empty_schema)
        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 0

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

        assert isinstance(polars_schema, pl.Schema)
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

        assert isinstance(polars_schema, pl.Schema)
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

        assert isinstance(polars_schema, pl.Schema)
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

        assert isinstance(polars_schema, pl.Schema)
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

        assert isinstance(polars_schema, pl.Schema)
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

        assert isinstance(polars_schema, pl.Schema)
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

        assert isinstance(polars_schema, pl.Schema)
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


class TestNullableFields:
    """Test nullable field handling."""

    def test_pyspark_nullable_fields(self):
        """Test that PySpark nullable fields are handled."""
        pyspark_schema = StructType(
            [
                StructField("required", StringType(), nullable=False),
                StructField("optional", StringType(), nullable=True),
            ]
        )
        # Conversion should succeed (nullable is not preserved in Polars schema)
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, pl.Schema)
        assert polars_schema["required"] == pl.String
        assert polars_schema["optional"] == pl.String

    def test_polars_to_pyspark_all_nullable(self):
        """Test that Polars schemas create nullable PySpark fields."""
        polars_schema = {
            "name": pl.String,
            "age": pl.Int32,
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        # All fields should be nullable by default
        assert pyspark_schema.fields[0].nullable is True
        assert pyspark_schema.fields[1].nullable is True


class TestDecimalAndBinaryTypes:
    """Test Decimal and Binary type conversions."""

    def test_decimal_type_polars_to_pyspark(self):
        """Test conversion of Decimal type from Polars to PySpark."""
        polars_schema = {
            "price": pl.Decimal,
            "quantity": pl.Decimal,
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert isinstance(pyspark_schema.fields[0].dataType, DecimalType)
        assert isinstance(pyspark_schema.fields[1].dataType, DecimalType)

    def test_binary_type_polars_to_pyspark(self):
        """Test conversion of Binary type from Polars to PySpark."""
        polars_schema = {
            "data": pl.Binary,
            "blob": pl.Binary,
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert isinstance(pyspark_schema.fields[0].dataType, BinaryType)
        assert isinstance(pyspark_schema.fields[1].dataType, BinaryType)

    def test_decimal_type_pyspark_to_polars(self):
        """Test conversion of DecimalType from PySpark to Polars."""
        pyspark_schema = StructType(
            [
                StructField("price", DecimalType()),
                StructField("total", DecimalType(18, 2)),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, pl.Schema)
        assert polars_schema["price"] == pl.Decimal
        assert polars_schema["total"] == pl.Decimal

    def test_binary_type_pyspark_to_polars(self):
        """Test conversion of BinaryType from PySpark to Polars."""
        pyspark_schema = StructType(
            [
                StructField("data", BinaryType()),
                StructField("blob", BinaryType()),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, pl.Schema)
        assert polars_schema["data"] == pl.Binary
        assert polars_schema["blob"] == pl.Binary

    def test_decimal_round_trip(self):
        """Test round-trip conversion of Decimal type."""
        original = {"price": pl.Decimal}
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)

        assert converted_back["price"] == pl.Decimal

    def test_binary_round_trip(self):
        """Test round-trip conversion of Binary type."""
        original = {"data": pl.Binary}
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)

        assert converted_back["data"] == pl.Binary


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_schema_polars_to_pyspark(self):
        """Test conversion of empty Polars schema."""
        polars_schema = {}
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert len(pyspark_schema.fields) == 0

    def test_empty_schema_pyspark_to_polars(self):
        """Test conversion of empty PySpark schema."""
        pyspark_schema = StructType([])
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, pl.Schema)
        assert len(polars_schema) == 0

    def test_single_field_schema(self):
        """Test schema with a single field."""
        polars_schema = {"id": pl.Int64}
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert len(pyspark_schema.fields) == 1
        assert pyspark_schema.fields[0].name == "id"

    def test_all_fields_nullable_pyspark(self):
        """Test PySpark schema with all fields nullable."""
        pyspark_schema = StructType(
            [
                StructField("a", StringType(), nullable=True),
                StructField("b", IntegerType(), nullable=True),
                StructField("c", BooleanType(), nullable=True),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert len(polars_schema) == 3
        # Types can be either classes or instances (Decimal/Datetime are instances in pl.Schema)
        # Check that all values are valid Polars types
        from polars.datatypes import DataType

        assert all(
            isinstance(t, type) or isinstance(t, DataType)
            for t in polars_schema.values()
        )

    def test_no_fields_nullable_pyspark(self):
        """Test PySpark schema with no fields nullable."""
        pyspark_schema = StructType(
            [
                StructField("a", StringType(), nullable=False),
                StructField("b", IntegerType(), nullable=False),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        # Should convert successfully (nullability not preserved)
        assert len(polars_schema) == 2


class TestNewTypes:
    """Test new type mappings."""

    def test_null_type_polars_to_pyspark(self):
        """Test conversion of Null type from Polars to PySpark."""
        polars_schema = {"null_field": pl.Null}
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert isinstance(pyspark_schema.fields[0].dataType, NullType)

    def test_null_type_pyspark_to_polars(self):
        """Test conversion of NullType from PySpark to Polars."""
        pyspark_schema = StructType([StructField("null_field", NullType())])
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, pl.Schema)
        assert polars_schema["null_field"] == pl.Null

    def test_timestampntz_type_pyspark_to_polars(self):
        """Test conversion of TimestampNTZType from PySpark to Polars."""
        pyspark_schema = StructType([StructField("ts", TimestampNTZType())])
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, pl.Schema)
        assert polars_schema["ts"] == pl.Datetime

    def test_categorical_enum_polars_to_pyspark(self):
        """Test conversion of Categorical and Enum types from Polars to PySpark."""
        polars_schema = {
            "category": pl.Categorical,
            "enum_val": pl.Enum,
        }
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert isinstance(pyspark_schema.fields[0].dataType, StringType)
        assert isinstance(pyspark_schema.fields[1].dataType, StringType)

    def test_varchar_char_pyspark_to_polars(self):
        """Test conversion of VarcharType and CharType from PySpark to Polars."""
        pyspark_schema = StructType(
            [
                StructField("varchar_field", VarcharType(100)),
                StructField("char_field", CharType(10)),
            ]
        )
        polars_schema = to_polars_schema(pyspark_schema)

        assert isinstance(polars_schema, pl.Schema)
        assert polars_schema["varchar_field"] == pl.String
        assert polars_schema["char_field"] == pl.String

    def test_int128_polars_to_pyspark(self):
        """Test conversion of Int128 type from Polars to PySpark."""
        polars_schema = {"big_int": pl.Int128}
        pyspark_schema = to_pyspark_schema(polars_schema)

        assert isinstance(pyspark_schema, StructType)
        assert isinstance(pyspark_schema.fields[0].dataType, DecimalType)

    def test_new_types_round_trip(self):
        """Test round-trip conversion for new types."""
        # Null
        original = {"null_field": pl.Null}
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)
        assert converted_back["null_field"] == pl.Null

        # Categorical
        original = {"category": pl.Categorical}
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)
        assert converted_back["category"] == pl.String  # Categorical becomes String

        # Enum
        original = {"enum_val": pl.Enum}
        pyspark = to_pyspark_schema(original)
        converted_back = to_polars_schema(pyspark)
        assert converted_back["enum_val"] == pl.String  # Enum becomes String


class TestInputValidation:
    """Test input validation."""

    def test_duplicate_field_names_polars(self):
        """Test that duplicate field names raise SchemaError."""
        # Note: Python dicts can't have duplicate keys, so we test by creating
        # a schema that would have duplicates if we iterated through items
        # Our validation checks during iteration, so we can test by manually
        # checking field names as we process them
        # Since dict literals can't have duplicates, we test the validation
        # by ensuring our code would catch duplicates if they existed
        # In practice, this is tested via PySpark schema duplicates
        pass  # Validation is tested via PySpark duplicate test

    def test_duplicate_field_names_pyspark(self):
        """Test that duplicate field names in PySpark schema raise SchemaError."""
        pyspark_schema = StructType(
            [
                StructField("field", StringType()),
                StructField("field", IntegerType()),
            ]
        )
        with pytest.raises(SchemaError, match="Duplicate field name"):
            to_polars_schema(pyspark_schema)

    def test_empty_field_name_polars(self):
        """Test that empty field names raise SchemaError."""
        with pytest.raises(SchemaError, match="cannot be empty"):
            to_pyspark_schema({"": pl.String})

    def test_empty_field_name_pyspark(self):
        """Test that empty field names in PySpark raise SchemaError."""
        pyspark_schema = StructType([StructField("", StringType())])
        with pytest.raises(SchemaError, match="cannot be empty"):
            to_polars_schema(pyspark_schema)

    def test_none_field_type(self):
        """Test that None field types raise SchemaError."""
        with pytest.raises(SchemaError, match="cannot be None"):
            to_pyspark_schema({"field": None})

    def test_non_string_field_name_polars(self):
        """Test that non-string field names raise SchemaError."""
        with pytest.raises(SchemaError, match="must be a string"):
            to_pyspark_schema({123: pl.String})

    def test_non_string_field_name_pyspark(self):
        """Test that non-string field names in PySpark raise SchemaError."""
        # Create a StructField with a non-string name (this would need special handling)
        # Actually, PySpark StructField requires a string name, so this test may not be possible
        # But let's test that our validation still works
        pass  # Skip this test as PySpark doesn't allow non-string names


class TestEdgeCaseIterableValidation:
    """Test edge cases for iterable format validation."""

    def test_empty_iterable_schema(self):
        """Test that empty iterable creates empty schema."""
        empty_list = []
        empty_tuple = ()

        pyspark_list = to_pyspark_schema(empty_list)
        pyspark_tuple = to_pyspark_schema(empty_tuple)

        assert isinstance(pyspark_list, StructType)
        assert isinstance(pyspark_tuple, StructType)
        assert len(pyspark_list.fields) == 0
        assert len(pyspark_tuple.fields) == 0

    def test_non_string_field_name_in_iterable(self):
        """Test that non-string field names in iterable raise SchemaError."""
        with pytest.raises(SchemaError, match="Field names must be strings"):
            to_pyspark_schema([(123, pl.String)])  # Integer field name

        with pytest.raises(SchemaError, match="Field names must be strings"):
            to_pyspark_schema([(None, pl.String)])  # None field name

    def test_empty_field_name_in_iterable(self):
        """Test that empty field names in iterable raise SchemaError."""
        with pytest.raises(SchemaError, match="cannot be empty strings"):
            to_pyspark_schema([("", pl.String)])

    def test_duplicate_field_names_in_iterable(self):
        """Test that duplicate field names in iterable raise SchemaError."""
        with pytest.raises(SchemaError, match="Duplicate field name"):
            to_pyspark_schema([("name", pl.String), ("name", pl.Int32)])

    def test_very_deeply_nested_structure(self):
        """Test conversion of very deeply nested structures."""
        # Create a 5-level deep nested struct
        deep_schema = {
            "level1": pl.Struct(
                [
                    pl.Field(
                        "level2",
                        pl.Struct(
                            [
                                pl.Field(
                                    "level3",
                                    pl.Struct(
                                        [
                                            pl.Field(
                                                "level4",
                                                pl.Struct(
                                                    [
                                                        pl.Field("level5", pl.String),
                                                    ]
                                                ),
                                            ),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                ]
            ),
        }

        pyspark_schema = to_pyspark_schema(deep_schema)
        assert isinstance(pyspark_schema, StructType)

        # Navigate through the levels
        level1 = pyspark_schema.fields[0]
        assert isinstance(level1.dataType, StructType)
        level2 = level1.dataType.fields[0]
        assert isinstance(level2.dataType, StructType)
        level3 = level2.dataType.fields[0]
        assert isinstance(level3.dataType, StructType)
        level4 = level3.dataType.fields[0]
        assert isinstance(level4.dataType, StructType)
        level5 = level4.dataType.fields[0]
        assert isinstance(level5.dataType, StringType)

    def test_deeply_nested_arrays(self):
        """Test conversion of deeply nested arrays."""
        # 4 levels of nested arrays
        deep_array_schema = {
            "deep": pl.List(pl.List(pl.List(pl.List(pl.String)))),
        }

        pyspark_schema = to_pyspark_schema(deep_array_schema)
        assert isinstance(pyspark_schema, StructType)

        deep_field = pyspark_schema.fields[0]
        assert isinstance(deep_field.dataType, ArrayType)

        # Navigate through nested arrays
        level1 = deep_field.dataType.elementType
        assert isinstance(level1, ArrayType)
        level2 = level1.elementType
        assert isinstance(level2, ArrayType)
        level3 = level2.elementType
        assert isinstance(level3, ArrayType)
        level4 = level3.elementType
        assert isinstance(level4, StringType)

    def test_iterable_with_wrong_tuple_length(self):
        """Test iterable with tuples of wrong length."""
        with pytest.raises(SchemaError, match="has length"):
            to_pyspark_schema([("name",)])  # Too short

        with pytest.raises(SchemaError, match="has length"):
            to_pyspark_schema([("name", pl.String, "extra")])  # Too long

    def test_iterable_with_non_tuple_items(self):
        """Test iterable with non-tuple items."""
        with pytest.raises(SchemaError, match="is not a tuple"):
            to_pyspark_schema(["not", "tuples"])

        with pytest.raises(SchemaError, match="is not a tuple"):
            to_pyspark_schema([("name", pl.String), {"not": "tuple"}])
