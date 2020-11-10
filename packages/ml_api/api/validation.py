from marshmallow import Schema, fields
from marshmallow import ValidationError

import typing as t
import json


class InvalidInputErrot(Exception):
    """Invalid model input."""


# List of column names to change before validation
SYNTAX_ERROR_FIELD_MAP = {}


class TitanicDataRequestSchema(Schema):
    pclass = fields.Integer()
    sex = fields.Str()
    age = fields.Float(allow_none=True)
    sibsp = fields.Integer(allow_none=True)
    parch = fields.Integer(allow_none=True)
    fare = fields.Float(allow_none=True)
    cabin = fields.Str(allow_none=True)
    embarked = fields.Str(allow_none=True)
    title = fields.Str(allow_none=True)


def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # Delete them in reverse order to don't
    # throw off the subsequent indexes
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    schema = TitanicDataRequestSchema(strict=True, many=True)

    # Convert syntax error field names (beginning with numbers)
    for dict in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict[value] = dict[key]
            del dict[key]

    errors = None
    try:
        schema.load(json.loads(input_data))
    except ValidationError as exc:
        errors = exc.messages
        print(f"ERROR MSG: {exc.messages}")
        print(f"ERROR DATA: {exc.data}")
        print(f"ERROR FIELDS: {exc.fields}")

    # convert syntax error field names back
    # NOTE: Never name your data fields with
    # numbers as the first letter
    for dict in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict[key] = dict[value]
            del dict[value]

    if errors:
        validated_input = _filter_error_rows(
            errors=errors, validated_input=input_data
        )
    else:
        validated_input = input_data

    return validated_input, errors
