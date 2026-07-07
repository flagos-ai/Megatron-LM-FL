import types

from typing import Literal, Optional, Union, get_args, get_origin

none_type = type(None)


def add_field_literal_choice(attr, new_choice):
    origin = get_origin(attr.type)
    type_tuple = get_args(attr.type)
    is_optional = False
    if origin in [types.UnionType, Union] and none_type in type_tuple:
        is_optional = True
        non_none_types = [t for t in type_tuple if t is not none_type]
        assert len(non_none_types) == 1, f"Unsupported type: {attr.type}"
        literal_type = non_none_types[0]
        origin = get_origin(literal_type)
    assert origin is Literal, f"Unsupported type: {attr.type}"
    if is_optional:
        type_tuple = get_args(literal_type)
    choices = list(dict.fromkeys(type_tuple))
    if new_choice in choices:
        return attr
    choices.append(new_choice)

    choices_types = [type(c) for c in choices]
    assert all(t == choices_types[0] for t in choices_types), (
        "Type of each choice in a Literal type should all be the same."
    )

    new_type_tuple = Literal[tuple(choices)]
    if is_optional:
        new_type_tuple = Optional[new_type_tuple]
    attr.type = new_type_tuple
    return attr
