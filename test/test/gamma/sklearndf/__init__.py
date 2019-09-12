import inspect
import re
from typing import *
from typing import Type


def get_classes(
    from_module, matching: str, excluding: Optional[Union[str, List[str]]] = None
) -> List[Type]:
    """ Helper to return all classes with matching name from a Python module """

    if isinstance(excluding, list):
        excluding = "|".join([f"({exclude_pattern})" for exclude_pattern in excluding])

    return [
        m[1]
        for m in inspect.getmembers(from_module)
        if isinstance(m[1], type)
        and re.match(matching, m[0])
        and not (excluding and re.match(excluding, m[0]))
    ]


def get_wrapped_counterpart(to_wrap: Type, from_package=None) -> Type:
    """ Helper to return the wrapped counterpart for a sklearn class """
    orig_name = to_wrap.__name__
    new_name = orig_name + "DF"

    if hasattr(from_package, new_name):
        return getattr(from_package, new_name)
