import inspect
import re
from typing import *
from typing import Type


def get_classes(from_module, regex: str, ignore: Iterable[str] = None) -> List[Type]:
    """ Helper to return all classes with matching name from a Python module """
    if ignore is None:
        ignore = []

    return [
        m[1]
        for m in inspect.getmembers(from_module)
        if re.match(regex, m[0]) and isinstance(m[1], type) and not m[0] in ignore
    ]


def get_wrapped_counterpart(to_wrap: Type, from_package=None) -> Type:
    """ Helper to return the wrapped counterpart for a sklearn class """
    orig_name = to_wrap.__name__
    new_name = orig_name + "DF"

    if hasattr(from_package, new_name):
        return getattr(from_package, new_name)
