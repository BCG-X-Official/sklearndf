import inspect
import re
from typing import *
from typing import Type


def get_classes(from_module, regex) -> List[Type]:
    return [m[1] for m in inspect.getmembers(from_module) if re.match(regex, m[0])]


def get_wrapped_counterpart(to_wrap: Type, from_package=None) -> Type:
    orig_name = to_wrap.__name__
    new_name = orig_name + "DF"

    if hasattr(from_package, new_name):
        return getattr(from_package, new_name)
