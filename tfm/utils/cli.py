import inspect
from typing import Callable, Dict

from tfm import cli


def get_cli() -> Dict[str, Callable]:
    # DO NOT REMOVE THIS IMPORTS, THEY ARE USED BY
    # `inspect.getmembers(cli, inspect.ismodule)`
    from tfm.cli import inference, training, tuning

    modules = inspect.getmembers(cli, inspect.ismodule)
    modules_ids = [id(i[1]) for i in modules]
    fire_mapping = {}

    for module in modules:
        fire_mapping.update(
            {
                function[0].replace("_", "-"): function[1]
                for function in inspect.getmembers(module[1], inspect.isfunction)
                if id(inspect.getmodule(function[1])) in modules_ids
                and not function[0].startswith("_")
            }
        )

    return fire_mapping
