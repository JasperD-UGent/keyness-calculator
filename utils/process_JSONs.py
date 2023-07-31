from ast import literal_eval
import codecs
import json
import os
import pathlib
import sys
from typing import Any, Union


def dump_json(path_to_direc: Union[str, pathlib.Path], fn: str, variable: Any, *, indent=None):
    """Dump given variable to given JSON filename on given path.
    :param path_to_direc: path to the directory in which the JSON needs to be dumped.
    :param fn: filename of the JSON.
    :param variable: variable which should be dumped into the JSON.
    :param indent: indent which needs to be applied.
    :return: `None`
    """

    if not os.path.isdir(path_to_direc):
        os.makedirs(path_to_direc)

    with codecs.open(os.path.join(path_to_direc, fn), "w", "utf-8") as f:
        json.dump(variable, f, indent=2) if indent is not None else json.dump(variable, f)
    f.close()


def load_json(
        path,
        *,
        encoding: str = "utf-8"
):
    """Load JSON file from given path.
    :param path: path to the JSON which needs to be loaded.
    :param encoding: encoding of the JSON.
    :return: the loaded JSON.
    """

    with codecs.open(path, "r", encoding) as f:
        f_loaded = json.load(f)
    f.close()

    print(f"Finished loading {path}.")

    return f_loaded


def load_json_str_to_obj(
        path,
        *,
        encoding: str = "utf-8"
):
    """Load JSON file from given path and apply literal evaluation to loaded variable.
    :param path: path to the JSON which needs to be loaded.
    :param encoding: encoding of the JSON.
    :return: the loaded JSON.
    """
    f_loaded = load_json(path, encoding=encoding)
    f_loaded_new = {}

    for string in f_loaded:
        f_loaded_new[literal_eval(string)] = f_loaded[string]

    print(f"Finished loading {path}.")

    return f_loaded_new
