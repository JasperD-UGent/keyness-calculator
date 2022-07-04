from ast import literal_eval
import codecs
import json
import os


def dump_json(path_to_direc, fn, variable):
    """Dump given variable to given JSON filename on given path."""

    if not os.path.isdir(path_to_direc):
        os.makedirs(path_to_direc)

    with codecs.open(os.path.join(path_to_direc, fn), "w", "utf-8") as f:
        json.dump(variable, f, indent=2)
    f.close()


def load_json(
        path,
        *,
        encoding: str = "utf-8"
):
    """Load JSON file from given path."""

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
    """Load JSON file from given path."""

    with codecs.open(path, "r", encoding) as f:
        f_loaded = json.load(f)
    f.close()

    f_loaded_new = {}

    for string in f_loaded:
        f_loaded_new[literal_eval(string)] = f_loaded[string]

    print(f"Finished loading {path}.")

    return f_loaded_new
