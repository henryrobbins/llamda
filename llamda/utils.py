# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/utils/utils.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import os
import inspect


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def print_hyperlink(path, text=None):
    """Print hyperlink to file or folder for convenient navigation"""
    # Format: \033]8;;file:///path/to/file\033\\text\033]8;;\033\\
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name
