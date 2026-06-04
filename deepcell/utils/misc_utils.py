"""Miscellaneous utility functions"""


import re


def sorted_nicely(ll):
    """Sort a list of strings by the numerical order of all substrings

    Args:
        l (list): List of strings to sort

    Returns:
        list: a sorted list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(ll, key=alphanum_key)


def get_sorted_keys(dict_to_sort):
    """Gets the keys from a dict and sorts them in ascending order.
    Assumes keys are of the form ``Ni``, where ``N`` is a letter and ``i``
    is an integer.

    Args:
        dict_to_sort (dict): dict whose keys need sorting

    Returns:
        list: list of sorted keys from ``dict_to_sort``
    """
    sorted_keys = list(dict_to_sort.keys())
    sorted_keys.sort(key=lambda x: int(x[1:]))
    return sorted_keys
