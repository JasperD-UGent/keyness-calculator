import sys


def multiple_250(counter):
    """Counter which prints every multiple of 250."""

    if (counter / 250).is_integer():
        print(counter)
