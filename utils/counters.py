import sys


def multiple_250(counter: int):
    """Function which prints every multiple of 250.
    :param counter: value of the counter.
    :return: `None`
    """

    if (counter / 250).is_integer():
        print(counter)
