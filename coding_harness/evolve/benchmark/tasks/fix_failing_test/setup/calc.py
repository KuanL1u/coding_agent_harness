"""A tiny calculator module (contains a deliberate bug in ``add``)."""


def add(a, b):
    # BUG: subtracts instead of adding.
    return a - b


def multiply(a, b):
    return a * b
