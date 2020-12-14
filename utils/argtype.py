import argparse
import re
import numpy as np


error = argparse.ArgumentTypeError


def boolean(x: str):
    return str(x).lower() == 'true'


def integer_or_inf(x: str):
    try:
        int(x)
        return int(x)
    except ValueError:
        if str(x).lower() == 'inf':
            return np.inf
        else:
            raise error("{x} is not proper")
