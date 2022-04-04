# -*- coding: utf-8 -*-
class OutOfBoundsError(ValueError):
    """
    Used to indicate a wrong value, for example δ < 1.
    """

    pass


class InsufficientSamplesError(ValueError):
    """
    Used to indicate the number of samples given is too small.
    """

    pass
