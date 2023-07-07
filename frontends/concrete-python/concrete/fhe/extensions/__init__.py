"""
Provide additional features that are not present in numpy.
"""

from .array import array
from .convolution import conv
from .maxpool import maxpool
from .ones import one, ones, ones_like
from .round_bit_pattern import AutoRounder, round_bit_pattern
from .table import LookupTable
from .tag import tag
from .univariate import univariate
from .zeros import zero, zeros, zeros_like
