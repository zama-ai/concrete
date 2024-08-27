"""
Provide additional features that are not present in numpy.
"""

from numpy import where as if_then_else

from .array import array
from .bits import bits
from .constant import constant
from .convolution import conv
from .hint import hint
from .identity import identity, refresh
from .maxpool import maxpool
from .multivariate import multivariate
from .ones import one, ones, ones_like
from .relu import relu
from .round_bit_pattern import AutoRounder, round_bit_pattern
from .table import LookupTable
from .tag import tag
from .truncate_bit_pattern import AutoTruncator, truncate_bit_pattern
from .univariate import univariate
from .zeros import zero, zeros, zeros_like
