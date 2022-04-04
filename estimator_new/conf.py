# -*- coding: utf-8 -*-
"""
Default values.
"""

from .reduction import Kyber, ABLR21
from .simulator import GSA

red_cost_model = Kyber
red_cost_model_classical_poly_space = ABLR21
red_shape_model = "gsa"
red_simulator = GSA
mitm_opt = "analytical"
