"""Finite element analysis of layered RC shells with the NLFEA and the NN (solver, elements, mesh, post-processing,
NN predictions). The repository root is added to sys.path for the imports from sampling/ and training/."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
