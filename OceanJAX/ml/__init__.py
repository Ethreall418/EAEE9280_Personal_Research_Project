"""
OceanJAX ML Package
===================
Machine-learning / parameterisation interfaces for OceanJAX.

Phase 1 — Closure interface (this package):
    A minimal, stable hook that connects learnable parameterisations to the
    physical time-stepping loop without altering the pure-physics code path.

    Public API:
        ClosureOutput    – named container for tracer-tendency corrections and
                           vertical diffusivity scaling.
        AbstractClosure  – base class; subclass and override __call__.
        NullClosure      – no-op; preserves pure-physics behaviour exactly.

Phase 2+ (future):
    Concrete neural closures (e.g. MLP, CNN, GNN parameterisations) will
    live here as additional modules.
"""

from OceanJAX.ml.closure import AbstractClosure, ClosureOutput, NullClosure

__all__ = ["AbstractClosure", "ClosureOutput", "NullClosure"]