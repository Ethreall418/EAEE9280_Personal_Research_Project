"""
OceanJAX Parallel Package
=========================
Batch / multi-GPU execution utilities.

Phase 1  — Ensemble / batch parallelism (this package):
    Distribute independent model instances across GPUs via vmap + NamedSharding.
    No inter-device communication required; grid and params are replicated.

Phase 2+ — Single-domain sharding / halo exchange (future):
    shard_map + explicit halo exchange for splitting one large domain across GPUs.
"""

from OceanJAX.parallel.ensemble import batch_step, batch_run, sharded_ensemble_run

__all__ = ["batch_step", "batch_run", "sharded_ensemble_run"]
