"""Internal components composed by :class:`ReIDTrainer`.

The public training API remains in ``boxmot.reid.training.trainer``.  This
package keeps the trainer implementation split by responsibility so the
orchestrator does not accumulate data, optimization, loss, and persistence
details in one module.
"""
