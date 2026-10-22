"""Monotonically increasing cubic schedule with warmup and recovery."""

def target_for_epoch(epoch: int, epochs: int, final_sparsity: float,
                     warmup: float = 0.20, end: float = 0.80) -> float:
    if not (0 <= final_sparsity < 1 and 0 <= warmup < end <= 1):
        raise ValueError('Invalid sparsity schedule')
    # Called at END of epoch: progress (epoch+1)/epochs.
    progress = (epoch + 1) / epochs
    if progress <= warmup:
        return 0.0
    fraction = min(1.0, (progress - warmup) / (end - warmup))
    return final_sparsity * (1 - (1 - fraction) ** 3)
