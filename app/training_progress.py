from typing import Optional

# Shared in-memory progress store, keyed by sensor uid.
# Written by the training thread, read by the /training/progress/{uid} endpoint.
# Structure: { phase, percent, epoch, total_epochs, eta_seconds }
training_progress: dict[str, dict] = {}


def set_phase(uid: str, phase: str, percent: int, eta_seconds: Optional[int] = None,
              epoch: Optional[int] = None, total_epochs: Optional[int] = None) -> None:
    training_progress[uid] = {
        "phase": phase,
        "percent": percent,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "eta_seconds": eta_seconds,
    }


def clear_progress(uid: str) -> None:
    training_progress.pop(uid, None)
