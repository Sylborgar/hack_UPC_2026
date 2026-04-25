"""Device helpers for PyTorch inference."""

from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Return a torch.device from a friendly user option."""
    option = (requested or "auto").strip().lower()

    if option == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if option in {"cpu", "cuda", "mps"}:
        if option == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested --device cuda but CUDA is not available")
        if option == "mps":
            mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            if not mps_available:
                raise ValueError("Requested --device mps but MPS is not available")
        return torch.device(option)

    raise ValueError(f"Unsupported device option: {requested}")
