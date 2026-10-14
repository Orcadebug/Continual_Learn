"""A-GEM-style gradient projection used by Exp18-Reconstructed.

This module is intentionally independent of Laya so its safety-critical
mathematics can be tested on CPU before an expensive GPU lifetime is launched.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Projection:
    gradient: torch.Tensor
    cosine: float
    dot: float
    projected: bool
    change_norm: float


def project_gradient(g_new: torch.Tensor, g_ref: torch.Tensor, epsilon: float = 1e-12) -> Projection:
    """Return a finite gradient non-interfering with `g_ref` to first order.

    Non-finite input is rejected rather than silently transformed: a malformed
    controller update must fail loudly and be logged by the caller.
    """
    if g_new.shape != g_ref.shape:
        raise ValueError("new and reference gradients must have identical shape")
    if not torch.isfinite(g_new).all() or not torch.isfinite(g_ref).all():
        raise FloatingPointError("non-finite gradient supplied to adaptive valve")
    dot_t = torch.dot(g_new, g_ref)
    new_norm = torch.linalg.vector_norm(g_new)
    ref_norm = torch.linalg.vector_norm(g_ref)
    cosine = 0.0 if float(new_norm * ref_norm) <= epsilon else float(dot_t / (new_norm * ref_norm))
    if float(dot_t) >= 0.0 or float(ref_norm) <= epsilon:
        return Projection(g_new.clone(), cosine, float(dot_t), False, 0.0)
    result = g_new - dot_t / (torch.dot(g_ref, g_ref) + epsilon) * g_ref
    return Projection(result, cosine, float(dot_t), True, float(torch.linalg.vector_norm(result - g_new)))


def flatten_gradients(grads: list[torch.Tensor | None], params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Make a parameter-aligned gradient vector, replacing unused entries by 0."""
    pieces = [torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1) if g is None else g.detach().reshape(-1)
              for g, p in zip(grads, params, strict=True)]
    return torch.cat(pieces) if pieces else torch.empty(0)


def assign_flat_gradient(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> None:
    """Assign a flattened controller output back to ``param.grad`` tensors."""
    offset = 0
    for param in params:
        count = param.numel()
        param.grad = flat[offset:offset + count].reshape_as(param).clone()
        offset += count
    if offset != flat.numel():
        raise ValueError("flat gradient does not match parameter count")
