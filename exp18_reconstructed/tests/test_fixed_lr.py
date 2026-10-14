import pytest
import torch


def update(lr: float) -> float:
    value = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([value], lr=lr, weight_decay=0.0)
    value.grad = torch.tensor([2.0])
    before = value.detach().clone()
    optimizer.step()
    return float((value.detach() - before).abs())


def test_fixed_lr_arm_is_a_real_adamw_learning_rate_reduction():
    normal, reduced = update(1e-3), update(3e-4)
    assert reduced / normal == pytest.approx(0.30, rel=1e-5)
