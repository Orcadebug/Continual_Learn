import pytest
import torch

from exp18_reconstructed.valve import project_gradient


def test_conflicting_gradients_are_projected_to_noninterference():
    result = project_gradient(torch.tensor([1.0, -2.0]), torch.tensor([-1.0, 0.0]))
    assert result.projected
    assert torch.dot(result.gradient, torch.tensor([-1.0, 0.0])) >= -1e-6
    assert result.change_norm > 0


def test_aligned_gradients_are_unchanged():
    proposed = torch.tensor([1.0, 2.0])
    result = project_gradient(proposed, torch.tensor([2.0, 4.0]))
    assert not result.projected
    assert torch.equal(result.gradient, proposed)


def test_zero_reference_gradient_is_safe_noop():
    proposed = torch.tensor([1.0, -3.0])
    result = project_gradient(proposed, torch.zeros(2))
    assert not result.projected
    assert torch.equal(result.gradient, proposed)


def test_nonfinite_gradient_fails_loudly():
    with pytest.raises(FloatingPointError):
        project_gradient(torch.tensor([float("nan")]), torch.tensor([1.0]))
