"""Sanity check harness implementing Steps 2, 4, and 5 of the Paper-to-PyTorch pipeline:

- assert_shape_contract: Verifies output tensor dimensions against expected shapes.
- audit_gradients: Validates gradient flow, checking for NaNs, Infs, and frozen target networks.
- overfit_single_batch: Automated single-batch overfit gate before full benchmark training.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def assert_shape_contract(
    module: nn.Module,
    dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    expected_output_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Verifies that passing dummy_input through module produces the expected output shape."""
    module.eval()
    with torch.no_grad():
        if isinstance(dummy_input, dict):
            output = module(**dummy_input)
        elif isinstance(dummy_input, (tuple, list)):
            output = module(*dummy_input)
        else:
            output = module(dummy_input)

    assert output.shape == expected_output_shape, (
        f"Shape contract violation: Expected shape {expected_output_shape}, "
        f"but received {output.shape}."
    )
    return output


def audit_gradients(
    model: nn.Module,
    expected_frozen_names: Optional[List[str]] = None,
) -> None:
    """Audits model parameters for NaN, Inf, and properly frozen target/non-target weights."""
    expected_frozen = set(expected_frozen_names or [])

    for name, param in model.named_parameters():
        is_expected_frozen = any(frozen_name in name for frozen_name in expected_frozen)

        if is_expected_frozen:
            assert param.grad is None or (param.grad == 0).all(), (
                f"Gradient leak detected! Parameter '{name}' was expected to be frozen, "
                f"but received non-zero gradients."
            )
        elif param.requires_grad:
            assert param.grad is not None, (
                f"Missing gradient! Parameter '{name}' requires_grad=True but received no grad."
            )
            assert not torch.isnan(param.grad).any(), (
                f"NaN gradient detected in parameter '{name}'!"
            )
            assert not torch.isinf(param.grad).any(), (
                f"Inf gradient detected in parameter '{name}'!"
            )


def overfit_single_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    batch_inputs: Any,
    batch_targets: Any,
    max_steps: int = 200,
    target_loss: float = 0.02,
    verbose: bool = True,
) -> float:
    """Trains on a single small batch for max_steps.

    The loss must drop below target_loss, proving there are no architectural, masking,
    or loss formulation bugs.
    """
    model.train()
    initial_loss: Optional[float] = None
    final_loss: float = float("inf")

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()

        if isinstance(batch_inputs, dict):
            predictions = model(**batch_inputs)
        elif isinstance(batch_inputs, (tuple, list)):
            predictions = model(*batch_inputs)
        else:
            predictions = model(batch_inputs)

        loss = loss_fn(predictions, batch_targets)
        loss_val = loss.item()

        if initial_loss is None:
            initial_loss = loss_val

        assert not torch.isnan(loss), f"Loss diverged to NaN at step {step}!"
        assert not torch.isinf(loss), f"Loss diverged to Inf at step {step}!"

        loss.backward()
        optimizer.step()
        final_loss = loss_val

        if verbose and (step == 1 or step % 50 == 0 or loss_val < target_loss):
            print(f"  [Step {step:03d}/{max_steps}] Loss: {loss_val:.6f}")

        if loss_val < target_loss:
            if verbose:
                print(f"  --> Overfit gate passed at step {step} with loss {loss_val:.6f} (< {target_loss}).")
            return final_loss

    assert final_loss < target_loss, (
        f"Overfit sanity check failed! Loss plateaued at {final_loss:.6f} "
        f"(Target was < {target_loss}). Check forward pass, activation scales, and loss gradients."
    )
    return final_loss
