import torch
import torch.nn as nn
from sanity.checks import assert_shape_contract, audit_gradients, overfit_single_batch


def test_assert_shape_contract():
    linear = nn.Linear(10, 5)
    x = torch.randn(4, 10)
    out = assert_shape_contract(linear, x, (4, 5))
    assert out.shape == (4, 5)


def test_audit_gradients():
    linear = nn.Linear(10, 5)
    x = torch.randn(4, 10)
    loss = linear(x).sum()
    loss.backward()
    audit_gradients(linear)


def test_overfit_single_batch():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    final_loss = overfit_single_batch(
        model, optimizer, loss_fn, x, y, max_steps=150, target_loss=0.01, verbose=False
    )
    assert final_loss < 0.01
