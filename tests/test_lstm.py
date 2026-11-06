"""Unit tests for the forward pass of RansomwareLSTM (KAN-26)."""
import pytest
import torch

from src.model.lstm import RansomwareLSTM

N_FEATURES = 24  # adjust to len(feature_cols) from the Spark pipeline


@pytest.fixture
def model():
    torch.manual_seed(0)
    return RansomwareLSTM(n_features=N_FEATURES, hidden_dim=32, num_layers=2).eval()


def test_output_shape_and_range(model):
    x = torch.randn(4, 50, N_FEATURES)
    score = model(x)
    assert score.shape == (4,)
    assert torch.all(score >= 0) and torch.all(score <= 1)


def test_variable_length_batch(model):
    x = torch.randn(3, 40, N_FEATURES)
    lengths = torch.tensor([40, 25, 5])
    score = model(x, lengths)
    assert score.shape == (3,)
    assert torch.isfinite(score).all()


def test_padding_does_not_change_score(model):
    seq = torch.randn(1, 20, N_FEATURES)
    padded = torch.cat([seq, torch.randn(1, 30, N_FEATURES) * 100], dim=1)
    s1 = model(seq, torch.tensor([20]))
    s2 = model(padded, torch.tensor([20]))
    assert torch.allclose(s1, s2, atol=1e-5)


def test_different_sequence_lengths_accepted(model):
    for T in (1, 10, 200):
        assert model(torch.randn(2, T, N_FEATURES)).shape == (2,)


def test_attention_weights_sum_to_one(model):
    x = torch.randn(2, 30, N_FEATURES)
    lengths = torch.tensor([30, 12])
    _, w = model(x, lengths, return_attention=True)
    assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)
    assert w[1, 12:].abs().sum() == 0  # no attention on padding


def test_gradients_flow(model):
    model.train()
    x = torch.randn(4, 30, N_FEATURES)
    loss = model(x).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
