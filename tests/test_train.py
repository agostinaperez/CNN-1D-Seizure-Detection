"""Tests de la función de pérdida (src/train.py), sin entrenar nada."""

import torch
import torch.nn.functional as F

from src.train import FocalLoss


def test_focal_loss_gamma_zero_is_scaled_bce():
    logits = torch.tensor([2.0, -1.0, 0.5])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = FocalLoss(gamma=0.0, alpha=0.5)(logits, targets)
    expected = 0.5 * F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(loss, expected)


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([5.0])  # p ≈ 0.993, correcto para target 1
    targets = torch.tensor([1.0])
    focal = FocalLoss(gamma=2.0, alpha=1.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert focal.item() >= 0.0
    # un ejemplo bien clasificado y confiable pesa (mucho) menos que en BCE
    assert focal.item() < bce.item()
