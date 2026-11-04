import torch

from src.compute_metrics import count_parameters
from src.config import CONV_CHANNELS, CONV_KERNELS, FC_UNITS, N_CHANNELS, WIN_SAMPLES
from src.evaluate import load_checkpoint, rebuild_model
from src.model import SeizureCNN


def test_forward_shape():
    model = SeizureCNN()
    model.eval()
    x = torch.randn(2, N_CHANNELS, WIN_SAMPLES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_parameter_count():
    model = SeizureCNN()
    info = count_parameters(model)
    assert info["n_params_total"] == 871489


def test_checkpoint_roundtrip(tmp_path):
    model = SeizureCNN()
    model.eval()
    checkpoint = {
        "model_config": {
            "in_channels": N_CHANNELS,
            "conv_channels": list(CONV_CHANNELS),
            "conv_kernels": list(CONV_KERNELS),
            "fc_units": FC_UNITS,
            "dropout": 0.4,
        },
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)

    loaded = load_checkpoint(path)
    rebuilt = rebuild_model(loaded)
    rebuilt.eval()

    x = torch.randn(1, N_CHANNELS, WIN_SAMPLES)
    with torch.no_grad():
        assert torch.allclose(model(x), rebuilt(x))
