import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

import src.api as api


def _make_fake_models(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # checkpoint mínimo válido para la ruta de listado
    torch.save({"loss": "focal"}, models_dir / "fake.pt")
    # subcarpeta (unused) que NO debe listarse
    (models_dir / "unused").mkdir()
    torch.save({"loss": "bce"}, models_dir / "unused" / "ignored.pt")
    return models_dir


def test_health_and_list_models(tmp_path, monkeypatch):
    models_dir = _make_fake_models(tmp_path)
    monkeypatch.setattr(api, "MODELS_DIR", models_dir)

    client = TestClient(api.app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["models"] == ["fake.pt"]

    listed = client.get("/api/models")
    assert listed.status_code == 200
    names = [m["name"] for m in listed.json()["models"]]
    assert names == ["fake.pt"]


def test_validate_model_name_rejects_traversal(tmp_path, monkeypatch):
    models_dir = _make_fake_models(tmp_path)
    monkeypatch.setattr(api, "MODELS_DIR", models_dir)

    with pytest.raises(HTTPException) as exc:
        api._validate_model_name("../secret.pt")
    assert exc.value.status_code == 400


def test_model_not_found(tmp_path, monkeypatch):
    models_dir = _make_fake_models(tmp_path)
    monkeypatch.setattr(api, "MODELS_DIR", models_dir)

    client = TestClient(api.app)
    assert client.get("/api/models/nope.pt").status_code == 404
