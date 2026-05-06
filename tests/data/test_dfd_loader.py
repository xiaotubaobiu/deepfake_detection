import json

import pytest

from deepfake_detection.data.builders import build_eval_loader


@pytest.fixture
def tiny_cfg(tmp_path):
    return {
        "dataset": {"root": str(tmp_path), "methods": ["Deepfakes"], "frames_per_video": 8},
        "train": {"per_gpu_batch": 2, "num_workers": 0, "prefetch_factor": 2, "seed": 42},
        "model": {"name": "clip_finetune"},
    }


def test_build_eval_loader_uses_dfd_json(tmp_path, tiny_cfg):
    json_path = tmp_path / "dataset_json" / "DeepFakeDetection.json"
    json_path.parent.mkdir(parents=True)
    payload = {
        "DeepFakeDetection": {
            "DFD_real": {"test": {"c23": {"real_vid": {"frames": ["/abs/real/000.png"], "landmarks": []}}}},
            "DFD_fake": {"test": {"c23": {"fake_vid": {"frames": ["/abs/fake/000.png"], "landmarks": []}}}},
        }
    }
    json_path.write_text(json.dumps(payload))

    loader = build_eval_loader(tiny_cfg, domain="dfd", distributed=False)

    assert len(loader.dataset.records) == 2
    assert {record.label for record in loader.dataset.records} == {0, 1}
    assert {record.video_id for record in loader.dataset.records} == {"real_vid", "fake_vid"}


def test_build_eval_loader_raises_when_dfd_missing(tiny_cfg):
    with pytest.raises(FileNotFoundError):
        build_eval_loader(tiny_cfg, domain="dfd", distributed=False)
