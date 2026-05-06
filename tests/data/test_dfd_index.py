import json
from pathlib import Path

import pytest

from deepfake_detection.data.index_ffpp import load_dfd_json_index


def test_load_dfd_json_index_reads_absolute_frame_paths(tmp_path):
    json_path = tmp_path / "dataset_json" / "DeepFakeDetection.json"
    json_path.parent.mkdir(parents=True)
    payload = {
        "DeepFakeDetection": {
            "DFD_real": {
                "test": {
                    "c23": {
                        "real_vid": {
                            "frames": ["/abs/real/000.png"],
                            "landmarks": [],
                        }
                    }
                }
            },
            "DFD_fake": {
                "test": {
                    "c23": {
                        "fake_vid": {
                            "frames": ["/abs/fake/000.png"],
                            "landmarks": ["/abs/fake/000.npy"],
                        }
                    }
                }
            },
        }
    }
    json_path.write_text(json.dumps(payload))

    real = load_dfd_json_index(str(tmp_path), "test", 0)
    fake = load_dfd_json_index(str(tmp_path), "test", 1)

    assert len(real) == 1
    assert real[0].frame_path == "/abs/real/000.png"
    assert real[0].landmark_path is None
    assert real[0].label == 0
    assert len(fake) == 1
    assert fake[0].frame_path == "/abs/fake/000.png"
    assert fake[0].landmark_path == "/abs/fake/000.npy"
    assert fake[0].label == 1
