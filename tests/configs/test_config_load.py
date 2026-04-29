from pathlib import Path
import sys
import os


def test_environment_file_declares_deepfake_detection_name():
    text = Path("environment.yml").read_text()
    assert "name: deepfake-detection" in text


def test_requirements_include_clip_and_albumentations():
    text = Path("requirements.txt").read_text()
    assert "albumentations" in text
    assert "git+https://github.com/openai/CLIP.git" in text


def test_exp3_config_loads_prompt_loss():
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    config_path = os.path.join(project_root, "configs", "exp3_clip_prompt.yaml")
    base_path = os.path.join(project_root, "configs", "base.yaml")
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(base_path) as f:
        base = yaml.safe_load(f)
    merged = {**base, **cfg}
    for k in set(list(base.keys()) + list(cfg.keys())):
        if k in base and k in cfg and isinstance(base[k], dict) and isinstance(cfg[k], dict):
            merged[k] = {**base[k], **cfg[k]}
    cfg = merged
    assert cfg["loss"]["name"] == "cross_entropy_plus_prompt"
    assert cfg["loss"]["beta"] == 0.1
    assert cfg["loss"]["alpha"] == 0.3
    assert cfg["loss"]["tau"] == 0.07
    assert cfg["train"]["per_gpu_batch"] == 128
    assert cfg["train"]["lr"] == 0.00002
    assert cfg["train"]["weight_decay"] == 0.0005
    assert cfg["train"]["epochs"] == 5
    assert cfg["train"]["patience"] == 3
