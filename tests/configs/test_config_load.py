from pathlib import Path
import sys


def test_environment_file_declares_deepfake_detection_name():
    text = Path("environment.yml").read_text()
    assert "name: deepfake-detection" in text


def test_requirements_include_clip_and_albumentations():
    text = Path("requirements.txt").read_text()
    assert "albumentations" in text
    assert "git+https://github.com/openai/CLIP.git" in text


def test_norm_shortcut_config_loads_from_experiments_configs():
    sys.path.insert(0, str(Path.cwd()))
    from train import load_config

    cfg = load_config("experiments/configs/norm_shortcut_raw_s42.yaml")
    assert cfg["experiment_name"] == "norm_shortcut_raw_s42"
    assert cfg["train"]["output_dir"] == "experiments/outputs"
    assert cfg["train"]["seed"] == 42
