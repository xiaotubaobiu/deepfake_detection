from pathlib import Path


def test_environment_file_declares_deepfake_detection_name():
    text = Path("environment.yml").read_text()
    assert "name: deepfake-detection" in text


def test_requirements_include_clip_and_albumentations():
    text = Path("requirements.txt").read_text()
    assert "albumentations" in text
    assert "git+https://github.com/openai/CLIP.git" in text
