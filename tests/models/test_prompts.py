from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def test_build_fixed_prompt_texts_returns_real_and_fake_prompts():
    from deepfake_detection.models.clip_prompt import build_fixed_prompt_texts
    real, fake = build_fixed_prompt_texts()
    assert real == REAL_PROMPTS
    assert fake == FAKE_PROMPTS
