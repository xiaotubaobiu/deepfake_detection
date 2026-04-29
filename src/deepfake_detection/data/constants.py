CLIP_MODEL_NAME = "ViT-B/16"
IMG_SIZE = 224
FACE_SWAP_METHODS = [
    "simswap", "inswap", "blendface", "faceswap",
    "fsgan", "mobileswap", "e4s", "facedancer",
]
REENACTMENT_METHODS = [
    "fomm", "facevid2vid", "wav2lip", "sadtalker",
    "MRAA", "pirender", "tpsm", "lia",
]
ALL_METHODS = FACE_SWAP_METHODS + REENACTMENT_METHODS
REAL_LABEL = 0
FAKE_LABEL = 1
REAL_PROMPTS = [
    "a real human face photo",
    "an authentic face image",
    "a natural face without manipulation",
]
FAKE_PROMPTS = [
    "a fake face photo",
    "a manipulated face image",
    "a deepfake face image",
]
