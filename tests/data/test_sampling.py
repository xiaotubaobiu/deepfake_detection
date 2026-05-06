from deepfake_detection.data.sampling import sample_uniform_frame_indices, balance_real_video_ids


def test_sample_uniform_frame_indices_returns_eight_sorted_positions():
    indices = sample_uniform_frame_indices(num_frames=32, num_samples=8)
    assert indices == [0, 4, 8, 13, 17, 22, 26, 31]


def test_balance_real_video_ids_oversamples_to_target_count():
    balanced = balance_real_video_ids(["r0", "r1"], target_count=5, seed=7)
    assert len(balanced) == 5
    assert set(balanced).issubset({"r0", "r1"})


def test_balance_real_video_ids_returns_early_if_sufficient():
    balanced = balance_real_video_ids(["r0", "r1", "r2", "r3"], target_count=3, seed=0)
    assert len(balanced) == 3


def test_sample_uniform_clamps_to_available_frames():
    assert sample_uniform_frame_indices(num_frames=4, num_samples=8) == [0, 1, 2, 3]


from deepfake_detection.data.index_cdf import normalize_cdf_method_name


def test_normalize_cdf_method_name_keeps_known_method_names():
    assert normalize_cdf_method_name("wav2lip") == "wav2lip"
    assert normalize_cdf_method_name("MRAA") == "MRAA"
