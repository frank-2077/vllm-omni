"""
E2E offline tests for MiniCPM-o 4.5 model with multimodal input and audio / text output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.assertions import assert_omni_text_responses_identical
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.minicpmo_encoder_cache import assert_audio_cache_trace, audio_cache_rpc
from tests.helpers.stage_config import get_deploy_config_path

models = ["openbmb/MiniCPM-o-4_5"]

_CI_DEPLOY = get_deploy_config_path("minicpmo_4_5.yaml")


test_params = [(model, _CI_DEPLOY, {"trust_remote_code": True}) for model in models]
cache_test_params = [
    (
        os.environ.get("MINICPMO_CACHE_TEST_MODEL", model),
        os.environ.get("MINICPMO_CACHE_TEST_DEPLOY", _CI_DEPLOY),
        {
            "trust_remote_code": True,
            "worker_extension_cls": "tests.helpers.minicpmo_encoder_cache.MiniCPMOEncoderCacheProbe",
        },
    )
    for model in models
]


@pytest.fixture
def audio_cache_probe(omni_runner):
    try:
        assert all(result is True for result in audio_cache_rpc(omni_runner, "start_audio_cache_probe"))
        yield lambda: audio_cache_rpc(omni_runner, "read_audio_cache_probe")
    finally:
        assert all(result is True for result in audio_cache_rpc(omni_runner, "stop_audio_cache_probe"))


def get_question(prompt_type: str = "text") -> str:
    prompts = {
        "text": "What is the capital of China? Answer in 20 words.",
        "audio": "Describe the audio briefly.",
        "image": "What color are the squares in this image?",
        "video": "Describe the video briefly.",
        "mix": "Describe what is in the image and audio.",
    }
    return prompts.get(prompt_type, prompts["text"])


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing text, generating text output."""
    request_config = {"prompts": get_question("text"), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing audio, generating text output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    request_config = {"prompts": get_question("audio"), "audios": (audio, 16000), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing image, generating text output."""
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {"prompts": get_question("image"), "images": image, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating text output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing text and generating audio through Talker and Code2Wav."""
    request_config = {"prompts": get_question("text"), "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing mixed modalities (image + audio), generating audio output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {
        "prompts": get_question("mix"),
        "audios": (audio, 16000),
        "images": image,
        "modalities": ["audio"],
    }
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)


def _cache_probe_audio(phrase: str):
    """Synthetic audio with a phrase-unique mm_hash.

    Use distinct first words because the generator truncates speech to one
    second. Tests also check that the resulting content hashes differ.
    """
    audio = generate_synthetic_audio(1, 1, 16000, phrase_text=phrase)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    return (audio, 16000)


@pytest.mark.full_model
@pytest.mark.omni
@pytest.mark.cache
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", cache_test_params, indirect=True)
def test_audio_encoder_cache_warm_repeat_parity(omni_runner, omni_runner_handler, audio_cache_probe) -> None:
    """Cold/warm requests must call Whisper [1, 0] and return identical text."""
    audio = _cache_probe_audio("Whisper warm repeat probe")
    request_config = {"prompts": get_question("audio"), "audios": audio, "modalities": ["text"]}
    responses, trace = [], []
    for _ in range(2):
        response = omni_runner_handler.send_omni_request(request_config)
        assert response.success and response.text_content, "Audio cache request failed or produced empty text"
        responses.append(response)
        trace.append(audio_cache_probe())
    assert_audio_cache_trace(trace, expected_calls=[1, 0], expected_items=[1, 0])
    for rank in range(len(trace[0])):
        audio_hash = trace[0][rank]["encoded_hashes"][0]
        assert all(step[rank]["gathered_hashes"] == [[audio_hash]] for step in trace)
    assert_omni_text_responses_identical(responses, context="audio warm repeat")


@pytest.mark.full_model
@pytest.mark.omni
@pytest.mark.cache
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", cache_test_params, indirect=True)
def test_audio_encoder_cache_mixed_hit_partition_parity(omni_runner, omni_runner_handler, audio_cache_probe) -> None:
    """Warm A, encode only C in [A, C], then reuse both; also check [C, A]."""
    audio_a = _cache_probe_audio("Alpha mixed cache probe")
    audio_c = _cache_probe_audio("Charlie mixed cache probe")
    assert not (audio_a[0] == audio_c[0]).all(), "Mixed probe requires distinct waveform content"

    warmup = omni_runner_handler.send_omni_request(
        {"prompts": get_question("audio"), "audios": audio_a, "modalities": ["text"]}
    )
    assert warmup.success and warmup.text_content, "Audio A warmup failed or produced empty text"
    trace = [audio_cache_probe()]
    mixed_config = {
        "prompts": "Describe the first audio clip, then describe the second audio clip.",
        "audios": [[audio_a, audio_c]],
        "modalities": ["text"],
    }
    responses = []
    for _ in range(2):
        response = omni_runner_handler.send_omni_request(mixed_config)
        assert response.success and response.text_content, "Mixed audio request failed or produced empty text"
        responses.append(response)
        trace.append(audio_cache_probe())
    assert_audio_cache_trace(trace, expected_calls=[1, 1, 0], expected_items=[1, 1, 0])
    for rank in range(len(trace[0])):
        audio_hash_a = trace[0][rank]["encoded_hashes"][0]
        audio_hash_c = trace[1][rank]["encoded_hashes"][0]
        assert audio_hash_a != audio_hash_c, "Mixed probe must encode C, not warm A"
        assert trace[0][rank]["gathered_hashes"] == [[audio_hash_a]]
        assert trace[1][rank]["gathered_hashes"] == [[audio_hash_a, audio_hash_c]]
        assert trace[2][rank]["gathered_hashes"] == [[audio_hash_a, audio_hash_c]]
    assert_omni_text_responses_identical(responses, context="audio mixed hit/miss vs fully cached")

    reversed_config = {**mixed_config, "audios": [[audio_c, audio_a]]}
    reversed_responses, reversed_trace = [], []
    for _ in range(2):
        response = omni_runner_handler.send_omni_request(reversed_config)
        assert response.success and response.text_content, "Reversed audio request failed or produced empty text"
        reversed_responses.append(response)
        reversed_trace.append(audio_cache_probe())
    assert_audio_cache_trace(reversed_trace, expected_calls=[0, 0], expected_items=[0, 0])
    for rank in range(len(trace[0])):
        expected = trace[1][rank]["gathered_hashes"][0][::-1]
        assert all(step[rank]["gathered_hashes"] == [expected] for step in reversed_trace)
    assert_omni_text_responses_identical(reversed_responses, context="reversed audio fully cached repeat")
