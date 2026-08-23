"""
E2E offline tests for MiniCPM-o 4.5 model with multimodal input and audio / text output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path

models = ["openbmb/MiniCPM-o-4_5"]

_CI_DEPLOY = get_deploy_config_path("minicpmo_4_5.yaml")


test_params = [(model, None, {"deploy_config": _CI_DEPLOY, "trust_remote_code": True}) for model in models]


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

    Each distinct phrase yields distinct waveform bytes, so the encoder-cache
    tests below control exactly which items are warm: they never collide with
    the default-phrase audio used by the other tests in this module.
    """
    audio = generate_synthetic_audio(1, 1, 16000, phrase_text=phrase)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    return (audio, 16000)


@pytest.mark.full_model
@pytest.mark.omni
@pytest.mark.cache
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_encoder_cache_warm_repeat_parity(omni_runner, omni_runner_handler) -> None:
    """Repeating an identical audio request must reproduce the same text.

    Regression for the audio half of the #5069 encoder-cache evaluation: the
    first request encodes the audio cold; the byte-identical second request is
    served from vLLM's content-addressed encoder cache (mm_hash) without
    re-entering the Whisper encoder. Both paths consume the same cached
    embedding tensor, so greedy text must match exactly. A divergence here
    means cached encoder outputs are corrupted or misapplied on the hit path.
    """
    from tests.helpers.assertions import assert_omni_text_responses_identical

    audio = _cache_probe_audio("encoder cache warm repeat probe")
    request_config = {"prompts": get_question("audio"), "audios": audio, "modalities": ["text"]}
    responses = [omni_runner_handler.send_omni_request(request_config) for _ in range(2)]
    assert_omni_text_responses_identical(responses, context="audio warm repeat")


@pytest.mark.full_model
@pytest.mark.omni
@pytest.mark.cache
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_encoder_cache_mixed_hit_partition_parity(omni_runner, omni_runner_handler) -> None:
    """Mixed hit/miss and fully-cached paths must agree, item order preserved.

    Regression for the #5069 audio attribution probe. Sequence:
      1. warm item A alone (fills its encoder-cache entry);
      2. request [A, C]: A is a cache hit, C is encoded live — the encoder
         receives only the missing item;
      3. repeat [A, C] byte-identically: both items are now cache hits and the
         encoder is not entered at all.
    Requests 2 and 3 consume identical embedding tensors, so their greedy text
    must match exactly. A divergence means cached embeddings were placed into
    the wrong placeholder positions (ordering) or partial-hit partitioning
    corrupted the batch.
    """
    from tests.helpers.assertions import assert_omni_text_responses_identical

    audio_a = _cache_probe_audio("encoder cache mixed probe alpha")
    audio_c = _cache_probe_audio("encoder cache mixed probe charlie")

    # Step 1: make A warm so step 2 exercises the partial-hit partition.
    omni_runner_handler.send_omni_request({"prompts": get_question("audio"), "audios": audio_a, "modalities": ["text"]})

    mixed_config = {
        "prompts": "Are these two audio clips the same?",
        # Nested list: one prompt carrying two audio items (two placeholders).
        "audios": [[audio_a, audio_c]],
        "modalities": ["text"],
    }
    responses = [omni_runner_handler.send_omni_request(mixed_config) for _ in range(2)]
    assert_omni_text_responses_identical(responses, context="audio mixed hit/miss vs fully cached")
