# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.multimodal.inputs import PlaceholderRange

from tests.helpers.assertions import assert_omni_text_responses_identical
from tests.helpers.minicpmo_encoder_cache import MiniCPMOEncoderCacheProbe, assert_audio_cache_trace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("success", "text", "message"),
    [(False, "same", "failed"), (True, "", "no text"), (True, None, "no text"), (True, "different", "diverged")],
)
def test_text_parity_rejects_failed_empty_or_different_response(success, text, message):
    responses = [
        SimpleNamespace(success=True, text_content="same"),
        SimpleNamespace(success=success, text_content=text),
    ]
    with pytest.raises(AssertionError, match=message):
        assert_omni_text_responses_identical(responses)


def _stats(calls, hashes):
    return {
        "calls": calls,
        "batch_sizes": [len(hashes)] if calls else [],
        "encoded_hashes": hashes,
        "gathered_hashes": [["A", "C"]],
    }


@pytest.mark.parametrize("broken_rank", [0, 1])
def test_trace_rejects_encoder_reexecution_on_any_rank(broken_rank):
    trace = [[_stats(1, ["A"]) for _ in range(2)], [_stats(0, []) for _ in range(2)]]
    trace[1][broken_rank] = _stats(1, ["A"])
    with pytest.raises(AssertionError, match=f"TP rank {broken_rank}: encoder calls"):
        assert_audio_cache_trace(trace, [1, 0], [1, 0])


def test_trace_rejects_reencoding_both_items_in_one_call():
    trace = [[_stats(1, ["A"])], [_stats(1, ["A", "C"])], [_stats(0, [])]]
    with pytest.raises(AssertionError, match="request 1"):
        assert_audio_cache_trace(trace, [1, 1, 0], [1, 1, 0])


@pytest.mark.parametrize("corruption", [None, "order", "mask"])
def test_worker_probe_checks_gather_and_restores_methods(corruption):
    encoder = torch.nn.Identity()
    cache = {"A": torch.full((2, 3), 1.0), "C": torch.full((2, 3), 2.0)}
    features = [
        SimpleNamespace(identifier=key, modality="audio", mm_position=PlaceholderRange(offset=offset, length=2))
        for key, offset in [("A", 1), ("C", 4)]
    ]

    def gather(_scheduler):
        mask = torch.tensor([False, True, True, False, True, True])
        embeddings = [cache["A"], cache["C"]]
        if corruption == "order":
            embeddings.reverse()
        if corruption == "mask":
            mask = mask.roll(1)
        return embeddings, mask

    def batch(_scheduler):
        return ["C"], [("audio", {})], []

    runner = SimpleNamespace(
        _gather_mm_embeddings=gather,
        _batch_mm_inputs_from_scheduler=batch,
        encoder_cache=cache,
        input_batch=SimpleNamespace(req_ids=["request"]),
        requests={"request": SimpleNamespace(num_computed_tokens=0, mm_features=features)},
    )
    worker = MiniCPMOEncoderCacheProbe()
    worker.model_runner = runner
    worker.get_model = lambda: SimpleNamespace(thinker=SimpleNamespace(apm=encoder))
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False))
    scheduler = SimpleNamespace(num_scheduled_tokens={"request": 6})
    worker.start_audio_cache_probe()
    try:
        encoder(torch.zeros(1, 80, 100))
        runner._batch_mm_inputs_from_scheduler(scheduler)
        if corruption:
            with pytest.raises(AssertionError, match="placeholder"):
                runner._gather_mm_embeddings(scheduler)
        else:
            runner._gather_mm_embeddings(scheduler)
            assert worker.read_audio_cache_probe() == _stats(1, ["C"])
            assert worker.read_audio_cache_probe()["calls"] == 0
    finally:
        worker.stop_audio_cache_probe()
    assert runner._gather_mm_embeddings is gather
    assert runner._batch_mm_inputs_from_scheduler is batch
    assert not encoder._forward_pre_hooks
