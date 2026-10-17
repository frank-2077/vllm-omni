# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Worker-side instrumentation for MiniCPM-o audio encoder cache tests."""


class MiniCPMOEncoderCacheProbe:
    """Installed through worker_extension_cls; RPCs target only Stage 0.

    Install after startup so profiling calls do not enter the measurements.
    Record each TP rank independently: summing ranks would overcount a single
    encoder invocation. The probe never changes encoder outputs or cache state.
    """

    def start_audio_cache_probe(self):
        import torch

        assert not hasattr(self, "_audio_cache_probe"), "Probe already installed"
        runner = self.model_runner
        encoder = self.get_model().thinker.apm
        assert encoder is not None, "Stage 0 has no Whisper encoder"
        assert not self.vllm_config.cache_config.enable_prefix_caching, "KV prefix hits would confound this test"
        state = {"calls": 0, "batch_sizes": [], "encoded_hashes": [], "gathered_hashes": []}
        originals = {
            "_batch_mm_inputs_from_scheduler": runner._batch_mm_inputs_from_scheduler,
            "_gather_mm_embeddings": runner._gather_mm_embeddings,
        }

        def count_encoder(_module, args):
            state["calls"] += 1
            state["batch_sizes"].append(int(args[0].shape[0]))

        def batch_inputs(*args, **kwargs):
            result = originals["_batch_mm_inputs_from_scheduler"](*args, **kwargs)
            hashes, mm_kwargs, _ = result
            assert all(modality == "audio" for modality, _ in mm_kwargs)
            state["encoded_hashes"].extend(hashes)
            return result

        def gather(scheduler_output, *args, **kwargs):
            embeddings, mask = originals["_gather_mm_embeddings"](scheduler_output, *args, **kwargs)
            # These probes use ordinary prefill (no speculative look-ahead).
            assert not args and not kwargs
            expected = []
            expected_mask = torch.zeros_like(mask)
            hashes = []
            batch_offset = 0
            for req_id in runner.input_batch.req_ids:
                req = runner.requests[req_id]
                start = req.num_computed_tokens
                count = scheduler_output.num_scheduled_tokens[req_id]
                for feature in req.mm_features:
                    pos = feature.mm_position
                    lo = max(start, pos.offset)
                    hi = min(start + count, pos.offset + pos.length)
                    if lo >= hi:
                        continue
                    assert feature.modality == "audio"
                    begin, end = pos.get_embeds_indices_in_range(lo - pos.offset, hi - pos.offset)
                    if begin == end:
                        continue
                    expected.append(runner.encoder_cache[feature.identifier][begin:end])
                    hashes.append(feature.identifier)
                    slot = slice(batch_offset + lo - start, batch_offset + hi - start)
                    if pos.is_embed is None:
                        expected_mask[slot] = True
                    else:
                        expected_mask[slot] = pos.is_embed[lo - pos.offset : hi - pos.offset]
                batch_offset += count
            assert len(embeddings) == len(expected), "Wrong number of gathered audio embeddings"
            for actual, reference in zip(embeddings, expected):
                assert torch.equal(actual, reference), "Audio embedding placed in the wrong placeholder"
            assert torch.equal(mask, expected_mask), "Audio placeholder mask differs from request positions"
            if hashes:
                state["gathered_hashes"].append(hashes)
            return embeddings, mask

        handle = encoder.register_forward_pre_hook(count_encoder)
        self._audio_cache_probe = (state, originals, handle)
        runner._batch_mm_inputs_from_scheduler = batch_inputs
        runner._gather_mm_embeddings = gather
        return True

    def read_audio_cache_probe(self):
        """Return and reset counters after a completed request."""
        state, _, _ = self._audio_cache_probe
        result = {key: value.copy() if isinstance(value, list) else value for key, value in state.items()}
        state.update(calls=0, batch_sizes=[], encoded_hashes=[], gathered_hashes=[])
        return result

    def stop_audio_cache_probe(self):
        if not hasattr(self, "_audio_cache_probe"):
            return True
        _, originals, handle = self._audio_cache_probe
        handle.remove()
        for name, method in originals.items():
            setattr(self.model_runner, name, method)
        del self._audio_cache_probe
        return True


def audio_cache_rpc(runner, method):
    results = runner.omni.engine.collective_rpc(method, stage_ids=[0], timeout=30)
    assert len(results) == 1, f"Expected one Stage 0 replica, got {results!r}"
    workers = results[0]
    assert isinstance(workers, list) and workers, f"Stage 0 RPC failed: {results!r}"
    return workers


def assert_audio_cache_trace(trace, expected_calls, expected_items):
    """Check each request on every TP worker, including full cache hits."""
    assert len(trace) == len(expected_calls) == len(expected_items)
    num_workers = len(trace[0])
    assert num_workers > 0
    assert all(len(step) == num_workers for step in trace), "Worker count changed during probe"
    for rank in range(num_workers):
        calls = [step[rank]["calls"] for step in trace]
        print(f"Stage 0 TP rank {rank}: Whisper calls per request = {calls}")
        assert calls == expected_calls, f"TP rank {rank}: encoder calls {calls}, expected {expected_calls}"
        for index, (step, items) in enumerate(zip(trace, expected_items)):
            stats = step[rank]
            assert len(stats["encoded_hashes"]) == items, f"TP rank {rank}, request {index}: {stats!r}"
            # Each one-second clip fits in one Whisper chunk.
            assert sum(stats["batch_sizes"]) == items, f"TP rank {rank}, request {index}: {stats!r}"
            assert stats["gathered_hashes"], f"TP rank {rank}, request {index}: no audio embeddings gathered"
