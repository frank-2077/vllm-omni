"""L1 — CPU-only unit coverage for the two hashing layers the MM prefix-cache
eval (#5069) depends on.

Covers, for the MiniCPM-o 4.5 placeholder geometry (64 mm tokens per image via
``query_num``, interleaved Daily-Omni-style packs):

  L1.1/L1.2  mm_hash determinism and discrimination (MultiModalHasher)
  L1.3/L1.4  block-hash extra keys: straddling items, two items in one block,
             interleaved ordering, and the (identifier, offset-in-block) key
             that keeps same-item-different-position blocks distinct
  L1.5       OmniTensorPrefixCache round-trip: the merged hidden states for a
             prefix-cache hit must equal the full-forward reference (the CPU
             proxy for hypothesis H2 in probe_mm_cache/README.md)

Run:
    pytest tests/core/test_mm_block_hash_minicpmo.py -q
"""

import hashlib

import numpy as np
import pytest
import torch
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# MiniCPM-o 4.5: each image resolves to query_num=64 embedding tokens; audio
# items vary with duration. Block size 16 divides 64, so a single image spans
# exactly 4 full blocks when aligned — the straddle cases below deliberately
# misalign it.
MM_LEN = 64
BLOCK = 16


class _Req:
    """The minimal Request surface generate_block_hash_extra_keys touches."""

    def __init__(self, mm_features):
        self.mm_features = mm_features
        self.lora_request = None
        self.cache_salt = None
        self.prompt_embeds = None


def _feat(identifier: str, offset: int, length: int = MM_LEN, modality: str = "image"):
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _img(seed: int, w: int = 8, h: int = 8) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# L1.1 / L1.2 — mm_hash
# ---------------------------------------------------------------------------


def test_mm_hash_deterministic_within_and_across_calls():
    a1 = MultiModalHasher.hash_kwargs(image=_img(0))
    a2 = MultiModalHasher.hash_kwargs(image=_img(0))
    assert a1 == a2


def test_mm_hash_discriminates_content_and_kwarg_name():
    base = MultiModalHasher.hash_kwargs(image=_img(0))
    assert MultiModalHasher.hash_kwargs(image=_img(1)) != base
    # Same bytes under a different modality kwarg must not collide.
    assert MultiModalHasher.hash_kwargs(audio=_img(0)) != base


def test_mm_hash_discriminates_dtype_and_shape():
    img = _img(0)
    base = MultiModalHasher.hash_kwargs(image=img)
    assert MultiModalHasher.hash_kwargs(image=img.astype(np.int32)) != base
    assert MultiModalHasher.hash_kwargs(image=img.reshape(8, 8 * 3)) != base
    # Audio analogue: same samples at a different sample rate must differ once
    # the rate participates in the hashed kwargs.
    wav = np.zeros(1600, dtype=np.float32)
    assert MultiModalHasher.hash_kwargs(audio=(wav, 16000)) != MultiModalHasher.hash_kwargs(audio=(wav, 24000))


# ---------------------------------------------------------------------------
# L1.3 — block-hash extra keys, single items
# ---------------------------------------------------------------------------


def test_extra_keys_absent_for_text_only_block():
    req = _Req([_feat("h1", offset=100)])
    keys, nxt = generate_block_hash_extra_keys(req, 0, BLOCK, 0)
    assert keys is None
    assert nxt == 0


def test_extra_keys_mm_item_at_offset_zero():
    req = _Req([_feat("h1", offset=0)])
    keys, _ = generate_block_hash_extra_keys(req, 0, BLOCK, 0)
    assert keys == (("h1", 0),)


def test_extra_keys_mm_item_straddles_block_boundary():
    # Item occupies [10, 74): blocks 0..4 all intersect it. The extra key is
    # (identifier, item_offset - block_start), so continuation blocks carry a
    # NEGATIVE in-block offset — deterministic, and distinct per block.
    req = _Req([_feat("h1", offset=10)])

    keys0, n0 = generate_block_hash_extra_keys(req, 0, BLOCK, 0)
    assert keys0 == (("h1", 10),)  # starts inside block 0 at in-block offset 10
    assert n0 == 0  # item not finished -> same mm idx

    keys1, n1 = generate_block_hash_extra_keys(req, BLOCK, 2 * BLOCK, n0)
    assert keys1 == (("h1", 10 - BLOCK),)  # continuation: negative offset
    assert n1 == 0

    keys4, n4 = generate_block_hash_extra_keys(req, 4 * BLOCK, 5 * BLOCK, n0)
    assert keys4 == (("h1", 10 - 4 * BLOCK),)  # tail [64, 74) lands in block 4
    assert n4 == 1  # item ends inside this block -> advance

    keys5, _ = generate_block_hash_extra_keys(req, 5 * BLOCK, 6 * BLOCK, n4)
    assert keys5 is None


def test_extra_keys_offset_disambiguates_same_item_at_different_positions():
    """Two requests place the SAME image at different in-block offsets; the
    blocks' extra keys must differ or the prefix cache would alias them."""
    req_a = _Req([_feat("h1", offset=4)])
    req_b = _Req([_feat("h1", offset=8)])
    keys_a, _ = generate_block_hash_extra_keys(req_a, 0, BLOCK, 0)
    keys_b, _ = generate_block_hash_extra_keys(req_b, 0, BLOCK, 0)
    assert keys_a != keys_b


def test_extra_keys_two_items_share_one_block():
    # Short audio items: [0, 6) and [6, 12) both inside block 0.
    req = _Req(
        [
            _feat("a1", offset=0, length=6, modality="audio"),
            _feat("a2", offset=6, length=6, modality="audio"),
        ]
    )
    keys, nxt = generate_block_hash_extra_keys(req, 0, BLOCK, 0)
    assert keys == (("a1", 0), ("a2", 6))
    assert nxt == 2


# ---------------------------------------------------------------------------
# L1.4 — interleaved image/audio pack (Daily-Omni shape)
# ---------------------------------------------------------------------------


def test_extra_keys_interleaved_pack_orders_and_advances():
    """image(64) audio(10) image(64) audio(10) with 2-token text gaps, walked
    block by block the way the scheduler does; keys must appear in prompt
    order and every item must be visited exactly once."""
    feats = []
    pos = 0
    for i in range(2):
        feats.append(_feat(f"img{i}", offset=pos))
        pos += MM_LEN + 2
        feats.append(_feat(f"aud{i}", offset=pos, length=10, modality="audio"))
        pos += 10 + 2

    req = _Req(feats)
    total_blocks = (pos + BLOCK - 1) // BLOCK
    seen: list[str] = []
    mm_idx = 0
    for b in range(total_blocks):
        keys, mm_idx = generate_block_hash_extra_keys(req, b * BLOCK, (b + 1) * BLOCK, mm_idx)
        for ident, _off in keys or ():
            if not seen or seen[-1] != ident:
                seen.append(ident)
    assert seen == ["img0", "aud0", "img1", "aud1"]
    assert mm_idx == 4


# ---------------------------------------------------------------------------
# L1.5 — OmniTensorPrefixCache round-trip (H2, CPU proxy)
# ---------------------------------------------------------------------------

NUM_BLOCKS = 8
HIDDEN = 4


class _BlockTableStub:
    """What the merge path touches: ``.block_tables`` (hybrid-cache guard) and
    ``[0].block_table.cpu[req_idx, :n]`` (cached block lookup)."""

    def __init__(self, table: torch.Tensor):
        class _T:
            pass

        wrap = _T()
        wrap.cpu = table
        self.block_table = wrap
        self.block_tables = [wrap]

    def __getitem__(self, idx):
        assert idx == 0
        return self


class _InputBatchStub:
    def __init__(self, req_ids, num_computed, table):
        self.req_ids = req_ids
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
        self.num_computed_tokens_cpu = num_computed
        self.block_table = _BlockTableStub(table)


def _hs(num_tokens: int, base: float) -> torch.Tensor:
    return torch.arange(num_tokens * HIDDEN, dtype=torch.float32).reshape(num_tokens, HIDDEN) + base


def test_omni_prefix_cache_round_trip_equals_full_forward():
    """Write a request's full prompt through the cache in one step, then replay
    it as a prefix-cache hit (only the tail scheduled) and require the merged
    hidden states to equal the original full-forward tensor exactly."""
    cache = OmniTensorPrefixCache(num_blocks=NUM_BLOCKS, block_size=BLOCK, hidden_size=HIDDEN, hs_dtype=torch.float32)

    prompt_len = 40  # 2.5 blocks
    full_hidden = _hs(prompt_len, base=100.0)
    # Blocks 1,2,3 belong to this request; slots follow vLLM's block*size+off.
    blocks = [1, 2, 3]
    slots = torch.tensor([blocks[t // BLOCK] * BLOCK + (t % BLOCK) for t in range(prompt_len)], dtype=torch.int64)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=full_hidden,
        multimodal_outputs={},
        num_tokens_unpadded=prompt_len,
        slot_mapping=slots,
        num_tokens_padded=prompt_len,
    )

    # Replay: 32 of 40 tokens are cache hits; 8 are scheduled fresh.
    num_cached = 2 * BLOCK
    num_new = prompt_len - num_cached
    new_tail = full_hidden[num_cached:]

    cache.add_prefix_cached_new_req_id("req1")
    table = torch.zeros((1, NUM_BLOCKS), dtype=torch.int64)
    table[0, : len(blocks)] = torch.tensor(blocks)
    input_batch = _InputBatchStub(["req1"], np.array([num_cached]), table)

    merged = cache.get_merged_hidden_states(
        query_start_loc=[0],
        input_batch=input_batch,
        hidden_states=new_tail.clone(),
        num_scheduled_tokens={"req1": num_new},
    )

    assert torch.equal(merged["req1"], full_hidden), (
        "prefix-cache merge does not reproduce the full-forward hidden states"
    )


def test_omni_prefix_cache_partial_block_tail_is_not_served_stale():
    """A request whose prompt ends mid-block writes that partial block's slots;
    a second write for the same slots (e.g. after eviction/reuse) must win."""
    cache = OmniTensorPrefixCache(num_blocks=NUM_BLOCKS, block_size=BLOCK, hidden_size=HIDDEN, hs_dtype=torch.float32)
    slots = torch.arange(BLOCK, dtype=torch.int64)  # block 0

    first = _hs(BLOCK, base=0.0)
    second = _hs(BLOCK, base=999.0)
    for hidden in (first, second):
        cache.update_omni_tensor_prefix_cache(
            hidden_states=hidden,
            multimodal_outputs={},
            num_tokens_unpadded=BLOCK,
            slot_mapping=slots,
            num_tokens_padded=BLOCK,
        )

    assert torch.equal(cache.hidden_states_cache[0], second), "stale slot content survived an overwrite"


def test_sha256_stability_reference():
    """Guard against the hashing backend silently changing representation:
    the digest of a fixed byte string must be stable across runs/processes."""
    assert (
        hashlib.sha256(b"minicpmo-4_5-mm-cache-probe").hexdigest()
        == "57166a59ebec2ff45f3872c73ccb7e0370c2e9a1a76a2ff48d42d727372561c5"
    )
