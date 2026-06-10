"""Split-KV attention for the SMC linear TARGET_VERIFY path.

The stock linear-verify path runs the 2-stage extend kernel with grid
``(bs, q_heads, ceil(ext/BLOCK_M))`` — at SMC decode widths that is on the
order of a hundred CTAs, each serially scanning the request's whole prefix,
so occupancy (and latency) degrades linearly with context length.

This module reuses the decode path's flash-decoding machinery instead:

  stage 1  ``_fwd_grouped_kernel_stage1`` (unmodified, from
           decode_attention.py) over per-TOKEN virtual rows: row
           ``(r, i)`` = verify token ``i`` of request ``r``, attending the
           request's prefix KV from the cache.  All ``ext`` rows of a
           request share the same prefix range — causality only matters
           within the in-flight block, which stage 1 never sees.
           Grid ``(bs*ext, q_heads/BLOCK_H, kv_splits)``.

  stage 2  ``_verify_inflight_merge_kernel`` (below): per (row, q_head),
           computes the tiny causal attention over the ≤ ext in-flight
           tokens (read from the layer's k/v inputs, exactly like the
           extend kernel's in-flight part — their KV cache writes are NOT
           in stage 1's prefix indices), then merges the stage-1 split
           partials with the standard LSE merge and writes the final
           output.  Mirrors ``_fwd_kernel_stage2`` semantics: stage-1
           partials are (acc/e_sum, e_max + log(e_sum)).

Metadata contract (built by the triton backend when SMC_VERIFY_KV_SPLITS=1):
  token_kv_indptr     (bs*ext + 1,)  cumsum of per-row prefix lengths
                      (= request prefix length repeated ext times)
  token_kv_indices    request r's prefix cache locations, repeated for each
                      of its ext rows
  token_num_kv_splits (bs*ext,) per-row split count (get_num_kv_splits)

Supported: plain dense GQA/MHA attention (no MLA/DPE head split, no
sliding window, no sinks, no fp8 KV descale, no xai temperature).  The
backend falls back to the stock extend path otherwise.
"""

import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_grouped_att_m_fwd,
)


@triton.jit
def _verify_inflight_merge_kernel(
    Q,
    K_Inflight,
    V_Inflight,
    O,
    Mid_O,
    Mid_Lse,
    kv_indptr,
    num_kv_splits,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_ob,
    stride_oh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    kv_group_num: tl.constexpr,
    EXT: tl.constexpr,
    BLOCK_EXT: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // kv_group_num
    # Rows are request-major and uniform (ext per request): the in-flight
    # keys visible to this row are k/v rows [row - pos, row].
    pos = row % EXT
    base = row - pos

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    q = tl.load(
        Q + row * stride_qb + head * stride_qh + offs_d, mask=mask_d, other=0.0
    ).to(tl.float32)

    # ---- in-flight causal partial (≤ EXT keys, from layer inputs) ----
    offs_j = tl.arange(0, BLOCK_EXT)
    mask_j = offs_j <= pos
    k = tl.load(
        K_Inflight
        + (base + offs_j)[:, None] * stride_kb
        + kv_head * stride_kh
        + offs_d[None, :],
        mask=mask_j[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    qk = tl.sum(q[None, :] * k, 1) * sm_scale
    if logit_cap > 0:
        qk = logit_cap * (2 * tl.sigmoid(2 * (qk / logit_cap)) - 1)
    qk = tl.where(mask_j, qk, float("-inf"))

    e_max = tl.max(qk, 0)
    p = tl.where(mask_j, tl.exp(qk - e_max), 0.0)
    v = tl.load(
        V_Inflight
        + (base + offs_j)[:, None] * stride_vb
        + kv_head * stride_vh
        + offs_dv[None, :],
        mask=mask_j[:, None] & mask_dv[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(p[:, None] * v, 0)
    e_sum = tl.sum(p, 0)

    # ---- merge stage-1 prefix split partials (same math as stage 2) ----
    prefix_len = tl.load(kv_indptr + row + 1) - tl.load(kv_indptr + row)
    kv_splits = tl.load(num_kv_splits + row)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(prefix_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, prefix_len)
        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O
                + row * stride_mid_ob
                + head * stride_mid_oh
                + split_kv_id * stride_mid_os
                + offs_dv,
                mask=mask_dv,
                other=0.0,
            )
            tlse = tl.load(
                Mid_Lse
                + row * stride_lse_b
                + head * stride_lse_h
                + split_kv_id * stride_lse_s
            )
            n_e_max = tl.maximum(tlse, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            exp_lse = tl.exp(tlse - n_e_max)
            acc = acc * old_scale + exp_lse * tv
            e_sum = e_sum * old_scale + exp_lse
            e_max = n_e_max

    tl.store(
        O + row * stride_ob + head * stride_oh + offs_dv,
        acc / e_sum,
        mask=mask_dv,
    )


def linear_verify_split_attention(
    q,
    k_inflight,
    v_inflight,
    o,
    k_buffer,
    v_buffer,
    token_kv_indptr,
    token_kv_indices,
    token_num_kv_splits,
    attn_logits,
    attn_lse,
    max_kv_splits,
    ext,
    sm_scale,
    logit_cap=0.0,
):
    """Split-KV linear TARGET_VERIFY attention.

    q:                     (bs*ext, q_heads, Lk)
    k_inflight/v_inflight: (bs*ext, kv_heads, Lk/Lv) — the layer's new k/v
    o:                     (bs*ext, q_heads, Lv) output
    attn_logits/attn_lse:  stage-1 partial buffers covering bs*ext rows
    """
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _MIN_BLOCK_KV,
    )

    num_tokens, num_heads = q.shape[0], q.shape[1]
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    kv_group_num = num_heads // k_inflight.shape[1]

    # Stage 1: prefix partials via the stock grouped split-KV decode kernel.
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        token_kv_indptr,
        token_kv_indices,
        token_num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )

    # Stage 2: in-flight causal part + split merge.
    grid = (num_tokens, num_heads)
    _verify_inflight_merge_kernel[grid](
        q,
        k_inflight,
        v_inflight,
        o,
        attn_logits,
        attn_lse,
        token_kv_indptr,
        token_num_kv_splits,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_inflight.stride(0),
        k_inflight.stride(1),
        v_inflight.stride(0),
        v_inflight.stride(1),
        o.stride(0),
        o.stride(1),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        attn_lse.stride(2),
        kv_group_num=kv_group_num,
        EXT=ext,
        BLOCK_EXT=triton.next_power_of_2(ext),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_DV=triton.next_power_of_2(Lv),
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )
