"""
Analyze dumped Mamba radix tree states.

Usage:
    python tmp/scripts/analyze_mamba_states.py /tmp/mamba_states

Expects a dump directory produced by POST /debug/dump_mamba_tree containing:
  - tree_info.json
  - node_*.safetensors
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def print_tree_structure(tree):
    print("=" * 70)
    print("TREE STRUCTURE")
    print("=" * 70)
    for n in tree["nodes"]:
        parent_str = (
            f"parent={n['parent_id']}" if n["parent_id"] is not None else "ROOT"
        )
        mamba_str = (
            f"mamba_idx={n.get('mamba_pool_idx', 'N/A')}"
            if n["has_mamba"]
            else "no_mamba"
        )
        children_str = f"children={n['children']}" if n["children"] else "leaf"
        print(
            f"  Node {n['id']:>3}: {parent_str:>12} | "
            f"tokens={n['token_count']:>5} | {mamba_str:>15} | {children_str}"
        )


def analyze_node(dump_dir, node_id):
    filepath = os.path.join(dump_dir, f"node_{node_id}.safetensors")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return None

    with safe_open(filepath, framework="pt") as f:
        metadata = f.metadata()

    state = load_file(filepath)

    # print(f"\n--- Node {node_id} ---")
    # print(f"  Metadata: {json.dumps(metadata, indent=4)}")
    # print(f"  Tensors:")
    # for k, v in sorted(state.items()):
    #     print(
    #         f"    {k}: shape={list(v.shape)}, dtype={v.dtype}, "
    #         f"norm={v.norm().item():.6f}, mean={v.float().mean().item():.6e}, "
    #         f"std={v.float().std().item():.6e}"
    #     )

    has_parent = metadata.get("has_parent_state", "False") == "True"
    if has_parent:
        child_temporal = state["child_temporal"]
        parent_temporal = state["parent_temporal"]
        diff = child_temporal.float() - parent_temporal.float()
        print(f"  SSM temporal drift:")
        print(f"    L2 norm of diff: {diff.norm().item():.6f}")
        print(f"    Max abs diff:    {diff.abs().max().item():.6e}")
        print(f"    Mean abs diff:   {diff.abs().mean().item():.6e}")
        print(
            f"    Relative diff:   "
            f"{diff.norm().item() / (parent_temporal.float().norm().item() + 1e-12):.6f}"
        )

        num_layers = child_temporal.shape[0]
        print(f"    Per-layer L2 drift (temporal, {num_layers} layers):")
        for layer in range(num_layers):
            layer_diff = diff[layer].norm().item()
            layer_parent_norm = parent_temporal[layer].float().norm().item()
            rel = layer_diff / (layer_parent_norm + 1e-12)
            print(f"      Layer {layer:>2}: abs={layer_diff:.6f}, rel={rel:.6f}")

        conv_keys_child = sorted([k for k in state if k.startswith("child_conv_")])
        conv_keys_parent = sorted([k for k in state if k.startswith("parent_conv_")])
        if conv_keys_child and conv_keys_parent:
            for ck, pk in zip(conv_keys_child, conv_keys_parent):
                cdiff = state[ck].float() - state[pk].float()
                print(
                    f"    Conv drift ({ck} vs {pk}): "
                    f"L2={cdiff.norm().item():.6f}, max={cdiff.abs().max().item():.6e}"
                )
    else:
        print(f"  No parent state (parent node has no mamba value)")

    return state


def linear_cka_centered_torch(kv1: torch.Tensor, kv2: torch.Tensor) -> torch.Tensor:
    """
    Centered linear CKA (Kornblith et al. 2019) for (L, D) tensors.
    Returns scalar similarity in [0, 1].
    """
    assert kv1.shape[1] == kv2.shape[1], "kv1, kv2 must have same embedding dimension."
    device = kv1.device

    kv1_centered = kv1 - kv1.mean(dim=0, keepdim=True)
    kv2_centered = kv2 - kv2.mean(dim=0, keepdim=True)

    xtx = (kv1_centered.T @ kv1_centered).norm(p="fro")
    yty = (kv2_centered.T @ kv2_centered).norm(p="fro")
    xty = (kv1_centered.T @ kv2_centered).norm(p="fro")

    if xtx == 0 or yty == 0:
        return torch.tensor(0.0, device=device, dtype=kv1.dtype)

    return (xty**2) / (xtx * yty)


def cka_analysis_grouped_layers(temporal, node_id, token_len, group_size=3):
    """
    Analyze intra-group CKA similarity of the raw SSM state.
    Every `group_size` SSM layers are grouped together.
    Within each group, compute CKA between layers 1/2 and 2/3.
    temporal shape: [num_layers, num_heads, head_dim, state_size]

    Computes CKA in two views:
      - Normal:     [num_heads*head_dim, state_size]  (rows = head channels, cols = state dims)
      - Transposed: [state_size, num_heads*head_dim]  (rows = state dims, cols = head channels)
    """
    t = temporal.float()
    num_layers = t.shape[0]
    state_size = t.shape[-1]
    num_groups = num_layers // group_size
    flat_rows = t[0].reshape(-1, state_size).shape[0]

    print(f"\n{'=' * 70}")
    print(f"INTRA-GROUP CKA (raw SSM state): Node {node_id} ({token_len} tok)")
    print(f"  {num_layers} layers grouped by {group_size} -> {num_groups} groups")
    print(
        f"  Normal view:     [{flat_rows}, {state_size}] (head channels x state dims)"
    )
    print(
        f"  Transposed view: [{state_size}, {flat_rows}] (state dims x head channels)"
    )
    print(f"{'=' * 70}")
    print()

    header = (
        f"  {'Group':>6} | {'Layers':>12} | "
        f"{'CKA(1,2)':>10} | {'CKA(2,3)':>10} | "
        f"{'CKA_T(1,2)':>12} | {'CKA_T(2,3)':>12}"
    )
    print(header)
    print(
        f"  {'-' * 6}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 12}"
    )

    for g in range(num_groups):
        base = g * group_size
        layers_in_group = list(range(base, base + group_size))

        # Normal: [heads*head_dim, state_size]
        mats = [t[l].reshape(-1, state_size) for l in layers_in_group]
        # Transposed: [state_size, heads*head_dim]
        mats_t = [m.T for m in mats]

        cka_12 = linear_cka_centered_torch(mats[0], mats[1]).item()
        cka_23 = linear_cka_centered_torch(mats[1], mats[2]).item()
        cka_t_12 = linear_cka_centered_torch(mats_t[0], mats_t[1]).item()
        cka_t_23 = linear_cka_centered_torch(mats_t[1], mats_t[2]).item()

        layers_str = f"{layers_in_group[0]},{layers_in_group[1]},{layers_in_group[2]}"
        print(
            f"  {g:>6} | {layers_str:>12} | "
            f"{cka_12:>10.6f} | {cka_23:>10.6f} | "
            f"{cka_t_12:>12.6f} | {cka_t_23:>12.6f}"
        )

    # All consecutive layer pairs
    print()
    print(f"  All consecutive pairs:")
    header2 = f"  {'Pair':>8} | {'CKA':>10} | {'CKA_T':>10}"
    print(header2)
    print(f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 10}")
    for l in range(num_layers - 1):
        mat_a = t[l].reshape(-1, state_size)
        mat_b = t[l + 1].reshape(-1, state_size)
        cka = linear_cka_centered_torch(mat_a, mat_b).item()
        cka_t = linear_cka_centered_torch(mat_a.T, mat_b.T).item()
        print(f"  {l:>3},{l + 1:<3} | {cka:>10.6f} | {cka_t:>10.6f}")


def _svd_stats(S):
    """Compute SVD energy percentages at 10/20/30/40% rank thresholds."""
    n = len(S)
    total_energy = (S**2).sum().item()
    if total_energy < 1e-12:
        return None
    cumulative_energy = torch.cumsum(S**2, dim=0) / total_energy
    pcts = {}
    for frac in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        k = max(1, int(round(n * frac)))
        pcts[frac] = cumulative_energy[min(k - 1, n - 1)].item() * 100
    return pcts


def svd_analysis_temporal(
    parent_temporal,
    child_temporal,
    parent_id,
    child_id,
    parent_token_len=None,
    child_token_len=None,
):
    t_a = parent_temporal.float()
    t_b = child_temporal.float()
    diff = t_b - t_a
    num_layers = diff.shape[0]
    state_size = diff.shape[-1]

    token_delta = None
    if parent_token_len is not None and child_token_len is not None:
        token_delta = child_token_len - parent_token_len

    print(f"\n{'=' * 70}")
    print(f"TEMPORAL SVD (per-layer): Parent {parent_id} -> Child {child_id}")
    if parent_token_len is not None:
        print(
            f"  Parent tokens: {parent_token_len}, Child tokens: {child_token_len}, Delta: {token_delta}"
        )
    print(
        f"  Shape per layer: {list(diff.shape[1:])} -> reshaped to [{diff[0].reshape(-1, state_size).shape[0]}, {state_size}]"
    )
    print(f"{'=' * 70}")
    print(f"  Overall diff L2: {diff.norm().item():.4f}")
    print()

    header = (
        f"  {'Layer':>5} | {'Diff L2':>10} | "
        f"{'Top10%':>8} | {'Top20%':>8} | {'Top30%':>8} | {'Top40%':>8} | {'Top50%':>8} | {'Top60%':>8} | {'Top70%':>8}"
    )
    print(header)
    print(
        f"  {'-' * 5}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}"
    )

    all_sv_ratios = []
    for layer in range(num_layers):
        mat = diff[layer].reshape(-1, state_size)
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

        pcts = _svd_stats(S)
        if pcts is None:
            print(
                f"  {layer:>5} | {'~zero':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"
            )
            continue

        layer_norm = diff[layer].norm().item()
        print(
            f"  {layer:>5} | {layer_norm:>10.4f} | "
            f"{pcts[0.10]:>7.1f}% | {pcts[0.20]:>7.1f}% | {pcts[0.30]:>7.1f}% | {pcts[0.40]:>7.1f}% | {pcts[0.50]:>7.1f}% | {pcts[0.60]:>7.1f}% | {pcts[0.70]:>7.1f}%"
        )
        all_sv_ratios.append(S / S[0])

    # if all_sv_ratios:
    #     avg_ratios = torch.stack(all_sv_ratios).mean(dim=0)
    #     print(f"\n  Avg normalized singular values (first 20 of {len(avg_ratios)}):")
    #     for i in range(min(20, len(avg_ratios))):
    #         bar = "#" * int(avg_ratios[i].item() * 50)
    #         print(f"    SV {i:>3}: {avg_ratios[i].item():.4f} |{bar}")

    # Cross-layer analysis: combine each pair of neighboring layers
    print(f"\n  {'=' * 70}")
    print(
        f"  CROSS-LAYER SVD (2 neighboring layers combined): Parent {parent_id} -> Child {child_id}"
    )
    if parent_token_len is not None:
        print(
            f"  Parent tokens: {parent_token_len}, Child tokens: {child_token_len}, Delta: {token_delta}"
        )
    print(
        f"  Concatenating layer[i] and layer[i+1] along last dim -> [{diff[0].reshape(-1, state_size).shape[0]}, {state_size * 2}]"
    )
    print(f"  {'=' * 70}")
    print()

    header2 = (
        f"  {'Layers':>8} | {'Diff L2':>10} | "
        f"{'Top10%':>8} | {'Top20%':>8} | {'Top30%':>8} | {'Top40%':>8} | {'Top50%':>8} | {'Top60%':>8} | {'Top70%':>8}"
    )
    print(header2)
    print(
        f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}"
    )

    cross_sv_ratios = []
    for layer in range(num_layers - 1):
        # Reshape each layer to [heads*head_dim, state_size], then concat along last dim
        mat_a = diff[layer].reshape(-1, state_size)  # [4096, 128]
        mat_b = diff[layer + 1].reshape(-1, state_size)  # [4096, 128]
        mat = torch.cat([mat_a, mat_b], dim=-1)  # [4096, 256]

        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

        pcts = _svd_stats(S)
        if pcts is None:
            print(
                f"  {layer:>3},{layer + 1:<3} | {'~zero':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"
            )
            continue

        pair_norm = mat.norm().item()
        print(
            f"  {layer:>3},{layer + 1:<3} | {pair_norm:>10.4f} | "
            f"{pcts[0.10]:>7.1f}% | {pcts[0.20]:>7.1f}% | {pcts[0.30]:>7.1f}% | {pcts[0.40]:>7.1f}% | {pcts[0.50]:>7.1f}% | {pcts[0.60]:>7.1f}% | {pcts[0.70]:>7.1f}%"
        )
        cross_sv_ratios.append(S / S[0])

    # Cross-layer analysis: combine each group of 3 neighboring layers (8 groups for 24 layers)
    group_size = 3
    num_groups = num_layers // group_size
    flat_rows = diff[0].reshape(-1, state_size).shape[0]

    print(f"\n  {'=' * 70}")
    print(
        f"  CROSS-LAYER SVD (3-layer groups): Parent {parent_id} -> Child {child_id}"
    )
    if parent_token_len is not None:
        print(
            f"  Parent tokens: {parent_token_len}, Child tokens: {child_token_len}, Delta: {token_delta}"
        )
    print(
        f"  Concatenating 3 layers along last dim -> [{flat_rows}, {state_size * group_size}]"
    )
    print(f"  {num_layers} layers / {group_size} = {num_groups} groups")
    print(f"  {'=' * 70}")
    print()

    header3 = (
        f"  {'Group':>6} | {'Layers':>12} | {'Diff L2':>10} | "
        f"{'Top10%':>8} | {'Top20%':>8} | {'Top30%':>8} | {'Top40%':>8} | {'Top50%':>8} | {'Top60%':>8} | {'Top70%':>8}"
    )
    print(header3)
    print(
        f"  {'-' * 6}-+-{'-' * 12}-+-{'-' * 10}-+-"
        f"{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}"
    )

    for g in range(num_groups):
        base = g * group_size
        layers_in_group = list(range(base, base + group_size))

        # Concat 3 layers along last dim -> [flat_rows, state_size * 3]
        mats = [diff[l].reshape(-1, state_size) for l in layers_in_group]
        mat = torch.cat(mats, dim=-1)

        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

        pcts = _svd_stats(S)
        layers_str = ",".join(str(l) for l in layers_in_group)
        if pcts is None:
            print(
                f"  {g:>6} | {layers_str:>12} | {'~zero':>10} | "
                f"{'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"
            )
            continue

        group_norm = mat.norm().item()
        print(
            f"  {g:>6} | {layers_str:>12} | {group_norm:>10.4f} | "
            f"{pcts[0.10]:>7.1f}% | {pcts[0.20]:>7.1f}% | {pcts[0.30]:>7.1f}% | {pcts[0.40]:>7.1f}% | {pcts[0.50]:>7.1f}% | {pcts[0.60]:>7.1f}% | {pcts[0.70]:>7.1f}%"
        )


def svd_analysis_conv(states, node_a, node_b):
    conv_keys = sorted([k for k in states[node_a] if k.startswith("child_conv_")])
    if not conv_keys:
        return

    for conv_key in conv_keys:
        c_a = states[node_a][conv_key].float()
        c_b = states[node_b][conv_key].float()
        diff = c_a - c_b
        num_layers = diff.shape[0]

        print(f"\n--- Conv SVD ({conv_key}): Node {node_a} vs Node {node_b} ---")
        print(
            f"  Conv shape per layer: {list(diff.shape[1:])}. "
            f"Max rank={min(diff.shape[1], diff.shape[2]) if diff.dim() == 3 else 'N/A'}"
        )
        print(f"  {'Layer':>5} | {'Diff L2':>10} | SV ratios | Top-1 %")
        for layer in range(num_layers):
            mat = diff[layer]
            if mat.dim() == 1:
                mat = mat.unsqueeze(-1)
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            layer_norm = diff[layer].norm().item()
            total_e = (S**2).sum().item()
            top1 = (S[0] ** 2).item() / (total_e + 1e-12) * 100
            sv_str = ", ".join([f"{s:.4f}" for s in S.tolist()])
            print(f"  {layer:>5} | {layer_norm:>10.4f} | [{sv_str}] top-1={top1:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze dumped Mamba tree states")
    parser.add_argument("dump_dir", help="Path to dump directory")
    args = parser.parse_args()

    dump_dir = args.dump_dir
    tree_path = os.path.join(dump_dir, "tree_info.json")
    if not os.path.exists(tree_path):
        print(f"Error: {tree_path} not found", file=sys.stderr)
        sys.exit(1)

    tree = json.load(open(tree_path))

    # Print tree
    print_tree_structure(tree)

    # Find nodes with mamba state
    mamba_node_ids = [n["id"] for n in tree["nodes"] if n["has_mamba"]]
    print(f"\nMamba nodes: {mamba_node_ids}")

    # Per-node analysis
    print("\n" + "=" * 70)
    print("STATE ANALYSIS PER NODE")
    print("=" * 70)

    states = {}
    for nid in mamba_node_ids:
        state = analyze_node(dump_dir, nid)
        if state is not None:
            states[nid] = state

    # Find parent-child pairs: nodes whose safetensors contain both parent and child state
    parent_child_pairs = []
    for nid in mamba_node_ids:
        if nid not in states:
            continue
        filepath = os.path.join(dump_dir, f"node_{nid}.safetensors")
        with safe_open(filepath, framework="pt") as f:
            metadata = f.metadata()
        if metadata.get("has_parent_state", "False") == "True":
            parent_id = int(metadata["parent_node_id"])
            parent_token_len = int(metadata.get("parent_token_path_len", 0))
            child_token_len = int(metadata.get("child_token_path_len", 0))
            parent_child_pairs.append(
                (parent_id, nid, parent_token_len, child_token_len)
            )

    if not parent_child_pairs:
        print(
            "\nNo parent-child pairs with both states found. "
            "The parent nodes are likely tombstones (no mamba state)."
        )
        return

    # Parent-child comparison
    print(f"\n{'=' * 70}")
    print("PARENT-CHILD COMPARISON")
    print("=" * 70)
    for parent_id, child_id, p_tokens, c_tokens in parent_child_pairs:
        state = states[child_id]
        parent_temporal = state["parent_temporal"].float()
        child_temporal = state["child_temporal"].float()
        diff = child_temporal - parent_temporal
        cosine = torch.nn.functional.cosine_similarity(
            parent_temporal.flatten(), child_temporal.flatten(), dim=0
        ).item()
        print(
            f"  Parent {parent_id} ({p_tokens} tok) -> Child {child_id} ({c_tokens} tok), "
            f"delta={c_tokens - p_tokens} tok: "
            f"temporal L2={diff.norm().item():.6f}, "
            f"max_abs={diff.abs().max().item():.6e}, cosine_sim={cosine:.6f}"
        )

    # SVD low-rank analysis
    print(f"\n{'=' * 70}")
    print("LOW-RANK ANALYSIS (SVD)")
    print("=" * 70)
    for parent_id, child_id, p_tokens, c_tokens in parent_child_pairs:
        state = states[child_id]
        svd_analysis_temporal(
            state["parent_temporal"],
            state["child_temporal"],
            parent_id,
            child_id,
            parent_token_len=p_tokens,
            child_token_len=c_tokens,
        )

    # # Intra-group CKA analysis on raw SSM states
    # print(f"\n{'=' * 70}")
    # print("INTRA-GROUP CKA ANALYSIS (raw SSM state, grouped by 3 layers)")
    # print("=" * 70)
    # for nid in mamba_node_ids:
    #     if nid not in states:
    #         continue
    #     filepath = os.path.join(dump_dir, f"node_{nid}.safetensors")
    #     with safe_open(filepath, framework="pt") as f:
    #         metadata = f.metadata()
    #     token_len = int(metadata.get("child_token_path_len", 0))
    #     cka_analysis_grouped_layers(
    #         states[nid]["child_temporal"], nid, token_len
    #     )


if __name__ == "__main__":
    main()
