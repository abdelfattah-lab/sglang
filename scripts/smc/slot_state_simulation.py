"""
Pure-Python simulation of the slot-based SMCSlotState data structure.

Exercises the full lifecycle with 2 requests x 3 particles:
  Batched Prefill → Materialize → Decode → Weight Update → Resample →
  Particle Finish → Group Finalize → Remaining Group Continues

Run: python3 scripts/smc/slot_state_simulation.py
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
#  Minimal mocks
# ──────────────────────────────────────────────────────────────


@dataclass
class MockReq:
    rid: str
    output_ids: List[int] = field(default_factory=list)
    kv_committed_len: int = 0
    kv_allocated_len: int = 0
    req_pool_idx: Optional[int] = None
    smc_group_id: Optional[str] = None
    smc_particle_idx: int = -1
    max_new_tokens: int = 100
    eos_token_ids: List[int] = field(default_factory=lambda: [2])  # EOS=2
    finished_reason: Optional[str] = None

    def finished(self) -> bool:
        return self.finished_reason is not None


@dataclass
class MockKVPool:
    """Simulates req_to_token_pool + token_to_kv_pool_allocator."""

    next_pool_idx: int = 0
    next_token_slot: int = 100
    block_table: Dict[int, List[int]] = field(default_factory=dict)
    refcounts: Dict[int, int] = field(default_factory=dict)

    def alloc_pool_row(self) -> int:
        idx = self.next_pool_idx
        self.next_pool_idx += 1
        self.block_table[idx] = []
        return idx

    def alloc_token_slots(self, n: int) -> List[int]:
        slots = list(range(self.next_token_slot, self.next_token_slot + n))
        self.next_token_slot += n
        for s in slots:
            self.refcounts[s] = 1
        return slots

    def assign_slots_to_pool(self, pool_idx: int, start: int, slots: List[int]):
        bt = self.block_table[pool_idx]
        while len(bt) < start + len(slots):
            bt.append(-1)
        for i, s in enumerate(slots):
            bt[start + i] = s

    def copy_block_table(self, src_pool: int, dst_pool: int, length: int):
        src_bt = self.block_table[src_pool]
        dst_bt = self.block_table.setdefault(dst_pool, [])
        while len(dst_bt) < length:
            dst_bt.append(-1)
        for i in range(length):
            slot = src_bt[i]
            dst_bt[i] = slot
            self.refcounts[slot] = self.refcounts.get(slot, 0) + 1

    def free_pool_row(self, pool_idx: int, length: int):
        bt = self.block_table.get(pool_idx, [])
        for i in range(min(length, len(bt))):
            slot = bt[i]
            if slot >= 0:
                self.refcounts[slot] -= 1
                if self.refcounts[slot] <= 0:
                    del self.refcounts[slot]
        del self.block_table[pool_idx]


# ──────────────────────────────────────────────────────────────
#  v2 smc_info API
# ──────────────────────────────────────────────────────────────


@dataclass
class SMCDecodeContext:
    """Per-decode-cycle state created by scheduler, consumed by worker.

    This is the bridge between scheduler-side KV allocation and
    worker-side ForwardBatch preparation (prepare_for_draft / prepare_for_verify).
    """

    orig_seq_lens: List[int]  # committed prefix BEFORE advance
    orig_seq_lens_sum: int
    new_seq_lens: List[int]  # AFTER advance by gamma+1 
    gamma: int

    @staticmethod
    def from_slot_gather(
        seq_lens: List[int],
        kv_allocated_lens: List[int],
        req_pool_indices: List[int],
        gamma_plus_1: int,
        kv_pool: MockKVPool,
    ) -> Tuple["SMCDecodeContext", List[int]]:
        """Vectorized KV allocation — replaces the Python loop in
        SMCDraftInput.prepare_for_decode L203-217.

        Returns (ctx, new_kv_allocated_lens).
        """
        bs = len(seq_lens)
        orig_seq_lens = list(seq_lens)
        orig_seq_lens_sum = sum(seq_lens)

        # Vectorized (would be torch ops in real code)
        alloc_start = [max(kv, sl) for kv, sl in zip(kv_allocated_lens, seq_lens)]
        needed_len = [sl + gamma_plus_1 for sl in seq_lens]
        new_alloc = [max(0, nl - al) for nl, al in zip(needed_len, alloc_start)]
        num_needed = sum(new_alloc)
        nxt_kv_lens = [al + na for al, na in zip(alloc_start, new_alloc)]

        if num_needed > 0:
            token_slots = kv_pool.alloc_token_slots(num_needed)
            offset = 0
            for i in range(bs):
                if new_alloc[i] > 0:
                    kv_pool.assign_slots_to_pool(
                        req_pool_indices[i], alloc_start[i],
                        token_slots[offset : offset + new_alloc[i]],
                    )
                    offset += new_alloc[i]

        ctx = SMCDecodeContext(
            orig_seq_lens=orig_seq_lens,
            orig_seq_lens_sum=orig_seq_lens_sum,
            new_seq_lens=[sl + gamma_plus_1 for sl in seq_lens],
            gamma=gamma_plus_1 - 1,
        )
        return ctx, nxt_kv_lens


@dataclass
class SMCDraftInputV2:
    """Pure data carrier — no prepare methods. Travels on batch.spec_info."""

    verified_id: List[int]
    logprob_diff: Optional[List[float]] = None
    num_tokens_per_req: int = -1
    decode_ctx: Optional[SMCDecodeContext] = None


@dataclass
class MockModelWorkerBatch:
    """What the worker receives — contiguous, gathered from slot state."""

    forward_mode: str
    input_ids: List[int]
    req_pool_indices: List[int]
    seq_lens: List[int]
    seq_lens_sum: int
    spec_info: SMCDraftInputV2
    batch_size: int


@dataclass
class MockBatchResult:
    """What the worker returns."""

    next_token_ids: List[List[int]]  # [bs][gamma+1] accepted tokens
    logprob_diff: List[float]  # [bs] per-particle
    bonus_ids: List[int]  # [bs] next verified_id


# ──────────────────────────────────────────────────────────────
#  SMCSlotState
# ──────────────────────────────────────────────────────────────

EMPTY = -1


class SMCSlotState:
    def __init__(self, max_slots: int, gamma_plus_1: int, max_output_len: int):
        self.max_slots = max_slots
        self.gamma_plus_1 = gamma_plus_1
        self.max_output_len = max_output_len

        # Slot lifecycle (CPU)
        self.free_slots: List[int] = list(range(max_slots))
        self.slot_to_req: Dict[int, MockReq] = {}
        self.slot_to_group_id: Dict[int, str] = {}

        # Per-slot state (simulating GPU tensors as lists, shape [max_slots])
        self.req_pool_indices = [EMPTY] * max_slots
        self.seq_lens = [0] * max_slots  # = kv_committed_len
        self.kv_allocated_lens = [0] * max_slots
        self.verified_ids = [0] * max_slots
        self.token_counts = [0] * max_slots
        self.group_indices = [EMPTY] * max_slots
        self.particle_indices = [EMPTY] * max_slots
        self.finished_mask = [False] * max_slots
        self.max_new_tokens_slot = [0] * max_slots
        self.eos_ids_slot: List[List[int]] = [[] for _ in range(max_slots)]

        # Token history (simulating [max_slots, max_output_len] 2D tensor)
        self.all_token_ids = [[EMPTY] * max_output_len for _ in range(max_slots)]

        # Active batch index
        self.active_slots: List[int] = []
        self.group_active_indptr: List[int] = [0]

        # Group tracking
        self.group_slot_lists: Dict[str, List[int]] = {}
        self.group_log_weights: Dict[str, List[float]] = {}
        self.group_interval_weights: Dict[str, List[float]] = {}
        self.group_n_particles: Dict[str, int] = {}

        # KV pool
        self.kv_pool = MockKVPool()

    # ────────────── Slot Allocation ──────────────

    def allocate_slots(
        self,
        group_id: str,
        group_idx: int,
        particle_reqs: List[MockReq],
        shared_seq_len: int,
        parent_pool_idx: int,
    ) -> List[int]:
        n = len(particle_reqs)
        assert len(self.free_slots) >= n, f"Need {n} slots, have {len(self.free_slots)}"
        slots = [self.free_slots.pop(0) for _ in range(n)]

        for slot, req in zip(slots, particle_reqs):
            pool_idx = self.kv_pool.alloc_pool_row()
            self.kv_pool.copy_block_table(parent_pool_idx, pool_idx, shared_seq_len)
            req.req_pool_idx = pool_idx

            self.slot_to_req[slot] = req
            self.slot_to_group_id[slot] = group_id
            self.req_pool_indices[slot] = pool_idx
            self.seq_lens[slot] = shared_seq_len
            self.kv_allocated_lens[slot] = shared_seq_len
            self.verified_ids[slot] = req.output_ids[-1] if req.output_ids else 0
            self.token_counts[slot] = len(req.output_ids)
            self.group_indices[slot] = group_idx
            self.particle_indices[slot] = req.smc_particle_idx
            self.finished_mask[slot] = False
            self.max_new_tokens_slot[slot] = req.max_new_tokens
            self.eos_ids_slot[slot] = list(req.eos_token_ids)
            for j, tok in enumerate(req.output_ids):
                self.all_token_ids[slot][j] = tok

        self.group_slot_lists[group_id] = slots
        self.group_log_weights[group_id] = [0.0] * n
        self.group_interval_weights[group_id] = [0.0] * n
        self.group_n_particles[group_id] = n
        self.rebuild_active_slots()
        return slots

    def free_group_slots(self, group_id: str):
        slots = self.group_slot_lists.pop(group_id, [])
        for slot in slots:
            pool_idx = self.req_pool_indices[slot]
            alloc_len = self.kv_allocated_lens[slot]
            if pool_idx != EMPTY and alloc_len > 0:
                self.kv_pool.free_pool_row(pool_idx, alloc_len)
            self.req_pool_indices[slot] = EMPTY
            self.seq_lens[slot] = 0
            self.kv_allocated_lens[slot] = 0
            self.verified_ids[slot] = 0
            self.token_counts[slot] = 0
            self.group_indices[slot] = EMPTY
            self.particle_indices[slot] = EMPTY
            self.finished_mask[slot] = False
            self.slot_to_req.pop(slot, None)
            self.slot_to_group_id.pop(slot, None)
            self.free_slots.append(slot)
        self.group_log_weights.pop(group_id, None)
        self.group_interval_weights.pop(group_id, None)
        self.group_n_particles.pop(group_id, None)
        self.rebuild_active_slots()

    def rebuild_active_slots(self):
        self.active_slots = []
        self.group_active_indptr = [0]
        for group_id in sorted(self.group_slot_lists.keys()):
            active = [s for s in self.group_slot_lists[group_id]
                      if not self.finished_mask[s]]
            self.active_slots.extend(active)
            self.group_active_indptr.append(len(self.active_slots))

    # ────────────── Decode Preparation ──────────────

    def prepare_for_decode(self) -> SMCDraftInputV2:
        active = self.active_slots
        if not active:
            return SMCDraftInputV2(verified_id=[], num_tokens_per_req=self.gamma_plus_1)

        # Gather contiguous from sparse slots
        seq_lens_g = [self.seq_lens[s] for s in active]
        kv_alloc_g = [self.kv_allocated_lens[s] for s in active]
        pool_idx_g = [self.req_pool_indices[s] for s in active]
        verified_g = [self.verified_ids[s] for s in active]

        # Vectorized KV allocation via SMCDecodeContext
        ctx, new_kv_alloc = SMCDecodeContext.from_slot_gather(
            seq_lens=seq_lens_g,
            kv_allocated_lens=kv_alloc_g,
            req_pool_indices=pool_idx_g,
            gamma_plus_1=self.gamma_plus_1,
            kv_pool=self.kv_pool,
        )

        # Scatter back to sparse slots
        for i, slot in enumerate(active):
            self.kv_allocated_lens[slot] = new_kv_alloc[i]
            self.seq_lens[slot] = ctx.new_seq_lens[i]

        return SMCDraftInputV2(
            verified_id=verified_g,
            num_tokens_per_req=self.gamma_plus_1,
            decode_ctx=ctx,
        )

    # ────────────── Build ModelWorkerBatch ──────────────

    def build_model_worker_batch(
        self, draft_input: SMCDraftInputV2
    ) -> MockModelWorkerBatch:
        """Gather sparse → contiguous for the worker."""
        active = self.active_slots
        # These are already contiguous from prepare_for_decode
        ctx = draft_input.decode_ctx
        return MockModelWorkerBatch(
            forward_mode="DECODE",
            input_ids=list(draft_input.verified_id),
            req_pool_indices=[self.req_pool_indices[s] for s in active],
            seq_lens=list(ctx.new_seq_lens),  # advanced
            seq_lens_sum=sum(ctx.new_seq_lens),
            spec_info=draft_input,
            batch_size=len(active),
        )

    # ────────────── Process Batch Result ──────────────

    def process_batch_result(self, result: MockBatchResult) -> Dict:
        active = self.active_slots
        assert len(result.next_token_ids) == len(active)

        # a. Write accepted tokens (Triton scatter in real code)
        for i, slot in enumerate(active):
            offset = self.token_counts[slot]
            for j, tok in enumerate(result.next_token_ids[i]):
                self.all_token_ids[slot][offset + j] = tok
            self.token_counts[slot] += len(result.next_token_ids[i])

        # b. Update verified_ids from bonus tokens
        for i, slot in enumerate(active):
            self.verified_ids[slot] = result.bonus_ids[i]

        # c. Batched finish check (GPU comparisons in real code)
        newly_finished = []
        for i, slot in enumerate(active):
            if self.finished_mask[slot]:
                continue
            if self.token_counts[slot] >= self.max_new_tokens_slot[slot]:
                self.finished_mask[slot] = True
                newly_finished.append(slot)
                continue
            eos_set = set(self.eos_ids_slot[slot])
            for tok in result.next_token_ids[i]:
                if tok in eos_set:
                    self.finished_mask[slot] = True
                    newly_finished.append(slot)
                    break

        # d. Sync finished to Reqs (Python loop, only for newly finished)
        for slot in newly_finished:
            req = self.slot_to_req[slot]
            count = self.token_counts[slot]
            req.output_ids = list(self.all_token_ids[slot][:count])
            req.kv_committed_len = self.seq_lens[slot]
            req.kv_allocated_len = self.kv_allocated_lens[slot]
            req.finished_reason = "EOS_or_LENGTH"

        # e. Extract group logprob diffs (zero-copy slices via indptr)
        group_diffs = {}
        sorted_groups = sorted(self.group_slot_lists.keys())
        for g_idx, group_id in enumerate(sorted_groups):
            start = self.group_active_indptr[g_idx]
            end = self.group_active_indptr[g_idx + 1]
            diffs = result.logprob_diff[start:end]
            pidxs = [self.particle_indices[active[j]] for j in range(start, end)]
            group_diffs[group_id] = (pidxs, diffs)

        # f. Update group log_weights
        for group_id, (pidxs, diffs) in group_diffs.items():
            lw = self.group_log_weights[group_id]
            iw = self.group_interval_weights[group_id]
            for pidx, diff in zip(pidxs, diffs):
                lw[pidx] += diff
                iw[pidx] += diff

        # g. Rebuild active_slots if finishes occurred
        if newly_finished:
            self.rebuild_active_slots()

        return {"newly_finished": newly_finished, "group_diffs": group_diffs}

    # ────────────── Resampling ──────────────

    def resample_copy_slot(self, dst_slot: int, src_slot: int):
        # GPU tensor row copies
        self.seq_lens[dst_slot] = self.seq_lens[src_slot]
        self.kv_allocated_lens[dst_slot] = self.kv_allocated_lens[src_slot]
        self.verified_ids[dst_slot] = self.verified_ids[src_slot]
        self.finished_mask[dst_slot] = self.finished_mask[src_slot]
        src_count = self.token_counts[src_slot]
        self.token_counts[dst_slot] = src_count
        self.all_token_ids[dst_slot][:src_count] = list(
            self.all_token_ids[src_slot][:src_count]
        )

        # KV block table copy (still through pool in real code)
        src_pool = self.req_pool_indices[src_slot]
        dst_pool = self.req_pool_indices[dst_slot]
        old_alloc = self.kv_allocated_lens[dst_slot]
        if dst_pool != EMPTY and dst_pool in self.kv_pool.block_table:
            self.kv_pool.free_pool_row(dst_pool, old_alloc)
            self.kv_pool.block_table[dst_pool] = []
        src_len = self.seq_lens[src_slot]
        self.kv_pool.copy_block_table(src_pool, dst_pool, src_len)

        # Req-level text state (cold, only for finalization)
        src_req = self.slot_to_req[src_slot]
        dst_req = self.slot_to_req[dst_slot]
        dst_req.output_ids = list(src_req.output_ids)

    def apply_resample(self, group_id: str, ancestor_indices: List[int]):
        slots = self.group_slot_lists[group_id]
        n = len(slots)
        pidx_to_slot = {self.particle_indices[s]: s for s in slots}

        # Snapshot sources BEFORE any copies (avoid clobbering)
        snapshots = {}
        for src_pidx in set(ancestor_indices):
            src_slot = pidx_to_slot[src_pidx]
            snapshots[src_pidx] = {
                "seq_lens": self.seq_lens[src_slot],
                "kv_allocated_lens": self.kv_allocated_lens[src_slot],
                "verified_ids": self.verified_ids[src_slot],
                "finished_mask": self.finished_mask[src_slot],
                "token_counts": self.token_counts[src_slot],
                "all_token_ids": list(
                    self.all_token_ids[src_slot][: self.token_counts[src_slot]]
                ),
            }

        for dst_pidx, src_pidx in enumerate(ancestor_indices):
            if src_pidx == dst_pidx:
                continue  # no copy needed
            dst_slot = pidx_to_slot[dst_pidx]
            snap = snapshots[src_pidx]
            self.seq_lens[dst_slot] = snap["seq_lens"]
            self.kv_allocated_lens[dst_slot] = snap["kv_allocated_lens"]
            self.verified_ids[dst_slot] = snap["verified_ids"]
            self.finished_mask[dst_slot] = snap["finished_mask"]
            self.token_counts[dst_slot] = snap["token_counts"]
            tc = snap["token_counts"]
            self.all_token_ids[dst_slot][:tc] = list(snap["all_token_ids"])

            # KV block table copy
            src_slot = pidx_to_slot[src_pidx]
            src_pool = self.req_pool_indices[src_slot]
            dst_pool = self.req_pool_indices[dst_slot]
            self.kv_pool.copy_block_table(src_pool, dst_pool, snap["seq_lens"])

        self.group_interval_weights[group_id] = [0.0] * n

    # ────────────── Finalization ──────────────

    def finalize_group(self, group_id: str) -> Tuple[int, List[int]]:
        lw = self.group_log_weights[group_id]
        slots = self.group_slot_lists[group_id]
        best_slot = max(slots, key=lambda s: lw[self.particle_indices[s]])
        best_pidx = self.particle_indices[best_slot]
        count = self.token_counts[best_slot]
        best_output = list(self.all_token_ids[best_slot][:count])
        self.free_group_slots(group_id)
        return best_pidx, best_output

    def group_has_active(self, group_id: str) -> bool:
        return any(
            not self.finished_mask[s] for s in self.group_slot_lists.get(group_id, [])
        )

    # ────────────── ESS / Resample Check ──────────────

    def check_and_resample(
        self, group_id: str, threshold: float = 0.5, method: str = "systematic"
    ) -> bool:
        lw = self.group_interval_weights[group_id]
        slots = self.group_slot_lists[group_id]
        active_pidxs = [
            self.particle_indices[s] for s in slots if not self.finished_mask[s]
        ]
        if len(active_pidxs) <= 1:
            return False

        active_lw = [lw[p] for p in active_pidxs]
        max_lw = max(active_lw)
        unnorm = [math.exp(w - max_lw) for w in active_lw]
        norm_sum = sum(unnorm)
        normalized = [w / norm_sum for w in unnorm]
        ess = 1.0 / sum(w ** 2 for w in normalized)
        n = len(active_pidxs)

        should = ess < n * threshold
        print(f"    ESS check: weights={[f'{w:.3f}' for w in normalized]}, "
              f"ESS={ess:.2f}, threshold={n*threshold:.1f}, resample={should}")

        if not should:
            return False

        # Systematic resample
        cdf = []
        running = 0.0
        for w in normalized:
            running += w
            cdf.append(running)

        import random
        step = 1.0 / n
        start = random.random() * step
        positions = [start + i * step for i in range(n)]
        ancestors = []
        for pos in positions:
            for j, c in enumerate(cdf):
                if pos <= c:
                    ancestors.append(active_pidxs[j])
                    break
            else:
                ancestors.append(active_pidxs[-1])

        print(f"    Ancestors: {ancestors} (maps particle indices)")
        # Convert to full-group ancestor indices
        full_ancestors = list(range(self.group_n_particles[group_id]))
        for i, pidx in enumerate(active_pidxs):
            full_ancestors[pidx] = ancestors[i]
        self.apply_resample(group_id, full_ancestors)
        return True

    # ────────────── Printing ──────────────

    def print_state(self, label: str = ""):
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        print(f"  active_slots: {self.active_slots}")
        print(f"  group_active_indptr: {self.group_active_indptr}")
        print(f"  free_slots: {sorted(self.free_slots)}")

        header = (
            f"  {'slot':>4} {'grp':>4} {'pidx':>4} {'pool':>5} "
            f"{'seqL':>5} {'kvAl':>5} {'vID':>5} {'tCnt':>5} {'fin':>4}  tokens"
        )
        print(header)
        print(f"  {'-' * (len(header) + 10)}")
        for s in range(self.max_slots):
            if self.req_pool_indices[s] == EMPTY:
                continue
            gid = self.slot_to_group_id.get(s, "?")
            tc = self.token_counts[s]
            toks = self.all_token_ids[s][: min(tc, 10)]
            tok_str = str(toks) + ("..." if tc > 10 else "")
            fin = "YES" if self.finished_mask[s] else ""
            print(
                f"  {s:>4} {gid:>4} {self.particle_indices[s]:>4} "
                f"{self.req_pool_indices[s]:>5} {self.seq_lens[s]:>5} "
                f"{self.kv_allocated_lens[s]:>5} {self.verified_ids[s]:>5} "
                f"{self.token_counts[s]:>5} {fin:>4}  {tok_str}"
            )
        for gid in sorted(self.group_log_weights):
            lw = [f"{w:.2f}" for w in self.group_log_weights[gid]]
            iw = [f"{w:.2f}" for w in self.group_interval_weights[gid]]
            print(f"  group {gid}: log_w={lw}  interval_w={iw}")
        print()


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def make_parent(rid, prompt_len, kv_pool, max_new_tokens=20, eos=2):
    req = MockReq(rid=rid, max_new_tokens=max_new_tokens, eos_token_ids=[eos])
    req.kv_committed_len = prompt_len
    req.kv_allocated_len = prompt_len
    pool_idx = kv_pool.alloc_pool_row()
    req.req_pool_idx = pool_idx
    kv_slots = kv_pool.alloc_token_slots(prompt_len)
    kv_pool.assign_slots_to_pool(pool_idx, 0, kv_slots)
    return req


def make_particles(parent, n, temperature=0.7):
    particles = []
    for pidx in range(n):
        req = MockReq(
            rid=f"{parent.rid}_p{pidx}",
            output_ids=list(parent.output_ids),
            kv_committed_len=parent.kv_committed_len,
            kv_allocated_len=parent.kv_allocated_len,
            smc_group_id=parent.rid,
            smc_particle_idx=pidx,
            max_new_tokens=parent.max_new_tokens,
            eos_token_ids=list(parent.eos_token_ids),
        )
        particles.append(req)
    return particles


def section(title):
    print(f"\n{'─'*72}")
    print(f"  >>> {title}")
    print(f"{'─'*72}")


# ──────────────────────────────────────────────────────────────
#  Simulation
# ──────────────────────────────────────────────────────────────


def run_simulation():
    import random
    random.seed(42)

    GAMMA = 2
    G1 = GAMMA + 1  # 3
    PROMPT_LEN = 5
    N_PARTICLES = 3

    state = SMCSlotState(max_slots=8, gamma_plus_1=G1, max_output_len=50)

    state.print_state("Initial: all 8 slots free")

    # ════════════════════════════════════════════════════════
    section("Step 1: BATCHED PREFILL — ReqA and ReqB together")
    # ════════════════════════════════════════════════════════

    parentA = make_parent("A", PROMPT_LEN, state.kv_pool, max_new_tokens=15)
    parentB = make_parent("B", PROMPT_LEN, state.kv_pool, max_new_tokens=30)

    # Simulate batched prefill: both in one forward pass
    # Draft model samples x0 per request
    x0_A, x0_B = 42, 99
    parentA.output_ids.append(x0_A)
    parentB.output_ids.append(x0_B)
    print(f"  Batched prefill: ReqA (x0={x0_A}), ReqB (x0={x0_B})")
    print(f"  Both parent reqs prefilled in ONE forward pass.")

    # ════════════════════════════════════════════════════════
    section("Step 2: MATERIALIZE both groups (sequentially after batched prefill)")
    # ════════════════════════════════════════════════════════

    particles_A = make_particles(parentA, N_PARTICLES)
    slots_A = state.allocate_slots("A", 0, particles_A, PROMPT_LEN, parentA.req_pool_idx)
    print(f"  Group A → slots {slots_A}")

    particles_B = make_particles(parentB, N_PARTICLES)
    slots_B = state.allocate_slots("B", 1, particles_B, PROMPT_LEN, parentB.req_pool_idx)
    print(f"  Group B → slots {slots_B}")

    state.print_state("After materialization: 6 particles in 6 slots")

    # ════════════════════════════════════════════════════════
    section("Step 3: FIRST DECODE CYCLE")
    # ════════════════════════════════════════════════════════

    print("  3a. prepare_for_decode (vectorized KV alloc)")
    draft_input = state.prepare_for_decode()
    ctx = draft_input.decode_ctx
    print(f"      orig_seq_lens = {ctx.orig_seq_lens}")
    print(f"      new_seq_lens  = {ctx.new_seq_lens}  (+{G1})")
    print(f"      verified_ids  = {draft_input.verified_id}")

    print("\n  3b. build_model_worker_batch (sparse → contiguous gather)")
    batch = state.build_model_worker_batch(draft_input)
    print(f"      batch.input_ids       = {batch.input_ids}")
    print(f"      batch.req_pool_indices = {batch.req_pool_indices}")
    print(f"      batch.seq_lens        = {batch.seq_lens}")
    print(f"      batch.batch_size      = {batch.batch_size}")

    print("\n  3c. Worker forward (simulated)")
    result1 = MockBatchResult(
        next_token_ids=[
            [71, 33, 55],  # A0
            [71, 45, 60],  # A1
            [71, 33, 55],  # A2
            [15, 28, 31],  # B0
            [15, 28, 32],  # B1
            [15, 10, 33],  # B2
        ],
        logprob_diff=[-0.3, -0.5, -0.2, -0.1, -0.4, -0.6],
        bonus_ids=[55, 60, 55, 31, 32, 33],
    )

    print("\n  3d. process_batch_result (write-back to slots)")
    out = state.process_batch_result(result1)
    print(f"      newly_finished: {out['newly_finished']}")

    print("\n  3e. Weight update + resample check")
    for gid in sorted(state.group_slot_lists):
        print(f"    Group {gid}:")
        state.check_and_resample(gid)

    state.print_state("After first decode cycle")

    # ════════════════════════════════════════════════════════
    section("Step 4: SECOND DECODE CYCLE — triggers Group B resample")
    # ════════════════════════════════════════════════════════

    draft_input = state.prepare_for_decode()
    batch = state.build_model_worker_batch(draft_input)

    result2 = MockBatchResult(
        next_token_ids=[
            [80, 81, 82],  # A0
            [80, 85, 86],  # A1
            [80, 81, 87],  # A2
            [40, 41, 42],  # B0
            [40, 41, 43],  # B1
            [40, 50, 51],  # B2
        ],
        # B1 and B2 get large negative diffs → triggers resample
        logprob_diff=[-0.3, -0.1, -0.2, -0.05, -4.0, -3.5],
        bonus_ids=[82, 86, 87, 42, 43, 51],
    )

    out = state.process_batch_result(result2)
    print(f"  newly_finished: {out['newly_finished']}")

    print("\n  Resample check:")
    for gid in sorted(state.group_slot_lists):
        print(f"    Group {gid}:")
        did_resample = state.check_and_resample(gid)
        if did_resample:
            print(f"    → Resampled! Slot data copied, interval weights reset.")

    state.print_state("After second decode (Group B resampled)")

    # ════════════════════════════════════════════════════════
    section("Step 5: THIRD DECODE — A2 hits EOS")
    # ════════════════════════════════════════════════════════

    draft_input = state.prepare_for_decode()
    batch = state.build_model_worker_batch(draft_input)
    print(f"  batch.batch_size = {batch.batch_size}")
    print(f"  batch.input_ids  = {batch.input_ids}")

    result3 = MockBatchResult(
        next_token_ids=[
            [90, 91, 92],  # A0
            [90, 95, 96],  # A1
            [90, 2, 0],    # A2 — EOS=2 at position 1!
            [60, 61, 62],  # B0
            [60, 61, 63],  # B1 (was copy of B0, now diverges)
            [60, 61, 64],  # B2 (was copy of B0, now diverges)
        ],
        logprob_diff=[-0.4, -0.3, -0.9, -0.2, -0.3, -0.25],
        bonus_ids=[92, 96, 0, 62, 63, 64],
    )

    out = state.process_batch_result(result3)
    print(f"  newly_finished: {out['newly_finished']} (slot 2 = A2 hit EOS)")

    for gid in sorted(state.group_slot_lists):
        print(f"    Group {gid}:")
        state.check_and_resample(gid)

    state.print_state("After third decode (A2 finished, removed from active)")

    # ════════════════════════════════════════════════════════
    section("Step 6: FOURTH DECODE — A0, A1 approach max_new_tokens=15")
    # ════════════════════════════════════════════════════════

    print(f"  Token counts: {[(s, state.token_counts[s]) for s in state.active_slots]}")

    draft_input = state.prepare_for_decode()
    batch = state.build_model_worker_batch(draft_input)

    result4 = MockBatchResult(
        next_token_ids=[
            [70, 71, 72],  # A0: count 10→13
            [70, 75, 76],  # A1: count 10→13
            [65, 66, 67],  # B0
            [65, 66, 68],  # B1
            [65, 66, 69],  # B2
        ],
        logprob_diff=[-0.1, -0.2, -0.15, -0.1, -0.12],
        bonus_ids=[72, 76, 67, 68, 69],
    )
    out = state.process_batch_result(result4)

    state.print_state("After fourth decode")

    # ════════════════════════════════════════════════════════
    section("Step 7: FIFTH DECODE — A0, A1 exceed max_new_tokens=15")
    # ════════════════════════════════════════════════════════

    print(f"  Token counts: {[(s, state.token_counts[s]) for s in state.active_slots]}")
    print(f"  A0 max_new_tokens={state.max_new_tokens_slot[0]}, "
          f"A1 max_new_tokens={state.max_new_tokens_slot[1]}")

    draft_input = state.prepare_for_decode()
    result5 = MockBatchResult(
        next_token_ids=[
            [50, 51, 52],  # A0: 13→16, exceeds 15
            [50, 55, 56],  # A1: 13→16, exceeds 15
            [75, 76, 77],  # B0
            [75, 76, 78],  # B1
            [75, 76, 79],  # B2
        ],
        logprob_diff=[-0.3, -0.2, -0.1, -0.15, -0.1],
        bonus_ids=[52, 56, 77, 78, 79],
    )
    out = state.process_batch_result(result5)
    print(f"  newly_finished: {out['newly_finished']}")

    state.print_state("After fifth decode (A0, A1 hit max_tokens)")

    # ════════════════════════════════════════════════════════
    section("Step 8: FINALIZE GROUP A — pick best particle")
    # ════════════════════════════════════════════════════════

    a_active = [s for s in state.group_slot_lists.get("A", [])
                if not state.finished_mask[s]]
    print(f"  Group A remaining active: {a_active}")
    assert not a_active, "All A particles should be finished"

    best_pidx, best_output = state.finalize_group("A")
    print(f"  Best particle: {best_pidx} (highest log_weight)")
    print(f"  Best output ({len(best_output)} tokens): {best_output}")

    state.print_state("After Group A finalized (slots 0,1,2 freed)")

    # ════════════════════════════════════════════════════════
    section("Step 9: GROUP B CONTINUES — another decode cycle")
    # ════════════════════════════════════════════════════════

    print(f"  active_slots: {state.active_slots}")
    print(f"  free_slots: {sorted(state.free_slots)}")

    draft_input = state.prepare_for_decode()
    batch = state.build_model_worker_batch(draft_input)
    print(f"  batch.batch_size = {batch.batch_size}")
    print(f"  batch.input_ids  = {batch.input_ids}")

    result6 = MockBatchResult(
        next_token_ids=[
            [85, 86, 87],  # B0
            [85, 86, 88],  # B1
            [85, 86, 89],  # B2
        ],
        logprob_diff=[-0.1, -0.15, -0.1],
        bonus_ids=[87, 88, 89],
    )
    out = state.process_batch_result(result6)

    for gid in sorted(state.group_slot_lists):
        print(f"    Group {gid}:")
        state.check_and_resample(gid)

    state.print_state("Group B continues independently")

    print("=" * 72)
    print("  SIMULATION COMPLETE")
    print("=" * 72)
    print()
    print("  Key takeaways:")
    print("  - Slots are stable: indices never move, only active_slots changes")
    print("  - Resample copies data between rows, slot indices unchanged")
    print("  - Group A finalized and freed slots; Group B unaffected")
    print("  - No sync_from_groups, no filter_batch, no merge_batch")
    print("  - prepare_for_decode is vectorized (no Python loop over reqs)")
    print()


if __name__ == "__main__":
    run_simulation()
