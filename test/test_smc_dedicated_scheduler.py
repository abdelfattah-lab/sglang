from collections import deque
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch

from sglang.srt.smc import dedicated_scheduler as dedicated_scheduler_mod
from sglang.srt.smc.v2.req_state import ScheduleBatchSMC
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")
register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")

SMCCoordinator = getattr(dedicated_scheduler_mod, "SMCCoordinator", None)
ScheduleGroupBatch = getattr(dedicated_scheduler_mod, "ScheduleGroupBatch", None)
SMCScheduler = getattr(dedicated_scheduler_mod, "SMCScheduler", None)
SMCCoordinatorV2 = dedicated_scheduler_mod.SMCCoordinatorV2
SMCSchedulerV2 = dedicated_scheduler_mod.SMCSchedulerV2
SequenceGroup = dedicated_scheduler_mod.SequenceGroup
_prepare_req_for_private_prefill = dedicated_scheduler_mod._prepare_req_for_private_prefill


class _FakeReq:
    def __init__(
        self,
        *,
        rid: str,
        particle_idx: int,
        req_pool_idx: int,
        output_ids: list[int],
        kv_indices: list[int],
        finished_reason=None,
        finished_len=None,
    ):
        self.rid = rid
        self.smc_particle_idx = particle_idx
        self.req_pool_idx = req_pool_idx
        self.output_ids = list(output_ids)
        self.kv_committed_len = len(kv_indices)
        self.kv_allocated_len = len(kv_indices)
        self.cache_protected_len = len(kv_indices)
        self.logprob_start_len = 0
        self.finished_reason = finished_reason
        self.finished_len = finished_len
        self.finished_output = False
        self.to_finish = None
        self.prefix_indices = torch.tensor(kv_indices, dtype=torch.int64)
        self.decoded_text = ""
        self.surr_offset = None
        self.read_offset = None
        self.surr_and_decode_ids = None
        self.cur_decode_ids_len = None
        self.smc_group_id = rid.split("_")[0]
        self.origin_input_ids = [1, 2, 3]

    def finished(self):
        return self.finished_reason is not None


class _FakePrefillReq:
    def __init__(self):
        self.origin_input_ids = [1, 2, 3]
        self.output_ids = [4]
        self.prefix_indices = torch.tensor([99, 100], dtype=torch.int64)
        self.last_node = object()
        self.last_host_node = object()
        self.last_host_backup_node = object()
        self.host_hit_length = 7
        self.mamba_branching_seqlen = 8
        self.cache_protected_len = 2
        self.fill_ids = []
        self.extend_input_len = -1

    def init_next_round_input(self, tree_cache=None):
        assert tree_cache is None
        self.fill_ids = self.origin_input_ids + self.output_ids
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)


class _FakeRuntimeReq:
    def __init__(
        self,
        *,
        group_id: str,
        particle_idx: int,
        req_pool_idx: int,
        finish_after: int | None,
        output_ids: list[int] | None = None,
    ):
        self.rid = f"{group_id}_p{particle_idx}"
        self.smc_group_id = group_id
        self.smc_particle_idx = particle_idx
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = [1, 2, 3]
        self.output_ids = list(output_ids or [])
        self.kv_committed_len = max(len(self.output_ids), 1)
        self.kv_allocated_len = self.kv_committed_len
        self.finished_reason = None
        self.finished_len = None
        self.finish_after = finish_after

    def finished(self):
        return self.finished_reason is not None

    def check_finished(self, new_accepted_len: int = 1):
        del new_accepted_len
        if self.finish_after is None:
            return
        if len(self.output_ids) >= self.finish_after:
            self.finished_reason = SimpleNamespace(type="length")
            self.finished_len = self.finish_after


class _FakeBootstrapReq:
    def __init__(
        self,
        *,
        rid: str,
        req_pool_idx: int,
        output_ids: list[int] | None = None,
        kv_indices: list[int] | None = None,
        finish_after: int | None = None,
    ):
        self.rid = rid
        self.req_pool_idx = req_pool_idx
        self.smc_group_id = None
        self.smc_particle_idx = None
        self.origin_input_text = rid
        self.origin_input_ids = [1, 2, 3]
        self.origin_input_ids_unpadded = tuple(self.origin_input_ids)
        self.sampling_params = SimpleNamespace(temperature=0.7, custom_params=None)
        self.output_ids = list(output_ids or [])
        self.kv_committed_len = len(kv_indices or [])
        self.kv_allocated_len = len(kv_indices or [])
        self.cache_protected_len = len(kv_indices or [])
        self.prefix_indices = torch.tensor(kv_indices or [], dtype=torch.int64)
        self.finished_reason = None
        self.finished_len = None
        self.finish_after = finish_after
        self.decoded_text = ""
        self.surr_offset = None
        self.read_offset = None
        self.tokenizer = None
        self.lora_id = None
        self.input_embeds = None
        self.token_type_ids = None
        self.custom_logit_processor = None
        self.require_reasoning = False
        self.eos_token_ids = []
        self.vocab_size = 32000
        self.priority = None
        self.extra_key = None
        self.routing_key = None
        self.dimensions = None
        self.last_node = None
        self.last_host_node = None
        self.last_host_backup_node = None
        self.host_hit_length = 0
        self.mamba_branching_seqlen = None
        self.fill_ids = []
        self.extend_input_len = -1
        self.time_stats = SimpleNamespace(set_completion_time=lambda: None)

    def init_next_round_input(self, tree_cache=None):
        del tree_cache
        self.fill_ids = self.origin_input_ids + self.output_ids
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    def finished(self):
        return self.finished_reason is not None

    def check_finished(self, new_accepted_len: int = 1):
        del new_accepted_len
        if self.finish_after is None:
            return
        if len(self.output_ids) >= self.finish_after:
            self.finished_reason = SimpleNamespace(type="length")
            self.finished_len = self.finish_after


class _FakeReqToTokenPool:
    def __init__(
        self,
        rows: list[list[int]],
        free_rows: list[int] | None = None,
        device: str = "cpu",
    ):
        self.req_to_token = torch.tensor(rows, dtype=torch.int32, device=device)
        self.device = device
        self.free_rows = deque(free_rows or [])

    def write(self, key, value):
        row, cols = key
        self.req_to_token[row, cols] = value

    def alloc(self, reqs):
        if len(self.free_rows) < len(reqs):
            return None
        for req in reqs:
            req.req_pool_idx = self.free_rows.popleft()
        return True

    def copy_block_table(self, src_row, dst_row, shared_seq_len, allocator):
        del allocator
        if shared_seq_len <= 0:
            return
        self.req_to_token[dst_row, :shared_seq_len] = self.req_to_token[
            src_row, :shared_seq_len
        ]

    def free(self, req):
        if req.req_pool_idx is None:
            return
        self.free_rows.append(req.req_pool_idx)
        req.req_pool_idx = None


class _FakeAllocator:
    def __init__(self, size: int = 256, device: str = "cpu"):
        self.inc_calls = []
        self.dec_calls = []
        self.free_calls = []
        self.free_group_depth = 0
        self.page_size = 1
        self.slot_ref_count = torch.zeros(size, dtype=torch.int32, device=device)

    def free_group_begin(self):
        self.free_group_depth += 1

    def free_group_end(self):
        self.free_group_depth -= 1

    def inc_ref(self, indices):
        self.inc_calls.append(indices.clone())

    def dec_ref_and_free(self, indices):
        self.dec_calls.append(indices.clone())

    def free(self, indices):
        self.free_calls.append(indices.clone())
        if indices.numel() > 0:
            self.slot_ref_count[indices] = 0


class _FakeCoordinator:
    def __init__(self):
        self.updates = []
        self.resamples = []

    def on_step_update(self, group, step_inputs):
        self.updates.append(
            (group.group_id, step_inputs.reqs, step_inputs.logprob_diffs.clone())
        )

    def maybe_resample(self, group, scheduler):
        del scheduler
        self.resamples.append(group.group_id)


def _make_runtime_group(
    group_id: str,
    finish_after: list[int | None],
    *,
    pool_idx_base: int = 0,
) -> SequenceGroup:
    reqs = {
        particle_idx: _FakeRuntimeReq(
            group_id=group_id,
            particle_idx=particle_idx,
            req_pool_idx=pool_idx_base + particle_idx,
            finish_after=limit,
        )
        for particle_idx, limit in enumerate(finish_after)
    }
    return SequenceGroup(
        parent_req=SimpleNamespace(rid=group_id),
        n_particles=len(finish_after),
        particle_temperature=0.7,
        particle_reqs=reqs,
        log_weights=torch.zeros(len(finish_after), dtype=torch.float64),
    )


@skipUnless(
    all(obj is not None for obj in (SMCCoordinator, ScheduleGroupBatch, SMCScheduler)),
    "V1 dedicated scheduler removed",
)
class TestDedicatedSMCV1(TestCase):
    def test_private_prefill_resets_prefix_cache_state(self):
        req = _FakePrefillReq()

        _prepare_req_for_private_prefill(req)

        self.assertEqual(req.fill_ids, [1, 2, 3, 4])
        self.assertEqual(req.extend_input_len, 4)
        self.assertEqual(req.cache_protected_len, 0)
        self.assertEqual(req.host_hit_length, 0)
        self.assertIsNone(req.last_node)
        self.assertIsNone(req.last_host_node)
        self.assertIsNone(req.last_host_backup_node)
        self.assertIsNone(req.mamba_branching_seqlen)
        self.assertEqual(req.prefix_indices.numel(), 0)

    def test_resample_copies_source_state_into_destination_slot(self):
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10, 11],
            kv_indices=[1, 2],
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[99],
            kv_indices=[3],
        )
        group = SequenceGroup(
            parent_req=SimpleNamespace(rid="g"),
            n_particles=2,
            particle_temperature=0.7,
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.tensor([0.0, -100.0], dtype=torch.float64),
        )
        scheduler = SimpleNamespace(
            req_to_token_pool=_FakeReqToTokenPool([[1, 2, 0], [3, 0, 0]]),
            token_to_kv_pool_allocator=_FakeAllocator(),
            device="cpu",
        )
        coordinator = SMCCoordinator(
            device="cpu",
            resample_threshold=0.75,
            resample_method="systematic",
        )

        coordinator.maybe_resample(group, scheduler)

        self.assertEqual(req1.output_ids, [10, 11])
        self.assertIsNone(req1.finished_reason)
        self.assertEqual(req1.kv_committed_len, 2)
        self.assertEqual(req1.kv_allocated_len, 2)
        self.assertTrue(
            torch.equal(
                scheduler.req_to_token_pool.req_to_token[1, :2],
                torch.tensor([1, 2], dtype=torch.int32),
            )
        )
        self.assertEqual(group.log_weights.tolist(), [0.0, -100.0])
        self.assertEqual(group.interval_log_weights.tolist(), [0.0, 0.0])
        self.assertEqual(len(scheduler.token_to_kv_pool_allocator.inc_calls), 1)
        self.assertEqual(len(scheduler.token_to_kv_pool_allocator.dec_calls), 1)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.free_group_depth, 0)

    def test_resample_only_uses_active_slots(self):
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[1],
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[99],
            kv_indices=[7],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=1,
        )
        req2 = _FakeReq(
            rid="g_p2",
            particle_idx=2,
            req_pool_idx=2,
            output_ids=[20, 21],
            kv_indices=[3, 4],
        )
        group = SequenceGroup(
            parent_req=SimpleNamespace(rid="g"),
            n_particles=3,
            particle_temperature=0.7,
            particle_reqs={0: req0, 1: req1, 2: req2},
            log_weights=torch.tensor([-100.0, 5.0, 0.0], dtype=torch.float64),
            interval_log_weights=torch.tensor([-100.0, 5.0, 0.0], dtype=torch.float64),
        )
        scheduler = SimpleNamespace(
            req_to_token_pool=_FakeReqToTokenPool([[1, 0, 0], [7, 0, 0], [3, 4, 0]]),
            token_to_kv_pool_allocator=_FakeAllocator(),
            device="cpu",
        )
        coordinator = SMCCoordinator(
            device="cpu",
            resample_threshold=0.75,
            resample_method="systematic",
        )

        coordinator.maybe_resample(group, scheduler)

        self.assertEqual(req0.output_ids, [20, 21])
        self.assertEqual(req1.output_ids, [99])
        self.assertEqual(req2.output_ids, [20, 21])
        self.assertEqual(group.interval_log_weights.tolist(), [0.0, 5.0, 0.0])
        self.assertTrue(
            torch.equal(
                scheduler.req_to_token_pool.req_to_token[0, :2],
                torch.tensor([3, 4], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                scheduler.req_to_token_pool.req_to_token[1, :1],
                torch.tensor([7], dtype=torch.int32),
            )
        )

    def test_pick_best_uses_visible_finished_length_as_tiebreak(self):
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[1, 2, 99, 100],
            kv_indices=[1, 2],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=2,
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[1, 2, 3],
            kv_indices=[4, 5, 6],
            finished_reason=SimpleNamespace(type="length"),
            finished_len=3,
        )
        group = SequenceGroup(
            parent_req=SimpleNamespace(rid="g"),
            n_particles=2,
            particle_temperature=0.7,
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.tensor([0.0, 0.0], dtype=torch.float64),
        )
        coordinator = SMCCoordinator(
            device="cpu",
            resample_threshold=0.5,
            resample_method="systematic",
        )

        best = coordinator.pick_best(group)

        self.assertIs(best, req1)

    def test_admit_prefill_groups_batches_multiple_groups_up_to_capacity(self):
        running_group = _make_runtime_group("running", [None, None], pool_idx_base=0)
        queued_group0 = _make_runtime_group("g0", [None, None], pool_idx_base=10)
        queued_group1 = _make_runtime_group("g1", [None, None], pool_idx_base=20)
        scheduler = SimpleNamespace(
            waiting_groups=deque([queued_group0, queued_group1]),
            running_groups=[running_group],
            max_running_requests=4,
        )
        scheduler._active_particle_count = (
            lambda groups=None: SMCScheduler._active_particle_count(scheduler, groups)
        )
        scheduler._emit_abort = lambda req, error_msg: self.fail(
            f"unexpected abort for {req.rid}: {error_msg}"
        )

        admitted = SMCScheduler._admit_prefill_groups(scheduler)

        self.assertEqual([group.group_id for group in admitted], ["g0"])
        self.assertEqual(
            [group.group_id for group in scheduler.waiting_groups],
            ["g1"],
        )

    def test_admit_prefill_groups_skips_oversized_group_and_keeps_progress(self):
        oversized_group = _make_runtime_group("too-big", [None] * 5, pool_idx_base=0)
        queued_group = _make_runtime_group("g0", [None, None], pool_idx_base=10)
        aborted = []
        scheduler = SimpleNamespace(
            waiting_groups=deque([oversized_group, queued_group]),
            running_groups=[],
            max_running_requests=4,
        )
        scheduler._active_particle_count = (
            lambda groups=None: SMCScheduler._active_particle_count(scheduler, groups)
        )
        scheduler._emit_abort = lambda req, error_msg: aborted.append((req.rid, error_msg))

        admitted = SMCScheduler._admit_prefill_groups(scheduler)

        self.assertEqual([group.group_id for group in admitted], ["g0"])
        self.assertEqual(
            aborted,
            [
                (
                    "too-big",
                    "SMC particle count exceeds max_running_requests for the dedicated scheduler.",
                )
            ],
        )
        self.assertEqual(len(scheduler.waiting_groups), 0)

    def test_materialize_group_from_bootstrap_parent_fans_out_parent_kv(self):
        parent_req = _FakeBootstrapReq(
            rid="g",
            req_pool_idx=0,
            output_ids=[77],
            kv_indices=[10, 11],
        )
        group = SequenceGroup(
            parent_req=parent_req,
            n_particles=2,
            particle_temperature=0.7,
        )
        scheduler = SimpleNamespace(
            device="cpu",
            model_worker=SimpleNamespace(
                materialize_smc_parent_draft_prefix=lambda req: None
            ),
            req_to_token_pool=_FakeReqToTokenPool(
                [[10, 11, 0], [0, 0, 0], [0, 0, 0]],
                free_rows=[1, 2],
            ),
            token_to_kv_pool_allocator=_FakeAllocator(),
            tree_cache=SimpleNamespace(dec_lock_ref=lambda *args, **kwargs: None),
        )

        def _clone_particle(parent_req, particle_idx, temperature, return_logprob):
            del temperature, return_logprob
            return _FakeReq(
                rid=f"{parent_req.rid}_p{particle_idx}",
                particle_idx=particle_idx,
                req_pool_idx=None,
                output_ids=list(parent_req.output_ids),
                kv_indices=[],
            )

        with (
            patch(
                "sglang.srt.smc.v2.scheduler.clone_req_for_smc_particle",
                side_effect=_clone_particle,
            ),
            patch(
                "sglang.srt.smc.v2.scheduler._release_smc_parent_req",
            ) as release_parent,
        ):
            error = SMCScheduler._materialize_group_from_bootstrap_parent(
                scheduler, group
            )

        self.assertIsNone(error)
        self.assertEqual(sorted(group.particle_reqs), [0, 1])
        self.assertEqual(group.log_weights.tolist(), [0.0, 0.0])
        self.assertEqual(group.interval_log_weights.tolist(), [0.0, 0.0])
        self.assertEqual(
            [group.particle_reqs[idx].req_pool_idx for idx in [0, 1]],
            [1, 2],
        )
        for idx in [0, 1]:
            particle_req = group.particle_reqs[idx]
            self.assertEqual(particle_req.output_ids, [77])
            self.assertEqual(particle_req.kv_committed_len, 2)
            self.assertEqual(particle_req.kv_allocated_len, 2)
            self.assertTrue(
                torch.equal(
                    particle_req.prefix_indices,
                    torch.tensor([10, 11], dtype=torch.int64),
                )
            )
        release_parent.assert_called_once()

    def test_process_prefill_group_result_promotes_unfinished_bootstrap_groups_only(self):
        finished_group = SequenceGroup(
            parent_req=_FakeBootstrapReq(rid="g0", req_pool_idx=0, finish_after=1),
            n_particles=2,
            particle_temperature=0.7,
        )
        running_group = SequenceGroup(
            parent_req=_FakeBootstrapReq(rid="g1", req_pool_idx=1, finish_after=None),
            n_particles=2,
            particle_temperature=0.7,
        )
        scheduler = SimpleNamespace(
            prefill_groups=[finished_group, running_group],
            running_groups=[],
        )
        finished = []
        materialized = []
        aborted = []
        scheduler._complete_finished_bootstrap_parent = (
            lambda group: finished.append(group.group_id)
        )
        scheduler._materialize_group_from_bootstrap_parent = (
            lambda group: materialized.append(group.group_id) or None
        )
        scheduler._abort_bootstrap_parent = (
            lambda group, error_msg: aborted.append((group.group_id, error_msg))
        )
        scheduler._sync_running_group_batch = lambda: None
        batch = SimpleNamespace(
            reqs=[
                finished_group.parent_req,
                running_group.parent_req,
            ]
        )
        result = GenerationBatchResult(next_token_ids=torch.tensor([11, 13]))

        SMCScheduler.process_prefill_group_result(scheduler, batch, result)

        self.assertEqual(finished, ["g0"])
        self.assertEqual(materialized, ["g1"])
        self.assertEqual(aborted, [])
        self.assertEqual([group.group_id for group in scheduler.running_groups], ["g1"])
        self.assertEqual(running_group.parent_req.output_ids, [13])

    def test_process_prefill_group_result_aborts_group_on_fanout_error(self):
        group = SequenceGroup(
            parent_req=_FakeBootstrapReq(rid="g0", req_pool_idx=0, finish_after=None),
            n_particles=2,
            particle_temperature=0.7,
        )
        scheduler = SimpleNamespace(
            prefill_groups=[group],
            running_groups=[],
        )
        aborted = []
        scheduler._complete_finished_bootstrap_parent = lambda group: self.fail(
            f"unexpected completion for {group.group_id}"
        )
        scheduler._materialize_group_from_bootstrap_parent = lambda group: "fanout failed"
        scheduler._abort_bootstrap_parent = (
            lambda group, error_msg: aborted.append((group.group_id, error_msg))
        )
        scheduler._sync_running_group_batch = lambda: None
        batch = SimpleNamespace(reqs=[group.parent_req])
        result = GenerationBatchResult(next_token_ids=torch.tensor([11]))

        SMCScheduler.process_prefill_group_result(scheduler, batch, result)

        self.assertEqual(aborted, [("g0", "fanout failed")])
        self.assertEqual(scheduler.running_groups, [])

    def test_process_decode_group_result_slices_by_group_and_finalizes_drained_groups(self):
        finished_group = _make_runtime_group("g0", [1, 1], pool_idx_base=0)
        running_group = _make_runtime_group("g1", [3], pool_idx_base=10)
        coordinator = _FakeCoordinator()
        scheduler = SimpleNamespace(
            running_groups=[finished_group, running_group],
            coordinator=coordinator,
            device="cpu",
        )
        scheduler._resolve_spec_overlap_token_ids = (
            lambda result, batch: [[21], [22], [23]]
        )
        scheduler._group_step_inputs = (
            lambda reqs, diffs: SMCScheduler._group_step_inputs(scheduler, reqs, diffs)
        )
        scheduler._sync_running_group_batch = lambda: None
        finalized = []
        scheduler._finalize_group = lambda group: finalized.append(group.group_id)
        batch = SimpleNamespace(
            reqs=[
                finished_group.particle_reqs[0],
                finished_group.particle_reqs[1],
                running_group.particle_reqs[0],
            ]
        )
        result = GenerationBatchResult(
            logprob_diff=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        )

        SMCScheduler.process_decode_group_result(scheduler, batch, result)

        self.assertEqual(finalized, ["g0"])
        self.assertEqual([group.group_id for group in scheduler.running_groups], ["g1"])
        self.assertEqual(coordinator.resamples, ["g1"])
        self.assertEqual([gid for gid, *_ in coordinator.updates], ["g0", "g1"])
        self.assertTrue(
            torch.equal(
                coordinator.updates[0][2],
                torch.tensor([0.1, 0.2], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                coordinator.updates[1][2],
                torch.tensor([0.3], dtype=torch.float32),
            )
        )

    def test_process_decode_group_result_uses_batch_csr_for_noncontiguous_rows(self):
        group0 = _make_runtime_group("g0", [3, 3], pool_idx_base=0)
        group1 = _make_runtime_group("g1", [3], pool_idx_base=10)
        coordinator = _FakeCoordinator()
        scheduler = SimpleNamespace(
            running_groups=[group0, group1],
            coordinator=coordinator,
            device="cpu",
        )
        scheduler._resolve_spec_overlap_token_ids = (
            lambda result, batch: [[31], [32], [33]]
        )
        scheduler._group_step_inputs = (
            lambda reqs, diffs: SMCScheduler._group_step_inputs(scheduler, reqs, diffs)
        )
        scheduler._sync_running_group_batch = lambda: None
        scheduler._finalize_group = lambda group: self.fail(
            f"unexpected finalization for {group.group_id}"
        )

        batch = ScheduleGroupBatch(
            reqs=[
                group0.particle_reqs[0],
                group1.particle_reqs[0],
                group0.particle_reqs[1],
            ],
            groups=[group0, group1],
            device="cpu",
        )
        batch._rebuild_group_index()
        result = GenerationBatchResult(
            logprob_diff=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        )

        SMCScheduler.process_decode_group_result(scheduler, batch, result)

        self.assertEqual(coordinator.resamples, ["g0", "g1"])
        self.assertEqual([gid for gid, *_ in coordinator.updates], ["g0", "g1"])
        self.assertTrue(
            torch.equal(
                coordinator.updates[0][2],
                torch.tensor([0.1, 0.3], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                coordinator.updates[1][2],
                torch.tensor([0.2], dtype=torch.float32),
            )
        )

    def test_sync_running_group_batch_refreshes_verified_ids_from_req_state(self):
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[1],
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[2],
        )
        group = SequenceGroup(
            parent_req=SimpleNamespace(rid="g"),
            n_particles=2,
            particle_temperature=0.7,
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            interval_log_weights=torch.zeros(2, dtype=torch.float64),
        )
        batch = ScheduleGroupBatch(
            reqs=[req0, req1],
            groups=[group],
            req_to_token_pool=_FakeReqToTokenPool([[1, 0], [2, 0]]),
            device="cpu",
            spec_info=SimpleNamespace(
                verified_id=torch.tensor([999, 888], dtype=torch.int32),
                new_seq_lens=torch.tensor([1, 1], dtype=torch.int64),
                num_tokens_per_req=9,
            ),
        )

        # Simulate in-place resample copying slot 1 state into slot 0.
        req0.output_ids = list(req1.output_ids)
        req0.kv_committed_len = req1.kv_committed_len
        req0.kv_allocated_len = req1.kv_allocated_len

        batch.sync_from_groups([group], num_tokens_per_req=9, vocab_size=32000)

        self.assertTrue(
            torch.equal(
                batch.spec_info.verified_id,
                torch.tensor([20, 20], dtype=torch.int32),
            )
        )

class TestDedicatedSMCV2(TestCase):
    def test_v2_admit_prefill_groups_uses_free_slot_capacity(self):
        queued_group = _make_runtime_group("g0", [None, None], pool_idx_base=10)
        scheduler = SimpleNamespace(
            waiting_groups=deque([queued_group]),
            max_running_requests=4,
            slot_state=SimpleNamespace(available_slot_count=lambda: 0),
        )
        scheduler._emit_abort = lambda req, error_msg: self.fail(
            f"unexpected abort for {req.rid}: {error_msg}"
        )

        admitted = SMCSchedulerV2._admit_prefill_groups(scheduler)

        self.assertEqual(admitted, [])
        self.assertEqual(
            [group.group_id for group in scheduler.waiting_groups],
            ["g0"],
        )

    def test_v2_resample_uses_full_population_and_copies_finished_state(self):
        slot_state = ScheduleBatchSMC(
            max_num_reqs=3,
            device="cpu",
            gamma_plus_1=2,
            vocab_size=32000,
            max_output_len=8,
            req_to_token_pool=_FakeReqToTokenPool(
                [[1, 0, 0], [7, 0, 0], [3, 4, 0]]
            ),
            token_to_kv_pool_allocator=_FakeAllocator(),
            tree_cache=SimpleNamespace(),
            model_config=SimpleNamespace(),
        )
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[1],
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[99],
            kv_indices=[7],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=1,
        )
        req2 = _FakeReq(
            rid="g_p2",
            particle_idx=2,
            req_pool_idx=2,
            output_ids=[20, 21],
            kv_indices=[3, 4],
        )
        slot_state.slot_to_req = {0: req0, 1: req1, 2: req2}
        slot_state.group_slot_lists = {"g": [0, 1, 2]}
        slot_state.group_log_weights = {
            "g": torch.tensor([-100.0, 5.0, 0.0], dtype=torch.float64)
        }
        slot_state.group_interval_weights = {
            "g": torch.tensor([-100.0, 5.0, 0.0], dtype=torch.float64)
        }
        slot_state.req_pool_indices[0] = 0
        slot_state.req_pool_indices[1] = 1
        slot_state.req_pool_indices[2] = 2
        slot_state.seq_lens[0] = 1
        slot_state.seq_lens[1] = 1
        slot_state.seq_lens[2] = 2
        slot_state.kv_allocated_lens[0] = 1
        slot_state.kv_allocated_lens[1] = 1
        slot_state.kv_allocated_lens[2] = 2
        slot_state.token_counts[0] = 1
        slot_state.token_counts[1] = 1
        slot_state.token_counts[2] = 2
        slot_state.particle_indices[0] = 0
        slot_state.particle_indices[1] = 1
        slot_state.particle_indices[2] = 2
        slot_state.finished_mask[1] = True
        slot_state.rebuild_active_slots()

        coordinator = SMCCoordinatorV2(
            device="cpu",
            resample_threshold=0.75,
            resample_method="systematic",
        )

        coordinator.maybe_resample("g", slot_state)

        self.assertEqual(req0.output_ids, [99])
        self.assertEqual(req2.output_ids, [99])
        self.assertEqual(req0.finished_reason.type, "stop")
        self.assertEqual(req2.finished_reason.type, "stop")
        self.assertEqual(req0.finished_len, 1)
        self.assertEqual(req2.finished_len, 1)
        self.assertTrue(slot_state.finished_mask[0].item())
        self.assertTrue(slot_state.finished_mask[1].item())
        self.assertTrue(slot_state.finished_mask[2].item())
        self.assertFalse(slot_state.group_has_active("g"))
        self.assertEqual(slot_state.active_particle_count(), 0)
        self.assertEqual(
            slot_state.group_log_weights["g"].tolist(),
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            slot_state.group_interval_weights["g"].tolist(),
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            int(slot_state.req_to_token_pool.req_to_token[0, 0].item()),
            7,
        )
        self.assertEqual(
            int(slot_state.req_to_token_pool.req_to_token[2, 0].item()),
            7,
        )
        self.assertEqual(len(slot_state.token_to_kv_pool_allocator.inc_calls), 2)
        self.assertEqual(len(slot_state.token_to_kv_pool_allocator.dec_calls), 2)
        self.assertEqual(
            sorted(indices.numel() for indices in slot_state.token_to_kv_pool_allocator.dec_calls),
            [1, 2],
        )

    # Tests for the removed `TensorizedResamplePlan` / `fused_resample` /
    # `fused_collect_resample_jobs` paths were deleted.  See git history
    # for the old versions; `--smc-fast-resample` replaces them with a
    # unified `BatchedResampleResult` hot path validated against the slow
    # path via end-to-end GSM8K runs.

    @skipUnless(False, "Deleted: path removed in --smc-fast-resample refactor")
    def _DELETED_test_v2_fused_resample_batches_tensor_metadata(self):
        device = "cuda"
        allocator = _FakeAllocator(size=64, device=device)
        allocator.slot_ref_count[
            torch.tensor([11, 12, 13, 21, 31, 32, 41, 42], dtype=torch.int64, device=device)
        ] = 1
        slot_state = ScheduleBatchSMC(
            max_num_reqs=4,
            device=device,
            gamma_plus_1=2,
            vocab_size=32000,
            max_output_len=6,
            req_to_token_pool=_FakeReqToTokenPool(
                [
                    [31, 32, 0, 0, 0, 0],
                    [41, 42, 0, 0, 0, 0],
                    [11, 12, 13, 0, 0, 0],
                    [21, 0, 0, 0, 0, 0],
                ],
                device=device,
            ),
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(),
            model_config=SimpleNamespace(),
        )
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[90, 91],
            kv_indices=[31, 32],
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[80, 81],
            kv_indices=[41, 42],
        )
        req2 = _FakeReq(
            rid="g_p2",
            particle_idx=2,
            req_pool_idx=2,
            output_ids=[7],
            kv_indices=[11, 12, 13],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=1,
        )
        req3 = _FakeReq(
            rid="g_p3",
            particle_idx=3,
            req_pool_idx=3,
            output_ids=[20],
            kv_indices=[21],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=1,
        )
        slot_state.slot_to_req = {0: req0, 1: req1, 2: req2, 3: req3}
        slot_state.group_slot_lists = {"g": [0, 1, 2, 3]}
        slot_state.group_log_weights = {
            "g": torch.tensor([3.0, 1.0, 0.0, -1.0], dtype=torch.float64, device=device)
        }
        slot_state.req_pool_indices[:] = torch.tensor(
            [0, 1, 2, 3], dtype=torch.int64, device=device
        )
        slot_state.particle_indices[:] = torch.tensor(
            [0, 1, 2, 3], dtype=torch.int32, device=device
        )
        slot_state.seq_lens[:] = torch.tensor([2, 2, 3, 1], dtype=torch.int64, device=device)
        slot_state.kv_allocated_lens[:] = torch.tensor(
            [2, 2, 3, 1], dtype=torch.int64, device=device
        )
        slot_state.verified_ids[:] = torch.tensor(
            [901, 801, 701, 201], dtype=torch.int32, device=device
        )
        slot_state.token_counts[:] = torch.tensor([2, 2, 3, 1], dtype=torch.int32, device=device)
        slot_state.finished_mask[2:] = True
        slot_state.all_token_ids[0] = torch.tensor(
            [90, 91, 900, 901, 902, 903], dtype=torch.int32, device=device
        )
        slot_state.all_token_ids[1] = torch.tensor(
            [80, 81, 800, 801, 802, 803], dtype=torch.int32, device=device
        )
        slot_state.all_token_ids[2] = torch.tensor(
            [7, 701, 702, 703, 704, 705], dtype=torch.int32, device=device
        )
        slot_state.all_token_ids[3] = torch.tensor(
            [20, 201, 202, 203, 204, 205], dtype=torch.int32, device=device
        )
        slot_state.rebuild_active_slots()

        coordinator = SMCCoordinatorV2(
            device=device,
            resample_threshold=0.75,
            resample_method="systematic",
            fused_resample=True,
        )

        coordinator.dispatch_resample_batch(
            dedicated_scheduler_mod.TensorizedResamplePlan(
                dst_slots=torch.tensor([0, 1], dtype=torch.int64, device=device),
                src_slots=torch.tensor([2, 3], dtype=torch.int64, device=device),
            ),
            slot_state,
        )

        self.assertEqual(slot_state.seq_lens[:2].tolist(), [3, 1])
        self.assertEqual(slot_state.kv_allocated_lens[:2].tolist(), [3, 1])
        self.assertEqual(slot_state.verified_ids[:2].tolist(), [701, 201])
        self.assertEqual(slot_state.token_counts[:2].tolist(), [3, 1])
        self.assertTrue(torch.equal(slot_state.all_token_ids[0], slot_state.all_token_ids[2]))
        self.assertTrue(torch.equal(slot_state.all_token_ids[1], slot_state.all_token_ids[3]))
        self.assertEqual(
            slot_state.all_token_ids[0, : int(slot_state.token_counts[0].item())].tolist(),
            [7, 701, 702],
        )
        self.assertEqual(
            slot_state.all_token_ids[1, : int(slot_state.token_counts[1].item())].tolist(),
            [20],
        )
        self.assertEqual(req0.output_ids, [7])
        self.assertEqual(req1.output_ids, [20])
        self.assertEqual(req0.finished_reason.type, "stop")
        self.assertEqual(req1.finished_reason.type, "stop")
        self.assertEqual(req0.finished_len, 1)
        self.assertEqual(req1.finished_len, 1)
        self.assertFalse(slot_state.group_has_active("g"))
        self.assertEqual(slot_state.active_particle_count(), 0)
        self.assertTrue(
            torch.equal(
                slot_state.req_to_token_pool.req_to_token[0],
                slot_state.req_to_token_pool.req_to_token[2],
            )
        )
        self.assertEqual(
            sorted(torch.cat(allocator.free_calls).tolist()),
            [31, 32, 41, 42],
        )

        freed = []
        slot_state.free_group_slots = lambda group_id: freed.append(group_id)
        parent_req = SimpleNamespace(output_ids=[], finished_reason=None, finished_len=None)
        finalized = slot_state.finalize_group("g", parent_req)

        self.assertIs(finalized, parent_req)
        self.assertEqual(parent_req.output_ids, [7])
        self.assertEqual(parent_req.finished_reason.type, "stop")
        self.assertEqual(parent_req.finished_len, 1)
        self.assertEqual(freed, ["g"])

    def _DELETED_test_v2_process_decode_result_uses_batched_collect_when_enabled(self):
        batched_calls = []
        serial_calls = []
        dispatch_calls = []
        rebuild_calls = []
        drain_calls = []

        coordinator = SimpleNamespace(
            fused_collect_resample_jobs=True,
            collect_resample_jobs_batch=lambda group_ids, slot_state: (
                batched_calls.append((list(group_ids), slot_state))
                or None
            ),
            collect_resample_jobs=lambda group_id, slot_state: (
                serial_calls.append((group_id, slot_state)) or None
            ),
            dispatch_resample_batch=lambda jobs, slot_state, rebuild_active=False: (
                dispatch_calls.append((jobs, slot_state, rebuild_active))
            ),
        )
        slot_state = SimpleNamespace(
            process_batch_result=lambda **kwargs: [],
            group_has_active=lambda group_id: group_id in {"g0", "g1"},
            rebuild_active_slots=lambda: rebuild_calls.append(True),
        )
        scheduler = SimpleNamespace(
            device="cpu",
            slot_state=slot_state,
            running_groups=[
                SimpleNamespace(group_id="g0"),
                SimpleNamespace(group_id="g1"),
                SimpleNamespace(group_id="g2"),
            ],
            coordinator=coordinator,
            _drain_finished_groups=lambda: drain_calls.append(True),
        )
        result = GenerationBatchResult(
            next_token_ids=torch.empty(0, dtype=torch.int32),
            accept_lens=torch.empty(0, dtype=torch.int32),
            next_draft_input=SimpleNamespace(
                verified_id=torch.empty(0, dtype=torch.int32)
            ),
            logprob_diff=torch.empty(0, dtype=torch.float32),
        )

        SMCSchedulerV2._process_decode_result(scheduler, result)

        self.assertEqual(
            batched_calls,
            [(["g0", "g1"], slot_state)],
        )
        self.assertEqual(serial_calls, [])
        self.assertEqual(len(dispatch_calls), 1)
        self.assertFalse(dispatch_calls[0][2])
        self.assertEqual(rebuild_calls, [True])
        self.assertEqual(drain_calls, [True])

    def test_v2_finalize_group_uses_visible_finished_length(self):
        slot_state = ScheduleBatchSMC(
            max_num_reqs=2,
            device="cpu",
            gamma_plus_1=2,
            vocab_size=32000,
            max_output_len=8,
            req_to_token_pool=_FakeReqToTokenPool([[1, 2, 3, 4], [5, 6, 7, 0]]),
            token_to_kv_pool_allocator=_FakeAllocator(),
            tree_cache=SimpleNamespace(),
            model_config=SimpleNamespace(),
        )
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[1, 2],
            kv_indices=[1, 2, 3, 4],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=2,
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[1, 2, 3],
            kv_indices=[5, 6, 7],
            finished_reason=SimpleNamespace(type="length"),
            finished_len=3,
        )
        slot_state.slot_to_req = {0: req0, 1: req1}
        slot_state.group_slot_lists = {"g": [0, 1]}
        slot_state.group_log_weights = {"g": torch.tensor([0.0, 0.0], dtype=torch.float64)}
        slot_state.req_pool_indices[0] = 0
        slot_state.req_pool_indices[1] = 1
        slot_state.kv_allocated_lens[0] = 4
        slot_state.kv_allocated_lens[1] = 3
        slot_state.particle_indices[0] = 0
        slot_state.particle_indices[1] = 1
        slot_state.token_counts[0] = 4
        slot_state.token_counts[1] = 3
        freed = []
        slot_state.free_group_slots = lambda group_id: freed.append(group_id)

        parent_req = SimpleNamespace(output_ids=[], finished_reason=None, finished_len=None)
        finalized = slot_state.finalize_group("g", parent_req)

        self.assertIs(finalized, parent_req)
        self.assertEqual(parent_req.output_ids, [1, 2, 3])
        self.assertEqual(parent_req.finished_len, 3)
        self.assertEqual(parent_req.finished_reason.type, "length")
        self.assertEqual(freed, ["g"])
