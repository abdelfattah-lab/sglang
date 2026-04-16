from types import SimpleNamespace
from unittest import TestCase

from sglang.srt.smc.v1.resampler import SMCResampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _FakeReq:
    def __init__(self, rid: str, particle_idx: int, group_id: str):
        self.rid = rid
        self.smc_particle_idx = particle_idx
        self.smc_group_id = group_id

    def finished(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = True

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def filter_batch(self, keep_indices):
        self.reqs = [self.reqs[i] for i in keep_indices]


class TestSMCResampler(TestCase):
    def test_remove_active_group_members_from_running_batch_dedupes_existing_reqs(self):
        active_req = _FakeReq("group_p0", 0, "group")
        other_req = _FakeReq("group_p1", 1, "group")
        running_batch = _FakeBatch([active_req, other_req, active_req])
        scheduler = SimpleNamespace(running_batch=running_batch)
        resampler = SMCResampler(SimpleNamespace(), "cpu")

        resampler._remove_active_group_members_from_running_batch(
            scheduler,
            [active_req],
        )

        self.assertEqual(scheduler.running_batch.reqs, [other_req])
        self.assertFalse(scheduler.running_batch.batch_is_full)
