"""SMC variant of ModelRunner.

Replaces the standard ``TokenToKVPoolAllocator`` with
``SMCRefCountedTokenAllocator`` so SMC particles can share KV slots via
refcounts.  The swap happens inside ``_init_pools`` immediately after
the standard allocator is constructed, before anything else can cache a
stale reference to it.
"""

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.smc.mem_cache.allocator import SMCRefCountedTokenAllocator


class SMCModelRunner(ModelRunner):
    def _init_pools(self):
        super()._init_pools()
        # Swap standard allocator for SMC refcount-tracking variant when the
        # standard one was constructed.  Skipped for the draft worker (which
        # is passed the target's allocator), and for SWA / paged / NPU
        # paths where a different allocator subclass was already chosen.
        if (
            type(self.token_to_kv_pool_allocator) is TokenToKVPoolAllocator
            and not self.is_draft_worker
        ):
            self.token_to_kv_pool_allocator = SMCRefCountedTokenAllocator(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            )
