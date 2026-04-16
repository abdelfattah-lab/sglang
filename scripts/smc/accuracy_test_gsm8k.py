"""GSM8K benchmark for SMC speculative decoding.

Four modes with identical preprocessing for fair comparison:
  - smc_engine: SMC via the dedicated SMCEngine (offline, no tokenizer manager)
  - smc:        engine-level SMC (single Engine, scheduler-integrated)
  - native:     Python-level SMC (two separate Engines, external SMC logic)
  - baseline:   vanilla generation (no speculative decoding)

Usage:
  # SMCEngine (dedicated offline engine, recommended)
  python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine -N 8 -g 32

  # Engine-level SMC
  python scripts/smc/accuracy_test_gsm8k.py --mode smc -N 8 -g 32

  # Native/external SMC (debugger reference)
  python scripts/smc/accuracy_test_gsm8k.py --mode native -N 8 -g 32

  # Baseline (no speculative decoding)
  python scripts/smc/accuracy_test_gsm8k.py --mode baseline

  # Compare engine vs native side-by-side
  python scripts/smc/accuracy_test_gsm8k.py --mode smc -N 8 -g 32 --num-questions 20
  python scripts/smc/accuracy_test_gsm8k.py --mode native -N 8 -g 32 --num-questions 20

  # Custom models
  python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --draft-model meta-llama/Llama-3.2-1B-Instruct \
      -N 8 -g 32
"""

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Shared preprocessing (identical across all modes)
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output or gold answer."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return numbers[-1].replace(",", "") if numbers else None


def format_instruction(question: str) -> str:
    """Build the instruction prompt for a GSM8K question."""
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k(tokenizer, num_questions: int):
    """Load GSM8K and build chat-template prompts + gold labels."""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    prompts = []
    labels = []
    for sample in dataset.select(range(num_questions)):
        instruction = format_instruction(sample["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        labels.append(extract_answer(sample["answer"]))
    assert all(l is not None for l in labels), "Some gold labels could not be parsed"
    return prompts, labels


# ---------------------------------------------------------------------------
# Native SMC decoder (two separate Engines, Python-level SMC logic)
# ---------------------------------------------------------------------------


@dataclass
class NativeSMCConfig:
    """Configuration for native SMC decoding."""

    n_particles: int = 4
    gamma: int = 8
    draft_temperature: float = 0.7
    target_lhts_temperature: float = 1.0
    resample_threshold: float = 0.5
    resample_method: str = "systematic"  # systematic | multinomial

class NativeSMCDecoder:
    """
    SMC decoder using two separate sglang Engine instances.

    This is a reference / debugger implementation.  The algorithm is
    identical to the engine-level SMC but orchestrated in Python so we
    can inspect every step.
    """

    def __init__(
        self,
        draft_model: str,
        target_model: str,
        cfg: NativeSMCConfig,
        draft_mem: float = 0.25,
        target_mem: float = 0.55,
    ):
        from sglang.srt.entrypoints.engine import Engine

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)

        print(f"[native] Loading draft model: {draft_model} (mem={draft_mem})")
        self.draft_engine = Engine(
            model_path=draft_model,
            mem_fraction_static=draft_mem,
            log_level="warning",
        )
        print(f"[native] Loading target model: {target_model} (mem={target_mem})")
        self.target_engine = Engine(
            model_path=target_model,
            mem_fraction_static=target_mem,
            log_level="warning",
        )
        print(
            f"[native] SMC config: N={cfg.n_particles}, γ={cfg.gamma}, "
            f"draft_temp={cfg.draft_temperature}, "
            f"target_lhts_temp={cfg.target_lhts_temperature}"
        )

    # ---- core decode loop ------------------------------------------------

    def decode(
        self, prompt: str, max_tokens: int = 512
    ) -> Tuple[str, Dict]:
        """Run SMC decoding on a single prompt. Returns (text, stats)."""
        cfg = self.cfg

        prompt_ids: List[int] = self.tokenizer.encode(
            prompt, add_special_tokens=False
        )
        prompt_len = len(prompt_ids)

        particle_ids: List[List[int]] = [
            prompt_ids[:] for _ in range(cfg.n_particles)
        ]
        particle_token_lens = [prompt_len] * cfg.n_particles
        log_weights = np.zeros(cfg.n_particles)
        finished = [False] * cfg.n_particles

        stats: Dict = {
            "steps": 0,
            "resample_count": 0,
            "total_draft_tokens": 0,
            "total_target_scores": 0,
            "skipped_target_calls": 0,
            "time_draft_gen": 0.0,
            "time_target_score": 0.0,
        }

        tokens_generated = 0
        start_time = time.time()
        first_step_end = None
        first_step_tokens = 0

        while tokens_generated < max_tokens:
            if all(finished):
                break

            active_idx = [i for i, f in enumerate(finished) if not f]
            active_input_ids = [particle_ids[i] for i in active_idx]

            draft_logprobs = np.zeros(cfg.n_particles)
            token_counts = [0] * cfg.n_particles
            output_ids: List[List[int]] = [[] for _ in range(cfg.n_particles)]

            # --- 1. Draft generation ---
            if active_input_ids:
                t0 = time.perf_counter()
                draft_result = self.draft_engine.generate(
                    input_ids=active_input_ids,
                    sampling_params={
                        "max_new_tokens": cfg.gamma,
                        "temperature": cfg.draft_temperature,
                    },
                    return_logprob=True,
                )
                stats["time_draft_gen"] += time.perf_counter() - t0

                draft_outputs = (
                    draft_result
                    if isinstance(draft_result, list)
                    else [draft_result]
                )

                for idx, ai in enumerate(active_idx):
                    out = draft_outputs[idx]
                    out_ids = list(out.get("output_ids", []) or [])
                    output_ids[ai] = out_ids

                    olps = out.get("meta_info", {}).get(
                        "output_token_logprobs", []
                    )
                    draft_logprobs[ai] = _sum_logprobs(olps)
                    token_counts[ai] = len(out_ids)

                    # EOS detection
                    meta = out.get("meta_info", {})
                    fr = meta.get("finish_reason", {})
                    is_stop = fr == "stop" or (
                        isinstance(fr, dict) and fr.get("type") == "stop"
                    )
                    eos_id = self.tokenizer.eos_token_id
                    hit_eos = eos_id is not None and eos_id in out_ids
                    if is_stop or hit_eos or len(out_ids) < cfg.gamma:
                        finished[ai] = True

            stats["total_draft_tokens"] += sum(token_counts)

            # --- 2. Target scoring ---
            target_logprobs = np.zeros(cfg.n_particles)

            # Skip target if all active particles are truly identical
            skip_target = False
            if active_idx:
                prefixes = [tuple(particle_ids[i]) for i in active_idx]
                extensions = [tuple(output_ids[i]) for i in active_idx]
                if (
                    len(set(prefixes)) == 1
                    and len(set(extensions)) == 1
                    and len(extensions[0]) > 0
                ):
                    skip_target = True
                    stats["skipped_target_calls"] += 1
                    for i in active_idx:
                        target_logprobs[i] = draft_logprobs[i]

            # Compute logprob_start_lens before extending
            if active_idx and not skip_target:
                logprob_start_lens = [
                    max(0, particle_token_lens[i] - 1) for i in active_idx
                ]

            # Extend particles in-place
            for i in range(cfg.n_particles):
                particle_ids[i].extend(output_ids[i])
            particle_token_lens = [len(ids) for ids in particle_ids]

            if active_idx and not skip_target:
                active_new_ids = [particle_ids[i] for i in active_idx]
                active_counts = [token_counts[i] for i in active_idx]

                t0 = time.perf_counter()
                target_result = self.target_engine.generate(
                    input_ids=active_new_ids,
                    sampling_params={"max_new_tokens": 0},
                    return_logprob=True,
                    logprob_start_len=logprob_start_lens,
                )
                stats["time_target_score"] += time.perf_counter() - t0

                target_outputs = (
                    target_result
                    if isinstance(target_result, list)
                    else [target_result]
                )

                for j, i in enumerate(active_idx):
                    out = target_outputs[j]
                    input_lps = out.get("meta_info", {}).get(
                        "input_token_logprobs", []
                    )
                    nc = active_counts[j]
                    if nc > 0 and len(input_lps) >= nc:
                        cont_lps = input_lps[-nc:]
                        target_logprobs[i] = _sum_logprobs(cont_lps)
                        target_logprobs[i] /= cfg.target_lhts_temperature

                stats["total_target_scores"] += len(active_idx)

            # --- 3. Importance weight update ---
            log_importance = target_logprobs - draft_logprobs
            for i in active_idx:
                log_weights[i] += log_importance[i]

            # --- 4. Resample if ESS too low ---
            weights = _normalize_weights(log_weights)
            ess = _effective_sample_size(weights)

            if ess < cfg.n_particles * cfg.resample_threshold:
                old_finished = finished.copy()
                old_lens = particle_token_lens[:]
                particle_ids, log_weights, indices = _resample(
                    particle_ids, weights, cfg.resample_method
                )
                finished = [old_finished[idx] for idx in indices]
                particle_token_lens = [old_lens[idx] for idx in indices]
                stats["resample_count"] += 1

            tokens_generated += cfg.gamma
            stats["steps"] += 1
            if first_step_end is None:
                first_step_end = time.time()
                first_step_tokens = cfg.gamma

        elapsed = time.time() - start_time
        stats["elapsed_time"] = elapsed
        stats["ttft"] = (
            (first_step_end - start_time)
            if first_step_end
            else elapsed
        )
        decode_time = elapsed - stats["ttft"]
        decode_tokens = tokens_generated - first_step_tokens
        stats["decode_tps"] = (
            decode_tokens / decode_time if decode_time > 0 else 0.0
        )

        # Select best particle
        best_idx = int(np.argmax(log_weights))
        generated_ids = particle_ids[best_idx][prompt_len:]
        stats["output_token_count"] = len(generated_ids)
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text, stats

    def shutdown(self):
        if hasattr(self, "draft_engine"):
            self.draft_engine.shutdown()
        if hasattr(self, "target_engine"):
            self.target_engine.shutdown()


# ---- SMC helpers ---------------------------------------------------------


def _sum_logprobs(logprobs_list) -> float:
    if not logprobs_list:
        return 0.0
    total = 0.0
    for item in logprobs_list:
        if item is None:
            continue
        if isinstance(item, (int, float)):
            total += float(item)
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            if item[0] is not None:
                total += float(item[0])
    return total


def _normalize_weights(log_w: np.ndarray) -> np.ndarray:
    w = np.exp(log_w - np.max(log_w))
    return w / w.sum()


def _effective_sample_size(weights: np.ndarray) -> float:
    return 1.0 / np.sum(weights**2)


def _resample(
    particles: List[List[int]],
    weights: np.ndarray,
    method: str,
) -> Tuple[List[List[int]], np.ndarray, List[int]]:
    n = len(particles)
    if method == "multinomial":
        indices = np.random.choice(n, size=n, replace=True, p=weights)
    else:  # systematic
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(weights)
        indices = np.clip(np.searchsorted(cumsum, positions), 0, n - 1)
    new_particles = [list(particles[i]) for i in indices]
    return new_particles, np.zeros(n), list(indices)


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


def run_smc_engine_eval(args, prompts, labels):
    """Evaluation using the dedicated SMCEngine (offline, no tokenizer manager)."""
    from sglang.srt.smc.engine import SMCEngine

    draft_model = args.draft_model or DEFAULT_DRAFT_MODEL
    engine_kwargs = dict(
        model_path=args.model,
        draft_model_path=draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.temperature,
        target_temperature=args.temperature,
        trust_remote_code=True,
        page_size=1,
        attention_backend=args.attention_backend,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.resample_threshold is not None:
        engine_kwargs["resample_threshold"] = args.resample_threshold
    if args.smc_fast_resample:
        engine_kwargs["smc_fast_resample"] = True
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    else:
        engine_kwargs["max_running_requests"] = max(args.particles + 4, 16)
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    with SMCEngine(**engine_kwargs) as engine:
        preds = []
        total_output_tokens = 0
        stage_timing = None
        if args.report_stage_timing:
            engine.reset_stage_timing_summary()
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["completion_tokens"]
                    print(f"--- Q{qi} ({ntok} tokens) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_answer(output["text"]))
                total_output_tokens += output["completion_tokens"]
            elapsed = time.perf_counter() - tic
            correct = sum(
                p == l for p, l in zip(preds, labels[: len(preds)])
            )
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic
        if args.report_stage_timing:
            with tempfile.NamedTemporaryFile(
                prefix="smc-stage-timing-",
                suffix=".json",
                delete=False,
            ) as fout:
                stage_timing_path = fout.name
            try:
                engine.dump_stage_timing_summary(stage_timing_path)
                with open(stage_timing_path, "r", encoding="utf-8") as fin:
                    stage_timing = json.load(fin)
            finally:
                if os.path.exists(stage_timing_path):
                    os.unlink(stage_timing_path)

    return preds, total_output_tokens, latency, stage_timing


def run_engine_eval(args, prompts, labels):
    """Engine-level SMC or baseline evaluation."""
    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
        #log_level="info",
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.mode == "smc":
        engine_kwargs["speculative_algorithm"] = "SMC"
        engine_kwargs["speculative_draft_model_path"] = (
            args.draft_model or DEFAULT_DRAFT_MODEL
        )
        engine_kwargs["smc_n_particles"] = args.particles
        engine_kwargs["smc_gamma"] = args.gamma
        engine_kwargs["smc_draft_temperature"] = args.temperature
        engine_kwargs["smc_target_temperature"] = args.temperature
        engine_kwargs["page_size"] = 1
        engine_kwargs["attention_backend"] = args.attention_backend
        if args.resample_threshold is not None:
            engine_kwargs["smc_resample_threshold"] = args.resample_threshold
        if args.smc_fast_resample:
            engine_kwargs["smc_fast_resample"] = True
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    elif args.mode == "smc":
        # Keep some headroom above particle count for SMC scheduling.
        engine_kwargs["max_running_requests"] = max(args.particles + 4, 16)

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    with sgl.Engine(**engine_kwargs) as engine:
        preds = []
        total_output_tokens = 0
        stage_timing = None
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["meta_info"]["completion_tokens"]
                    print(f"--- Q{qi} ({ntok} tokens) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_answer(output["text"]))
                total_output_tokens += output["meta_info"][
                    "completion_tokens"
                ]
            elapsed = time.perf_counter() - tic
            correct = sum(
                p == l for p, l in zip(preds, labels[: len(preds)])
            )
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic
        if args.mode == "smc" and args.report_stage_timing:
            server_info = engine.get_server_info()
            internal_states = server_info.get("internal_states") or [{}]
            stage_timing = internal_states[0].get("forward_stage_timing_summary")

    return preds, total_output_tokens, latency, stage_timing


def run_native_eval(args, prompts, labels):
    """Native (Python-level) SMC evaluation."""
    cfg = NativeSMCConfig(
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.temperature,
        target_lhts_temperature=1.0,
        resample_threshold=0.5,
        resample_method="systematic",
    )
    draft_model = args.draft_model or DEFAULT_DRAFT_MODEL
    decoder = NativeSMCDecoder(
        draft_model=draft_model,
        target_model=args.model,
        cfg=cfg,
        draft_mem=args.draft_mem,
        target_mem=args.target_mem,
    )

    try:
        preds = []
        total_output_tokens = 0
        tic = time.perf_counter()
        for qi, prompt in enumerate(prompts):
            text, stats = decoder.decode(prompt, max_tokens=args.max_new_tokens)
            if qi < 3:
                ntok = stats.get("output_token_count", 0)
                print(f"--- Q{qi} ({ntok} tokens) ---")
                print(text[:400])
                print()
            preds.append(extract_answer(text))
            total_output_tokens += stats.get("output_token_count", 0)
            elapsed = time.perf_counter() - tic
            correct = sum(
                p == l for p, l in zip(preds, labels[: len(preds)])
            )
            decode_tps = stats.get("decode_tps", 0.0)
            print(
                f"[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"decode_tps={decode_tps:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic
    finally:
        decoder.shutdown()

    return preds, total_output_tokens, latency, None


def print_stage_timing_summary(stage_timing, wall_time, total_tokens):
    prefill = stage_timing["prefill"]
    decode = stage_timing["decode"]
    accounted_sec = stage_timing["total_accounted_sec"]
    remaining_sec = max(wall_time - accounted_sec, 0.0)
    decode_share = (
        decode["total_sec"] / accounted_sec if accounted_sec > 0 else 0.0
    )
    decode_run_tps = (
        total_tokens / decode["run_batch_sec"] if decode["run_batch_sec"] > 0 else 0.0
    )

    print("  Stage timing (scheduler-attributed):")
    print(
        "    Prefill:"
        f" total={prefill['total_sec']:.2f}s"
        f" prepare={prefill['prepare_sec']:.2f}s"
        f" run={prefill['run_batch_sec']:.2f}s"
        f" process={prefill['process_result_sec']:.2f}s"
        f" batches={prefill['batches']}"
        f" avg_rows={prefill['avg_rows_per_batch']:.1f}"
    )
    print(
        "    Decode:"
        f" total={decode['total_sec']:.2f}s"
        f" prepare={decode['prepare_sec']:.2f}s"
        f" run={decode['run_batch_sec']:.2f}s"
        f" process={decode['process_result_sec']:.2f}s"
        f" batches={decode['batches']}"
        f" avg_rows={decode['avg_rows_per_batch']:.1f}"
    )
    print(
        "    Other scheduler:"
        f" empty_schedule={stage_timing['empty_schedule_sec']:.2f}s"
        f" idle={stage_timing['idle_sec']:.2f}s"
    )
    print(
        "    Accounted scheduler time:"
        f" {accounted_sec:.2f}s ({100 * accounted_sec / wall_time:.1f}% of wall)"
    )
    print(
        "    Remaining wall time outside scheduler buckets:"
        f" {remaining_sec:.2f}s ({100 * remaining_sec / wall_time:.1f}% of wall)"
    )
    print(f"    Decode share of accounted scheduler time: {100 * decode_share:.1f}%")
    print(f"    Completion tokens / decode run time: {decode_run_tps:.1f} tok/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    mode_label = {
        "smc_engine": "SMCEngine (dedicated offline)",
        "smc": "Engine-level SMC",
        "native": "Native (Python) SMC",
        "baseline": "Baseline (vanilla)",
    }
    print(f"Mode: {mode_label[args.mode]} | Model: {args.model}")
    if args.mode in ("smc_engine", "smc", "native"):
        draft = args.draft_model or DEFAULT_DRAFT_MODEL
        print(
            f"  particles={args.particles}, gamma={args.gamma}, "
            f"temperature={args.temperature}, draft={draft}"
        )
    print(
        f"  num_questions={args.num_questions}, max_new_tokens={args.max_new_tokens}, "
        f"ignore_eos={args.ignore_eos}"
    )
    print()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load tokenizer and data (shared across all modes)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, labels = load_gsm8k(tokenizer, args.num_questions)

    # Run evaluation
    if args.mode == "smc_engine":
        preds, total_tokens, latency, stage_timing = run_smc_engine_eval(
            args, prompts, labels
        )
    elif args.mode == "native":
        preds, total_tokens, latency, stage_timing = run_native_eval(
            args, prompts, labels
        )
    else:
        preds, total_tokens, latency, stage_timing = run_engine_eval(
            args, prompts, labels
        )

    # Report
    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p is None for p in preds)
    n = len(preds)

    print(f"\n{'=' * 55}")
    print(f"  {mode_label[args.mode]}")
    if args.mode in ("smc_engine", "smc", "native"):
        print(f"  N={args.particles}, γ={args.gamma}, temp={args.temperature}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:          {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"  Invalid:           {invalid}/{n} ({100 * invalid / n:.1f}%)")
    print(f"  Output throughput: {total_tokens / latency:.1f} tok/s")
    print(f"  Total tokens:      {total_tokens}")
    print(f"  Wall time:         {latency:.1f}s")
    if stage_timing is not None:
        print_stage_timing_summary(stage_timing, latency, total_tokens)
    print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument(
        "--mode",
        choices=["baseline", "smc", "smc_engine", "native"],
        default="smc_engine",
        help="baseline = vanilla, smc = engine-level, smc_engine = dedicated SMCEngine, native = Python-level (default: smc_engine)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"target model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
        help=f"draft model path (default: {DEFAULT_DRAFT_MODEL})",
    )

    # SMC parameters (used by both smc and native modes)
    smc_grp = parser.add_argument_group("SMC parameters")
    smc_grp.add_argument("--particles", "-N", type=int, default=4)
    smc_grp.add_argument("--gamma", "-g", type=int, default=4)
    smc_grp.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="draft temperature (default: 0.7)",
    )
    smc_grp.add_argument(
        "--seed", type=int, default=None, help="numpy seed for reproducibility"
    )
    smc_grp.add_argument(
        "--resample-threshold", type=float, default=None,
        help="ESS resample threshold (default: 0.5, use 0 to disable resampling)",
    )
    smc_grp.add_argument(
        "--smc-fast-resample",
        action="store_true",
        default=False,
        help=(
            "Use the fused SMC resample path (single Triton kernel for "
            "normalize + ESS + systematic resample + dst/src compaction, "
            "feeding the fused KV-copy kernel directly). "
            "Requires systematic resampling + CUDA.  "
            "Default (off) is the slow per-group Python path (golden truth)."
        ),
    )
    # Benchmark
    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--num-questions", type=int, default=80)
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pass ignore_eos through engine sampling_params for throughput comparisons",
    )
    bench.add_argument(
        "--report-stage-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print scheduler timing split by prefill vs decode for smc_engine or smc",
    )

    # Engine overrides (smc / baseline modes)
    eng = parser.add_argument_group("engine overrides (smc/baseline)")
    eng.add_argument("--attention-backend", type=str, default="triton",
                      choices=["triton", "fa3"],
                      help="attention backend for SMC mode (default: triton)")
    eng.add_argument("--mem-fraction-static", type=float, default=0.4)
    eng.add_argument("--cuda-graph-max-bs", type=int, default=128)
    eng.add_argument("--max-running-requests", type=int, default=128)

    # Native mode memory splits
    nat = parser.add_argument_group("native mode memory")
    nat.add_argument(
        "--draft-mem",
        type=float,
        default=0.25,
        help="GPU mem fraction for draft engine (default: 0.25)",
    )
    nat.add_argument(
        "--target-mem",
        type=float,
        default=0.55,
        help="GPU mem fraction for target engine (default: 0.55)",
    )

    args = parser.parse_args()
    main(args)
