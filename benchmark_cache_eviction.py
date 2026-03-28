#!/usr/bin/env python3
"""Benchmark SGLang cache eviction policies sequentially on a single GPU.

This script runs two benchmarks back to back:
1. Fresh server with radix eviction policy `lru`
2. Fresh server with radix eviction policy `seglen`

For each policy it will:
- start a fresh server
- wait for the server to be ready
- run `python -m sglang.bench_serving`
- save benchmark results to a JSONL file
- save the server log to a log file
- stop the server before moving to the next policy

All outputs are written under ./output relative to the directory where this
script is launched from.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from urllib import error, request


@dataclass(frozen=True)
class PolicyRun:
    name: str
    server_args: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang cache eviction policies lru vs seglen."
    )
    parser.add_argument("--sglang-bin", default="sglang", help="Path to the sglang CLI.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run sglang.bench_serving.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for both the SGLang server and benchmark client.",
    )
    parser.add_argument(
        "--port", type=int, default=30000, help="Port for the SGLang server."
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=900,
        help="Seconds to wait for the server to become ready.",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=int,
        default=60,
        help="Seconds to wait for the server to stop gracefully.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between readiness checks.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="output",
        help="Output directory name created under the launch directory.",
    )

    # Server options
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--mamba-scheduler-strategy", default="extra_buffer"
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.70)
    parser.add_argument("--seglen-eff-weight", type=float, default=0.85)
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="If set, pass --max-running-requests to sglang serve for both policies.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass --trust-remote-code to sglang serve.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Do not pass --trust-remote-code to sglang serve.",
    )

    # Benchmark options
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--dataset-name", default="sharegpt")
    parser.add_argument(
        "--dataset-path",
        default="multi_group_shared_prefix_dataset_6k_10k_compact.json",
    )
    parser.add_argument(
        "--sharegpt-subset-seed",
        type=int,
        default=None,
        help=(
            "If set with --dataset-name sharegpt, create one fixed shuffled subset "
            "and reuse it for both policy runs."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=164)
    parser.add_argument("--sharegpt-output-len", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="If set, pass --max-concurrency to bench_serving for both policies.",
    )
    parser.add_argument(
        "--output-details",
        action="store_true",
        default=True,
        help="Pass --output-details to bench_serving.",
    )
    parser.add_argument(
        "--no-output-details",
        dest="output_details",
        action="store_false",
        help="Do not pass --output-details to bench_serving.",
    )

    return parser.parse_args()


def build_policy_runs(args: argparse.Namespace) -> list[PolicyRun]:
    common_server_args = [
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--mamba-scheduler-strategy",
        args.mamba_scheduler_strategy,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if args.trust_remote_code:
        common_server_args.append("--trust-remote-code")
    if args.max_running_requests is not None:
        common_server_args.extend(
            ["--max-running-requests", str(args.max_running_requests)]
        )

    return [
        PolicyRun(
            name="lru",
            server_args=common_server_args
            + ["--radix-eviction-policy", "lru"],
        ),
        PolicyRun(
            name="seglen",
            server_args=common_server_args
            + [
                "--radix-eviction-policy",
                "seglen",
                "--seglen-eff-weight",
                str(args.seglen_eff_weight),
            ],
        ),
    ]


def build_benchmark_command(
    args: argparse.Namespace, output_file: Path, dataset_path: str
) -> list[str]:
    command = [
        args.python_bin,
        "-m",
        "sglang.bench_serving",
        "--backend",
        args.backend,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ready-check-timeout-sec",
        "0",
        "--dataset-name",
        args.dataset_name,
        "--dataset-path",
        dataset_path,
        "--num-prompts",
        str(args.num_prompts),
        "--sharegpt-output-len",
        str(args.sharegpt_output_len),
        "--request-rate",
        str(args.request_rate),
        "--seed",
        str(args.seed),
        "--output-file",
        str(output_file),
    ]
    if args.output_details:
        command.append("--output-details")
    if args.max_concurrency is not None:
        command.extend(["--max-concurrency", str(args.max_concurrency)])
    return command


def build_sharegpt_subset(
    args: argparse.Namespace, output_dir: Path, run_prefix: str
) -> str:
    if args.dataset_name != "sharegpt" or args.sharegpt_subset_seed is None:
        return args.dataset_path

    if not args.dataset_path:
        raise ValueError(
            "--sharegpt-subset-seed currently requires --dataset-path for sharegpt."
        )

    source_path = Path(args.dataset_path)
    with source_path.open("r", encoding="utf-8") as fp:
        dataset = json.load(fp)

    eligible_examples = [
        item
        for item in dataset
        if len(item.get("conversations", item.get("conversation", []))) >= 2
    ]
    if len(eligible_examples) < args.num_prompts:
        raise ValueError(
            f"Requested {args.num_prompts} prompts, but only "
            f"{len(eligible_examples)} ShareGPT examples have at least two turns."
        )

    rng = random.Random(args.sharegpt_subset_seed)
    rng.shuffle(eligible_examples)
    subset_examples = eligible_examples[: args.num_prompts]

    subset_path = (
        output_dir
        / (
            f"{run_prefix}_sharegpt_subset_seed{args.sharegpt_subset_seed}_"
            f"n{args.num_prompts}.json"
        )
    )
    with subset_path.open("w", encoding="utf-8") as fp:
        json.dump(subset_examples, fp)

    return str(subset_path)


def wait_for_server_ready(
    host: str,
    port: int,
    timeout_sec: int,
    poll_interval_sec: float,
    proc: subprocess.Popen[bytes],
) -> None:
    urls = [
        f"http://{host}:{port}/health_generate",
        f"http://{host}:{port}/health",
        f"http://{host}:{port}/v1/models",
    ]
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            raise RuntimeError(f"Server exited before becoming ready (code {return_code}).")

        for url in urls:
            try:
                with request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (error.URLError, TimeoutError):
                continue

        time.sleep(poll_interval_sec)

    raise TimeoutError(
        f"Server did not become ready within {timeout_sec}s on {host}:{port}."
    )


def terminate_process_tree(
    proc: subprocess.Popen[bytes], shutdown_timeout: int
) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + shutdown_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=10)


def format_command(command: Iterable[str]) -> str:
    return shlex.join(list(command))


def run_policy(
    policy: PolicyRun,
    args: argparse.Namespace,
    output_dir: Path,
    run_prefix: str,
    dataset_path: str,
) -> None:
    server_log = output_dir / f"{run_prefix}_{policy.name}_server.log"
    bench_log = output_dir / f"{run_prefix}_{policy.name}_benchmark.log"
    result_file = output_dir / f"{run_prefix}_{policy.name}_benchmark.jsonl"

    server_command = [args.sglang_bin, "serve", *policy.server_args]
    bench_command = build_benchmark_command(args, result_file, dataset_path)

    print(f"[{policy.name}] Starting server")
    print(f"[{policy.name}] Server command: {format_command(server_command)}")

    with server_log.open("w", encoding="utf-8") as server_log_fp:
        server_log_fp.write(f"Command: {format_command(server_command)}\n\n")
        server_log_fp.flush()
        server_proc = subprocess.Popen(
            server_command,
            stdout=server_log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            wait_for_server_ready(
                host=args.host,
                port=args.port,
                timeout_sec=args.startup_timeout,
                poll_interval_sec=args.poll_interval,
                proc=server_proc,
            )
            print(f"[{policy.name}] Server is ready")

            print(f"[{policy.name}] Running benchmark")
            print(f"[{policy.name}] Benchmark command: {format_command(bench_command)}")

            with bench_log.open("w", encoding="utf-8") as bench_log_fp:
                bench_log_fp.write(f"Command: {format_command(bench_command)}\n\n")
                bench_log_fp.flush()
                completed = subprocess.run(
                    bench_command,
                    stdout=bench_log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Benchmark failed for policy {policy.name} with exit code "
                    f"{completed.returncode}. See {bench_log}."
                )

            print(f"[{policy.name}] Benchmark complete")
            print(f"[{policy.name}] Result file: {result_file}")
            print(f"[{policy.name}] Server log: {server_log}")
            print(f"[{policy.name}] Benchmark log: {bench_log}")
        finally:
            print(f"[{policy.name}] Stopping server")
            terminate_process_tree(server_proc, args.shutdown_timeout)


def main() -> int:
    args = parse_args()

    output_dir = Path.cwd() / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_prefix = time.strftime("%Y%m%d_%H%M%S")
    print(f"Writing outputs under {output_dir}")
    dataset_path = build_sharegpt_subset(args, output_dir, run_prefix)
    if dataset_path != args.dataset_path:
        print(f"Using fixed ShareGPT subset: {dataset_path}")

    for policy in build_policy_runs(args):
        run_policy(policy, args, output_dir, run_prefix, dataset_path)

    print("All benchmarks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
