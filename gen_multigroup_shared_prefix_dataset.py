import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


SEED = 42
OUTPUT_PATH = Path("multi_group_shared_prefix_dataset.json")

# Pool sizes from prior logs. These are only used for rough printed estimates.
KV_POOL_TOKENS = 263_865
MAMBA_POOL_SLOTS = 151

# Default dataset shape:
# - 12 shared-prefix groups with different prefix lengths
# - 180 prompts per group -> 2160 shared-prefix prompts
# - 600 random noise prompts -> 21.7% noise
# - Total prompts = 2760
DEFAULT_NUM_GROUPS = 12
DEFAULT_PROMPTS_PER_GROUP = 180
DEFAULT_NUM_NOISE_PROMPTS = 600

# Different shared-prefix lengths so policies see a mix of replay costs.
DEFAULT_PREFIX_LEN_WORDS = [
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
]

# Group-specific suffix lengths. Longer suffixes create larger leaf segments and
# more branching variety within each group.
SUFFIX_LEN_WORDS = [48, 64, 80, 96, 112, 128]
NOISE_PROMPT_LEN_WORDS = [96, 128, 160, 224, 320]
COMPLETION_LEN_WORDS = 16


@dataclass(frozen=True)
class GroupConfig:
    group_id: int
    prefix_len_words: int
    prompts_per_group: int


def make_group_prefix(group: GroupConfig) -> str:
    header = (
        f"[Group {group.group_id}] "
        f"You are assisting with benchmark scenario {group.group_id}. "
        f"Follow the shared background context carefully. "
    )
    words = [f"g{group.group_id}_ctx_{j}" for j in range(group.prefix_len_words)]
    return header + " ".join(words) + " User request follows: "


def make_group_suffix(group: GroupConfig, req_id: int) -> str:
    suffix_len = SUFFIX_LEN_WORDS[(group.group_id + req_id) % len(SUFFIX_LEN_WORDS)]
    words = [f"g{group.group_id}_r{req_id}_q_{j}" for j in range(suffix_len)]
    return " ".join(words) + " Please answer precisely."


def make_noise_prompt(prompt_id: int) -> str:
    prompt_len = NOISE_PROMPT_LEN_WORDS[prompt_id % len(NOISE_PROMPT_LEN_WORDS)]
    header = f"[Noise {prompt_id}] This is an unrelated standalone request. "
    words = [f"noise_{prompt_id}_{j}" for j in range(prompt_len)]
    return header + " ".join(words) + " Please answer precisely."


def estimate_token_count(prompt: str) -> int:
    # A simple word-count proxy is enough for rough benchmark-size estimates.
    return len(prompt.split())


def make_linear_spaced_lengths(
    *, num_groups: int, min_words: int, max_words: int
) -> List[int]:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if min_words <= 0 or max_words <= 0:
        raise ValueError("prefix lengths must be positive")
    if min_words > max_words:
        raise ValueError("prefix_min_words must be <= prefix_max_words")
    if num_groups == 1:
        return [min_words]
    return [
        int(round(min_words + i * (max_words - min_words) / (num_groups - 1)))
        for i in range(num_groups)
    ]


def resolve_prefix_lengths(
    *,
    num_groups: int,
    prefix_scale: float,
    prefix_lengths: Optional[str],
    prefix_min_words: Optional[int],
    prefix_max_words: Optional[int],
) -> List[int]:
    if prefix_lengths:
        lengths = [int(x.strip()) for x in prefix_lengths.split(",") if x.strip()]
        if len(lengths) != num_groups:
            raise ValueError(
                f"--prefix-lengths provided {len(lengths)} values, expected {num_groups}"
            )
        return lengths

    if prefix_min_words is not None or prefix_max_words is not None:
        if prefix_min_words is None or prefix_max_words is None:
            raise ValueError(
                "--prefix-min-words and --prefix-max-words must be provided together"
            )
        return make_linear_spaced_lengths(
            num_groups=num_groups,
            min_words=prefix_min_words,
            max_words=prefix_max_words,
        )

    if num_groups == len(DEFAULT_PREFIX_LEN_WORDS):
        return [
            max(1, int(round(x * prefix_scale))) for x in DEFAULT_PREFIX_LEN_WORDS
        ]

    return make_linear_spaced_lengths(
        num_groups=num_groups,
        min_words=max(1, int(round(DEFAULT_PREFIX_LEN_WORDS[0] * prefix_scale))),
        max_words=max(1, int(round(DEFAULT_PREFIX_LEN_WORDS[-1] * prefix_scale))),
    )


def build_dataset(
    *,
    prefix_len_words: List[int],
    prompts_per_group: int,
    num_noise_prompts: int,
) -> List[dict]:
    random.seed(SEED)

    groups = [
        GroupConfig(
            group_id=group_id,
            prefix_len_words=prefix_len_words[group_id],
            prompts_per_group=prompts_per_group,
        )
        for group_id in range(len(prefix_len_words))
    ]

    completion = " ".join(["answer"] * COMPLETION_LEN_WORDS) + "."
    dataset: List[dict] = []

    for group in groups:
        prefix = make_group_prefix(group)
        for req_id in range(group.prompts_per_group):
            dataset.append(
                {
                    "id": f"shared_g{group.group_id}_r{req_id}",
                    "conversations": [
                        {"from": "human", "value": prefix + make_group_suffix(group, req_id)},
                        {"from": "gpt", "value": completion},
                    ],
                    "metadata": {
                        "kind": "shared_prefix",
                        "group_id": group.group_id,
                        "prefix_len_words": group.prefix_len_words,
                    },
                }
            )

    for prompt_id in range(num_noise_prompts):
        dataset.append(
            {
                "id": f"noise_{prompt_id}",
                "conversations": [
                    {"from": "human", "value": make_noise_prompt(prompt_id)},
                    {"from": "gpt", "value": completion},
                ],
                "metadata": {
                    "kind": "noise",
                },
            }
        )

    random.shuffle(dataset)
    return dataset


def write_dataset(dataset: List[dict], output_path: Path, output_format: str) -> None:
    if output_format == "json":
        output_path.write_text(json.dumps(dataset), encoding="utf-8")
        return

    if output_format == "jsonl":
        lines = [json.dumps(item) for item in dataset]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    raise ValueError(f"Unsupported output_format: {output_format}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a multi-group shared-prefix benchmark dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--prefix-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to all shared-prefix lengths.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=DEFAULT_NUM_GROUPS,
        help="Number of shared-prefix groups to generate.",
    )
    parser.add_argument(
        "--prompts-per-group",
        type=int,
        default=DEFAULT_PROMPTS_PER_GROUP,
        help="Number of prompts generated for each shared-prefix group.",
    )
    parser.add_argument(
        "--num-noise-prompts",
        type=int,
        default=DEFAULT_NUM_NOISE_PROMPTS,
        help="Number of unrelated noise prompts to include.",
    )
    parser.add_argument(
        "--prefix-lengths",
        type=str,
        default=None,
        help="Comma-separated list of exact shared-prefix lengths in words. Overrides --prefix-scale.",
    )
    parser.add_argument(
        "--prefix-min-words",
        type=int,
        default=None,
        help="Minimum shared-prefix length in words for linear spacing.",
    )
    parser.add_argument(
        "--prefix-max-words",
        type=int,
        default=None,
        help="Maximum shared-prefix length in words for linear spacing.",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "jsonl"),
        default="json",
        help="Output dataset format. Use jsonl for --dataset-name custom.",
    )
    args = parser.parse_args()

    scaled_prefix_lengths = resolve_prefix_lengths(
        num_groups=args.num_groups,
        prefix_scale=args.prefix_scale,
        prefix_lengths=args.prefix_lengths,
        prefix_min_words=args.prefix_min_words,
        prefix_max_words=args.prefix_max_words,
    )

    dataset = build_dataset(
        prefix_len_words=scaled_prefix_lengths,
        prompts_per_group=args.prompts_per_group,
        num_noise_prompts=args.num_noise_prompts,
    )
    write_dataset(dataset, args.output, args.output_format)

    shared_total = args.num_groups * args.prompts_per_group
    total_prompts = shared_total + args.num_noise_prompts
    noise_ratio = args.num_noise_prompts / total_prompts

    est_prefix_tokens = sum(scaled_prefix_lengths)
    est_shared_suffix_tokens = 0
    for group_id in range(args.num_groups):
        for req_id in range(args.prompts_per_group):
            est_shared_suffix_tokens += SUFFIX_LEN_WORDS[
                (group_id + req_id) % len(SUFFIX_LEN_WORDS)
            ]
    est_noise_tokens = 0
    for prompt_id in range(args.num_noise_prompts):
        est_noise_tokens += NOISE_PROMPT_LEN_WORDS[
            prompt_id % len(NOISE_PROMPT_LEN_WORDS)
        ]
    est_total_tokens = est_prefix_tokens + est_shared_suffix_tokens + est_noise_tokens

    print(
        f"Generated {total_prompts} samples "
        f"({shared_total} shared-prefix + {args.num_noise_prompts} noise) -> {args.output}"
    )
    print(
        f"Groups: {args.num_groups}, prompts/group: {args.prompts_per_group}, "
        f"noise ratio: {noise_ratio:.1%}"
    )
    print(
        "Shared prefix lengths (words): "
        + ", ".join(str(x) for x in scaled_prefix_lengths)
    )
    print("Estimated cached tokens:")
    print(f"  prefix nodes : {est_prefix_tokens:>8,}")
    print(f"  shared leaves: {est_shared_suffix_tokens:>8,}")
    print(f"  noise prompts: {est_noise_tokens:>8,}")
    print(
        f"  total        : {est_total_tokens:>8,}  "
        f"vs KV pool {KV_POOL_TOKENS:,} ({est_total_tokens / KV_POOL_TOKENS:.2f}x)"
    )
    print(
        f"  total nodes  : ~{args.num_groups + shared_total + args.num_noise_prompts:,}  "
        f"vs mamba pool {MAMBA_POOL_SLOTS} "
        f"({(args.num_groups + shared_total + args.num_noise_prompts) / MAMBA_POOL_SLOTS:.1f}x)"
    )

    first_prompt = dataset[0]["conversations"][0]["value"]
    print(f"First prompt estimated words: {estimate_token_count(first_prompt)}")


if __name__ == "__main__":
    main()
