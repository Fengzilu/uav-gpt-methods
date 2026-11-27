import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocess_drone_v51 import load_small_mat_streams, normalize_iq_batch
from scripts3.adaptive_pretrain_routing import (
    candidate_features,
    distortion,
    holdout_mask,
    load_index,
    rate_spec,
    shard_windows,
    spectral_probe,
)
from scripts3.pretrain_engine import CFG
from scripts3.source_io import publish


BRANCHES = ("low", "mid", "high")
LENGTH_GRID = (512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 12288, 16384, 20480)


def archive_windows(root, reference):
    for path in sorted((root / "kaggle").rglob("*.mat")):
        group = "Kaggle/" + path.parent.relative_to(root / "kaggle").as_posix()
        for samples, _ in load_small_mat_streams(path):
            if len(samples) >= reference["high"]:
                yield "Kaggle", group, normalize_iq_batch(samples[None, :reference["high"]])[0]
    for path in sorted(root.glob("gent_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            samples = archive["raw_bytes_0"].view("<c8")
            if len(samples) >= reference["mid"]:
                yield "Gent", f"Gent/{path.stem}", normalize_iq_batch(samples[None, :reference["mid"]])[0]
    with np.load(root / "radioml_examples.npz", allow_pickle=False) as archive:
        for key in sorted(archive.files):
            values = archive[key]
            if values.shape == (reference["low"], 2):
                samples = values[:, 0] + 1j * values[:, 1]
                yield "RadioML", f"RadioML/{key}", normalize_iq_batch(samples[None, :])[0]


def frozen_windows(root, heldout_hz, max_groups, windows_per_group, seed):
    index = load_index(root / "pretrain_index.parquet")
    held = holdout_mask(index.sampling_rate, heldout_hz)
    excluded = index.group_key.isin(index.loc[held, "group_key"])
    train = index[index.role.eq("P_train") & ~excluded & index.valid_length.eq(index.window_len)]
    rng = np.random.default_rng(seed)
    strata = train.groupby("group_key").window_len.max()
    groups = [rng.permutation(group.index.to_numpy())[:max_groups] for _, group in strata.groupby(strata)]
    if not groups:
        raise ValueError("No full P_train windows after physical-rate exclusion")
    sampled = train[train.group_key.isin(np.concatenate(groups))].sort_values(["shard", "local_idx"])
    pieces = [group.iloc[np.linspace(0, len(group) - 1, min(len(group), windows_per_group), dtype=int)]
              for _, group in sampled.groupby("group_key", sort=False)]
    for row, iq in shard_windows(pd.concat(pieces), root / "shards"):
        yield str(strata.loc[row.group_key]), row.group_key, iq


def measure(windows, lengths, min_retained, max_distortion):
    sources, groups, feasible = [], [], []
    for source, group, iq in windows:
        observed = iq.shape[-1]
        if not np.isfinite(iq).all():
            raise ValueError(f"Non-finite IQ in {group}")
        spectrum = spectral_probe(iq, observed)
        fits = np.zeros(len(lengths), dtype=bool)
        for index, length in enumerate(lengths):
            if length == observed:

                fits[index] = True
            elif length < observed and spectrum is not None:
                outside = candidate_features(spectrum, length / observed)[2]
                if 1.0 - outside >= min_retained:
                    fits[index] = distortion(iq, observed, length) <= max_distortion
        sources.append(source)
        groups.append(group)
        feasible.append(fits)
    if not feasible:
        raise ValueError("No valid windows for branch-length selection")
    return np.array(sources), np.array(groups), np.stack(feasible)


def split_recordings(sources, groups, seed):
    if len(groups) != len(sources) or any(len(set(sources[groups == group])) != 1 for group in np.unique(groups)):
        raise ValueError("A recording group must belong to exactly one source stratum")
    design_groups = set()
    rng = np.random.default_rng(seed)
    for source in np.unique(sources):
        ids = rng.permutation(np.unique(groups[sources == source]))
        if len(ids) < 2:
            raise ValueError(f"Need at least two recording groups in {source}")
        audit_count = max(1, len(ids) // 4)
        design_groups.update(ids[audit_count:])
    return np.isin(groups, sorted(design_groups))


def route(fits, indices):
    eligible = fits[:, indices]
    return np.where(eligible.any(axis=1), eligible.argmax(axis=1), -1)


def search(fits, groups, design, lengths, strides):
    ranked = []
    design_fits = fits[design]
    _, inverse, group_sizes = np.unique(groups[design], return_inverse=True, return_counts=True)
    weights = 1.0 / group_sizes[inverse] / len(group_sizes)
    for indices in combinations(range(len(lengths)), 3):
        selected_lengths = tuple(lengths[index] for index in indices)
        if any(length % stride for length, stride in zip(selected_lengths, strides)):
            continue
        tokens = [length // stride for length, stride in zip(selected_lengths, strides)]
        if not tokens[0] < tokens[1] < tokens[2]:
            continue
        costs = np.square(tokens)
        routes = route(design_fits, indices)
        counts = Counter(routes)
        if -1 in counts or len(counts) != 3:
            continue
        ranked.append((float(weights @ costs[routes]), selected_lengths, indices, costs))
    if not ranked:
        raise ValueError("No three active branches cover all design windows under the supplied budgets")
    ranked.sort(key=lambda item: (round(item[0], 10), item[1]))

    return dict(zip(BRANCHES, ranked[0][1]))


def main():
    parser = argparse.ArgumentParser(description="Select three input lengths from unlabeled IQ")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data", type=Path, help="Frozen preprocessing directory")
    mode.add_argument("--archive", type=Path, help="Local Kaggle, Gent and RadioML evidence subset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTH_GRID)
    parser.add_argument("--holdout-hz", nargs="*", type=float, default=[])
    parser.add_argument("--min-retained", type=float, default=0.90)
    parser.add_argument("--max-distortion", type=float, default=0.10)
    parser.add_argument("--records-per-branch", type=int, default=128)
    parser.add_argument("--windows-per-record", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.out.exists() or min(args.lengths) <= 0 or not 0 < args.min_retained <= 1
            or not np.isfinite(args.max_distortion) or args.max_distortion < 0
            or min(args.records_per_branch, args.windows_per_record) < 1
            or any(not np.isfinite(hz) or hz <= 0 for hz in args.holdout_hz)
            or (args.archive and args.holdout_hz)):
        parser.error("Use a new output path, valid budgets and lengths, and hold out rates only with a frozen index")
    reference, _ = rate_spec()
    strides = tuple(int(np.prod(CFG.RATE_SPEC[branch]["strides"])) for branch in BRANCHES)
    lengths = sorted(set(args.lengths) | set(reference.values()))
    if args.archive:
        windows = archive_windows(args.archive, reference)
    else:
        windows = frozen_windows(args.data, args.holdout_hz, args.records_per_branch,
                                 args.windows_per_record, args.seed)
    sources, groups, fits = measure(windows, lengths, args.min_retained, args.max_distortion)
    design = split_recordings(sources, groups, args.seed)
    result = search(fits, groups, design, lengths, strides)
    publish(args.out, dict(lengths=result, strides=dict(zip(BRANCHES, strides))))
    print(f"Selected input lengths: {result}", flush=True)


if __name__ == "__main__":
    main()
