import argparse
from fractions import Fraction
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample_poly
from tqdm import tqdm

from scripts3.source_io import digest, output_directory, publish


ROLES = {"P_train": "train", "P_val": "val"}
COLUMNS = ["role", "split", "group_key", "sampling_rate", "rate_class",
           "shard", "local_idx", "window_len", "valid_length"]
PROBE_POINTS = 512
FOLD_CAL_FRACTION = 0.25
EPS = 1e-12


def rate_spec():
    from scripts3.pretrain_engine import CFG

    lengths = {name: spec["L"] for name, spec in CFG.RATE_SPEC.items()}
    costs = {name: (spec["L"] / math.prod(spec["strides"])) ** 2
             for name, spec in CFG.RATE_SPEC.items()}
    return lengths, costs


def convert(iq, observed, length):
    if not 0 < length <= observed <= iq.shape[-1]:
        raise ValueError("The candidate view requires a valid, non-upsampled interval")
    ratio = Fraction(length, int(observed))
    segment = np.asarray(iq[..., :observed], dtype=np.float32)
    return resample_poly(segment, ratio.numerator, ratio.denominator, axis=-1)


def distortion(iq, observed, length):
    target = np.asarray(iq[:, :observed], dtype=np.float32)
    candidate = convert(target, observed, length)
    ratio = Fraction(int(observed), length)
    restored = resample_poly(candidate, ratio.numerator, ratio.denominator, axis=-1)
    return float(np.sum((restored - target) ** 2) / max(float(np.sum(target ** 2)), EPS))


def spectral_probe(iq, observed):
    width = min(int(observed), PROBE_POINTS)
    starts = np.linspace(0, observed - width, min(4, 1 + observed // width), dtype=int)

    blocks = np.stack([iq[:, start:start + width] for start in starts]).astype(np.float32)
    spectrum = np.abs(np.fft.fft((blocks[:, 0] + 1j * blocks[:, 1]) * np.hanning(width))) ** 2
    total = spectrum.sum(axis=1)
    if not np.isfinite(total).all() or total.sum() <= EPS:
        return None
    return np.abs(np.fft.fftfreq(width)), spectrum[total > EPS] / total[total > EPS, None]


def candidate_features(spectrum, ratio):
    frequencies, power = spectrum
    outside = power[:, frequencies > 0.45 * ratio].sum(axis=1)
    mean_power = power.mean(axis=0)
    flatness = np.exp(np.log(np.maximum(mean_power, EPS)).mean()) * len(mean_power)
    return np.array([1.0, outside.mean(), outside.max(), flatness, ratio])


def fitted_model(x, y, groups, fit_mask, alpha):
    if not fit_mask.any() or fit_mask.all():
        return None
    train = x[fit_mask]
    penalty = np.diag([0.0, 1e-3, 1e-3, 1e-3, 1e-3])
    coefficients = np.linalg.solve(train.T @ train + penalty, train.T @ y[fit_mask])
    cal_x, cal_y, cal_groups = x[~fit_mask], y[~fit_mask], groups[~fit_mask]
    residual = cal_y - np.maximum(0, cal_x @ coefficients)
    maxima = pd.Series(residual).groupby(cal_groups).max().sort_values().to_numpy()
    rank = math.ceil((len(maxima) + 1) * (1 - alpha))
    if rank > len(maxima):
        return None
    return dict(coefficients=coefficients.tolist(), margin=float(max(0, maxima[rank - 1])),
                fit_recordings=len(np.unique(groups[fit_mask])), calibration_recordings=len(maxima))


def choose(iq, observed, reference, policy, available=None):
    if available is None:
        available = policy["lengths"]
    lengths, costs = policy["lengths"], policy["costs"]
    alternatives = [key for key in policy["models"] if key in available
                    and lengths[key] < observed and costs[key] < costs[reference]]
    if not alternatives or (spectrum := spectral_probe(iq, observed)) is None:
        return reference
    for key in sorted(alternatives, key=costs.__getitem__):
        features = candidate_features(spectrum, lengths[key] / observed)
        if 1.0 - features[2] < policy["min_retained"]:
            continue
        model = policy["models"][key]
        upper = max(0, features @ model["coefficients"]) + model["margin"]
        if upper <= policy["max_distortion"]:
            return key
    return reference


def holdout_mask(rates, held_out):
    mask = np.zeros(len(rates), dtype=bool)
    for hz in held_out:
        mask |= np.isclose(rates, hz, rtol=1e-6, atol=1e-3)
    return mask


def load_index(path):
    frame = pd.read_parquet(path, columns=COLUMNS)
    for column in ("role", "split", "group_key", "rate_class", "shard"):
        frame[column] = frame[column].astype("string")
    if frame.empty or frame[COLUMNS].isna().any().any() or frame.group_key.eq("").any():
        raise ValueError("The frozen index needs complete metadata for each window")
    if frame.groupby("group_key").role.nunique().gt(1).any():
        raise ValueError("A recording appears in multiple roles")
    expected = frame.role.map(ROLES).fillna("excl")
    if not frame.split.eq(expected).all():
        raise ValueError("The frozen index contains inconsistent roles and splits")
    if (frame.sampling_rate.le(0).any() or not np.isfinite(frame.sampling_rate).all()
            or frame.valid_length.le(0).any() or frame.valid_length.gt(frame.window_len).any()
            or frame.local_idx.lt(0).any() or frame.duplicated(["shard", "local_idx"]).any()):
        raise ValueError("Invalid acquisition rate, window length, or shard position")
    lengths, _ = rate_spec()
    if not frame.rate_class.map(lengths).eq(frame.window_len).all():
        raise ValueError("Source IQ must use the native window lengths of the current pretraining engine")
    return frame


def shard_windows(frame, root):
    for shard, group in frame.groupby("shard", sort=True):
        with h5py.File(root / shard, "r") as handle:
            array = handle["iq"]
            for row in group.itertuples():
                yield row, array[row.local_idx]


def fit_policy(pool, root, args):
    lengths, costs = rate_spec()
    generator = np.random.default_rng(args.seed)

    records = pool.assign(cost=pool.rate_class.map(costs)).groupby("group_key").cost.max()
    fit_ids, cal_ids = set(), set()
    for _, stratum in records.groupby(records):
        ids = generator.permutation(stratum.index.to_numpy())[:args.records_per_branch]
        n_cal = min(len(ids) - 1, math.ceil(FOLD_CAL_FRACTION * len(ids)))
        if n_cal > 0:
            cal_ids.update(ids[:n_cal])
        fit_ids.update(ids[max(0, n_cal):])
    samples = pool[pool.group_key.isin(fit_ids | cal_ids)]

    samples = samples.sort_values(["shard", "local_idx"]).groupby("group_key", sort=False)
    pieces = [group.iloc[np.linspace(0, len(group) - 1, min(len(group), args.windows_per_record), dtype=int)]
              for _, group in samples]
    if not pieces:
        raise ValueError("No eligible P_train windows for learning the router")
    examples = {key: [] for key in lengths}
    for row, iq in shard_windows(pd.concat(pieces), root):
        spectrum = spectral_probe(iq, row.valid_length)
        if spectrum is None:
            continue
        for key, length in lengths.items():
            if length < row.valid_length and costs[key] < costs[row.rate_class]:
                features = candidate_features(spectrum, length / row.valid_length)
                examples[key].append((features, distortion(iq, row.valid_length, length), row.group_key))
    models = {}
    for key, rows in examples.items():
        if not rows:
            continue
        x, y, groups = zip(*rows)
        groups = np.array(groups)
        model = fitted_model(np.stack(x), np.array(y), groups, np.isin(groups, list(fit_ids)), args.alpha)
        if model is not None:
            models[key] = model
    if not models:
        raise ValueError("No alternative could be calibrated. Check recording counts, eligible windows, and alpha.")
    return dict(lengths=lengths, costs=costs, models=models, min_retained=args.min_retained,
                max_distortion=args.max_distortion, alpha=args.alpha, seed=args.seed,
                fit_recording_ids=sorted(fit_ids), calibration_recording_ids=sorted(cal_ids))


def output_indices(index, frame, root, policy, work):
    keys = pd.MultiIndex.from_frame(frame[["shard", "local_idx"]])
    positions = pd.Series(np.arange(len(frame)), index=keys)
    routes = frame.rate_class.copy()
    candidates = frame.valid_length.gt(min(policy["lengths"].values()))
    source = frame[candidates & frame.rate_class.map(policy["costs"]).gt(min(policy["costs"].values()))]
    for row, iq in tqdm(shard_windows(source, root), total=len(source), desc="Route pretraining windows"):
        routes.at[row.Index] = choose(iq, row.valid_length, row.rate_class, policy)

    schema = pq.ParquetFile(index).schema_arrow.remove_metadata()
    adaptive_schema = schema.append(pa.field("routing_input_samples", pa.int32())).append(
        pa.field("effective_sampling_rate", pa.float64()))
    with pq.ParquetWriter(work / "adaptive.parquet", adaptive_schema, compression="zstd") as adaptive:
        for batch in pq.ParquetFile(index).iter_batches(batch_size=65536):
            block = batch.to_pandas()
            loc = positions.reindex(pd.MultiIndex.from_frame(block[["shard", "local_idx"]])).to_numpy()
            keep = np.isfinite(loc)
            if not keep.any():
                continue
            block = block.loc[keep].copy()
            selected = routes.iloc[loc[keep].astype(int)].to_numpy()
            changed = block.rate_class.ne(selected)
            sizes = pd.Series(selected, index=block.index).map(policy["lengths"])
            block["routing_input_samples"] = block.valid_length.where(changed, 0)
            block["effective_sampling_rate"] = block.sampling_rate.where(
                ~changed, block.sampling_rate * sizes / block.valid_length)
            block["valid_length"] = block.valid_length.where(~changed, sizes)
            block["rate_class"], block["window_len"] = selected, sizes
            if "rate_id" in block:
                from scripts.preprocess_drone_v51 import RATE_ID_MAP
                block["rate_id"] = block.rate_class.map(RATE_ID_MAP)
            adaptive.write_table(pa.Table.from_pandas(block, schema=adaptive_schema, preserve_index=False))


def prepare(args):
    index = args.data / "pretrain_index.parquet"
    root = args.data / "shards"
    with output_directory(args.out) as work:
        metadata = load_index(index)
        held = holdout_mask(metadata.sampling_rate, args.holdout_hz)

        excluded = metadata.group_key.isin(metadata.loc[held, "group_key"])
        pretrain = metadata[metadata.role.isin(ROLES) & ~excluded]
        policy = fit_policy(pretrain[pretrain.role.eq("P_train")], root, args)
        output_indices(index, pretrain, root, policy, work)
        publish(work / "routing.json", dict(
            policy=policy, source_index_sha256=digest(index),
            adaptive_index_sha256=digest(work / "adaptive.parquet")))
    print(f"Prepared the adaptive pretraining index under {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare an adaptive pretraining index")
    parser.add_argument("--data", type=Path, required=True, help="Frozen preprocessing root (index and shards)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the adaptive pretraining index")
    parser.add_argument("--holdout-hz", type=float, nargs="*", default=[])
    parser.add_argument("--min-retained", type=float, default=0.90)
    parser.add_argument("--max-distortion", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--records-per-branch", type=int, default=512)
    parser.add_argument("--windows-per-record", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (not 0 < args.min_retained <= 1 or not 0 < args.alpha < 1
            or not np.isfinite(args.max_distortion) or args.max_distortion < 0
            or min(args.records_per_branch, args.windows_per_record) < 1
            or any(not np.isfinite(hz) or hz <= 0 for hz in args.holdout_hz)):
        parser.error("Invalid budget, calibration level, physical holdout rate, or sample limit")
    prepare(args)


if __name__ == "__main__":
    main()
