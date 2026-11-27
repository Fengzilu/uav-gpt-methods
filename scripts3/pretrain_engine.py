from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.checkpoint import checkpoint as _ckpt
from tqdm import tqdm


MODEL_SIZES = {
    "small":  dict(D_MODEL=512,  N_LAYERS=12, N_HEADS=8,  DEC_DIM=256, DEC_LAYERS=2),
    "base":   dict(D_MODEL=768,  N_LAYERS=12, N_HEADS=12, DEC_DIM=384, DEC_LAYERS=2),
    "large":  dict(D_MODEL=1024, N_LAYERS=16, N_HEADS=16, DEC_DIM=512, DEC_LAYERS=3),
    "xlarge": dict(D_MODEL=1280, N_LAYERS=20, N_HEADS=16, DEC_DIM=640, DEC_LAYERS=3),
}


class CFG:
    DATA_ROOT = "datasets/preprocess_v51"
    INDEX_NAME = "pretrain_index.parquet"
    INDEX_SHA256 = ""
    SHARD_SUBDIR = "shards"
    PAIR_INPUT_LENGTHS = False

    DROP_EXCL = True
    INACTIVE_KEEP_RATIO = 0.4

    RATE_SPEC = {
        "high": dict(L=20480, strides=(4, 8, 8)),
        "mid":  dict(L=2048,  strides=(2, 8, 4)),
        "low":  dict(L=1024,  strides=(4, 4, 4)),
    }

    SIZE = "large"
    D_MODEL = 1024
    N_LAYERS = 16
    N_HEADS = 16
    FFN_RATIO = 4
    DROPOUT = 0.10
    DROP_PATH = 0.10
    GRAD_CKPT = False

    USE_DECODER = True
    DEC_DIM = 512
    DEC_LAYERS = 3
    DEC_HEADS = 8

    SPEC_BINS = 64
    PROJ_DIM = 128
    CONTRAST_T = 0.07
    CONTRAST_PER_RATE = True

    BATCH = {"high": 96, "mid": 16, "low": 24}
    VIEWS_PER_GROUP = {"high": 4, "mid": 4, "low": 2}
    NEIGHBOR_SPAN = 24

    GRAD_ACCUM = 2
    NUM_WORKERS = 5
    NUM_WORKERS_PER_RATE = {"high": 7, "mid": 2, "low": 3}
    VAL_NUM_WORKERS = 2
    PREFETCH = 4
    PIN_MEMORY = True
    SHARD_CACHE_CAP = 32

    MAX_STEPS = 400_000
    WARMUP_STEPS = 8_000
    LR = 2.0e-4
    MIN_LR = 1.5e-5
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.95)
    GRAD_CLIP = 1.0
    AMP_DTYPE = "bf16"
    EMA_DECAY = 0.999
    COMPILE = False

    MASK_RATIO = 0.35
    MASK_MODE_PROB = {"random": 0.40, "block": 0.35, "tail": 0.25}
    TAIL_RATIO = 0.25
    NORM_PIX_LOSS = True
    MASK_INPUT = True

    USE_DATASET_ID = False
    DOMAIN_DROPOUT = 0.25
    DOMAIN_EMB_ON_CLS = False

    W_PRESENCE = 0.15
    W_TECH = 0.15
    PRESENCE_POS_WEIGHT = True

    ENERGY_SD_FLOOR = 0.10

    AUG_PROB = 0.8
    AUG_PHASE = True
    AUG_FREQ_OFF = True
    AUG_FREQ_MAX = 0.02
    AUG_AMP_SCALE = (0.7, 1.4)
    AUG_NOISE_SNR_DB = (8.0, 30.0)
    AUG_TIME_SHIFT = True
    AUG_TIME_SHIFT_MAX = 0.10

    PHASES = [
        (0.00, ("high",),              dict(recon=1.0, pred=0.5,  spec=0.5, energy=0.5, contrast=0.10, domain=0.05)),
        (0.10, ("high", "mid"),        dict(recon=0.8, pred=0.4,  spec=0.8, energy=0.5, contrast=0.35, domain=0.10)),
        (0.25, ("high", "mid", "low"), dict(recon=0.6, pred=0.3,  spec=1.0, energy=0.5, contrast=0.50, domain=0.12)),
        (0.60, ("high", "mid", "low"), dict(recon=0.5, pred=0.25, spec=1.0, energy=0.4, contrast=0.40, domain=0.10)),
        (0.85, ("high", "mid", "low"), dict(recon=0.4, pred=0.2,  spec=0.8, energy=0.3, contrast=0.25, domain=0.06)),
    ]

    OUT_DIR = "runs/uavgpt_large"
    LOG_EVERY = 50
    VAL_EVERY = 2_500
    VAL_STEPS = 50
    SAVE_EVERY = 5_000
    KEEP_LAST_N = 3
    USE_TENSORBOARD = True
    BEST_KEYS = ("spec/high", "energy/high")

    SEED = 42


def apply_size(name: str):
    cfg = MODEL_SIZES[name]
    CFG.SIZE = name
    for k, v in cfg.items():
        setattr(CFG, k, v)

    CFG.DEC_HEADS = max(4, CFG.DEC_DIM // 64)

    if name in ("large", "xlarge"):
        CFG.WARMUP_STEPS = max(CFG.WARMUP_STEPS, 8_000)
        CFG.LR = min(CFG.LR, 2.0e-4)

    if name == "xlarge":
        CFG.BATCH = {"high": 64, "mid": 12, "low": 16}
        CFG.GRAD_ACCUM = 3
        CFG.WARMUP_STEPS = max(CFG.WARMUP_STEPS, 10_000)
        CFG.LR = min(CFG.LR, 1.5e-4)


N_DATASET, N_BAND, N_SCENE, N_DEVICE, N_RATE = 4, 10, 5, 4, 3
RATE_TO_ID = {"high": 0, "mid": 1, "low": 2}

TECH_CLASSES = [
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK",
    "BPSK", "8PSK", "AM-SSB-SC", "4ASK", "16PSK", "64APSK", "128QAM",
    "128APSK", "AM-DSB-SC", "AM-SSB-WC", "64QAM", "QPSK", "256QAM",
    "AM-DSB-WC", "OOK", "16QAM", "DVB-T", "LTE", "WiFi",
]
TECH_TO_ID = {n: i for i, n in enumerate(TECH_CLASSES)}
N_TECH = len(TECH_CLASSES)
TECH_IGNORE = -100


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def dist_setup() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg: str):
    if is_main():
        print(msg, flush=True)


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def raise_fd_limit(target: int = 65536) -> int:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = min(hard, target)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            return want
        return soft
    except Exception:
        return -1


_LABEL_VOCAB: Optional[List[str]] = None


def label_vocab(data_root: Optional[str] = None) -> List[str]:
    global _LABEL_VOCAB
    if _LABEL_VOCAB is not None:
        return _LABEL_VOCAB

    root = data_root or CFG.DATA_ROOT
    path = os.path.join(root, CFG.INDEX_NAME)

    if not os.path.isfile(path):
        _LABEL_VOCAB = []
        return _LABEL_VOCAB

    v = pd.read_parquet(path, columns=["drone_type"])["drone_type"]
    _LABEL_VOCAB = sorted(v.fillna("").astype(str).unique().tolist())
    return _LABEL_VOCAB


def _unwrap_model(model: nn.Module) -> nn.Module:
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


class ShardHandleCache:
    def __init__(self, root: str, capacity: Optional[int] = None):
        self.root = root
        self.capacity = capacity or CFG.SHARD_CACHE_CAP
        self.cache: "OrderedDict[str, h5py.File]" = OrderedDict()

    def get(self, shard: str) -> h5py.File:
        f = self.cache.get(shard)
        if f is not None:
            self.cache.move_to_end(shard)
            return f

        path = os.path.join(self.root, shard)
        for attempt, delay in enumerate((2, 5, 15, 30, 0), start=1):
            try:
                f = h5py.File(path, "r", libver="latest")
                break
            except OSError as exc:
                if not delay:
                    raise OSError(f"Cannot open shard after {attempt} attempts: {path}") from exc
                print(f"[shard open] {path}: {exc}; retry {attempt} in {delay}s", flush=True)
                time.sleep(delay)
        self.cache[shard] = f

        if len(self.cache) > self.capacity:
            _, old = self.cache.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass

        return f

    def close(self):
        for f in self.cache.values():
            try:
                f.close()
            except Exception:
                pass
        self.cache.clear()


class RateWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, shard_root: str, rate_class: str,
                 label_names: List[str]):
        self.rate_class = rate_class
        self.shard_root = shard_root
        self.L = int(CFG.RATE_SPEC[rate_class]["L"])

        cat = pd.Categorical(df["shard"].astype(str))
        self.shard_names = list(cat.categories)
        self.shard_code = np.asarray(cat.codes, dtype=np.int32)
        self.local_idx = df["local_idx"].to_numpy(np.int32)

        self.dataset_id = df["dataset_id"].to_numpy(np.int16)
        self.band_id = df["band_id"].to_numpy(np.int16)
        self.scene_id = df["scene_id"].to_numpy(np.int16)
        self.device_id = df["device_id"].to_numpy(np.int16)
        self.rate_id = np.full(len(df), RATE_TO_ID[rate_class], dtype=np.int16)
        self.valid_length = df["valid_length"].to_numpy(np.int32) if CFG.PAIR_INPUT_LENGTHS else None

        rate_column = "effective_sampling_rate" if "effective_sampling_rate" in df else "sampling_rate"
        self.log_fs = np.log10(df[rate_column].to_numpy(np.float32) + 1.0)
        self.routing_input_samples = (
            df["routing_input_samples"].to_numpy(np.int32) if "routing_input_samples" in df else None
        )
        if self.routing_input_samples is not None:
            from scripts3.adaptive_pretrain_routing import convert
            self.convert_view = convert
        self.fc_ghz = df["center_freq"].to_numpy(np.float32) / 1e9
        self.gain_db = df["gain_db"].to_numpy(np.float32)

        skey = (
            df["dataset"].astype(str) + "|" +
            df["source_file"].astype(str) + "|" +
            df["channel"].astype(str)
        )
        gcat = pd.Categorical(skey)
        self.group_id = np.asarray(gcat.codes, dtype=np.int64)

        self.label_names = label_names
        lut = {n: i for i, n in enumerate(label_names)}
        self.label = (
            df["drone_type"].fillna("").astype(str)
            .map(lut).fillna(-1).to_numpy(np.int16)
        )

        self.is_active = df["is_active_global"].to_numpy(np.int8)
        self.weight = df["sample_weight"].to_numpy(np.float32)

        mod = df["modulation"].fillna("").astype(str)
        self.tech = mod.map(
            lambda x: TECH_TO_ID.get(x, TECH_IGNORE)
        ).to_numpy(np.int64)

        self._cache: Optional[ShardHandleCache] = None

    def __len__(self):
        return len(self.local_idx)

    def _handles(self) -> ShardHandleCache:
        if self._cache is None:
            self._cache = ShardHandleCache(self.shard_root)
        return self._cache

    def __getitem__(self, i: int) -> Dict[str, Any]:
        f = self._handles().get(self.shard_names[self.shard_code[i]])
        x = np.asarray(f["iq"][int(self.local_idx[i])], dtype=np.float32)

        if self.routing_input_samples is not None and self.routing_input_samples[i]:
            x = self.convert_view(x, int(self.routing_input_samples[i]), self.L)
        if x.shape[-1] != self.L:
            fixed = np.zeros((2, self.L), np.float32)
            n = min(self.L, x.shape[-1])
            fixed[:, :n] = x[:, :n]
            x = fixed

        item = {
            "iq": torch.from_numpy(x),
            "dom": torch.tensor(
                [
                    self.dataset_id[i],
                    self.band_id[i],
                    self.scene_id[i],
                    self.device_id[i],
                    self.rate_id[i],
                ],
                dtype=torch.long,
            ),
            "cont": torch.tensor(
                [
                    self.log_fs[i],
                    self.fc_ghz[i],
                    self.gain_db[i] / 50.0,
                ],
                dtype=torch.float32,
            ),
            "group": torch.tensor(self.group_id[i], dtype=torch.long),
            "label": torch.tensor(int(self.label[i]), dtype=torch.long),
            "active": torch.tensor(int(self.is_active[i]), dtype=torch.long),
            "tech": torch.tensor(int(self.tech[i]), dtype=torch.long),
        }
        if self.valid_length is not None:
            item["valid_length"] = torch.tensor(int(self.valid_length[i]), dtype=torch.long)
        return item


class GroupedBatchSampler(Sampler[List[int]]):
    def __init__(self, ds: RateWindowDataset, batch_size: int, views: int,
                 span: int, seed: int = 0, rank: int = 0, world: int = 1):
        assert batch_size % views == 0, (
            f"BATCH[{ds.rate_class}]={batch_size} 必须能被 views={views} 整除"
        )

        self.batch_size = batch_size
        self.views = views
        self.span = span
        self.n_groups_per_batch = batch_size // views
        self.seed = seed
        self.rank = rank
        self.world = world
        self.epoch = 0
        self.skip_batches = 0

        order = np.argsort(ds.group_id, kind="stable")
        gsorted = ds.group_id[order]
        bounds = np.flatnonzero(np.diff(gsorted)) + 1
        self.group_slices = [g for g in np.split(order, bounds) if len(g) >= 1]

        if not self.group_slices:
            raise RuntimeError(f"[{ds.rate_class}] 没有可用 group")

        w = np.array(
            [float(ds.weight[g[0]]) * len(g) for g in self.group_slices],
            dtype=np.float64,
        )
        self.gprob = (
            w / w.sum()
            if w.sum() > 0
            else np.full(len(w), 1.0 / len(w))
        )
        self.gcdf = np.cumsum(self.gprob)
        self.gcdf[-1] = 1.0
        self.n_batches = max(1, len(ds) // (batch_size * world))

        n_single = sum(1 for g in self.group_slices if len(g) == 1)
        log(
            f"    [{ds.rate_class}] {len(ds):,} 窗口 / "
            f"{len(self.group_slices):,} stream "
            f"(单窗口组 {n_single:,}) / {self.n_batches:,} batch/epoch / "
            f"views={views}"
        )

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + self.rank)
        G = len(self.group_slices)

        for batch_index in range(self.n_batches):
            gsel = np.searchsorted(self.gcdf, rng.random(self.n_groups_per_batch))
            gsel = np.clip(gsel, 0, G - 1)

            batch: List[int] = []
            for gi in gsel:
                arr = self.group_slices[gi]
                n = len(arr)

                if n <= self.views:
                    pick = rng.choice(n, size=self.views, replace=True)
                else:
                    anchor = int(rng.integers(0, n))
                    lo = max(0, anchor - self.span)
                    hi = min(n, anchor + self.span + 1)

                    if hi - lo >= self.views:
                        pick = rng.choice(
                            np.arange(lo, hi),
                            size=self.views,
                            replace=False,
                        )
                    else:
                        pick = rng.choice(n, size=self.views, replace=False)

                batch.extend(int(arr[p]) for p in pick)

            if batch_index >= self.skip_batches:
                yield batch


def _views_for_rate(rc: str) -> int:
    if isinstance(CFG.VIEWS_PER_GROUP, dict):
        return int(CFG.VIEWS_PER_GROUP[rc])
    return int(CFG.VIEWS_PER_GROUP)


def load_index(split: str) -> Dict[str, pd.DataFrame]:
    path = os.path.join(CFG.DATA_ROOT, CFG.INDEX_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到索引: {path}")

    cols = [
        "dataset", "source_file", "channel", "rate_class", "band", "scene",
        "device", "drone_type", "modulation", "split",
        "sampling_rate", "center_freq", "gain_db",
        "dataset_id", "band_id", "scene_id", "device_id",
        "window_len", "start_sample", "shard", "local_idx",
        "is_active_global", "sample_weight",
    ]
    if CFG.INDEX_SHA256:
        import pyarrow.parquet as pq

        names = pq.ParquetFile(path).schema_arrow.names
        cols.extend(name for name in ("routing_input_samples", "effective_sampling_rate") if name in names)
    if CFG.PAIR_INPUT_LENGTHS:
        cols.append("valid_length")

    df = pd.read_parquet(path, columns=cols, filters=[("split", "==", split)])

    if CFG.DROP_EXCL:
        df = df[df["split"] != "excl"]

    df = df[df["split"] == split]

    if CFG.INACTIVE_KEEP_RATIO is not None and split == "train":
        inact = df["is_active_global"] <= 0
        n_in = int(inact.sum())

        if n_in > 0:
            keep_n = int(n_in * CFG.INACTIVE_KEEP_RATIO)
            if keep_n < n_in:
                drop_idx = df[inact].sample(
                    n=n_in - keep_n,
                    random_state=CFG.SEED,
                ).index
                df = df.drop(index=drop_idx)

            log(f"  静默窗口下采样: {n_in:,} -> {keep_n:,}")

    for c in [
        "dataset", "source_file", "channel", "rate_class", "band",
        "scene", "device", "drone_type", "modulation", "split", "shard",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    out = {}
    for rc in CFG.RATE_SPEC:
        sub = df[df["rate_class"] == rc]
        sub = sub[sub["window_len"] == CFG.RATE_SPEC[rc]["L"]]
        if len(sub):
            out[rc] = sub.reset_index(drop=True)

    return out


def collate(items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def build_loaders(split: str, rank: int, world: int,
                  num_workers: Optional[int] = None,
                  cached_label_names: Optional[List[str]] = None,
                  ) -> Tuple[Dict[str, DataLoader], Dict[str, RateWindowDataset]]:
    dfs = load_index(split)
    shard_root = os.path.join(CFG.DATA_ROOT, CFG.SHARD_SUBDIR)
    nw = CFG.NUM_WORKERS if num_workers is None else num_workers
    lnames = cached_label_names if cached_label_names is not None else label_vocab()

    loaders, dsets = {}, {}
    log(f"  [{split}] 索引载入完成:")

    for rc, df in dfs.items():
        ds = RateWindowDataset(df, shard_root, rc, lnames)
        sampler = GroupedBatchSampler(
            ds,
            CFG.BATCH[rc],
            _views_for_rate(rc),
            CFG.NEIGHBOR_SPAN,
            CFG.SEED,
            rank,
            world,
        )

        nw_rc = (
            CFG.NUM_WORKERS_PER_RATE.get(rc, nw)
            if num_workers is None
            else nw
        )

        loaders[rc] = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=nw_rc,
            pin_memory=CFG.PIN_MEMORY,
            collate_fn=collate,
            persistent_workers=nw_rc > 0,
            prefetch_factor=CFG.PREFETCH if nw_rc > 0 else None,
        )
        dsets[rc] = ds

    return loaders, dsets


class MultiRateIterator:
    def __init__(self, loaders: Dict[str, DataLoader]):
        self.loaders = loaders
        self.iters = {k: iter(v) for k, v in loaders.items()}
        self.epoch = {k: 0 for k in loaders}
        self.consumed = {k: 0 for k in loaders}

    def state_dict(self):
        return {rc: (self.epoch[rc], self.consumed[rc]) for rc in self.loaders}

    def restore(self, state):
        for rc, (epoch, consumed) in state.items():
            sampler = self.loaders[rc].batch_sampler
            assert 0 <= consumed <= len(sampler)
            sampler.set_epoch(epoch)
            sampler.skip_batches = consumed
            self.epoch[rc], self.consumed[rc] = epoch, consumed
            self.iters[rc] = iter(self.loaders[rc])

    def next(self, rates: Tuple[str, ...]) -> Dict[str, Dict[str, torch.Tensor]]:
        out = {}

        for rc in rates:
            if rc not in self.loaders:
                continue

            try:
                out[rc] = next(self.iters[rc])
            except StopIteration:
                self.epoch[rc] += 1
                self.consumed[rc] = 0
                sampler = self.loaders[rc].batch_sampler

                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.epoch[rc])
                    sampler.skip_batches = 0

                self.iters[rc] = iter(self.loaders[rc])
                out[rc] = next(self.iters[rc])

            self.consumed[rc] += 1

        return out


@torch.no_grad()
def augment_batch(x: torch.Tensor) -> torch.Tensor:
    B, _, L = x.shape
    dev = x.device

    x_f = x.float()
    apply = (torch.rand(B, device=dev) < CFG.AUG_PROB).float().view(B, 1)

    z_r = x_f[:, 0]
    z_i = x_f[:, 1]


    if CFG.AUG_PHASE:
        ph = torch.rand(B, 1, device=dev) * (2.0 * math.pi) * apply
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)

        z_r_new = z_r * cos_ph - z_i * sin_ph
        z_i_new = z_r * sin_ph + z_i * cos_ph
        z_r, z_i = z_r_new, z_i_new


    if CFG.AUG_FREQ_OFF:
        f = (
            (torch.rand(B, 1, device=dev) * 2.0 - 1.0)
            * CFG.AUG_FREQ_MAX
            * apply
        )
        n = torch.arange(L, device=dev, dtype=torch.float32).view(1, L)
        ph = 2.0 * math.pi * f * n

        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)

        z_r_new = z_r * cos_ph - z_i * sin_ph
        z_i_new = z_r * sin_ph + z_i * cos_ph
        z_r, z_i = z_r_new, z_i_new


    if CFG.AUG_TIME_SHIFT:
        max_sh = max(1, int(L * CFG.AUG_TIME_SHIFT_MAX))
        sh = (
            torch.rand(B, device=dev)
            * max_sh
            * apply.squeeze(1)
        ).long()

        idx = (
            torch.arange(L, device=dev).view(1, L)
            + sh.view(B, 1)
        ) % L

        z_r = torch.gather(z_r, 1, idx)
        z_i = torch.gather(z_i, 1, idx)


    lo, hi = CFG.AUG_AMP_SCALE
    amp = 1.0 + (
        torch.rand(B, 1, device=dev) * (hi - lo) + lo - 1.0
    ) * apply

    z_r = z_r * amp
    z_i = z_i * amp


    slo, shi = CFG.AUG_NOISE_SNR_DB
    snr = torch.rand(B, 1, device=dev) * (shi - slo) + slo

    sig_p = (
        z_r.square() + z_i.square()
    ).mean(dim=1, keepdim=True).clamp_min(1e-8)

    noise_p = sig_p / torch.pow(10.0, snr / 10.0)
    sigma = torch.sqrt(noise_p / 2.0) * apply

    z_r = z_r + sigma * torch.randn(B, L, device=dev)
    z_i = z_i + sigma * torch.randn(B, L, device=dev)

    out = torch.stack([z_r, z_i], dim=1)
    return torch.clamp(out, -8.0, 8.0).to(x.dtype)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x

        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()


class Block(nn.Module):
    def __init__(self, d: int, h: int, ratio: int, drop: float, dpr: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d * ratio, d),
        )

        self.h = h
        self.dh = d // h
        self.drop = drop
        self.dp = DropPath(dpr)

    def forward(self, x):
        B, N, D = x.shape

        y = self.n1(x)
        q = self.q(y).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.k(y).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.v(y).view(B, N, self.h, self.dh).transpose(1, 2)

        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop if self.training else 0.0,
        )

        a = a.transpose(1, 2).reshape(B, N, D)
        x = x + self.dp(self.o(a))
        x = x + self.dp(self.ff(self.n2(x)))

        return x


def _safe_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class MultiRateStem(nn.Module):
    def __init__(self, strides: Tuple[int, ...], d_model: int):
        super().__init__()
        mid = max(64, d_model // 8)
        chans = [2, mid, mid * 2, d_model]

        layers = []
        for i, s in enumerate(strides):
            k = 2 * s + 1
            layers += [
                nn.Conv1d(
                    chans[i],
                    chans[i + 1],
                    kernel_size=k,
                    stride=s,
                    padding=k // 2,
                ),
                _safe_group_norm(chans[i + 1]),
                nn.GELU(),
            ]

        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv1d(d_model, d_model, 1)
        self.total_stride = int(np.prod(strides))

    def forward(self, x):
        return self.proj(self.net(x)).transpose(1, 2)


class DomainEmbedding(nn.Module):
    SIZES = (N_DATASET, N_BAND, N_SCENE, N_DEVICE, N_RATE)

    def __init__(self, d_model: int):
        super().__init__()
        self.e_dataset = nn.Embedding(N_DATASET + 1, d_model)
        self.e_band = nn.Embedding(N_BAND + 1, d_model)
        self.e_scene = nn.Embedding(N_SCENE + 1, d_model)
        self.e_device = nn.Embedding(N_DEVICE + 1, d_model)
        self.e_rate = nn.Embedding(N_RATE + 1, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.null_emb = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.null_emb, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dom, cont, drop: Optional[torch.Tensor] = None):
        d = dom.clone()

        for j, n in enumerate(self.SIZES):
            invalid = (d[:, j] < 0) | (d[:, j] >= n)
            d[:, j] = torch.where(
                invalid,
                torch.full_like(d[:, j], n),
                d[:, j],
            )

        e = (
            self.e_band(d[:, 1])
            + self.e_scene(d[:, 2])
            + self.e_device(d[:, 3])
            + self.e_rate(d[:, 4])
            + self.mlp(cont)
        )

        if CFG.USE_DATASET_ID:
            e = e + self.e_dataset(d[:, 0])

        e = self.norm(e)

        if drop is not None and drop.any():
            m = drop.float().unsqueeze(-1)
            e = e * (1.0 - m) + self.null_emb.expand_as(e) * m

        return e


class _ParamHolder(nn.Module):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.param = nn.Parameter(data)
        nn.init.trunc_normal_(self.param, std=0.02)

    def forward(self):
        return self.param


class Decoder(nn.Module):
    def __init__(self, d_in: int, d_dec: int, n_layers: int, n_heads: int,
                 patch_samples: Dict[str, int]):
        super().__init__()

        self.embed = nn.Linear(d_in, d_dec)
        self.pos = nn.ModuleDict()

        for rc, spec in CFG.RATE_SPEC.items():
            n_tok = spec["L"] // patch_samples[rc]
            self.pos[rc] = _ParamHolder(torch.zeros(1, n_tok, d_dec))

        self.blocks = nn.ModuleList([
            Block(d_dec, n_heads, 4, 0.0, 0.0)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_dec)

        self.time_heads = nn.ModuleDict({
            rc: nn.Linear(d_dec, 2 * patch_samples[rc])
            for rc in CFG.RATE_SPEC
        })
        self.spec_head = nn.Linear(d_dec, CFG.SPEC_BINS)
        self.energy_head = nn.Linear(d_dec, 1)

    def forward(self, tok: torch.Tensor, rate: str):
        h = self.embed(tok)
        h = h + self.pos[rate].param[:, :h.shape[1]]

        for blk in self.blocks:
            h = blk(h)

        h = self.norm(h)

        return (
            self.time_heads[rate](h),
            self.spec_head(h),
            self.energy_head(h).squeeze(-1),
        )


class DroneRFMultiRate(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG.D_MODEL
        self.d_model = d

        self.stems = nn.ModuleDict({
            rc: MultiRateStem(tuple(spec["strides"]), d)
            for rc, spec in CFG.RATE_SPEC.items()
        })

        self.patch_samples = {
            rc: self.stems[rc].total_stride
            for rc in CFG.RATE_SPEC
        }

        self.domain = DomainEmbedding(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos = nn.ModuleDict()
        for rc, spec in CFG.RATE_SPEC.items():
            n_tok = spec["L"] // self.patch_samples[rc]
            self.pos[rc] = _ParamHolder(torch.zeros(1, n_tok + 1, d))

        dpr = torch.linspace(0, CFG.DROP_PATH, CFG.N_LAYERS).tolist()
        self.blocks = nn.ModuleList([
            Block(
                d,
                CFG.N_HEADS,
                CFG.FFN_RATIO,
                CFG.DROPOUT,
                dpr[i],
            )
            for i in range(CFG.N_LAYERS)
        ])

        self.norm = nn.LayerNorm(d)

        if CFG.USE_DECODER:
            self.decoder = Decoder(
                d,
                CFG.DEC_DIM,
                CFG.DEC_LAYERS,
                CFG.DEC_HEADS,
                self.patch_samples,
            )
            self.recon_heads = None
            self.spec_head = None
            self.energy_head = None
        else:
            self.decoder = None
            self.recon_heads = nn.ModuleDict({
                rc: nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Linear(d, 2 * self.patch_samples[rc]),
                )
                for rc in CFG.RATE_SPEC
            })
            self.spec_head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, CFG.SPEC_BINS),
            )
            self.energy_head = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Linear(d // 2, 1),
            )

        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, CFG.PROJ_DIM),
        )

        self.presence_head = nn.Linear(d, 1)
        self.tech_head = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Linear(256, N_TECH),
        )

        self.domain_clf = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Linear(256, N_DATASET),
        )

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x, dom, cont, rate, mask=None, domain_drop=None,
               return_hidden: bool = False):
        t = self.stems[rate](x)
        B, N, D = t.shape

        if mask is not None:
            t = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.expand(B, N, D),
                t,
            )

        dvec = self.domain(dom, cont, domain_drop).unsqueeze(1)
        cls = self.cls.expand(B, 1, D)

        if CFG.DOMAIN_EMB_ON_CLS:
            cls = cls + dvec

        h = torch.cat([cls, t + dvec], dim=1)
        h = h + self.pos[rate].param[:, :N + 1]

        hidden = []
        for blk in self.blocks:
            if CFG.GRAD_CKPT and self.training:
                h = _ckpt(blk, h, use_reentrant=False)
            else:
                h = blk(h)

            if return_hidden:
                hidden.append(h)

        h = self.norm(h)

        if return_hidden:
            return h[:, 0], h[:, 1:], hidden

        return h[:, 0], h[:, 1:]

    def reconstruct(self, tok: torch.Tensor, rate: str):
        if self.decoder is not None:
            return self.decoder(tok, rate)

        return (
            self.recon_heads[rate](tok),
            self.spec_head(tok),
            self.energy_head(tok).squeeze(-1),
        )


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    B, C, L = x.shape
    N = L // p
    t = x[:, :, :N * p].reshape(B, C, N, p).permute(0, 2, 1, 3)
    return t.reshape(B, N, C * p)


def spectral_target(x: torch.Tensor, p: int, bins: int) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        B, _, L = x.shape
        N = L // p

        z = torch.complex(x[:, 0], x[:, 1])[:, :N * p].reshape(B, N, p)
        S = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
        mag = torch.log1p(S.abs())

        mag = F.adaptive_avg_pool1d(
            mag.reshape(B * N, 1, p),
            bins,
        ).reshape(B, N, bins)

        mu = mag.mean(dim=-1, keepdim=True)
        sd = mag.std(dim=-1, keepdim=True).clamp_min(1e-6)

        return (mag - mu) / sd


def energy_target(x: torch.Tensor, p: int) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        B, _, L = x.shape
        N = L // p

        pw = (
            x[:, 0] ** 2 + x[:, 1] ** 2
        )[:, :N * p].reshape(B, N, p).mean(-1)

        e = torch.log10(pw.clamp_min(1e-8))
        mu = e.mean(dim=1, keepdim=True)
        sd = e.std(dim=1, keepdim=True).clamp_min(CFG.ENERGY_SD_FLOOR)

        return (e - mu) / sd


_MASK_MODES = ("random", "block", "tail")


def build_mask(B: int, N: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    p = torch.tensor(
        [CFG.MASK_MODE_PROB[m] for m in _MASK_MODES],
        device=device,
        dtype=torch.float32,
    )

    mode = torch.multinomial(p, B, replacement=True)
    ar = torch.arange(N, device=device).view(1, N)

    n_rand = max(1, int(N * CFG.MASK_RATIO))
    n_tail = max(1, int(N * CFG.TAIL_RATIO))

    rank = torch.rand(B, N, device=device).argsort(1).argsort(1)
    m_rand = rank < n_rand

    n_blk = 2
    blen = max(1, n_rand // n_blk)
    hi = max(1, N - blen + 1)

    m_blk = torch.zeros(B, N, dtype=torch.bool, device=device)
    for _ in range(n_blk):
        s = torch.randint(0, hi, (B, 1), device=device)
        m_blk |= (ar >= s) & (ar < s + blen)

    m_tail = ar >= (N - n_tail)
    md = mode.view(B, 1)

    mask = torch.where(
        md == 0,
        m_rand,
        torch.where(md == 1, m_blk, m_tail),
    )

    return mask, (mode == 2)


def apply_input_mask(x: torch.Tensor, mask: torch.Tensor, p: int) -> torch.Tensor:
    B, C, L = x.shape
    N = mask.shape[1]

    keep = (
        (~mask).to(x.dtype)
        .unsqueeze(-1)
        .expand(B, N, p)
        .reshape(B, 1, N * p)
    )

    if N * p < L:
        keep = F.pad(keep, (0, L - N * p), value=1.0)

    return x * keep


def masked_mse(pred, tgt, mask, sample_sel=None):
    m = mask.float()

    if sample_sel is not None:
        m = m * sample_sel.float().unsqueeze(1)

    denom = m.sum().clamp_min(1.0)

    if pred.dim() == 3:
        per_tok = ((pred - tgt) ** 2).mean(dim=-1)
    else:
        per_tok = (pred - tgt) ** 2

    return (per_tok * m).sum() / denom


def supcon_loss(z, group, temp):
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / temp

    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)

    pos = (group.unsqueeze(0) == group.unsqueeze(1)) & (~eye)
    valid = pos.any(dim=1)

    if valid.sum() == 0:
        return z.new_zeros(())

    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_f = pos.float()
    mean_lp = (logprob * pos_f).sum(1) / pos_f.sum(1).clamp_min(1.0)

    return -mean_lp[valid].mean()


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def phase_of(progress: float):
    rates, w = CFG.PHASES[0][1], CFG.PHASES[0][2]

    for thr, r, ww in CFG.PHASES:
        if progress >= thr:
            rates, w = r, ww

    return rates, w


def lr_at(step: int) -> float:
    if step < CFG.WARMUP_STEPS:
        return CFG.LR * step / max(1, CFG.WARMUP_STEPS)

    p = (
        (step - CFG.WARMUP_STEPS)
        / max(1, CFG.MAX_STEPS - CFG.WARMUP_STEPS)
    )
    p = min(1.0, p)

    return CFG.MIN_LR + 0.5 * (
        CFG.LR - CFG.MIN_LR
    ) * (1 + math.cos(math.pi * p))


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        core = _unwrap_model(model)

        self.shadow = {
            k: v.detach().clone().float()
            for k, v in core.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        core = _unwrap_model(model)

        current = core.state_dict()
        shadows = list(self.shadow.values())
        sources = [current[k].detach().float() for k in self.shadow]
        torch._foreach_mul_(shadows, self.decay)
        torch._foreach_add_(shadows, sources, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow


def forward_losses(model, batches, device, weights, grl_lambda, train=True):
    core = _unwrap_model(model)
    stats: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device)

    per_rate_z, per_rate_g, per_rate_cls = [], [], []
    per_rate_grl_target = []

    n_tokens = 0

    for rc, b in batches.items():
        x = b["iq"].to(device, non_blocking=True)
        dom = b["dom"].to(device, non_blocking=True)
        cont = b["cont"].to(device, non_blocking=True)
        B = x.shape[0]

        if train:
            x = augment_batch(x)

        domain_drop = None
        if train and CFG.DOMAIN_DROPOUT > 0:
            domain_drop = (
                torch.rand(B, device=device) < CFG.DOMAIN_DROPOUT
            )

        p = core.patch_samples[rc]
        N = x.shape[-1] // p
        n_tokens += B * (N + 1)

        tgt_time = patchify(x, p)
        if CFG.NORM_PIX_LOSS:
            mu = tgt_time.mean(dim=-1, keepdim=True)
            sd = tgt_time.std(dim=-1, keepdim=True).clamp_min(1e-6)
            tgt_time = (tgt_time - mu) / sd

        tgt_spec = spectral_target(x, p, CFG.SPEC_BINS)
        tgt_ener = energy_target(x, p)

        mask, is_tail = build_mask(B, N, device)
        x_in = apply_input_mask(x, mask, p) if CFG.MASK_INPUT else x

        cls, tok = core.encode(
            x_in,
            dom,
            cont,
            rc,
            mask,
            domain_drop,
        )

        pred_time, pred_spec, pred_ener = core.reconstruct(tok, rc)

        l_rec = masked_mse(pred_time, tgt_time, mask, ~is_tail)
        l_prd = masked_mse(pred_time, tgt_time, mask, is_tail)
        l_spc = masked_mse(pred_spec, tgt_spec, mask)
        l_eng = masked_mse(pred_ener, tgt_ener, mask)

        total = total + (
            weights["recon"] * l_rec
            + weights["pred"] * l_prd
            + weights["spec"] * l_spc
            + weights["energy"] * l_eng
        )

        stats[f"recon/{rc}"] = l_rec.detach()
        stats[f"pred/{rc}"] = l_prd.detach()
        stats[f"spec/{rc}"] = l_spc.detach()
        stats[f"energy/{rc}"] = l_eng.detach()

        if CFG.W_PRESENCE > 0:
            y_a = b["active"].to(device).float()
            logit_a = core.presence_head(cls).squeeze(-1)

            if CFG.PRESENCE_POS_WEIGHT and train:
                pr = y_a.mean().clamp(0.02, 0.98)
                pw = ((1.0 - pr) / pr).clamp(0.1, 10.0)
                l_pres = F.binary_cross_entropy_with_logits(
                    logit_a,
                    y_a,
                    pos_weight=pw,
                )
            else:
                l_pres = F.binary_cross_entropy_with_logits(logit_a, y_a)

            total = total + CFG.W_PRESENCE * l_pres

            stats[f"pres/{rc}"] = l_pres.detach()
            stats[f"pres_acc/{rc}"] = (
                ((logit_a > 0).float() == y_a)
                .float()
                .mean()
                .detach()
            )
            stats[f"pres_pos/{rc}"] = y_a.mean().detach()

        if CFG.W_TECH > 0:
            y_t = b["tech"].to(device)
            lg_t = core.tech_head(cls)

            if (b["tech"] != TECH_IGNORE).any():
                l_tech = F.cross_entropy(
                    lg_t,
                    y_t,
                    ignore_index=TECH_IGNORE,
                )

                stats[f"tech/{rc}"] = l_tech.detach()
                stats[f"tech_acc/{rc}"] = (
                    (lg_t.argmax(-1) == y_t)[y_t != TECH_IGNORE]
                    .float()
                    .mean()
                    .detach()
                )
            else:
                l_tech = lg_t.sum() * 0.0

            total = total + CFG.W_TECH * l_tech

        per_rate_z.append(core.proj(cls))
        per_rate_g.append(b["group"].to(device))
        per_rate_cls.append(cls)

        per_rate_grl_target.append((dom[:, 0].clamp(0, N_DATASET - 1),
                                    b["dom"][:, 0].clamp(0, N_DATASET - 1).unique().numel() > 1))

    if weights["contrast"] > 0 and per_rate_z:
        if CFG.CONTRAST_PER_RATE:
            losses = [
                supcon_loss(z, g, CFG.CONTRAST_T)
                for z, g in zip(per_rate_z, per_rate_g)
            ]
            l_con = torch.stack(losses).mean()
        else:
            z = torch.cat(per_rate_z, 0)
            g = torch.cat([
                gg * 8 + RATE_TO_ID[rc]
                for gg, rc in zip(per_rate_g, batches.keys())
            ], 0)
            l_con = supcon_loss(z, g, CFG.CONTRAST_T)

        total = total + weights["contrast"] * l_con
        stats["contrast"] = l_con.detach()

    if weights["domain"] > 0 and per_rate_cls:
        d_losses, d_accs = [], []

        for c, (y, mixed) in zip(per_rate_cls, per_rate_grl_target):
            if not mixed:
                continue

            logit_d = core.domain_clf(
                GradReverse.apply(c, grl_lambda)
            )

            d_losses.append(F.cross_entropy(logit_d, y))
            d_accs.append(
                (logit_d.argmax(-1) == y).float().mean()
            )

        if d_losses:
            l_dom = torch.stack(d_losses).mean()
            total = total + weights["domain"] * l_dom
            stats["domain"] = l_dom.detach()
            stats["domain_acc"] = torch.stack(d_accs).mean().detach()

        stats["domain_fired"] = torch.as_tensor(
            float(len(d_losses)),
            device=device,
        )

    stats["loss"] = total.detach()
    return total, stats, n_tokens


def _try_resume(model, ck) -> bool:
    try:
        model.load_state_dict(ck["model"], strict=True)
        return True
    except Exception as e:
        log(
            "[警告] checkpoint 与当前模型结构不兼容"
            "(可能改了 --size), 跳过续训:\n"
            f"  {e}"
        )
        return False


def save_ckpt(path, model, opt, scaler, ema, step, best_val, world,
              full=True, keep_ema=True, streams=None):
    core = _unwrap_model(model)

    obj = {
        "model": core.state_dict(),
        "ema": (
            {k: v.cpu() for k, v in ema.state_dict().items()}
            if (ema and keep_ema)
            else None
        ),
        "step": int(step),
        "best_val": float(best_val),
        "size": CFG.SIZE,
        "cfg": {
            k: v
            for k, v in CFG.__dict__.items()
            if k.isupper() and isinstance(v, (int, float, str, bool))
        },
    }

    if full:
        obj["opt"] = opt.state_dict()
        obj["scaler"] = (
            scaler.state_dict()
            if scaler.is_enabled()
            else None
        )

    if full and streams:
        obj["streams"] = [it.state_dict() if it else None for it in streams]
        obj["rng"] = (random.getstate(), np.random.get_state(),
                      torch.get_rng_state(), torch.cuda.get_rng_state_all())
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def prune_ckpts(d: str, keep: int):
    cks = sorted([
        f for f in os.listdir(d)
        if f.startswith("step_")
    ])

    for f in cks[:-keep]:
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


def run_pretrain(args, comparison=None):
    rank, world, local_rank = dist_setup()

    if world > 1:
        raise RuntimeError(
            "本脚本 forward 直接调用 core.encode 绕过了 DDP.forward，"
            "多卡梯度不会同步。\n"
            "请单卡运行；若要多卡，需要将 forward_losses 改为走 DDP wrapper。"
        )

    set_seed(CFG.SEED, rank)

    device = torch.device(
        f"cuda:{local_rank}"
        if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    config_path = os.path.join(CFG.OUT_DIR, "config.json")
    if os.path.isfile(os.path.join(CFG.OUT_DIR, "last.pt")) and os.path.isfile(config_path):
        with open(config_path) as f:
            previous_config = json.load(f)
        if previous_config.get("INDEX_SHA256", "") != CFG.INDEX_SHA256:
            raise ValueError("Resume directory uses a different pretraining index")

    if is_main():
        with open(
            config_path,
            "w",
        ) as f:
            json.dump(
                {
                    k: (
                        v
                        if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v)
                    )
                    for k, v in CFG.__dict__.items()
                    if k.isupper()
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    log("=" * 88)
    log(f"DroneRF-MultiRate 多率自监督预训练 v1.5.1 [size={CFG.SIZE}]")
    log("=" * 88)
    log(f"数据      : {CFG.DATA_ROOT}")
    log(f"输出      : {CFG.OUT_DIR}")
    log(f"设备      : {device}   world={world}")
    log(
        f"结构      : d={CFG.D_MODEL} L={CFG.N_LAYERS} H={CFG.N_HEADS} "
        f"decoder={'%dL/d%d' % (CFG.DEC_LAYERS, CFG.DEC_DIM) if CFG.USE_DECODER else 'off'}"
    )
    log(
        f"batch     : {CFG.BATCH}  "
        f"views={CFG.VIEWS_PER_GROUP}  accum={CFG.GRAD_ACCUM}"
    )
    log(
        f"workers   : {CFG.NUM_WORKERS_PER_RATE} "
        f"(val={CFG.VAL_NUM_WORKERS})"
    )
    log(
        f"步数      : {CFG.MAX_STEPS:,}  "
        f"warmup={CFG.WARMUP_STEPS:,}  lr={CFG.LR}"
    )
    log(f"grad_ckpt : {CFG.GRAD_CKPT}")
    log("=" * 88)

    cached_lnames = label_vocab()

    train_loaders, _ = build_loaders(
        "train",
        rank,
        world,
        cached_label_names=cached_lnames,
    )

    val_loaders, _ = build_loaders(
        "val",
        rank,
        world,
        num_workers=CFG.VAL_NUM_WORKERS,
        cached_label_names=cached_lnames,
    )

    if not train_loaders:
        raise RuntimeError("训练集为空，请检查索引与 rate_class/window_len")

    train_it = MultiRateIterator(train_loaders)
    val_it = MultiRateIterator(val_loaders) if val_loaders else None

    model = DroneRFMultiRate().to(device)
    if comparison is not None:
        comparison.record_initial_weights(model)

    n_par = sum(p.numel() for p in model.parameters())
    n_bb = sum(
        p.numel()
        for n, p in model.named_parameters()
        if any(
            n.startswith(pf)
            for pf in (
                "stems.", "domain.", "cls", "mask_token",
                "pos.", "blocks.", "norm.",
            )
        )
    )

    log(
        f"参数量    : 总 {n_par / 1e6:.2f} M   "
        f"(backbone {n_bb / 1e6:.2f} M)"
    )

    for rc in CFG.RATE_SPEC:
        L = CFG.RATE_SPEC[rc]["L"]
        ps = model.patch_samples[rc]
        log(f"  {rc:<5} L={L:<6} patch={ps:<4} tokens={L // ps}")

    if CFG.COMPILE:
        log("torch.compile 编译中 ...")
        model = torch.compile(model)

    decay, no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if (
            p.ndim <= 1
            or "pos" in n
            or "token" in n
            or "cls" in n
            or "null_emb" in n
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": CFG.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=CFG.LR,
        betas=CFG.BETAS,
    )

    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[CFG.AMP_DTYPE]

    use_amp = (
        CFG.AMP_DTYPE != "fp32"
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            CFG.AMP_DTYPE == "fp16"
            and device.type == "cuda"
        ),
    )

    ema = EMA(model, CFG.EMA_DECAY) if CFG.EMA_DECAY else None

    start_step, best_val = 0, float("inf")
    last_ck = os.path.join(CFG.OUT_DIR, "last.pt")

    if os.path.isfile(last_ck):
        ck = _torch_load(last_ck, map_location="cpu")
        if ck.get("cfg", {}).get("INDEX_SHA256", "") != CFG.INDEX_SHA256:
            raise ValueError("Resume checkpoint uses a different pretraining index")
        target_model = _unwrap_model(model)

        target_model.load_state_dict(ck["model"], strict=True)
        if "streams" not in ck or "rng" not in ck:
            raise ValueError("Incomplete resume state in this experiment checkpoint")
        if "opt" not in ck:
            raise ValueError("Missing optimizer state")
        opt.load_state_dict(ck["opt"])

        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])

        if ema and ck.get("ema"):
            ema.shadow = {
                k: v.to(device)
                for k, v in ck["ema"].items()
            }

        start_step = int(ck["step"]) + 1
        best_val = ck.get("best_val", best_val)

        log(
            f"[续训] 从 step {start_step} 恢复, "
            f"best={best_val:.4f}"
        )

        for iterator, state in zip((train_it, val_it), ck["streams"]):
            if iterator:
                iterator.restore(state)
        py, np_state, cpu, cuda = ck["rng"]
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        del ck

    writer = None
    if is_main() and CFG.USE_TENSORBOARD:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(CFG.OUT_DIR, "tb"))
        except Exception:
            writer = None

    logf = (
        open(os.path.join(CFG.OUT_DIR, "train_log.jsonl"), "a")
        if is_main()
        else None
    )

    log("\n>>> 开始训练\n")
    model.train()

    stop_after = CFG.MAX_STEPS if comparison is None else comparison.stop_after
    pbar = tqdm(
        range(start_step, stop_after),
        initial=start_step,
        total=stop_after,
        disable=not is_main(),
        smoothing=0.05,
    )

    run_stats: Dict[str, torch.Tensor] = {}
    run_cnt: Dict[str, int] = {}
    run_n = 0
    tok_acc = 0
    t_win = time.time()
    t0 = time.time()

    autocast_dev = "cuda" if device.type == "cuda" else "cpu"

    for step in pbar:
        progress = step / CFG.MAX_STEPS
        rates, weights = phase_of(progress)
        if CFG.INDEX_SHA256 and not any(rate in train_loaders for rate in rates):
            rates = (next(rate for rate in CFG.RATE_SPEC if rate in train_loaders),)
        grl = 2.0 / (1.0 + math.exp(-10 * progress)) - 1.0

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        opt.zero_grad(set_to_none=True)

        for micro in range(CFG.GRAD_ACCUM):
            if comparison is None:
                batches = train_it.next(rates)
            else:
                batches = comparison.training_batch(train_it, step, micro, rates)

            if not batches:
                continue

            with nullcontext():
                with torch.autocast(
                    autocast_dev,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    loss_function = forward_losses if comparison is None else comparison.training_loss
                    loss, st, ntok = loss_function(model, batches, device, weights, grl, train=True)
                    loss = loss / CFG.GRAD_ACCUM

                scaler.scale(loss).backward()

            for k, v in st.items():
                run_stats[k] = run_stats.get(k, 0) + v
                run_cnt[k] = run_cnt.get(k, 0) + 1

            tok_acc += ntok
            run_n += 1

        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CFG.GRAD_CLIP,
        )

        scaler.step(opt)
        scaler.update()

        if ema:
            ema.update(model)

        if step % CFG.LOG_EVERY == 0 and run_n > 0:
            ks = list(run_stats.keys())
            vals = torch.stack([
                run_stats[k].float()
                for k in ks
            ]).detach().cpu().tolist()

            avg = {
                k: v / max(1, run_cnt[k])
                for k, v in zip(ks, vals)
            }

            dt = max(1e-6, time.time() - t_win)
            ips = CFG.LOG_EVERY / dt if step > start_step else 0.0
            tps = tok_acc / dt

            run_stats.clear()
            run_cnt.clear()
            run_n = 0
            tok_acc = 0
            t_win = time.time()

            if is_main():
                pbar.set_postfix_str(
                    f"L={avg.get('loss', 0):.3f} "
                    f"sp={avg.get('spec/high', 0):.3f} "
                    f"en={avg.get('energy/high', 0):.3f} "
                    f"df={avg.get('domain_fired', 0):.1f} "
                    f"lr={lr_at(step):.1e} "
                    f"gn={float(gn):.1f} "
                    f"{ips:.1f}it/s "
                    f"ph={'+'.join(r[0] for r in rates)}"
                )

                rec = {
                    "step": step,
                    "lr": lr_at(step),
                    "grad_norm": float(gn),
                    "it_s": round(ips, 3),
                    "tok_s": round(tps, 1),
                    "rates": list(rates),
                    **{
                        k: round(float(v), 5)
                        for k, v in avg.items()
                    },
                }

                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                if writer:
                    for k, v in avg.items():
                        writer.add_scalar(f"train/{k}", float(v), step)

                    writer.add_scalar("train/lr", lr_at(step), step)
                    writer.add_scalar("perf/it_per_s", ips, step)

        if val_it and step > 0 and step % CFG.VAL_EVERY == 0:
            model.eval()

            vs: Dict[str, torch.Tensor] = {}
            vc: Dict[str, int] = {}
            vn = 0

            with torch.no_grad():
                for _ in range(CFG.VAL_STEPS):
                    vb = val_it.next(rates)

                    if not vb:
                        continue

                    with torch.autocast(
                        autocast_dev,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        _, st, _ = forward_losses(
                            model,
                            vb,
                            device,
                            weights,
                            grl,
                            train=False,
                        )

                    for k, v in st.items():
                        vs[k] = vs.get(k, 0) + v
                        vc[k] = vc.get(k, 0) + 1

                    vn += 1

            model.train()

            if vn > 0:
                ks = list(vs.keys())
                vals = torch.stack([
                    vs[k].float()
                    for k in ks
                ]).detach().cpu().tolist()

                vavg = {
                    k: v / max(1, vc[k])
                    for k, v in zip(ks, vals)
                }

                sel_terms = [
                    vavg[k]
                    for k in CFG.BEST_KEYS
                    if k in vavg
                ]

                if sel_terms:
                    sel = sum(sel_terms) / len(sel_terms)
                else:
                    sel = vavg.get("loss", float("inf"))

                vavg["sel"] = sel

                log(
                    f"\n[val @ {step}] "
                    + "  ".join(
                        f"{k}={v:.4f}"
                        for k, v in sorted(vavg.items())
                    )
                )

                if writer:
                    for k, v in vavg.items():
                        writer.add_scalar(f"val/{k}", float(v), step)

                if is_main() and sel < best_val:
                    best_val = sel

                    save_ckpt(
                        os.path.join(CFG.OUT_DIR, "best.pt"),
                        model,
                        opt,
                        scaler,
                        ema,
                        step,
                        best_val,
                        world,
                        full=True,
                        keep_ema=True,
                        streams=(train_it, val_it),
                    )

                    log(f"[best] sel={best_val:.4f} 已保存")

        completed = step + 1
        milestone = comparison is not None and completed in comparison.milestones
        periodic = (step > 0 and step % CFG.SAVE_EVERY == 0 if comparison is None
                    else completed % CFG.SAVE_EVERY == 0)
        if is_main() and (periodic or milestone):
            save_ckpt(
                last_ck,
                model,
                opt,
                scaler,
                ema,
                step,
                best_val,
                world,
                full=True,
                keep_ema=True,
                streams=(train_it, val_it),
            )

            if comparison is None or milestone:
                checkpoint_step = step if comparison is None else completed
                save_ckpt(
                    os.path.join(CFG.OUT_DIR, f"step_{checkpoint_step:07d}.pt"),
                    model, opt, scaler, ema, step, best_val, world, full=False, keep_ema=False,
                )
            if comparison is None:
                prune_ckpts(CFG.OUT_DIR, CFG.KEEP_LAST_N)

    if is_main():
        if comparison is None or stop_after % CFG.SAVE_EVERY != 0:
            save_ckpt(
                last_ck, model, opt, scaler, ema, stop_after - 1, best_val, world,
                full=True, keep_ema=True, streams=(train_it, val_it),
            )

        log(
            f"\n训练完成, 用时 {(time.time() - t0) / 3600:.2f} 小时, "
            f"best={best_val:.4f}"
        )

        if logf:
            logf.close()

        if writer:
            writer.close()


def run_smoke(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CFG.NUM_WORKERS = 2
    CFG.VAL_NUM_WORKERS = 1

    full_batch = getattr(args, "smoke_full_batch", False)

    if not full_batch:
        CFG.BATCH = {"high": 12, "mid": 8, "low": 8}
        CFG.VIEWS_PER_GROUP = {"high": 4, "mid": 4, "low": 2}
        CFG.NUM_WORKERS_PER_RATE = {"high": 2, "mid": 1, "low": 1}

    CFG.GRAD_ACCUM = 1

    log(
        f">>> 冒烟测试 [size={CFG.SIZE}] "
        f"{'(正式 batch, 测真实峰值显存)' if full_batch else '(小 batch)'}"
    )
    log(f"    batch={CFG.BATCH} grad_ckpt={CFG.GRAD_CKPT}")

    cached_lnames = label_vocab()

    loaders, _ = build_loaders(
        "train",
        0,
        1,
        cached_label_names=cached_lnames,
    )

    it = MultiRateIterator(loaders)
    rates = tuple(loaders.keys())

    log(f"可用 rate: {rates}")

    model = DroneRFMultiRate().to(device)

    n_par = sum(p.numel() for p in model.parameters())
    n_bb = sum(
        p.numel()
        for n, p in model.named_parameters()
        if any(
            n.startswith(pf)
            for pf in (
                "stems.", "domain.", "cls", "mask_token",
                "pos.", "blocks.", "norm.",
            )
        )
    )

    log(
        f"参数量: 总 {n_par / 1e6:.2f} M "
        f"(backbone {n_bb / 1e6:.2f} M, "
        f"d={CFG.D_MODEL} L={CFG.N_LAYERS} H={CFG.N_HEADS})"
    )

    b = it.next(rates)

    for rc, bb in b.items():
        x = bb["iq"]

        log(
            f"[{rc}] iq={tuple(x.shape)} "
            f"mean={x.mean():.4f} std={x.std():.4f} "
            f"nan={int(torch.isnan(x).sum())} "
            f"uniq_group={len(set(bb['group'].tolist()))}/{len(bb['group'])} "
            f"tech_valid={(bb['tech'] != TECH_IGNORE).sum().item()} "
            f"active_pos={bb['active'].float().mean().item():.2f} "
            f"dataset_uniq={bb['dom'][:, 0].unique().tolist()}"
        )


    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    w = dict(
        recon=1.0,
        pred=0.5,
        spec=1.0,
        energy=0.5,
        contrast=0.5,
        domain=0.1,
    )

    autocast_dev = "cuda" if device.type == "cuda" else "cpu"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i in range(4):
        b = it.next(rates)
        t = time.time()

        with torch.autocast(
            autocast_dev,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            loss, st, _ntok = forward_losses(
                model,
                b,
                device,
                w,
                0.5,
                True,
            )

        loss.backward()

        gn = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        opt.step()
        opt.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        log(
            f"step{i} ({time.time() - t:.2f}s): "
            + " ".join(
                f"{k}={float(v):.4f}"
                for k, v in sorted(st.items())
            )
            + f" gn={float(gn):.2f}"
        )

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"\n峰值显存: {peak:.2f} GB")

        if full_batch:
            log(
                "  (正式 batch 真实峰值) "
                "< 20GB -> 正常，可不开 grad-ckpt；"
                "> 28GB -> 加 --grad-ckpt 或 --batch-high 64"
            )
        else:
            log(
                "  (smoke 小 batch，正式 batch 请用 "
                "--smoke-full-batch 测真实峰值)"
            )

    log("\n冒烟测试通过。重点核对:")
    log("  domain_fired 应为 1.0 (high 同时包含 dataset 0 / 1 时会触发)")
    log("  pres_pos/high 若长期偏高，保持 PRESENCE_POS_WEIGHT=True")
    log("  energy/* 应整体下降；少量 batch 波动是正常的")
    log("  recon/* 长期约 0.85~0.98 属正常")
    log("  tech/* 初始约 ln(27)=3.30")


def main():
    ap = argparse.ArgumentParser("DroneRF-MultiRate 预训练 v1.5.1 (单卡)")

    ap.add_argument(
        "--mode",
        default="pretrain",
        choices=["pretrain", "smoke"],
    )
    ap.add_argument(
        "--size",
        default="large",
        choices=list(MODEL_SIZES.keys()),
    )

    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--index", type=str, default=None, help="Use a prepared pretraining index with the original shards")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument(
        "--amp",
        type=str,
        default=None,
        choices=["bf16", "fp16", "fp32"],
    )

    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-decoder", action="store_true")

    ap.add_argument("--batch-high", type=int, default=None)
    ap.add_argument("--batch-mid", type=int, default=None)
    ap.add_argument("--batch-low", type=int, default=None)


    ap.add_argument(
        "--smoke-full-batch",
        action="store_true",
        help="smoke 使用正式 batch，测试真实峰值显存",
    )

    args = ap.parse_args()

    fd = raise_fd_limit()

    apply_size(args.size)

    if args.data:
        CFG.DATA_ROOT = args.data
    if args.index:
        from pathlib import Path
        from scripts3.source_io import digest

        path = Path(args.index).resolve()
        CFG.INDEX_NAME = str(path)
        CFG.INDEX_SHA256 = digest(path)

    if args.out:
        CFG.OUT_DIR = args.out

    if args.steps:
        CFG.MAX_STEPS = args.steps
        CFG.WARMUP_STEPS = min(
            CFG.WARMUP_STEPS,
            max(50, args.steps // 20),
        )


    if args.workers is not None:
        CFG.NUM_WORKERS = args.workers
        CFG.NUM_WORKERS_PER_RATE = {
            "high": args.workers,
            "mid": args.workers,
            "low": args.workers,
        }
        CFG.VAL_NUM_WORKERS = min(
            CFG.VAL_NUM_WORKERS,
            args.workers,
        )

    if args.grad_accum is not None:
        CFG.GRAD_ACCUM = args.grad_accum

    if args.lr is not None:
        CFG.LR = args.lr

    if args.amp:
        CFG.AMP_DTYPE = args.amp

    if args.grad_ckpt:
        CFG.GRAD_CKPT = True

    if args.compile:
        CFG.COMPILE = True

    if args.no_decoder:
        CFG.USE_DECODER = False

    if args.batch_high is not None:
        CFG.BATCH["high"] = args.batch_high

    if args.batch_mid is not None:
        CFG.BATCH["mid"] = args.batch_mid

    if args.batch_low is not None:
        CFG.BATCH["low"] = args.batch_low

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    log(f"fd limit: {fd}")
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_pretrain(args)


if __name__ == "__main__":
    main()
