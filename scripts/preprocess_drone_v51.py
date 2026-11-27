from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pads
from tqdm import tqdm

warnings.filterwarnings("ignore")


try:
    h5py.get_config().complex_names = ("real", "imag")
except Exception:
    pass


class CFG:

    DRONERFA_DIR = os.environ.get("UAVGPT_DRONERFA_DIR", "data/DroneRFa")
    RF_UAVS_DIR = os.environ.get("UAVGPT_RF_UAVS_DIR", "data/RF_UAVs")
    RADIOML18_H5 = os.environ.get("UAVGPT_RADIOML18_H5", "data/RadioML/GOLD_XYZ_OSC.0001_1024.hdf5")
    GENT_WIFI_DIR = os.environ.get("UAVGPT_GENT_WIFI_DIR", "data/Gent")


    OUTPUT_DIR = r"datasets/preprocess_v51"


    DO_DRONERFA  = True
    DO_RF_UAVS   = True
    DO_RADIOML18 = True
    DO_GENT_WIFI = True


    DISK_BUDGET_GB = 700.0
    MIN_FREE_GB    = 80.0


    NUM_WORKERS       = 8
    MAXTASKSPERCHILD  = 16


    WINDOW_LEN_HIGH = 20480
    WINDOW_LEN_MID  = 2048
    WINDOW_LEN_LOW  = 1024
    OVERLAP_RATIO   = 0.5

    FS_HIGH_THRESH = 50e6
    FS_MID_THRESH  = 500e3


    IO_CHUNK_SAMPLES = 4_000_000


    ACTIVE_SCAN  = True
    PRESCAN_MODE = "exact"

    NOISE_FLOOR_Q         = 0.20
    PEAK_Q                = 0.95
    ACTIVE_SNR_TH_DB      = 3.0
    MIN_DYNAMIC_RANGE_DB  = 1.5
    NEG_RATIO             = 0.15
    NOISE_ONLY_KEEP       = 200
    FLAT_CHANNEL_KEEP     = 400


    ACTIVE_PROBE_LEN       = 2048
    PRESCAN_MAX_CANDIDATES = 20000


    DRONERFA_CHANNEL_MODE  = "both"
    MAX_WINDOWS_PER_STREAM = 2000


    RF_UAVS_ACTIVE_SCAN = False

    RF_UAVS_FILES_PER_UNIT = 256
    RF_UAVS_FILE_STRIDE    = 1


    RADIOML18_STRIDE      = 10
    RADIOML18_CHUNK       = 2048
    RADIOML18_UNIT_FRAMES = 100_000
    RADIOML18_DEBUG_CAP   = None


    GENT_BIN_DTYPE   = "float32"
    GENT_INT16_SCALE = 32768.0


    GENT_ACTIVE_SCAN = False
    GENT_MAX_WINDOWS_PER_FILE = 8000


    NORM_METHOD = "robust_std"
    NORM_CLIP   = 5.0
    REMOVE_DC   = True


    STORE_DTYPE      = "float16"
    HDF5_COMPRESSION = None
    HDF5_GZIP_LEVEL  = 1

    SHARD_SIZE_HIGH = 1024
    SHARD_SIZE_MID  = 4096
    SHARD_SIZE_LOW  = 8192


    BATCH_SIZE_HIGH = 256
    BATCH_SIZE_MID  = 512
    BATCH_SIZE_LOW  = 1024


    FILTER_ZERO_WINDOWS = True
    ZERO_ABS_MAX_TH     = 1e-10


    VAL_RATIO                  = 0.03
    SINGLE_GROUP_TIME_SPLIT    = True
    MIN_WINDOWS_FOR_TIME_SPLIT = 200


    TARGET_MIX = {
        "DroneRFa":     0.55,
        "Gent_WiFi":    0.15,
        "RadioML_2018": 0.20,
        "RF_UAVs":      0.10,
    }
    MAX_SAMPLE_WEIGHT = 20.0


    GLOBAL_FLOOR_Q = 0.10


    MAX_FILES_PER_DATASET = None


    DEFAULT_FS_DRONERFA = 100e6
    DEFAULT_FS_RF_UAVS  = 100e6
    DEFAULT_FS_GENT     = 1e6
    DEFAULT_FS_RML18    = 64e3

    GENT_FS_SOURCE = "confirmed"

    FC_915M = 915e6
    FC_24G  = 2.44e9
    FC_58G  = 5.8e9

    FC_RF_UAVS  = 5.775e9
    FC_GENT     = 2.4e9
    FC_BASEBAND = 0.0

    SEED = 42
    DRY_RUN = False


DATASET_ID_MAP = {"DroneRFa": 0, "RF_UAVs": 1, "Gent_WiFi": 2, "RadioML_2018": 3}
DATASET_NAME_OF = {v: k for k, v in DATASET_ID_MAP.items()}


BAND_ID_MAP = {
    "915M": 0, "2.4G": 1, "5.8G": 2, "baseband": 3,
    "UHF": 4, "other": 5, "unknown": 6,
    "VHF": 7, "L_S": 8, "C": 9,
}
SCENE_ID_MAP = {"outdoor": 0, "indoor": 1, "urban_background": 2,
                "lab_simulated": 3, "unknown": 4}
DEVICE_ID_MAP = {"USRP-2955": 0, "SDR": 1, "simulated": 2, "unknown": 3}
RATE_ID_MAP = {"high": 0, "mid": 1, "low": 2}

RML18_CLASSES = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
    'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
    '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
    'AM-DSB-WC', 'OOK', '16QAM'
]

RF_UAVS_DRONE_MAP = {
    "1": "Parrot_Anafi", "2": "Fimi_X8SE", "3": "DJI_Phantom_20",
    "4": "DJI_Mavic_Air", "5": "DJI_Mavic_Mini", "6": "DJI_Phantom_10",
}

DRONERFA_T_NAME_MAP = {
    "T0000": "Background", "T0010": "DJI_Phantom_4_Pro", "T0011": "DJI_Matrice_200",
    "T0101": "DJI_Air_2S", "T0110": "DJI_Mini_3_Pro", "T0111": "DJI_Inspire_2",
    "T1001": "DJI_Mini_2", "T1011": "DJI_Matrice_300", "T1101": "DJI_Matrice_30T",
    "T10010": "FrSky_X20", "T10100": "Taranis_Plus",
}

DRONERFA_24_THEN_58 = {"T0010", "T0011", "T0101", "T0110", "T0111", "T1001"}
DRONERFA_58_THEN_24 = {"T1011", "T1101"}
DRONERFA_915_24 = {"T10010", "T10100"}


INDEX_SCHEMA = pa.schema([
    ("dataset", pa.string()),
    ("unit_id", pa.string()),
    ("source_file", pa.string()),
    ("channel", pa.string()),
    ("t_code", pa.string()),
    ("d_code", pa.string()),
    ("s_code", pa.string()),
    ("active_band", pa.string()),
    ("active_band_conf", pa.string()),
    ("band", pa.string()),
    ("scene", pa.string()),
    ("device", pa.string()),
    ("rate_class", pa.string()),
    ("drone_type", pa.string()),
    ("distance", pa.string()),
    ("modulation", pa.string()),
    ("segment_id", pa.string()),
    ("group_key", pa.string()),
    ("split", pa.string()),
    ("fc_source", pa.string()),
    ("sampling_rate", pa.float64()),
    ("center_freq", pa.float64()),
    ("gain_db", pa.float32()),
    ("dataset_id", pa.int16()),
    ("band_id", pa.int16()),
    ("scene_id", pa.int16()),
    ("device_id", pa.int16()),
    ("rate_id", pa.int16()),
    ("window_idx", pa.int64()),
    ("start_sample", pa.int64()),
    ("end_sample", pa.int64()),
    ("valid_length", pa.int32()),
    ("original_length", pa.int64()),
    ("window_len", pa.int32()),
    ("raw_power", pa.float32()),
    ("norm_power", pa.float32()),
    ("est_snr_db", pa.float32()),
    ("is_active", pa.int8()),
    ("snr_db", pa.int32()),
    ("shard", pa.string()),
    ("local_idx", pa.int32()),
])
INDEX_FIELDS = INDEX_SCHEMA.names
STR_FIELDS = [f.name for f in INDEX_SCHEMA if pa.types.is_string(f.type)]

FINAL_SCHEMA = (INDEX_SCHEMA
                .append(pa.field("snr_global_db", pa.float32()))
                .append(pa.field("is_active_global", pa.int8()))
                .append(pa.field("sample_weight", pa.float32())))


@dataclass
class DomainMeta:
    dataset: str
    sampling_rate: float
    center_freq: float
    gain_db: float = 0.0
    rate_class: str = "unknown"
    scene: str = "unknown"
    band: str = "unknown"
    device: str = "unknown"
    drone_type: str = ""
    distance: str = ""
    segment_id: str = ""
    modulation: str = ""
    snr_db: Optional[int] = None
    source_file: str = ""
    channel: str = ""
    t_code: str = ""
    d_code: str = ""
    s_code: str = ""
    active_band: str = ""
    active_band_conf: str = "unknown"
    fc_source: str = "default"
    group_key: str = ""


@dataclass
class WinItem:
    iq: Optional[np.ndarray]
    meta: DomainMeta
    start: int
    valid_len: int
    orig_len: int
    est_snr_db: float = -999.0
    is_active: int = 1
    window_len: int = 0


@dataclass
class Unit:
    dataset: str
    unit_id: str
    path: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def stable_unit_frac(s: str) -> float:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_split(group_key: str) -> str:
    return "val" if stable_unit_frac("split|" + group_key) < CFG.VAL_RATIO else "train"


def get_rate_class(fs: float) -> str:
    if fs >= CFG.FS_HIGH_THRESH:
        return "high"
    if fs >= CFG.FS_MID_THRESH:
        return "mid"
    return "low"


def get_target_len(rate_class: str) -> int:
    return {"high": CFG.WINDOW_LEN_HIGH,
            "mid": CFG.WINDOW_LEN_MID,
            "low": CFG.WINDOW_LEN_LOW}[rate_class]


def get_batch_size(rate_class: str) -> int:
    return {"high": CFG.BATCH_SIZE_HIGH,
            "mid": CFG.BATCH_SIZE_MID,
            "low": CFG.BATCH_SIZE_LOW}[rate_class]


def get_shard_size(rate_class: str) -> int:
    return {"high": CFG.SHARD_SIZE_HIGH,
            "mid": CFG.SHARD_SIZE_MID,
            "low": CFG.SHARD_SIZE_LOW}.get(rate_class, CFG.SHARD_SIZE_MID)


def window_stride(target_len: int) -> int:
    s = max(1, int(round(target_len * (1.0 - CFG.OVERLAP_RATIO))))
    while target_len % s and s > 1:
        s -= 1
    return max(1, s)


def band_from_fc(fc: float) -> str:
    if fc <= 0:
        return "baseband"
    g = fc / 1e9
    if g < 0.03:
        return "baseband"
    if g < 0.30:
        return "VHF"
    if g < 0.80:
        return "UHF"
    if g <= 1.05:
        return "915M"
    if g < 2.30:
        return "L_S"
    if g <= 2.55:
        return "2.4G"
    if g < 5.10:
        return "C"
    if g <= 6.00:
        return "5.8G"
    return "other"


def encode_domain_ids(meta: DomainMeta) -> Dict[str, int]:
    return {
        "dataset_id": DATASET_ID_MAP.get(meta.dataset, -1),
        "band_id": BAND_ID_MAP.get(meta.band, -1),
        "scene_id": SCENE_ID_MAP.get(meta.scene, -1),
        "device_id": DEVICE_ID_MAP.get(meta.device, -1),
        "rate_id": RATE_ID_MAP.get(meta.rate_class, -1),
    }


def store_dtype():
    return np.float16 if CFG.STORE_DTYPE == "float16" else np.float32


def store_itemsize() -> int:
    return 2 if CFG.STORE_DTYPE == "float16" else 4


def h5_kwargs() -> Dict[str, Any]:
    if CFG.HDF5_COMPRESSION is None:
        return {}
    if CFG.HDF5_COMPRESSION == "gzip":
        return {"compression": "gzip", "compression_opts": CFG.HDF5_GZIP_LEVEL}
    if CFG.HDF5_COMPRESSION == "lzf":
        return {"compression": "lzf"}
    return {}


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def human_bytes(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{n:.2f} PB"


def dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def fsync_file(path: str):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def fsync_dir(path: str):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _cap_or(a: int, b: Optional[int]) -> int:
    return min(a, b) if b else a


def parse_tds_from_filename(fname: str) -> Tuple[str, str, str, Optional[int]]:
    m = re.search(r"(T\d+)\s*[_-]\s*(D\d+)\s*[_-]\s*(S\d+)", fname, re.I)
    if not m:
        return "", "", "", None
    t, d, s = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
    try:
        sv = int(s[1:], 2)
    except Exception:
        sv = None
    return t, d, s, sv


def infer_dronerfa_active_band(t_code: str, s_value: Optional[int]) -> Tuple[str, str]:
    if not t_code or s_value is None:
        return "", "unknown"
    if t_code == "T0000":
        return "", "unknown"
    if t_code in DRONERFA_915_24:
        return "", "unknown"
    if t_code in DRONERFA_24_THEN_58:
        return ("2.4G" if s_value <= 7 else "5.8G"), "doc"
    if t_code in DRONERFA_58_THEN_24:
        return ("5.8G" if s_value <= 5 else "2.4G"), "doc"
    return "", "unknown"


def dronerfa_channel_freq(t_code: str, channel: str) -> float:
    if t_code in DRONERFA_915_24:
        return CFG.FC_915M if channel == "RF0" else CFG.FC_24G
    return CFG.FC_24G if channel == "RF0" else CFG.FC_58G


def should_keep_dronerfa_channel(t_code, drone_type, channel_band, active_band) -> bool:
    if CFG.DRONERFA_CHANNEL_MODE == "both":
        return True
    if drone_type == "Background" or t_code == "T0000":
        return True
    if t_code in DRONERFA_915_24:
        return True
    if not active_band:
        return True
    return channel_band == active_band


def normalize_iq_batch(batch_complex: np.ndarray) -> np.ndarray:
    batch_complex = np.nan_to_num(batch_complex, nan=0.0, posinf=0.0,
                                  neginf=0.0).astype(np.complex64)
    iq = np.stack([batch_complex.real, batch_complex.imag], axis=1).astype(np.float32)

    if CFG.NORM_METHOD == "robust_std":
        if CFG.REMOVE_DC:
            iq = iq - np.median(iq, axis=2, keepdims=True)
        mad = np.median(np.abs(iq), axis=(1, 2), keepdims=True)
        scale = 1.4826 * np.maximum(mad, 1e-10)
        out = np.where(mad < 1e-10, 0.0, iq / scale)
        out = np.clip(out, -CFG.NORM_CLIP, CFG.NORM_CLIP)

    elif CFG.NORM_METHOD == "unit_power":
        p = np.mean(iq ** 2, axis=(1, 2), keepdims=True) * 2.0
        out = np.where(p < 1e-10, 0.0, iq / np.sqrt(np.maximum(p, 1e-10)))
        out = np.clip(out, -CFG.NORM_CLIP * 4, CFG.NORM_CLIP * 4)
    else:
        raise ValueError(f"未知 NORM_METHOD: {CFG.NORM_METHOD}")

    return out.astype(np.float32)


def compute_power(batch_complex: np.ndarray, iq_norm: np.ndarray):
    raw = np.mean(batch_complex.real ** 2 + batch_complex.imag ** 2, axis=1).astype(np.float32)
    nrm = np.mean(iq_norm[:, 0, :] ** 2 + iq_norm[:, 1, :] ** 2, axis=1).astype(np.float32)
    return raw, nrm


class H5Vec:

    def __init__(self, ds: h5py.Dataset):
        self.ds = ds
        if ds.ndim == 1:
            self.mode, self.n = 0, ds.shape[0]
        elif ds.ndim == 2:
            if ds.shape[0] == 1:
                self.mode, self.n = 1, ds.shape[1]
            elif ds.shape[1] == 1:
                self.mode, self.n = 2, ds.shape[0]
            elif ds.shape[0] >= ds.shape[1]:
                self.mode, self.n = 2, ds.shape[0]
            else:
                self.mode, self.n = 1, ds.shape[1]
        else:
            raise ValueError(f"不支持的维度: {ds.shape}")

    def read(self, s: int, e: int) -> np.ndarray:
        e = min(e, self.n)
        if e <= s:
            return np.empty(0, np.float32)
        if self.mode == 0:
            a = self.ds[s:e]
        elif self.mode == 1:
            a = self.ds[0, s:e]
        else:
            a = self.ds[s:e, 0]
        return np.asarray(a, dtype=np.float32)


class IQSource:
    n: int = 0

    def read_iq(self, s: int, e: int) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


class H5IQSource(IQSource):
    def __init__(self, i_ds, q_ds):
        self.i, self.q = H5Vec(i_ds), H5Vec(q_ds)
        self.n = min(self.i.n, self.q.n)

    def read_iq(self, s, e):
        e = min(e, self.n)
        if e <= s:
            return np.empty(0, np.complex64)
        a, b = self.i.read(s, e), self.q.read(s, e)
        L = min(a.size, b.size)
        out = np.empty(L, np.complex64)
        out.real = a[:L]
        out.imag = b[:L]
        return out


class MemmapIQSource(IQSource):

    def __init__(self, path, dtype, scale=1.0):
        self.mm = np.memmap(path, dtype=dtype, mode="r")
        self.n = int(self.mm.size // 2)
        self.scale = float(scale)

    def read_iq(self, s, e):
        e = min(e, self.n)
        if e <= s:
            return np.empty(0, np.complex64)
        seg = np.asarray(self.mm[2 * s:2 * e], dtype=np.float32)
        if self.scale != 1.0:
            seg = seg / self.scale
        L = seg.size // 2
        out = np.empty(L, np.complex64)
        out.real = seg[0:2 * L:2]
        out.imag = seg[1:2 * L:2]
        return out

    def close(self):
        try:
            del self.mm
        except Exception:
            pass


class ArrayIQSource(IQSource):
    def __init__(self, arr: np.ndarray):
        self.a = np.asarray(arr, dtype=np.complex64).reshape(-1)
        self.n = self.a.size

    def read_iq(self, s, e):
        return self.a[s:min(e, self.n)]


def make_window_starts(total_len: int, target_len: int) -> np.ndarray:
    if total_len <= 0:
        return np.empty(0, np.int64)
    if total_len < target_len:
        return np.zeros(1, np.int64)
    stride = window_stride(target_len)
    return np.arange(0, total_len - target_len + 1, stride, dtype=np.int64)


def uniform_cap(starts: np.ndarray, cap: Optional[int]) -> np.ndarray:
    if cap is None or len(starts) <= cap:
        return starts
    idx = np.linspace(0, len(starts) - 1, cap).astype(np.int64)
    return starts[idx]


def exact_window_power(src: IQSource, N: int, target_len: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    stride = window_stride(target_len)
    k = target_len // stride
    n_cells = N // stride
    if n_cells < k:
        return np.empty(0, np.int64), np.empty(0, np.float64)

    cell_pw = np.zeros(n_cells, np.float64)
    per_read = max(1, CFG.IO_CHUNK_SAMPLES // stride)

    for c0 in range(0, n_cells, per_read):
        c1 = min(n_cells, c0 + per_read)
        z = src.read_iq(c0 * stride, c1 * stride)
        L = (z.size // stride) * stride
        if L <= 0:
            break
        r = z.real[:L]
        i = z.imag[:L]
        p = (r.astype(np.float64) ** 2 + i.astype(np.float64) ** 2)
        p = p.reshape(-1, stride).mean(axis=1)
        cell_pw[c0:c0 + p.size] = p
        del z, r, i, p

    cs = np.concatenate(([0.0], np.cumsum(cell_pw)))
    win_pw = (cs[k:] - cs[:-k]) / float(k)
    starts = np.arange(win_pw.size, dtype=np.int64) * stride
    ok = (starts + target_len) <= N
    return starts[ok], win_pw[ok]


def probe_power(src: IQSource, starts: np.ndarray, probe_len: int) -> np.ndarray:
    pw = np.zeros(len(starts), np.float64)
    for k, s in enumerate(starts):
        s = int(s)
        z = src.read_iq(s, min(s + probe_len, src.n))
        if z.size:
            r = z.real.astype(np.float64)
            i = z.imag.astype(np.float64)
            pw[k] = float(np.mean(r * r + i * i))
    return pw


def select_starts_by_power(starts: np.ndarray, pw: np.ndarray,
                           rng: np.random.Generator,
                           max_windows: Optional[int]):
    n = len(starts)
    if n == 0:
        return starts, np.empty(0, np.float32), np.empty(0, np.int8)

    valid = pw > 0
    if not valid.any():
        sel = uniform_cap(starts, _cap_or(CFG.NOISE_ONLY_KEEP, max_windows))
        return (sel,
                np.zeros(len(sel), np.float32),
                np.zeros(len(sel), np.int8))

    q_lo = max(float(np.quantile(pw[valid], CFG.NOISE_FLOOR_Q)), 1e-30)
    q_hi = max(float(np.quantile(pw[valid], CFG.PEAK_Q)), 1e-30)
    dyn_db = 10.0 * np.log10(q_hi / q_lo)

    snr = (10.0 * np.log10(np.maximum(pw, 1e-30) / q_lo)).astype(np.float32)

    if dyn_db < CFG.MIN_DYNAMIC_RANGE_DB:
        sel = uniform_cap(starts, _cap_or(CFG.FLAT_CHANNEL_KEEP, max_windows))
        idx = np.searchsorted(starts, sel)
        return sel, snr[idx], np.zeros(len(sel), np.int8)

    active = snr >= CFG.ACTIVE_SNR_TH_DB
    pos = starts[active]
    neg_pool = starts[~active]

    if len(pos) == 0:
        sel = uniform_cap(starts, _cap_or(CFG.NOISE_ONLY_KEEP, max_windows))
    else:
        n_neg = int(len(pos) * CFG.NEG_RATIO)
        if n_neg > 0 and len(neg_pool) > 0:
            neg = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False)
            sel = np.sort(np.concatenate([pos, neg]))
        else:
            sel = pos
        sel = uniform_cap(sel, max_windows)

    idx = np.searchsorted(starts, sel)
    return sel, snr[idx], active[idx].astype(np.int8)


def group_starts_into_chunks(sel, target_len, chunk_samples, N):
    i, m = 0, len(sel)
    while i < m:
        cs = int(sel[i])
        ce = min(N, cs + max(chunk_samples, target_len))
        j = i
        while j < m and int(sel[j]) + target_len <= ce:
            j += 1
        if j == i:
            ce = min(N, cs + target_len)
            j = i + 1
        yield cs, ce, sel[i:j]
        i = j


def iter_windows_from_source(src: IQSource,
                             meta: DomainMeta,
                             target_len: int,
                             dry: bool,
                             do_active_scan: bool,
                             max_windows: Optional[int],
                             seed_key: str) -> Iterator[WinItem]:
    N = src.n
    if N <= 0:
        return

    rng = np.random.default_rng(int(stable_unit_frac(seed_key) * (2 ** 31)))

    starts = make_window_starts(N, target_len)
    if len(starts) == 0:
        return

    use_scan = (do_active_scan and CFG.ACTIVE_SCAN
                and CFG.PRESCAN_MODE != "off"
                and N >= target_len and len(starts) > 4)

    if use_scan:
        if CFG.PRESCAN_MODE == "exact":
            starts, pw = exact_window_power(src, N, target_len)
            if starts.size == 0:
                return
        else:
            starts = uniform_cap(starts, CFG.PRESCAN_MAX_CANDIDATES)
            pw = probe_power(src, starts, CFG.ACTIVE_PROBE_LEN)
        sel, sel_snr, sel_act = select_starts_by_power(starts, pw, rng, max_windows)
    else:
        sel = uniform_cap(starts, max_windows)
        sel_snr = np.full(len(sel), -999.0, np.float32)
        sel_act = np.ones(len(sel), np.int8)

    if len(sel) == 0:
        return

    snr_of = {int(s): float(sel_snr[k]) for k, s in enumerate(sel)}
    act_of = {int(s): int(sel_act[k]) for k, s in enumerate(sel)}

    if dry:
        for s in sel:
            s = int(s)
            vl = min(target_len, N - s)
            yield WinItem(None, meta, s, vl, N, snr_of[s], act_of[s], target_len)
        return

    for cs, ce, group in group_starts_into_chunks(sel, target_len,
                                                  CFG.IO_CHUNK_SAMPLES, N):
        buf = src.read_iq(cs, ce)
        blen = buf.size
        for s in group:
            s = int(s)
            ls = s - cs
            le = ls + target_len
            if le <= blen:
                w = buf[ls:le].astype(np.complex64, copy=True)
                vl = target_len
            else:
                w = np.zeros(target_len, np.complex64)
                seg = buf[ls:blen]
                w[:seg.size] = seg
                vl = int(seg.size)
            if vl <= 0:
                continue
            yield WinItem(w, meta, s, vl, N, snr_of[s], act_of[s], target_len)
        del buf
    gc.collect()


def _basename_lower(k: str) -> str:
    return k.split("/")[-1].lower()


def _pair_iq_keys(keys: List[str]) -> List[Tuple[str, str, str]]:
    bm = {_basename_lower(k): k for k in keys}
    cands = [
        ("rf0_i", "rf0_q", "RF0"), ("rf1_i", "rf1_q", "RF1"),
        ("rf0i", "rf0q", "RF0"), ("rf1i", "rf1q", "RF1"),
        ("ch0_i", "ch0_q", "RF0"), ("ch1_i", "ch1_q", "RF1"),
        ("i", "q", ""), ("i_data", "q_data", ""), ("data_i", "data_q", ""),
    ]
    return [(bm[i], bm[q], t) for i, q, t in cands if i in bm and q in bm]


def _try_2d_iq(arr: np.ndarray, tag: str) -> List[Tuple[np.ndarray, str]]:
    out = []
    if (np.issubdtype(arr.dtype, np.number) and arr.ndim == 2
            and min(arr.shape) in (2, 4) and arr.size > 200):
        a = (arr if arr.shape[0] <= arr.shape[1] else arr.T).astype(np.float32)
        if a.shape[0] == 2:
            out.append(((a[0] + 1j * a[1]).astype(np.complex64), tag))
        elif a.shape[0] >= 4:
            out.append(((a[0] + 1j * a[1]).astype(np.complex64), "RF0"))
            out.append(((a[2] + 1j * a[3]).astype(np.complex64), "RF1"))
    return out


def load_small_mat_streams(path: str) -> List[Tuple[np.ndarray, str]]:
    out: List[Tuple[np.ndarray, str]] = []
    try:
        f = h5py.File(path, "r")
        is_h5 = True
    except OSError:
        is_h5 = False

    if is_h5:
        try:
            dsets: Dict[str, h5py.Dataset] = {}
            f.visititems(lambda n, o: dsets.__setitem__(n, o)
                         if isinstance(o, h5py.Dataset) else None)

            pairs = _pair_iq_keys(list(dsets.keys()))
            if pairs:
                for ik, qk, tag in pairs:
                    a = np.squeeze(dsets[ik][()]).astype(np.float32).reshape(-1)
                    b = np.squeeze(dsets[qk][()]).astype(np.float32).reshape(-1)
                    L = min(a.size, b.size)
                    if L > 100:
                        out.append(((a[:L] + 1j * b[:L]).astype(np.complex64), tag))
                return out

            for k in sorted(dsets.keys()):
                try:
                    arr = np.squeeze(dsets[k][()])
                except Exception:
                    continue
                tag = k.split("/")[-1]
                if arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
                    arr = np.asarray(arr["real"]) + 1j * np.asarray(arr["imag"])
                if np.iscomplexobj(arr) and arr.size > 100:
                    out.append((arr.astype(np.complex64).reshape(-1), tag))
                    continue
                out.extend(_try_2d_iq(arr, tag))
        finally:
            f.close()
        return out

    from scipy.io import loadmat
    mat = loadmat(path)
    for k in sorted(kk for kk in mat.keys() if not kk.startswith("__")):
        arr = np.squeeze(np.asarray(mat[k]))
        if np.iscomplexobj(arr) and arr.size > 100:
            out.append((arr.astype(np.complex64).reshape(-1), k))
            continue
        out.extend(_try_2d_iq(arr, k))
    return out


def build_units() -> List[Unit]:
    units: List[Unit] = []

    if CFG.DO_DRONERFA:
        if CFG.DRONERFA_DIR and os.path.isdir(CFG.DRONERFA_DIR):
            fs = sorted(glob.glob(os.path.join(CFG.DRONERFA_DIR, "**", "*.mat"),
                                  recursive=True))
            if CFG.MAX_FILES_PER_DATASET:
                fs = fs[:CFG.MAX_FILES_PER_DATASET]
            for fp in fs:
                rel = os.path.relpath(fp, CFG.DRONERFA_DIR)
                units.append(Unit("DroneRFa", f"dronerfa::{rel}", fp, {"rel": rel}))
        else:
            print(f"[警告] DroneRFa 目录不存在: {CFG.DRONERFA_DIR}")


    if CFG.DO_RF_UAVS:
        if CFG.RF_UAVS_DIR and os.path.isdir(CFG.RF_UAVS_DIR):
            fs = sorted(glob.glob(os.path.join(CFG.RF_UAVS_DIR, "**", "*.mat"),
                                  recursive=True))
            st = max(1, int(CFG.RF_UAVS_FILE_STRIDE))
            if st > 1:
                fs = fs[::st]
            if CFG.MAX_FILES_PER_DATASET:
                fs = fs[:CFG.MAX_FILES_PER_DATASET]
            rels = [os.path.relpath(fp, CFG.RF_UAVS_DIR) for fp in fs]

            g = max(1, int(CFG.RF_UAVS_FILES_PER_UNIT))
            n_groups = 0
            for i in range(0, len(rels), g):
                chunk = rels[i:i + g]
                uid = f"rfuav_grp::{i:08d}::{short_hash('|'.join(chunk))}"
                units.append(Unit("RF_UAVs", uid, "", {"rels": chunk}))
                n_groups += 1
            if rels:
                print(f"[RF_UAVs] {len(rels)} 个文件 -> {n_groups} "
                      f"个处理单位 (每单位 {g} 文件)")
        else:
            print(f"[警告] RF_UAVs 目录不存在: {CFG.RF_UAVS_DIR}")

    if CFG.DO_GENT_WIFI:
        if CFG.GENT_WIFI_DIR and os.path.isdir(CFG.GENT_WIFI_DIR):
            fs: List[str] = []
            for ext in ("*.bin", "*.dat", "*.iq"):
                fs += glob.glob(os.path.join(CFG.GENT_WIFI_DIR, "**", ext),
                                recursive=True)
            fs = sorted(set(fs))
            if CFG.MAX_FILES_PER_DATASET:
                fs = fs[:CFG.MAX_FILES_PER_DATASET]
            for fp in fs:
                rel = os.path.relpath(fp, CFG.GENT_WIFI_DIR)
                units.append(Unit("Gent_WiFi", f"gent::{rel}", fp, {"rel": rel}))
        else:
            print(f"[警告] Gent 目录不存在: {CFG.GENT_WIFI_DIR}")

    if CFG.DO_RADIOML18:
        if CFG.RADIOML18_H5 and os.path.isfile(CFG.RADIOML18_H5):
            with h5py.File(CFG.RADIOML18_H5, "r") as f:
                N = int(f["X"].shape[0])
            if CFG.RADIOML18_DEBUG_CAP:
                N = min(N, int(CFG.RADIOML18_DEBUG_CAP))
            step = int(CFG.RADIOML18_UNIT_FRAMES)
            blocks = list(range(0, N, step))
            if CFG.MAX_FILES_PER_DATASET:
                blocks = blocks[:CFG.MAX_FILES_PER_DATASET]
            for a in blocks:
                b = min(N, a + step)
                units.append(Unit("RadioML_2018", f"rml18::{a}-{b}",
                                  CFG.RADIOML18_H5, {"a": a, "b": b}))
        else:
            print(f"[警告] RadioML18 文件不存在: {CFG.RADIOML18_H5}")

    return units


def iter_unit_dronerfa(unit: Unit, dry: bool) -> Iterator[WinItem]:
    fp, rel = unit.path, unit.extra["rel"]
    fname = os.path.basename(fp)
    low = fp.lower()

    t_code, d_code, s_code, s_val = parse_tds_from_filename(fname)
    drone = DRONERFA_T_NAME_MAP.get(t_code, t_code or "Unknown")
    is_bg = (t_code == "T0000") or (drone == "Background")
    if is_bg:
        drone, scene = "Background", "urban_background"
    elif "indoor" in low:
        scene = "indoor"
    elif "outdoor" in low:
        scene = "outdoor"
    else:
        scene = "unknown"

    active_band, ab_conf = infer_dronerfa_active_band(t_code, s_val)
    target_len = CFG.WINDOW_LEN_HIGH

    with h5py.File(fp, "r") as f:
        dsets: Dict[str, h5py.Dataset] = {}
        f.visititems(lambda n, o: dsets.__setitem__(n, o)
                     if isinstance(o, h5py.Dataset) else None)
        pairs = _pair_iq_keys(list(dsets.keys()))
        if not pairs:
            return

        for ik, qk, channel in pairs:
            src = H5IQSource(dsets[ik], dsets[qk])
            if src.n <= target_len:
                continue

            fc = dronerfa_channel_freq(t_code, channel)
            band = band_from_fc(fc)
            if not should_keep_dronerfa_channel(t_code, drone, band, active_band):
                continue

            meta = DomainMeta(
                dataset="DroneRFa",
                sampling_rate=CFG.DEFAULT_FS_DRONERFA,
                center_freq=fc, gain_db=0.0,
                rate_class=get_rate_class(CFG.DEFAULT_FS_DRONERFA),
                scene=scene, band=band, device="USRP-2955",
                drone_type=drone, distance=d_code,
                segment_id=f"{s_code}_{channel}",
                modulation="", snr_db=None, source_file=rel, channel=channel,
                t_code=t_code, d_code=d_code, s_code=s_code,
                active_band=active_band, active_band_conf=ab_conf,
                fc_source="fixed",
                group_key=f"DroneRFa|{t_code}|{d_code}|{s_code}",
            )

            yield from iter_windows_from_source(
                src, meta, target_len, dry,
                do_active_scan=True,
                max_windows=CFG.MAX_WINDOWS_PER_STREAM,
                seed_key=f"{unit.unit_id}|{channel}",
            )


def iter_unit_rf_uavs(unit: Unit, dry: bool) -> Iterator[WinItem]:
    rels = unit.extra.get("rels") or [unit.extra.get("rel", "")]
    fs = CFG.DEFAULT_FS_RF_UAVS
    rc = get_rate_class(fs)
    target_len = get_target_len(rc)
    band = band_from_fc(CFG.FC_RF_UAVS)

    for rel in rels:
        if not rel:
            continue
        fp = os.path.join(CFG.RF_UAVS_DIR, rel)
        if not os.path.isfile(fp):
            continue

        low = fp.lower()
        parent = os.path.basename(os.path.dirname(fp))


        m = re.search(r"(?<!\d)(\d{1,3})\s*m(?![a-z0-9])", low)
        distance = f"{m.group(1)}m" if m else ""
        scene = "indoor" if distance == "1m" else ("outdoor" if distance else "unknown")

        m = re.match(r"(\d+)", parent)
        drone = RF_UAVS_DRONE_MAP.get(m.group(1), parent) if m else parent

        try:
            streams = load_small_mat_streams(fp)
        except Exception as e:

            print(f"  [RF_UAVs 跳过] {rel}: {type(e).__name__}: {e}", flush=True)
            continue

        for arr, tag in streams:
            meta = DomainMeta(
                dataset="RF_UAVs", sampling_rate=fs, center_freq=CFG.FC_RF_UAVS,
                rate_class=rc, scene=scene, band=band,
                device="USRP-2955", drone_type=drone, distance=distance,
                segment_id=tag, source_file=rel, channel=tag,
                active_band="5.8G", active_band_conf="doc", fc_source="fixed",
                group_key=f"RF_UAVs|{rel}",
            )
            yield from iter_windows_from_source(
                ArrayIQSource(arr), meta, target_len, dry,
                do_active_scan=CFG.RF_UAVS_ACTIVE_SCAN,
                max_windows=None,
                seed_key=f"{unit.unit_id}|{rel}|{tag}",
            )
            del arr
        del streams


def _parse_gent_fc(name_lower: str, full_lower: str) -> Tuple[float, str]:
    m = re.search(r"(?:^|[_\-])(?:fc|cf|f)[_\-]?(\d+(?:\.\d+)?)\s*(ghz|mhz|khz)",
                  name_lower)
    if m:
        v, u = float(m.group(1)), m.group(2)
        return v * {"ghz": 1e9, "mhz": 1e6, "khz": 1e3}[u], "parsed"

    for m in re.finditer(r"(\w{0,4})[_\-]?(\d+(?:\.\d+)?)\s*(ghz|mhz)", name_lower):
        pre = m.group(1)
        if any(k in pre for k in ("bw", "sr", "fs", "rate", "sp")):
            continue
        v, u = float(m.group(2)), m.group(3)
        fc = v * (1e9 if u == "ghz" else 1e6)
        if fc >= 100e6:
            return fc, "parsed"

    for key, val in (("dvb", 650e6), ("lte", 1.85e9),
                     ("wifi", 2.437e9), ("wi-fi", 2.437e9)):
        if key in full_lower:
            return val, "parsed"

    return CFG.FC_GENT, "default"


def iter_unit_gent(unit: Unit, dry: bool) -> Iterator[WinItem]:
    fp, rel = unit.path, unit.extra["rel"]
    name = os.path.basename(fp).lower()
    full = fp.lower()


    fs = CFG.DEFAULT_FS_GENT
    m = re.search(r"(\d+(?:\.\d+)?)\s*msps", name)
    if m:
        fs = float(m.group(1)) * 1e6
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*ksps", name)
        if m:
            fs = float(m.group(1)) * 1e3

    gain = 0.0
    m = re.search(r"g(?:ain)?[_-]?(\d+)", name)
    if m:
        gain = float(m.group(1))

    fc, fc_src = _parse_gent_fc(name, full)

    tech = ""
    for k, v in [("dvb", "DVB-T"), ("lte", "LTE"), ("wifi", "WiFi"), ("wi-fi", "WiFi")]:
        if k in name or k in full:
            tech = v
            break

    dtype = np.float32 if CFG.GENT_BIN_DTYPE == "float32" else np.int16
    scale = 1.0 if CFG.GENT_BIN_DTYPE == "float32" else CFG.GENT_INT16_SCALE
    src = MemmapIQSource(fp, dtype, scale)
    if src.n < 200:
        src.close()
        return

    rc = get_rate_class(fs)
    meta = DomainMeta(
        dataset="Gent_WiFi", sampling_rate=fs, center_freq=fc, gain_db=gain,
        rate_class=rc, scene="urban_background", band=band_from_fc(fc),
        device="SDR", drone_type="Background", distance="", segment_id="",
        modulation=tech, source_file=rel, channel="",
        active_band=band_from_fc(fc), active_band_conf="guess", fc_source=fc_src,
        group_key=f"Gent_WiFi|{rel}",
    )
    try:
        yield from iter_windows_from_source(
            src, meta, get_target_len(rc), dry,
            do_active_scan=CFG.GENT_ACTIVE_SCAN,
            max_windows=CFG.GENT_MAX_WINDOWS_PER_FILE,
            seed_key=unit.unit_id,
        )
    finally:
        src.close()


def iter_unit_radioml(unit: Unit, dry: bool) -> Iterator[WinItem]:
    a, b = int(unit.extra["a"]), int(unit.extra["b"])
    stride = max(1, CFG.RADIOML18_STRIDE)
    target_len = CFG.WINDOW_LEN_LOW
    blk = CFG.RADIOML18_CHUNK * stride

    with h5py.File(unit.path, "r") as f:
        X, Y, Z = f["X"], f["Y"], f["Z"]
        for bs in range(a, b, blk):
            be = min(b, bs + blk)

            if dry:
                cnt = len(range(0, be - bs, stride))
                meta = DomainMeta(
                    dataset="RadioML_2018", sampling_rate=CFG.DEFAULT_FS_RML18,
                    center_freq=CFG.FC_BASEBAND, rate_class="low",
                    scene="lab_simulated", band="baseband", device="simulated",
                    source_file="rml18", active_band="baseband",
                    active_band_conf="doc", fc_source="fixed",
                    group_key=f"RadioML_2018|{bs}",
                )
                for _ in range(cnt):
                    yield WinItem(None, meta, 0, target_len, target_len,
                                  -999.0, 1, target_len)
                continue

            xb = X[bs:be][::stride]
            yb = Y[bs:be][::stride]
            zb = Z[bs:be][::stride]
            for j in range(xb.shape[0]):
                gidx = bs + j * stride
                mi = int(np.argmax(yb[j]))
                mod = RML18_CLASSES[mi] if mi < len(RML18_CLASSES) else f"mod_{mi}"
                meta = DomainMeta(
                    dataset="RadioML_2018", sampling_rate=CFG.DEFAULT_FS_RML18,
                    center_freq=CFG.FC_BASEBAND, rate_class="low",
                    scene="lab_simulated", band="baseband", device="simulated",
                    drone_type="", distance="", segment_id=str(gidx),
                    modulation=mod, snr_db=int(zb[j][0]),
                    source_file=f"rml18::{gidx}", channel="",
                    active_band="baseband", active_band_conf="doc",
                    fc_source="fixed",


                    group_key=f"RadioML_2018|{gidx}",
                )
                s = xb[j]
                iq = np.zeros(target_len, np.complex64)
                n = min(target_len, s.shape[0])
                iq.real[:n] = s[:n, 0]
                iq.imag[:n] = s[:n, 1]
                yield WinItem(iq, meta, 0, n, int(s.shape[0]), -999.0, 1, target_len)
            del xb, yb, zb
            gc.collect()


def iter_unit_windows(unit: Unit, dry: bool) -> Iterator[WinItem]:
    if unit.dataset == "DroneRFa":
        yield from iter_unit_dronerfa(unit, dry)
    elif unit.dataset == "RF_UAVs":
        yield from iter_unit_rf_uavs(unit, dry)
    elif unit.dataset == "Gent_WiFi":
        yield from iter_unit_gent(unit, dry)
    elif unit.dataset == "RadioML_2018":
        yield from iter_unit_radioml(unit, dry)
    else:
        raise ValueError(unit.dataset)


def meta_base_dict(meta: DomainMeta) -> Dict[str, Any]:
    d = getattr(meta, "_cached_dict", None)
    if d is None:
        d = asdict(meta)
        d.update(encode_domain_ids(meta))
        d["split"] = assign_split(meta.group_key)
        for k in STR_FIELDS:
            if k in d:
                d[k] = safe_str(d[k])
        object.__setattr__(meta, "_cached_dict", d)
    return d


def make_record(item: WinItem, window_idx: int, raw_power: float,
                norm_power: float, unit_id: str) -> Dict[str, Any]:
    m = meta_base_dict(item.meta).copy()
    base = m.get("segment_id", "")
    m["segment_id"] = f"{base}_w{window_idx}" if base else f"w{window_idx}"
    m["unit_id"] = unit_id
    m["window_idx"] = int(window_idx)
    m["start_sample"] = int(item.start)
    m["end_sample"] = int(item.start + item.valid_len)
    m["valid_length"] = int(item.valid_len)
    m["original_length"] = int(item.orig_len)
    m["window_len"] = int(item.window_len)
    m["raw_power"] = float(raw_power)
    m["norm_power"] = float(norm_power)
    m["est_snr_db"] = float(item.est_snr_db)
    m["is_active"] = int(item.is_active)
    m["snr_db"] = -999 if item.meta.snr_db is None else int(item.meta.snr_db)
    m["shard"] = ""
    m["local_idx"] = -1
    return m


class UnitWriter:
    def __init__(self, unit: Unit, out_root: str, worker_tag: str):
        self.unit = unit
        self.uhash = short_hash(unit.unit_id)
        self.out_root = out_root
        self.shards_dir = os.path.join(out_root, "shards")
        self.index_dir = os.path.join(out_root, "index")
        self.staging = os.path.join(out_root, "_staging", worker_tag, self.uhash)
        if os.path.isdir(self.staging):
            shutil.rmtree(self.staging, ignore_errors=True)
        os.makedirs(self.staging, exist_ok=True)

        self.buffers: Dict[str, Dict[str, list]] = {}
        self.seq = 0
        self.records: List[Dict[str, Any]] = []
        self.staged: List[str] = []
        self.bytes_written = 0

    @staticmethod
    def _key(rec) -> str:
        return f"{rec['rate_class']}_fs{int(rec['sampling_rate'])}_len{int(rec['window_len'])}"

    def add(self, iq: np.ndarray, rec: Dict[str, Any]):
        k = self._key(rec)
        if k not in self.buffers:
            self.buffers[k] = {"iq": [], "rec": []}

        self.buffers[k]["iq"].append(np.ascontiguousarray(iq, dtype=store_dtype()))
        self.buffers[k]["rec"].append(rec)
        if len(self.buffers[k]["rec"]) >= get_shard_size(rec["rate_class"]):
            self.flush_key(k)

    def flush_key(self, k: str):
        buf = self.buffers.get(k)
        if not buf or not buf["rec"]:
            return
        name = f"shard_{k}_{self.uhash}_{self.seq:03d}.h5"
        path = os.path.join(self.staging, name)
        arr = np.stack(buf["iq"], axis=0)
        n, L = arr.shape[0], arr.shape[-1]

        with h5py.File(path, "w") as hf:
            hf.attrs["rate_key"] = k
            hf.attrs["num_samples"] = n
            hf.attrs["store_dtype"] = CFG.STORE_DTYPE
            hf.attrs["unit_id"] = self.unit.unit_id
            hf.attrs["norm_method"] = CFG.NORM_METHOD
            hf.attrs["norm_clip"] = float(CFG.NORM_CLIP)
            hf.attrs["remove_dc"] = int(bool(CFG.REMOVE_DC))

            hf.create_dataset("iq", data=arr, chunks=(1, 2, L), **h5_kwargs())
            for fld in ("dataset_id", "band_id", "scene_id", "device_id", "rate_id",
                        "sampling_rate", "center_freq", "gain_db", "start_sample",
                        "end_sample", "valid_length", "window_len", "raw_power",
                        "norm_power", "est_snr_db", "is_active", "snr_db", "window_idx"):
                hf.create_dataset(fld, data=np.asarray(
                    [r.get(fld, -999) if r.get(fld) is not None else -999
                     for r in buf["rec"]]))
            hf.flush()
        fsync_file(path)

        for i, r in enumerate(buf["rec"]):
            r["shard"] = name
            r["local_idx"] = i
            self.records.append(r)

        self.staged.append(name)
        self.bytes_written += os.path.getsize(path)
        self.seq += 1
        self.buffers[k] = {"iq": [], "rec": []}
        del arr
        gc.collect()

    def commit(self) -> Dict[str, Any]:
        for k in list(self.buffers.keys()):
            self.flush_key(k)

        if not self.records:
            shutil.rmtree(self.staging, ignore_errors=True)
            return {"unit_id": self.unit.unit_id, "status": "empty",
                    "n_windows": 0, "bytes": 0, "shards": [], "index_part": ""}

        os.makedirs(self.shards_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)


        for name in self.staged:
            os.replace(os.path.join(self.staging, name),
                       os.path.join(self.shards_dir, name))
        fsync_dir(self.shards_dir)


        part = f"part_{self.uhash}.parquet"
        ppath = os.path.join(self.index_dir, part)
        tmp = ppath + ".tmp"
        df = pd.DataFrame(self.records)
        for f in INDEX_FIELDS:
            if f not in df.columns:
                df[f] = None
        df = df[INDEX_FIELDS]
        tbl = pa.Table.from_pandas(df, schema=INDEX_SCHEMA, preserve_index=False)
        pq.write_table(tbl, tmp, compression="snappy")
        fsync_file(tmp)
        os.replace(tmp, ppath)
        fsync_dir(self.index_dir)

        self.bytes_written += os.path.getsize(ppath)

        shutil.rmtree(self.staging, ignore_errors=True)
        res = {"unit_id": self.unit.unit_id, "status": "ok",
               "n_windows": len(self.records), "bytes": int(self.bytes_written),
               "shards": self.staged, "index_part": part}
        self.records = []
        del df, tbl
        gc.collect()
        return res

    def abort(self):
        shutil.rmtree(self.staging, ignore_errors=True)


class Batcher:
    def __init__(self, writer: Optional[UnitWriter], unit_id: str):
        self.writer = writer
        self.unit_id = unit_id
        self.items: List[WinItem] = []
        self.cur_len = None
        self.counter = 0
        self.dry_stats: Dict[str, Dict[str, int]] = {}

    def _dry_count(self, item: WinItem):
        rc = item.meta.rate_class
        d = self.dry_stats.setdefault(rc, {"n": 0, "bytes": 0, "active": 0})
        d["n"] += 1
        d["bytes"] += 2 * item.window_len * store_itemsize()
        d["active"] += int(item.is_active)

    def add(self, item: WinItem):
        if item.iq is None:
            self._dry_count(item)
            self.counter += 1
            return

        if CFG.FILTER_ZERO_WINDOWS:
            th2 = CFG.ZERO_ABS_MAX_TH ** 2
            r, i = item.iq.real, item.iq.imag
            if float(np.max(r * r + i * i)) < th2:
                return

        if self.cur_len is not None and item.window_len != self.cur_len:
            self.flush()
        self.cur_len = item.window_len
        self.items.append(item)
        if len(self.items) >= get_batch_size(item.meta.rate_class):
            self.flush()

    def flush(self):
        if not self.items:
            self.cur_len = None
            return
        bc = np.stack([it.iq for it in self.items], axis=0).astype(np.complex64)
        nz = normalize_iq_batch(bc)
        rp, npw = compute_power(bc, nz)
        for i, it in enumerate(self.items):
            rec = make_record(it, self.counter, float(rp[i]), float(npw[i]),
                              self.unit_id)
            if self.writer is not None:
                self.writer.add(nz[i], rec)
            self.counter += 1
        self.items = []
        self.cur_len = None
        del bc, nz, rp, npw
        gc.collect()


class Ledger:
    def __init__(self, out_root: str):
        self.dir = os.path.join(out_root, "_state")
        os.makedirs(self.dir, exist_ok=True)

    def load(self) -> Dict[str, Dict[str, Any]]:
        done: Dict[str, Dict[str, Any]] = {}
        for fp in sorted(glob.glob(os.path.join(self.dir, "done_*.jsonl"))):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("unit_id"):
                        done[rec["unit_id"]] = rec
        return done

    def writer_path(self, tag: str) -> str:
        return os.path.join(self.dir, f"done_{tag}.jsonl")

    @staticmethod
    def append(path: str, rec: Dict[str, Any]):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


class DiskGuard:
    def __init__(self, out_root: str, base_bytes: int, counter=None):
        self.out_root = out_root
        self.base = base_bytes
        self.counter = counter

    def add(self, n: int):
        if self.counter is None:
            self.base += n
        else:
            with self.counter.get_lock():
                self.counter.value += n

    def used(self) -> int:
        return self.base + (self.counter.value if self.counter is not None else 0)

    def check(self) -> Tuple[bool, str]:
        free = shutil.disk_usage(self.out_root).free
        if free < CFG.MIN_FREE_GB * 1024 ** 3:
            return False, f"剩余空间 {human_bytes(free)} < MIN_FREE_GB={CFG.MIN_FREE_GB}GB"
        if self.used() > CFG.DISK_BUDGET_GB * 1024 ** 3:
            return False, (f"已用 {human_bytes(self.used())} > "
                           f"DISK_BUDGET_GB={CFG.DISK_BUDGET_GB}GB")
        return True, ""


def cleanup_orphans(out_root: str, done: Dict[str, Dict[str, Any]]):
    stg = os.path.join(out_root, "_staging")
    if os.path.isdir(stg):
        shutil.rmtree(stg, ignore_errors=True)

    ref_shards, ref_parts = set(), set()
    for rec in done.values():
        ref_shards.update(rec.get("shards", []))
        if rec.get("index_part"):
            ref_parts.add(rec["index_part"])

    n_s = n_p = 0
    sd = os.path.join(out_root, "shards")
    if os.path.isdir(sd):
        for fn in os.listdir(sd):
            if fn.endswith(".h5") and fn not in ref_shards:
                os.remove(os.path.join(sd, fn))
                n_s += 1
    idd = os.path.join(out_root, "index")
    if os.path.isdir(idd):
        for fn in os.listdir(idd):
            if fn.endswith(".tmp"):
                os.remove(os.path.join(idd, fn))
                n_p += 1
            elif fn.endswith(".parquet") and fn not in ref_parts:
                os.remove(os.path.join(idd, fn))
                n_p += 1
    if n_s or n_p:
        print(f"[清理] 删除孤儿 shard {n_s} 个 / 孤儿 index part {n_p} 个（上次崩溃残留）")


def validate_done(out_root: str, done: Dict[str, Dict[str, Any]]
                  ) -> Dict[str, Dict[str, Any]]:
    ok = {}
    for uid, rec in done.items():
        if rec.get("status") == "empty":
            ok[uid] = rec
            continue
        part = rec.get("index_part", "")
        if part and not os.path.isfile(os.path.join(out_root, "index", part)):
            continue
        miss = [s for s in rec.get("shards", [])
                if not os.path.isfile(os.path.join(out_root, "shards", s))]
        if miss:
            continue
        ok[uid] = rec
    if len(ok) != len(done):
        print(f"[校验] {len(done) - len(ok)} 个单位的产物不完整，将重新处理")
    return ok


def reconcile_units(out_root: str, done: Dict[str, Dict[str, Any]],
                    unit_ids: set) -> Dict[str, Dict[str, Any]]:
    stale = [u for u in done if u not in unit_ids]
    if not stale:
        return done

    print(f"[对账] 账本中 {len(stale)} 个条目不在当前任务集合内"
          f"（粒度或源文件已变更），清除中 ...")
    n_file = 0
    for uid in stale:
        rec = done.pop(uid)
        for fn in rec.get("shards", []):
            p = os.path.join(out_root, "shards", fn)
            if os.path.exists(p):
                os.remove(p)
                n_file += 1
        part = rec.get("index_part", "")
        if part:
            p = os.path.join(out_root, "index", part)
            if os.path.exists(p):
                os.remove(p)
                n_file += 1

    sd = os.path.join(out_root, "_state")
    tmp = os.path.join(sd, "done_compact.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in done.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    for old in glob.glob(os.path.join(sd, "done_*.jsonl")):
        os.remove(old)
    os.replace(tmp, os.path.join(sd, "done_compact.jsonl"))
    print(f"[对账] 删除 {n_file} 个文件，账本已压实为 done_compact.jsonl")
    return done


_G: Dict[str, Any] = {}


def cfg_snapshot() -> Dict[str, Any]:
    return {k: v for k, v in CFG.__dict__.items() if k.isupper()}


def apply_cfg(d: Dict[str, Any]):
    for k, v in d.items():
        setattr(CFG, k, v)


def pool_init(cfg_dict, out_root, base_bytes, counter):
    apply_cfg(cfg_dict)
    warnings.filterwarnings("ignore")
    try:
        h5py.get_config().complex_names = ("real", "imag")
    except Exception:
        pass
    _G["out_root"] = out_root
    _G["guard"] = DiskGuard(out_root, base_bytes, counter)
    _G["ledger"] = Ledger(out_root)
    _G["tag"] = f"p{os.getpid()}"
    _G["ledger_path"] = _G["ledger"].writer_path(_G["tag"])
    np.random.seed(CFG.SEED + os.getpid())


def process_unit(unit: Unit) -> Dict[str, Any]:
    t0 = time.time()
    if CFG.DRY_RUN:
        b = Batcher(None, unit.unit_id)
        try:
            for it in iter_unit_windows(unit, dry=True):
                b.add(it)
        except Exception as e:
            return {"unit_id": unit.unit_id, "dataset": unit.dataset,
                    "status": "error", "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1500:]}
        return {"unit_id": unit.unit_id, "dataset": unit.dataset, "status": "dry",
                "dry": b.dry_stats, "sec": time.time() - t0}

    guard: DiskGuard = _G["guard"]
    ok, why = guard.check()
    if not ok:
        return {"unit_id": unit.unit_id, "dataset": unit.dataset,
                "status": "disk_stop", "error": why}

    w = UnitWriter(unit, _G["out_root"], _G["tag"])
    try:
        b = Batcher(w, unit.unit_id)
        for it in iter_unit_windows(unit, dry=False):
            b.add(it)
        b.flush()
        res = w.commit()
    except Exception as e:
        w.abort()
        return {"unit_id": unit.unit_id, "dataset": unit.dataset, "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:]}

    guard.add(res["bytes"])
    res["dataset"] = unit.dataset
    res["sec"] = time.time() - t0
    Ledger.append(_G["ledger_path"], res)
    return res


PASS1_COLS = ["dataset", "rate_class", "band", "scene", "device", "split",
              "active_band_conf", "fc_source", "dataset_id", "band_id",
              "channel", "drone_type", "modulation", "snr_db",
              "group_key", "raw_power", "start_sample", "is_active"]


def plan_split(strat_groups: Dict[str, Dict[str, int]],
               group_maxstart: Dict[str, int]) -> Tuple[set, Dict[str, int]]:
    val_groups: set = set()
    time_thresh: Dict[str, int] = {}

    for strat, gmap in strat_groups.items():
        if len(gmap) >= 2:
            keys = sorted(gmap.keys(), key=lambda k: stable_unit_frac("split|" + k))
            total = sum(gmap.values())
            target = total * CFG.VAL_RATIO
            acc = 0
            for k in keys:
                if acc >= target and acc > 0:
                    break
                val_groups.add(k)
                acc += gmap[k]
        else:
            k = next(iter(gmap))
            n = gmap[k]
            if CFG.SINGLE_GROUP_TIME_SPLIT and n >= CFG.MIN_WINDOWS_FOR_TIME_SPLIT:
                mx = int(group_maxstart.get(k, 0))
                if mx > 0:
                    time_thresh[k] = int(mx * (1.0 - CFG.VAL_RATIO))
    return val_groups, time_thresh


def finalize(out_root: str, elapsed: float = 0.0):
    index_dir = os.path.join(out_root, "index")
    parts = sorted(glob.glob(os.path.join(index_dir, "part_*.parquet")))
    if not parts:
        print("[finalize] 没有找到任何 index part，跳过")
        return

    print(f"\n[finalize] 扫描 {len(parts)} 个 index part ...")
    ds = pads.dataset(parts, format="parquet", schema=INDEX_SCHEMA)


    counts = {k: Counter() for k in
              ["dataset", "rate_class", "band", "scene", "device",
               "active_band_conf", "fc_source"]}
    per_ds_total = Counter()
    strat_groups: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_maxstart: Dict[str, int] = {}
    group_ds: Dict[str, int] = {}
    floor_pool: Dict[Tuple[int, int, str], List[np.ndarray]] = defaultdict(list)
    total = 0
    n_active_stream = 0

    for batch in tqdm(ds.to_batches(columns=PASS1_COLS, batch_size=200_000),
                      desc="[finalize] pass1"):
        df = batch.to_pandas()
        total += len(df)
        for col in counts:
            counts[col].update(df[col].astype(str).values.tolist())
        per_ds_total.update(df["dataset_id"].astype(int).values.tolist())
        n_active_stream += int((df["is_active"].values > 0).sum())

        strat = (df["dataset"].fillna("") + "|" + df["drone_type"].fillna("")
                 + "|" + df["modulation"].fillna("") + "|"
                 + df["snr_db"].astype(int).astype(str))
        gb = df.assign(_strat=strat).groupby(["_strat", "group_key"], sort=False)
        for (st, gk), sub in gb:
            strat_groups[st][gk] += len(sub)
            mx = int(sub["start_sample"].max())
            if mx > group_maxstart.get(gk, -1):
                group_maxstart[gk] = mx
            group_ds[gk] = int(sub["dataset_id"].iloc[0])

        for (did, bid, ch), sub in df.groupby(["dataset_id", "band_id", "channel"],
                                              sort=False):
            floor_pool[(int(did), int(bid), str(ch))].append(
                sub["raw_power"].to_numpy(np.float32))
        del df, batch
    gc.collect()


    floors: Dict[Tuple[int, int, str], float] = {}
    for k, arrs in floor_pool.items():
        v = np.concatenate(arrs)
        v = v[v > 0]
        floors[k] = float(np.quantile(v, CFG.GLOBAL_FLOOR_Q)) if v.size else 1e-20
    del floor_pool
    gc.collect()


    val_groups, time_thresh = plan_split(strat_groups, group_maxstart)


    val_ds_count = Counter()
    for st, gmap in strat_groups.items():
        for gk, c in gmap.items():
            if gk in val_groups:
                val_ds_count[group_ds.get(gk, -1)] += c


    train_ds_count = {d: max(1, per_ds_total[d] - val_ds_count.get(d, 0))
                      for d in per_ds_total}
    train_total = sum(train_ds_count.values())
    mix = {d: CFG.TARGET_MIX.get(DATASET_NAME_OF.get(d, ""), 0.0)
           for d in per_ds_total}
    msum = sum(mix.values()) or 1.0
    wmap = {d: float(min(CFG.MAX_SAMPLE_WEIGHT,
                         (mix[d] / msum) * train_total / train_ds_count[d]))
            for d in per_ds_total}

    print(f"[finalize] val group 数 : {len(val_groups)}   "
          f"时间切分 group 数 : {len(time_thresh)}")
    print(f"[finalize] sample_weight: "
          f"{ {DATASET_NAME_OF.get(d, d): round(w, 4) for d, w in wmap.items()} }")


    merged = os.path.join(out_root, "pretrain_index.parquet")
    tmp = merged + ".tmp"
    wr = pq.ParquetWriter(tmp, FINAL_SCHEMA, compression="snappy")

    split_counter = Counter()
    n_active_global = 0

    for batch in tqdm(ds.to_batches(batch_size=100_000), desc="[finalize] pass2"):
        df = pa.Table.from_batches([batch], schema=INDEX_SCHEMA).to_pandas()

        gk = df["group_key"].fillna("")
        sp = np.where(gk.isin(val_groups).to_numpy(), "val", "train").astype(object)

        if time_thresh:
            th = gk.map(time_thresh).astype("float64").to_numpy()
            has = ~np.isnan(th)
            if has.any():
                ss = df["start_sample"].to_numpy(np.int64)
                wl = df["window_len"].to_numpy(np.int64)
                is_val = has & (ss >= np.nan_to_num(th, nan=np.inf))

                is_excl = has & (~is_val) & ((ss + wl) > np.nan_to_num(th, nan=np.inf))
                sp = np.where(is_val, "val", sp)
                sp = np.where(is_excl, "excl", sp)
        df["split"] = sp
        split_counter.update(pd.Series(sp).value_counts().to_dict())

        keys = list(zip(df["dataset_id"].astype(int),
                        df["band_id"].astype(int),
                        df["channel"].fillna("").astype(str)))
        fl = np.array([floors.get(k, 1e-20) for k in keys], dtype=np.float64)
        rp = np.maximum(df["raw_power"].to_numpy(np.float64), 1e-30)
        snr_g = (10.0 * np.log10(rp / np.maximum(fl, 1e-30))).astype(np.float32)
        act_g = (snr_g >= CFG.ACTIVE_SNR_TH_DB).astype(np.int8)
        n_active_global += int(act_g.sum())

        df["snr_global_db"] = snr_g
        df["is_active_global"] = act_g
        df["sample_weight"] = df["dataset_id"].astype(int).map(wmap)\
                                              .fillna(1.0).astype(np.float32)

        for f in FINAL_SCHEMA.names:
            if f not in df.columns:
                df[f] = None
        df = df[FINAL_SCHEMA.names]
        wr.write_table(pa.Table.from_pandas(df, schema=FINAL_SCHEMA,
                                            preserve_index=False))
        del df, batch
        gc.collect()

    wr.close()
    fsync_file(tmp)
    os.replace(tmp, merged)

    shard_bytes = dir_size_bytes(os.path.join(out_root, "shards"))
    index_bytes = dir_size_bytes(index_dir)
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "unknown"

    manifest = {
        "version": "5.2",
        "git_commit": git,
        "python": sys.version.split()[0],
        "numpy": np.__version__, "h5py": h5py.__version__,
        "pyarrow": pa.__version__, "pandas": pd.__version__,
        "platform": platform.platform(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 1),
        "config": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                       else str(v)) for k, v in cfg_snapshot().items()},
        "id_maps": {"dataset": DATASET_ID_MAP, "band": BAND_ID_MAP,
                    "scene": SCENE_ID_MAP, "device": DEVICE_ID_MAP,
                    "rate": RATE_ID_MAP},
        "num_windows": int(total),
        "num_active_stream": int(n_active_stream),
        "num_active_global": int(n_active_global),
        "active_ratio_global": round(n_active_global / max(total, 1), 4),
        "num_shards": len(glob.glob(os.path.join(out_root, "shards", "*.h5"))),
        "shard_bytes": int(shard_bytes),
        "index_bytes": int(index_bytes),
        "total_size_human": human_bytes(shard_bytes + index_bytes),
        "dataset_counts": {DATASET_NAME_OF.get(d, str(d)): int(c)
                           for d, c in per_ds_total.items()},
        "dataset_sample_weight": {DATASET_NAME_OF.get(d, str(d)): round(w, 6)
                                  for d, w in wmap.items()},
        "split_counts": dict(split_counter),
        "num_val_groups": len(val_groups),
        "num_time_split_groups": len(time_thresh),
        "global_noise_floor": {f"{DATASET_NAME_OF.get(k[0], k[0])}|band{k[1]}|{k[2]}":
                               float(v) for k, v in floors.items()},
        "counts": {k: dict(v) for k, v in counts.items()},
        "notes": {
            "gent_fs": f"{CFG.DEFAULT_FS_GENT:.0f} Hz ({CFG.GENT_FS_SOURCE})",
            "is_active": "逐流相对门限, 平坦通道置 0",
            "is_active_global": "数据集级绝对噪底判定, 推荐训练时用这一列",
            "split_excl": "时间切分保护带内的窗口, 训练与验证都应排除",
            "rf_uavs_packing": f"{CFG.RF_UAVS_FILES_PER_UNIT} 个文件/单位 "
                               f"(stride={CFG.RF_UAVS_FILE_STRIDE})",
        },
    }
    with open(os.path.join(out_root, "preprocess_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[finalize] 合并索引 : {merged}  ({total:,} 行)")
    print(f"[finalize] 落盘体积 : {human_bytes(shard_bytes + index_bytes)}")
    print(f"[finalize] active(global) 占比 : "
          f"{n_active_global / max(total, 1) * 100:.1f}%")
    print(f"[finalize] split    : {dict(split_counter)}")
    for col in ["dataset", "rate_class", "band"]:
        print(f"  {col}: {dict(counts[col])}")


def prepare_dirs(out_root: str):
    for sub in ["", "shards", "index", "_state", "_staging"]:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    legacy = glob.glob(os.path.join(out_root, "shard_*.h5"))
    if legacy:
        raise RuntimeError(
            f"检测到 {len(legacy)} 个旧版 shard 直接躺在 {out_root} 根目录。\n"
            f"v4.x 的产物归一化有误（I/Q 分别 MAD），无法复用。\n"
            f"请 rm -rf 该目录，或把 CFG.OUTPUT_DIR 换成一个全新目录。")


def run_dry(units: List[Unit]):
    print(f"\n>>> DRY-RUN 容量预估（不写任何数据），共 {len(units)} 个单位")
    if CFG.PRESCAN_MODE == "exact":
        print("    [注意] exact 预扫描会完整读一遍源数据，DroneRFa 全量可能要 1~2 小时。")
        print("           想快速估算请用: --limit 20  或  --prescan probe\n")

    agg: Dict[str, Dict[str, Dict[str, int]]] = {}
    errors = []
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    snap = cfg_snapshot()
    nw = max(1, min(CFG.NUM_WORKERS, len(units)))

    with ctx.Pool(nw, initializer=pool_init,
                  initargs=(snap, CFG.OUTPUT_DIR, 0, None),
                  maxtasksperchild=CFG.MAXTASKSPERCHILD) as pool:
        for res in tqdm(pool.imap_unordered(process_unit, units, chunksize=1),
                        total=len(units), desc="[dry-run]"):
            if res.get("status") == "error":
                errors.append(res)
                continue
            d = agg.setdefault(res["dataset"], {})
            for rc, st in res.get("dry", {}).items():
                t = d.setdefault(rc, {"n": 0, "bytes": 0, "active": 0})
                t["n"] += st["n"]
                t["bytes"] += st["bytes"]
                t["active"] += st["active"]

    print("\n" + "=" * 88)
    print(f"{'dataset':<16}{'rate':<8}{'windows':>12}{'active':>12}{'est.size':>16}")
    print("-" * 88)
    grand_n = grand_b = 0
    for dsname in sorted(agg):
        for rc in sorted(agg[dsname]):
            t = agg[dsname][rc]
            grand_n += t["n"]
            grand_b += t["bytes"]
            print(f"{dsname:<16}{rc:<8}{t['n']:>12,}{t['active']:>12,}"
                  f"{human_bytes(t['bytes']):>16}")
    print("-" * 88)
    print(f"{'TOTAL':<24}{grand_n:>12,}{'':>12}{human_bytes(grand_b):>16}")
    print("=" * 88)

    if CFG.MAX_FILES_PER_DATASET:
        print(f"\n[提示] 本次仅抽样了每个数据集的前 {CFG.MAX_FILES_PER_DATASET} 个单位，"
              f"请按实际文件数线性外推。")

    free = shutil.disk_usage(CFG.OUTPUT_DIR).free
    print(f"\n当前可用磁盘 : {human_bytes(free)}")
    print(f"预算上限     : {CFG.DISK_BUDGET_GB} GB")
    if grand_b > CFG.DISK_BUDGET_GB * 1024 ** 3:
        print("\n[!] 预计输出超过预算，建议任选：")
        print(f"    - 降低 CFG.MAX_WINDOWS_PER_STREAM（当前 {CFG.MAX_WINDOWS_PER_STREAM}）")
        print(f"    - 提高 CFG.ACTIVE_SNR_TH_DB（当前 {CFG.ACTIVE_SNR_TH_DB} dB）"
              f"或降低 CFG.NEG_RATIO（当前 {CFG.NEG_RATIO}）")
        print("    - 把 CFG.DRONERFA_CHANNEL_MODE 改成 'active'")
        print(f"    - 调大 CFG.RF_UAVS_FILE_STRIDE（当前 {CFG.RF_UAVS_FILE_STRIDE}）以抽样省盘")
    else:
        print("\n[OK] 预计输出在预算内，可以执行 --mode run")

    if errors:
        print(f"\n[!] {len(errors)} 个单位预扫描报错，前 5 条：")
        for e in errors[:5]:
            print(f"    {e['unit_id']}: {e['error']}")


def run_full(units: List[Unit]):
    ledger = Ledger(CFG.OUTPUT_DIR)
    done = validate_done(CFG.OUTPUT_DIR, ledger.load())


    done = reconcile_units(CFG.OUTPUT_DIR, done, {u.unit_id for u in units})
    cleanup_orphans(CFG.OUTPUT_DIR, done)

    todo = [u for u in units if u.unit_id not in done]
    print(f"\n>>> 总单位 {len(units)}，已完成 {len(done)}，待处理 {len(todo)}")
    if not todo:
        print(">>> 全部已完成，直接进入 finalize")
        finalize(CFG.OUTPUT_DIR)
        return

    base = (dir_size_bytes(os.path.join(CFG.OUTPUT_DIR, "shards"))
            + dir_size_bytes(os.path.join(CFG.OUTPUT_DIR, "index")))
    print(f">>> 已有产物 {human_bytes(base)}，可用磁盘 "
          f"{human_bytes(shutil.disk_usage(CFG.OUTPUT_DIR).free)}")

    order = {"DroneRFa": 0, "Gent_WiFi": 1, "RF_UAVs": 2, "RadioML_2018": 3}
    todo.sort(key=lambda u: order.get(u.dataset, 9))

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    counter = ctx.Value("q", 0)
    snap = cfg_snapshot()
    nw = max(1, min(CFG.NUM_WORKERS, len(todo)))
    t0 = time.time()

    n_ok = n_err = n_empty = 0
    n_win = 0
    stopped = False
    errors = []


    done_by_ds: Counter = Counter()
    total_by_ds: Counter = Counter(u.dataset for u in todo)
    ds_order = ["DroneRFa", "RF_UAVs", "Gent_WiFi", "RadioML_2018"]

    with ctx.Pool(nw, initializer=pool_init,
                  initargs=(snap, CFG.OUTPUT_DIR, base, counter),
                  maxtasksperchild=CFG.MAXTASKSPERCHILD) as pool:
        it = pool.imap_unordered(process_unit, todo, chunksize=1)
        pbar = tqdm(total=len(todo), desc="[preprocess]", smoothing=0.02)
        try:
            for res in it:
                pbar.update(1)
                st = res.get("status")
                dsn = res.get("dataset", "?")

                if st in ("ok", "empty"):
                    done_by_ds[dsn] += 1
                if st == "ok":
                    n_ok += 1
                    n_win += res["n_windows"]
                elif st == "empty":
                    n_empty += 1
                elif st == "disk_stop":
                    stopped = True
                    print(f"\n[!] 磁盘保护触发：{res['error']}")
                    print("[!] 正在优雅停止，已完成的单位均已落盘，"
                          "释放空间后重跑同一命令即可续跑。")
                    pool.terminate()
                    break
                else:
                    n_err += 1
                    errors.append(res)

                prog = " ".join(f"{k[:4]}{done_by_ds[k]}/{total_by_ds[k]}"
                                for k in ds_order if total_by_ds[k])
                pbar.set_postfix_str(
                    f"{prog} win={n_win:,} disk={human_bytes(base + counter.value)}")
        except KeyboardInterrupt:
            print("\n[!] 收到 Ctrl-C，正在停止（已完成的单位已落盘，可续跑）")
            pool.terminate()
            stopped = True
        finally:
            pbar.close()

    el = time.time() - t0
    print(f"\n>>> 本轮完成 {n_ok} 个单位 / 空 {n_empty} / 失败 {n_err}，"
          f"新增 {n_win:,} 个窗口，用时 {el/60:.1f} 分钟")
    print(f">>> 按数据集完成情况: "
          f"{ {k: f'{done_by_ds[k]}/{total_by_ds[k]}' for k in ds_order if total_by_ds[k]} }")
    if errors:
        print("[!] 失败单位（前 10）：")
        for e in errors[:10]:
            print(f"    {e['unit_id']}: {e['error']}")
        with open(os.path.join(CFG.OUTPUT_DIR, "_state", "errors.json"),
                  "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

    finalize(CFG.OUTPUT_DIR, el)
    if stopped:
        print("\n[提醒] 提前结束，仍有单位未处理。重跑同一命令即可续跑。")


def run_verify(out_root: str):
    idx = os.path.join(out_root, "pretrain_index.parquet")
    if not os.path.isfile(idx):
        print("未找到 pretrain_index.parquet，请先 --mode finalize")
        return
    df = pd.read_parquet(idx, columns=["shard", "local_idx", "window_len",
                                       "split", "group_key", "dataset"])
    g = df.groupby("shard").agg(n=("local_idx", "size"),
                                mx=("local_idx", "max"),
                                L=("window_len", "first"))
    sdir = os.path.join(out_root, "shards")
    bad = []
    for shard, row in tqdm(g.iterrows(), total=len(g), desc="[verify]"):
        p = os.path.join(sdir, shard)
        if not os.path.isfile(p):
            bad.append((shard, "shard 文件缺失"))
            continue
        try:
            with h5py.File(p, "r") as f:
                sh = f["iq"].shape
        except Exception as e:
            bad.append((shard, f"无法打开: {e}"))
            continue
        if sh[0] != row.n or sh[0] != row.mx + 1:
            bad.append((shard, f"样本数 {sh[0]} != 索引 {row.n} (max_idx={row.mx})"))
        if sh[1] != 2 or sh[2] != row.L:
            bad.append((shard, f"形状 {sh} 与 window_len={row.L} 不符"))

    on_disk = {fn for fn in os.listdir(sdir) if fn.endswith(".h5")}
    orphan = on_disk - set(g.index)

    tr = set(df.loc[df.split == "train", "group_key"])
    va = set(df.loc[df.split == "val", "group_key"])
    leak = tr & va

    print(f"\n总窗口         : {len(df):,}")
    print(f"索引引用 shard : {len(g)}")
    print(f"磁盘 shard     : {len(on_disk)}")
    print(f"孤儿 shard     : {len(orphan)}")
    print(f"异常 shard     : {len(bad)}")
    print(f"train/val 泄漏 : {len(leak)} 个 group")
    print(f"split 分布     : {df['split'].value_counts().to_dict()}")
    ds_split = df.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    print(f"\n各数据集 split:\n{ds_split.to_string()}")
    for s, why in bad[:20]:
        print(f"   {s}: {why}")
    if not bad and not orphan and not leak:
        print("\n✅ 全部一致，可以开始预训练。")
    else:
        print("\n❌ 存在问题，请先修复。")


def run_check(out_root: str, n: int = 12):
    idx = os.path.join(out_root, "pretrain_index.parquet")
    if not os.path.isfile(idx):
        print("未找到 pretrain_index.parquet，请先 --mode finalize")
        return
    df = pd.read_parquet(idx, columns=["dataset", "shard", "local_idx", "rate_class",
                                       "band", "is_active", "is_active_global",
                                       "est_snr_db", "snr_global_db", "split",
                                       "raw_power", "modulation", "sample_weight"])
    print(f"总窗口 {len(df):,}")
    print(df.groupby(["dataset", "rate_class"]).size().to_string())
    print(f"\nsplit            : {df['split'].value_counts().to_dict()}")
    print(f"is_active(stream): {df['is_active'].value_counts().to_dict()}")
    print(f"is_active(global): {df['is_active_global'].value_counts().to_dict()}")
    print(f"sample_weight    : "
          f"{df.groupby('dataset')['sample_weight'].first().round(4).to_dict()}")

    smp = df.sample(min(n, len(df)), random_state=0)
    print(f"\n抽检 {len(smp)} 个窗口：")
    ratios = []
    for _, r in smp.iterrows():
        p = os.path.join(out_root, "shards", r["shard"])
        with h5py.File(p, "r") as f:
            x = np.asarray(f["iq"][int(r["local_idx"])], dtype=np.float32)
        z = x[0] + 1j * x[1]
        ratio = float(np.std(x[0]) / max(np.std(x[1]), 1e-9))
        ratios.append(ratio)
        print(f"  {r['dataset']:<13} {r['rate_class']:<5} L={x.shape[-1]:<6} "
              f"|z|mean={np.mean(np.abs(z)):.3f} stdI/stdQ={ratio:.4f} "
              f"act={r['is_active']}/{r['is_active_global']} "
              f"snrG={r['snr_global_db']:.1f}dB nan={int(np.isnan(x).sum())}")

    ra = np.asarray(ratios)
    print(f"\nstdI/stdQ  mean={ra.mean():.4f}  std={ra.std():.4f}  "
          f"range=[{ra.min():.4f}, {ra.max():.4f}]")
    if ra.std() < 1e-4:
        print("❌ stdI/stdQ 恒定 -> I/Q 仍在被【分别】归一化，P0-2 未修复！")
    elif 0.7 <= ra.mean() <= 1.35:
        print("✅ I/Q 联合归一化正确（比值围绕 1.0 自然抖动）")
    else:
        print("⚠ 比值偏离 1.0 较多，请检查数据源是否本身存在强 IQ 不平衡")


def main():
    ap = argparse.ArgumentParser(
        description="无人机射频 IQ 基础模型预处理 v5.2（断点续跑 / 磁盘预算 / RF_UAVs 打包）")
    ap.add_argument("--mode", default="run",
                    choices=["dry", "run", "finalize", "check", "verify",
                             "clean-staging"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="每个数据集最多处理多少个单位（调试用）")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--budget-gb", type=float, default=None)
    ap.add_argument("--prescan", type=str, default=None,
                    choices=["exact", "probe", "off"])
    ap.add_argument("--channel-mode", type=str, default=None,
                    choices=["both", "active"])
    ap.add_argument("--rf-uavs-per-unit", type=int, default=None,
                    help="RF_UAVs 每个处理单位打包的文件数 (默认 256)")
    ap.add_argument("--rf-uavs-stride", type=int, default=None,
                    help="RF_UAVs 文件抽样步长, >1 用于快速省盘 (默认 1, 不抽样)")
    args = ap.parse_args()

    if args.out:
        CFG.OUTPUT_DIR = args.out
    if args.workers:
        CFG.NUM_WORKERS = args.workers
    if args.limit is not None:
        CFG.MAX_FILES_PER_DATASET = args.limit
    if args.budget_gb is not None:
        CFG.DISK_BUDGET_GB = args.budget_gb
    if args.prescan:
        CFG.PRESCAN_MODE = args.prescan
    if args.channel_mode:
        CFG.DRONERFA_CHANNEL_MODE = args.channel_mode
    if args.rf_uavs_per_unit:
        CFG.RF_UAVS_FILES_PER_UNIT = args.rf_uavs_per_unit
    if args.rf_uavs_stride:
        CFG.RF_UAVS_FILE_STRIDE = args.rf_uavs_stride
    CFG.DRY_RUN = (args.mode == "dry")

    np.random.seed(CFG.SEED)
    prepare_dirs(CFG.OUTPUT_DIR)

    if args.mode == "clean-staging":
        shutil.rmtree(os.path.join(CFG.OUTPUT_DIR, "_staging"), ignore_errors=True)
        os.makedirs(os.path.join(CFG.OUTPUT_DIR, "_staging"), exist_ok=True)
        print("staging 已清空")
        return
    if args.mode == "finalize":
        finalize(CFG.OUTPUT_DIR)
        return
    if args.mode == "check":
        run_check(CFG.OUTPUT_DIR)
        return
    if args.mode == "verify":
        run_verify(CFG.OUTPUT_DIR)
        return

    print("=" * 88)
    print("无人机射频 IQ 基础模型预处理 v5.2")
    print("=" * 88)
    print(f"输出目录        : {CFG.OUTPUT_DIR}")
    print(f"模式            : {args.mode}   worker: {CFG.NUM_WORKERS}")
    print(f"归一化          : {CFG.NORM_METHOD} (I/Q 联合 scale, "
          f"remove_dc={CFG.REMOVE_DC}, clip=±{CFG.NORM_CLIP})")
    print(f"存储            : {CFG.STORE_DTYPE}, 压缩={CFG.HDF5_COMPRESSION}, "
          f"chunk=(1,2,L)")
    print(f"DroneRFa        : channel={CFG.DRONERFA_CHANNEL_MODE}, "
          f"每流上限={CFG.MAX_WINDOWS_PER_STREAM}")
    print(f"RF_UAVs         : 每单位 {CFG.RF_UAVS_FILES_PER_UNIT} 文件, "
          f"stride={CFG.RF_UAVS_FILE_STRIDE}")
    print(f"功率预筛        : {CFG.PRESCAN_MODE}  阈值={CFG.ACTIVE_SNR_TH_DB}dB  "
          f"neg={CFG.NEG_RATIO}  noise_keep={CFG.NOISE_ONLY_KEEP}  "
          f"flat_keep={CFG.FLAT_CHANNEL_KEEP}")
    print(f"Gent 采样率     : {CFG.DEFAULT_FS_GENT/1e6:.3f} MS/s "
          f"[{CFG.GENT_FS_SOURCE}] -> rate_class="
          f"{get_rate_class(CFG.DEFAULT_FS_GENT)}, 窗口="
          f"{get_target_len(get_rate_class(CFG.DEFAULT_FS_GENT))}")
    print(f"磁盘            : 预算 {CFG.DISK_BUDGET_GB} GB / 保留 {CFG.MIN_FREE_GB} GB / "
          f"当前可用 {human_bytes(shutil.disk_usage(CFG.OUTPUT_DIR).free)}")
    print(f"划分            : VAL_RATIO={CFG.VAL_RATIO} (分层+分组, "
          f"单组类别用带保护带的时间切分)")
    print("=" * 88)

    units = build_units()
    if not units:
        print("[错误] 没有枚举到任何数据单位，请检查输入路径。")
        return
    print("单位统计: " + str(dict(Counter(u.dataset for u in units))))

    if args.mode == "dry":
        run_dry(units)
    else:
        run_full(units)


if __name__ == "__main__":
    main()
