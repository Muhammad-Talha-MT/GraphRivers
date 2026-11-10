#!/usr/bin/env python3
"""GraphRivers pilot: historical AORC -> 40 six-hour NWM discharge forecasts.

Imports GWNet from the repository containing this script. This is an integration
experiment; outlet/NWM routing verification remains pending. No future weather
or discharge histories are model inputs. Resume last.pt automatically, including
the next batch within an epoch. SIGINT/SIGTERM stop after a complete batch.
"""
import argparse
from collections import OrderedDict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import xarray as xr

VARIABLES = ["APCP_surface", "TMP_2maboveground", "SPFH_2maboveground",
             "PRES_surface", "DSWRF_surface", "DLWRF_surface",
             "UGRD_10maboveground", "VGRD_10maboveground"]
HISTORY, HORIZON = 120, 40


class MonthlyReader:
    def __init__(self, root, ids, start, stop, cache_months=16):
        self.ids = ids
        self.cache = OrderedDict()
        self.cache_months = cache_months
        self.paths, self.positions, self.offsets = [], [], [0]
        time_parts = []
        for first in pd.date_range(start, stop, freq="MS", inclusive="left"):
            last = first + pd.offsets.MonthBegin(1)
            stamp = f"{first:%Y%m%dT%H}_{last:%Y%m%dT%H}"
            path = root / "processed/six_hour_nodes" / str(first.year) / f"tennessee_{stamp}_six_hour_nodes.nc"
            if not path.is_file():
                raise FileNotFoundError(path)
            with xr.open_dataset(path, cache=False, engine="netcdf4") as ds:
                times = pd.DatetimeIndex(ds.time.values)
                wanted = pd.date_range(first, last, freq="6h", inclusive="left")
                if not times.equals(wanted):
                    raise ValueError(f"Time mismatch: {path}")
                index = pd.Index(ds.feature_id.values)
                if not index.is_unique:
                    raise ValueError(f"Duplicate COMIDs: {path}")
                positions = index.get_indexer(ids)
                if (positions < 0).any():
                    raise ValueError(f"Pilot COMID absent: {path}")
                for variable in VARIABLES + ["streamflow"]:
                    for name in [variable, variable + "_valid_hours"]:
                        if ds[name].dims != ("time", "feature_id"):
                            raise ValueError(f"Unexpected dimensions: {path}: {name}")
            self.paths.append(path)
            self.positions.append(positions)
            time_parts.append(times.values)
            self.offsets.append(self.offsets[-1] + len(times))
        self.offsets = np.asarray(self.offsets)
        self.times = pd.DatetimeIndex(np.concatenate(time_parts))
        if not self.times.equals(pd.date_range(start, stop, freq="6h", inclusive="left")):
            raise ValueError("Monthly archive has a gap")

    def month(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        with xr.open_dataset(self.paths[number], cache=False, engine="netcdf4") as ds:
            ds = ds.isel(feature_id=self.positions[number])
            x = np.stack([ds[v].values for v in VARIABLES], axis=-1).astype(np.float32)
            counts = np.stack([ds[v + "_valid_hours"].values for v in VARIABLES], axis=-1)
            xm = np.isfinite(x) & (counts == 6)
            q = ds.streamflow.values.astype(np.float32)
            qm = np.isfinite(q) & (ds.streamflow_valid_hours.values == 6)
            if (q[qm] < 0).any():
                raise ValueError(f"Negative valid discharge: {self.paths[number]}")
        result = x, xm, q, qm
        self.cache[number] = result
        while len(self.cache) > self.cache_months:
            self.cache.popitem(last=False)
        return result

    def window(self, start, stop):
        if not 0 <= start < stop <= len(self.times):
            raise IndexError((start, stop))
        parts = [[], [], [], []]
        cursor = start
        while cursor < stop:
            number = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
            end = min(stop, int(self.offsets[number + 1]))
            left, right = cursor - self.offsets[number], end - self.offsets[number]
            for destination, values in zip(parts, self.month(number)):
                destination.append(values[left:right])
            cursor = end
        return tuple(np.concatenate(p, axis=0) for p in parts)


def fit_normalization(reader, progress=False):
    count = np.zeros(9, dtype=np.int64)
    total, squares = np.zeros(9), np.zeros(9)
    for number in range(len(reader.paths)):
        x, xm, q, qm = reader.month(number)
        for feature in range(9):
            values = (x[..., feature][xm[..., feature]].astype(np.float64)
                      if feature < 8 else np.log1p(q[qm].astype(np.float64)))
            count[feature] += values.size
            total[feature] += values.sum()
            squares[feature] += np.square(values).sum()
        if progress and ((number + 1) % 12 == 0 or number + 1 == len(reader.paths)):
            print(f"Training normalization: {number+1}/{len(reader.paths)} months", flush=True)
    if not (count > 0).all():
        raise ValueError("A training variable has no valid values")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "count": count.tolist()}


class ForecastDataset(Dataset):
    def __init__(self, reader, normalization, target_start, target_stop):
        self.reader = reader
        self.mean = np.asarray(normalization["mean"])
        self.std = np.asarray(normalization["std"])
        # The entire target interval must lie within its calendar split.
        left = max(HISTORY, int(reader.times.searchsorted(target_start)))
        right = int(reader.times.searchsorted(target_stop)) - HORIZON + 1
        self.issues = np.arange(left, right, dtype=np.int64)
        if len(self.issues) == 0:
            raise ValueError("No forecasting windows in requested split")

    def __len__(self):
        return len(self.issues)

    def __getitem__(self, index):
        issue = int(self.issues[index])
        x, xm, _, _ = self.reader.window(issue - HISTORY, issue)
        _, _, q, qm = self.reader.window(issue, issue + HORIZON)
        safe_x = np.where(xm, x, self.mean[:8])
        normalized_x = ((safe_x - self.mean[:8]) / self.std[:8]).astype(np.float32)
        normalized_x[~xm] = 0
        inputs = np.concatenate([normalized_x, xm.astype(np.float32)], axis=-1)
        safe_q = np.where(qm, q, 0).astype(np.float64)
        y = ((np.log1p(safe_q) - self.mean[8]) / self.std[8]).astype(np.float32)
        y[~qm] = 0
        return {"x": torch.from_numpy(inputs), "y": torch.from_numpy(y[..., None]),
                "mask": torch.from_numpy(qm[..., None]), "issue": issue}


def make_model(ids, edges, device):
    from gwnetmodel import GWNet
    from utils_wnet import load_adj
    n = len(ids)
    adjacency = np.eye(n, dtype=np.float32)
    adjacency[edges.source.values, edges.target.values] = 1
    supports = [torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
                for a in load_adj(adjacency, "doubletransition")]
    model = GWNet(device=device, num_nodes=n, supports=supports, addaptadj=False,
                  in_dim=16, out_dim=40, blocks=4, layers=5,
                  residual_channels=8, dilation_channels=8,
                  skip_channels=32, end_channels=64)
    # Reuse the tested fix even if the source still has legacy convolution types.
    for name in ("gate_convs", "residual_convs", "skip_convs"):
        modules = []
        for layer in getattr(model, name):
            if isinstance(layer, nn.Conv1d):
                corrected = nn.Conv2d(layer.in_channels, layer.out_channels,
                    kernel_size=layer.kernel_size, stride=layer.stride,
                    padding=layer.padding, dilation=layer.dilation,
                    groups=layer.groups, bias=layer.bias is not None)
                corrected.load_state_dict(layer.state_dict())
                layer = corrected
            modules.append(layer)
        setattr(model, name, nn.ModuleList(modules))
    return model.to(device)


def predict(model, x):
    raw = model(x)
    if raw.shape != (x.shape[0], 1, x.shape[2], HORIZON):
        raise ValueError(f"Unexpected model output {tuple(raw.shape)}")
    return raw[:, 0].permute(0, 2, 1).unsqueeze(-1)


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def epoch_order(size, seed, epoch):
    return np.random.default_rng(seed + epoch).permutation(size).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument("--pilot", default="rosn7")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--cache-months", type=int, default=16)
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-stop", default="2018-07-01", help="Exclusive; no later than 2019-01-01")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables; otherwise stop after this many non-improving validation epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.checkpoint_every, args.cache_months) < 1:
        parser.error("Epochs, batch size, checkpoint interval and cache size must be positive")
    train_start, train_stop = pd.Timestamp(args.train_start), pd.Timestamp(args.train_stop)
    val_start, val_stop = pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")
    if (train_start.tzinfo is not None or train_stop.tzinfo is not None
        or train_start != train_start.to_period("M").start_time
        or train_stop != train_stop.to_period("M").start_time
        or not pd.Timestamp("1979-02-01") <= train_start < train_stop <= val_start):
        parser.error("Training dates must be month starts within [1979-02-01, 2019-01-01]")
    if args.early_stop_patience < 0:
        parser.error("Early-stop patience must be nonnegative")
    out = args.out or args.root / "models" / f"{args.pilot}_forecast_pilot"
    out.mkdir(parents=True, exist_ok=True)
    run_lock = (out / ".run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another training process is using {out}") from None
    last_path, best_path = out / "last.pt", out / "best.pt"
    print("INTEGRATION PILOT: outlet mapping and NWM routing remain provisional.", flush=True)
    graph = args.root / "processed/pilot_graph" / args.pilot
    nodes = pd.read_parquet(graph / "nodes.parquet").sort_values("node_index")
    edges = pd.read_parquet(graph / "edges.parquet")
    ids = nodes.comid.to_numpy(dtype=np.int64)
    if not nodes.comid.is_unique or not np.array_equal(nodes.node_index.values, np.arange(len(ids))):
        raise ValueError("Invalid pilot node ordering")
    source, target = edges.source.to_numpy(dtype=np.int64), edges.target.to_numpy(dtype=np.int64)
    if not (((source >= 0) & (source < len(ids))).all() and ((target >= 0) & (target < len(ids))).all()):
        raise ValueError("Invalid edge indices")
    if not (np.array_equal(ids[source], edges.fromcomid.values) and np.array_equal(ids[target], edges.tocomid.values)):
        raise ValueError("Edge COMIDs disagree with indices")
    train_reader = MonthlyReader(args.root, ids, train_start, train_stop, args.cache_months)
    # Prior-year history is allowed; all validation targets remain in 2019.
    val_context_start = (val_start - pd.Timedelta(hours=6 * HISTORY)).to_period("M").start_time
    val_reader = MonthlyReader(args.root, ids, val_context_start, val_stop, args.cache_months)
    model_path = Path(__file__).resolve().parent / "gwnetmodel.py"
    config = {"pilot": args.pilot, "batch_size": args.batch_size, "lr": args.lr,
              "seed": args.seed, "train": [str(train_start), str(train_stop)],
              "validation": [str(val_start), str(val_stop)], "history": HISTORY,
              "horizon": HORIZON, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "model_settings": "GWNet:16/40,4blocks,5layers,8/8/32/64,no_adaptive",
              "node_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
              "edge_sha256": hashlib.sha256(np.column_stack([source, target]).astype("<i8").tobytes()).hexdigest(),
              "sources": [[str(p), p.stat().st_size, p.stat().st_mtime_ns]
                          for p in train_reader.paths + val_reader.paths]}
    saved = None
    if last_path.exists():
        # Load only checkpoints created by this script in your trusted output folder.
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume configuration/source files changed. Use a different --out for a new run.")
    print(f"Training targets: {train_start} to {train_stop} (exclusive)", flush=True)
    print(f"Validation targets: {val_start} to {val_stop} (exclusive)", flush=True)
    normalization = saved["normalization"] if saved else fit_normalization(train_reader, progress=True)
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2))
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    train = ForecastDataset(train_reader, normalization, train_start, train_stop)
    val = ForecastDataset(val_reader, normalization, val_start, val_stop)
    print(f"Nodes={len(ids)}; train windows={len(train)}; validation windows={len(val)}", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model = make_model(ids, edges, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    state = {"epoch": 0, "next_batch": 0, "step": 0, "bad_epochs": 0,
             "train_sse": 0.0, "train_count": 0, "best_val": math.inf}
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for parameters in optimizer.state.values():
            for key, value in parameters.items():
                if torch.is_tensor(value):
                    parameters[key] = value.to(device)
        state = saved["progress"]
        restore_rng(saved["rng"])
        print(f"Resuming epoch {state['epoch'] + 1}, batch {state['next_batch'] + 1}", flush=True)
    requested_stop = [False]
    def stop_handler(signum, frame):
        requested_stop[0] = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    def checkpoint():
        atomic_save({"config": config, "normalization": normalization,
                     "progress": dict(state), "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": rng_state()}, last_path)

    checkpoint()
    for epoch in range(state["epoch"], args.epochs):
        order = epoch_order(len(train), args.seed, epoch)
        first_batch = state["next_batch"]
        remaining = order[first_batch * args.batch_size:]
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(Subset(train, remaining), batch_size=args.batch_size,
                            num_workers=0, generator=generator, pin_memory=device.type == "cuda")
        model.train()
        for batch_number, batch in enumerate(loader, start=first_batch):
            x, y, mask = [batch[k].to(device) for k in ("x", "y", "mask")]
            valid_count = int(mask.sum().item())
            if valid_count:
                optimizer.zero_grad(set_to_none=True)
                prediction = predict(model, x)
                error = prediction[mask] - y[mask]
                loss = error.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; resume from last saved checkpoint")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                state["train_sse"] += float(error.detach().square().sum().item())
                state["train_count"] += valid_count
                state["step"] += 1
            state["next_batch"] = batch_number + 1
            if state["next_batch"] % args.checkpoint_every == 0 or requested_stop[0]:
                checkpoint()
                print(f"Epoch {epoch+1}: saved after batch {state['next_batch']}", flush=True)
            if requested_stop[0]:
                print("Stopped safely. Rerun the same command to resume.", flush=True)
                return
        # Save completed training batches before validation, which can be rerun.
        checkpoint()
        model.eval()
        val_sse, val_count = 0.0, 0
        with torch.no_grad():
            loader = DataLoader(val, batch_size=args.batch_size, num_workers=0,
                                generator=torch.Generator().manual_seed(args.seed))
            for batch in loader:
                x, y, mask = [batch[k].to(device) for k in ("x", "y", "mask")]
                prediction = predict(model, x)
                error = prediction[mask] - y[mask]
                if not torch.isfinite(error).all():
                    raise FloatingPointError("Nonfinite validation prediction")
                val_sse += float(error.square().sum().item())
                val_count += int(mask.sum().item())
                if requested_stop[0]:
                    checkpoint()
                    print("Stopped in validation; validation restarts on resume.", flush=True)
                    return
        if not val_count or not state["train_count"]:
            raise ValueError("No valid training or validation targets")
        train_mse, val_mse = state["train_sse"] / state["train_count"], val_sse / val_count
        improved = val_mse < state["best_val"]
        if improved:
            state["best_val"] = val_mse
            state["bad_epochs"] = 0
            atomic_save({"config": config, "normalization": normalization,
                         "model": model.state_dict(), "epoch": epoch + 1,
                         "val_mse": val_mse}, best_path)
        else:
            state["bad_epochs"] = state.get("bad_epochs", 0) + 1
        record = {"epoch": epoch+1, "train_mse": train_mse, "val_mse": val_mse,
                  "val_valid_targets": val_count, "best": improved}
        print(json.dumps(record), flush=True)
        # Epoch-numbered reports avoid duplicate CSV rows after interruption.
        (out / f"epoch_{epoch+1:03d}.json").write_text(json.dumps(record, indent=2))
        state.update(epoch=epoch+1, next_batch=0, train_sse=0.0, train_count=0)
        checkpoint()
        if args.early_stop_patience and state.get("bad_epochs", 0) >= args.early_stop_patience:
            print(f"Early stopping after {state['bad_epochs']} non-improving epochs. Best model: {best_path}", flush=True)
            break
    print(f"Pilot complete. Best validation MSE={state['best_val']:.6f}; outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
