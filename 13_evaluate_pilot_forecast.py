#!/usr/bin/env python3
"""Evaluate the GraphRivers integration pilot against NWM simulation targets.

Persistence uses the last available six-hour NWM discharge interval immediately
before issue time, a simulated-state diagnostic baseline (not an operational
observation baseline). All methods use common available verification samples.
Lead labels are interval end hours: lead 6 covers [issue, issue+6h).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_training_module():
    path = Path(__file__).resolve().parent / "12_train_pilot_forecast.py"
    if not path.is_file():
        raise FileNotFoundError(f"Keep evaluation beside the training script: {path}")
    spec = importlib.util.spec_from_file_location("pilot_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metric_values(count, squared_error, absolute_error, obs_sum, obs_squared_sum):
    count = np.asarray(count)
    denominator = obs_squared_sum - np.divide(obs_sum**2, count,
                         out=np.zeros_like(obs_sum, dtype=float), where=count > 0)
    mse = np.divide(squared_error, count, out=np.full_like(squared_error, np.nan), where=count > 0)
    mae = np.divide(absolute_error, count, out=np.full_like(absolute_error, np.nan), where=count > 0)
    nse = np.full_like(squared_error, np.nan)
    usable = (count > 1) & (denominator > 1e-12 * np.maximum(obs_squared_sum, 1))
    np.divide(squared_error, denominator, out=nse, where=usable)
    nse[usable] = 1 - nse[usable]
    return mae, np.sqrt(mse), nse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument("--pilot", default="rosn7")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    api = load_training_module()
    run = args.run or args.root / "models" / f"{args.pilot}_forecast_pilot"
    checkpoint_path = run / "best.pt"
    # Read only the trusted checkpoint created by our training script.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, norm = checkpoint["config"], checkpoint["normalization"]
    if config["pilot"] != args.pilot:
        raise ValueError("Pilot differs from the saved model")
    model_path = Path(__file__).resolve().parent / "gwnetmodel.py"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != config["model_sha256"]:
        raise ValueError("Model source changed since training")
    # Ensure the data have not been replaced since training/validation.
    for source_path, size, modified in config["sources"]:
        stat = Path(source_path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, modified):
            raise ValueError(f"Data source changed: {source_path}")
    graph = args.root / "processed/pilot_graph" / args.pilot
    nodes = pd.read_parquet(graph / "nodes.parquet").sort_values("node_index")
    edges = pd.read_parquet(graph / "edges.parquet")
    ids = nodes.comid.to_numpy(dtype=np.int64)
    if not nodes.comid.is_unique or not np.array_equal(nodes.node_index.values, np.arange(len(ids))):
        raise ValueError("Invalid node ordering")
    source = edges.source.to_numpy(dtype=np.int64)
    target = edges.target.to_numpy(dtype=np.int64)
    if hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() != config["node_sha256"]:
        raise ValueError("Node ordering changed")
    edge_hash = hashlib.sha256(np.column_stack([source, target]).astype("<i8").tobytes()).hexdigest()
    if edge_hash != config["edge_sha256"]:
        raise ValueError("Graph edges changed")
    report = json.loads((graph / "pilot_graph_report.json").read_text())
    outlet_id = int(report["outlet_comid"])
    matches = np.flatnonzero(ids == outlet_id)
    if len(matches) != 1:
        raise ValueError("Outlet absent or duplicated")
    outlet = int(matches[0])
    train_start, train_stop = map(pd.Timestamp, config["train"])
    val_start, val_stop = map(pd.Timestamp, config["validation"])
    train_reader = api.MonthlyReader(args.root, ids, train_start, train_stop)
    # Arithmetic mean of valid raw training discharge separately for each node.
    train_flow_sum = np.zeros(len(ids), dtype=np.float64)
    train_flow_count = np.zeros(len(ids), dtype=np.int64)
    for number in range(len(train_reader.paths)):
        _, _, flow, valid = train_reader.month(number)
        train_flow_sum += np.where(valid, flow, 0).astype(np.float64).sum(axis=0)
        train_flow_count += valid.sum(axis=0)
    if not (train_flow_count > 0).all():
        raise ValueError("A pilot node lacks training data for its mean baseline")
    train_mean = train_flow_sum / train_flow_count
    context_start = (val_start - pd.Timedelta(hours=6 * api.HISTORY)).to_period("M").start_time
    reader = api.MonthlyReader(args.root, ids, context_start, val_stop)
    dataset = api.ForecastDataset(reader, norm, val_start, val_stop)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    device = torch.device(args.device)
    model = api.make_model(ids, edges, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    methods = ["gnn", "nwm_persistence", "training_mean"]
    shape = (len(methods), api.HORIZON, len(ids))
    count = np.zeros(shape, dtype=np.int64)
    sse, sae, obs_sum, obs_sq, normalized_sse = [np.zeros(shape) for _ in range(5)]
    clipped, candidate_targets, common_targets = 0, 0, 0
    q_mean, q_std = norm["mean"][8], norm["std"][8]
    print(f"Best checkpoint: epoch {checkpoint['epoch']}; saved validation MSE={checkpoint['val_mse']:.6f}", flush=True)
    print(f"Validation windows={len(dataset)}; candidate outlet COMID={outlet_id}", flush=True)
    with torch.no_grad():
        for number, batch in enumerate(loader):
            normalized_pred = api.predict(model, batch["x"].to(device)).cpu().numpy()[..., 0].astype(np.float64)
            normalized_target = batch["y"].numpy()[..., 0].astype(np.float64)
            target_mask = batch["mask"].numpy()[..., 0]
            # Recover raw targets directly; do not inverse-transform placeholders.
            observations, last_flows, last_masks = [], [], []
            for issue in batch["issue"].tolist():
                _, _, flow, mask = reader.window(issue, issue + api.HORIZON)
                _, _, last_flow, last_mask = reader.window(issue - 1, issue)
                observations.append(flow)
                last_flows.append(last_flow[0])
                last_masks.append(last_mask[0])
            observations = np.stack(observations).astype(np.float64)
            last_flows, last_masks = np.stack(last_flows), np.stack(last_masks)
            common = target_mask & last_masks[:, None, :]
            candidate_targets += int(target_mask.sum())
            common_targets += int(common.sum())
            with np.errstate(over="raise", invalid="raise"):
                raw_gnn = np.expm1(normalized_pred * q_std + q_mean)
            if not np.isfinite(raw_gnn).all():
                raise FloatingPointError("Nonfinite inverse-transformed GNN discharge")
            clipped += int(((raw_gnn < 0) & common).sum())
            # Enforce nonnegative discharge for raw-unit evaluation only.
            raw_gnn = np.maximum(raw_gnn, 0)
            persistence = np.broadcast_to(last_flows[:, None, :], observations.shape)
            training_mean = np.broadcast_to(train_mean[None, None, :], observations.shape)
            for method, prediction in enumerate([raw_gnn, persistence, training_mean]):
                if method == 0:
                    normalized_method = normalized_pred
                else:
                    normalized_method = (np.log1p(np.where(common, prediction, 0)) - q_mean) / q_std
                error = np.where(common, prediction - observations, 0)
                normalized_error = np.where(common, normalized_method - normalized_target, 0)
                obs = np.where(common, observations, 0)
                count[method] += common.sum(axis=0)
                sse[method] += np.square(error).sum(axis=0)
                sae[method] += np.abs(error).sum(axis=0)
                normalized_sse[method] += np.square(normalized_error).sum(axis=0)
                obs_sum[method] += obs.sum(axis=0)
                obs_sq[method] += np.square(obs).sum(axis=0)
            if (number + 1) % 50 == 0:
                print(f"Evaluated {min((number+1)*args.batch_size, len(dataset))}/{len(dataset)} windows", flush=True)
    if not common_targets:
        raise ValueError("No common valid evaluation targets")
    mae, rmse, nse = metric_values(count, sse, sae, obs_sum, obs_sq)
    node_rows, lead_rows, summary_rows = [], [], []
    for method, name in enumerate(methods):
        for lead in range(api.HORIZON):
            total_count = int(count[method, lead].sum())
            node_nse = nse[method, lead]
            lead_rows.append({"method": name, "lead_end_hours": (lead+1)*6,
                "count": total_count,
                "mae_m3_s": float(sae[method, lead].sum()/total_count) if total_count else None,
                "rmse_m3_s": float(np.sqrt(sse[method, lead].sum()/total_count)) if total_count else None,
                "normalized_log_mse": float(normalized_sse[method, lead].sum()/total_count) if total_count else None,
                "median_node_nse": float(np.median(node_nse[np.isfinite(node_nse)])) if np.isfinite(node_nse).any() else None})
            for node, comid in enumerate(ids):
                node_rows.append({"method": name, "lead_end_hours": (lead+1)*6,
                    "comid": int(comid), "candidate_outlet": node == outlet,
                    "count": int(count[method, lead, node]),
                    "mae_m3_s": mae[method, lead, node], "rmse_m3_s": rmse[method, lead, node],
                    "nse": nse[method, lead, node]})
        for label, selection in [("all_nodes", slice(None)), ("candidate_outlet", slice(outlet, outlet+1))]:
            total_count = int(count[method, :, selection].sum())
            summary_rows.append({"method": name, "scope": label, "count": total_count,
                "mae_m3_s": float(sae[method, :, selection].sum()/total_count) if total_count else None,
                "rmse_m3_s": float(np.sqrt(sse[method, :, selection].sum()/total_count)) if total_count else None,
                "normalized_log_mse": float(normalized_sse[method, :, selection].sum()/total_count) if total_count else None})
    out = run / "evaluation_best"
    out.mkdir(parents=True, exist_ok=True)
    node_metrics = pd.DataFrame(node_rows)
    lead_metrics = pd.DataFrame(lead_rows)
    summary = pd.DataFrame(summary_rows)
    node_metrics.to_csv(out / "node_lead_metrics.csv", index=False)
    lead_metrics.to_csv(out / "network_lead_metrics.csv", index=False)
    node_metrics.loc[node_metrics.candidate_outlet].to_csv(out / "outlet_lead_metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame({"comid": ids, "training_mean_m3_s": train_mean,
                  "valid_training_intervals": train_flow_count}).to_csv(out / "training_node_means.csv", index=False)
    metadata = {"checkpoint": str(checkpoint_path), "best_epoch": checkpoint["epoch"],
        "validation_windows": len(dataset), "available_target_instances": candidate_targets,
        "common_target_instances": common_targets, "excluded_without_persistence_history": candidate_targets-common_targets,
        "negative_gnn_predictions_clipped": clipped,
        "training_mean_definition": "node-specific arithmetic mean of raw discharge in the training period",
        "candidate_outlet_comid": outlet_id,
        "routing_and_outlet_status": "provisional",
        "persistence_source": "last historical six-hour NWM discharge; diagnostic simulated-state baseline",
        "lead_convention": "interval end hours; lead 6 represents [issue, issue+6h)",
        "note": "Overlapping windows reuse target times. Counts are forecast instances, not independent observations. Raw metrics clip negative GNN discharge to zero; normalized MSE uses original model output."}
    (out / "evaluation_report.json").write_text(json.dumps(metadata, indent=2))
    print("\nMatched validation summary:")
    print(summary.to_string(index=False))
    print("\nCandidate outlet, selected leads:")
    selected = node_metrics.candidate_outlet & node_metrics.lead_end_hours.isin([6,24,72,120,240])
    print(node_metrics.loc[selected, ["method","lead_end_hours","count","mae_m3_s","rmse_m3_s","nse"]].to_string(index=False))
    print(f"\nNegative GNN predictions clipped for raw metrics: {clipped}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
