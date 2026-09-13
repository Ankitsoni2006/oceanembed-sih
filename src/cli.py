"""
SIH26066 — OceanEmbed Master Command-Line Interface (CLI)
Unified entry point for data inspection, downloading, pipeline processing,
model training, evaluation, independent ARGO validation, and interactive demo.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import torch

from src.data.catalog import DATA_CATALOG, TARGET_GRID, TARGET_DEPTHS, COMMON_USABLE_PERIOD
from src.data.acquisition import CopernicusAcquisitionEngine
from src.preprocessing.pipeline import OceanDataPipeline
from src.data.chunked_dataset import ChunkedOceanDataset
from src.preprocessing.normalization import OceanStandardScaler


def cli_data_inspect(args):
    """Prints detailed metadata about the catalog, grid, depths, and local storage."""
    print("=" * 70)
    print("SIH26066 — OCEANEMBED DATA CATALOG & SYSTEM STATUS")
    print("=" * 70)
    print(f"Region: North Indian Ocean ({TARGET_GRID.lat_min}°N–{TARGET_GRID.lat_max}°N, {TARGET_GRID.lon_min}°E–{TARGET_GRID.lon_max}°E)")
    print(f"Target Resolution: {TARGET_GRID.resolution}° x {TARGET_GRID.resolution}°")
    print(f"Target Grid Dimensions: {TARGET_GRID.shape[0]} Latitudes x {TARGET_GRID.shape[1]} Longitudes ({TARGET_GRID.shape[0]*TARGET_GRID.shape[1]} total cells)")
    print(f"Target Subsurface Depths ({TARGET_DEPTHS.num_depths}): {list(TARGET_DEPTHS.depths)}")
    print(f"Common Usable Overlap: {COMMON_USABLE_PERIOD[0]} to {COMMON_USABLE_PERIOD[1]}")
    
    print("\n--- SURFACE & REFERENCE DATASETS ---")
    for key, spec in DATA_CATALOG.items():
        print(f"[{key:8s}] {spec.name}")
        print(f"           Dataset ID: {spec.dataset_id}")
        print(f"           Variables:  {spec.variables} (Native res: {spec.native_spatial_res_deg}°, Freq: {spec.temporal_frequency})")
        print(f"           Coverage:   {spec.temporal_coverage[0]} to {spec.temporal_coverage[1]}")

    # Check local index manifest
    index_path = "data/processed/dataset_index.json"
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)
        print(f"\n--- PROCESSED DATASET CHUNKS ({len(idx.get('chunks', {}))}) ---")
        for cname, cinfo in idx.get("chunks", {}).items():
            print(f"  Chunk: {cname} | Days: {cinfo['num_days']} ({cinfo['date_start']} to {cinfo['date_end']}) | Size: {cinfo['size_bytes']:,} bytes")
        print(f"Total processed samples indexed: {len(idx.get('samples', []))}")
    else:
        print("\nNo processed dataset chunks found in data/processed/.")


def cli_data_test_access(args):
    """Performs a small, live authentication and metadata verification test."""
    print("Testing Copernicus Marine live connection and authentication...")
    engine = CopernicusAcquisitionEngine()
    test_date = args.date or "2020-01-01"
    try:
        path = engine.fetch_dataset_slice("SSH", test_date, test_date)
        print(f"SUCCESS! Verified live acquisition on SSH for {test_date}: {path} ({os.path.getsize(path):,} bytes)")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


def cli_data_build(args):
    """Builds aligned (X, Y) dataset chunks for a specified date range."""
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    chunk_name = args.chunk_name or f"chunk_{args.start.replace('-', '')}_{args.end.replace('-', '')}"

    pipeline = OceanDataPipeline()
    x_list = []
    y_list = []
    dates = []
    manifests = []

    cur_dt = start_dt
    while cur_dt <= end_dt:
        date_str = cur_dt.strftime("%Y-%m-%d")
        try:
            x, y, meta = pipeline.process_day(date_str)
            x_list.append(x)
            y_list.append(y)
            dates.append(date_str)
            manifests.append(meta)
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            if not args.ignore_errors:
                raise e
        cur_dt += timedelta(days=1)

    if len(x_list) == 0:
        print("No samples were successfully generated.")
        sys.exit(1)

    chunk_path = ChunkedOceanDataset.save_chunk(
        processed_dir="data/processed",
        chunk_name=chunk_name,
        x_list=x_list,
        y_list=y_list,
        dates=dates,
        metadata_extra={"daily_manifests": manifests}
    )
    print(f"\nBUILD COMPLETE! Created chunk '{chunk_name}' with {len(dates)} days at: {chunk_path}")


def cli_train(args):
    from src.training.train import run_training
    run_training(
        model_type=args.model,
        dataset_path=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir
    )


def cli_evaluate(args):
    from src.evaluation.spatial_eval import SpatialEvaluator
    raw_pt = torch.load(args.dataset, weights_only=False)
    x = raw_pt["X"]
    y = raw_pt["Y"]
    dates = raw_pt.get("dates", [])
    evaluator = SpatialEvaluator(checkpoint_path=args.checkpoint, model_type=args.model)
    summary = evaluator.evaluate_dataset(x, y, dates)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation report written to: {args.output_json}")


def cli_validate_argo(args):
    from src.validation.argo_eval import ArgoValidator
    raw_pt = torch.load(args.dataset, weights_only=False)
    x_sample = raw_pt["X"][0:1]
    validator = ArgoValidator(checkpoint_path=args.checkpoint, model_type=args.model, argo_file=args.argo_file)
    results = validator.run_validation(x_sample)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"ARGO in-situ validation report written to: {args.output_json}")


def cli_predict(args):
    from src.inference.predict import OceanEmbedPredictor
    raw_pt = torch.load(args.dataset, weights_only=False)
    x = raw_pt["X"][args.sample_idx]
    date_str = raw_pt.get("dates", ["2020-01-01"])[args.sample_idx]
    predictor = OceanEmbedPredictor(checkpoint_path=args.checkpoint)
    pred_3d = predictor.predict_tensor(x)[0]
    indices = predictor.compute_oceanographic_indices(pred_3d)
    predictor.export_to_netcdf(pred_3d, args.output_nc, date_str=date_str, extra_vars=indices)
    print(f"Prediction NetCDF created at: {args.output_nc}")


def cli_serve(args):
    import uvicorn
    print(f"Starting OceanEmbed API Server on {args.host}:{args.port}...")
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="SIH26066 — OceanEmbed Master CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # data subparsers
    data_parser = subparsers.add_parser("data", help="Data operations")
    data_sub = data_parser.add_subparsers(dest="data_action")

    # data inspect
    p_inspect = data_sub.add_parser("inspect", help="Inspect catalog and local dataset")
    p_inspect.set_defaults(func=cli_data_inspect)

    # data test-access
    p_test = data_sub.add_parser("test-access", help="Test live data API access")
    p_test.add_argument("--date", default="2020-01-01", help="Date to test (YYYY-MM-DD)")
    p_test.set_defaults(func=cli_data_test_access)

    # data build
    p_build = data_sub.add_parser("build", help="Build synchronized (X, Y) dataset chunk")
    p_build.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_build.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_build.add_argument("--chunk-name", default=None, help="Name of output chunk")
    p_build.add_argument("--ignore-errors", action="store_true", help="Continue on single-day error")
    p_build.set_defaults(func=cli_data_build)

    # train
    p_train = subparsers.add_parser("train", help="Train subsurface reconstruction models")
    p_train.add_argument("--model", default="oceanembed", choices=["oceanembed", "mlp", "simple_cnn"])
    p_train.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--val-ratio", type=float, default=0.2)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--checkpoint-dir", default="checkpoints")
    p_train.set_defaults(func=cli_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute depth-wise and spatial metrics")
    p_eval.add_argument("--checkpoint", default="checkpoints/oceanembed_best.pt")
    p_eval.add_argument("--model", default="oceanembed", choices=["oceanembed", "mlp", "simple_cnn"])
    p_eval.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt")
    p_eval.add_argument("--output-json", default="reports/evaluation_summary.json")
    p_eval.set_defaults(func=cli_evaluate)

    # validate-argo
    p_argo = subparsers.add_parser("validate-argo", help="Independent validation against in-situ ARGO profiling floats")
    p_argo.add_argument("--checkpoint", default="checkpoints/oceanembed_best.pt")
    p_argo.add_argument("--model", default="oceanembed", choices=["oceanembed", "mlp", "simple_cnn"])
    p_argo.add_argument("--argo-file", default="data/argo/20221101_prof.nc")
    p_argo.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt")
    p_argo.add_argument("--output-json", default="reports/argo_validation.json")
    p_argo.set_defaults(func=cli_validate_argo)

    # predict
    p_pred = subparsers.add_parser("predict", help="Reconstruct 3D subsurface temperature and compute physical indices")
    p_pred.add_argument("--checkpoint", default="checkpoints/oceanembed_best.pt")
    p_pred.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt")
    p_pred.add_argument("--sample-idx", type=int, default=0)
    p_pred.add_argument("--output-nc", default="data/processed/prediction_output.nc")
    p_pred.set_defaults(func=cli_predict)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI REST API server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.set_defaults(func=cli_serve)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

