# run_pipeline.py
import os, sys, yaml, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

from src.core.pipeline_core import Row, PipelineRunner, CSVLogger, ensure_dir
from src.core.components import FrameSampler, YOLOFaceDetect, FaceFilter, CropExporter, ResourceRegistry

def load_cfg(path="pipeline.yml"):
    return yaml.safe_load(open(path))

def build_pipeline(cfg, cache_root: Path, logs: CSVLogger):
    comps = [
        FrameSampler(cfg.get("frame_sampler",{}), cache_root, logs),
        YOLOFaceDetect(cfg.get("yolo_face",{}), cache_root, logs),
        # FaceFilter(cfg.get("face_filter",{}), cache_root, logs),
        CropExporter(cfg.get("export_crops",{}), cache_root, logs),
    ]
    return PipelineRunner(comps)

def list_inputs(videos_dir: Path):
    rows = []
    for p in sorted(videos_dir.glob("*.*")):
        asset_id = hashlib.sha256(p.read_bytes()[:4096]).hexdigest()[:16]  # fast-ish stable id
        rows.append(Row(asset_id=asset_id, uri=str(p), stage="ingest"))
    return rows

# -- worker bootstrap to preload heavy models once per process
def worker_bootstrap(model_cfg):
    from ultralytics import YOLO
    key = f"ultralytics:{model_cfg.get('model','yolov8n-face.pt')}"
    ResourceRegistry.get(key, lambda: YOLO(model_cfg.get("model","yolov8n-face.pt")))

def run_shard(shard_rows, cfg, cache_root, logs_root):
    logs = CSVLogger(Path(logs_root))
    runner = build_pipeline(cfg, Path(cache_root), logs)
    stream = runner.run(iter(shard_rows))
    # materialize only final outputs (still streaming)
    out_manifest = []
    for r in stream:
        out_manifest.append({
            "asset_id": r.asset_id, "uri": r.uri, "stage": r.stage,
            "frame_idx": r.frame_idx, "image_path": r.image_path,
            "boxes": r.boxes, **{f"meta.{k}": v for k,v in r.meta.items()}
        })
    return out_manifest

if __name__ == "__main__":
    import argparse, hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="pipeline.yaml")
    ap.add_argument("--videos", default="./videos")
    ap.add_argument("--cache", default="./.cache")
    ap.add_argument("--logs", default="./logs")
    ap.add_argument("--out", default="./out/manifest.parquet")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--shard_size", type=int, default=200)   # tune for ~2000 files
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    ensure_dir(Path(args.cache)); ensure_dir(Path(args.logs)); ensure_dir(Path(args.out).parent)

    rows = list_inputs(Path(args.videos))

    # shard inputs so each worker reuses loaded model across a big chunk
    shards = [rows[i:i+args.shard_size] for i in range(0, len(rows), args.shard_size)]

    manifest_rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=worker_bootstrap,
                             initargs=(cfg.get("yolo_face",{}),)) as ex:
        futures = [ex.submit(run_shard, shard, cfg, args.cache, args.logs) for shard in shards]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="shards"):
            manifest_rows.extend(fut.result())

    # write master manifest (Parquet is nice for scale; CSV optional)
    pd.DataFrame(manifest_rows).to_parquet(args.out, index=False)
    pd.DataFrame(manifest_rows).to_csv(args.out.replace(".parquet", ".csv"), index=False)
    print("Done:", args.out)
