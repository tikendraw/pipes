# components.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Any, Dict
import subprocess, cv2, math
from dataclasses import asdict
from ultralytics import YOLO
import hashlib

from .pipeline_core import Component, FilterComponent, Row, ensure_dir, ResourceRegistry

# --------- Component 1: FrameSampler ---------
class FrameSampler(Component):
    name = "frame_sampler"
    # cfg: { fps: 2, out_dir: "./.cache/frames" }

    def process_one(self, row: Row) -> Iterable[Row]:
        cfg = self.cfg
        fps = cfg.get("fps", 2)
        frames_dir = Path(cfg.get("out_dir", ".cache/frames")) / row.asset_id
        ensure_dir(frames_dir)
        # ffmpeg extracts once per asset (idempotent)
        # select fps to avoid huge memory use
        cmd = [
            "ffmpeg","-hide_banner","-loglevel","error","-y",
            "-i", row.uri, "-vf", f"fps={fps}", str(frames_dir / "%06d.jpg")
        ]
        subprocess.run(cmd, check=True)
        for p in sorted(frames_dir.glob("*.jpg")):
            idx = int(p.stem)
            yield Row(
                asset_id=row.asset_id, uri=row.uri,
                stage=self.name, frame_idx=idx, image_path=str(p),
                meta={"fps": fps}
            )

# --------- Component 2: YOLOFaceDetect ---------
class YOLOFaceDetect(Component):
    name = "yolo_face"
    # cfg: { model: "yolov8n-face.pt", conf: 0.5 }

    def __init__(self, cfg, cache_root, logs):
        super().__init__(cfg, cache_root, logs)
        self.model_key = f"ultralytics:{cfg.get('model','yolov8n-face.pt')}"

    def _model(self):
        return ResourceRegistry.get(self.model_key, lambda: YOLO()) #self.cfg.get("model","yolov8n-face.pt")))

    def process(self, rows: Iterable[Row]) -> Iterable[Row]:
        model = self._model()
        conf = float(self.cfg.get("conf", 0.5))
        for r in rows:
            if not r.image_path:
                continue
            res = model.predict(source=r.image_path, conf=conf, verbose=False)[0]
            boxes = []
            for b in res.boxes.xyxy.cpu().numpy().tolist():
                x1,y1,x2,y2 = map(int, b)
                boxes.append((x1,y1,x2,y2))
            out = Row(**{**asdict(r), "stage": self.name, "boxes": boxes})
            # lightweight metrics to meta
            if boxes:
                img = cv2.imread(r.image_path)
                H, W = img.shape[:2]
                areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes]
                out.meta.update({
                    "face_count": len(boxes),
                    "max_face_pct": 100.0 * max(areas)/float(W*H),
                    "W": W, "H": H
                })
            else:
                out.meta.update({"face_count": 0, "max_face_pct": 0.0})
            yield out

# --------- Component 3: FaceFilter ---------
class FaceFilter(FilterComponent):
    name = "face_filter"
    # cfg: { min_face_pct: 4, max_faces_per_frame: 2 }

    def keep(self, r: Row) -> bool:
        mfp = float(self.cfg.get("min_face_pct", 4))
        maxc = int(self.cfg.get("max_faces_per_frame", 2))
        return (r.meta.get("max_face_pct", 0.0) >= mfp) and (r.meta.get("face_count", 0) <= maxc)

# --------- Component 4: CropExporter ---------
class CropExporter(Component):
    name = "export_crops"
    # cfg: { out_dir: "./out/faces", size: [224,224], margin_pct: 15 }

    def process(self, rows: Iterable[Row]) -> Iterable[Row]:
        import numpy as np
        out_dir = Path(self.cfg.get("out_dir","./out/faces"))
        ensure_dir(out_dir)
        size = tuple(self.cfg.get("size",[224,224]))
        margin = float(self.cfg.get("margin_pct", 15))/100.0

        for r in rows:
            if not r.boxes:
                continue
            img = cv2.imread(r.image_path)
            H, W = img.shape[:2]
            for i,(x1,y1,x2,y2) in enumerate(r.boxes):
                bw, bh = (x2-x1), (y2-y1)
                cx, cy = x1 + bw/2, y1 + bh/2
                nw, nh = int(bw*(1+margin)), int(bh*(1+margin))
                xx1, yy1 = max(0,int(cx-nw/2)), max(0,int(cy-nh/2))
                xx2, yy2 = min(W,int(cx+nw/2)), min(H,int(cy+nh/2))
                crop = img[yy1:yy2, xx1:xx2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, size)
                fid = hashlib.sha256(f"{r.asset_id}:{r.frame_idx}:{i}:{xx1}:{yy1}:{xx2}:{yy2}".encode()).hexdigest()[:20]
                fp = out_dir / f"{fid}.jpg"
                cv2.imwrite(str(fp), crop)
                meta = {**r.meta, "face_idx": i, "export_path": str(fp)}
                yield Row(
                    asset_id=r.asset_id, uri=r.uri, stage=self.name,
                    frame_idx=r.frame_idx, image_path=str(fp), boxes=[(xx1,yy1,xx2,yy2)],
                    meta=meta
                )
