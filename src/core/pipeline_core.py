# pipeline_core.py
from __future__ import annotations
import os, json, hashlib, pickle, csv, uuid, time, shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Iterable, Dict, Any, Optional, Iterator, List, Tuple
from contextlib import contextmanager
import pandas as pd

# ---------- Common Row ----------
@dataclass
class Row:
    asset_id: str                 # stable ID per source video
    uri: str                      # path/URL to video or image
    stage: str                    # name of stage that produced this row
    frame_idx: Optional[int] = None
    image_path: Optional[str] = None
    boxes: Optional[List[Tuple[int,int,int,int]]] = None  # xyxy
    meta: Dict[str, Any] = field(default_factory=dict)

    def shallow_hash(self) -> str:
        # hash only stable fields likely to affect downstream compute
        payload = json.dumps({
            "asset_id": self.asset_id,
            "uri": self.uri,
            "frame_idx": self.frame_idx,
            "image_path": self.image_path,
            "meta_keys": sorted(list(self.meta.keys())),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

# ---------- Utilities ----------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def dict_hash(d: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- CSV Logger ----------
class CSVLogger:
    def __init__(self, root: Path):
        self.root = root; ensure_dir(root)

    def _fp(self, component_name: str) -> Path:
        return self.root / f"{component_name}.csv"

    def write_header_if_needed(self, component_name: str, fieldnames: List[str]):
        fp = self._fp(component_name)
        if not fp.exists():
            with open(fp, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log_row(self, component_name: str, data: Dict[str, Any]):
        fp = self._fp(component_name)
        fieldnames = sorted(data.keys())
        self.write_header_if_needed(component_name, fieldnames)
        with open(fp, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(data)

# ---------- Resource Registry (per-process) ----------
class ResourceRegistry:
    _store: Dict[str, Any] = {}

    @classmethod
    def get(cls, key: str, factory):
        if key not in cls._store:
            cls._store[key] = factory()
        return cls._store[key]

# ---------- Caching ----------
class CacheMixin:
    cache_root: Path

    def cache_key(self, component_name: str, cfg: Dict[str, Any], row: Row) -> str:
        token = f"{component_name}:{dict_hash(cfg)}:{row.shallow_hash()}"
        return sha256_bytes(token.encode())[:24]

    def cache_paths(self, key: str) -> Tuple[Path, Path]:
        base = self.cache_root / key
        return base / "stream.pkl", base / "done"

    def stream_from_cache(self, key: str) -> Optional[Iterator[Row]]:
        stream_fp, done_fp = self.cache_paths(key)
        if done_fp.exists() and stream_fp.exists():
            def _gen():
                with open(stream_fp, "rb") as f:
                    while True:
                        try:
                            yield pickle.load(f)
                        except EOFError:
                            break
            return _gen()
        return None

    def write_to_cache(self, key: str, rows: Iterable[Row]) -> Iterator[Row]:
        stream_fp, done_fp = self.cache_paths(key)
        ensure_dir(stream_fp.parent)
        with open(stream_fp, "wb") as f:
            for r in rows:
                pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
                yield r
        done_fp.write_text("ok")

# ---------- Component Base ----------
class Component(CacheMixin):
    name: str = "component"

    def __init__(self, cfg: Dict[str, Any], cache_root: Path, logs: CSVLogger):
        self.cfg = cfg
        self.cache_root = cache_root / self.name
        ensure_dir(self.cache_root)
        self.logs = logs

    def process(self, rows: Iterable[Row]) -> Iterable[Row]:
        """Override: consume iterable rows, yield new rows."""
        raise NotImplementedError

    # wraps process() with per-row caching + logging
    def run(self, rows: Iterable[Row]) -> Iterable[Row]:
        def _gen():
            for row in rows:
                key = self.cache_key(self.name, self.cfg, row)
                cached = self.stream_from_cache(key)
                if cached is not None:
                    for c in cached:
                        self._log(c)
                        yield c
                    continue
                produced = self.process_one(row)  # generator
                cached_stream = self.write_to_cache(key, produced)
                for c in cached_stream:
                    self._log(c)
                    yield c
        return _gen()

    # per-row default: expand a row into zero-or-more rows
    def process_one(self, row: Row) -> Iterable[Row]:
        # default implementation calls process() with single-row generator.
        def single():
            yield row
        return self.process(single())

    def _log(self, r: Row):
        data = asdict(r)
        # flatten some meta for CSV friendliness
        flat = {**data, **{f"meta.{k}": v for k, v in r.meta.items()}}
        self.logs.log_row(self.name, flat)

# ---------- Simple Filter base ----------
class FilterComponent(Component):
    name = "filter"

    def keep(self, r: Row) -> bool:
        raise NotImplementedError

    def process(self, rows: Iterable[Row]) -> Iterable[Row]:
        for r in rows:
            if self.keep(r):
                r2 = Row(**{**asdict(r), "stage": self.name})
                yield r2

# ---------- Runner ----------
class PipelineRunner:
    def __init__(self, components: List[Component]):
        self.components = components

    def run(self, rows: Iterable[Row]) -> Iterable[Row]:
        stream = rows
        for c in self.components:
            stream = c.run(stream)
        return stream
