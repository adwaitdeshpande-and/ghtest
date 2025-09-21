# app/services/anpr.py
from __future__ import annotations

import os
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# YOLOv8 (ultralytics) + OCR
from ultralytics import YOLO  # type: ignore
import easyocr  # type: ignore


# --------------------------
# Paths / Config
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # app/
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
ANPR_JSON_DIR = PROJECT_ROOT / "data" / "anpr"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"

# You can set a custom weights file via env:
#   export ANPR_YOLO_WEIGHTS=/workspaces/ghtest/models/yolov8n-license-plate.pt
ANPR_YOLO_WEIGHTS="keremberke/yolov8n-license-plate"

# If your plate model has a different class name/id mapping, adjust here:
PLATE_CLASS_NAMES = {"license-plate", "licence-plate", "number-plate", "plate"}  # best-effort


# --------------------------
# Utilities
# --------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm_rel(path: Path) -> str:
    """Return project-relative path string."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path.resolve())


def _alnum_up(text: str) -> str:
    """Keep only A-Z/0-9 and uppercase for plate-like strings."""
    allowed = set(string.ascii_letters + string.digits)
    return "".join([c for c in text if c in allowed]).upper()


def _best_easyocr_text(reader: easyocr.Reader, crop: np.ndarray, min_conf: float) -> Tuple[str, float]:
    # EasyOCR returns [ [bbox, text, conf], ... ]
    # We'll pick the highest conf text, post-process to alnum+upper
    result = reader.readtext(crop)
    best_text, best_conf = "", 0.0
    for _bbox, txt, conf in result:
        if conf is None:
            continue
        norm = _alnum_up(txt)
        if conf > best_conf and norm:
            best_text, best_conf = norm, float(conf)
    if best_conf < min_conf:
        return "", best_conf
    return best_text, best_conf


# --------------------------
# Main ANPR pipeline
# --------------------------
@dataclass
class ANPRParams:
    conf: float = 0.25
    iou: float = 0.45
    ocr_min_conf: float = 0.4
    max_frames: Optional[int] = None  # None = all
    device: Optional[str] = None      # "cpu" or "cuda" (if supported)


class ANPRService:
    def __init__(self, weights_path: str | Path = ANPR_YOLO_WEIGHTS, device: Optional[str] = None):
        # Accept either a local path OR a hub model id (e.g., "keremberke/yolov8n-license-plate")
        fallback_hub_id = "keremberke/yolov8n-license-plate"

        def _load_model(spec: str):
            # Let Ultralytics resolve either a path or a hub id; it auto-downloads if needed
            return YOLO(spec)

        w = Path(str(weights_path))
        try:
            if w.exists():
                self.model = _load_model(str(w))
            else:
                # If it doesn't look like an existing local file, try as hub id first
                try:
                    self.model = _load_model(str(weights_path))
                except Exception:
                    # Last resort: known public plate model
                    self.model = _load_model(fallback_hub_id)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load ANPR model from '{weights_path}'. "
                f"Tried hub fallback '{fallback_hub_id}'. Root cause: {e}"
            )
        # EasyOCR default languages; add others if needed (e.g., 'en', 'hi')
        self.reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        self.device = device

        # Try to detect which class indices correspond to plates
        self._plate_class_ids = set()
        try:
            names = self.model.model.names  # dict[int, str]
            for cls_id, name in (names or {}).items():
                if str(name).strip().lower() in PLATE_CLASS_NAMES:
                    self._plate_class_ids.add(int(cls_id))
            # If model has single-class (0) detector for plates, include 0
            if not self._plate_class_ids and len(names or {}) == 1:
                self._plate_class_ids.add(0)
        except Exception:
            # Fallback: assume single-class model
            self._plate_class_ids = {0}

    def run_on_video(self, stored_name: str, params: ANPRParams) -> Dict[str, Any]:
        """
        Runs ANPR on a stored video in data/uploads/{stored_name}.
        Returns a dict compatible with your UI's advanced blocks:
          {
            "ok": true,
            "source_path": "data/uploads/video.mp4",
            "annotated_path": "data/annotated/video.anpr.mp4",
            "json_path": "data/anpr/video.anpr.json",
            "num_frames": 123,
            "num_plates": 7,
            "meta": { ... }
          }
        """
        src_path = (UPLOADS_DIR / stored_name).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Source video not found: {src_path}")

        _ensure_dir(ANPR_JSON_DIR)
        _ensure_dir(ANNOTATED_DIR)

        stem = Path(stored_name).stem
        json_path = (ANPR_JSON_DIR / f"{stem}.anpr.json").resolve()
        annotated_path = (ANNOTATED_DIR / f"{stem}.anpr.mp4").resolve()

        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {src_path}")

        # Video meta
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Writer (mp4v; if you need H.264 use ffmpeg step later)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open VideoWriter for annotated output.")

        detections: List[Dict[str, Any]] = []
        frame_idx = 0
        max_frames = params.max_frames if params.max_frames and params.max_frames > 0 else total_frames

        try:
            while True:
                if frame_idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                # Run YOLO inference
                results = self.model.predict(
                    source=frame,
                    conf=params.conf,
                    iou=params.iou,
                    verbose=False,
                    device=self.device or "cpu"
                )

                # Iterate boxes
                for r in results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    try:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        continue

                    for (x1, y1, x2, y2), sc, cls_id in zip(boxes, scores, classes):
                        if self._plate_class_ids and int(cls_id) not in self._plate_class_ids:
                            continue
                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        w = max(1, x2i - x1i)
                        h = max(1, y2i - y1i)
                        bbox = [x1i, y1i, w, h]

                        # Crop for OCR
                        crop = frame[max(0, y1i):max(0, y1i)+h, max(0, x1i):max(0, x1i)+w]
                        plate_text, ocr_conf = _best_easyocr_text(self.reader, crop, params.ocr_min_conf)

                        # Store detection
                        detections.append({
                            "frame_idx": frame_idx,
                            "bbox": bbox,
                            "conf": float(sc),
                            "label": plate_text or "plate",
                            "ocr_conf": float(ocr_conf),
                            "cls_id": int(cls_id),
                        })

                        # Draw overlay
                        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                        label_txt = (plate_text if plate_text else "plate")
                        cv2.putText(
                            frame,
                            f"{label_txt} {sc:.2f}",
                            (x1i, max(0, y1i - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                writer.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        # Build JSON
        payload: Dict[str, Any] = {
            "ok": True,
            "source_path": _norm_rel(src_path),
            "annotated_path": _norm_rel(annotated_path),
            "json_path": _norm_rel(json_path),
            "num_frames": int(min(max_frames, total_frames)),
            "num_plates": int(sum(1 for d in detections if (d.get("label") and d["label"] != "plate"))),
            "meta": {
                "source_file": stored_name,
                "video_width": width,
                "video_height": height,
                "fps": float(fps),
                "total_frames": total_frames,
                "params": {
                    "conf": params.conf,
                    "iou": params.iou,
                    "ocr_min_conf": params.ocr_min_conf,
                    "max_frames": params.max_frames,
                    "device": params.device,
                },
            },
            "detections": detections,
            "tracks": [],  # not doing tracking in ANPR v1; can be extended to track plates
        }

        # Save JSON
        _ensure_dir(json_path.parent)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return payload


# Convenience function used by router
def run_anpr_by_filename(
    stored_name: str,
    conf: float = 0.25,
    iou: float = 0.45,
    ocr_min_conf: float = 0.4,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    svc = ANPRService(weights_path=ANPR_YOLO_WEIGHTS, device=device)
    params = ANPRParams(conf=conf, iou=iou, ocr_min_conf=ocr_min_conf, max_frames=max_frames, device=device)
    return svc.run_on_video(stored_name, params)
