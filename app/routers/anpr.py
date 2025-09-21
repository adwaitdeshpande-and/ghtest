# app/routers/anpr.py
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Path, Query, Body
from pydantic import BaseModel

from app.services.anpr import run_anpr_by_filename

router = APIRouter(tags=["anpr"])

class ANPRRequest(BaseModel):
    conf: float = 0.25
    iou: float = 0.45
    ocr_min_conf: float = 0.40
    max_frames: Optional[int] = None
    device: Optional[str] = None  # "cpu" or "cuda"


@router.post("/anpr/by-filename/{stored_name}")
def anpr_by_filename(
    stored_name: str = Path(..., description="Filename stored under data/uploads"),
    req: ANPRRequest = Body(default=ANPRRequest()),
):
    """
    Run ANPR (license-plate detection + OCR) on a previously uploaded video.

    Returns a dict with:
      ok: bool
      source_path: project-relative path to the source file
      annotated_path: project-relative path to an annotated MP4
      json_path: project-relative path to the ANPR JSON
      num_frames: number of frames processed
      num_plates: count of detections with non-empty plate_text
      meta: info like fps, width/height, params
      detections: list of {frame_idx, bbox[x,y,w,h], conf, label(plate_text or 'plate'), ocr_conf, cls_id}
      tracks: [] (reserved for future tracking)
    """
    try:
        out = run_anpr_by_filename(
            stored_name=stored_name,
            conf=req.conf,
            iou=req.iou,
            ocr_min_conf=req.ocr_min_conf,
            max_frames=req.max_frames,
            device=req.device,
        )
        return out
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # You can log e with traceback here if desired
        raise HTTPException(status_code=500, detail=f"ANPR failed: {e}")
