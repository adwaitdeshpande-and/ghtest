# ui/dashboard.py
from __future__ import annotations
import json, mimetypes, datetime, hashlib, os
from pathlib import Path
from typing import Dict, Any, List, DefaultDict
from collections import defaultdict

import cv2, numpy as np, requests, streamlit as st
from PIL import Image

# --- config ---
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")  # root (no /api) to match /files endpoints
PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="AI CCTV & Digital Media Forensic Tool — MVP", layout="wide")
st.title("AI CCTV & Digital Media Forensic Tool — MVP")
st.caption("Hackathon demo UI (uploads, forensics, motion/YOLO detection, faces index & search).")

# =========================
# HTTP helpers
# =========================
def api_health() -> dict:
    r = requests.get(f"{API_BASE}/health", timeout=5); r.raise_for_status(); return r.json()

def api_ingest(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    r = requests.post(f"{API_BASE}/ingest", files=files, timeout=60); r.raise_for_status(); return r.json()

def _api_base_api() -> str:
    # helper: ensure we can hit either root or /api
    return API_BASE if API_BASE.rstrip("/").endswith("/api") else API_BASE.rstrip("/") + "/api"

def api_list_files() -> dict:
    # prefer existing root route /files; if 404, fallback to /api/files
    url1 = f"{API_BASE}/files"
    url2 = f"{_api_base_api()}/files"
    try:
        r = requests.get(url1, timeout=10)
        if r.status_code == 404:
            r2 = requests.get(url2, timeout=10); r2.raise_for_status(); return r2.json()
        r.raise_for_status(); return r.json()
    except requests.HTTPError:
        # final fallback
        r2 = requests.get(url2, timeout=10); r2.raise_for_status(); return r2.json()

def api_delete_file(stored_name: str, deep: bool) -> dict:
    # try root; if 404, try /api
    params = {"deep": str(deep).lower()}
    url1 = f"{API_BASE}/files/{stored_name}"
    url2 = f"{_api_base_api()}/files/{stored_name}"
    # first attempt
    r = requests.delete(url1, params=params, timeout=30)
    if r.status_code == 404:
        r = requests.delete(url2, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_forensics_files() -> dict:
    r = requests.get(f"{API_BASE}/forensics/files", timeout=10); r.raise_for_status(); return r.json()

def api_forensics_by_filename(stored_name: str) -> dict:
    r = requests.get(f"{API_BASE}/forensics/summary/by-filename/{stored_name}", timeout=60); r.raise_for_status(); return r.json()

def api_detect_options() -> dict:
    r = requests.get(f"{API_BASE}/detect/options", timeout=10); r.raise_for_status(); return r.json()

def api_detect_by_filename(stored_name: str, params: dict) -> dict:
    r = requests.post(f"{API_BASE}/detect/run/by-filename/{stored_name}", json=params, timeout=1200)
    r.raise_for_status(); return r.json()

# Faces APIs
def api_faces_extract_by_filename(stored_name: str, sample_fps: float, max_frames: int) -> dict:
    r = requests.post(f"{API_BASE}/faces/extract/by-filename/{stored_name}", json={"sample_fps": float(sample_fps), "max_frames": int(max_frames)}, timeout=600)
    r.raise_for_status(); return r.json()

def api_faces_index_reset() -> dict:
    r = requests.post(f"{API_BASE}/faces/index/reset", timeout=10); r.raise_for_status(); return r.json()

def api_faces_index_add_by_filename(stored_name: str, sample_fps: float, max_frames: int, max_faces: int) -> dict:
    r = requests.post(f"{API_BASE}/faces/index/add/by-filename/{stored_name}", json={"sample_fps": float(sample_fps), "max_frames": int(max_frames), "max_faces": int(max_faces)}, timeout=1200)
    r.raise_for_status(); return r.json()

def api_faces_index_stats() -> dict:
    r = requests.get(f"{API_BASE}/faces/index/stats", timeout=10); r.raise_for_status(); return r.json()

def api_faces_search(file_bytes: bytes, filename: str, top_k: int) -> dict:
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    data = {"top_k": str(int(top_k))}
    r = requests.post(f"{API_BASE}/faces/search", files=files, data=data, timeout=120); r.raise_for_status(); return r.json()

def _api_base_api() -> str:
    """Return API base ending with /api (for backends mounted under /api)."""
    return API_BASE if API_BASE.rstrip("/").endswith("/api") else API_BASE.rstrip("/") + "/api"

def _sanitize_embedding(vec) -> List[float]:
    """Ensure JSON-serializable list[float] and finite values."""
    out: List[float] = []
    for v in (vec or []):
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f):
            out.append(f)
    return out

def api_faces_search_by_embedding(embedding: List[float], top_k: int) -> dict:
    """
    Tries /faces/search/by-embedding on both roots:
    1) {API_BASE}/faces/search/by-embedding
    2) {API_BASE or .../api}/faces/search/by-embedding
    """
    payload = {"embedding": _sanitize_embedding(embedding), "top_k": int(top_k)}
    if not payload["embedding"]:
        raise ValueError("Empty/invalid embedding vector.")

    # Try root first
    url1 = f"{API_BASE}/faces/search/by-embedding"
    url2 = f"{_api_base_api()}/faces/search/by-embedding"

    r = requests.post(url1, json=payload, timeout=60)
    if r.status_code == 404:
        r = requests.post(url2, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def api_ela_by_filename(stored_name: str, jpeg_quality: int, hi_thresh: int, frame_idx: int | None) -> dict:
    params = {"jpeg_quality": int(jpeg_quality), "hi_thresh": int(hi_thresh)}
    if frame_idx is not None: params["frame_idx"] = int(frame_idx)
    r = requests.post(f"{API_BASE}/forensics/ela/by-filename/{stored_name}", params=params, timeout=120)
    r.raise_for_status(); return r.json()

def api_report_generate(payload: dict) -> dict:
    r = requests.post(f"{API_BASE}/report/generate", json=payload, timeout=120); r.raise_for_status(); return r.json()

# Verify endpoints (UI tab will call /api/verify/pdf)
def api_verify_pdf(pdf_bytes: bytes, public_key_pem: bytes | None, bundle_json: bytes | None) -> dict:
    api_base_api = API_BASE if API_BASE.rstrip("/").endswith("/api") else API_BASE.rstrip("/") + "/api"
    files = {"pdf": ("report.pdf", pdf_bytes, "application/pdf")}
    if public_key_pem: files["public_key_pem"] = ("public.pem", public_key_pem, "application/x-pem-file")
    if bundle_json: files["bundle_json"] = ("bundle.json", bundle_json, "application/json")
    r = requests.post(f"{api_base_api}/verify/pdf", files=files, timeout=120)
    r.raise_for_status(); return r.json()

# =========================
# Local preview helpers
# =========================
def try_show_media(stored_path: str):
    abs_path = (PROJECT_ROOT / stored_path).resolve()
    if not abs_path.exists():
        st.info("File saved, but preview not found on this path."); return
    suffix = abs_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = Image.open(abs_path); st.image(img, caption=abs_path.name, use_container_width=True)
    elif suffix in {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}:
        try: st.video(str(abs_path))
        except Exception:
            with abs_path.open("rb") as f: st.video(f.read())
    else:
        st.write("Preview not supported for this file type.")

def draw_overlays_on_frame(video_path: str, frame_idx: int, dets_this_frame: List[Dict[str, Any]]) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video for preview: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ok, frame = cap.read(); cap.release()
    if not ok or frame is None: raise RuntimeError(f"Could not read frame {frame_idx}")
    for d in dets_this_frame:
        x, y, w, h = d["bbox"]; x2, y2 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d.get('label','obj')} {d.get('conf', 0):.2f}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); return Image.fromarray(frame_rgb)

def group_detections_by_frame(detections: List[Dict[str, Any]]) -> DefaultDict[int, List[Dict[str, Any]]]:
    grouped: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in detections: grouped[int(d["frame_idx"])].append(d)
    return grouped

def crop_from_source(source_file: str, bbox: List[int], frame_idx: int | None) -> Image.Image | None:
    src_path = (PROJECT_ROOT / "data" / "uploads" / source_file).resolve()
    if not src_path.exists(): return None
    x, y, w, h = map(int, bbox)
    if src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(src_path)); 
        if img is None: return None
        H, W = img.shape[:2]; x2, y2 = min(x+w, W-1), min(y+h, H-1)
        crop = img[max(0,y):y2, max(0,x):x2, :]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    else:
        if frame_idx is None: return None
        cap = cv2.VideoCapture(str(src_path)); 
        if not cap.isOpened(): return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)); ok, frame = cap.read(); cap.release()
        if not ok or frame is None: return None
        H, W = frame.shape[:2]; x2, y2 = min(x+w, W-1), min(y+h, H-1)
        crop = frame[max(0,y):y2, max(0,x):x2, :]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# =========================
# Header
# =========================
health_col, files_col = st.columns([1, 2])
with health_col:
    st.subheader("Server Health")
    health = api_health()
    yolo_avail = health.get("yolo_available", "no")
    if health.get("status") == "ok": st.success(f"Backend is running ✅  |  YOLO: {yolo_avail}")
    else: st.error(f"Backend issue: {health.get('status')}")

with files_col:
    st.subheader("Stored Files")
    if st.button("Refresh list", key="refresh_files"): pass
    listing = api_list_files()
    files = listing.get("files", [])
    st.write(f"Count: **{listing.get('count', 0)}**")

    if not files:
        st.info("No files yet. Upload one below.")
    else:
        # nice table with delete controls
        for f in files:
            row = st.container(border=True)
            c1, c2, c3, c4 = row.columns([6, 2, 2, 3])
            ext = Path(f).suffix.lower()
            icon = "🎥" if ext in {".mp4",".mov",".mkv",".avi",".m4v",".webm"} else "🖼️"
            c1.markdown(f"{icon} `{f}`")
            preview = c2.button("Open", key=f"open_{f}")
            deepdel = c3.checkbox("deep", value=False, key=f"deep_{f}", help="Also delete annotated/results tracks etc.")
            delete = c4.button("Delete", type="secondary", key=f"del_{f}")

            if preview:
                try:
                    # show quick preview using Forensics summary path
                    summ = api_forensics_by_filename(f)
                    rel_path = summ.get("path")
                    if rel_path:
                        rp = Path(rel_path)
                        if not rp.is_absolute(): rp = (PROJECT_ROOT / rel_path).resolve()
                        st.caption(f"Path: `{rp}`")
                        try_show_media(str(rp.relative_to(PROJECT_ROOT)))
                except Exception as e:
                    st.warning(f"Could not preview: {e}")

            if delete:
                with row:
                    st.warning(f"Type the exact filename to confirm delete of `{f}`")
                    confirm = st.text_input("Confirm name", key=f"confirm_{f}")
                    go = st.button("Confirm delete", key=f"go_{f}", type="primary")
                    if go and confirm == f:
                        try:
                            res = api_delete_file(f, deepdel)
                            st.success(f"Deleted {res.get('deleted')}. Related: {len(res.get('related_deleted', []))}")
                            st.rerun()
                        except requests.HTTPError as e:
                            detail = e.response.text if getattr(e, "response", None) else ""
                            st.error(f"Delete failed ({getattr(e.response, 'status_code', 'HTTP')}): {detail or ''}")

st.markdown("---")

# =========================
# Tabs
# =========================
tab_upload, tab_forensics, tab_detect, tab_faces, tab_report, tab_verify = st.tabs(
    ["📤 Upload & Hash", "🧪 Forensics Metadata", "🎯 Detection", "👤 Faces (Index & Search)", "📄 Report & ELA", "🔒 Verify PDF"]
)

# --------- Upload Tab ---------
with tab_upload:
    st.subheader("Upload & Hash")
    uploaded = st.file_uploader(
        "Select an image/video to ingest",
        type=["jpg","jpeg","png","bmp","webp","mp4","mov","mkv","avi","m4v","webm"],
        key="uploader_main",
    )
    if uploaded is not None:
        col_a, _ = st.columns([1, 1])
        with col_a:
            st.write("**Selected file:**", uploaded.name)
            if uploaded.type.startswith("image/"): st.image(uploaded, caption="Preview", use_container_width=True)
            elif uploaded.type.startswith("video/"): st.info("Video selected (preview after upload).")
        if st.button("Upload & Compute SHA-256", type="primary", key="upload_and_hash_btn"):
            resp = api_ingest(uploaded.getvalue(), uploaded.name)
            st.success("Uploaded successfully!"); st.json(resp)
            st.markdown("**Local Preview (from stored path):**"); try_show_media(resp.get("stored_path", ""))

# --------- Forensics Tab ---------
with tab_forensics:
    st.subheader("Forensic Metadata Viewer")
    left, right = st.columns([2, 3])
    with left:
        f_listing = api_forensics_files()
        filenames: List[str] = f_listing.get("files", [])
        selected = st.selectbox("Stored filename", options=["-- select --"] + filenames, index=0, key="forensics_file_sel")
        cols = st.columns([1,1])
        load = cols[0].button("Load metadata", type="primary", disabled=(selected == "-- select --"), key="forensics_load_btn")
        do_del = cols[1].button("Delete file", disabled=(selected == "-- select --"), key="forensics_del_btn")
        deep = st.checkbox("Deep delete related artifacts", value=False, key="forensics_del_deep")

        if load: st.session_state["_selected_forensics_file"] = selected
        if do_del and selected != "-- select --":
            try:
                res = api_delete_file(selected, deep)
                st.success(f"Deleted {res.get('deleted')}"); st.rerun()
            except requests.HTTPError as e:
                detail = e.response.text if getattr(e, "response", None) else ""
                st.error(f"Delete failed ({getattr(e.response,'status_code','HTTP')}): {detail or ''}")

    with right:
        sel = st.session_state.get("_selected_forensics_file")
        if sel and sel != "-- select --":
            summary = api_forensics_by_filename(sel)
            st.markdown(f"**File:** `{sel}`")
            base_info_cols = st.columns(4)
            base_info_cols[0].metric("Type", summary.get("kind"))
            base_info_cols[1].metric("MIME", summary.get("mime"))
            size_b = summary.get("size_bytes", 0)
            base_info_cols[2].metric("Size (bytes)", f"{size_b:,}")
            base_info_cols[3].markdown(f"`SHA-256`:\n\n`{summary.get('sha256')}`")
            st.divider()
            rel_path = summary.get("path")
            if rel_path:
                try_show_media(Path(rel_path).relative_to(PROJECT_ROOT) if rel_path.startswith(str(PROJECT_ROOT)) else rel_path)
            details = summary.get("details", {}); kind = summary.get("kind")
            if kind == "image":
                st.markdown("### Image Details")
                img_cols = st.columns(3)
                img_cols[0].write(f"**Width:** {details.get('width')}")
                img_cols[1].write(f"**Height:** {details.get('height')}")
                img_cols[2].write(f"**Format/Mode:** {details.get('format')}/{details.get('mode')}")
                exif = details.get("exif", {})
                if exif: st.markdown("#### EXIF"); st.json(exif)
                if "error" in details: st.warning(details["error"])
            elif kind == "video":
                st.markdown("### Video Details")
                v_cols = st.columns(4)
                v_cols[0].write(f"**Duration (s):** {details.get('duration_sec')}")
                v_cols[1].write(f"**Bitrate:** {details.get('bit_rate')}")
                v_cols[2].write(f"**Resolution:** {details.get('width')}×{details.get('height')}")
                v_cols[3].write(f"**FPS:** {details.get('fps')}")
                st.write(f"**Codec:** {details.get('codec_name')}")
                if "ffprobe" in details:
                    if "ffprobe_error" in details["ffprobe"]:
                        st.error(details["ffprobe"]["ffprobe_error"]); st.info("Tip: Install ffprobe (ffmpeg) for richer video metadata.")
                    else:
                        with st.expander("Raw ffprobe JSON"): st.code(json.dumps(details["ffprobe"], indent=2), language="json")
            else:
                st.info("File type not recognized as image/video. Raw details:"); st.json(details)
            st.divider()
            with st.expander("Full Forensic Summary (raw JSON)"):
                st.code(json.dumps(summary, indent=2), language="json")

# --------- Detection Tab ---------
with tab_detect:
    st.subheader("Detection & Tracking")

    # 1) Simple controls (unchanged)
    f_listing = api_forensics_files(); all_files: List[str] = f_listing.get("files", [])
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
    video_files = [f for f in all_files if Path(f).suffix.lower() in video_exts]
    left, right = st.columns([2, 3])

    with left:
        if not video_files: st.info("Upload a video in the **Upload & Hash** tab first.")
        selected_video = st.selectbox("Select a video (stored filename)", ["-- select --"] + video_files, index=0, key="detect_video_sel")
        defaults = api_detect_options()
        engine = st.selectbox("Engine", options=["motion","yolov8"], index=0 if defaults.get("engine","motion")=="motion" else 1, key="detect_engine_sel")
        sample_fps = st.slider("Sample FPS", 0.1, 30.0, float(defaults.get("sample_fps",2.0)), 0.1, key="detect_samplefps")
        max_frames = st.number_input("Max frames (None = all)", min_value=1, value=int(defaults.get("max_frames",200)), key="detect_maxframes")

        if engine == "motion":
            min_area = st.number_input("Min area (motion contour)", min_value=10, value=int(defaults.get("min_area",500)), key="detect_minarea")
            conf_thresh = None; iou_thresh_track = None
        else:
            st.info("YOLOv8 selected — ensure ultralytics + torch are installed and server restarted.")
            conf_thresh = st.slider("(YOLO) Confidence threshold", 0.01, 0.95, float(defaults.get("conf_thresh",0.25)), 0.01, key="detect_conf")
            iou_thresh_track = st.slider("(YOLO) IOU for track association", 0.1, 0.9, float(defaults.get("iou_thresh_track",0.4)), 0.05, key="detect_iou")
            min_area = None

        run_btn = st.button("Run detection", type="primary", disabled=(selected_video=="-- select --"), key="detect_run_btn")
        if run_btn:
            params = {"engine": engine, "sample_fps": float(sample_fps), "max_frames": int(max_frames)}
            if engine == "motion": params["min_area"] = int(min_area if min_area is not None else 500)
            else:
                params["conf_thresh"] = float(conf_thresh if conf_thresh is not None else 0.25)
                params["iou_thresh_track"] = float(iou_thresh_track if iou_thresh_track is not None else 0.4)
            result = api_detect_by_filename(selected_video, params)
            st.session_state["_detect_result"] = result; st.session_state["_detect_file"] = selected_video
            st.success(f"Detection complete. Engine: {result.get('meta',{}).get('engine','motion')}")

    with right:
        result = st.session_state.get("_detect_result"); sel_file = st.session_state.get("_detect_file")
        if result and sel_file:
            meta = result.get("meta", {}); dets = result.get("detections", []); tracks = result.get("tracks", [])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Processed frames", f"{meta.get('processed_frames', 0)}")
            m2.metric("Detections", f"{len(dets)}")
            m3.metric("Tracks", f"{len(tracks)}")
            m4.metric("Sample FPS", f"{meta.get('fps_sampling', 0)}")

            video_path = meta.get("video_path")
            if not video_path or not Path(video_path).exists():
                st.warning("Video path not accessible for preview. (Still fine; JSON below)")
            else:
                det_by_frame = group_detections_by_frame(dets); frames_available = sorted(det_by_frame.keys())
                if frames_available:
                    frame_choice = st.slider("Preview frame index", min_value=frames_available[0], max_value=frames_available[-1], value=frames_available[0], step=1, key="detect_preview_frame")
                    img_prev = draw_overlays_on_frame(video_path, frame_choice, det_by_frame.get(frame_choice, []))
                    st.image(img_prev, caption=f"Frame {frame_choice} with overlays", use_container_width=True)
                else: st.info("No detections found on sampled frames.")

            st.divider(); st.markdown("### Track Events"); st.json(tracks) if tracks else st.write("No tracks created.")
            st.divider(); st.markdown("### Raw Detection JSON")
            with st.expander("Show JSON"): st.code(json.dumps(result, indent=2), language="json")
            st.download_button(label="Download detection JSON", data=json.dumps(result, indent=2), file_name=f"{Path(sel_file).stem}_detection.json", mime="application/json", key="detect_download_json")

    # 2) Advanced Detection & Tracking (Stored) — inline
    st.markdown("---")
    with st.expander("Advanced Detection & Tracking (Stored uploads: data/uploads)", expanded=True):
        API_BASE_API = API_BASE if API_BASE.rstrip("/").endswith("/api") else (API_BASE.rstrip("/") + "/api")

        def _post_api(url_tail: str, data: Dict[str, Any], files: Dict[str, Any] | None = None) -> Dict[str, Any]:
            url = f"{API_BASE_API}{url_tail}"
            r = requests.post(url, data=data, files=files, timeout=1200)
            if not r.ok: raise RuntimeError(f"{url} failed ({r.status_code}): {r.text}")
            return r.json()

        def _norm_path(p: str | None) -> str | None:
            if not p: return None
            pp = Path(p); 
            if not pp.is_absolute(): pp = (PROJECT_ROOT / p).resolve()
            return str(pp)

        def _media_preview(path_like: str | None, title: str):
            st.markdown(f"**{title}**")
            if not path_like: st.info("No output path."); return
            abspath = _norm_path(path_like)
            if not abspath or not Path(abspath).exists(): st.warning(f"Not found: {path_like}"); return
            suf = Path(abspath).suffix.lower(); st.caption(f"Path: `{abspath}` • exists: **{Path(abspath).exists()}**")
            if suf in {".jpg",".jpeg",".png",".bmp",".webp"}: st.image(abspath, use_container_width=True); return
            if suf in {".mp4",".mov",".m4v",".webm"}:
                try: st.video(abspath)
                except Exception:
                    with open(abspath, "rb") as f: st.video(f.read())
                return
            st.write(f"Saved: `{abspath}`")

        stored_files = [f for f in all_files]
        imgs = [f for f in stored_files if Path(f).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
        vids = [f for f in stored_files if Path(f).suffix.lower() in {".mp4",".mov",".m4v",".webm",".mkv",".avi"}]

        sub = st.tabs(["Detect (Image, stored)", "Detect (Video, stored)", "Track (Video, stored)"])

        with sub[0]:
            pick = st.selectbox("Stored image", ["-- select --"] + imgs, index=0, key="adv_det_img_pick")
            colA, colB, colC = st.columns(3)
            conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="adv_det_img_conf")
            iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="adv_det_img_iou")
            model = colC.selectbox("Model", ["yolov8l.pt","yolov8x.pt"], index=0, key="adv_det_img_model")
            if st.button("Run detection (by-filename)", disabled=(pick=="-- select --"), key="adv_det_img_run"):
                res = _post_api(f"/detect/image/by-filename/{pick}", data={"model_name": model, "conf": conf, "iou": iou})
                st.json(res); _media_preview(res.get("annotated_path"), "Annotated image")

        with sub[1]:
            pick = st.selectbox("Stored video", ["-- select --"] + vids, index=0, key="adv_det_vid_pick")
            colA, colB, colC, colD = st.columns(4)
            conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="adv_det_vid_conf")
            iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="adv_det_vid_iou")
            stride = colC.number_input("Frame stride", 1, 8, 1, step=1, key="adv_det_vid_stride")
            maxf   = colD.number_input("Max frames", 0, 5000, 0, step=50, key="adv_det_vid_maxf", help="0 = no cap")
            model = st.selectbox("Model", ["yolov8l.pt","yolov8x.pt"], index=0, key="adv_det_vid_model")
            if st.button("Run detection (by-filename)", disabled=(pick=="-- select --"), key="adv_det_vid_run"):
                data = {"model_name": model, "conf": conf, "iou": iou, "stride": int(stride), "max_frames": None if int(maxf)==0 else int(maxf)}
                res = _post_api(f"/detect/video/by-filename/{pick}", data=data)
                st.json(res); _media_preview(res.get("annotated_path"), "Annotated video")

        with sub[2]:
            pick = st.selectbox("Stored video", ["-- select --"] + vids, index=0, key="adv_trk_vid_pick")
            colA, colB, colC = st.columns(3)
            conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="adv_trk_conf")
            iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="adv_trk_iou")
            minlen = colC.number_input("Min track length (frames)", 1, 100, 5, step=1, key="adv_trk_minlen")
            model = st.selectbox("Model", ["yolov8l.pt","yolov8x.pt"], index=0, key="adv_trk_model")
            device = st.selectbox("Device", ["cpu"], index=0, key="adv_trk_device")
            if st.button("Run tracking (by-filename)", disabled=(pick=="-- select --"), key="adv_trk_run"):
                data = {"model_name": model, "conf": conf, "iou": iou, "min_track_len": int(minlen), "device": None if device=="cpu" else device}
                res = _post_api(f"/track/video/by-filename/{pick}", data=data)
                st.json(res); _media_preview(res.get("annotated_path"), "Annotated tracking video")
                # optional: pull findings into report session
                try:
                    fj = res.get("findings_json_path")
                    if fj:
                        fpath = Path(_norm_path(fj))
                        if fpath.exists():
                            payload = json.loads(fpath.read_text(encoding="utf-8"))
                            findings = payload.get("findings", [])
                            if findings and st.button("Add these Findings to the Report", key=f"adv_trk_add_{hash(fj)}"):
                                st.session_state.setdefault("rep_findings", []).extend(findings)
                                st.success(f"Added {len(findings)} finding(s) to report session.")
                except Exception as e:
                    st.warning(f"Could not read findings: {e}")

# --------- Faces Tab ---------
with tab_faces:
    st.subheader("Faces: Build Index & Search")

    # fetch list of stored files
    f_listing = api_forensics_files()
    all_files: List[str] = f_listing.get("files", [])

    left, right = st.columns([2, 3])

    # ---------------- Left: Index + Preview ----------------
    with left:
        st.markdown("#### 1) Index Builder")
        st.caption("Add faces from a stored image/video to the in-memory index.")
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".webm"}
        video_exts = {".mp4", ".mov", ".mkv", ".avi"}
        faceable_files = [f for f in all_files if Path(f).suffix.lower() in image_exts.union(video_exts)]

        idx_file = st.selectbox(
            "Stored filename",
            options=["-- select --"] + faceable_files,
            index=0,
            key="faces_index_sel",
        )
        sample_fps = st.slider("Sample FPS (video only)", 0.1, 10.0, 1.0, 0.1, key="faces_samplefps")
        max_frames = st.number_input("Max frames to sample (video)", min_value=1, value=50, key="faces_maxframes")
        max_faces = st.number_input("Max faces to add", min_value=1, max_value=10000, value=500, key="faces_maxfaces")

        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Add faces to index", type="primary", disabled=(idx_file == "-- select --"), key="faces_add_btn"):
            resp = api_faces_index_add_by_filename(
                idx_file, sample_fps=float(sample_fps), max_frames=int(max_frames), max_faces=int(max_faces)
            )
            st.success(f"Added: {resp.get('added', 0)}  |  Total in index: {resp.get('total', 0)}")
            st.session_state["_face_index_stats"] = api_faces_index_stats()

        if col_btn2.button("Reset index", type="secondary", key="faces_reset_btn"):
            api_faces_index_reset()
            st.session_state["_face_index_stats"] = {"count": 0}
            st.info("Face index reset.")

        # stats
        stats = st.session_state.get("_face_index_stats")
        if not stats:
            stats = api_faces_index_stats()
            st.session_state["_face_index_stats"] = stats
        st.metric("Index size (faces)", stats.get("count", 0))

        st.markdown("---")
        st.markdown("#### 2) Extract (Preview Only, optional)")
        st.caption("See detected faces from a stored file (does not change index).")
        prev_file = st.selectbox(
            "Pick file to preview faces",
            options=["-- select --"] + faceable_files,
            index=0,
            key="faces_preview_sel",
        )
        if st.button("Preview detected faces", key="faces_preview_btn"):
            if prev_file == "-- select --":
                st.warning("Pick a stored file first.")
            else:
                out = api_faces_extract_by_filename(prev_file, sample_fps=float(sample_fps), max_frames=int(max_frames))
                faces = out.get("faces", [])
                st.session_state["_faces_preview_list"] = faces
                st.session_state["_faces_preview_source"] = prev_file
                st.write(f"Found faces: **{len(faces)}** (showing up to 12)")

        # Render preview gallery with "Use this face as query"
        faces_prev = st.session_state.get("_faces_preview_list", [])
        faces_src = st.session_state.get("_faces_preview_source")
        if faces_prev and faces_src:
            show_n = min(12, len(faces_prev))
            for i in range(show_n):
                f = faces_prev[i]
                crop = crop_from_source(faces_src, f.get("bbox", [0, 0, 0, 0]), f.get("frame_idx"))
                c1, c2 = st.columns([1, 1])
                with c1:
                    if crop is not None:
                        st.image(
                            crop,
                            caption=f"Face {i+1} | score≈{f.get('det_score', 0):.2f} | sharp={f.get('sharpness', 0):.1f}",
                            use_container_width=True,
                        )
                    else:
                        st.text(f"Face {i+1} (preview unavailable)")
                with c2:
                    st.write(f"**BBox:** {f.get('bbox')}")
                    st.write(f"**Frame:** {f.get('frame_idx')}")
                    st.write(f"**Timestamp (s):** {f.get('ts_sec')}")
                    topk = st.slider(f"Top-K (Face {i+1})", 1, 20, 5, 1, key=f"topk_face_{i}")
                    if st.button("Use this face as query", key=f"use_face_{i}", type="primary"):
                        embedding = f.get("normed_embedding", [])
                        if not embedding:
                            st.warning("No embedding attached to this face.")
                        else:
                            # Ensure index has entries first
                            stats = st.session_state.get("_face_index_stats") or api_faces_index_stats()
                            if (stats or {}).get("count", 0) <= 0:
                                st.warning("Face index is empty. Add faces to the index first (left pane).")
                            else:
                                try:
                                    # Many backends expect a specific dimension (e.g., 512). We sanitize but don't force-resize.
                                    resp = api_faces_search_by_embedding(embedding, int(topk))
                                    st.session_state[f"_face_query_result_{i}"] = resp
                                    st.success(f"Search ran for Face {i+1}. See results below.")
                                except requests.HTTPError as e:
        # Show server’s 4xx/5xx text so you can see the reason (dim mismatch / no face / bad payload, etc.)
                                    status = getattr(e.response, "status_code", "HTTP")
                                    detail = ""
                                    try:
                                        detail = e.response.text
                                    except Exception:
                                        pass
                                    st.error(f"Search failed ({status}). {detail or 'See server logs.'}")
                                except ValueError as ve:
                                    st.warning(f"{ve}")
                                except Exception as ex:
                                    st.error(f"Search failed: {ex}")


                # Show results for this face if present
                res = st.session_state.get(f"_face_query_result_{i}")
                if res:
                    results = res.get("results", [])
                    st.markdown(f"**Results for Face {i+1} (top {res.get('top_k', len(results))}):**")
                    if not results:
                        st.info("No matches.")
                    else:
                        for j, r in enumerate(results):
                            rc1, rc2 = st.columns([1, 1])
                            with rc1:
                                cropm = crop_from_source(r.get("source_file", ""), r.get("bbox", [0, 0, 0, 0]), r.get("frame_idx"))
                                if cropm is not None:
                                    st.image(cropm, caption=f"Match {j+1} | score={r.get('score', 0):.3f}", use_container_width=True)
                                else:
                                    st.text(f"Match {j+1} (preview unavailable)")
                            with rc2:
                                st.write(f"**Source file:** `{r.get('source_file')}`")
                                st.write(f"**Frame:** {r.get('frame_idx')}")
                                st.write(f"**Timestamp (s):** {r.get('ts_sec')}")
                                st.write(f"**BBox:** {r.get('bbox')}")

    # ---------------- Right: Query Image Search ----------------
    with right:
        st.markdown("#### 3) Search by Query Image")
        st.caption("Upload a face image; we’ll detect the largest face and cosine-search the index.")
        query = st.file_uploader("Query face image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="face_query_uploader")
        top_k = st.slider("Top-K results", 1, 20, 5, 1, key="face_query_topk")
        if st.button("Search", type="primary", disabled=(query is None), key="face_query_searchbtn"):
            try:
                resp = api_faces_search(query.getvalue(), query.name, int(top_k))
                st.success(f"Faces in query: {resp.get('query_faces_found', 0)}")
                results = resp.get("results", [])
                if not results:
                    st.info("No matches returned (index might be empty).")
                else:
                    for i, r in enumerate(results):
                        col_img, col_meta = st.columns([1, 1])
                        with col_img:
                            crop = crop_from_source(r.get("source_file", ""), r.get("bbox", [0, 0, 0, 0]), r.get("frame_idx"))
                            if crop is not None:
                                st.image(crop, caption=f"Match {i+1}  |  score={r.get('score', 0):.3f}", use_container_width=True)
                            else:
                                st.text(f"Match {i+1}: (preview unavailable)")
                        with col_meta:
                            st.write(f"**Source file:** `{r.get('source_file')}`")
                            st.write(f"**Frame:** {r.get('frame_idx')}")
                            st.write(f"**Timestamp (s):** {r.get('ts_sec')}")
                            st.write(f"**BBox:** {r.get('bbox')}")
            except requests.HTTPError as e:
                # Show backend detail (e.g., "Face index is empty" or "No face detected...")
                detail = ""
                try:
                    detail = e.response.text
                except Exception:
                    pass
                st.error(f"Search failed ({e.response.status_code}). {detail or 'See server logs.'}")

# --------- Report & ELA Tab ---------
with tab_report:
    st.subheader("Report & ELA")
    listing = api_forensics_files(); all_files: List[str] = listing.get("files", [])
    image_exts = {".jpg",".jpeg",".png",".bmp",".webp"}; video_exts = {".mp4",".mov",".mkv",".avi",".m4v",".webm"}

    st.markdown("### ELA (Error Level Analysis)")
    col_ela_l, col_ela_r = st.columns([2, 3])
    with col_ela_l:
        pick = st.selectbox("Stored file (image or video)", options=["-- select --"] + all_files, index=0, key="ela_pick")
        jpeg_quality = st.slider("JPEG quality (recompress)", 10, 95, 90, 1, key="ela_q")
        hi_thresh = st.slider("Highlight threshold (0-255)", 0, 255, 40, 1, key="ela_thr")
        frame_idx = None
        if pick != "-- select --" and Path(pick).suffix.lower() in video_exts:
            frame_idx = st.number_input("Frame index for video", min_value=0, value=0, step=1, key="ela_frame")
        run_ela_btn = st.button("Run ELA", type="primary", key="ela_run")

    with col_ela_r:
        if run_ela_btn:
            try:
                res = api_ela_by_filename(pick, jpeg_quality=jpeg_quality, hi_thresh=hi_thresh, frame_idx=frame_idx)
                st.session_state["_ela_last"] = res; st.success("ELA generated.")
            except requests.HTTPError as e:
                detail = ""; 
                try: detail = e.response.text
                except Exception: pass
                st.error(f"ELA failed ({getattr(e.response,'status_code','HTTP')}): {detail or ''}")

        ela_res = st.session_state.get("_ela_last")
        if ela_res:
            st.write("**Stats:**"); st.json({k: v for k, v in ela_res.items() if k not in {"original_path","ela_path"}})
            def _load(path_rel: str): return Image.open((PROJECT_ROOT / path_rel).resolve())
            try:
                c1, c2 = st.columns(2)
                with c1: st.image(_load(ela_res["original_path"]), caption="Original", use_container_width=True)
                with c2: st.image(_load(ela_res["ela_path"]), caption="ELA heatmap", use_container_width=True)
            except Exception as e: st.warning(f"Could not preview ELA images: {e}")
            if st.button("Add this ELA image to report thumbnails", key="ela_add_thumb"):
                thumbs = st.session_state.get("_report_ela_thumbs", []); thumbs.append(ela_res["ela_path"])
                st.session_state["_report_ela_thumbs"] = thumbs; st.info(f"Added. Thumbs in report: {len(thumbs)}")

    st.divider(); st.markdown("### Build PDF Report")
    # Header
    h_case_id = st.text_input("Case ID", key="rep_case_id")
    h_investigator = st.text_input("Investigator", key="rep_investigator")
    h_station = st.text_input("Station/Unit", key="rep_station")
    h_contact = st.text_input("Contact", key="rep_contact")
    h_notes = st.text_area("Case Notes", key="rep_notes")

    # Evidence
    ev_pick = st.multiselect("Select evidence files", options=all_files, default=[], key="rep_ev_pick")
    evidence_payload: List[Dict[str, Any]] = []
    for f in ev_pick:
        summ = api_forensics_by_filename(f); det = summ.get("details", {})
        dur = float(det.get("duration_sec", 0.0) or 0.0) if summ.get("kind") == "video" else None
        camera_id = st.text_input(f"Camera ID for {f}", value="unknown", key=f"rep_cam_{f}")
        evidence_payload.append({"filename": f, "sha256": summ.get("sha256",""), "ingest_time": datetime.datetime.utcnow().isoformat()+"Z", "camera_id": camera_id, "duration": dur})

    # Findings
    st.markdown("**Findings (optional)**"); st.caption("Add one or more events. Representative frame can be any stored image path (or leave blank).")
    if "rep_findings" not in st.session_state: st.session_state["rep_findings"] = []
    if st.button("Add empty finding", key="rep_f_add"):
        st.session_state["rep_findings"].append({"time_window":"", "track_id":None, "object_type":"person", "representative_frame_path":"", "bbox":None, "matched_offender_id":None, "matched_offender_name":None, "similarity_score":None, "verification_status":"unverified"})
    new_list = []
    for idx, f in enumerate(st.session_state["rep_findings"]):
        st.markdown(f"**Event #{idx+1}**"); c1, c2 = st.columns(2)
        with c1:
            f["time_window"] = st.text_input("Time window", value=f.get("time_window",""), key=f"rep_f_tw_{idx}")
            f["track_id"] = st.number_input("Track ID (optional)", min_value=0, value=int(f.get("track_id") or 0), step=1, key=f"rep_f_tid_{idx}") if f.get("track_id") is not None else None
            f["object_type"] = st.selectbox("Object type", ["person","car","motion","object"], index=["person","car","motion","object"].index(f.get("object_type","person")), key=f"rep_f_obj_{idx}")
            f["representative_frame_path"] = st.text_input("Representative image path (optional)", value=f.get("representative_frame_path",""), key=f"rep_f_repr_{idx}")
        with c2:
            f["matched_offender_id"] = st.text_input("Matched offender ID (optional)", value=f.get("matched_offender_id") or "", key=f"rep_f_oid_{idx}") or None
            f["matched_offender_name"] = st.text_input("Matched offender Name (optional)", value=f.get("matched_offender_name") or "", key=f"rep_f_oname_{idx}") or None
            sim = st.text_input("Similarity score (0..1, optional)", value="" if f.get("similarity_score") is None else str(f.get("similarity_score")), key=f"rep_f_sim_{idx}")
            f["similarity_score"] = float(sim) if sim.strip() else None
            f["verification_status"] = st.selectbox("Verification status", ["verified","unverified","unknown"], index=["verified","unverified","unknown"].index(f.get("verification_status","unverified")), key=f"rep_f_ver_{idx}")
        bx = st.text_input("BBox x,y,w,h (optional)", value=(",".join(map(str, f.get("bbox") or [])) if f.get("bbox") else ""), key=f"rep_f_bbox_{idx}")
        if bx.strip():
            try: x,y,w,h = [int(v.strip()) for v in bx.split(",")]; f["bbox"] = [x,y,w,h]
            except Exception: st.warning("Invalid bbox format; expected x,y,w,h"); f["bbox"] = None
        if st.button("Remove this event", key=f"rep_f_rm_{idx}"): pass
        else: new_list.append(f); st.markdown("---")
    st.session_state["rep_findings"] = new_list

    st.markdown("**Forensics block**")
    meta_summary = api_forensics_by_filename(ev_pick[0]) if ev_pick else {}
    thumbs = st.session_state.get("_report_ela_thumbs", []); st.write(f"ELA thumbnails queued: **{len(thumbs)}**")
    deepfake = st.slider("Deepfake heuristic score (0..1; optional)", 0.0, 1.0, 0.0, 0.01, key="rep_df")

    if st.button("Generate PDF Report", type="primary", key="rep_generate"):
        try:
            payload = {"header":{"case_id":h_case_id,"investigator":h_investigator,"station_unit":h_station,"contact":h_contact,"case_notes":h_notes},
                       "evidence":evidence_payload,
                       "findings":st.session_state["rep_findings"],
                       "forensics":{"metadata_summary":meta_summary,"tamper_flags":[],"ela_thumbnails":thumbs,"deepfake_score":float(deepfake)},
                       "bundle_json":{}}
            out = api_report_generate(payload); st.success("Report generated.")
            st.json({k: v for k,v in out.items() if k not in {"pdf_path","json_path","qr_path"}})
            _abs = lambda p: str((PROJECT_ROOT / p).resolve())
            st.markdown(f"[Download PDF]({_abs(out['pdf_path'])})")
            st.markdown(f"[Download JSON bundle]({_abs(out['json_path'])})")
            st.markdown(f"[QR image]({_abs(out['qr_path'])})")
        except requests.HTTPError as e:
            detail = e.response.text if getattr(e, "response", None) else ""
            st.error(f"Report generation failed ({getattr(e.response,'status_code','HTTP')}): {detail or ''}")

# --------- Verify PDF Tab (right-most) ---------
with tab_verify:
    st.subheader("Verify Report (PDF)")
    st.caption("Upload a generated PDF. Optionally add a public key (PEM) and/or the report’s JSON bundle for cross-check.")
    c1, c2 = st.columns([2,1])
    with c1:
        up_pdf = st.file_uploader("Report PDF", type=["pdf"], key="ver_pdf")
        up_pem = st.file_uploader("Public key (PEM, optional)", type=["pem"], key="ver_pem")
        up_json = st.file_uploader("Bundle JSON (optional)", type=["json"], key="ver_json")
        run = st.button("Run verification", type="primary", disabled=(up_pdf is None))
    with c2:
        api_base_api = API_BASE if API_BASE.rstrip("/").endswith("/api") else API_BASE.rstrip("/") + "/api"
        st.write("**API_BASE**:", f"`{api_base_api}`")

    if run and up_pdf is not None:
        try:
            res = api_verify_pdf(up_pdf.getvalue(), up_pem.getvalue() if up_pem else None, up_json.getvalue() if up_json else None)
            ok = res.get("ok"); st.success("Verification PASSED") if ok else st.error("Verification FAILED")
            st.json(res)
            st.code(f"printed report_sha256: {res.get('printed_sha256')}\ncomputed sha256:     {res.get('computed_sha256')}", language="text")
        except requests.HTTPError as e:
            detail = e.response.text if getattr(e, "response", None) else ""
            st.error(f"Verify failed ({getattr(e.response,'status_code','HTTP')}): {detail or ''}")
