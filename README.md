# AI-BASED CCTV & DIGITAL MEDIA FORENSIC ANALYSIS TOOL

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red)](https://streamlit.io/)  
[![FastAPI](https://img.shields.io/badge/fastapi-backend-green)](https://fastapi.tiangolo.com/)  

Prototype system developed for **Goa Police Hackathon 2025**.  
It combines automated **object/face detection, tracking, metadata verification, and forensic reporting** into a single workflow.  
Backend runs on **FastAPI**, frontend on **Streamlit**.

---

## Features
- YOLOv8-based **object and face detection** (`yolov8n.pt`, `yolov8l.pt` included).
- **Tracking** of persons/objects across frames.
- **Metadata & integrity checks** (EXIF, codec info, error-level analysis).
- **Forensic reports** generated in JSON + PDF with QR codes and ELA images.
- Modular design with backend (API), frontend (UI), and data storage.

---

## Project Structure
```
app/        # FastAPI backend + services
ui/         # Streamlit dashboard
data/       # Detections, frames, tracks, reports
requirements.txt
yolov8n.pt  # YOLOv8 nano model (fast, lightweight)
yolov8l.pt  # YOLOv8 large model (higher accuracy)
```

---

## Setup

### 1. Clone the repository
```bash
   git clone https://github.com/adwaitdeshpande-and/ghtest.git
   cd ghtest
```

### 2. Create & activate virtual environment
```bash
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows
   .\.venv\Scripts\Activate.ps1
```

### 3. Windows users

#### Install FFmpeg
For video processing, install FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html):

1. Download the **full build** for Windows.  
2. Extract the archive (e.g., `C:\ffmpeg`).  
3. Add the `bin` folder (e.g., `C:\ffmpeg\bin`) to your system **Environment Variables → Path**.  
4. Confirm installation:  
```bash
  ffmpeg -version
```

#### Install Microsoft Visual C++ Build Tools
The `insightface` package requires Microsoft Visual C++ to compile C extensions on Windows.  
You can download the required build tools from the official site:  
[https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Install and restart your system after installation. This ensures dependencies like `insightface` can work correctly.

### 4. Install Python dependencies
```bash
  pip install -r requirements.txt
```

---

## Running the project

Open **two terminals** in the project root.

**Terminal 1 → Start FastAPI backend**
```bash
   # Activate virtual environment
   # Linux / macOS
   source .venv/bin/activate
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   # Ensure you are in project root
   cd ghtest
   
   # Start backend
   uvicorn app.main:app --reload
```
Backend runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

**Terminal 2 → Start Streamlit dashboard**
```bash
   # Activate virtual environment
   # Linux / macOS
   source .venv/bin/activate
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   streamlit run ui/dashboard.py
```
Dashboard runs at: [http://localhost:8501](http://localhost:8501)

---

## Technical Architecture

The system is organized in a modular pipeline:  

```
[ Streamlit UI ]  →  [ FastAPI Backend ]  →  [ Services Layer ]  →  [ Vision Modules ]  →  [ Data Storage ]
```

- **UI (ui/)** → Streamlit dashboard for uploading media, controlling detection, viewing results, and exporting reports.  
- **Backend (app/)** → FastAPI app that routes requests from UI to the services.  
- **Services (app/services/)** → Implements business logic such as detection, face analysis, metadata extraction, report generation.  
- **Vision (app/vision/)** → Core computer vision modules for detection and tracking (YOLOv8 + trackers).  
- **Data (data/)** → Stores outputs including detections, frames, tracks, and forensic reports.  

---

## Supported Features

### Object & Person Detection
- Uses **YOLOv8 (COCO pretrained)** for detecting 80 object categories.  
- Lightweight (`yolov8n.pt`) and accurate (`yolov8l.pt`) models included.  

### Tracking
- Multi-object tracking with frame-by-frame consistency.  
- Generates cropped frames for each track under `data/frames/tracks/`.  

### Face Detection & Embeddings
- Uses specialized models (ArcFace/InsightFace) for extracting normalized embeddings.  
- Enables potential matching across different frames or videos.  

### Metadata Extraction
- Extracts video properties (codec, resolution, duration, bitrate).  
- Pulls EXIF metadata from images if available.  

### Error Level Analysis (ELA)
- Detects potential tampering by recompressing and highlighting anomalies.  
- Stores ELA results in `data/reports/assets/`.  

### Forensic Report Generation
- Produces **JSON (raw)** and **PDF (human-readable)** reports.  
- Reports include case IDs, evidence hashes (SHA-256), and QR codes for verification.  
- Demo signature block included in PDF to simulate digital signing.  

---

## Workflow

1. Upload a video/image via the dashboard.  
2. System runs detection + tracking.  
3. Metadata + integrity analysis is performed.  
4. **Report is generated** in `data/reports/`:
   - **JSON** → raw results.  
   - **PDF** → formatted forensic report with case ID, findings, signatures.  
   - **Assets** → QR codes and ELA thumbnails.  

Example report output (simplified):
```
   Case ID: auto-generated
   Findings: Objects/Faces tracked
   Metadata: codec, resolution, hashes
   Report files: PDF + JSON + QR + ELA images
```

---

## Models

- `yolov8n.pt` → Fast, low-resource (included).  
- `yolov8l.pt` → Larger, more accurate (included).  
- Both models are **pretrained on the COCO dataset** (80 object classes).  
- Other YOLOv8 weights can be downloaded from [Ultralytics](https://github.com/ultralytics/ultralytics).

---

## Reports

- PDF report contains case header, evidence metadata, detections, and QR-code integrity verification.  
- ELA images highlight possible tampering.  
- Reports stored in: `data/reports/`.

---

## Notes
- Python **3.12** recommended.  
- Local setup supported on **Windows & Linux**.  
- For research/educational use only.  
