# Deploying the Person Re-ID & Tracking console

## What you have

```
reid_app/
├── app.py                  # Streamlit frontend (the console UI)
├── reid_core.py             # Refactored pipeline logic (no Colab/UI code)
├── requirements.txt         # Python deps (pip)
├── packages.txt             # OS-level deps (apt, for Streamlit Cloud)
└── .streamlit/config.toml   # Theme + upload size
```

`app.py` never touches `cv2`/`torch` logic directly — it calls into
`reid_core.py`, which is a straight refactor of your original notebook
script (same detection/tracking/re-ID/distance math, same drawing style),
just with Colab uploads and `print()` swapped for function arguments and a
progress callback.

## ⚠️ Read this before you pick a host

This pipeline runs **YOLOv8 + DeepSORT + FaceNet + a ResNet50 encoder, per
frame, on every person in the frame.** That's real compute. A few things
that matter when choosing where to deploy:

| Constraint | Why it matters |
|---|---|
| **No GPU on Streamlit Community Cloud** | Everything runs on CPU. `yolov8n.pt` is usable; `yolov8m.pt` will be slow (seconds per frame, not real-time) on a free instance. |
| **~1 GB RAM on Streamlit Community Cloud's free tier** | Loading YOLO + FaceNet + ResNet50 + DeepSORT simultaneously is tight. Use `yolov8n.pt` and expect occasional "Oh no" reboots if you process a long video. |
| **200 MB upload limit, hard-capped by the platform** | Streamlit Community Cloud overrides `maxUploadSize` and will not accept video files bigger than ~200 MB, no matter what `.streamlit/config.toml` says. This is the single biggest practical blocker for a video app on the free tier. |
| **Ephemeral filesystem** | Every redeploy/restart wipes the container. The first YOLO/FaceNet weights download happens again on cold start (~1–2 min). |

**Bottom line:** Streamlit Community Cloud is great for a *demo* with short
clips (a few seconds, `yolov8n`, `skip_frames=2-3`). For anything resembling
production use — full-length security footage, the `m` model, multiple
concurrent users — use Hugging Face Spaces (GPU tier) or your own
Docker/VM, both covered below.

---

## Option A — Streamlit Community Cloud (free, fastest to set up, best for a demo)

1. **Push this folder to a public (or private, on paid GitHub) GitHub repo.**
   ```bash
   cd reid_app
   git init
   git add .
   git commit -m "Person Re-ID Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   Do **not** commit large `.pt` weight files or sample videos — `ultralytics`
   auto-downloads `yolov8n.pt`/`yolov8s.pt`/`yolov8m.pt` on first use, and
   `facenet-pytorch` auto-downloads the `vggface2` weights the same way.

2. **Go to** [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick the repo, branch `main`, main file path `app.py`.

3. Streamlit Cloud automatically reads `requirements.txt` (Python deps) and
   `packages.txt` (apt deps — this is what gets you `ffmpeg` and the
   `libGL`/`libglib` libraries OpenCV needs on a headless box). No extra
   config needed.

4. Click **Deploy**. First boot will take a few minutes (installing torch,
   downloading model weights). Subsequent restarts within the same session
   are faster; a full redeploy re-downloads weights since the filesystem is
   ephemeral.

5. In the sidebar, set **YOLO model size → `yolov8n.pt`**, bump
   **"Process every Nth frame"** to 2–3, and use **"Limit processing to
   first N seconds"** for a quick demo before committing to a full clip.

**If the app crashes / greys out:** that's almost always RAM. Open the
"Manage app" menu → **Reboot app**, and reduce frame size / use the `n`
model.

---

## Option B — Hugging Face Spaces (recommended if you need it to actually run well)

Spaces supports a free CPU tier (same constraints as above) **and** a paid
GPU tier you can switch on per-Space — a much better fit for `yolov8m.pt` +
FaceNet running at a reasonable speed.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   SDK = **Streamlit**.
2. Push the same files (`app.py`, `reid_core.py`, `requirements.txt`,
   `packages.txt`, `.streamlit/`) to the Space's git repo the same way as
   GitHub:
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git push space main
   ```
3. Spaces reads `requirements.txt` and `packages.txt` the same way Streamlit
   Cloud does — no changes needed.
4. In the Space's **Settings → Hardware**, upgrade to a GPU tier (e.g.
   `T4 small`) if you want `yolov8m.pt` to run smoothly. If you do, you can
   delete the `--extra-index-url https://download.pytorch.org/whl/cpu` line
   from `requirements.txt` so torch pulls CUDA-enabled wheels instead.
5. Spaces' free-tier upload limits are friendlier than Streamlit Cloud's,
   but check current limits in HF's docs before relying on large uploads.

---

## Option C — Your own Docker container (a cloud VM, on-prem box, etc.)

This is the option with no upload caps, no shared free-tier RAM limits, and
full control over GPU access. Example `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py reid_core.py .
COPY .streamlit/ .streamlit/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t reid-console .
docker run -p 8501:8501 reid-console
```

For GPU access on a CUDA-capable host, use `nvidia/cuda:12.1-runtime-ubuntu22.04`
as the base image instead, install Python on top, drop the
`--extra-index-url ...cpu` line from `requirements.txt`, and run the
container with `docker run --gpus all ...`. Any VM with an NVIDIA GPU
(AWS `g4dn.xlarge`, GCP `n1-standard-4` + T4, etc.) works.

---

## Running it locally first (do this before deploying anything)

```bash
cd reid_app
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`. Confirm a short test clip works end to
end (reference photos → run → annotated video + report) before pushing to
any host — it's much faster to debug locally than in a redeploy loop.

---

## Tuning for whatever host you pick

- **`yolov8n.pt`** for CPU hosts; only reach for `m` with a GPU.
- **"Process every Nth frame" = 2–3** roughly halves/thirds processing time
  with a small tracking-smoothness cost.
- **"Limit processing to first N seconds"** lets you sanity-check settings
  on a 10–15 s slice before running the full video.
- Calibrate `focal_px` once for your actual camera (instructions are in the
  sidebar tooltip) — without it, the on-screen distance numbers are rough
  estimates only.

## Responsible use

This tool performs face- and body-based re-identification of real people.
Wherever you deploy it: get consent from anyone you intend to track,
restrict access to the app and its outputs (video, embeddings, alert logs
all count as personal/biometric data), and check what your local laws
(e.g. GDPR in the EU, BIPA in Illinois, similar biometric-privacy statutes
elsewhere) require before pointing it at footage of real people you don't
have clear authorization to monitor.
