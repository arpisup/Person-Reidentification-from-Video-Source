# -*- coding: utf-8 -*-
"""
app.py — Streamlit console for the Person Re-ID & Tracking pipeline.

Run locally:
    streamlit run app.py

See DEPLOYMENT.md for hosting options (Streamlit Cloud, Hugging Face
Spaces, Docker on a GPU box, etc).
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import reid_core as core

# ──────────────────────────────────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ──────────────────────────────────────────────────────────────────────────
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SENTRY · Person Re-ID Console",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# THEME — tactical / surveillance console
#   Void      #0A0E12   background
#   Panel     #121821   cards / surfaces
#   Phosphor  #39FF7A   target acquired (matches the on-video green box)
#   Amber     #FFB454   searching / neutral persons (matches orange box)
#   Alert     #FF4757   danger / close range (matches the red box)
#   Steel     #7C8B99   muted labels
# Display face: Space Grotesk · Body: Inter · Telemetry: JetBrains Mono
# ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --void: #0A0E12;
    --panel: #121821;
    --panel-2: #161D27;
    --line: #232C38;
    --phosphor: #39FF7A;
    --amber: #FFB454;
    --alert: #FF4757;
    --steel: #7C8B99;
    --text: #E8EDF2;
}

html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--void) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background-color: transparent; }
[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

/* Mono telemetry text helper class */
.tele { font-family: 'JetBrains Mono', monospace; }

/* ---- HUD strip (mirrors the on-video draw_hud overlay) ---- */
.hud-strip {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(180deg, #0D131A 0%, #0A0E12 100%);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 22px;
    margin-bottom: 18px;
}
.hud-left { display:flex; align-items:center; gap: 14px; }
.hud-dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink:0; }
.hud-dot.idle { background: var(--steel); }
.hud-dot.scanning { background: var(--amber); animation: pulse 1.4s infinite; }
.hud-dot.acquired { background: var(--phosphor); animation: pulse 1.1s infinite; box-shadow: 0 0 10px var(--phosphor); }
@keyframes pulse { 0%{opacity:1;} 50%{opacity:.35;} 100%{opacity:1;} }
.hud-status { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; letter-spacing:.03em; }
.hud-sub { font-family:'JetBrains Mono',monospace; color: var(--steel); font-size:0.78rem; }
.hud-right { font-family:'JetBrains Mono',monospace; color: var(--steel); font-size:0.8rem; text-align:right; }

/* ---- Section eyebrow / step numbering (real sequence: setup -> run -> report) ---- */
.eyebrow {
    font-family:'JetBrains Mono',monospace; color: var(--phosphor);
    font-size: 0.72rem; letter-spacing:.18em; text-transform:uppercase;
    margin-bottom: 2px;
}

/* ---- Cards ---- */
.console-card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 14px;
}

/* ---- Metric tiles ---- */
[data-testid="stMetric"] {
    background: var(--panel-2); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--steel) !important; font-family:'JetBrains Mono',monospace; font-size:0.72rem !important; letter-spacing:.05em; text-transform:uppercase; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif !important; color: var(--text); }

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {
    background: var(--phosphor); color: #06140C; border: none;
    font-family:'Space Grotesk',sans-serif; font-weight:700;
    border-radius: 8px; padding: 0.55em 1.2em; letter-spacing:.02em;
    transition: filter .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.12); color:#06140C; }
button[kind="secondary"] { background: var(--panel-2) !important; color: var(--text) !important; border: 1px solid var(--line) !important; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-family:'Space Grotesk',sans-serif; font-weight:600; color: var(--steel);
    padding: 10px 4px;
}
.stTabs [aria-selected="true"] { color: var(--phosphor) !important; border-bottom: 2px solid var(--phosphor) !important; }

/* ---- Badges ---- */
.badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-family:'JetBrains Mono',monospace; font-size: 0.72rem; font-weight:700; letter-spacing:.04em; }
.badge.green  { background: rgba(57,255,122,0.12); color: var(--phosphor); border: 1px solid rgba(57,255,122,0.35); }
.badge.amber  { background: rgba(255,180,84,0.12); color: var(--amber); border: 1px solid rgba(255,180,84,0.35); }
.badge.red    { background: rgba(255,71,87,0.12); color: var(--alert); border: 1px solid rgba(255,71,87,0.35); }
.badge.steel  { background: rgba(124,139,153,0.12); color: var(--steel); border: 1px solid rgba(124,139,153,0.35); }

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel-2) !important; border: 1px dashed var(--line) !important; border-radius: 10px;
}
hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────

defaults = {
    "stage": "setup",          # setup -> running -> done
    "workdir": None,
    "ref_paths": [],
    "video_path": None,
    "video_name": None,
    "results": None,
    "preview_h264": None,
    "stop_requested": False,
    "cfg": core.build_cfg(),
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def workdir() -> Path:
    if st.session_state.workdir is None:
        st.session_state.workdir = tempfile.mkdtemp(prefix="reid_")
    return Path(st.session_state.workdir)


@st.cache_resource(show_spinner=False)
def get_models(yolo_model_name: str):
    return core.load_models(yolo_model_name)


# ──────────────────────────────────────────────────────────────────────────
# HUD STRIP  (signature element — same status language as the video overlay)
# ──────────────────────────────────────────────────────────────────────────

def render_hud():
    stage = st.session_state.stage
    device = core.get_device().upper()
    cam = st.session_state.cfg["camera_id"]
    now = time.strftime("%Y-%m-%d  %H:%M:%S")

    if stage == "running":
        dot, status, status_color = "scanning", "SCANNING FEED", "var(--amber)"
    elif stage == "done":
        n_alerts = len(st.session_state.results["alert_log"]) if st.session_state.results else 0
        if n_alerts > 0:
            dot, status, status_color = "acquired", "TARGET ACQUIRED", "var(--phosphor)"
        else:
            dot, status, status_color = "idle", "NO MATCH FOUND", "var(--steel)"
    else:
        dot, status, status_color = "idle", "IDLE — AWAITING INPUT", "var(--steel)"

    st.markdown(f"""
    <div class="hud-strip">
        <div class="hud-left">
            <div class="hud-dot {dot}"></div>
            <div>
                <div class="hud-status" style="color:{status_color};">{status}</div>
                <div class="hud-sub">{cam} · DEVICE:{device}</div>
            </div>
        </div>
        <div class="hud-right">{now}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROL PANEL
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 SENTRY")
    st.markdown(
        "<span class='hud-sub'>PERSON RE-ID & TRACKING CONSOLE</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("<div class='eyebrow'>Detector</div>", unsafe_allow_html=True)
    yolo_choice = st.selectbox(
        "YOLO model size", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="n = fastest (best for CPU / Streamlit Cloud). m = most accurate, needs a GPU to stay real-time.",
    )
    yolo_conf = st.slider("Detection confidence", 0.10, 0.90, 0.40, 0.05)
    yolo_iou = st.slider("Detection IoU (NMS)", 0.10, 0.90, 0.45, 0.05)

    st.markdown("<div class='eyebrow'>Re-ID matching</div>", unsafe_allow_html=True)
    reid_threshold = st.slider("Match threshold", 0.30, 0.95, 0.60, 0.01,
                                help="Cosine similarity needed to flag someone as the target.")
    reid_exit_margin = st.slider("Exit margin", 0.0, 0.40, 0.15, 0.01,
                                  help="Once acquired, a track only loses target status if its rolling average drops below (threshold − margin).")
    face_weight = st.slider("Face vs. body weight", 0.0, 1.0, 0.55, 0.05,
                             help="How much a face match counts vs. a body/clothing match, when a face is visible.")

    with st.expander("Advanced — tracker & quality gates"):
        max_age = st.slider("Tracker max age (frames)", 5, 90, 30)
        n_init = st.slider("Tracker n_init", 1, 10, 3)
        max_cosine_dist = st.slider("Tracker max cosine distance", 0.05, 0.80, 0.40, 0.05)
        min_crop_h = st.slider("Min crop height (px)", 20, 200, 70)
        min_crop_w = st.slider("Min crop width (px)", 10, 100, 28)
        score_hist_len = st.slider("Score smoothing window (frames)", 1, 20, 7)

    with st.expander("Advanced — distance & output"):
        known_height = st.number_input("Assumed person height (m)", 1.0, 2.2, 1.70, 0.01)
        focal_px = st.number_input(
            "Camera focal length (px)", 100, 5000, 750, 10,
            help="distance_m = (known_height_m × focal_px) / bbox_height_px. "
                 "Calibrate: stand a person at a known distance D, note bbox_height_px from a run, "
                 "then focal_px = D × bbox_height_px / known_height_m.",
        )
        camera_id = st.text_input("Camera ID", "CAM-01")
        _default_skip = 2 if (IS_HF_SPACE or core.get_device() == "cpu") else 1
        skip_frames = st.slider("Process every Nth frame", 1, 10, _default_skip,
                                 help="Higher = faster but choppier tracking. Use 2–3 on CPU-only hosts.")
        max_seconds = st.number_input(
            "Limit processing to first N seconds (0 = full video)", 0, 36000, 0,
            help="Useful for a quick demo run on limited hardware before committing to the full video.",
        )

    st.session_state.cfg = core.build_cfg(dict(
        yolo_model=yolo_choice, yolo_conf=yolo_conf, yolo_iou=yolo_iou,
        reid_threshold=reid_threshold, reid_exit_margin=reid_exit_margin,
        face_weight=face_weight, max_age=max_age, n_init=n_init,
        max_cosine_dist=max_cosine_dist, min_crop_height_px=min_crop_h,
        min_crop_width_px=min_crop_w, score_history_len=score_hist_len,
        known_height_m=known_height, focal_px=focal_px, camera_id=camera_id,
        skip_frames=skip_frames,
    ))

    st.markdown("---")
    device_label = core.get_device().upper()
    if IS_HF_SPACE:
        space_id = os.environ.get("SPACE_ID", "unknown")
        st.caption(f"☁️ Running on **Hugging Face Spaces** · `{space_id}`")
    st.caption("🖥️ Running on **%s**" % device_label)
    if core.get_device() == "cpu":
        st.caption("No GPU detected — keep videos short and use the **n** model size.")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

render_hud()

tab_setup, tab_run, tab_report = st.tabs([
    "📡 01 · TARGET SETUP", "🛰️ 02 · RUN SURVEILLANCE", "📊 03 · MISSION REPORT",
])

# ---- TAB 1: SETUP -----------------------------------------------------
with tab_setup:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("<div class='eyebrow'>Reference photos</div>", unsafe_allow_html=True)
        st.markdown("#### Who are we looking for?")
        st.caption("Upload 2–5 clear photos of the target — different angles and lighting make matching far more robust.")
        ref_files = st.file_uploader(
            "Reference images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if ref_files:
            cols = st.columns(min(len(ref_files), 5))
            for i, f in enumerate(ref_files):
                with cols[i % len(cols)]:
                    st.image(f, use_container_width=True)
            st.markdown(f"<span class='badge green'>{len(ref_files)} REFERENCE IMAGE(S) LOADED</span>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='eyebrow'>Footage</div>", unsafe_allow_html=True)
        st.markdown("#### Which video do we search?")
        st.caption("Upload the footage to scan for the target. MP4 / MOV / AVI supported.")
        video_file = st.file_uploader(
            "Video", type=["mp4", "mov", "avi", "mkv"],
            label_visibility="collapsed",
        )
        if video_file:
            st.video(video_file)
            st.markdown(f"<span class='badge green'>{video_file.name}</span>", unsafe_allow_html=True)

    st.markdown("")
    ready = bool(ref_files) and bool(video_file)
    if not ready:
        st.info("Upload at least one reference image and one video to continue.")
    else:
        if st.button("➡️  Stage for surveillance run", type="primary"):
            wd = workdir()
            ref_paths = []
            for i, f in enumerate(ref_files):
                p = wd / f"ref_{i}_{f.name}"
                p.write_bytes(f.getvalue())
                ref_paths.append(str(p))
            vid_path = wd / f"input_{video_file.name}"
            vid_path.write_bytes(video_file.getvalue())

            st.session_state.ref_paths = ref_paths
            st.session_state.video_path = str(vid_path)
            st.session_state.video_name = video_file.name
            st.session_state.stage = "staged"
            st.success("Staged. Head to **02 · RUN SURVEILLANCE** to start the pass.")

# ---- TAB 2: RUN --------------------------------------------------------
with tab_run:
    if not st.session_state.video_path:
        st.info("Nothing staged yet — complete **01 · TARGET SETUP** first.")
    else:
        st.markdown("<div class='eyebrow'>Mission brief</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reference images", len(st.session_state.ref_paths))
        c2.metric("Footage", st.session_state.video_name or "—")
        c3.metric("Detector", st.session_state.cfg["yolo_model"])

        run_col, _ = st.columns([1, 3])
        start = run_col.button("▶  START SURVEILLANCE RUN", type="primary",
                                disabled=(st.session_state.stage == "running"))

        progress_bar = st.progress(0.0)
        status_line = st.empty()
        metric_cols = st.columns(4)
        m_frame = metric_cols[0].empty()
        m_alerts = metric_cols[1].empty()
        m_persons = metric_cols[2].empty()
        m_dist = metric_cols[3].empty()
        preview_slot = st.empty()

        if start:
            st.session_state.stage = "running"
            cfg = dict(st.session_state.cfg)
            cfg["output_path"] = str(workdir() / "reid_output.mp4")
            if max_seconds and max_seconds > 0:
                # convert seconds -> frame cap using the source video's fps
                cap_probe = cv2.VideoCapture(st.session_state.video_path)
                src_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
                cap_probe.release()
                cfg["max_frames"] = int(max_seconds * src_fps)

            with st.spinner("Loading models (first run downloads weights — this can take a minute)…"):
                models_ = get_models(cfg["yolo_model"])

            try:
                with st.spinner("Building reference gallery from uploaded photos…"):
                    gallery = core.build_reference_gallery(models_, st.session_state.ref_paths)
            except ValueError as e:
                st.error(str(e))
                st.session_state.stage = "staged"
                st.stop()

            n_no_face = sum(1 for g in gallery if not g["face_found"])
            if n_no_face:
                st.warning(f"No face detected in {n_no_face} reference image(s) — those will rely on body/clothing similarity only.")

            t0 = time.time()

            def on_frame(frame_no, total, info):
                pct = min(frame_no / total, 1.0) if total else 0.0
                progress_bar.progress(pct)
                elapsed = time.time() - t0
                eta = (elapsed / frame_no) * (total - frame_no) if frame_no and total else 0
                status_line.markdown(
                    f"<span class='tele'>FRAME {frame_no}/{total or '?'} · "
                    f"{pct*100:5.1f}% · ETA {eta:0.0f}s</span>",
                    unsafe_allow_html=True,
                )
                m_frame.metric("Frame", f"{frame_no}/{total or '?'}")
                m_alerts.metric("Target alerts", info["alert_count"])
                m_persons.metric("People in frame", info["n_persons"])
                dist_txt = f"{info['target_dist']:.1f} m" if info["target_found"] and info["target_dist"] > 0 else "—"
                m_dist.metric("Target distance", dist_txt)

                if info["preview_bgr"] is not None:
                    rgb = cv2.cvtColor(info["preview_bgr"], cv2.COLOR_BGR2RGB)
                    preview_slot.image(rgb, channels="RGB", use_container_width=True,
                                        caption="Live annotated preview")

            try:
                results = core.run_pipeline(
                    models_, cfg, gallery, st.session_state.video_path,
                    on_frame=on_frame, preview_every=8,
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.session_state.stage = "staged"
                st.stop()

            st.session_state.results = results
            st.session_state.results["gallery"] = gallery
            st.session_state.stage = "done"
            progress_bar.progress(1.0)
            st.success(f"Run complete in {results['elapsed_sec']:.1f}s — see **03 · MISSION REPORT**.")
            st.rerun()

# ---- TAB 3: REPORT ------------------------------------------------------
with tab_report:
    if not st.session_state.results:
        st.info("No run yet — finish **02 · RUN SURVEILLANCE** first.")
    else:
        res = st.session_state.results
        cfg = st.session_state.cfg
        summary = core.summarize_alert_log(res["alert_log"], cfg)

        st.markdown("<div class='eyebrow'>Outcome</div>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total alerts", summary["total_alerts"])
        k2.metric("Unique tracks flagged", summary["unique_tracks"])
        k3.metric("Avg. confidence", f"{summary['avg_confidence']:.1%}" if summary["avg_confidence"] is not None else "—")
        k4.metric("Closest approach", f"{summary['min_distance_m']:.2f} m" if summary["min_distance_m"] is not None else "—")

        st.markdown("")

        if summary["total_alerts"] == 0:
            st.warning(
                f"No target detections recorded. Try lowering the **match threshold** "
                f"(currently {cfg['reid_threshold']:.2f}) in the sidebar and re-running."
            )
        else:
            df = pd.DataFrame(res["alert_log"])

            colL, colR = st.columns([3, 2], gap="large")
            with colL:
                st.markdown("#### 🎬 Annotated footage")
                if st.session_state.preview_h264 is None:
                    with st.spinner("Encoding browser-playable preview…"):
                        h264_out = str(workdir() / "reid_output_h264.mp4")
                        st.session_state.preview_h264 = core.make_preview_h264(
                            res["output_path"], output_path=h264_out
                        )
                video_path = st.session_state.preview_h264
                if Path(video_path).exists():
                    st.video(video_path)
                st.caption("🟢 target match · 🟠 other person · 🔴 target within 1.5 m")

            with colR:
                st.markdown("#### 📈 Confidence over time")
                st.line_chart(df.set_index("time_sec")["confidence"], height=180)
                st.markdown("#### 📉 Distance over time")
                dist_df = df[df["distance_m"] > 0]
                if not dist_df.empty:
                    st.line_chart(dist_df.set_index("time_sec")["distance_m"], height=180)
                else:
                    st.caption("No valid distance readings (check focal length calibration).")

            st.markdown("#### 🧾 Alert log")
            st.dataframe(
                df[["frame", "time_sec", "track_id", "confidence", "distance_m", "centroid_px"]]
                .rename(columns={
                    "time_sec": "time (s)", "track_id": "track ID",
                    "confidence": "confidence", "distance_m": "distance (m)",
                    "centroid_px": "centroid (px)",
                }),
                use_container_width=True, height=260,
            )

            st.markdown("#### ⬇️ Export")
            e1, e2, e3 = st.columns(3)
            with e1:
                if Path(res["output_path"]).exists():
                    st.download_button(
                        "Annotated video (.mp4)",
                        data=Path(st.session_state.preview_h264 or res["output_path"]).read_bytes(),
                        file_name="reid_output.mp4", mime="video/mp4",
                        use_container_width=True,
                    )
            with e2:
                report_json = json.dumps(res["alert_log"], indent=2).encode()
                st.download_button(
                    "Alert log (.json)", data=report_json,
                    file_name="reid_alert_report.json", mime="application/json",
                    use_container_width=True,
                )
            with e3:
                pkl_path = str(workdir() / "reid_model.pkl")
                core.save_model_pkl(
                    pkl_path, st.session_state.ref_paths, res["gallery"],
                    res["alert_log"], res["track_face_emb"], res["track_body_emb"],
                    res["track_is_target"], cfg,
                )
                st.download_button(
                    "Model + embeddings (.pkl)", data=Path(pkl_path).read_bytes(),
                    file_name="reid_model.pkl", mime="application/octet-stream",
                    use_container_width=True,
                )

        st.markdown("---")
        if st.button("🔄  Start a new run", type="secondary"):
            wd = st.session_state.workdir
            for k in ["stage", "ref_paths", "video_path", "video_name", "results", "preview_h264"]:
                st.session_state[k] = defaults[k]
            if wd:
                shutil.rmtree(wd, ignore_errors=True)
            st.session_state.workdir = None
            st.rerun()
