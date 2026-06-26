# -*- coding: utf-8 -*-
"""
reid_core.py
────────────
Re-ID + tracking pipeline, refactored out of the original Colab notebook
script so it can be imported by a Streamlit app (or anything else).

Differences from the original script:
  • No Colab detection, no `files.upload()`, no argparse — every input is a
    plain function argument (file paths, arrays, or a config dict).
  • Models are NOT loaded at import time. Call `load_models()` once and
    reuse the returned `ReIDModels` object (Streamlit wraps this with
    `st.cache_resource` so it only happens once per session).
  • `run_pipeline()` reports progress through an `on_frame` callback instead
    of `print()`, and accepts a `should_stop` callable so a UI can offer a
    Stop button.
  • Nothing here imports `streamlit` — this module has zero UI dependencies
    and can be unit-tested or reused in a CLI on its own.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = dict(
    # Detection
    yolo_model="yolov8n.pt",   # n = fastest (CPU-friendly), m = more accurate (needs GPU for real-time)
    yolo_conf=0.40,
    yolo_iou=0.45,
    # Re-ID matching
    reid_threshold=0.60,
    reid_exit_margin=0.15,
    score_history_len=7,
    face_weight=0.55,
    min_crop_height_px=70,
    min_crop_width_px=28,
    # Tracker
    max_age=30,
    n_init=3,
    max_cosine_dist=0.40,
    # Processing
    skip_frames=1,
    max_frames=0,               # 0 = process the whole video
    # Distance estimation
    known_height_m=1.70,
    focal_px=750,
    # Output
    camera_id="CAM-01",
    output_path="reid_output.mp4",
    # Colours (BGR, matches the on-screen HUD)
    color_target=(0, 255, 0),
    color_other=(255, 128, 0),
    color_danger=(0, 0, 255),
)


def build_cfg(overrides: Optional[dict] = None) -> dict:
    cfg = dict(DEFAULT_CFG)
    if overrides:
        cfg.update(overrides)
    return cfg


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ReIDModels:
    yolo: YOLO
    mtcnn: MTCNN
    facenet: InceptionResnetV1
    body_encoder: nn.Module
    body_transform: transforms.Compose
    device: str


def load_models(yolo_model_name: str = "yolov8n.pt", device: Optional[str] = None) -> ReIDModels:
    """Loads every model used by the pipeline. Expensive — call once and
    cache the result (e.g. with st.cache_resource in the Streamlit app)."""
    device = device or get_device()

    yolo = YOLO(yolo_model_name)

    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=15,
        thresholds=[0.5, 0.6, 0.6],
        post_process=True,
        select_largest=False,
    )
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    body_encoder = nn.Sequential(*list(resnet50.children())[:-1]).eval().to(device)

    body_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ReIDModels(
        yolo=yolo, mtcnn=mtcnn, facenet=facenet,
        body_encoder=body_encoder, body_transform=body_transform, device=device,
    )


# ──────────────────────────────────────────────────────────────────────────
# EMBEDDING & DISTANCE UTILITIES
# ──────────────────────────────────────────────────────────────────────────

def estimate_distance_m(bbox_height_px: int, cfg: dict) -> float:
    if bbox_height_px <= 0:
        return -1.0
    return round((cfg["known_height_m"] * cfg["focal_px"]) / bbox_height_px, 2)


def extract_face_embedding(m: ReIDModels, img_rgb: np.ndarray):
    try:
        faces = m.mtcnn(Image.fromarray(img_rgb))
    except Exception:
        return None
    if faces is None:
        return None
    face_tensor = faces[0].unsqueeze(0).to(m.device)
    with torch.no_grad():
        emb = m.facenet(face_tensor)
    return F.normalize(emb, p=2, dim=1).cpu().numpy().flatten()


def extract_body_embedding(m: ReIDModels, img_rgb: np.ndarray):
    tensor = m.body_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(m.device)
    with torch.no_grad():
        feat = m.body_encoder(tensor)
    feat = feat.flatten(1)
    return F.normalize(feat, p=2, dim=1).cpu().numpy().flatten()


def extract_embeddings(m: ReIDModels, img_rgb: np.ndarray):
    body_emb = extract_body_embedding(m, img_rgb)
    face_emb = extract_face_embedding(m, img_rgb)
    return face_emb, body_emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_score(face_emb, body_emb, gallery: list, face_weight: float) -> float:
    best = -1.0
    for g in gallery:
        body_sim = cosine_similarity(body_emb, g["body"])
        if face_emb is not None and g["face"] is not None:
            face_sim = cosine_similarity(face_emb, g["face"])
            score = face_weight * face_sim + (1.0 - face_weight) * body_sim
        else:
            score = body_sim
        best = max(best, score)
    return best


def ema_update(old_emb, new_emb, alpha=0.25):
    updated = (1 - alpha) * old_emb + alpha * new_emb
    return updated / (np.linalg.norm(updated) + 1e-8)


def crop_is_good_quality(x1, y1, x2, y2, cfg: dict) -> bool:
    h, w = y2 - y1, x2 - x1
    return h >= cfg["min_crop_height_px"] and w >= cfg["min_crop_width_px"]


# ──────────────────────────────────────────────────────────────────────────
# DRAWING / ANNOTATION HELPERS  (identical visual language to the source
# script, parametrised on `cfg` instead of reading a module-level global)
# ──────────────────────────────────────────────────────────────────────────

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.45):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_corner_box(img, x1, y1, x2, y2, color, arm=18, thickness=3):
    for sx, sy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                            (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (sx, sy), (sx + dx * arm, sy), color, thickness)
        cv2.line(img, (sx, sy), (sx, sy + dy * arm), color, thickness)


def annotate_target(frame, tid, bbox, sim, dist_m, cam_id, frame_no, fps, cfg):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    color = cfg["color_danger"] if (0 < dist_m < 1.5) else cfg["color_target"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_corner_box(frame, x1, y1, x2, y2, color)
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 22, 2)

    top = max(y1 - 70, 0)
    alpha_rect(frame, x1, top, x1 + 260, y1, (0, 60, 0))

    sec = frame_no / max(fps, 1)
    dist_str = f"{dist_m:.1f} m" if dist_m > 0 else "N/A"
    lines = [
        (f"TARGET  ID:{tid}", (0, 255, 80), 0.55, 2),
        (f"Conf:{sim:.1%}  {cam_id}", (180, 255, 180), 0.44, 1),
        (f"XY:({cx},{cy})  t:{sec:.1f}s", (180, 255, 180), 0.40, 1),
        (f"Distance: {dist_str}", (255, 220, 100), 0.44, 1),
    ]
    for i, (text, col, scale, thick) in enumerate(lines):
        cv2.putText(frame, text, (x1 + 5, top + 16 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def annotate_other(frame, tid, bbox, dist_m, cfg):
    x1, y1, x2, y2 = bbox
    color = cfg["color_other"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    dist_str = f"{dist_m:.1f}m" if dist_m > 0 else ""
    cv2.putText(frame, f"P{tid} {dist_str}",
                (x1 + 4, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_trajectory(frame, traj, color, max_points=60):
    pts = traj[-max_points:]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(frame, pts[i - 1], pts[i], faded, 2)


def draw_hud(frame, frame_no, fps, n_persons, target_found, cam_id, target_dist):
    alpha_rect(frame, 0, 0, 400, 95, (0, 0, 0))
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    status = "TARGET DETECTED" if target_found else "SEARCHING ..."
    col = (0, 255, 80) if target_found else (60, 80, 255)
    dist_info = f"  |  Dist: {target_dist:.1f} m" if (target_found and target_dist > 0) else ""

    hud_lines = [
        (f"{cam_id}  |  {ts}", (200, 200, 200), 0.45),
        (f"Frame {frame_no}  |  Persons: {n_persons}", (200, 200, 200), 0.45),
        (f"{status}{dist_info}", col, 0.58),
    ]
    for i, (text, color, scale) in enumerate(hud_lines):
        thick = 2 if i == 2 else 1
        cv2.putText(frame, text, (8, 20 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────
# REFERENCE GALLERY
# ──────────────────────────────────────────────────────────────────────────

def build_reference_gallery(m: ReIDModels, ref_paths: list[str]) -> list[dict]:
    """One gallery entry per reference image: {face, body, source}."""
    gallery = []
    for path in ref_paths:
        ref_bgr = cv2.imread(path)
        if ref_bgr is None:
            continue
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        face_emb, body_emb = extract_embeddings(m, ref_rgb)
        gallery.append({"face": face_emb, "body": body_emb, "source": path,
                         "face_found": face_emb is not None})
    if not gallery:
        raise ValueError("No usable reference images — all failed to load.")
    return gallery


# ──────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(
    m: ReIDModels,
    cfg: dict,
    ref_gallery: list,
    video_path: str,
    on_frame: Optional[Callable[[int, int, dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    preview_every: int = 5,
) -> dict:
    """
    Runs detection + tracking + re-ID over the whole video and writes an
    annotated copy to cfg["output_path"].

    on_frame(frame_no, total, info) is called every frame, where info =
        {alert_count, target_found, target_dist, preview_bgr}
    preview_bgr is only populated every `preview_every` frames (else None)
    to keep the callback cheap.

    should_stop() is polled once per frame; if it returns True the loop
    breaks early (partial results are still returned).
    """
    tracker = DeepSort(
        max_age=cfg["max_age"],
        n_init=cfg["n_init"],
        max_cosine_distance=cfg["max_cosine_dist"],
        nn_budget=100,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cfg.get("max_frames"):
        total = min(total, cfg["max_frames"]) if total > 0 else cfg["max_frames"]

    out = cv2.VideoWriter(
        cfg["output_path"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H),
    )

    track_face_emb: dict = {}
    track_body_emb: dict = {}
    track_is_target: dict = {}
    track_trajectory = defaultdict(list)
    track_last_sim: dict = {}
    track_score_hist = defaultdict(lambda: deque(maxlen=cfg["score_history_len"]))

    alert_log = []
    alert_frame_count = 0
    frame_no = 0
    start_time = time.time()
    exit_threshold = cfg["reid_threshold"] - cfg["reid_exit_margin"]
    stopped_early = False

    while True:
        if should_stop and should_stop():
            stopped_early = True
            break

        ret, bgr = cap.read()
        if not ret:
            break
        frame_no += 1
        if cfg.get("max_frames") and frame_no > cfg["max_frames"]:
            frame_no -= 1
            break

        if frame_no % cfg["skip_frames"] != 0:
            out.write(bgr)
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        results = m.yolo(bgr, conf=cfg["yolo_conf"], iou=cfg["yolo_iou"],
                          classes=[0], verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=bgr)

        target_this_frame = False
        target_dist_frame = -1.0

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1 = max(0, int(ltrb[0])); y1 = max(0, int(ltrb[1]))
            x2 = min(W, int(ltrb[2])); y2 = min(H, int(ltrb[3]))
            if x2 <= x1 or y2 <= y1:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bbox_h = y2 - y1
            dist_m = estimate_distance_m(bbox_h, cfg)

            crop = rgb[y1:y2, x1:x2]
            face_emb, body_emb = None, None
            if crop.size > 0:
                try:
                    face_emb, body_emb = extract_embeddings(m, crop)
                except Exception:
                    face_emb, body_emb = None, None

            good_quality = crop_is_good_quality(x1, y1, x2, y2, cfg)

            if body_emb is not None and good_quality:
                track_body_emb[tid] = (
                    ema_update(track_body_emb[tid], body_emb)
                    if tid in track_body_emb else body_emb
                )
            if face_emb is not None:
                track_face_emb[tid] = (
                    ema_update(track_face_emb[tid], face_emb)
                    if tid in track_face_emb else face_emb
                )

            if tid in track_body_emb:
                sim = match_score(track_face_emb.get(tid), track_body_emb[tid],
                                   ref_gallery, cfg["face_weight"])
                track_score_hist[tid].append(sim)
                avg_recent = sum(track_score_hist[tid]) / len(track_score_hist[tid])

                was_target = track_is_target.get(tid, False)
                if not was_target:
                    is_target = sim >= cfg["reid_threshold"]
                else:
                    is_target = avg_recent >= exit_threshold
                track_last_sim[tid] = sim
            else:
                sim = track_last_sim.get(tid, 0.0)
                is_target = track_is_target.get(tid, False)

            track_is_target[tid] = is_target
            track_trajectory[tid].append((cx, cy))
            trail_color = cfg["color_target"] if is_target else (80, 80, 80)
            draw_trajectory(bgr, track_trajectory[tid], trail_color)

            if is_target:
                target_this_frame = True
                target_dist_frame = dist_m
                alert_frame_count += 1
                annotate_target(bgr, tid, (x1, y1, x2, y2), sim, dist_m,
                                 cfg["camera_id"], frame_no, fps, cfg)
                alert_log.append({
                    "frame": frame_no,
                    "timestamp": datetime.now().isoformat(),
                    "camera_id": cfg["camera_id"],
                    "track_id": int(tid),
                    "confidence": round(float(sim), 4),
                    "bbox": [x1, y1, x2, y2],
                    "centroid_px": [cx, cy],
                    "distance_m": round(float(dist_m), 2),
                    "time_sec": round(frame_no / fps, 2),
                })
            else:
                annotate_other(bgr, tid, (x1, y1, x2, y2), dist_m, cfg)

        draw_hud(bgr, frame_no, fps, len(tracks), target_this_frame,
                 cfg["camera_id"], target_dist_frame)
        out.write(bgr)

        if on_frame:
            preview = bgr.copy() if (frame_no % preview_every == 0) else None
            on_frame(frame_no, total, {
                "alert_count": alert_frame_count,
                "target_found": target_this_frame,
                "target_dist": target_dist_frame,
                "n_persons": len(tracks),
                "preview_bgr": preview,
            })

    cap.release()
    out.release()
    elapsed = time.time() - start_time

    return {
        "alert_log": alert_log,
        "track_face_emb": track_face_emb,
        "track_body_emb": track_body_emb,
        "track_is_target": track_is_target,
        "output_path": cfg["output_path"],
        "fps": fps,
        "width": W,
        "height": H,
        "frames_processed": frame_no,
        "total_frames": total,
        "elapsed_sec": elapsed,
        "stopped_early": stopped_early,
    }


# ──────────────────────────────────────────────────────────────────────────
# REPORTING / PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────

def summarize_alert_log(alert_log: list, cfg: dict) -> dict:
    if not alert_log:
        return {
            "camera_id": cfg["camera_id"], "total_alerts": 0, "unique_tracks": 0,
            "avg_confidence": None, "max_confidence": None,
            "min_distance_m": None, "avg_distance_m": None, "first_seen": {},
        }
    confs = [a["confidence"] for a in alert_log]
    dists = [a["distance_m"] for a in alert_log if a["distance_m"] > 0]
    first_seen = {}
    for a in alert_log:
        first_seen.setdefault(a["track_id"], a)

    return {
        "camera_id": cfg["camera_id"],
        "total_alerts": len(alert_log),
        "unique_tracks": len(first_seen),
        "avg_confidence": float(np.mean(confs)),
        "max_confidence": float(np.max(confs)),
        "min_distance_m": float(min(dists)) if dists else None,
        "avg_distance_m": float(np.mean(dists)) if dists else None,
        "first_seen": first_seen,
    }


def save_alert_report_json(alert_log: list, path: str = "reid_alert_report.json") -> str:
    with open(path, "w") as f:
        json.dump(alert_log, f, indent=2)
    return path


def save_model_pkl(
    path: str, ref_paths: list, ref_gallery: list, alert_log: list,
    track_face_emb: dict, track_body_emb: dict, track_is_target: dict, cfg: dict,
) -> str:
    payload = {
        "version": "4.0.0",
        "created_at": datetime.now().isoformat(),
        "reference_images": ref_paths,
        "reference_gallery": ref_gallery,
        "config": cfg,
        "alert_log": alert_log,
        "track_face_embeddings": track_face_emb,
        "track_body_embeddings": track_body_emb,
        "track_is_target": track_is_target,
        "models_used": {
            "detector": "YOLOv8 (ultralytics)",
            "face_encoder": "InceptionResnetV1 vggface2 (facenet-pytorch)",
            "body_encoder": "ResNet50 ImageNet (torchvision)",
            "tracker": "DeepSORT (deep-sort-realtime)",
        },
        "distance_config": {
            "method": "bounding_box_height_heuristic",
            "known_height_m": cfg["known_height_m"],
            "focal_px": cfg["focal_px"],
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def make_preview_h264(input_path: str, output_path: str = "reid_output_h264.mp4") -> str:
    """Re-encode to H.264 so the video plays inline in a browser. Falls back
    to the raw file if ffmpeg isn't available (e.g. not yet installed)."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
            "-movflags", "+faststart", output_path,
        ], check=True)
        return output_path
    except Exception:
        return input_path
