---
<<<<<<< HEAD
title: SENTRY — Person Re-ID & Tracking Console
emoji: 🎯
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8501
suggested_hardware: "cpu-upgrade"
short_description: AI-powered person re-identification and tracking from video
tags:
  - computer-vision
  - person-reidentification
  - yolov8
  - deepsort
  - facenet
  - streamlit
  - surveillance
pinned: false
---

# 🎯 SENTRY — Person Re-ID & Tracking Console

AI-powered surveillance system that identifies and tracks a specific target person across video footage using reference photos.

## What It Does

1. **Upload** 2–5 reference photos of the target person
2. **Upload** video footage to scan
3. **Run** the pipeline — YOLO detects people, FaceNet + ResNet50 match identity, DeepSORT tracks across frames
4. **Review** annotated video, confidence/distance charts, alert logs, and export results

## Tech Stack

| Component | Model / Library |
|---|---|
| Person Detection | YOLOv8 (ultralytics) |
| Face Encoding | InceptionResnetV1 — VGGFace2 (facenet-pytorch) |
| Body Encoding | ResNet50 — ImageNet (torchvision) |
| Multi-Object Tracking | DeepSORT (deep-sort-realtime) |
| Distance Estimation | Bounding-box height heuristic |
| Frontend | Streamlit |

## Performance Tips

- On **CPU** (free tier): use `yolov8n.pt`, set "Process every Nth frame" to 2–3, limit to first 10–15 seconds for a quick test
- On **GPU** (T4 small or above): `yolov8m.pt` runs smoothly at near real-time

## Responsible Use

This tool performs face- and body-based re-identification of real people. Get consent from anyone you intend to track, restrict access to outputs (video, embeddings, alert logs are personal/biometric data), and check local laws (GDPR, BIPA, etc.) before use.
=======
title: Sentry
emoji: 📊
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 0767e6e9520435edde5a15b1f5ce523b20a72190
