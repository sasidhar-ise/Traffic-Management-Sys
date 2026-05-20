"""
Smart Traffic Violation Detector — LIVE STREAMING + OPTIMIZED
Pushes annotated frames into a queue for real-time browser display.
Run via app.py or: python detect.py --video your_video.mp4
"""

import cv2
import numpy as np
import argparse
import json
import time
import queue
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------

# Approximate pixel‑speed threshold for "speeding" on resized frames
# (tuned so not every tiny motion becomes a "speeding" event)
SPEED_THRESHOLD_PX  = 10

# COCO class ids (for default yolov8n.pt)
PHONE_CLASS_IDS       = [67]            # cell phone
VEHICLE_CLASS_IDS     = [2, 3, 5, 7]    # car, motorcycle, bus, truck
PERSON_CLASS_IDS      = [0]             # person

# Heuristic threshold for "risky maneuver" based on lateral jitter
LATERAL_JITTER_THRESH = 25.0

PROCESS_EVERY_N     = 3
RESIZE_W, RESIZE_H  = 640, 360

COLORS = {
    "speeding":  (0, 60, 255),
    "risky":     (0, 140, 255),
    "phone":     (255, 0, 180),
    "normal":    (0, 220, 80),
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class ViolationTracker:
    def __init__(self):
        self.track_history = defaultdict(list)
        self.violations    = defaultdict(set)
        self.violation_log = []

    def update_position(self, tid, cx, cy):
        h = self.track_history[tid]
        h.append((cx, cy))
        if len(h) > 20: h.pop(0)

    def get_speed(self, tid):
        h = self.track_history[tid]
        # Need at least a small history to estimate motion
        if len(h) < 3:
            return 0.0
        # Compare current pos with a few steps back
        step = min(4, len(h) - 1)
        dx = h[-1][0] - h[-step][0]
        dy = h[-1][1] - h[-step][1]
        return float(np.sqrt(dx**2 + dy**2) / step)

    def get_lateral_jitter(self, tid):
        """
        Simple measure of how much the object is weaving left/right.
        Returns the standard deviation of the last few x positions.
        """
        h = self.track_history[tid]
        if len(h) < 8:
            return 0.0
        xs = np.array([p[0] for p in h[-12:]])  # up to last 12 positions
        return float(xs.std())

    def log_violation(self, tid, vtype, frame_no, cx, cy):
        self.violation_log.append({
            "track_id": int(tid),
            "violation": vtype,
            "frame": int(frame_no),
            "position": [int(cx), int(cy)],
        })



def detect_violations_live(video_path: str, output_path: str,
                            frame_queue=None, job: dict = None):
    """
    Main detection function.
    frame_queue: queue.Queue — push annotated frames here for MJPEG streaming
    job: dict — updated with progress, summary, events for status polling
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device.upper()}")

    if job: job["message"] = "Loading YOLOv8 model..."
    model = YOLO("yolov8n.pt")
    model.to(device)
    if job: job["message"] = "Model loaded. Starting analysis..."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    FPS   = cap.get(cv2.CAP_PROP_FPS) or 30
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H  = RESIZE_W, RESIZE_H

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    tracker     = ViolationTracker()
    total_viols = defaultdict(int)
    frame_no    = 0
    last_frame  = None
    start_time  = time.time()

    print(f"[INFO] {TOTAL} frames | every {PROCESS_EVERY_N} frames | {W}x{H}\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_no += 1
        frame = cv2.resize(frame, (W, H))

        # Lightweight frame skipping to keep video from becoming too slow on CPU.
        # We only run the YOLO model every PROCESS_EVERY_N frames, but still
        # stream intermediate frames so the browser view feels smoother.
        if frame_no % PROCESS_EVERY_N != 0:
            # Reuse the last annotated frame so boxes don't "blink" on and off.
            show = last_frame if last_frame is not None else frame
            out.write(show)
            if frame_queue:
                try:
                    frame_queue.put_nowait(show.copy())
                except queue.Full:
                    pass
            # We still update basic progress so the bar moves
            if job:
                pct = int(frame_no/TOTAL*100) if TOTAL else 0
                elapsed  = time.time()-start_time
                fps_live = frame_no/elapsed if elapsed>0 else 0
                eta      = int((TOTAL-frame_no)/fps_live) if fps_live>0 else 0
                job["progress"] = pct
                job["message"]  = f"Frame {frame_no}/{TOTAL} | {fps_live:.1f} fps | ETA {eta}s"
            continue

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=VEHICLE_CLASS_IDS
                    + PERSON_CLASS_IDS
                    + PHONE_CLASS_IDS,
            device=device,
        )

        detected_phones   = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, classes):
                if cls in PHONE_CLASS_IDS:
                    detected_phones.append(box)

            # -----------------------------
            # VEHICLE / PERSON LOGIC
            # -----------------------------
            for box, tid, cls in zip(boxes, ids, classes):
                x1,y1,x2,y2 = map(int, box)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                label_parts, viol_types = [], set()

                tracker.update_position(tid, cx, cy)

                if cls in VEHICLE_CLASS_IDS:
                    spd = tracker.get_speed(tid)
                    if spd > SPEED_THRESHOLD_PX:
                        viol_types.add("speeding")
                        label_parts.append(f"SPEED:{spd:.0f}")

                    # Heuristic "risky maneuver" violation:
                    # high lateral jitter (weaving) while moving
                    jitter = tracker.get_lateral_jitter(tid)
                    if jitter > LATERAL_JITTER_THRESH and spd > SPEED_THRESHOLD_PX * 0.5:
                        viol_types.add("risky")
                        label_parts.append("RISKY MOVE")

                if cls in PERSON_CLASS_IDS:
                    for pb in detected_phones:
                        ox = max(0, min(x2,int(pb[2]))-max(x1,int(pb[0])))
                        oy = max(0, min(y2,int(pb[3]))-max(y1,int(pb[1])))
                        if ox*oy > 400:
                            viol_types.add("phone")
                            label_parts.append("PHONE")

                if viol_types:
                    # Pick a consistent color based on primary violation type
                    primary = list(viol_types)[0]
                    color = COLORS.get(primary, (0, 0, 255))
                    # Only count/log each violation type ONCE per track id
                    for v in viol_types:
                        if v not in tracker.violations[tid]:
                            tracker.violations[tid].add(v)
                            total_viols[v] += 1
                            tracker.log_violation(tid, v, frame_no, cx, cy)
                else:
                    color = COLORS["normal"]

                # -----------------------------
                # SIMPLE "RISK" INDICATOR
                # -----------------------------
                # Risk based on speed and how erratic the movement is
                if cls in VEHICLE_CLASS_IDS:
                    spd = tracker.get_speed(tid)
                    jitter = tracker.get_lateral_jitter(tid)
                    # Combine normalized speed and jitter
                    norm_spd    = min(1.0, spd / (SPEED_THRESHOLD_PX * 1.5))
                    norm_jitter = min(1.0, jitter / (LATERAL_JITTER_THRESH * 1.5))
                    risk = min(1.0, 0.6 * norm_spd + 0.4 * norm_jitter)
                    label_parts.append(f"RISK:{risk:.2f}")

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                label = f"ID:{tid} " + "|".join(label_parts) if label_parts else f"ID:{tid}"
                (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1,y1-lh-6),(x1+lw+4,y1), color, -1)
                cv2.putText(frame, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # HUD
        elapsed  = time.time()-start_time
        fps_live = frame_no/elapsed if elapsed>0 else 0
        eta      = int((TOTAL-frame_no)/fps_live) if fps_live>0 else 0
        pct      = int(frame_no/TOTAL*100) if TOTAL else 0

        cv2.rectangle(frame,(0,0),(310,118),(0,0,0),-1)
        hy=20
        for txt,col in [
            (f"Progress:{pct}%  ETA:{eta}s",(180,180,180)),
            (f"Speed: {fps_live:.1f} fps",(180,180,180)),
            (f"Speeding:  {total_viols['speeding']}",COLORS['speeding']),
            (f"Risky:     {total_viols['risky']}",COLORS['risky']),
            (f"Phone Use: {total_viols['phone']}",COLORS['phone']),
        ]:
            cv2.putText(frame,txt,(8,hy),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1); hy+=22

        last_frame = frame.copy()
        out.write(frame)

        # ✅ Push to live stream queue
        if frame_queue:
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        # Update job status for polling
        if job:
            job["progress"] = pct
            job["message"]  = f"Frame {frame_no}/{TOTAL} | {fps_live:.1f} fps | ETA {eta}s"
            job["summary"]  = {k: int(v) for k, v in total_viols.items()}
            job["events"]   = tracker.violation_log[-200:]  # last 200 events

        if frame_no % 30 == 0:
            print(f"  {pct}% | {fps_live:.1f}fps | ETA {eta}s | "
                  f"Speed:{total_viols['speeding']} Risky:{total_viols['risky']} Phone:{total_viols['phone']}")

    cap.release()
    out.release()

    # Final report JSON
    report = {
        "video": video_path,
        "total_frames": int(frame_no),
        "fps": float(FPS),
        "device": device,
        "summary": {k: int(v) for k, v in total_viols.items()},
        "events": tracker.violation_log,
    }
    report_path = Path(output_path).with_suffix(".json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NpEncoder)

    if job:
        job["summary"] = report["summary"]
        job["events"]  = report["events"]

    t = int(time.time()-start_time)
    print(f"\n✅ Done in {t}s! Violations: {sum(total_viols.values())}")
    return report


# CLI wrapper
def detect_violations(video_path, output_path, model_path="yolov8n.pt"):
    return detect_violations_live(video_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--skip",   type=int, default=3)
    args = parser.parse_args()
    PROCESS_EVERY_N = args.skip
    detect_violations(args.video, args.output)
