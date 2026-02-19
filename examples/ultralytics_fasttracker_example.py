"""Example program: run Ultralytics YOLO detections through FastTracker.

This example is intentionally simple and mirrors the integration entrypoint,
while staying easy to customize in user projects.
"""

from types import SimpleNamespace
import json
import os
import sys
import cv2
import numpy as np

from ultralytics import YOLO

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FILE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from yolox.tracker.fasttracker import Fasttracker


def yolo_to_fasttracker_input(result):
    """Convert Ultralytics result boxes to FastTracker Nx5 [x1,y1,x2,y2,score]."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
    return np.concatenate([xyxy, conf], axis=1)


def main():
    # 1) Load detector + tracker config
    model = YOLO("yolov8n.pt")
    with open("configs/004_default.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # FastTracker currently reads args.mot20
    args = SimpleNamespace(mot20=False)
    tracker = Fasttracker(args, cfg, frame_rate=30)

    # 2) Open input video and output writer
    cap = cv2.VideoCapture("input.mp4")
    if not cap.isOpened():
        raise RuntimeError("Failed to open input.mp4")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30

    writer = cv2.VideoWriter(
        "tracked_output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # 3) Run detect -> track -> visualize loop
    frame_id = 0
    with open("track_results.txt", "w", encoding="utf-8") as mot_f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            result = model.predict(source=frame, conf=0.25, iou=0.7, verbose=False)[0]
            dets = yolo_to_fasttracker_input(result)

            # FastTracker uses [img_h, img_w] and img_size for scaling logic.
            img_info = [frame.shape[0], frame.shape[1]]
            img_size = (frame.shape[0], frame.shape[1])

            tracks = tracker.update(dets, img_info, img_size)
            for t in tracks:
                x, y, bw, bh = t.tlwh
                tid = int(t.track_id)
                score = float(t.score)

                # MOT line format
                mot_f.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},{score:.4f},-1,-1,-1\n")

                # Draw
                x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 2)

            writer.write(frame)

    cap.release()
    writer.release()
    print("Done: tracked_output.mp4 and track_results.txt saved")


if __name__ == "__main__":
    main()
