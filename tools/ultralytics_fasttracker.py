import argparse
import json
import os
from types import SimpleNamespace
import logging
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FILE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal envs
    logger = logging.getLogger("ultralytics_fasttracker")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser("Ultralytics YOLO + FastTracker integration")
    parser.add_argument("--source", required=True, help="Video path, image path, image folder, or webcam index.")
    parser.add_argument("--model", required=True, help="Ultralytics model path (e.g., yolov8n.pt).")
    parser.add_argument("--config", default="configs/004_default.json", help="FastTracker JSON config.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", default="cpu", help="Ultralytics device string, e.g. cpu or 0.")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional class IDs to keep.")
    parser.add_argument("--fps", type=int, default=30, help="Tracker frame rate for webcam/images fallback.")
    parser.add_argument("--mot20", action="store_true", help="Enable MOT20 matching behavior in tracker.")
    parser.add_argument("--class-aware", action="store_true", help="Use class-aware FastTracker variant.")
    parser.add_argument("--save-video", default="runs/ultralytics_fasttracker/out.mp4", help="Output video path.")
    parser.add_argument("--save-mot", default="runs/ultralytics_fasttracker/track_results.txt", help="Output MOT txt path.")
    parser.add_argument("--show", action="store_true", help="Show live visualization window.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_tracker(args, config):
    from yolox.tracker.fasttracker import Fasttracker as FastTracker
    from yolox.tracker.fasttracker_cls import Fasttracker as FastTrackerCLS

    tracker_args = SimpleNamespace(mot20=args.mot20)
    tracker_cls = FastTrackerCLS if args.class_aware else FastTracker
    return tracker_cls(tracker_args, config, frame_rate=args.fps)


def yolo_result_to_tracker_input(result, class_aware=False):
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 7 if class_aware else 5), dtype=np.float32)

    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)

    if class_aware:
        cls = result.boxes.cls.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        ones = np.ones_like(conf, dtype=np.float32)
        # FastTrackerCLS expects score columns and class id as last column.
        return np.concatenate([xyxy, conf, ones, cls], axis=1)

    return np.concatenate([xyxy, conf], axis=1)


def draw_tracks(frame, tracks):
    import cv2
    for t in tracks:
        x, y, w, h = t.tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        tid = int(t.track_id)
        score = float(t.score)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(
            frame,
            f"ID {tid} {score:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 240, 50),
            2,
        )


def iter_image_paths(source):
    if os.path.isfile(source):
        return [source]
    paths = []
    for name in sorted(os.listdir(source)):
        p = os.path.join(source, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in IMAGE_EXT:
            paths.append(p)
    return paths


def run_image_sequence(source, model, tracker, args):
    import cv2
    image_paths = iter_image_paths(source)
    if not image_paths:
        raise RuntimeError(f"No images found in source: {source}")

    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError(f"Could not read image: {image_paths[0]}")
    h, w = first.shape[:2]
    writer = get_writer(args.save_video, args.fps, w, h)

    os.makedirs(os.path.dirname(args.save_mot), exist_ok=True)
    with open(args.save_mot, "w", encoding="utf-8") as mot_f:
        for frame_id, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"Skipping unreadable image: {image_path}")
                continue

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                classes=args.classes,
                verbose=False,
            )[0]
            dets = yolo_result_to_tracker_input(result, class_aware=args.class_aware)
            img_info = [frame.shape[0], frame.shape[1]]
            img_size = (frame.shape[0], frame.shape[1])

            online_targets = tracker.update(dets, img_info, img_size)
            for t in online_targets:
                x, y, bw, bh = t.tlwh
                mot_f.write(f"{frame_id},{t.track_id},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},{t.score:.4f},-1,-1,-1\n")

            draw_tracks(frame, online_targets)
            writer.write(frame)
            if args.show:
                cv2.imshow("Ultralytics FastTracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    writer.release()


def get_writer(path, fps, width, height):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def run_video(source, model, tracker, args):
    import cv2
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else args.fps

    writer = get_writer(args.save_video, fps, width, height)
    os.makedirs(os.path.dirname(args.save_mot), exist_ok=True)

    frame_id = 0
    with open(args.save_mot, "w", encoding="utf-8") as mot_f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                classes=args.classes,
                verbose=False,
            )[0]

            dets = yolo_result_to_tracker_input(result, class_aware=args.class_aware)
            img_info = [frame.shape[0], frame.shape[1]]
            img_size = (frame.shape[0], frame.shape[1])

            online_targets = tracker.update(dets, img_info, img_size)
            for t in online_targets:
                x, y, bw, bh = t.tlwh
                mot_f.write(f"{frame_id},{t.track_id},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},{t.score:.4f},-1,-1,-1\n")

            draw_tracks(frame, online_targets)
            writer.write(frame)

            if args.show:
                cv2.imshow("Ultralytics FastTracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if frame_id % 50 == 0:
                logger.info(f"Processed {frame_id} frames")

    cap.release()
    writer.release()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required. Install with: pip install ultralytics") from exc

    config = load_config(args.config)
    tracker = make_tracker(args, config)
    model = YOLO(args.model)

    if os.path.isdir(args.source) or os.path.splitext(args.source.lower())[1] in IMAGE_EXT:
        run_image_sequence(args.source, model, tracker, args)
    else:
        run_video(args.source, model, tracker, args)

    if args.show:
        import cv2
        cv2.destroyAllWindows()

    logger.info(f"Done. Video: {args.save_video}")
    logger.info(f"Done. MOT results: {args.save_mot}")


if __name__ == "__main__":
    main()
