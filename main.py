# main.py

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|"
    "reconnect_streamed;1|"
    "reconnect_at_eof;1|"
    "reconnect_delay_max;2|"
    "rw_timeout;5000000"
)

import cv2
import numpy as np
import time
import threading
import torch
from collections import deque

cv2.setNumThreads(1)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from ultralytics import YOLO

from config import (
    STREAM_URL,
    CAMERA_NAME,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    YOLO_MODEL,
    CONF_THRESHOLD,
    MODEL_IMGSZ,
    DETECTION_TARGET_FPS,
    DISPLAY_TARGET_FPS,
    PLAYBACK_TARGET_FPS,
    SOURCE_TARGET_FPS,
    PLAYBACK_QUEUE_SIZE,
    MIN_BOX_AREA,
    MAX_BOX_AREA_RATIO,
    MIN_BOX_WIDTH,
    MIN_BOX_HEIGHT,
    MIN_MOTOR_CONF,
    MIN_MOBIL_CONF,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    ROI_POLYGON,
    SHOW_ZONE_DEBUG,
    IN_ZONE_POLYGON,
    OUT_ZONE_POLYGON,
    VIOLATION_ZONE_POLYGON,
    MAX_DETECTION_SEQ_GAP,
    TRAJECTORY_MAX_POINTS,
    TRACK_MATCH_DISTANCE,
    TRACK_MAX_MISSED_FRAMES,
    DIRECTION_MIN_MOVE_PX,
    TRACK_DRAW_HOLD_FRAMES,
    DRAW_BOX_SHRINK_RATIO,
    WRONG_WAY_RECENT_UP_MOVE_PX,
    WRONG_WAY_NET_UP_MOVE_PX,
    TRACKER_CONFIG,
    VEHICLE_CLASS_IDS,
)

WINDOW_NAME = "TRAFFIC_EYE AI - Smooth Preview Detection"


class PlaybackFrameBuffer:
    def __init__(self, max_queue_size=6):
        self.lock = threading.Lock()

        self.max_queue_size = max_queue_size
        self.queue = deque()
        self.dropped_playback_frames = 0

        self.latest_frame = None
        self.latest_seq = 0

        # Frame yang benar-benar sedang diputar/display.
        # Detector akan baca ini supaya box lebih sinkron dengan preview.
        self.playback_frame = None
        self.playback_seq = -1
        self.playback_connected = False

        self.connected = False
        self.source_fps = 0.0
        self.read_ms = 0.0
        self.frame_gap_ms = 0.0
        self.queue_size = 0

    def set(self, frame, connected=True, read_ms=0.0, frame_gap_ms=0.0):
        with self.lock:
            self.latest_seq += 1

            item = {
                "frame": frame,
                "seq": self.latest_seq,
            }

            if len(self.queue) < self.max_queue_size:
                self.queue.append(item)
            else:
                # Queue penuh: jangan buang frame lama.
                # Buang frame baru supaya playback tetap jalan berurut.
                self.dropped_playback_frames += 1

            self.latest_frame = frame
            self.connected = connected
            self.read_ms = read_ms
            self.frame_gap_ms = frame_gap_ms
            self.queue_size = len(self.queue)

    def pop_playback_frame(self):
        with self.lock:
            if not self.queue:
                return None, self.latest_seq, self.connected

            item = self.queue.popleft()
            self.queue_size = len(self.queue)

            return item["frame"].copy(), item["seq"], self.connected

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.latest_seq, self.connected

            return self.latest_frame.copy(), self.latest_seq, self.connected

    def set_playback_frame(self, frame, seq, connected):
        with self.lock:
            if frame is None:
                return

            self.playback_frame = frame.copy()
            self.playback_seq = seq
            self.playback_connected = connected

    def get_playback_frame(self):
        with self.lock:
            if self.playback_frame is None:
                return None, self.playback_seq, self.playback_connected

            return (
                self.playback_frame.copy(),
                self.playback_seq,
                self.playback_connected,
            )

    def set_connected(self, connected):
        with self.lock:
            self.connected = connected

    def set_source_fps(self, source_fps):
        with self.lock:
            self.source_fps = source_fps

    def get_stream_metrics(self):
        with self.lock:
            return (
                self.source_fps,
                self.read_ms,
                self.frame_gap_ms,
                self.queue_size,
            )

    def get_queue_status(self):
        with self.lock:
            return len(self.queue), self.latest_seq, self.connected


class DetectionBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
        self.infer_ms = 0.0
        self.detect_fps = 0.0
        self.last_seq = -1

    def set(self, detections, infer_ms, detect_fps, seq):
        with self.lock:
            self.detections = detections
            self.infer_ms = infer_ms
            self.detect_fps = detect_fps
            self.last_seq = seq

    def get(self):
        with self.lock:
            return (
                list(self.detections),
                self.infer_ms,
                self.detect_fps,
                self.last_seq,
            )


class TrajectoryTracker:
    def __init__(self):
        self.next_fallback_track_id = 100000
        self.tracks = {}

    def point_inside_polygon(self, point, polygon):
        if not polygon:
            return False

        px, py = point
        poly = np.array(polygon, dtype=np.int32)

        return (
            cv2.pointPolygonTest(
                poly,
                (float(px), float(py)),
                False,
            )
            >= 0
        )

    def get_zone_name(self, center):
        # Finish zone merah dicek dulu supaya kalau polygon overlap,
        # merah tetap menang.
        if self.point_inside_polygon(center, VIOLATION_ZONE_POLYGON):
            return "violation_zone"

        if self.point_inside_polygon(center, IN_ZONE_POLYGON):
            return "in_zone"

        if self.point_inside_polygon(center, OUT_ZONE_POLYGON):
            return "out_zone"

        return "road_zone"

    def get_fallback_track_id(self, detection):
        # Fallback kalau ByteTrack belum kasih ID.
        # Tetap pakai center supaya tidak crash, tapi ini bukan tracking final.
        center = detection.get("center")

        if center is None:
            self.next_fallback_track_id += 1
            return self.next_fallback_track_id

        cx, cy = center
        best_track_id = None
        best_distance = TRACK_MATCH_DISTANCE

        for track_id, track in self.tracks.items():
            if not str(track_id).startswith("fallback_"):
                continue

            if track["vehicle_type"] != detection.get("vehicle_type"):
                continue

            last_x, last_y = track["history"][-1]
            distance = ((cx - last_x) ** 2 + (cy - last_y) ** 2) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        if best_track_id is not None:
            return best_track_id

        self.next_fallback_track_id += 1
        return f"fallback_{self.next_fallback_track_id}"

    def classify_direction(self, track):
        history = list(track["history"])
        zones = list(track["zones"])

        if len(history) < 4:
            return track.get("final_status", "unknown")

        # Kalau status sudah pernah terkunci, jangan berubah-ubah lagi.
        if track.get("final_status", "unknown") != "unknown":
            return track["final_status"]

        current_zone = zones[-1]

        start_x, start_y = history[0]
        end_x, end_y = history[-1]

        net_dx = end_x - start_x
        net_dy = end_y - start_y
        move_dist = ((net_dx**2) + (net_dy**2)) ** 0.5

        # Jangan nilai status kalau trajectory belum cukup bergerak.
        if move_dist < DIRECTION_MIN_MOVE_PX:
            return "unknown"

        # INI BAGIAN YANG LU TANYA.
        # Taruh di sini.
        origin_zone = zones[0]

        if current_zone == origin_zone:
            return "unknown"

        if current_zone == "violation_zone":
            track["final_status"] = "wrong_way"
            return "wrong_way"

        if current_zone == "in_zone":
            track["final_status"] = "in"
            return "in"

        if current_zone == "out_zone":
            track["final_status"] = "out"
            return "out"

        return "unknown"

    def update(self, detections):
        for track in self.tracks.values():
            track["missed"] += 1

        updated_detections = []

        for detection in detections:
            center = detection.get("center")
            vehicle_type = detection.get("vehicle_type")

            if center is None:
                updated_detections.append(detection)
                continue

            track_id = detection.get("track_id")

            if track_id is None:
                track_id = self.get_fallback_track_id(detection)

            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "vehicle_type": vehicle_type,
                    "history": deque(maxlen=TRAJECTORY_MAX_POINTS),
                    "zones": deque(maxlen=TRAJECTORY_MAX_POINTS),
                    "missed": 0,
                    # Zone pertama saat object muncul.
                    # Ini dipakai sebagai asal trajectory.
                    "origin_zone": None,
                    # Kalau sudah IN/OUT/VIOLATION, status dikunci.
                    "final_status": "unknown",
                }

            track = self.tracks[track_id]
            track["vehicle_type"] = vehicle_type
            track["history"].append(center)
            track["zones"].append(self.get_zone_name(center))
            track["missed"] = 0

            if track["origin_zone"] is None:
                track["origin_zone"] = track["zones"][-1]

            direction = self.classify_direction(track)

            detection = dict(detection)
            detection["track_id"] = track_id
            detection["trajectory"] = list(track["history"])
            detection["zone"] = track["zones"][-1]
            detection["direction"] = direction

            updated_detections.append(detection)

        stale_track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > TRACK_MAX_MISSED_FRAMES
        ]

        for track_id in stale_track_ids:
            del self.tracks[track_id]

        return updated_detections


class DetectionStabilizer:
    def __init__(self):
        self.tracks = {}

    def get_key(self, detection):
        track_id = detection.get("track_id")

        if track_id is not None:
            return track_id

        center = detection.get("center")
        vehicle_type = detection.get("vehicle_type", "unknown")

        if center is None:
            return None

        cx, cy = center

        # Fallback kalau track_id tidak ada.
        # Dibulatkan supaya object dekat tidak bikin key terlalu liar.
        return f"{vehicle_type}_{cx // 40}_{cy // 40}"

    def update(self, detections):
        for track in self.tracks.values():
            track["missed"] += 1

        for detection in detections:
            key = self.get_key(detection)

            if key is None:
                continue

            detection = dict(detection)
            detection["missed"] = 0
            detection["is_hold"] = False

            self.tracks[key] = {
                "detection": detection,
                "missed": 0,
            }

        stable_detections = []

        stale_keys = []

        for key, track in self.tracks.items():
            missed = track["missed"]

            if missed > TRACK_DRAW_HOLD_FRAMES:
                stale_keys.append(key)
                continue

            detection = dict(track["detection"])

            if missed > 0:
                detection["is_hold"] = True
                detection["missed"] = missed

            stable_detections.append(detection)

        for key in stale_keys:
            del self.tracks[key]

        return stable_detections


class CCTVReaderThread:
    def __init__(self, stream_url, frame_buffer):
        self.stream_url = stream_url
        self.frame_buffer = frame_buffer
        self.cap = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.cap:
            self.cap.release()

    def open_stream(self):
        cap = cv2.VideoCapture(self.stream_url)

        if not cap.isOpened():
            print("[ERROR] Stream tidak bisa dibuka.")
            return None

        # Jangan paksa buffer size 1 untuk HLS.
        # Buffer terlalu kecil bikin stream terasa stop-start.
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[OK] Stream terbuka: {CAMERA_NAME}")
        return cap

    def loop(self):
        source_fps_timer = time.perf_counter()
        source_fps_count = 0
        last_frame_time = None
        source_interval = 1.0 / max(SOURCE_TARGET_FPS, 1)
        next_read_at = time.perf_counter()

        while self.running:
            now_limit = time.perf_counter()

            if now_limit < next_read_at:
                time.sleep(min(next_read_at - now_limit, 0.02))
                continue

            next_read_at = time.perf_counter() + source_interval
            loop_start = time.perf_counter()

            if self.cap is None:
                self.frame_buffer.set_connected(False)
                self.cap = self.open_stream()

                if self.cap is None:
                    time.sleep(2)
                    continue

                self.frame_buffer.set_connected(True)

            try:
                read_start = time.perf_counter()
                ret, frame = self.cap.read()
                read_ms = (time.perf_counter() - read_start) * 1000
            except Exception as e:
                print(f"[WARN] Exception saat baca frame: {e}. Reconnect...")

                self.frame_buffer.set_connected(False)

                if self.cap:
                    self.cap.release()

                self.cap = None
                time.sleep(1)
                continue

            if not ret or frame is None:
                print("[WARN] Frame gagal dibaca. Reconnect...")

                self.frame_buffer.set_connected(False)

                if self.cap:
                    self.cap.release()

                self.cap = None
                time.sleep(1)
                continue

            now_frame_time = time.perf_counter()

            if last_frame_time is None:
                frame_gap_ms = 0.0
            else:
                frame_gap_ms = (now_frame_time - last_frame_time) * 1000

            last_frame_time = now_frame_time

            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            self.frame_buffer.set(
                frame,
                connected=True,
                read_ms=read_ms,
                frame_gap_ms=frame_gap_ms,
            )
            source_fps_count += 1
            source_fps_now = time.perf_counter()

            if source_fps_now - source_fps_timer >= 1.0:
                source_fps = source_fps_count / (source_fps_now - source_fps_timer)
                self.frame_buffer.set_source_fps(source_fps)

                source_fps_timer = source_fps_now
                source_fps_count = 0


class DetectorThread:
    def __init__(self, model, frame_buffer, detection_buffer):
        self.model = model
        self.frame_buffer = frame_buffer
        self.detection_buffer = detection_buffer
        self.running = False
        self.thread = None
        self.last_processed_seq = -1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

    def map_vehicle_type(self, class_id):
        if class_id == 3:
            return "motor"

        if class_id in [2, 5, 7]:
            return "mobil"

        return "unknown"

    def is_inside_roi(self, cx, cy):
        if not ROI_POLYGON:
            return True

        roi = np.array(ROI_POLYGON, dtype=np.int32)

        return (
            cv2.pointPolygonTest(
                roi,
                (float(cx), float(cy)),
                False,
            )
            >= 0
        )

    def is_valid_detection_box(self, x1, y1, x2, y2, vehicle_type, confidence):
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)

        # Filter khusus per tipe kendaraan.
        # Motor di CCTV biasanya kecil, blur, dan box-nya sering tidak stabil.
        if vehicle_type == "motor":
            min_box_width = 5
            min_box_height = 5
            min_box_area = 80
            min_conf = MIN_MOTOR_CONF
            min_aspect_ratio = 0.18
            max_aspect_ratio = 6.50
        else:
            min_box_width = MIN_BOX_WIDTH
            min_box_height = MIN_BOX_HEIGHT
            min_box_area = MIN_BOX_AREA
            min_conf = MIN_MOBIL_CONF
            min_aspect_ratio = MIN_ASPECT_RATIO
            max_aspect_ratio = MAX_ASPECT_RATIO

        if box_w < min_box_width or box_h < min_box_height:
            return False

        box_area = box_w * box_h

        if box_area < min_box_area:
            return False

        frame_area = DISPLAY_WIDTH * DISPLAY_HEIGHT
        max_box_area = frame_area * MAX_BOX_AREA_RATIO

        if box_area > max_box_area:
            return False

        aspect_ratio = box_w / max(box_h, 1)

        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            return False

        if confidence < min_conf:
            return False

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if not self.is_inside_roi(cx, cy):
            return False

        return True

    def run_detection(self, frame):
        results = self.model.track(
            frame,
            conf=CONF_THRESHOLD,
            classes=VEHICLE_CLASS_IDS,
            imgsz=MODEL_IMGSZ,
            verbose=False,
            device="cpu",
            persist=True,
            tracker=TRACKER_CONFIG,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if box.id is None:
                    track_id = None
                else:
                    track_id = int(box.id[0])

                vehicle_type = self.map_vehicle_type(class_id)

                if vehicle_type == "unknown":
                    continue

                if not self.is_valid_detection_box(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    vehicle_type=vehicle_type,
                    confidence=confidence,
                ):
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "vehicle_type": vehicle_type,
                        "confidence": confidence,
                        "direction": "unknown",
                        "track_id": track_id,
                        "trajectory": [],
                    }
                )

        return detections

    def loop(self):
        min_interval = 1.0 / max(DETECTION_TARGET_FPS, 1)
        last_detect_time = 0.0

        fps_timer = time.perf_counter()
        fps_count = 0
        detect_fps = 0.0

        while self.running:
            now = time.perf_counter()

            if now - last_detect_time < min_interval:
                time.sleep(0.005)
                continue

            frame, seq, connected = self.frame_buffer.get_playback_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            if seq == self.last_processed_seq:
                time.sleep(0.005)
                continue

            self.last_processed_seq = seq
            last_detect_time = now

            infer_start = time.perf_counter()
            detections = self.run_detection(frame)
            infer_ms = (time.perf_counter() - infer_start) * 1000

            fps_count += 1
            fps_now = time.perf_counter()

            if fps_now - fps_timer >= 1.0:
                detect_fps = fps_count / (fps_now - fps_timer)
                fps_timer = fps_now
                fps_count = 0

            self.detection_buffer.set(
                detections=detections,
                infer_ms=infer_ms,
                detect_fps=detect_fps,
                seq=seq,
            )


def count_vehicle_types(detections):
    counts = {
        "motor": 0,
        "mobil": 0,
        "pelanggaran": 0,
    }

    for det in detections:
        vehicle_type = det["vehicle_type"]
        direction = det.get("direction", "unknown")

        if vehicle_type in counts:
            counts[vehicle_type] += 1

        if direction == "wrong_way":
            counts["pelanggaran"] += 1

    return counts


def get_direction_label(direction):
    if direction == "in":
        return "IN"

    if direction == "out":
        return "OUT"

    if direction == "wrong_way":
        return "LAWAN_ARAH"

    return "TRACK"


def get_detection_color(detection):
    vehicle_type = detection.get("vehicle_type")
    direction = detection.get("direction", "unknown")

    # OpenCV pakai BGR, bukan RGB.
    if direction == "wrong_way":
        return (0, 0, 255)  # merah

    if vehicle_type == "motor":
        return (0, 165, 255)  # oranye

    if vehicle_type == "mobil":
        return (255, 220, 0)  # biru

    return (0, 255, 0)  # fallback hijau


def draw_polygon_outline(frame, points, color, label):
    if not points:
        return

    polygon = np.array(points, dtype=np.int32)

    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    label_x, label_y = points[0]

    cv2.putText(
        frame,
        label,
        (label_x + 5, label_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_zone_debug(frame):
    if not SHOW_ZONE_DEBUG:
        return

    draw_polygon_outline(
        frame,
        ROI_POLYGON,
        (255, 255, 255),
        "ROI",
    )

    draw_polygon_outline(
        frame,
        IN_ZONE_POLYGON,
        (255, 0, 0),
        "IN_ZONE",
    )

    draw_polygon_outline(
        frame,
        OUT_ZONE_POLYGON,
        (0, 165, 255),
        "OUT_ZONE",
    )

    draw_polygon_outline(
        frame,
        VIOLATION_ZONE_POLYGON,
        (0, 0, 255),
        "VIOLATION_ZONE",
    )


def draw_trajectory(frame, detection):
    trajectory = detection.get("trajectory", [])

    if len(trajectory) < 2:
        return

    color = get_detection_color(detection)

    for i in range(1, len(trajectory)):
        cv2.line(
            frame,
            trajectory[i - 1],
            trajectory[i],
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.arrowedLine(
        frame,
        trajectory[-2],
        trajectory[-1],
        color,
        2,
        cv2.LINE_AA,
        tipLength=0.35,
    )


def shrink_box_for_draw(box, vehicle_type=None):
    x1, y1, x2, y2 = box

    # Motor jangan dikecilin lagi. Box motor sudah kecil dari sananya.
    if vehicle_type == "motor":
        return box

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    shrink_x = int(box_w * DRAW_BOX_SHRINK_RATIO)
    shrink_y = int(box_h * DRAW_BOX_SHRINK_RATIO)

    new_x1 = x1 + shrink_x
    new_y1 = y1 + shrink_y
    new_x2 = x2 - shrink_x
    new_y2 = y2 - shrink_y

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return box

    return new_x1, new_y1, new_x2, new_y2


def draw_detection(frame, detection):
    vehicle_type = detection["vehicle_type"]
    x1, y1, x2, y2 = shrink_box_for_draw(detection["box"], vehicle_type)
    confidence = detection["confidence"]
    track_id = detection.get("track_id")
    direction = detection.get("direction", "unknown")

    color = get_detection_color(detection)
    direction_label = get_direction_label(direction)

    draw_trajectory(frame, detection)

    thickness = 1 if detection.get("is_hold") else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if track_id is None:
        text = f"{vehicle_type} {direction_label} {confidence:.2f}"
    else:
        text = f"ID:{track_id} {vehicle_type} {direction_label} {confidence:.2f}"

    text_pos = (x1, max(y1 - 8, 20))

    # Outline hitam supaya tulisan tetap kebaca di kendaraan terang/gelap.
    cv2.putText(
        frame,
        text,
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_hud(
    frame,
    counts,
    display_fps,
    source_fps,
    read_ms,
    frame_gap_ms,
    queue_size,
    detect_fps,
    infer_ms,
    connected,
    frame_seq,
    detect_seq,
):
    status_text = "CONNECTED" if connected else "RECONNECTING"

    cv2.putText(
        frame,
        CAMERA_NAME,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    counter_text = (
        f"Motor: {counts['motor']} | "
        f"Mobil: {counts['mobil']} | "
        f"Pelanggaran: {counts['pelanggaran']}"
    )

    cv2.putText(
        frame,
        counter_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    perf_text = (
        f"Display FPS: {display_fps:.1f} | "
        f"Source FPS: {source_fps:.1f} | "
        f"Read: {read_ms:.0f} ms | "
        f"Gap: {frame_gap_ms:.0f} ms | "
        f"Queue: {queue_size} | "
        f"Detect FPS: {detect_fps:.1f} | "
        f"Infer: {infer_ms:.0f} ms"
    )

    cv2.putText(
        frame,
        perf_text,
        (20, DISPLAY_HEIGHT - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        status_text,
        (20, DISPLAY_HEIGHT - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if connected else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    print("[INFO] Load YOLO model...")
    model = YOLO(YOLO_MODEL)
    print(f"[OK] Model loaded: {YOLO_MODEL}")

    try:
        model.fuse()
    except Exception:
        pass

    frame_buffer = PlaybackFrameBuffer(max_queue_size=PLAYBACK_QUEUE_SIZE)
    detection_buffer = DetectionBuffer()
    trajectory_tracker = TrajectoryTracker()
    detection_stabilizer = DetectionStabilizer()

    reader = CCTVReaderThread(STREAM_URL, frame_buffer)
    detector = DetectorThread(model, frame_buffer, detection_buffer)

    reader.start()
    detector.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    fps_timer = time.perf_counter()
    fps_count = 0
    display_fps = 0.0

    render_delay = 1.0 / max(DISPLAY_TARGET_FPS, 1)
    playback_delay = 1.0 / max(PLAYBACK_TARGET_FPS, 1)

    last_display_frame = None
    last_playback_pop_time = 0.0
    last_frame_seq = -1
    last_tracked_detect_seq = -1
    last_tracked_detections = []

    try:
        while True:
            loop_start = time.perf_counter()

            queue_size_now, latest_seq_now, connected_now = (
                frame_buffer.get_queue_status()
            )

            now_playback = time.perf_counter()
            should_pop_new_frame = (
                last_display_frame is None
                or now_playback - last_playback_pop_time >= playback_delay
            )

            # Kalau waktunya belum ambil frame baru,
            # render ulang frame terakhir saja.
            # Ini bikin window bisa 35 FPS tanpa mempercepat video.
            if not should_pop_new_frame and last_display_frame is not None:
                frame = last_display_frame.copy()
                frame_seq = last_frame_seq
                connected = connected_now
            else:
                # Kalau belum pernah ada frame tampil sama sekali,
                # tunggu diam sampai frame pertama masuk.
                if queue_size_now <= 0 and last_display_frame is None:
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                    time.sleep(0.01)
                    continue

                # Kalau queue kosong setelah pernah tampil,
                # tahan frame terakhir tanpa teks buffering.
                if queue_size_now <= 0 and last_display_frame is not None:
                    frame = last_display_frame.copy()
                    frame_seq = last_frame_seq
                    connected = connected_now
                else:
                    # Ini baru ambil frame CCTV berikutnya secara FIFO.
                    frame, frame_seq, connected = frame_buffer.pop_playback_frame()

                    last_playback_pop_time = now_playback
                    last_frame_seq = frame_seq

            if frame is None:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                time.sleep(0.03)
                continue

            # Kirim frame yang benar-benar sedang tampil ke detector.
            # Ini bikin box tidak ngikut latest stream yang beda timing.
            frame_buffer.set_playback_frame(frame, frame_seq, connected)

            detections, infer_ms, detect_fps, detect_seq = detection_buffer.get()

            if detect_seq >= 0:
                seq_gap = abs(frame_seq - detect_seq)
            else:
                seq_gap = 999999

            # Kalau detection terlalu beda jauh dari frame yang sedang tampil,
            # jangan gambar box. Ini mencegah box terlihat telat/ngawur.
            if seq_gap > MAX_DETECTION_SEQ_GAP:
                detections = detection_stabilizer.update([])
            else:
                if detect_seq != last_tracked_detect_seq:
                    last_tracked_detections = trajectory_tracker.update(detections)
                    last_tracked_detect_seq = detect_seq
                    detections = detection_stabilizer.update(last_tracked_detections)
                else:
                    detections = detection_stabilizer.update([])

            counts = count_vehicle_types(detections)

            draw_zone_debug(frame)

            for detection in detections:
                draw_detection(frame, detection)

            fps_count += 1
            now = time.perf_counter()

            if now - fps_timer >= 1.0:
                display_fps = fps_count / (now - fps_timer)
                fps_timer = now
                fps_count = 0
            source_fps, read_ms, frame_gap_ms, queue_size = (
                frame_buffer.get_stream_metrics()
            )

            draw_hud(
                frame=frame,
                counts=counts,
                display_fps=display_fps,
                source_fps=source_fps,
                read_ms=read_ms,
                frame_gap_ms=frame_gap_ms,
                queue_size=queue_size,
                detect_fps=detect_fps,
                infer_ms=infer_ms,
                connected=connected,
                frame_seq=frame_seq,
                detect_seq=detect_seq,
            )

            last_display_frame = frame.copy()
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elapsed = time.perf_counter() - loop_start
            sleep_time = render_delay - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        detector.stop()
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
