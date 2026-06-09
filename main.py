import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|"
    "reconnect_streamed;1|"
    "reconnect_at_eof;1|"
    "reconnect_delay_max;2|"
    "rw_timeout;5000000"
)

import cv2
import time
import threading
import torch

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
    MIN_BOX_AREA,
    VEHICLE_CLASS_IDS,
)

WINDOW_NAME = "TRAFFIC_EYE AI - Smooth Preview Detection"


class LatestFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.connected = False
        self.source_fps = 0.0

    def set(self, frame, connected=True):
        with self.lock:
            self.frame = frame
            self.seq += 1
            self.connected = connected

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, self.seq, self.connected

            return self.frame.copy(), self.seq, self.connected

    def set_connected(self, connected):
        with self.lock:
            self.connected = connected

    def set_source_fps(self, source_fps):
        with self.lock:
            self.source_fps = source_fps

    def get_source_fps(self):
        with self.lock:
            return self.source_fps


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
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("[ERROR] Stream tidak bisa dibuka.")
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[OK] Stream terbuka: {CAMERA_NAME}")
        return cap

    def loop(self):
        source_fps_timer = time.perf_counter()
        source_fps_count = 0
        while self.running:
            if self.cap is None:
                self.frame_buffer.set_connected(False)
                self.cap = self.open_stream()

                if self.cap is None:
                    time.sleep(2)
                    continue

                self.frame_buffer.set_connected(True)

            try:
                ret, frame = self.cap.read()
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

            self.frame_buffer.set(frame, connected=True)
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

        if class_id == 2:
            return "mobil"

        if class_id in [5, 7]:
            return "kendaraan_besar"

        return "unknown"

    def run_detection(self, frame):
        results = self.model.predict(
            frame,
            conf=CONF_THRESHOLD,
            classes=VEHICLE_CLASS_IDS,
            imgsz=MODEL_IMGSZ,
            verbose=False,
            device="cpu",
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                vehicle_type = self.map_vehicle_type(class_id)

                if vehicle_type == "unknown":
                    continue

                box_area = max(0, x2 - x1) * max(0, y2 - y1)

                if box_area < MIN_BOX_AREA:
                    continue

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "vehicle_type": vehicle_type,
                        "confidence": confidence,
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

            frame, seq, connected = self.frame_buffer.get()

            if frame is None:
                time.sleep(0.05)
                continue

            if seq == self.last_processed_seq:
                time.sleep(0.005)
                continue

            self.last_processed_seq = seq
            last_detect_time = now

            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

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
        "kendaraan_besar": 0,
    }

    for det in detections:
        vehicle_type = det["vehicle_type"]

        if vehicle_type in counts:
            counts[vehicle_type] += 1

    return counts


def draw_detection(frame, detection):
    x1, y1, x2, y2 = detection["box"]
    vehicle_type = detection["vehicle_type"]
    confidence = detection["confidence"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{vehicle_type} {confidence:.2f}"

    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_hud(
    frame,
    counts,
    display_fps,
    source_fps,
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
        f"Besar: {counts['kendaraan_besar']}"
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

    frame_buffer = LatestFrameBuffer()
    detection_buffer = DetectionBuffer()

    reader = CCTVReaderThread(STREAM_URL, frame_buffer)
    detector = DetectorThread(model, frame_buffer, detection_buffer)

    reader.start()
    detector.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    fps_timer = time.perf_counter()
    fps_count = 0
    display_fps = 0.0

    frame_delay = 1.0 / max(DISPLAY_TARGET_FPS, 1)

    try:
        while True:
            loop_start = time.perf_counter()

            frame, frame_seq, connected = frame_buffer.get()

            if frame is None:
                frame = 255 * cv2.UMat(DISPLAY_HEIGHT, DISPLAY_WIDTH, cv2.CV_8UC3).get()

                cv2.putText(
                    frame,
                    "Waiting for CCTV stream...",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                detections = []
                infer_ms = 0.0
                detect_fps = 0.0
                detect_seq = -1
            else:
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                detections, infer_ms, detect_fps, detect_seq = detection_buffer.get()

            counts = count_vehicle_types(detections)

            for detection in detections:
                draw_detection(frame, detection)

            fps_count += 1
            now = time.perf_counter()

            if now - fps_timer >= 1.0:
                display_fps = fps_count / (now - fps_timer)
                fps_timer = now
                fps_count = 0
            source_fps = frame_buffer.get_source_fps()

            draw_hud(
                frame=frame,
                counts=counts,
                display_fps=display_fps,
                source_fps=source_fps,
                detect_fps=detect_fps,
                infer_ms=infer_ms,
                connected=connected,
                frame_seq=frame_seq,
                detect_seq=detect_seq,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_delay - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        detector.stop()
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
