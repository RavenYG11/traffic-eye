import cv2
import time
from ultralytics import YOLO

from config import (
    STREAM_URL,
    CAMERA_NAME,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    YOLO_MODEL,
    CONF_THRESHOLD,
    VEHICLE_CLASS_IDS,
)


WINDOW_NAME = "TRAFFIC_EYE AI - CCTV Vehicle Detection"


def map_vehicle_type(class_id: int) -> str:
    if class_id == 3:
        return "motor"

    if class_id == 2:
        return "mobil"

    if class_id in [5, 7]:
        return "kendaraan_besar"

    return "unknown"


def open_stream():
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("[ERROR] Stream tidak bisa dibuka.")
        return None

    print(f"[OK] Stream terbuka: {CAMERA_NAME}")
    return cap


def draw_detection(frame, box, label, confidence):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{label} {confidence:.2f}"

    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    print("[INFO] Load YOLO model...")
    model = YOLO(YOLO_MODEL)
    print(f"[OK] Model loaded: {YOLO_MODEL}")

    cap = open_stream()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        if cap is None:
            print("[INFO] Reconnect stream dalam 3 detik...")
            time.sleep(3)
            cap = open_stream()
            continue

        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] Frame gagal dibaca. Reconnect...")
            cap.release()
            cap = None
            continue

        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )

        vehicle_count = {
            "motor": 0,
            "mobil": 0,
            "kendaraan_besar": 0,
        }

        for result in results:
            boxes = result.boxes

            for detected_box in boxes:
                class_id = int(detected_box.cls[0])
                confidence = float(detected_box.conf[0])
                xyxy = detected_box.xyxy[0].tolist()

                vehicle_type = map_vehicle_type(class_id)

                if vehicle_type == "unknown":
                    continue

                vehicle_count[vehicle_type] += 1

                draw_detection(
                    frame=frame,
                    box=xyxy,
                    label=vehicle_type,
                    confidence=confidence,
                )

        cv2.putText(
            frame,
            CAMERA_NAME,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        counter_text = (
            f"Motor: {vehicle_count['motor']} | "
            f"Mobil: {vehicle_count['mobil']} | "
            f"Besar: {vehicle_count['kendaraan_besar']}"
        )

        cv2.putText(
            frame,
            counter_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if cap:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()