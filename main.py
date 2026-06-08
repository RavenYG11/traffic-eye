import cv2
import time
from config import STREAM_URL, CAMERA_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT


def open_stream():
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("[ERROR] Stream tidak bisa dibuka.")
        return None

    print(f"[OK] Stream terbuka: {CAMERA_NAME}")
    return cap


def main():
    cap = open_stream()

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

        cv2.putText(
            frame,
            CAMERA_NAME,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("TRAFFIC_EYE AI - CCTV Preview", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if cap:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()