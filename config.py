STREAM_URL = "https://eofficev2.bekasikota.go.id/backupcctv/m3/Perempatan_Caman.m3u8"

CAMERA_NAME = "Perempatan Caman"

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

YOLO_MODEL = "models/yolov8n.pt"

CONF_THRESHOLD = 0.35
MODEL_IMGSZ = 416

# Detection target FPS. Ini bukan display FPS.
DETECTION_TARGET_FPS = 4

# Display FPS Target
DISPLAY_TARGET_FPS = 30

# Filter box kecil supaya objek jauh/noise tidak terlalu banyak
MIN_BOX_AREA = 450

RECONNECT_DELAY_SEC = 3

# COCO class id:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
