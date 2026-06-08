STREAM_URL = "https://eofficev2.bekasikota.go.id/backupcctv/m3/Perempatan_Caman.m3u8"

CAMERA_NAME = "Perempatan Caman"

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

YOLO_MODEL = "models/yolov8n.pt"
CONF_THRESHOLD = 0.35

# COCO class id:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]