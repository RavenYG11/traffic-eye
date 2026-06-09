STREAM_URL = "https://eofficev2.bekasikota.go.id/backupcctv/m3/Perempatan_Caman.m3u8"

CAMERA_NAME = "Perempatan Caman"

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

YOLO_MODEL = "models/yolo26n.pt"

CONF_THRESHOLD = 0.35
MODEL_IMGSZ = 640

# Detection target FPS. Ini bukan display FPS.
DETECTION_TARGET_FPS = 4

# Display FPS Target
DISPLAY_TARGET_FPS = 30

# Playback FPS.
# Ini yang mengatur speed pemutaran frame CCTV FIFO.
# Jangan terlalu tinggi supaya gerakan tidak jadi speed 2x.
PLAYBACK_TARGET_FPS = 25

# Reader FPS target.
# Jangan terlalu tinggi untuk HLS .m3u8, supaya buffer video tidak cepat habis.
SOURCE_TARGET_FPS = 50

# 72 frame di 12 FPS = sekitar 6 detik buffer maksimal
PLAYBACK_QUEUE_SIZE = 600

# Tunggu buffer terkumpul dulu sebelum mulai play.
# 24 frame di 12 FPS = sekitar 2 detik delay awal
PLAYBACK_START_BUFFER_FRAMES = 1

# Kalau buffer turun di bawah ini, pause sebentar untuk isi ulang.
# 12 frame di 12 FPS = sekitar 1 detik buffer minimal
PLAYBACK_RESUME_BUFFER_FRAMES = 1

# Filter box kecil supaya objek jauh/noise tidak terlalu banyak
MIN_BOX_AREA = 450

RECONNECT_DELAY_SEC = 3

# COCO class id:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
