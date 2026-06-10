STREAM_URL = "https://eofficev2.bekasikota.go.id/backupcctv/m3/Perempatan_Caman.m3u8"

CAMERA_NAME = "Perempatan Caman"

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

YOLO_MODEL = "models/yolo26n.pt"

CONF_THRESHOLD = 0.22
MODEL_IMGSZ = 640

# Detection target FPS. Ini bukan display FPS.
DETECTION_TARGET_FPS = 5

# Display FPS Target
DISPLAY_TARGET_FPS = 30

# Playback FPS.
# Ini yang mengatur speed pemutaran frame CCTV FIFO.
# Jangan terlalu tinggi supaya gerakan tidak jadi speed 2x.
PLAYBACK_TARGET_FPS = 15

# Reader FPS target.
# Jangan terlalu tinggi untuk HLS .m3u8, supaya buffer video tidak cepat habis.
SOURCE_TARGET_FPS = 15

# 72 frame di 12 FPS = sekitar 6 detik buffer maksimal
PLAYBACK_QUEUE_SIZE = 90

# Tunggu buffer terkumpul dulu sebelum mulai play.
# 24 frame di 12 FPS = sekitar 2 detik delay awal
PLAYBACK_START_BUFFER_FRAMES = 3

# Kalau buffer turun di bawah ini, pause sebentar untuk isi ulang.
# 12 frame di 12 FPS = sekitar 1 detik buffer minimal
PLAYBACK_RESUME_BUFFER_FRAMES = 3

# Filter box kecil supaya objek jauh/noise tidak terlalu banyak
MIN_BOX_AREA = 220

# Buang box yang terlalu besar dibanding frame.
# Ini bantu cegah area jalan / gabungan kendaraan kebaca sebagai 1 object besar.
MAX_BOX_AREA_RATIO = 0.28

# Ukuran minimum box.
MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 8

# Confidence per tipe kendaraan.
# Motor boleh sedikit lebih rendah karena motor CCTV sering kecil/jauh.
MIN_MOTOR_CONF = 0.28
MIN_MOBIL_CONF = 0.30

# Filter bentuk box.
# Terlalu gepeng / terlalu tinggi biasanya noise atau salah deteksi.
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 4.80

# ROI area jalan untuk Perempatan Caman.
# Format: titik polygon dalam koordinat frame 960x540.
# Ini masih dibuat lebar dulu, nanti bisa disempitkan setelah test.
# ROI utama object detection.
# Dibikin lebih fokus ke area jalan utama Perempatan Caman.
# Area background atas/semak/gedung dikurangi supaya false detection turun.
ROI_POLYGON = [
    (0, 145),
    (150, 125),
    (360, 145),
    (650, 125),
    (960, 110),
    (960, 540),
    (0, 540),
]

# Debug garis zone.
# True dulu buat tuning titik polygon.
SHOW_ZONE_DEBUG = True

# Area biru:
# Kendaraan yang masuk ke area ini dianggap IN,
# asal tidak sedang bergerak balik ke zona atas.
IN_ZONE_POLYGON = [
    (0, 315),
    (150, 275),
    (355, 285),
    (535, 430),
    (520, 540),
    (0, 540),
]

# Area oranye:
# Kendaraan yang masuk ke area ini dianggap OUT.
OUT_ZONE_POLYGON = [
    (0, 145),
    (210, 125),
    (390, 185),
    (360, 255),
    (155, 245),
    (0, 230),
]

# Zona atas / arah datang kendaraan normal.
# Kalau track pernah masuk area biru/oranye lalu balik ke sini,
# itu kandidat LAWAN_ARAH.
UPSTREAM_ZONE_POLYGON = [
    (0, 105),
    (960, 90),
    (960, 230),
    (500, 235),
    (250, 185),
    (0, 230),
]

VIOLATION_ZONE_POLYGON = [
    (390, 95),
    (960, 95),
    (960, 335),
    (720, 280),
    (520, 220),
    (390, 160),
]

# Kalau jarak seq detection terlalu jauh dari frame tampil,
# jangan gambar box supaya tidak terlihat ngawur/telat.
# Karena detector sekarang baca playback_frame,
# seq gap bisa dibuat lebih ketat lagi.
MAX_DETECTION_SEQ_GAP = 25

# Tracking/trajectory.
TRAJECTORY_MAX_POINTS = 80
TRACK_MATCH_DISTANCE = 90
TRACK_MAX_MISSED_FRAMES = 60

# Minimal perpindahan untuk fallback arah.
DIRECTION_MIN_MOVE_PX = 35

# Kalau YOLO miss sebentar, box tetap ditahan beberapa frame
# supaya tidak kedip-kedip.
TRACK_DRAW_HOLD_FRAMES = 45

# Kecilkan box hanya untuk tampilan.
# Ini tidak mengubah center tracking / logic direction.
DRAW_BOX_SHRINK_RATIO = 0.10

# Deteksi pelanggaran lawan arah.
# Object dianggap lawan arah kalau sudah pernah masuk zone IN/OUT,
# lalu bergerak naik cukup jauh.
WRONG_WAY_RECENT_UP_MOVE_PX = 18
WRONG_WAY_NET_UP_MOVE_PX = 30

# Tracker bawaan Ultralytics.
TRACKER_CONFIG = "botsort.yaml"


RECONNECT_DELAY_SEC = 3

# COCO class id:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
