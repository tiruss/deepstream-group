try:
    from env_helper import *
except ImportError:
    from .env_helper import *

MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000

TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080

TRACKER_CONFIG_FILE = "configs/tracker.txt"

TRACKER_CONFIG_INT_KEYS = [
    "tracker-width",
    "tracker-height",
    "gpu-id",
]

UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF

USE_NEW_NVSTREAMMUX = get_bool("USE_NEW_NVSTREAMMUX", False)
FIX_FPS = get_bool("FIX_FPS", False)

# 0: CPU, 1: GPU
OSD_PROCESS_MODE = 1
OSD_DISPLAY_TEXT = 1
OSD_DISPLAY_BBOX = 1

SUPPORT_HW_ENC = True
MAX_HW_ENC = 3

CODEC = "H265"
BITRATE = 4000000
UDPSINK_PORT_NUM = 5000
RTSP_PORT_NUM = 8554

MAX_QUEUE_SIZE = 0

TRACKER_TIMEOUT = 10  # unused

# Database, meta, thumbnail configuration
ENABLE_DB = get_bool("ENABLE_DB", False)

HOST_IP = get_str("HOST_IP", "175.126.184.64")

DB_IP = HOST_IP
DB_PORT = 27017
DB_USER = "root"
DB_PASSWORD = "etri"
DB_NAME = "customslab"
COLLECTION_META = "meta"
COLLECTION_THUMBNAIL = "thumbnail"
COLLECTION_MATCH = "match"

META_INTERVAL = 15
META_BATCH_TIMEOUT = 0.1
META_MAX_BATCH_SIZE = 256

ENABLE_KAFKA = get_bool("ENABLE_KAFKA", False)

KAFKA_SERVER = HOST_IP + ":9092"
KAFKA_TOPIC_REID = "customslab-reid"
KAFKA_TOPIC_FACE = "customslab-face"
