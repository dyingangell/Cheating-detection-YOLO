import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Add project root and project_files/ to path.
# project_files/ is needed because worker.py uses bare `from newArch import ...`
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "project_files"))

# ---------------------------------------------------------------------------
# Patch heavy imports BEFORE project_files modules are loaded.
# This prevents CUDA / Redis / GStreamer from being required during tests.
# ---------------------------------------------------------------------------
_yolo_mock = MagicMock()
_redis_mock = MagicMock()
_shm_mock = MagicMock()

sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("msgpack", MagicMock())
sys.modules.setdefault("redis", MagicMock())


def _make_engine():
    """
    Build a ProctoringEngine without touching GPU, Redis, or shared memory.
    All external I/O is replaced with MagicMock.
    """
    from project_files.newArch import ProctoringEngine

    engine = ProctoringEngine.__new__(ProctoringEngine)

    # Tuneable pose parameters (mirrors __init__ defaults after our fixes)
    engine.pose_base_radius = 0.38
    engine.threshold = 10.0
    engine.pose_calib_s = 30.0
    engine.pose_max_away_dist = 1.2
    engine.pose_walk_speed_threshold = 30.0
    engine.pose_score_k = 1.0
    engine.pose_angle_base = 0.45
    engine.pose_angle_weight = 0.0       # disabled for overhead cameras
    engine.pose_combine_mode = "max"
    engine.pose_foreshortening_strength = 0.4
    engine.pose_frame_height = 720.0
    engine.pose_debug = False
    engine.pose_save_clips = False

    # Runtime state
    engine.pose_state = {}
    engine.warned_persons = set()
    engine.person_gate = {}
    engine.warned_boxes_by_cam = {}
    engine.warned_iou_thr = 0.25
    engine.pose_warn_center_px = 140.0
    engine.decisions_key = "proctor_decisions"
    engine.confirm_cooldown_s = 120.0
    engine.pending_ttl_s = 3600.0
    engine.warned_box_ttl_s = 86400.0
    engine.start_time = 0.0
    engine.frameCount = 0
    engine.peopleAVG = 0
    engine.peopleMax = 0
    engine.cooldown = 3
    engine.last_save = {}
    engine.phone_counters = {}
    engine.detections = []
    engine.save_dir = "evidence_folder"

    # Mocked external deps
    engine.r = MagicMock()
    engine.pose_model = MagicMock()
    engine.shared_array = MagicMock()

    return engine


@pytest.fixture
def engine():
    return _make_engine()


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

def make_kpts(nose_x, nose_y, l_shoulder_x, l_shoulder_y, r_shoulder_x, r_shoulder_y,
              conf=0.9) -> np.ndarray:
    """
    Build a minimal COCO-17 keypoint array (17, 3).
    Only nose (0), left shoulder (5) and right shoulder (6) are set to real
    values; the rest are zeroed out (confidence 0 → ignored by the engine).
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [nose_x, nose_y, conf]           # nose
    kpts[5] = [l_shoulder_x, l_shoulder_y, conf]  # left shoulder
    kpts[6] = [r_shoulder_x, r_shoulder_y, conf]  # right shoulder
    return kpts


def make_box(cx, cy, w=100, h=160):
    """Build an (x1, y1, x2, y2) bounding box around a centre point."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
