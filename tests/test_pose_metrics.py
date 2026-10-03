"""
Tests for pure static methods in ProctoringEngine:
  - _pose_suspicion_from_kpts
  - _pose_suspicion_with_angle
  - _bbox_iou_xyxy

These require no GPU, Redis, or file system.
"""
import math
import numpy as np
import pytest

from pf.newArch import ProctoringEngine
from tests.conftest import make_kpts


# ── _pose_suspicion_from_kpts ────────────────────────────────────────────────

class TestPoseSuspicionFromKpts:

    def test_centered_nose_returns_zero_rel_x(self):
        # Nose exactly between shoulders → rel_nose_x == 0
        kpts = make_kpts(nose_x=640, nose_y=300,
                         l_shoulder_x=580, l_shoulder_y=400,
                         r_shoulder_x=700, r_shoulder_y=400)
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert ok
        assert abs(rx) < 1e-5

    def test_nose_shifted_right(self):
        # Nose shifted 60px right of centre; shoulder_w = 120 → rel_x ≈ 0.5
        kpts = make_kpts(nose_x=700, nose_y=400,
                         l_shoulder_x=580, l_shoulder_y=400,
                         r_shoulder_x=700, r_shoulder_y=400)
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert ok
        assert abs(rx - 0.5) < 1e-4

    def test_nose_shifted_left(self):
        kpts = make_kpts(nose_x=580, nose_y=400,
                         l_shoulder_x=580, l_shoulder_y=400,
                         r_shoulder_x=700, r_shoulder_y=400)
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert ok
        assert abs(rx - (-0.5)) < 1e-4

    def test_low_nose_confidence_returns_none(self):
        kpts = make_kpts(640, 300, 580, 400, 700, 400, conf=0.9)
        kpts[0, 2] = 0.1  # nose confidence too low
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert not ok
        assert rx is None

    def test_low_shoulder_confidence_returns_none(self):
        kpts = make_kpts(640, 300, 580, 400, 700, 400, conf=0.9)
        kpts[5, 2] = 0.1  # left shoulder confidence too low
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert not ok

    def test_shoulders_too_close_returns_none(self):
        # Left and right shoulder at almost the same x → shoulder_w < 1
        kpts = make_kpts(640, 300, 640, 400, 640.5, 400)
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert not ok

    def test_empty_keypoints_returns_none(self):
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(np.array([]))
        assert not ok

    def test_none_keypoints_returns_none(self):
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(None)
        assert not ok

    def test_shoulder_width_correct(self):
        kpts = make_kpts(640, 300, 580, 400, 700, 400)
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert ok
        assert abs(sw - 120.0) < 1e-4

    def test_rel_nose_y_upward(self):
        # Nose above shoulder midpoint → negative rel_y
        kpts = make_kpts(640, 300, 580, 400, 700, 400)
        rx, ry, sw, ok = ProctoringEngine._pose_suspicion_from_kpts(kpts)
        assert ok
        # mid_y = 400, nose_y = 300 → rel_y = (300-400)/120 ≈ -0.833
        assert ry < 0


# ── _pose_suspicion_with_angle ───────────────────────────────────────────────

class TestPoseSuspicionWithAngle:

    def test_returns_five_values_on_valid_kpts(self):
        kpts = make_kpts(640, 300, 580, 400, 700, 400)
        result = ProctoringEngine._pose_suspicion_with_angle(kpts)
        assert len(result) == 5
        rx, ry, sw, angle, ok = result
        assert ok
        assert isinstance(angle, float)

    def test_angle_is_in_range(self):
        kpts = make_kpts(640, 300, 580, 400, 700, 400)
        _, _, _, angle, ok = ProctoringEngine._pose_suspicion_with_angle(kpts)
        assert ok
        assert -math.pi <= angle <= math.pi

    def test_low_confidence_returns_false(self):
        kpts = make_kpts(640, 300, 580, 400, 700, 400)
        kpts[6, 2] = 0.05  # right shoulder confidence too low
        _, _, _, _, ok = ProctoringEngine._pose_suspicion_with_angle(kpts)
        assert not ok

    def test_empty_array_returns_false(self):
        _, _, _, _, ok = ProctoringEngine._pose_suspicion_with_angle(np.array([]))
        assert not ok


# ── _bbox_iou_xyxy ───────────────────────────────────────────────────────────

class TestBboxIou:

    def test_identical_boxes_iou_one(self):
        box = (100, 100, 200, 200)
        assert abs(ProctoringEngine._bbox_iou_xyxy(box, box) - 1.0) < 1e-6

    def test_no_overlap_iou_zero(self):
        a = (0, 0, 100, 100)
        b = (200, 200, 300, 300)
        assert ProctoringEngine._bbox_iou_xyxy(a, b) == 0.0

    def test_half_overlap(self):
        a = (0, 0, 100, 100)   # area = 10 000
        b = (50, 0, 150, 100)  # area = 10 000, overlap = 50×100 = 5000
        iou = ProctoringEngine._bbox_iou_xyxy(a, b)
        # union = 10000 + 10000 - 5000 = 15000 → iou = 5000/15000 ≈ 0.333
        assert abs(iou - 1 / 3) < 1e-4

    def test_contained_box(self):
        outer = (0, 0, 200, 200)   # area = 40 000
        inner = (50, 50, 150, 150)  # area = 10 000, fully inside outer
        iou = ProctoringEngine._bbox_iou_xyxy(outer, inner)
        # intersection = 10000, union = 40000 → iou = 0.25
        assert abs(iou - 0.25) < 1e-4

    def test_touching_edges_iou_zero(self):
        a = (0, 0, 100, 100)
        b = (100, 0, 200, 100)  # shares only an edge, no area overlap
        assert ProctoringEngine._bbox_iou_xyxy(a, b) == 0.0

    def test_bad_coords_returns_zero(self):
        assert ProctoringEngine._bbox_iou_xyxy(("a", "b", "c", "d"), (0, 0, 1, 1)) == 0.0
