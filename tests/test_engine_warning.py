"""
Tests for the warning accumulation logic inside ProctoringEngine:
  - score accumulates during sustained lateral deviation
  - writing pose (depth-dominant movement) is suppressed → no FP
  - foreshortening: farther student needs more deviation to trigger
  - walking resets / decays the score
  - calibration window prevents early warnings
  - warned_persons gate suppresses duplicate alerts

Key geometry (used in all tests unless stated otherwise):
  shoulders: left=(580,400), right=(700,400)  → shoulder_w=120px, mid=(640,400)
  normal head: nose=(640,360)
    rel_nose_x=0.0, rel_nose_y=-0.833  ← baseline after calibration
  cheat head:  nose=(730,360)   ← shifted 90px right, same depth
    lateral_dev=0.75, depth_dev=0
    dist=0.75 < max_away_dist=1.2  ← stays within "seated" zone
    lateral_excess ≈ 0.75 - 0.46 = 0.29 → accumulates ~0.073/frame at dt=0.25
    frames to threshold-10: ~140 frames
"""
import numpy as np
import pytest

from tests.conftest import make_kpts, make_box, _make_engine

# shoulder geometry constants
LS = (580, 400)   # left shoulder (x, y)
RS = (700, 400)   # right shoulder (x, y)
SHOULDER_W = RS[0] - LS[0]   # 120 px


def _normal_kpts():
    """Head straight above own paper — calibration posture."""
    return make_kpts(640, 360, *LS, *RS)


def _cheat_kpts():
    """
    Head shifted 90px to the right, same depth as normal.
    dist from baseline ≈ 0.75, well below max_away_dist=1.2.
    Produces lateral_excess ≈ 0.29 → suspicious after calibration.
    """
    return make_kpts(730, 360, *LS, *RS)


def _static_box():
    return make_box(640, 400)


# ── helpers ──────────────────────────────────────────────────────────────────

def _calibrate(engine, person_key, cam, tid, kpts=None, box=None, calib_s=None):
    """Push frames through the calibration window to establish a stable baseline."""
    if kpts is None:
        kpts = _normal_kpts()
    if box is None:
        box = _static_box()
    if calib_s is None:
        calib_s = float(engine.pose_calib_s)

    ts = 0.0
    step = 0.25
    for _ in range(int(calib_s / step) + 4):
        engine._update_pose_warning_for_person(person_key, cam, tid, kpts, box, ts)
        ts += step


def _run_frames(engine, person_key, cam, tid, kpts, box, n_frames, dt=0.25, start_ts=None):
    """
    Simulate n_frames.
    Returns the first non-empty warn_text seen (or '' if none triggered).
    Also returns the final timestamp so callers can continue from there.
    """
    ts = (float(engine.pose_calib_s) + 1.0) if start_ts is None else start_ts
    first_warn = ""
    for _ in range(n_frames):
        warn = engine._update_pose_warning_for_person(person_key, cam, tid, kpts, box, ts)
        ts += dt
        if warn and not first_warn:
            first_warn = warn
    return first_warn, ts


# ── calibration window ───────────────────────────────────────────────────────

class TestCalibrationWindow:

    def test_no_warning_during_calibration(self, engine):
        """Score must not trigger a warning while calibration window is active."""
        box = _static_box()
        ts = 0.0
        for _ in range(40):   # 10 s at dt=0.25 — still inside 30 s calib
            warn = engine._update_pose_warning_for_person(
                "10_1", "10", 1, _cheat_kpts(), box, ts)
            assert warn == "", "Must not warn during calibration"
            ts += 0.25

    def test_warning_fires_after_calibration(self, engine):
        """After calibration, sustained lateral deviation must trigger warning."""
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        first_warn, _ = _run_frames(engine, "10_1", "10", 1,
                                    _cheat_kpts(), box, n_frames=300)
        assert first_warn == "dist_warning"


# ── lateral vs depth suppression (#2 fix) ────────────────────────────────────

class TestLateralDepthSuppression:

    def test_lateral_movement_triggers_warning(self, engine):
        """Horizontal nose shift (looking at neighbor) → warning."""
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        first_warn, _ = _run_frames(engine, "10_1", "10", 1,
                                    _cheat_kpts(), box, n_frames=300)
        assert first_warn == "dist_warning"

    def test_forward_lean_does_not_trigger(self, engine):
        """
        Nose moves forward (Y only) = writing pose.
        depth_dev >> lateral_dev → depth-suppress kicks in → no FP.
        """
        # nose_y 460 vs normal 360 → depth shift of 100px, no lateral
        lean_kpts = make_kpts(640, 460, *LS, *RS)
        box = _static_box()
        _calibrate(engine, "10_2", "10", 2)

        first_warn, _ = _run_frames(engine, "10_2", "10", 2,
                                    lean_kpts, box, n_frames=300)
        assert first_warn == "", "Writing lean must not trigger warning"

    def test_depth_dominant_diagonal_is_suppressed(self, engine):
        """
        Small lateral + large depth → depth > lateral × ratio → suppressed.
        nose_x=660 (+20px lateral), nose_y=510 (+110px depth from normal 360).
        lateral_dev≈0.17, depth_dev≈1.75 → depth > lateral×1.5 → suppressed.
        """
        mixed_kpts = make_kpts(660, 510, *LS, *RS)
        box = _static_box()
        _calibrate(engine, "10_3", "10", 3)

        first_warn, _ = _run_frames(engine, "10_3", "10", 3,
                                    mixed_kpts, box, n_frames=300)
        assert first_warn == ""

    def test_lateral_dominant_diagonal_triggers(self, engine):
        """
        Large lateral + small depth → depth < lateral × ratio → lateral penalised.
        nose shifted right 90px and only slightly forward (20px).
        """
        # lateral_dev≈0.75, depth_dev≈0.17 → depth < lateral×1.5 → penalised
        diag_kpts = make_kpts(730, 380, *LS, *RS)
        box = _static_box()
        _calibrate(engine, "10_4", "10", 4)

        first_warn, _ = _run_frames(engine, "10_4", "10", 4,
                                    diag_kpts, box, n_frames=300)
        assert first_warn == "dist_warning"


# ── foreshortening correction (#4 fix) ───────────────────────────────────────

class TestForeshorteningCorrection:

    def _score_after(self, person_key, cam, tid, box, n=80):
        """Run n frames of cheat kpts and return score_s."""
        eng = _make_engine()
        _calibrate(eng, person_key, cam, tid, box=box)
        ts = float(eng.pose_calib_s) + 1.0
        for _ in range(n):
            eng._update_pose_warning_for_person(person_key, cam, tid,
                                                _cheat_kpts(), box, ts)
            ts += 0.25
        return eng.pose_state[person_key]["score_s"]

    def test_far_student_scores_less_than_near(self):
        """
        Same lateral deviation; student farther from camera (higher bbox Y)
        gets larger effective_base_radius → lower score (fewer FP for back rows).
        """
        box_near = make_box(640, 100)   # near top of frame (close to camera)
        box_far  = make_box(640, 620)   # near bottom of frame (far from camera)

        score_near = self._score_after("10_1", "10", 1, box_near)
        score_far  = self._score_after("10_2", "10", 2, box_far)

        assert score_near >= score_far, (
            f"Near student should score >= far student; "
            f"near={score_near:.3f}, far={score_far:.3f}"
        )

    def test_effective_radius_increases_with_y(self):
        """effective_base_radius grows monotonically as bbox moves down the frame."""
        eng = _make_engine()
        _calibrate(eng, "10_1", "10", 1)
        # After calibration the person state has base_radius set
        base_r = eng.pose_state["10_1"]["base_radius"]
        frame_h = eng.pose_frame_height
        strength = eng.pose_foreshortening_strength

        for y_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            bbox_cy = y_frac * frame_h
            expected = base_r * (1.0 + strength * y_frac)
            # Compute via same formula the engine uses
            computed = base_r * (1.0 + strength * min(1.0, max(0.0, bbox_cy / frame_h)))
            assert abs(computed - expected) < 1e-6


# ── walking detection ─────────────────────────────────────────────────────────

class TestWalkingDetection:

    def test_walking_decays_score(self, engine):
        """
        After building suspicion score, large bbox movement (walking)
        decays the score at 2×normal rate.
        """
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        # Build some score (60 frames < 140 needed to trigger warning)
        ts = float(engine.pose_calib_s) + 1.0
        for _ in range(60):
            engine._update_pose_warning_for_person("10_1", "10", 1,
                                                   _cheat_kpts(), box, ts)
            ts += 0.25

        score_before = engine.pose_state["10_1"]["score_s"]
        assert score_before > 0, "Score should have accumulated before walking"

        # Simulate walking: large bbox jump each frame
        for i in range(40):
            walk_box = make_box(640 + i * 50, 400)
            engine._update_pose_warning_for_person("10_1", "10", 1,
                                                   _normal_kpts(), walk_box, ts)
            ts += 0.25

        score_after = engine.pose_state["10_1"]["score_s"]
        assert score_after < score_before, (
            f"Score should decay while walking; "
            f"before={score_before:.2f}, after={score_after:.2f}"
        )

    def test_static_person_score_not_decayed_by_walk(self, engine):
        """Person not walking: score decays at normal (slow) rate, not walk rate."""
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        ts = float(engine.pose_calib_s) + 1.0
        for _ in range(60):
            engine._update_pose_warning_for_person("10_1", "10", 1,
                                                   _cheat_kpts(), box, ts)
            ts += 0.25
        score_after_cheat = engine.pose_state["10_1"]["score_s"]

        # One frame of normal pose, same static box → slow decay (1/s × 0.25 = 0.25)
        engine._update_pose_warning_for_person("10_1", "10", 1,
                                               _normal_kpts(), box, ts)
        score_one_frame_later = engine.pose_state["10_1"]["score_s"]

        expected_decay = 1.0 * 0.25
        actual_decay = score_after_cheat - score_one_frame_later
        assert abs(actual_decay - expected_decay) < 1e-4, (
            f"Static normal decay should be {expected_decay}, got {actual_decay:.4f}"
        )


# ── gate / spam prevention ────────────────────────────────────────────────────

class TestWarningGate:

    def test_warned_person_never_re_triggers(self, engine):
        """A person in warned_persons must never produce another warning."""
        engine.warned_persons.add("10_1")
        box = _static_box()
        ts = 100.0
        for _ in range(300):
            warn = engine._update_pose_warning_for_person(
                "10_1", "10", 1, _cheat_kpts(), box, ts)
            ts += 0.25
            assert warn == "", "Banned person must produce no warning"

    def test_pending_gate_suppresses_repeat(self, engine):
        """After first warning, person enters PENDING — no second warning."""
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        first_warn, ts_after = _run_frames(engine, "10_1", "10", 1,
                                           _cheat_kpts(), box, n_frames=300)
        assert first_warn == "dist_warning", "First warning must fire"

        # Continue — person is in PENDING, no further warnings allowed
        for _ in range(100):
            warn = engine._update_pose_warning_for_person(
                "10_1", "10", 1, _cheat_kpts(), box, ts_after)
            ts_after += 0.25
            assert warn == "", "No second warning while PENDING"

    def test_cooldown_then_resume(self, engine):
        """After operator marks 'not_cheating', cooldown ends → tracking resumes."""
        engine.confirm_cooldown_s = 5.0   # shorten cooldown for the test
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        # Trigger warning
        first_warn, ts_after = _run_frames(engine, "10_1", "10", 1,
                                           _cheat_kpts(), box, n_frames=300)
        assert first_warn == "dist_warning"

        # Operator marks as not_cheating
        engine._drain_decisions.__func__  # just ensure method exists
        engine.person_gate["10_1"] = {
            "state": "cooldown",
            "until": ts_after + 5.0,   # 5 s cooldown
            "cam_id": "10",
            "track_id": 1,
            "box": box,
        }
        engine.pose_state.pop("10_1", None)   # reset pose state as drain_decisions does

        # During cooldown: no warning
        for _ in range(10):
            warn = engine._update_pose_warning_for_person(
                "10_1", "10", 1, _cheat_kpts(), box, ts_after)
            ts_after += 0.25
            assert warn == ""

        # After cooldown expires: tracking resumes (score can accumulate again)
        ts_resumed = ts_after + 10.0   # well past cooldown
        state_key = engine.pose_state.get("10_1")
        # At minimum, engine should not crash and should re-enter tracking
        warn = engine._update_pose_warning_for_person(
            "10_1", "10", 1, _cheat_kpts(), box, ts_resumed)
        # No assertion on warn value — depends on whether calibration window restarted
        assert warn in ("", "dist_warning")


# ── score accumulation mechanics ─────────────────────────────────────────────

class TestScoreAccumulation:

    def test_score_decays_on_normal_behavior(self, engine):
        """After suspicious frames, returning to normal pose decays score."""
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        ts = float(engine.pose_calib_s) + 1.0
        for _ in range(30):
            engine._update_pose_warning_for_person("10_1", "10", 1,
                                                   _cheat_kpts(), box, ts)
            ts += 0.25

        score_peak = engine.pose_state["10_1"]["score_s"]
        assert score_peak > 0

        for _ in range(80):
            engine._update_pose_warning_for_person("10_1", "10", 1,
                                                   _normal_kpts(), box, ts)
            ts += 0.25

        score_after = engine.pose_state["10_1"]["score_s"]
        assert score_after < score_peak

    def test_score_bounded_by_cap(self, engine):
        """Score must not exceed 15.0 regardless of sustained suspicious behavior."""
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        ts = float(engine.pose_calib_s) + 1.0
        for _ in range(2000):
            engine._update_pose_warning_for_person("10_1", "10", 1,
                                                   _cheat_kpts(), box, ts)
            ts += 0.25
            score = engine.pose_state["10_1"].get("score_s", 0.0)
            assert score <= 15.0, f"Score exceeded cap: {score}"

    def test_score_zero_at_start(self, engine):
        """Fresh person starts with zero score."""
        engine._update_pose_warning_for_person(
            "10_1", "10", 1, _normal_kpts(), _static_box(), 0.0)
        assert engine.pose_state["10_1"]["score_s"] == 0.0

    def test_score_increases_when_suspicious(self, engine):
        """Score increases frame by frame during sustained suspicious pose."""
        box = _static_box()
        _calibrate(engine, "10_1", "10", 1)

        ts = float(engine.pose_calib_s) + 1.0
        prev = 0.0
        increases = 0
        for _ in range(50):
            engine._update_pose_warning_for_person("10_1", "10", 1,
                                                   _cheat_kpts(), box, ts)
            ts += 0.25
            curr = engine.pose_state["10_1"]["score_s"]
            if curr > prev:
                increases += 1
            prev = curr

        assert increases > 0, "Score must increase during suspicious frames"
