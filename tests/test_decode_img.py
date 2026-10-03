"""
Tests for decode_img in worker.py:
  - raw JPEG bytes (legacy format)
  - JSON + base64 format (current format)
  - empty / None input
  - invalid data
"""
import base64
import json
import numpy as np
import cv2
import pytest

from pf.worker import decode_img


def _make_jpeg(width=64, height=64, color=(100, 150, 200)) -> bytes:
    """Create a small solid-colour JPEG frame and return raw bytes."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


# ── raw JPEG (legacy) ─────────────────────────────────────────────────────────

class TestDecodeRawJpeg:

    def test_returns_frame_and_unknown(self):
        jpeg = _make_jpeg()
        result = decode_img(jpeg)
        assert result is not None
        frame, cam_id = result
        assert cam_id == "unknown"
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3

    def test_frame_has_correct_shape(self):
        jpeg = _make_jpeg(width=80, height=60)
        frame, _ = decode_img(jpeg)
        assert frame.shape == (60, 80, 3)

    def test_jpeg_magic_bytes_detected(self):
        # JPEG starts with 0xFF 0xD8
        jpeg = _make_jpeg()
        assert jpeg[0] == 0xFF and jpeg[1] == 0xD8


# ── JSON + base64 (current format) ───────────────────────────────────────────

class TestDecodeJsonFormat:

    def _make_payload(self, cam_id="5", width=64, height=64) -> bytes:
        jpeg = _make_jpeg(width, height)
        obj = {
            "cam_id": cam_id,
            "img": base64.b64encode(jpeg).decode("ascii"),
        }
        return json.dumps(obj).encode("utf-8")

    def test_returns_correct_cam_id(self):
        payload = self._make_payload(cam_id="42")
        frame, cam_id = decode_img(payload)
        assert cam_id == "42"

    def test_returns_ndarray_frame(self):
        payload = self._make_payload()
        frame, _ = decode_img(payload)
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3

    def test_frame_shape(self):
        payload = self._make_payload(width=80, height=60)
        frame, _ = decode_img(payload)
        assert frame.shape == (60, 80, 3)

    def test_string_input_also_works(self):
        # decode_img should handle str as well as bytes
        payload = self._make_payload(cam_id="7").decode("utf-8")
        frame, cam_id = decode_img(payload)
        assert cam_id == "7"
        assert frame is not None

    def test_cam_id_is_always_string(self):
        jpeg = _make_jpeg()
        obj = {"cam_id": 99, "img": base64.b64encode(jpeg).decode("ascii")}
        payload = json.dumps(obj).encode("utf-8")
        frame, cam_id = decode_img(payload)
        assert isinstance(cam_id, str)
        assert cam_id == "99"


# ── edge cases ────────────────────────────────────────────────────────────────

class TestDecodeEdgeCases:

    def test_none_returns_none(self):
        assert decode_img(None) is None

    def test_empty_bytes_returns_none(self):
        assert decode_img(b"") is None

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            decode_img(b"not json and not jpeg")
