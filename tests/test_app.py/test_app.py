import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def make_landmarks():
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
    landmarks[10].y = 0.1
    landmarks[152].y = 0.9
    landmarks[234].x = 0.2
    landmarks[454].x = 0.8
    landmarks[13].y = 0.49
    landmarks[14].y = 0.51
    landmarks[61].x = 0.42
    landmarks[291].x = 0.58
    landmarks[159].y = 0.40
    landmarks[145].y = 0.44
    landmarks[105].y = 0.50
    landmarks[33].x = 0.35
    landmarks[133].x = 0.45
    landmarks[386].y = 0.40
    landmarks[374].y = 0.44
    landmarks[334].y = 0.50
    landmarks[362].x = 0.55
    landmarks[263].x = 0.65
    return landmarks


class FaceMemeStudioTests(unittest.TestCase):
    def test_face_signature_has_one_value_per_measurement(self):
        signature = app.extract_face_signature(make_landmarks())
        self.assertEqual(len(signature), len(app.FEATURE_SCALES))

    def test_nearest_match_selects_each_photo_individually(self):
        neutral_landmarks = make_landmarks()
        open_mouth_landmarks = make_landmarks()
        open_mouth_landmarks[14].y = 0.30
        neutral_entry = app.MemeEntry(
            Path("meme_001.jpg"), app.extract_face_signature(neutral_landmarks), None
        )
        open_mouth_entry = app.MemeEntry(
            Path("meme_002.jpg"), app.extract_face_signature(open_mouth_landmarks), None
        )

        self.assertIs(
            app.find_best_meme(neutral_entry.signature, [neutral_entry, open_mouth_entry]),
            neutral_entry,
        )
        self.assertIs(
            app.find_best_meme(open_mouth_entry.signature, [neutral_entry, open_mouth_entry]),
            open_mouth_entry,
        )

    def test_feature_distance_is_lower_for_matching_faces(self):
        first = app.extract_face_signature(make_landmarks())
        different_landmarks = make_landmarks()
        different_landmarks[14].y = 0.30
        second = app.extract_face_signature(different_landmarks)
        self.assertEqual(app.meme_feature_distance(first, first), 0.0)
        self.assertGreater(app.meme_feature_distance(first, second), 0.0)

    def test_smoothing_reduces_single_frame_changes(self):
        smoothed = app.smooth_signature((0.0, 0.0), (1.0, 1.0), weight=0.25)
        self.assertEqual(smoothed, (0.25, 0.25))

    def test_match_switch_requires_a_clear_improvement(self):
        self.assertFalse(app.is_clear_match_improvement(0.45, 0.60))
        self.assertTrue(app.is_clear_match_improvement(0.30, 0.60))
        self.assertTrue(app.is_clear_match_improvement(0.30, None))

    def test_photos_without_faces_are_not_auto_matched(self):
        entry = app.MemeEntry(Path("landscape.jpg"), None, None)
        self.assertIsNone(app.find_best_meme((0.0,) * len(app.FEATURE_SCALES), [entry]))

    def test_face_box_stays_inside_frame(self):
        box = app.find_face_box(make_landmarks(), 640, 480)
        self.assertIsNotNone(box)
        left, top, right, bottom = box
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 640)
        self.assertLessEqual(bottom, 480)

    def test_resize_to_cover_has_target_size(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = app.resize_to_cover(image, 80, 120)
        self.assertEqual(resized.shape, (120, 80, 3))

    def test_flat_gallery_keeps_photos_independent(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            root = Path(temporary_folder)
            (root / "meme_001.png").touch()
            (root / "meme_002.jpg").touch()
            (root / "notes.txt").touch()
            (root / "nested").mkdir()
            (root / "nested" / "meme_003.jpg").touch()
            photos = app.find_meme_files(root)
            self.assertEqual([path.name for path in photos], ["meme_001.png", "meme_002.jpg"])

    def test_load_meme_image_keeps_context_around_face(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            path = Path(temporary_folder) / "portrait.png"
            cv2.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint8))
            entry = app.MemeEntry(path, None, (0.25, 0.25, 0.75, 0.75))
            image = app.load_meme_image(entry)
            self.assertEqual(image.shape[:2], (100, 100))

    def test_face_mesh_has_full_feature_connections(self):
        self.assertGreater(len(app.FACE_CONNECTIONS), 2000)


if __name__ == "__main__":
    unittest.main()
