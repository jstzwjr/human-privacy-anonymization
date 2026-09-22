"""COCO person 框评测的 CPU 回归测试。"""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_coco_person import evaluate_results


class CocoPersonEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.annotations_path = self.root / "instances.json"
        annotations = {
            "info": {},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "person.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "empty.jpg", "width": 100, "height": 100},
            ],
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            ],
        }
        self.annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
        self.metadata = {
            "format": "coco-person-bbox-v1",
            "annotation_sha256": hashlib.sha256(self.annotations_path.read_bytes()).hexdigest(),
            "image_ids": [1, 2],
            "category_id": 1,
            "score_threshold": 0.001,
            "model": "L",
        }
        self.results_path = self.root / "predictions.jsonl"
        self.output_path = self.root / "metrics.json"

    def evaluate(self, rows, metadata=None):
        records = [self.metadata if metadata is None else metadata, *rows]
        self.results_path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate_results(self.results_path, self.annotations_path, self.output_path)

    @staticmethod
    def perfect_rows():
        return [
            {"image_id": 1, "detections": [{"bbox": [10, 10, 20, 20], "score": 0.9}]},
            {"image_id": 2, "detections": []},
        ]

    def test_perfect_boxes_score_one_and_report_is_saved(self):
        result = self.evaluate(self.perfect_rows())
        self.assertAlmostEqual(result["metrics"]["AP"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP50"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP75"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AR100"], 1.0)
        saved = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["metrics"], result["metrics"])

    def test_empty_predictions_score_zero(self):
        rows = [{"image_id": image_id, "detections": []} for image_id in (1, 2)]
        result = self.evaluate(rows)
        self.assertEqual(result["metrics"]["AP50"], 0.0)
        self.assertEqual(result["metrics"]["AR100"], 0.0)

    def test_false_positive_on_negative_image_lowers_ap(self):
        rows = self.perfect_rows()
        rows[1]["detections"] = [{"bbox": [10, 10, 20, 20], "score": 0.99}]
        result = self.evaluate(rows)
        self.assertAlmostEqual(result["metrics"]["AP50"], 0.5)
        self.assertAlmostEqual(result["metrics"]["AR100"], 1.0)

    def test_zero_area_false_positive_is_retained_in_metrics(self):
        rows = self.perfect_rows()
        rows[1]["detections"] = [{"bbox": [10, 10, 0, 20], "score": 0.99}]
        result = self.evaluate(rows)
        self.assertAlmostEqual(result["metrics"]["AP50"], 0.5)
        self.assertAlmostEqual(result["metrics"]["AR100"], 1.0)

    def test_missing_image_is_rejected_even_when_it_has_no_person(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.perfect_rows()[:1])

    def test_duplicate_image_is_rejected(self):
        rows = self.perfect_rows()
        with self.assertRaises(ValueError):
            self.evaluate([rows[0], *rows])

    def test_unknown_image_is_rejected(self):
        metadata = dict(self.metadata, image_ids=[1, 3])
        rows = self.perfect_rows()
        rows[1]["image_id"] = 3
        with self.assertRaises(ValueError):
            self.evaluate(rows, metadata)

    def test_wrong_annotation_hash_is_rejected(self):
        metadata = dict(self.metadata, annotation_sha256="0" * 64)
        with self.assertRaises(ValueError):
            self.evaluate(self.perfect_rows(), metadata)

    def test_non_person_category_is_rejected(self):
        metadata = dict(self.metadata, category_id=2)
        with self.assertRaises(ValueError):
            self.evaluate(self.perfect_rows(), metadata)

    def test_nonfinite_detection_values_are_rejected(self):
        for field in ("score", "bbox"):
            for value in (float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    rows = self.perfect_rows()
                    detection = rows[0]["detections"][0]
                    if field == "score":
                        detection["score"] = value
                    else:
                        detection["bbox"][0] = value
                    with self.assertRaises(ValueError):
                        self.evaluate(rows)

    def test_invalid_boxes_and_scores_are_rejected(self):
        invalid_detections = [
            {"bbox": [10, 10, 20], "score": 0.9},
            {"bbox": [10, 10, -20, 20], "score": 0.9},
            {"bbox": [10, 10, 20, 20], "score": 1.1},
            {"bbox": [10, 10, 20, 20], "score": -0.1},
        ]
        for detection in invalid_detections:
            with self.subTest(detection=detection):
                rows = self.perfect_rows()
                rows[0]["detections"] = [detection]
                with self.assertRaises(ValueError):
                    self.evaluate(rows)


if __name__ == "__main__":
    unittest.main()
