"""SCRFD 人脸框与 COCO 人体框 IoA 关联指标的 CPU 回归测试。"""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_scrfd_coco_person import evaluate_results


class ScrfdCocoPersonEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.annotations_path = self.root / "instances.json"
        self.results_path = self.root / "predictions.jsonl"
        self.output_path = self.root / "metrics.json"
        self.annotations = {
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
                    "bbox": [10, 10, 40, 80],
                    "area": 3200,
                    "iscrowd": 0,
                }
            ],
        }

    def evaluate(self, rows, *, metadata_changes=None, ioa_threshold=0.9):
        self.annotations_path.write_text(json.dumps(self.annotations), encoding="utf-8")
        metadata = {
            "format": "scrfd-coco-person-ioa-v1",
            "annotation_sha256": hashlib.sha256(self.annotations_path.read_bytes()).hexdigest(),
            "image_ids": [1, 2],
            "category_id": 1,
            "score_threshold": 0.5,
            "model": "10g",
        }
        metadata.update(metadata_changes or {})
        self.results_path.write_text(
            "\n".join(json.dumps(record) for record in [metadata, *rows]) + "\n",
            encoding="utf-8",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate_results(
                self.results_path,
                self.annotations_path,
                self.output_path,
                ioa_threshold=ioa_threshold,
            )

    @staticmethod
    def perfect_rows():
        return [
            {"image_id": 1, "detections": [{"bbox": [20, 20, 10, 10], "score": 0.9}]},
            {"image_id": 2, "detections": []},
        ]

    def add_crowd(self, image_id):
        self.annotations["annotations"].append({
            "id": 2,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [0, 0, 100, 100],
            "area": 10000,
            "iscrowd": 1,
        })

    def test_small_face_inside_large_person_is_correct_and_report_is_saved(self):
        result = self.evaluate(self.perfect_rows())
        for value in result["metrics"].values():
            self.assertAlmostEqual(value, 1.0)
        self.assertEqual(result["counts"], {
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "ignored_detections": 0,
            "noncrowd_person_count": 1,
            "crowd_person_count": 0,
            "detection_count": 1,
        })
        saved = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["metrics"], result["metrics"])
        self.assertEqual(saved["counts"], result["counts"])

    def test_exact_ioa_threshold_is_inclusive(self):
        rows = self.perfect_rows()
        rows[0]["detections"][0]["bbox"] = [9, 20, 10, 10]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 1.0)

    def test_overlap_just_below_threshold_is_false_positive(self):
        rows = self.perfect_rows()
        rows[0]["detections"][0]["bbox"] = [8.9, 20, 10, 10]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["true_positives"], 0)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["counts"]["false_negatives"], 1)
        for value in result["metrics"].values():
            self.assertEqual(value, 0.0)

    def test_custom_threshold_changes_matching(self):
        rows = self.perfect_rows()
        rows[0]["detections"][0]["bbox"] = [9, 20, 10, 10]
        result = self.evaluate(rows, ioa_threshold=0.95)
        self.assertEqual(result["counts"]["true_positives"], 0)
        self.assertEqual(result["counts"]["false_positives"], 1)

    def test_threshold_one_accepts_fully_contained_face(self):
        result = self.evaluate(self.perfect_rows(), ioa_threshold=1.0)
        self.assertEqual(result["counts"]["true_positives"], 1)

    def test_duplicate_faces_match_one_person_only_once(self):
        rows = self.perfect_rows()
        rows[0]["detections"].append({"bbox": [21, 21, 10, 10], "score": 0.8})
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["counts"]["false_negatives"], 0)
        self.assertAlmostEqual(result["metrics"]["association_precision"], 0.5)
        self.assertAlmostEqual(result["metrics"]["person_recall"], 1.0)
        self.assertAlmostEqual(result["metrics"]["association_F1"], 2 / 3)
        # 满召回后的低分误检影响最终 precision，但不会降低插值 AP。
        self.assertAlmostEqual(result["metrics"]["association_AP"], 1.0)

    def test_partial_recall_uses_one_hundred_one_point_ap(self):
        self.annotations["annotations"].append({
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [60, 60, 20, 20],
            "area": 400,
            "iscrowd": 0,
        })
        result = self.evaluate(self.perfect_rows())
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["false_negatives"], 1)
        self.assertAlmostEqual(result["metrics"]["person_recall"], 0.5)
        self.assertAlmostEqual(result["metrics"]["association_precision"], 1.0)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 51 / 101)
        self.assertAlmostEqual(result["metrics"]["association_F1"], 2 / 3)

    def test_false_positive_on_negative_image_lowers_ap(self):
        rows = self.perfect_rows()
        rows[1]["detections"] = [{"bbox": [20, 20, 10, 10], "score": 0.99}]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 0.5)
        self.assertAlmostEqual(result["metrics"]["association_precision"], 0.5)
        self.assertAlmostEqual(result["metrics"]["person_recall"], 1.0)

    def test_empty_predictions_count_all_unmatched_people(self):
        result = self.evaluate([
            {"image_id": 1, "detections": []},
            {"image_id": 2, "detections": []},
        ])
        self.assertEqual(result["counts"]["false_negatives"], 1)
        self.assertEqual(result["counts"]["detection_count"], 0)
        for value in result["metrics"].values():
            self.assertEqual(value, 0.0)

    def test_crowd_can_ignore_multiple_faces_without_increasing_recall(self):
        self.add_crowd(image_id=2)
        rows = self.perfect_rows()
        rows[1]["detections"] = [
            {"bbox": [20, 20, 10, 10], "score": 0.99},
            {"bbox": [60, 60, 10, 10], "score": 0.98},
        ]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["false_positives"], 0)
        self.assertEqual(result["counts"]["ignored_detections"], 2)
        self.assertEqual(result["counts"]["noncrowd_person_count"], 1)
        self.assertEqual(result["counts"]["crowd_person_count"], 1)
        self.assertEqual(result["counts"]["detection_count"], 3)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 1.0)
        self.assertAlmostEqual(result["metrics"]["association_precision"], 1.0)
        self.assertAlmostEqual(result["metrics"]["person_recall"], 1.0)

    def test_regular_person_takes_priority_over_overlapping_crowd(self):
        self.add_crowd(image_id=1)
        self.annotations["annotations"].reverse()
        result = self.evaluate(self.perfect_rows())
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["ignored_detections"], 0)
        self.assertEqual(result["counts"]["false_negatives"], 0)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 1.0)

    def test_crowd_only_has_no_evaluable_person_gt(self):
        self.annotations["annotations"] = []
        self.add_crowd(image_id=1)
        result = self.evaluate(self.perfect_rows())
        self.assertEqual(result["counts"]["noncrowd_person_count"], 0)
        self.assertEqual(result["counts"]["crowd_person_count"], 1)
        self.assertEqual(result["counts"]["ignored_detections"], 1)
        self.assertEqual(result["counts"]["true_positives"], 0)
        self.assertEqual(result["counts"]["false_negatives"], 0)
        self.assertEqual(result["metrics"]["association_precision"], 0.0)
        for name in ("association_AP", "person_recall", "association_F1"):
            self.assertEqual(result["metrics"][name], -1.0)

    def test_no_gt_still_counts_false_positives(self):
        self.annotations["annotations"] = []
        result = self.evaluate(self.perfect_rows())
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["counts"]["ignored_detections"], 0)
        self.assertEqual(result["metrics"]["association_precision"], 0.0)
        for name in ("association_AP", "person_recall", "association_F1"):
            self.assertEqual(result["metrics"][name], -1.0)

    def test_more_than_one_hundred_faces_are_not_truncated(self):
        rows = self.perfect_rows()
        rows[1]["detections"] = [
            {"bbox": [90, 90, 1, 1], "score": 0.99} for _ in range(150)
        ]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["false_positives"], 150)
        self.assertEqual(result["counts"]["detection_count"], 151)
        self.assertAlmostEqual(result["metrics"]["association_precision"], 1 / 151)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 1 / 151)

    def test_zero_area_face_is_retained_as_false_positive(self):
        rows = self.perfect_rows()
        rows[1]["detections"] = [{"bbox": [20, 20, 0, 10], "score": 0.99}]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["counts"]["detection_count"], 2)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 0.5)

    def test_face_outside_image_keeps_original_area_denominator(self):
        self.annotations["annotations"][0].update(bbox=[0, 0, 100, 100], area=10000)
        rows = self.perfect_rows()
        rows[0]["detections"][0]["bbox"] = [-10, 10, 20, 10]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["true_positives"], 0)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["counts"]["false_negatives"], 1)

    def test_matching_uses_score_order_instead_of_saved_order(self):
        rows = self.perfect_rows()
        rows[0]["detections"] = [
            {"bbox": [20, 20, 10, 10], "score": 0.6},
            {"bbox": [21, 21, 10, 10], "score": 0.9},
        ]
        result = self.evaluate(rows)
        self.assertEqual(result["counts"]["true_positives"], 1)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertAlmostEqual(result["metrics"]["association_AP"], 1.0)

    def test_invalid_ioa_threshold_is_rejected(self):
        for threshold in (0.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.evaluate(self.perfect_rows(), ioa_threshold=threshold)

    def test_missing_image_is_rejected_even_if_it_has_no_person(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.perfect_rows()[:1])

    def test_duplicate_image_is_rejected(self):
        rows = self.perfect_rows()
        with self.assertRaises(ValueError):
            self.evaluate([rows[0], *rows])

    def test_unknown_image_is_rejected(self):
        rows = self.perfect_rows()
        rows[1]["image_id"] = 3
        with self.assertRaises(ValueError):
            self.evaluate(rows, metadata_changes={"image_ids": [1, 3]})

    def test_wrong_annotation_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.perfect_rows(), metadata_changes={"annotation_sha256": "0" * 64})

    def test_wrong_result_format_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.perfect_rows(), metadata_changes={"format": "coco-person-bbox-v1"})

    def test_non_person_category_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.perfect_rows(), metadata_changes={"category_id": 2})

    def test_invalid_detection_values_are_rejected(self):
        invalid_detections = [
            {"bbox": [20, 20, 10], "score": 0.9},
            {"bbox": [20, 20, -10, 10], "score": 0.9},
            {"bbox": [20, 20, 10, 10], "score": 1.1},
            {"bbox": [20, 20, 10, 10], "score": -0.1},
            {"bbox": [float("nan"), 20, 10, 10], "score": 0.9},
            {"bbox": [20, 20, float("inf"), 10], "score": 0.9},
            {"bbox": [20, 20, 10, 10], "score": float("nan")},
            {"bbox": [20, 20, 10, 10], "score": float("inf")},
        ]
        for detection in invalid_detections:
            rows = self.perfect_rows()
            rows[0]["detections"] = [detection]
            with self.subTest(detection=detection), self.assertRaises(ValueError):
                self.evaluate(rows)

    def test_existing_report_is_not_overwritten(self):
        self.output_path.write_text("已有结果", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.evaluate(self.perfect_rows())
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "已有结果")


if __name__ == "__main__":
    unittest.main()
