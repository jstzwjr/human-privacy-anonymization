"""需设置 SCRFD_TEST_CUDA=1 的真实 GPU 回归测试，使用项目已有模型和图片。"""
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scrfd_detector import SCRFDDetector

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("SCRFD_TEST_CUDA") == "1", "需设置 SCRFD_TEST_CUDA=1 并提供 GPU、模型和测试图片")
class ScrfdCudaTests(unittest.TestCase):
    def test_cuda_initialization_and_detections_match_cpu(self):
        import onnxruntime

        images = [np.zeros((640, 640, 3), dtype=np.uint8)]
        for name in ("bus.jpg", "zidane.jpg"):
            with Image.open(ROOT / "test_images" / name) as image:
                images.append(np.asarray(image.convert("RGB")))
        for model in ("500m", "10g"):
            with self.subTest(model=model):
                model_path = ROOT / "model" / "scrfd" / f"det_{model}.onnx"
                # 创建 GPU 会话前不导入 torch、不手动预加载，覆盖此次动态库加载故障。
                gpu = SCRFDDetector(model_path, intra_op_num_threads=1)
                self.assertEqual(gpu.active_provider(), "CUDAExecutionProvider")
                with patch.object(onnxruntime, "get_available_providers", return_value=["CPUExecutionProvider"]):
                    cpu = SCRFDDetector(model_path, intra_op_num_threads=1)
                self.assertEqual(cpu.active_provider(), "CPUExecutionProvider")
                for index, image in enumerate(images):
                    with self.subTest(image=index):
                        gpu_faces, cpu_faces = gpu.detect(image), cpu.detect(image)
                        self.assertEqual(len(gpu_faces), 0 if index == 0 else 2)
                        self.assertEqual(len(gpu_faces), len(cpu_faces))
                        for actual, expected in zip(gpu_faces, cpu_faces):
                            actual_values = list(asdict(actual).values())
                            expected_values = list(asdict(expected).values())
                            np.testing.assert_allclose(actual_values[:4], expected_values[:4], rtol=0, atol=0.1)
                            self.assertAlmostEqual(actual.score, expected.score, delta=0.001)


if __name__ == "__main__":
    unittest.main()
