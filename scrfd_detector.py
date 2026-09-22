"""从 World Studio 迁移的独立 SCRFD 人脸检测接口，保留原推理行为。"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

# 保留 World Studio 的固定参数；预处理和解码常量与原 SCRFD 接口一致。
_DET_THRESHOLD = 0.5
_NMS_THRESHOLD = 0.4
_INPUT_SIZE = (640, 640)  # (width, height)
_FEAT_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
_FEATURE_MAP_COUNT = 3  # number of stride levels; SCRFD groups outputs as
# [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
_INPUT_MEAN = 127.5
_INPUT_STD = 128.0


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    w: float
    h: float
    score: float


class SCRFDDetector:
    """加载 SCRFD ONNX 模型，以固定尺寸、单次前向检测 RGB 图像中的人脸。"""

    def __init__(
        self,
        model_path: Path,
        *,
        intra_op_num_threads: int | None = None,
    ) -> None:
        # 保留延迟导入，只有创建检测器时才加载 ONNX Runtime。
        import onnxruntime

        available_providers = set(onnxruntime.get_available_providers())
        providers = [
            provider
            for provider in ('CUDAExecutionProvider', 'CPUExecutionProvider')
            if provider in available_providers
        ]
        if intra_op_num_threads is None:
            self._session = onnxruntime.InferenceSession(
                str(model_path),
                providers=providers,
            )
        else:
            session_options = onnxruntime.SessionOptions()
            session_options.intra_op_num_threads = intra_op_num_threads
            self._session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]
        if len(self._output_names) != 9:
            raise ValueError(
                f'Expected a 9-output SCRFD export (score/bbox/kps x 3 strides), '
                f'got {len(self._output_names)} outputs.'
            )

    def active_provider(self) -> str:
        return self._session.get_providers()[0]

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """输入 H×W×3、uint8、RGB 数组，返回原图坐标下的人脸框。"""
        det_image, det_scale = _resize_with_padding(image, _INPUT_SIZE)
        blob = _to_blob(det_image)
        outputs = self._session.run(self._output_names, {self._input_name: blob})

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for level, stride in enumerate(_FEAT_STRIDES):
            scores = outputs[level].reshape(-1)
            bbox_deltas = outputs[level + _FEATURE_MAP_COUNT] * stride
            anchor_centers = _anchor_centers(det_image.shape[0], det_image.shape[1], stride)
            keep = np.where(scores >= _DET_THRESHOLD)[0]
            if keep.size == 0:
                continue
            all_boxes.append(_distance_to_bbox(anchor_centers[keep], bbox_deltas[keep]))
            all_scores.append(scores[keep])

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes) / det_scale
        scores = np.concatenate(all_scores)
        keep = _nms(boxes, scores, _NMS_THRESHOLD)
        return [
            FaceBox(
                x=float(boxes[i, 0]),
                y=float(boxes[i, 1]),
                w=float(boxes[i, 2] - boxes[i, 0]),
                h=float(boxes[i, 3] - boxes[i, 1]),
                score=float(scores[i]),
            )
            for i in keep
        ]


def _resize_with_padding(
    image: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """等比例缩放并在右侧、底部补零，保持原 SCRFD 的坐标还原方式。"""
    height, width = image.shape[0], image.shape[1]
    target_width, target_height = input_size
    image_ratio = height / width
    model_ratio = target_height / target_width
    if image_ratio > model_ratio:
        new_height = target_height
        new_width = int(new_height / image_ratio)
    else:
        new_width = target_width
        new_height = int(new_width * image_ratio)
    det_scale = new_height / height
    resized = np.asarray(
        Image.fromarray(image).resize((new_width, new_height), Image.BILINEAR)
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[:new_height, :new_width, :] = resized
    return canvas, det_scale


def _to_blob(det_image: np.ndarray) -> np.ndarray:
    # 转 float32 后原位归一化，提供连续 NCHW 数组，避免推理时再复制转置视图。
    normalized = np.subtract(det_image, _INPUT_MEAN, dtype=np.float32)
    normalized *= 1.0 / _INPUT_STD
    return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[np.newaxis, ...])


@lru_cache(maxsize=None)
def _anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    grid_height, grid_width = height // stride, width // stride
    centers = np.stack(np.mgrid[:grid_height, :grid_width][::-1], axis=-1).astype(np.float32)
    centers = (centers * stride).reshape(-1, 2)
    if _NUM_ANCHORS > 1:
        centers = np.stack([centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)
    return centers


def _distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        overlap_w = np.maximum(0.0, xx2 - xx1 + 1)
        overlap_h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = overlap_w * overlap_h
        union = areas[i] + areas[order[1:]] - intersection
        overlap = intersection / union
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep
