"""Reproducible three-image RF-DETR test; synchronized predict API timings."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRSegMedium, RFDETRSegLarge, RFDETRSeg2XLarge
from rfdetr.assets.coco_classes import COCO_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['M', 'L', '2XL'], required=True)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--repeats', type=int, default=10)
args = parser.parse_args()
assert args.warmup >= 1 and args.repeats >= 1
root = Path(__file__).resolve().parent
cls, filename = {'M': (RFDETRSegMedium, 'rf-detr-seg-medium.pt'), 'L': (RFDETRSegLarge, 'rf-detr-seg-large.pt'), '2XL': (RFDETRSeg2XLarge, 'rf-detr-seg-xxlarge.pt')}[args.model]
assert torch.cuda.is_available()
model = cls(pretrain_weights=str(root / 'model' / filename), device='cuda')
out = root / 'outputs' / 'smoke' / args.model
out.mkdir(parents=True, exist_ok=True)
reports = []
for name in ['bus.jpg', 'zidane.jpg', 'coco_val2017_000000000785.jpg']:
    with Image.open(root / 'test_images' / name) as source:
        image = source.convert('RGB')
    for _ in range(args.warmup):
        model.predict(image, threshold=0.3)
    torch.cuda.synchronize()
    timings = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        detections = model.predict(image, threshold=0.3)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    people = np.array([COCO_CLASSES[int(i)] == 'person' for i in detections.class_id], dtype=bool)
    w, h = image.size
    if people.any():
        assert detections.mask is not None
        masks = detections.mask[people]
        mask = np.any(masks, axis=0).astype(np.uint8) * 255
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
        mask_shape = list(masks.shape)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_shape = [0, h, w]
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(frame, (101, 101), 35)
    result = np.where(mask[..., None] > 0, blur, frame)
    stem = Path(name).stem
    assert cv2.imwrite(str(out / f'{stem}_mask.png'), mask)
    assert cv2.imwrite(str(out / f'{stem}_blurred.jpg'), result)
    report = {
        'model': args.model, 'image': name, 'device': torch.cuda.get_device_name(0),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'people': int(people.sum()), 'mask_shape': mask_shape, 'output_size': [w, h],
        'warmup_per_image': args.warmup, 'repeats': args.repeats,
        'predict_ms_mean': float(np.mean(timings)), 'predict_ms_median': float(np.median(timings)),
        'predict_ms_p95': float(np.percentile(timings, 95)), 'predict_ms_min': min(timings),
        'predict_ms_max': max(timings), 'predict_ms_samples': timings,
        'threshold': 0.3, 'backend': 'PyTorch default predict; no explicit FP16/compile/TensorRT optimization',
        'timing_scope': 'Synchronized model.predict(PIL image): includes API preprocessing and mask postprocessing; excludes model load, image decode, dilation, blur, and disk writes.',
        'status': 'person_masks_generated' if people.any() else 'no_person_detected_requires_review'
    }
    (out / f'{stem}_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    reports.append(report)
    print(json.dumps({k: report[k] for k in ['model', 'image', 'people', 'predict_ms_mean', 'predict_ms_median', 'predict_ms_p95']}), flush=True)
(out / 'benchmark_results.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
