"""Single-image GPU inference and privacy-mask smoke check."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRSegMedium, RFDETRSegLarge, RFDETRSeg2XLarge
from rfdetr.assets.coco_classes import COCO_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['M', 'L', '2XL'], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
cls, filename = {'M': (RFDETRSegMedium, 'rf-detr-seg-medium.pt'), 'L': (RFDETRSegLarge, 'rf-detr-seg-large.pt'), '2XL': (RFDETRSeg2XLarge, 'rf-detr-seg-xxlarge.pt')}[args.model]
assert torch.cuda.is_available(), 'CUDA unavailable'
model = cls(pretrain_weights=str(root / 'model' / filename), device='cuda')
source = root / 'test_images' / 'bus.jpg'
image = Image.open(source).convert('RGB')
torch.cuda.synchronize()
start = time.perf_counter()
detections = model.predict(image, threshold=0.3)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
people = np.array([COCO_CLASSES[int(i)] == 'person' for i in detections.class_id], dtype=bool)
assert detections.mask is not None, 'No segmentation output'
masks = detections.mask[people]
assert len(masks) > 0, 'No people found in bus sample'
w, h = image.size
mask = np.any(masks, axis=0).astype(np.uint8) * 255
if mask.shape != (h, w):
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
blur = cv2.GaussianBlur(frame, (101, 101), 35)
output = np.where(mask[..., None] > 0, blur, frame)
out = root / 'outputs' / 'smoke' / args.model
out.mkdir(parents=True, exist_ok=True)
assert cv2.imwrite(str(out / 'bus_mask.png'), mask)
assert cv2.imwrite(str(out / 'bus_blurred.jpg'), output)
report = {'model': args.model, 'device': torch.cuda.get_device_name(0), 'people': int(people.sum()), 'mask_shape': list(masks.shape), 'output_size': [w, h], 'first_predict_seconds': elapsed, 'note': 'First call includes startup; not a throughput benchmark. Visual result is not privacy certification.'}
(out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
