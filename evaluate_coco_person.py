"""先保存 RF-DETR 的 person 检测框，再独立计算 COCO bbox 指标。"""

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path('/home/wjr/mount/dataset/coco')
FORMAT = 'coco-person-bbox-v1'
METRIC_NAMES = (
    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large',
)


def annotation_info(path):
    """读取原始标注并定位 person，摘要用于防止评测时混用标签。"""
    raw = path.read_bytes()
    annotations = json.loads(raw)
    person_ids = [c['id'] for c in annotations['categories'] if c['name'] == 'person']
    if len(person_ids) != 1:
        raise ValueError('标注必须包含唯一的 person 类别')
    return annotations, person_ids[0], hashlib.sha256(raw).hexdigest()


def predict(args):
    """逐图持久化，空预测也写一行；中断文件不能冒充完整评测。"""
    import torch
    from PIL import Image
    from rfdetr import RFDETRSegMedium, RFDETRSegLarge, RFDETRSeg2XLarge
    from rfdetr.assets.coco_classes import COCO_CLASSES

    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError('--threshold 必须在 0 到 1 之间')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit 必须大于 0')
    output = args.output or ROOT / 'outputs' / 'coco_person' / args.model / 'predictions.jsonl'
    if output.exists():
        raise FileExistsError(f'结果已存在，请指定新的 --output：{output}')
    annotations_path = args.data_root / 'annotations' / 'instances_val2017.json'
    annotations, person_id, digest = annotation_info(annotations_path)
    images = sorted(annotations['images'], key=lambda image: image['id'])
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise ValueError('没有可评测图片')
    for image in images:
        path = args.data_root / 'val2017' / image['file_name']
        if not path.is_file():
            raise FileNotFoundError(path)

    model_class, filename = {
        'M': (RFDETRSegMedium, 'rf-detr-seg-medium.pt'),
        'L': (RFDETRSegLarge, 'rf-detr-seg-large.pt'),
        '2XL': (RFDETRSeg2XLarge, 'rf-detr-seg-xxlarge.pt'),
    }[args.model]
    weights = ROOT / 'model' / filename
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = model_class(pretrain_weights=str(weights), device=args.device)
    model_person_id = next(key for key, value in COCO_CLASSES.items() if value == 'person')
    metadata = {
        'format': FORMAT,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'model_class': model_class.__name__,
        'weights': str(weights),
        'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'annotations': str(annotations_path.resolve()),
        'annotation_sha256': digest,
        'category_id': person_id,
        'image_ids': [image['id'] for image in images],
        'score_threshold': args.threshold,
        'device': args.device,
        'torch_version': str(torch.__version__),
        'model_resolution': model.model.resolution,
        'num_select': model.model.args.num_select,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    detection_count = 0
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        stream.flush()
        for index, image_info in enumerate(images, 1):
            path = args.data_root / 'val2017' / image_info['file_name']
            with Image.open(path) as source:
                image = source.convert('RGB')
            if image.size != (image_info['width'], image_info['height']):
                raise ValueError(f'图片尺寸与标注不符：{path}')
            detections = model.predict(image, threshold=args.threshold, include_source_image=False)
            people = []
            for box, score, category in zip(detections.xyxy, detections.confidence, detections.class_id):
                if int(category) == model_person_id:
                    x1, y1, x2, y2 = map(float, box)
                    people.append({'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(score)})
            stream.write(json.dumps({'image_id': image_info['id'], 'detections': people}, allow_nan=False) + '\n')
            stream.flush()
            detection_count += len(people)
            del detections
            if index == 1 or index % 100 == 0 or index == len(images):
                print(f'已保存 {index}/{len(images)} 张，person 框 {detection_count} 个，'
                      f'耗时 {time.perf_counter() - start:.1f} 秒', flush=True)
    print(f'预测完成：{output}', flush=True)


def load_predictions(results_path, annotations_path, expected_format=FORMAT):
    """读取逐图预测并校验标注摘要、图像完整性和检测数值，供两种评测复用。"""
    annotations, person_id, digest = annotation_info(annotations_path)
    all_image_ids = {image['id'] for image in annotations['images']}
    records = []
    seen = set()
    with results_path.open(encoding='utf-8') as stream:
        metadata = json.loads(next(stream, '{}'))
        if metadata.get('format') != expected_format:
            raise ValueError('结果格式不匹配')
        if metadata.get('annotation_sha256') != digest:
            raise ValueError('结果与当前标注的 SHA-256 不一致')
        if metadata.get('category_id') != person_id:
            raise ValueError('结果类别不是标注中的 person')
        image_ids = metadata['image_ids']
        if not image_ids or len(set(image_ids)) != len(image_ids) or not set(image_ids) <= all_image_ids:
            raise ValueError('评测图像 ID 为空、重复或不在标注中')
        expected = set(image_ids)
        for line in stream:
            row = json.loads(line)
            image_id = row['image_id']
            if image_id not in expected or image_id in seen:
                raise ValueError(f'重复或未声明的图像 ID：{image_id}')
            seen.add(image_id)
            for detection in row['detections']:
                box, score = detection['bbox'], detection['score']
                if (len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value)
                                             for value in [*box, score])
                        or box[2] < 0 or box[3] < 0 or not 0 <= score <= 1):
                    raise ValueError(f'无效的检测框或置信度：image_id={image_id}')
                # 原图边界裁剪可能产生零面积框，保留给官方评测计误检，不能删掉来抬高 AP。
                records.append({'image_id': image_id, 'category_id': person_id, 'bbox': box, 'score': score})
    if seen != expected:
        raise ValueError(f'预测不完整：缺少 {len(expected - seen)} 张图片，不能计算完整指标')
    return annotations, person_id, digest, metadata, records


def evaluate_results(results_path, annotations_path, output_path):
    """只依赖保存结果与标注，按官方 COCO person bbox 协议评测。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if output_path.exists():
        raise FileExistsError(f'指标已存在，请指定新的 --output：{output_path}')
    annotations, person_id, digest, metadata, records = load_predictions(results_path, annotations_path)
    image_ids = metadata['image_ids']
    expected = set(image_ids)
    all_image_ids = {image['id'] for image in annotations['images']}

    ground_truth = COCO(str(annotations_path))
    if records:
        predictions = ground_truth.loadRes(records)
    else:
        # 官方 loadRes([]) 会索引空列表，显式构造空结果仍须计入所有漏检。
        predictions = COCO()
        predictions.dataset = {'images': annotations['images'], 'categories': annotations['categories'],
                               'annotations': []}
        predictions.createIndex()
    evaluator = COCOeval(ground_truth, predictions, iouType='bbox')
    evaluator.params.catIds = [person_id]
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    report = {
        'results': str(results_path.resolve()),
        'annotations': str(annotations_path.resolve()),
        'annotation_sha256': digest,
        'model': metadata.get('model'),
        'score_threshold': metadata.get('score_threshold'),
        'image_count': len(image_ids),
        'full_validation_set': expected == all_image_ids,
        'detection_count': len(records),
        'category_id': person_id,
        'iou_type': 'bbox',
        'max_dets': list(evaluator.params.maxDets),
        'metrics': dict(zip(METRIC_NAMES, map(float, evaluator.stats))),
        'note': '指标为 0–1；-1 表示该范围没有可评测真值。仅评测 person 检测框，不是 mask 或脱敏效果。',
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'指标已保存：{output_path}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inference = commands.add_parser('predict', help='运行模型，逐图保存 person 框')
    inference.add_argument('--data-root', type=Path, default=DATA_ROOT)
    inference.add_argument('--model', choices=['M', 'L', '2XL'], default='L')
    inference.add_argument('--device', default='cuda')
    inference.add_argument('--threshold', type=float, default=0.001)
    inference.add_argument('--limit', type=int, help='仅用于小规模验证；按图像 ID 排序取前 N 张')
    inference.add_argument('--output', type=Path, help='JSONL 路径；默认 outputs/coco_person/<模型>/predictions.jsonl')
    evaluation = commands.add_parser('evaluate', help='从已保存结果计算指标，无需 GPU 或加载模型')
    evaluation.add_argument('--results', type=Path, required=True)
    evaluation.add_argument('--annotations', type=Path, default=DATA_ROOT / 'annotations' / 'instances_val2017.json')
    evaluation.add_argument('--output', type=Path, help='默认在结果文件旁保存 .metrics.json')
    args = parser.parse_args()
    if args.command == 'predict':
        predict(args)
    else:
        evaluate_results(args.results, args.annotations, args.output or args.results.with_suffix('.metrics.json'))


if __name__ == '__main__':
    main()
