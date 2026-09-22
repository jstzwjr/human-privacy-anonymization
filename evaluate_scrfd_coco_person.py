"""保存 SCRFD 人脸预测，再按人脸框与 COCO 人体框的 IoA 计算关联指标。"""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluate_coco_person import DATA_ROOT, ROOT, annotation_info, load_predictions


FORMAT = 'scrfd-coco-person-ioa-v1'


def predict(args):
    """复用现有检测器，每张图保存一次预测，包含无人图片和空检测。"""
    import numpy as np
    import onnxruntime
    from PIL import Image
    import scrfd_detector

    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit 必须大于 0')
    output = args.output or ROOT / 'outputs' / 'scrfd_coco_person' / args.model / 'predictions.jsonl'
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
    weights = ROOT / 'model' / 'scrfd' / f'det_{args.model}.onnx'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    detector = scrfd_detector.SCRFDDetector(weights, intra_op_num_threads=1)
    metadata = {
        'format': FORMAT,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'model_class': 'SCRFDDetector',
        'weights': str(weights),
        'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'annotations': str(annotations_path.resolve()),
        'annotation_sha256': digest,
        'category_id': person_id,
        'prediction_target': 'face',
        'ground_truth_target': 'person',
        'image_ids': [image['id'] for image in images],
        'score_threshold': scrfd_detector._DET_THRESHOLD,
        'nms_threshold': scrfd_detector._NMS_THRESHOLD,
        'input_size': list(scrfd_detector._INPUT_SIZE),
        'provider': detector.active_provider(),
        'onnxruntime_version': onnxruntime.__version__,
        'intra_op_num_threads': 1,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'模型 SCRFD {args.model}，实际执行后端 {detector.active_provider()}', flush=True)
    start = time.perf_counter()
    detection_count = 0
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        for index, image_info in enumerate(images, 1):
            path = args.data_root / 'val2017' / image_info['file_name']
            with Image.open(path) as source:
                image = source.convert('RGB')
            if image.size != (image_info['width'], image_info['height']):
                raise ValueError(f'图片尺寸与标注不符：{path}')
            faces = detector.detect(np.asarray(image))
            detections = [{'bbox': [face.x, face.y, face.w, face.h], 'score': face.score}
                          for face in faces]
            stream.write(json.dumps({'image_id': image_info['id'], 'detections': detections},
                                    allow_nan=False) + '\n')
            stream.flush()
            detection_count += len(detections)
            if index == 1 or index % 100 == 0 or index == len(images):
                print(f'已保存 {index}/{len(images)} 张，人脸框 {detection_count} 个，'
                      f'耗时 {time.perf_counter() - start:.1f} 秒', flush=True)
    print(f'预测完成：{output}', flush=True)


def evaluate_results(results_path, annotations_path, output_path, ioa_threshold=0.9):
    """仅用保存结果与人体标注评测；此关联代理指标不是标准 COCO AP。"""
    import numpy as np
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not math.isfinite(ioa_threshold) or not 0 < ioa_threshold <= 1:
        raise ValueError('--ioa-threshold 必须大于 0 且不超过 1')
    if output_path.exists():
        raise FileExistsError(f'指标已存在，请指定新的 --output：{output_path}')
    annotations, person_id, digest, metadata, records = load_predictions(
        results_path, annotations_path, expected_format=FORMAT,
    )
    image_ids = metadata['image_ids']
    expected = set(image_ids)
    ground_truth = COCO(str(annotations_path))
    if records:
        predictions = ground_truth.loadRes(records)
    else:
        predictions = COCO()
        predictions.dataset = {'images': annotations['images'], 'categories': annotations['categories'],
                               'annotations': []}
        predictions.createIndex()

    class FacePersonIoAEval(COCOeval):
        def computeIoU(self, imgId, catId):
            gt = self._gts[imgId, catId]
            dt = sorted(self._dts[imgId, catId], key=lambda detection: -detection['score'])
            dt = dt[:self.params.maxDets[-1]]
            # 此参数仅让交集除以检测框面积；真实 GT 的 iscrowd 保持原样供匹配使用。
            return mask_utils.iou([d['bbox'] for d in dt], [g['bbox'] for g in gt], [1] * len(gt))

    evaluator = FacePersonIoAEval(ground_truth, predictions, iouType='bbox')
    evaluator.params.catIds = [person_id]
    evaluator.params.imgIds = image_ids
    evaluator.params.iouThrs = np.array([ioa_threshold])
    evaluator.params.areaRng = [[0, float('inf')]]
    evaluator.params.areaRngLbl = ['all']
    # 用实际最大框数作为内部上限，所有保存的预测都参加评测，不丢弃第 101 个及以后的框。
    max_dets = max(Counter(record['image_id'] for record in records).values(), default=0)
    evaluator.params.maxDets = [max(1, max_dets)]
    evaluator.evaluate()
    evaluator.accumulate()

    tp = fp = ignored = person_count = 0
    for result in evaluator.evalImgs:
        if result is None:
            continue
        matches = result['dtMatches'][0] > 0
        ignore = result['dtIgnore'][0].astype(bool)
        tp += int(np.count_nonzero(matches & ~ignore))
        fp += int(np.count_nonzero(~matches & ~ignore))
        ignored += int(np.count_nonzero(ignore))
        person_count += int(np.count_nonzero(result['gtIgnore'] == 0))
    fn = person_count - tp
    ap_values = evaluator.eval['precision'][0, :, 0, 0, 0]
    ap_values = ap_values[ap_values >= 0]
    metrics = {
        'association_AP': float(ap_values.mean()) if ap_values.size else -1.0,
        'association_precision': tp / (tp + fp) if tp + fp else 0.0,
        'person_recall': tp / person_count if person_count else -1.0,
        'association_F1': 2 * tp / (2 * tp + fp + fn) if person_count else -1.0,
    }
    crowd_count = sum(1 for annotation in annotations['annotations']
                      if annotation['image_id'] in expected and annotation['category_id'] == person_id
                      and annotation.get('iscrowd', 0))
    report = {
        'results': str(results_path.resolve()),
        'annotations': str(annotations_path.resolve()),
        'annotation_sha256': digest,
        'model': metadata.get('model'),
        'provider': metadata.get('provider'),
        'score_threshold': metadata.get('score_threshold'),
        'image_count': len(image_ids),
        'full_validation_set': expected == {image['id'] for image in annotations['images']},
        'category_id': person_id,
        'matching': 'intersection(face_bbox, person_bbox) / area(face_bbox) >= ioa_threshold',
        'ioa_threshold': ioa_threshold,
        'matching_policy': 'score_descending_one_to_one_regular_then_ignore_crowd',
        'area_group': 'all',
        'max_dets_per_image': None,
        'ap_recall_points': len(evaluator.params.recThrs),
        'counts': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'ignored_detections': ignored,
            'noncrowd_person_count': person_count,
            'crowd_person_count': crowd_count,
            'detection_count': len(records),
        },
        'metrics': metrics,
        'note': '这是人脸框与人体框的关联代理指标，不是标准 COCO AP 或真实人脸 AP。'
                '人体内部的错误人脸框也可能算 TP；背身或脸不可见的人仍在人体召回率分母中。'
                'AP 仅基于已保存的置信度截断预测；指标范围 0–1，-1 表示无可评测普通人体。'
                '无有效预测时 precision 为 0；匹配 crowd 的检测忽略，不计 TP/FP。',
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'IoA >= {ioa_threshold:g}，TP={tp}，FP={fp}，FN={fn}，crowd 忽略={ignored}', flush=True)
    for name, value in metrics.items():
        print(f'{name}: {value:.6f}', flush=True)
    print(f'关联指标已保存：{output_path}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inference = commands.add_parser('predict', help='运行 SCRFD，逐图保存人脸框')
    inference.add_argument('--data-root', type=Path, default=DATA_ROOT)
    inference.add_argument('--model', choices=['500m', '10g'], default='10g')
    inference.add_argument('--limit', type=int, help='按图像 ID 排序取前 N 张，仅用于小规模验证')
    inference.add_argument('--output', type=Path, help='默认 outputs/scrfd_coco_person/<模型>/predictions.jsonl')
    evaluation = commands.add_parser('evaluate', help='读取结果并计算 IoA 关联指标，无需模型或 GPU')
    evaluation.add_argument('--results', type=Path, required=True)
    evaluation.add_argument('--annotations', type=Path, default=DATA_ROOT / 'annotations' / 'instances_val2017.json')
    evaluation.add_argument('--ioa-threshold', type=float, default=0.9, help='匹配要求 IoA >= 此值，默认 0.9')
    evaluation.add_argument('--output', type=Path, help='默认在结果文件旁保存 .metrics.json')
    args = parser.parse_args()
    if args.command == 'predict':
        predict(args)
    else:
        evaluate_results(args.results, args.annotations, args.output or args.results.with_suffix('.metrics.json'),
                         args.ioa_threshold)


if __name__ == '__main__':
    main()
