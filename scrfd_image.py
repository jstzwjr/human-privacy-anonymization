"""使用迁移的 SCRFD 接口测试单张图片，保存人脸框 JSON 与可视化图片。"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scrfd_detector import SCRFDDetector


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='待检测图片路径')
    parser.add_argument('--model', choices=['500m', '10g'], default='10g', help='默认对应 World Studio thorough 档位')
    parser.add_argument('--output-dir', type=Path, help='新建结果目录；已有目录拒绝覆盖')
    args = parser.parse_args()
    model_path = ROOT / 'model' / 'scrfd' / f'det_{args.model}.onnx'
    output_dir = args.output_dir or ROOT / 'outputs' / 'scrfd' / args.model / args.image.stem
    if output_dir.exists():
        raise FileExistsError(f'输出目录已存在，请指定新的 --output-dir：{output_dir}')
    if not model_path.is_file():
        raise FileNotFoundError(f'缺少 SCRFD 权重，请参照 README 准备文件：{model_path}')
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    # 单图 CPU 测试使用一个推理线程，检测逻辑与原接口相同。
    detector = SCRFDDetector(model_path, intra_op_num_threads=1)
    faces = detector.detect(np.asarray(image))
    start = time.perf_counter()
    for _ in range(100):
        faces = detector.detect(np.asarray(image))
    elapsed = time.perf_counter() - start
    report = {
        'image': str(args.image.resolve()),
        'image_size': list(image.size),
        'model': str(model_path),
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'provider': detector.active_provider(),
        'face_count': len(faces),
        'faces': [asdict(face) for face in faces],
        'first_detect_seconds': elapsed,
        'note': 'x/y/w/h 为原图像素坐标。首轮 detect 含预处理和后处理，不是稳态性能基准。',
    }
    draw = ImageDraw.Draw(image)
    for face in faces:
        draw.rectangle((face.x, face.y, face.x + face.w, face.y + face.h), outline='red', width=3)
        draw.text((max(0, face.x), max(0, face.y - 12)), f'{face.score:.3f}', fill='red')
    output_dir.mkdir(parents=True, exist_ok=False)
    image.save(output_dir / 'detections.png')
    (output_dir / 'result.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    )
    print(f'检测完成：{len(faces)} 张人脸，执行后端 {report["provider"]}，首轮耗时 {elapsed:.3f} 秒')
    print(f'检测框与置信度：{output_dir / "result.json"}')
    print(f'可视化图片：{output_dir / "detections.png"}')


if __name__ == '__main__':
    main()
