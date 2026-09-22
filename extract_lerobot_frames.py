"""将 LeRobot v2 数据集的三路视频逐帧导出为固定宽度命名的 JPEG。"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path


CAMERAS = {
    'image': 'head',
    'left_wrist_image': 'left_wrist',
    'right_wrist_image': 'right_wrist',
}
DEFAULT_DATASET = Path('/home/wjr/mount/dataset/ml')


def image_name(camera, global_index, timestamp):
    if not 0 <= global_index < 100_000_000:
        raise ValueError('全局帧号超出 8 位编号范围')
    timestamp = Decimal(timestamp).quantize(Decimal('0.000001'))
    if not timestamp.is_finite() or not 0 <= timestamp < 1_000_000:
        raise ValueError('视频相对时间超出固定宽度命名范围')
    return f'{camera}_{global_index:08d}_{timestamp:013.6f}.jpg'


def load_episodes(dataset_root, max_episodes=None):
    info = json.loads((dataset_root / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in
                (dataset_root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    episodes.sort(key=lambda item: item['episode_index'])
    if not episodes or len({item['episode_index'] for item in episodes}) != len(episodes):
        raise ValueError('Episode 清单为空或编号重复')
    offset = 0
    for episode in episodes:
        if episode['length'] <= 0:
            raise ValueError(f"Episode {episode['episode_index']} 的帧数无效")
        episode['global_offset'] = offset
        offset += episode['length']
    if offset != info['total_frames']:
        raise ValueError('Episode 长度之和与 meta/info.json 的 total_frames 不一致')
    return info, episodes[:max_episodes] if max_episodes is not None else episodes


def video_path(dataset_root, info, episode, video_key):
    return dataset_root / info['video_path'].format(
        episode_chunk=episode['episode_index'] // info['chunks_size'],
        video_key=video_key, episode_index=episode['episode_index'],
    )


def frame_timestamps(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-threads', '2', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    frames = json.loads(result.stdout)['frames']
    timestamps = [Decimal(frame['best_effort_timestamp_time']) for frame in frames]
    if not timestamps or any(not value.is_finite() for value in timestamps):
        raise ValueError(f'视频没有有效逐帧时间戳：{path}')
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f'视频时间戳未按显示顺序递增：{path}')
    return timestamps


def extract_camera(dataset_root, info, episodes, staging, video_key, camera, stop):
    output = staging / camera
    output.mkdir()
    count = byte_count = 0
    fields = ['filename', 'video_key', 'episode_index', 'frame_index',
              'global_frame_index', 'timestamp_seconds', 'video_pts_seconds', 'source_video']
    try:
        with (staging / f'{camera}_index.csv').open('x', newline='') as index_file:
            writer = csv.DictWriter(index_file, fieldnames=fields)
            writer.writeheader()
            for number, episode in enumerate(episodes, 1):
                if stop.is_set():
                    raise RuntimeError('另一相机导出失败，停止本路处理')
                source = video_path(dataset_root, info, episode, video_key)
                timestamps = frame_timestamps(source)
                if len(timestamps) != episode['length']:
                    raise ValueError(f'{source}：解码帧数 {len(timestamps)} 与元数据 '
                                     f"{episode['length']} 不一致")
                with tempfile.TemporaryDirectory(prefix=f'.{camera}-', dir=staging) as temp:
                    temporary = Path(temp)
                    command = [
                        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
                        '-n', '-threads', '2', '-i', str(source), '-map', '0:v:0',
                        '-an', '-vsync', '0', '-c:v', 'mjpeg', '-q:v', '2',
                        '-pix_fmt', 'yuvj420p', '-threads', '2', '-start_number', '0',
                        str(temporary / 'frame_%08d.jpg'),
                    ]
                    subprocess.run(command, check=True)
                    if len(list(temporary.glob('*.jpg'))) != len(timestamps):
                        raise ValueError(f'JPEG 数量与逐帧时间戳数量不一致：{source}')
                    for local_index, pts in enumerate(timestamps):
                        global_index = episode['global_offset'] + local_index
                        relative = pts - timestamps[0]
                        name = image_name(camera, global_index, relative)
                        frame = temporary / f'frame_{local_index:08d}.jpg'
                        size = frame.stat().st_size
                        if not size:
                            raise ValueError(f'JPEG 文件为空：{frame}')
                        # 同文件系统内硬链接发布，不允许覆盖同名文件。
                        os.link(frame, output / name)
                        writer.writerow({
                            'filename': f'{camera}/{name}', 'video_key': video_key,
                            'episode_index': episode['episode_index'], 'frame_index': local_index,
                            'global_frame_index': global_index,
                            'timestamp_seconds': f'{relative:.6f}',
                            'video_pts_seconds': f'{pts:.6f}',
                            'source_video': str(source.relative_to(dataset_root)),
                        })
                        count += 1
                        byte_count += size
                index_file.flush()
                print(f"[{camera}] {number}/{len(episodes)} 段完成，"
                      f"Episode {episode['episode_index']:06d}，累计 {count} 张", flush=True)
    except Exception as error:
        print(f'[{camera}] 导出失败：{error}', flush=True)
        stop.set()
        raise
    return {'camera': camera, 'frames': count, 'jpeg_bytes': byte_count}


def export_frames(dataset_root, output_root, max_episodes=None, workers=3):
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).absolute()
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError('max_episodes 必须为正数')
    if not 1 <= workers <= 3:
        raise ValueError('workers 必须在 1 到 3 之间')
    if output_root.is_symlink() or (output_root.exists() and
                                   (not output_root.is_dir() or any(output_root.iterdir()))):
        raise ValueError(f'输出目录必须不存在或为空，拒绝覆盖：{output_root}')
    for program in ('ffmpeg', 'ffprobe'):
        if shutil.which(program) is None:
            raise RuntimeError(f'找不到已有的 {program}，请提供包含它的 PATH')
    info, episodes = load_episodes(dataset_root, max_episodes)
    for episode in episodes:
        for video_key in CAMERAS:
            source = video_path(dataset_root, info, episode, video_key)
            if not source.is_file():
                raise FileNotFoundError(f'缺少相机视频：{source}')
    expected = sum(item['length'] for item in episodes)
    print(f'准备导出 {len(episodes)} 段，每路 {expected} 张，共 {expected * 3} 张', flush=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}-extract-', dir=output_root.parent))
    stop = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(extract_camera, dataset_root, info, episodes, staging,
                                   key, camera, stop) for key, camera in CAMERAS.items()]
            results = [future.result() for future in as_completed(futures)]
        if any(item['frames'] != expected for item in results):
            raise ValueError('最终相机图片数量校验失败')
        summary = {
            'dataset_root': str(dataset_root), 'episodes': len(episodes),
            'frames_per_camera': expected, 'total_images': expected * 3,
            'filename_pattern': '<camera>_<global_frame_index:08d>_<relative_seconds:013.6f>.jpg',
            'timestamp_origin': '每段视频首帧 PTS；非机器人采集时的绝对时钟',
            'jpeg_quality': 'FFmpeg MJPEG q:v=2, yuvj420p；不缩放',
            'cameras': results,
        }
        (staging / 'extraction_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        # 全部成功后原子替换空目标目录；目标若新增内容，rename 会失败而不会覆盖。
        os.rename(staging, output_root)
        print(f'导出完成：{output_root}，共 {expected * 3} 张', flush=True)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output-root', type=Path, help='默认 <dataset-root>/images；必须为空')
    parser.add_argument('--max-episodes', type=int, help='仅导出前 N 段，用于试跑')
    parser.add_argument('--workers', type=int, default=3, help='并行相机数，1～3，默认 3')
    args = parser.parse_args()
    export_frames(args.dataset_root, args.output_root or args.dataset_root / 'images',
                  args.max_episodes, args.workers)


if __name__ == '__main__':
    main()
