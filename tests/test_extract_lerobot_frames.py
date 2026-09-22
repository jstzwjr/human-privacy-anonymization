"""校验跨 Episode 命名、视频时间戳和输出保护。"""

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from extract_lerobot_frames import CAMERAS, export_frames, image_name


class FrameExportTests(unittest.TestCase):
    def test_fixed_width_names(self):
        first = image_name('head', 0, '0')
        later = image_name('head', 100, '3.333333')
        self.assertEqual(first, 'head_00000000_000000.000000.jpg')
        self.assertEqual(later, 'head_00000100_000003.333333.jpg')
        self.assertEqual(len(first), len(later))

    def make_dataset(self, root):
        (root / 'meta').mkdir()
        info = {'fps': 30, 'total_frames': 5, 'chunks_size': 1000,
                'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
        (root / 'meta/info.json').write_text(json.dumps(info))
        (root / 'meta/episodes.jsonl').write_text(
            json.dumps({'episode_index': 0, 'length': 3}) + '\n' +
            json.dumps({'episode_index': 1, 'length': 2}) + '\n')
        for key in CAMERAS:
            folder = root / 'videos/chunk-000' / key
            folder.mkdir(parents=True)
            for episode, count in enumerate((3, 2)):
                # 视频实际 5 FPS，而元数据写 30；用此确认没有按 metadata FPS 猜时间。
                subprocess.run([
                    'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                    '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=5',
                    '-frames:v', str(count), '-c:v', 'libx264', '-threads', '1',
                    str(folder / f'episode_{episode:06d}.mp4'),
                ], check=True)
        return info

    def test_two_episodes_three_cameras_and_video_pts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            output = root / 'images'
            output.mkdir()
            summary = export_frames(root, output)
            self.assertEqual(summary['total_images'], 15)
            for camera in CAMERAS.values():
                self.assertEqual(len(list((output / camera).glob('*.jpg'))), 5)
                with (output / f'{camera}_index.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([int(row['global_frame_index']) for row in rows], list(range(5)))
                self.assertEqual([row['timestamp_seconds'] for row in rows],
                                 ['0.000000', '0.200000', '0.400000', '0.000000', '0.200000'])
                self.assertEqual(rows[3]['frame_index'], '0')
                self.assertEqual(rows[3]['episode_index'], '1')
                self.assertEqual(Path(rows[3]['filename']).name,
                                 f'{camera}_00000003_000000.000000.jpg')
                for row in rows:
                    self.assertTrue((output / row['filename']).read_bytes().startswith(b'\xff\xd8'))
            sentinel = output / 'private.txt'
            sentinel.write_text('保留原数据')
            with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
                export_frames(root, output)
            self.assertEqual(sentinel.read_text(), '保留原数据')

    def test_bad_frame_count_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            info = self.make_dataset(root)
            info['total_frames'] = 6
            (root / 'meta/info.json').write_text(json.dumps(info))
            (root / 'meta/episodes.jsonl').write_text(
                json.dumps({'episode_index': 0, 'length': 4}) + '\n' +
                json.dumps({'episode_index': 1, 'length': 2}) + '\n')
            output = root / 'images'
            output.mkdir()
            with self.assertRaises(ValueError):
                export_frames(root, output, workers=1)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(root.glob('.images-extract-*')), [])


if __name__ == '__main__':
    unittest.main()
