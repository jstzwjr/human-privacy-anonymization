# 三相机视频逐帧导出

脚本：`/home/wjr/mount/code/human-privacy-anonymization/extract_lerobot_frames.py`。

仅使用 Python 3 标准库以及已有的 `ffmpeg`、`ffprobe`，不安装或修改系统环境。读取 LeRobot v2 的 `meta/info.json`、`meta/episodes.jsonl` 和视频，不修改源视频或 Parquet。

## 执行

```bash
cd /home/wjr/mount/code/human-privacy-anonymization
mkdir -p logs
set -o pipefail
python3 -u extract_lerobot_frames.py \
  --dataset-root /home/wjr/mount/dataset/ml \
  --output-root /home/wjr/mount/dataset/ml/images \
  2>&1 | tee logs/extract_lerobot_frames.log
```

目标目录必须不存在或为空；已有图片时拒绝覆盖。默认并行处理三路相机，每个解码/编码进程使用两条线程。可用 `--workers 1` 降低并发，或用 `--max-episodes 2 --output-root /tmp/独立试跑目录` 先导出前两段。

## 文件组织与命名

| 原始 video key | 输出子目录 |
| --- | --- |
| `image` | `head` |
| `left_wrist_image` | `left_wrist` |
| `right_wrist_image` | `right_wrist` |

默认输出根目录为 `/home/wjr/mount/dataset/ml/images`。示例：

```text
head/head_00000000_000000.000000.jpg
head/head_00000100_000003.333333.jpg
left_wrist/left_wrist_00000100_000003.333333.jpg
right_wrist/right_wrist_00000100_000003.333333.jpg
```

- 帧号固定 8 位，三路相机分别使用相同的全局编号范围。按 `episode_index` 排序，以各段 `length` 累计偏移，避免不同 Episode 的第 0 帧重名。
- 时间戳为当前 MP4 中该帧的显示时间 PTS 减去首帧 PTS，单位秒，整数部分 6 位、小数部分 6 位。每个 Episode 重新从零计时；不是机器人采集时的绝对时间，也不根据 FPS 推测。
- 时间戳由 `ffprobe` 的逐帧 `best_effort_timestamp_time` 获取，帧按解码后的显示顺序对应。FFmpeg 使用 `-vsync 0`，不补帧、不丢帧、不缩放。
- 输出 JPEG 使用 FFmpeg `-q:v 2 -pix_fmt yuvj420p`，是高质量有损编码。

根目录还包含每路相机的 `<camera>_index.csv`，记录输出相对路径、原相机 key、Episode 编号、段内帧号、全局帧号、相对秒、原始视频 PTS 和源视频相对路径。

`extraction_summary.json` 记录数据集、数量、命名约定和 JPEG 总大小。本次完整数据集已导出并验证：214 段，每路 142,373 张，总计 427,119 张；JPEG 合计约 18.714 GiB。

## 完整性与失败处理

每段视频的解码帧数、JPEG 数量必须与 Episode 元数据一致。所有数据先写到输出目录旁的独立临时目录；全部完成并通过数量校验后，再原子发布到最终目录。正常异常退出会清理本次临时目录，保留命令运行日志。强制杀进程可能留下旁边的隐藏临时目录，应确认没有任务运行后只清理该次目录。

不提供自动跳过或恢复半成品，避免将残缺文件当成完成结果。目标目录已有内容时请改用新的空目录。

## 验证

```bash
cd /home/wjr/mount/code/human-privacy-anonymization
mkdir -p logs
set -o pipefail
python3 -m unittest discover -s tests -p 'test_extract_lerobot_frames.py' -v \
  2>&1 | tee logs/test_extract_lerobot_frames.log
```

测试实际构建两段三路视频，验证跨段编号、来自视频而非声明 FPS 的时间戳、图片数量、同名输出保护，以及帧数不匹配时不发布半成品。

全量验证已核对三路所有图片非空、文件名格式、全局编号连续性、CSV 逐行对应关系及各 Episode 帧数。验证结果保存在 `logs/verify_extract_lerobot_frames.json`；导出日志为 `logs/extract_lerobot_frames.log`。
