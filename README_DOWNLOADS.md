# RF-DETR 模型、推理代码与测试素材下载记录

> 最新状态（2026-09-18）：已创建并验证 rf-detr Conda 环境，M/L/2XL 已完成单图 GPU 推理；详见 [环境说明](README_ENVIRONMENT.md)。下文未安装/未推理描述属于下载阶段记录。

整理日期：2026-09-18

项目目录：`/home/ubuntu/data_1/chenxiang/human-privacy-anonymization`

## 当前状态

- 三个模型已下载，均通过官方权重注册表 MD5 校验。
- 官方推理代码已克隆，未修改源码。
- 三张测试图片已通过 Pillow 完整性检查；COCO 139 下载遇到 TLS 错误，留待手动下载。
- 尚未安装推理环境、运行模型或生成模糊效果图。当前系统 python3 未检测到 torch。

## 目录

```text
model/                 三个 COCO 预训练实例分割权重
code/rf-detr/          官方推理、训练和导出代码
test_images/           原始测试图片
README.md              调研总结
README_DOWNLOADS.md    本文件
download_assets.py     批量下载脚本（含重试和官方 MD5 校验）
assets_manifest.json   来源、状态、尺寸和校验信息
SHA256SUMS             已验证文件的 SHA-256
```

## 官方代码及复现命令

- 仓库：https://github.com/roboflow/rf-detr
- 固定 commit：`5669b7c8e3ac961a9fac3c3dc3967c328c7ffec3`
- 权重地址与 MD5 来源：[固定版本注册表](https://github.com/roboflow/rf-detr/blob/5669b7c8e3ac961a9fac3c3dc3967c328c7ffec3/src/rfdetr/assets/model_weights.py)
- 推理示例：`code/rf-detr/docs/learn/run/segmentation.md`
- 推理实现：`code/rf-detr/src/rfdetr/`

以下 clone 命令用于新环境；当前目录已克隆，不必重复执行。

```bash
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
mkdir -p model code test_images
git clone https://github.com/roboflow/rf-detr.git code/rf-detr
git -C code/rf-detr checkout 5669b7c8e3ac961a9fac3c3dc3967c328c7ffec3
```

## 三个模型

| 模型 | 本地文件 | 实际字节数 | 官方 MD5 |
|---|---|---:|---|
| RF-DETR-Seg-M | `model/rf-detr-seg-medium.pt` | 143024058 | `a49af1562c3719227ad43d0ca53b4c7a` |
| RF-DETR-Seg-L | `model/rf-detr-seg-large.pt` | 145055866 | `275f7b094909544ed2841c94a677d07e` |
| RF-DETR-Seg-2XL | `model/rf-detr-seg-xxlarge.pt` | 154851262 | `040bc3412af840fa8a47e0ff69b552ba` |

这些是带实例 mask 输出的 COCO 预训练分割模型，不是只输出框的 detection 模型。2XL 官方本地文件名为 `xxlarge`，Python 类名为 `RFDETRSeg2XLarge`。下载文件大小不是运行显存。

### 下载命令

使用临时文件，成功后才替换正式文件，避免把失败下载误当作完整权重。

```bash
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 'https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth' -o model/rf-detr-seg-medium.pt.part && mv model/rf-detr-seg-medium.pt.part model/rf-detr-seg-medium.pt
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 'https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth' -o model/rf-detr-seg-large.pt.part && mv model/rf-detr-seg-large.pt.part model/rf-detr-seg-large.pt
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 'https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth' -o model/rf-detr-seg-xxlarge.pt.part && mv model/rf-detr-seg-xxlarge.pt.part model/rf-detr-seg-xxlarge.pt
```

```bash
printf '%s\n' \
  'a49af1562c3719227ad43d0ca53b4c7a  model/rf-detr-seg-medium.pt' \
  '275f7b094909544ed2841c94a677d07e  model/rf-detr-seg-large.pt' \
  '040bc3412af840fa8a47e0ff69b552ba  model/rf-detr-seg-xxlarge.pt' | md5sum -c -
```

## 测试图片

| 文件 | 状态 | 尺寸（宽×高） | 用途 |
|---|---|---|---|
| `test_images/bus.jpg` | verified | 810×1080 | 街道行人与公交车，观察多人全身覆盖 |
| `test_images/zidane.jpg` | verified | 1280×720 | 近景人物，观察局部人体与头部覆盖 |
| `test_images/coco_val2017_000000000139.jpg` | pending_manual_download | 待下载 | COCO 验证集样例 139，检查室内人物 |
| `test_images/coco_val2017_000000000785.jpg` | verified | 640×425 | COCO 验证集样例 785，检查运动人物 |

来源：bus/zidane 来自 Ultralytics 官方示例资产；COCO 图片来自官方数据集存储。用途描述用于初步测试设计，不代表已经运行分割验证。公开示例可能属于预训练模型见过的数据分布，只用于查看效果，不能代替独立隐私脱敏验收集。图片授权应以各来源与原始图片许可为准，模型许可证不自动覆盖图片。

### 图片来源和下载命令

```bash
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg' -o test_images/bus.jpg.part && mv test_images/bus.jpg.part test_images/bus.jpg
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg' -o test_images/zidane.jpg.part && mv test_images/zidane.jpg.part test_images/zidane.jpg
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 'https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000139.jpg' -o test_images/coco_val2017_000000000139.jpg.part && mv test_images/coco_val2017_000000000139.jpg.part test_images/coco_val2017_000000000139.jpg
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 'https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000785.jpg' -o test_images/coco_val2017_000000000785.jpg.part && mv test_images/coco_val2017_000000000785.jpg.part test_images/coco_val2017_000000000785.jpg
```

COCO 139 是本次唯一未完成的素材。不要使用 `curl -k` 绕过证书检查。

## 后续环境安装与推理入口（尚未执行）

以下在项目独立虚拟环境安装，不修改系统 Python。需先确认驱动与所安装 PyTorch CUDA 构建兼容。

```bash
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./code/rf-detr
```

使用本地权重的官方 API 示例：

```python
from rfdetr import RFDETRSegMedium, RFDETRSegLarge, RFDETRSeg2XLarge

# 每次加载一个模型，避免同时占用显存。
model = RFDETRSegLarge(pretrain_weights="model/rf-detr-seg-large.pt")
detections = model.predict("test_images/bus.jpg", threshold=0.3)

# 另外两个模型的对应构造方式：
# RFDETRSegMedium(pretrain_weights="model/rf-detr-seg-medium.pt")
# RFDETRSeg2XLarge(pretrain_weights="model/rf-detr-seg-xxlarge.pt")
```

该片段仅展示加载和分割调用，尚未运行验证。实际脱敏需要按官方 COCO 类别映射筛选 person，合并掩码、适度外扩、还原原图坐标，再执行强模糊。threshold=0.3 仅为待调试起点，不是经过验证的隐私保护阈值。

## 文件校验

```bash
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
sha256sum -c SHA256SUMS
```

SHA256SUMS 记录本次已成功下载的六个文件。手动补充 COCO 139 后，可另行计算其校验值并更新清单；本次记录保留下载时的真实状态。
