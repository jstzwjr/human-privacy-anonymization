# COCO val2017 person 检测框评测

`evaluate_coco_person.py` 使用本地 RF-DETR-Seg-M/L/2XL 权重预测 person 检测框，先逐图保存结果，再独立计算官方 COCO bbox 指标。默认模型为 L。本评测不计算分割 mask 精度或隐私脱敏充分性。

## 环境与数据

```bash
ssh wjr
source /home/wjr/miniconda3/etc/profile.d/conda.sh
conda activate rf-detr
cd /home/wjr/mount/code/human-privacy-anonymization
mkdir -p logs
set -o pipefail
```

默认数据根目录是 `/home/wjr/mount/dataset/coco`，包含 `val2017/` 和 `annotations/instances_val2017.json`。当前验证集共 5,000 张图片，其中 2,693 张有人体标注，包含 11,004 个人体标注（10,777 个非 crowd）。不下载或使用训练图片。

推理沿用现有 `rf-detr` 环境，指标阶段额外依赖 `pycocotools`；当前环境已安装 2.0.11。新环境可执行 `python -m pip install pycocotools`。指标阶段不导入 torch/rfdetr，不需要 GPU、模型权重或原图片。

## 先保存预测

```bash
HF_HUB_OFFLINE=1 python evaluate_coco_person.py predict --model L \
  2>&1 | tee logs/coco_person_L_predict.log
```

- 默认处理全部 5,000 张图片，包括无人图片；按图像 ID 排序。
- `--model M|L|2XL` 选择现有分割模型，使用其 person 框；保留官方 `predict()` 路径，不修改模型源码或额外添加 NMS。
- `--threshold` 默认 0.001，为 AP 保留低分候选，不使用展示脚本的 0.3。仍低于阈值的预测已被舍弃，需要时可设为 0 后重新推理；不能从已截断结果恢复低分框。
- 默认输入尺寸及全类别 top-k 沿用模型配置。M/L 的 top-k 为 200，2XL 为 300；筛选 person 后不补回其他候选。
- 当前分割模型没有公开的跳过 mask 参数，推理内部仍计算 mask，但文件只保存检测框和置信度。
- `--limit 8` 可先做小规模验证；指标文件会明确标记是否为完整验证集，子集结果不可当作全量结果。
- `--data-root` 可修改数据根目录，`--device` 默认 `cuda`，`--output` 可指定新的 JSONL 文件路径。

默认结果为 `outputs/coco_person/L/predictions.jsonl`。首行保存运行配置、权重和标注 SHA-256、完整的待评测图像 ID 列表。之后每张图一行：

```json
{"image_id": 139, "detections": [{"bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.95}]}
```

`bbox` 为原图像素坐标 `[x, y, width, height]`。所有记录均为 person，其 COCO 类别 ID 保存于首行。没有检出人时仍保存 `{"image_id": 139, "detections": []}`。JSONL 外层是本脚本的逐图格式，不应直接传给 `COCO.loadRes()`；评测阶段会转换为标准 COCO 检测记录。

每张图完成后立即写入并 flush；不支持断点续跑。结果已存在时拒绝覆盖，应为新实验指定新的 `--output`。发生推理异常会退出，不把失败图片伪装成空预测；评测会拒绝缺图或截断的结果文件。

## 再从结果计算指标

```bash
python evaluate_coco_person.py evaluate \
  --results outputs/coco_person/L/predictions.jsonl \
  2>&1 | tee logs/coco_person_L_evaluate.log
```

默认读取上述原始 COCO 标注，并保存 `outputs/coco_person/L/predictions.metrics.json`。可通过 `--annotations` 指定同一份标注的其他路径、通过 `--output` 指定新的指标文件；已有指标文件同样拒绝覆盖。

评测使用官方 `pycocotools.COCOeval(iouType='bbox')`，限定 `catIds=[person]`，但 `imgIds` 保留所有已选图片，所以无人图片上的误检也会影响 AP。crowd 匹配、面积分组和 IoU 匹配均由官方实现处理。全空预测仍计算为漏检；边界裁剪产生的零面积预测保留参与官方评测，不通过删除预测抬高 AP。

输出 12 个标准指标：

| 字段 | 含义 |
|---|---|
| AP | IoU 0.50:0.05:0.95 的平均 AP，maxDets=100 |
| AP50 / AP75 | IoU 0.50 / 0.75 的 AP |
| AP_small / AP_medium / AP_large | COCO 面积分组下的 AP |
| AR1 / AR10 / AR100 | 每图最多保留 1 / 10 / 100 个 person 预测时的平均召回率 |
| AR_small / AR_medium / AR_large | COCO 面积分组下的 AR，maxDets=100 |

指标范围为 0–1，乘以 100 为百分数；`-1` 表示该面积范围没有可评测真值。AP 是置信度排序后的精确率—召回率曲线指标，不是某一业务置信度阈值下的 precision。默认 0.001 的预测截断可能降低可达到的召回率，跨实验比较须保留相同评测口径。

## 脚本回归验证

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_coco_person.py -v \
  2>&1 | tee logs/coco_person_unit_tests.log
```

测试覆盖完美框、空预测、无人图片高分误检、零面积误检、结果落盘、缺图/重复图/未知图 ID、标注摘要不匹配、类别不匹配及非法数值。

## 2026-09-22 全量验证结果

服务器 wjr，RTX 3090，Python 3.12.14，PyTorch 2.14.0 / CUDA 13.0，RF-DETR 1.11.0.dev0，pycocotools 2.0.11。使用 Seg-L 本地权重、默认 504×504 输入、阈值 0.001、官方未显式优化的 predict 路径。

全部 5,000 张图片完成，保存 156,830 个 person 候选框（含低分框，不能解释为人数）；从结果文件独立计算指标成功。11 项 CPU 回归测试通过。

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
|---:|---:|---:|---:|---:|---:|---:|
| 63.48% | 86.74% | 68.99% | 39.40% | 71.76% | 86.06% | 73.26% |

完整指标及原始预测保存在 `outputs/coco_person/L/`，推理和评测日志为 `logs/coco_person_L_predict.log`、`logs/coco_person_L_evaluate.log`。官方 checkpoint 加载器出现缺少 `args.num_queries / args.group_detr` 的兼容回退警告，原样保留在日志中；以上为本环境实测结果。
