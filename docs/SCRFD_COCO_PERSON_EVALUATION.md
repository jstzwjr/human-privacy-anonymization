# SCRFD 与 COCO person 的 IoA 关联评测

`evaluate_scrfd_coco_person.py` 使用本项目 `scrfd_detector.py` 检测人脸，先保存预测，再独立计算人脸框与人体框的关联指标。它用于观察现有人脸方案能关联到多少人体、多少检测落在人体外。COCO person 没有人脸框标签，因此这不是标准 COCO bbox AP，也不能作为真实人脸检测精度。

## 环境与执行

```bash
ssh wjr
source /home/wjr/miniconda3/etc/profile.d/conda.sh
conda activate rf-detr
cd /home/wjr/mount/code/human-privacy-anonymization
mkdir -p logs
set -o pipefail

scrfd_run="outputs/scrfd_coco_person/10g_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$scrfd_run"

# 第一步：整个数据集只推理一次，每张图片保存一行，空检测也保存。
python evaluate_scrfd_coco_person.py predict --model 10g \
  --data-root /home/wjr/mount/dataset/coco \
  --output "$scrfd_run/predictions.jsonl" \
  2>&1 | tee "$scrfd_run/predict.log"

# 第二步：只读取预测与人体标注，不加载模型，不需要 GPU 或原图片。
python evaluate_scrfd_coco_person.py evaluate \
  --results "$scrfd_run/predictions.jsonl" --ioa-threshold 0.9 \
  --output "$scrfd_run/metrics.json" \
  2>&1 | tee "$scrfd_run/evaluate.log"

# 比较更严格的阈值，无需重新推理。
python evaluate_scrfd_coco_person.py evaluate \
  --results "$scrfd_run/predictions.jsonl" --ioa-threshold 0.95 \
  --output "$scrfd_run/metrics_ioa95.json" \
  2>&1 | tee "$scrfd_run/evaluate_ioa95.log"
```

默认数据目录是 `/home/wjr/mount/dataset/coco`，包含 `val2017/` 和 `annotations/instances_val2017.json`。默认评测全部 5,000 张图片，包含无人图片；按图像 ID 排序。`predict --limit 8` 仅用于快速验证，指标文件会标明是否为完整验证集。

模型可选 `--model 10g`（默认）或 `--model 500m`，分别读取 `model/scrfd/det_10g.onnx`、`model/scrfd/det_500m.onnx`。整个预测过程复用同一检测器；保留当前接口的 RGB 输入、640×640 等比例缩放补零、置信度阈值 0.5、NMS 阈值 0.4，以及 CUDA 优先、CPU 回退。实际后端会打印并写入预测首行。

推理依赖现有 `numpy`、`Pillow`、ONNX Runtime；评测只依赖 `numpy` 和 `pycocotools`，不导入 torch、RF-DETR 或 ONNX Runtime。两种评测共用 `evaluate_coco_person.py` 中的标注读取、预测文件完整性和数值校验。

## 匹配规则

对于人脸框 F 与人体框 P：

`IoA(F, P) = area(F ∩ P) / area(F)`

例如，10×10 的人脸框完全位于 100×100 的人体框内，IoA 为 1，IoU 仅为 0.01。本脚本使用 IoA，默认匹配条件是 `IoA >= 0.9`；`--ioa-threshold` 必须大于 0 且不超过 1。

1. 仅使用所选图片中的 person 标注，按检测置信度降序匹配。
2. 每个人脸框最多匹配一个普通人体框，每个普通人体框最多匹配一张脸。存在多个可匹配人体时选择 IoA 最大者；相同分数和重叠值的处理沿用 pycocotools 的稳定排序及匹配规则。
3. 成功匹配普通人体计 TP；未匹配的普通人体计 FN。重复检测同一人体，超出的检测通常计 FP。
4. 没匹配到普通人体，但满足 crowd 人体框 IoA 条件的检测计 ignored；一个 crowd 区域可吸收多个检测，crowd 不进入人体召回率分母。普通人体匹配优先于 crowd。
5. 其余检测计 FP，无人体标注图片上的检测同样计 FP。所有已保存检测都参与评测，不设每图 100 框截断，也不做小/中/大面积分组。
6. 人脸框保留原接口坐标，不裁剪到图像边界，IoA 分母使用原始人脸框面积。零面积预测的 IoA 为 0，保留计 FP；负宽高和非有限数值拒绝评测。

实现复用 pycocotools 的一对一/crowd 匹配与 PR 累计逻辑，仅将重叠矩阵改为人脸框面积作分母。计算重叠时使用其 crowd-IoA 公式，但不修改真实 GT 的 `iscrowd`。

## 指标与解释

| 字段 | 含义 |
|---|---|
| `association_AP` | 固定 IoA 阈值下，按置信度排序的 PR 曲线在 101 个 recall 点的插值平均 |
| `association_precision` | TP / (TP + FP)，crowd ignored 不进入分母 |
| `person_recall` | TP / 非 crowd 人体数 |
| `association_F1` | 2×TP / (2×TP + FP + FN) |
| `counts` | TP、FP、FN、ignored、普通人体数、crowd 人体数、保存检测总数 |

指标为 0–1；没有普通人体 GT 时，AP/recall/F1 为 -1。没有有效预测时 precision 为 0；有普通人体但预测全空时，AP/precision/recall/F1 均为 0。

需要注意三点：

- 检测框只要位于人体内，即使落在衣服或躯干上也可能算 TP；重叠的人体框也可能造成关联歧义。因此高 precision 不能证明它是真实人脸。
- COCO 中背身、脸不可见、遮挡的人仍进入普通人体数量，低 recall 不一定意味着漏检可见人脸。该 recall 表示“能通过一张检测脸关联到的人体比例”，不表示模糊覆盖全身的比例。
- SCRFD 接口在保存前已经过滤 score < 0.5，AP 仅基于剩余预测。降低 IoA 阈值不能恢复被过滤的低分人脸。此 AP 不与 RF-DETR 的标准 person bbox AP 直接比较，也不是多个 IoA 阈值的 mAP。

## 结果文件

默认预测路径为 `outputs/scrfd_coco_person/<模型>/predictions.jsonl`。首行格式标识为 `scrfd-coco-person-ioa-v1`，记录模型权重及标注 SHA-256、待评测图像 ID 列表、阈值、输入尺寸、实际 provider 和 ONNX Runtime 版本。后续每张图一行：

```json
{"image_id": 139, "detections": [{"bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.95}]}
```

`bbox` 是原图像素坐标 `[x, y, width, height]`，内容是人脸框；首行 `category_id` 指向关联目标 person。无人脸时写空 `detections`。逐图写入并 flush，不支持断点续跑；推理失败会退出，不伪装成空预测。

默认指标路径是在预测文件旁生成 `.metrics.json`；`--annotations` 可指定同一份标注的其他位置。指标保存 IoA 阈值、匹配规则、阈值截断信息、完整/子集标识及 counts/metrics。预测和指标文件均拒绝覆盖，换阈值评测时请使用新的 `--output`。

不完整、重复或未知图像 ID、不同标注摘要、不同结果格式、错误类别和非法检测数值会被拒绝，避免误将中断预测或 RF-DETR 人体框用于此评测。

## 回归测试

```bash
python -m unittest discover -s tests -p test_scrfd_coco_person.py -v \
  2>&1 | tee logs/scrfd_coco_person_tests.log
```

测试使用合成标签和保存结果，不需要 GPU 或模型。覆盖 IoA 与 IoU 区别、阈值边界、一对一与 crowd 匹配、无人图误检、101 点 AP、超过 100 个检测、零面积/越界框、空预测以及文件校验。

## 2026-09-22 全量验证

wjr 服务器、RTX 3090、ONNX Runtime GPU 1.30.0；SCRFD 10g 使用 CUDAExecutionProvider，在全部 5,000 张 COCO val2017 图片上保存 6,274 个人脸框，置信度下限为 0.5。模型初始化后的图片读取、检测和逐图写入耗时约 39.7 秒，此耗时不是纯模型前向时间。

| IoA 阈值 | 关联 AP | 关联 precision | 人体 recall | F1 | TP | FP | FN | crowd ignored |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 36.85% | 86.06% | 38.16% | 52.87% | 4,112 | 666 | 6,665 | 1,496 |
| 0.95 | 35.77% | 85.00% | 37.65% | 52.19% | 4,058 | 716 | 6,719 | 1,500 |

两组指标使用同一预测文件，分母包含 10,777 个普通人体，另有 227 个 crowd 标注。指标含义和限制见上文，不能将这里的关联 precision 解释为真实人脸准确率。

完整预测及两组指标位于 `outputs/scrfd_coco_person/10g/`：`predictions.jsonl`、`predictions.metrics.json`、`predictions.ioa95.metrics.json`。推理日志为 `logs/scrfd_coco_person_10g_predict.log`，默认评测日志为 `logs/scrfd_coco_person_10g_evaluate.log`，离线重算及完整性验证日志为 `logs/scrfd_coco_person_offline_validation.log`。

500m 也完成了 8 张图片的 GPU 流程验证，结果位于 `outputs/scrfd_coco_person/500m_smoke/`，仅作运行验证。新增 26 项 IoA 回归和原有 11 项 COCO 回归全部通过；完整测试发现 41 项，40 项通过，独立的 SCRFD GPU 对照测试因未设置开关而跳过。本次真实推理已验证两种模型实际使用 CUDA。
