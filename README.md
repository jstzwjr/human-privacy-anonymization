# Human Privacy Anonymization — 人体像素级分割与数据脱敏


## 仓库快速开始

本仓库包含项目脚本、调研文档、历史实测报告及固定版本的 RF-DETR submodule。模型、图片、运行输出和本机环境快照不进入 Git。

```bash
git clone --recurse-submodules https://github.com/cx1628555321/human-privacy-anonymization.git
cd human-privacy-anonymization
# 已克隆但未初始化 submodule 时：
git submodule update --init --recursive

# 在已配置的服务器上
conda activate rf-detr
# 新环境请参考 README_ENVIRONMENT.md；安装官方源码：
python -m pip install -e ./code/rf-detr
python download_assets.py
python smoke_inference.py --model L
python benchmark_images.py --model L
```

- [下载与校验](README_DOWNLOADS.md)
- [已验证环境](README_ENVIRONMENT.md)
- [三模型三图片历史计时](docs/BENCHMARK_RESULTS.md)
- [FP32 优化审计](docs/SPEED_AUDIT.md)
- [COCO person 检测框评测：保存预测与独立计算指标](docs/COCO_PERSON_EVALUATION.md)

文档中的服务器绝对路径是本次部署记录；Python 脚本按所在目录定位项目资源。`assets_manifest.json` 与 `SHA256SUMS` 是下载记录，重新下载后会更新。下载脚本遇到网络错误会退出，重试命令见下载说明。

## 在 wjr 服务器运行脚本

以下命令使用当前服务器的 `rf-detr` 环境。先执行一次：

```bash
ssh wjr
source /home/wjr/miniconda3/etc/profile.d/conda.sh
conda activate rf-detr
cd /home/wjr/mount/code/human-privacy-anonymization
mkdir -p logs
set -o pipefail
```

### COCO person 检测框：先保存预测，再计算指标

数据目录为 `/home/wjr/mount/dataset/coco`，包含 `val2017/` 和 `annotations/instances_val2017.json`。默认评测全部 5,000 张图片，包含无人图片；只计算 person 的 bbox AP/AR。

```bash
# 新建一次实验的目录，避免覆盖已有 L 模型结果。
eval_run="outputs/coco_person/L_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$eval_run"

# 第一步：GPU 推理，逐图保存框和置信度。
HF_HUB_OFFLINE=1 python evaluate_coco_person.py predict \
  --model L --data-root /home/wjr/mount/dataset/coco \
  --output "$eval_run/predictions.jsonl" \
  2>&1 | tee "$eval_run/predict.log"

# 第二步：只读取结果和标注，不加载模型、不需要 GPU。
python evaluate_coco_person.py evaluate \
  --results "$eval_run/predictions.jsonl" \
  --output "$eval_run/metrics.json" \
  2>&1 | tee "$eval_run/evaluate.log"
```

`--model` 支持 `M`、`L`、`2XL`；`predict --limit 8` 可以先跑小样本。默认置信度下限为 `0.001`，为 AP 保留低分预测。现有全量预测是 `outputs/coco_person/L/predictions.jsonl`；只重算它时，将第二步的 `--results` 指向该文件，并给 `--output` 指定新的文件名即可。

预测文件或指标文件已存在时，脚本拒绝覆盖。输出的 `AP` 为 IoU 0.50:0.05:0.95 的平均 AP，`AP50/AP75` 为固定 IoU 指标，`AR100` 中 100 表示每图最多保留 100 个预测框。文件中的指标为 0–1。完整格式与参数见 [COCO 评测说明](docs/COCO_PERSON_EVALUATION.md)。

### SCRFD 单张图片人脸检测

`scrfd_detector.py` 从 World Studio 的 `processing/face_anonymization/detector.py` 独立迁移，保留 `FaceBox`、`SCRFDDetector`、`detect()` 和 `active_provider()` 接口，不依赖 World Studio 或 Django。检测逻辑保持一致：RGB 输入、640×640 等比例缩放补零、置信度阈值 0.5、NMS 阈值 0.4。

当前服务器已将两个原模型复制到本项目，后续推理无需访问 World Studio 目录：

| 参数 | 本项目权重 | 原 World Studio 档位 |
|---|---|---|
| `--model 10g`（默认） | `model/scrfd/det_10g.onnx` | thorough |
| `--model 500m` | `model/scrfd/det_500m.onnx` | fast |

当前 `rf-detr` 环境使用 `onnxruntime-gpu 1.30.0`，依赖 CUDA 13 和 cuDNN 9；已在 RTX 3090 上验证 GPU 推理。新环境需要 `numpy`、`Pillow`，以及 CPU 版 `onnxruntime` 或 GPU 版 `onnxruntime-gpu`（二选一；GPU 版还需匹配的 CUDA/cuDNN 运行库）。

检测器保留 CUDA 优先、CPU 回退的选择。对于支持 `preload_dlls()` 的 CUDA 版 ONNX Runtime，初始化时先按[官方预加载方式](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls)加载当前 Python 环境中的 CUDA/cuDNN 库，再创建会话。此次 `libcublasLt.so.13: cannot open shared object file` 是库已安装在 `site-packages/nvidia/cu13/lib/`、但未被动态加载器找到，通过预加载修复。判断是否真正使用 GPU，请查看终端或 `result.json` 中的 `provider`，应为 `CUDAExecutionProvider`；仅在可用 provider 列表中看到 CUDA 并不能证明加载成功。

```bash
# 保存检测框、置信度与原尺寸可视化图片；输入图片不变。
python scrfd_image.py --image test_images/bus.jpg --model 10g \
  --output-dir "outputs/scrfd/bus_10g_$(date +%Y%m%d_%H%M%S)" \
  2>&1 | tee "logs/scrfd_bus_10g_$(date +%Y%m%d_%H%M%S).log"

# 也可以传入任意单张图片的绝对路径，或改用 --model 500m。
```

结果目录包含 `result.json`（原图像素坐标 `x/y/w/h`、score、模型摘要、实际执行后端、首轮耗时）和 `detections.png`（红色人脸框与置信度）。没有人脸时保存空列表和未画框的图片。已有输出目录拒绝覆盖；不指定 `--output-dir` 时，默认写入 `outputs/scrfd/<模型>/<图片名>/`。该入口只检测和画框，不执行模糊，也不计算 COCO person AP。

GPU 回归使用两个 SCRFD 模型，检查实际后端，并比较 bus、zidane、空白图的 CPU/GPU 检测结果（坐标容差 0.1 像素、置信度容差 0.001）：

```bash
SCRFD_TEST_CUDA=1 python -m unittest discover -s tests -p test_scrfd_cuda.py -v \
  2>&1 | tee logs/scrfd_cuda_regression.log
```

此测试需要本地模型、测试图片和可用 GPU；未设置 `SCRFD_TEST_CUDA=1` 时，常规测试会跳过它。

在本项目 Python 代码中可以直接调用：

```python
from pathlib import Path
import numpy as np
from PIL import Image
from scrfd_detector import SCRFDDetector

detector = SCRFDDetector(Path("model/scrfd/det_10g.onnx"))
with Image.open("test_images/bus.jpg") as image:
    faces = detector.detect(np.asarray(image.convert("RGB")))
for face in faces:
    print(face.x, face.y, face.w, face.h, face.score)
```

权重是已有文件的本地副本，不进入 Git；新克隆需自行放入上表路径。原权重来源为 InsightFace v0.7 的 `buffalo_l` / `buffalo_sc` 模型包，原项目记录其用途为非商业研究，详见 [上游模型说明](https://github.com/deepinsight/insightface#license)。

---

> 最新状态（2026-09-18）：已创建并验证 rf-detr Conda 环境，M/L/2XL 已完成单图 GPU 推理；详见 [环境说明](README_ENVIRONMENT.md)。下文未安装/未推理描述属于下载阶段记录。

调研整理日期：2026-09-18
项目目录：`/home/ubuntu/data_1/chenxiang/human-privacy-anonymization`
部署目标：用户指定的 RTX 3090 服务器（具体 GPU、显存及软件环境待实测确认）。

> 当前阶段：调研与方案设计。尚未安装模型、下载权重或运行本地性能评测。本文的公开性能数字不是该服务器的实测结果。

## 1. 项目目标

对图片和视频数据中的所有可见人体进行像素级分割，并对人体覆盖区域执行强模糊，降低数据使用、共享过程中的人员隐私暴露风险。

- 处理范围包括头发、脸、衣物、手臂、手、腿、鞋等可见人体区域。
- 背身、蹲坐、遮挡、远处小人及画面边缘的局部人体均应纳入处理；不要求头和脚同时出现。
- 默认处理画面中所有人，输出保持原图或原视频帧分辨率。
- 正常情况沿人体轮廓处理；分割异常时允许保守扩大覆盖，必要时对整个检测框处理。
- 模糊后的图像不是无条件的匿名化保证。验收应检查处理后是否仍保留可辨认细节。若不需要保留人体纹理，可使用不透明掩码覆盖。
- 姓名牌、屏幕文字、其他区域的个人信息及视频音轨不由人体分割自动解决，需要按数据范围另行处理。

## 2. 选型结论

| 角色 | 推荐模型 | 选择理由 |
|---|---|---|
| 首版主模型 | **RF-DETR-Seg-L** | 像素分割质量与处理成本较均衡 |
| 精度对照 | **RF-DETR-Seg-2XL** | 适合离线数据处理，验证更高计算量是否减少人体遗漏 |
| 吞吐对照 | **RF-DETR-Seg-M** | 大规模数据处理时评估速度与覆盖完整性的折中 |
| 算力受限备选 | **YOLO26-seg-S/M** | 模型较轻，部署生态完善；需考虑授权 |
| 困难样本辅助 | **SAM 3.1** | 评估复杂视频的分割与跟踪质量，不预设其人体精度必然最高 |

这些是工程实测优先级，不是已经证明的隐私保护效果排名。通用 COCO 分割 AP 无法直接代表人体专项召回率或脱敏完整性。

## 3. 验收指标优先级

1. **人员漏检率**：真实可见人员中，完全没有被处理的比例。
2. **人体未覆盖像素比例**：真实人体像素中未被最终脱敏掩码覆盖的比例；同时检查小目标和最差样本，避免大人物掩盖小人物失败。
3. **连续漏分帧数和漏处理片段数**：视频中短暂漏检也会暴露清晰画面。
4. **脱敏强度**：脸部、衣服文字、纹理等是否仍清晰可辨。
5. **端到端性能**：包括解码、预处理、推理、掩码还原、模糊及编码的吞吐量、P50/P95 延迟和显存峰值。
6. **背景误覆盖比例与边界观感**：在覆盖完整的前提下尽量保留背景。

对本项目，漏处理人的代价高于多处理少量背景。mAP、IoU 和边界指标可作辅助，但不能替代以上指标。

## 4. 技术路线与候选比较

### 4.1 实例分割、语义分割和抠图

- **实例分割**：每个人独立输出像素掩码，通常同时给出框、类别和置信度；适合多人发现、质量检查及后续跟踪。
- **人体语义分割**：输出全部人体的统一前景掩码；若无需区分个人，也能用于脱敏。
- **人体抠图（Matting）**：输出连续透明度 alpha，用于发丝和半透明边缘。边缘精细不等于能自动找齐所有人。
- **普通人体检测**：只输出框，不满足主流程的像素轮廓要求，仅保留作兜底。

### 4.2 模型优劣

| 模型 | 输出/优势 | 局限 | 适用性 |
|---|---|---|---|
| RF-DETR-Seg | 多人实例 mask、框、类别、置信度；公开 GPU 精度/延迟表现强 | Nano 也约 33.6M 参数；较低输入尺寸需重点测小人 | GPU 离线/在线主线 |
| YOLO26-seg | 多人实例 mask；轻量变体、跟踪和导出工具完整 | 细小肢体与边界需验证；AGPL/Enterprise | 轻量部署对照 |
| SAM 3.1 | 文本提示驱动的视频分割与跟踪 | 重模型、部署复杂、权重访问条件；不能保证找齐所有人 | 离线质量对照/困难样本 |
| 检测器 + SAM 2.1 | 框提示细化 mask，视频传播；SAM 2.1 采用 Apache-2.0 | 两阶段成本；需补检新入场目标、防止传播漂移 | 宽松许可证的视频备选 |
| BiRefNet-portrait | 人像抠图、细致边缘；模型卡标注 MIT | 不是完整多人检测跟踪系统；远景需验证 | 近景图片精细化 |
| MatAnyone 2 | 首帧 mask 引导视频抠图，输出前景和 alpha | 需要初始 mask；新入场人员需额外发现；商业使用需许可 | 视频边缘精细化研究 |
| Robust Video Matting（RVM） | 时序记忆、无额外提示的人体视频抠图，多种部署格式 | 不能据抠图演示认定复杂场景全员召回；GPL-3.0 | 近景视频性能对照 |
| PP-HumanSegV2 / MediaPipe Selfie | 轻量人像分割 | 自拍/人像定位，不能直接推定适合远景、拥挤场景 | 场景限定的轻量备选 |
| Sapiens2 | 高分辨率人体部位分割，可合并前景类别 | 计算量高；许可证禁止 surveillance 等用途 | 不作为本项目默认部署候选 |

## 5. 公开精度和速度排名

下表按分割精度降序排列。来源为 **Roboflow 自测**的统一对比，不是独立第三方评测。

- 精度：COCO val2017、Mask AP50:95，多类别平均指标，越高越好。
- 延迟：NVIDIA T4、TensorRT、FP16、batch size=1，越低越好。
- 参数量：部署模型口径；与训练 checkpoint 参数/体积可能不同。
- 不将其他硬件、数据集或抠图指标混入本表；Ultralytics 官方自测数值可能因口径而不同。

| 精度排名 | 模型 | Mask AP | 延迟 ms | 参数量 M | 输入尺寸 |
|---:|---|---:|---:|---:|---|
| 1 | RF-DETR-Seg-2XL | 49.9 | 21.80 | 38.6 | 768×768 |
| 2 | RF-DETR-Seg-XL | 48.8 | 13.50 | 38.1 | 624×624 |
| 3 | RF-DETR-Seg-L | 47.1 | 8.80 | 36.2 | 504×504 |
| 4 | YOLO26-seg-X | 46.8 | 12.92 | 62.8 | 640×640 |
| 5 | YOLO26-seg-L | 45.5 | 7.58 | 28.0 | 640×640 |
| 6 | RF-DETR-Seg-M | 45.3 | 5.90 | 35.7 | 432×432 |
| 7 | YOLO26-seg-M | 44.0 | 6.32 | 23.6 | 640×640 |
| 8 | RF-DETR-Seg-S | 43.1 | 4.40 | 33.7 | 384×384 |
| 9 | RF-DETR-Seg-N | 40.3 | 3.40 | 33.6 | 312×312 |
| 10 | YOLO26-seg-S | 40.2 | 3.47 | 10.4 | 640×640 |
| 11 | YOLO26-seg-N | 34.7 | 2.31 | 2.7 | 640×640 |

速度由快到慢：

`YOLO26-N → RF-Seg-N → YOLO26-S → RF-Seg-S → RF-Seg-M → YOLO26-M → YOLO26-L → RF-Seg-L → YOLO26-X → RF-Seg-XL → RF-Seg-2XL`

以上 YOLO 均指 seg 版本。源数据：[RF-DETR Benchmarks](https://github.com/roboflow/rf-detr#benchmarks)。

**不能将该排序直接作为人体脱敏排序，也不能将 T4 数字当作 RTX 3090 性能。** SAM 3.1、SAM 2.1、BiRefNet、RVM、MatAnyone 2 缺少与该表一致的公开条件，暂不做跨任务数值排名。

## 6. 模型大小、输入与输出

### 6.1 主候选模型大小

| 模型 | 参数量 | FP16 纯参数存储估算 | 默认/基准输入 |
|---|---:|---:|---|
| RF-DETR-Seg-M | 35.7M | 71.4 MB | 432×432 |
| RF-DETR-Seg-L | 36.2M | 72.4 MB | 504×504 |
| RF-DETR-Seg-2XL | 38.6M | 77.2 MB | 768×768 |
| YOLO26-seg-S | 10.4M | 20.8 MB | 640×640 |
| YOLO26-seg-M | 23.6M | 47.2 MB | 640×640 |
| SAM 3.1 | 本次未核实独立精确参数量 | 官方 checkpoint 约 3.5 GB，不等于 FP16 参数量 | 标准内部图像处理尺寸约 1008×1008 |

MB 使用十进制。估算为参数量×2 字节，**不是下载体积、TensorRT 引擎大小或显存需求**。SAM 3 的 848M 参数不能未经核实直接作为 SAM 3.1 的精确参数数目。

RF-DETR 当前 Seg-M/L 等变体使用 24 的倍数作为自定义正方形输入约束，更通用的规则为 `patch_size × num_windows`，应以锁定版本为准。YOLO 可通过 `imgsz` 调整，需满足步长与导出后端约束。固定形状 TensorRT 引擎改变输入配置后需重建或配置相应 profile。

### 6.2 输出尺寸

原图尺寸、推理尺寸和成品尺寸是三件不同的事。所有主候选均可构建输出 720p、1080p、1440p、4K 原尺寸成品的流程：

```text
原图 3840×2160
  → 缩放到模型推理尺寸
  → 输出分割预测并还原为原图坐标/尺寸的掩码
  → 在保留的 3840×2160 原图上模糊
  → 输出 3840×2160 成品
```

4K 输出不代表原生 4K 分割精度。低分辨率推理丢失的细节无法靠放大 mask 恢复；需要评估更大输入、切片或局部细化的成本。

YOLO Python 接口可使用 `retina_masks=True` 返回与原图尺寸一致的 mask；默认关闭时，必须正确处理缩放和 padding。ONNX/TensorRT 原始输出需要匹配相应的解码、阈值处理与坐标还原。

### 6.3 统一业务输出接口建议

以 1920×1080 图像中有 N 个人为例：

```text
boxes:       [N, 4]             原图像素坐标 xyxy
scores:      [N]                目标置信度
masks:       [N, 1080, 1920]    每个人的二值掩码
track_ids:   [N]                可选，跨帧 ID
union_mask:  [1080, 1920]       全部人体的联合掩码
output:      [1080, 1920, 3]    后处理生成的脱敏图像
```

这是统一接口设计，不是各模型原始 API 的字段声明。SAM 的提示概念不等同于 COCO 固定 class ID；不同库的 person ID 编码也应核对，不应跨库硬编码同一 ID。

分割模型不直接输出模糊图片；模糊由后处理执行。一个 4K uint8 mask 约 8.3 MB，应及时合并和释放实例 mask，避免多人视频内存累积。

## 7. 脱敏实现方案

```text
解码图片/视频
  → 检测并分割所有可见人体
  → 按原图坐标还原 mask
  → 合并人体 mask、适度外扩和修补异常孔洞
  → 视频时序处理与持续新目标检测
  → 正常样本按扩张 mask 强模糊
  → 异常样本扩大覆盖或进入复核
  → 编码脱敏成品 + 处理日志
```

关键策略：

- 通过验证集调节检测和 mask 阈值，优先覆盖，不直接使用面向展示效果的默认阈值。
- 外扩幅度随人物尺度调整，覆盖头发、手脚边缘和运动模糊区域；不盲目填满所有背景孔洞。
- 分割异常但检测到人时，可采用扩大后的框进行保守遮挡；检测也漏人时，框兜底无效，需要困难样本评测、补充检测或复核。
- 可评估人脸/头部检测作为额外检查，但只能补足相应可见区域，不能替代全身分割。
- 视频历史 mask 必须进行运动对齐或可靠传播；不能简单平均上一帧掩码。
- 持续发现新入场人员；跟踪不能取代检测，也不能因为单帧失败立即撤销覆盖。
- 模糊强度随人物大小调整。Matting 的 alpha 不直接作为脱敏混合权重，避免边缘混回清晰原图。
- 推理失败或异常结果不自动输出未处理原图；记录错误、进入复核或采取预设保守覆盖。
- 处理日志记录版本、阈值、模型、输入输出关联与失败状态；原始数据不混入脱敏交付目录。

## 8. RTX 3090 首轮验证计划

1. 检查 GPU 型号、可用显存、驱动、CUDA、磁盘空间及现有 Python 环境。
2. 固定模型代码版本、权重来源、输入尺寸、预处理、后处理与导出配置。
3. 准备代表性图片及连续视频，覆盖多人、遮挡、远景、背身、蹲坐、画面边缘、低光、运动和新人入场。
4. 标注全体可见人体和困难局部，划分阈值调优集与独立验收集。
5. 首轮比较 RF-DETR-Seg-M/L/2XL 与 YOLO26-seg-S，使用一致的成品分辨率、模糊策略和验收指标。
6. 分别记录模型推理延迟和完整流水线成本；暖机后统计，单独记录冷启动。
7. 如增加精度仍不能解决遗漏，再评估 SAM 3.1、较高输入尺寸、切片及局部细化。
8. 检查 TensorRT/FP16 导出前后的人体覆盖差异，不能只验证能成功运行。

是否采用批处理、量化、多进程或多路视频并发，应在首轮基线之后决定。本文不承诺 3090 上的具体 FPS、显存或零漏检。

## 9. 许可证摘要

| 项目 | 当前调研结论 |
|---|---|
| RF-DETR-Seg | 官方表中 Seg-N 至 Seg-2XL 为 Apache-2.0；不要与部分 detection Plus 模型的 PML 授权混淆 |
| YOLO26 | AGPL-3.0 / Enterprise，应按实际集成与交付方式确认 |
| SAM 2.1 | 官方模型 checkpoint、训练和 demo 代码采用 Apache-2.0，部分 demo 资产另有许可证 |
| SAM 3 / 3.1 | 自定义 SAM License，HF 访问需满足条件；不能称为 Apache-2.0 |
| BiRefNet-portrait | HF 模型卡标注 MIT |
| RVM | GPL-3.0 |
| MatAnyone 2 | S-Lab License，非商业使用限制，商业使用需许可 |
| Sapiens2 | 自定义许可，明确限制 surveillance、biometric processing 等用途，不作为默认方案 |

最终以实际采用版本的代码、权重和依赖许可证为准。

## 10. 官方资料与代码入口

- [RF-DETR 代码、模型与统一基准](https://github.com/roboflow/rf-detr)
- [RF-DETR 导出文档](https://rfdetr.roboflow.com/latest/learn/export/)
- [RF-DETR 输入尺寸与变体说明](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/variants.py)
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [YOLO 实例分割输出与规格](https://docs.ultralytics.com/tasks/segment/)
- [YOLO 推理与 retina_masks](https://docs.ultralytics.com/modes/predict/)
- [SAM 3 / 3.1 官方代码](https://github.com/facebookresearch/sam3)
- [SAM 3.1 发布说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [SAM 3.1 Hugging Face 权重](https://huggingface.co/facebook/sam3.1/tree/main)
- [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [SAM 2.1 官方代码与基准](https://github.com/facebookresearch/sam2)
- [BiRefNet 官方代码](https://github.com/ZhengPeng7/BiRefNet)
- [BiRefNet-portrait 模型卡](https://huggingface.co/ZhengPeng7/BiRefNet-portrait)
- [MatAnyone 2 官方代码](https://github.com/pq-yang/MatAnyone2)
- [MatAnyone 2 Hugging Face](https://huggingface.co/PeiqingYang/MatAnyone2)
- [MatAnyone 2 许可证](https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt)
- [RVM 官方代码](https://github.com/PeterL1n/RobustVideoMatting)
- [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/contrib/PP-HumanSeg/README_cn.md)
- [MediaPipe 人像分割](https://developers.google.cn/edge/mediapipe/solutions/vision/image_segmenter)
- [Sapiens2 官方代码](https://github.com/facebookresearch/sapiens2)
- [Sapiens2 许可证](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)

## 11. 后续产出

- 可复现的依赖环境与锁定版本。
- 图片/视频人体分割及脱敏脚本。
- 单一配置文件管理阈值、尺寸、模糊与兜底策略。
- RTX 3090 实测报告、失败样本分析和最终选型。
- 不包含未脱敏原始数据的交付目录与处理记录。
