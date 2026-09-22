set -o pipefail
scrfd_run="outputs/scrfd_coco_person/10g_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$scrfd_run"

# 保存人脸预测，模型可选 10g 或 500m
python evaluate_scrfd_coco_person.py predict --model 10g \
  --output "$scrfd_run/predictions.jsonl" \
  2>&1 | tee "$scrfd_run/predict.log"

# 计算关联指标，无需 GPU
python evaluate_scrfd_coco_person.py evaluate \
  --results "$scrfd_run/predictions.jsonl" \
  --ioa-threshold 0.9 \
  --output "$scrfd_run/metrics.json" \
  2>&1 | tee "$scrfd_run/evaluate.log"
