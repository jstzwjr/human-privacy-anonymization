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
