# 三模型 × 三图片推理测试

测试日期：2026-09-18；环境 `rf-detr`；RTX 3090；模型顺序执行，不并行竞争 GPU。

每张图片单独预热 3 次、计时 10 次，计时前后执行 CUDA synchronize。统计 model.predict(PIL image) 调用，包括 API 内部预处理、模型计算和分割后处理；不包括模型加载、文件解码、掩码外扩、模糊、图片保存。未显式启用 FP16、compile 或 TensorRT。不是端到端脱敏流水线耗时。

| 模型 | 图片 | 人体实例数 | 平均 ms | 中位数 ms | P95 ms |
|---|---|---:|---:|---:|---:|
| M | bus.jpg | 4 | 17.72 | 17.07 | 20.52 |
| M | zidane.jpg | 3 | 17.58 | 17.52 | 18.17 |
| M | coco_val2017_000000000785.jpg | 1 | 16.37 | 16.11 | 17.65 |
| L | bus.jpg | 4 | 22.25 | 21.58 | 25.18 |
| L | zidane.jpg | 2 | 21.89 | 21.56 | 23.46 |
| L | coco_val2017_000000000785.jpg | 1 | 20.92 | 20.54 | 22.09 |
| 2XL | bus.jpg | 4 | 49.55 | 49.18 | 52.60 |
| 2XL | zidane.jpg | 2 | 49.00 | 48.85 | 49.89 |
| 2XL | coco_val2017_000000000785.jpg | 1 | 47.30 | 47.07 | 48.36 |

每个模型目录保存 `<图片名>_mask.png`、`<图片名>_blurred.jpg`、`<图片名>_result.json`。总表为 timings.csv；每张图的全部 10 次计时保存在对应 result JSON 中。原来的 result.json 是前一轮 bus 首帧测试记录，请使用新生成的逐图记录和 benchmark_results.json 查看本轮结果。

阈值 0.3；膨胀核 11×11；高斯模糊核 101×101，sigma=35；全部输出保持原图尺寸。

M 在 zidane.jpg 返回 3 个人体实例，L/2XL 返回 2 个；这属于需要复核的预测差异，不能将实例数直接当作召回率或精度。三张公开图片仅供检查效果，不足以证明隐私脱敏覆盖率。官方 checkpoint 兼容回退警告保留在 logs/benchmark_*.log 中。

复现：

```bash
conda activate rf-detr
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
for model in M L 2XL; do
  HF_HUB_OFFLINE=1 python benchmark_images.py --model "$model" --warmup 3 --repeats 10
done
```
