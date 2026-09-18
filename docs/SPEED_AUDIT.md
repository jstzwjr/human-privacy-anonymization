# FP32 推理速度与 GPU 审计

日期：2026-09-18。未修改原 benchmark_images.py、模型权重、阈值或输入尺寸。

方法：每张图片每种模式预热 5 次、计时 20 次；CUDA 同步；三个模型顺序运行；模式顺序为 baseline、no_source、optimized_fp32。固定顺序和共享 GPU 可能存在时钟/负载偏差，因此结果是短测观察，不是严格独立性能基准。

optimized_fp32 = model.inference(compile=False, dtype=torch.float32) + predict(include_source_image=False)。

| 模型 | 图片 | 基线均值 ms | FP32 优化均值 ms | 延迟下降 | 掩码一致 |
|---|---|---:|---:|---:|---|
| M | bus.jpg | 17.11 | 15.46 | 9.6% | True |
| M | zidane.jpg | 17.60 | 15.10 | 14.2% | True |
| M | coco_val2017_000000000785.jpg | 16.18 | 14.32 | 11.5% | True |
| L | bus.jpg | 21.85 | 19.34 | 11.5% | True |
| L | zidane.jpg | 22.01 | 19.43 | 11.7% | True |
| L | coco_val2017_000000000785.jpg | 21.04 | 18.84 | 10.4% | True |
| 2XL | bus.jpg | 49.30 | 44.59 | 9.6% | True |
| 2XL | zidane.jpg | 48.90 | 44.86 | 8.3% | True |
| 2XL | coco_val2017_000000000785.jpg | 47.45 | 43.04 | 9.3% | True |

## GPU 与结果一致性

- forward pre-hook 实测三个模型参数和输入均为 cuda:0 / torch.float32，training=False，inference_mode=True。不是仅依靠 cuda.is_available 或显卡名称推断。
- 三模型×三图片：FP32 优化后类别、框、二值 mask 与基线完全一致，mask_changed_pixels=0；置信度最大差异约 1.79e-7，因此不声称所有浮点输出逐位一致或所有未来图片都无变化。
- no_source 模式输出完全一致，但短测没有稳定速度收益；它不改变模型计算，可用于减少不必要的元数据持有。
- 实测 Torch matmul TF32=True，cuDNN TF32=True，均保持环境默认值；没有主动打开新的降精度选项。

## 推荐

优先考虑官方 FP32 推理路径，但在隐私敏感业务上线前扩大困难样本一致性验证。异步解码/写文件、模型常驻及减少重复文件 I/O 可改善端到端吞吐，当前计时不包含这些步骤，不能据本表量化收益。

当前官方 predict 已带 inference_mode、自动 eval、pin_memory 和 non_blocking 传输，重复添加没有必要。batch 与编译可再评估，不能预先保证逐像素一致。FP16/BF16、TF32 变更、INT8/TensorRT、降低输入分辨率、提高阈值或跳帧均不列作已证明无精度影响的优化。

无需通过删除 torch.cuda.synchronize 来制造更短计时；这会漏算异步 GPU 工作。性能生产流水线可以减少不必要的全局同步，但计时仍需正确覆盖任务完成。

复现：
```bash
conda activate rf-detr
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
for model in M L 2XL; do
  HF_HUB_OFFLINE=1 python audit_inference_speed.py --model "$model"
done
```

详细每次计时和一致性字段：outputs/speed_audit/{M,L,2XL}.json；日志：logs/speed_audit_*.log。
