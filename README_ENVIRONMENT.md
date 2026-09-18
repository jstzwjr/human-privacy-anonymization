# RF-DETR Conda 推理环境

配置与验证日期：2026-09-18。

## 结果

已从 `py312` 克隆创建 `rf-detr`，安装项目官方源码及推理依赖。原始 `py312` 未修改。

- 环境：`/home/ubuntu/miniconda3/envs/rf-detr`
- Python 3.12.13
- PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128
- transformers 5.17.0 / supervision 0.30.4 / pyDeprecate 0.9.0
- RF-DETR 1.11.0.dev0，官方源码 commit `5669b7c8e3ac961a9fac3c3dc3967c328c7ffec3`
- GPU：NVIDIA GeForce RTX 3090 24GB
- `pip check` 通过。
- M、L、2XL 均完成本地权重加载和 GPU 分割，每次独立进程加载一个模型。
- 测试图 `bus.jpg`：三个模型均返回 4 个人，实例掩码形状 `[4,1080,810]`，成品尺寸 810×1080。

克隆继承的 `mdm 1.0.0` 要求 torch 2.6.0 和未安装的 xformers，与本项目无关且依赖冲突，因此仅在新环境卸载其包注册。没有改动原始环境。

## 激活与执行

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate rf-detr
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization

HF_HUB_OFFLINE=1 python smoke_inference.py --model L
HF_HUB_OFFLINE=1 python smoke_inference.py --model M
HF_HUB_OFFLINE=1 python smoke_inference.py --model 2XL
```

无需激活环境也可以使用绝对路径：

```bash
cd /home/ubuntu/data_1/chenxiang/human-privacy-anonymization
HF_HUB_OFFLINE=1 /home/ubuntu/miniconda3/envs/rf-detr/bin/python smoke_inference.py --model L
```

## 输出

每个模型输出到 `outputs/smoke/<M|L|2XL>/`：

- `bus_mask.png`：合并并外扩后的人体二值掩码。
- `bus_blurred.jpg`：原尺寸人体模糊图。
- `result.json`：GPU、人数、尺寸、首轮预测耗时。

运行日志：`logs/smoke_M.log`、`logs/smoke_L.log`、`logs/smoke_2XL.log`。

这是单图推理验证脚本，不是生产批处理流水线。当前使用阈值 0.3、11×11 掩码膨胀、101×101 高斯模糊核及 sigma=35，参数仅用于展示。尚未完成困难场景覆盖率、隐私脱敏充分性或稳态吞吐测试。

## 安装过程记录

```bash
/home/ubuntu/miniconda3/bin/conda create -y -n rf-detr --clone py312
/home/ubuntu/miniconda3/envs/rf-detr/bin/python -m pip install -e ./code/rf-detr
/home/ubuntu/miniconda3/envs/rf-detr/bin/python -m pip uninstall -y mdm
/home/ubuntu/miniconda3/envs/rf-detr/bin/python -m pip check
```

安装命令中的源码路径相对于项目目录。日志见 `logs/environment_install.log`。版本快照见 `environment/rf-detr-pip-freeze.txt` 与 `environment/rf-detr-conda.yml`，快照包含继承包及本机路径，不是跨机器通用锁文件。

## 官方警告与验证边界

- 官方加载器提示 checkpoint 缺少 `args.num_queries / args.group_detr`，采用兼容回退。三个模型的单图推理均成功，但这不代表已复现官方精度；日志保留该警告，后续正式评测应检查兼容性。
- positional encoding / patch size 提示出现在构建 backbone 时，本次加载的是完整 RF-DETR checkpoint，并已完成离线预测。
- 尚未调用 FP16 推理优化或 TensorRT 导出；当前结果用于验证环境与权重可用性。
- 首轮预测包含启动开销，不能据 `first_predict_seconds` 给模型作速度排名。
