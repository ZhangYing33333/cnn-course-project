# 神经网络与深度学习 CNN 大作业

## 1. 项目说明

项目实现 CIFAR 适配版 ResNet-18，并在 CIFAR-10 上训练一个基线模型和三个改进模型。CIFAR-10 官方训练集按类别分层划分为训练集和验证集；验证集用于逐轮选 checkpoint。CIFAR-10 官方测试集不参与训练、选轮数或选模型。四个最终 checkpoint 仅在训练和模型选择完成后用于 CIFAR-10.1 v6 场景推理对比。

## 2. 模型

输入为 `3×32×32`，输出 10 类。主干采用 ResNet-18 BasicBlock 配置 `[2,2,2,2]`。为适配 CIFAR 图像，stem 使用 `3×3` 卷积、步长 1，不使用 ImageNet ResNet 开头的最大池化层；四个阶段通道数为 64、128、256、512，末端使用全局平均池化和全连接层。模型从随机权重开始训练，不使用预训练参数。

| `--model` 选项 | 相对基线的改动 |
|---|---|
| `resnet18` | 标准卷积、BatchNorm、ReLU，作为基线 |
| `resnet18_se` | 在残差块加入 SE 通道注意力 |
| `resnet18_dw` | 3×3 标准卷积换成 depthwise + pointwise 卷积 |
| `resnet18_gelu` | 把基线中的 ReLU 替换为 GELU |

除表中所列结构变化外，四组训练使用相同的数据划分、轮数、批大小、学习率和随机种子。GELU 是单独的激活函数对照，不与 SE 或深度可分离卷积组合。

## 3. 项目目录

```text
cnn-course-project/
├── train.py
├── evaluate_cifar10_test.py
├── compare_cifar10_1.py
├── requirements.txt
├── src/
│   ├── models.py
│   └── data.py
├── data/
│   ├── cifar-10-python.tar.gz
│   └── cifar10_1/
│       ├── cifar10.1_v6_data.npy
│       └── cifar10.1_v6_labels.npy
└── outputs/
```

## 4. 环境

项目已在 Windows、Python 3.14.6、PyTorch 2.15.0+cpu、torchvision 0.29.2+cpu 环境完成模型前向检查。CPU 环境安装命令如下：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision numpy --index-url https://download.pytorch.org/whl/cpu
$py = ".\.venv\Scripts\python.exe"
```

若已有可用虚拟环境，将 `$py` 设置为该环境的 `python.exe` 完整路径即可。

## 5. 数据准备

### CIFAR-10 训练集

下载地址https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz，将 `cifar-10-python.tar.gz` 放到项目的 `data/` 目录。首次构建训练集时，torchvision 会校验并解压该文件。数据应位于 `data/cifar-10-batches-py/`。官方压缩包 MD5 为 `c58f30108f718f92721af3b95e74349a`；可用下列命令核验：

```powershell
(Get-FileHash .\data\cifar-10-python.tar.gz -Algorithm MD5).Hash
```

### CIFAR-10.1 v6 场景数据

从 [CIFAR-10.1 官方仓库](https://github.com/modestyachts/CIFAR-10.1) 下载 `datasets` 目录中的 `cifar10.1_v6_data.npy` 和 `cifar10.1_v6_labels.npy`，放进 `data\cifar10_1\`。这两个 `.npy` 文件无需解压。目录应为：

```text
data/cifar10_1/cifar10.1_v6_data.npy
data/cifar10_1/cifar10.1_v6_labels.npy
```

## 6. 训练集/验证集划分

默认 `--val-fraction 0.1`，从 CIFAR-10 官方训练集的 50,000 张图像中按类别分层划分 10% 作验证集：每类 4,500 张用于训练、500 张用于验证，总计 45,000 张训练图像和 5,000 张验证图像。分层切分由 `--seed` 固定，四个模型使用相同索引。

训练集采用随机裁剪和水平翻转增强；验证集不做随机增强，只做张量转换和标准化。训练期间每轮计算训练和验证损失、准确率，并只根据验证准确率保存最佳 checkpoint。官方 CIFAR-10 测试集不被训练脚本加载。CIFAR-10.1 v6 只用于最后的场景对比。

## 7. 训练命令

在项目根目录打开 PowerShell，先做语法和命令行检查：

```powershell
& $py -m py_compile .\train.py .\evaluate_cifar10_test.py .\compare_cifar10_1.py .\src\models.py .\src\data.py
& $py .\train.py --help
```

可先运行 1 轮检查环境和数据读取，确认后再正式训练。正式对照使用 20 轮、批大小 64、随机种子 42、初始学习率 0.1：

```powershell
& $py .\train.py --model resnet18 --epochs 20 --batch-size 64 --seed 42
& $py .\train.py --model resnet18_se --epochs 20 --batch-size 64 --seed 42
& $py .\train.py --model resnet18_dw --epochs 20 --batch-size 64 --seed 42
& $py .\train.py --model resnet18_gelu --epochs 20 --batch-size 64 --seed 42
```

命令逐个运行。训练脚本默认 SGD、动量 0.9、权重衰减 0.0005、CosineAnnealingLR 和交叉熵损失。CPU 训练耗时较长。新流程会覆盖同名旧 checkpoint 和日志；请确认新 checkpoint 对应的 summary 中含有 `val_fraction`、`train_images` 和 `val_images` 字段。

## 8. 训练输出

每个模型在 `outputs/` 生成：

- `<model>_metrics.csv`：每轮训练/验证损失、训练/验证准确率、耗时和学习率。
- `<model>_summary.json`：训练设备、数据划分、参数量、MACs、最佳轮数和最佳验证准确率。
- `<model>_best.pt`：按验证准确率保存的最佳权重。

参数量统计可训练参数。MACs 通过 Conv2d 和 Linear 层的 hook 估算，不计 BatchNorm、激活、池化和数据加载开销；MACs 与 FLOPs 是不同口径。


## 9. CIFAR-10 官方测试集最终评估

模型结构、训练轮数和其他选择全部确定，四个训练任务完成后，再对 CIFAR-10 官方测试集进行一次最终评估。训练脚本不读取该测试集；最终评估脚本只计算损失、准确率、参数量和 MACs，不依据测试结果重新选择轮数、结构或 checkpoint。

在项目根目录运行：

```powershell
& $py .\evaluate_cifar10_test.py --device cpu --batch-size 64
```

输出为 `outputs\cifar10_final_test_summary.csv`。为 CIFAR-10 上的最终测试结果；如需再次调整模型，应另行保留新的测试集，不能继续把该测试结果用于选模。
## 10. 场景应用推理对比

四个模型训练完成并确认 checkpoint 来自新验证划分流程后，运行：

```powershell
& $py .\compare_cifar10_1.py --device cpu --batch-size 64
```

脚本从 `outputs/` 加载四个 `*_best.pt`，对CIFAR-10.1 v6 图像计算准确率、平均单张前向时间、吞吐量、参数量和 MACs，不进行训练或权重更新。输出文件为：

- `outputs/cifar10_1_scene_summary.csv`：每个模型一行汇总。
- `outputs/cifar10_1_scene_predictions.csv`：每张图像的真实类别、预测类别和正确与否。

推理时间只计模型前向，不含文件读取和预处理；实际速度依赖硬件、批大小和系统负载。



