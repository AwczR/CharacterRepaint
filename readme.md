# 古籍拓片去噪与修复扩散模型

本项目旨在使用 U-Net 架构的扩散模型（Denoising Diffusion Probabilistic Models, DDPM）对古籍拓片的扫描图像进行去噪和修复。

## 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [环境设置](#环境设置)
- [数据准备](#数据准备)
  - [数据集结构](#数据集结构-1)
  - [预处理脚本 (`create_metadata.py`)](#预处理脚本-create_metadatapy)
    - [使用方法](#使用方法)
    - [输出](#输出)
- [模型训练 (`train.py`)](#模型训练-trainpy)
  - [配置文件 (`config.yaml`)](#配置文件-configyaml)
  - [开始训练](#开始训练)
  - [训练过程](#训练过程)
- [结果与可视化](#结果与可视化)
- [后续工作与展望](#后续工作与展望)

## 项目概述

古籍拓片由于年代久远、保存条件等因素，常常存在噪声、破损、模糊等问题，影响其研究价值和观赏性。本项目利用深度学习中的扩散模型，通过学习从噪声图像恢复到清晰图像的逆过程，实现对拓片图像的自动去噪和修复。

模型主要基于 U-Net 结构，结合时间步嵌入（Sinusoidal Positional Timestep Embedding）和残差块（ResNet Blocks）来预测在扩散过程中每一步添加的噪声，从而在反向采样时逐步去除噪声。

## 项目结构

```
your_project_directory/
├── model.py                 # 定义 U-Net 扩散模型结构
├── create_metadata.py       # 数据集预处理脚本，生成元数据文件
├── train.py                 # 模型训练脚本
├── config.yaml              # 训练配置文件 (超参数、路径等)
├── dataset_metadata.json    # (示例) 由 create_metadata.py 生成
├── README.md                # 本文档
│
├── data/                    # (示例) 存放原始图像数据
│   ├── noisy_images/
│   │   ├── 001.png
│   │   └── ...
│   └── clean_images/
│       ├── 001.png
│       └── ...
│
└── training_results/        # (示例) 训练脚本的输出目录
    ├── checkpoints/
    │   └── model_epoch_XXX.pth
    └── visualizations/
        ├── loss_plot.png
        └── epoch_XXXX_samples.png
```

## 环境设置

在开始之前，请确保你已安装以下 Python 库：

- Python (建议 3.8+)
- PyTorch (建议 1.10+，并确保与你的 CUDA 版本兼容，如果使用 GPU)
- Torchvision
- Pillow (PIL)
- tqdm
- Matplotlib
- PyYAML (用于读取配置文件)
- NumPy

你可以使用 pip 安装它们：

```bash
pip install torch torchvision torchaudio pillow tqdm matplotlib pyyaml numpy
```

或者，如果你的 `requirements.txt` 文件已准备好，可以使用：
```bash
pip install -r requirements.txt
```
(你可能需要根据实际安装的包版本生成 `requirements.txt`)

## 数据准备

在开始训练去噪模型之前，需要对原始的有噪声和无噪声（清晰）的古籍拓片图像数据集进行预处理。

### 数据集结构

预处理脚本期望你的数据集遵循以下结构：

```
dataset_root/                 # 例如 ./data/
├── noisy_images/             # 存放有噪声的古籍拓片扫描图
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── clean_images/             # 存放对应的无噪声（清晰）的古籍拓片图
    ├── 001.png
    ├── 002.png
    └── ...
```

- 有噪声图像和无噪声图像应通过**相同的文件名**进行配对。
- 所有图像应为灰度图（单通道）。脚本默认会对图像进行校验（尺寸默认为 288x288 像素，模式为 'L'）。

### 预处理脚本 (`create_metadata.py`)

此脚本用于扫描图像文件夹、匹配文件对、可选地校验图像，并生成一个 JSON 格式的元数据文件。这个元数据文件将在后续的模型训练阶段被直接加载。

#### 使用方法

通过命令行运行脚本：

```bash
python create_metadata.py --noisy_dir <path_to_noisy_images> --clean_dir <path_to_clean_images> [options]
```

**参数说明：**

*   `--noisy_dir <path_to_noisy_images>`: **必需**。包含有噪声图像的文件夹路径。
*   `--clean_dir <path_to_clean_images>`: **必需**。包含无噪声（清晰）图像的文件夹路径。
*   `--output_file <filename.json>`: 可选。指定输出的元数据 JSON 文件的名称和路径。默认为 `dataset_metadata.json`。
*   `--skip_verification`: 可选。如果设置此标志，脚本将跳过对每个图像文件进行尺寸和模式的校验。

**示例：**

```bash
python create_metadata.py --noisy_dir ./data/noisy_images --clean_dir ./data/clean_images --output_file dataset_metadata.json
```

#### 输出

脚本成功运行后，会生成一个 JSON 格式的元数据文件（例如 `dataset_metadata.json`）。该文件包含了训练所需的所有图像路径信息，其结构大致如下：

```json
{
    "base_noisy_dir": "/full/path/to/your/noisy_images",
    "base_clean_dir": "/full/path/to/your/clean_images",
    "expected_size": [288, 288],
    "expected_mode": "L",
    "image_pairs": [
        {
            "filename": "001.png",
            "noisy_path_relative": "001.png",
            "clean_path_relative": "001.png"
        }
        // ... 更多图像对
    ],
    "num_valid_pairs": 1234
}
```

## 模型训练 (`train.py`)

`train.py` 脚本负责加载数据、初始化模型、并执行训练循环。

### 配置文件 (`config.yaml`)

所有训练相关的参数，如数据集路径、模型超参数、学习率、批大小、训练轮数等，都通过 `config.yaml` 文件进行管理。在开始训练前，请务必检查并根据你的需求修改此文件。

一个 `config.yaml` 的示例结构如下：

```yaml
dataset:
  metadata_path: "dataset_metadata.json" # create_metadata.py 的输出
  batch_size: 4
  num_workers: 2

model_params:
  image_size: 288
  in_channels: 1
  out_channels: 1
  model_channels: 64
  channel_mult: [1, 2, 4, 8]
  num_res_blocks: 2
  time_emb_dim_ratio: 4
  groups: 32

diffusion_params:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

training_params:
  epochs: 100
  lr: 0.0001
  output_dir: "training_results" # 结果保存路径
  save_checkpoint_freq: 10
  save_visualization_freq: 5
  device: "cuda" # "cuda" or "cpu"

visualization_params:
  num_samples_to_visualize: 4
  fixed_noise_for_samples: True
```

### 开始训练

确保你的 `dataset_metadata.json` (或你在 `config.yaml` 中指定的路径) 已经生成，并且 `config.yaml` 文件配置正确。然后运行：

```bash
python train.py --config config.yaml
```
或者，如果你的配置文件名为 `config.yaml` (默认值)，可以直接运行：
```bash
python train.py
```

### 训练过程

- **数据加载**: 训练脚本会根据 `config.yaml` 中的 `dataset.metadata_path` 加载元数据，并创建 PyTorch `Dataset` 和 `DataLoader`。
- **模型初始化**: 根据 `config.yaml` 中的 `model_params` 初始化 `UNetDenoisingModel`。
- **扩散过程**: 训练时，对每个干净图像 `x_0`，脚本会：
    1. 随机采样一个时间步 `t`。
    2. 生成高斯噪声 `ε`。
    3. 计算带噪图像 `x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε` (前向扩散)。
    4. 将 `x_t` 和 `t` 输入模型，模型预测噪声 `ε_θ(x_t, t)`。
    5. 计算损失 (通常是 `predicted_noise` 和真实添加的 `ε` 之间的 MSE Loss)。
- **优化**: 使用 AdamW 优化器更新模型参数。
- **进度显示**: 在命令行/终端，每个 epoch 内会显示一个 `tqdm` 进度条，实时更新已处理批次、当前批次损失、处理速度和预计剩余时间。

## 结果与可视化

训练过程中产生的结果会保存在 `config.yaml` 中 `training_params.output_dir` 指定的目录下 (默认为 `training_results/`)。

- **`checkpoints/`**: 存放模型检查点 (`.pth` 文件)。
    - 文件名格式: `model_epoch_XXX.pth`。
    - 包含模型权重、优化器状态、epoch 数和该 epoch 的损失。
    - 保存频率由 `save_checkpoint_freq` 控制。
- **`visualizations/`**: 存放可视化结果。
    - **`loss_plot.png`**: 训练损失随 epoch 变化的曲线图，会定期更新。
    - **`epoch_XXXX_samples.png`**: 使用当前 epoch 的模型从噪声生成的去噪样本图像，用于评估模型生成效果。生成频率由 `save_visualization_freq` 控制。
    - (可选) **`epoch_XXXX_denoising_process.gif`**: 如果在代码中启用了该功能，会保存去噪过程的GIF动画。

## 后续工作与展望

- **模型评估**: 使用独立的测试集对训练好的模型进行定量（如 PSNR, SSIM）和定性评估。
- **超参数调优**: 进一步调整学习率、模型大小、扩散步数等超参数以获得更优性能。
- **采样策略**: 尝试更高级的采样方法 (如 DDIM) 可能以更少的采样步数获得更好的生成质量。
- **条件扩散**: 如果有额外的条件信息（如拓片类型、破损程度等），可以探索条件扩散模型。
- **应用部署**: 将训练好的模型集成到实际的应用中，例如一个用户友好的拓片修复工具。

---
