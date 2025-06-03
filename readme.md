您好，我想您可能是想让我输出更新后的 `README.md` 文件内容，而不是 `readme.py`。`.md` 是 Markdown 文件的扩展名，通常用于编写文档，比如 README。

根据我们之前讨论的 `inference.py` 的修改（加入了对输入图像预先叠加高斯噪声的选项），我会更新 `README.md` 中关于 `inference.py` 使用说明的部分。

以下是更新后的完整 `README.md` 内容：

```markdown
# 古籍拓片去噪与修复扩散模型

本项目旨在使用 U-Net 架构的扩散模型（Denoising Diffusion Probabilistic Models, DDPM）对古籍拓片的扫描图像进行去噪和修复。

## 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [环境设置](#环境设置)
- [数据准备](#数据准备)
  - [数据集结构](#数据集结构-1)
  - [预处理脚本 (`create_metadata.py`)](#预处理脚本-create_metadatapy)
- [模型训练 (`train.py`)](#模型训练-trainpy)
  - [配置文件 (`config.yaml`)](#配置文件-configyaml)
  - [学习率调度器](#学习率调度器)
  - [开始训练](#开始训练)
  - [训练过程](#训练过程)
- [结果与可视化](#结果与可视化)
- [模型推理 (`inference.py`)](#模型推理-inferencepy)
  - [推理方法说明 (SDEdit)](#推理方法说明-sdedit)
  - [可选：对输入图像预加高斯噪声](#可选对输入图像预加高斯噪声)
  - [使用方法](#使用方法-1)
- [后续工作与展望](#后续工作与展望)

## 项目概述

古籍拓片由于年代久远、保存条件等因素，常常存在噪声、破损、模糊等问题，影响其研究价值和观赏性。本项目利用深度学习中的扩散模型，通过学习从噪声图像恢复到清晰图像的逆过程，实现对拓片图像的自动去噪和修复。

模型主要基于 U-Net 结构，结合时间步嵌入（Sinusoidal Positional Timestep Embedding）和残差块（ResNet Blocks）来预测在扩散过程中每一步添加的噪声，从而在反向采样时逐步去除噪声。

## 项目结构

```
your_project_directory/
├── model.py                 # 定义 U-Net 扩散模型结构
├── create_metadata.py       # 数据集预处理脚本，生成元数据文件
├── train.py                 # 模型训练脚本 (支持学习率调度器)
├── inference.py             # 模型推理脚本 (使用SDEdit方法，支持对输入预加噪声)
├── config.yaml              # 训练配置文件 (超参数、路径、学习率调度器等)
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

此脚本用于扫描图像文件夹、匹配文件对、可选地校验图像，并生成一个 JSON 格式的元数据文件。这个元数据文件将在后续的模型训练阶段被直接加载。具体使用方法请参考该脚本内部的注释或运行 `python create_metadata.py --help`。

**基本使用示例：**

```bash
python create_metadata.py --noisy_dir ./data/noisy_images --clean_dir ./data/clean_images --output_file dataset_metadata.json
```

脚本成功运行后，会生成一个 JSON 格式的元数据文件，包含训练所需的所有图像路径信息。

## 模型训练 (`train.py`)

`train.py` 脚本负责加载数据、初始化模型、并执行训练循环，现在支持学习率调度器。

### 配置文件 (`config.yaml`)

所有训练相关的参数，如数据集路径、模型超参数、学习率、批大小、训练轮数以及学习率调度器配置，都通过 `config.yaml` 文件进行管理。在开始训练前，请务必检查并根据你的需求修改此文件。

一个 `config.yaml` 的示例结构（包含学习率调度器）如下：

```yaml
dataset:
  metadata_path: "/openbayes/home/CharacterRepaint/scripts/dataset_metadata.json"
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
  scheduler: # 可选的学习率调度器配置
    name: "StepLR"      # 支持 "StepLR", "Cosine", "Exponential"
    step_size: 30       # (用于 StepLR) 每 step_size 个 epoch 将 LR 乘以 gamma
    gamma: 0.5          # (用于 StepLR, ExponentialLR) 衰减倍数
    # t_max: 100        # (用于 CosineAnnealingLR) 余弦周期的最大迭代次数
    # eta_min: 0        # (用于 CosineAnnealingLR) 学习率的最小值
  output_dir: "training_results/experiment_001" # 结果保存路径
  save_checkpoint_freq: 10
  save_visualization_freq: 5
  device: "cuda"

visualization_params:
  num_samples_to_visualize: 4
  fixed_noise_for_samples: True
```

### 学习率调度器

`train.py` 现在支持通过 `config.yaml` 文件配置学习率调度器。在 `training_params` 下添加 `scheduler` 字段即可启用。

支持的调度器类型 (`name` 字段)：
- `StepLR`: 每隔 `step_size` 个 epoch，学习率乘以 `gamma`。
- `CosineAnnealingLR`: 使用余弦退火调整学习率。需要参数 `t_max` (通常设为总 epochs) 和 `eta_min`。
- `ExponentialLR`: 每个 epoch 后，学习率乘以 `gamma`。

如果 `training_params` 中没有 `scheduler` 字段，则不使用学习率调度器。

### 开始训练

确保你的元数据文件 (例如 `dataset_metadata.json`) 已经生成，并且 `config.yaml` 文件配置正确。然后运行：

```bash
python train.py --config config.yaml
```
或者，如果你的配置文件名为 `config.yaml` (默认值)，可以直接运行：
```bash
python train.py
```

### 训练过程

- **数据加载、模型初始化、扩散过程**：与之前版本类似。
- **优化**: 使用 AdamW 优化器。如果配置了学习率调度器，则在每个 epoch 结束后调用 `scheduler.step()` 更新学习率。
- **进度显示**: 在命令行/终端，每个 epoch 内会显示一个 `tqdm` 进度条，实时更新已处理批次、当前批次损失。每个 epoch 结束后会打印平均损失和当前学习率。

## 结果与可视化

训练过程中产生的结果会保存在 `config.yaml` 中 `training_params.output_dir` 指定的目录下。

- **`checkpoints/`**: 存放模型检查点 (`.pth` 文件)。
    - 文件名格式: `model_epoch_XXX.pth`。
    - 包含模型权重、优化器状态、(若使用)学习率调度器状态、epoch 数和该 epoch 的损失。
- **`visualizations/`**: 存放可视化结果。
    - **`loss_plot.png`**: 训练损失随 epoch 变化的曲线图。
    - **`epoch_XXXX_samples.png`**: 使用当前 epoch 的模型从噪声生成的去噪样本图像。

## 模型推理 (`inference.py`)

训练完成后，可以使用 `inference.py` 脚本对单张有噪声的图像进行去噪处理。该脚本使用了一种称为 SDEdit (Stochastic Denoising Editing) 的方法，并支持对输入图像预先叠加一层随机高斯噪声。

### 推理方法说明 (SDEdit)

标准的扩散模型采样是从纯随机噪声开始生成图像。而 SDEdit 方法允许我们基于一张已有的输入图像（在这里是你的带噪拓片）进行生成/修复。其基本步骤是：

1.  **(可选) 预加高斯噪声**: 对输入的原始带噪图像，可以先额外叠加一层可控强度的随机高斯噪声。
2.  **前向加噪 (SDEdit 核心)**: 对（可能已预加噪的）图像，执行一小部分（由 `sde_strength` 参数控制）的前向扩散步骤，使其变得更模糊，更接近高斯噪声。
3.  **反向去噪**: 然后，从这个加噪后的图像和对应的时间步开始，执行扩散模型的标准反向采样过程，逐步去噪，直到生成最终的修复图像。

`sde_strength` 参数控制了对输入图像的“信任度”与模型“创造性修复”之间的平衡。预加高斯噪声则引入了更多的随机性和潜在的生成多样性。

### 可选：对输入图像预加高斯噪声

通过命令行参数，你可以在 SDEdit 流程开始前，为输入的原始带噪图像额外添加一层高斯噪声。这可以：
- 引入更多的随机性，使得对于同一输入，每次运行的结果可能略有不同。
- 鼓励模型进行更大胆的重构，可能有助于处理某些类型的顽固噪声或大面积破损。
- 需要通过实验调整 `--initial_noise_std` 参数来控制预加噪声的强度。

### 使用方法

通过命令行运行推理脚本：

```bash
python inference.py \
    --config_path <path_to_training_config.yaml> \
    --checkpoint_path <path_to_model_checkpoint.pth> \
    --input_image_path <path_to_your_noisy_image.png> \
    [options]
```

**必需参数：**

*   `--config_path <path_to_training_config.yaml>`: **必需**。指向你训练模型时使用的 `config.yaml` 文件。
*   `--checkpoint_path <path_to_model_checkpoint.pth>`: **必需**。指向训练好的模型检查点文件。
*   `--input_image_path <path_to_your_noisy_image.png>`: **必需**。你想要进行去噪处理的原始带噪图像的路径。

**可选参数：**

*   `--output_image_path <path_to_save_output.png>`: 可选。去噪后图像的保存路径。默认为 `denoised_output.png`。
*   `--sde_strength <integer>`: 可选。SDEdit 的强度参数，表示在（可能已预加噪的）输入图像上先进行多少步前向加噪。**这是一个关键参数，需要根据效果进行调整。** 默认值为 `250`。该值必须小于 `config.yaml` 中定义的 `diffusion_params.num_timesteps` (通常是1000)，且应为非负数。
    *   **较小的值** (例如 100-200): 结果更忠实于原始输入图像的结构，但去噪可能不彻底。
    *   **较大的值** (例如 300-500+): 去噪效果可能更强，但可能会更多地依赖模型“幻想”内容，细节可能与原图有偏差。
    *   如果为 `0`，则不执行 SDEdit 的前向加噪步骤，直接从（可能已预加噪的）输入图像开始（反向采样循环将不运行，结果是输入或预加噪后的输入）。
*   `--device <cuda|cpu>`: 可选。指定推理设备。默认为 `cuda` (如果可用)。
*   `--add_initial_gaussian`: 可选。布尔标志。如果设置，则在 SDEdit 流程开始前，对输入图像预先叠加一层高斯噪声。
*   `--initial_noise_std <float>`: 可选。如果设置了 `--add_initial_gaussian`，此参数定义了预加高斯噪声的标准差。默认为 `0.05`。可以尝试 `0.01` 到 `0.2` 范围内的值。

**示例：**

1.  **基本 SDEdit 推理：**
    ```bash
    python inference.py \
        --config_path ./config.yaml \
        --checkpoint_path ./training_results/experiment_001/checkpoints/model_epoch_100.pth \
        --input_image_path ./data/noisy_sample.png \
        --output_image_path ./denoised_sample_sde200.png \
        --sde_strength 200
    ```

2.  **SDEdit 推理，并对输入图像预加高斯噪声：**
    ```bash
    python inference.py \
        --config_path ./config.yaml \
        --checkpoint_path ./training_results/experiment_001/checkpoints/model_epoch_100.pth \
        --input_image_path ./data/noisy_sample.png \
        --output_image_path ./denoised_sample_sde200_initnoise008.png \
        --sde_strength 200 \
        --add_initial_gaussian \
        --initial_noise_std 0.08
    ```
脚本执行后，会显示原始输入图像和去噪后的图像，并将去噪结果保存到指定路径。

## 后续工作与展望

- **模型评估**: 使用独立的测试集对训练好的模型进行定量（如 PSNR, SSIM）和定性评估。
- **超参数调优**: 进一步调整学习率、模型大小、扩散步数、SDEdit强度、预加噪声强度等超参数。
- **采样策略**: 探索更高级的采样方法 (如 DDIM) 可能以更少的采样步数获得更好的生成质量或更快的推理速度。
- **条件扩散**: 如果有额外的条件信息，可以探索条件扩散模型。
- **应用部署**: 将训练好的模型集成到实际的应用中。

---
```

**这次更新的主要内容集中在 `模型推理 (inference.py)` 章节：**

1.  **项目结构图**：更新了 `inference.py` 的描述，提到支持对输入预加噪声。
2.  **推理方法说明 (SDEdit)**：增加了对可选的“预加高斯噪声”步骤的描述。
3.  **新增章节：“可选：对输入图像预加高斯噪声”**：详细解释了这个新功能的含义和作用。
4.  **使用方法 -> 可选参数**：
    *   添加了 `--add_initial_gaussian` 和 `--initial_noise_std` 两个新参数的说明。
    *   稍微调整了对 `--sde_strength` 为 `0` 时的行为描述，使其更准确。
5.  **使用方法 -> 示例**：增加了一个使用预加高斯噪声功能的命令行示例。

希望这个版本的 README 更能满足你的需求！
