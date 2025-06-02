好的，这是一个针对预处理部分的 `README.md` 文件内容：

```markdown
# 古籍拓片去噪模型 - 数据预处理

本项目旨在使用扩散模型对有噪声的古籍拓片扫描图进行去噪。此 `README.md` 文档的这一部分详细介绍了数据预处理步骤。

## 概述

在开始训练去噪模型之前，需要对原始的有噪声和无噪声（清晰）的古籍拓片图像数据集进行预处理。此步骤主要包括：

1.  **扫描图像文件夹**：定位有噪声和对应的无噪声图像。
2.  **文件匹配**：基于文件名将有噪声图像与其对应的无噪声清晰图像配对。
3.  **（可选）图像校验**：检查图像文件是否存在、是否可以正常打开、图像尺寸是否符合预期（例如 288x288 像素）、图像模式是否为灰度图（'L'模式）。
4.  **生成元数据文件**：将所有有效的图像对信息（如文件路径）以及数据集的整体信息（如基础路径、图像尺寸）存储在一个 JSON 文件中。这个元数据文件将在后续的模型训练阶段被 PyTorch `Dataset` 类直接加载，以提高数据加载效率并确保数据的一致性。

## 先决条件

在运行预处理脚本之前，请确保你已安装以下 Python 库：

*   **Pillow (PIL)**: 用于图像文件的打开和基本处理。
*   **tqdm**: 用于在处理文件时显示美观的进度条。

你可以使用 pip 安装它们：

```bash
pip install Pillow tqdm
```

## 数据集结构

预处理脚本期望你的数据集遵循以下结构：

```
dataset_root/
├── noisy_images/          # 存放有噪声的古籍拓片扫描图
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── clean_images/          # 存放对应的无噪声（清晰）的古籍拓片图
    ├── 001.png
    ├── 002.png
    └── ...
```

-   有噪声图像和无噪声图像应通过**相同的文件名**进行配对（例如，`noisy_images/001.png` 对应 `clean_images/001.png`）。
-   所有图像应为灰度图（单通道）。
-   建议图像尺寸统一，例如本项目默认的 288x288 像素。

## 预处理脚本

预处理脚本为 `preprocess_dataset.py`。

### 使用方法

通过命令行运行脚本：

```bash
python preprocess_dataset.py --noisy_dir <path_to_noisy_images> --clean_dir <path_to_clean_images> [options]
```

**参数说明：**

*   `--noisy_dir <path_to_noisy_images>`: **必需**。包含有噪声图像的文件夹路径。
*   `--clean_dir <path_to_clean_images>`: **必需**。包含无噪声（清晰）图像的文件夹路径。
*   `--output_file <filename.json>`: 可选。指定输出的元数据 JSON 文件的名称和路径。默认为 `dataset_metadata.json`。
*   `--skip_verification`: 可选。如果设置此标志，脚本将跳过对每个图像文件进行尺寸和模式的校验。这会加快预处理速度，但如果图像不符合预期，可能会在训练阶段导致错误。

### 示例

```bash
python preprocess_dataset.py --noisy_dir ./data/rubbings_noisy --clean_dir ./data/rubbings_clean --output_file ./metadata/rubbings_train_metadata.json
```

这条命令会：
1. 扫描 `./data/rubbings_noisy` 和 `./data/rubbings_clean` 文件夹。
2. 匹配文件名为 `*.png` 的图像对。
3. 对每张图像进行校验（确保是 288x288 的灰度图）。
4. 将有效的图像对信息和数据集元数据保存到 `./metadata/rubbings_train_metadata.json` 文件中。

### 输出

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
        },
        // ... 更多图像对
    ],
    "num_valid_pairs": 1234
}
```

这个元数据文件将在后续的训练脚本中被 `Dataset` 类直接引用。

## 注意事项

-   如果你的原始图像数据发生变动（例如，添加、删除或修改了图像文件），你需要重新运行预处理脚本以更新元数据文件，确保训练时使用最新的数据信息。
-   脚本会报告在有噪声和无噪声文件夹中找到的任何未匹配的文件，这有助于你检查数据集的完整性。
-   图像校验过程（尺寸、模式）有助于在训练开始前发现潜在的数据问题。如果你的图像已经过严格处理并符合要求，可以使用 `--skip_verification` 选项以节省时间。

---
(后续部分将介绍模型训练、评估和推断等内容)
```

你可以将以上内容复制粘贴到你的 `README.md` 文件中。它为其他用户（或未来的你）提供了清晰的指导，说明了如何准备用于训练模型的数据集。# CharacterRepaint
