import os
import json
from PIL import Image # 用于图像校验
import argparse
from tqdm import tqdm # 用于显示进度条

def verify_image(image_path, expected_size=(288, 288), expected_mode="L"):
    """
    校验单个图像是否存在、是否能打开、尺寸和模式是否符合预期。
    """
    if not os.path.exists(image_path):
        print(f"警告: 图像文件不存在 {image_path}")
        return False
    try:
        img = Image.open(image_path)
        if img.size != expected_size:
            print(f"警告: 图像 {image_path} 尺寸不符，期望 {expected_size}，实际 {img.size}")
            return False
        if img.mode != expected_mode:
            # 有些灰度图可能是 '1' (binary), 'L' (luminance/grayscale), 'LA' (L with alpha)
            # 如果你的灰度图不是严格'L'，可以放宽条件，例如: if img.mode not in ['L', '1']:
            # 但通常我们会先统一转换为 'L'
            print(f"警告: 图像 {image_path} 模式不符，期望 {expected_mode}，实际 {img.mode}")
            # 你可以在这里尝试转换：img = img.convert(expected_mode) 然后再检查
            return False
        return True
    except Exception as e:
        print(f"警告: 打开或处理图像 {image_path} 失败: {e}")
        return False

def create_metadata(noisy_dir, clean_dir, metadata_output_file,
                    expected_size=(288, 288), expected_mode="L",
                    perform_verification=True):
    """
    扫描有噪声和无噪声图像文件夹，匹配文件对，并存储元数据。

    假设:
    - 有噪声和无噪声图像通过相同的文件名进行匹配。
    - 例如: noisy_dir/001.png 对应 clean_dir/001.png
    """
    noisy_files = set(os.listdir(noisy_dir))
    clean_files = set(os.listdir(clean_dir))

    matched_filenames = sorted(list(noisy_files.intersection(clean_files)))
    
    if not matched_filenames:
        print("错误: 在两个文件夹中没有找到任何匹配的文件名。请检查文件命名和路径。")
        return

    print(f"找到了 {len(matched_filenames)} 个匹配的文件名。")
    
    dataset_metadata = {
        "base_noisy_dir": os.path.abspath(noisy_dir),
        "base_clean_dir": os.path.abspath(clean_dir),
        "expected_size": list(expected_size),
        "expected_mode": expected_mode,
        "image_pairs": []
    }

    valid_pairs_count = 0
    for filename in tqdm(matched_filenames, desc="处理图像对"):
        noisy_path = os.path.join(noisy_dir, filename)
        clean_path = os.path.join(clean_dir, filename)

        # 可选的图像校验
        if perform_verification:
            is_noisy_valid = verify_image(noisy_path, expected_size, expected_mode)
            is_clean_valid = verify_image(clean_path, expected_size, expected_mode)
            if not (is_noisy_valid and is_clean_valid):
                print(f"跳过文件对: {filename} 因为图像校验失败。")
                continue
        
        dataset_metadata["image_pairs"].append({
            "filename": filename,
            "noisy_path_relative": filename, # 存储相对路径，配合base_dir使用
            "clean_path_relative": filename
        })
        valid_pairs_count += 1

    dataset_metadata["num_valid_pairs"] = valid_pairs_count
    print(f"处理完成。共找到 {valid_pairs_count} 对有效的图像。")

    # 检查是否有未匹配的文件
    unmatched_noisy = noisy_files - set(matched_filenames)
    if unmatched_noisy:
        print(f"\n警告: 在有噪声文件夹中找到 {len(unmatched_noisy)} 个未匹配的文件:")
        for f in list(unmatched_noisy)[:5]: print(f"  - {f}") # 最多显示5个
        if len(unmatched_noisy) > 5: print("  ...")

    unmatched_clean = clean_files - set(matched_filenames)
    if unmatched_clean:
        print(f"\n警告: 在无噪声文件夹中找到 {len(unmatched_clean)} 个未匹配的文件:")
        for f in list(unmatched_clean)[:5]: print(f"  - {f}")
        if len(unmatched_clean) > 5: print("  ...")

    # 保存元数据到JSON文件
    try:
        with open(metadata_output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, indent=4, ensure_ascii=False)
        print(f"\n元数据成功保存到: {metadata_output_file}")
    except IOError as e:
        print(f"错误: 无法写入元数据文件 {metadata_output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为古籍拓片去噪任务预处理数据集并生成元数据文件。")
    parser.add_argument("--noisy_dir", type=str, required=True, help="包含有噪声图像的文件夹路径。")
    parser.add_argument("--clean_dir", type=str, required=True, help="包含无噪声图像的文件夹路径。")
    parser.add_argument("--output_file", type=str, default="dataset_metadata.json", help="输出的元数据JSON文件名。")
    parser.add_argument("--skip_verification", action="store_true", help="跳过图像校验步骤（尺寸、模式）。")
    
    args = parser.parse_args()

    create_metadata(
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        metadata_output_file=args.output_file,
        perform_verification=not args.skip_verification
    )

    # 示例：如何加载元数据 (可以在训练脚本中使用)
    # with open(args.output_file, 'r', encoding='utf-8') as f:
    #     loaded_metadata = json.load(f)
    # print("\n加载的元数据示例 (前2对):")
    # print(f"基础有噪声路径: {loaded_metadata['base_noisy_dir']}")
    # print(f"基础无噪声路径: {loaded_metadata['base_clean_dir']}")
    # for pair in loaded_metadata['image_pairs'][:2]:
    #     full_noisy_path = os.path.join(loaded_metadata['base_noisy_dir'], pair['noisy_path_relative'])
    #     full_clean_path = os.path.join(loaded_metadata['base_clean_dir'], pair['clean_path_relative'])
    #     print(f"  - {pair['filename']}: {full_noisy_path} <-> {full_clean_path}")