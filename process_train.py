import os
import random
import shutil
import numpy as np
import cv2

# 定义路径
root_path = r"D:\Wyy_FuTransHNet\TrainDataset"
images_path = os.path.join(root_path, "image")
masks_path = os.path.join(root_path, "mask")
output_path = "D:\Wyy_FuTransHNet\TrainData"  # 输出文件夹保持在同一根路径下

# 定义保存路径
data_train_path = os.path.join(output_path, "data_train")
data_val_path = os.path.join(output_path, "data_val")
mask_train_path = os.path.join(output_path, "mask_train")
mask_val_path = os.path.join(output_path, "mask_val")

# 确保目标文件夹存在
os.makedirs(data_train_path, exist_ok=True)
os.makedirs(data_val_path, exist_ok=True)
os.makedirs(mask_train_path, exist_ok=True)
os.makedirs(mask_val_path, exist_ok=True)

# 获取所有文件名（假设 images 和 masks 目录下的文件名一致）
image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
mask_files = [f for f in os.listdir(masks_path) if f.endswith('.png')]

# 检查文件数量是否匹配
if len(image_files) != len(mask_files):
    raise ValueError("Images and masks file count do not match!")

# 打乱文件顺序以保证随机性
combined = list(zip(image_files, mask_files))
random.shuffle(combined)
image_files, mask_files = zip(*combined)

# 按照 8:2 划分数据集
split_index = int(0.8 * len(image_files))
train_image_files = image_files[:split_index]
val_image_files = image_files[split_index:]
train_mask_files = mask_files[:split_index]
val_mask_files = mask_files[split_index:]

# 图像和掩码统一调整的大小
height = 352
width = 352

# 将文件复制到相应的文件夹
def copy_files(src_folder, dest_folder, files):
    for file_name in files:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy(src_path, dest_path)

# 复制训练集和验证集
copy_files(images_path, data_train_path, train_image_files)
copy_files(images_path, data_val_path, val_image_files)
copy_files(masks_path, mask_train_path, train_mask_files)
copy_files(masks_path, mask_val_path, val_mask_files)

# 处理并保存为.npy文件
def process_and_save(image_files, mask_files, image_folder, mask_folder, save_root, save_name):
    count = 0
    length = len(image_files)
    imgs = np.uint8(np.zeros([length, height, width, 3]))  # 存储图像
    masks = np.uint8(np.zeros([length, height, width]))   # 存储掩码

    for i in range(length):
        # 加载图像
        img_path = os.path.join(image_folder, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))

        # 加载掩码
        mask_path = os.path.join(mask_folder, mask_files[i])
        mask = cv2.imread(mask_path, 0)  # 读取为灰度图
        mask = cv2.resize(mask, (width, height))

        # 存储处理后的图像和掩码
        imgs[count] = img
        masks[count] = mask

        count += 1
        print(f'Processed {count}/{length} images')

    # 保存处理后的数据到指定路径
    np.save(f'{save_root}/data_{save_name}.npy', imgs)
    np.save(f'{save_root}/mask_{save_name}.npy', masks)

# 保存路径
save_root_train = os.path.join(output_path, "TrainData")
os.makedirs(save_root_train, exist_ok=True)  # 创建保存训练集的文件夹

save_root_val = os.path.join(output_path, "ValData")
os.makedirs(save_root_val, exist_ok=True)  # 创建保存验证集的文件夹

# 处理并保存训练集和验证集
process_and_save(train_image_files, train_mask_files, data_train_path, mask_train_path, save_root_train, "train")
process_and_save(val_image_files, val_mask_files, data_val_path, mask_val_path, save_root_val, "val")

print("训练集和验证集处理及保存完成！")
