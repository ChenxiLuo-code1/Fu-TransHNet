import numpy as np
import cv2
import os

# 数据集根目录
base_root = 'D:/Wyy_FuTransHNet/TrainDataset/TestDataset/'

# 数据集名称列表
data_names = ['CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB', 'Kvasir']

# 图像和掩码的相对路径
data_f = ['/images/']
mask_f = ['/masks/']

# 验证集大小
set_size = [380]

# 保存的文件名称
save_name = ['val']

# 图像和掩码统一调整的大小
height = 352
width = 352

# 定义保存路径的根目录
base_save_root = 'D:/Wyy_FuTransHNet/TestData'

# 遍历每个数据集
for data_name in data_names:
    print(f'Processing dataset: {data_name}...')

    # 数据集的根路径
    root = os.path.join(base_root, data_name)

    for j in range(len(data_f)):
        print('Processing ' + data_f[j] + '......')
        count = 0
        length = set_size[j]
        imgs = np.uint8(np.zeros([length, height, width, 3]))
        masks = np.uint8(np.zeros([length, height, width]))

        path = root + data_f[j]
        mask_p = root + mask_f[j]

        # 设置具体保存路径
        save_root = os.path.join(base_save_root, data_name, save_name[j])
        os.makedirs(save_root, exist_ok=True)  # 创建保存路径

        for i in os.listdir(path):
            if len(i.split('_')) == 2:
                img = cv2.imread(path + i)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (width, height))

                m_path = mask_p + i.replace('.jpg', '_segmentation.png')
                mask = cv2.imread(m_path, 0)
                mask = cv2.resize(mask, (width, height))

                imgs[count] = img
                masks[count] = mask

                count += 1
                print(f'Processed {count}/{length} images')

        # 保存处理后的数据到指定路径
        np.save('{}/data_{}.npy'.format(save_root, save_name[j]), imgs)
        np.save('{}/mask_{}.npy'.format(save_root, save_name[j]), masks)

    print(f'Finished processing dataset: {data_name}')
