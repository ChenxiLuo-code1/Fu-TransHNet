import numpy as np
import cv2
import os

root = 'D:/wyy/polyp/TransFuse/TestData/CVC-EndoSceneStill/' # change to your data folder path
data_f = ['CVC-362images/']
mask_f = ['CVC-362masks/']
set_size = [362]
save_name = ['CVC-362']

height = 352
width = 352

for j in range(1):

	print('processing ' + data_f[j] + '......')
	count = 0
	length = set_size[j]
	imgs = np.uint8(np.zeros([length, height, width, 3]))  #创建长宽高，3通道的四维全零数组，像素大小为8位无符号整数
	masks = np.uint8(np.zeros([length, height, width]))

	path = root + data_f[j]
	mask_p = root + mask_f[j]

	for i in os.listdir(path):
		#print(i)
		#if len(i.split('_'))==2:    #data.split()通过指定分隔符对字符串进行切片
		img = cv2.imread(path+i)
		#print(img.shape)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (width, height))   #重置图像大小

		#m_path = mask_p + i.replace('.png', '_segmentation.png')
		#mask = cv2.imread(m_path, 0)
		mask = cv2.imread(mask_p+i, 0)

		mask = cv2.resize(mask, (width, height))

		imgs[count] = img
		masks[count] = mask

		count +=1
		print(count)


	np.save('{}/data_{}.npy'.format(root, save_name[j]), imgs)
	np.save('{}/mask_{}.npy'.format(root, save_name[j]), masks)
