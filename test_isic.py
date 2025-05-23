import torch
import numpy as np
import os, argparse

from lib.TransFuse import TransFuse_S
from utils.dataloader import test_dataset
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str,
                        default='D:/Wyy_FuTransHNet/results/n3_Trans49.pth')
    parser.add_argument('--test_path', type=str,
                        default='/home/415/wyy/polyp/TestData', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='D:/Wyy_FuTransHNet/results',
                        help='path to save inference segmentation')

    opt = parser.parse_args()

    model = TransFuse_S(pretrained=True).cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    # print('evaluating model: ', opt.ckpt_path)
    j = 0
for _data_name in ['CVC-ClinicDB', 'CVC-ColonDB', 'CVC-T', 'ETIS-LaribPolypDB', 'Kvasir']:
    # CVC-ClinicDB  CVC-ColonDB	CVC-T ETIS-LaribPolypDB	Kvasir

    image_root = '/home/415/wyy/polyp/TestData/data_{}.npy'.format(_data_name)
    gt_root = 'D:/Wyy_FuTransHNet/mask_{}.npy'.format(_data_name)
    pred_save_path = 'D:/Wyy_FuTransHNet/results/pred/DSS/{}/'.format(_data_name)
    gt_save_path = 'D:/Wyy_FuTransHNet/results/gt/{}/'.format(_data_name)

    test_loader = test_dataset(image_root, gt_root)
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(gt_save_path, exist_ok=True)
    dice_bank = []
    iou_bank = []

    b = 0.0
    for i in range(test_loader.size):
        image, gt = test_loader.load_data()


        gt = 1 * (gt > 0.5)
        image = image.cuda()

        with torch.no_grad():
            res1, res2, res3 = model(image)

        w1 = 0.34230294885896795
        w2 = 0.3268978957442428
        w3 = 0.3307991553967893



        res =w1* res3 +w2* res2 +w3* res1

        res = res.sigmoid().data.cpu().numpy().squeeze() 
        res = 1 * (res > 0.5)
        if pred_save_path and gt_save_path is not None:
            imageio.imwrite(pred_save_path + '/' + str(j) + '.png', res)
            imageio.imwrite(gt_save_path+'/'+str(j)+'.png', gt)
            j=j+1

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)

        dice_bank.append(dice)
        iou_bank.append(iou)

    print('Dice: {:.4f}, IoU: {:.4f}'.
          format(np.mean(dice_bank), np.mean(iou_bank)))
