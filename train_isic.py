import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.TransFuse import TransFuse_S
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np

import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def update_w(loss1, loss2, loss3, n, **kwargs):
    """
    计算每一次loss的视角权重
    """
    w1=(math.e**(-loss1/n))/((math.e**(-loss1/n))+(math.e**(-loss2/n))+(math.e**(-loss3/n)))
    w2=(math.e**(-loss2/n))/((math.e**(-loss1/n))+(math.e**(-loss2/n))+(math.e**(-loss3/n)))
    w3=(math.e**(-loss3/n))/((math.e**(-loss1/n))+(math.e**(-loss2/n))+(math.e**(-loss3/n)))

    return w1,w2,w3

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  #torch.abs()求绝对值

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))   #wbce=[20,1]

    pred = torch.sigmoid(pred)    #激活函数，值域范围[0,1],值为分类成目标类别的概率
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)  #wiou=[20,1]

    return (wbce + wiou).mean()     #.mean()返回20*1的平均值，是一个实数。


def train(train_loader, model, optimizer, epoch, best_loss):
    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        n=1
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).to("cuda")
        gts = Variable(gts).to("cuda")

        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)     #返回预测值   CNN预测、Transformer预测、融合后预测

        loss4 = structure_loss(lateral_map_4, gts)   #预测值和真实值计算损失
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)

        w1, w2, w3 = update_w(loss2,loss3,loss4,n)
        loss = w1*loss2+w2*loss3+w3*loss4+n*(w1*math.log(w1,math.e)+w2*math.log(w2,math.e)+w3*math.log(w3,math.e))

        # ---- backward ----
        loss.backward()    #反向传播，根据loss计算得到每个参数的梯度值
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()    #根据计算得到的参数梯度值对网络中的参数进行更新，起到降低loss值的作用
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-4: {:0.4f}], lateral-2: {:0.4f}], lateral-3: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record4.show(), loss_record2.show(), loss_record3.show()))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'n9_Trans%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'n9_Trans%d.pth' % epoch)
        file = open('D:/test/w_value/n9-value', 'a')
        file.write(str(w1.item()))
        file.write('\r\n')
        file.write(str(w2.item()))
        file.write('\r\n')
        file.write(str(w3.item()))
        file.write('\r\n')
        file.write(str("*********"))
        file.write('\r\n')

        file.close()

    return best_loss


def test(model, path):

    model.eval()
    mean_loss = []


    image_root = '{}/data_{}.npy'.format(path, 'val')
    gt_root = '{}/mask_{}.npy'.format(path,'val')
    test_loader = test_dataset(image_root, gt_root)

    dice_bank = []
    iou_bank = []
    loss_bank = []
    acc_bank = []

    for i in range(test_loader.size):
        image, gt = test_loader.load_data()
        image = image.cuda()

        with torch.no_grad():
             _, _, res = model(image)

        loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

        res = res.sigmoid().data.cpu().numpy().squeeze()
        gt = 1*(gt>0.5)
        res = 1*(res > 0.5)

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        loss_bank.append(loss.item())
        dice_bank.append(dice)
        iou_bank.append(iou)
        acc_bank.append(acc)


    print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
        format(path, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

    mean_loss.append(np.mean(loss_bank))

    return mean_loss[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='D:/Wyy_FuTransHNet/TrainData/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='D:/Wyy_FuTransHNet/TrainData/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='TransFuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer') #一阶矩估计的指数衰减率
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')#二阶矩估计的指数衰减率

    opt = parser.parse_args()

    # ---- build models ----

    model = TransFuse_S(pretrained=True).to("cuda")

    params = model.parameters()    #用来保存模型中的w和bias值
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
    image_root = '{}/data_train.npy'.format(opt.train_path)
    gt_root = '{}/mask_train.npy'.format(opt.train_path)


    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    total_step = len(train_loader)


    print("#"*20, "Start Training", "#"*20)

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_loss)
