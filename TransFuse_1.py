import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
from lib.DeiT import  deit_small_patch16_224 as deit
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.HarDMSEG import HarDMSEG






class RCFM(nn.Module):
    def __init__(self, channel):
        super(RCFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.transform1=BasicConv2d(128, channel, 3, padding=1)
        self.transform2=BasicConv2d(64, channel, 3, padding=1)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3,padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3,padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):           #x1=[20,256,22,22],x2=[20,256,44,44],x3=[20,256,88,88]
        x1_1 = x1
        x2=self.transform1(x2)
        x3=self.transform2(x3)

        x2_1 = self.conv_upsample1(self.upsample(x1)) + x2    #[16,256,44,44]
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) + x3
        #x3_1=self.upsample(x2_1)+x3
        #print(x3_1.shape)
        #x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1)))* self.conv_upsample3(self.upsample(x2)) * x3       #[16,256,88,88]

        x2_1=self.conv_upsample1(x2_1)
        x3_1=self.conv_upsample1(x3_1)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)   #[16,512,44,44]

        x2_2 = self.conv_concat2(x2_2)      #[16,512,44,44]
        #print(x2_2.shape)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)   #[16,768,88,88]
        # print("*****")
        # print(x3_2.shape)
        x3_2 = self.conv_concat3(x3_2)    #[16,768,88,88]


        x1 = self.conv4(x3_2)         #[16,256,88,88]
        #print(x1.shape)
        return x1


def conv1x1(in_channels, out_channels, stride=1):
    ''' 1x1 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    ''' 3x3 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                     bias=False)


def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    ''' 7x7 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, dilation=dilation,
                     bias=False)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio, dilation=1):
        super(CBAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = conv7x7(2, 1, stride=1, dilation=self.dilation)

    def forward(self, x):
        ''' Channel attention '''
        avgOut = self.globalAvgPool(x)
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        maxOut = self.globalMaxPool(x)
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)
        # sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        Mc = self.sigmoid(avgOut + maxOut)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)
        Mf1 = Mc * x

        ''' Spatial attention. '''
        # sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        maxOut = torch.max(Mf1, 1)[0].unsqueeze(1)
        avgOut = torch.mean(Mf1, 1).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        Ms = self.conv1(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2), Ms.size(3))
        Mf2 = Ms * Mf1
        return Mf2


class NewBifusion(nn.Module):
    def __init__(self, ch_1, ch_2, ch_out, drop_rate=0.):
        super(NewBifusion, self).__init__()

        # channel attention for t, use cbam
        self.convt = Conv(ch_2, ch_out, 1, padding=0, bn=True, relu=False)  # 变成相同通道
        self.cbam1 = CBAM(ch_out, 16)

        # spatial attention for g
        self.cbam2 = CBAM(ch_out, 16)

        # bi-linear modelling for both
        self.conv11 = Conv(ch_1 * 2, ch_1, 1, padding=0, bn=False, relu=True)
        self.conv12 = Conv(ch_1, ch_1 // 2, 3, padding=1, bn=False, relu=True)
        self.conv13 = Conv(ch_1 // 2, 2, 3, padding=1, bn=False, relu=False)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        x = self.convt(x)
        bp = torch.cat((g, x), dim=1)
        bp = self.conv11(bp)   #[16,256,22,22]

        bp = self.conv12(bp)   #[16,128,22,22]

        att = self.conv13(bp)  #[16,2,22,22]

        att = F.softmax(att, dim=1)  #[16,2,22,22]
        #print("***")
        att_1 = att[:, 0, :, :].unsqueeze(1)
        #print(att_1.shape)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        #print(att_2.shape)
        g = self.cbam2(g)

        x = self.cbam1(x)
        # print(x.shape)
        # print((att_1*g).shape)
        # print((att_2 * x).shape)
        fuse = att_1 * g + att_2 * x
        #print(fuse.shape)
        fuse = self.residual(fuse)
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)  # dim表示按列连接


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)  # [16,256,22,22]
        W_x = self.W_x(x)  # [16,256,22,22]
        bp = self.W(W_g * W_x)  # [16,256,22,22]

        # print("#")
        # print(bp.shape)  #bp.shape=[16,256,22,22],[16,256,44,44],[16,256,88,88]

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)  # [16,2,22,22]
        g = self.spatial(g)  # [16,1,22,22]
        g = self.sigmoid(g) * g_in  # [16,256,22,22]
        # print("##")
        # print(g.shape)   #g.shape=[16,256,22,22],[16,256,44,44],[16,256,88,88]

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)  # [16,384,1,1]
        x = self.fc1(x)  # [16,64,1,1]
        x = self.relu(x)  # [16,64,1,1]
        x = self.fc2(x)  # [16,384,1,1]
        x = self.sigmoid(x) * x_in  # [16,384,22,22]
        fuse = self.residual(torch.cat([g, x, bp], 1))
        # print("###")
        # print(fuse.shape)   #fuse.shape=[16,256,22,22],[16,256,44,44],[16,256,88,88]

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class TransFuse_S(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_S, self).__init__()

        self.net = HarDMSEG()
        if pretrained:
            self.net.load_state_dict(torch.load('D:/wyy/HarDNet-MSEG-master/HarDNet-MSEG-master/snapshots/HarD-MSEG-best/HarDNet-MSEG-best.pth'))
        self.net.fc = nn.Identity()
        self.net.layer4 = nn.Identity()
        # self.net.bn1 = nn.BatchNorm2d()
        self.transformer = deit(pretrained=pretrained)

        self.CFM = RCFM(256)

        # self.Translayer2_0 = BasicConv2d(384, 64, 1)
        # self.Translayer2_1 = BasicConv2d(128, 64, 1)
        # self.Translayer3_1 = BasicConv2d(64, 64, 1)

        # 在输入到CFM中先对其进行通道转换，转换为256
        # self.f_0 = BasicConv2d(256, 256, 1)
        # self.f_1 = BasicConv2d(128, 256, 1)
        # self.f_2 = BasicConv2d(64, 256, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)
        self.up3 = Up(1, 64)
        self.up4 = Up(64, 128)
        self.up5 = Up(128, 256)
        # self.rfbt_1 = RFB_modified(384, 256)

        self.t_2 = nn.Sequential(
            Conv(64, 64, kernel_size=3, stride=2, padding=2, dilation=2, bias=True, bn=True, relu=True),
            Conv(64, 64, kernel_size=3, stride=1, padding=1, bias=True, bn=True, relu=True)

        )
        self.t_1 = BatchRelu(128)

        self.t_0 = nn.Sequential(
            nn.ConvTranspose2d(384, 384, 2, stride=2, padding=0, output_padding=0, bias=True),
            Conv(384, 384, kernel_size=3, stride=1, padding=1, bias=True, bn=True, relu=True)

        )

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),              #nn.relu()：将小于0的置为0，大于0的不变
            Conv(64, num_classes, 3, bn=False, relu=False)    #因为已经变为单通道了，不需要在进行nn.relu()激活函数了
        )

        self.final_2 = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256,
                                   drop_rate=drop_rate / 2)  # [16,256,12,16]+[16,384,12,16]->[16,256,12,16]

        self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128,
                                       drop_rate=drop_rate / 2)  # [16,128,24,32]+[16,128,24,32]->[16,128,24,32]
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64,
                                       drop_rate=drop_rate / 2)  # [16,64,48,64]+[16,64,48,64]->[16,64,48,64]
        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.fusion1 = NewBifusion(ch_1=256, ch_2=384, ch_out=256, drop_rate=drop_rate / 2)

        self.fusion2 = NewBifusion(ch_1=128, ch_2=128, ch_out=128, drop_rate=drop_rate / 2)

        self.fusion3 = NewBifusion(ch_1=64, ch_2=64, ch_out=64, drop_rate=drop_rate / 2)



        self.drop =nn.Dropout2d(drop_rate)  # Dropout2d的赋值对象是彩色图像数据，即输入为（N,C,H,W）对每一个通道维度C按概率赋值为0

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        # print("########")                  #此时传进来的imgs.size()=[batchsize,channels,height,width]   [16,3,352,352]
        # print(imgs.size())
        x_b = self.transformer(imgs)  # 此时x_b.size()=[16,192,384]   经过transformer4维变量变成3维向量   z_0 [16,484,384]
        x_b = torch.transpose(x_b, 1, 2)  # 此时x_b.size()=[16,384,192]  torch.transpose()是用来交换维数的  [16,384,484]
        x_b = x_b.view(x_b.shape[0], -1, 22, 22)  # 此时x_b.size()=[16,384,12,16]      [16,384,22,22]
        x_b = self.drop(x_b)  # 此时   t_0=[16,384,22,22]

        # x_b=self.rfbt_1(x_b)     #此时t_0=[16,256,22,22]

        x_b_1 = self.up1(x_b)  # 此时x_b_1.size()=[16,128,24,32]  [16,128,44,44]
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # 此时x_b_2.size()=[16,64,48,64]     [16,64,88,88]



        # x_u = self.net.conv1(imgs)   #此时x_u.size()=[16,64,96,128]
        # x_u = self.net(imgs)  # x_u.shape=[16,1,352,352]
        #
        # x_u = F.interpolate(x_u, scale_factor=0.125)  # [16,1,8,8]    [16,1 44,44]
        #
        #
        # x_u_2 = self.up3(x_u)
        #
        # x_u_2 = self.drop(x_u_2)  # 此时x_u.size()=[16,64,48,64] [16,64,88,88]
        #
        # x_u_1 = self.up4(x_u_2)
        # x_u_1 = F.interpolate(x_u_1, scale_factor=0.25)
        # x_u_1 = self.drop(x_u_1)  # 此时x_u.size()=[16,128,24,32]  [16,128,44,44]
        # # print('******')
        # # print(x_u_1.shape)
        # x_u = self.up5(x_u_1)
        # x_u = F.interpolate(x_u, scale_factor=0.25)
        # x_u = self.drop(x_u)  # 此时x_u.size()=[16,256,12,16]    [16,256,22,22]
        # # print("########")
        # # print(x_u.size())
        #
        # # joint path
        # x_c = self.fusion1(x_u, x_b)  # x_u,x_b进行融合 f0[16,256,22,22]
        # # print("########")
        # # print(x_c.size())
        #
        # x_c_1_1 = self.fusion2(x_u_1, x_b_1)  # x_u_1,x_b_1进行融合    此时x_c_1_1.size()=[16,128,44,44] f1
        #
        #
        # x_c_2_1 = self.fusion3(x_u_2, x_b_2)  # x_u_1,x_b_1进行融合    此时x_c_2_1.size()=[16,64,88,88] f2
        #
        # # 把每次融合的结果再次融合起来
        #
        # F_all = self.CFM(x_c, x_c_1_1, x_c_2_1)


        # decoder part
        #map_x = F.interpolate(self.final_x(x_u), scale_factor=16, mode='bilinear')  # CNN融合后的预测值  [16,256,22,22]
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear')  # Transformer branch 的预测值
        #map_2 = F.interpolate(self.final_2(F_all), scale_factor=4, mode='bilinear')  #融合branch的预测值
        return  map_1

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        # self.se1 = SEBlock(F_g)
        # self.se2 = SEBlock(F_l)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g1 = self.se1(g)
        # x1 = self.se2(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1, bn=False, relu=True,
                 bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BatchRelu(nn.Module):
    def __init__(self, out_channels):
        super(BatchRelu, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)

        x = self.relu(x)
        return x


#