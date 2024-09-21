from src.model import common
import torch
import torch.nn as nn


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self,
                 conv,
                 n_feat,
                 kernel_size,
                 reduction,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)

        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
                 n_resblock):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)\
            for i in range(n_resblock)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):

    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = int(args.scale)
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_mean, rgb_std)

        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(conv,
                          n_feats,
                          kernel_size,
                          reduction,
                          act=act,
                          res_scale=args.res_scale,
                          n_resblock=n_resblocks) for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


